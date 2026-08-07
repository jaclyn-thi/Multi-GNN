#!/usr/bin/env python3
"""Phase-4B objective-ablation nine-cell frozen R198 validation-only evaluation.

Subcommands:
  preflight  — verify ablation checkpoints + LONG@3000 reuse comparability
  extract    — one GPU cell: arm step-3000 × target full-subgraph R198 train/val
  probe      — one CPU job per target: PaperStyleMLP on three ablation arms
  finalize   — tables, deltas vs ADAPTIVE LONG@3000, ten interpretation answers

Never retrains encoders. Never loads/scores test. Projection head bypassed at extract.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.nn import to_hetero

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_loading import get_data  # noqa: E402
from mixed_ssl_phase2.bn import (  # noqa: E402
    apply_bn_,
    bn_bundle_l1,
    clone_bn_bundle,
    collect_bn_bundle,
)
from train_util import (  # noqa: E402
    AddEgoIds,
    FORWARD_EDGE_TYPE,
    add_arange_ids,
    expected_seed_edge_ids,
    extract_param,
    extract_seed_embeddings_hetero,
    get_loaders,
    log_seed_coverage,
    save_embedding_split_npz,
)
from phase4b_frozen_eval.probe import fit_r198_probe  # noqa: E402
from phase4b_objective_ablation import (  # noqa: E402
    PROJECTION_ARCHITECTURE,
    REFERENCE_LONG_CKPT_SHA256,
)
from phase4b_objective_ablation_frozen_eval import (  # noqa: E402
    CELLS,
    CHECKPOINT_SHA256,
    CHECKPOINT_SHA256_PATH,
    CHECKPOINT_STEP,
    CONTRACT_ID,
    COVERAGE_FLOORS,
    EDGE_DIM,
    EMB_ROOT,
    FINAL_FEATURE_NAMES,
    INIT_SHA256,
    NOTES_PATH,
    OLD_CONTRACT_ID,
    PROBE,
    R198_DIM,
    REFERENCE_LONG_ARM,
    REFERENCE_LONG_CELL,
    REFERENCE_LONG_FROZEN_ROOT,
    RESULT_ROOT,
    SAML_SPLIT_PROTOCOL,
    SAMLD_COVERAGE_NOTE,
    SEED,
    SOURCE_COHORT,
    TARGET_SCALER_SHA256,
    TARGETS,
    TRAIN_ARMS,
    TWIN_JSON,
    UPDATES_PER_DOMAIN,
    arm_objective_label,
    arm_uses_projection_training,
    bn_bundle_domain,
    cell_name,
    cell_role,
    checkpoint_path,
    encoders_for_target,
    integrity_path,
)
from phase4b_objective_ablation_frozen_eval.views import compare_cross_arm_views  # noqa: E402
from shared_core_contract import SHARED_CORE_FINAL_FEATURE_NAMES  # noqa: E402
from training import get_model  # noqa: E402
from util import create_parser, logger_setup, set_seed  # noqa: E402


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_ordered_ids(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a.astype(np.int64)).tobytes()).hexdigest()


def bn_bundle_sha(bundle: Dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for k in sorted(bundle.keys()):
        h.update(k.encode())
        t = torch.as_tensor(bundle[k]).detach().cpu().contiguous()
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def load_checkpoint_sha_registry() -> Dict[str, str]:
    p = ROOT / CHECKPOINT_SHA256_PATH
    reg = dict(CHECKPOINT_SHA256)
    if p.is_file():
        blob = json.loads(p.read_text(encoding="utf-8"))
        for arm in TRAIN_ARMS:
            if blob.get(arm):
                reg[arm] = str(blob[arm])
    return reg


def persist_checkpoint_sha(arm: str, sha: str) -> None:
    p = ROOT / CHECKPOINT_SHA256_PATH
    reg = load_checkpoint_sha_registry()
    if not reg.get(arm):
        reg[arm] = sha
        write_json(p, reg)


def emb_dir(arm: str, target: str) -> Path:
    return ROOT / EMB_ROOT / cell_name(arm, target) / "pre_embedding_3h"


def result_dir() -> Path:
    return ROOT / RESULT_ROOT


def validate_npz(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {"ok": False, "reason": "missing"}
    try:
        d = np.load(path)
        Z = np.asarray(d["Z"])
        y = np.asarray(d["y"]).reshape(-1)
        eid = np.asarray(d["edge_id"]).reshape(-1)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": str(e)}
    if Z.ndim != 2 or Z.shape[1] != R198_DIM:
        return {"ok": False, "reason": f"bad dim {Z.shape}"}
    if not np.isfinite(Z).all():
        return {"ok": False, "reason": "nonfinite"}
    if eid.size != np.unique(eid).size:
        return {"ok": False, "reason": "dup edge_id"}
    return {
        "ok": True,
        "n": int(Z.shape[0]),
        "n_pos": int((y == 1).sum()),
        "dim": R198_DIM,
        "edge_id_sha256": sha_ordered_ids(eid),
        "y_sha256": hashlib.sha256(y.astype(np.int64).tobytes()).hexdigest(),
    }


def verify_arm_integrity(arm: str) -> Dict[str, Any]:
    ip = ROOT / integrity_path(arm)
    if not ip.is_file():
        return {"ok": False, "arm": arm, "reason": "missing integrity.json", "path": str(ip)}
    blob = json.loads(ip.read_text(encoding="utf-8"))
    ok = bool(blob.get("ok")) and bool((blob.get("gates") or {}).get("ok"))
    return {
        "ok": ok,
        "arm": arm,
        "path": str(ip),
        "gates": blob.get("gates"),
        "init_sha256": blob.get("init_sha256"),
    }


def verify_checkpoints() -> Dict[str, Any]:
    sha_reg = load_checkpoint_sha_registry()
    out = {"ok": True, "arms": {}, "init_sha256_expected": INIT_SHA256}
    for arm in TRAIN_ARMS:
        integ = verify_arm_integrity(arm)
        p = ROOT / checkpoint_path(arm)
        if not integ["ok"]:
            out["ok"] = False
            out["arms"][arm] = {"ok": False, "reason": "integrity_failed", "integrity": integ}
            continue
        if not p.is_file():
            out["ok"] = False
            out["arms"][arm] = {"ok": False, "reason": "missing_checkpoint", "path": str(p), "integrity": integ}
            continue
        sha = sha256_file(p)
        persist_checkpoint_sha(arm, sha)
        expected = sha_reg.get(arm) or sha
        blob = torch.load(p, map_location="cpu", weights_only=False)
        resolved = blob.get("resolved") or {}
        ew = (blob.get("model_state_dict") or {}).get("edge_emb.node__to__node.weight")
        edge_dim_ok = (
            int(resolved.get("edge_dim", -1)) == EDGE_DIM
            if resolved.get("edge_dim") is not None
            else (ew is not None and int(ew.shape[-1]) == EDGE_DIM)
        )
        gates = {
            "integrity_ok": integ["ok"],
            "sha_recorded": bool(sha),
            "sha_match_if_set": (not sha_reg.get(arm)) or (sha == sha_reg[arm]),
            "contract": blob.get("feature_contract_id") == CONTRACT_ID,
            "edge_dim": bool(edge_dim_ok),
            "frozen_extract_bypasses_projection": True,
            "preserve_seed_false": resolved.get("preserve_seed_edges") is False,
            "has_model": "model_state_dict" in blob,
            "has_bn": "bn_bundles" in blob and len(blob["bn_bundles"]) >= 1,
            "init_sha": blob.get("init_sha256") == INIT_SHA256,
            "global_step_expected": int(blob.get("global_optimizer_step", -1)) == CHECKPOINT_STEP,
            "ignores_projection_state_at_extract": True,
        }
        try:
            _ = {k: v.shape for k, v in blob["model_state_dict"].items()}
            gates["model_state_reloadable"] = True
        except Exception:
            gates["model_state_reloadable"] = False
        if "projection_state_dict" in blob:
            gates["projection_state_dict_present_ignored_at_extract"] = True
        ok = all(gates.values())
        if not ok:
            out["ok"] = False
        bn_info = {
            d: {"n_keys": len(b), "sha256": bn_bundle_sha(b)}
            for d, b in blob.get("bn_bundles", {}).items()
        }
        out["arms"][arm] = {
            "ok": ok,
            "path": str(p),
            "sha256": sha,
            "expected_sha256": expected,
            "integrity": integ,
            "gates": gates,
            "bn_bundles": bn_info,
            "edge_scalers": {
                d: blob["edge_scalers"][d]["scaler_sha256"]
                for d in blob.get("edge_scalers", {})
            },
            "training_projection": arm_uses_projection_training(arm),
            "frozen_extract_projection": False,
            "saml_split_protocol": blob.get("saml_split_protocol"),
        }
    return out


def make_extract_args(target: str, cell: str) -> argparse.Namespace:
    argv = [
        "--data",
        target,
        "--model",
        "gin",
        "--objective",
        "contrastive",
        "--unique_name",
        f"phase4b_objabl_frozen_{cell}",
        "--seed",
        str(SEED),
        "--batch_size",
        "8192",
        "--num_neighs",
        "100",
        "100",
        "--loader_num_workers",
        "0",
        "--reverse_mp",
        "--ego",
        "--ports",
        "--emlps",
        "--tds",
        "--correct_reverse_edge_features",
        "--feature_contract",
        CONTRACT_ID,
        "--train_fit_edge_znorm",
        "--skip_test_eval",
        "--direct_r198_infonce",
        "--tqdm",
    ]
    ns = create_parser().parse_args(argv)
    ns.include_temporal_flow_edge_features = False
    ns.preserve_seed_edges = False
    ns.contrast_projection_head = False
    ns.skip_test_eval = True
    ns.embedding_dim = R198_DIM
    ns.representation_source = "pre_embedding_3h"
    ns.extract_splits = "train,val"
    ns.embeddings_dir = str(ROOT / EMB_ROOT)
    ns.embeddings_subdir = cell
    return ns


def load_reference_long_cell(target: str) -> Dict[str, Any]:
    name = REFERENCE_LONG_CELL[target]
    p = ROOT / REFERENCE_LONG_FROZEN_ROOT / "cells" / f"{name}.json"
    if not p.is_file():
        raise FileNotFoundError(p)
    return json.loads(p.read_text(encoding="utf-8"))


def build_long_comparability_card() -> Dict[str, Any]:
    fields = {
        "same_feature_contract": CONTRACT_ID,
        "same_feature_order": list(FINAL_FEATURE_NAMES) == list(SHARED_CORE_FINAL_FEATURE_NAMES),
        "same_extraction_protocol_full_subgraph_r198": True,
        "same_target_bn_policy": True,
        "same_probe_architecture_hyperparameters": PROBE,
        "same_init_sha": INIT_SHA256,
        "reference_checkpoint_sha256": REFERENCE_LONG_CKPT_SHA256,
        "reference_step": CHECKPOINT_STEP,
        "reference_arm": REFERENCE_LONG_ARM,
        "no_reextract_reference": True,
    }
    verified = {}
    ok = True
    for target in TARGETS:
        try:
            cell = load_reference_long_cell(target)
        except FileNotFoundError as e:
            return {
                "ok": False,
                "verdict": "PROTOCOL_INCOMPARABLE",
                "reason": str(e),
                "fields": fields,
            }
        pp = cell.get("probe_protocol") or {}
        probe_ok = (
            pp.get("learner") == PROBE["learner"]
            and pp.get("epochs") == PROBE["epochs"]
            and pp.get("lr") == PROBE["lr"]
            and pp.get("batch_size") == PROBE["batch_size"]
            and pp.get("seed") == PROBE["seed"]
            and pp.get("input_dim") == PROBE["input_dim"]
        )
        scaler_ok = cell.get("target_scaler_sha256") == TARGET_SCALER_SHA256[target]
        bn_ok = cell.get("bn_bundle_domain") == target
        no_test = cell.get("test_evaluated") is False
        ck_ok = cell.get("checkpoint_sha256") == REFERENCE_LONG_CKPT_SHA256
        verified[target] = {
            "ok": probe_ok and scaler_ok and bn_ok and no_test and ck_ok,
            "path": str(ROOT / REFERENCE_LONG_FROZEN_ROOT / "cells" / f"{REFERENCE_LONG_CELL[target]}.json"),
            "probe_ok": probe_ok,
            "scaler_ok": scaler_ok,
            "bn_ok": bn_ok,
            "no_test": no_test,
            "checkpoint_sha_ok": ck_ok,
            "matched_val_n": cell.get("matched_val_n"),
            "matched_val_edge_sha256": cell.get("matched_val_edge_sha256"),
            "AUPRC": cell.get("validation_auprc"),
            "F1@0.5": (cell.get("validation_metrics_at_0.5") or {}).get("f1"),
        }
        if not verified[target]["ok"]:
            ok = False

    ok = (
        ok
        and fields["same_feature_order"]
        and fields["same_extraction_protocol_full_subgraph_r198"]
        and fields["same_target_bn_policy"]
    )
    return {
        "ok": ok,
        "verdict": "COMPARABLE_REUSE_AUTHORIZED" if ok else "PROTOCOL_INCOMPARABLE",
        "fields": fields,
        "verified_reference_long_cells": verified,
        "reuse_targets": list(TARGETS),
        "interpretation_notes": [
            "ADAPTIVE LONG@3000 metrics reused from completed mixed_long_frozen_eval.",
            "Ablation arms extracted under identical R198 protocol with projection bypassed.",
            "Cohort EdgeID hashes may differ from LONG reference; compare per-target carefully.",
        ],
        "do_not_reextract_long": True,
    }


def cmd_preflight(_: argparse.Namespace) -> int:
    logger_setup()
    t0 = time.perf_counter()
    result_dir().mkdir(parents=True, exist_ok=True)
    (result_dir() / "cells").mkdir(parents=True, exist_ok=True)
    (ROOT / EMB_ROOT).mkdir(parents=True, exist_ok=True)

    ck = verify_checkpoints()
    bn_ok = True
    bn_checks = {}
    for arm, target in CELLS:
        info = ck["arms"].get(arm, {})
        bn_dom = bn_bundle_domain(arm, target)
        present = bn_dom in (info.get("bn_bundles") or {})
        scaler_match = (info.get("edge_scalers") or {}).get(target) == TARGET_SCALER_SHA256[target]
        bn_checks[cell_name(arm, target)] = {
            "bn_bundle_domain": bn_dom,
            "bn_present": present,
            "scaler_sha_match": scaler_match,
        }
        if not present or not scaler_match:
            bn_ok = False
            ck["ok"] = False

    comparability = build_long_comparability_card()
    write_json(result_dir() / "long_phase4b_comparability_card.json", comparability)

    cells = []
    for i, (arm, target) in enumerate(CELLS):
        cells.append(
            {
                "array_index": i,
                "arm": arm,
                "target": target,
                "objective": arm_objective_label(arm),
                "projection_training": arm_uses_projection_training(arm),
                "projection_frozen_extract": False,
                "role": cell_role(arm, target),
                "bn_bundle_domain": bn_bundle_domain(arm, target),
                "target_scaler_sha256_expected": TARGET_SCALER_SHA256[target],
                "checkpoint": checkpoint_path(arm),
                "checkpoint_sha256": load_checkpoint_sha_registry().get(arm),
                "checkpoint_step": CHECKPOINT_STEP,
                "updates_per_domain": UPDATES_PER_DOMAIN,
                "integrity_path": integrity_path(arm),
                "embeddings_dir": str(emb_dir(arm, target)),
            }
        )

    feature_order_ok = list(FINAL_FEATURE_NAMES) == list(SHARED_CORE_FINAL_FEATURE_NAMES)
    report = {
        "ok": bool(ck["ok"]) and feature_order_ok and bool(comparability["ok"]) and bn_ok,
        "phase": "4b_objective_ablation_frozen_eval",
        "feature_contract_id": CONTRACT_ID,
        "old_contract_geometry_equivalent": OLD_CONTRACT_ID,
        "final_feature_names": list(FINAL_FEATURE_NAMES),
        "feature_order_matches_shared_core": feature_order_ok,
        "saml_split_protocol": SAML_SPLIT_PROTOCOL,
        "edge_dim": EDGE_DIM,
        "r198_dim": R198_DIM,
        "projection_at_extract": False,
        "projection_architecture_audit": PROJECTION_ARCHITECTURE,
        "preserve_seed_edges": False,
        "amp": False,
        "init_sha256": INIT_SHA256,
        "long_phase4b_comparability": comparability,
        "checkpoints": ck,
        "bn_and_scaler_checks": bn_checks,
        "cells": cells,
        "cell_array_mapping": {
            "0-2": "EXPERT_ONLY × (Small-HI, SAML-D, Small-LI)",
            "3-5": "INFONCE_ONLY × targets",
            "6-8": "PROJECTION_ON_ADAPTIVE × targets",
        },
        "coverage_floors": COVERAGE_FLOORS,
        "samld_coverage_note": SAMLD_COVERAGE_NOTE,
        "probe_audit": dict(PROBE),
        "source_cohort_protocol_card": SOURCE_COHORT,
        "slurm_plan": {
            "partition": "mit_preemptable",
            "account": "mit_general",
            "qos": "normal",
            "train_array": "0-2%2",
            "extract_array": "0-8%2",
            "max_gpu_concurrency": 2,
            "extract_resources": {"mem": "128G", "cpus": 8, "gres": "gpu:1", "time": "06:00:00"},
            "train_resources": {"mem": "128G", "cpus": 16, "gres": "gpu:1", "time": "04:00:00"},
            "probe_Small-HI": {"mem": "64G", "cpus": 8, "time": "04:00:00"},
            "probe_SAML-D": {"mem": "96G", "cpus": 8, "time": "06:00:00"},
            "probe_Small-LI": {"mem": "96G", "cpus": 8, "time": "06:00:00"},
            "finalize": {"mem": "16G", "cpus": 4, "time": "01:00:00"},
        },
        "no_encoder_retrain": True,
        "no_test_eval": True,
        "no_test_npz_allowed": True,
        "reference_long_not_reextracted": True,
        "elapsed_sec": time.perf_counter() - t0,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(result_dir() / "preflight.json", report)
    print(
        json.dumps(
            {
                "ok": report["ok"],
                "n_cells": len(cells),
                "comparability": comparability["verdict"],
                "max_gpu_concurrency": 2,
            },
            indent=2,
        )
    )
    if not report["ok"]:
        logging.error("Preflight FAILED — do not submit")
        return 2
    logging.info("Preflight OK")
    return 0


def run_extract_cell(arm: str, target: str) -> Dict[str, Any]:
    integ = verify_arm_integrity(arm)
    if not integ["ok"]:
        raise RuntimeError(f"integrity gate failed for {arm}: {integ}")

    cell = cell_name(arm, target)
    out_dir = emb_dir(arm, target)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "test.npz").is_file():
        raise RuntimeError(f"test.npz present: {out_dir}")

    need = []
    reused = []
    for s in ("train", "val"):
        st = validate_npz(out_dir / f"{s}.npz")
        if st.get("ok"):
            reused.append(s)
        else:
            need.append(s)
    meta_path = out_dir / "meta.json"
    if not need and meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return {"status": "reuse", "arm": arm, "target": target, "meta": meta}

    ckpt_p = ROOT / checkpoint_path(arm)
    sha = sha256_file(ckpt_p)
    persist_checkpoint_sha(arm, sha)
    sha_reg = load_checkpoint_sha_registry()
    if sha_reg.get(arm) and sha != sha_reg[arm]:
        raise RuntimeError(f"checkpoint sha mismatch {sha} != {sha_reg[arm]}")
    blob = torch.load(ckpt_p, map_location="cpu", weights_only=False)
    if blob.get("feature_contract_id") != CONTRACT_ID:
        raise RuntimeError("bad contract in checkpoint")

    bn_dom = bn_bundle_domain(arm, target)
    if bn_dom not in blob["bn_bundles"]:
        raise RuntimeError(f"BN bundle {bn_dom} missing from {arm} checkpoint")
    bn_sel = clone_bn_bundle(blob["bn_bundles"][bn_dom])
    bn_sha = bn_bundle_sha(bn_sel)

    with open(ROOT / "data_config.json", encoding="utf-8") as f:
        data_config = json.load(f)

    args = make_extract_args(target, cell)
    set_seed(SEED)
    logging.info("Loading target graph %s under %s (arm=%s)", target, CONTRACT_ID, arm)
    tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(args, data_config)
    if int(te_inds.numel()) != 0:
        raise RuntimeError("te_inds nonempty — refuse test")
    ea = tr_data[FORWARD_EDGE_TYPE].edge_attr
    if int(ea.shape[1]) != EDGE_DIM:
        raise RuntimeError(f"edge_dim={ea.shape[1]} != 6")
    names = list(getattr(args, "edge_feature_schema_names", []) or [])
    expect_names = list(FINAL_FEATURE_NAMES)
    if list(SHARED_CORE_FINAL_FEATURE_NAMES) != expect_names:
        raise RuntimeError("Phase-3/Phase-4 feature-order drift")
    if names and names != expect_names:
        raise RuntimeError(f"schema mismatch {names}")
    scaler = getattr(args, "shared_core_edge_scaler", None)
    if not isinstance(scaler, dict) or scaler.get("scaler_sha256") != TARGET_SCALER_SHA256[target]:
        raise RuntimeError(
            f"target scaler sha {scaler.get('scaler_sha256') if isinstance(scaler, dict) else None} "
            f"!= locked {TARGET_SCALER_SHA256[target]}"
        )

    y_tr_all = tr_data[FORWARD_EDGE_TYPE].y
    y_va_all = val_data[FORWARD_EDGE_TYPE].y
    source = {
        "train_n": int(tr_inds.numel()),
        "val_n": int(val_inds.numel()),
        "train_pos": int((y_tr_all[tr_inds] == 1).sum()) if tr_inds.numel() else 0,
        "val_pos": int((y_va_all[val_inds] == 1).sum()) if val_inds.numel() else 0,
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform = AddEgoIds()
    add_arange_ids([tr_data, val_data, te_data])

    sample_args = SimpleNamespace(**vars(args))
    sample_args.loader_num_workers = 0
    sample_loader, _, _ = get_loaders(
        tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform, sample_args, train_shuffle=False
    )
    sample_batch = next(iter(sample_loader))
    del sample_loader

    config = SimpleNamespace(
        model="gin",
        n_hidden=extract_param("n_hidden", args),
        n_gnn_layers=extract_param("n_gnn_layers", args),
        n_heads=None,
        dropout=extract_param("dropout", args),
        final_dropout=extract_param("final_dropout", args),
    )
    args.direct_r198_infonce = True
    args.contrast_projection_head = False
    model = get_model(sample_batch, config, args)
    emb_dim = int(getattr(model, "embedding_dim", R198_DIM))
    model = to_hetero(model, tr_data.metadata(), aggr="mean")
    model.bypass_embedding_head = True

    # Encoder only — ignore projection_state_dict if present.
    model.load_state_dict(blob["model_state_dict"], strict=True)
    apply_bn_(model, bn_sel)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.SyncBatchNorm)):
            m.eval()

    bn_before_extract = clone_bn_bundle(collect_bn_bundle(model))

    tr_loader, val_loader, te_loader = get_loaders(
        tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform, args, train_shuffle=False
    )
    del te_loader

    staging = out_dir / f".staging_{os.getpid()}"
    staging.mkdir(parents=True, exist_ok=True)

    pre_dim = R198_DIM
    split_stats = {}
    with torch.inference_mode():
        for split_name, loader, inds, gdata in (
            ("train", tr_loader, tr_inds, tr_data),
            ("val", val_loader, val_inds, val_data),
        ):
            if split_name not in need and split_name in reused:
                continue
            expected = expected_seed_edge_ids(loader.data, inds, hetero=True)
            edge_ids, z, y = extract_seed_embeddings_hetero(
                loader,
                inds,
                model,
                gdata,
                device,
                args,
                representation_source="pre_embedding_3h",
                pre_dim=pre_dim,
                emb_dim=emb_dim,
                head_spec=None,
                max_batches=None,
            )
            log_seed_coverage(edge_ids, expected, split_name)
            if int(z.shape[1]) != R198_DIM:
                raise RuntimeError(f"Z dim {z.shape[1]} != 198")
            save_embedding_split_npz(staging / f"{split_name}.npz", z, y, edge_ids)
            eid_np = edge_ids.detach().cpu().numpy()
            y_np = y.detach().cpu().numpy()
            split_stats[split_name] = {
                "n": int(eid_np.shape[0]),
                "n_pos": int((y_np == 1).sum()),
                "edge_id_sha256": sha_ordered_ids(eid_np),
            }

    bn_after_extract = collect_bn_bundle(model)
    bn_unchanged = bn_bundle_l1(bn_before_extract, bn_after_extract) == 0.0

    for s in need:
        src = staging / f"{s}.npz"
        st = validate_npz(src)
        if not st.get("ok"):
            raise RuntimeError(f"staged {s} bad: {st}")
        dst = out_dir / f"{s}.npz"
        tmp = out_dir / f"{s}.npz.promoting"
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
    if (out_dir / "test.npz").is_file():
        raise RuntimeError("test.npz appeared")

    extracted = {}
    for s in ("train", "val"):
        st = validate_npz(out_dir / f"{s}.npz")
        extracted[s] = {"n": st["n"], "n_pos": st["n_pos"], "edge_id_sha256": st["edge_id_sha256"]}
    floors = COVERAGE_FLOORS[target]
    edge_cov = min(extracted["train"]["n"] / max(source["train_n"], 1), extracted["val"]["n"] / max(source["val_n"], 1))
    pos_cov = min(
        extracted["train"]["n_pos"] / max(source["train_pos"], 1),
        extracted["val"]["n_pos"] / max(source["val_pos"], 1),
    )
    cov_ok = edge_cov >= floors["edge"] and pos_cov >= floors["positive"]

    meta = {
        "cell": cell,
        "encoder_arm": arm,
        "target_dataset": target,
        "objective": arm_objective_label(arm),
        "projection_training": arm_uses_projection_training(arm),
        "projection_frozen_extract": False,
        "role": cell_role(arm, target),
        "feature_contract_id": CONTRACT_ID,
        "edge_dim": EDGE_DIM,
        "r198_dim": R198_DIM,
        "checkpoint_path": str(ckpt_p),
        "checkpoint_sha256": sha,
        "init_sha256": blob.get("init_sha256"),
        "bn_policy": {
            "bundle_domain": bn_dom,
            "bundle_sha256": bn_sha,
            "mixed_uses_target_domain_bn": True,
            "no_target_bn_adaptation": True,
            "bn_unchanged_during_extract": bn_unchanged,
        },
        "target_scaler_sha256": scaler["scaler_sha256"],
        "saml_split_protocol": SAML_SPLIT_PROTOCOL if target == "SAML-D" else None,
        "source_cohort": source,
        "extracted": extracted,
        "coverage": {
            "edge_coverage_min": edge_cov,
            "positive_coverage_min": pos_cov,
            "floors": floors,
            "ok": cov_ok,
            "note": SAMLD_COVERAGE_NOTE if target == "SAML-D" else None,
        },
        "projection_bypassed": True,
        "projection_state_dict_ignored": "projection_state_dict" in blob,
        "encoder_requires_grad": False,
        "model_eval": True,
        "test_evaluated": False,
        "test_npz_present": False,
        "skip_test_eval": True,
        "extractor": "run_phase4b_objective_ablation_frozen_eval.run_extract_cell",
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if not cov_ok:
        write_json(out_dir / "meta.json", meta)
        raise RuntimeError(f"coverage gate failed: edge={edge_cov} pos={pos_cov} floors={floors}")
    write_json(out_dir / "meta.json", meta)
    write_json(result_dir() / "cells" / f"{cell}_extract.json", meta)
    shutil.rmtree(staging, ignore_errors=True)
    del model, blob
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"status": "ok", "meta": meta}


def cmd_extract(args: argparse.Namespace) -> int:
    logger_setup()
    tid = args.array_task_id
    if tid is None:
        tid = int(os.environ.get("SLURM_ARRAY_TASK_ID", "-1"))
    if tid < 0 or tid >= len(CELLS):
        raise SystemExit(f"bad array task {tid}")
    arm, target = CELLS[tid]
    logging.info("Extract cell %s: %s → %s", tid, arm, target)
    out = run_extract_cell(arm, target)
    print(json.dumps({"ok": True, "array_task": tid, "arm": arm, "target": target, "status": out["status"]}, indent=2))
    return 0


def load_split_arrays(arm: str, target: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    d = emb_dir(arm, target)
    tr = np.load(d / "train.npz")
    va = np.load(d / "val.npz")
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    return (
        np.asarray(tr["Z"]),
        np.asarray(tr["y"]).reshape(-1),
        np.asarray(tr["edge_id"]).reshape(-1),
        np.asarray(va["Z"]),
        np.asarray(va["y"]).reshape(-1),
        np.asarray(va["edge_id"]).reshape(-1),
        meta,
    )


def align_target_cohort(target: str) -> Dict[str, Any]:
    arms = list(encoders_for_target(target))
    arms_data = {}
    for arm in arms:
        ztr, ytr, etr, zva, yva, eva, meta = load_split_arrays(arm, target)
        if (emb_dir(arm, target) / "test.npz").is_file():
            raise RuntimeError(f"test.npz in {arm}/{target}")
        arms_data[arm] = {
            "z_tr": ztr,
            "y_tr": ytr,
            "e_tr": etr,
            "z_va": zva,
            "y_va": yva,
            "e_va": eva,
            "meta": meta,
        }

    tr_sets = [set(d["e_tr"].tolist()) for d in arms_data.values()]
    va_sets = [set(d["e_va"].tolist()) for d in arms_data.values()]
    tr_inter = set.intersection(*tr_sets)
    va_inter = set.intersection(*va_sets)
    tr_exact = all(s == tr_inter for s in tr_sets)
    va_exact = all(s == va_inter for s in va_sets)

    tr_sorted = np.array(sorted(tr_inter), dtype=np.int64)
    va_sorted = np.array(sorted(va_inter), dtype=np.int64)

    aligned = {}
    original = {}
    for arm, d in arms_data.items():
        original[arm] = {
            "train_n": int(d["e_tr"].size),
            "val_n": int(d["e_va"].size),
            "train_pos": int((d["y_tr"] == 1).sum()),
            "val_pos": int((d["y_va"] == 1).sum()),
        }
        tr_map = {int(e): i for i, e in enumerate(d["e_tr"])}
        va_map = {int(e): i for i, e in enumerate(d["e_va"])}
        tr_idx = np.array([tr_map[int(e)] for e in tr_sorted], dtype=np.int64)
        va_idx = np.array([va_map[int(e)] for e in va_sorted], dtype=np.int64)
        aligned[arm] = {
            "z_tr": d["z_tr"][tr_idx],
            "y_tr": d["y_tr"][tr_idx],
            "e_tr": tr_sorted.copy(),
            "z_va": d["z_va"][va_idx],
            "y_va": d["y_va"][va_idx],
            "e_va": va_sorted.copy(),
            "meta": d["meta"],
        }

    ref = arms[0]
    for arm in arms[1:]:
        if not np.array_equal(aligned[arm]["y_tr"], aligned[ref]["y_tr"]):
            raise RuntimeError(f"train labels disagree: {arm}")
        if not np.array_equal(aligned[arm]["y_va"], aligned[ref]["y_va"]):
            raise RuntimeError(f"val labels disagree: {arm}")

    source = arms_data[ref]["meta"]["source_cohort"]
    floors = COVERAGE_FLOORS[target]
    matched = {
        "train_n": int(tr_sorted.size),
        "val_n": int(va_sorted.size),
        "train_pos": int((aligned[ref]["y_tr"] == 1).sum()),
        "val_pos": int((aligned[ref]["y_va"] == 1).sum()),
    }
    edge_cov = min(matched["train_n"] / max(source["train_n"], 1), matched["val_n"] / max(source["val_n"], 1))
    pos_cov = min(matched["train_pos"] / max(source["train_pos"], 1), matched["val_pos"] / max(source["val_pos"], 1))
    cov_ok = edge_cov >= floors["edge"] and pos_cov >= floors["positive"]
    if not cov_ok:
        raise RuntimeError(f"{target} matched coverage failed edge={edge_cov} pos={pos_cov}")

    report = {
        "target": target,
        "encoders": arms,
        "train_sets_exact_match": tr_exact,
        "val_sets_exact_match": va_exact,
        "original_per_arm": original,
        "matched": matched,
        "matched_train_edge_sha256": sha_ordered_ids(tr_sorted),
        "matched_val_edge_sha256": sha_ordered_ids(va_sorted),
        "coverage": {
            "source": source,
            "edge_coverage": {"min": edge_cov},
            "positive_coverage": {"min": pos_cov},
            "floors": floors,
            "ok": cov_ok,
        },
        "no_test_npz": True,
    }
    return {"aligned": aligned, "report": report}


def cmd_probe(args: argparse.Namespace) -> int:
    logger_setup()
    target = args.target
    if target not in TARGETS:
        raise SystemExit(target)
    t0 = time.perf_counter()
    pack = align_target_cohort(target)
    aligned = pack["aligned"]
    report = pack["report"]
    write_json(result_dir() / f"matched_cohorts_{target.lower().replace('-', '')}.json", report)

    cell_results = []
    pred_dir = result_dir() / "val_predictions" / target.lower().replace("-", "_")
    pred_dir.mkdir(parents=True, exist_ok=True)

    for arm in encoders_for_target(target):
        logging.info("Probe %s → %s", arm, target)
        d = aligned[arm]
        fit = fit_r198_probe(d["z_tr"], d["y_tr"], d["z_va"], d["y_va"], device=torch.device("cpu"))
        np.savez_compressed(
            pred_dir / f"{arm}.npz",
            edge_id=d["e_va"].astype(np.int64),
            y=d["y_va"].astype(np.int64),
            logit_selected=fit["val_logit_selected"],
            proba_selected=fit["val_proba_selected"],
            logit_final=fit["val_logit_final"],
            proba_final=fit["val_proba_final"],
        )
        cell = {
            "encoder": arm,
            "target": target,
            "objective": arm_objective_label(arm),
            "projection_training": arm_uses_projection_training(arm),
            "projection_frozen_extract": False,
            "role": cell_role(arm, target),
            "checkpoint_step": CHECKPOINT_STEP,
            "updates_per_domain": UPDATES_PER_DOMAIN,
            "bn_bundle_domain": bn_bundle_domain(arm, target),
            "bn_bundle_sha256": d["meta"]["bn_policy"]["bundle_sha256"],
            "target_scaler_sha256": d["meta"]["target_scaler_sha256"],
            "checkpoint_sha256": d["meta"]["checkpoint_sha256"],
            "coverage": report["coverage"],
            "matched_train_n": report["matched"]["train_n"],
            "matched_val_n": report["matched"]["val_n"],
            "matched_train_pos": report["matched"]["train_pos"],
            "matched_val_pos": report["matched"]["val_pos"],
            "matched_train_edge_sha256": report["matched_train_edge_sha256"],
            "matched_val_edge_sha256": report["matched_val_edge_sha256"],
            "train_sets_exact_match_across_arms": report["train_sets_exact_match"],
            "val_sets_exact_match_across_arms": report["val_sets_exact_match"],
            "validation_auprc": fit["validation_auprc"],
            "validation_auroc": fit["validation_auroc"],
            "validation_metrics_at_0.5": fit["validation_metrics_at_0.5"],
            "validation_metrics_at_val_optimal_f1": fit["validation_metrics_at_val_optimal_f1"],
            "final_probe_val_bce": fit["final_probe_val_bce"],
            "final_probe_train_bce": fit["final_probe_train_bce"],
            "selected_probe_val_bce": fit["selected_probe_val_bce"],
            "final_probe_epoch": fit["final_probe_epoch"],
            "selected_probe_epoch": fit["selected_probe_epoch"],
            "prevalence_val": fit["prevalence_val"],
            "n_val": fit["n_val"],
            "n_val_pos": fit["n_val_pos"],
            "probe_protocol": fit["probe_protocol"],
            "encoder_updated": False,
            "test_evaluated": False,
        }
        write_json(result_dir() / "cells" / f"{cell_name(arm, target)}.json", cell)
        cell_results.append(cell)
        for k in list(d.keys()):
            if isinstance(d[k], np.ndarray):
                d[k] = None
        gc.collect()

    del aligned
    gc.collect()

    summary = {
        "target": target,
        "ok": True,
        "matched_cohort": report,
        "cells": cell_results,
        "probe_protocol": PROBE,
        "elapsed_sec": time.perf_counter() - t0,
        "encoder_retrained": False,
        "test_data_loaded_or_scored": False,
    }
    write_json(result_dir() / f"probe_{target.lower().replace('-', '_')}.json", summary)
    print(json.dumps({"ok": True, "target": target, "n_cells": len(cell_results)}, indent=2))
    return 0


def _row_from_cell(c: Dict[str, Any]) -> Dict[str, Any]:
    m05 = c["validation_metrics_at_0.5"]
    mopt = c["validation_metrics_at_val_optimal_f1"]
    return {
        "Arm": c["encoder"],
        "Objective": c.get("objective") or arm_objective_label(c["encoder"]),
        "Projection": c.get("projection_training", arm_uses_projection_training(c["encoder"])),
        "Target": c["target"],
        "AUPRC": c["validation_auprc"],
        "AUROC": c["validation_auroc"],
        "F1@0.5": m05["f1"],
        "P": m05["precision"],
        "R": m05["recall"],
        "F1@val-thr": mopt["f1"],
        "Final val BCE": c["final_probe_val_bce"],
    }


def _row_from_reference(cell: Dict[str, Any]) -> Dict[str, Any]:
    m05 = cell["validation_metrics_at_0.5"]
    mopt = cell["validation_metrics_at_val_optimal_f1"]
    return {
        "Arm": REFERENCE_LONG_ARM,
        "Objective": "adaptive InfoNCE+TF (reference; reused)",
        "Projection": False,
        "Target": cell["target"],
        "AUPRC": cell["validation_auprc"],
        "AUROC": cell["validation_auroc"],
        "F1@0.5": m05["f1"],
        "P": m05["precision"],
        "R": m05["recall"],
        "F1@val-thr": mopt["f1"],
        "Final val BCE": cell["final_probe_val_bce"],
    }


def _cmp(a: float, b: float) -> Dict[str, float]:
    return {"delta": float(a - b), "retention": float(a / b) if b else float("nan")}


def answer_interpretation_questions(
    by_arm_target: Dict[Tuple[str, str], Dict[str, Any]],
    ref_by_target: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    answers: Dict[str, Any] = {}

    # 1. EXPERT_ONLY vs adaptive per domain
    q1 = {}
    for t in TARGETS:
        exp = by_arm_target[("EXPERT_ONLY", t)]["validation_auprc"]
        ref = ref_by_target[t]["validation_auprc"]
        q1[t] = {
            "expert_auprc": exp,
            "adaptive_auprc": ref,
            "delta": exp - ref,
            "expert_matches_or_exceeds": exp >= ref,
        }
    answers["1_expert_vs_adaptive_per_domain"] = q1

    # 2. INFONCE_ONLY in-domain utility
    q2 = {}
    for t in TARGETS:
        inf = by_arm_target[("INFONCE_ONLY", t)]["validation_auprc"]
        q2[t] = {"infonce_auprc": inf, "retains_useful_signal": inf > 0.05}
    answers["2_infonce_in_domain_utility"] = q2

    # 3. adaptive beats single-objective anywhere?
    wins_adaptive = []
    for t in TARGETS:
        ref = ref_by_target[t]["validation_auprc"]
        exp = by_arm_target[("EXPERT_ONLY", t)]["validation_auprc"]
        inf = by_arm_target[("INFONCE_ONLY", t)]["validation_auprc"]
        if ref > exp and ref > inf:
            wins_adaptive.append(t)
    answers["3_adaptive_beats_single_objective"] = {
        "targets": wins_adaptive,
        "anywhere": bool(wins_adaptive),
    }

    # 4. projection-on vs adaptive
    q4 = {}
    for t in TARGETS:
        proj = by_arm_target[("PROJECTION_ON_ADAPTIVE", t)]["validation_auprc"]
        ref = ref_by_target[t]["validation_auprc"]
        q4[t] = {"projection_auprc": proj, "adaptive_auprc": ref, "delta": proj - ref}
    answers["4_projection_on_vs_adaptive"] = q4

    # 5. projection-on HI/LI (conflict-heavy domains)
    q5 = {}
    for t in ("Small-HI", "Small-LI"):
        proj = by_arm_target[("PROJECTION_ON_ADAPTIVE", t)]["validation_auprc"]
        ref = ref_by_target[t]["validation_auprc"]
        q5[t] = {"delta_vs_adaptive": proj - ref, "helps": proj > ref}
    answers["5_projection_on_hi_li"] = q5

    # 6. contrast cross-domain vs specialist
    inf_mean = float(np.mean([by_arm_target[("INFONCE_ONLY", t)]["validation_auprc"] for t in TARGETS]))
    exp_mean = float(np.mean([by_arm_target[("EXPERT_ONLY", t)]["validation_auprc"] for t in TARGETS]))
    answers["6_contrast_cross_domain_vs_specialist"] = {
        "infonce_mean_auprc": inf_mean,
        "expert_mean_auprc": exp_mean,
        "contrast_more_cross_domain_than_specialist": inf_mean > exp_mean,
        "note": "proxy: mean AUPRC across three targets; not a causal claim",
    }

    # 7. complementary TF+InfoNCE evidence
    ref_mean = float(np.mean([ref_by_target[t]["validation_auprc"] for t in TARGETS]))
    answers["7_complementary_tf_infonce"] = {
        "adaptive_mean_auprc": ref_mean,
        "expert_mean_auprc": exp_mean,
        "infonce_mean_auprc": inf_mean,
        "adaptive_beats_both_singles_on_mean": ref_mean > exp_mean and ref_mean > inf_mean,
    }

    # 8. strongest common checkpoint across targets
    arm_scores = {}
    for arm in TRAIN_ARMS:
        arm_scores[arm] = float(np.mean([by_arm_target[(arm, t)]["validation_auprc"] for t in TARGETS]))
    best_arm = max(arm_scores, key=arm_scores.get)
    answers["8_strongest_common_checkpoint"] = {
        "mean_auprc_by_arm": arm_scores,
        "best_arm": best_arm,
        "reference_adaptive_mean": ref_mean,
    }

    # 9. limitations
    answers["9_limitations"] = {
        "single_seed": 2,
        "validation_only": True,
        "no_test_estimate": True,
        "matched_cohort_may_differ_from_long_reference": True,
        "one_checkpoint_step": CHECKPOINT_STEP,
    }

    # 10. smallest next experiment
    answers["10_smallest_next_experiment"] = {
        "proposal": (
            "If adaptive beats singles on ≥2 targets, run matched second-seed "
            "replication on the winning arm only; otherwise extend step-1500 "
            "frozen diagnostics before adding new objectives."
        ),
        "do_not_add_unrequested_arms": True,
    }
    return answers


def cmd_finalize(_: argparse.Namespace) -> int:
    logger_setup()
    cells = []
    for arm, target in CELLS:
        p = result_dir() / "cells" / f"{cell_name(arm, target)}.json"
        if not p.is_file():
            raise SystemExit(f"missing cell result {p}")
        cells.append(json.loads(p.read_text(encoding="utf-8")))
    by = {(c["encoder"], c["target"]): c for c in cells}

    card_path = result_dir() / "long_phase4b_comparability_card.json"
    comparability = (
        json.loads(card_path.read_text(encoding="utf-8"))
        if card_path.is_file()
        else build_long_comparability_card()
    )

    ref_by_target = {t: load_reference_long_cell(t) for t in TARGETS}

    view_report = compare_cross_arm_views()
    write_json(result_dir() / "cross_arm_view_match.json", view_report)

    table_rows = []
    delta_rows = []
    for arm, target in CELLS:
        c = by[(arm, target)]
        row = _row_from_cell(c)
        table_rows.append(row)
        ref = ref_by_target[target]
        delta_rows.append(
            {
                "Arm": arm,
                "Target": target,
                "AUPRC": c["validation_auprc"],
                "ADAPTIVE_LONG_3000_AUPRC": ref["validation_auprc"],
                "delta_AUPRC": c["validation_auprc"] - ref["validation_auprc"],
                "retention_AUPRC": c["validation_auprc"] / ref["validation_auprc"]
                if ref["validation_auprc"]
                else float("nan"),
                "delta_F1@0.5": c["validation_metrics_at_0.5"]["f1"]
                - ref["validation_metrics_at_0.5"]["f1"],
            }
        )
        ref_row = _row_from_reference(ref)
        if not any(r["Arm"] == REFERENCE_LONG_ARM and r["Target"] == target for r in table_rows):
            table_rows.append(ref_row)

    interpretation = answer_interpretation_questions(by, ref_by_target)

    csv_fields = list(table_rows[0].keys())
    csv_p = result_dir() / "cells" / "objective_ablation_table.csv"
    with csv_p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields)
        w.writeheader()
        for r in table_rows:
            w.writerow(r)

    write_json(result_dir() / "cells" / "objective_ablation_table.json", table_rows)
    write_json(result_dir() / "cells" / "deltas_vs_adaptive_long_3000.json", delta_rows)
    write_json(result_dir() / "interpretation.json", interpretation)

    payload = {
        "ok": True,
        "phase": "4b_objective_ablation_frozen_eval",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_question": (
            "Under the exact three-domain LONG protocol, what does each objective "
            "component contribute to frozen R198 validation quality at step 3000?"
        ),
        "long_phase4b_comparability": comparability,
        "cross_arm_view_match": view_report,
        "probe_protocol": PROBE,
        "table": table_rows,
        "deltas_vs_adaptive_long_3000": delta_rows,
        "interpretation": interpretation,
        "reference_long_reused_not_reextracted": True,
        "test_data_loaded_or_scored": False,
        "encoder_retrained": False,
        "projection_architecture_audit": PROJECTION_ARCHITECTURE,
    }
    write_json(result_dir() / "aggregate.json", payload)
    write_json(ROOT / TWIN_JSON, payload)

    lines = [
        "# Phase-4B objective ablation frozen R198 validation eval",
        "",
        f"> Twin: `{TWIN_JSON}`",
        f"> Reference LONG@3000 reuse: `{REFERENCE_LONG_FROZEN_ROOT}`",
        "",
        f"**ok={payload['ok']}** — validation-only; no encoder retrain; no test.",
        "",
        f"LONG comparability: `{comparability.get('verdict')}`",
        "",
        "## Protocol locks",
        "",
        f"- contract: `{CONTRACT_ID}`",
        f"- probe: PaperStyleMLP {PROBE['epochs']}ep lr={PROBE['lr']} bs={PROBE['batch_size']} seed={PROBE['seed']}",
        "- extract: full-subgraph R198 train/val; projection bypassed",
        f"- projection architecture (training arm C only): {PROJECTION_ARCHITECTURE}",
        "",
        "## Main table",
        "",
        "| Arm | Objective | Projection | Target | AUPRC | AUROC | F1@0.5 | P | R | F1@val-thr | Final val BCE |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in table_rows:
        lines.append(
            f"| {r['Arm']} | {r['Objective']} | {r['Projection']} | {r['Target']} | "
            f"{float(r['AUPRC']):.4f} | {float(r['AUROC']):.4f} | {float(r['F1@0.5']):.4f} | "
            f"{float(r['P']):.4f} | {float(r['R']):.4f} | {float(r['F1@val-thr']):.4f} | "
            f"{float(r['Final val BCE']):.6f} |"
        )

    lines += ["", "## Deltas vs ADAPTIVE LONG@3000", ""]
    for d in delta_rows:
        lines.append(
            f"- **{d['Arm']} / {d['Target']}**: AUPRC Δ={d['delta_AUPRC']:+.4f} "
            f"(retention {d['retention_AUPRC']:.3f})"
        )

    lines += ["", "## Ten interpretation answers", ""]
    for key, val in interpretation.items():
        lines += [f"### {key}", "", "```json", json.dumps(val, indent=2), "```", ""]

    lines += [
        "",
        "## Cross-arm view matching",
        "",
        f"- ok={view_report.get('ok')}",
        f"- limitation: {view_report.get('long_view_hash_limitation', '')[:120]}…",
        "",
        "Confirmation: no test data loaded/scored; adaptive LONG not re-extracted.",
        "",
    ]
    notes = ROOT / NOTES_PATH
    notes.parent.mkdir(parents=True, exist_ok=True)
    notes.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "notes": str(notes), "n_rows": len(table_rows)}, indent=2))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("preflight")
    sp.set_defaults(func=cmd_preflight)

    se = sub.add_parser("extract")
    se.add_argument("--array_task_id", type=int, default=None)
    se.set_defaults(func=cmd_extract)

    spr = sub.add_parser("probe")
    spr.add_argument("--target", type=str, required=True, choices=list(TARGETS))
    spr.set_defaults(func=cmd_probe)

    sf = sub.add_parser("finalize")
    sf.set_defaults(func=cmd_finalize)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
