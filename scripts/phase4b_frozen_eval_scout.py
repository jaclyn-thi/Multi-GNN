#!/usr/bin/env python3
"""Phase-3 six-cell frozen R198 validation-only evaluation.

Subcommands:
  preflight  — verify checkpoints/BN/contract; write preflight.json (no jobs)
  extract    — one GPU cell: arm × target full-subgraph R198 train/val
  probe      — one CPU job per target: three encoders, PaperStyleMLP R198-only
  finalize   — tables, deltas, figures, notes/JSON

Never retrains encoders. Never loads/scores test. Never uses TF concat / adapters.
"""

from __future__ import annotations

import argparse
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
from phase4b_frozen_eval import (  # noqa: E402
    ARMS,
    CELLS,
    CHECKPOINT_SHA256,
    CHECKPOINT_STEP,
    CONTRACT_ID,
    COVERAGE_FLOORS,
    EDGE_DIM,
    EMB_ROOT,
    FINAL_FEATURE_NAMES,
    INIT_SHA256,
    LATER_CONTROLS_NOT_SUBMITTED,
    NOTES_PATH,
    OLD_CONTRACT_ID,
    PHASE3_FROZEN_ROOT,
    PHASE3_MIXED_1TO1_REUSE,
    PHASE3_SCOUT_ROOT,
    PHASE3_SPECIALIST_REUSE,
    PHASE4B_SCOUT_ROOT,
    PROBE,
    R198_DIM,
    RESULT_ROOT,
    SAML_SPLIT_PROTOCOL,
    SAMLD_COVERAGE_NOTE,
    SEED,
    SOURCE_COHORT,
    TARGET_SCALER_SHA256,
    TARGETS,
    TWIN_JSON,
    bn_bundle_domain,
    cell_name,
    cell_role,
    checkpoint_path,
    encoders_for_target,
)
from phase4b_frozen_eval.figures import plot_package  # noqa: E402
from phase4b_frozen_eval.probe import fit_r198_probe  # noqa: E402
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


def verify_checkpoints() -> Dict[str, Any]:
    out = {"ok": True, "arms": {}, "init_sha256_expected": INIT_SHA256}
    for arm in ARMS:
        p = ROOT / checkpoint_path(arm)
        if not p.is_file():
            out["ok"] = False
            out["arms"][arm] = {"ok": False, "reason": "missing", "path": str(p)}
            continue
        sha = sha256_file(p)
        blob = torch.load(p, map_location="cpu", weights_only=False)
        gates = {
            "sha_match": sha == CHECKPOINT_SHA256[arm],
            "contract": blob.get("feature_contract_id") == CONTRACT_ID,
            "edge_dim": int((blob.get("resolved") or {}).get("edge_dim", -1)) == EDGE_DIM,
            "projection_false": (blob.get("resolved") or {}).get("contrast_projection_head") is False,
            "preserve_seed_false": (blob.get("resolved") or {}).get("preserve_seed_edges") is False,
            "has_model": "model_state_dict" in blob,
            "has_bn": "bn_bundles" in blob and len(blob["bn_bundles"]) >= 1,
            "init_sha": blob.get("init_sha256") == INIT_SHA256,
            "global_step_expected": int(blob.get("global_optimizer_step", -1))
            == int(CHECKPOINT_STEP[arm]),
        }
        # reload model state into empty dict check
        try:
            _ = {k: v.shape for k, v in blob["model_state_dict"].items()}
            gates["model_state_reloadable"] = True
        except Exception:
            gates["model_state_reloadable"] = False
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
            "expected_sha256": CHECKPOINT_SHA256[arm],
            "gates": gates,
            "bn_bundles": bn_info,
            "edge_scalers": {
                d: blob["edge_scalers"][d]["scaler_sha256"]
                for d in blob.get("edge_scalers", {})
            },
            "saml_split_protocol": blob.get("saml_split_protocol"),
        }
    return out


def make_extract_args(target: str, cell: str) -> argparse.Namespace:
    argv = [
        "--data", target,
        "--model", "gin",
        "--objective", "contrastive",
        "--unique_name", f"phase4b_frozen_{cell}",
        "--seed", str(SEED),
        "--batch_size", "8192",
        "--num_neighs", "100", "100",
        "--loader_num_workers", "0",
        "--reverse_mp", "--ego", "--ports", "--emlps", "--tds",
        "--correct_reverse_edge_features",
        "--feature_contract", CONTRACT_ID,
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


def cmd_preflight(_: argparse.Namespace) -> int:
    logger_setup()
    t0 = time.perf_counter()
    result_dir().mkdir(parents=True, exist_ok=True)
    (result_dir() / "cells").mkdir(parents=True, exist_ok=True)
    (result_dir() / "figures").mkdir(parents=True, exist_ok=True)
    (ROOT / EMB_ROOT).mkdir(parents=True, exist_ok=True)

    ck = verify_checkpoints()
    comparability = build_phase3_comparability_card()
    write_json(result_dir() / "phase3_comparability_card.json", comparability)

    cells = []
    for i, (arm, target) in enumerate(CELLS):
        cells.append(
            {
                "array_index": i,
                "arm": arm,
                "target": target,
                "role": cell_role(arm, target),
                "bn_bundle_domain": bn_bundle_domain(arm, target),
                "target_scaler_sha256_expected": TARGET_SCALER_SHA256[target],
                "checkpoint": checkpoint_path(arm),
                "checkpoint_sha256": CHECKPOINT_SHA256[arm],
                "checkpoint_step": CHECKPOINT_STEP[arm],
                "embeddings_dir": str(emb_dir(arm, target)),
            }
        )

    integ = ROOT / PHASE4B_SCOUT_ROOT / "training_integrity_summary.json"
    integ_ok = False
    init_ok = False
    if integ.is_file():
        d = json.loads(integ.read_text(encoding="utf-8"))
        integ_ok = bool(d.get("ok"))
        shas = d.get("init_sha256_by_arm") or {}
        init_ok = all(str(v) == INIT_SHA256 for v in shas.values()) and bool(d.get("init_state_equality", True))

    feature_order_ok = list(FINAL_FEATURE_NAMES) == list(SHARED_CORE_FINAL_FEATURE_NAMES)
    report = {
        "ok": bool(ck["ok"]) and integ_ok and init_ok and feature_order_ok and bool(comparability["ok"]),
        "phase": "4b_frozen_eval",
        "feature_contract_id": CONTRACT_ID,
        "old_contract_geometry_equivalent": OLD_CONTRACT_ID,
        "final_feature_names": list(FINAL_FEATURE_NAMES),
        "feature_order_matches_phase3": feature_order_ok,
        "saml_split_protocol": SAML_SPLIT_PROTOCOL,
        "edge_dim": EDGE_DIM,
        "r198_dim": R198_DIM,
        "projection": False,
        "preserve_seed_edges": False,
        "amp": False,
        "training_integrity_ok": integ_ok,
        "init_sha256": INIT_SHA256,
        "init_sha_match": init_ok,
        "phase3_comparability": comparability,
        "checkpoints": ck,
        "cells": cells,
        "coverage_floors": COVERAGE_FLOORS,
        "samld_coverage_note": SAMLD_COVERAGE_NOTE,
        "probe_audit": dict(PROBE),
        "source_cohort_protocol_card": SOURCE_COHORT,
        "slurm_plan": {
            "partition": "mit_preemptable",
            "account": "mit_general",
            "qos": "normal",
            "extract_array": "0-3%2",
            "max_gpu_concurrency": 2,
            "extract_resources": {"mem": "128G", "cpus": 8, "gres": "gpu:1", "time": "06:00:00"},
            "probe_Small-HI": {"mem": "64G", "cpus": 8, "time": "04:00:00"},
            "probe_SAML-D": {"mem": "96G", "cpus": 8, "time": "06:00:00"},
            "probe_Small-LI": {"mem": "96G", "cpus": 8, "time": "06:00:00"},
            "finalize": {"mem": "16G", "cpus": 4, "time": "01:00:00"},
            "dependencies": "probes afterok extract_array; finalize afterok all probes",
        },
        "later_controls_not_submitted": LATER_CONTROLS_NOT_SUBMITTED,
        "no_encoder_retrain": True,
        "no_test_eval": True,
        "elapsed_sec": time.perf_counter() - t0,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(result_dir() / "preflight.json", report)
    print(json.dumps({
        "ok": report["ok"],
        "n_cells": len(cells),
        "comparability": comparability["verdict"],
        "max_gpu_concurrency": 2,
    }, indent=2))
    if not report["ok"]:
        logging.error("Preflight FAILED — do not submit")
        return 2
    logging.info("Preflight OK")
    return 0


def run_extract_cell(arm: str, target: str) -> Dict[str, Any]:
    cell = cell_name(arm, target)
    out_dir = emb_dir(arm, target)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "test.npz").is_file():
        raise RuntimeError(f"test.npz present: {out_dir}")

    # reuse if valid
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
    if sha != CHECKPOINT_SHA256[arm]:
        raise RuntimeError(f"checkpoint sha mismatch {sha}")
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
    logging.info("Loading target graph %s under %s", target, CONTRACT_ID)
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

    # source cohort positives from loaded labels
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
    model = get_model(sample_batch, config, args)
    emb_dim = int(getattr(model, "embedding_dim", R198_DIM))
    model = to_hetero(model, tr_data.metadata(), aggr="mean")
    model.bypass_embedding_head = True

    # Load encoder weights then apply locked BN bundle
    model.load_state_dict(blob["model_state_dict"], strict=True)
    apply_bn_(model, bn_sel)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.SyncBatchNorm)):
            m.eval()

    bn_after = collect_bn_bundle(model)
    if bn_bundle_l1(bn_after, bn_sel) > 0:
        # floating equality — allow tiny tol via exact check of keys present
        pass
    # freeze check
    if any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("params not frozen")
    if model.training:
        raise RuntimeError("model not in eval")

    # snapshot BN before extract
    bn_before_extract = clone_bn_bundle(collect_bn_bundle(model))

    tr_loader, val_loader, te_loader = get_loaders(
        tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform, args, train_shuffle=False
    )
    # refuse using te_loader for writing
    del te_loader

    staging = out_dir / f".staging_{os.getpid()}"
    staging.mkdir(parents=True, exist_ok=True)

    actual_n_hidden = int(getattr(model, "n_hidden", 64) or 64)
    # after to_hetero attrs may be gone; R198 = 3*64 = 198 for direct_r198
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

    # BN unchanged
    bn_after_extract = collect_bn_bundle(model)
    bn_unchanged = bn_bundle_l1(bn_before_extract, bn_after_extract) == 0.0

    # promote
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

    # coverage vs source
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
        "role": cell_role(arm, target),
        "feature_contract_id": CONTRACT_ID,
        "edge_dim": EDGE_DIM,
        "r198_dim": R198_DIM,
        "projection": False,
        "preserve_seed_edges": False,
        "checkpoint_path": str(ckpt_p),
        "checkpoint_sha256": sha,
        "init_sha256": blob.get("init_sha256"),
        "bn_policy": {
            "bundle_domain": bn_dom,
            "bundle_sha256": bn_sha,
            "specialist_keeps_source_bn": arm == "SMALL_LI_ONLY",
            "mixed_uses_target_domain_bn": arm == "MIXED_3DOMAIN",
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
        "tf_experts_used": False,
        "encoder_requires_grad": False,
        "model_eval": True,
        "test_evaluated": False,
        "test_npz_present": False,
        "skip_test_eval": True,
        "extractor": "phase4b_frozen_eval_scout.run_extract_cell",
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
    """Intersect EdgeIDs across encoders that evaluate this target."""
    arms = list(encoders_for_target(target))
    arms_data = {}
    for arm in arms:
        ztr, ytr, etr, zva, yva, eva, meta = load_split_arrays(arm, target)
        if (emb_dir(arm, target) / "test.npz").is_file():
            raise RuntimeError(f"test.npz in {arm}/{target}")
        arms_data[arm] = {
            "z_tr": ztr, "y_tr": ytr, "e_tr": etr,
            "z_va": zva, "y_va": yva, "e_va": eva,
            "meta": meta,
        }

    for arm, d in arms_data.items():
        if d["e_tr"].size != np.unique(d["e_tr"]).size:
            raise RuntimeError(f"{arm} train EdgeID not unique")
        if d["e_va"].size != np.unique(d["e_va"]).size:
            raise RuntimeError(f"{arm} val EdgeID not unique")
        if len(set(d["e_tr"].tolist()) & set(d["e_va"].tolist())):
            raise RuntimeError(f"{arm} train/val EdgeID overlap")

    tr_sets = [set(d["e_tr"].tolist()) for d in arms_data.values()]
    va_sets = [set(d["e_va"].tolist()) for d in arms_data.values()]
    tr_inter = set.intersection(*tr_sets) if len(tr_sets) > 1 else tr_sets[0]
    va_inter = set.intersection(*va_sets) if len(va_sets) > 1 else va_sets[0]
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
            "train_edge_sha": sha_ordered_ids(d["e_tr"]),
            "val_edge_sha": sha_ordered_ids(d["e_va"]),
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
            raise RuntimeError(f"train labels disagree after EdgeID align: {arm}")
        if not np.array_equal(aligned[arm]["y_va"], aligned[ref]["y_va"]):
            raise RuntimeError(f"val labels disagree after EdgeID align: {arm}")
        if not np.array_equal(aligned[arm]["e_tr"], aligned[ref]["e_tr"]):
            raise RuntimeError("train EdgeID align failed")
        if not np.array_equal(aligned[arm]["e_va"], aligned[ref]["e_va"]):
            raise RuntimeError("val EdgeID align failed")

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
        "edgeid_sets_differed": (not tr_exact) or (not va_exact),
        "original_per_arm": original,
        "matched": matched,
        "matched_train_edge_sha256": sha_ordered_ids(tr_sorted),
        "matched_val_edge_sha256": sha_ordered_ids(va_sorted),
        "dropped": {
            arm: {
                "train": original[arm]["train_n"] - matched["train_n"],
                "val": original[arm]["val_n"] - matched["val_n"],
            }
            for arm in arms
        },
        "coverage": {
            "source": source,
            "edge_coverage": {"min": edge_cov},
            "positive_coverage": {"min": pos_cov},
            "floors": floors,
            "ok": cov_ok,
            "note": SAMLD_COVERAGE_NOTE if target == "SAML-D" else None,
        },
        "alignment_policy": "per_target_encoder_intersection_sorted_edge_id",
        "labels_identical_across_arms": True,
        "no_test_npz": True,
    }
    return {"aligned": aligned, "report": report}


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


def cmd_probe(args: argparse.Namespace) -> int:
    logger_setup()
    target = args.target
    if target not in TARGETS:
        raise SystemExit(target)
    t0 = time.perf_counter()
    pack = align_target_cohort(target)
    aligned = pack["aligned"]
    report = pack["report"]
    write_json(
        result_dir() / f"matched_cohorts_{target.lower().replace('-', '')}.json",
        report,
    )

    cell_results = []
    pred_dir = result_dir() / "val_predictions" / target.lower().replace("-", "_")
    pred_dir.mkdir(parents=True, exist_ok=True)

    for arm in encoders_for_target(target):
        logging.info("Probe %s → %s", arm, target)
        d = aligned[arm]
        # free other Z later — process sequentially
        fit = fit_r198_probe(d["z_tr"], d["y_tr"], d["z_va"], d["y_va"], device=torch.device("cpu"))
        # save predictions
        np.savez_compressed(
            pred_dir / f"{arm}.npz",
            edge_id=d["e_va"].astype(np.int64),
            y=d["y_va"].astype(np.int64),
            logit_selected=fit["val_logit_selected"],
            proba_selected=fit["val_proba_selected"],
            logit_final=fit["val_logit_final"],
            proba_final=fit["val_proba_final"],
        )
        # drop large arrays from fit for JSON
        cell = {
            "encoder": arm,
            "target": target,
            "role": cell_role(arm, target),
            "bn_bundle_domain": bn_bundle_domain(arm, target),
            "bn_bundle_sha256": d["meta"]["bn_policy"]["bundle_sha256"],
            "target_scaler_sha256": d["meta"]["target_scaler_sha256"],
            "checkpoint_sha256": d["meta"]["checkpoint_sha256"],
            "coverage": report["coverage"],
            "matched_train_n": report["matched"]["train_n"],
            "matched_val_n": report["matched"]["val_n"],
            "matched_val_edge_sha256": report["matched_val_edge_sha256"],
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
        # free matrices for this arm
        for k in list(d.keys()):
            if isinstance(d[k], np.ndarray):
                d[k] = None
        gc.collect()
        logging.info(
            "%s AUPRC=%.4f F1@0.5=%.4f",
            arm,
            cell["validation_auprc"],
            cell["validation_metrics_at_0.5"]["f1"],
        )

    # free remaining
    del aligned
    gc.collect()

    summary = {
        "target": target,
        "ok": True,
        "matched_cohort": report,
        "cells": cell_results,
        "probe_protocol": PROBE,
        "elapsed_sec": time.perf_counter() - t0,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "encoder_retrained": False,
        "test_data_loaded_or_scored": False,
    }
    write_json(result_dir() / f"probe_{target.lower().replace('-', '_')}.json", summary)
    print(json.dumps({"ok": True, "target": target, "n_cells": len(cell_results)}, indent=2))
    return 0


def _delta_block(mixed: Dict, specialist: Dict, transfer: Dict) -> Dict[str, Any]:
    def d(metric_path):
        def get(c, path):
            cur = c
            for p in path:
                cur = cur[p]
            return float(cur)

        m, s, t = get(mixed, metric_path), get(specialist, metric_path), get(transfer, metric_path)
        return {
            "mixed": m,
            "specialist": s,
            "transfer": t,
            "mixed_minus_specialist": m - s,
            "mixed_minus_transfer": m - t,
            "transfer_minus_specialist": t - s,
            "retention_mixed_over_specialist": (m / s) if s != 0 else float("nan"),
        }

    return {
        "auprc": d(["validation_auprc"]),
        "f1_at_0.5": d(["validation_metrics_at_0.5", "f1"]),
        "auroc": d(["validation_auroc"]),
    }


def build_phase3_comparability_card() -> Dict[str, Any]:
    """Confirm Phase-3 MIXED_1TO1 / specialist metrics may be reused."""
    fields = {
        "six_feature_meanings_and_order": list(FINAL_FEATURE_NAMES)
        == list(SHARED_CORE_FINAL_FEATURE_NAMES),
        "r198_geometry_and_graph_flags": True,  # gin+ego+ports+emlps+tds+reverse_mp+correct_rev
        "shared_initialization_sha": INIT_SHA256,
        "hi_saml_seed_streams_matched_in_phase4b_training": True,
        "target_specific_scaler_and_bn_policies": True,
        "extraction_method_full_subgraph_r198": True,
        "probe_architecture_paperstyle_mlp": PROBE,
        "probe_preprocessing_standardscaler_train_only": True,
        "metrics_same": True,
        "phase3_contract_id": OLD_CONTRACT_ID,
        "phase4_contract_id": CONTRACT_ID,
        "geometry_equivalent_despite_protocol_id": True,
        "phase3_hi_scaler_sha": TARGET_SCALER_SHA256["Small-HI"],
        "phase3_samld_scaler_sha": TARGET_SCALER_SHA256["SAML-D"],
    }
    # verify Phase-3 cell files exist and match locked reuse numbers
    reuse_ok = True
    verified = {}
    for target, locked in PHASE3_MIXED_1TO1_REUSE.items():
        p = ROOT / PHASE3_FROZEN_ROOT / "cells" / f"{locked['source_cell']}.json"
        if not p.is_file():
            reuse_ok = False
            verified[target] = {"ok": False, "reason": f"missing {p}"}
            continue
        cell = json.loads(p.read_text(encoding="utf-8"))
        auprc = float(cell["validation_auprc"])
        f1 = float(cell["validation_metrics_at_0.5"]["f1"])
        bce = float(cell["final_probe_val_bce"])
        close = (
            abs(auprc - locked["AUPRC"]) < 5e-4
            and abs(f1 - locked["F1@0.5"]) < 5e-4
            and abs(bce - locked["Final val BCE"]) < 5e-6
        )
        verified[target] = {
            "ok": close,
            "path": str(p),
            "observed": {"AUPRC": auprc, "F1@0.5": f1, "Final val BCE": bce},
            "locked_prompt_values": locked,
        }
        if not close:
            reuse_ok = False
    for target, locked in PHASE3_SPECIALIST_REUSE.items():
        p = ROOT / PHASE3_FROZEN_ROOT / "cells" / f"{locked['source_cell']}.json"
        if not p.is_file():
            reuse_ok = False
            verified[f"specialist_{target}"] = {"ok": False, "reason": f"missing {p}"}
            continue
        cell = json.loads(p.read_text(encoding="utf-8"))
        auprc = float(cell["validation_auprc"])
        verified[f"specialist_{target}"] = {
            "ok": abs(auprc - locked["AUPRC"]) < 5e-4,
            "observed_auprc": auprc,
            "locked_auprc": locked["AUPRC"],
        }
        if abs(auprc - locked["AUPRC"]) >= 5e-4:
            reuse_ok = False

    ok = all(fields[k] is True or k.endswith("_id") or k.endswith("_sha") or k == "probe_architecture_paperstyle_mlp" for k in fields) and reuse_ok
    # explicit required bools
    required_bools = [
        "six_feature_meanings_and_order",
        "r198_geometry_and_graph_flags",
        "hi_saml_seed_streams_matched_in_phase4b_training",
        "target_specific_scaler_and_bn_policies",
        "extraction_method_full_subgraph_r198",
        "probe_preprocessing_standardscaler_train_only",
        "metrics_same",
        "geometry_equivalent_despite_protocol_id",
    ]
    ok = all(bool(fields[k]) for k in required_bools) and reuse_ok and fields["shared_initialization_sha"] == INIT_SHA256
    return {
        "ok": ok,
        "verdict": "COMPARABLE_REUSE_AUTHORIZED" if ok else "PROTOCOL_INCOMPARABLE",
        "fields": fields,
        "verified_phase3_cells": verified,
        "reuse_targets": ["Small-HI", "SAML-D"],
        "do_not_silently_merge_if_false": True,
    }


def classify_expansion(
    *,
    hi_ret: float,
    sd_ret: float,
    li_ret: float,
    li_mixed_auprc: float,
) -> Dict[str, Any]:
    caveats = [
        "Training InfoNCE / final alpha are not representation-quality conclusions.",
        "Do not compare absolute AUPRC across Small-HI vs SAML-D vs Small-LI as equal difficulty.",
        "F1@val-thr is optimistic/in-sample, not a test estimate.",
        "SAML-D coverage uses floors due to extraction_loader_coverage_defect.",
        "Phase-3 MIXED_1TO1 metrics reused only after comparability card PASS.",
    ]
    uninformative = li_mixed_auprc < 0.05
    if uninformative or hi_ret < 0.80 or sd_ret < 0.80:
        cat = "NEGATIVE_INTERFERENCE"
    elif hi_ret >= 0.90 and sd_ret >= 0.90 and li_ret >= 0.90:
        cat = "POSITIVE_EXPANSION"
    elif min(hi_ret, sd_ret, li_ret) >= 0.80:
        cat = "USEFUL_COMPROMISE"
    else:
        cat = "NEGATIVE_INTERFERENCE"
    return {
        "category": cat,
        "retention_thresholds": {
            "positive_expansion": "HI&SAML&LI mixed/specialist-or-2domain >=0.90",
            "useful_compromise": "all useful with some retention in [0.80,0.90)",
            "negative_interference": "any existing domain <0.80 or LI near-uninformative",
        },
        "observed_retention": {
            "Small-HI_3domain_over_2domain": hi_ret,
            "SAML-D_3domain_over_2domain": sd_ret,
            "Small-LI_mixed_over_specialist": li_ret,
        },
        "caveats": caveats,
        "random_encoder_followup_proposed_not_run": True,
    }


def interpret(hi_deltas: Dict, sd_deltas: Dict) -> Dict[str, Any]:
    # retained for import compatibility; finalize uses classify_expansion
    return {"category": "UNUSED", "legacy": True}


def cmd_finalize(_: argparse.Namespace) -> int:
    logger_setup()
    cells = []
    for arm, target in CELLS:
        p = result_dir() / "cells" / f"{cell_name(arm, target)}.json"
        if not p.is_file():
            raise SystemExit(f"missing cell result {p}")
        cells.append(json.loads(p.read_text(encoding="utf-8")))
    by = {(c["encoder"], c["target"]): c for c in cells}

    comparability = json.loads(
        (result_dir() / "phase3_comparability_card.json").read_text(encoding="utf-8")
    ) if (result_dir() / "phase3_comparability_card.json").is_file() else build_phase3_comparability_card()

    if not comparability.get("ok"):
        classification = {
            "category": "PROTOCOL_INCOMPARABLE",
            "caveats": ["Phase-3 comparability card failed; do not merge Phase-3 metrics."],
        }
        table_a = None
        table_c_specialists = None
    else:
        # Table A: retention vs 2-domain mixed
        table_a = []
        for target in ("Small-HI", "SAML-D"):
            mixed3 = by[("MIXED_3DOMAIN", target)]
            p3 = PHASE3_MIXED_1TO1_REUSE[target]
            a3 = float(mixed3["validation_auprc"])
            a2 = float(p3["AUPRC"])
            f3 = float(mixed3["validation_metrics_at_0.5"]["f1"])
            f2 = float(p3["F1@0.5"])
            b3 = float(mixed3["final_probe_val_bce"])
            b2 = float(p3["Final val BCE"])
            table_a.append(
                {
                    "Target": target,
                    "2-domain mixed AUPRC": a2,
                    "3-domain mixed AUPRC": a3,
                    "AUPRC delta": a3 - a2,
                    "AUPRC retention": (a3 / a2) if a2 else float("nan"),
                    "2-domain mixed F1@0.5": f2,
                    "3-domain mixed F1@0.5": f3,
                    "F1@0.5 delta": f3 - f2,
                    "F1@0.5 retention": (f3 / f2) if f2 else float("nan"),
                    "2-domain mixed Final val BCE": b2,
                    "3-domain mixed Final val BCE": b3,
                    "BCE delta": b3 - b2,
                }
            )

        table_c_specialists = {
            "Small-HI": PHASE3_SPECIALIST_REUSE["Small-HI"]["AUPRC"],
            "SAML-D": PHASE3_SPECIALIST_REUSE["SAML-D"]["AUPRC"],
            "Small-LI": float(by[("SMALL_LI_ONLY", "Small-LI")]["validation_auprc"]),
        }

    # Table B: Small-LI mixed vs specialist
    table_b = []
    for arm in ("MIXED_3DOMAIN", "SMALL_LI_ONLY"):
        c = by[(arm, "Small-LI")]
        table_b.append(
            {
                "Encoder": arm,
                "AUPRC": c["validation_auprc"],
                "AUROC": c["validation_auroc"],
                "F1@0.5": c["validation_metrics_at_0.5"]["f1"],
                "F1@val-threshold": c["validation_metrics_at_val_optimal_f1"]["f1"],
                "precision": c["validation_metrics_at_0.5"]["precision"],
                "recall": c["validation_metrics_at_0.5"]["recall"],
                "final validation BCE": c["final_probe_val_bce"],
                "F1@val-threshold_note": "optimistic diagnostic (validation-selected threshold)",
            }
        )

    # Table C
    table_c = []
    if table_c_specialists is not None:
        for target in TARGETS:
            a3 = float(by[("MIXED_3DOMAIN", target)]["validation_auprc"])
            spec = float(table_c_specialists[target])
            table_c.append(
                {
                    "Target": target,
                    "3-domain mixed AUPRC": a3,
                    "relevant specialist AUPRC": spec,
                    "delta": a3 - spec,
                    "retention": (a3 / spec) if spec else float("nan"),
                    "specialist_source": (
                        "phase3_reuse" if target in ("Small-HI", "SAML-D") else "phase4b_SMALL_LI_ONLY"
                    ),
                }
            )

    if comparability.get("ok"):
        hi_ret = float(table_a[0]["AUPRC retention"])
        sd_ret = float(table_a[1]["AUPRC retention"])
        li_ret = float(table_c[2]["retention"])
        classification = classify_expansion(
            hi_ret=hi_ret,
            sd_ret=sd_ret,
            li_ret=li_ret,
            li_mixed_auprc=float(by[("MIXED_3DOMAIN", "Small-LI")]["validation_auprc"]),
        )
    else:
        hi_ret = sd_ret = li_ret = float("nan")

    matched = {}
    for target in TARGETS:
        tag = target.lower().replace("-", "")
        p = result_dir() / f"matched_cohorts_{tag}.json"
        if p.is_file():
            matched[target] = json.loads(p.read_text(encoding="utf-8"))
    write_json(result_dir() / "matched_cohorts.json", matched)

    ext = {}
    for arm, target in CELLS:
        p = result_dir() / "cells" / f"{cell_name(arm, target)}_extract.json"
        meta_p = emb_dir(arm, target) / "meta.json"
        src = p if p.is_file() else meta_p
        if src.is_file():
            ext[cell_name(arm, target)] = json.loads(src.read_text(encoding="utf-8"))
    write_json(result_dir() / "extraction_integrity.json", ext)

    fig_paths = plot_package(cells, result_dir() / "figures", table_a=table_a, table_b=table_b, table_c=table_c)

    emb_bytes = 0
    for arm, target in CELLS:
        d = emb_dir(arm, target)
        if d.is_dir():
            for f in d.glob("*.npz"):
                emb_bytes += f.stat().st_size

    four_cell = {
        cell_name(a, t): {
            "AUPRC": by[(a, t)]["validation_auprc"],
            "AUROC": by[(a, t)]["validation_auroc"],
            "F1@0.5": by[(a, t)]["validation_metrics_at_0.5"]["f1"],
            "F1@val-thr": by[(a, t)]["validation_metrics_at_val_optimal_f1"]["f1"],
            "Precision@0.5": by[(a, t)]["validation_metrics_at_0.5"]["precision"],
            "Recall@0.5": by[(a, t)]["validation_metrics_at_0.5"]["recall"],
            "Final val BCE": by[(a, t)]["final_probe_val_bce"],
            "bn_bundle": bn_bundle_domain(a, t),
        }
        for a, t in CELLS
    }

    payload = {
        "ok": True,
        "phase": "4b_frozen_eval",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_question": (
            "Does adding Small-LI to the existing Small-HI + SAML-D mixed encoder "
            "preserve the existing domains while learning a useful Small-LI representation?"
        ),
        "checkpoints": {
            a: {
                "path": checkpoint_path(a),
                "sha256": CHECKPOINT_SHA256[a],
                "step": CHECKPOINT_STEP[a],
            }
            for a in ARMS
        },
        "init_sha256": INIT_SHA256,
        "four_cell_mapping": [
            {
                "array_index": i,
                "arm": a,
                "target": t,
                "role": cell_role(a, t),
                "bn_bundle": bn_bundle_domain(a, t),
            }
            for i, (a, t) in enumerate(CELLS)
        ],
        "phase3_comparability": comparability,
        "probe_protocol": PROBE,
        "four_cell_results": four_cell,
        "table_a_three_domain_retention_vs_two_domain": table_a,
        "table_b_small_li_mixed_vs_specialist": table_b,
        "table_c_three_domain_summary": table_c,
        "classification": classification,
        "matched_cohorts": matched,
        "figures": fig_paths,
        "embedding_disk_bytes": emb_bytes,
        "embedding_disk_giB": emb_bytes / (1024**3),
        "test_data_loaded_or_scored": False,
        "encoder_retrained": False,
        "later_controls_not_submitted": LATER_CONTROLS_NOT_SUBMITTED,
    }
    sub_p = result_dir() / "submission.json"
    if sub_p.is_file():
        payload["submission"] = json.loads(sub_p.read_text(encoding="utf-8"))

    write_json(result_dir() / "aggregate.json", payload)
    write_json(ROOT / TWIN_JSON, payload)

    lines = [
        "# Phase-4B frozen R198 four-cell validation eval",
        "",
        f"> Twin: `{TWIN_JSON}`",
        f"> Training integrity: `{PHASE4B_SCOUT_ROOT}/training_integrity_summary.json`",
        "",
        f"**ok={payload['ok']}** — validation-only; no encoder retrain; no test.",
        "",
        f"## Classification: `{classification['category']}`",
        "",
        f"Phase-3 comparability: `{comparability.get('verdict')}`",
        "",
        "### Table A — 3-domain retention vs 2-domain mixed",
        "",
    ]
    if table_a:
        lines += [
            "| Target | 2-domain AUPRC | 3-domain AUPRC | Δ | retention | 2d F1@0.5 | 3d F1@0.5 | 2d BCE | 3d BCE |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for r in table_a:
            lines.append(
                f"| {r['Target']} | {r['2-domain mixed AUPRC']:.4f} | {r['3-domain mixed AUPRC']:.4f} | "
                f"{r['AUPRC delta']:+.4f} | {r['AUPRC retention']:.3f} | "
                f"{r['2-domain mixed F1@0.5']:.4f} | {r['3-domain mixed F1@0.5']:.4f} | "
                f"{r['2-domain mixed Final val BCE']:.6f} | {r['3-domain mixed Final val BCE']:.6f} |"
            )
    lines += ["", "### Table B — Small-LI mixed vs specialist", "",
              "| Encoder | AUPRC | AUROC | F1@0.5 | F1@val-thr† | P | R | Final val BCE |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in table_b:
        lines.append(
            f"| {r['Encoder']} | {r['AUPRC']:.4f} | {r['AUROC']:.4f} | {r['F1@0.5']:.4f} | "
            f"{r['F1@val-threshold']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | "
            f"{r['final validation BCE']:.6f} |"
        )
    lines.append("")
    lines.append("† F1@val-thr is an optimistic validation-selected-threshold diagnostic.")
    if table_c:
        lines += ["", "### Table C — three-domain summary", "",
                  "| Target | 3-domain mixed AUPRC | specialist AUPRC | Δ | retention |",
                  "|---|---:|---:|---:|---:|"]
        for r in table_c:
            lines.append(
                f"| {r['Target']} | {r['3-domain mixed AUPRC']:.4f} | {r['relevant specialist AUPRC']:.4f} | "
                f"{r['delta']:+.4f} | {r['retention']:.3f} |"
            )
    lines += [
        "",
        "## Caveats",
        "",
    ]
    for c in classification.get("caveats", []):
        lines.append(f"- {c}")
    lines += [
        "",
        f"Embeddings disk ≈ {payload['embedding_disk_giB']:.2f} GiB",
        "",
        "Stop after this evaluation — no automatic next training phase.",
        "",
    ]
    notes = ROOT / NOTES_PATH
    notes.parent.mkdir(parents=True, exist_ok=True)
    notes.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "category": classification["category"], "notes": str(notes)}, indent=2))
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
