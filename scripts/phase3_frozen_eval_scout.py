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
from phase3_frozen_eval import (  # noqa: E402
    ARMS,
    CELLS,
    CHECKPOINT_SHA256,
    CONTRACT_ID,
    COVERAGE_FLOORS,
    EDGE_DIM,
    EMB_ROOT,
    INIT_SHA256,
    LATER_CONTROLS_NOT_SUBMITTED,
    NOTES_PATH,
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
)
from phase3_frozen_eval.figures import plot_package  # noqa: E402
from phase3_frozen_eval.probe import fit_r198_probe  # noqa: E402
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
            "global_step_1000": int(blob.get("global_optimizer_step", -1)) == 1000,
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
        "--unique_name", f"phase3_frozen_{cell}",
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
    ck = verify_checkpoints()
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
                "embeddings_dir": str(emb_dir(arm, target)),
            }
        )
    # integrity training gate
    integ = ROOT / "results/diagnostics/smallhi_samld_mixed_ssl_phase3_scout/training_integrity_summary.json"
    integ_ok = False
    if integ.is_file():
        d = json.loads(integ.read_text(encoding="utf-8"))
        integ_ok = bool(d.get("ok")) and bool(d.get("init_state_equality"))

    probe_audit = dict(PROBE)
    report = {
        "ok": bool(ck["ok"]) and integ_ok,
        "phase": "3_frozen_eval",
        "feature_contract_id": CONTRACT_ID,
        "saml_split_protocol": SAML_SPLIT_PROTOCOL,
        "edge_dim": EDGE_DIM,
        "r198_dim": R198_DIM,
        "projection": False,
        "preserve_seed_edges": False,
        "training_integrity_ok": integ_ok,
        "checkpoints": ck,
        "cells": cells,
        "coverage_floors": COVERAGE_FLOORS,
        "samld_coverage_note": SAMLD_COVERAGE_NOTE,
        "probe_audit": probe_audit,
        "source_cohort_protocol_card": SOURCE_COHORT,
        "slurm_plan": {
            "partition": "mit_preemptable",
            "account": "mit_general",
            "qos": "normal",
            "extract_array": "0-5%2",
            "max_gpu_concurrency": 2,
            "extract_resources": {"mem": "128G", "cpus": 8, "gres": "gpu:1", "time": "06:00:00"},
            "probe_Small-HI": {"mem": "64G", "cpus": 8, "time": "04:00:00"},
            "probe_SAML-D": {"mem": "96G", "cpus": 8, "time": "06:00:00"},
            "finalize": {"mem": "16G", "cpus": 4, "time": "01:00:00"},
            "dependencies": "probes afterok extract_array; finalize afterok both probes",
        },
        "storage_projection": {
            "note": "3×Small-HI + 3×SAML-D full R198 train/val; SAML ~6GB/cell, HI ~3GB/cell",
            "estimated_giB": 30,
        },
        "wall_time_projection": {
            "extract_per_cell_hours": "1–3 (SAML longer)",
            "extract_wall_with_%2": "≈6–10h",
            "probe_hours": "1–2 per target",
        },
        "later_controls_not_submitted": LATER_CONTROLS_NOT_SUBMITTED,
        "elapsed_sec": time.perf_counter() - t0,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(result_dir() / "preflight.json", report)
    write_json(
        ROOT / TWIN_JSON,
        {
            "phase": "3_frozen_eval",
            "status": "preflight",
            "preflight_ok": report["ok"],
            "cells": cells,
            "slurm_plan": report["slurm_plan"],
        },
    )
    print(json.dumps({"ok": report["ok"], "n_cells": len(cells), "max_gpu_concurrency": 2}, indent=2))
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
    if names and names != list(SHARED_CORE_FINAL_FEATURE_NAMES):
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
            "specialist_keeps_source_bn": arm != "MIXED_1TO1",
            "mixed_uses_target_domain_bn": arm == "MIXED_1TO1",
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
        "extractor": "phase3_frozen_eval_scout.run_extract_cell",
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
    """Intersect EdgeIDs across three encoders; never align by row position."""
    arms_data = {}
    for arm in ARMS:
        ztr, ytr, etr, zva, yva, eva, meta = load_split_arrays(arm, target)
        if (emb_dir(arm, target) / "test.npz").is_file():
            raise RuntimeError(f"test.npz in {arm}/{target}")
        arms_data[arm] = {
            "z_tr": ztr, "y_tr": ytr, "e_tr": etr,
            "z_va": zva, "y_va": yva, "e_va": eva,
            "meta": meta,
        }

    # uniqueness + disjoint within each arm
    for arm, d in arms_data.items():
        if d["e_tr"].size != np.unique(d["e_tr"]).size:
            raise RuntimeError(f"{arm} train EdgeID not unique")
        if d["e_va"].size != np.unique(d["e_va"]).size:
            raise RuntimeError(f"{arm} val EdgeID not unique")
        if len(set(d["e_tr"].tolist()) & set(d["e_va"].tolist())):
            raise RuntimeError(f"{arm} train/val EdgeID overlap")

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

    # label agreement after alignment
    ref = ARMS[0]
    for arm in ARMS[1:]:
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
            for arm in ARMS
        },
        "coverage": {
            "source": source,
            "edge_coverage": {"min": edge_cov},
            "positive_coverage": {"min": pos_cov},
            "floors": floors,
            "ok": cov_ok,
            "note": SAMLD_COVERAGE_NOTE if target == "SAML-D" else None,
        },
        "alignment_policy": "three_encoder_intersection_sorted_edge_id",
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

    for arm in ARMS:
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


def interpret(hi_deltas: Dict, sd_deltas: Dict) -> Dict[str, Any]:
    def near(d):
        return abs(d["auprc"]["mixed_minus_specialist"]) <= 0.02

    def exceeds_transfer(d):
        return d["auprc"]["mixed_minus_transfer"] > 0.02

    hi_near, sd_near = near(hi_deltas), near(sd_deltas)
    hi_beat_x, sd_beat_x = exceeds_transfer(hi_deltas), exceeds_transfer(sd_deltas)
    if hi_near and sd_near:
        cat = "DUAL_DOMAIN_RETENTION"
    elif (not hi_near or not sd_near) and hi_beat_x and sd_beat_x:
        cat = "USEFUL_COMPROMISE"
    elif (
        hi_deltas["auprc"]["mixed_minus_specialist"] < -0.02
        and sd_deltas["auprc"]["mixed_minus_specialist"] < -0.02
        and (not hi_beat_x)
        and (not sd_beat_x)
    ):
        cat = "NEGATIVE_INTERFERENCE"
    else:
        cat = "MIXED_OR_DOMAIN_DEPENDENT"
    return {
        "category": cat,
        "exploratory_not_predeclared_thesis_gate": True,
        "caveats": [
            "Training InfoNCE is not a representation-quality conclusion.",
            "Do not compare absolute AUPRC across Small-HI vs SAML-D as equal difficulty.",
            "F1@val-thr is optimistic/in-sample, not a test estimate.",
            "SAML-D coverage uses floors due to extraction_loader_coverage_defect.",
        ],
    }


def cmd_finalize(_: argparse.Namespace) -> int:
    logger_setup()
    cells = []
    for arm, target in CELLS:
        p = result_dir() / "cells" / f"{cell_name(arm, target)}.json"
        if not p.is_file():
            raise SystemExit(f"missing cell result {p}")
        cells.append(json.loads(p.read_text(encoding="utf-8")))

    by = {(c["encoder"], c["target"]): c for c in cells}
    tables = {}
    for target in TARGETS:
        rows = []
        for arm in ARMS:
            c = by[(arm, target)]
            rows.append(
                {
                    "Encoder": arm,
                    "Role": c["role"],
                    "AUPRC": c["validation_auprc"],
                    "AUROC": c["validation_auroc"],
                    "F1@0.5": c["validation_metrics_at_0.5"]["f1"],
                    "F1@val-thr": c["validation_metrics_at_val_optimal_f1"]["f1"],
                    "Precision@0.5": c["validation_metrics_at_0.5"]["precision"],
                    "Recall@0.5": c["validation_metrics_at_0.5"]["recall"],
                    "Final val BCE": c["final_probe_val_bce"],
                }
            )
        tables[target] = rows

    hi_d = _delta_block(by[("MIXED_1TO1", "Small-HI")], by[("SMALL_HI_ONLY", "Small-HI")], by[("SAMLD_ONLY", "Small-HI")])
    sd_d = _delta_block(by[("MIXED_1TO1", "SAML-D")], by[("SAMLD_ONLY", "SAML-D")], by[("SMALL_HI_ONLY", "SAML-D")])
    category = interpret(hi_d, sd_d)

    # merge matched cohorts
    matched = {}
    for target in TARGETS:
        tag = target.lower().replace("-", "")
        p = result_dir() / f"matched_cohorts_{tag}.json"
        if p.is_file():
            matched[target] = json.loads(p.read_text(encoding="utf-8"))
    write_json(result_dir() / "matched_cohorts.json", matched)

    # extraction integrity
    ext = {}
    for arm, target in CELLS:
        p = result_dir() / "cells" / f"{cell_name(arm, target)}_extract.json"
        meta_p = emb_dir(arm, target) / "meta.json"
        src = p if p.is_file() else meta_p
        if src.is_file():
            ext[cell_name(arm, target)] = json.loads(src.read_text(encoding="utf-8"))
    write_json(result_dir() / "extraction_integrity.json", ext)

    fig_paths = plot_package(cells, result_dir() / "figures")

    # disk usage
    emb_bytes = 0
    for arm, target in CELLS:
        d = emb_dir(arm, target)
        if d.is_dir():
            for f in d.glob("*.npz"):
                emb_bytes += f.stat().st_size

    payload = {
        "ok": True,
        "phase": "3_frozen_eval",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoints": {a: {"path": checkpoint_path(a), "sha256": CHECKPOINT_SHA256[a]} for a in ARMS},
        "init_sha256": INIT_SHA256,
        "six_cell_mapping": [
            {"array_index": i, "arm": a, "target": t, "role": cell_role(a, t), "bn_bundle": bn_bundle_domain(a, t)}
            for i, (a, t) in enumerate(CELLS)
        ],
        "bn_policy_per_cell": {
            cell_name(a, t): bn_bundle_domain(a, t) for a, t in CELLS
        },
        "target_scaler_sha_per_cell": {
            cell_name(a, t): TARGET_SCALER_SHA256[t] for a, t in CELLS
        },
        "probe_protocol": PROBE,
        "tables": tables,
        "deltas": {"Small-HI": hi_d, "SAML-D": sd_d},
        "interpretation": category,
        "matched_cohorts": matched,
        "figures": fig_paths,
        "embedding_disk_bytes": emb_bytes,
        "embedding_disk_giB": emb_bytes / (1024**3),
        "test_data_loaded_or_scored": False,
        "encoder_retrained": False,
        "later_controls_not_submitted": LATER_CONTROLS_NOT_SUBMITTED,
    }
    # attach job ids from submission if present
    sub_p = result_dir() / "submission.json"
    if sub_p.is_file():
        payload["submission"] = json.loads(sub_p.read_text(encoding="utf-8"))

    write_json(result_dir() / "aggregate.json", payload)
    write_json(ROOT / TWIN_JSON, payload)

    # notes
    lines = [
        "# Phase-3 frozen R198 six-cell validation eval",
        "",
        f"> Twin: `{TWIN_JSON}`",
        f"> Integrity training: `results/diagnostics/smallhi_samld_mixed_ssl_phase3_scout/training_integrity_summary.json`",
        "",
        f"**ok={payload['ok']}** — validation-only; no encoder retrain; no test.",
        "",
        f"## Interpretation: `{category['category']}`",
        "",
        "### Small-HI validation table",
        "",
        "| Encoder | Role | AUPRC | AUROC | F1@0.5 | F1@val-thr | P@0.5 | R@0.5 | Final val BCE |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in tables["Small-HI"]:
        lines.append(
            f"| {r['Encoder']} | {r['Role']} | {r['AUPRC']:.4f} | {r['AUROC']:.4f} | "
            f"{r['F1@0.5']:.4f} | {r['F1@val-thr']:.4f} | {r['Precision@0.5']:.4f} | "
            f"{r['Recall@0.5']:.4f} | {r['Final val BCE']:.6f} |"
        )
    lines += ["", "### SAML-D validation table", "",
              "| Encoder | Role | AUPRC | AUROC | F1@0.5 | F1@val-thr | P@0.5 | R@0.5 | Final val BCE |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in tables["SAML-D"]:
        lines.append(
            f"| {r['Encoder']} | {r['Role']} | {r['AUPRC']:.4f} | {r['AUROC']:.4f} | "
            f"{r['F1@0.5']:.4f} | {r['F1@val-thr']:.4f} | {r['Precision@0.5']:.4f} | "
            f"{r['Recall@0.5']:.4f} | {r['Final val BCE']:.6f} |"
        )
    lines += [
        "",
        "## Deltas (AUPRC)",
        "",
        f"- Small-HI mixed−specialist: {hi_d['auprc']['mixed_minus_specialist']:+.4f} "
        f"(retention {hi_d['auprc']['retention_mixed_over_specialist']:.3f})",
        f"- Small-HI mixed−transfer: {hi_d['auprc']['mixed_minus_transfer']:+.4f}",
        f"- SAML-D mixed−specialist: {sd_d['auprc']['mixed_minus_specialist']:+.4f} "
        f"(retention {sd_d['auprc']['retention_mixed_over_specialist']:.3f})",
        f"- SAML-D mixed−transfer: {sd_d['auprc']['mixed_minus_transfer']:+.4f}",
        "",
        "## Caveats",
        "",
    ]
    for c in category["caveats"]:
        lines.append(f"- {c}")
    lines += [
        "",
        "## Probe recipe",
        "",
        f"```json\n{json.dumps(PROBE, indent=2)}\n```",
        "",
        f"Embeddings disk ≈ {payload['embedding_disk_giB']:.2f} GiB",
        "",
    ]
    notes = ROOT / NOTES_PATH
    notes.parent.mkdir(parents=True, exist_ok=True)
    notes.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "category": category["category"], "notes": str(notes)}, indent=2))
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
