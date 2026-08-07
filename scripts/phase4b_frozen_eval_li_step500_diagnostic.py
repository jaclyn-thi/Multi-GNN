#!/usr/bin/env python3
"""Phase-4B addendum: SMALL_LI_ONLY step-500 → Small-LI frozen probe diagnostic.

Does not overwrite existing step-1000 or MIXED cells.
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
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch_geometric.nn import to_hetero

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_loading import get_data  # noqa: E402
from embedding_extraction import (  # noqa: E402
    expected_seed_edge_ids,
    extract_seed_embeddings_hetero,
    get_loaders,
    log_seed_coverage,
    save_embedding_split_npz,
)
from mixed_ssl_phase2.bn import (  # noqa: E402
    apply_bn_,
    bn_bundle_l1,
    clone_bn_bundle,
    collect_bn_bundle,
)
from phase4b_frozen_eval import (  # noqa: E402
    CONTRACT_ID,
    COVERAGE_FLOORS,
    EDGE_DIM,
    EMB_ROOT,
    FINAL_FEATURE_NAMES,
    INIT_SHA256,
    PROBE,
    R198_DIM,
    RESULT_ROOT,
    SEED,
    TARGET_SCALER_SHA256,
)
from phase4b_frozen_eval.probe import fit_r198_probe  # noqa: E402
from shared_core_contract import SHARED_CORE_FINAL_FEATURE_NAMES  # noqa: E402
from train_util import AddEgoIds, FORWARD_EDGE_TYPE, add_arange_ids, extract_param  # noqa: E402
from training import get_model  # noqa: E402
from util import create_parser, logger_setup, set_seed  # noqa: E402

CELL = "SMALL_LI_ONLY_step500__smallli"
CKPT_REL = (
    "results/checkpoints/financial_multidataset_shared_core_phase4b_scout_seed2/"
    "small_li_only/checkpoint_step_0500.tar"
)
EXPECTED_CKPT_SHA = "4eb7c8b434b821e0e8ef324a1b30ec5bf99fa177f379f185e36ddcb3b47f11a9"
ADDENDUM_DIR = Path(RESULT_ROOT) / "addendum_li_step500"
EXISTING_LI_CELL = "SMALL_LI_ONLY__smallli"
MIXED_LI_CELL = "MIXED_3DOMAIN__smallli"


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
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


def emb_dir() -> Path:
    return ROOT / EMB_ROOT / CELL / "pre_embedding_3h"


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
    }


def make_extract_args() -> argparse.Namespace:
    argv = [
        "--data", "Small-LI",
        "--model", "gin",
        "--objective", "contrastive",
        "--unique_name", f"phase4b_frozen_{CELL}",
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
    ns.embeddings_subdir = CELL
    return ns


def run_extract() -> Dict[str, Any]:
    out_dir = emb_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    # Refuse colliding with existing cells
    for forbidden in (EXISTING_LI_CELL, MIXED_LI_CELL):
        if CELL == forbidden:
            raise RuntimeError("refusing to overwrite existing cell name")
        if (ROOT / EMB_ROOT / forbidden).resolve() == out_dir.resolve():
            raise RuntimeError("embedding path collision with existing cell")
    if (out_dir / "test.npz").is_file():
        raise RuntimeError("test.npz present")

    need, reused = [], []
    for s in ("train", "val"):
        st = validate_npz(out_dir / f"{s}.npz")
        if st.get("ok"):
            reused.append(s)
        else:
            need.append(s)
    meta_path = out_dir / "meta.json"
    if not need and meta_path.is_file():
        return {"status": "reuse", "meta": json.loads(meta_path.read_text(encoding="utf-8"))}

    ckpt_p = ROOT / CKPT_REL
    if not ckpt_p.is_file():
        raise FileNotFoundError(ckpt_p)
    sha = sha256_file(ckpt_p)
    if sha != EXPECTED_CKPT_SHA:
        raise RuntimeError(f"checkpoint sha {sha} != locked {EXPECTED_CKPT_SHA}")
    blob = torch.load(ckpt_p, map_location="cpu", weights_only=False)
    if int(blob.get("global_optimizer_step", -1)) != 500:
        raise RuntimeError(f"expected step 500, got {blob.get('global_optimizer_step')}")
    if blob.get("init_sha256") != INIT_SHA256:
        raise RuntimeError("init sha mismatch")
    if blob.get("feature_contract_id") != CONTRACT_ID:
        raise RuntimeError("contract mismatch")
    if blob.get("test_evaluated") is True:
        raise RuntimeError("test_evaluated true in checkpoint")
    if "Small-LI" not in blob["bn_bundles"]:
        raise RuntimeError("Small-LI BN missing")
    bn_sel = clone_bn_bundle(blob["bn_bundles"]["Small-LI"])
    bn_sha = bn_bundle_sha(bn_sel)

    with open(ROOT / "data_config.json", encoding="utf-8") as f:
        data_config = json.load(f)
    args = make_extract_args()
    set_seed(SEED)
    logging.info("Loading Small-LI under %s for step-500 diagnostic", CONTRACT_ID)
    tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(args, data_config)
    if int(te_inds.numel()) != 0:
        raise RuntimeError("te_inds nonempty — refuse test")
    ea = tr_data[FORWARD_EDGE_TYPE].edge_attr
    if int(ea.shape[1]) != EDGE_DIM:
        raise RuntimeError(f"edge_dim={ea.shape[1]}")
    names = list(getattr(args, "edge_feature_schema_names", []) or [])
    if names and names != list(FINAL_FEATURE_NAMES):
        raise RuntimeError(f"schema {names}")
    if list(SHARED_CORE_FINAL_FEATURE_NAMES) != list(FINAL_FEATURE_NAMES):
        raise RuntimeError("feature-order drift vs Phase-3")
    scaler = getattr(args, "shared_core_edge_scaler", None)
    if not isinstance(scaler, dict) or scaler.get("scaler_sha256") != TARGET_SCALER_SHA256["Small-LI"]:
        raise RuntimeError("Small-LI scaler sha mismatch")

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
    model.load_state_dict(blob["model_state_dict"], strict=True)
    apply_bn_(model, bn_sel)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.SyncBatchNorm)):
            m.eval()

    bn_before = clone_bn_bundle(collect_bn_bundle(model))
    tr_loader, val_loader, te_loader = get_loaders(
        tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform, args, train_shuffle=False
    )
    del te_loader

    staging = out_dir / f".staging_{os.getpid()}"
    staging.mkdir(parents=True, exist_ok=True)
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
                pre_dim=R198_DIM,
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

    bn_unchanged = bn_bundle_l1(bn_before, collect_bn_bundle(model)) == 0.0
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
    floors = COVERAGE_FLOORS["Small-LI"]
    edge_cov = min(
        extracted["train"]["n"] / max(source["train_n"], 1),
        extracted["val"]["n"] / max(source["val_n"], 1),
    )
    pos_cov = min(
        extracted["train"]["n_pos"] / max(source["train_pos"], 1),
        extracted["val"]["n_pos"] / max(source["val_pos"], 1),
    )
    cov_ok = edge_cov >= floors["edge"] and pos_cov >= floors["positive"]
    if not cov_ok:
        raise RuntimeError(f"coverage failed edge={edge_cov} pos={pos_cov}")

    meta = {
        "cell": CELL,
        "encoder_arm": "SMALL_LI_ONLY",
        "checkpoint_step": 500,
        "target_dataset": "Small-LI",
        "feature_contract_id": CONTRACT_ID,
        "edge_dim": EDGE_DIM,
        "r198_dim": R198_DIM,
        "checkpoint_path": str(ckpt_p),
        "checkpoint_sha256": sha,
        "init_sha256": INIT_SHA256,
        "bn_policy": {
            "bundle_domain": "Small-LI",
            "bundle_sha256": bn_sha,
            "bn_unchanged_during_extract": bn_unchanged,
        },
        "target_scaler_sha256": scaler["scaler_sha256"],
        "source_cohort": source,
        "extracted": extracted,
        "coverage": {
            "edge_coverage_min": edge_cov,
            "positive_coverage_min": pos_cov,
            "floors": floors,
            "ok": cov_ok,
        },
        "projection": False,
        "preserve_seed_edges": False,
        "test_evaluated": False,
        "encoder_retrained": False,
        "does_not_overwrite_existing_cells": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": os.environ.get("SLURM_JOB_ID"),
    }
    write_json(out_dir / "meta.json", meta)
    write_json(ROOT / RESULT_ROOT / "cells" / f"{CELL}_extract.json", meta)
    write_json(ADDENDUM_DIR / "extract_meta.json", meta)
    shutil.rmtree(staging, ignore_errors=True)
    return {"status": "extracted", "meta": meta}


def load_npz_split(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = np.load(path)
    return (
        np.asarray(d["Z"], dtype=np.float32),
        np.asarray(d["y"]).reshape(-1).astype(np.int64),
        np.asarray(d["edge_id"]).reshape(-1).astype(np.int64),
    )


def align_to_existing_cohort() -> Dict[str, Any]:
    """Match EdgeIDs to the completed Phase-4B Small-LI matched cohort."""
    matched_path = ROOT / RESULT_ROOT / "matched_cohorts_smallli.json"
    if not matched_path.is_file():
        raise FileNotFoundError(matched_path)
    matched_report = json.loads(matched_path.read_text(encoding="utf-8"))

    # Use existing step-1000 embeddings as cohort reference (exact match to mixed LI)
    ref_dir = ROOT / EMB_ROOT / EXISTING_LI_CELL / "pre_embedding_3h"
    ztr_r, ytr_r, etr_r = load_npz_split(ref_dir / "train.npz")
    zva_r, yva_r, eva_r = load_npz_split(ref_dir / "val.npz")

    new_dir = emb_dir()
    ztr, ytr, etr = load_npz_split(new_dir / "train.npz")
    zva, yva, eva = load_npz_split(new_dir / "val.npz")

    for name, e in (("ref_tr", etr_r), ("ref_va", eva_r), ("new_tr", etr), ("new_va", eva)):
        if e.size != np.unique(e).size:
            raise RuntimeError(f"{name} EdgeID not unique")
    if len(set(etr.tolist()) & set(eva.tolist())):
        raise RuntimeError("new train/val EdgeID overlap")
    if len(set(etr_r.tolist()) & set(eva_r.tolist())):
        raise RuntimeError("ref train/val EdgeID overlap")

    tr_inter = set(etr.tolist()) & set(etr_r.tolist())
    va_inter = set(eva.tolist()) & set(eva_r.tolist())
    tr_sorted = np.array(sorted(tr_inter), dtype=np.int64)
    va_sorted = np.array(sorted(va_inter), dtype=np.int64)

    def subset(z, y, e, order):
        m = {int(x): i for i, x in enumerate(e)}
        idx = np.array([m[int(x)] for x in order], dtype=np.int64)
        return z[idx], y[idx], order.copy()

    ztr_a, ytr_a, etr_a = subset(ztr, ytr, etr, tr_sorted)
    zva_a, yva_a, eva_a = subset(zva, yva, eva, va_sorted)
    _, ytr_ref, _ = subset(ztr_r, ytr_r, etr_r, tr_sorted)
    _, yva_ref, _ = subset(zva_r, yva_r, eva_r, va_sorted)
    if not np.array_equal(ytr_a, ytr_ref) or not np.array_equal(yva_a, yva_ref):
        raise RuntimeError("labels disagree after EdgeID align vs existing LI cohort")

    source = matched_report["coverage"]["source"]
    floors = COVERAGE_FLOORS["Small-LI"]
    matched = {
        "train_n": int(tr_sorted.size),
        "val_n": int(va_sorted.size),
        "train_pos": int((ytr_a == 1).sum()),
        "val_pos": int((yva_a == 1).sum()),
    }
    edge_cov = min(matched["train_n"] / max(source["train_n"], 1), matched["val_n"] / max(source["val_n"], 1))
    pos_cov = min(matched["train_pos"] / max(source["train_pos"], 1), matched["val_pos"] / max(source["val_pos"], 1))
    cov_ok = edge_cov >= floors["edge"] and pos_cov >= floors["positive"]
    if not cov_ok:
        raise RuntimeError(f"matched coverage failed edge={edge_cov} pos={pos_cov}")

    exact_tr = set(etr.tolist()) == set(etr_r.tolist())
    exact_va = set(eva.tolist()) == set(eva_r.tolist())
    report = {
        "reference_cell": EXISTING_LI_CELL,
        "new_cell": CELL,
        "train_sets_exact_match": exact_tr,
        "val_sets_exact_match": exact_va,
        "matched": matched,
        "matched_train_edge_sha256": sha_ordered_ids(tr_sorted),
        "matched_val_edge_sha256": sha_ordered_ids(va_sorted),
        "reference_matched_val_edge_sha256": matched_report.get("matched_val_edge_sha256"),
        "val_edge_sha_equals_phase4b_matched": sha_ordered_ids(va_sorted)
        == matched_report.get("matched_val_edge_sha256"),
        "coverage": {
            "source": source,
            "edge_coverage": {"min": edge_cov},
            "positive_coverage": {"min": pos_cov},
            "floors": floors,
            "ok": cov_ok,
        },
        "no_test_npz": True,
    }
    write_json(ADDENDUM_DIR / "matched_cohort.json", report)
    return {
        "z_tr": ztr_a,
        "y_tr": ytr_a,
        "e_tr": etr_a,
        "z_va": zva_a,
        "y_va": yva_a,
        "e_va": eva_a,
        "report": report,
        "meta": json.loads((emb_dir() / "meta.json").read_text(encoding="utf-8")),
    }


def row_from_cell(c: Dict[str, Any], *, encoder: str, li_updates: int, prevalence: float) -> Dict[str, Any]:
    auprc = float(c["validation_auprc"])
    return {
        "Encoder/checkpoint": encoder,
        "LI updates": li_updates,
        "AUPRC": auprc,
        "lift": auprc / prevalence if prevalence > 0 else float("nan"),
        "AUROC": float(c["validation_auroc"]),
        "F1@0.5": float(c["validation_metrics_at_0.5"]["f1"]),
        "precision@0.5": float(c["validation_metrics_at_0.5"]["precision"]),
        "recall@0.5": float(c["validation_metrics_at_0.5"]["recall"]),
        "F1@val-thr": float(c["validation_metrics_at_val_optimal_f1"]["f1"]),
        "final BCE": float(c["final_probe_val_bce"]),
    }


def run_probe_and_finalize() -> Dict[str, Any]:
    ADDENDUM_DIR.mkdir(parents=True, exist_ok=True)
    pack = align_to_existing_cohort()
    fit = fit_r198_probe(
        pack["z_tr"], pack["y_tr"], pack["z_va"], pack["y_va"], device=torch.device("cpu")
    )
    meta = pack["meta"]
    report = pack["report"]
    prevalence = float(fit["prevalence_val"])
    cell = {
        "encoder": "SMALL_LI_ONLY",
        "checkpoint_step": 500,
        "target": "Small-LI",
        "cell": CELL,
        "bn_bundle_domain": "Small-LI",
        "bn_bundle_sha256": meta["bn_policy"]["bundle_sha256"],
        "target_scaler_sha256": meta["target_scaler_sha256"],
        "checkpoint_sha256": meta["checkpoint_sha256"],
        "init_sha256": INIT_SHA256,
        "coverage": report["coverage"],
        "matched_train_n": report["matched"]["train_n"],
        "matched_val_n": report["matched"]["val_n"],
        "matched_val_edge_sha256": report["matched_val_edge_sha256"],
        "validation_auprc": fit["validation_auprc"],
        "validation_auroc": fit["validation_auroc"],
        "validation_metrics_at_0.5": fit["validation_metrics_at_0.5"],
        "validation_metrics_at_val_optimal_f1": {
            **fit["validation_metrics_at_val_optimal_f1"],
            "optimistic_diagnostic": True,
            "not_a_test_estimate": True,
        },
        "final_probe_val_bce": fit["final_probe_val_bce"],
        "final_probe_train_bce": fit["final_probe_train_bce"],
        "selected_probe_val_bce": fit["selected_probe_val_bce"],
        "final_probe_epoch": fit["final_probe_epoch"],
        "selected_probe_epoch": fit["selected_probe_epoch"],
        "prevalence_val": prevalence,
        "auprc_over_prevalence_lift": float(fit["validation_auprc"]) / prevalence if prevalence else float("nan"),
        "n_val": fit["n_val"],
        "n_val_pos": fit["n_val_pos"],
        "probe_protocol": fit["probe_protocol"],
        "encoder_updated": False,
        "test_evaluated": False,
        "does_not_overwrite_existing_cells": True,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    # Write under addendum + cells with distinct name (never overwrite step1000)
    write_json(ADDENDUM_DIR / "cell_SMALL_LI_ONLY_step500__smallli.json", cell)
    write_json(ROOT / RESULT_ROOT / "cells" / f"{CELL}.json", cell)

    pred_dir = ADDENDUM_DIR / "val_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        pred_dir / "SMALL_LI_ONLY_step500.npz",
        edge_id=pack["e_va"],
        y=pack["y_va"],
        logit_selected=fit["val_logit_selected"],
        proba_selected=fit["val_proba_selected"],
        logit_final=fit["val_logit_final"],
        proba_final=fit["val_proba_final"],
    )

    # Reuse existing metrics
    c1000 = json.loads((ROOT / RESULT_ROOT / "cells" / f"{EXISTING_LI_CELL}.json").read_text())
    cmix = json.loads((ROOT / RESULT_ROOT / "cells" / f"{MIXED_LI_CELL}.json").read_text())
    # Prefer matched-cohort prevalence (identical)
    prev = prevalence
    table = [
        row_from_cell(cell, encoder="SMALL_LI_ONLY step 500", li_updates=500, prevalence=prev),
        row_from_cell(c1000, encoder="SMALL_LI_ONLY step 1000", li_updates=1000, prevalence=prev),
        row_from_cell(cmix, encoder="MIXED_3DOMAIN step 1500", li_updates=500, prevalence=prev),
    ]
    a500 = table[0]["AUPRC"]
    a1000 = table[1]["AUPRC"]
    amix = table[2]["AUPRC"]
    deltas = {
        "step500_minus_step1000_AUPRC": a500 - a1000,
        "mixed_minus_LI_step500_AUPRC": amix - a500,
        "specialist_late_training_degradation": a500 > a1000,
        "degradation_note": (
            "SPECIALIST DEGRADED between 500→1000"
            if a500 > a1000
            else "No late-training degradation (step1000 ≥ step500 AUPRC)"
        ),
    }
    summary = {
        "ok": True,
        "diagnostic": "SMALL_LI_ONLY_step500_vs_step1000_vs_mixed_exposure_matched",
        "checkpoint": CKPT_REL,
        "checkpoint_sha256": EXPECTED_CKPT_SHA,
        "init_sha256": INIT_SHA256,
        "validation_prevalence": prev,
        "step500_cell": cell,
        "comparison_table": table,
        "deltas": deltas,
        "matched_cohort": report,
        "probe_protocol": PROBE,
        "reused_existing_cells": [EXISTING_LI_CELL, MIXED_LI_CELL],
        "overwrote_existing_cells": False,
        "test_data_loaded_or_scored": False,
        "encoder_retrained": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": os.environ.get("SLURM_JOB_ID"),
    }
    write_json(ADDENDUM_DIR / "summary.json", summary)
    write_json(ROOT / RESULT_ROOT / "addendum_li_step500_summary.json", summary)

    lines = [
        "# Addendum: SMALL_LI_ONLY step-500 vs step-1000 vs MIXED (exposure-matched)",
        "",
        f"> Parent: `{RESULT_ROOT}/`",
        f"> Summary: `{RESULT_ROOT}/addendum_li_step500/summary.json`",
        "",
        "Validation-only diagnostic. No encoder retrain. No test. Did not overwrite existing cells.",
        "",
        f"**Validation prevalence** = {prev:.6e} ({cell['n_val_pos']}/{cell['n_val']})",
        "",
        "## Comparison table",
        "",
        "| Encoder/checkpoint | LI updates | AUPRC | lift | AUROC | F1@0.5 | F1@val-thr† | final BCE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in table:
        lines.append(
            f"| {r['Encoder/checkpoint']} | {r['LI updates']} | {r['AUPRC']:.4f} | {r['lift']:.1f} | "
            f"{r['AUROC']:.4f} | {r['F1@0.5']:.4f} | {r['F1@val-thr']:.4f} | {r['final BCE']:.6f} |"
        )
    lines += [
        "",
        "† F1@val-thr is an optimistic validation-selected-threshold diagnostic.",
        "",
        "## Deltas",
        "",
        f"- step500 − step1000 AUPRC = **{deltas['step500_minus_step1000_AUPRC']:+.4f}**",
        f"- mixed − LI-step500 AUPRC = **{deltas['mixed_minus_LI_step500_AUPRC']:+.4f}**",
        f"- Late-training degradation? **{deltas['specialist_late_training_degradation']}** — {deltas['degradation_note']}",
        "",
        "## Step-500 cell metrics",
        "",
        f"- AUPRC = {cell['validation_auprc']:.4f} (lift {cell['auprc_over_prevalence_lift']:.1f}×)",
        f"- AUROC = {cell['validation_auroc']:.4f}",
        f"- F1@0.5 = {cell['validation_metrics_at_0.5']['f1']:.4f} "
        f"(P={cell['validation_metrics_at_0.5']['precision']:.4f}, "
        f"R={cell['validation_metrics_at_0.5']['recall']:.4f})",
        f"- F1@val-thr† = {cell['validation_metrics_at_val_optimal_f1']['f1']:.4f}",
        f"- Final epoch-20 val BCE = {cell['final_probe_val_bce']:.6f}",
        "",
    ]
    notes = ADDENDUM_DIR / "README.md"
    notes.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Append short pointer to parent notes
    parent_notes = ROOT / "notes" / "financial_multidataset_shared_core_phase4b_frozen_eval.md"
    if parent_notes.is_file():
        text = parent_notes.read_text(encoding="utf-8")
        marker = "## Addendum: LI specialist step-500 diagnostic"
        if marker not in text:
            text = text.rstrip() + (
                "\n\n"
                + marker
                + "\n\n"
                + f"See [`{RESULT_ROOT}/addendum_li_step500/README.md`](../{RESULT_ROOT}/addendum_li_step500/README.md) "
                + f"and [`addendum_li_step500_summary.json`](../{RESULT_ROOT}/addendum_li_step500_summary.json).\n"
            )
            parent_notes.write_text(text, encoding="utf-8")
    print(json.dumps({"ok": True, "deltas": deltas, "table": table}, indent=2, default=str))
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract", "probe_finalize", "preflight"])
    args = ap.parse_args()
    logger_setup()
    logging.getLogger().setLevel(logging.INFO)
    ADDENDUM_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / RESULT_ROOT / "cells").mkdir(parents=True, exist_ok=True)

    if args.cmd == "preflight":
        p = ROOT / CKPT_REL
        sha = sha256_file(p) if p.is_file() else None
        ok = (
            p.is_file()
            and sha == EXPECTED_CKPT_SHA
            and (ROOT / RESULT_ROOT / "cells" / f"{EXISTING_LI_CELL}.json").is_file()
            and (ROOT / RESULT_ROOT / "cells" / f"{MIXED_LI_CELL}.json").is_file()
        )
        blob = torch.load(p, map_location="cpu", weights_only=False) if p.is_file() else {}
        report = {
            "ok": ok,
            "checkpoint": CKPT_REL,
            "sha256": sha,
            "expected_sha256": EXPECTED_CKPT_SHA,
            "step": blob.get("global_optimizer_step"),
            "init_sha256": blob.get("init_sha256"),
            "bn_keys": list(blob.get("bn_bundles", {}).keys()) if blob else [],
            "will_not_overwrite": [EXISTING_LI_CELL, MIXED_LI_CELL],
            "new_cell": CELL,
        }
        write_json(ADDENDUM_DIR / "preflight.json", report)
        print(json.dumps(report, indent=2))
        return 0 if ok else 2

    if args.cmd == "extract":
        out = run_extract()
        print(json.dumps({"ok": True, "status": out["status"], "cell": CELL}, indent=2))
        return 0

    if args.cmd == "probe_finalize":
        run_probe_and_finalize()
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
