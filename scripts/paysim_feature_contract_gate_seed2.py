#!/usr/bin/env python3
"""PaySim feature-contract validation gate (seed-2, validation metrics only).

Subcommands: smoke | contract_cell | aggregate

Never evaluates PaySim test. Never trains/finetunes encoders.
Never writes historical embeddings/paysim/* trees.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch_geometric.nn import to_hetero

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_loading import get_data  # noqa: E402
from data_util import apply_corrected_reverse_edge_attr  # noqa: E402
from feature_contracts import (  # noqa: E402
    CONTRACT_LEGACY,
    CONTRACT_STRUCTURE_ONLY,
    CONTRACT_TYPE_ONLY,
    PAYSIM_V1_CONTRACT_IDS,
    SLOT_CURRENCY,
    SLOT_PAYMENT_FORMAT,
    ensure_contract_output_dir,
    get_feature_contract,
)
from linear_probe import load_embedding_npz  # noqa: E402
from ranking_metrics import alert_budget_metrics  # noqa: E402
from train_util import (  # noqa: E402
    AddEgoIds,
    add_arange_ids,
    extract_param,
    get_loaders,
    load_checkpoint_weights,
)
from training import get_model  # noqa: E402
from util import create_parser, logger_setup, set_seed  # noqa: E402

TAG = "paysim_feature_contract_gate_seed2"
EMBED_ROOT = ROOT / "embeddings" / TAG
RESULT_ROOT = ROOT / "results" / "diagnostics" / TAG
CELLS = RESULT_ROOT / "cells"
NOTES_MD = ROOT / "notes" / f"{TAG}.md"
FINAL_JSON = ROOT / "results" / "diagnostics" / f"{TAG}.json"

CKPT = ROOT / (
    "saved-models/checkpoint_gin_emlps_ports_tds_corrected_asym_proj_8192neg_queue0_40ep_seed2.tar"
)
UNIQUE = "gin_emlps_ports_tds_corrected_asym_proj_8192neg_queue0_40ep_seed2"
EXPECTED_SHA_PREFIX = "18e06f555aa4"
EXPECTED_SHA_FULL = "18e06f555aa4880dfc1e95caa3f54a207e5aa186d266887772640feb93a06ae6"
EXPECTED_EDGE_DIM = 8
FORWARD_EDGE = ("node", "to", "node")
ENCODER_SEED = 2
RANDOM_INIT_SEED = 2
DOWNSTREAM_SEED = 1
FORBIDDEN_WRITE_PREFIXES = (
    ROOT / "embeddings" / "paysim",
    ROOT / "embeddings" / "paysim_dplus_transfer_final",
    ROOT / "embeddings" / "paysim_regression_audit",
    ROOT / "embeddings" / "paysim_preserve_normalization_ablation",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ids_hash(ids: np.ndarray) -> Dict[str, Any]:
    ids = np.asarray(ids, dtype=np.int64)
    return {
        "n": int(ids.shape[0]),
        "n_unique": int(np.unique(ids).shape[0]),
        "edge_id_sum": int(ids.sum()) if ids.size else 0,
        "sha256_of_ids_bytes": hashlib.sha256(ids.tobytes()).hexdigest() if ids.size else None,
    }


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n")


def assert_allowed_write(path: Path) -> None:
    resolved = path.resolve()
    for bad in FORBIDDEN_WRITE_PREFIXES:
        try:
            resolved.relative_to(bad.resolve())
        except ValueError:
            continue
        raise RuntimeError(f"Refusing write under forbidden prefix {bad}: {path}")


def code_provenance() -> Dict[str, Any]:
    def _run(cmd: List[str]) -> str:
        try:
            return subprocess.check_output(cmd, cwd=str(ROOT), stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return ""

    head = _run(["git", "rev-parse", "HEAD"])
    porcelain = _run(["git", "status", "--porcelain"])
    dirty_files = [ln[3:] for ln in porcelain.splitlines() if ln.strip()]
    src_files = [
        "feature_contracts.py",
        "data_loading.py",
        "util.py",
        "embedding_extraction.py",
        "scripts/paysim_feature_contract_gate_seed2.py",
    ]
    hashes = {}
    for rel in src_files:
        p = ROOT / rel
        if p.is_file():
            hashes[rel] = sha256_file(p)
    return {
        "git_commit": head or None,
        "dirty_file_count": len(dirty_files),
        "dirty_tree_manifest": dirty_files[:200],
        "source_file_sha256": hashes,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def gin_model_class_weight() -> Dict[int, float]:
    args = create_parser().parse_args(["--data", "PaySim", "--model", "gin", "--testing"])
    return {0: float(extract_param("w_ce1", args)), 1: float(extract_param("w_ce2", args))}


def tune_thr_max_f1(y: np.ndarray, proba: np.ndarray) -> float:
    y = y.astype(np.int64)
    if len(np.unique(y)) < 2:
        return 0.5
    prec, rec, thrs = precision_recall_curve(y, proba)
    if thrs.size == 0:
        return 0.5
    f1 = (2 * prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-12)
    return float(thrs[int(np.argmax(f1))])


def metrics_block(y: np.ndarray, proba: np.ndarray, thr: float) -> Dict[str, Any]:
    y = y.astype(np.int64)
    pred = (proba >= float(thr)).astype(np.int64)
    out: Dict[str, Any] = {
        "auroc": float(roc_auc_score(y, proba)) if len(np.unique(y)) > 1 else float("nan"),
        "auprc": float(average_precision_score(y, proba)) if len(np.unique(y)) > 1 else float("nan"),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "threshold": float(thr),
        "positive_prediction_rate": float(pred.mean()) if y.size else 0.0,
        "tp": int(((pred == 1) & (y == 1)).sum()),
        "fp": int(((pred == 1) & (y == 0)).sum()),
        "tn": int(((pred == 0) & (y == 0)).sum()),
        "fn": int(((pred == 0) & (y == 1)).sum()),
        "n": int(y.shape[0]),
        "n_positives": int(y.sum()),
        "positive_rate": float(y.mean()) if y.size else 0.0,
    }
    out.update(alert_budget_metrics(y, proba))
    return out


def hash_state_dict(sd: Dict[str, torch.Tensor], *, learned_only: bool = True) -> str:
    h = hashlib.sha256()
    for name in sorted(sd.keys()):
        if learned_only and (
            name.endswith("running_mean")
            or name.endswith("running_var")
            or name.endswith("num_batches_tracked")
        ):
            continue
        h.update(name.encode())
        t = sd[name].detach().cpu().contiguous()
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def hetero_edge_dim(data) -> int:
    store = data[FORWARD_EDGE]
    return int(store.edge_attr.shape[1])


def forward_edge_attr(data) -> torch.Tensor:
    return data[FORWARD_EDGE].edge_attr


def verify_checkpoint() -> Dict[str, Any]:
    if not CKPT.is_file():
        raise SystemExit(f"missing checkpoint {CKPT}")
    sha = sha256_file(CKPT)
    if not sha.startswith(EXPECTED_SHA_PREFIX):
        raise SystemExit(
            f"checkpoint sha prefix mismatch: got {sha[:12]} expected {EXPECTED_SHA_PREFIX}"
        )
    if sha != EXPECTED_SHA_FULL:
        logging.warning("Full sha differs from historical full string but prefix OK: %s", sha)
    blob = torch.load(CKPT, map_location="cpu")
    sd = blob["model_state_dict"]
    edim = None
    for k, v in sd.items():
        if "edge_emb" in k and k.endswith("weight"):
            edim = int(v.shape[-1])
            break
    if edim != EXPECTED_EDGE_DIM:
        raise SystemExit(f"checkpoint edge_dim={edim}")
    return {
        "path": str(CKPT),
        "sha256": sha,
        "sha_prefix_ok": True,
        "edge_dim": edim,
        "epoch": blob.get("epoch"),
        "learned_weight_sha256": hash_state_dict(sd, learned_only=True),
    }


def base_extract_argv(
    *,
    contract_id: str,
    emb_subdir: str,
    random_init: bool,
    batch_size: int,
    extract_splits: str = "train,val",
) -> List[str]:
    argv = [
        "--data", "PaySim",
        "--model", "gin",
        "--testing",
        "--tqdm",
        "--unique_name", UNIQUE,
        "--embeddings_dir", str(EMBED_ROOT),
        "--embeddings_subdir", emb_subdir,
        "--batch_size", str(batch_size),
        "--loader_num_workers", "0",
        "--num_neighs", "100", "100",
        "--representation_source", "post_embedding",
        "--extract_splits", extract_splits,
        "--reverse_mp", "--ego", "--ports", "--tds", "--emlps",
        "--correct_reverse_edge_features",
        "--train_fit_edge_znorm",
        "--feature_contract", contract_id,
        "--seed", str(ENCODER_SEED if not random_init else RANDOM_INIT_SEED),
    ]
    if random_init:
        argv.append("--random_init")
    return argv


def parse_extract_args(argv: List[str]):
    p = create_parser()
    p.add_argument("--embeddings_dir", type=str, default="embeddings")
    p.add_argument("--random_init", action="store_true")
    p.add_argument("--checkpoint_suffix", type=str, default="")
    p.add_argument("--embeddings_subdir", type=str, default=None)
    p.add_argument("--representation_source", type=str, default="post_embedding")
    p.add_argument("--extract_splits", type=str, default="train,val,test")
    return p.parse_args(argv)


def _build_model_for_data(args, te_data, tr_data, val_data, tr_inds, val_inds, te_inds, device):
    from types import SimpleNamespace

    config = SimpleNamespace(
        model=args.model,
        n_hidden=extract_param("n_hidden", args),
        n_gnn_layers=extract_param("n_gnn_layers", args),
        n_heads=extract_param("n_heads", args) if args.model == "gat" else None,
        dropout=extract_param("dropout", args),
        final_dropout=extract_param("final_dropout", args),
    )
    transform = AddEgoIds() if args.ego else None
    add_arange_ids([tr_data, val_data, te_data])
    sample_args = SimpleNamespace(**vars(args))
    sample_args.loader_num_workers = 0
    sample_loader, _, _ = get_loaders(
        tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform, sample_args, train_shuffle=False
    )
    sample_batch = next(iter(sample_loader))
    del sample_loader
    model = get_model(sample_batch, config, args)
    model = to_hetero(model, te_data.metadata(), aggr="mean")
    model.to(device)
    return model, transform


def run_logistic_val_only(
    emb_dir: Path,
    *,
    contract_id: str,
    encoder_role: str,
    ckpt_meta: Optional[Dict[str, Any]],
    random_weight_sha: Optional[str],
) -> Dict[str, Any]:
    """Fit logistic on train H; score validation only (never touch test)."""
    if (emb_dir / "test.npz").is_file():
        raise SystemExit(f"Refusing: test.npz present under {emb_dir} (test eval forbidden)")
    z_tr, y_tr, ids_tr = load_embedding_npz(emb_dir / "train.npz")
    z_va, y_va, ids_va = load_embedding_npz(emb_dir / "val.npz")
    cw = gin_model_class_weight()
    set_seed(DOWNSTREAM_SEED)
    clf = LogisticRegression(
        class_weight=cw,
        max_iter=1000,
        random_state=DOWNSTREAM_SEED,
        solver="lbfgs",
        n_jobs=1,
        C=1.0,
    )
    clf.fit(z_tr, y_tr)
    proba_tr = clf.predict_proba(z_tr)[:, 1].astype(np.float64)
    proba_va = clf.predict_proba(z_va)[:, 1].astype(np.float64)
    thr_diag = tune_thr_max_f1(y_va, proba_va)
    rep = {
        "scope": "seed2_validation_gate_only",
        "not_a_final_transfer_result": True,
        "test_evaluated": False,
        "feature_contract_id": contract_id,
        "feature_contract": get_feature_contract(contract_id).summary(),
        "encoder_role": encoder_role,
        "embeddings_dir": str(emb_dir),
        "normalization_protocol": "paysim_train_fit_edge_znorm",
        "bn_protocol": "frozen_aml_bn",
        "learner": "LogisticRegression",
        "feature_stack": "H_only_post128",
        "class_weight_mode": "model",
        "class_weight": {str(k): float(v) for k, v in cw.items()},
        "C": 1.0,
        "downstream_seed": DOWNSTREAM_SEED,
        "h_dim": int(z_tr.shape[1]),
        "checkpoint": ckpt_meta,
        "random_encoder_learned_weight_sha256": random_weight_sha,
        "ids": {"train": ids_hash(ids_tr), "val": ids_hash(ids_va)},
        "coverage": {
            "train": {
                "n": int(y_tr.shape[0]),
                "n_positives": int(y_tr.sum()),
                "positive_rate": float(y_tr.mean()),
            },
            "val": {
                "n": int(y_va.shape[0]),
                "n_positives": int(y_va.sum()),
                "positive_rate": float(y_va.mean()),
            },
        },
        "validation_metrics_at_0.5": metrics_block(y_va, proba_va, 0.5),
        "validation_metrics_at_val_optimal_f1": {
            **metrics_block(y_va, proba_va, thr_diag),
            "note": "within-validation diagnostic only; threshold selected on the same val split",
        },
        "train_fit_diagnostics": {
            "train_auroc": float(roc_auc_score(y_tr, proba_tr)) if len(np.unique(y_tr)) > 1 else None,
            "train_auprc": float(average_precision_score(y_tr, proba_tr))
            if len(np.unique(y_tr)) > 1
            else None,
        },
        "code_provenance": code_provenance(),
    }
    return rep


def cmd_smoke(args: argparse.Namespace) -> None:
    logger_setup()
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    EMBED_ROOT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt_meta = verify_checkpoint()
    contract_id = CONTRACT_TYPE_ONLY
    t0 = time.perf_counter()

    argv = base_extract_argv(
        contract_id=contract_id,
        emb_subdir="_smoke_type_only",
        random_init=False,
        batch_size=int(args.batch_size),
        extract_splits="train,val",
    )
    ns = parse_extract_args(argv)
    set_seed(ns.seed)
    with open("data_config.json") as f:
        data_config = json.load(f)
    t_data0 = time.perf_counter()
    tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(ns, data_config)
    t_data = time.perf_counter() - t_data0
    edim = hetero_edge_dim(tr_data)
    if edim != EXPECTED_EDGE_DIM:
        raise SystemExit(f"edge_dim={edim}")
    ea = forward_edge_attr(tr_data)
    cur_std = float(ea[:, SLOT_CURRENCY].std())
    typ_std = float(ea[:, SLOT_PAYMENT_FORMAT].std())
    cur_absmax = float(ea[:, SLOT_CURRENCY].abs().max())
    if cur_absmax > 1e-5:
        raise SystemExit(f"currency channel not ~0 after z-norm: absmax={cur_absmax}")
    if typ_std <= 1e-6:
        raise SystemExit(f"type channel has no variance after z-norm: std={typ_std}")
    if not torch.isfinite(ea).all():
        raise SystemExit("non-finite edge_attr")

    # reverse swap sanity on train attrs
    rev, _ = apply_corrected_reverse_edge_attr(ea, ports=True, tds=True)
    if not torch.equal(rev[:, :4], ea[:, :4]):
        raise SystemExit("reverse altered base columns")

    model, transform = _build_model_for_data(
        ns, te_data, tr_data, val_data, tr_inds, val_inds, te_inds, device
    )
    ns.unique_name = UNIQUE
    load_checkpoint_weights(model, device, ns, data_config)
    after = hash_state_dict(model.state_dict(), learned_only=True)
    if sha256_file(CKPT) != ckpt_meta["sha256"]:
        raise SystemExit("encoder checkpoint file mutated")

    tr_loader, val_loader, _ = get_loaders(
        tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform, ns, train_shuffle=False
    )
    from train_util import _extract_seed_embeddings_hetero_impl

    model.eval()
    t_ext0 = time.perf_counter()
    edge_ids, z, y, _ = _extract_seed_embeddings_hetero_impl(
        tr_loader,
        tr_inds,
        model,
        tr_data,
        device,
        ns,
        None,
        dual=False,
        max_batches=int(args.max_batches),
    )
    t_ext = time.perf_counter() - t_ext0
    z_np_check = z.detach().cpu().numpy() if torch.is_tensor(z) else np.asarray(z)
    if not np.isfinite(z_np_check).all():
        raise SystemExit("non-finite embeddings")
    ids_tr = ids_hash(
        edge_ids.detach().cpu().numpy() if torch.is_tensor(edge_ids) else np.asarray(edge_ids)
    )
    ids_va_full = ids_hash(
        val_inds.numpy() if hasattr(val_inds, "numpy") else np.asarray(val_inds)
    )

    # one logistic step on smoke embeddings (tiny)
    set_seed(DOWNSTREAM_SEED)
    cw = gin_model_class_weight()
    y_np = y.detach().cpu().numpy().astype(np.int64) if torch.is_tensor(y) else np.asarray(y)
    z_np = z.detach().cpu().numpy() if torch.is_tensor(z) else np.asarray(z)
    if len(np.unique(y_np)) < 2:
        logging.warning("smoke batch lacks both classes; skipping AUROC")
        probe_ok = True
        probe_auroc = None
    else:
        clf = LogisticRegression(
            class_weight=cw, max_iter=200, random_state=DOWNSTREAM_SEED, solver="lbfgs", n_jobs=1, C=1.0
        )
        clf.fit(z_np, y_np)
        proba = clf.predict_proba(z_np)[:, 1]
        probe_auroc = float(roc_auc_score(y_np, proba))
        probe_ok = np.isfinite(probe_auroc)

    n_train = int(tr_inds.numel() if hasattr(tr_inds, "numel") else len(tr_inds))
    batches_full = max(1, (n_train + int(args.batch_size) - 1) // int(args.batch_size))
    per_batch = t_ext / max(1, int(args.max_batches))
    proj_extract_s = per_batch * batches_full * 2  # train+val rough
    proj_total_h = (t_data + proj_extract_s + 600) / 3600.0  # +10min probe buffer
    under_6h = proj_total_h < 6.0

    report = {
        "passed": bool(probe_ok and under_6h and cur_absmax <= 1e-5 and typ_std > 1e-6),
        "feature_contract_id": contract_id,
        "feature_contract": get_feature_contract(contract_id).summary(),
        "checkpoint": ckpt_meta,
        "edge_dim": edim,
        "currency_channel_absmax_after_znorm": cur_absmax,
        "currency_channel_std_after_znorm": cur_std,
        "type_channel_std_after_znorm": typ_std,
        "embeddings_finite": True,
        "encoder_file_sha256_unchanged": True,
        "train_ids_smoke": ids_tr,
        "val_inds_hash": ids_va_full,
        "logistic_probe_ok": probe_ok,
        "logistic_probe_auroc_smoke_batches": probe_auroc,
        "timing": {
            "setup_get_data_sec": t_data,
            "extract_max_batches_sec": t_ext,
            "max_batches": int(args.max_batches),
            "per_batch_extract_sec": per_batch,
            "projected_train_val_extract_sec": proj_extract_s,
            "projected_total_hours_one_contract_cell": proj_total_h,
            "under_six_hours": under_6h,
            "wall_smoke_sec": time.perf_counter() - t0,
        },
        "code_provenance": code_provenance(),
        "model_state_learned_sha_after_load": after,
    }
    out = RESULT_ROOT / "smoke.json"
    assert_allowed_write(out)
    write_json(out, report)
    if not report["passed"]:
        raise SystemExit(f"smoke failed: {json.dumps(report['timing'], indent=2)}")
    logging.info("SMOKE PASSED → %s", out)


def extract_role(
    *,
    contract_id: str,
    random_init: bool,
    batch_size: int,
    device: torch.device,
    ckpt_meta: Dict[str, Any],
) -> Dict[str, Any]:
    import embedding_extraction as ee

    role = "random" if random_init else "pretrained"
    emb_subdir = f"{contract_id}/{role}_post128"
    out_dir = EMBED_ROOT / emb_subdir
    ensure_contract_output_dir(out_dir, contract_id)
    assert_allowed_write(out_dir)

    argv = base_extract_argv(
        contract_id=contract_id,
        emb_subdir=emb_subdir,
        random_init=random_init,
        batch_size=batch_size,
        extract_splits="train,val",
    )
    ns = parse_extract_args(argv)
    set_seed(ns.seed)
    with open("data_config.json") as f:
        data_config = json.load(f)
    tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(ns, data_config)
    if hetero_edge_dim(tr_data) != EXPECTED_EDGE_DIM:
        raise SystemExit("edge_dim != 8")
    ea = forward_edge_attr(tr_data)
    channel_checks = {
        "currency_absmax": float(ea[:, SLOT_CURRENCY].abs().max()),
        "currency_std": float(ea[:, SLOT_CURRENCY].std()),
        "payment_format_std": float(ea[:, SLOT_PAYMENT_FORMAT].std()),
    }
    if contract_id == CONTRACT_TYPE_ONLY and channel_checks["currency_absmax"] > 1e-5:
        raise SystemExit("type_only currency not neutralized")
    if contract_id == CONTRACT_STRUCTURE_ONLY and (
        channel_checks["currency_absmax"] > 1e-5
        or float(ea[:, SLOT_PAYMENT_FORMAT].abs().max()) > 1e-5
    ):
        raise SystemExit("structure_only categorical slots not neutralized")

    # Full extract via embedding_extraction (train+val only)
    out = ee.run_embedding_extraction(
        tr_data, val_data, te_data, tr_inds, val_inds, te_inds, ns, data_config
    )
    if (Path(out) / "test.npz").is_file():
        raise SystemExit("test.npz was written — forbidden")

    # Enrich meta
    meta_path = Path(out) / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    random_sha = None
    if random_init:
        # Hash a freshly seeded model (same seed/architecture across contracts).
        set_seed(RANDOM_INIT_SEED)
        model_r, _ = _build_model_for_data(
            ns, te_data, tr_data, val_data, tr_inds, val_inds, te_inds, device
        )
        random_sha = hash_state_dict(model_r.state_dict(), learned_only=True)
        del model_r
    meta.update(
        {
            "feature_contract_id": contract_id,
            "feature_contract": get_feature_contract(contract_id).summary(),
            "normalization_protocol": "paysim_train_fit_edge_znorm",
            "bn_protocol": "frozen_aml_bn",
            "checkpoint": None if random_init else ckpt_meta,
            "random_encoder_learned_weight_sha256": random_sha,
            "channel_checks_after_znorm": channel_checks,
            "extract_splits": "train,val",
            "test_extracted": False,
            "code_provenance": code_provenance(),
            "gate_tag": TAG,
        }
    )
    write_json(meta_path, meta)

    probe = run_logistic_val_only(
        Path(out),
        contract_id=contract_id,
        encoder_role=role,
        ckpt_meta=None if random_init else ckpt_meta,
        random_weight_sha=random_sha,
    )
    cell_path = CELLS / f"{contract_id}__{role}.json"
    assert_allowed_write(cell_path)
    write_json(cell_path, probe)
    return {
        "embeddings_dir": str(out),
        "cell": str(cell_path),
        "channel_checks": channel_checks,
        "random_encoder_learned_weight_sha256": random_sha,
        "validation_auprc": probe["validation_metrics_at_0.5"]["auprc"],
        "validation_auroc": probe["validation_metrics_at_0.5"]["auroc"],
        "validation_f1_0.5": probe["validation_metrics_at_0.5"]["f1"],
    }


def cmd_contract_cell(args: argparse.Namespace) -> None:
    logger_setup()
    contract_id = args.feature_contract
    if contract_id not in PAYSIM_V1_CONTRACT_IDS:
        raise SystemExit(f"bad contract {contract_id}")
    CELLS.mkdir(parents=True, exist_ok=True)
    EMBED_ROOT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt_meta = verify_checkpoint()

    logging.info("=== contract_cell %s pretrained ===", contract_id)
    pre = extract_role(
        contract_id=contract_id,
        random_init=False,
        batch_size=int(args.batch_size),
        device=device,
        ckpt_meta=ckpt_meta,
    )
    logging.info("=== contract_cell %s random ===", contract_id)
    rnd = extract_role(
        contract_id=contract_id,
        random_init=True,
        batch_size=int(args.batch_size),
        device=device,
        ckpt_meta=ckpt_meta,
    )
    summary = {
        "feature_contract_id": contract_id,
        "pretrained": pre,
        "random": rnd,
        "delta_pretrained_minus_random": {
            "validation_auprc": float(pre["validation_auprc"] - rnd["validation_auprc"]),
            "validation_auroc": float(pre["validation_auroc"] - rnd["validation_auroc"]),
            "validation_f1_0.5": float(pre["validation_f1_0.5"] - rnd["validation_f1_0.5"]),
        },
        "checkpoint": ckpt_meta,
        "code_provenance": code_provenance(),
        "test_evaluated": False,
        "scope": "seed2_validation_gate_only",
    }
    out = RESULT_ROOT / f"cell_summary_{contract_id}.json"
    assert_allowed_write(out)
    write_json(out, summary)
    logging.info("Wrote %s", out)


def cmd_aggregate(_: argparse.Namespace) -> None:
    logger_setup()
    rows = []
    random_shas = []
    for cid in PAYSIM_V1_CONTRACT_IDS:
        path = RESULT_ROOT / f"cell_summary_{cid}.json"
        if not path.is_file():
            raise SystemExit(f"missing {path}")
        summary = json.loads(path.read_text(encoding="utf-8"))
        pre_cell = json.loads(Path(summary["pretrained"]["cell"]).read_text(encoding="utf-8"))
        rnd_cell = json.loads(Path(summary["random"]["cell"]).read_text(encoding="utf-8"))
        if pre_cell.get("test_evaluated") or rnd_cell.get("test_evaluated"):
            raise SystemExit("test metrics found — abort")
        rsa = summary["random"].get("random_encoder_learned_weight_sha256")
        if rsa:
            random_shas.append(rsa)
        rows.append(
            {
                "feature_contract_id": cid,
                "pretrained_val_auprc": pre_cell["validation_metrics_at_0.5"]["auprc"],
                "pretrained_val_auroc": pre_cell["validation_metrics_at_0.5"]["auroc"],
                "pretrained_val_f1_0.5": pre_cell["validation_metrics_at_0.5"]["f1"],
                "pretrained_val_f1_at_val_opt": pre_cell["validation_metrics_at_val_optimal_f1"]["f1"],
                "pretrained_val_opt_threshold": pre_cell["validation_metrics_at_val_optimal_f1"]["threshold"],
                "random_val_auprc": rnd_cell["validation_metrics_at_0.5"]["auprc"],
                "random_val_auroc": rnd_cell["validation_metrics_at_0.5"]["auroc"],
                "random_val_f1_0.5": rnd_cell["validation_metrics_at_0.5"]["f1"],
                "delta_auprc": summary["delta_pretrained_minus_random"]["validation_auprc"],
                "delta_auroc": summary["delta_pretrained_minus_random"]["validation_auroc"],
                "delta_f1_0.5": summary["delta_pretrained_minus_random"]["validation_f1_0.5"],
                "pretrained_cell": summary["pretrained"]["cell"],
                "random_cell": summary["random"]["cell"],
                "pretrained_full_val_0.5": pre_cell["validation_metrics_at_0.5"],
                "pretrained_full_val_opt": pre_cell["validation_metrics_at_val_optimal_f1"],
            }
        )

    if random_shas and len(set(random_shas)) != 1:
        logging.warning("Random encoder weight hashes differ across contracts: %s", set(random_shas))

    # Selection
    def sort_key(r):
        return (r["pretrained_val_auprc"], r["pretrained_val_f1_0.5"])

    ranked = sorted(rows, key=sort_key, reverse=True)
    numerical_winner = ranked[0]
    selected = None
    selection_notes = []
    for r in ranked:
        if r["pretrained_val_auprc"] > r["random_val_auprc"]:
            selected = r
            selection_notes.append(
                f"{r['feature_contract_id']} beats matched random on val AUPRC "
                f"({r['pretrained_val_auprc']:.4f} > {r['random_val_auprc']:.4f})"
            )
            break
        selection_notes.append(
            f"{r['feature_contract_id']} fails random beat "
            f"({r['pretrained_val_auprc']:.4f} <= {r['random_val_auprc']:.4f})"
        )
    if selected is None:
        selected = numerical_winner
        selection_notes.append("FALLBACK: no contract beat random; reporting numerical winner with warning")

    type_only = next(r for r in rows if r["feature_contract_id"] == CONTRACT_TYPE_ONLY)
    prefer_type_only = False
    if (
        selected["feature_contract_id"] != CONTRACT_TYPE_ONLY
        and type_only["pretrained_val_auprc"] > type_only["random_val_auprc"]
        and abs(type_only["pretrained_val_auprc"] - selected["pretrained_val_auprc"]) <= 0.005
        and abs(type_only["pretrained_val_f1_0.5"] - selected["pretrained_val_f1_0.5"]) <= 0.01
    ):
        prefer_type_only = True
        selection_notes.append(
            "Preferring paysim_type_only_v1 as semantically defensible "
            f"(within 0.005 AUPRC and 0.01 F1 of {selected['feature_contract_id']})"
        )
        selected = type_only

    final = {
        "title": "PaySim feature-contract validation gate (seed-2)",
        "scope": "seed2_validation_gate_only",
        "not_a_final_transfer_result": True,
        "test_evaluated": False,
        "checkpoint": verify_checkpoint(),
        "protocol": {
            "encoder": "frozen corrected/no-preserve seed-2 GIN",
            "representation": "post-128 H",
            "normalization": "strict-inductive train-fit edge z-norm",
            "bn": "frozen AML BatchNorm",
            "learner": "LogisticRegression class_weight=model C=1",
            "metrics_split": "validation only",
        },
        "contracts": rows,
        "ranking_by_pretrained_val_auprc": [r["feature_contract_id"] for r in ranked],
        "numerical_winner": numerical_winner["feature_contract_id"],
        "selected_contract": selected["feature_contract_id"],
        "prefer_type_only_applied": prefer_type_only,
        "selection_notes": selection_notes,
        "random_encoder_sha_agreement": {
            "unique_hashes": sorted(set(random_shas)),
            "all_equal": len(set(random_shas)) <= 1,
        },
        "code_provenance": code_provenance(),
    }
    assert_allowed_write(FINAL_JSON)
    write_json(FINAL_JSON, final)
    write_json(RESULT_ROOT / "aggregate.json", final)

    lines = [
        f"# PaySim feature-contract validation gate (seed-2)",
        "",
        "> **Scope:** seed-2 validation gate only. **Not** a final transfer result. "
        "PaySim test was never evaluated.",
        "",
        f"- Checkpoint: `{CKPT.name}` sha256 `{final['checkpoint']['sha256'][:12]}…`",
        f"- Selected contract: **`{final['selected_contract']}`**",
        f"- Numerical winner (val AUPRC): `{final['numerical_winner']}`",
        "",
        "## Validation metrics (pretrained vs random)",
        "",
        "| Contract | Pre AUROC | Pre AUPRC | Pre F1@0.5 | Pre F1@val-opt | Rand AUPRC | Δ AUPRC |",
        "|----------|----------:|----------:|-----------:|---------------:|-----------:|--------:|",
    ]
    for r in ranked:
        lines.append(
            f"| `{r['feature_contract_id']}` | {r['pretrained_val_auroc']:.4f} | "
            f"{r['pretrained_val_auprc']:.4f} | {r['pretrained_val_f1_0.5']:.4f} | "
            f"{r['pretrained_val_f1_at_val_opt']:.4f} | {r['random_val_auprc']:.4f} | "
            f"{r['delta_auprc']:+.4f} |"
        )
    lines.extend(["", "## Selection notes", ""])
    for n in selection_notes:
        lines.append(f"- {n}")
    lines.extend(
        [
            "",
            "## Protocol",
            "",
            "- Frozen corrected/no-preserve seed-2 GIN; post-128 H; train-fit z-norm; frozen AML BN",
            "- LogisticRegression `class_weight=model`, `C=1`; validation metrics only",
            "- Matched random edge-dim-8 re-extracted under each contract",
            "",
        ]
    )
    NOTES_MD.parent.mkdir(parents=True, exist_ok=True)
    NOTES_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logging.info("Wrote %s and %s", FINAL_JSON, NOTES_MD)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    sm = sub.add_parser("smoke")
    sm.add_argument("--batch_size", type=int, default=4096)
    sm.add_argument("--max_batches", type=int, default=2)
    sm.set_defaults(func=cmd_smoke)

    cc = sub.add_parser("contract_cell")
    cc.add_argument("--feature_contract", type=str, required=True, choices=list(PAYSIM_V1_CONTRACT_IDS))
    cc.add_argument("--batch_size", type=int, default=4096)
    cc.set_defaults(func=cmd_contract_cell)

    ag = sub.add_parser("aggregate")
    ag.set_defaults(func=cmd_aggregate)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
