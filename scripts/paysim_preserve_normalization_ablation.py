#!/usr/bin/env python3
"""PaySim preserve-vs-normalization ablation (Slurm compute nodes only).

Subcommands: smoke | extract_probe | amlworld_eval_a | bn_recal_probe | aggregate

Never writes historical embeddings/paysim/*, paysim_dplus_transfer_final/*,
or paysim_regression_audit/* (reuse is read-only + pointer JSONs).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch_geometric.nn import to_hetero

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_loading import get_data  # noqa: E402
from gcpal_txn_node.eval_mlp import PaperStyleMLP, _predict_proba  # noqa: E402
from linear_probe import load_embedding_npz  # noqa: E402
from ranking_metrics import alert_budget_metrics  # noqa: E402
from train_util import (  # noqa: E402
    AddEgoIds,
    add_arange_ids,
    expected_seed_edge_ids,
    extract_param,
    extract_seed_embeddings_hetero,
    get_loaders,
    load_checkpoint_weights,
    log_seed_coverage,
    save_embedding_split_npz,
)
from training import get_model  # noqa: E402
from util import create_parser, logger_setup, set_seed  # noqa: E402

EMBED_ROOT = ROOT / "embeddings" / "paysim_preserve_normalization_ablation"
RESULT_ROOT = ROOT / "results" / "diagnostics" / "paysim_preserve_normalization_ablation"
CELLS = RESULT_ROOT / "cells"
FORBIDDEN_WRITE_PREFIXES = (
    ROOT / "embeddings" / "paysim",
    ROOT / "embeddings" / "paysim_dplus_transfer_final",
    ROOT / "embeddings" / "paysim_regression_audit",
)

CKPT_A = ROOT / (
    "saved-models/checkpoint_gin_emlps_ports_tds_corrected_asym_proj_8192neg_queue0_40ep_seed2.tar"
)
CKPT_B = ROOT / (
    "saved-models/checkpoint_gin_emlps_ports_tds_corrected_preserve_seed_"
    "asym_proj_8192neg_queue0_40ep_seed2.tar"
)
UNIQUE_A = "gin_emlps_ports_tds_corrected_asym_proj_8192neg_queue0_40ep_seed2"
UNIQUE_B = (
    "gin_emlps_ports_tds_corrected_preserve_seed_asym_proj_8192neg_queue0_40ep_seed2"
)
EXPECTED_SHA_A = "18e06f555aa4880dfc1e95caa3f54a207e5aa186d266887772640feb93a06ae6"
EXPECTED_SHA_B = "a320920141f585c5825cbd63ce760a845fb434a9b162d4c87270dc72b0442b87"
EXPECTED_EDGE_DIM = 8
FORWARD_EDGE_TYPE = ("node", "to", "node")
ENCODER_SEED = 2
RANDOM_INIT_SEED = 2
DOWNSTREAM_SEED = 1
MLP_EPOCHS = 15
MLP_LR = 1e-3
MLP_BS = 8192
TF_CACHE = ROOT / "results/cache/temporal_flow_causal/Small-HI"


def hetero_edge_dim(data) -> int:
    """Edge feature width on hetero graphs (attrs live on edge stores, not top-level)."""
    store = data[FORWARD_EDGE_TYPE]
    if not hasattr(store, "edge_attr") or store.edge_attr is None:
        raise AttributeError(f"missing edge_attr on {FORWARD_EDGE_TYPE}")
    return int(store.edge_attr.shape[1])

# Read-only reusable artifacts (verified in aggregate / smoke provenance)
REUSE_A_PERGRAPH_EMB = ROOT / "embeddings/paysim_regression_audit/lineage_corrected_post128"
REUSE_A_PERGRAPH_CELL = (
    ROOT / "results/diagnostics/paysim_regression_audit/cells/L_corrected_logistic_H.json"
)
REUSE_B_TRAINFIT_EMB = (
    ROOT / "embeddings/paysim_dplus_transfer_final/dplus_seed2/post_embedding_128"
)
REUSE_B_TRAINFIT_CELL = (
    ROOT
    / "results/diagnostics/paysim_regression_audit/cells/C_dplus_seed2_post128_logistic_model.json"
)
REUSE_B_TRAINFIT_ROLE = (
    ROOT / "results/diagnostics/paysim_dplus_transfer_final/role_seed2.json"
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


def metrics_block(y: np.ndarray, proba: np.ndarray, thr: float) -> Dict[str, float]:
    y = y.astype(np.int64)
    pred = (proba >= float(thr)).astype(np.int64)
    out = {
        "auroc": float(roc_auc_score(y, proba)) if len(np.unique(y)) > 1 else float("nan"),
        "auprc": float(average_precision_score(y, proba)) if len(np.unique(y)) > 1 else float("nan"),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "threshold": float(thr),
        "positive_prediction_rate": float(pred.mean()) if y.size else 0.0,
        "tp": float(((pred == 1) & (y == 1)).sum()),
        "fp": float(((pred == 1) & (y == 0)).sum()),
        "tn": float(((pred == 0) & (y == 0)).sum()),
        "fn": float(((pred == 0) & (y == 1)).sum()),
        "n": float(y.shape[0]),
        "n_positives": float(int(y.sum())),
        "positive_rate": float(y.mean()) if y.size else 0.0,
    }
    out.update(alert_budget_metrics(y, proba))
    return out


def hash_state_dict(sd: Dict[str, torch.Tensor], *, include: Optional[str] = None) -> str:
    h = hashlib.sha256()
    for name in sorted(sd.keys()):
        if include == "bn_stats":
            if not (name.endswith("running_mean") or name.endswith("running_var") or name.endswith("num_batches_tracked")):
                continue
        elif include == "learned":
            if name.endswith("running_mean") or name.endswith("running_var") or name.endswith("num_batches_tracked"):
                continue
        h.update(name.encode())
        t = sd[name].detach().cpu().contiguous()
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def edge_dim_from_state(sd: Dict[str, torch.Tensor]) -> int:
    for k, v in sd.items():
        if "edge_emb" in k and k.endswith("weight"):
            return int(v.shape[-1])
    raise KeyError("no edge_emb.*.weight in state dict")


def base_extract_argv(
    *,
    data: str,
    unique: str,
    emb_dir: str,
    emb_subdir: str,
    seed: int,
    train_fit: bool,
    random_init: bool = False,
    representation_source: str = "post_embedding",
    batch_size: int = 4096,
) -> List[str]:
    argv = [
        "--data", data,
        "--model", "gin",
        "--testing",
        "--tqdm",
        "--unique_name", unique,
        "--embeddings_dir", emb_dir,
        "--embeddings_subdir", emb_subdir,
        "--batch_size", str(batch_size),
        "--loader_num_workers", "0",
        "--num_neighs", "100", "100",
        "--representation_source", representation_source,
        "--reverse_mp", "--ego", "--ports", "--tds", "--emlps",
        "--correct_reverse_edge_features",
        "--seed", str(seed),
    ]
    if train_fit:
        argv.append("--train_fit_edge_znorm")
    if random_init:
        argv.append("--random_init")
    return argv


def run_logistic_pair(
    emb_dir: Path,
    *,
    label_prefix: str,
    normalization: str,
    bn_protocol: str,
    checkpoint_path: Optional[str],
    checkpoint_sha: Optional[str],
    encoder_role: str,
) -> Dict[str, Any]:
    splits = {}
    for sp in ("train", "val", "test"):
        z, y, ids = load_embedding_npz(emb_dir / f"{sp}.npz")
        splits[sp] = {"Z": z, "y": y, "ids": ids}

    reports = {}
    for cw_mode in ("model", "none"):
        cw: Any = gin_model_class_weight() if cw_mode == "model" else None
        set_seed(DOWNSTREAM_SEED)
        clf = LogisticRegression(
            class_weight=cw, max_iter=1000, random_state=DOWNSTREAM_SEED, solver="lbfgs", n_jobs=1, C=1.0
        )
        clf.fit(splits["train"]["Z"], splits["train"]["y"])
        proba = {sp: clf.predict_proba(splits[sp]["Z"])[:, 1].astype(np.float64) for sp in splits}
        thr = tune_thr_max_f1(splits["val"]["y"], proba["val"])
        key = f"{label_prefix}_logistic_cw_{cw_mode}"
        rep = {
            "label": key,
            "encoder_role": encoder_role,
            "embeddings_dir": str(emb_dir),
            "normalization_protocol": normalization,
            "bn_protocol": bn_protocol,
            "learner": "LogisticRegression",
            "feature_stack": "H_only",
            "class_weight_mode": cw_mode,
            "class_weight": cw if not isinstance(cw, dict) else {str(k): float(v) for k, v in cw.items()},
            "C": 1.0,
            "seed": DOWNSTREAM_SEED,
            "h_dim": int(splits["train"]["Z"].shape[1]),
            "checkpoint_path": checkpoint_path,
            "checkpoint_sha256": checkpoint_sha,
            "ids": {sp: ids_hash(splits[sp]["ids"]) for sp in splits},
            "coverage": {
                sp: {
                    "n": int(splits[sp]["y"].shape[0]),
                    "n_positives": int(splits[sp]["y"].sum()),
                    "positive_rate": float(splits[sp]["y"].mean()),
                }
                for sp in splits
            },
            "validation_selected_threshold": thr,
            "threshold_0.5": metrics_block(splits["test"]["y"], proba["test"], 0.5),
            "threshold_val_selected": metrics_block(splits["test"]["y"], proba["test"], thr),
            "val_ranking": {
                "auroc": float(roc_auc_score(splits["val"]["y"], proba["val"])),
                "auprc": float(average_precision_score(splits["val"]["y"], proba["val"])),
            },
        }
        out = CELLS / f"{key}.json"
        assert_allowed_write(out)
        write_json(out, rep)
        reports[cw_mode] = {"path": str(out), "threshold_0.5": rep["threshold_0.5"]}
        logging.info("Wrote %s auroc=%.4f", out, rep["threshold_0.5"]["auroc"])
    return reports


def _build_model_for_data(args, te_data, tr_data, val_data, tr_inds, val_inds, te_inds, device):
    config = type("C", (), {})()
    config.model = args.model
    config.n_hidden = extract_param("n_hidden", args)
    config.n_gnn_layers = extract_param("n_gnn_layers", args)
    config.n_heads = extract_param("n_heads", args) if args.model == "gat" else None
    config.dropout = extract_param("dropout", args)
    config.final_dropout = extract_param("final_dropout", args)
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
    return model, transform


def recalibrate_bn_train_only(
    model: nn.Module,
    tr_loader,
    tr_data,
    device,
    args,
    *,
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    """Update BN running stats on PaySim train batches only; no optimizer / backward."""
    before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    learned_before = hash_state_dict(before, include="learned")
    bn_before = hash_state_dict(before, include="bn_stats")

    for p in model.parameters():
        p.requires_grad = False
    model.train()
    n_batches = 0
    store = ("node", "to", "node")
    rev = ("node", "rev_to", "node")
    with torch.no_grad():
        for batch_i, batch in enumerate(tr_loader):
            if max_batches is not None and batch_i >= int(max_batches):
                break
            batch[store].edge_attr = batch[store].edge_attr[:, 1:]
            batch[rev].edge_attr = batch[rev].edge_attr[:, 1:]
            batch.to(device)
            model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
            n_batches += 1
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    after = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    learned_after = hash_state_dict(after, include="learned")
    bn_after = hash_state_dict(after, include="bn_stats")
    if learned_before != learned_after:
        raise RuntimeError("BN recalibration changed learned parameters")
    # byte-identical check on learned tensors
    for k in before:
        if k.endswith("running_mean") or k.endswith("running_var") or k.endswith("num_batches_tracked"):
            continue
        if not torch.equal(before[k], after[k]):
            raise RuntimeError(f"learned tensor changed: {k}")
    return {
        "n_batches": n_batches,
        "learned_hash_before": learned_before,
        "learned_hash_after": learned_after,
        "bn_stats_hash_before": bn_before,
        "bn_stats_hash_after": bn_after,
        "learned_unchanged": learned_before == learned_after,
        "bn_stats_changed": bn_before != bn_after,
    }


def cmd_smoke(args: argparse.Namespace) -> int:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    CELLS.mkdir(parents=True, exist_ok=True)
    EMBED_ROOT.mkdir(parents=True, exist_ok=True)
    gates: Dict[str, Any] = {"passed": False}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for label, path, expected in (("A", CKPT_A, EXPECTED_SHA_A), ("B", CKPT_B, EXPECTED_SHA_B)):
        if not path.is_file():
            raise SystemExit(f"missing {path}")
        sha = sha256_file(path)
        ckpt = torch.load(path, map_location="cpu")
        sd = ckpt["model_state_dict"]
        edim = edge_dim_from_state(sd)
        gates[f"ckpt_{label}"] = {
            "path": str(path),
            "sha256": sha,
            "sha_ok": sha == expected,
            "edge_dim": edim,
            "edge_dim_ok": edim == EXPECTED_EDGE_DIM,
            "epoch": ckpt.get("epoch"),
        }
        if sha != expected or edim != EXPECTED_EDGE_DIM:
            raise SystemExit(f"checkpoint gate failed for {label}: {gates[f'ckpt_{label}']}")

    # Exercise both norms + BN recalib on a few train batches (encoder A)
    import embedding_extraction as ee

    for train_fit, tag in ((False, "pergraph"), (True, "trainfit")):
        argv = base_extract_argv(
            data="PaySim",
            unique=UNIQUE_A,
            emb_dir=str(EMBED_ROOT / "_smoke"),
            emb_subdir=f"A_{tag}_smoke",
            seed=ENCODER_SEED,
            train_fit=train_fit,
            batch_size=int(args.batch_size),
        )
        # Build args via create_parser of embedding_extraction
        p = create_parser()
        # re-add extraction-only flags from ee.main
        p.add_argument("--embeddings_dir", type=str, default="embeddings")
        p.add_argument("--random_init", action="store_true")
        p.add_argument("--checkpoint_suffix", type=str, default="")
        p.add_argument("--embeddings_subdir", type=str, default=None)
        p.add_argument("--representation_source", type=str, default="post_embedding")
        p.add_argument("--extract_splits", type=str, default="train,val,test")
        ns = p.parse_args(argv)
        set_seed(ns.seed)
        with open("data_config.json") as f:
            data_config = json.load(f)
        tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(ns, data_config)
        edim = hetero_edge_dim(tr_data)
        if edim != EXPECTED_EDGE_DIM:
            raise SystemExit(f"runtime edge_dim={edim} != 8 for {tag}")
        model, transform = _build_model_for_data(ns, te_data, tr_data, val_data, tr_inds, val_inds, te_inds, device)
        # Load A weights by temporarily setting unique_name
        ns.unique_name = UNIQUE_A
        load_checkpoint_weights(model, device, ns, data_config)
        model.eval()
        tr_loader, _, _ = get_loaders(
            tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform, ns, train_shuffle=False
        )
        bn_info = recalibrate_bn_train_only(
            model, tr_loader, tr_data, device, ns, max_batches=int(args.max_batches)
        )
        if not bn_info["learned_unchanged"]:
            raise SystemExit("smoke BN changed learned weights")
        if not bn_info["bn_stats_changed"]:
            logging.warning("BN stats hash unchanged on smoke batches (possible but unusual)")
        # few-batch post-128 extract via capped impl
        from train_util import (
            PreEmbeddingCapture,
            _extract_seed_embeddings_hetero_impl,
            resolve_embedding_head_linear,
        )

        model.eval()
        # post-only: capture=None
        edge_ids, z_post, y, _z_pre = _extract_seed_embeddings_hetero_impl(
            tr_loader,
            tr_inds,
            model,
            tr_data,
            device,
            ns,
            capture=None,
            dual=False,
            max_batches=int(args.max_batches),
        )
        z = z_post
        if not torch.isfinite(z).all():
            raise SystemExit("non-finite embeddings in smoke")
        if edge_ids.numel() != z.shape[0] or z.shape[0] != y.shape[0]:
            raise SystemExit("ID alignment failure in smoke")
        gates[f"norm_{tag}"] = {
            "edge_dim": edim,
            "train_fit_edge_znorm": train_fit,
            "bn": bn_info,
            "n_rows_smoke": int(z.shape[0]),
            "z_dim": int(z.shape[1]),
            "finite": True,
            "id_aligned": True,
        }

    gates["passed"] = True
    gates["host"] = __import__("socket").gethostname()
    gates["job_id"] = __import__("os").environ.get("SLURM_JOB_ID")
    gates["timestamp"] = datetime.now(timezone.utc).isoformat()
    out_json = RESULT_ROOT / "smoke.json"
    out_md = RESULT_ROOT / "smoke.md"
    assert_allowed_write(out_json)
    write_json(out_json, gates)
    out_md.write_text(
        "# PaySim preserve/normalization ablation — smoke\n\n"
        f"- passed: **{gates['passed']}**\n"
        f"- job: {gates['job_id']} host: {gates['host']}\n"
        f"- ckpt A sha ok: {gates['ckpt_A']['sha_ok']} edge_dim={gates['ckpt_A']['edge_dim']}\n"
        f"- ckpt B sha ok: {gates['ckpt_B']['sha_ok']} edge_dim={gates['ckpt_B']['edge_dim']}\n"
        f"- pergraph/trainfit few-batch + BN gates recorded in smoke.json\n"
    )
    logging.info("Smoke PASSED → %s", out_json)
    return 0


def cmd_extract_probe(args: argparse.Namespace) -> int:
    """CELL: A_trainfit | B_pergraph | random_both"""
    import embedding_extraction as ee

    CELLS.mkdir(parents=True, exist_ok=True)
    EMBED_ROOT.mkdir(parents=True, exist_ok=True)
    cell = args.cell
    with open("data_config.json") as f:
        data_config = json.load(f)

    def _extract_one(unique, subdir, train_fit, random_init, seed, ckpt_path, ckpt_sha, role, norm):
        argv = base_extract_argv(
            data="PaySim",
            unique=unique,
            emb_dir=str(EMBED_ROOT),
            emb_subdir=subdir,
            seed=seed,
            train_fit=train_fit,
            random_init=random_init,
            batch_size=int(args.batch_size),
        )
        p = create_parser()
        p.add_argument("--embeddings_dir", type=str, default="embeddings")
        p.add_argument("--random_init", action="store_true")
        p.add_argument("--checkpoint_suffix", type=str, default="")
        p.add_argument("--embeddings_subdir", type=str, default=None)
        p.add_argument("--representation_source", type=str, default="post_embedding")
        p.add_argument("--extract_splits", type=str, default="train,val,test")
        ns = p.parse_args(argv)
        set_seed(ns.seed)
        # If not random, ensure unique points at the right checkpoint filename
        if not random_init and unique not in (UNIQUE_A, UNIQUE_B):
            raise SystemExit(f"unexpected unique {unique}")
        tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(ns, data_config)
        edim = hetero_edge_dim(tr_data)
        if edim != EXPECTED_EDGE_DIM:
            raise SystemExit(f"edge_dim={edim}")
        out = ee.run_embedding_extraction(
            tr_data, val_data, te_data, tr_inds, val_inds, te_inds, ns, data_config
        )
        assert_allowed_write(out)
        # annotate meta
        meta_path = out / "meta.json"
        meta = json.loads(meta_path.read_text())
        meta["train_fit_edge_znorm"] = bool(train_fit)
        meta["correct_reverse_edge_features"] = True
        meta["checkpoint_sha256"] = ckpt_sha
        meta["normalization_protocol"] = norm
        meta["encoder_role"] = role
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        run_logistic_pair(
            out,
            label_prefix=subdir,
            normalization=norm,
            bn_protocol="none_frozen_eval",
            checkpoint_path=str(ckpt_path) if ckpt_path else None,
            checkpoint_sha=ckpt_sha,
            encoder_role=role,
        )
        return str(out)

    summary: Dict[str, Any] = {"cell": cell, "job_id": __import__("os").environ.get("SLURM_JOB_ID")}

    if cell == "A_trainfit":
        sha = sha256_file(CKPT_A)
        if sha != EXPECTED_SHA_A:
            raise SystemExit("A hash mismatch")
        summary["out"] = _extract_one(
            UNIQUE_A, "A_corrected_trainfit_post128", True, False, ENCODER_SEED, CKPT_A, sha, "A_corrected_no_preserve", "paysim_train_fit_edge_znorm"
        )
    elif cell == "B_pergraph":
        sha = sha256_file(CKPT_B)
        if sha != EXPECTED_SHA_B:
            raise SystemExit("B hash mismatch")
        summary["out"] = _extract_one(
            UNIQUE_B, "B_dplus_pergraph_post128", False, False, ENCODER_SEED, CKPT_B, sha, "B_dplus_preserve", "per_graph_edge_znorm"
        )
    elif cell == "random_both":
        # Record seed BEFORE any metrics
        seed_rec = {
            "random_init_seed": RANDOM_INIT_SEED,
            "architecture": "gin+ports+tds+emlps+correct_reverse edge_dim=8",
            "recorded_before_metrics": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        write_json(RESULT_ROOT / "random_init_seed_record.json", seed_rec)
        outs = {}
        for train_fit, sub, norm in (
            (True, "random_edge8_trainfit_post128", "paysim_train_fit_edge_znorm"),
            (False, "random_edge8_pergraph_post128", "per_graph_edge_znorm"),
        ):
            outs[sub] = _extract_one(
                f"random_init_edge8_{sub}",
                sub,
                train_fit,
                True,
                RANDOM_INIT_SEED,
                None,
                None,
                "random_init_edge_dim8",
                norm,
            )
        summary["outs"] = outs
        summary["seed_record"] = str(RESULT_ROOT / "random_init_seed_record.json")
    else:
        raise SystemExit(f"unknown cell {cell}")

    write_json(CELLS / f"_summary_{cell}.json", summary)
    return 0


def cmd_bn_recal_probe(args: argparse.Namespace) -> int:
    """BN recalibration then full post-128 extract + logistic for A or B."""
    import embedding_extraction as ee
    from types import SimpleNamespace

    role = args.role  # A or B
    if role == "A":
        ckpt, unique, expected_sha, train_fit, sub = (
            CKPT_A, UNIQUE_A, EXPECTED_SHA_A, True, "A_corrected_trainfit_bnrecal_post128"
        )
        enc_role = "A_corrected_no_preserve"
        norm = "paysim_train_fit_edge_znorm"
    else:
        ckpt, unique, expected_sha, train_fit, sub = (
            CKPT_B, UNIQUE_B, EXPECTED_SHA_B, False, "B_dplus_pergraph_bnrecal_post128"
        )
        enc_role = "B_dplus_preserve"
        norm = "per_graph_edge_znorm"

    sha = sha256_file(ckpt)
    if sha != expected_sha:
        raise SystemExit("checkpoint hash mismatch")

    argv = base_extract_argv(
        data="PaySim",
        unique=unique,
        emb_dir=str(EMBED_ROOT),
        emb_subdir=sub,
        seed=ENCODER_SEED,
        train_fit=train_fit,
        batch_size=int(args.batch_size),
    )
    p = create_parser()
    p.add_argument("--embeddings_dir", type=str, default="embeddings")
    p.add_argument("--random_init", action="store_true")
    p.add_argument("--checkpoint_suffix", type=str, default="")
    p.add_argument("--embeddings_subdir", type=str, default=None)
    p.add_argument("--representation_source", type=str, default="post_embedding")
    p.add_argument("--extract_splits", type=str, default="train,val,test")
    ns = p.parse_args(argv)
    set_seed(ns.seed)
    with open("data_config.json") as f:
        data_config = json.load(f)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(ns, data_config)
    model, transform = _build_model_for_data(ns, te_data, tr_data, val_data, tr_inds, val_inds, te_inds, device)
    load_checkpoint_weights(model, device, ns, data_config)
    tr_loader, val_loader, te_loader = get_loaders(
        tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform, ns, train_shuffle=False
    )
    bn_info = recalibrate_bn_train_only(model, tr_loader, tr_data, device, ns, max_batches=None)
    if not bn_info["learned_unchanged"]:
        raise SystemExit("learned params changed")
    if not bn_info["bn_stats_changed"]:
        logging.warning("BN stats hash unchanged after full train pass")

    # Extract all splits with frozen/eval model (BN stats already updated)
    model.eval()
    out_dir = EMBED_ROOT / sub
    assert_allowed_write(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_checksums = {}
    for split_name, loader, inds, graph in (
        ("train", tr_loader, tr_inds, tr_data),
        ("val", val_loader, val_inds, val_data),
        ("test", te_loader, te_inds, te_data),
    ):
        expected = expected_seed_edge_ids(loader.data, inds, hetero=True)
        edge_ids, z, y = extract_seed_embeddings_hetero(
            loader, inds, model, graph, device, ns,
            representation_source="post_embedding", pre_dim=None, emb_dim=128, head_spec=None,
        )
        log_seed_coverage(edge_ids, expected, split_name)
        save_embedding_split_npz(out_dir / f"{split_name}.npz", z, y, edge_ids)
        eid_np = edge_ids.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        split_checksums[split_name] = {
            "num_rows": int(eid_np.shape[0]),
            "num_positives": int(y_np.sum()),
            "positive_rate": float(y_np.mean()) if eid_np.shape[0] else float("nan"),
            "edge_id_sum": int(eid_np.astype("int64").sum()),
        }
    meta = {
        "unique_name": sub,
        "source_unique_name": unique,
        "checkpoint_path": str(ckpt),
        "checkpoint_sha256": sha,
        "bn_recalibration": bn_info,
        "normalization_protocol": norm,
        "bn_protocol": "target_train_only_running_stats",
        "train_fit_edge_znorm": train_fit,
        "split_checksums": split_checksums,
        "encoder_role": enc_role,
    }
    write_json(out_dir / "meta.json", meta)
    write_json(CELLS / f"{sub}_bn_meta.json", meta)
    run_logistic_pair(
        out_dir,
        label_prefix=sub,
        normalization=norm,
        bn_protocol="target_train_only_running_stats",
        checkpoint_path=str(ckpt),
        checkpoint_sha=sha,
        encoder_role=enc_role,
    )
    return 0


def cmd_amlworld_eval_a(args: argparse.Namespace) -> int:
    """Locked Small-HI frozen eval for encoder A: pre-3h H and H+X+TF MLP."""
    import embedding_extraction as ee
    import importlib.util

    sha = sha256_file(CKPT_A)
    if sha != EXPECTED_SHA_A:
        raise SystemExit("A hash mismatch")

    emb_subdir = "A_corrected_amlworld_pre3h"
    argv = base_extract_argv(
        data="Small-HI",
        unique=UNIQUE_A,
        emb_dir=str(EMBED_ROOT),
        emb_subdir=emb_subdir,
        seed=ENCODER_SEED,
        train_fit=False,
        representation_source="pre_embedding_3h",
        batch_size=int(args.batch_size),
    )
    # Small-HI typically uses per-graph / default z-norm (no train_fit)
    p = create_parser()
    p.add_argument("--embeddings_dir", type=str, default="embeddings")
    p.add_argument("--random_init", action="store_true")
    p.add_argument("--checkpoint_suffix", type=str, default="")
    p.add_argument("--embeddings_subdir", type=str, default=None)
    p.add_argument("--representation_source", type=str, default="post_embedding")
    p.add_argument("--extract_splits", type=str, default="train,val,test")
    ns = p.parse_args(argv)
    set_seed(ns.seed)
    with open("data_config.json") as f:
        data_config = json.load(f)
    tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(ns, data_config)
    out = ee.run_embedding_extraction(
        tr_data, val_data, te_data, tr_inds, val_inds, te_inds, ns, data_config
    )
    # pre_embedding_3h nests under subdir
    emb_dir = out if (out / "train.npz").is_file() else out / "pre_embedding_3h"
    if not (emb_dir / "train.npz").is_file():
        # embedding_extraction routes pre_embedding into subdir under embeddings_subdir
        cand = EMBED_ROOT / emb_subdir / "pre_embedding_3h"
        if cand.is_dir():
            emb_dir = cand
    assert (emb_dir / "train.npz").is_file(), emb_dir

    spec = importlib.util.spec_from_file_location(
        "probe_feature_ablation", ROOT / "scripts" / "probe_feature_ablation.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["probe_feature_ablation"] = mod
    spec.loader.exec_module(mod)
    df, df_train, tr_ids, va_ids, te_ids, dspec = mod.load_dataset_frames("Small-HI", str(ROOT / "data_config.json"))
    y_all = df[dspec.label_col].to_numpy().astype(np.int64)
    x_raw, _, _, _ = mod.build_full_feature_matrix(
        df, df_train, ("edge_native",), categorical_encoding="one_hot"
    )
    tf_feat = np.load(TF_CACHE / "features.npy").astype(np.float32)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    def train_mlp_best_val_auprc(x_tr, y_tr, x_va, y_va, x_te, y_te):
        torch.manual_seed(2)
        np.random.seed(2)
        model = PaperStyleMLP(int(x_tr.shape[1])).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=MLP_LR)
        x_t = torch.from_numpy(x_tr.astype(np.float32))
        y_t = torch.from_numpy(y_tr.astype(np.float32))
        n = x_tr.shape[0]
        best_auprc, best_state, best_ep = -1.0, None, -1
        for ep in range(MLP_EPOCHS):
            model.train()
            perm = np.random.RandomState(2 * 1009 + ep).permutation(n)
            for start in range(0, n, MLP_BS):
                idx = perm[start : start + MLP_BS]
                opt.zero_grad(set_to_none=True)
                loss = nn.functional.binary_cross_entropy_with_logits(
                    model(x_t[idx].to(device)), y_t[idx].to(device)
                )
                loss.backward()
                opt.step()
            pva = _predict_proba(model, x_va, batch_size=MLP_BS, device=device)
            auprc = float(average_precision_score(y_va, pva))
            if auprc > best_auprc + 1e-12:
                best_auprc = auprc
                best_ep = ep + 1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        model.to(device)
        pva = _predict_proba(model, x_va, batch_size=MLP_BS, device=device)
        pte = _predict_proba(model, x_te, batch_size=MLP_BS, device=device)
        thr = tune_thr_max_f1(y_va, pva)
        return {
            "best_epoch_by_val_auprc": best_ep,
            "best_val_auprc": best_auprc,
            "validation_selected_threshold": thr,
            "threshold_0.5": metrics_block(y_te, pte, 0.5),
            "threshold_val_selected": metrics_block(y_te, pte, thr),
            "val_ranking": {
                "auroc": float(roc_auc_score(y_va, pva)),
                "auprc": float(average_precision_score(y_va, pva)),
            },
            "learner": "PaperStyleMLP",
            "note": "secondary_to_paysim_H_only_logistic_primary",
        }

    stacks_out = {}
    for stack_name in ("pre3h_H_only", "pre3h_HxXTF"):
        feats = {}
        for sp, expected_ids in (("train", tr_ids), ("val", va_ids), ("test", te_ids)):
            z, y, ids = load_embedding_npz(emb_dir / f"{sp}.npz")
            if not np.array_equal(y, y_all[ids]):
                raise SystemExit(f"label mismatch {sp}")
            if stack_name == "pre3h_H_only":
                mat = z.astype(np.float32)
            else:
                mat = np.concatenate([z, x_raw[ids], tf_feat[ids]], axis=1).astype(np.float32)
            feats[sp] = {"X": mat, "y": y, "ids": ids}
        scaler = StandardScaler()
        x_tr = scaler.fit_transform(feats["train"]["X"]).astype(np.float32)
        x_va = scaler.transform(feats["val"]["X"]).astype(np.float32)
        x_te = scaler.transform(feats["test"]["X"]).astype(np.float32)
        stacks_out[stack_name] = train_mlp_best_val_auprc(
            x_tr, feats["train"]["y"], x_va, feats["val"]["y"], x_te, feats["test"]["y"]
        )
        stacks_out[stack_name]["feature_dim"] = int(x_tr.shape[1])
        stacks_out[stack_name]["ids"] = {sp: ids_hash(feats[sp]["ids"]) for sp in feats}

    report = {
        "label": "amlworld_eval_A_corrected_no_preserve",
        "checkpoint_path": str(CKPT_A),
        "checkpoint_sha256": sha,
        "embeddings_dir": str(emb_dir),
        "dataset": "Small-HI",
        "stacks": stacks_out,
        "primary_diagnostic_note": "AMLWorld locked MLP is secondary; PaySim H-only logistic is primary for preserve/norm ablation",
    }
    out = CELLS / "amlworld_A_corrected_mlp.json"
    assert_allowed_write(out)
    write_json(out, report)
    logging.info("Wrote %s", out)
    return 0


def _load_cell(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def cmd_aggregate(_: argparse.Namespace) -> int:
    """CPU aggregation: verify cells, write final MD/JSON, append registry rows."""
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    smoke = _load_cell(RESULT_ROOT / "smoke.json")
    if not smoke.get("passed"):
        raise SystemExit("smoke.json missing or not passed")

    # Verify reusable artifacts
    reuse = {}
    meta_a = _load_cell(REUSE_A_PERGRAPH_EMB / "meta.json")
    if UNIQUE_A not in str(meta_a.get("source_unique_name") or meta_a.get("checkpoint_path") or ""):
        # source_unique_name should match
        if meta_a.get("source_unique_name") != UNIQUE_A:
            raise SystemExit(f"reuse A unique mismatch: {meta_a.get('source_unique_name')}")
    cell_a = _load_cell(REUSE_A_PERGRAPH_CELL)
    reuse["A_pergraph"] = {
        "embeddings": str(REUSE_A_PERGRAPH_EMB),
        "cell": str(REUSE_A_PERGRAPH_CELL),
        "checkpoint_path": meta_a.get("checkpoint_path"),
        "verified_unique": meta_a.get("source_unique_name"),
        "test_auroc": cell_a.get("threshold_0.5", {}).get("auroc"),
        "normalization": "per_graph_edge_znorm",
        "reuse": True,
    }

    role_b = _load_cell(REUSE_B_TRAINFIT_ROLE)
    if role_b.get("load", {}).get("sha256") != EXPECTED_SHA_B:
        raise SystemExit("reuse B role sha mismatch")
    if not role_b.get("integrity", {}).get("train_fit_edge_znorm"):
        raise SystemExit("reuse B not train_fit")
    cell_b = _load_cell(REUSE_B_TRAINFIT_CELL)
    # ID hash from cell
    reuse["B_trainfit"] = {
        "embeddings": str(REUSE_B_TRAINFIT_EMB),
        "cell": str(REUSE_B_TRAINFIT_CELL),
        "role": str(REUSE_B_TRAINFIT_ROLE),
        "checkpoint_sha256": EXPECTED_SHA_B,
        "test_auroc": cell_b.get("threshold_0.5", {}).get("auroc"),
        "normalization": "paysim_train_fit_edge_znorm",
        "reuse": True,
    }

    required_new = [
        "A_corrected_trainfit_post128_logistic_cw_model.json",
        "A_corrected_trainfit_post128_logistic_cw_none.json",
        "B_dplus_pergraph_post128_logistic_cw_model.json",
        "B_dplus_pergraph_post128_logistic_cw_none.json",
        "random_edge8_trainfit_post128_logistic_cw_model.json",
        "random_edge8_pergraph_post128_logistic_cw_model.json",
        "A_corrected_trainfit_bnrecal_post128_logistic_cw_model.json",
        "B_dplus_pergraph_bnrecal_post128_logistic_cw_model.json",
        "amlworld_A_corrected_mlp.json",
    ]
    cells = {}
    missing = []
    for name in required_new:
        p = CELLS / name
        if not p.is_file():
            missing.append(name)
        else:
            cells[name] = _load_cell(p)
    if missing:
        raise SystemExit(f"refuse aggregate; missing cells: {missing}")

    # Pointer JSONs for reused (do not copy large embeds)
    write_json(CELLS / "reuse_A_pergraph_pointer.json", reuse["A_pergraph"])
    write_json(CELLS / "reuse_B_trainfit_pointer.json", reuse["B_trainfit"])

    def auroc(name: str) -> Optional[float]:
        return cells.get(name, {}).get("threshold_0.5", {}).get("auroc")

    table = {
        "A_pergraph_reused": reuse["A_pergraph"]["test_auroc"],
        "A_trainfit_new": auroc("A_corrected_trainfit_post128_logistic_cw_model.json"),
        "B_pergraph_new": auroc("B_dplus_pergraph_post128_logistic_cw_model.json"),
        "B_trainfit_reused": reuse["B_trainfit"]["test_auroc"],
        "random_trainfit": auroc("random_edge8_trainfit_post128_logistic_cw_model.json"),
        "random_pergraph": auroc("random_edge8_pergraph_post128_logistic_cw_model.json"),
        "A_trainfit_bnrecal": auroc("A_corrected_trainfit_bnrecal_post128_logistic_cw_model.json"),
        "B_pergraph_bnrecal": auroc("B_dplus_pergraph_bnrecal_post128_logistic_cw_model.json"),
    }

    # 2x2 deltas
    analysis = {
        "preserve_effect_under_trainfit": None
        if table["B_trainfit_reused"] is None or table["A_trainfit_new"] is None
        else float(table["B_trainfit_reused"]) - float(table["A_trainfit_new"]),
        "preserve_effect_under_pergraph": None
        if table["B_pergraph_new"] is None or table["A_pergraph_reused"] is None
        else float(table["B_pergraph_new"]) - float(table["A_pergraph_reused"]),
        "norm_effect_under_A": None
        if table["A_trainfit_new"] is None or table["A_pergraph_reused"] is None
        else float(table["A_trainfit_new"]) - float(table["A_pergraph_reused"]),
        "norm_effect_under_B": None
        if table["B_trainfit_reused"] is None or table["B_pergraph_new"] is None
        else float(table["B_trainfit_reused"]) - float(table["B_pergraph_new"]),
        "bnrecal_delta_A_trainfit": None
        if table["A_trainfit_bnrecal"] is None or table["A_trainfit_new"] is None
        else float(table["A_trainfit_bnrecal"]) - float(table["A_trainfit_new"]),
        "bnrecal_delta_B_pergraph": None
        if table["B_pergraph_bnrecal"] is None or table["B_pergraph_new"] is None
        else float(table["B_pergraph_bnrecal"]) - float(table["B_pergraph_new"]),
    }

    aml = cells["amlworld_A_corrected_mlp.json"]

    payload = {
        "title": "paysim_preserve_normalization_ablation",
        "primary_metric": "PaySim post-128 H-only logistic AUROC/AUPRC",
        "checkpoints": {
            "A": {"path": str(CKPT_A), "sha256": EXPECTED_SHA_A, "preserve_seed_edges": False},
            "B": {"path": str(CKPT_B), "sha256": EXPECTED_SHA_B, "preserve_seed_edges": True},
        },
        "reuse": reuse,
        "auroc_table_cw_model": table,
        "analysis_deltas_auroc": analysis,
        "amlworld_A": {
            "pre3h_H_only": aml["stacks"]["pre3h_H_only"]["threshold_0.5"],
            "pre3h_HxXTF": aml["stacks"]["pre3h_HxXTF"]["threshold_0.5"],
        },
        "cells_dir": str(CELLS),
        "smoke": {"path": str(RESULT_ROOT / "smoke.json"), "passed": True},
        "no_encoder_training": True,
        "historical_artifacts_overwritten": False,
    }
    final_json = ROOT / "results/diagnostics/paysim_preserve_normalization_ablation.json"
    final_md = ROOT / "notes/paysim_preserve_normalization_ablation.md"
    write_json(final_json, payload)

    def fmt(x):
        return "NA" if x is None else f"{x:.4f}"

    md = f"""# PaySim preserve vs normalization ablation

No encoder training. Primary diagnostic: **post-128 H-only logistic** (class_weight=model).

## 2×2 AUROC (cw=model @0.5)

| | per-graph z-norm | PaySim train-fit z-norm |
|--|-----------------:|------------------------:|
| **A corrected, no preserve** | {fmt(table['A_pergraph_reused'])} (reused) | {fmt(table['A_trainfit_new'])} (new) |
| **B D+ + preserve** | {fmt(table['B_pergraph_new'])} (new) | {fmt(table['B_trainfit_reused'])} (reused) |
| **random edge_dim=8** | {fmt(table['random_pergraph'])} | {fmt(table['random_trainfit'])} |

## BN recalibration (target-train-only running stats)

| Cell | AUROC |
|------|------:|
| A train-fit + BN recal | {fmt(table['A_trainfit_bnrecal'])} |
| B per-graph + BN recal | {fmt(table['B_pergraph_bnrecal'])} |

Deltas: A {fmt(analysis['bnrecal_delta_A_trainfit'])}, B {fmt(analysis['bnrecal_delta_B_pergraph'])}

## AMLWorld locked eval of A (secondary)

- pre-3h H: AUROC {aml['stacks']['pre3h_H_only']['threshold_0.5']['auroc']:.4f} AUPRC {aml['stacks']['pre3h_H_only']['threshold_0.5']['auprc']:.4f}
- pre-3h H+X+TF: AUROC {aml['stacks']['pre3h_HxXTF']['threshold_0.5']['auroc']:.4f} AUPRC {aml['stacks']['pre3h_HxXTF']['threshold_0.5']['auprc']:.4f}

## Deltas

- preserve | train-fit: {fmt(analysis['preserve_effect_under_trainfit'])}
- preserve | per-graph: {fmt(analysis['preserve_effect_under_pergraph'])}
- norm | A: {fmt(analysis['norm_effect_under_A'])}
- norm | B: {fmt(analysis['norm_effect_under_B'])}

## Artifacts

- `{final_md}`
- `{final_json}`
- cells: `{CELLS}/`
- embeds: `{EMBED_ROOT}/`
"""
    final_md.write_text(md)

    # Append diagnostic rows to experiment registry (JSON list if present)
    reg_path = ROOT / "results/diagnostics/thesis_experiment_registry.json"
    if reg_path.is_file():
        reg = json.loads(reg_path.read_text())
        rows = reg if isinstance(reg, list) else reg.get("rows") or reg.get("experiments") or []
        if not isinstance(rows, list):
            rows = []
        new_rows = [
            {
                "run_id": "paysim_preserve_norm_ablation|diagnostic|A_trainfit|logistic_H",
                "dataset": "PaySim",
                "status": "diagnostic",
                "preserve_seed_edges": False,
                "train_fit_edge_znorm": True,
                "test_auroc": table["A_trainfit_new"],
                "source": str(final_json),
            },
            {
                "run_id": "paysim_preserve_norm_ablation|diagnostic|B_pergraph|logistic_H",
                "dataset": "PaySim",
                "status": "diagnostic",
                "preserve_seed_edges": True,
                "train_fit_edge_znorm": False,
                "test_auroc": table["B_pergraph_new"],
                "source": str(final_json),
            },
        ]
        # Avoid duplicating if re-run
        existing_ids = {r.get("run_id") for r in rows if isinstance(r, dict)}
        for nr in new_rows:
            if nr["run_id"] not in existing_ids:
                rows.append(nr)
        if isinstance(reg, list):
            write_json(reg_path, rows)
        else:
            key = "rows" if "rows" in reg else ("experiments" if "experiments" in reg else None)
            if key:
                reg[key] = rows
                write_json(reg_path, reg)
            else:
                write_json(RESULT_ROOT / "registry_rows_appended.json", new_rows)

    # notes registry markdown append
    notes_reg = ROOT / "notes/thesis_experiment_registry.md"
    if notes_reg.is_file():
        with notes_reg.open("a") as f:
            f.write(
                "\n\n## PaySim preserve/normalization ablation (diagnostic)\n\n"
                f"- See `{final_md}` / `{final_json}`\n"
                f"- A train-fit AUROC={fmt(table['A_trainfit_new'])}; "
                f"B per-graph AUROC={fmt(table['B_pergraph_new'])}\n"
            )

    logging.info("Wrote %s and %s", final_json, final_md)
    return 0


def main() -> None:
    logger_setup()
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("smoke")
    ps.add_argument("--batch_size", type=int, default=4096)
    ps.add_argument("--max_batches", type=int, default=2)
    ps.set_defaults(func=cmd_smoke)

    pe = sub.add_parser("extract_probe")
    pe.add_argument("--cell", required=True, choices=("A_trainfit", "B_pergraph", "random_both"))
    pe.add_argument("--batch_size", type=int, default=4096)
    pe.set_defaults(func=cmd_extract_probe)

    pb = sub.add_parser("bn_recal_probe")
    pb.add_argument("--role", required=True, choices=("A", "B"))
    pb.add_argument("--batch_size", type=int, default=4096)
    pb.set_defaults(func=cmd_bn_recal_probe)

    pa = sub.add_parser("amlworld_eval_a")
    pa.add_argument("--batch_size", type=int, default=8192)
    pa.add_argument("--device", default="cuda:0")
    pa.set_defaults(func=cmd_amlworld_eval_a)

    pg = sub.add_parser("aggregate")
    pg.set_defaults(func=cmd_aggregate)

    args = p.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
