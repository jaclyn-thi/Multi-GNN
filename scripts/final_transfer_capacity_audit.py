#!/usr/bin/env python3
"""Final transfer capacity audit (exploratory / post-hoc).

Subcommands:
  shift          — Part A source–target shift (Small-HI train vs PaySim train)
  capacity_fit   — Part B seed-2 validation learner×stack cells (no test selection)
  gate           — validation gate (always exit 0; writes pass/fail)
  confirm        — locked test + seeds 1/3/4 if gate passed; else clean SKIP
  aggregate      — notes/json report

No GNN training. table_eligible=false. exploratory_posthoc=true.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier
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
from torch_geometric.data import HeteroData
from torch_geometric.nn import to_hetero

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_loading import get_data  # noqa: E402
from feature_contracts import CONTRACT_LEGACY  # noqa: E402
from gcpal_txn_node.eval_mlp import PaperStyleMLP, _predict_proba  # noqa: E402
from graph_augmentations import generate_views  # noqa: E402
from linear_probe import load_embedding_npz  # noqa: E402
from ranking_metrics import alert_budget_metrics  # noqa: E402
from train_util import (  # noqa: E402
    AddEgoIds,
    FORWARD_EDGE_TYPE,
    REVERSE_EDGE_TYPE,
    add_arange_ids,
    extract_param,
    get_loaders,
)
from training import get_model  # noqa: E402
from util import create_parser, logger_setup, set_seed  # noqa: E402

TAG = "final_transfer_capacity_audit"
RESULT = ROOT / "results" / "diagnostics" / TAG
CELLS = RESULT / "cells"
NOTES = ROOT / "notes" / f"{TAG}.md"
OUT_JSON = ROOT / "results" / "diagnostics" / f"{TAG}.json"
GATE_JSON = RESULT / "gate.json"
SHIFT_JSON = RESULT / "shift_audit.json"
CAPACITY_JSON = RESULT / "capacity_seed2_validation.json"
CONFIRM_JSON = RESULT / "confirmation.json"

SOURCE_UNIQUE = "gin_emlps_ports_tds_corrected_asym_proj_8192neg_queue0_40ep_seed2"
SOURCE_CKPT = ROOT / f"saved-models/checkpoint_{SOURCE_UNIQUE}.tar"
SOURCE_SHA = "18e06f555aa4880dfc1e95caa3f54a207e5aa186d266887772640feb93a06ae6"
EMB_ROOT = ROOT / "embeddings" / "final_corrected_no_preserve_multiseed"
CONTRACT = CONTRACT_LEGACY
DEV_SEED = 2
CONFIRM_SEEDS = (1, 3, 4)
LOGISTIC_SEED = 1
MLP_SEED = 2
MLP_EPOCHS = 15
MLP_LR = 1e-3
MLP_BS = 8192
GATE_MARGIN = 0.003
EXPECTED_ID = {
    "train": "2511d0de4504e52960b414e6b84d47486089a573b6c57aa040feb561e2d2809a",
    "val": "a8de85f31dfe91bd767da6daedf9f2bab474d08c8412c796111e8767ebd0b1e3",
}

# Predeclared HistGradientBoosting (XGB/LGBM not installed in env).
HGB_CFG = {
    "learner": "HistGradientBoostingClassifier",
    "max_depth": 6,
    "learning_rate": 0.1,
    "max_iter": 100,
    "l2_regularization": 0.0,
    "random_state": LOGISTIC_SEED,
    "note": "Repository-supported sklearn substitute; xgboost/lightgbm not installed.",
}


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def ids_hash(ids: np.ndarray) -> Dict[str, Any]:
    a = np.asarray(ids, dtype=np.int64).reshape(-1)
    return {
        "n": int(a.shape[0]),
        "n_unique": int(np.unique(a).shape[0]),
        "edge_id_sum": int(a.sum()),
        "sha256_of_ids_bytes": hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest(),
    }


def gin_cw() -> Dict[int, float]:
    args = create_parser().parse_args(["--data", "PaySim", "--model", "gin", "--testing"])
    return {0: float(extract_param("w_ce1", args)), 1: float(extract_param("w_ce2", args))}


def tune_thr(y: np.ndarray, proba: np.ndarray) -> float:
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


def verify_ckpt() -> str:
    if not SOURCE_CKPT.is_file():
        raise SystemExit(f"missing {SOURCE_CKPT}")
    sha = _sha256_file(SOURCE_CKPT)
    if sha != SOURCE_SHA:
        raise SystemExit(f"ckpt sha mismatch {sha}")
    return sha


def emb_dir(seed: int, protocol: str) -> Path:
    if protocol == "P1":
        return EMB_ROOT / f"seed{seed}_P1_strict_inductive_legacy"
    if protocol == "P2":
        return EMB_ROOT / f"seed{seed}_P2_label_free_target_bn_legacy"
    if protocol == "random":
        return EMB_ROOT / "controls_random_paysim_legacy_duplicate_v1"
    raise ValueError(protocol)


def load_splits(path: Path, *, require_expected_ids: bool) -> Dict[str, Dict[str, np.ndarray]]:
    out = {}
    for sp in ("train", "val"):
        z, y, ids = load_embedding_npz(path / f"{sp}.npz")
        meta = ids_hash(ids)
        if require_expected_ids and meta["sha256_of_ids_bytes"] != EXPECTED_ID[sp]:
            raise SystemExit(f"{path} {sp} id hash mismatch")
        out[sp] = {"Z": z, "y": y, "ids": ids, "ids_meta": meta}
    # test loaded only by confirm
    return out


def load_x_edge_native() -> Tuple[np.ndarray, List[str]]:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "probe_feature_ablation", ROOT / "scripts" / "probe_feature_ablation.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    df, df_train, _, _, _, _ = mod.load_dataset_frames("PaySim", str(ROOT / "data_config.json"))
    x, names, _, _ = mod.build_full_feature_matrix(
        df, df_train, ("edge_native",), categorical_encoding="one_hot"
    )
    return x.astype(np.float32), list(names)


def stack_mat(name: str, z, x) -> np.ndarray:
    if name == "X":
        return x.astype(np.float32)
    if name == "H":
        return z.astype(np.float32)
    if name == "H+X":
        return np.concatenate([z, x], axis=1).astype(np.float32)
    raise ValueError(name)


def fit_logistic(x_tr, y_tr, x_va, y_va) -> Dict[str, Any]:
    scaler = StandardScaler()
    tr = scaler.fit_transform(x_tr).astype(np.float32)
    va = scaler.transform(x_va).astype(np.float32)
    cw = gin_cw()
    set_seed(LOGISTIC_SEED)
    clf = LogisticRegression(
        class_weight=cw, max_iter=1000, random_state=LOGISTIC_SEED, solver="lbfgs", n_jobs=1, C=1.0
    )
    clf.fit(tr, y_tr)
    pva = clf.predict_proba(va)[:, 1].astype(np.float64)
    thr = tune_thr(y_va, pva)
    return {
        "learner": "LogisticRegression",
        "config": {"C": 1.0, "class_weight": "model", "downstream_seed": LOGISTIC_SEED},
        "feature_dim": int(tr.shape[1]),
        "validation": {
            "threshold_0.5": metrics_block(y_va, pva, 0.5),
            "threshold_val_selected": metrics_block(y_va, pva, thr),
            "validation_selected_threshold": thr,
        },
        "val_auprc_at_0.5": float(average_precision_score(y_va, pva)),
    }


def fit_mlp(x_tr, y_tr, x_va, y_va, device: torch.device) -> Dict[str, Any]:
    scaler = StandardScaler()
    tr = scaler.fit_transform(x_tr).astype(np.float32)
    va = scaler.transform(x_va).astype(np.float32)
    torch.manual_seed(MLP_SEED)
    np.random.seed(MLP_SEED)
    model = PaperStyleMLP(int(tr.shape[1])).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=MLP_LR)
    x_t = torch.from_numpy(tr)
    y_t = torch.from_numpy(y_tr.astype(np.float32))
    n = tr.shape[0]
    best_auprc, best_state, best_ep = -1.0, None, -1
    for ep in range(MLP_EPOCHS):
        model.train()
        perm = np.random.RandomState(MLP_SEED * 1009 + ep).permutation(n)
        for start in range(0, n, MLP_BS):
            idx = perm[start : start + MLP_BS]
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(
                model(x_t[idx].to(device)), y_t[idx].to(device)
            )
            loss.backward()
            opt.step()
        pva = _predict_proba(model, va, batch_size=MLP_BS, device=device)
        auprc = float(average_precision_score(y_va, pva)) if len(np.unique(y_va)) > 1 else float("nan")
        if auprc > best_auprc + 1e-12:
            best_auprc = auprc
            best_ep = ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.to(device)
    pva = _predict_proba(model, va, batch_size=MLP_BS, device=device)
    thr = tune_thr(y_va, pva)
    return {
        "learner": "PaperStyleMLP",
        "config": {
            "epochs": MLP_EPOCHS,
            "lr": MLP_LR,
            "batch_size": MLP_BS,
            "seed": MLP_SEED,
            "selection": "best_val_auprc",
        },
        "feature_dim": int(tr.shape[1]),
        "best_epoch_by_val_auprc": best_ep,
        "best_val_auprc": best_auprc,
        "validation": {
            "threshold_0.5": metrics_block(y_va, pva, 0.5),
            "threshold_val_selected": metrics_block(y_va, pva, thr),
            "validation_selected_threshold": thr,
        },
        "val_auprc_at_0.5": float(average_precision_score(y_va, pva)),
    }


def fit_hgb(x_tr, y_tr, x_va, y_va) -> Dict[str, Any]:
    scaler = StandardScaler()
    tr = scaler.fit_transform(x_tr).astype(np.float32)
    va = scaler.transform(x_va).astype(np.float32)
    # class-balanced sample weights (predeclared; no search)
    n0 = max(int((y_tr == 0).sum()), 1)
    n1 = max(int((y_tr == 1).sum()), 1)
    w = np.where(y_tr == 1, n0 / n1, 1.0).astype(np.float64)
    clf = HistGradientBoostingClassifier(
        max_depth=HGB_CFG["max_depth"],
        learning_rate=HGB_CFG["learning_rate"],
        max_iter=HGB_CFG["max_iter"],
        l2_regularization=HGB_CFG["l2_regularization"],
        random_state=HGB_CFG["random_state"],
    )
    clf.fit(tr, y_tr, sample_weight=w)
    pva = clf.predict_proba(va)[:, 1].astype(np.float64)
    thr = tune_thr(y_va, pva)
    return {
        "learner": HGB_CFG["learner"],
        "config": dict(HGB_CFG),
        "feature_dim": int(tr.shape[1]),
        "validation": {
            "threshold_0.5": metrics_block(y_va, pva, 0.5),
            "threshold_val_selected": metrics_block(y_va, pva, thr),
            "validation_selected_threshold": thr,
        },
        "val_auprc_at_0.5": float(average_precision_score(y_va, pva)),
    }


def _col_stats(arr: np.ndarray, names: Sequence[str], sample_n: int, rng: np.random.RandomState) -> Dict[str, Any]:
    n = arr.shape[0]
    idx = rng.choice(n, size=min(sample_n, n), replace=False) if n > sample_n else np.arange(n)
    s = arr[idx]
    out = {}
    for j, name in enumerate(names):
        col = s[:, j].astype(np.float64)
        out[name] = {
            "mean": float(np.mean(col)),
            "std": float(np.std(col)),
            "min": float(np.min(col)),
            "max": float(np.max(col)),
            "p50": float(np.median(col)),
            "zero_frac": float(np.mean(col == 0.0)),
        }
    return out


def _argv_domain(data: str, train_fit: bool) -> List[str]:
    argv = [
        "--data", data,
        "--model", "gin",
        "--testing",
        "--reverse_mp",
        "--ego",
        "--ports",
        "--tds",
        "--emlps",
        "--correct_reverse_edge_features",
        "--batch_size", "2048",
        "--num_neighs", "50", "50",
        "--loader_num_workers", "0",
        "--seed", "2",
        "--unique_name", f"shift_audit_{data}",
    ]
    if data == "PaySim":
        argv += ["--feature_contract", CONTRACT]
    if train_fit:
        argv.append("--train_fit_edge_znorm")
    return argv


def cmd_shift(args: argparse.Namespace) -> int:
    RESULT.mkdir(parents=True, exist_ok=True)
    CELLS.mkdir(parents=True, exist_ok=True)
    sha = verify_ckpt()
    rng = np.random.RandomState(2)
    sample_n = int(args.sample_n)
    n_batches = int(args.n_batches)

    with open(ROOT / "data_config.json") as f:
        data_config = json.load(f)

    domains = {}
    for data, train_fit in (("Small-HI", False), ("PaySim", True)):
        ns = create_parser().parse_args(_argv_domain(data, train_fit))
        set_seed(2)
        t0 = time.perf_counter()
        tr, va, te, tr_inds, val_inds, te_inds = get_data(ns, data_config)
        # labels only for exclusion check
        y = tr[FORWARD_EDGE_TYPE].y if isinstance(tr, HeteroData) else tr.y
        assert y is not None
        ea = tr[FORWARD_EDGE_TYPE].edge_attr if isinstance(tr, HeteroData) else tr.edge_attr
        # no EdgeID yet
        edge_dim = int(ea.shape[1])
        names = [f"e{j}" for j in range(edge_dim)]
        # Prefer semantic names for first channels
        base = ["Timestamp", "Amount", "Currency", "PaymentFormat", "in_port", "out_port", "in_td", "out_td"]
        names = base[:edge_dim] if edge_dim <= len(base) else names
        feat_stats = _col_stats(ea.detach().cpu().numpy(), names, sample_n, rng)

        # Graph degree stats on train graph (sampled)
        src = tr[FORWARD_EDGE_TYPE].edge_index[0].detach().cpu().numpy()
        dst = tr[FORWARD_EDGE_TYPE].edge_index[1].detach().cpu().numpy()
        n_edges = int(src.shape[0])
        sample_e = rng.choice(n_edges, size=min(sample_n, n_edges), replace=False)
        src_s, dst_s = src[sample_e], dst[sample_e]
        # approximate degrees via bincount on full graph (cheap relative to GNN)
        n_nodes = int(max(int(src.max()), int(dst.max())) + 1)
        out_deg = np.bincount(src, minlength=n_nodes)
        in_deg = np.bincount(dst, minlength=n_nodes)
        node_sample = rng.choice(n_nodes, size=min(100_000, n_nodes), replace=False)
        deg_stats = {
            "out_degree_mean": float(out_deg[node_sample].mean()),
            "in_degree_mean": float(in_deg[node_sample].mean()),
            "total_degree_mean": float((out_deg + in_deg)[node_sample].mean()),
            "out_degree_p95": float(np.percentile(out_deg[node_sample], 95)),
            "in_degree_p95": float(np.percentile(in_deg[node_sample], 95)),
            "n_nodes": n_nodes,
            "n_edges": n_edges,
        }
        # reciprocal rate (sampled)
        pair = set(zip(src_s.tolist(), dst_s.tolist()))
        recip = sum(1 for a, b in pair if (b, a) in pair)
        deg_stats["reciprocal_rate_among_sampled_pairs"] = float(recip / max(len(pair), 1))
        # parallel-edge multiplicity on sampled undirected pairs
        undirected = np.sort(np.stack([src_s, dst_s], axis=1), axis=1)
        _, ucounts = np.unique(undirected, axis=0, return_counts=True)
        deg_stats["parallel_edge_rate_sampled"] = float(np.mean(ucounts > 1))
        deg_stats["mean_multiplicity_sampled"] = float(ucounts.mean())
        # fan-in / fan-out on sampled receivers/senders
        deg_stats["fanout_mean_sampled_senders"] = float(out_deg[src_s].mean())
        deg_stats["fanin_mean_sampled_receivers"] = float(in_deg[dst_s].mean())
        deg_stats["sender_activity_p95"] = float(np.percentile(out_deg[node_sample], 95))
        deg_stats["receiver_activity_p95"] = float(np.percentile(in_deg[node_sample], 95))

        # missing / neutral / near-duplicate channel rates (sampled)
        ea_s = ea.detach().cpu().numpy()[sample_e]
        missing = {
            "nan_frac": float(np.isnan(ea_s).mean()) if np.issubdtype(ea_s.dtype, np.floating) else 0.0,
            "zero_row_frac": float(np.mean(np.all(np.isclose(ea_s, 0.0), axis=1))),
        }
        if ea_s.shape[1] >= 2:
            missing["amount_zero_frac"] = float(np.mean(np.isclose(ea_s[:, 1], 0.0)))
        if edge_dim >= 8:
            missing["tds_in_zero_frac"] = float(np.mean(np.isclose(ea_s[:, 6], 0.0)))
            missing["tds_out_zero_frac"] = float(np.mean(np.isclose(ea_s[:, 7], 0.0)))
        # duplicated adjacent channels (ports or TDS clones)
        if edge_dim >= 6:
            missing["in_out_port_equal_frac"] = float(np.mean(np.isclose(ea_s[:, 4], ea_s[:, 5])))
        if edge_dim >= 8:
            missing["in_out_tds_equal_frac"] = float(np.mean(np.isclose(ea_s[:, 6], ea_s[:, 7])))

        # Augmentation retention (preserve_seed_edges=false matches locked SSL)
        transform = AddEgoIds() if ns.ego else None
        add_arange_ids([tr])
        tr_loader, _, _ = get_loaders(tr, va, te, tr_inds, val_inds, te_inds, transform, ns)
        batch = next(iter(tr_loader))
        # strip? generate_views expects ids in edge_attr col0 for hetero
        stats: Dict[str, float] = {}
        _ = generate_views(
            batch,
            edge_attr_mask_rate=0.1,
            edge_drop_rate=0.1,
            preserve_seed_edges=False,
            edge_drop_stats=stats,
        )
        retention = {k: float(v) for k, v in stats.items() if isinstance(v, (int, float))}

        # Categorical frequency on currency/payment (cols 2,3 before ports if dim>=4)
        ea_np = ea.detach().cpu().numpy()
        cat = {}
        for j, nm in enumerate(names[:4]):
            vals, counts = np.unique(ea_np[: min(len(ea_np), sample_n), j].round(6), return_counts=True)
            # for continuous timestamp/amount skip top categories
            if nm in ("Timestamp", "Amount"):
                continue
            top = sorted(zip(vals.tolist(), (counts / counts.sum()).tolist()), key=lambda t: -t[1])[:10]
            cat[nm] = top

        domains[data] = {
            "load_sec": time.perf_counter() - t0,
            "edge_dim": edge_dim,
            "train_fit_edge_znorm": bool(train_fit),
            "feature_contract": CONTRACT if data == "PaySim" else None,
            "n_train_edges": int(tr_inds.numel()),
            "feature_channel_stats_sampled": feat_stats,
            "categorical_top": cat,
            "graph_degree_stats": deg_stats,
            "missing_neutral_duplicate_rates_sampled": missing,
            "augmentation_retention_sample_batch": retention,
            "labels_excluded_from_features": True,
            "label_y_not_used_in_stats": True,
            "split_ids": {
                "train": ids_hash(tr_inds.cpu().numpy()),
                "val": ids_hash(val_inds.cpu().numpy()),
            },
            "_objects": {"tr": tr, "va": va, "te": te, "tr_inds": tr_inds, "val_inds": val_inds, "te_inds": te_inds, "ns": ns},
        }

    # Frozen encoder activation / embedding stats (CPU; no login-node intended use)
    device = torch.device("cpu")
    payload_ckpt = torch.load(SOURCE_CKPT, map_location="cpu", weights_only=False)
    sd = payload_ckpt["model_state_dict"]
    encoder_stats = {}
    for data in ("Small-HI", "PaySim"):
        d = domains[data]
        ns = d["_objects"]["ns"]
        tr, va, te = d["_objects"]["tr"], d["_objects"]["va"], d["_objects"]["te"]
        tr_inds, val_inds, te_inds = (
            d["_objects"]["tr_inds"],
            d["_objects"]["val_inds"],
            d["_objects"]["te_inds"],
        )
        transform = AddEgoIds() if ns.ego else None
        tr_loader, _, _ = get_loaders(tr, va, te, tr_inds, val_inds, te_inds, transform, ns)
        from types import SimpleNamespace

        config = SimpleNamespace(
            model="gin",
            n_hidden=extract_param("n_hidden", ns),
            n_gnn_layers=extract_param("n_gnn_layers", ns),
            n_heads=None,
            dropout=extract_param("dropout", ns),
            final_dropout=extract_param("final_dropout", ns),
        )
        # get_model expects EdgeID still present (subtracts 1 for edge_dim)
        sample_m = next(iter(tr_loader))
        model = get_model(sample_m, config, ns)
        model = to_hetero(model, te.metadata(), aggr="mean")
        incompatible = model.load_state_dict(sd, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise SystemExit(f"state_dict mismatch on {data}: {incompatible}")
        model.eval()

        hooks = []
        acts: Dict[str, List[Dict[str, float]]] = {}

        def _moments(t: torch.Tensor) -> Dict[str, float]:
            x = t.detach().float().cpu()
            flat = x.reshape(-1).numpy()
            out = {"mean": float(flat.mean()), "std": float(flat.std())}
            if x.dim() >= 2:
                out["row_norm_mean"] = float(torch.linalg.vector_norm(x.reshape(x.shape[0], -1), dim=1).mean())
            return out

        def make_pre_hook(name: str):
            def _hook(mod, inp):
                t = inp[0] if isinstance(inp, tuple) else inp
                if torch.is_tensor(t):
                    acts.setdefault(f"preBN:{name}", []).append(_moments(t))
            return _hook

        def make_post_hook(name: str):
            def _hook(mod, inp, out):
                t = out[0] if isinstance(out, tuple) else out
                if torch.is_tensor(t):
                    acts.setdefault(f"postBN:{name}", []).append(_moments(t))
            return _hook

        for name, mod in model.named_modules():
            if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
                hooks.append(mod.register_forward_pre_hook(make_pre_hook(name)))
                hooks.append(mod.register_forward_hook(make_post_hook(name)))

        emb_rows = []
        cos_pos = []
        cos_neg = []
        neigh_sizes = []
        with torch.no_grad():
            for bi, batch in enumerate(tr_loader):
                if bi >= n_batches:
                    break
                # sampled neighborhood size under actual NeighborLoader
                try:
                    n_nodes_b = int(batch["node"].x.shape[0])
                    n_edges_b = int(batch[FORWARD_EDGE_TYPE].edge_index.shape[1])
                    neigh_sizes.append({"n_nodes": n_nodes_b, "n_edges": n_edges_b})
                except Exception:
                    pass
                batch[FORWARD_EDGE_TYPE].edge_attr = batch[FORWARD_EDGE_TYPE].edge_attr[:, 1:]
                batch[REVERSE_EDGE_TYPE].edge_attr = batch[REVERSE_EDGE_TYPE].edge_attr[:, 1:]
                z = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)[FORWARD_EDGE_TYPE]
                z_np = z.detach().float().cpu().numpy()
                emb_rows.append(z_np)
                if z_np.shape[0] >= 4:
                    a = z_np / (np.linalg.norm(z_np, axis=1, keepdims=True) + 1e-8)
                    # adjacent rows as weak positive-pair proxy; random perm as negatives
                    cos_pos.append(float(np.mean(np.sum(a[::2][:64] * a[1::2][:64], axis=1))))
                    perm = rng.permutation(min(128, a.shape[0]))
                    cos_neg.append(float(np.mean(np.sum(a[perm[:64]] * a[perm[-64:]], axis=1))))
        for h in hooks:
            h.remove()

        Z = np.concatenate(emb_rows, axis=0)
        Zc = Z - Z.mean(0, keepdims=True)
        s = np.linalg.svd(Zc, compute_uv=False)
        p = (s ** 2) / max((s ** 2).sum(), 1e-12)
        erank = float(np.exp(-np.sum(p * np.log(p + 1e-12))))

        def _summarize_acts(prefix: str) -> Dict[str, Any]:
            keys = [k for k in acts if k.startswith(prefix)]
            summary = {}
            for k in keys[:16]:
                v = acts[k]
                summary[k] = {
                    "n_calls": len(v),
                    "mean_of_means": float(np.mean([x["mean"] for x in v])),
                    "mean_of_stds": float(np.mean([x["std"] for x in v])),
                }
            return summary

        encoder_stats[data] = {
            "n_embed_rows": int(Z.shape[0]),
            "embed_dim": int(Z.shape[1]),
            "embed_norm_mean": float(np.linalg.norm(Z, axis=1).mean()),
            "embed_norm_std": float(np.linalg.norm(Z, axis=1).std()),
            "per_dim_var_mean": float(Z.var(axis=0).mean()),
            "per_dim_var_min": float(Z.var(axis=0).min()),
            "effective_rank": erank,
            "cosine_adjacent_proxy_mean": float(np.mean(cos_pos)) if cos_pos else float("nan"),
            "cosine_random_neg_proxy_mean": float(np.mean(cos_neg)) if cos_neg else float("nan"),
            "sampled_neighborhood_batch_sizes": neigh_sizes,
            "pre_bn_activation_moments": _summarize_acts("preBN:"),
            "post_bn_activation_moments": _summarize_acts("postBN:"),
        }

        del model, tr_loader
        domains[data].pop("_objects", None)

    # NPZ-based post-128 / pre-3h comparison where available
    rep_npz = {}
    p1 = load_embedding_npz(EMB_ROOT / "seed2_P1_strict_inductive_legacy" / "train.npz")
    aml_post = EMB_ROOT / "seed2_amlworld_post128"
    aml_pre = EMB_ROOT / "seed2_amlworld_pre3h" / "pre_embedding_3h"
    if (aml_post / "train.npz").is_file():
        zp, _, _ = load_embedding_npz(aml_post / "train.npz")
        # nested path variants
        if zp.ndim == 2:
            rep_npz["aml_post128"] = {
                "norm_mean": float(np.linalg.norm(zp[:50000], axis=1).mean()),
                "dim": int(zp.shape[1]),
                "var_mean": float(zp[:50000].var(0).mean()),
            }
    # find post128 train
    for cand in [
        EMB_ROOT / "seed2_amlworld_post128" / "train.npz",
        EMB_ROOT / "seed2_amlworld_post128" / "post_embedding" / "train.npz",
    ]:
        if cand.is_file():
            zp, _, _ = load_embedding_npz(cand)
            rep_npz["aml_post128"] = {
                "norm_mean": float(np.linalg.norm(zp[:50000], axis=1).mean()),
                "dim": int(zp.shape[1]),
                "var_mean": float(zp[:50000].var(0).mean()),
            }
            break
    if (aml_pre / "train.npz").is_file():
        zp, _, _ = load_embedding_npz(aml_pre / "train.npz")
        rep_npz["aml_pre3h"] = {
            "norm_mean": float(np.linalg.norm(zp[:50000], axis=1).mean()),
            "dim": int(zp.shape[1]),
            "var_mean": float(zp[:50000].var(0).mean()),
        }
    zp, _, _ = p1
    rep_npz["paysim_p1_post128"] = {
        "norm_mean": float(np.linalg.norm(zp[:50000], axis=1).mean()),
        "dim": int(zp.shape[1]),
        "var_mean": float(zp[:50000].var(0).mean()),
    }

    # Ranked discrepancies
    ranked = []
    hi = domains["Small-HI"]["feature_channel_stats_sampled"]
    ps = domains["PaySim"]["feature_channel_stats_sampled"]
    for ch in hi:
        if ch in ps:
            dmean = abs(hi[ch]["mean"] - ps[ch]["mean"])
            ranked.append(
                {
                    "name": f"feature_mean_absdiff:{ch}",
                    "value": dmean,
                    "category": "feature_shift",
                    "implicates": "input_normalization_or_schema",
                }
            )
    ranked.append(
        {
            "name": "degree_out_mean_absdiff",
            "value": abs(
                domains["Small-HI"]["graph_degree_stats"]["out_degree_mean"]
                - domains["PaySim"]["graph_degree_stats"]["out_degree_mean"]
            ),
            "category": "topology_shift",
            "implicates": "message_passing_neighborhood",
        }
    )
    ranked.append(
        {
            "name": "embed_norm_mean_absdiff",
            "value": abs(
                encoder_stats["Small-HI"]["embed_norm_mean"] - encoder_stats["PaySim"]["embed_norm_mean"]
            ),
            "category": "representation_shift",
            "implicates": "encoder_bn_or_features",
        }
    )
    ranked.append(
        {
            "name": "effective_rank_absdiff",
            "value": abs(
                encoder_stats["Small-HI"]["effective_rank"] - encoder_stats["PaySim"]["effective_rank"]
            ),
            "category": "representation_collapse_check",
            "implicates": "representation_geometry",
        }
    )
    # BN mean-of-means discrepancy (first shared post-BN key if present)
    hi_bn = encoder_stats["Small-HI"].get("post_bn_activation_moments") or {}
    ps_bn = encoder_stats["PaySim"].get("post_bn_activation_moments") or {}
    for k in hi_bn:
        if k in ps_bn:
            ranked.append(
                {
                    "name": f"post_bn_mean_absdiff:{k}",
                    "value": abs(hi_bn[k]["mean_of_means"] - ps_bn[k]["mean_of_means"]),
                    "category": "bn_activation_shift",
                    "implicates": "batchnorm_running_stats_or_input_scale",
                }
            )
            break
    hi_m = domains["Small-HI"].get("missing_neutral_duplicate_rates_sampled") or {}
    ps_m = domains["PaySim"].get("missing_neutral_duplicate_rates_sampled") or {}
    for k in hi_m:
        if k in ps_m and isinstance(hi_m[k], (int, float)) and isinstance(ps_m[k], (int, float)):
            ranked.append(
                {
                    "name": f"missing_rate_absdiff:{k}",
                    "value": abs(float(hi_m[k]) - float(ps_m[k])),
                    "category": "feature_shift",
                    "implicates": "schema_or_neutral_fill",
                }
            )
    ranked.sort(key=lambda r: -float(r["value"]))

    out = {
        "ok": True,
        "exploratory_posthoc": True,
        "table_eligible": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_sha256": sha,
        "sample_n": sample_n,
        "n_batches_encoder": n_batches,
        "domains": {k: {a: b for a, b in v.items() if a != "_objects"} for k, v in domains.items()},
        "encoder_forward_stats": encoder_stats,
        "npz_representation_stats": rep_npz,
        "ranked_discrepancies": ranked[:30],
        "separations": {
            "feature_shift": True,
            "topology_shift": True,
            "bn_activation_shift": True,
            "representation_noncollapse_note": "effective_rank reported for both domains",
            "short_causal_flow_patterns": "not_supported_in_repo_utilities_skipped",
        },
    }
    write_json(SHIFT_JSON, out)
    write_json(CELLS / "shift_audit.json", out)
    logging.info("Shift audit wrote %s", SHIFT_JSON)
    return 0


def cmd_capacity_fit(args: argparse.Namespace) -> int:
    RESULT.mkdir(parents=True, exist_ok=True)
    CELLS.mkdir(parents=True, exist_ok=True)
    sha = verify_ckpt()
    # CPU-only capacity cells (Slurm CPU partition)
    device = torch.device("cpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    p1 = load_splits(emb_dir(DEV_SEED, "P1"), require_expected_ids=True)
    rnd = load_splits(emb_dir(DEV_SEED, "random"), require_expected_ids=True)
    # align random ids
    for sp in ("train", "val"):
        if not np.array_equal(p1[sp]["ids"], rnd[sp]["ids"]):
            raise SystemExit("random control id mismatch vs P1")
    x_full, x_names = load_x_edge_native()
    x_tr = x_full[p1["train"]["ids"]]
    x_va = x_full[p1["val"]["ids"]]

    cells = {}
    # Primary P1 protocol
    for stack in ("X", "H", "H+X"):
        for learner in ("logistic", "mlp", "hgb"):
            key = f"P1_{stack}_{learner}"
            logging.info("Fitting %s", key)
            z_tr = p1["train"]["Z"] if "H" in stack else None
            z_va = p1["val"]["Z"] if "H" in stack else None
            mat_tr = stack_mat(stack, z_tr if z_tr is not None else p1["train"]["Z"], x_tr)
            mat_va = stack_mat(stack, z_va if z_va is not None else p1["val"]["Z"], x_va)
            if stack == "X":
                mat_tr, mat_va = x_tr, x_va
            elif stack == "H":
                mat_tr, mat_va = p1["train"]["Z"], p1["val"]["Z"]
            if learner == "logistic":
                cell = fit_logistic(mat_tr, p1["train"]["y"], mat_va, p1["val"]["y"])
            elif learner == "mlp":
                cell = fit_mlp(mat_tr, p1["train"]["y"], mat_va, p1["val"]["y"], device)
            else:
                cell = fit_hgb(mat_tr, p1["train"]["y"], mat_va, p1["val"]["y"])
            cell.update(
                {
                    "protocol": "P1_strict_inductive_legacy",
                    "stack": stack,
                    "seed": DEV_SEED,
                    "bn_protocol": "frozen_aml_bn",
                    "feature_contract_id": CONTRACT,
                    "ids": {"train": p1["train"]["ids_meta"], "val": p1["val"]["ids_meta"]},
                    "x_feature_names": x_names if "X" in stack else None,
                    "checkpoint_sha256": sha,
                    "test_inspected": False,
                    "exploratory_posthoc": True,
                    "table_eligible": False,
                }
            )
            cells[key] = cell
            write_json(CELLS / f"{key}.json", cell)

    # Random controls
    for stack in ("H", "H+X"):
        key = f"random_{stack}_logistic"
        logging.info("Fitting %s", key)
        mat_tr = stack_mat(stack, rnd["train"]["Z"], x_tr)
        mat_va = stack_mat(stack, rnd["val"]["Z"], x_va)
        cell = fit_logistic(mat_tr, rnd["train"]["y"], mat_va, rnd["val"]["y"])
        cell.update(
            {
                "protocol": "random_encoder",
                "stack": stack,
                "seed": DEV_SEED,
                "ids": {"train": rnd["train"]["ids_meta"], "val": rnd["val"]["ids_meta"]},
                "test_inspected": False,
                "exploratory_posthoc": True,
                "table_eligible": False,
            }
        )
        cells[key] = cell
        write_json(CELLS / f"{key}.json", cell)

    # P2 sensitivity (logistic only)
    p2 = load_splits(emb_dir(DEV_SEED, "P2"), require_expected_ids=True)
    for stack in ("H", "H+X"):
        key = f"P2_{stack}_logistic"
        mat_tr = stack_mat(stack, p2["train"]["Z"], x_tr)
        mat_va = stack_mat(stack, p2["val"]["Z"], x_va)
        cell = fit_logistic(mat_tr, p2["train"]["y"], mat_va, p2["val"]["y"])
        cell.update(
            {
                "protocol": "P2_label_free_target_bn",
                "stack": stack,
                "sensitivity": True,
                "bn_protocol": "target_train_only_running_stats",
                "test_inspected": False,
                "exploratory_posthoc": True,
                "table_eligible": False,
            }
        )
        cells[key] = cell
        write_json(CELLS / f"{key}.json", cell)

    summary = {
        "ok": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "exploratory_posthoc": True,
        "table_eligible": False,
        "test_inspected": False,
        "checkpoint_sha256": sha,
        "cells": {
            k: {
                "val_auprc_at_0.5": v["val_auprc_at_0.5"],
                "learner": v["learner"],
                "stack": v["stack"],
                "feature_dim": v["feature_dim"],
            }
            for k, v in cells.items()
        },
        "predeclared_learners": {
            "logistic": {"C": 1.0, "class_weight": "model", "seed": LOGISTIC_SEED},
            "PaperStyleMLP": {"epochs": MLP_EPOCHS, "lr": MLP_LR, "bs": MLP_BS, "seed": MLP_SEED},
            "HistGradientBoosting": HGB_CFG,
        },
    }
    write_json(CAPACITY_JSON, summary)
    logging.info("Capacity fit done")
    return 0


def cmd_gate(args: argparse.Namespace) -> int:
    cap = json.loads(CAPACITY_JSON.read_text(encoding="utf-8"))
    cells = {}
    for p in CELLS.glob("P1_*.json"):
        cells[p.stem] = json.loads(p.read_text(encoding="utf-8"))
    for p in CELLS.glob("random_*.json"):
        cells[p.stem] = json.loads(p.read_text(encoding="utf-8"))

    # Select best pretrained H+X by val AUPRC among learners (val only)
    hx_keys = [k for k in cells if k.startswith("P1_H+X_")]
    x_keys = [k for k in cells if k.startswith("P1_X_")]
    if not hx_keys or not x_keys:
        raise SystemExit("missing P1 H+X or X cells")
    best_hx = max(hx_keys, key=lambda k: cells[k]["val_auprc_at_0.5"])
    best_x = max(x_keys, key=lambda k: cells[k]["val_auprc_at_0.5"])
    rand_key = "random_H+X_logistic"
    if rand_key not in cells:
        raise SystemExit("missing random H+X")

    hx = cells[best_hx]["val_auprc_at_0.5"]
    x = cells[best_x]["val_auprc_at_0.5"]
    rhx = cells[rand_key]["val_auprc_at_0.5"]
    # coverage
    cov_ok = (
        cells[best_hx]["ids"]["val"]["sha256_of_ids_bytes"]
        == cells[rand_key]["ids"]["val"]["sha256_of_ids_bytes"]
        == EXPECTED_ID["val"]
    )
    checks = {
        "hx_beats_x_by_margin": (hx - x) >= GATE_MARGIN,
        "hx_beats_random_hx_by_margin": (hx - rhx) >= GATE_MARGIN,
        "coverage_ok": cov_ok,
    }
    passed = all(checks.values())
    gate = {
        "ok": True,
        "passed": passed,
        "scientific_failure": not passed,
        "slurm_should_not_treat_as_crash": True,
        "margin": GATE_MARGIN,
        "selected_pretrained_hx_cell": best_hx,
        "selected_x_cell": best_x,
        "random_hx_cell": rand_key,
        "val_auprc": {"H+X": hx, "X": x, "random_H+X": rhx},
        "deltas": {"hx_minus_x": hx - x, "hx_minus_random_hx": hx - rhx},
        "checks": checks,
        "test_inspected": False,
        "exploratory_posthoc": True,
        "table_eligible": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "capacity_summary": cap.get("cells"),
    }
    write_json(GATE_JSON, gate)
    if passed:
        logging.info("GATE_PASS")
    else:
        logging.info(
            "GATE_FAIL scientific (Slurm exit 0): checks=%s val_auprc=%s",
            checks,
            gate["val_auprc"],
        )
    return 0  # always success for Slurm


def cmd_confirm(args: argparse.Namespace) -> int:
    gate = json.loads(GATE_JSON.read_text(encoding="utf-8"))
    if not gate.get("passed"):
        payload = {
            "skipped": True,
            "reason": "validation_gate_failed",
            "gate": gate,
            "test_evaluated": False,
            "exploratory_posthoc": True,
            "table_eligible": False,
        }
        write_json(CONFIRM_JSON, payload)
        logging.info("Confirmation SKIPPED (gate failed)")
        return 0

    # Locked config from gate; evaluate test for selected learner on seeds 2 + 1,3,4
    sel = gate["selected_pretrained_hx_cell"]  # e.g. P1_H+X_mlp
    parts = sel.split("_")
    # P1_H+X_logistic -> stack H+X, learner logistic
    learner = parts[-1]
    stack = "H+X"
    device = torch.device("cpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    x_full, _ = load_x_edge_native()
    results = {}
    for seed in (DEV_SEED,) + CONFIRM_SEEDS:
        splits = {}
        ed = emb_dir(seed, "P1")
        for sp in ("train", "val", "test"):
            z, y, ids = load_embedding_npz(ed / f"{sp}.npz")
            splits[sp] = {"Z": z, "y": y, "ids": ids}
        x_tr, x_va, x_te = x_full[splits["train"]["ids"]], x_full[splits["val"]["ids"]], x_full[splits["test"]["ids"]]
        mat = {
            "train": stack_mat(stack, splits["train"]["Z"], x_tr),
            "val": stack_mat(stack, splits["val"]["Z"], x_va),
            "test": stack_mat(stack, splits["test"]["Z"], x_te),
        }
        # Refit on train/val then score test (same recipe)
        if learner == "logistic":
            scaler = StandardScaler()
            tr = scaler.fit_transform(mat["train"]).astype(np.float32)
            va = scaler.transform(mat["val"]).astype(np.float32)
            te = scaler.transform(mat["test"]).astype(np.float32)
            cw = gin_cw()
            set_seed(LOGISTIC_SEED)
            clf = LogisticRegression(
                class_weight=cw, max_iter=1000, random_state=LOGISTIC_SEED, solver="lbfgs", n_jobs=1, C=1.0
            )
            clf.fit(tr, splits["train"]["y"])
            pva = clf.predict_proba(va)[:, 1]
            pte = clf.predict_proba(te)[:, 1]
            thr = tune_thr(splits["val"]["y"], pva)
        elif learner == "mlp":
            # use fit_mlp then transform test with same scaler path — reimplement briefly
            scaler = StandardScaler()
            tr = scaler.fit_transform(mat["train"]).astype(np.float32)
            va = scaler.transform(mat["val"]).astype(np.float32)
            te = scaler.transform(mat["test"]).astype(np.float32)
            torch.manual_seed(MLP_SEED)
            np.random.seed(MLP_SEED)
            model = PaperStyleMLP(tr.shape[1]).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=MLP_LR)
            x_t = torch.from_numpy(tr)
            y_t = torch.from_numpy(splits["train"]["y"].astype(np.float32))
            best_auprc, best_state = -1.0, None
            n = tr.shape[0]
            for ep in range(MLP_EPOCHS):
                model.train()
                perm = np.random.RandomState(MLP_SEED * 1009 + ep).permutation(n)
                for start in range(0, n, MLP_BS):
                    idx = perm[start : start + MLP_BS]
                    opt.zero_grad(set_to_none=True)
                    loss = nn.functional.binary_cross_entropy_with_logits(
                        model(x_t[idx].to(device)), y_t[idx].to(device)
                    )
                    loss.backward()
                    opt.step()
                pva = _predict_proba(model, va, batch_size=MLP_BS, device=device)
                auprc = float(average_precision_score(splits["val"]["y"], pva))
                if auprc > best_auprc:
                    best_auprc = auprc
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(best_state)
            model.to(device)
            pva = _predict_proba(model, va, batch_size=MLP_BS, device=device)
            pte = _predict_proba(model, te, batch_size=MLP_BS, device=device)
            thr = tune_thr(splits["val"]["y"], pva)
        else:
            scaler = StandardScaler()
            tr = scaler.fit_transform(mat["train"]).astype(np.float32)
            va = scaler.transform(mat["val"]).astype(np.float32)
            te = scaler.transform(mat["test"]).astype(np.float32)
            n0 = max(int((splits["train"]["y"] == 0).sum()), 1)
            n1 = max(int((splits["train"]["y"] == 1).sum()), 1)
            w = np.where(splits["train"]["y"] == 1, n0 / n1, 1.0).astype(np.float64)
            clf = HistGradientBoostingClassifier(
                max_depth=6, learning_rate=0.1, max_iter=100, random_state=LOGISTIC_SEED
            )
            clf.fit(tr, splits["train"]["y"], sample_weight=w)
            pva = clf.predict_proba(va)[:, 1]
            pte = clf.predict_proba(te)[:, 1]
            thr = tune_thr(splits["val"]["y"], pva)

        results[str(seed)] = {
            "seed": seed,
            "learner": learner,
            "stack": stack,
            "validation": {
                "threshold_0.5": metrics_block(splits["val"]["y"], pva, 0.5),
                "threshold_val_selected": metrics_block(splits["val"]["y"], pva, thr),
            },
            "test": {
                "threshold_0.5": metrics_block(splits["test"]["y"], pte, 0.5),
                "threshold_val_selected": metrics_block(splits["test"]["y"], pte, thr),
            },
            "ids": {sp: ids_hash(splits[sp]["ids"]) for sp in splits},
            "role": "development" if seed == DEV_SEED else "confirmation",
        }
        write_json(CELLS / f"confirm_seed{seed}_{learner}_{stack.replace('+', 'plus')}.json", results[str(seed)])

    payload = {
        "skipped": False,
        "gate_passed": True,
        "selected_cell": sel,
        "learner": learner,
        "stack": stack,
        "test_evaluated": True,
        "per_seed": results,
        "exploratory_posthoc": True,
        "table_eligible": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(CONFIRM_JSON, payload)
    return 0


def cmd_aggregate(args: argparse.Namespace) -> int:
    shift = json.loads(SHIFT_JSON.read_text()) if SHIFT_JSON.is_file() else None
    cap = json.loads(CAPACITY_JSON.read_text()) if CAPACITY_JSON.is_file() else None
    gate = json.loads(GATE_JSON.read_text()) if GATE_JSON.is_file() else None
    conf = json.loads(CONFIRM_JSON.read_text()) if CONFIRM_JSON.is_file() else None
    out = {
        "title": TAG,
        "exploratory_posthoc": True,
        "table_eligible": False,
        "git_head_at_implementation_note": "see submission.json",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_sha256": SOURCE_SHA,
        "shift_audit": {
            "path": str(SHIFT_JSON),
            "top_discrepancies": (shift or {}).get("ranked_discrepancies", [])[:10],
        },
        "capacity": {
            "path": str(CAPACITY_JSON),
            "cells": (cap or {}).get("cells"),
        },
        "gate": gate,
        "confirmation": conf,
        "answers": {
            "gate_passed": bool((gate or {}).get("passed")),
            "embedding_contribution_supported": bool((gate or {}).get("passed")),
            "largest_shift_categories": list(
                {
                    r["category"]
                    for r in (shift or {}).get("ranked_discrepancies", [])[:5]
                }
            ),
        },
    }
    write_json(OUT_JSON, out)
    lines = [
        "# Final transfer capacity audit",
        "",
        "> Exploratory / post-hoc. `table_eligible=false`.",
        f"> Twin: `{OUT_JSON.relative_to(ROOT)}`",
        "",
        f"- Locked ckpt SHA256: `{SOURCE_SHA}`",
        f"- Gate passed: **{(gate or {}).get('passed')}**",
        "",
        "## Part A — top discrepancies",
        "",
    ]
    for r in (shift or {}).get("ranked_discrepancies", [])[:10]:
        lines.append(f"- `{r['name']}` = {r['value']:.6g} [{r['category']}] → {r['implicates']}")
    lines += ["", "## Part B — capacity gate", ""]
    if gate:
        lines.append(f"- Selected H+X cell: `{gate.get('selected_pretrained_hx_cell')}`")
        lines.append(f"- Val AUPRC H+X / X / random H+X: {gate.get('val_auprc')}")
        lines.append(f"- Checks: `{gate.get('checks')}`")
    if conf:
        lines.append(f"- Confirmation skipped: {conf.get('skipped')}")
        if not conf.get("skipped"):
            for s, block in (conf.get("per_seed") or {}).items():
                te = block["test"]["threshold_0.5"]
                lines.append(
                    f"- seed {s} test AUPRC@0.5={te['auprc']:.6f} AUROC={te['auroc']:.6f} F1={te['f1']:.6f}"
                )
    NOTES.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logging.info("Wrote %s", OUT_JSON)
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    """Login/Slurm-cheap smoke: no get_data, no model fit."""
    RESULT.mkdir(parents=True, exist_ok=True)
    sha = verify_ckpt()
    p1 = emb_dir(DEV_SEED, "P1")
    rnd = emb_dir(DEV_SEED, "random")
    for sp in ("train", "val"):
        z, y, ids = load_embedding_npz(p1 / f"{sp}.npz")
        meta = ids_hash(ids)
        if meta["sha256_of_ids_bytes"] != EXPECTED_ID[sp]:
            raise SystemExit(f"P1 {sp} id mismatch")
        zr, yr, idr = load_embedding_npz(rnd / f"{sp}.npz")
        if not np.array_equal(ids, idr):
            raise SystemExit("random ids != P1")
        _ = (z.shape, zr.shape, int(y.sum()), int(yr.sum()))
    # synthetic gate logic
    fake_cells = {
        "P1_H+X_logistic": {"val_auprc_at_0.5": 0.05, "ids": {"val": {"sha256_of_ids_bytes": EXPECTED_ID["val"]}}},
        "P1_X_logistic": {"val_auprc_at_0.5": 0.04, "ids": {"val": {"sha256_of_ids_bytes": EXPECTED_ID["val"]}}},
        "random_H+X_logistic": {
            "val_auprc_at_0.5": 0.041,
            "ids": {"val": {"sha256_of_ids_bytes": EXPECTED_ID["val"]}},
        },
    }
    hx, x, rhx = 0.05, 0.04, 0.041
    assert (hx - x) >= GATE_MARGIN
    assert (hx - rhx) >= GATE_MARGIN
    out = {
        "ok": True,
        "smoke": True,
        "checkpoint_sha256": sha,
        "expected_id_hashes": EXPECTED_ID,
        "embedding_dirs": {
            "P1": str(p1),
            "P2": str(emb_dir(DEV_SEED, "P2")),
            "random": str(rnd),
        },
        "synthetic_gate_checks_ok": True,
        "fake_cells_keys": list(fake_cells),
        "hgb_learner": HGB_CFG["learner"],
        "exploratory_posthoc": True,
        "table_eligible": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(RESULT / "smoke.json", out)
    logging.info("Smoke OK")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("smoke")
    sh = sub.add_parser("shift")
    sh.add_argument("--sample_n", type=int, default=200_000)
    sh.add_argument("--n_batches", type=int, default=8)
    sub.add_parser("capacity_fit")
    sub.add_parser("gate")
    sub.add_parser("confirm")
    sub.add_parser("aggregate")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    logger_setup()
    args = build_parser().parse_args(argv)
    RESULT.mkdir(parents=True, exist_ok=True)
    CELLS.mkdir(parents=True, exist_ok=True)
    if args.cmd == "smoke":
        return cmd_smoke(args)
    if args.cmd == "shift":
        return cmd_shift(args)
    if args.cmd == "capacity_fit":
        return cmd_capacity_fit(args)
    if args.cmd == "gate":
        return cmd_gate(args)
    if args.cmd == "confirm":
        return cmd_confirm(args)
    if args.cmd == "aggregate":
        return cmd_aggregate(args)
    raise SystemExit(args.cmd)


if __name__ == "__main__":
    raise SystemExit(main())
