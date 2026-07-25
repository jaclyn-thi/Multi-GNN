#!/usr/bin/env python3
"""Performance-focused full-stack + published-GCPAL challenge evaluation.

No GNN training. Candidates from validation/provenance only.
NOT an exact GCPAL reproduction.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from dataset_specs import get_dataset_spec
from gcpal_txn_node.data import load_small_hi_frame
from gcpal_txn_node.eval_mlp import PaperStyleMLP, _predict_proba, _select_threshold_f1
from gcpal_txn_node.extraction import (
    JOINT_FULL_GRAPH_RANDOM40_V1,
    TEMPORAL_EXPANDING_WINDOW_V1,
    extract_joint_full_graph,
    joint_full_graph_random40_config,
    load_encoder_from_checkpoint,
    sha256_file,
)
from gcpal_txn_node.features import fit_feature_preprocessor
from gcpal_txn_node.adjacency import build_directed_flow_adjacency
from linear_probe import load_embedding_npz, tune_threshold_max_f1
from ranking_metrics import alert_budget_metrics
from util import set_seed

# Import feature builders from probe_feature_ablation without package path issues.
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "probe_feature_ablation",
    _ROOT / "scripts" / "probe_feature_ablation.py",
)
_pfa = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules["probe_feature_ablation"] = _pfa
_spec.loader.exec_module(_pfa)
build_full_feature_matrix = _pfa.build_full_feature_matrix
load_dataset_frames = _pfa.load_dataset_frames
from transaction_knn.features import resolve_amount_column

# Predetermined stratified seeds for GCPAL-style label ratios (validation selection never uses test).
GCPAL_SPLIT_SEEDS = (2, 11, 23, 42, 77)
GCPAL_TARGET_F1_40 = 0.581
GCPAL_TARGET_F1_60 = 0.658
# Published GCPAL Table 2 targets (method); raw baselines undocumented in repo → gate uses our X-only.
LEARNERS = ("logistic", "mlp", "hist_gbm")
WEIGHT_MODES = ("none", "balanced")  # focal only with mlp
FOCAL_GAMMA = 2.0

EDGE_POST = Path(
    "embeddings/gin_emlps_ports_tds_corrected_preserve_seed_asym_proj_8192neg_queue0_40ep_seed2"
)
EDGE_PRE3H = EDGE_POST / "pre_embedding_3h"
EDGE_CKPT = Path(
    "saved-models/checkpoint_gin_emlps_ports_tds_corrected_preserve_seed_asym_proj_8192neg_queue0_40ep_seed2.tar"
)
D_CKPT = Path("checkpoints/gcpal_txn_node_posagg_B_D_supcon_5ep_seed2/epoch_05.pt")
D_H_DIR = Path("embeddings/gcpal_txn_node_posagg/B_supcon_mean_logprob/ep05")
TF_CACHE = Path("results/cache/temporal_flow_causal/Small-HI")


def _metrics_block(y: np.ndarray, proba: np.ndarray, thr: float) -> Dict[str, float]:
    y = y.astype(np.int64)
    pred = (proba >= float(thr)).astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    n = int(y.shape[0])
    out = {
        "auroc": float(roc_auc_score(y, proba)) if len(np.unique(y)) > 1 else float("nan"),
        "auprc": float(average_precision_score(y, proba)) if len(np.unique(y)) > 1 else float("nan"),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "threshold": float(thr),
        "positive_prediction_rate": float(pred.mean()) if n else 0.0,
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
        "n": float(n),
    }
    out.update(alert_budget_metrics(y, proba))
    return out


def _ranking_only(y: np.ndarray, proba: np.ndarray) -> Dict[str, float]:
    y = y.astype(np.int64)
    return {
        "auroc": float(roc_auc_score(y, proba)) if len(np.unique(y)) > 1 else float("nan"),
        "auprc": float(average_precision_score(y, proba)) if len(np.unique(y)) > 1 else float("nan"),
        "n": float(y.shape[0]),
        "positive_rate": float(y.mean()) if y.size else 0.0,
    }


class FocalBCE(nn.Module):
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
        p = torch.sigmoid(logits)
        pt = torch.where(y > 0.5, p, 1.0 - p)
        return (((1.0 - pt) ** self.gamma) * bce).mean()


def train_mlp(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_va: np.ndarray,
    y_va: np.ndarray,
    x_te: np.ndarray,
    y_te: np.ndarray,
    *,
    device: torch.device,
    seed: int,
    focal: bool = False,
    epochs: int = 15,
    batch_size: int = 8192,
    lr: float = 1e-3,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = PaperStyleMLP(int(x_tr.shape[1])).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit: nn.Module = FocalBCE(FOCAL_GAMMA) if focal else None  # type: ignore
    x_t = torch.from_numpy(x_tr.astype(np.float32))
    y_t = torch.from_numpy(y_tr.astype(np.float32))
    n = x_tr.shape[0]
    model.train()
    for ep in range(epochs):
        perm = np.random.RandomState(seed * 1009 + ep).permutation(n)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            xb = x_t[idx].to(device)
            yb = y_t[idx].to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            if focal:
                loss = crit(logits, yb)
            else:
                loss = nn.functional.binary_cross_entropy_with_logits(logits, yb)
            loss.backward()
            opt.step()
    proba_te = _predict_proba(model, x_te, batch_size=batch_size, device=device)
    proba_va = _predict_proba(model, x_va, batch_size=batch_size, device=device)
    thr = _select_threshold_f1(y_va, proba_va)
    return {
        "val_ranking": _ranking_only(y_va, proba_va),
        "threshold_0.5": _metrics_block(y_te, proba_te, 0.5),
        "threshold_val_selected": {
            **_metrics_block(y_te, proba_te, thr),
            "validation_selected_threshold": float(thr),
        },
        "val_at_selected_threshold": _metrics_block(y_va, proba_va, thr),
        "learner": "mlp_focal" if focal else "mlp",
    }


def train_logistic(
    x_tr, y_tr, x_va, y_va, x_te, y_te, *, seed: int, class_weight
) -> Dict[str, Any]:
    clf = LogisticRegression(
        class_weight=class_weight,
        max_iter=1000,
        random_state=seed,
        solver="lbfgs",
        n_jobs=8,
        C=1.0,
    )
    clf.fit(x_tr, y_tr)
    proba_va = clf.predict_proba(x_va)[:, 1]
    proba_te = clf.predict_proba(x_te)[:, 1]
    thr, _ = tune_threshold_max_f1(y_va, proba_va)
    return {
        "val_ranking": _ranking_only(y_va, proba_va),
        "threshold_0.5": _metrics_block(y_te, proba_te, 0.5),
        "threshold_val_selected": {
            **_metrics_block(y_te, proba_te, thr),
            "validation_selected_threshold": float(thr),
        },
        "val_at_selected_threshold": _metrics_block(y_va, proba_va, thr),
        "learner": "logistic",
        "class_weight": class_weight if class_weight is not None else "none",
    }


def train_hgb(
    x_tr, y_tr, x_va, y_va, x_te, y_te, *, seed: int, class_weight
) -> Dict[str, Any]:
    # sklearn HistGradientBoosting: use sample_weight for class balance
    if class_weight == "balanced":
        n_pos = max(int((y_tr == 1).sum()), 1)
        n_neg = max(int((y_tr == 0).sum()), 1)
        w_pos = n_neg / n_pos
        sw = np.where(y_tr == 1, w_pos, 1.0).astype(np.float64)
    else:
        sw = None
    clf = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.05,
        max_iter=200,
        random_state=seed,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
    )
    clf.fit(x_tr, y_tr, sample_weight=sw)
    proba_va = clf.predict_proba(x_va)[:, 1]
    proba_te = clf.predict_proba(x_te)[:, 1]
    thr, _ = tune_threshold_max_f1(y_va, proba_va)
    return {
        "val_ranking": _ranking_only(y_va, proba_va),
        "threshold_0.5": _metrics_block(y_te, proba_te, 0.5),
        "threshold_val_selected": {
            **_metrics_block(y_te, proba_te, thr),
            "validation_selected_threshold": float(thr),
        },
        "val_at_selected_threshold": _metrics_block(y_va, proba_va, thr),
        "learner": "hist_gbm",
        "class_weight": class_weight if class_weight is not None else "none",
        "tree_backend_note": "sklearn HistGradientBoostingClassifier (lightgbm/xgboost not installed)",
    }


def fit_eval(
    x_tr, y_tr, x_va, y_va, x_te, y_te, *, learner: str, weight: str, device, seed: int
) -> Dict[str, Any]:
    # Scale non-constant columns with train stats
    scaler = StandardScaler()
    x_tr_s = scaler.fit_transform(x_tr).astype(np.float32)
    x_va_s = scaler.transform(x_va).astype(np.float32)
    x_te_s = scaler.transform(x_te).astype(np.float32)
    cw = "balanced" if weight == "balanced" else None
    if learner == "logistic":
        return train_logistic(x_tr_s, y_tr, x_va_s, y_va, x_te_s, y_te, seed=seed, class_weight=cw)
    if learner == "hist_gbm":
        return train_hgb(x_tr_s, y_tr, x_va_s, y_va, x_te_s, y_te, seed=seed, class_weight=cw)
    if learner == "mlp":
        return train_mlp(
            x_tr_s, y_tr, x_va_s, y_va, x_te_s, y_te, device=device, seed=seed, focal=False
        )
    if learner == "mlp_focal":
        return train_mlp(
            x_tr_s, y_tr, x_va_s, y_va, x_te_s, y_te, device=device, seed=seed, focal=True
        )
    raise ValueError(learner)


def stack_matrix(parts: List[np.ndarray]) -> np.ndarray:
    return np.concatenate(parts, axis=1).astype(np.float32)


def random_label_split(
    y: np.ndarray, *, train_frac: float, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified: train_frac of all labeled rows for train+val pool; rest test.
    Inner: 75% of pool → train, 25% → val (same as txn-node random-40 diagnostic).
    """
    sss = StratifiedShuffleSplit(n_splits=1, train_size=train_frac, random_state=seed)
    tr_r, te_r = next(sss.split(np.arange(len(y)), y))
    sss_inner = StratifiedShuffleSplit(n_splits=1, train_size=0.75, random_state=seed + 1)
    tr_r2, va_r = next(sss_inner.split(tr_r, y[tr_r]))
    return tr_r[tr_r2], tr_r[va_r], te_r


def summarize_seeds(rows: List[dict], key_path: Sequence[str]) -> Dict[str, float]:
    vals = []
    for r in rows:
        cur: Any = r
        for k in key_path:
            cur = cur[k]
        vals.append(float(cur))
    a = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(a.mean()),
        "sample_sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        "median": float(np.median(a)),
        "values": vals,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_config", default="data_config.json")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=2)
    p.add_argument(
        "--output_json",
        default="results/diagnostics/gcpal_challenge_fullstack_eval.json",
    )
    p.add_argument(
        "--output_md",
        default="notes/gcpal_challenge_fullstack_eval.md",
    )
    p.add_argument("--skip_joint_extract", action="store_true")
    args = p.parse_args()
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    if out_json.exists() or out_md.exists():
        raise SystemExit("Refusing overwrite existing deliverables")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    t0 = time.perf_counter()

    # --- Candidate provenance (validation-only selection among A/B/C/D) ---
    candidates_meta = {
        "edge_centric_Dplus": {
            "name": "edge_centric_Dplus_corrected_preserve",
            "checkpoint": str(EDGE_CKPT),
            "selection": (
                "Among A/B/C/D contrastive arms, D+ (corrected reverse + preserve_seed_edges) "
                "selected by validation embedding-probe F1/AUPRC/AUROC; not test."
            ),
            "post128_dir": str(EDGE_POST),
            "pre3h_dir": str(EDGE_PRE3H),
            "job": "18514684",
            "flags": "gin+emlps+tds+ports+ego+reverse_mp+correct_reverse+preserve_seed_edges",
        },
        "txn_node_D_supcon": {
            "name": "txn_node_D_supcon_ep5",
            "checkpoint": str(D_CKPT),
            "selection": (
                "Positive-aggregation ablation: among B/C/D, SupCon selected by temporal "
                "validation HxX AUPRC (expanding-window); not test."
            ),
            "extraction_temporal": TEMPORAL_EXPANDING_WINDOW_V1,
            "h_dir": str(D_H_DIR),
            "job": "18669618",
        },
    }
    assert EDGE_CKPT.is_file() and D_CKPT.is_file()
    assert (EDGE_POST / "train.npz").is_file() and (EDGE_PRE3H / "train.npz").is_file()

    # --- Load frames / features ---
    df, df_train, tr, va, te, spec = load_dataset_frames("Small-HI", args.data_config)
    y_all = df[spec.label_col].to_numpy().astype(np.int64)
    n_all = len(df)
    logging.info("Loaded Small-HI n=%d tr/va/te=%d/%d/%d", n_all, len(tr), len(va), len(te))

    x_raw_full, raw_names, _, raw_meta = build_full_feature_matrix(
        df, df_train, ("edge_native",), categorical_encoding="one_hot"
    )
    x_morph_full, morph_names, _, morph_meta = build_full_feature_matrix(
        df,
        df_train,
        ("degree_fan", "flow_balance", "temporal_behavior"),
        categorical_encoding="one_hot",
    )
    tf_feat = np.load(TF_CACHE / "features.npy").astype(np.float32)
    assert tf_feat.shape[0] == n_all

    # Txn-node style X (for D stacks) — train-fit preprocessor
    prep = fit_feature_preprocessor(df_train, amount_col=resolve_amount_column(df))
    x_txn = prep.transform(df).astype(np.float32)

    # Edge-centric H aligned to temporal splits (intersection with available embeddings)
    def load_edge_split(emb_root: Path, ids: np.ndarray):
        z_map = {}
        y_map = {}
        for split, split_ids in (("train", tr), ("val", va), ("test", te)):
            z, y, eid = load_embedding_npz(emb_root / f"{split}.npz")
            for i, e in enumerate(eid.tolist()):
                z_map[int(e)] = z[i]
                y_map[int(e)] = int(y[i])
        keep = np.array([int(i) in z_map for i in ids.tolist()], dtype=bool)
        ids_k = ids[keep]
        z = np.stack([z_map[int(i)] for i in ids_k.tolist()], axis=0)
        y = np.array([y_map[int(i)] for i in ids_k.tolist()], dtype=np.int64)
        return z, y, ids_k, float(keep.mean())

    h_post = {}
    h_pre = {}
    coverage = {}
    for name, root in (("post128", EDGE_POST), ("pre3h", EDGE_PRE3H)):
        for split, ids in (("train", tr), ("val", va), ("test", te)):
            z, y, ids_k, cov = load_edge_split(root, ids)
            target = h_post if name == "post128" else h_pre
            target[split] = {"z": z, "y": y, "ids": ids_k}
            coverage[f"{name}_{split}"] = {
                "coverage": cov,
                "n": int(ids_k.shape[0]),
                "dim": int(z.shape[1]),
            }
            # label consistency check
            assert np.array_equal(y, y_all[ids_k])

    # D expanding-window H
    d_h = {
        "train": np.load(D_H_DIR / "h_train.npy"),
        "val": np.load(D_H_DIR / "h_val.npy"),
        "test": np.load(D_H_DIR / "h_test.npy"),
    }
    d_ids = {
        "train": np.load(D_H_DIR / "ids_train.npy"),
        "val": np.load(D_H_DIR / "ids_val.npy"),
        "test": np.load(D_H_DIR / "ids_test.npy"),
    }
    for s in ("train", "val", "test"):
        assert np.array_equal(d_ids[s], {"train": tr, "val": va, "test": te}[s])

    # Joint full-graph H for D (random protocol)
    joint_path = Path("embeddings/gcpal_txn_node_posagg/B_supcon_mean_logprob/ep05_joint_full/h_all.npy")
    if joint_path.is_file():
        d_h_all = np.load(joint_path)
        logging.info("Loaded cached joint H %s", joint_path)
    elif args.skip_joint_extract:
        raise SystemExit("Missing joint H and --skip_joint_extract set")
    else:
        logging.info("Extracting joint full-graph H for D (no GNN train)...")
        flow_full, _ = build_directed_flow_adjacency(
            df["from_id"].to_numpy(),
            df["to_id"].to_numpy(),
            df["Timestamp"].astype(float).to_numpy(),
            policy="immediate_next",
        )
        enc, _ = load_encoder_from_checkpoint(
            D_CKPT, in_dim=int(x_txn.shape[1]), emb_dim=128, map_location=str(device)
        )
        enc.to(device)
        all_ids = np.arange(n_all, dtype=np.int64)
        joint = extract_joint_full_graph(
            encoder=enc,
            x_all=x_txn,
            flow_ei=flow_full,
            all_node_ids=all_ids,
            device=device,
            config=joint_full_graph_random40_config(seed=args.seed),
            checkpoint_path=D_CKPT,
            verify_expected_edges=True,
        )
        d_h_all = joint["embeddings"]["all"]
        joint_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(joint_path, d_h_all)
        np.save(joint_path.parent / "ids_all.npy", all_ids)
        logging.info("Wrote %s", joint_path)

    def feats_for_ids(ids: np.ndarray, kind: str) -> np.ndarray:
        if kind == "X":
            return x_raw_full[ids]
        if kind == "X_txn":
            return x_txn[ids]
        if kind == "TF":
            return tf_feat[ids]
        if kind == "morph":
            return x_morph_full[ids]
        raise ValueError(kind)

    # --- Temporal stack definitions ---
    # Edge-centric: H is post128 or pre3h; never call txn H pre-3h
    temporal_configs: List[Dict[str, Any]] = []

    def add_edge_configs(rep_name: str, hstore: dict):
        # Use intersection ids per split already in hstore
        stacks = [
            ("X", ["X"]),
            ("TF", ["TF"]),
            ("X+TF", ["X", "TF"]),
            ("H", ["H"]),
            ("H+X", ["H", "X"]),
            ("H+TF", ["H", "TF"]),
            ("H+X+TF", ["H", "X", "TF"]),
            ("morph", ["morph"]),
            ("H+X+morph", ["H", "X", "morph"]),
        ]
        if rep_name == "edge_pre3h":
            stacks.append(("H+X+TF", ["H", "X", "TF"]))  # already have; ensure listed
        for stack_name, parts in stacks:
            temporal_configs.append(
                {
                    "candidate": rep_name,
                    "stack": stack_name,
                    "parts": parts,
                    "hstore": hstore,
                    "id_key": "ids",
                    "x_key": "X",
                }
            )

    add_edge_configs("edge_post128", h_post)
    add_edge_configs("edge_pre3h", h_pre)
    # Explicit pre3h + X + TF already in list

    # Feature-only controls (no H) — use full temporal ids
    for stack_name, parts in (
        ("X", ["X"]),
        ("TF", ["TF"]),
        ("X+TF", ["X", "TF"]),
        ("morph", ["morph"]),
        ("X+morph", ["X", "morph"]),
    ):
        temporal_configs.append(
            {
                "candidate": "features_only",
                "stack": stack_name,
                "parts": parts,
                "hstore": None,
                "ids": {"train": tr, "val": va, "test": te},
                "x_key": "X",
            }
        )

    # Txn-node D expanding
    for stack_name, parts in (
        ("X", ["X_txn"]),
        ("TF", ["TF"]),
        ("X+TF", ["X_txn", "TF"]),
        ("H", ["H"]),
        ("H+X", ["H", "X_txn"]),
        ("H+TF", ["H", "TF"]),
        ("H+X+TF", ["H", "X_txn", "TF"]),
        ("morph", ["morph"]),
        ("H+X+morph", ["H", "X_txn", "morph"]),
    ):
        temporal_configs.append(
            {
                "candidate": "txn_D_supcon_expanding",
                "stack": stack_name,
                "parts": parts,
                "h_direct": d_h,
                "ids": d_ids,
                "x_key": "X_txn",
            }
        )

    def build_split_xy(cfg: dict, split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if cfg.get("hstore") is not None:
            ids = cfg["hstore"][split]["ids"]
            y = cfg["hstore"][split]["y"]
            H = cfg["hstore"][split]["z"]
        elif cfg.get("h_direct") is not None:
            ids = cfg["ids"][split]
            y = y_all[ids]
            H = cfg["h_direct"][split]
        else:
            ids = cfg["ids"][split]
            y = y_all[ids]
            H = None
        blocks = []
        for part in cfg["parts"]:
            if part == "H":
                assert H is not None
                blocks.append(H)
            else:
                blocks.append(feats_for_ids(ids, part))
        return stack_matrix(blocks), y, ids

    # Deduplicate configs by (candidate, stack)
    seen = set()
    uniq_cfgs = []
    for c in temporal_configs:
        key = (c["candidate"], c["stack"])
        if key in seen:
            continue
        seen.add(key)
        uniq_cfgs.append(c)
    temporal_configs = uniq_cfgs

    # Phase 1: logistic on all stacks (fast). Phase 2: mlp/hgb/focal on shortlist.
    phase1_grid = [("logistic", "none"), ("logistic", "balanced")]
    phase2_learners = [
        ("mlp", "none"),
        ("mlp_focal", "none"),
        ("hist_gbm", "none"),
        ("hist_gbm", "balanced"),
    ]

    temporal_results = []
    logging.info("Phase-1 logistic grid: %d stacks", len(temporal_configs))
    for cfg in temporal_configs:
        try:
            x_tr, y_tr, _ = build_split_xy(cfg, "train")
            x_va, y_va, _ = build_split_xy(cfg, "val")
            x_te, y_te, _ = build_split_xy(cfg, "test")
        except Exception as e:
            logging.exception("skip %s/%s: %s", cfg["candidate"], cfg["stack"], e)
            continue
        for learner, weight in phase1_grid:
            tag = f"{cfg['candidate']}|{cfg['stack']}|{learner}|{weight}"
            logging.info("TEMPORAL %s X=%s", tag, x_tr.shape)
            try:
                res = fit_eval(
                    x_tr,
                    y_tr,
                    x_va,
                    y_va,
                    x_te,
                    y_te,
                    learner=learner,
                    weight=weight,
                    device=device,
                    seed=args.seed,
                )
            except Exception as e:
                logging.exception("fail %s: %s", tag, e)
                continue
            temporal_results.append(
                {
                    "tag": tag,
                    "candidate": cfg["candidate"],
                    "stack": cfg["stack"],
                    "learner": learner,
                    "weight": weight,
                    "n_features": int(x_tr.shape[1]),
                    "n_train": int(x_tr.shape[0]),
                    "metrics": res,
                    "val_auprc": res["val_ranking"]["auprc"],
                    "val_f1_at_selected": res["val_at_selected_threshold"]["f1"],
                    "phase": 1,
                    "_xy": (x_tr, y_tr, x_va, y_va, x_te, y_te),
                }
            )

    # Shortlist: top 8 by val AUPRC + forced baselines (keep any phase-1 row with _xy)
    temporal_results.sort(
        key=lambda r: (r["val_auprc"], r["val_f1_at_selected"]), reverse=True
    )
    forced = {
        ("features_only", "X"),
        ("edge_post128", "H+X"),
        ("edge_pre3h", "H+X+TF"),
        ("txn_D_supcon_expanding", "H+X"),
        ("txn_D_supcon_expanding", "H"),
    }
    shortlist_keys = {(r["candidate"], r["stack"]) for r in temporal_results[:8]}
    shortlist_keys |= forced
    # Prefer balanced logistic row for each key when available (has _xy)
    by_cs: Dict[Tuple[str, str], dict] = {}
    for r in temporal_results:
        if "_xy" not in r:
            continue
        k = (r["candidate"], r["stack"])
        if k not in shortlist_keys:
            continue
        prev = by_cs.get(k)
        if prev is None or (r["weight"] == "balanced" and prev["weight"] != "balanced"):
            by_cs[k] = r
        elif prev["weight"] == r["weight"] and r["val_auprc"] > prev["val_auprc"]:
            by_cs[k] = r
    phase2_base = list(by_cs.values())

    logging.info("Phase-2 rich learners on %d stacks", len(phase2_base))
    for base in phase2_base:
        x_tr, y_tr, x_va, y_va, x_te, y_te = base["_xy"]
        for learner, weight in phase2_learners:
            tag = f"{base['candidate']}|{base['stack']}|{learner}|{weight}"
            logging.info("TEMPORAL %s", tag)
            try:
                res = fit_eval(
                    x_tr,
                    y_tr,
                    x_va,
                    y_va,
                    x_te,
                    y_te,
                    learner=learner,
                    weight=weight,
                    device=device,
                    seed=args.seed,
                )
            except Exception as e:
                logging.exception("fail %s: %s", tag, e)
                continue
            temporal_results.append(
                {
                    "tag": tag,
                    "candidate": base["candidate"],
                    "stack": base["stack"],
                    "learner": learner,
                    "weight": weight,
                    "n_features": int(x_tr.shape[1]),
                    "n_train": int(x_tr.shape[0]),
                    "metrics": res,
                    "val_auprc": res["val_ranking"]["auprc"],
                    "val_f1_at_selected": res["val_at_selected_threshold"]["f1"],
                    "phase": 2,
                }
            )

    # Drop cached matrices before JSON
    for r in temporal_results:
        r.pop("_xy", None)

    # Selection: max val AUPRC, tie-break val F1
    temporal_results.sort(
        key=lambda r: (r["val_auprc"], r["val_f1_at_selected"]), reverse=True
    )
    best_temporal = temporal_results[0] if temporal_results else None

    # --- Raw baseline gate (X-only logistic + mlp on temporal) ---
    raw_gate_rows = [r for r in temporal_results if r["candidate"] == "features_only" and r["stack"] == "X"]
    # GCPAL paper does not document raw F1 baselines in-repo → PARTIAL reconstruction
    comparability = {
        "verdict": "PARTIAL",
        "reason": (
            "GCPAL published raw/feature baselines are not recoverable from the paper text "
            "available in-repo (no code release). We report X-only logistic/MLP under our "
            "temporal protocol as the reconstruction gate, and compare method F1 to Table-2 "
            "targets 0.581 (40%) / 0.658 (60%) only under our reconstructed random-label protocol."
        ),
        "x_only_temporal": [
            {
                "learner": r["learner"],
                "weight": r["weight"],
                "val_auprc": r["val_auprc"],
                "test_f1_0.5": r["metrics"]["threshold_0.5"]["f1"],
                "test_f1_val_thr": r["metrics"]["threshold_val_selected"]["f1"],
                "test_auprc": r["metrics"]["threshold_0.5"]["auprc"],
            }
            for r in raw_gate_rows
        ],
    }

    # --- GCPAL random 40/60 ---
    # D: joint full-graph H; edge: temporal-frozen H with documented scope caveat
    random_protocols = {}
    for ratio, target in ((0.4, GCPAL_TARGET_F1_40), (0.6, GCPAL_TARGET_F1_60)):
        ratio_key = f"random_{int(ratio*100)}"
        random_protocols[ratio_key] = {
            "train_label_fraction": ratio,
            "target_f1": target,
            "split_seeds": list(GCPAL_SPLIT_SEEDS),
            "label_handling": (
                "StratifiedShuffleSplit over ALL transactions: "
                f"{int(ratio*100)}% → train+val pool, remainder → test; "
                "inner 75/25 train/val on the pool (seed+1). "
                "Same construction as txn-node random-40 diagnostic."
            ),
            "by_candidate": {},
        }

        # Evaluate a shortlist: best temporal config's candidate family + D joint + edge post H+X+TF + X-only
        shortlist = []
        if best_temporal:
            shortlist.append(best_temporal)
        # Always include key baselines
        for r in temporal_results:
            if (r["candidate"], r["stack"], r["learner"], r["weight"]) in {
                ("txn_D_supcon_expanding", "H+X", "mlp", "none"),
                ("txn_D_supcon_expanding", "H+X", "logistic", "balanced"),
                ("edge_post128", "H+X", "logistic", "balanced"),
                ("edge_pre3h", "H+X+TF", "logistic", "balanced"),
                ("edge_pre3h", "H+X+TF", "mlp", "none"),
                ("features_only", "X", "logistic", "balanced"),
                ("features_only", "X", "mlp", "none"),
            }:
                shortlist.append(r)
        # unique by tag
        seen_t = set()
        sl = []
        for r in shortlist:
            if r["tag"] in seen_t:
                continue
            seen_t.add(r["tag"])
            sl.append(r)

        for r in sl:
            seed_rows = []
            for s in GCPAL_SPLIT_SEEDS:
                # Build feature matrix on all rows for this candidate/stack
                cand = r["candidate"]
                stack = r["stack"]
                # Find matching cfg
                cfg = next(
                    c
                    for c in temporal_configs
                    if c["candidate"] == cand and c["stack"] == stack
                )
                # For random: use joint H for txn D; for edge use concatenated temporal H by edge id
                if cand == "txn_D_supcon_expanding":
                    # joint full-graph H
                    H_all = d_h_all
                    ids_all = np.arange(n_all, dtype=np.int64)
                    extraction_scope = {
                        "mode": JOINT_FULL_GRAPH_RANDOM40_V1,
                        "note": "Joint full-graph encode then random label split",
                    }
                    blocks = []
                    # map stack parts
                    part_map = {
                        "H": H_all,
                        "X_txn": x_txn,
                        "X": x_raw_full,
                        "TF": tf_feat,
                        "morph": x_morph_full,
                    }
                    for part in cfg["parts"]:
                        if part == "H":
                            blocks.append(H_all)
                        elif part == "X_txn":
                            blocks.append(x_txn)
                        else:
                            blocks.append(part_map[part] if part in part_map else feats_for_ids(ids_all, part))
                    X_all = stack_matrix(blocks)
                    y = y_all
                elif cand.startswith("edge_"):
                    # Documented: reuse temporally extracted embeddings (not joint full-graph MP)
                    extraction_scope = {
                        "mode": "temporal_split_frozen_embeddings_then_random_labels",
                        "note": (
                            "Edge-centric H was extracted under temporal induce-per-split "
                            "(established SSL protocol). Random label splits are applied to "
                            "those frozen vectors — NOT a joint full-graph re-encode. "
                            "Diagnostic-only for GCPAL Table-2 style ratios."
                        ),
                    }
                    hstore = h_post if cand == "edge_post128" else h_pre
                    # Build full-length matrix with NaN fill then keep rows present in any split
                    dim_h = hstore["train"]["z"].shape[1]
                    H_full = np.zeros((n_all, dim_h), dtype=np.float32)
                    present = np.zeros(n_all, dtype=bool)
                    for sp in ("train", "val", "test"):
                        ids = hstore[sp]["ids"]
                        H_full[ids] = hstore[sp]["z"]
                        present[ids] = True
                    ids_keep = np.where(present)[0]
                    blocks = []
                    for part in cfg["parts"]:
                        if part == "H":
                            blocks.append(H_full[ids_keep])
                        else:
                            blocks.append(feats_for_ids(ids_keep, part if part != "X_txn" else "X"))
                    X_all = stack_matrix(blocks)
                    y = y_all[ids_keep]
                    ids_all = ids_keep
                else:
                    # features only
                    extraction_scope = {"mode": "features_only_random_labels"}
                    ids_all = np.arange(n_all, dtype=np.int64)
                    blocks = [feats_for_ids(ids_all, part if part != "X_txn" else "X") for part in cfg["parts"]]
                    X_all = stack_matrix(blocks)
                    y = y_all

                tr_i, va_i, te_i = random_label_split(y, train_frac=ratio, seed=s)
                res = fit_eval(
                    X_all[tr_i],
                    y[tr_i],
                    X_all[va_i],
                    y[va_i],
                    X_all[te_i],
                    y[te_i],
                    learner=r["learner"],
                    weight=r["weight"],
                    device=device,
                    seed=s,
                )
                seed_rows.append(
                    {
                        "seed": s,
                        "metrics": res,
                        "n_train": int(tr_i.shape[0]),
                        "n_val": int(va_i.shape[0]),
                        "n_test": int(te_i.shape[0]),
                        "extraction_scope": extraction_scope,
                    }
                )
            # aggregates on val AUPRC and test F1 (both thresholds)
            random_protocols[ratio_key]["by_candidate"][r["tag"]] = {
                "per_seed": seed_rows,
                "agg_val_auprc": summarize_seeds(seed_rows, ("metrics", "val_ranking", "auprc")),
                "agg_test_f1_0.5": summarize_seeds(seed_rows, ("metrics", "threshold_0.5", "f1")),
                "agg_test_f1_val_thr": summarize_seeds(
                    seed_rows, ("metrics", "threshold_val_selected", "f1")
                ),
                "agg_test_auprc": summarize_seeds(seed_rows, ("metrics", "threshold_0.5", "auprc")),
                "exceeds_target_f1_0.5_mean": float(
                    summarize_seeds(seed_rows, ("metrics", "threshold_0.5", "f1"))["mean"]
                )
                > target,
                "exceeds_target_f1_val_thr_mean": float(
                    summarize_seeds(seed_rows, ("metrics", "threshold_val_selected", "f1"))["mean"]
                )
                > target,
                "target_f1": target,
            }

    # Select best random config per ratio by mean val AUPRC (never test)
    for ratio_key, block in random_protocols.items():
        if not block["by_candidate"]:
            continue
        best_tag = max(
            block["by_candidate"].items(),
            key=lambda kv: (kv[1]["agg_val_auprc"]["mean"], kv[1]["agg_test_f1_val_thr"]["mean"]),
        )[0]
        # secondary key mistakenly used test f1 — fix to val f1
        best_tag = max(
            block["by_candidate"].items(),
            key=lambda kv: (
                kv[1]["agg_val_auprc"]["mean"],
                float(
                    np.mean(
                        [
                            s["metrics"]["val_at_selected_threshold"]["f1"]
                            for s in kv[1]["per_seed"]
                        ]
                    )
                ),
            ),
        )[0]
        block["selected_by_val_auprc"] = best_tag
        block["selected_test_metrics_after_selection"] = block["by_candidate"][best_tag]

    # Recommendation
    recommendation = {
        "next_finetune_experiment": None,
        "rationale": None,
    }
    if best_temporal:
        if best_temporal["candidate"].startswith("edge"):
            recommendation = {
                "next_finetune_experiment": (
                    "Fine-tune the D+ edge-centric encoder (corrected reverse + preserve_seed) "
                    "with a light supervised head on the winning feature stack "
                    f"`{best_temporal['stack']}` under temporal val AUPRC early-stopping; "
                    "do not change reverse/preserve semantics."
                ),
                "rationale": "Best temporal validation AUPRC came from edge-centric D+ frozen stack.",
            }
        elif best_temporal["candidate"].startswith("txn"):
            recommendation = {
                "next_finetune_experiment": (
                    "Extend txn-node D SupCon beyond 5 epochs with count-normalized or SupCon "
                    "aggregation under expanding-window val HxX AUPRC selection; keep joint "
                    "full-graph diagnostic separate."
                ),
                "rationale": "Best temporal validation AUPRC came from txn-node D SupCon stacks.",
            }
        else:
            recommendation = {
                "next_finetune_experiment": (
                    "SSL embeddings did not beat strong feature-only stacks on val AUPRC; "
                    "next experiment should improve representation learning (objective/augmentation) "
                    "rather than probe tweaks."
                ),
                "rationale": "Feature-only won validation selection.",
            }

    payload = {
        "not_exact_reproduction": True,
        "gnn_training_occurred": False,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "wall_seconds": time.perf_counter() - t0,
        "candidates": candidates_meta,
        "embedding_coverage": coverage,
        "d_checkpoint_sha256": sha256_file(D_CKPT),
        "comparability_gate": comparability,
        "temporal_primary": {
            "selection_rule": "max val AUPRC, secondary val F1 at selected threshold; never test",
            "selected": best_temporal,
            "all_results_sorted": temporal_results[:80],
            "n_configs_evaluated": len(temporal_results),
        },
        "random_protocols": random_protocols,
        "published_targets": {
            "40pct_f1": GCPAL_TARGET_F1_40,
            "60pct_f1": GCPAL_TARGET_F1_60,
            "configs_exceeding": {
                k: {
                    tag: {
                        "f1_0.5_mean": v["agg_test_f1_0.5"]["mean"],
                        "f1_val_thr_mean": v["agg_test_f1_val_thr"]["mean"],
                        "exceeds_0.5": v["exceeds_target_f1_0.5_mean"],
                        "exceeds_val_thr": v["exceeds_target_f1_val_thr_mean"],
                    }
                    for tag, v in block["by_candidate"].items()
                    if v["exceeds_target_f1_0.5_mean"] or v["exceeds_target_f1_val_thr_mean"]
                }
                for k, block in random_protocols.items()
            },
        },
        "recommendation": recommendation,
        "notes": [
            "LightGBM/XGBoost not installed; used sklearn HistGradientBoostingClassifier.",
            "Txn-node H is never labeled pre-3h.",
            "Edge-centric random protocol reuses temporally extracted H (documented PARTIAL).",
            "Txn-node random protocol uses joint full-graph extraction.",
        ],
    }

    def _json_safe(o: Any) -> Any:
        if isinstance(o, dict):
            return {str(k): _json_safe(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_json_safe(v) for v in o]
        if isinstance(o, (np.floating, float)):
            f = float(o)
            return None if not math.isfinite(f) else f
        if isinstance(o, (np.integer, int)):
            return int(o)
        if isinstance(o, (np.bool_, bool)):
            return bool(o)
        if isinstance(o, np.ndarray):
            return None
        return o

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n")

    # Markdown
    lines = [
        "# GCPAL challenge full-stack evaluation",
        "",
        "**No GNN training.** NOT an exact GCPAL reproduction.",
        "",
        f"Companion: [`{out_json}`](../{out_json})",
        f"Job: `{os.environ.get('SLURM_JOB_ID')}`",
        "",
        "## Candidates (validation provenance)",
        "",
        "- **Edge D+**: corrected reverse + `preserve_seed_edges`, post-128 & pre-3h "
        f"(job 18514684; val-selected among A/B/C/D)",
        "- **Txn D SupCon ep5**: job 18669618; val HxX AUPRC among B/C/D aggregations",
        "- **Feature controls**: X / TF / morph",
        "",
        "## Comparability gate",
        "",
        f"**{comparability['verdict']}** — {comparability['reason']}",
        "",
        "## Temporal primary (selected)",
        "",
    ]
    if best_temporal:
        bt = best_temporal
        m = bt["metrics"]
        lines += [
            f"**Selected:** `{bt['tag']}`",
            f"- val AUPRC={bt['val_auprc']:.6f}  val F1@sel={bt['val_f1_at_selected']:.6f}",
            f"- test AUPRC@0.5={m['threshold_0.5']['auprc']:.6f} AUROC={m['threshold_0.5']['auroc']:.6f}",
            f"- test F1@0.5={m['threshold_0.5']['f1']:.6f} F1@val-thr={m['threshold_val_selected']['f1']:.6f}",
            f"- P@100/500/1000={m['threshold_0.5'].get('precision_at_100'):.3f}/"
            f"{m['threshold_0.5'].get('precision_at_500'):.3f}/"
            f"{m['threshold_0.5'].get('precision_at_1000'):.3f}",
            "",
            "### Top-10 by val AUPRC",
            "",
            "| tag | val AUPRC | test AUPRC@0.5 | F1@val-thr |",
            "|-----|----------:|---------------:|-----------:|",
        ]
        for r in temporal_results[:10]:
            lines.append(
                f"| `{r['tag']}` | {r['val_auprc']:.4f} | "
                f"{r['metrics']['threshold_0.5']['auprc']:.4f} | "
                f"{r['metrics']['threshold_val_selected']['f1']:.4f} |"
            )
    lines += ["", "## Random-40 / Random-60 (diagnostic)", ""]
    for rk, block in random_protocols.items():
        lines.append(f"### {rk} (target F1={block['target_f1']})")
        lines.append("")
        lines.append(block["label_handling"])
        sel = block.get("selected_by_val_auprc")
        if sel:
            s = block["by_candidate"][sel]
            lines.append(
                f"- **Selected by val AUPRC:** `{sel}` mean val AUPRC="
                f"{s['agg_val_auprc']['mean']:.4f}±{s['agg_val_auprc']['sample_sd']:.4f}"
            )
            lines.append(
                f"- Test F1@0.5 mean={s['agg_test_f1_0.5']['mean']:.4f}±{s['agg_test_f1_0.5']['sample_sd']:.4f} "
                f"(exceeds target: {s['exceeds_target_f1_0.5_mean']})"
            )
            lines.append(
                f"- Test F1@val-thr mean={s['agg_test_f1_val_thr']['mean']:.4f}±{s['agg_test_f1_val_thr']['sample_sd']:.4f} "
                f"(exceeds target: {s['exceeds_target_f1_val_thr_mean']})"
            )
        lines.append("")
    lines += [
        "## Recommendation",
        "",
        recommendation.get("next_finetune_experiment") or "n/a",
        "",
        f"Rationale: {recommendation.get('rationale')}",
        "",
        "## Confirmation",
        "",
        "- No GNN training in this job.",
        "- No automatic fine-tune submissions.",
        "",
    ]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n")
    logging.info("Wrote %s and %s", out_json, out_md)


if __name__ == "__main__":
    main()
