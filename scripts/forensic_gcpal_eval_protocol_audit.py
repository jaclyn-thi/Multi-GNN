#!/usr/bin/env python3
"""Forensic GCPAL-versus-current evaluation-protocol audit (no GNN retraining).

Uses corrected-TDS + preserve-seed D seed-2 embeddings, Small-HI raw features, and
causal temporal-flow downstream features. Separates temporal (thesis) vs random
diagnostic splits; LR vs MLP; weighting and threshold rules.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
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
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataset_specs import get_dataset_spec
from linear_probe import load_embedding_npz, tune_threshold_max_f1
from morphology.temporal_flow_causal import TEMPORAL_FLOW_CAUSAL_FEATURE_NAMES
from transaction_knn.features import load_data_config, resolve_amount_column
from util import logger_setup, set_seed

GCPAL_TABLE2 = {
    "40": {
        "Logistic Regression": {"P": 0.535, "R": 0.441, "F1": 0.483},
        "MLP": {"P": 0.385, "R": 0.420, "F1": 0.392},
        "GCPAL": {"P": 0.600, "R": 0.564, "F1": 0.581},
    },
    "60": {
        "Logistic Regression": {"P": 0.613, "R": 0.497, "F1": 0.548},
        "MLP": {"P": 0.451, "R": 0.489, "F1": 0.418},
        "GCPAL": {"P": 0.684, "R": 0.634, "F1": 0.658},
    },
}

MODEL_CE_WEIGHTS = {0: 1.0000182882773443, 1: 6.275014431494497}  # gin Small-HI thesis
MLP_HIDDEN = 128
MLP_EPOCHS = 15
MLP_BATCH = 8192
MLP_LR = 1e-3
MLP_DROPOUT = 0.1
CLASSIFIER_SEEDS = (0, 1, 2, 3, 4)
SPLIT_SEEDS = (0, 1, 2, 3, 4)


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _agg(rows: List[Dict[str, float]], keys: Sequence[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"n": len(rows)}
    for k in keys:
        xs = np.asarray([r[k] for r in rows if r.get(k) is not None and np.isfinite(r[k])], dtype=float)
        if xs.size == 0:
            out[k] = {"mean": None, "std_sample": None}
        else:
            out[k] = {
                "mean": float(xs.mean()),
                "std_sample": float(xs.std(ddof=1)) if xs.size > 1 else 0.0,
                "values": xs.tolist(),
            }
    return out


def _metrics_at_threshold(y: np.ndarray, proba: np.ndarray, thr: float) -> Dict[str, float]:
    y = y.astype(np.int64)
    pred = (proba >= thr).astype(np.int64)
    if len(np.unique(y)) < 2:
        auroc = float("nan")
        auprc = float("nan")
    else:
        auroc = float(roc_auc_score(y, proba))
        auprc = float(average_precision_score(y, proba))
    return {
        "auroc": auroc,
        "auprc": auprc,
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "threshold": float(thr),
        "n": float(y.shape[0]),
        "positive_rate": float(y.mean()) if y.size else float("nan"),
    }


def _safe_log1p(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.maximum(x.astype(np.float64), 0.0)).astype(np.float32)


def load_raw_columns(df: pd.DataFrame) -> Dict[str, Any]:
    amount_col = resolve_amount_column(df)
    return {
        "timestamp": df["Timestamp"].astype(float).to_numpy().astype(np.float32),
        "log1p_amount": _safe_log1p(df[amount_col].astype(float).to_numpy()),
        "currency": df["Received Currency"].fillna("__missing__").astype(str).to_numpy(),
        "payment_format": df["Payment Format"].fillna("__missing__").astype(str).to_numpy(),
        "amount_col": amount_col,
        "names": [
            "Timestamp",
            f"log1p_{amount_col}",
            "Received Currency_ordinal",
            "Payment Format_ordinal",
        ],
    }


def encode_raw_train_fit(
    raw: Dict[str, Any], train_idx: np.ndarray, row_idx: np.ndarray
) -> Tuple[np.ndarray, StandardScaler, Dict[str, Dict[str, int]]]:
    """Fit ordinal maps + scaler on train rows; transform requested rows."""
    cur_map = {c: i for i, c in enumerate(sorted(np.unique(raw["currency"][train_idx])))}
    pay_map = {c: i for i, c in enumerate(sorted(np.unique(raw["payment_format"][train_idx])))}

    def pack(idx: np.ndarray) -> np.ndarray:
        cur = np.asarray([cur_map.get(v, -1) for v in raw["currency"][idx]], dtype=np.float32)
        pay = np.asarray([pay_map.get(v, -1) for v in raw["payment_format"][idx]], dtype=np.float32)
        return np.column_stack(
            [raw["timestamp"][idx], raw["log1p_amount"][idx], cur, pay]
        ).astype(np.float32)

    x_train = pack(train_idx)
    scaler = StandardScaler().fit(x_train)
    x = scaler.transform(pack(row_idx)).astype(np.float32)
    return x, scaler, {"currency": cur_map, "payment_format": pay_map}


class TorchMLP(nn.Module):
    def __init__(self, d_in: int, hidden: int = MLP_HIDDEN, dropout: float = MLP_DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def fit_predict_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: Dict[str, np.ndarray],
    *,
    seed: int,
    device: torch.device,
    class_weight: Optional[Dict[int, float]],
    epochs: int = MLP_EPOCHS,
) -> Dict[str, np.ndarray]:
    set_seed(seed)
    model = TorchMLP(x_train.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=MLP_LR)
    y_t = torch.from_numpy(y_train.astype(np.float32))
    x_t = torch.from_numpy(x_train.astype(np.float32))
    sample_w = None
    if class_weight is not None:
        w = torch.tensor([class_weight[0], class_weight[1]], dtype=torch.float32)
        sample_w = torch.where(y_t > 0.5, w[1], w[0])

    n = x_train.shape[0]
    model.train()
    for _ep in range(epochs):
        perm = np.random.RandomState(seed * 10007 + _ep).permutation(n)
        for start in range(0, n, MLP_BATCH):
            idx = perm[start : start + MLP_BATCH]
            xb = x_t[idx].to(device)
            yb = y_t[idx].to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            if sample_w is None:
                loss = nn.functional.binary_cross_entropy_with_logits(logits, yb)
            else:
                sw = sample_w[idx].to(device)
                loss = nn.functional.binary_cross_entropy_with_logits(logits, yb, weight=sw)
            loss.backward()
            opt.step()

    model.eval()
    out = {}
    with torch.no_grad():
        for name, xe in x_eval.items():
            probs = []
            xe_t = torch.from_numpy(xe.astype(np.float32))
            for start in range(0, xe_t.shape[0], MLP_BATCH):
                logits = model(xe_t[start : start + MLP_BATCH].to(device))
                probs.append(torch.sigmoid(logits).cpu().numpy())
            out[name] = np.concatenate(probs, axis=0) if probs else np.zeros((0,), dtype=np.float32)
    return out


def fit_predict_lr(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: Dict[str, np.ndarray],
    *,
    seed: int,
    class_weight: Any,
    max_iter: int = 1000,
) -> Dict[str, np.ndarray]:
    clf = LogisticRegression(
        class_weight=class_weight,
        max_iter=max_iter,
        random_state=seed,
        solver="lbfgs",
        n_jobs=8,
        C=1.0,
    )
    clf.fit(x_train, y_train)
    return {k: clf.predict_proba(v)[:, 1] for k, v in x_eval.items()}


def align_embeddings_by_edge_id(
    emb_edge_id: np.ndarray, emb_z: np.ndarray, emb_y: np.ndarray, target_edge_ids: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return Z,y for target_edge_ids; drop missing; coverage fraction."""
    order = np.argsort(emb_edge_id)
    e_sorted = emb_edge_id[order]
    z_sorted = emb_z[order]
    y_sorted = emb_y[order]
    pos = np.searchsorted(e_sorted, target_edge_ids)
    valid = pos < e_sorted.shape[0]
    valid &= e_sorted[np.clip(pos, 0, len(e_sorted) - 1)] == target_edge_ids
    # also handle empty
    hit = np.where(valid)[0]
    pos_hit = pos[hit]
    return (
        z_sorted[pos_hit],
        y_sorted[pos_hit],
        target_edge_ids[hit],
        float(hit.size / max(target_edge_ids.size, 1)),
    )


def load_temporal_embeddings(emb_dir: Path) -> Dict[str, Dict[str, np.ndarray]]:
    out = {}
    for split in ("train", "val", "test"):
        z, y, e = load_embedding_npz(emb_dir / f"{split}.npz")
        out[split] = {"Z": z, "y": y, "edge_id": e.astype(np.int64)}
    return out


def run_temporal_protocol(
    *,
    emb_post: Dict[str, Dict[str, np.ndarray]],
    emb_pre: Optional[Dict[str, Dict[str, np.ndarray]]],
    raw: Dict[str, Any],
    tf_features: np.ndarray,
    tf_edge_id: np.ndarray,
    device: torch.device,
) -> Dict[str, Any]:
    # TF cache edge_id is dense arange over CSV rows; embedding edge_id == CSV row index.
    if not np.array_equal(tf_edge_id, np.arange(tf_edge_id.shape[0], dtype=tf_edge_id.dtype)):
        raise RuntimeError("Expected dense TF edge_id arange for indexing")

    tr_e = emb_post["train"]["edge_id"]

    def raw_for(ids: np.ndarray) -> np.ndarray:
        x, _, _ = encode_raw_train_fit(raw, tr_e, ids)
        return x

    x_raw = {s: raw_for(emb_post[s]["edge_id"]) for s in ("train", "val", "test")}

    def tf_for(ids: np.ndarray) -> np.ndarray:
        return tf_features[ids.astype(np.int64)].astype(np.float32)

    x_tf = {s: tf_for(emb_post[s]["edge_id"]) for s in ("train", "val", "test")}
    tf_scaler = StandardScaler().fit(x_tf["train"])
    x_tf = {s: tf_scaler.transform(x_tf[s]).astype(np.float32) for s in x_tf}

    y = {s: emb_post[s]["y"] for s in ("train", "val", "test")}
    z_post = {s: emb_post[s]["Z"] for s in ("train", "val", "test")}
    z_pre = None
    if emb_pre is not None:
        # align pre to post edge ids
        z_pre = {}
        for s in ("train", "val", "test"):
            zp, yp, ep, cov = align_embeddings_by_edge_id(
                emb_pre[s]["edge_id"], emb_pre[s]["Z"], emb_pre[s]["y"], emb_post[s]["edge_id"]
            )
            if cov < 0.99:
                logging.warning("pre3h coverage for %s = %.4f", s, cov)
            # reorder zp to match emb_post order exactly
            # align function returns only hits in target order — if cov~1, lengths match
            if zp.shape[0] != emb_post[s]["Z"].shape[0]:
                raise RuntimeError(f"pre3h alignment failed for {s}: {zp.shape[0]} vs {emb_post[s]['Z'].shape[0]}")
            z_pre[s] = zp

    counts = {
        s: {
            "n": int(y[s].shape[0]),
            "n_pos": int(y[s].sum()),
            "positive_rate": float(y[s].mean()),
        }
        for s in ("train", "val", "test")
    }

    arms = {
        "raw": lambda s: x_raw[s],
        "embedding_post": lambda s: z_post[s],
        "embedding_post+raw": lambda s: np.concatenate([z_post[s], x_raw[s]], axis=1),
    }
    if z_pre is not None:
        arms["pre3h+raw"] = lambda s: np.concatenate([z_pre[s], x_raw[s]], axis=1)
        arms["pre3h+raw+temporal_flow"] = lambda s: np.concatenate(
            [z_pre[s], x_raw[s], x_tf[s]], axis=1
        )

    results: Dict[str, Any] = {"sample_counts": counts, "arms": {}}

    def eval_arm(arm_name: str, make_x) -> Dict[str, Any]:
        xtr, xva, xte = make_x("train"), make_x("val"), make_x("test")
        ytr, yva, yte = y["train"], y["val"], y["test"]
        arm_out: Dict[str, Any] = {"dim": int(xtr.shape[1]), "classifiers": {}}

        # Logistic: unweighted + thesis model weights
        for cw_name, cw in (("unweighted", None), ("thesis_model_weights", MODEL_CE_WEIGHTS)):
            seed_rows_05 = []
            seed_rows_val = []
            for seed in CLASSIFIER_SEEDS:
                proba = fit_predict_lr(
                    xtr, ytr, {"val": xva, "test": xte}, seed=seed, class_weight=cw
                )
                m05 = _metrics_at_threshold(yte, proba["test"], 0.5)
                thr, _ = tune_threshold_max_f1(yva, proba["val"])
                mval = _metrics_at_threshold(yte, proba["test"], thr)
                m05["seed"] = seed
                mval["seed"] = seed
                seed_rows_05.append(m05)
                seed_rows_val.append(mval)
            arm_out["classifiers"][f"logistic_{cw_name}"] = {
                "fixed_0.5": _agg(seed_rows_05, ["f1", "precision", "recall", "auroc", "auprc"]),
                "validation_tuned_threshold": _agg(
                    seed_rows_val, ["f1", "precision", "recall", "auroc", "auprc"]
                ),
                "per_seed_fixed_0.5": seed_rows_05,
                "per_seed_validation_tuned": seed_rows_val,
                "note_validation_tuned": "Threshold selected on temporal validation only; NOT paper-compatible.",
            }

        # MLP: paper-primary BCE unweighted + weighted secondary
        for cw_name, cw in (("bce_unweighted", None), ("bce_thesis_weights", MODEL_CE_WEIGHTS)):
            seed_rows_05 = []
            seed_rows_val = []
            for seed in CLASSIFIER_SEEDS:
                proba = fit_predict_mlp(
                    xtr,
                    ytr,
                    {"val": xva, "test": xte},
                    seed=seed,
                    device=device,
                    class_weight=cw,
                )
                m05 = _metrics_at_threshold(yte, proba["test"], 0.5)
                thr, _ = tune_threshold_max_f1(yva, proba["val"])
                mval = _metrics_at_threshold(yte, proba["test"], thr)
                m05["seed"] = seed
                mval["seed"] = seed
                seed_rows_05.append(m05)
                seed_rows_val.append(mval)
            arm_out["classifiers"][f"mlp_{cw_name}"] = {
                "fixed_0.5": _agg(seed_rows_05, ["f1", "precision", "recall", "auroc", "auprc"]),
                "validation_tuned_threshold": _agg(
                    seed_rows_val, ["f1", "precision", "recall", "auroc", "auprc"]
                ),
                "per_seed_fixed_0.5": seed_rows_05,
                "per_seed_validation_tuned": seed_rows_val,
                "mlp_config": {
                    "hidden": MLP_HIDDEN,
                    "epochs": MLP_EPOCHS,
                    "batch_size": MLP_BATCH,
                    "lr": MLP_LR,
                    "dropout": MLP_DROPOUT,
                    "loss": "BCEWithLogits" + ("" if cw is None else "+sample_class_weights"),
                },
            }
        return arm_out

    for name, fn in arms.items():
        logging.info("Temporal arm %s ...", name)
        t0 = time.perf_counter()
        results["arms"][name] = eval_arm(name, fn)
        results["arms"][name]["runtime_seconds"] = time.perf_counter() - t0
    return results


def run_random_protocol(
    *,
    z_all: np.ndarray,
    y_all: np.ndarray,
    edge_id_all: np.ndarray,
    raw: Dict[str, Any],
    train_frac: float,
    device: torch.device,
) -> Dict[str, Any]:
    """Stratified random train/test; fixed 0.5; no test tuning. Label: random_transductive_diagnostic."""
    results = {
        "train_frac": train_frac,
        "label": "random_transductive_diagnostic",
        "decision_rule_primary": "fixed_0.5",
        "split_seeds": list(SPLIT_SEEDS),
        "splits": [],
    }
    n = y_all.shape[0]
    for split_seed in SPLIT_SEEDS:
        sss = StratifiedShuffleSplit(
            n_splits=1, train_size=train_frac, random_state=split_seed
        )
        tr_idx, te_idx = next(sss.split(np.zeros(n), y_all))
        # optional internal val from train only (secondary)
        y_tr = y_all[tr_idx]
        train_eids = edge_id_all[tr_idx]
        test_eids = edge_id_all[te_idx]

        # Fit ordinal maps + scaler once on train edge ids; transform train/test.
        x_raw_tr, scaler, maps = encode_raw_train_fit(raw, train_eids, train_eids)

        def _pack_raw(idx: np.ndarray) -> np.ndarray:
            cur = np.asarray(
                [maps["currency"].get(v, -1) for v in raw["currency"][idx]], dtype=np.float32
            )
            pay = np.asarray(
                [maps["payment_format"].get(v, -1) for v in raw["payment_format"][idx]],
                dtype=np.float32,
            )
            packed = np.column_stack(
                [raw["timestamp"][idx], raw["log1p_amount"][idx], cur, pay]
            ).astype(np.float32)
            return scaler.transform(packed).astype(np.float32)

        x_raw_te = _pack_raw(test_eids)

        z_tr, z_te = z_all[tr_idx], z_all[te_idx]
        y_te = y_all[te_idx]
        y_tr_full = y_all[tr_idx]

        split_res = {
            "split_seed": split_seed,
            "n_train": int(tr_idx.size),
            "n_test": int(te_idx.size),
            "train_pos_rate": float(y_tr_full.mean()),
            "test_pos_rate": float(y_te.mean()),
            "arms": {},
        }

        arm_specs = {
            "raw": (x_raw_tr, x_raw_te),
            "embedding_post": (z_tr, z_te),
            "embedding_post+raw": (
                np.concatenate([z_tr, x_raw_tr], axis=1),
                np.concatenate([z_te, x_raw_te], axis=1),
            ),
        }
        for arm_name, (xtr, xte) in arm_specs.items():
            arm_out: Dict[str, Any] = {}
            for cw_name, cw in (("unweighted", None), ("thesis_model_weights", MODEL_CE_WEIGHTS)):
                rows = []
                for cseed in CLASSIFIER_SEEDS:
                    proba = fit_predict_lr(xtr, y_tr_full, {"test": xte}, seed=cseed, class_weight=cw)
                    m = _metrics_at_threshold(y_te, proba["test"], 0.5)
                    m["classifier_seed"] = cseed
                    rows.append(m)
                arm_out[f"logistic_{cw_name}_fixed_0.5"] = _agg(
                    rows, ["f1", "precision", "recall", "auroc", "auprc"]
                )
                arm_out[f"logistic_{cw_name}_fixed_0.5_per_seed"] = rows
            for cw_name, cw in (("bce_unweighted", None), ("bce_thesis_weights", MODEL_CE_WEIGHTS)):
                rows = []
                for cseed in CLASSIFIER_SEEDS:
                    proba = fit_predict_mlp(
                        xtr, y_tr_full, {"test": xte}, seed=cseed, device=device, class_weight=cw
                    )
                    m = _metrics_at_threshold(y_te, proba["test"], 0.5)
                    m["classifier_seed"] = cseed
                    rows.append(m)
                arm_out[f"mlp_{cw_name}_fixed_0.5"] = _agg(
                    rows, ["f1", "precision", "recall", "auroc", "auprc"]
                )
                arm_out[f"mlp_{cw_name}_fixed_0.5_per_seed"] = rows
            split_res["arms"][arm_name] = arm_out
        results["splits"].append(split_res)
        logging.info(
            "Random %.0f%% seed=%d done (train=%d test=%d)",
            100 * train_frac,
            split_seed,
            tr_idx.size,
            te_idx.size,
        )

    # pool across split seeds for summary
    pooled = {}
    for arm in ("raw", "embedding_post", "embedding_post+raw"):
        pooled[arm] = {}
        for metric_key in (
            "logistic_unweighted_fixed_0.5",
            "mlp_bce_unweighted_fixed_0.5",
            "logistic_thesis_model_weights_fixed_0.5",
            "mlp_bce_thesis_weights_fixed_0.5",
        ):
            f1s = []
            for sp in results["splits"]:
                block = sp["arms"][arm][metric_key]["f1"]
                if block.get("mean") is not None:
                    f1s.append(block["mean"])
            pooled[arm][metric_key] = {
                "f1_mean_over_split_seeds": float(np.mean(f1s)) if f1s else None,
                "f1_std_over_split_seeds": float(np.std(f1s, ddof=1)) if len(f1s) > 1 else None,
            }
    results["pooled_over_split_seeds"] = pooled
    return results


def decide(payload: Dict[str, Any]) -> Dict[str, Any]:
    temp = payload["temporal_protocol"]["arms"]
    # Prefer validation-tuned thesis LR on emb+raw as "current thesis-like", and fixed-0.5 MLP unweighted as paper-like
    def f1(arm, clf, rule):
        try:
            return temp[arm]["classifiers"][clf][rule]["f1"]["mean"]
        except Exception:
            return None

    raw_lr_val = f1("raw", "logistic_thesis_model_weights", "validation_tuned_threshold")
    emb_raw_lr_val = f1("embedding_post+raw", "logistic_thesis_model_weights", "validation_tuned_threshold")
    emb_raw_mlp_05 = f1("embedding_post+raw", "mlp_bce_unweighted", "fixed_0.5")
    raw_mlp_05 = f1("raw", "mlp_bce_unweighted", "fixed_0.5")
    pre_tf = None
    if "pre3h+raw+temporal_flow" in temp:
        pre_tf = f1("pre3h+raw+temporal_flow", "mlp_bce_unweighted", "fixed_0.5")
    pre_raw = f1("pre3h+raw", "mlp_bce_unweighted", "fixed_0.5") if "pre3h+raw" in temp else None

    rand60 = payload.get("random_protocol_60", {})
    rand40 = payload.get("random_protocol_40", {})
    if rand60.get("skipped") or not rand60.get("pooled_over_split_seeds"):
        return {
            "verdict": "D",
            "rationale": (
                "Random-split diagnostic protocols unavailable; cannot adjudicate GCPAL "
                "comparability. Temporal results alone are insufficient for verdict A–C."
            ),
            "deltas": {},
            "key_scores": {
                "temporal_raw_lr_val_tuned_f1": raw_lr_val,
                "temporal_embraw_lr_val_tuned_f1": emb_raw_lr_val,
                "temporal_embraw_mlp_unweighted_fixed0.5_f1": emb_raw_mlp_05,
                "temporal_raw_mlp_unweighted_fixed0.5_f1": raw_mlp_05,
                "temporal_pre3h_raw_mlp_fixed0.5_f1": pre_raw,
                "temporal_pre3h_raw_tf_mlp_fixed0.5_f1": pre_tf,
                "gcpal_table2_60_f1": GCPAL_TABLE2["60"]["GCPAL"]["F1"],
                "gcpal_table2_40_f1": GCPAL_TABLE2["40"]["GCPAL"]["F1"],
            },
        }

    rand60p = rand60["pooled_over_split_seeds"]
    rand40p = rand40["pooled_over_split_seeds"]
    r60_embraw_mlp = rand60p["embedding_post+raw"]["mlp_bce_unweighted_fixed_0.5"]["f1_mean_over_split_seeds"]
    r60_raw_mlp = rand60p["raw"]["mlp_bce_unweighted_fixed_0.5"]["f1_mean_over_split_seeds"]
    r40_embraw_mlp = rand40p["embedding_post+raw"]["mlp_bce_unweighted_fixed_0.5"]["f1_mean_over_split_seeds"]

    gcpal60 = GCPAL_TABLE2["60"]["GCPAL"]["F1"]
    gcpal40 = GCPAL_TABLE2["40"]["GCPAL"]["F1"]

    deltas = {
        "random60_minus_temporal_mlp_embraw_fixed0.5": (
            None
            if r60_embraw_mlp is None or emb_raw_mlp_05 is None
            else r60_embraw_mlp - emb_raw_mlp_05
        ),
        "mlp_minus_logistic_temporal_embraw_fixed0.5": (
            None
            if emb_raw_mlp_05 is None
            else emb_raw_mlp_05
            - (f1("embedding_post+raw", "logistic_unweighted", "fixed_0.5") or float("nan"))
        ),
        "embraw_minus_raw_temporal_mlp_fixed0.5": (
            None if emb_raw_mlp_05 is None or raw_mlp_05 is None else emb_raw_mlp_05 - raw_mlp_05
        ),
        "pre3h_minus_post_temporal_mlp_raw_fixed0.5": (
            None
            if pre_raw is None or emb_raw_mlp_05 is None
            else pre_raw - emb_raw_mlp_05
        ),
        "tf_stack_minus_pre3h_raw_temporal_mlp_fixed0.5": (
            None if pre_tf is None or pre_raw is None else pre_tf - pre_raw
        ),
        "remaining_gap_random60_embraw_mlp_to_gcpal60": (
            None if r60_embraw_mlp is None else gcpal60 - r60_embraw_mlp
        ),
        "remaining_gap_random40_embraw_mlp_to_gcpal40": (
            None if r40_embraw_mlp is None else gcpal40 - r40_embraw_mlp
        ),
    }

    # Verdict
    gap60 = deltas["remaining_gap_random60_embraw_mlp_to_gcpal60"]
    split_delta = deltas["random60_minus_temporal_mlp_embraw_fixed0.5"]
    if gap60 is None or r60_embraw_mlp is None:
        verdict = "D"
        rationale = "Missing critical random-split or temporal MLP results."
    elif gap60 <= 0.03 and (split_delta or 0) >= 0.05:
        verdict = "A"
        rationale = (
            f"Under random 60% diagnostic, emb+raw MLP F1={r60_embraw_mlp:.3f} approaches "
            f"GCPAL {gcpal60:.3f} (gap={gap60:.3f}); much of the temporal gap is protocol."
        )
    elif gap60 <= 0.12:
        verdict = "B"
        rationale = (
            f"Random 60% emb+raw MLP F1={r60_embraw_mlp:.3f} vs GCPAL {gcpal60:.3f} "
            f"(gap={gap60:.3f}); protocol helps but a representation gap remains."
        )
    elif gap60 > 0.12:
        verdict = "C"
        rationale = (
            f"Even under random 60% emb+raw MLP (F1={r60_embraw_mlp:.3f}), GCPAL {gcpal60:.3f} "
            f"is not approached (gap={gap60:.3f})."
        )
    else:
        verdict = "D"
        rationale = "Borderline/insufficient evidence."

    return {"verdict": verdict, "rationale": rationale, "deltas": deltas, "key_scores": {
        "temporal_raw_lr_val_tuned_f1": raw_lr_val,
        "temporal_embraw_lr_val_tuned_f1": emb_raw_lr_val,
        "temporal_embraw_mlp_unweighted_fixed0.5_f1": emb_raw_mlp_05,
        "temporal_raw_mlp_unweighted_fixed0.5_f1": raw_mlp_05,
        "temporal_pre3h_raw_mlp_fixed0.5_f1": pre_raw,
        "temporal_pre3h_raw_tf_mlp_fixed0.5_f1": pre_tf,
        "random60_embraw_mlp_unweighted_fixed0.5_f1": r60_embraw_mlp,
        "random60_raw_mlp_unweighted_fixed0.5_f1": r60_raw_mlp,
        "random40_embraw_mlp_unweighted_fixed0.5_f1": r40_embraw_mlp,
        "gcpal_table2_60_f1": gcpal60,
        "gcpal_table2_40_f1": gcpal40,
    }}


def write_md(path: Path, payload: Dict[str, Any]) -> None:
    d = payload["decision"]
    lines = [
        "# Forensic GCPAL vs current evaluation-protocol audit",
        "",
        "No GNN retraining. Random-split results are **diagnostic only** (not thesis-primary).",
        "",
        "## Embedding provenance",
        "",
        f"```json\n{json.dumps(payload['provenance'], indent=2)}\n```",
        "",
        "## Unresolved paper details (not guessed)",
        "",
        "- AMLWorld split (temporal vs random)",
        "- Validation set / early stopping",
        "- Decision threshold",
        "- Class weighting",
        "- Feature processing for X",
        "- Optimizer / LR schedule details beyond 'MLP + BCE'",
        "",
        "## GCPAL Table 2 targets",
        "",
        f"```json\n{json.dumps(GCPAL_TABLE2, indent=2)}\n```",
        "",
        "## Decision",
        "",
        f"**{d['verdict']}** — {d['rationale']}",
        "",
        "### Key scores",
        "",
        f"```json\n{json.dumps(d['key_scores'], indent=2)}\n```",
        "",
        "### Quantified deltas",
        "",
        f"```json\n{json.dumps(d['deltas'], indent=2)}\n```",
        "",
        "## Temporal protocol (thesis boundaries)",
        "",
        "Validation-tuned thresholds are reported separately from fixed-0.5.",
        "",
    ]
    for arm, block in payload["temporal_protocol"]["arms"].items():
        lines.append(f"### `{arm}` (dim={block.get('dim')})")
        for clf, cres in block["classifiers"].items():
            f05 = cres["fixed_0.5"]["f1"]
            fval = cres["validation_tuned_threshold"]["f1"]
            m05 = f05.get("mean")
            mval = fval.get("mean")
            s05 = f05.get("std_sample")
            sval = fval.get("std_sample")
            lines.append(
                f"- **{clf}**: fixed0.5 F1="
                f"{'n/a' if m05 is None else f'{m05:.4f}±{s05:.4f}'}; "
                f"val-tuned F1="
                f"{'n/a' if mval is None else f'{mval:.4f}±{sval:.4f}'}"
            )
        lines.append("")

    lines += [
        "## Random 60% train diagnostic (`random_transductive_diagnostic`)",
        "",
        "Fixed 0.5 only; no test tuning. Pooled over 5 split seeds (each with 5 classifier seeds).",
        "",
    ]
    if payload["random_protocol_60"].get("skipped"):
        lines.append("_Skipped in this run._")
        lines.append("")
    else:
        for arm, block in payload["random_protocol_60"]["pooled_over_split_seeds"].items():
            lines.append(f"### `{arm}`")
            for k, v in block.items():
                lines.append(
                    f"- {k}: F1={v.get('f1_mean_over_split_seeds')} ± {v.get('f1_std_over_split_seeds')}"
                )
            lines.append("")

    lines += [
        "## Random 40% train diagnostic",
        "",
    ]
    if payload["random_protocol_40"].get("skipped"):
        lines.append("_Skipped in this run._")
        lines.append("")
    else:
        for arm, block in payload["random_protocol_40"]["pooled_over_split_seeds"].items():
            lines.append(f"### `{arm}`")
            for k, v in block.items():
                lines.append(
                    f"- {k}: F1={v.get('f1_mean_over_split_seeds')} ± {v.get('f1_std_over_split_seeds')}"
                )
            lines.append("")

    lines += [
        "## Classifier configs",
        "",
        f"```json\n{json.dumps(payload['classifier_configs'], indent=2)}\n```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    logger_setup()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--embedding_dir",
        default="embeddings/gin_emlps_ports_tds_corrected_preserve_seed_asym_proj_8192neg_queue0_40ep_seed2",
    )
    p.add_argument(
        "--fullgraph_embedding_dir",
        default="embeddings/gin_emlps_ports_tds_corrected_preserve_seed_asym_proj_8192neg_queue0_40ep_seed2_fullgraph_random_transductive_diagnostic",
    )
    p.add_argument("--data_config", default="data_config.json")
    p.add_argument("--tf_cache_dir", default="results/cache/temporal_flow_causal/Small-HI")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output_json", required=True)
    p.add_argument("--output_md", required=True)
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny subsample for syntax/path validation only (not for reported results).",
    )
    p.add_argument("--smoke_n", type=int, default=4000)
    p.add_argument(
        "--skip_random",
        action="store_true",
        help="Skip random-split protocols (temporal only).",
    )
    args = p.parse_args()

    # Smoke overrides (module-level constants used by helpers).
    if args.smoke:
        global MLP_EPOCHS, CLASSIFIER_SEEDS, SPLIT_SEEDS
        MLP_EPOCHS = 2
        CLASSIFIER_SEEDS = (0, 1)
        SPLIT_SEEDS = (0, 1)
        logging.warning("SMOKE MODE enabled: reduced epochs/seeds/sample sizes")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    emb_dir = Path(args.embedding_dir)
    fg_dir = Path(args.fullgraph_embedding_dir)
    pre_dir = emb_dir / "pre_embedding_3h"

    meta = json.loads((emb_dir / "meta.json").read_text())
    provenance = {
        "run": meta.get("unique_name") or meta.get("source_unique_name"),
        "checkpoint_epoch": meta.get("checkpoint_epoch"),
        "representation_source_temporal": meta.get("representation_source"),
        "ports": meta.get("ports"),
        "tds": meta.get("tds"),
        "include_temporal_flow_edge_features": meta.get("include_temporal_flow_edge_features"),
        "checkpoint_path": meta.get("checkpoint_path"),
        "correct_reverse_expected": True,
        "preserve_seed_expected": True,
        "pre3h_temporal_available": (pre_dir / "train.npz").is_file(),
        "fullgraph_post_available": (fg_dir / "all.npz").is_file(),
        "fullgraph_label": "random_transductive_diagnostic",
        "incompatible_mix_warning": (
            "Temporal train/val/test embeddings are NOT concatenated for random splits; "
            "random protocols use dedicated full-graph inference only."
        ),
    }
    # verify checkpoint flags if possible
    ckpt_path = Path(str(meta.get("checkpoint_path") or ""))
    if ckpt_path.is_file():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        provenance["checkpoint_flags"] = {
            "correct_reverse_edge_features": ck.get("correct_reverse_edge_features"),
            "reverse_edge_feature_semantics": ck.get("reverse_edge_feature_semantics"),
            "preserve_seed_edges": ck.get("preserve_seed_edges"),
            "ports": ck.get("ports"),
            "tds": ck.get("tds"),
            "epoch": ck.get("epoch"),
        }

    if not provenance["fullgraph_post_available"] and not args.skip_random:
        raise SystemExit(
            f"Missing full-graph embeddings at {fg_dir / 'all.npz'} "
            "(required unless --skip_random)"
        )

    logging.info("Loading CSV + temporal embeddings...")
    cfg = load_data_config(args.data_config)
    spec = get_dataset_spec("Small-HI")
    df = pd.read_csv(Path(cfg["paths"]["aml_data"]) / "Small-HI" / spec.formatted_csv_name())
    df["Timestamp"] = df["Timestamp"] - df["Timestamp"].min()
    raw = load_raw_columns(df)

    emb_post = load_temporal_embeddings(emb_dir)
    emb_pre = load_temporal_embeddings(pre_dir) if provenance["pre3h_temporal_available"] else None

    tf_dir = Path(args.tf_cache_dir)
    tf_features = np.load(tf_dir / "features.npy").astype(np.float32)
    tf_edge_id = np.load(tf_dir / "edge_id.npy").astype(np.int64)
    assert tf_features.shape[0] == tf_edge_id.shape[0]

    if args.skip_random:
        z_all = y_all = e_all = None
    else:
        z_all, y_all, e_all = load_embedding_npz(fg_dir / "all.npz")
        e_all = e_all.astype(np.int64)
        y_csv = df[spec.label_col].to_numpy().astype(np.int64)
        y_check = y_csv[e_all]
        if not np.array_equal(y_check, y_all.astype(np.int64)):
            logging.warning("Full-graph y mismatch vs CSV; using CSV labels for random protocol")
            y_all = y_check

    if args.smoke:
        logging.warning("SMOKE MODE: subsampled data; do not use for GCPAL comparison")
        rng = np.random.RandomState(0)

        def _sub_split(d: Dict[str, np.ndarray], n: int) -> Dict[str, np.ndarray]:
            n = min(n, d["y"].shape[0])
            pos = np.where(d["y"] == 1)[0]
            neg = np.where(d["y"] == 0)[0]
            n_pos = min(max(20, n // 50), pos.size, n // 2)
            n_neg = min(n - n_pos, neg.size)
            idx = np.concatenate(
                [rng.choice(pos, n_pos, replace=False), rng.choice(neg, n_neg, replace=False)]
            )
            rng.shuffle(idx)
            return {k: v[idx] for k, v in d.items()}

        emb_post = {s: _sub_split(emb_post[s], args.smoke_n) for s in emb_post}
        if emb_pre is not None:
            emb_pre = {s: _sub_split(emb_pre[s], args.smoke_n) for s in emb_pre}
        if z_all is not None:
            n_fg = min(args.smoke_n * 3, z_all.shape[0])
            pos = np.where(y_all == 1)[0]
            neg = np.where(y_all == 0)[0]
            n_pos = min(max(50, n_fg // 50), pos.size, n_fg // 2)
            n_neg = min(n_fg - n_pos, neg.size)
            fg_idx = np.concatenate(
                [rng.choice(pos, n_pos, replace=False), rng.choice(neg, n_neg, replace=False)]
            )
            rng.shuffle(fg_idx)
            z_all, y_all, e_all = z_all[fg_idx], y_all[fg_idx], e_all[fg_idx]
        provenance["smoke"] = {
            "smoke_n": args.smoke_n,
            "mlp_epochs": MLP_EPOCHS,
            "classifier_seeds": list(CLASSIFIER_SEEDS),
            "split_seeds": list(SPLIT_SEEDS),
        }

    classifier_configs = {
        "logistic": {
            "impl": "sklearn.linear_model.LogisticRegression",
            "solver": "lbfgs",
            "C": 1.0,
            "max_iter": 1000,
            "seeds": list(CLASSIFIER_SEEDS),
        },
        "mlp": {
            "impl": "torch MLP",
            "hidden": MLP_HIDDEN,
            "dropout": MLP_DROPOUT,
            "epochs": MLP_EPOCHS,
            "batch_size": MLP_BATCH,
            "lr": MLP_LR,
            "seeds": list(CLASSIFIER_SEEDS),
            "paper_primary_loss": "BCE unweighted",
            "secondary_loss": "BCE with thesis gin class weights",
        },
        "feature_stacks": {
            "raw_native": "Timestamp, log1p(Amount), ordinal currency/format; scaler fit train-only",
            "raw_ports_tds": (
                "Not exported as a separate downstream matrix here; ports/TDS were encoder "
                "message-passing features (edge_dim=8) for D, not the thesis probe 'raw' stack."
            ),
            "temporal_flow_causal": list(TEMPORAL_FLOW_CAUSAL_FEATURE_NAMES),
            "embedding_post": "128-d post_embedding from D",
            "pre_embedding_3h": "198-d pre-head representation when available",
        },
    }

    logging.info("Running temporal protocol...")
    t0 = time.perf_counter()
    temporal = run_temporal_protocol(
        emb_post=emb_post,
        emb_pre=emb_pre,
        raw=raw,
        tf_features=tf_features,
        tf_edge_id=tf_edge_id,
        device=device,
    )
    logging.info("Temporal done in %.1fs", time.perf_counter() - t0)

    if args.skip_random:
        logging.warning("Skipping random protocols (--skip_random)")
        random60 = {"skipped": True, "pooled_over_split_seeds": {}}
        random40 = {"skipped": True, "pooled_over_split_seeds": {}}
    else:
        logging.info("Running random 60% protocol...")
        t0 = time.perf_counter()
        random60 = run_random_protocol(
            z_all=z_all, y_all=y_all, edge_id_all=e_all, raw=raw, train_frac=0.6, device=device
        )
        logging.info("Random60 done in %.1fs", time.perf_counter() - t0)

        logging.info("Running random 40% protocol...")
        t0 = time.perf_counter()
        random40 = run_random_protocol(
            z_all=z_all, y_all=y_all, edge_id_all=e_all, raw=raw, train_frac=0.4, device=device
        )
        logging.info("Random40 done in %.1fs", time.perf_counter() - t0)

    payload = {
        "title": "Forensic GCPAL vs evaluation-protocol audit",
        "provenance": provenance,
        "gcpal_table2": GCPAL_TABLE2,
        "unresolved_paper_details": [
            "AMLWorld split type",
            "validation protocol",
            "threshold rule",
            "class weighting",
            "X feature processing",
            "optimizer details",
        ],
        "classifier_configs": classifier_configs,
        "temporal_protocol": temporal,
        "random_protocol_60": random60,
        "random_protocol_40": random40,
    }
    payload["decision"] = decide(payload)

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(_jsonable(payload), indent=2) + "\n", encoding="utf-8")
    write_md(out_md, payload)
    logging.info("DECISION %s", payload["decision"]["verdict"])
    print(out_json)
    print(out_md)


if __name__ == "__main__":
    main()
