#!/usr/bin/env python3
"""Bounded frozen D+ improvement sprint (no encoder training).

Experiment A: equal-weight probability ensemble of locked best-score seeds.
Experiment B: validation-only gate for seed1/3 epoch-40 ``_last`` checkpoints.

Hard constraints: frozen encoder only; no new architecture/objective/sweeps;
no test-driven selection; do not modify verified final analysis metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import importlib.util

from gcpal_txn_node.eval_mlp import PaperStyleMLP, _predict_proba, _select_threshold_f1
from linear_probe import load_embedding_npz
from ranking_metrics import alert_budget_metrics
from util import logger_setup, set_seed

_spec = importlib.util.spec_from_file_location(
    "probe_feature_ablation", ROOT / "scripts" / "probe_feature_ablation.py"
)
_pfa = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules["probe_feature_ablation"] = _pfa
_spec.loader.exec_module(_pfa)
build_full_feature_matrix = _pfa.build_full_feature_matrix
load_dataset_frames = _pfa.load_dataset_frames

TF_CACHE = ROOT / "results/cache/temporal_flow_causal/Small-HI"
STACK_DIM = 227
MLP_EPOCHS = 15
MLP_LR = 1e-3
MLP_BS = 8192
DOWNSTREAM_SEED = 2
METRIC_TOL = 1e-6

LOCKED_FINAL = ROOT / "results/diagnostics/final_dplus_multiseed_and_finetune_analysis.json"
SCORE_DIR = ROOT / "results/diagnostics/frozen_dplus_sprint_scores"
SPRINT_JSON = ROOT / "results/diagnostics/final_frozen_dplus_improvement_sprint.json"
SPRINT_MD = ROOT / "notes/final_frozen_dplus_improvement_sprint.md"

BEST_SEEDS = {
    1: {
        "unique_name": "edge_dplus_corrected_preserve_40ep_seed1_final",
        "emb_dir": ROOT
        / "embeddings/edge_dplus_corrected_preserve_40ep_seed1_final/pre_embedding_3h",
        "ckpt": ROOT
        / "saved-models/checkpoint_edge_dplus_corrected_preserve_40ep_seed1_final.tar",
        "epoch": 34,
        "sha256": "7bc393f02e552063524671837974991294423ab902786a898e3128489f68afb7",
    },
    2: {
        "unique_name": "gin_emlps_ports_tds_corrected_preserve_seed_asym_proj_8192neg_queue0_40ep_seed2",
        "emb_dir": ROOT
        / "embeddings/gin_emlps_ports_tds_corrected_preserve_seed_asym_proj_8192neg_queue0_40ep_seed2/pre_embedding_3h",
        "ckpt": ROOT
        / "saved-models/checkpoint_gin_emlps_ports_tds_corrected_preserve_seed_asym_proj_8192neg_queue0_40ep_seed2.tar",
        "epoch": 40,
        "sha256": "a320920141f585c5825cbd63ce760a845fb434a9b162d4c87270dc72b0442b87",
    },
    3: {
        "unique_name": "edge_dplus_corrected_preserve_40ep_seed3_final",
        "emb_dir": ROOT
        / "embeddings/edge_dplus_corrected_preserve_40ep_seed3_final/pre_embedding_3h",
        "ckpt": ROOT
        / "saved-models/checkpoint_edge_dplus_corrected_preserve_40ep_seed3_final.tar",
        "epoch": 29,
        "sha256": "c8f95e982e1e46f83cdcd0adc4533ddac6b996030669fee0b961def9a868e36b",
    },
}

LAST_SEEDS = {
    1: {
        "unique_name": "edge_dplus_corrected_preserve_40ep_seed1_final",
        "embeddings_subdir": "edge_dplus_corrected_preserve_40ep_seed1_final_last",
        "checkpoint_suffix": "_last",
        "ckpt": ROOT
        / "saved-models/checkpoint_edge_dplus_corrected_preserve_40ep_seed1_final_last.tar",
        "expected_epoch": 40,
    },
    3: {
        "unique_name": "edge_dplus_corrected_preserve_40ep_seed3_final",
        "embeddings_subdir": "edge_dplus_corrected_preserve_40ep_seed3_final_last",
        "checkpoint_suffix": "_last",
        "ckpt": ROOT
        / "saved-models/checkpoint_edge_dplus_corrected_preserve_40ep_seed3_final_last.tar",
        "expected_epoch": 40,
    },
}

LOCKED_VAL_AUPRC = {1: 0.5193476063333158, 2: 0.5500001348997521, 3: 0.5409943503221283}
LOCKED_AGG_VAL_AUPRC = 0.5367806971850654
GATE_DELTA = 0.005
PUBLISHED_MULTIGIN_EU = 0.6479


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
        "n_duplicate_rows": int(ids.shape[0] - np.unique(ids).shape[0]),
        "edge_id_sum": int(ids.sum()),
        "sha256_of_ids_bytes": hashlib.sha256(ids.tobytes()).hexdigest(),
    }


def metrics_block(y: np.ndarray, proba: np.ndarray, thr: float) -> Dict[str, float]:
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


def _fit_mlp(x_tr, y_tr, device, seed: int = DOWNSTREAM_SEED) -> PaperStyleMLP:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = PaperStyleMLP(int(x_tr.shape[1])).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=MLP_LR)
    x_t = torch.from_numpy(x_tr.astype(np.float32))
    y_t = torch.from_numpy(y_tr.astype(np.float32))
    n = x_tr.shape[0]
    model.train()
    for ep in range(MLP_EPOCHS):
        perm = np.random.RandomState(seed * 1009 + ep).permutation(n)
        for start in range(0, n, MLP_BS):
            idx = perm[start : start + MLP_BS]
            xb = x_t[idx].to(device)
            yb = y_t[idx].to(device)
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(model(xb), yb)
            loss.backward()
            opt.step()
    return model


def train_mlp_return_proba(
    x_tr, y_tr, x_va, y_va, x_te, y_te, device, seed: int = DOWNSTREAM_SEED
):
    model = _fit_mlp(x_tr, y_tr, device, seed=seed)
    proba_te = _predict_proba(model, x_te, batch_size=MLP_BS, device=device)
    proba_va = _predict_proba(model, x_va, batch_size=MLP_BS, device=device)
    thr = _select_threshold_f1(y_va, proba_va)
    metrics = {
        "val_ranking": {
            "auprc": float(average_precision_score(y_va, proba_va)),
            "auroc": float(roc_auc_score(y_va, proba_va)),
            "n": float(y_va.shape[0]),
        },
        "val_at_selected_threshold": metrics_block(y_va, proba_va, thr),
        "threshold_0.5": metrics_block(y_te, proba_te, 0.5),
        "threshold_val_selected": {
            **metrics_block(y_te, proba_te, thr),
            "validation_selected_threshold": float(thr),
        },
        "validation_selected_threshold": float(thr),
        "learner": "mlp",
        "weight": "none",
        "mlp_epochs": MLP_EPOCHS,
        "mlp_lr": MLP_LR,
        "downstream_seed": seed,
    }
    return metrics, proba_va, proba_te


def train_mlp_val_only(x_tr, y_tr, x_va, y_va, device, seed: int = DOWNSTREAM_SEED):
    """Fit MLP and score validation only (no test forward pass)."""
    model = _fit_mlp(x_tr, y_tr, device, seed=seed)
    proba_va = _predict_proba(model, x_va, batch_size=MLP_BS, device=device)
    thr = _select_threshold_f1(y_va, proba_va)
    return {
        "model": model,
        "proba_va": proba_va,
        "thr": float(thr),
        "val_ranking": {
            "auprc": float(average_precision_score(y_va, proba_va)),
            "auroc": float(roc_auc_score(y_va, proba_va)),
            "n": float(y_va.shape[0]),
        },
        "val_at_selected_threshold": metrics_block(y_va, proba_va, thr),
        "validation_selected_threshold": float(thr),
    }


def load_hxxtf_splits(
    emb_dir: Path, *, row_order: str = "npz"
) -> Dict[str, Dict[str, np.ndarray]]:
    """Load H+X+TF stacks.

    row_order:
      - ``npz``: preserve each split ``.npz`` row order (matches
        ``eval_frozen_dplus_hxxtf_mlp.py`` / seeds 1 & 3 locked evals).
      - ``temporal_split``: reorder to temporal split ID order with embedding
        intersection (matches ``gcpal_challenge_fullstack_eval.py`` / seed-2
        job 18678029). Seeded MLP minibatches are index-based, so row order
        changes the training trajectory.
    """
    if row_order not in ("npz", "temporal_split"):
        raise ValueError(row_order)
    if not (TF_CACHE / "features.npy").is_file():
        raise SystemExit(f"missing TF cache {TF_CACHE / 'features.npy'}")
    df, df_train, tr_ids, va_ids, te_ids, spec = load_dataset_frames(
        "Small-HI", str(ROOT / "data_config.json")
    )
    y_all = df[spec.label_col].to_numpy().astype(np.int64)
    x_raw, _, _, _ = build_full_feature_matrix(
        df, df_train, ("edge_native",), categorical_encoding="one_hot"
    )
    tf_feat = np.load(TF_CACHE / "features.npy").astype(np.float32)
    assert x_raw.shape[1] == 24 and tf_feat.shape[1] == 5

    if row_order == "npz":
        splits = {}
        for sp in ("train", "val", "test"):
            z, y, ids = load_embedding_npz(emb_dir / f"{sp}.npz")
            if z.shape[1] != 198:
                raise SystemExit(f"expected H dim 198, got {z.shape[1]} in {emb_dir}/{sp}.npz")
            if not np.array_equal(y, y_all[ids]):
                raise SystemExit(f"label mismatch on {sp} in {emb_dir}")
            hxxtf = np.concatenate([z, x_raw[ids], tf_feat[ids]], axis=1).astype(np.float32)
            if hxxtf.shape[1] != STACK_DIM:
                raise SystemExit(f"stack dim {hxxtf.shape[1]} != {STACK_DIM}")
            splits[sp] = {"X": hxxtf, "y": y.astype(np.int64), "ids": ids.astype(np.int64)}
        return splits

    # temporal_split order (fullstack 18678029)
    z_map: Dict[int, np.ndarray] = {}
    y_map: Dict[int, int] = {}
    for sp in ("train", "val", "test"):
        z, y, eid = load_embedding_npz(emb_dir / f"{sp}.npz")
        if z.shape[1] != 198:
            raise SystemExit(f"expected H dim 198, got {z.shape[1]}")
        for i, e in enumerate(eid.tolist()):
            z_map[int(e)] = z[i]
            y_map[int(e)] = int(y[i])
    splits = {}
    for sp, expected_ids in (("train", tr_ids), ("val", va_ids), ("test", te_ids)):
        expected = np.asarray(expected_ids, dtype=np.int64)
        keep = np.array([int(i) in z_map for i in expected.tolist()], dtype=bool)
        ids = expected[keep]
        z = np.stack([z_map[int(i)] for i in ids.tolist()], axis=0)
        y = np.array([y_map[int(i)] for i in ids.tolist()], dtype=np.int64)
        if not np.array_equal(y, y_all[ids]):
            raise SystemExit(f"label mismatch on temporal-aligned {sp}")
        hxxtf = np.concatenate([z, x_raw[ids], tf_feat[ids]], axis=1).astype(np.float32)
        if hxxtf.shape[1] != STACK_DIM:
            raise SystemExit(f"stack dim {hxxtf.shape[1]} != {STACK_DIM}")
        splits[sp] = {"X": hxxtf, "y": y, "ids": ids}
    return splits


def _within_tol(a: float, b: float, tol: float = METRIC_TOL) -> bool:
    return abs(float(a) - float(b)) <= tol


def reproduce_best_seed(seed: int, device: torch.device, force: bool = False) -> Dict[str, Any]:
    cfg = BEST_SEEDS[seed]
    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    score_path = SCORE_DIR / f"seed{seed}_best_scores.npz"
    meta_path = SCORE_DIR / f"seed{seed}_best_meta.json"
    if score_path.is_file() and meta_path.is_file() and not force:
        meta = json.loads(meta_path.read_text())
        logging.info("Reusing existing scores %s", score_path)
        return meta

    emb_dir = cfg["emb_dir"]
    for sp in ("train", "val", "test"):
        if not (emb_dir / f"{sp}.npz").is_file():
            raise SystemExit(f"missing verified embedding {emb_dir / f'{sp}.npz'}")
    ckpt_sha = sha256_file(cfg["ckpt"])
    if ckpt_sha != cfg["sha256"]:
        raise SystemExit(f"seed{seed} ckpt sha mismatch: {ckpt_sha} != {cfg['sha256']}")
    ckpt = torch.load(cfg["ckpt"], map_location="cpu")
    epoch = int(ckpt.get("epoch", -1))
    if epoch != cfg["epoch"]:
        raise SystemExit(f"seed{seed} epoch {epoch} != expected {cfg['epoch']}")

    set_seed(DOWNSTREAM_SEED)
    # Seed 2 locked metrics come from fullstack job 18678029 (temporal ID order).
    # Seeds 1/3 locked metrics come from eval_frozen_dplus_hxxtf_mlp (npz order).
    row_order = "temporal_split" if seed == 2 else "npz"
    splits = load_hxxtf_splits(emb_dir, row_order=row_order)
    scaler = StandardScaler()
    x_tr = scaler.fit_transform(splits["train"]["X"]).astype(np.float32)
    x_va = scaler.transform(splits["val"]["X"]).astype(np.float32)
    x_te = scaler.transform(splits["test"]["X"]).astype(np.float32)
    metrics, proba_va, proba_te = train_mlp_return_proba(
        x_tr,
        splits["train"]["y"],
        x_va,
        splits["val"]["y"],
        x_te,
        splits["test"]["y"],
        device=device,
        seed=DOWNSTREAM_SEED,
    )

    locked = json.loads(LOCKED_FINAL.read_text())
    locked_seed = next(s for s in locked["per_seed"] if int(s["encoder_seed"]) == seed)
    checks = {
        "val_auprc": (metrics["val_ranking"]["auprc"], locked_seed["val_auprc"]),
        "test_auprc": (metrics["threshold_0.5"]["auprc"], locked_seed["test_auprc"]),
        "test_f1_0.5": (metrics["threshold_0.5"]["f1"], locked_seed["test_f1_0.5"]),
        "test_auroc": (metrics["threshold_0.5"]["auroc"], locked_seed["test_auroc"]),
    }
    mismatches = {
        k: {"reproduced": a, "locked": b, "abs_diff": abs(a - b)}
        for k, (a, b) in checks.items()
        if not _within_tol(a, b)
    }
    if mismatches:
        raise SystemExit(
            f"ABORT seed{seed}: reproduced metrics disagree with locked final: "
            + json.dumps(mismatches, indent=2)
        )

    np.savez_compressed(
        score_path,
        val_ids=splits["val"]["ids"],
        val_y=splits["val"]["y"],
        val_proba=proba_va.astype(np.float64),
        test_ids=splits["test"]["ids"],
        test_y=splits["test"]["y"],
        test_proba=proba_te.astype(np.float64),
    )
    meta = {
        "encoder_seed": seed,
        "checkpoint_epoch": epoch,
        "checkpoint_sha256": ckpt_sha,
        "unique_name": cfg["unique_name"],
        "embeddings_dir": str(emb_dir),
        "score_path": str(score_path),
        "metrics": metrics,
        "locked_match": True,
        "metric_tol": METRIC_TOL,
        "checks": {k: {"reproduced": a, "locked": b} for k, (a, b) in checks.items()},
        "val_ids": ids_hash(splits["val"]["ids"]),
        "test_ids": ids_hash(splits["test"]["ids"]),
        "n_pos_test": int((splits["test"]["y"] == 1).sum()),
        "row_order": row_order,
        "gnn_training_occurred": False,
        "aml_labels_updated_encoder": False,
        "extraction_rerun": False,
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    logging.info("Wrote %s and %s", score_path, meta_path)
    return meta


def _align_maps(ids: np.ndarray, y: np.ndarray, proba: np.ndarray) -> Dict[int, Tuple[int, float]]:
    out: Dict[int, Tuple[int, float]] = {}
    for i, yi, p in zip(ids.tolist(), y.tolist(), proba.tolist()):
        ii = int(i)
        if ii in out and (out[ii][0] != int(yi) or abs(out[ii][1] - float(p)) > 1e-12):
            raise SystemExit(f"duplicate id {ii} with inconsistent label/proba")
        out[ii] = (int(yi), float(p))
    return out


def run_ensemble() -> Dict[str, Any]:
    metas = {}
    maps_val = {}
    maps_te = {}
    for seed in (1, 2, 3):
        meta_path = SCORE_DIR / f"seed{seed}_best_meta.json"
        score_path = SCORE_DIR / f"seed{seed}_best_scores.npz"
        if not meta_path.is_file() or not score_path.is_file():
            raise SystemExit(f"missing reproduced scores for seed{seed}")
        metas[seed] = json.loads(meta_path.read_text())
        if not metas[seed].get("locked_match"):
            raise SystemExit(f"seed{seed} locked_match false")
        z = np.load(score_path)
        maps_val[seed] = _align_maps(z["val_ids"], z["val_y"], z["val_proba"])
        maps_te[seed] = _align_maps(z["test_ids"], z["test_y"], z["test_proba"])

    inter_val = set(maps_val[1]) & set(maps_val[2]) & set(maps_val[3])
    inter_te = set(maps_te[1]) & set(maps_te[2]) & set(maps_te[3])
    val_ids = np.array(sorted(inter_val), dtype=np.int64)
    te_ids = np.array(sorted(inter_te), dtype=np.int64)

    def gather(maps, ids):
        y = np.array([maps[1][int(i)][0] for i in ids], dtype=np.int64)
        for s in (2, 3):
            y_s = np.array([maps[s][int(i)][0] for i in ids], dtype=np.int64)
            if not np.array_equal(y, y_s):
                raise SystemExit(f"label mismatch across seeds on aligned IDs ({s})")
        probs = np.stack(
            [
                np.array([maps[s][int(i)][1] for i in ids], dtype=np.float64)
                for s in (1, 2, 3)
            ],
            axis=0,
        )
        return y, probs.mean(axis=0)

    y_va, p_va = gather(maps_val, val_ids)
    y_te, p_te = gather(maps_te, te_ids)
    n_pos_te = int((y_te == 1).sum())
    # all positives from each seed retained?
    pos_sets = {}
    for s in (1, 2, 3):
        z = np.load(SCORE_DIR / f"seed{s}_best_scores.npz")
        pos_sets[s] = set(z["test_ids"][z["test_y"] == 1].astype(np.int64).tolist())
    pos_union = pos_sets[1] | pos_sets[2] | pos_sets[3]
    pos_inter_seeds = pos_sets[1] & pos_sets[2] & pos_sets[3]
    pos_in_ensemble = set(te_ids[y_te == 1].tolist())
    if pos_inter_seeds - pos_in_ensemble:
        raise SystemExit("ensemble dropped some positive IDs present in all seeds")
    if n_pos_te != 1611 or len(pos_inter_seeds) != 1611:
        # still report; fail hard only if positives missing from intersection of seeds
        if len(pos_inter_seeds) != 1611:
            raise SystemExit(f"expected 1611 positives in all seeds, got {len(pos_inter_seeds)}")

    thr = _select_threshold_f1(y_va, p_va)
    report = {
        "classification": (
            "Post-hoc equal-weight ensemble of independently self-supervised frozen D+ encoders."
        ),
        "role": "SECONDARY_frozen_equal_weight_ensemble",
        "ensemble_rule": "mean(seed1_probability, seed2_probability, seed3_probability)",
        "weights_searched": False,
        "logit_averaging_compared": False,
        "seed_dropped_by_test": False,
        "intersection": {
            "val": {
                **ids_hash(val_ids),
                "n_pos": int((y_va == 1).sum()),
            },
            "test": {
                **ids_hash(te_ids),
                "n_pos": n_pos_te,
                "all_1611_positives_retained": n_pos_te == 1611,
                "pos_intersection_across_seeds": len(pos_inter_seeds),
                "pos_union_across_seeds": len(pos_union),
            },
            "labels_identical_for_aligned_ids": True,
            "unique_ids": True,
        },
        "per_seed_reproduced": {
            str(s): {
                "checkpoint_epoch": metas[s]["checkpoint_epoch"],
                "test_f1_0.5": metas[s]["metrics"]["threshold_0.5"]["f1"],
                "test_auprc": metas[s]["metrics"]["threshold_0.5"]["auprc"],
                "val_auprc": metas[s]["metrics"]["val_ranking"]["auprc"],
                "locked_match": True,
            }
            for s in (1, 2, 3)
        },
        "validation_selected_threshold": float(thr),
        "val_ranking": {
            "auprc": float(average_precision_score(y_va, p_va)),
            "auroc": float(roc_auc_score(y_va, p_va)),
            "n": float(y_va.shape[0]),
        },
        "test_metrics_threshold_0.5": metrics_block(y_te, p_te, 0.5),
        "test_metrics_val_threshold": {
            **metrics_block(y_te, p_te, thr),
            "validation_selected_threshold": float(thr),
        },
        "comparisons": {
            "frozen_three_seed_mean_f1_0.5": 0.6392612830279665,
            "frozen_three_seed_sd_f1_0.5": 0.018588281152646165,
            "frozen_seed2_f1_0.5": 0.6559172497718284,
            "published_multigin_eu_f1": PUBLISHED_MULTIGIN_EU,
        },
        "gnn_training_occurred": False,
        "aml_labels_updated_encoder": False,
        "test_used_for_selection": False,
        "does_not_replace_three_seed_mean_robustness": True,
    }
    f1 = report["test_metrics_threshold_0.5"]["f1"]
    report["numerically_exceeds_multigin_eu_f1"] = bool(f1 > PUBLISHED_MULTIGIN_EU)
    report["delta_vs_multigin_eu_f1"] = float(f1 - PUBLISHED_MULTIGIN_EU)
    out = SCORE_DIR / "equal_weight_ensemble.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    logging.info("Wrote ensemble %s", out)
    return report


def extract_last(seed: int) -> Path:
    cfg = LAST_SEEDS[seed]
    emb_dir = ROOT / "embeddings" / cfg["embeddings_subdir"] / "pre_embedding_3h"
    if all((emb_dir / f"{sp}.npz").is_file() for sp in ("train", "val", "test")):
        logging.info("Reusing existing _last embeddings %s", emb_dir)
        return emb_dir
    ckpt = torch.load(cfg["ckpt"], map_location="cpu")
    epoch = int(ckpt.get("epoch", -1))
    if epoch != cfg["expected_epoch"]:
        raise SystemExit(f"seed{seed} _last epoch {epoch} != {cfg['expected_epoch']}")
    cmd = [
        sys.executable,
        str(ROOT / "embedding_extraction.py"),
        "--data",
        "Small-HI",
        "--model",
        "gin",
        "--tqdm",
        "--batch_size",
        "8192",
        "--num_neighs",
        "100",
        "100",
        "--loader_num_workers",
        "0",
        "--seed",
        str(seed),
        "--reverse_mp",
        "--ego",
        "--ports",
        "--emlps",
        "--tds",
        "--correct_reverse_edge_features",
        "--unique_name",
        cfg["unique_name"],
        "--checkpoint_suffix",
        cfg["checkpoint_suffix"],
        "--embeddings_subdir",
        cfg["embeddings_subdir"],
        "--representation_source",
        "pre_embedding_3h",
        "--testing",
    ]
    logging.info("Extracting _last seed%s: %s", seed, " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(ROOT))
    return emb_dir


def eval_last_val_only(seed: int, device: torch.device, force: bool = False) -> Dict[str, Any]:
    """Extract + train/val only. No test forward pass during the gate."""
    out = SCORE_DIR / f"seed{seed}_last_val_gate.json"
    model_path = SCORE_DIR / f"seed{seed}_last_mlp_state.pt"
    if out.is_file() and model_path.is_file() and not force:
        return json.loads(out.read_text())
    cfg = LAST_SEEDS[seed]
    emb_dir = extract_last(seed)
    ckpt_sha = sha256_file(cfg["ckpt"])
    ckpt = torch.load(cfg["ckpt"], map_location="cpu")
    epoch = int(ckpt.get("epoch", -1))
    set_seed(DOWNSTREAM_SEED)
    splits = load_hxxtf_splits(emb_dir)
    scaler = StandardScaler()
    x_tr = scaler.fit_transform(splits["train"]["X"]).astype(np.float32)
    x_va = scaler.transform(splits["val"]["X"]).astype(np.float32)
    pack = train_mlp_val_only(
        x_tr, splits["train"]["y"], x_va, splits["val"]["y"], device=device, seed=DOWNSTREAM_SEED
    )
    # Persist scaler + MLP for optional post-gate test only
    torch.save(
        {
            "model_state": pack["model"].cpu().state_dict(),
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "thr": pack["thr"],
            "emb_dir": str(emb_dir),
            "d_in": STACK_DIM,
        },
        model_path,
    )
    np.savez_compressed(
        SCORE_DIR / f"seed{seed}_last_val_scores.npz",
        val_ids=splits["val"]["ids"],
        val_y=splits["val"]["y"],
        val_proba=pack["proba_va"].astype(np.float64),
    )
    report = {
        "encoder_seed": seed,
        "checkpoint": str(cfg["ckpt"]),
        "checkpoint_sha256": ckpt_sha,
        "checkpoint_epoch": epoch,
        "embeddings_dir": str(emb_dir),
        "val_auprc": pack["val_ranking"]["auprc"],
        "val_auroc": pack["val_ranking"]["auroc"],
        "val_f1_at_selected": pack["val_at_selected_threshold"]["f1"],
        "validation_selected_threshold": pack["validation_selected_threshold"],
        "locked_best_val_auprc": LOCKED_VAL_AUPRC[seed],
        "delta_val_auprc_vs_locked_best": float(
            pack["val_ranking"]["auprc"] - LOCKED_VAL_AUPRC[seed]
        ),
        "test_metrics_inspected_for_gate": False,
        "test_forward_pass_during_gate": False,
        "gnn_training_occurred": False,
        "aml_labels_updated_encoder": False,
        "mlp_state_path": str(model_path),
    }
    out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def _score_last_test_after_gate(seed: int, device: torch.device) -> Dict[str, Any]:
    """Only called if validation gate passes. Re-fits scaler on train (deterministic) then scores test."""
    model_path = SCORE_DIR / f"seed{seed}_last_mlp_state.pt"
    blob = torch.load(model_path, map_location="cpu")
    emb_dir = Path(blob["emb_dir"])
    # Verify saved scaler matches a fresh train fit (protocol safety)
    splits = load_hxxtf_splits(emb_dir)
    scaler = StandardScaler()
    scaler.fit(splits["train"]["X"])
    if not np.allclose(scaler.mean_, blob["scaler_mean"]) or not np.allclose(
        scaler.scale_, blob["scaler_scale"]
    ):
        raise SystemExit(f"seed{seed} scaler mismatch vs gate-time fit")
    x_te = scaler.transform(splits["test"]["X"]).astype(np.float32)
    x_va = scaler.transform(splits["val"]["X"]).astype(np.float32)
    model = PaperStyleMLP(STACK_DIM)
    model.load_state_dict(blob["model_state"])
    model.to(device)
    model.eval()
    proba_te = _predict_proba(model, x_te, batch_size=MLP_BS, device=device)
    proba_va = _predict_proba(model, x_va, batch_size=MLP_BS, device=device)
    thr = float(blob["thr"])
    metrics = {
        "val_ranking": {
            "auprc": float(average_precision_score(splits["val"]["y"], proba_va)),
            "auroc": float(roc_auc_score(splits["val"]["y"], proba_va)),
            "n": float(splits["val"]["y"].shape[0]),
        },
        "threshold_0.5": metrics_block(splits["test"]["y"], proba_te, 0.5),
        "threshold_val_selected": {
            **metrics_block(splits["test"]["y"], proba_te, thr),
            "validation_selected_threshold": thr,
        },
        "validation_selected_threshold": thr,
    }
    np.savez_compressed(
        SCORE_DIR / f"seed{seed}_last_scores.npz",
        val_ids=splits["val"]["ids"],
        val_y=splits["val"]["y"],
        val_proba=proba_va.astype(np.float64),
        test_ids=splits["test"]["ids"],
        test_y=splits["test"]["y"],
        test_proba=proba_te.astype(np.float64),
    )
    return metrics


def run_gate_and_optional_test(device: torch.device) -> Dict[str, Any]:
    g1 = eval_last_val_only(1, device)
    g3 = eval_last_val_only(3, device)
    s2_val = LOCKED_VAL_AUPRC[2]
    vals = [g1["val_auprc"], s2_val, g3["val_auprc"]]
    agg = float(np.mean(vals))
    threshold = LOCKED_AGG_VAL_AUPRC + GATE_DELTA
    passed = agg >= threshold
    gate = {
        "policy": "fixed_horizon_epoch_40_last",
        "seed1_last_val_auprc": g1["val_auprc"],
        "seed2_ep40_val_auprc": s2_val,
        "seed3_last_val_auprc": g3["val_auprc"],
        "fixed_ep40_val_auprc_mean": agg,
        "locked_best_val_auprc_mean": LOCKED_AGG_VAL_AUPRC,
        "required_improvement": GATE_DELTA,
        "pass_threshold": threshold,
        "gate_passed": passed,
        "test_evaluation_permitted": passed,
        "test_inspected_during_gate": False,
        "test_forward_pass_during_gate": False,
        "checkpoint_selection_if_pass": "downstream-validation-informed",
        "seed1_delta_vs_best34": g1["delta_val_auprc_vs_locked_best"],
        "seed3_delta_vs_best29": g3["delta_val_auprc_vs_locked_best"],
    }
    (SCORE_DIR / "epoch40_val_gate.json").write_text(json.dumps(gate, indent=2) + "\n")

    test_block: Optional[Dict[str, Any]] = None
    if passed:
        s1m = _score_last_test_after_gate(1, device)
        s3m = _score_last_test_after_gate(3, device)
        s2m = json.loads((SCORE_DIR / "seed2_best_meta.json").read_text())["metrics"]
        per = []
        for seed, metrics, epoch in (
            (1, s1m, 40),
            (2, s2m, 40),
            (3, s3m, 40),
        ):
            per.append(
                {
                    "encoder_seed": seed,
                    "checkpoint_epoch": epoch,
                    "val_auprc": metrics["val_ranking"]["auprc"],
                    "test_auprc": metrics["threshold_0.5"]["auprc"],
                    "test_auroc": metrics["threshold_0.5"]["auroc"],
                    "test_f1_0.5": metrics["threshold_0.5"]["f1"],
                    "test_f1_val_thr": metrics["threshold_val_selected"]["f1"],
                    "P100": metrics["threshold_0.5"]["precision_at_100"],
                    "P500": metrics["threshold_0.5"]["precision_at_500"],
                    "P1000": metrics["threshold_0.5"]["precision_at_1000"],
                }
            )
        f1s = [p["test_f1_0.5"] for p in per]
        auprcs = [p["test_auprc"] for p in per]
        test_block = {
            "label": "fixed-horizon epoch-40 aggregate (secondary; does not replace best-score)",
            "per_seed": per,
            "aggregate": {
                "test_f1_0.5_mean": float(np.mean(f1s)),
                "test_f1_0.5_sample_std": float(np.std(f1s, ddof=1)),
                "test_auprc_mean": float(np.mean(auprcs)),
                "test_auprc_sample_std": float(np.std(auprcs, ddof=1)),
                "val_auprc_mean": agg,
            },
            "does_not_replace_best_score_primary": True,
            "checkpoint_selection": "downstream-validation-informed",
        }
        (SCORE_DIR / "epoch40_test_aggregate.json").write_text(
            json.dumps(test_block, indent=2) + "\n"
        )
    else:
        logging.info(
            "Gate FAILED (agg val AUPRC %.6f < %.6f); no _last test evaluation.",
            agg,
            threshold,
        )
    return {"gate": gate, "fixed_ep40_test": test_block}


def write_deliverables(ensemble: Dict[str, Any], gate_pack: Dict[str, Any]) -> None:
    gate = gate_pack["gate"]
    ens_f1 = ensemble["test_metrics_threshold_0.5"]["f1"]
    ens_auprc = ensemble["test_metrics_threshold_0.5"]["auprc"]
    exceeds = ensemble["numerically_exceeds_multigin_eu_f1"]
    primary = (
        "Locked three-seed frozen D+ best-score aggregate "
        "(test F1@0.5 0.6393 ± 0.0186) remains the primary robustness result."
    )
    wording = (
        "Primary: a self-supervised contrastive Multi-GIN encoder (D+) evaluated with "
        "the encoder frozen and a supervised downstream MLP on pre-3h H+X+TF achieves "
        "test F1@0.5 of 0.639 ± 0.019 over three encoder seeds (best-score checkpoints). "
        "Secondary: a post-hoc equal-weight ensemble of the same three frozen encoders "
        f"reaches test F1@0.5 of {ens_f1:.4f} and AUPRC of {ens_auprc:.4f}; this is not "
        "the robustness statistic. "
        + (
            "A fixed-horizon epoch-40 (_last) policy was validation-gated and "
            + (
                "passed; its test aggregate is reported separately and does not replace "
                "the best-score primary."
                if gate["gate_passed"]
                else "failed the +0.005 val-AUPRC gate, so _last test was not evaluated."
            )
        )
    )
    payload = {
        "title": "final_frozen_dplus_improvement_sprint",
        "constraints": {
            "no_gnn_or_encoder_training": True,
            "no_supervised_gradients_into_encoder": True,
            "no_new_architecture": True,
            "no_new_contrastive_objective": True,
            "no_feature_learner_weighting_sweep": True,
            "no_test_driven_selection": True,
            "no_automatic_followup_jobs": True,
            "used_existing_checkpoints_and_cached_embeddings_where_possible": True,
        },
        "primary_unchanged": {
            "result": "locked_three_seed_frozen_best_score_aggregate",
            "test_f1_0.5_mean_pm_sd": "0.6393 ± 0.0186",
            "source": str(LOCKED_FINAL),
            "modified": False,
        },
        "experiment_A_equal_weight_ensemble": ensemble,
        "experiment_B_epoch40_last_val_gate": gate_pack,
        "answers": {
            "1_ensemble_fixed_0.5_f1_and_auprc": {
                "f1": ens_f1,
                "auprc": ens_auprc,
            },
            "2_ensemble_numerically_exceeds_multigin_eu": exceeds,
            "3_fixed_epoch40_validation_gate": {
                "val_auprc_mean": gate["fixed_ep40_val_auprc_mean"],
                "locked_best_val_auprc_mean": gate["locked_best_val_auprc_mean"],
                "delta": gate["fixed_ep40_val_auprc_mean"] - gate["locked_best_val_auprc_mean"],
                "required_delta": GATE_DELTA,
                "passed": gate["gate_passed"],
            },
            "4_last_test_evaluation_permitted": gate["test_evaluation_permitted"],
            "5_primary_frozen_result": primary,
            "6_exact_thesis_wording": wording,
            "7_no_encoder_received_supervised_updates": True,
            "8_no_new_training_or_automatic_followup": True,
        },
    }
    SPRINT_JSON.parent.mkdir(parents=True, exist_ok=True)
    SPRINT_MD.parent.mkdir(parents=True, exist_ok=True)
    SPRINT_JSON.write_text(json.dumps(payload, indent=2) + "\n")

    te05 = ensemble["test_metrics_threshold_0.5"]
    tev = ensemble["test_metrics_val_threshold"]
    lines = [
        "# Final frozen D+ improvement sprint",
        "",
        "Secondary frozen evaluations only. **Does not modify** "
        "`final_dplus_multiseed_and_finetune_analysis` primary metrics.",
        "",
        "## Constraints",
        "",
        "- No GNN/encoder training; no supervised gradients into any encoder.",
        "- No new architecture, contrastive objective, or feature/learner/weighting sweep.",
        "- No test-driven selection; no automatic follow-up jobs.",
        "- Used existing best-score embeddings; `_last` extract only for seeds 1 and 3.",
        "",
        "## Primary (unchanged)",
        "",
        primary,
        "",
        "## Experiment A — Equal-weight frozen D+ ensemble",
        "",
        ensemble["classification"],
        "",
        f"- Rule: `{ensemble['ensemble_rule']}` (no weight search; no logit averaging).",
        f"- Common test IDs: **{ensemble['intersection']['test']['n']}** "
        f"(sha256 `{ensemble['intersection']['test']['sha256_of_ids_bytes'][:16]}…`); "
        f"positives retained: **{ensemble['intersection']['test']['n_pos']}** / 1611.",
        f"- Val-selected threshold: **{ensemble['validation_selected_threshold']:.2f}**",
        "",
        "| Split / thr | AUROC | AUPRC | F1 | P | R | PPR |",
        "|-------------|------:|------:|---:|--:|--:|----:|",
        f"| test @0.5 | {te05['auroc']:.4f} | {te05['auprc']:.4f} | **{te05['f1']:.4f}** | "
        f"{te05['precision']:.4f} | {te05['recall']:.4f} | {te05['positive_prediction_rate']:.6f} |",
        f"| test @val-thr | {tev['auroc']:.4f} | {tev['auprc']:.4f} | {tev['f1']:.4f} | "
        f"{tev['precision']:.4f} | {tev['recall']:.4f} | {tev['positive_prediction_rate']:.6f} |",
        "",
        f"- Confusion @0.5: TP={te05['tp']:.0f} FP={te05['fp']:.0f} "
        f"TN={te05['tn']:.0f} FN={te05['fn']:.0f}",
        f"- P@100/500/1000 @0.5: {te05['precision_at_100']:.3f} / "
        f"{te05['precision_at_500']:.3f} / {te05['precision_at_1000']:.3f}",
        "",
        "### Comparisons (fixed-0.5 F1)",
        "",
        f"- Frozen three-seed mean: 0.6393 ± 0.0186",
        f"- Frozen seed-2: 0.6559",
        f"- Multi-GIN+EU: 0.6479 ± 0.0122",
        f"- Ensemble: **{ens_f1:.4f}** (Δ vs Multi-GIN+EU = {ensemble['delta_vs_multigin_eu_f1']:+.4f}; "
        f"exceeds={exceeds})",
        "",
        "## Experiment B — `_last` / fixed epoch-40 validation gate",
        "",
        f"| Seed | Checkpoint | Val AUPRC | vs locked best |",
        f"|-----:|------------|----------:|---------------:|",
        f"| 1 | `_last` ep40 | {gate['seed1_last_val_auprc']:.6f} | "
        f"{gate['seed1_delta_vs_best34']:+.6f} |",
        f"| 2 | best=ep40 | {gate['seed2_ep40_val_auprc']:.6f} | +0.000000 |",
        f"| 3 | `_last` ep40 | {gate['seed3_last_val_auprc']:.6f} | "
        f"{gate['seed3_delta_vs_best29']:+.6f} |",
        f"| **mean** | fixed ep40 | **{gate['fixed_ep40_val_auprc_mean']:.6f}** | "
        f"Δ vs best-mean {gate['fixed_ep40_val_auprc_mean'] - gate['locked_best_val_auprc_mean']:+.6f} |",
        "",
        f"- Gate requires mean val AUPRC ≥ {gate['pass_threshold']:.6f} "
        f"(locked best mean {LOCKED_AGG_VAL_AUPRC:.6f} + {GATE_DELTA}).",
        f"- **Gate passed: {gate['gate_passed']}**. "
        f"`_last` test evaluation permitted: **{gate['test_evaluation_permitted']}**.",
        f"- Test was not inspected during the gate.",
        "",
    ]
    if gate_pack["fixed_ep40_test"] is not None:
        t = gate_pack["fixed_ep40_test"]
        lines += [
            "### Fixed-horizon epoch-40 test aggregate (secondary only)",
            "",
            f"- mean test F1@0.5: {t['aggregate']['test_f1_0.5_mean']:.4f} ± "
            f"{t['aggregate']['test_f1_0.5_sample_std']:.4f}",
            f"- mean test AUPRC: {t['aggregate']['test_auprc_mean']:.4f} ± "
            f"{t['aggregate']['test_auprc_sample_std']:.4f}",
            "- Does **not** replace the best-score primary aggregate.",
            "- Checkpoint selection described as **downstream-validation-informed**.",
            "",
        ]
    else:
        lines += [
            "Gate failed → retained existing best-score checkpoint policy; "
            "no `_last` test evaluation.",
            "",
        ]
    lines += [
        "## Final answers",
        "",
        f"1. Ensemble fixed-0.5 F1 / AUPRC: **{ens_f1:.6f}** / **{ens_auprc:.6f}**",
        f"2. Numerically exceeds Multi-GIN+EU: **{exceeds}**",
        f"3. Fixed-ep40 val gate: mean={gate['fixed_ep40_val_auprc_mean']:.6f} "
        f"(Δ={gate['fixed_ep40_val_auprc_mean'] - LOCKED_AGG_VAL_AUPRC:+.6f}; "
        f"need ≥+{GATE_DELTA}); passed=**{gate['gate_passed']}**",
        f"4. `_last` test permitted: **{gate['test_evaluation_permitted']}**",
        f"5. Primary: locked three-seed best-score frozen aggregate",
        f"6. Wording: {wording}",
        "7. No encoder received supervised updates: **true**",
        "8. No new GNN training / automatic follow-up: **true**",
        "",
    ]
    SPRINT_MD.write_text("\n".join(lines) + "\n")
    logging.info("Wrote %s and %s", SPRINT_JSON, SPRINT_MD)


def main() -> None:
    logger_setup()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--stage",
        required=True,
        choices=(
            "reproduce_seed",
            "ensemble",
            "last_val_gate_seed",
            "gate_decide",
            "all",
        ),
    )
    p.add_argument("--encoder_seed", type=int, default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logging.info("device=%s stage=%s", device, args.stage)

    if args.stage == "reproduce_seed":
        if args.encoder_seed not in (1, 2, 3):
            raise SystemExit("--encoder_seed must be 1/2/3")
        reproduce_best_seed(args.encoder_seed, device, force=args.force)
    elif args.stage == "ensemble":
        run_ensemble()
    elif args.stage == "last_val_gate_seed":
        if args.encoder_seed not in (1, 3):
            raise SystemExit("--encoder_seed must be 1 or 3 for _last gate")
        eval_last_val_only(args.encoder_seed, device, force=args.force)
    elif args.stage == "gate_decide":
        # assumes last val gates already done; still needs device if promoting
        run_gate_and_optional_test(device)
    elif args.stage == "all":
        for s in (1, 2, 3):
            reproduce_best_seed(s, device, force=args.force)
        ensemble = run_ensemble()
        gate_pack = run_gate_and_optional_test(device)
        write_deliverables(ensemble, gate_pack)
    else:
        raise SystemExit(args.stage)


if __name__ == "__main__":
    main()
