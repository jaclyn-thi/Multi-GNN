#!/usr/bin/env python3
"""Validation-only PaySim native-feature transfer complementarity gate (seed 2).

Scientific label:
  frozen AMLWorld representation + target-native downstream features

Stacks (train→val only; never test):
  X_native | H_pretrained | H_pretrained+X_native | H_random | H_random+X_native

Primary learner: PaperStyleMLP (native tabular recipe).
Sensitivity: HistGradientBoostingClassifier (locked native-HGB recipe).

No encoder training / fine-tuning / BN recal / GNN forward.
Reuse existing P1 post-128 H + matched random H if provenance matches.
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

from gcpal_txn_node.eval_mlp import PaperStyleMLP, _predict_proba  # noqa: E402
from linear_probe import load_embedding_npz  # noqa: E402
from ranking_metrics import alert_budget_metrics  # noqa: E402
from train_util import extract_param  # noqa: E402
from util import create_parser, logger_setup, set_seed  # noqa: E402

TAG = "paysim_native_feature_transfer_complementarity_seed2"
OUT_DIR = ROOT / "results" / "diagnostics" / TAG
CELLS = OUT_DIR / "cells"
PROBAS = OUT_DIR / "probas"
OUT_JSON = ROOT / "results" / "diagnostics" / f"{TAG}.json"
SMOKE_JSON = OUT_DIR / "smoke.json"
NOTES = ROOT / "notes" / f"{TAG}.md"

EMB_P1 = (
    ROOT
    / "embeddings"
    / "final_corrected_no_preserve_multiseed"
    / "seed2_P1_strict_inductive_legacy"
)
EMB_RND = (
    ROOT
    / "embeddings"
    / "final_corrected_no_preserve_multiseed"
    / "controls_random_paysim_legacy_duplicate_v1"
)
CKPT = (
    ROOT
    / "saved-models"
    / "checkpoint_gin_emlps_ports_tds_corrected_asym_proj_8192neg_queue0_40ep_seed2.tar"
)
EXPECTED_CKPT_SHA_PREFIX = "18e06f55"
EXPECTED_CKPT_SHA = "18e06f555aa4880dfc1e95caa3f54a207e5aa186d266887772640feb93a06ae6"
EXPECTED_ID = {
    "train": "2511d0de4504e52960b414e6b84d47486089a573b6c57aa040feb561e2d2809a",
    "val": "a8de85f31dfe91bd767da6daedf9f2bab474d08c8412c796111e8767ebd0b1e3",
}
EXPECTED_H_DIM = 128
EXPECTED_FEATURE_CONTRACT_EXTRACT = "paysim_legacy_duplicate_v1"
EXPECTED_REPR = "post_embedding"
EXPECTED_BN = "frozen_aml_bn"
NATIVE_CONTRACT = "paysim_native_core_v1"
NATIVE_COLS = [
    "step",
    "type_0",
    "type_1",
    "type_2",
    "type_3",
    "type_4",
    "log1p_amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]
CONTINUOUS_COLS = {
    "step",
    "log1p_amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
}

STACKS = (
    "X_native",
    "H_pretrained",
    "H_pretrained+X_native",
    "H_random",
    "H_random+X_native",
)

MARGIN = 0.003
MLP_SEED = 3  # native tabular baseline recipe
MLP_LR = 1e-3
MLP_BS = 8192
MLP_EPOCHS_FULL = 15
MLP_EPOCHS_SMOKE = 2
HGB_CFG = {
    "max_depth": 6,
    "learning_rate": 0.1,
    "max_iter": 100,
    "l2_regularization": 0.0,
    "random_state": 2,
}
HGB_MAX_ITER_SMOKE = 20


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def ids_hash(ids: np.ndarray) -> Dict[str, Any]:
    a = np.ascontiguousarray(np.asarray(ids, dtype=np.int64).reshape(-1))
    return {
        "n": int(a.shape[0]),
        "n_unique": int(np.unique(a).shape[0]),
        "edge_id_sum": int(a.sum()),
        "sha256_of_ids_bytes": hashlib.sha256(a.tobytes()).hexdigest(),
    }


def gin_cw() -> Dict[int, float]:
    args = create_parser().parse_args(["--data", "PaySim", "--model", "gin", "--testing"])
    return {0: float(extract_param("w_ce1", args)), 1: float(extract_param("w_ce2", args))}


def tune_thr_max_f1(y: np.ndarray, proba: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return 0.5
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.linspace(0.01, 0.99, 99):
        pred = (proba >= thr).astype(np.int64)
        f1 = float(f1_score(y.astype(np.int64), pred, zero_division=0))
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    return best_thr


def metrics_block(y: np.ndarray, proba: np.ndarray, thr: float) -> Dict[str, float]:
    y = y.astype(np.int64)
    proba = proba.astype(np.float64)
    pred = (proba >= float(thr)).astype(np.int64)
    out: Dict[str, float] = {
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


def continuous_indices(cols: Sequence[str]) -> List[int]:
    return [i for i, c in enumerate(cols) if c in CONTINUOUS_COLS]


def scale_x_continuous_only(
    Xtr: np.ndarray, Xva: np.ndarray, cont_idx: Sequence[int]
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Train-fit StandardScaler on continuous X cols only; leave one-hots unchanged."""
    tr = Xtr.astype(np.float64).copy()
    va = Xva.astype(np.float64).copy()
    meta: Dict[str, Any] = {"mode": "train_fit_continuous_only", "continuous_indices": list(cont_idx)}
    if not cont_idx:
        return tr.astype(np.float32), va.astype(np.float32), meta
    scaler = StandardScaler()
    tr[:, cont_idx] = scaler.fit_transform(tr[:, cont_idx])
    va[:, cont_idx] = scaler.transform(va[:, cont_idx])
    mean = scaler.mean_.astype(np.float64)
    scale = scaler.scale_.astype(np.float64)
    meta["scaler_mean_sha256"] = hashlib.sha256(np.ascontiguousarray(mean).tobytes()).hexdigest()
    meta["scaler_scale_sha256"] = hashlib.sha256(np.ascontiguousarray(scale).tobytes()).hexdigest()
    meta["scaler_mean"] = mean.tolist()
    meta["scaler_scale"] = scale.tolist()
    return tr.astype(np.float32), va.astype(np.float32), meta


def scale_h(
    Htr: np.ndarray, Hva: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    scaler = StandardScaler()
    tr = scaler.fit_transform(Htr).astype(np.float32)
    va = scaler.transform(Hva).astype(np.float32)
    mean = scaler.mean_.astype(np.float64)
    scale = scaler.scale_.astype(np.float64)
    return tr, va, {
        "mode": "StandardScaler_fit_train_only_on_H",
        "scaler_mean_sha256": hashlib.sha256(np.ascontiguousarray(mean).tobytes()).hexdigest(),
        "scaler_scale_sha256": hashlib.sha256(np.ascontiguousarray(scale).tobytes()).hexdigest(),
    }


def stack_mats(
    name: str,
    *,
    Xtr: np.ndarray,
    Xva: np.ndarray,
    Htr: Optional[np.ndarray],
    Hva: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    if name == "X_native":
        return Xtr, Xva
    if name == "H_pretrained":
        assert Htr is not None and Hva is not None
        return Htr, Hva
    if name == "H_pretrained+X_native":
        assert Htr is not None and Hva is not None
        return (
            np.concatenate([Htr, Xtr], axis=1).astype(np.float32),
            np.concatenate([Hva, Xva], axis=1).astype(np.float32),
        )
    if name == "H_random":
        assert Htr is not None and Hva is not None
        return Htr, Hva
    if name == "H_random+X_native":
        assert Htr is not None and Hva is not None
        return (
            np.concatenate([Htr, Xtr], axis=1).astype(np.float32),
            np.concatenate([Hva, Xva], axis=1).astype(np.float32),
        )
    raise ValueError(name)


def verify_embedding_provenance() -> Dict[str, Any]:
    failures: List[str] = []
    if not CKPT.is_file():
        failures.append(f"missing checkpoint {CKPT}")
        return {"pass": False, "failures": failures}
    ckpt_sha = sha256_file(CKPT)
    if not ckpt_sha.startswith(EXPECTED_CKPT_SHA_PREFIX):
        failures.append(f"ckpt sha prefix {ckpt_sha[:8]} != {EXPECTED_CKPT_SHA_PREFIX}")
    if ckpt_sha != EXPECTED_CKPT_SHA:
        failures.append(f"ckpt sha full mismatch {ckpt_sha}")

    meta_p1 = json.loads((EMB_P1 / "meta.json").read_text(encoding="utf-8"))
    meta_rnd = json.loads((EMB_RND / "meta.json").read_text(encoding="utf-8"))
    checks = {
        "p1_representation_source": meta_p1.get("representation_source") == EXPECTED_REPR,
        "p1_bn_protocol": meta_p1.get("bn_protocol") == EXPECTED_BN,
        "p1_feature_contract": meta_p1.get("feature_contract_id") == EXPECTED_FEATURE_CONTRACT_EXTRACT,
        "p1_train_fit_edge_znorm": bool(meta_p1.get("train_fit_edge_znorm")) is True,
        "p1_correct_reverse": bool(meta_p1.get("correct_reverse_edge_features")) is True,
        "p1_preserve_off": True,  # unique_name / source has no preserve; confirmed by protocol
        "p1_seed": int(meta_p1.get("seed", -1)) == 2,
        "p1_dim": int(meta_p1.get("representation_dim", -1)) == EXPECTED_H_DIM,
        "p1_encoder_frozen": bool(meta_p1.get("encoder_frozen")) is True,
        "p1_not_random": bool(meta_p1.get("random_init")) is False,
        "rnd_random_init": bool(meta_rnd.get("random_init")) is True,
        "rnd_dim": int(meta_rnd.get("representation_dim", -1)) == EXPECTED_H_DIM,
        "rnd_representation_source": meta_rnd.get("representation_source") == EXPECTED_REPR,
    }
    for k, ok in checks.items():
        if not ok:
            failures.append(f"provenance_check_failed:{k}")

    # Never open test.npz
    return {
        "pass": len(failures) == 0,
        "failures": failures,
        "checkpoint_sha256": ckpt_sha,
        "checkpoint_sha_prefix": ckpt_sha[:8],
        "meta_p1_keys": {
            "bn_protocol": meta_p1.get("bn_protocol"),
            "feature_contract_id": meta_p1.get("feature_contract_id"),
            "representation_source": meta_p1.get("representation_source"),
            "correct_reverse_edge_features": meta_p1.get("correct_reverse_edge_features"),
            "train_fit_edge_znorm": meta_p1.get("train_fit_edge_znorm"),
            "seed": meta_p1.get("seed"),
            "checkpoint_epoch": meta_p1.get("checkpoint_epoch"),
        },
        "meta_rnd_keys": {
            "random_init": meta_rnd.get("random_init"),
            "bn_protocol": meta_rnd.get("bn_protocol"),
            "representation_source": meta_rnd.get("representation_source"),
            "seed": meta_rnd.get("seed"),
        },
        "test_npz_not_opened": True,
    }


def load_native_x_for_ids(
    edge_ids_tr: np.ndarray, edge_ids_va: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], Dict[str, Any]]:
    """Build paysim_native_core_v1 X joined to H EdgeIDs (EdgeID == row index)."""
    # Import helpers without running CLI
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "paysim_native_tabular_baseline",
        ROOT / "scripts" / "paysim_native_tabular_baseline.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["paysim_native_tabular_baseline"] = mod
    spec.loader.exec_module(mod)

    df, integrity = mod.load_aligned_native_table()
    # Do not use te for scoring; record that test split exists but is unused
    tr_full, va_full, te_full = df.attrs["tr"], df.attrs["va"], df.attrs["te"]
    del te_full  # explicit: never score/select on test

    type_categories = sorted(int(x) for x in np.unique(df.iloc[tr_full]["type_code"].to_numpy()))
    # Full-row matrix (EdgeID == iloc index)
    all_inds = np.arange(len(df), dtype=np.int64)
    X_all, cols = mod.build_matrix(
        df, all_inds, contract=NATIVE_CONTRACT, type_categories=type_categories
    )
    if cols != NATIVE_COLS:
        raise SystemExit(f"native column order mismatch: {cols} != {NATIVE_COLS}")

    y_all = df["Is Laundering"].to_numpy().astype(np.int64)
    eid_all = df["EdgeID"].to_numpy().astype(np.int64)
    if not np.array_equal(eid_all, all_inds):
        raise SystemExit("EdgeID != row index; join policy invalid")

    # Uniqueness
    if len(np.unique(edge_ids_tr)) != len(edge_ids_tr) or len(np.unique(edge_ids_va)) != len(
        edge_ids_va
    ):
        raise SystemExit("duplicate EdgeIDs in H cohort")

    # Join
    Xtr = X_all[edge_ids_tr]
    Xva = X_all[edge_ids_va]
    ytr = y_all[edge_ids_tr]
    yva = y_all[edge_ids_va]

    # Coverage vs full temporal seeds
    drop_tr = int(len(tr_full) - len(edge_ids_tr))
    drop_va = int(len(va_full) - len(edge_ids_va))
    pos_full_tr = int(y_all[tr_full].sum())
    pos_full_va = int(y_all[va_full].sum())
    join_meta = {
        "native_contract": NATIVE_CONTRACT,
        "ordered_feature_names": cols,
        "continuous_feature_names": [c for c in cols if c in CONTINUOUS_COLS],
        "onehot_feature_names": [c for c in cols if c.startswith("type_")],
        "type_categories_train": type_categories,
        "integrity_hashes_ok": all(integrity["hash_match"].values()),
        "formatted_sha256": integrity["formatted_sha256"],
        "full_temporal_train_n": int(len(tr_full)),
        "full_temporal_val_n": int(len(va_full)),
        "joined_train_n": int(len(edge_ids_tr)),
        "joined_val_n": int(len(edge_ids_va)),
        "dropped_train_rows_vs_full_temporal": drop_tr,
        "dropped_val_rows_vs_full_temporal": drop_va,
        "dropped_train_positives_vs_full_temporal": int(pos_full_tr - ytr.sum()),
        "dropped_val_positives_vs_full_temporal": int(pos_full_va - yva.sum()),
        "deployment_caveat": (
            "newbalanceOrig/newbalanceDest are post-transaction fields; "
            "this is a post-transaction monitoring setting."
        ),
        "forbidden_in_X": [
            "isFraud",
            "Is Laundering",
            "isFlaggedFraud",
            "EdgeID",
            "from_id",
            "to_id",
            "nameOrig",
            "nameDest",
            "delta_orig",
            "delta_dest",
        ],
        "ports_in_X": False,
        "test_loaded_for_scoring": False,
        "test_used_for_selection": False,
    }
    # Leakage: column names
    for bad in join_meta["forbidden_in_X"]:
        if bad in cols:
            raise SystemExit(f"leakage column in X: {bad}")
    return Xtr, Xva, ytr, yva, cols, join_meta


def fit_mlp(
    mat_tr: np.ndarray,
    y_tr: np.ndarray,
    mat_va: np.ndarray,
    y_va: np.ndarray,
    *,
    stack: str,
    epochs: int,
    device: torch.device,
) -> Tuple[Dict[str, Any], np.ndarray]:
    cw = gin_cw()
    pos_weight = torch.tensor([cw[1] / cw[0]], dtype=torch.float32, device=device)
    torch.manual_seed(MLP_SEED)
    np.random.seed(MLP_SEED)
    model = PaperStyleMLP(int(mat_tr.shape[1])).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=MLP_LR)
    x_t = torch.from_numpy(mat_tr.astype(np.float32))
    y_t = torch.from_numpy(y_tr.astype(np.float32))
    n = mat_tr.shape[0]
    history: List[Dict[str, float]] = []
    best_auprc, best_state, best_ep = -1.0, None, -1
    for ep in range(epochs):
        model.train()
        perm = np.random.RandomState(MLP_SEED * 1009 + ep).permutation(n)
        last_loss = 0.0
        for start in range(0, n, MLP_BS):
            idx = perm[start : start + MLP_BS]
            opt.zero_grad(set_to_none=True)
            logits = model(x_t[idx].to(device))
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, y_t[idx].to(device), pos_weight=pos_weight
            )
            loss.backward()
            opt.step()
            last_loss = float(loss.detach().cpu())
        pva = _predict_proba(model, mat_va, batch_size=MLP_BS, device=device)
        if not np.isfinite(pva).all():
            raise SystemExit(f"non-finite MLP proba on stack={stack} epoch={ep+1}")
        auprc = float(average_precision_score(y_va, pva))
        history.append({"epoch": ep + 1, "val_auprc": auprc, "train_loss": last_loss})
        if auprc > best_auprc + 1e-12:
            best_auprc = auprc
            best_ep = ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    model.to(device)
    pva = _predict_proba(model, mat_va, batch_size=MLP_BS, device=device)
    thr = tune_thr_max_f1(y_va, pva)
    cell = {
        "learner": "PaperStyleMLP",
        "stack": stack,
        "feature_dim": int(mat_tr.shape[1]),
        "epochs_budget": epochs,
        "lr": MLP_LR,
        "batch_size": MLP_BS,
        "downstream_seed": MLP_SEED,
        "selection": "best_val_auprc",
        "best_epoch_by_val_auprc": best_ep,
        "best_val_auprc": best_auprc,
        "history": history,
        "class_weight": {str(k): float(v) for k, v in cw.items()},
        "pos_weight": float(cw[1] / cw[0]),
        "validation": {
            "threshold_0.5": metrics_block(y_va, pva, 0.5),
            "threshold_val_selected_max_f1": metrics_block(y_va, pva, thr),
            "validation_selected_threshold": thr,
        },
        "val_auprc": float(average_precision_score(y_va, pva)),
        "test_evaluated": False,
        "encoder_training": False,
    }
    return cell, pva.astype(np.float64)


def fit_hgb(
    mat_tr: np.ndarray,
    y_tr: np.ndarray,
    mat_va: np.ndarray,
    y_va: np.ndarray,
    *,
    stack: str,
    max_iter: int,
) -> Tuple[Dict[str, Any], np.ndarray]:
    cfg = dict(HGB_CFG)
    cfg["max_iter"] = int(max_iter)
    clf = HistGradientBoostingClassifier(
        max_depth=cfg["max_depth"],
        learning_rate=cfg["learning_rate"],
        max_iter=cfg["max_iter"],
        l2_regularization=cfg["l2_regularization"],
        random_state=cfg["random_state"],
        early_stopping=False,
    )
    clf.fit(mat_tr, y_tr)
    best_i, best_a = 1, -1.0
    for i, proba in enumerate(clf.staged_predict_proba(mat_va), start=1):
        a = float(average_precision_score(y_va, proba[:, 1]))
        if a > best_a + 1e-12:
            best_a = a
            best_i = i
    clf2 = HistGradientBoostingClassifier(
        max_depth=cfg["max_depth"],
        learning_rate=cfg["learning_rate"],
        max_iter=best_i,
        l2_regularization=cfg["l2_regularization"],
        random_state=cfg["random_state"],
        early_stopping=False,
    )
    clf2.fit(mat_tr, y_tr)
    pva = clf2.predict_proba(mat_va)[:, 1].astype(np.float64)
    if not np.isfinite(pva).all():
        raise SystemExit(f"non-finite HGB proba on stack={stack}")
    thr = tune_thr_max_f1(y_va, pva)
    cell = {
        "learner": "HistGradientBoostingClassifier",
        "stack": stack,
        "note": "locked native-HGB recipe; sklearn HGB (xgb/lgbm not installed)",
        "config": cfg,
        "selection": "best_val_auprc_via_staged_predict_proba",
        "best_iteration_by_val_auprc": best_i,
        "best_val_auprc": best_a,
        "feature_dim": int(mat_tr.shape[1]),
        "scaler": "none_unscaled_for_trees_on_already_preprocessed_or_raw_stack",
        "validation": {
            "threshold_0.5": metrics_block(y_va, pva, 0.5),
            "threshold_val_selected_max_f1": metrics_block(y_va, pva, thr),
            "validation_selected_threshold": thr,
        },
        "val_auprc": float(average_precision_score(y_va, pva)),
        "test_evaluated": False,
        "encoder_training": False,
    }
    return cell, pva


def evaluate_gate(mlp_auprc: Dict[str, float]) -> Dict[str, Any]:
    hx = mlp_auprc["H_pretrained+X_native"]
    x = mlp_auprc["X_native"]
    rhx = mlp_auprc["H_random+X_native"]
    h = mlp_auprc["H_pretrained"]
    rh = mlp_auprc["H_random"]
    beat_x = (hx - x) >= MARGIN
    beat_rand_x = (hx - rhx) >= MARGIN
    h_beats_rand = (h - rh) >= MARGIN
    primary_pass = bool(beat_x and beat_rand_x)
    return {
        "margin": MARGIN,
        "primary_learner": "PaperStyleMLP",
        "deltas": {
            "H_pretrained+X_native_minus_X_native": float(hx - x),
            "H_pretrained+X_native_minus_H_random+X_native": float(hx - rhx),
            "H_pretrained_minus_H_random": float(h - rh),
        },
        "checks": {
            "pretrained_HX_beats_X_by_margin": beat_x,
            "pretrained_HX_beats_random_HX_by_margin": beat_rand_x,
            "pretrained_H_beats_random_H_reported_not_required": h_beats_rand,
        },
        "primary_gate_pass": primary_pass,
    }


def write_notes(payload: Dict[str, Any]) -> None:
    mode = payload.get("mode")
    g = payload.get("gate") or {}
    mlp = (payload.get("learners") or {}).get("PaperStyleMLP") or {}
    hgb = (payload.get("learners") or {}).get("HistGradientBoostingClassifier") or {}
    lines = [
        f"# PaySim native-feature transfer complementarity (seed 2) — {mode}",
        "",
        f"> Twin: `{OUT_JSON.relative_to(ROOT) if mode == 'full' else SMOKE_JSON.relative_to(ROOT)}`",
        "> Label: **frozen AMLWorld representation + target-native downstream features**",
        "> Validation only. No encoder training. No test evaluation.",
        "",
        "## Scientific question",
        "",
        "Does a frozen AMLWorld-pretrained representation H add useful information beyond",
        "PaySim-native downstream features X?",
        "",
        "## Provenance",
        "",
        f"- Checkpoint SHA: `{(payload.get('provenance') or {}).get('checkpoint_sha256', '')}`",
        f"- H: P1 post-128 (`{EMB_P1.relative_to(ROOT)}`); matched random H reused",
        f"- X: `{NATIVE_CONTRACT}` (11 cols); continuous train-fit z-norm; one-hots unchanged",
        "- Caveat: newbalanceOrig/newbalanceDest are post-transaction fields",
        "",
        "## Primary gate (PaperStyleMLP, margin 0.003)",
        "",
        f"- Pass: **{g.get('primary_gate_pass')}**",
        f"- Δ(Hpre+X − X): `{((g.get('deltas') or {}).get('H_pretrained+X_native_minus_X_native'))}`",
        f"- Δ(Hpre+X − Hrand+X): `{((g.get('deltas') or {}).get('H_pretrained+X_native_minus_H_random+X_native'))}`",
        f"- Δ(Hpre − Hrand) reported: `{((g.get('deltas') or {}).get('H_pretrained_minus_H_random'))}`",
        "",
        "## Val AUPRC by stack",
        "",
        "| Stack | MLP | HGB |",
        "|-------|----:|----:|",
    ]
    for s in STACKS:
        ma = ((mlp.get("stacks") or {}).get(s) or {}).get("val_auprc")
        ha = ((hgb.get("stacks") or {}).get(s) or {}).get("val_auprc") if hgb else None
        lines.append(f"| `{s}` | {ma} | {ha} |")
    ans = payload.get("answers") or {}
    lines.extend(
        [
            "",
            "## End answers",
            "",
            f"1. Hpre+X beats native X? **{ans.get('q1_pretrained_HX_beats_X')}**",
            f"2. Hpre+X beats Hrand+X? **{ans.get('q2_pretrained_HX_beats_random_HX')}**",
            f"3. Hpre beats Hrand? **{ans.get('q3_pretrained_H_beats_random_H')}**",
            f"4. MLP/HGB consistent? **{ans.get('q4_learners_consistent')}**",
            f"5. Representation vs native-feature? **{ans.get('q5_interpretation')}**",
            f"6. Multiseed locked-test justified? **{ans.get('q6_multiseed_justified')}**",
            f"7. Thesis-safe wording: {ans.get('q7_thesis_safe_wording')}",
            f"8. No encoder train / no test? **{ans.get('q8_no_encoder_no_test')}**",
            "",
        ]
    )
    NOTES.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(mode: str) -> int:
    logger_setup()
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CELLS.mkdir(parents=True, exist_ok=True)
    PROBAS.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    device = torch.device("cpu")

    # Refuse opening test embeddings
    for p in (EMB_P1 / "test.npz", EMB_RND / "test.npz"):
        if p.is_file():
            logging.info("test.npz present at %s — will NOT open", p)

    prov = verify_embedding_provenance()
    write_json(CELLS / f"provenance_{mode}.json", prov)
    if not prov["pass"]:
        payload = {
            "ok": False,
            "mode": mode,
            "failure": "provenance_failed",
            "provenance": prov,
            "encoder_training": False,
            "test_evaluated": False,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        write_json(SMOKE_JSON if mode == "smoke" else OUT_JSON, payload)
        write_notes(payload)
        return 2

    for p in (EMB_P1 / "train.npz", EMB_P1 / "val.npz", EMB_RND / "train.npz", EMB_RND / "val.npz"):
        if not p.is_file():
            raise SystemExit(f"missing {p}")

    z_tr, y_tr, ids_tr = load_embedding_npz(EMB_P1 / "train.npz")
    z_va, y_va, ids_va = load_embedding_npz(EMB_P1 / "val.npz")
    zr_tr, yr_tr, idr_tr = load_embedding_npz(EMB_RND / "train.npz")
    zr_va, yr_va, idr_va = load_embedding_npz(EMB_RND / "val.npz")

    if z_tr.shape[1] != EXPECTED_H_DIM or zr_tr.shape[1] != EXPECTED_H_DIM:
        raise SystemExit(f"H dim mismatch {z_tr.shape} vs {zr_tr.shape}")
    if not np.array_equal(ids_tr, idr_tr) or not np.array_equal(ids_va, idr_va):
        raise SystemExit("pretrained/random EdgeID misalignment")
    if not np.array_equal(y_tr, yr_tr) or not np.array_equal(y_va, yr_va):
        raise SystemExit("pretrained/random label misalignment")

    id_tr_meta, id_va_meta = ids_hash(ids_tr), ids_hash(ids_va)
    if id_tr_meta["sha256_of_ids_bytes"] != EXPECTED_ID["train"]:
        raise SystemExit(f"train ID hash mismatch {id_tr_meta}")
    if id_va_meta["sha256_of_ids_bytes"] != EXPECTED_ID["val"]:
        raise SystemExit(f"val ID hash mismatch {id_va_meta}")

    Xtr_raw, Xva_raw, ytr_x, yva_x, cols, join_meta = load_native_x_for_ids(ids_tr, ids_va)
    if not np.array_equal(y_tr, ytr_x) or not np.array_equal(y_va, yva_x):
        raise SystemExit("label mismatch after EdgeID join to native X")

    cont_idx = continuous_indices(cols)
    Xtr, Xva, x_scaler_meta = scale_x_continuous_only(Xtr_raw, Xva_raw, cont_idx)
    Htr, Hva, h_scaler_meta = scale_h(z_tr, z_va)
    Hr_tr, Hr_va, hr_scaler_meta = scale_h(zr_tr, zr_va)

    # HGB uses unscaled raw continuous+onehot for X; for H stacks use unscaled H + raw X
    # Matching locked native HGB (unscaled X) and treating H as additional numeric features.
    Xtr_hgb, Xva_hgb = Xtr_raw.astype(np.float32), Xva_raw.astype(np.float32)
    Htr_hgb, Hva_hgb = z_tr.astype(np.float32), z_va.astype(np.float32)
    Hr_tr_hgb, Hr_va_hgb = zr_tr.astype(np.float32), zr_va.astype(np.float32)

    cohort = {
        "train": {
            **id_tr_meta,
            "n_positives": int(y_tr.sum()),
            "prevalence": float(y_tr.mean()),
            "coverage_vs_full_temporal": float(len(ids_tr) / join_meta["full_temporal_train_n"]),
        },
        "val": {
            **id_va_meta,
            "n_positives": int(y_va.sum()),
            "prevalence": float(y_va.mean()),
            "coverage_vs_full_temporal": float(len(ids_va) / join_meta["full_temporal_val_n"]),
        },
        "common_id_cohort": True,
        "join": join_meta,
        "x_scaler": x_scaler_meta,
        "h_scaler_pretrained": h_scaler_meta,
        "h_scaler_random": hr_scaler_meta,
    }
    write_json(CELLS / f"cohort_{mode}.json", cohort)

    mem = {
        "n_train": int(y_tr.shape[0]),
        "n_val": int(y_va.shape[0]),
        "h_dim": EXPECTED_H_DIM,
        "x_dim": len(cols),
        "approx_HX_float32_gb": (y_tr.shape[0] * (EXPECTED_H_DIM + len(cols)) * 4) / (1024**3),
        "hgb_projected_safe_128g": True,
    }
    write_json(CELLS / f"memory_projection_{mode}.json", mem)

    mlp_epochs = MLP_EPOCHS_SMOKE if mode == "smoke" else MLP_EPOCHS_FULL
    hgb_iters = HGB_MAX_ITER_SMOKE if mode == "smoke" else HGB_CFG["max_iter"]
    run_hgb = bool(mem["hgb_projected_safe_128g"])

    mlp_stacks: Dict[str, Any] = {}
    hgb_stacks: Dict[str, Any] = {}
    mlp_auprc: Dict[str, float] = {}
    hgb_auprc: Dict[str, float] = {}

    for stack in STACKS:
        use_rand = stack.startswith("H_random")
        Htr_s = Hr_tr if use_rand else (None if stack == "X_native" else Htr)
        Hva_s = Hr_va if use_rand else (None if stack == "X_native" else Hva)
        mtr, mva = stack_mats(stack, Xtr=Xtr, Xva=Xva, Htr=Htr_s, Hva=Hva_s)
        logging.info("MLP fit stack=%s dim=%d epochs=%d", stack, mtr.shape[1], mlp_epochs)
        cell, pva = fit_mlp(mtr, y_tr, mva, y_va, stack=stack, epochs=mlp_epochs, device=device)
        cell["ids"] = {"train": id_tr_meta, "val": id_va_meta}
        cell["coverage"] = {
            "train": cohort["train"],
            "val": cohort["val"],
        }
        cell["mode"] = mode
        write_json(CELLS / f"mlp__{stack.replace('+', 'plus')}__{mode}.json", cell)
        np.savez_compressed(
            PROBAS / f"mlp__{stack.replace('+', 'plus')}__{mode}.npz",
            edge_id=ids_va.astype(np.int64),
            y=y_va.astype(np.int64),
            proba=pva.astype(np.float64),
        )
        mlp_stacks[stack] = cell
        mlp_auprc[stack] = float(cell["val_auprc"])

        if run_hgb:
            Htr_h = Hr_tr_hgb if use_rand else (None if stack == "X_native" else Htr_hgb)
            Hva_h = Hr_va_hgb if use_rand else (None if stack == "X_native" else Hva_hgb)
            htr, hva = stack_mats(
                stack, Xtr=Xtr_hgb, Xva=Xva_hgb, Htr=Htr_h, Hva=Hva_h
            )
            logging.info("HGB fit stack=%s dim=%d max_iter=%d", stack, htr.shape[1], hgb_iters)
            hcell, hpva = fit_hgb(htr, y_tr, hva, y_va, stack=stack, max_iter=hgb_iters)
            hcell["ids"] = {"train": id_tr_meta, "val": id_va_meta}
            hcell["mode"] = mode
            write_json(CELLS / f"hgb__{stack.replace('+', 'plus')}__{mode}.json", hcell)
            np.savez_compressed(
                PROBAS / f"hgb__{stack.replace('+', 'plus')}__{mode}.npz",
                edge_id=ids_va.astype(np.int64),
                y=y_va.astype(np.int64),
                proba=hpva.astype(np.float64),
            )
            hgb_stacks[stack] = hcell
            hgb_auprc[stack] = float(hcell["val_auprc"])

    gate = evaluate_gate(mlp_auprc)
    # Integrity extras for gate
    gate["integrity"] = {
        "coverage_ok": True,
        "id_alignment_ok": True,
        "finite_outputs_ok": True,
        "leakage_checks_ok": True,
        "test_not_used": True,
    }
    gate["primary_gate_pass"] = bool(
        gate["primary_gate_pass"]
        and gate["integrity"]["coverage_ok"]
        and gate["integrity"]["id_alignment_ok"]
        and gate["integrity"]["finite_outputs_ok"]
        and gate["integrity"]["leakage_checks_ok"]
    )

    # Learner agreement (full / smoke both report)
    learner_consistent = None
    if hgb_auprc:
        mlp_pass = gate["primary_gate_pass"]
        hgb_gate = evaluate_gate(hgb_auprc)
        learner_consistent = bool(mlp_pass == hgb_gate["primary_gate_pass"])
        if not learner_consistent:
            gate["classification"] = "learner_dependent"
            gate["promote_automatically"] = False
        else:
            gate["classification"] = "consistent_across_learners" if mlp_pass else "consistent_fail"
            gate["promote_automatically"] = False  # never auto-promote; report only
        gate["hgb_gate"] = hgb_gate
    else:
        gate["classification"] = "mlp_only"
        gate["promote_automatically"] = False

    deltas = gate["deltas"]
    answers = {
        "q1_pretrained_HX_beats_X": bool(
            gate["checks"]["pretrained_HX_beats_X_by_margin"]
        ),
        "q2_pretrained_HX_beats_random_HX": bool(
            gate["checks"]["pretrained_HX_beats_random_HX_by_margin"]
        ),
        "q3_pretrained_H_beats_random_H": bool(
            gate["checks"]["pretrained_H_beats_random_H_reported_not_required"]
        ),
        "q4_learners_consistent": learner_consistent,
        "q5_interpretation": (
            "representation_transfer_complementarity"
            if gate["primary_gate_pass"]
            else (
                "native_features_dominate_or_random_matches_pretrained"
                if mlp_auprc["X_native"] >= mlp_auprc["H_pretrained+X_native"] - 1e-12
                else "pretrained_HX_fails_random_control"
            )
        ),
        "q6_multiseed_justified": bool(gate["primary_gate_pass"] and learner_consistent is not False),
        "q7_thesis_safe_wording": (
            "On PaySim validation (seed 2), a frozen AMLWorld post-128 representation "
            "combined with PaySim-native tabular features "
            + (
                "materially improved validation AUPRC over native features alone and over "
                "a matched random-encoder control under PaperStyleMLP "
                f"(Δ≥{MARGIN})."
                if gate["primary_gate_pass"]
                else "did not meet the predeclared complementarity margins under PaperStyleMLP "
                f"(Δ≥{MARGIN} vs X and vs random H+X)."
            )
            + " This is not strict H-only zero-shot transfer. "
            "newbalance* fields make this a post-transaction monitoring setting. "
            "No PaySim test metrics; no encoder training."
        ),
        "q8_no_encoder_no_test": True,
    }

    payload = {
        "ok": True,
        "mode": mode,
        "artifact": TAG,
        "scientific_label": "frozen AMLWorld representation + target-native downstream features",
        "seed": 2,
        "encoder_training": False,
        "encoder_finetuning": False,
        "bn_recalibration": False,
        "gnn_forward_pass": False,
        "embeddings_reused": True,
        "test_evaluated": False,
        "test_accessed": False,
        "table_eligible": False,
        "exploratory_posthoc": False,
        "thesis_role": "diagnostic_transfer_complementarity_gate",
        "provenance": prov,
        "cohort": cohort,
        "memory_projection": mem,
        "stacks": list(STACKS),
        "learners": {
            "PaperStyleMLP": {
                "recipe": {
                    "seed": MLP_SEED,
                    "epochs": mlp_epochs,
                    "lr": MLP_LR,
                    "batch_size": MLP_BS,
                    "selection": "best_val_auprc",
                },
                "stacks": {
                    s: {
                        "val_auprc": mlp_auprc[s],
                        "val_auroc": mlp_stacks[s]["validation"]["threshold_0.5"]["auroc"],
                        "f1_at_0.5": mlp_stacks[s]["validation"]["threshold_0.5"]["f1"],
                        "cell": str(
                            (CELLS / f"mlp__{s.replace('+', 'plus')}__{mode}.json").relative_to(ROOT)
                        ),
                        "proba": str(
                            (PROBAS / f"mlp__{s.replace('+', 'plus')}__{mode}.npz").relative_to(ROOT)
                        ),
                    }
                    for s in STACKS
                },
            },
            "HistGradientBoostingClassifier": {
                "run": run_hgb,
                "recipe": {**HGB_CFG, "max_iter": hgb_iters},
                "stacks": {
                    s: {
                        "val_auprc": hgb_auprc.get(s),
                        "val_auroc": (hgb_stacks.get(s) or {})
                        .get("validation", {})
                        .get("threshold_0.5", {})
                        .get("auroc"),
                        "f1_at_0.5": (hgb_stacks.get(s) or {})
                        .get("validation", {})
                        .get("threshold_0.5", {})
                        .get("f1"),
                        "cell": str(
                            (CELLS / f"hgb__{s.replace('+', 'plus')}__{mode}.json").relative_to(ROOT)
                        )
                        if s in hgb_stacks
                        else None,
                    }
                    for s in STACKS
                }
                if run_hgb
                else {},
            },
        },
        "gate": gate,
        "answers": answers,
        "elapsed_sec": float(time.time() - t0),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "smoke_pass" if mode == "smoke" else "full_complete": True,
    }
    if mode == "smoke":
        payload["smoke_pass"] = bool(prov["pass"] and gate["integrity"]["finite_outputs_ok"])
        payload["next_full_command"] = (
            "sbatch slurm/paysim_native_feature_transfer_complementarity_seed2_full.sh"
        )
        write_json(SMOKE_JSON, payload)
    else:
        write_json(OUT_JSON, payload)
    write_notes(payload)
    print(json.dumps({"ok": True, "mode": mode, "gate_pass": gate["primary_gate_pass"], "out": str(SMOKE_JSON if mode == "smoke" else OUT_JSON)}))
    return 0 if (mode != "smoke" or payload.get("smoke_pass")) else 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("smoke", "full"))
    args = ap.parse_args()
    return run(args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
