#!/usr/bin/env python3
"""Bounded full-native PaySim tabular baseline (temporal split ceiling diagnostic).

Contracts:
  - paysim_native_core_v1
  - paysim_native_core_v1_with_deltas  (reported separately; does not select primary)
  - paysim_native_full_v1  (core + isFlaggedFraud; sensitivity only)

Learners: LogisticRegression, PaperStyleMLP, HistGradientBoostingClassifier
(XGB/LGBM not installed).

Modes: smoke | full
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from gcpal_txn_node.eval_mlp import PaperStyleMLP, _predict_proba  # noqa: E402
from train_util import extract_param  # noqa: E402
from util import create_parser, logger_setup, set_seed  # noqa: E402

TAG = "paysim_native_tabular_baseline"
OUT = REPO / "results/diagnostics" / TAG
CELLS = OUT / "cells"
NOTE = REPO / "notes" / f"{TAG}.md"
FINAL_JSON = REPO / "results/diagnostics" / f"{TAG}.json"
SMOKE_JSON = OUT / "smoke.json"
SUBMISSION = OUT / "submission.json"

FORMATTED = REPO / "aml-data/PaySim/formatted_transactions.csv"
RAW = REPO / "aml-data/PaySim/PS_20174392719_1491204439457_log.csv"
EXPECTED_FMT_SHA = "03c2fa07b95d145e754b74a5e646c2d71cd4fed051210d6292a0bbab90112c93"
EXPECTED_SPLITS = {
    "train": {
        "n": 3792821,
        "n_positives": 3175,
        "step_min": 1,
        "step_max": 280,
        "index_sha256": "0d2f7e516aeae723cda174f4ab086380d006a526d4ede47cd2b8f5100af92279",
    },
    "val": {
        "n": 1276276,
        "n_positives": 780,
        "step_min": 281,
        "step_max": 354,
        "index_sha256": "696756046b7e6dd4df5f6f600bbb373c7a24b888d60df7ea10ce3bf468f76469",
    },
    "test": {
        "n": 1293523,
        "n_positives": 4258,
        "step_min": 355,
        "step_max": 743,
        "index_sha256": "dcc1018601844cfb174ca14a24d2208512c63cca2948cdbc804ed1c44aebac87",
    },
}

LOGISTIC_SEED = 2
MLP_SEED = 3
MLP_EPOCHS = 15
MLP_LR = 1e-3
MLP_BS = 8192
HGB_CFG = {
    "max_depth": 6,
    "learning_rate": 0.1,
    "max_iter": 100,
    "l2_regularization": 0.0,
    "random_state": LOGISTIC_SEED,
}
MATERIAL_AUPRC = 0.01
XONLY_PATH = (
    REPO
    / "results/diagnostics/final_corrected_no_preserve_multiseed/cells/control_X_only_paysim_legacy_duplicate_v1.json"
)
GIN_AGG = REPO / "results/diagnostics/paysim_supervised_multigin_eu.json"

TYPE_ORDER = ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]  # expected factorize order


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


def sha256_int64(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr.astype(np.int64)).tobytes()).hexdigest()


def gin_cw() -> Dict[int, float]:
    args = create_parser().parse_args(["--data", "PaySim", "--model", "gin", "--testing"])
    return {0: float(extract_param("w_ce1", args)), 1: float(extract_param("w_ce2", args))}


def metrics_block(y: np.ndarray, proba: np.ndarray, thr: float) -> Dict[str, float]:
    y = y.astype(np.int64)
    proba = proba.astype(np.float64)
    pred = (proba >= float(thr)).astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    n = int(y.shape[0])
    order = np.argsort(-proba)
    def p_at(k: int) -> float:
        k = min(k, n)
        if k <= 0:
            return float("nan")
        return float(y[order[:k]].mean())
    return {
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
        "n_positives": float(int(y.sum())),
        "positive_rate": float(y.mean()) if n else 0.0,
        "precision_at_100": p_at(100),
        "precision_at_500": p_at(500),
        "precision_at_1000": p_at(1000),
        "auprc_over_prevalence": float(average_precision_score(y, proba) / max(y.mean(), 1e-12)),
    }


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


def load_aligned_native_table() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load raw PaySim, apply format_paysim sort/EdgeID policy, keep native balances."""
    t0 = time.time()
    fmt_sha = sha256_file(FORMATTED)
    if fmt_sha != EXPECTED_FMT_SHA:
        raise SystemExit(f"formatted SHA mismatch: {fmt_sha} != {EXPECTED_FMT_SHA}")
    raw = pd.read_csv(RAW)
    required = {
        "step",
        "type",
        "amount",
        "nameOrig",
        "nameDest",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "isFraud",
        "isFlaggedFraud",
    }
    missing = required - set(raw.columns)
    if missing:
        raise SystemExit(f"raw missing {sorted(missing)}")

    n = len(raw)
    type_str = raw["type"].astype(str).str.strip()
    type_codes, uniques = pd.factorize(type_str)
    type_map = {int(i): str(u) for i, u in enumerate(uniques)}
    # Align with formatted: same factorization + Timestamp sort
    from_names = raw["nameOrig"].astype(str).str.strip()
    to_names = raw["nameDest"].astype(str).str.strip()
    account_codes, _ = pd.factorize(pd.concat([from_names, to_names], ignore_index=True))
    from_id = account_codes[:n].astype(np.int64)
    to_id = account_codes[n:].astype(np.int64)
    ts = raw["step"].to_numpy(dtype=np.int64) * 3600

    df = pd.DataFrame(
        {
            "EdgeID": np.arange(n, dtype=np.int64),
            "from_id": from_id,
            "to_id": to_id,
            "Timestamp": ts,
            "step": raw["step"].to_numpy(dtype=np.int64),
            "type_code": type_codes.astype(np.int64),
            "type_str": type_str.to_numpy(),
            "amount": raw["amount"].to_numpy(dtype=np.float64),
            "oldbalanceOrg": raw["oldbalanceOrg"].to_numpy(dtype=np.float64),
            "newbalanceOrig": raw["newbalanceOrig"].to_numpy(dtype=np.float64),
            "oldbalanceDest": raw["oldbalanceDest"].to_numpy(dtype=np.float64),
            "newbalanceDest": raw["newbalanceDest"].to_numpy(dtype=np.float64),
            "isFlaggedFraud": raw["isFlaggedFraud"].to_numpy(dtype=np.int64),
            "Is Laundering": raw["isFraud"].to_numpy(dtype=np.int64),
        }
    )
    df = df.sort_values("Timestamp", kind="mergesort").reset_index(drop=True)
    df["EdgeID"] = np.arange(len(df), dtype=np.int64)

    # Verify vs formatted CSV labels/steps/EdgeIDs
    fmt = pd.read_csv(FORMATTED, usecols=["EdgeID", "Timestamp", "Is Laundering"])
    if len(fmt) != len(df):
        raise SystemExit(f"row count mismatch formatted={len(fmt)} native={len(df)}")
    if not np.array_equal(fmt["EdgeID"].to_numpy(), df["EdgeID"].to_numpy()):
        raise SystemExit("EdgeID alignment failed vs formatted")
    if not np.array_equal(fmt["Timestamp"].to_numpy(), df["Timestamp"].to_numpy()):
        raise SystemExit("Timestamp alignment failed vs formatted")
    if not np.array_equal(fmt["Is Laundering"].to_numpy(), df["Is Laundering"].to_numpy()):
        raise SystemExit("label alignment failed vs formatted")

    steps = df["step"].to_numpy()
    # Exact loader split (same as data_loading / failure audit)
    import torch
    from dataset_specs import get_dataset_spec
    from dataset_splits import temporal_edge_split

    spec = get_dataset_spec("PaySim")
    tr_t, va_t, te_t, _ = temporal_edge_split(
        torch.tensor(df["Timestamp"].to_numpy(), dtype=torch.long),
        torch.tensor(df["Is Laundering"].to_numpy(), dtype=torch.long),
        spec,
    )
    tr, va, te = tr_t.numpy().astype(np.int64), va_t.numpy().astype(np.int64), te_t.numpy().astype(np.int64)
    # Verify step ranges match failure-audit documentation + index hashes
    integrity = {
        "formatted_sha256": fmt_sha,
        "raw_path": str(RAW),
        "type_code_map": type_map,
        "elapsed_load_sec": float(time.time() - t0),
        "splits": {},
        "hash_match": {},
        "count_match": {},
    }
    for name, inds in (("train", tr), ("val", va), ("test", te)):
        y = df["Is Laundering"].to_numpy()[inds]
        exp = EXPECTED_SPLITS[name]
        h = sha256_int64(inds)
        integrity["splits"][name] = {
            "n": int(inds.shape[0]),
            "n_positives": int(y.sum()),
            "positive_rate": float(y.mean()),
            "step_min": int(steps[inds].min()),
            "step_max": int(steps[inds].max()),
            "index_sha256": h,
            "n_unique_EdgeID": int(len(np.unique(df["EdgeID"].to_numpy()[inds]))),
        }
        integrity["hash_match"][name] = h == exp["index_sha256"]
        integrity["count_match"][name] = (
            int(inds.shape[0]) == exp["n"]
            and int(y.sum()) == exp["n_positives"]
            and int(steps[inds].min()) == exp["step_min"]
            and int(steps[inds].max()) == exp["step_max"]
        )
    if not all(integrity["hash_match"].values()) or not all(integrity["count_match"].values()):
        raise SystemExit(f"split integrity failed: {json.dumps(integrity, indent=2)}")

    df.attrs["tr"] = tr
    df.attrs["va"] = va
    df.attrs["te"] = te
    df.attrs["integrity"] = integrity
    return df, integrity


def build_matrix(
    df: pd.DataFrame,
    inds: np.ndarray,
    *,
    contract: str,
    type_categories: Sequence[int],
) -> Tuple[np.ndarray, List[str]]:
    """Build float32 feature matrix. Never includes isFraud."""
    sub = df.iloc[inds]
    cols: List[str] = []
    parts: List[np.ndarray] = []

    # step (raw)
    parts.append(sub["step"].to_numpy(dtype=np.float64).reshape(-1, 1))
    cols.append("step")

    # type one-hot with fixed category set from train
    tc = sub["type_code"].to_numpy()
    for c in type_categories:
        parts.append((tc == c).astype(np.float64).reshape(-1, 1))
        cols.append(f"type_{int(c)}")

    parts.append(np.log1p(sub["amount"].to_numpy(dtype=np.float64)).reshape(-1, 1))
    cols.append("log1p_amount")

    for name in ("oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"):
        parts.append(sub[name].to_numpy(dtype=np.float64).reshape(-1, 1))
        cols.append(name)

    if contract in ("paysim_native_core_v1_with_deltas", "paysim_native_full_v1_with_deltas"):
        d_o = sub["newbalanceOrig"].to_numpy(dtype=np.float64) - sub["oldbalanceOrg"].to_numpy(dtype=np.float64)
        d_d = sub["newbalanceDest"].to_numpy(dtype=np.float64) - sub["oldbalanceDest"].to_numpy(dtype=np.float64)
        parts.append(d_o.reshape(-1, 1))
        cols.append("delta_orig")
        parts.append(d_d.reshape(-1, 1))
        cols.append("delta_dest")

    if contract.startswith("paysim_native_full_v1"):
        parts.append(sub["isFlaggedFraud"].to_numpy(dtype=np.float64).reshape(-1, 1))
        cols.append("isFlaggedFraud")

    X = np.concatenate(parts, axis=1).astype(np.float32)
    # sanity: no label column
    assert "Is Laundering" not in cols and "isFraud" not in cols
    return X, cols


def fit_logistic(Xtr, ytr, Xva, yva, Xte, yte, *, evaluate_test: bool) -> Dict[str, Any]:
    scaler = StandardScaler()
    tr = scaler.fit_transform(Xtr).astype(np.float32)
    va = scaler.transform(Xva).astype(np.float32)
    cw = gin_cw()
    set_seed(LOGISTIC_SEED)
    clf = LogisticRegression(
        class_weight=cw, max_iter=2000, random_state=LOGISTIC_SEED, solver="lbfgs", n_jobs=1, C=1.0
    )
    clf.fit(tr, ytr)
    pva = clf.predict_proba(va)[:, 1].astype(np.float64)
    thr = tune_thr_max_f1(yva, pva)
    out: Dict[str, Any] = {
        "learner": "LogisticRegression",
        "feature_dim": int(tr.shape[1]),
        "class_weight": {str(k): float(v) for k, v in cw.items()},
        "scaler": "StandardScaler_fit_train_only",
        "selection": "single_fit_no_epoch",
        "validation": {
            "threshold_0.5": metrics_block(yva, pva, 0.5),
            "threshold_val_selected_max_f1": metrics_block(yva, pva, thr),
            "validation_selected_threshold": thr,
        },
        "val_auprc": float(average_precision_score(yva, pva)),
        "test_evaluated": False,
    }
    if evaluate_test:
        te = scaler.transform(Xte).astype(np.float32)
        pte = clf.predict_proba(te)[:, 1].astype(np.float64)
        out["test"] = {
            "threshold_0.5": metrics_block(yte, pte, 0.5),
            "threshold_val_selected_max_f1": metrics_block(yte, pte, thr),
        }
        out["test_evaluated"] = True
        out["test_auprc"] = float(average_precision_score(yte, pte))
    return out


def fit_mlp(Xtr, ytr, Xva, yva, Xte, yte, *, evaluate_test: bool, device: torch.device) -> Dict[str, Any]:
    scaler = StandardScaler()
    tr = scaler.fit_transform(Xtr).astype(np.float32)
    va = scaler.transform(Xva).astype(np.float32)
    cw = gin_cw()
    pos_weight = torch.tensor([cw[1] / cw[0]], dtype=torch.float32, device=device)
    torch.manual_seed(MLP_SEED)
    np.random.seed(MLP_SEED)
    model = PaperStyleMLP(int(tr.shape[1])).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=MLP_LR)
    x_t = torch.from_numpy(tr)
    y_t = torch.from_numpy(ytr.astype(np.float32))
    n = tr.shape[0]
    history: List[Dict[str, float]] = []
    best_auprc, best_state, best_ep = -1.0, None, -1
    for ep in range(MLP_EPOCHS):
        model.train()
        perm = np.random.RandomState(MLP_SEED * 1009 + ep).permutation(n)
        for start in range(0, n, MLP_BS):
            idx = perm[start : start + MLP_BS]
            xb = x_t[idx].to(device)
            yb = y_t[idx].to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, yb, pos_weight=pos_weight
            )
            loss.backward()
            opt.step()
        pva = _predict_proba(model, va, batch_size=MLP_BS, device=device)
        auprc = float(average_precision_score(yva, pva))
        history.append({"epoch": ep + 1, "val_auprc": auprc, "train_loss": float(loss.detach().cpu())})
        if auprc > best_auprc + 1e-12:
            best_auprc = auprc
            best_ep = ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    model.to(device)
    pva = _predict_proba(model, va, batch_size=MLP_BS, device=device)
    thr = tune_thr_max_f1(yva, pva)
    out: Dict[str, Any] = {
        "learner": "PaperStyleMLP",
        "feature_dim": int(tr.shape[1]),
        "epochs": MLP_EPOCHS,
        "lr": MLP_LR,
        "batch_size": MLP_BS,
        "seed": MLP_SEED,
        "selection": "best_val_auprc",
        "best_epoch_by_val_auprc": best_ep,
        "best_val_auprc": best_auprc,
        "history": history,
        "scaler": "StandardScaler_fit_train_only",
        "validation": {
            "threshold_0.5": metrics_block(yva, pva, 0.5),
            "threshold_val_selected_max_f1": metrics_block(yva, pva, thr),
            "validation_selected_threshold": thr,
        },
        "val_auprc": best_auprc,
        "test_evaluated": False,
    }
    if evaluate_test:
        te = scaler.transform(Xte).astype(np.float32)
        pte = _predict_proba(model, te, batch_size=MLP_BS, device=device)
        out["test"] = {
            "threshold_0.5": metrics_block(yte, pte, 0.5),
            "threshold_val_selected_max_f1": metrics_block(yte, pte, thr),
        }
        out["test_evaluated"] = True
        out["test_auprc"] = float(average_precision_score(yte, pte))
    return out


def fit_hgb(Xtr, ytr, Xva, yva, Xte, yte, *, evaluate_test: bool) -> Dict[str, Any]:
    # Trees: unscaled numerics; categorical already one-hot
    clf = HistGradientBoostingClassifier(
        max_depth=HGB_CFG["max_depth"],
        learning_rate=HGB_CFG["learning_rate"],
        max_iter=HGB_CFG["max_iter"],
        l2_regularization=HGB_CFG["l2_regularization"],
        random_state=HGB_CFG["random_state"],
        early_stopping=False,
    )
    clf.fit(Xtr, ytr)
    best_i, best_a = 1, -1.0
    for i, proba in enumerate(clf.staged_predict_proba(Xva), start=1):
        a = float(average_precision_score(yva, proba[:, 1]))
        if a > best_a + 1e-12:
            best_a = a
            best_i = i
    # Refit with selected iterations
    clf2 = HistGradientBoostingClassifier(
        max_depth=HGB_CFG["max_depth"],
        learning_rate=HGB_CFG["learning_rate"],
        max_iter=best_i,
        l2_regularization=HGB_CFG["l2_regularization"],
        random_state=HGB_CFG["random_state"],
        early_stopping=False,
    )
    clf2.fit(Xtr, ytr)
    pva = clf2.predict_proba(Xva)[:, 1].astype(np.float64)
    thr = tune_thr_max_f1(yva, pva)
    out: Dict[str, Any] = {
        "learner": "HistGradientBoostingClassifier",
        "note": "xgboost/lightgbm not installed; sklearn HGB predeclared substitute",
        "config": dict(HGB_CFG),
        "selection": "best_val_auprc_via_staged_predict_proba",
        "best_iteration_by_val_auprc": best_i,
        "best_val_auprc": best_a,
        "feature_dim": int(Xtr.shape[1]),
        "scaler": "none_unscaled_for_trees",
        "validation": {
            "threshold_0.5": metrics_block(yva, pva, 0.5),
            "threshold_val_selected_max_f1": metrics_block(yva, pva, thr),
            "validation_selected_threshold": thr,
        },
        "val_auprc": best_a,
        "test_evaluated": False,
    }
    if evaluate_test:
        pte = clf2.predict_proba(Xte)[:, 1].astype(np.float64)
        out["test"] = {
            "threshold_0.5": metrics_block(yte, pte, 0.5),
            "threshold_val_selected_max_f1": metrics_block(yte, pte, thr),
        }
        out["test_evaluated"] = True
        out["test_auprc"] = float(average_precision_score(yte, pte))
    return out


def memory_projection(n_train: int, n_feat: int) -> Dict[str, Any]:
    bytes_x = n_train * n_feat * 4  # float32
    # raw df rough + 3 matrices + sklearn copies
    proj_gb = (bytes_x * 6 + 2_000_000_000) / (1024**3)
    return {
        "n_train": n_train,
        "n_features_example": n_feat,
        "approx_feature_matrix_gb_float32": bytes_x / (1024**3),
        "projected_peak_gb_conservative": proj_gb,
        "safe_for_64g": proj_gb < 48,
        "safe_for_128g": proj_gb < 100,
    }


def run_contract(
    df: pd.DataFrame,
    contract: str,
    *,
    learners: Sequence[str],
    evaluate_test: bool,
    device: torch.device,
    role: str,
) -> Dict[str, Any]:
    tr, va, te = df.attrs["tr"], df.attrs["va"], df.attrs["te"]
    type_categories = sorted(int(x) for x in np.unique(df.iloc[tr]["type_code"].to_numpy()))
    Xtr, cols = build_matrix(df, tr, contract=contract, type_categories=type_categories)
    Xva, _ = build_matrix(df, va, contract=contract, type_categories=type_categories)
    Xte, _ = build_matrix(df, te, contract=contract, type_categories=type_categories)
    ytr = df["Is Laundering"].to_numpy()[tr].astype(np.int64)
    yva = df["Is Laundering"].to_numpy()[va].astype(np.int64)
    yte = df["Is Laundering"].to_numpy()[te].astype(np.int64)

    cells: Dict[str, Any] = {}
    for learner in learners:
        logging.info("Fitting %s on %s (test=%s)", learner, contract, evaluate_test)
        if learner == "logistic":
            cell = fit_logistic(Xtr, ytr, Xva, yva, Xte, yte, evaluate_test=evaluate_test)
        elif learner == "mlp":
            cell = fit_mlp(Xtr, ytr, Xva, yva, Xte, yte, evaluate_test=evaluate_test, device=device)
        elif learner == "hgb":
            cell = fit_hgb(Xtr, ytr, Xva, yva, Xte, yte, evaluate_test=evaluate_test)
        else:
            raise ValueError(learner)
        cell["feature_contract_id"] = contract
        cell["feature_columns"] = cols
        cell["role"] = role
        cell["split"] = "temporal_hourly_steps_1_280_281_354_355_743"
        cell["no_resampling"] = True
        cell["table_eligible"] = False
        cell["exploratory_posthoc"] = False
        cell["thesis_role"] = "supervised_ceiling_diagnostic"
        path = CELLS / f"{contract}__{learner}.json"
        write_json(path, cell)
        cells[learner] = {"path": str(path.relative_to(REPO)), "val_auprc": cell["val_auprc"], "cell": cell}
        logging.info("  %s val_auprc=%.6f", learner, cell["val_auprc"])
    return {
        "contract": contract,
        "role": role,
        "feature_columns": cols,
        "type_categories_train": type_categories,
        "memory": memory_projection(int(Xtr.shape[0]), int(Xtr.shape[1])),
        "learners": cells,
    }


def load_comparators() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if XONLY_PATH.is_file():
        x = json.loads(XONLY_PATH.read_text())
        out["x_only_legacy_duplicate"] = {
            "path": str(XONLY_PATH.relative_to(REPO)),
            "val_auprc": float(x["validation"]["threshold_0.5"]["auprc"]),
            "test_auprc": float(x["test"]["threshold_0.5"]["auprc"]),
            "test_f1_at_0.5": float(x["test"]["threshold_0.5"]["f1"]),
            "protocol_note": "compatibility features only; train-fit StandardScaler; not native balances",
        }
    if GIN_AGG.is_file():
        g = json.loads(GIN_AGG.read_text())
        # aggregate keys may vary
        out["supervised_multigin_eu"] = {
            "path": str(GIN_AGG.relative_to(REPO)),
            "test_paper_argmax_f1_mean": 0.2020,
            "test_auprc_mean": 0.2553,
            "val_paper_argmax_f1_mean": 0.1907,
            "protocol_note": "ports+emlps Multi-GIN; paysim_legacy_duplicate_v1; paper_argmax",
        }
        # try to get val auprc mean from per_seed if present
        try:
            vals = []
            for s in ("1", "2", "3"):
                vals.append(float(g["per_seed_eval"][s]["splits"]["val"]["auprc"]))
            out["supervised_multigin_eu"]["val_auprc_mean"] = float(np.mean(vals))
        except Exception:
            out["supervised_multigin_eu"]["val_auprc_mean"] = None
    out["prevalence_baseline"] = {
        "train": EXPECTED_SPLITS["train"]["n_positives"] / EXPECTED_SPLITS["train"]["n"],
        "val": EXPECTED_SPLITS["val"]["n_positives"] / EXPECTED_SPLITS["val"]["n"],
        "test": EXPECTED_SPLITS["test"]["n_positives"] / EXPECTED_SPLITS["test"]["n"],
        "note": "AUPRC of random ranking equals prevalence",
    }
    return out


def gates(primary: Dict[str, Any], comparators: Dict[str, Any], sensitivity: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    x_val = float(comparators.get("x_only_legacy_duplicate", {}).get("val_auprc", float("nan")))
    gin_val = comparators.get("supervised_multigin_eu", {}).get("val_auprc_mean")
    # primary learner by val AUPRC among core learners
    learner_scores = {k: float(v["val_auprc"]) for k, v in primary["learners"].items()}
    best_learner = max(learner_scores, key=learner_scores.get)
    best_val = learner_scores[best_learner]
    flagged_delta = None
    if sensitivity is not None and best_learner in sensitivity["learners"]:
        flagged_delta = float(sensitivity["learners"][best_learner]["val_auprc"]) - best_val
    improves_x = (best_val - x_val) >= MATERIAL_AUPRC if math.isfinite(x_val) else None
    exceeds_gin = (best_val - float(gin_val)) >= MATERIAL_AUPRC if gin_val is not None else None
    return {
        "material_threshold_abs_auprc": MATERIAL_AUPRC,
        "primary_contract": primary["contract"],
        "primary_learner_by_val_auprc": best_learner,
        "primary_val_auprc": best_val,
        "learner_val_auprc": learner_scores,
        "q1_native_balances_improve_over_xonly": improves_x,
        "delta_vs_xonly_val_auprc": best_val - x_val if math.isfinite(x_val) else None,
        "q2_exceeds_supervised_gin_val_auprc": exceeds_gin,
        "delta_vs_gin_val_auprc": (best_val - float(gin_val)) if gin_val is not None else None,
        "q3_isFlaggedFraud_delta_val_auprc": flagged_delta,
        "q4_gin_primarily_feature_contract_limited": bool(improves_x) and bool(exceeds_gin) if improves_x is not None and exceeds_gin is not None else None,
        "q5_native_multigin_justified": bool(improves_x) if improves_x is not None else None,
        "interpretation_notes": [
            "Selection uses validation AUPRC only on paysim_native_core_v1.",
            "full/isFlaggedFraud is sensitivity-only.",
            "Test inspected once after primary choices locked.",
        ],
    }


def write_note(payload: Dict[str, Any]) -> None:
    g = payload["gates"]
    prim = payload["primary"]
    lines = [
        f"# PaySim native tabular baseline (`{TAG}`)",
        "",
        "> Supervised ceiling diagnostic under the locked temporal split. "
        "`table_eligible=false`. Not transfer evaluation.",
        "",
        f"- Formatted SHA256 verified: `{EXPECTED_FMT_SHA}`",
        f"- Split: steps train 1–280 / val 281–354 / test 355–743 (hashes matched failure audit)",
        f"- Tree learner: HistGradientBoostingClassifier (xgboost/lightgbm not installed)",
        f"- Jobs: see `{OUT.relative_to(REPO)}/submission.json`",
        "",
        "## Primary contract results (`paysim_native_core_v1`)",
        "",
        f"| Learner | val AUPRC | selected |",
        f"|---------|----------:|----------|",
    ]
    for k, v in prim["learners"].items():
        mark = " **primary**" if k == g["primary_learner_by_val_auprc"] else ""
        lines.append(f"| {k} | {v['val_auprc']:.6f} |{mark} |")
    pl = g["primary_learner_by_val_auprc"]
    cell = prim["learners"][pl]["cell"]
    lines += [
        "",
        f"**Primary learner:** `{pl}` (val AUPRC={g['primary_val_auprc']:.6f})",
        "",
        "### Locked test (after selection)",
        "",
    ]
    if cell.get("test_evaluated"):
        t05 = cell["test"]["threshold_0.5"]
        tv = cell["test"]["threshold_val_selected_max_f1"]
        lines += [
            f"| rule | AUROC | AUPRC | F1 | P | R | PPR | AUPRC/π |",
            f"|------|------:|------:|---:|--:|--:|----:|--------:|",
            f"| 0.5 | {t05['auroc']:.4f} | {t05['auprc']:.4f} | {t05['f1']:.4f} | {t05['precision']:.4f} | {t05['recall']:.4f} | {t05['positive_prediction_rate']:.6f} | {t05['auprc_over_prevalence']:.1f} |",
            f"| val-tuned | {tv['auroc']:.4f} | {tv['auprc']:.4f} | {tv['f1']:.4f} | {tv['precision']:.4f} | {tv['recall']:.4f} | {tv['positive_prediction_rate']:.6f} | {tv['auprc_over_prevalence']:.1f} |",
            "",
            f"P@100/500/1000 @ scores: {t05['precision_at_100']:.3f} / {t05['precision_at_500']:.3f} / {t05['precision_at_1000']:.3f}",
        ]
    lines += [
        "",
        "## Gates",
        "",
        f"1. Native vs X-only val AUPRC improve ≥{MATERIAL_AUPRC}? **{g['q1_native_balances_improve_over_xonly']}** (Δ={g['delta_vs_xonly_val_auprc']})",
        f"2. Exceeds Multi-GIN val AUPRC by ≥{MATERIAL_AUPRC}? **{g['q2_exceeds_supervised_gin_val_auprc']}** (Δ={g['delta_vs_gin_val_auprc']})",
        f"3. isFlaggedFraud Δ val AUPRC: **{g['q3_isFlaggedFraud_delta_val_auprc']}**",
        f"4. GIN primarily feature-contract-limited? **{g['q4_gin_primarily_feature_contract_limited']}**",
        f"5. Native Multi-GIN justified? **{g['q5_native_multigin_justified']}**",
        "",
        "## Protocol caveats",
        "",
        "- Temporal full-imbalance cohort (no resampling).",
        "- Compatibility X-only and Multi-GIN use `paysim_legacy_duplicate_v1` (no balances).",
        "- This diagnostic uses raw native balances; not an AMLWorld transfer setup.",
        "",
        "## Artifacts",
        "",
        f"- `{FINAL_JSON.relative_to(REPO)}`",
        f"- cells: `{CELLS.relative_to(REPO)}/`",
    ]
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_registry(payload: Dict[str, Any]) -> None:
    reg_path = REPO / "results/diagnostics/thesis_experiment_registry.json"
    if not reg_path.is_file():
        return
    reg = json.loads(reg_path.read_text())
    rows = reg.get("rows", [])
    existing = {r.get("run_id") for r in rows if isinstance(r, dict)}
    g = payload["gates"]
    pl = g["primary_learner_by_val_auprc"]
    cell = payload["primary"]["learners"][pl]["cell"]
    run_id = f"{TAG}|paysim_native_core_v1|{pl}|seed2"
    if run_id in existing:
        return
    row = {
        "run_id": run_id,
        "dataset": "PaySim",
        "objective": "supervised_tabular_ceiling_diagnostic",
        "encoder": "none_tabular",
        "seed": 2,
        "thesis_role": "thesis_supporting",
        "feature_contract_id": "paysim_native_core_v1",
        "test_auprc": cell.get("test_auprc"),
        "test_auroc": cell.get("test", {}).get("threshold_0.5", {}).get("auroc") if cell.get("test") else None,
        "source": str(FINAL_JSON),
        "table_eligible": False,
        "preserve_seed_edges": False,
    }
    rows.append(row)
    reg["rows"] = rows
    reg["row_count"] = len(rows)
    write_json(reg_path, reg)
    notes_reg = REPO / "notes/thesis_experiment_registry.md"
    if notes_reg.is_file():
        with notes_reg.open("a") as f:
            f.write(
                f"\n\n## {TAG}\n\n"
                f"- Appended `{run_id}` (table_eligible=false); historical rows unchanged.\n"
                f"- See `{NOTE.relative_to(REPO)}`.\n"
            )


def cmd_smoke(args: argparse.Namespace) -> int:
    logger_setup()
    OUT.mkdir(parents=True, exist_ok=True)
    CELLS.mkdir(parents=True, exist_ok=True)
    df, integrity = load_aligned_native_table()
    # memory projection for core features
    tr = df.attrs["tr"]
    type_categories = sorted(int(x) for x in np.unique(df.iloc[tr]["type_code"].to_numpy()))
    Xtr, cols = build_matrix(df, tr, contract="paysim_native_core_v1", type_categories=type_categories)
    mem = memory_projection(Xtr.shape[0], Xtr.shape[1])
    if not mem["safe_for_128g"]:
        write_json(
            SMOKE_JSON,
            {"ok": False, "reason": "memory_projection_unsafe", "integrity": integrity, "memory": mem},
        )
        raise SystemExit("memory projection unsafe; refusing to subsample")
    # tiny end-to-end: logistic on core, val only (no test)
    ytr = df["Is Laundering"].to_numpy()[tr].astype(np.int64)
    va = df.attrs["va"]
    Xva, _ = build_matrix(df, va, contract="paysim_native_core_v1", type_categories=type_categories)
    yva = df["Is Laundering"].to_numpy()[va].astype(np.int64)
    cell = fit_logistic(Xtr, ytr, Xva, yva, Xva, yva, evaluate_test=False)
    payload = {
        "ok": True,
        "mode": "smoke",
        "integrity": integrity,
        "memory": mem,
        "feature_columns_core": cols,
        "smoke_logistic_val_auprc": cell["val_auprc"],
        "comparators": load_comparators(),
    }
    write_json(SMOKE_JSON, payload)
    logging.info("SMOKE PASS val_auprc=%.6f mem=%s", cell["val_auprc"], mem)
    return 0


def cmd_full(args: argparse.Namespace) -> int:
    logger_setup()
    OUT.mkdir(parents=True, exist_ok=True)
    CELLS.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    df, integrity = load_aligned_native_table()
    comparators = load_comparators()

    # Phase 1: val-only fits for selection (no test)
    logging.info("Phase 1: validation selection (no test)")
    core = run_contract(
        df,
        "paysim_native_core_v1",
        learners=("logistic", "mlp", "hgb"),
        evaluate_test=False,
        device=device,
        role="primary",
    )
    core_deltas = run_contract(
        df,
        "paysim_native_core_v1_with_deltas",
        learners=("logistic", "mlp", "hgb"),
        evaluate_test=False,
        device=device,
        role="primary_with_deltas_report_only",
    )
    full = run_contract(
        df,
        "paysim_native_full_v1",
        learners=("logistic", "mlp", "hgb"),
        evaluate_test=False,
        device=device,
        role="sensitivity_isFlaggedFraud",
    )

    # Lock primary learner on core (no deltas)
    learner_scores = {k: float(v["val_auprc"]) for k, v in core["learners"].items()}
    primary_learner = max(learner_scores, key=learner_scores.get)
    logging.info("Locked primary learner=%s val_auprc=%.6f", primary_learner, learner_scores[primary_learner])

    # Phase 2: refit primary learner on core with test unlocked once
    logging.info("Phase 2: locked test evaluation for primary learner only (+ sensitivity companion)")
    core_test = run_contract(
        df,
        "paysim_native_core_v1",
        learners=(primary_learner,),
        evaluate_test=True,
        device=device,
        role="primary_locked_test",
    )
    # update core primary cell with test
    core["learners"][primary_learner] = core_test["learners"][primary_learner]

    # Sensitivity: same learner on full contract with test (does not change primary)
    full_test = run_contract(
        df,
        "paysim_native_full_v1",
        learners=(primary_learner,),
        evaluate_test=True,
        device=device,
        role="sensitivity_locked_test",
    )
    full["learners"][primary_learner] = full_test["learners"][primary_learner]

    # Optional: deltas primary learner test for report
    deltas_test = run_contract(
        df,
        "paysim_native_core_v1_with_deltas",
        learners=(primary_learner,),
        evaluate_test=True,
        device=device,
        role="deltas_report_locked_test",
    )
    core_deltas["learners"][primary_learner] = deltas_test["learners"][primary_learner]

    gate = gates(core, comparators, full)
    payload = {
        "artifact": TAG,
        "table_eligible": False,
        "thesis_role": "supervised_ceiling_diagnostic",
        "integrity": integrity,
        "comparators": comparators,
        "primary": {
            "contract": core["contract"],
            "feature_columns": core["feature_columns"],
            "learners": {
                k: {"path": v["path"], "val_auprc": v["val_auprc"], "cell": v["cell"]}
                for k, v in core["learners"].items()
            },
        },
        "primary_with_deltas": {
            "contract": core_deltas["contract"],
            "learners": {
                k: {"path": v["path"], "val_auprc": v["val_auprc"]}
                for k, v in core_deltas["learners"].items()
            },
        },
        "sensitivity_full": {
            "contract": full["contract"],
            "learners": {
                k: {"path": v["path"], "val_auprc": v["val_auprc"]}
                for k, v in full["learners"].items()
            },
        },
        "gates": gate,
        "hgb_note": "HistGradientBoostingClassifier used; lightgbm/xgboost not installed",
        "material_threshold_abs_auprc": MATERIAL_AUPRC,
    }
    # strip bulky nested cell copies from top-level except primary learner test cell already in path
    for block in ("primary",):
        for k, v in payload[block]["learners"].items():
            # keep cell for primary only
            if k != gate["primary_learner_by_val_auprc"]:
                v.pop("cell", None)
    write_json(FINAL_JSON, payload)
    write_note(payload)
    append_registry(payload)
    logging.info("Wrote %s and %s", FINAL_JSON, NOTE)
    logging.info("Gates: %s", json.dumps({k: gate[k] for k in gate if k.startswith("q")}, indent=2))
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    ps = sub.add_parser("smoke")
    ps.set_defaults(func=cmd_smoke)
    pf = sub.add_parser("full")
    pf.add_argument("--device", default="cpu")
    pf.set_defaults(func=cmd_full)
    args = p.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
