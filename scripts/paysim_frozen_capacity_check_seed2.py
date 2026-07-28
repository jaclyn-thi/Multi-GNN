#!/usr/bin/env python3
"""Bounded PaySim frozen-embedding capacity check (seed-2 validation only).

Single-process audit:
  - reuse exact existing logistic cells where present
  - fit PaperStyleMLP (≤5 epochs) for all five stacks
  - no GNN training, no embedding extraction, no test evaluation

exploratory_posthoc=true, table_eligible=false
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gcpal_txn_node.eval_mlp import PaperStyleMLP, _predict_proba  # noqa: E402
from linear_probe import load_embedding_npz  # noqa: E402
from ranking_metrics import alert_budget_metrics  # noqa: E402
from train_util import extract_param  # noqa: E402
from util import create_parser, logger_setup, set_seed  # noqa: E402

TAG = "paysim_frozen_capacity_check_seed2"
RESULT_DIR = ROOT / "results" / "diagnostics" / TAG
CELLS = RESULT_DIR / "cells"
OUT_JSON = ROOT / "results" / "diagnostics" / f"{TAG}.json"
NOTES = ROOT / "notes" / f"{TAG}.md"
SUBMISSION = RESULT_DIR / "submission.json"

EMB_P1 = ROOT / "embeddings" / "final_corrected_no_preserve_multiseed" / "seed2_P1_strict_inductive_legacy"
EMB_RND = (
    ROOT
    / "embeddings"
    / "final_corrected_no_preserve_multiseed"
    / "controls_random_paysim_legacy_duplicate_v1"
)

EXPECTED_ID = {
    "train": "2511d0de4504e52960b414e6b84d47486089a573b6c57aa040feb561e2d2809a",
    "val": "a8de85f31dfe91bd767da6daedf9f2bab474d08c8412c796111e8767ebd0b1e3",
}

# Exact logistic cells (same P1 IDs / recipe) to reuse — not refit.
LOGISTIC_REUSE = {
    "X": ROOT
    / "results/diagnostics/paysim_temporal_flow_downstream/cells/seed2_X_validation.json",
    "H": ROOT
    / "results/diagnostics/paysim_temporal_flow_downstream/cells/seed2_H_validation.json",
    "H+X": ROOT
    / "results/diagnostics/paysim_temporal_flow_downstream/cells/seed2_HplusX_validation.json",
    "random_H": ROOT
    / "results/diagnostics/final_corrected_no_preserve_multiseed/cells/control_random_paysim_legacy_duplicate_v1.json",
}

MARGIN = 0.003
LOGISTIC_SEED = 1
MLP_SEED = 2
MLP_EPOCHS = 5
MLP_LR = 1e-3
MLP_BS = 8192
STACKS = ("X", "H", "H+X", "random_H", "random_H+X")


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


def load_x_edge_native() -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "probe_feature_ablation", ROOT / "scripts" / "probe_feature_ablation.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Required before exec_module so dataclasses can resolve ClassVar annotations.
    sys.modules["probe_feature_ablation"] = mod
    spec.loader.exec_module(mod)
    df, df_train, tr_ids, va_ids, te_ids, dspec = mod.load_dataset_frames(
        "PaySim", str(ROOT / "data_config.json")
    )
    x, names, _, meta = mod.build_full_feature_matrix(
        df, df_train, ("edge_native",), categorical_encoding="one_hot"
    )
    return x.astype(np.float32), list(names), {
        "x_source": "edge_native_one_hot_train_fit",
        "n_features": int(x.shape[1]),
        "label_col": dspec.label_col,
        "split_counts": {"train": int(len(tr_ids)), "val": int(len(va_ids)), "test": int(len(te_ids))},
        "meta": meta,
        "note": "test split listed for coverage only; test features/labels never used",
    }


def stack_mat(name: str, z: Optional[np.ndarray], x: np.ndarray) -> np.ndarray:
    core = name.replace("random_", "")
    if core == "X":
        return x.astype(np.float32)
    if core == "H":
        assert z is not None
        return z.astype(np.float32)
    if core == "H+X":
        assert z is not None
        return np.concatenate([z, x], axis=1).astype(np.float32)
    raise ValueError(name)


def failure_report(reason: str, detail: Dict[str, Any]) -> int:
    payload = {
        "ok": False,
        "failure": reason,
        "detail": detail,
        "test_evaluated": False,
        "encoder_training": False,
        "exploratory_posthoc": True,
        "table_eligible": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(OUT_JSON, payload)
    NOTES.write_text(
        "\n".join(
            [
                "# PaySim frozen capacity check (seed 2)",
                "",
                f"> **FAILED preflight:** {reason}",
                f"> Twin: `{OUT_JSON.relative_to(ROOT)}`",
                "",
                "```json",
                json.dumps(detail, indent=2),
                "```",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    logging.error("PREFLIGHT_FAIL %s", reason)
    return 2


def verify_logistic_reuse(path: Path, stack: str) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    cell = json.loads(path.read_text(encoding="utf-8"))
    ids = cell.get("ids") or {}
    for sp in ("train", "val"):
        sha = ((ids.get(sp) or {}).get("sha256_of_ids_bytes"))
        if sha != EXPECTED_ID[sp]:
            logging.warning("reuse %s id mismatch on %s — skip reuse", stack, sp)
            return None
    # Accept TF-ablation cells (stack name) and multiseed random H control
    if stack == "random_H":
        if cell.get("feature_stack") not in (None, "H_only_post128") and "H" not in str(
            cell.get("feature_stack", "H")
        ):
            # control file is H-only by construction
            pass
        if cell.get("learner") != "LogisticRegression":
            return None
    else:
        if cell.get("stack") != stack and cell.get("feature_stack") != stack:
            # TF cells use stack key
            if cell.get("stack") != stack:
                return None
        if cell.get("learner") != "LogisticRegression":
            return None
    auprc = None
    if "val_auprc_at_0.5" in cell:
        auprc = float(cell["val_auprc_at_0.5"])
    elif "validation" in cell and "threshold_0.5" in cell["validation"]:
        auprc = float(cell["validation"]["threshold_0.5"]["auprc"])
    if auprc is None:
        return None
    return {
        "reused": True,
        "source_path": str(path.relative_to(ROOT)),
        "learner": "LogisticRegression",
        "stack": stack,
        "val_auprc_at_0.5": auprc,
        "validation": cell.get("validation"),
        "ids": cell.get("ids"),
        "coverage": cell.get("coverage"),
        "class_weight": cell.get("class_weight"),
        "C": cell.get("C", 1.0),
        "downstream_seed": cell.get("downstream_seed", LOGISTIC_SEED),
        "test_evaluated": False,
        "exploratory_posthoc": True,
        "table_eligible": False,
    }


def fit_logistic(mat_tr, y_tr, mat_va, y_va, ids_tr, ids_va, stack: str) -> Dict[str, Any]:
    scaler = StandardScaler()
    tr = scaler.fit_transform(mat_tr).astype(np.float32)
    va = scaler.transform(mat_va).astype(np.float32)
    cw = gin_cw()
    set_seed(LOGISTIC_SEED)
    clf = LogisticRegression(
        class_weight=cw, max_iter=1000, random_state=LOGISTIC_SEED, solver="lbfgs", n_jobs=1, C=1.0
    )
    clf.fit(tr, y_tr)
    pva = clf.predict_proba(va)[:, 1].astype(np.float64)
    thr = tune_thr_max_f1(y_va, pva)
    return {
        "reused": False,
        "learner": "LogisticRegression",
        "stack": stack,
        "feature_dim": int(tr.shape[1]),
        "class_weight_mode": "model",
        "class_weight": {str(k): float(v) for k, v in cw.items()},
        "C": 1.0,
        "downstream_seed": LOGISTIC_SEED,
        "scaler": "StandardScaler_fit_train_only",
        "ids": {"train": ids_hash(ids_tr), "val": ids_hash(ids_va)},
        "coverage": {
            "train": {"n": int(y_tr.shape[0]), "n_positives": int(y_tr.sum()), "positive_rate": float(y_tr.mean())},
            "val": {"n": int(y_va.shape[0]), "n_positives": int(y_va.sum()), "positive_rate": float(y_va.mean())},
        },
        "validation": {
            "threshold_0.5": metrics_block(y_va, pva, 0.5),
            "threshold_val_selected_max_f1": metrics_block(y_va, pva, thr),
            "validation_selected_threshold": thr,
        },
        "val_auprc_at_0.5": float(average_precision_score(y_va, pva)),
        "test_evaluated": False,
        "exploratory_posthoc": True,
        "table_eligible": False,
    }


def fit_mlp(mat_tr, y_tr, mat_va, y_va, ids_tr, ids_va, stack: str, device: torch.device) -> Dict[str, Any]:
    scaler = StandardScaler()
    tr = scaler.fit_transform(mat_tr).astype(np.float32)
    va = scaler.transform(mat_va).astype(np.float32)
    cw = gin_cw()
    # Established GIN model class weights → BCE pos_weight = w_pos / w_neg
    pos_weight = torch.tensor([cw[1] / cw[0]], dtype=torch.float32, device=device)

    torch.manual_seed(MLP_SEED)
    np.random.seed(MLP_SEED)
    model = PaperStyleMLP(int(tr.shape[1])).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=MLP_LR)
    x_t = torch.from_numpy(tr)
    y_t = torch.from_numpy(y_tr.astype(np.float32))
    n = tr.shape[0]
    history: List[Dict[str, float]] = []
    best_auprc, best_state, best_ep = -1.0, None, -1

    for ep in range(MLP_EPOCHS):
        model.train()
        perm = np.random.RandomState(MLP_SEED * 1009 + ep).permutation(n)
        for start in range(0, n, MLP_BS):
            idx = perm[start : start + MLP_BS]
            opt.zero_grad(set_to_none=True)
            logits = model(x_t[idx].to(device))
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, y_t[idx].to(device), pos_weight=pos_weight
            )
            loss.backward()
            opt.step()
        pva = _predict_proba(model, va, batch_size=MLP_BS, device=device)
        auprc = float(average_precision_score(y_va, pva)) if len(np.unique(y_va)) > 1 else float("nan")
        auroc = float(roc_auc_score(y_va, pva)) if len(np.unique(y_va)) > 1 else float("nan")
        history.append({"epoch": float(ep + 1), "val_auprc": auprc, "val_auroc": auroc})
        if auprc > best_auprc + 1e-12:
            best_auprc = auprc
            best_ep = ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)
    model.to(device)
    pva = _predict_proba(model, va, batch_size=MLP_BS, device=device)
    thr = tune_thr_max_f1(y_va, pva)
    return {
        "reused": False,
        "learner": "PaperStyleMLP",
        "stack": stack,
        "feature_dim": int(tr.shape[1]),
        "class_weight_mode": "model_pos_weight",
        "class_weight": {str(k): float(v) for k, v in cw.items()},
        "pos_weight": float(cw[1] / cw[0]),
        "downstream_seed": MLP_SEED,
        "mlp_epochs_max": MLP_EPOCHS,
        "mlp_lr": MLP_LR,
        "mlp_batch_size": MLP_BS,
        "loader_num_workers": 0,
        "scaler": "StandardScaler_fit_train_only",
        "selection": "best_epoch_by_validation_auprc",
        "best_epoch_by_val_auprc": best_ep,
        "best_val_auprc": best_auprc,
        "epoch_history": history,
        "ids": {"train": ids_hash(ids_tr), "val": ids_hash(ids_va)},
        "coverage": {
            "train": {"n": int(y_tr.shape[0]), "n_positives": int(y_tr.sum()), "positive_rate": float(y_tr.mean())},
            "val": {"n": int(y_va.shape[0]), "n_positives": int(y_va.sum()), "positive_rate": float(y_va.mean())},
        },
        "validation": {
            "threshold_0.5": metrics_block(y_va, pva, 0.5),
            "threshold_val_selected_max_f1": metrics_block(y_va, pva, thr),
            "validation_selected_threshold": thr,
        },
        "val_auprc_at_0.5": float(average_precision_score(y_va, pva)),
        "test_evaluated": False,
        "test_accessed": False,
        "exploratory_posthoc": True,
        "table_eligible": False,
    }


def auprc_of(cell: Optional[Dict[str, Any]]) -> Optional[float]:
    if cell is None:
        return None
    if "val_auprc_at_0.5" in cell:
        return float(cell["val_auprc_at_0.5"])
    try:
        return float(cell["validation"]["threshold_0.5"]["auprc"])
    except Exception:
        return None


def main() -> int:
    logger_setup()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    CELLS.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    # ---- preflight: artifacts exist; never open test.npz ----
    for p in (EMB_P1 / "train.npz", EMB_P1 / "val.npz", EMB_RND / "train.npz", EMB_RND / "val.npz"):
        if not p.is_file():
            return failure_report("missing_embedding_npz", {"path": str(p)})

    # Refuse test access explicitly
    for p in (EMB_P1 / "test.npz", EMB_RND / "test.npz"):
        if p.is_file():
            logging.info("test.npz present at %s but will not be opened", p)

    z_tr, y_tr, ids_tr = load_embedding_npz(EMB_P1 / "train.npz")
    z_va, y_va, ids_va = load_embedding_npz(EMB_P1 / "val.npz")
    zr_tr, yr_tr, idr_tr = load_embedding_npz(EMB_RND / "train.npz")
    zr_va, yr_va, idr_va = load_embedding_npz(EMB_RND / "val.npz")

    meta_tr, meta_va = ids_hash(ids_tr), ids_hash(ids_va)
    if meta_tr["sha256_of_ids_bytes"] != EXPECTED_ID["train"]:
        return failure_report("p1_train_id_hash_mismatch", {"got": meta_tr, "expected": EXPECTED_ID["train"]})
    if meta_va["sha256_of_ids_bytes"] != EXPECTED_ID["val"]:
        return failure_report("p1_val_id_hash_mismatch", {"got": meta_va, "expected": EXPECTED_ID["val"]})
    if not np.array_equal(ids_tr, idr_tr) or not np.array_equal(ids_va, idr_va):
        return failure_report(
            "random_control_id_misaligned_vs_p1",
            {"p1_train": meta_tr, "random_train": ids_hash(idr_tr), "p1_val": meta_va, "random_val": ids_hash(idr_va)},
        )
    if not np.array_equal(y_tr, yr_tr) or not np.array_equal(y_va, yr_va):
        return failure_report("random_control_label_misaligned_vs_p1", {})

    coverage = {
        "train": {
            "n": int(y_tr.shape[0]),
            "n_positives": int(y_tr.sum()),
            "positive_rate": float(y_tr.mean()),
            "ids": meta_tr,
        },
        "val": {
            "n": int(y_va.shape[0]),
            "n_positives": int(y_va.sum()),
            "positive_rate": float(y_va.mean()),
            "ids": meta_va,
        },
        "p1_vs_random_id_match": True,
        "coverage_ok": True,
    }
    write_json(CELLS / "preflight_coverage.json", coverage)
    logging.info("Preflight coverage OK: train n=%s pos=%s; val n=%s pos=%s",
                 coverage["train"]["n"], coverage["train"]["n_positives"],
                 coverage["val"]["n"], coverage["val"]["n_positives"])

    # X features (train/val only via edge_id index)
    x_full, x_names, x_meta = load_x_edge_native()
    if int(ids_tr.max()) >= x_full.shape[0] or int(ids_va.max()) >= x_full.shape[0]:
        return failure_report("x_matrix_too_short_for_edge_ids", {"x_rows": int(x_full.shape[0])})
    x_tr, x_va = x_full[ids_tr], x_full[ids_va]

    logistic_cells: Dict[str, Dict[str, Any]] = {}
    mlp_cells: Dict[str, Dict[str, Any]] = {}

    # Reuse logistic where exact; do not invent logistic fits for missing stacks
    # (missing stacks are covered by PaperStyleMLP below).
    for stack, path in LOGISTIC_REUSE.items():
        reused = verify_logistic_reuse(path, stack)
        if reused is not None:
            logistic_cells[stack] = reused
            write_json(CELLS / f"logistic_{stack.replace('+', 'plus')}_reused.json", reused)
            logging.info("Reused logistic %s val_auprc=%.6f from %s", stack, reused["val_auprc_at_0.5"], path.name)
        else:
            logging.info("No exact logistic reuse for %s — MLP will cover", stack)

    # MLP for all five stacks (covers missing logistic cells)
    for stack in STACKS:
        logging.info("Fitting MLP %s (max %d epochs)", stack, MLP_EPOCHS)
        z_use_tr = zr_tr if stack.startswith("random") else z_tr
        z_use_va = zr_va if stack.startswith("random") else z_va
        if stack in ("X",):
            mat_tr, mat_va = x_tr, x_va
        else:
            mat_tr = stack_mat(stack, z_use_tr, x_tr)
            mat_va = stack_mat(stack, z_use_va, x_va)
        cell = fit_mlp(mat_tr, y_tr, mat_va, y_va, ids_tr, ids_va, stack, device)
        mlp_cells[stack] = cell
        write_json(CELLS / f"mlp_{stack.replace('+', 'plus')}.json", cell)
        logging.info("MLP %s best_ep=%s val_auprc=%.6f", stack, cell["best_epoch_by_val_auprc"], cell["val_auprc_at_0.5"])

    # ---- answers (material = abs delta >= 0.003) ----
    def delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
        if a is None or b is None:
            return None
        return float(a - b)

    def material(d: Optional[float]) -> Optional[bool]:
        if d is None:
            return None
        return bool(d >= MARGIN)

    log_hx = auprc_of(logistic_cells.get("H+X"))
    log_x = auprc_of(logistic_cells.get("X"))
    log_h = auprc_of(logistic_cells.get("H"))
    log_rh = auprc_of(logistic_cells.get("random_H"))
    log_rhx = auprc_of(logistic_cells.get("random_H+X"))
    mlp_hx = auprc_of(mlp_cells.get("H+X"))
    mlp_x = auprc_of(mlp_cells.get("X"))
    mlp_h = auprc_of(mlp_cells.get("H"))
    mlp_rh = auprc_of(mlp_cells.get("random_H"))
    mlp_rhx = auprc_of(mlp_cells.get("random_H+X"))

    q = {
        "1_pretrained_HxX_exceeds_X": {
            "logistic_delta": delta(log_hx, log_x),
            "logistic_material": material(delta(log_hx, log_x)),
            "mlp_delta": delta(mlp_hx, mlp_x),
            "mlp_material": material(delta(mlp_hx, mlp_x)),
            "answer_logistic": bool(log_hx is not None and log_x is not None and log_hx > log_x),
            "answer_mlp": bool(mlp_hx is not None and mlp_x is not None and mlp_hx > mlp_x),
        },
        "2_pretrained_HxX_exceeds_random_HxX": {
            "logistic_delta": delta(log_hx, log_rhx),
            "logistic_material": material(delta(log_hx, log_rhx)),
            "mlp_delta": delta(mlp_hx, mlp_rhx),
            "mlp_material": material(delta(mlp_hx, mlp_rhx)),
            "answer_logistic": bool(log_hx is not None and log_rhx is not None and log_hx > log_rhx),
            "answer_mlp": bool(mlp_hx is not None and mlp_rhx is not None and mlp_hx > mlp_rhx),
        },
        "3_mlp_vs_logistic_on_pretrained_HxX": {
            "delta_mlp_minus_logistic": delta(mlp_hx, log_hx),
            "material": material(delta(mlp_hx, log_hx)),
            "answer_mlp_materially_better": bool(
                mlp_hx is not None and log_hx is not None and (mlp_hx - log_hx) >= MARGIN
            ),
        },
        "4_H_useful_above_random_H": {
            "logistic_delta": delta(log_h, log_rh),
            "logistic_material": material(delta(log_h, log_rh)),
            "mlp_delta": delta(mlp_h, mlp_rh),
            "mlp_material": material(delta(mlp_h, mlp_rh)),
            "answer_logistic": bool(log_h is not None and log_rh is not None and (log_h - log_rh) >= MARGIN),
            "answer_mlp": bool(mlp_h is not None and mlp_rh is not None and (mlp_h - mlp_rh) >= MARGIN),
        },
    }

    # Interpretation: probe under-capacity vs weak H
    mlp_helps = q["3_mlp_vs_logistic_on_pretrained_HxX"]["answer_mlp_materially_better"]
    hx_beats_x = q["1_pretrained_HxX_exceeds_X"]["logistic_material"] or q["1_pretrained_HxX_exceeds_X"]["mlp_material"]
    hx_beats_rand = (
        q["2_pretrained_HxX_exceeds_random_HxX"]["logistic_material"]
        or q["2_pretrained_HxX_exceeds_random_HxX"]["mlp_material"]
    )
    h_beats_rand = q["4_H_useful_above_random_H"]["logistic_material"] or q["4_H_useful_above_random_H"]["mlp_material"]

    if mlp_helps and not hx_beats_rand:
        interpretation = "probe_undercapacity_possible_but_H_still_weak_vs_random"
    elif mlp_helps and hx_beats_rand:
        interpretation = "probe_undercapacity_logistic_underfit_mlp_recovers_signal"
    elif not mlp_helps and not hx_beats_rand and not h_beats_rand:
        interpretation = "weak_transferred_H"
    elif not mlp_helps and hx_beats_x and hx_beats_rand:
        interpretation = "embedding_contribution_present_logistic_sufficient"
    else:
        interpretation = "mixed_see_deltas"

    q["5_interpretation"] = {
        "label": interpretation,
        "margin": MARGIN,
        "notes": [
            "Material improvement requires abs val AUPRC delta >= 0.003.",
            "Primary ranking metric: validation AUPRC at decision scores (threshold 0.5 block auprc).",
        ],
    }

    out = {
        "ok": True,
        "title": TAG,
        "exploratory_posthoc": True,
        "table_eligible": False,
        "encoder_training": False,
        "embedding_extraction": False,
        "test_evaluated": False,
        "test_accessed": False,
        "seed": 2,
        "protocol": "P1_strict_inductive_legacy_post128",
        "feature_contract": "paysim_legacy_duplicate_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reused_paths": {
            "embeddings_p1": str(EMB_P1.relative_to(ROOT)),
            "embeddings_random": str(EMB_RND.relative_to(ROOT)),
            "logistic_reuse": {k: str(v.relative_to(ROOT)) for k, v in LOGISTIC_REUSE.items()},
            "x_source": x_meta.get("x_source"),
        },
        "coverage": coverage,
        "x_meta": {k: v for k, v in x_meta.items() if k != "meta"},
        "x_feature_names": x_names,
        "predeclared": {
            "logistic": {"C": 1.0, "class_weight": "model", "seed": LOGISTIC_SEED},
            "PaperStyleMLP": {
                "epochs_max": MLP_EPOCHS,
                "lr": MLP_LR,
                "batch_size": MLP_BS,
                "seed": MLP_SEED,
                "class_weight": "model_pos_weight",
                "loader_num_workers": 0,
                "selection": "best_val_auprc_epoch_then_val_max_f1_threshold",
            },
            "material_margin_auprc": MARGIN,
        },
        "logistic_cells": {
            k: {
                "reused": v.get("reused"),
                "val_auprc_at_0.5": auprc_of(v),
                "source_path": v.get("source_path"),
            }
            for k, v in logistic_cells.items()
        },
        "mlp_cells": {
            k: {
                "val_auprc_at_0.5": auprc_of(v),
                "best_epoch_by_val_auprc": v.get("best_epoch_by_val_auprc"),
                "best_val_auprc": v.get("best_val_auprc"),
                "epoch_history": v.get("epoch_history"),
            }
            for k, v in mlp_cells.items()
        },
        "questions": q,
        "full_logistic": logistic_cells,
        "full_mlp": mlp_cells,
    }
    write_json(OUT_JSON, out)

    lines = [
        "# PaySim frozen capacity check (seed 2)",
        "",
        "> Exploratory / post-hoc. `table_eligible=false`. Validation only; test not accessed.",
        f"> Twin: `{OUT_JSON.relative_to(ROOT)}`",
        "",
        "## Coverage",
        "",
        f"- Train: n={coverage['train']['n']}, positives={coverage['train']['n_positives']}",
        f"- Val: n={coverage['val']['n']}, positives={coverage['val']['n_positives']}",
        f"- ID hashes match expected P1 / random control: **yes**",
        "",
        "## Val AUPRC @ 0.5",
        "",
        "| Stack | Logistic | MLP |",
        "|-------|----------|-----|",
    ]
    for stack in STACKS:
        lv = auprc_of(logistic_cells.get(stack))
        mv = auprc_of(mlp_cells.get(stack))
        if stack in logistic_cells:
            lr = "reused" if logistic_cells[stack].get("reused") else "fit"
            lvs = f"{lv:.6f} ({lr})" if lv is not None else f"n/a ({lr})"
        else:
            lvs = "n/a (no exact cell)"
        mvs = f"{mv:.6f}" if mv is not None else "n/a"
        lines.append(f"| {stack} | {lvs} | {mvs} |")
    lines += [
        "",
        "## Answers",
        "",
        f"1. Pretrained H+X > X? logistic={q['1_pretrained_HxX_exceeds_X']['answer_logistic']} "
        f"(Δ={q['1_pretrained_HxX_exceeds_X']['logistic_delta']}, material={q['1_pretrained_HxX_exceeds_X']['logistic_material']}); "
        f"mlp={q['1_pretrained_HxX_exceeds_X']['answer_mlp']} "
        f"(Δ={q['1_pretrained_HxX_exceeds_X']['mlp_delta']}, material={q['1_pretrained_HxX_exceeds_X']['mlp_material']})",
        f"2. Pretrained H+X > random H+X? logistic={q['2_pretrained_HxX_exceeds_random_HxX']['answer_logistic']} "
        f"(Δ={q['2_pretrained_HxX_exceeds_random_HxX']['logistic_delta']}, material={q['2_pretrained_HxX_exceeds_random_HxX']['logistic_material']}); "
        f"mlp={q['2_pretrained_HxX_exceeds_random_HxX']['answer_mlp']} "
        f"(Δ={q['2_pretrained_HxX_exceeds_random_HxX']['mlp_delta']}, material={q['2_pretrained_HxX_exceeds_random_HxX']['mlp_material']})",
        f"3. MLP materially > logistic on pretrained H+X? "
        f"{q['3_mlp_vs_logistic_on_pretrained_HxX']['answer_mlp_materially_better']} "
        f"(Δ={q['3_mlp_vs_logistic_on_pretrained_HxX']['delta_mlp_minus_logistic']})",
        f"4. H useful above random H (Δ≥{MARGIN})? "
        f"logistic={q['4_H_useful_above_random_H']['answer_logistic']}, mlp={q['4_H_useful_above_random_H']['answer_mlp']}",
        f"5. Interpretation: **{interpretation}**",
        "",
    ]
    NOTES.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logging.info("Wrote %s and %s", OUT_JSON, NOTES)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
