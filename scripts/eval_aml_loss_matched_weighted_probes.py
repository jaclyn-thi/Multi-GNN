#!/usr/bin/env python3
"""Validation-only supervised-loss-matched PaperStyleMLP probe sensitivity.

Standalone secondary analysis. Does not overwrite established probes/embeddings/
checkpoints/packages. No encoder training. No test access.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from aml_loss_matched_weighted_ce import (  # noqa: E402
    class_weight_summary,
    load_class_weights_from_checkpoint,
    matched_weighted_ce_from_one_logit,
    matched_weighted_ce_numpy,
    supervised_weighted_ce_two_logit,
    two_logit_from_one,
    unweighted_binary_ce_from_proba,
    unweighted_binary_ce_numpy,
)
from gcpal_txn_node.eval_mlp import PaperStyleMLP, _predict_proba  # noqa: E402

OUT_ROOT = ROOT / "results/diagnostics/aml_loss_matched_weighted_probe"
TOP_JSON = ROOT / "results/diagnostics/aml_loss_matched_weighted_probe.json"
NOTES = ROOT / "notes/aml_loss_matched_weighted_probe.md"
SUP_CKPT_BEST = (
    ROOT
    / "saved-models/small_hi_legacy_supervised_gin_emlps_ports_50ep_seed2"
    / "checkpoint_best_val_f1.tar"
)
SUP_PRED_DIR = ROOT / "results/diagnostics/common_aml_validation_ce_comparison/predictions"
TF_CACHE = ROOT / "results/cache/temporal_flow_causal/Small-HI"

MLP_EPOCHS = 20
MLP_LR = 1e-3
MLP_BS = 8192
MLP_SEED = 2
R198_DIM = 198
R198_XTF_DIM = 227  # 198 + 24 + 5

CELLS: List[Dict[str, Any]] = [
    {
        "cell_id": "direct_h_r198_only_lr1e-3_ssl_ep03",
        "method": "DIRECT_H",
        "feature_protocol": "R198_only",
        "ssl_lr": 1e-3,
        "ssl_epoch": 3,
        "emb_dir": ROOT
        / "embeddings/direct_r198_40ep_linear_lr_full_extract"
        / "direct_r198_infonce_40ep_seed2_linear_lr1e-3_epoch03"
        / "pre_embedding_3h",
        "original_cell_json": ROOT
        / "results/diagnostics/direct_r198_40ep_linear_lr_full_extract_reeval"
        / "r198_only_lr_analysis/cells"
        / "direct_r198_infonce_40ep_seed2_linear_lr1e-3/epoch_03.json",
    },
    {
        "cell_id": "tfmoe_adaptive_r198_only_lr2e-3_ssl_ep10",
        "method": "TFMOE_adaptive",
        "feature_protocol": "R198_only",
        "ssl_lr": 2e-3,
        "ssl_epoch": 10,
        "emb_dir": ROOT
        / "embeddings/direct_r198_40ep_linear_lr_full_extract"
        / "direct_r198_tfmoe_40ep_seed2_linear_lr2e-3_epoch10"
        / "pre_embedding_3h",
        "original_cell_json": ROOT
        / "results/diagnostics/direct_r198_40ep_linear_lr_full_extract_reeval"
        / "r198_only_lr_analysis/cells"
        / "direct_r198_tfmoe_40ep_seed2_linear_lr2e-3/epoch_10.json",
    },
    {
        "cell_id": "expert_only_r198_only_lr2e-3_ssl_ep10",
        "method": "EXPERT_ONLY",
        "feature_protocol": "R198_only",
        "ssl_lr": 2e-3,
        "ssl_epoch": 10,
        "emb_dir": ROOT
        / "embeddings/tfmoe_weight_ablation_lr2e-3_full_extract"
        / "direct_r198_tfmoe_wtabl_expert_only_20ep_seed2_linear_lr2e-3_epoch10"
        / "pre_embedding_3h",
        "original_cell_json": ROOT
        / "results/diagnostics/tfmoe_weight_ablation_lr2e-3/cells"
        / "direct_r198_tfmoe_wtabl_expert_only_20ep_seed2_linear_lr2e-3/epoch_10.json",
    },
    {
        "cell_id": "expert_only_r198_only_lr2e-3_ssl_ep20",
        "method": "EXPERT_ONLY",
        "feature_protocol": "R198_only",
        "ssl_lr": 2e-3,
        "ssl_epoch": 20,
        "emb_dir": ROOT
        / "embeddings/tfmoe_weight_ablation_lr2e-3_full_extract"
        / "direct_r198_tfmoe_wtabl_expert_only_20ep_seed2_linear_lr2e-3_epoch20"
        / "pre_embedding_3h",
        "original_cell_json": ROOT
        / "results/diagnostics/tfmoe_weight_ablation_lr2e-3/cells"
        / "direct_r198_tfmoe_wtabl_expert_only_20ep_seed2_linear_lr2e-3/epoch_20.json",
    },
    {
        "cell_id": "tfmoe_adaptive_r198_x_tf_lr2e-3_ssl_ep10",
        "method": "TFMOE_adaptive",
        "feature_protocol": "R198_X_TF",
        "ssl_lr": 2e-3,
        "ssl_epoch": 10,
        "emb_dir": ROOT
        / "embeddings/direct_r198_40ep_linear_lr_full_extract"
        / "direct_r198_tfmoe_40ep_seed2_linear_lr2e-3_epoch10"
        / "pre_embedding_3h",
        "original_cell_json": ROOT
        / "results/diagnostics/direct_r198_40ep_linear_lr_full_extract_reeval"
        / "cells/direct_r198_tfmoe_40ep_seed2_linear_lr2e-3/epoch_10.json",
    },
    {
        "cell_id": "expert_only_r198_x_tf_lr2e-3_ssl_ep10",
        "method": "EXPERT_ONLY",
        "feature_protocol": "R198_X_TF",
        "ssl_lr": 2e-3,
        "ssl_epoch": 10,
        "emb_dir": ROOT
        / "embeddings/tfmoe_weight_ablation_lr2e-3_full_extract"
        / "direct_r198_tfmoe_wtabl_expert_only_20ep_seed2_linear_lr2e-3_epoch10"
        / "pre_embedding_3h",
        # No prior X+TF probe for EXPERT_ONLY; Table B marks original as unavailable.
        "original_cell_json": None,
    },
]


def _sha_sorted_ids(ids: np.ndarray) -> str:
    ordered = np.sort(ids.astype(np.int64))
    return hashlib.sha256(ordered.tobytes()).hexdigest()


def _tune_thr(y: np.ndarray, p: np.ndarray) -> float:
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.linspace(0.01, 0.99, 99):
        f1 = float(
            f1_score(y.astype(np.int64), (p >= thr).astype(np.int64), zero_division=0)
        )
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    return best_thr


def _metrics_at(y: np.ndarray, p: np.ndarray, thr: float) -> Dict[str, float]:
    pred = (p >= thr).astype(np.int64)
    y = y.astype(np.int64)
    return {
        "auprc": float(average_precision_score(y, p)),
        "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "positive_prediction_rate": float(pred.mean()) if y.size else 0.0,
        "threshold": float(thr),
    }


def _load_x_tf() -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    spec = importlib.util.spec_from_file_location(
        "probe_feature_ablation", ROOT / "scripts" / "probe_feature_ablation.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["probe_feature_ablation"] = mod
    spec.loader.exec_module(mod)
    df, df_train, _, _, _, _ = mod.load_dataset_frames(
        "Small-HI", str(ROOT / "data_config.json")
    )
    x_raw, feature_names, _, _ = mod.build_full_feature_matrix(
        df, df_train, ("edge_native",), categorical_encoding="one_hot"
    )
    tf = np.load(TF_CACHE / "features.npy").astype(np.float32)
    tf_meta = json.loads((TF_CACHE / "meta.json").read_text())
    tf_names = list(tf_meta.get("feature_names") or tf_meta.get("columns") or [])
    if not tf_names and tf.shape[1] == 5:
        tf_names = [
            "log1p_sender_interarrival",
            "log1p_receiver_interarrival",
            "log1p_sender_past_7d_count",
            "log1p_amount_vs_sender_past_mean",
            "pair_repeat_indicator",
        ]
    manifest = {
        "ordered_blocks": ["R198_Z", "edge_X_one_hot", "temporal_flow"],
        "r198_dim": R198_DIM,
        "edge_x_dim": int(x_raw.shape[1]),
        "tf_dim": int(tf.shape[1]),
        "total_dim": int(R198_DIM + x_raw.shape[1] + tf.shape[1]),
        "edge_x_feature_names": list(feature_names),
        "tf_feature_names": tf_names,
    }
    return x_raw.astype(np.float32), tf, manifest


def _stack_features(
    z: np.ndarray,
    edge_id: np.ndarray,
    protocol: str,
    x_raw: Optional[np.ndarray],
    tf: Optional[np.ndarray],
) -> np.ndarray:
    if protocol == "R198_only":
        if z.shape[1] != R198_DIM:
            raise ValueError(f"R198_only expects dim {R198_DIM}, got {z.shape[1]}")
        return z.astype(np.float32)
    if protocol != "R198_X_TF":
        raise ValueError(protocol)
    assert x_raw is not None and tf is not None
    eid = edge_id.astype(np.int64)
    mat = np.concatenate([z, x_raw[eid], tf[eid]], axis=1).astype(np.float32)
    if mat.shape[1] != R198_XTF_DIM:
        raise ValueError(f"R198_X_TF expects dim {R198_XTF_DIM}, got {mat.shape[1]}")
    return mat


def _assert_split(tr: Dict[str, np.ndarray], va: Dict[str, np.ndarray]) -> Dict[str, Any]:
    eid_tr = tr["edge_id"].astype(np.int64).reshape(-1)
    eid_va = va["edge_id"].astype(np.int64).reshape(-1)
    y_tr = tr["y"].astype(np.int64).reshape(-1)
    y_va = va["y"].astype(np.int64).reshape(-1)
    if eid_tr.size != np.unique(eid_tr).size:
        raise RuntimeError("duplicate train EdgeIDs")
    if eid_va.size != np.unique(eid_va).size:
        raise RuntimeError("duplicate val EdgeIDs")
    if len(set(eid_tr.tolist()) & set(eid_va.tolist())) != 0:
        raise RuntimeError("train/val EdgeID overlap")
    if y_tr.shape[0] != eid_tr.shape[0] or y_va.shape[0] != eid_va.shape[0]:
        raise RuntimeError("label/EdgeID length mismatch")
    return {
        "n_train": int(eid_tr.size),
        "n_val": int(eid_va.size),
        "positives_train": int((y_tr == 1).sum()),
        "positives_val": int((y_va == 1).sum()),
        "prevalence_val": float(y_va.mean()),
        "train_id_hash": _sha_sorted_ids(eid_tr),
        "val_id_hash": _sha_sorted_ids(eid_va),
    }


def _logits_from_model(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    outs: List[np.ndarray] = []
    with torch.no_grad():
        xt = torch.from_numpy(x.astype(np.float32))
        for start in range(0, x.shape[0], MLP_BS):
            outs.append(
                model(xt[start : start + MLP_BS].to(device)).detach().cpu().numpy()
            )
    return np.concatenate(outs, axis=0).astype(np.float64)


def _pack_metrics(
    y: np.ndarray,
    logit: np.ndarray,
    w0: float,
    w1: float,
    *,
    classifier_meaning: str,
    best_probe_epoch: Optional[int],
) -> Dict[str, Any]:
    p = 1.0 / (1.0 + np.exp(-np.clip(logit, -50, 50)))
    thr = _tune_thr(y, p)
    m05 = _metrics_at(y, p, 0.5)
    mthr = _metrics_at(y, p, thr)
    return {
        "classifier_meaning": classifier_meaning,
        "best_probe_epoch": best_probe_epoch,
        "validation_auprc": m05["auprc"],
        "validation_auroc": m05["auroc"],
        "f1_at_0.5": m05["f1"],
        "precision_at_0.5": m05["precision"],
        "recall_at_0.5": m05["recall"],
        "positive_prediction_rate_at_0.5": m05["positive_prediction_rate"],
        "f1_at_val_thr": mthr["f1"],
        "precision_at_val_thr": mthr["precision"],
        "recall_at_val_thr": mthr["recall"],
        "positive_prediction_rate_at_val_thr": mthr["positive_prediction_rate"],
        "val_selected_threshold": thr,
        "f1_at_val_thr_optimistic_diagnostic": True,
        "native_matched_weighted_val_ce": matched_weighted_ce_numpy(logit, y, [w0, w1]),
        "common_unweighted_val_ce": unweighted_binary_ce_numpy(logit, y),
        "n": int(y.shape[0]),
        "positives": int((y.astype(np.int64) == 1).sum()),
        "prevalence": float(y.astype(np.float64).mean()),
    }


def fit_weighted_probe(
    mat_tr: np.ndarray,
    y_tr: np.ndarray,
    mat_va: np.ndarray,
    y_va: np.ndarray,
    w0: float,
    w1: float,
    device: torch.device,
) -> Dict[str, Any]:
    scaler = StandardScaler()
    tr = scaler.fit_transform(mat_tr).astype(np.float32)
    va = scaler.transform(mat_va).astype(np.float32)
    torch.manual_seed(MLP_SEED)
    np.random.seed(MLP_SEED)
    model = PaperStyleMLP(tr.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=MLP_LR)
    x_t = torch.from_numpy(tr)
    y_t = torch.from_numpy(y_tr.astype(np.int64))
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
            loss = matched_weighted_ce_from_one_logit(
                logits, y_t[idx].to(device), [w0, w1]
            )
            loss.backward()
            opt.step()

        tr_log = _logits_from_model(model, tr, device)
        va_log = _logits_from_model(model, va, device)
        pva = 1.0 / (1.0 + np.exp(-np.clip(va_log, -50, 50)))
        auprc = float(average_precision_score(y_va, pva))
        history.append(
            {
                "epoch": ep + 1,
                "train_matched_weighted_ce": matched_weighted_ce_numpy(tr_log, y_tr, [w0, w1]),
                "val_matched_weighted_ce": matched_weighted_ce_numpy(va_log, y_va, [w0, w1]),
                "train_unweighted_ce": unweighted_binary_ce_numpy(tr_log, y_tr),
                "val_unweighted_ce": unweighted_binary_ce_numpy(va_log, y_va),
                "val_auprc": auprc,
            }
        )
        if auprc > best_auprc + 1e-12:
            best_auprc = auprc
            best_ep = ep + 1
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

    assert best_state is not None
    final_va = _logits_from_model(model, va, device)
    final_tr = _logits_from_model(model, tr, device)
    model.load_state_dict(best_state)
    model.to(device)
    sel_va = _logits_from_model(model, va, device)
    # Also confirm proba helper path matches for selected
    p_sel = _predict_proba(model, va, batch_size=MLP_BS, device=device)

    return {
        "input_dim": int(tr.shape[1]),
        "best_probe_epoch": int(best_ep),
        "epoch_history": history,
        "final_logits_val": final_va,
        "selected_logits_val": sel_va,
        "final_logits_train": final_tr,
        "selected_proba_val": p_sel.astype(np.float64),
        "scaler_mean": scaler.mean_.astype(np.float64),
        "scaler_scale": scaler.scale_.astype(np.float64),
    }


def _original_probe_block(path: Optional[Path], protocol: str) -> Optional[Dict[str, Any]]:
    if path is None or not path.is_file():
        return None
    cell = json.loads(path.read_text())
    if protocol == "R198_only":
        if int((cell.get("primary") or {}).get("input_dim", -1)) == 198:
            return cell["primary"]
        return cell.get("diagnostic") or cell.get("primary")
    p = cell["primary"]
    if int(p.get("input_dim", -1)) != R198_XTF_DIM:
        raise ValueError(f"{path}: expected primary dim {R198_XTF_DIM}")
    return p


def _align_common(
    probe_eid: np.ndarray,
    probe_y: np.ndarray,
    probe_logit: np.ndarray,
    sup_eid: np.ndarray,
    sup_y: np.ndarray,
    sup_logits2: np.ndarray,
) -> Dict[str, Any]:
    # Deterministic order: sorted common EdgeIDs
    set_p = set(probe_eid.tolist())
    set_s = set(sup_eid.tolist())
    common = np.array(sorted(set_p & set_s), dtype=np.int64)
    lost_p = len(set_p) - common.size
    lost_s = len(set_s) - common.size
    map_p = {int(e): i for i, e in enumerate(probe_eid.tolist())}
    map_s = {int(e): i for i, e in enumerate(sup_eid.tolist())}
    idx_p = np.array([map_p[int(e)] for e in common], dtype=np.int64)
    idx_s = np.array([map_s[int(e)] for e in common], dtype=np.int64)
    y_p = probe_y[idx_p].astype(np.int64)
    y_s = sup_y[idx_s].astype(np.int64)
    if not np.array_equal(y_p, y_s):
        raise RuntimeError("label mismatch on common EdgeID intersection")
    return {
        "edge_id": common,
        "y": y_p,
        "probe_logit": probe_logit[idx_p],
        "sup_logits2": sup_logits2[idx_s],
        "n": int(common.size),
        "positives": int((y_p == 1).sum()),
        "id_hash": _sha_sorted_ids(common),
        "coverage_lost_probe": int(lost_p),
        "coverage_lost_supervised": int(lost_s),
        "n_probe_source": int(probe_eid.size),
        "n_supervised_source": int(sup_eid.size),
    }


def preflight() -> Dict[str, Any]:
    issues: List[str] = []
    if OUT_ROOT.exists():
        # Allow empty/partial new namespace; forbid overwriting completed prior package
        if (OUT_ROOT / "manifest.json").is_file():
            issues.append(f"output namespace already has manifest: {OUT_ROOT}")
    if TOP_JSON.exists():
        issues.append(f"top-level JSON already exists: {TOP_JSON}")
    if not SUP_CKPT_BEST.is_file():
        issues.append(f"missing supervised ckpt: {SUP_CKPT_BEST}")
    for name in (
        "supervised_final_epoch_50_val.npz",
        "supervised_best_validation_f1_val.npz",
    ):
        if not (SUP_PRED_DIR / name).is_file():
            issues.append(f"missing supervised preds: {SUP_PRED_DIR / name}")
    for c in CELLS:
        emb = Path(c["emb_dir"])
        for split in ("train.npz", "val.npz"):
            if not (emb / split).is_file():
                issues.append(f"missing {emb / split}")
        if (emb / "test.npz").is_file():
            issues.append(f"test.npz present (forbidden): {emb}")
        if c["feature_protocol"] == "R198_X_TF":
            if not (TF_CACHE / "features.npy").is_file():
                issues.append(f"missing TF cache: {TF_CACHE / 'features.npy'}")
        oj = c.get("original_cell_json")
        if oj is not None and not Path(oj).is_file():
            issues.append(f"missing original cell json: {oj}")

    # Quick weight load
    w0 = w1 = None
    if SUP_CKPT_BEST.is_file():
        w0, w1, src = load_class_weights_from_checkpoint(str(SUP_CKPT_BEST))
    else:
        src = None

    report = {
        "ok": len(issues) == 0,
        "issues": issues,
        "n_cells": len(CELLS),
        "class_weights": {"w0": w0, "w1": w1, "source": src},
        "out_root": str(OUT_ROOT),
        "test_path_accessed": False,
        "encoder_retrain": False,
        "overwrite_existing_artifacts": False,
        "slurm_template": {
            "partition": "mit_preemptable",
            "account": "mit_general",
            "qos": "normal",
            "mem": "128G",
            "cpus": 8,
            "gres": "gpu:1",
            "time": "06:00:00",
            "estimate": (
                "6 PaperStyleMLP probes (~5–15 min each on existing embeddings) "
                "+ X+TF feature build once; expect <3h wall, 128G mem matches prior probe jobs"
            ),
        },
    }
    return report


def run_all(device: torch.device) -> Dict[str, Any]:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "cells").mkdir(exist_ok=True)
    (OUT_ROOT / "predictions").mkdir(exist_ok=True)
    (OUT_ROOT / "tables").mkdir(exist_ok=True)

    w0, w1, w_src = load_class_weights_from_checkpoint(str(SUP_CKPT_BEST))
    weight_info = class_weight_summary(w0, w1, w_src)

    x_raw = tf = None
    xtf_manifest = None
    if any(c["feature_protocol"] == "R198_X_TF" for c in CELLS):
        logging.info("Loading edge X + temporal-flow for R198_X_TF cells")
        x_raw, tf, xtf_manifest = _load_x_tf()
        if xtf_manifest["total_dim"] != R198_XTF_DIM:
            raise RuntimeError(
                f"feature manifest total_dim={xtf_manifest['total_dim']} != {R198_XTF_DIM}"
            )

    sup_final = np.load(SUP_PRED_DIR / "supervised_final_epoch_50_val.npz")
    sup_best = np.load(SUP_PRED_DIR / "supervised_best_validation_f1_val.npz")

    cell_results: List[Dict[str, Any]] = []
    table_a: List[Dict[str, Any]] = []
    table_b: List[Dict[str, Any]] = []
    table_c: List[Dict[str, Any]] = []

    # Supervised rows on their own full cohorts, then also on common intersections later
    for label, pack, meaning in (
        (
            "supervised_final_epoch_50",
            sup_final,
            "final_supervised_epoch_50",
        ),
        (
            "supervised_best_validation_f1_epoch_43",
            sup_best,
            "best_validation_supervised_epoch_43",
        ),
    ):
        y = pack["y"].astype(np.int64)
        logits2 = pack["logits"].astype(np.float64)
        eid = pack["edge_id"].astype(np.int64)
        p = torch.softmax(torch.as_tensor(logits2), dim=-1)[:, 1].numpy()
        table_c.append(
            {
                "method": "supervised_MultiGIN",
                "model_checkpoint_meaning": meaning,
                "feature_protocol": "supervised_raw_edge_features",
                "ssl_epoch": "",
                "common_unweighted_val_ce": unweighted_binary_ce_from_proba(y, p),
                "common_supervised_weighted_val_ce": supervised_weighted_ce_two_logit(
                    y, logits2, [w0, w1]
                ),
                "n": int(y.size),
                "positives": int((y == 1).sum()),
                "id_hash": _sha_sorted_ids(eid),
                "note": "full supervised validation cohort from job 19458946 (pre-alignment)",
            }
        )

    for c in CELLS:
        logging.info("=== cell %s ===", c["cell_id"])
        emb = Path(c["emb_dir"])
        if (emb / "test.npz").is_file():
            raise RuntimeError(f"test.npz present: {emb}")
        tr_npz = np.load(emb / "train.npz")
        va_npz = np.load(emb / "val.npz")
        tr = {
            "Z": tr_npz["Z"].astype(np.float32),
            "y": tr_npz["y"].astype(np.int64),
            "edge_id": tr_npz["edge_id"].astype(np.int64),
        }
        va = {
            "Z": va_npz["Z"].astype(np.float32),
            "y": va_npz["y"].astype(np.int64),
            "edge_id": va_npz["edge_id"].astype(np.int64),
        }
        cohort = _assert_split(tr, va)
        if tr["Z"].shape[1] != R198_DIM or va["Z"].shape[1] != R198_DIM:
            raise RuntimeError(f"Z dim != 198 in {emb}")

        mat_tr = _stack_features(
            tr["Z"], tr["edge_id"], c["feature_protocol"], x_raw, tf
        )
        mat_va = _stack_features(
            va["Z"], va["edge_id"], c["feature_protocol"], x_raw, tf
        )
        expected_dim = R198_DIM if c["feature_protocol"] == "R198_only" else R198_XTF_DIM
        if mat_tr.shape[1] != expected_dim:
            raise RuntimeError(f"feature dim {mat_tr.shape[1]} != {expected_dim}")

        fit = fit_weighted_probe(
            mat_tr, tr["y"], mat_va, va["y"], w0, w1, device
        )
        final_pack = _pack_metrics(
            va["y"],
            fit["final_logits_val"],
            w0,
            w1,
            classifier_meaning="final_probe_epoch_20",
            best_probe_epoch=None,
        )
        sel_pack = _pack_metrics(
            va["y"],
            fit["selected_logits_val"],
            w0,
            w1,
            classifier_meaning="validation_selected_best_auprc",
            best_probe_epoch=fit["best_probe_epoch"],
        )
        p_final = 1.0 / (1.0 + np.exp(-np.clip(fit["final_logits_val"], -50, 50)))
        p_sel = 1.0 / (1.0 + np.exp(-np.clip(fit["selected_logits_val"], -50, 50)))
        if abs(float(average_precision_score(va["y"], p_sel)) - sel_pack["validation_auprc"]) > 1e-10:
            raise RuntimeError("selected AUPRC inconsistency")
        pred_final = OUT_ROOT / "predictions" / f"{c['cell_id']}_final_val.npz"
        pred_sel = OUT_ROOT / "predictions" / f"{c['cell_id']}_selected_val.npz"
        np.savez_compressed(
            pred_final,
            edge_id=va["edge_id"].astype(np.int64),
            y=va["y"].astype(np.int64),
            logit=fit["final_logits_val"].astype(np.float64),
            probability=p_final.astype(np.float64),
            classifier_epoch=np.array([20], dtype=np.int64),
            classifier_meaning=np.array(["final_probe_epoch_20"]),
        )
        np.savez_compressed(
            pred_sel,
            edge_id=va["edge_id"].astype(np.int64),
            y=va["y"].astype(np.int64),
            logit=fit["selected_logits_val"].astype(np.float64),
            probability=p_sel.astype(np.float64),
            classifier_epoch=np.array([fit["best_probe_epoch"]], dtype=np.int64),
            classifier_meaning=np.array(["validation_selected_best_auprc"]),
        )

        # Align each probe classifier to each supervised checkpoint (common intersection)
        align_rows = []
        for meaning, logit in (
            ("final_probe_epoch_20", fit["final_logits_val"]),
            ("validation_selected_best_auprc", fit["selected_logits_val"]),
        ):
            # Prefer intersection with supervised final; report hash; also compute vs best
            for sup_name, sup_pack, sup_meaning in (
                ("final_epoch_50", sup_final, "aligned_to_supervised_final_epoch_50"),
                (
                    "best_val_f1_epoch_43",
                    sup_best,
                    "aligned_to_supervised_best_validation_f1_epoch_43",
                ),
            ):
                al = _align_common(
                    va["edge_id"],
                    va["y"],
                    logit,
                    sup_pack["edge_id"].astype(np.int64),
                    sup_pack["y"].astype(np.int64),
                    sup_pack["logits"].astype(np.float64),
                )
                p_al = 1.0 / (1.0 + np.exp(-np.clip(al["probe_logit"], -50, 50)))
                row = {
                    "method": c["method"],
                    "model_checkpoint_meaning": f"{c['cell_id']}::{meaning}::{sup_meaning}",
                    "feature_protocol": c["feature_protocol"],
                    "ssl_epoch": c["ssl_epoch"],
                    "probe_classifier": meaning,
                    "supervised_reference": sup_name,
                    "common_unweighted_val_ce": unweighted_binary_ce_numpy(
                        al["probe_logit"], al["y"]
                    ),
                    "common_supervised_weighted_val_ce": matched_weighted_ce_numpy(
                        al["probe_logit"], al["y"], [w0, w1]
                    ),
                    "n": al["n"],
                    "positives": al["positives"],
                    "id_hash": al["id_hash"],
                    "coverage_lost_probe": al["coverage_lost_probe"],
                    "coverage_lost_supervised": al["coverage_lost_supervised"],
                    "n_probe_source": al["n_probe_source"],
                    "n_supervised_source": al["n_supervised_source"],
                    "common_unweighted_val_ce_from_proba": unweighted_binary_ce_from_proba(
                        al["y"], p_al
                    ),
                }
                align_rows.append(row)
                # Primary Table C rows: final probe vs supervised-final intersection
                if meaning == "final_probe_epoch_20" and sup_name == "final_epoch_50":
                    table_c.append(
                        {
                            "method": c["method"],
                            "model_checkpoint_meaning": f"{c['cell_id']}_final_probe_epoch_20",
                            "feature_protocol": c["feature_protocol"],
                            "ssl_epoch": c["ssl_epoch"],
                            "common_unweighted_val_ce": row["common_unweighted_val_ce"],
                            "common_supervised_weighted_val_ce": row[
                                "common_supervised_weighted_val_ce"
                            ],
                            "n": row["n"],
                            "positives": row["positives"],
                            "id_hash": row["id_hash"],
                            "coverage_lost_probe": row["coverage_lost_probe"],
                            "coverage_lost_supervised": row["coverage_lost_supervised"],
                        }
                    )
                if meaning == "validation_selected_best_auprc" and sup_name == "final_epoch_50":
                    table_c.append(
                        {
                            "method": c["method"],
                            "model_checkpoint_meaning": (
                                f"{c['cell_id']}_validation_selected_best_auprc"
                            ),
                            "feature_protocol": c["feature_protocol"],
                            "ssl_epoch": c["ssl_epoch"],
                            "common_unweighted_val_ce": row["common_unweighted_val_ce"],
                            "common_supervised_weighted_val_ce": row[
                                "common_supervised_weighted_val_ce"
                            ],
                            "n": row["n"],
                            "positives": row["positives"],
                            "id_hash": row["id_hash"],
                            "coverage_lost_probe": row["coverage_lost_probe"],
                            "coverage_lost_supervised": row["coverage_lost_supervised"],
                        }
                    )

        # Also add supervised metrics on the same common intersection used by probes
        # (recomputed once from first alignment of this cell's final)
        al0 = _align_common(
            va["edge_id"],
            va["y"],
            fit["final_logits_val"],
            sup_final["edge_id"].astype(np.int64),
            sup_final["y"].astype(np.int64),
            sup_final["logits"].astype(np.float64),
        )

        orig = _original_probe_block(
            Path(c["original_cell_json"]) if c["original_cell_json"] else None,
            c["feature_protocol"],
        )
        orig_auprc = orig_f1 = None
        if orig is not None:
            orig_auprc = float(orig["validation_auprc"])
            orig_f1 = float((orig.get("validation_metrics_at_0.5") or {}).get("f1", float("nan")))

        for pack, tag in ((final_pack, "final"), (sel_pack, "selected")):
            table_a.append(
                {
                    "method": c["method"],
                    "feature_protocol": c["feature_protocol"],
                    "ssl_epoch": c["ssl_epoch"],
                    "probe_objective": "matched_weighted_one_logit_CE",
                    "final_or_selected": pack["classifier_meaning"],
                    "validation_auprc": pack["validation_auprc"],
                    "f1_at_0.5": pack["f1_at_0.5"],
                    "f1_at_val_thr": pack["f1_at_val_thr"],
                    "weighted_val_ce": pack["native_matched_weighted_val_ce"],
                    "unweighted_val_ce": pack["common_unweighted_val_ce"],
                    "validation_auroc": pack["validation_auroc"],
                    "precision_at_0.5": pack["precision_at_0.5"],
                    "recall_at_0.5": pack["recall_at_0.5"],
                    "positive_prediction_rate_at_0.5": pack[
                        "positive_prediction_rate_at_0.5"
                    ],
                    "n": pack["n"],
                    "positives": pack["positives"],
                    "prevalence": pack["prevalence"],
                    "val_id_hash": cohort["val_id_hash"],
                }
            )

        table_b.append(
            {
                "method": c["method"],
                "feature_protocol": c["feature_protocol"],
                "ssl_epoch": c["ssl_epoch"],
                "original_unweighted_probe_auprc": orig_auprc,
                "original_unweighted_probe_f1_at_0.5": orig_f1,
                "original_source": (
                    str(c["original_cell_json"]) if c["original_cell_json"] else "unavailable_no_prior_xtf_probe"
                ),
                "original_label": "reused_existing_artifact",
                "matched_weighted_probe_auprc_selected": sel_pack["validation_auprc"],
                "matched_weighted_probe_f1_at_0.5_selected": sel_pack["f1_at_0.5"],
                "delta_auprc_selected_minus_original": (
                    None
                    if orig_auprc is None
                    else float(sel_pack["validation_auprc"] - orig_auprc)
                ),
                "delta_f1_selected_minus_original": (
                    None if orig_f1 is None else float(sel_pack["f1_at_0.5"] - orig_f1)
                ),
            }
        )

        cell_out = {
            "cell_id": c["cell_id"],
            "method": c["method"],
            "probe_feature_protocol": c["feature_protocol"],
            "ssl_lr": c["ssl_lr"],
            "ssl_epoch": c["ssl_epoch"],
            "emb_dir": str(emb),
            "cohort": cohort,
            "class_weights": weight_info,
            "probe_protocol": {
                "learner": "PaperStyleMLP",
                "mlp_epochs": MLP_EPOCHS,
                "mlp_lr": MLP_LR,
                "mlp_batch_size": MLP_BS,
                "mlp_seed": MLP_SEED,
                "loss": "matched_weighted_one_logit_CE",
                "selection_within_probe": "best_val_auprc",
                "encoder_frozen": True,
                "test_evaluated": False,
            },
            "feature_dim": int(mat_tr.shape[1]),
            "xtf_manifest": xtf_manifest if c["feature_protocol"] == "R198_X_TF" else None,
            "final": final_pack,
            "selected": sel_pack,
            "epoch_history": fit["epoch_history"],
            "predictions": {
                "final_val": str(pred_final),
                "selected_val": str(pred_sel),
            },
            "common_alignment_rows": align_rows,
            "common_intersection_with_supervised_final": {
                "n": al0["n"],
                "positives": al0["positives"],
                "id_hash": al0["id_hash"],
                "coverage_lost_probe": al0["coverage_lost_probe"],
                "coverage_lost_supervised": al0["coverage_lost_supervised"],
            },
            "original_unweighted_probe": {
                "path": str(c["original_cell_json"]) if c["original_cell_json"] else None,
                "validation_auprc": orig_auprc,
                "f1_at_0.5": orig_f1,
            },
        }
        cell_path = OUT_ROOT / "cells" / f"{c['cell_id']}.json"
        cell_path.write_text(json.dumps(cell_out, indent=2) + "\n")
        cell_results.append(cell_out)
        logging.info(
            "%s selected AUPRC=%.6f F1@0.5=%.6f weightedCE=%.6f",
            c["cell_id"],
            sel_pack["validation_auprc"],
            sel_pack["f1_at_0.5"],
            sel_pack["native_matched_weighted_val_ce"],
        )

    # Recompute supervised CE on the common intersection shared with probes
    # Use first probe cell's val EdgeIDs (all R198 embeds share same val EdgeIDs).
    probe_eid0 = np.load(Path(CELLS[0]["emb_dir"]) / "val.npz")["edge_id"].astype(np.int64)
    probe_y0 = np.load(Path(CELLS[0]["emb_dir"]) / "val.npz")["y"].astype(np.int64)
    # dummy logits for alignment helper
    dummy = np.zeros(probe_eid0.shape[0], dtype=np.float64)
    for sup_pack, meaning in (
        (sup_final, "final_supervised_epoch_50_on_common_intersection"),
        (sup_best, "best_validation_supervised_epoch_43_on_common_intersection"),
    ):
        al = _align_common(
            probe_eid0,
            probe_y0,
            dummy,
            sup_pack["edge_id"].astype(np.int64),
            sup_pack["y"].astype(np.int64),
            sup_pack["logits"].astype(np.float64),
        )
        p = torch.softmax(torch.as_tensor(al["sup_logits2"]), dim=-1)[:, 1].numpy()
        table_c.append(
            {
                "method": "supervised_MultiGIN",
                "model_checkpoint_meaning": meaning,
                "feature_protocol": "supervised_raw_edge_features",
                "ssl_epoch": "",
                "common_unweighted_val_ce": unweighted_binary_ce_from_proba(al["y"], p),
                "common_supervised_weighted_val_ce": supervised_weighted_ce_two_logit(
                    al["y"], al["sup_logits2"], [w0, w1]
                ),
                "n": al["n"],
                "positives": al["positives"],
                "id_hash": al["id_hash"],
                "coverage_lost_probe": al["coverage_lost_probe"],
                "coverage_lost_supervised": al["coverage_lost_supervised"],
            }
        )

    _write_csv(OUT_ROOT / "tables" / "table_a_matched_probe_performance.csv", table_a)
    _write_csv(OUT_ROOT / "tables" / "table_b_loss_effect.csv", table_b)
    _write_csv(OUT_ROOT / "tables" / "table_c_common_cohort_ce.csv", table_c)

    answers = _answers(cell_results, table_b, table_c, weight_info)
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_purpose": (
            "Secondary sensitivity: PaperStyleMLP probe trained with "
            "supervised-loss-matched weighted one-logit CE; established "
            "unweighted probes preserved."
        ),
        "guardrails": {
            "encoder_retrained": False,
            "test_accessed": False,
            "existing_artifacts_overwritten": False,
            "replacement_for_established_probe": False,
            "primary_ranking_metric": "validation_auprc",
            "f1_at_val_thr": "optimistic_diagnostic",
            "supervised_is_end_to_end": True,
            "probes_use_frozen_embeddings": True,
        },
        "class_weights": weight_info,
        "xtf_manifest": xtf_manifest,
        "cells": cell_results,
        "table_a": table_a,
        "table_b": table_b,
        "table_c": table_c,
        "answers": answers,
        "job_id": os.environ.get("SLURM_JOB_ID"),
    }
    TOP_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    (OUT_ROOT / "aml_loss_matched_weighted_probe.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    manifest = {
        "n_cells": len(cell_results),
        "out_root": str(OUT_ROOT),
        "top_json": str(TOP_JSON),
        "notes": str(NOTES),
        "class_weights": weight_info,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "test_accessed": False,
        "encoder_retrained": False,
        "overwrite": False,
    }
    (OUT_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    _write_notes(payload)
    return payload


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    keys: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _answers(
    cells: List[Dict[str, Any]],
    table_b: List[Dict[str, Any]],
    table_c: List[Dict[str, Any]],
    weight_info: Dict[str, Any],
) -> Dict[str, Any]:
    # 1 loss match — affirmed by tests + runtime formula
    # 2 deltas
    deltas = [
        {
            "method": r["method"],
            "feature_protocol": r["feature_protocol"],
            "ssl_epoch": r["ssl_epoch"],
            "delta_auprc": r["delta_auprc_selected_minus_original"],
            "delta_f1": r["delta_f1_selected_minus_original"],
        }
        for r in table_b
    ]
    r198 = [
        c
        for c in cells
        if c["probe_feature_protocol"] == "R198_only"
    ]
    best_r198 = max(r198, key=lambda c: c["selected"]["validation_auprc"])
    xtf = [c for c in cells if c["probe_feature_protocol"] == "R198_X_TF"]
    adapt_xtf = next(c for c in xtf if c["method"] == "TFMOE_adaptive")
    expert_xtf = next(c for c in xtf if c["method"] == "EXPERT_ONLY")

    common_ce = [
        r
        for r in table_c
        if "on_common_intersection" in str(r.get("model_checkpoint_meaning", ""))
        or (
            r.get("method") != "supervised_MultiGIN"
            and "final_probe_epoch_20" in str(r.get("model_checkpoint_meaning", ""))
        )
    ]
    return {
        "exact_weighted_one_logit_matches_two_logit_ce": True,
        "class_weights_loaded": weight_info,
        "weighted_probe_changed_auprc_or_f1": deltas,
        "strongest_r198_only_selected": {
            "method": best_r198["method"],
            "cell_id": best_r198["cell_id"],
            "ssl_epoch": best_r198["ssl_epoch"],
            "validation_auprc": best_r198["selected"]["validation_auprc"],
            "f1_at_0.5": best_r198["selected"]["f1_at_0.5"],
            "expert_only_remains_strongest": best_r198["method"] == "EXPERT_ONLY",
        },
        "expert_only_xtf_vs_adaptive_xtf": {
            "expert_only_auprc": expert_xtf["selected"]["validation_auprc"],
            "adaptive_auprc": adapt_xtf["selected"]["validation_auprc"],
            "delta_expert_minus_adaptive": float(
                expert_xtf["selected"]["validation_auprc"]
                - adapt_xtf["selected"]["validation_auprc"]
            ),
            "expert_improves_over_adaptive": bool(
                expert_xtf["selected"]["validation_auprc"]
                > adapt_xtf["selected"]["validation_auprc"]
            ),
        },
        "common_cohort_ce_rows": common_ce,
        "encoders_retrained": False,
        "test_accessed": False,
        "jobs_submitted_max_one": True,
        "existing_artifacts_overwritten": False,
    }


def _write_notes(payload: Dict[str, Any]) -> None:
    a = payload["answers"]
    w = payload["class_weights"]
    lines = [
        "# AML loss-matched weighted PaperStyleMLP probe sensitivity",
        "",
        "## Purpose",
        "",
        "Secondary validation-only sensitivity: train PaperStyleMLP with a one-logit",
        "weighted CE that is mathematically equivalent to the supervised Multi-GIN",
        "`CrossEntropyLoss(weight=[w0,w1], reduction='mean')` on logits `[0, z]`.",
        "",
        "This does **not** retrain or finetune any encoder.",
        "This is a downstream-loss sensitivity, **not** a replacement for the established",
        "unweighted PaperStyleMLP probe.",
        "Supervised Multi-GIN remains end-to-end trained; these probes use frozen embeddings.",
        "A lower unweighted CE on an extremely imbalanced cohort does not by itself imply",
        "better minority detection. **AUPRC remains the primary ranking metric.**",
        "F1 at a validation-selected threshold is optimistic/diagnostic.",
        "**No test claim** is made. Probe protocol must not be selected by test performance.",
        "",
        "## Class weights (from supervised checkpoint config)",
        "",
        f"- Source: `{SUP_CKPT_BEST}`",
        f"- w0={w['w0']}",
        f"- w1={w['w1']}",
        f"- Formula: `{w['formula']}`",
        "",
        "## Protocol",
        "",
        "- PaperStyleMLP, 20 epochs, lr=1e-3, bs=8192, seed=2",
        "- Train-only StandardScaler",
        "- Selection: best validation AUPRC (selected); final = epoch 20",
        "- Existing full-subgraph embeddings only; no extract/train",
        "- Standard Slurm: partition=mit_preemptable, account=mit_general, qos=normal",
        "",
        "## Tables",
        "",
        f"- `{OUT_ROOT / 'tables/table_a_matched_probe_performance.csv'}`",
        f"- `{OUT_ROOT / 'tables/table_b_loss_effect.csv'}`",
        f"- `{OUT_ROOT / 'tables/table_c_common_cohort_ce.csv'}`",
        f"- Full JSON: `{TOP_JSON}`",
        "",
        "## Direct answers",
        "",
        f"1. Exact weighted one-logit loss match two-logit CE? **{a['exact_weighted_one_logit_matches_two_logit_ce']}**",
        f"2. Weighted probe changed AUPRC/F1? See deltas: `{json.dumps(a['weighted_probe_changed_auprc_or_f1'], indent=2)}`",
        f"3. Expert-only strongest R198-only? **{a['strongest_r198_only_selected']['expert_only_remains_strongest']}** "
        f"({a['strongest_r198_only_selected']['method']} AUPRC="
        f"{a['strongest_r198_only_selected']['validation_auprc']:.6f})",
        f"4. Expert-only R198+X+TF improve over adaptive? "
        f"**{a['expert_only_xtf_vs_adaptive_xtf']['expert_improves_over_adaptive']}** "
        f"(ΔAUPRC={a['expert_only_xtf_vs_adaptive_xtf']['delta_expert_minus_adaptive']:.6f})",
        "5. Exact common-cohort CE: see Table C / `answers.common_cohort_ce_rows` in JSON.",
        "6. Encoders retrained? **no**",
        "7. Test accessed? **no**",
        "8. Jobs submitted? **one** (this sensitivity job)",
        "9. Existing artifacts overwritten? **no**",
        "",
    ]
    # Table A markdown
    lines += ["## Table A — matched probe performance", ""]
    lines.append(
        "| Method | Feature protocol | SSL epoch | Probe objective | Final/selected | Val AUPRC | F1@0.5 | F1@val-thr | Weighted val CE | Unweighted val CE |"
    )
    lines.append("|---|---|---:|---|---|---:|---:|---:|---:|---:|")
    for r in payload["table_a"]:
        lines.append(
            f"| {r['method']} | {r['feature_protocol']} | {r['ssl_epoch']} | "
            f"{r['probe_objective']} | {r['final_or_selected']} | "
            f"{r['validation_auprc']:.6f} | {r['f1_at_0.5']:.6f} | {r['f1_at_val_thr']:.6f} | "
            f"{r['weighted_val_ce']:.6f} | {r['unweighted_val_ce']:.6f} |"
        )
    lines += ["", "## Table B — effect of changing the probe loss", ""]
    lines.append(
        "| Method | Feature protocol | SSL epoch | Original unweighted AUPRC/F1 | Matched weighted AUPRC/F1 | ΔAUPRC | ΔF1 |"
    )
    lines.append("|---|---|---:|---|---|---:|---:|")
    for r in payload["table_b"]:
        o = (
            "N/A"
            if r["original_unweighted_probe_auprc"] is None
            else f"{r['original_unweighted_probe_auprc']:.6f}/{r['original_unweighted_probe_f1_at_0.5']:.6f}"
        )
        m = f"{r['matched_weighted_probe_auprc_selected']:.6f}/{r['matched_weighted_probe_f1_at_0.5_selected']:.6f}"
        da = "N/A" if r["delta_auprc_selected_minus_original"] is None else f"{r['delta_auprc_selected_minus_original']:+.6f}"
        df = "N/A" if r["delta_f1_selected_minus_original"] is None else f"{r['delta_f1_selected_minus_original']:+.6f}"
        lines.append(
            f"| {r['method']} | {r['feature_protocol']} | {r['ssl_epoch']} | {o} (reused) | {m} | {da} | {df} |"
        )
    lines += ["", "## Table C — exact common-cohort CE", ""]
    lines.append(
        "| Method | Model/checkpoint meaning | Common unweighted val CE | Common supervised-weighted val CE | n | positives | ID hash |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---|")
    for r in payload["table_c"]:
        lines.append(
            f"| {r['method']} | {r['model_checkpoint_meaning']} | "
            f"{r['common_unweighted_val_ce']:.6f} | {r['common_supervised_weighted_val_ce']:.6f} | "
            f"{r['n']} | {r['positives']} | `{str(r['id_hash'])[:16]}` |"
        )
    NOTES.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.preflight or not args.run:
        report = preflight()
        print(json.dumps(report, indent=2))
        if args.preflight and not args.run:
            return 0 if report["ok"] else 2

    if args.run:
        pf = preflight()
        if not pf["ok"]:
            logging.error("preflight failed: %s", pf["issues"])
            return 2
        device = torch.device(args.device)
        payload = run_all(device)
        print(
            json.dumps(
                {
                    "status": "ok",
                    "n_cells": len(payload["cells"]),
                    "answers": payload["answers"],
                    "out": str(OUT_ROOT),
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
