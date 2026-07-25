#!/usr/bin/env python3
"""One locked test evaluation of partial-FT best ckpt. No retraining / no MLP refit.

Secondary result: SSL-pretrained D+ with supervised partial fine-tuning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import importlib.util

from data_loading import get_data
from dplus_partial_finetune import (
    DPLUS_CKPT_SHA256,
    GradPreEmbeddingCapture,
    STACK_DIM,
    assert_dplus_checkpoint,
    build_graph_args,
    forward_pre3h_from_loader_batch,
    load_dplus_hetero_encoder,
    load_finetune_checkpoint,
    pack_online_features,
    set_encoder_modes,
)
from gcpal_txn_node.eval_mlp import PaperStyleMLP
from ranking_metrics import alert_budget_metrics
from train_util import AddEgoIds, get_loaders
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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def metrics_block(y, proba, thr):
    from sklearn.metrics import f1_score, precision_score, recall_score

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


@torch.inference_mode()
def predict_split(encoder, clf, loader, split_inds, capture, x_raw, tf_feat, scaler, device):
    set_encoder_modes(encoder, "warmup")  # full eval modes
    encoder.eval()
    clf.eval()
    probs, ys, eids_all = [], [], []
    for batch in loader:
        h, yb, eids = forward_pre3h_from_loader_batch(
            encoder, batch, split_inds, loader.data, capture, device
        )
        feats = pack_online_features(h, eids, x_raw, tf_feat, scaler, device)
        logits = clf(feats)
        probs.append(torch.sigmoid(logits).detach().cpu().numpy())
        ys.append(yb.detach().cpu().numpy())
        eids_all.append(eids.detach().cpu().numpy().astype(np.int64))
    proba = np.concatenate(probs)
    y = np.concatenate(ys).astype(np.int64)
    ids = np.concatenate(eids_all)
    # dedupe by first occurrence (stable)
    _, uniq_idx = np.unique(ids, return_index=True)
    uniq_idx = np.sort(uniq_idx)
    return proba[uniq_idx], y[uniq_idx], ids[uniq_idx]


def main() -> None:
    logger_setup()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint",
        default="saved-models/dplus_partial_finetune_hxxtf_seed2/checkpoint_best_val_auprc.tar",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output_json", required=True)
    p.add_argument("--output_md", required=True)
    p.add_argument("--data_config", default="data_config.json")
    args = p.parse_args()
    set_seed(2)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt_path = ROOT / args.checkpoint if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint)
    sha = sha256_file(ckpt_path)
    payload = torch.load(ckpt_path, map_location="cpu")
    early = payload.get("early_stop") or {}
    best_epoch = int(early.get("best_epoch", -1))
    best_auprc = float(early.get("best_auprc", -1))
    if best_epoch != 18:
        raise SystemExit(f"expected best_epoch=18, got {best_epoch}")
    if abs(best_auprc - 0.600) > 0.01:
        raise SystemExit(f"expected best val AUPRC≈0.600, got {best_auprc}")
    if "classifier_state_dict" not in payload or "encoder_state_dict" not in payload:
        raise SystemExit("checkpoint missing encoder/classifier")
    if "scaler_mean" not in payload or "scaler_scale" not in payload:
        raise SystemExit("checkpoint missing scaler")
    protocol = payload.get("protocol") or {}
    if protocol.get("source_checkpoint_sha256") != DPLUS_CKPT_SHA256:
        raise SystemExit("source D+ sha mismatch in FT protocol metadata")

    # Stored validation-selected threshold from history at best epoch
    thr = None
    for row in payload.get("history") or []:
        if int(row.get("epoch", -1)) == best_epoch and "val" in row:
            thr = float(row["val"].get("threshold"))
            break
    if thr is None:
        raise SystemExit("could not recover stored validation threshold from history")

    # Features for packing
    df, df_train, _, _, _, spec = load_dataset_frames("Small-HI", str(ROOT / args.data_config))
    x_raw, _, _, _ = build_full_feature_matrix(
        df, df_train, ("edge_native",), categorical_encoding="one_hot"
    )
    tf_feat = np.load(TF_CACHE / "features.npy").astype(np.float32)
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(payload["scaler_mean"], dtype=np.float64)
    scaler.scale_ = np.asarray(payload["scaler_scale"], dtype=np.float64)
    scaler.n_features_in_ = int(scaler.mean_.shape[0])
    if scaler.mean_.shape[0] != STACK_DIM:
        raise SystemExit(f"scaler dim {scaler.mean_.shape[0]} != {STACK_DIM}")

    with open(ROOT / args.data_config, encoding="utf-8") as f:
        data_config = json.load(f)
    graph_args = build_graph_args(seed=2, loader_num_workers=0)
    tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(graph_args, data_config)
    # Build encoder architecture then overwrite with FT weights (not D+ only)
    encoder, head_spec, pre_dim, _, _ = load_dplus_hetero_encoder(
        tr_data, val_data, te_data, tr_inds, val_inds, te_inds, graph_args, data_config, device
    )
    clf = PaperStyleMLP(STACK_DIM, hidden=128, dropout=0.1).to(device)
    state = load_finetune_checkpoint(ckpt_path, encoder, clf, optimizer=None)
    logging.info("Loaded FT state stage=%s global_epoch=%s", state.stage, state.global_epoch)

    transform = AddEgoIds() if graph_args.ego else None
    _, val_loader, te_loader = get_loaders(
        tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform, graph_args, train_shuffle=False
    )
    capture = GradPreEmbeddingCapture(encoder, pre_dim=pre_dim, emb_dim=128, head_spec=head_spec)

    # Optional val re-score for diagnostics (not for selection)
    val_proba, val_y, val_ids = predict_split(
        encoder, clf, val_loader, val_inds, capture, x_raw, tf_feat, scaler, device
    )
    val_auprc_online = float(average_precision_score(val_y, val_proba))

    te_proba, te_y, te_ids = predict_split(
        encoder, clf, te_loader, te_inds, capture, x_raw, tf_feat, scaler, device
    )
    capture.remove()

    n_dup = int(te_ids.shape[0] - np.unique(te_ids).shape[0])  # after dedupe should be 0
    report = {
        "role": "SECONDARY_ssl_pretrained_dplus_supervised_partial_finetune",
        "description": "SSL-pretrained D+ with supervised partial fine-tuning.",
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": sha,
        "best_epoch": best_epoch,
        "stored_best_val_auprc": best_auprc,
        "stored_best_val_f1": float(early.get("best_f1")),
        "stored_validation_selected_threshold": thr,
        "online_val_auprc_diagnostic": val_auprc_online,
        "online_val_n": int(val_y.shape[0]),
        "protocol": protocol,
        "stage": payload.get("stage"),
        "partial_unfreeze": protocol.get("stage2_trainable_prefixes"),
        "embedding_head_frozen": protocol.get("embedding_head_frozen"),
        "aml_labels_updated_encoder": True,
        "aml_labels_updated_classifier": True,
        "classifier_refit": False,
        "test_used_for_selection": False,
        "test_metrics_threshold_0.5": metrics_block(te_y, te_proba, 0.5),
        "test_metrics_val_threshold": {
            **metrics_block(te_y, te_proba, thr),
            "validation_selected_threshold": thr,
        },
        "test_ids": {
            "n": int(te_ids.shape[0]),
            "n_unique": int(np.unique(te_ids).shape[0]),
            "n_duplicate_after_dedupe": n_dup,
            "edge_id_sum": int(te_ids.sum()),
            "sha256_of_ids_bytes": hashlib.sha256(te_ids.astype(np.int64).tobytes()).hexdigest(),
        },
        "frozen_seed2_reference_18678029": {
            "val_auprc": 0.550,
            "test_auprc": 0.674,
            "test_f1_fixed_0.5": 0.656,
        },
        "comparison_to_frozen_seed2": {
            "delta_stored_val_auprc": best_auprc - 0.550,
            "delta_test_auprc_vs_ref": float(metrics_block(te_y, te_proba, 0.5)["auprc"] - 0.674),
            "delta_test_f1_0.5_vs_ref": float(metrics_block(te_y, te_proba, 0.5)["f1"] - 0.656),
        },
    }
    out_j = Path(args.output_json)
    out_m = Path(args.output_md)
    out_j.parent.mkdir(parents=True, exist_ok=True)
    out_m.parent.mkdir(parents=True, exist_ok=True)
    out_j.write_text(json.dumps(report, indent=2) + "\n")
    t05 = report["test_metrics_threshold_0.5"]
    tv = report["test_metrics_val_threshold"]
    lines = [
        "# Partial fine-tune seed-2 — locked test evaluation (SECONDARY)",
        "",
        report["description"],
        "",
        f"- checkpoint: `{ckpt_path}`",
        f"- sha256: `{sha}`",
        f"- best epoch: **{best_epoch}** (stored val AUPRC **{best_auprc:.6f}**)",
        f"- stored val threshold: **{thr}**",
        f"- online val AUPRC diagnostic: {val_auprc_online:.6f} (neighbor-sampled; not for selection)",
        f"- test AUPRC: **{t05['auprc']:.6f}**",
        f"- test AUROC: **{t05['auroc']:.6f}**",
        f"- test F1@0.5: **{t05['f1']:.6f}**",
        f"- test F1@val-thr: **{tv['f1']:.6f}**",
        f"- P@100/500/1000: {t05['precision_at_100']:.3f} / {t05['precision_at_500']:.3f} / {t05['precision_at_1000']:.3f}",
        "",
        "Classifier was **not** refit. Test was **not** used for selection.",
        "",
    ]
    out_m.write_text("\n".join(lines) + "\n")
    logging.info("Wrote %s", out_j)


if __name__ == "__main__":
    main()
