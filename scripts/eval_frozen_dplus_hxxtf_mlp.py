#!/usr/bin/env python3
"""Frozen D+ pre-3h H+X+TF MLP eval (18678029 recipe). No GNN training.

Self-supervised contrastive encoder evaluated using a supervised downstream
classifier, with the encoder frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

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
DOWNSTREAM_SEED = 2  # locked with job 18678029


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ids_hash(ids: np.ndarray) -> Dict[str, Any]:
    ids = np.asarray(ids).astype(np.int64)
    return {
        "n": int(ids.shape[0]),
        "n_unique": int(np.unique(ids).shape[0]),
        "n_duplicate_rows": int(ids.shape[0] - np.unique(ids).shape[0]),
        "edge_id_sum": int(ids.sum()),
        "edge_id_first": int(ids[0]) if ids.size else None,
        "edge_id_last": int(ids[-1]) if ids.size else None,
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


def train_mlp_18678029(x_tr, y_tr, x_va, y_va, x_te, y_te, device, seed: int = DOWNSTREAM_SEED):
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
    proba_te = _predict_proba(model, x_te, batch_size=MLP_BS, device=device)
    proba_va = _predict_proba(model, x_va, batch_size=MLP_BS, device=device)
    thr = _select_threshold_f1(y_va, proba_va)
    return {
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


def main() -> None:
    logger_setup()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--unique_name", required=True)
    p.add_argument("--checkpoint", required=True, help="Path to contrastive best checkpoint")
    p.add_argument("--encoder_seed", type=int, required=True)
    p.add_argument("--embeddings_dir", default=None, help="Default: embeddings/{unique_name}/pre_embedding_3h")
    p.add_argument("--skip_extract", action="store_true")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output_json", required=True)
    p.add_argument("--output_md", required=True)
    p.add_argument("--data_config", default="data_config.json")
    args = p.parse_args()
    set_seed(DOWNSTREAM_SEED)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise SystemExit(f"missing checkpoint {ckpt_path}")
    ckpt_sha = sha256_file(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt_epoch = int(ckpt.get("epoch", -1))

    emb_dir = Path(args.embeddings_dir) if args.embeddings_dir else (
        ROOT / "embeddings" / args.unique_name / "pre_embedding_3h"
    )

    if not args.skip_extract:
        import subprocess

        cmd = [
            sys.executable,
            str(ROOT / "embedding_extraction.py"),
            "--data", "Small-HI", "--model", "gin", "--tqdm",
            "--batch_size", "8192", "--num_neighs", "100", "100",
            "--loader_num_workers", "0",
            "--seed", str(args.encoder_seed),
            "--reverse_mp", "--ego", "--ports", "--emlps", "--tds",
            "--correct_reverse_edge_features",
            "--unique_name", args.unique_name,
            "--representation_source", "pre_embedding_3h",
            "--testing",
        ]
        logging.info("Extracting: %s", " ".join(cmd))
        subprocess.check_call(cmd, cwd=str(ROOT))

    for sp in ("train", "val", "test"):
        if not (emb_dir / f"{sp}.npz").is_file():
            raise SystemExit(f"missing embedding {emb_dir / f'{sp}.npz'}")

    df, df_train, tr_ids, va_ids, te_ids, spec = load_dataset_frames("Small-HI", str(ROOT / args.data_config))
    y_all = df[spec.label_col].to_numpy().astype(np.int64)
    x_raw, _, _, _ = build_full_feature_matrix(
        df, df_train, ("edge_native",), categorical_encoding="one_hot"
    )
    tf_feat = np.load(TF_CACHE / "features.npy").astype(np.float32)
    assert x_raw.shape[1] == 24 and tf_feat.shape[1] == 5

    splits = {}
    coverage = {}
    for sp, expected_ids in (("train", tr_ids), ("val", va_ids), ("test", te_ids)):
        z, y, ids = load_embedding_npz(emb_dir / f"{sp}.npz")
        if z.shape[1] != 198:
            raise SystemExit(f"expected H dim 198, got {z.shape[1]}")
        # coverage vs temporal split ids
        id_set = set(ids.astype(np.int64).tolist())
        exp = set(np.asarray(expected_ids).astype(np.int64).tolist()) if expected_ids is not None else id_set
        missing = sorted(exp - id_set)
        extra = sorted(id_set - exp)
        coverage[sp] = {
            "coverage": float(len(id_set & exp) / max(len(exp), 1)),
            "n_emb": int(len(ids)),
            "n_expected": int(len(exp)),
            "n_missing": int(len(missing)),
            "n_extra": int(len(extra)),
            "missing_ids_head": missing[:20],
            "ids": ids_hash(ids),
        }
        if not np.array_equal(y, y_all[ids]):
            raise SystemExit(f"label mismatch on {sp}")
        hxxtf = np.concatenate([z, x_raw[ids], tf_feat[ids]], axis=1).astype(np.float32)
        if hxxtf.shape[1] != STACK_DIM:
            raise SystemExit(f"stack dim {hxxtf.shape[1]} != {STACK_DIM}")
        splits[sp] = {"X": hxxtf, "y": y, "ids": ids, "Z": z}

    scaler = StandardScaler()
    x_tr = scaler.fit_transform(splits["train"]["X"]).astype(np.float32)
    x_va = scaler.transform(splits["val"]["X"]).astype(np.float32)
    x_te = scaler.transform(splits["test"]["X"]).astype(np.float32)

    metrics = train_mlp_18678029(
        x_tr, splits["train"]["y"], x_va, splits["val"]["y"], x_te, splits["test"]["y"],
        device=device, seed=DOWNSTREAM_SEED,
    )

    report = {
        "role": "PRIMARY_frozen_dplus_ssl_encoder_supervised_downstream",
        "description": (
            "Self-supervised contrastive encoder evaluated using a supervised "
            "downstream classifier, with the encoder frozen."
        ),
        "encoder_seed": args.encoder_seed,
        "unique_name": args.unique_name,
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": ckpt_sha,
        "checkpoint_epoch": ckpt_epoch,
        "correct_reverse_edge_features": bool(ckpt.get("correct_reverse_edge_features")),
        "preserve_seed_edges": bool(ckpt.get("preserve_seed_edges")),
        "reverse_edge_feature_semantics": ckpt.get("reverse_edge_feature_semantics"),
        "representation_source": "pre_embedding_3h",
        "stack": "H+X+TF",
        "stack_dim": STACK_DIM,
        "embeddings_dir": str(emb_dir),
        "coverage": coverage,
        "metrics": metrics,
        "val_auprc": metrics["val_ranking"]["auprc"],
        "val_f1_at_selected": metrics["val_at_selected_threshold"]["f1"],
        "test_auprc": metrics["threshold_0.5"]["auprc"],
        "test_auroc": metrics["threshold_0.5"]["auroc"],
        "test_f1_fixed_0.5": metrics["threshold_0.5"]["f1"],
        "test_f1_val_threshold": metrics["threshold_val_selected"]["f1"],
        "gnn_training_occurred": False,
        "aml_labels_updated_encoder": False,
        "test_used_for_selection": False,
        "recipe_match_18678029": {
            "mlp_epochs": MLP_EPOCHS,
            "mlp_lr": MLP_LR,
            "downstream_seed": DOWNSTREAM_SEED,
            "weight": "none",
            "focal": False,
            "feature_order": "H||X||TF",
            "scaler": "StandardScaler fit train-only",
        },
    }
    out_j = Path(args.output_json)
    out_m = Path(args.output_md)
    out_j.parent.mkdir(parents=True, exist_ok=True)
    out_m.parent.mkdir(parents=True, exist_ok=True)
    out_j.write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        f"# Frozen D+ H+X+TF MLP — encoder seed {args.encoder_seed}",
        "",
        report["description"],
        "",
        f"- unique_name: `{args.unique_name}`",
        f"- checkpoint epoch: {ckpt_epoch}",
        f"- checkpoint sha256: `{ckpt_sha}`",
        f"- val AUPRC: **{report['val_auprc']:.6f}**",
        f"- val F1@sel: **{report['val_f1_at_selected']:.6f}**",
        f"- test AUPRC: **{report['test_auprc']:.6f}**",
        f"- test AUROC: **{report['test_auroc']:.6f}**",
        f"- test F1@0.5: **{report['test_f1_fixed_0.5']:.6f}**",
        f"- test F1@val-thr: **{report['test_f1_val_threshold']:.6f}** (thr={metrics['validation_selected_threshold']})",
        f"- P@100/500/1000: {metrics['threshold_0.5']['precision_at_100']:.3f} / "
        f"{metrics['threshold_0.5']['precision_at_500']:.3f} / "
        f"{metrics['threshold_0.5']['precision_at_1000']:.3f}",
        "",
    ]
    out_m.write_text("\n".join(lines) + "\n")
    logging.info("Wrote %s and %s", out_j, out_m)


if __name__ == "__main__":
    main()
