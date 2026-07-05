#!/usr/bin/env python3
"""Post-hoc evaluation for supervised GNN checkpoints."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch_geometric.data import HeteroData
from torch_geometric.nn import to_hetero

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data_loading import get_data
from train_util import (
    AddEgoIds,
    FORWARD_EDGE_TYPE,
    add_arange_ids,
    edge_classifier_logits,
    extract_param,
    get_loaders,
)
from training import get_model
from util import create_parser, logger_setup, set_seed

ALERT_BUDGET_KS = (100, 500, 1000)


def build_model_config(args) -> SimpleNamespace:
    return SimpleNamespace(
        model=args.model,
        n_hidden=extract_param("n_hidden", args),
        n_gnn_layers=extract_param("n_gnn_layers", args),
        n_heads=extract_param("n_heads", args) if args.model == "gat" else None,
        dropout=extract_param("dropout", args),
        final_dropout=extract_param("final_dropout", args),
    )


def tune_threshold(y: np.ndarray, proba: np.ndarray) -> Tuple[float, float]:
    if len(np.unique(y)) < 2:
        pred = (proba >= 0.5).astype(np.int64)
        return 0.5, float(f1_score(y, pred, zero_division=0))
    precisions, recalls, thresholds = precision_recall_curve(y, proba)
    if thresholds.size == 0:
        pred = (proba >= 0.5).astype(np.int64)
        return 0.5, float(f1_score(y, pred, zero_division=0))
    scores = (2 * precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-12)
    idx = int(np.argmax(scores))
    return float(thresholds[idx]), float(scores[idx])


def alert_budget_metrics(y: np.ndarray, proba: np.ndarray) -> Dict[str, float]:
    y = y.astype(np.int64)
    n = int(y.shape[0])
    positives = int(y.sum())
    prevalence = float(y.mean()) if n else float("nan")
    out: Dict[str, float] = {}
    if n == 0:
        return out
    order = np.argsort(-proba)
    for k in ALERT_BUDGET_KS:
        kk = min(k, n)
        top = order[:kk]
        tp = int(y[top].sum())
        precision = float(tp / kk) if kk else float("nan")
        recall = float(tp / positives) if positives else float("nan")
        lift = float(precision / prevalence) if prevalence > 0 else float("nan")
        out[f"precision_at_{k}"] = precision
        out[f"recall_at_{k}"] = recall
        out[f"lift_at_{k}"] = lift
    return out


def split_metrics(y: np.ndarray, proba: np.ndarray, threshold: float) -> Dict[str, float]:
    pred = (proba >= threshold).astype(np.int64)
    pred_05 = (proba >= 0.5).astype(np.int64)
    out = {
        "n": float(y.shape[0]),
        "positive_rate": float(y.mean()) if y.shape[0] else float("nan"),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1_at_0_5": float(f1_score(y, pred_05, zero_division=0)),
        "threshold": float(threshold),
    }
    if len(np.unique(y)) < 2:
        out["auroc"] = float("nan")
        out["auprc"] = float("nan")
    else:
        out["auroc"] = float(roc_auc_score(y, proba))
        out["auprc"] = float(average_precision_score(y, proba))
    out.update(alert_budget_metrics(y, proba))
    return out


@torch.no_grad()
def collect_split_predictions(loader, split_inds, model, data, device, args) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    y_chunks: List[torch.Tensor] = []
    p_chunks: List[torch.Tensor] = []
    expected = 0
    seen = 0
    hetero = isinstance(data, HeteroData)
    split_inds_cpu = split_inds.detach().cpu()

    for batch in loader:
        if hetero:
            store = FORWARD_EDGE_TYPE
            batch_edge_inds = split_inds_cpu[batch[store].input_id.detach().cpu()]
            batch_edge_ids = loader.data[store].edge_attr.detach().cpu()[batch_edge_inds, 0]
            edge_ids = batch[store].edge_attr[:, 0].detach().cpu()
            mask = torch.isin(edge_ids, batch_edge_ids)
            batch[store].edge_attr = batch[store].edge_attr[:, 1:]
            batch[("node", "rev_to", "node")].edge_attr = batch[("node", "rev_to", "node")].edge_attr[:, 1:]
            batch.to(device)
            z = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)[store]
            logits = edge_classifier_logits(model, z)[mask.to(device)]
            y = batch[store].y[mask.to(device)]
        else:
            batch_edge_inds = split_inds_cpu[batch.input_id.detach().cpu()]
            batch_edge_ids = loader.data.edge_attr.detach().cpu()[batch_edge_inds, 0]
            edge_ids = batch.edge_attr[:, 0].detach().cpu()
            mask = torch.isin(edge_ids, batch_edge_ids)
            batch.edge_attr = batch.edge_attr[:, 1:]
            batch.to(device)
            z = model(batch.x, batch.edge_index, batch.edge_attr)
            logits = model.classifier(z)[mask.to(device)]
            y = batch.y[mask.to(device)]
        expected += int(batch_edge_ids.numel())
        seen += int(mask.sum().item())
        proba = torch.softmax(logits, dim=-1)[:, 1]
        p_chunks.append(proba.detach().cpu())
        y_chunks.append(y.detach().cpu().long())

    y_np = torch.cat(y_chunks).numpy().astype(np.int64)
    p_np = torch.cat(p_chunks).numpy().astype(np.float64)
    return y_np, p_np, {
        "expected_seed_edges": expected,
        "scored_seed_edges": seen,
        "coverage": float(seen / max(expected, 1)),
    }


def parse_training_log(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "status": "missing"}
    epochs = []
    cur: Dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m_train = re.search(r"Train F1: ([0-9.]+)", line)
        m_val = re.search(r"Validation F1: ([0-9.]+)", line)
        m_test = re.search(r"Test F1: ([0-9.]+)", line)
        if m_train:
            if cur:
                epochs.append(cur)
            cur = {"epoch_index": len(epochs) + 1, "train_f1_argmax": float(m_train.group(1))}
        elif m_val:
            cur["val_f1_argmax"] = float(m_val.group(1))
        elif m_test:
            cur["test_f1_argmax"] = float(m_test.group(1))
    if cur:
        epochs.append(cur)
    best = max(epochs, key=lambda r: r.get("val_f1_argmax", float("-inf"))) if epochs else None
    return {
        "path": str(path),
        "epochs": epochs,
        "best_by_val_f1_argmax": best,
        "note": "Canonical supervised checkpoint is the final saved checkpoint; best epoch is parsed from logs only.",
    }


def write_markdown(path: Path, payload: Dict[str, Any]) -> None:
    lines = [
        "# Supervised Small-LI GINe Baseline",
        "",
        "Small-LI comparison run using the repo's canonical supervised CE path.",
        "",
        f"- **run:** `{payload['run_name']}`",
        f"- **checkpoint:** `{payload['checkpoint_path']}`",
        f"- **CE class weights:** `{payload['ce_class_weight']}`",
        f"- **checkpoint epoch:** {payload['checkpoint_epoch']}",
        "",
        "| Split | AUROC | AUPRC | F1 | F1@0.5 | Precision | Recall | P@500 | R@500 | lift@500 | Pos Rate |",
        "|-------|------:|------:|---:|-------:|----------:|-------:|------:|------:|---------:|---------:|",
    ]
    for split in ("train", "val", "test"):
        row = payload["splits"][split]
        lines.append(
            f"| {split} | {row['auroc']:.4f} | {row['auprc']:.4f} | {row['f1']:.4f} | "
            f"{row['f1_at_0_5']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row.get('precision_at_500', float('nan')):.4f} | {row.get('recall_at_500', float('nan')):.4f} | "
            f"{row.get('lift_at_500', float('nan')):.1f} | {row['positive_rate']:.6f} |"
        )
    best = payload.get("training_log", {}).get("best_by_val_f1_argmax")
    if best:
        lines.extend(
            [
                "",
                f"Best log epoch by canonical argmax Validation F1: epoch {best.get('epoch_index')} "
                f"(val {best.get('val_f1_argmax'):.4f}, test {best.get('test_f1_argmax', float('nan')):.4f}).",
            ]
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = create_parser()
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    parser.add_argument("--training_log", default=None)
    args = parser.parse_args()

    logger_setup()
    set_seed(args.seed)
    with open(args.data_config if hasattr(args, "data_config") else "data_config.json", "r", encoding="utf-8") as f:
        data_config = json.load(f)

    tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(args, data_config)
    transform = AddEgoIds() if args.ego else None
    add_arange_ids([tr_data, val_data, te_data])
    tr_loader, val_loader, te_loader = get_loaders(
        tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform, args, train_shuffle=False
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sample_batch = next(iter(tr_loader))
    config = build_model_config(args)
    model = get_model(sample_batch, config, args)
    if args.reverse_mp:
        model = to_hetero(model, te_data.metadata(), aggr="mean")
    checkpoint_path = Path(data_config["paths"]["model_to_load"]) / f"checkpoint_{args.unique_name}.tar"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    y_val, p_val, cov_val = collect_split_predictions(val_loader, val_inds, model, val_data, device, args)
    threshold, val_f1 = tune_threshold(y_val, p_val)

    split_payload = {}
    coverage = {"val": cov_val}
    split_payload["val"] = split_metrics(y_val, p_val, threshold)
    for name, loader, inds, data in (
        ("train", tr_loader, tr_inds, tr_data),
        ("test", te_loader, te_inds, te_data),
    ):
        y, p, cov = collect_split_predictions(loader, inds, model, data, device, args)
        split_payload[name] = split_metrics(y, p, threshold)
        coverage[name] = cov

    payload = {
        "run_name": args.unique_name,
        "data": args.data,
        "model": args.model,
        "objective": "supervised",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "ce_class_weight": {
            "0": float(extract_param("w_ce1", args)),
            "1": float(extract_param("w_ce2", args)),
        },
        "classification_threshold": {
            "method": "max_f1_on_val",
            "value": float(threshold),
            "val_f1_at_selection": float(val_f1),
        },
        "splits": split_payload,
        "coverage": coverage,
        "training_log": parse_training_log(Path(args.training_log)) if args.training_log else None,
    }
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(Path(args.output_md), payload)
    print(out_json)


if __name__ == "__main__":
    main()
