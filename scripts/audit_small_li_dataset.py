#!/usr/bin/env python3
"""Lightweight sanity audit for the AMLWorld Small-LI dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataset_specs import FORMATTED_TRANSACTION_COLUMNS, get_dataset_spec, spec_summary
from dataset_splits import temporal_edge_split
from pattern_metadata import default_pattern_metadata_path
from scripts.probe_feature_ablation import build_full_feature_matrix
from transaction_knn.features import load_data_config, resolve_amount_column


def split_summary(df: pd.DataFrame, y: torch.Tensor, inds: torch.Tensor) -> Dict[str, Any]:
    labels = y[inds].numpy().astype(np.int64)
    return {
        "n_edges": int(labels.shape[0]),
        "n_positive": int(labels.sum()),
        "positive_rate": float(labels.mean()) if labels.shape[0] else None,
        "timestamp_min": float(df.iloc[inds.numpy()]["Timestamp"].min()) if labels.shape[0] else None,
        "timestamp_max": float(df.iloc[inds.numpy()]["Timestamp"].max()) if labels.shape[0] else None,
    }


def write_markdown(path: Path, payload: Dict[str, Any]) -> None:
    spec = payload["dataset_spec"]
    splits = payload["splits"]
    feature_check = payload["feature_generation_check"]
    lines = [
        "# Small-LI Dataset Audit",
        "",
        "| Item | Result |",
        "|------|--------|",
        f"| Dataset CLI key | `{payload['dataset_key']}` |",
        f"| CSV path | `{payload['csv_path']}` |",
        f"| Label column | `{spec['label_col']}` |",
        f"| Edge feature columns | `{', '.join(spec['edge_feature_cols'])}` |",
        f"| Split mode | `{spec['split_mode']}` ({spec['split_fractions']}) |",
        f"| Total edges | {payload['total_edges']:,} |",
        f"| Nodes/accounts | {payload['n_nodes']:,} |",
        f"| Overall positive rate | {payload['overall_positive_rate']:.6f} |",
        f"| Pattern metadata | {payload['pattern_metadata']['status']} |",
        f"| Raw/morph feature generation | {feature_check['status']} |",
        "",
        "## Splits",
        "",
        "| Split | Edges | Positives | Positive Rate |",
        "|-------|------:|----------:|--------------:|",
    ]
    for name in ("train", "val", "test"):
        row = splits[name]
        lines.append(
            f"| {name} | {row['n_edges']:,} | {row['n_positive']:,} | "
            f"{row['positive_rate']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Feature Generation",
            "",
            f"- Raw feature matrix shape: `{feature_check.get('raw_shape')}`",
            f"- Morph feature matrix shape: `{feature_check.get('morph_shape')}`",
            f"- Raw+morph feature matrix shape: `{feature_check.get('raw_morph_shape')}`",
            f"- Amount column: `{feature_check.get('amount_column')}`",
            "",
            f"Submission gate: **{payload['submission_gate']['status']}** - "
            f"{payload['submission_gate']['reason']}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="Small-LI")
    parser.add_argument("--data_config", default="data_config.json")
    parser.add_argument(
        "--output_json",
        default="results/diagnostics/small_li_dataset_audit.json",
    )
    parser.add_argument(
        "--output_md",
        default="results/diagnostics/small_li_dataset_audit.md",
    )
    args = parser.parse_args()

    cfg = load_data_config(args.data_config)
    spec = get_dataset_spec(args.data)
    csv_path = Path(cfg["paths"]["aml_data"]) / args.data / spec.formatted_csv_name()
    payload: Dict[str, Any] = {
        "dataset_key": args.data,
        "csv_path": str(csv_path),
        "dataset_spec": dict(spec_summary(spec)),
    }

    if not csv_path.is_file():
        payload["submission_gate"] = {
            "status": "STOP",
            "reason": f"missing formatted CSV: {csv_path}",
        }
    else:
        df = pd.read_csv(csv_path)
        missing = [col for col in FORMATTED_TRANSACTION_COLUMNS if col not in df.columns]
        edge_missing = [col for col in spec.edge_feature_cols if col not in df.columns]
        label_missing = spec.label_col not in df.columns
        df["Timestamp"] = df["Timestamp"] - df["Timestamp"].min()
        y = torch.LongTensor(df[spec.label_col].to_numpy()) if not label_missing else torch.LongTensor([])

        if not label_missing:
            timestamps = torch.Tensor(df["Timestamp"].to_numpy())
            tr_inds, val_inds, te_inds, split_buckets = temporal_edge_split(timestamps, y, spec)
            splits = {
                "train": split_summary(df, y, tr_inds),
                "val": split_summary(df, y, val_inds),
                "test": split_summary(df, y, te_inds),
            }
        else:
            tr_inds = val_inds = te_inds = torch.LongTensor([])
            split_buckets = [[], [], []]
            splits = {}

        pattern_path = default_pattern_metadata_path(cfg, args.data)
        pattern_status = "available" if pattern_path.is_file() else "not found"
        pattern_rows = None
        if pattern_path.is_file():
            pattern_rows = int(pd.read_csv(pattern_path, usecols=[0]).shape[0])

        feature_check: Dict[str, Any]
        try:
            if label_missing or edge_missing:
                raise ValueError("label or edge feature columns missing")
            df_train = df.iloc[tr_inds.numpy()].reset_index(drop=True)
            raw, _, _, _ = build_full_feature_matrix(
                df, df_train, ("edge_native",), categorical_encoding="ordinal"
            )
            morph, _, _, _ = build_full_feature_matrix(
                df,
                df_train,
                ("degree_fan", "flow_balance", "temporal_behavior"),
                categorical_encoding="ordinal",
            )
            raw_morph, _, _, _ = build_full_feature_matrix(
                df,
                df_train,
                ("edge_native", "degree_fan", "flow_balance", "temporal_behavior"),
                categorical_encoding="ordinal",
            )
            feature_check = {
                "status": "ok",
                "raw_shape": list(raw.shape),
                "morph_shape": list(morph.shape),
                "raw_morph_shape": list(raw_morph.shape),
                "amount_column": resolve_amount_column(df),
            }
        except Exception as exc:  # noqa: BLE001 - audit should record failure reason.
            feature_check = {
                "status": "failed",
                "error": repr(exc),
                "amount_column": resolve_amount_column(df) if "df" in locals() else None,
            }

        total_edges = int(df.shape[0])
        total_pos = int(y.sum().item()) if y.numel() else 0
        n_nodes = int(df.loc[:, ["from_id", "to_id"]].to_numpy().max() + 1)
        train_edges = splits.get("train", {}).get("n_edges", 0)
        train_pos_rate = splits.get("train", {}).get("positive_rate")
        val_pos_rate = splits.get("val", {}).get("positive_rate")
        test_pos_rate = splits.get("test", {}).get("positive_rate")

        gate_status = "OK"
        gate_reasons = []
        if missing:
            gate_status = "STOP"
            gate_reasons.append(f"missing formatted columns: {missing}")
        if edge_missing:
            gate_status = "STOP"
            gate_reasons.append(f"missing edge feature columns: {edge_missing}")
        if label_missing:
            gate_status = "STOP"
            gate_reasons.append(f"missing label column: {spec.label_col}")
        if total_pos == 0:
            gate_status = "STOP"
            gate_reasons.append("no positive labels")
        if any(rate in (0.0, 1.0) for rate in (train_pos_rate, val_pos_rate, test_pos_rate)):
            gate_status = "STOP"
            gate_reasons.append("at least one split has an absent or all-positive label class")
        if feature_check["status"] != "ok":
            gate_status = "STOP"
            gate_reasons.append(f"raw/morph feature generation failed: {feature_check.get('error')}")
        if train_edges < 8192:
            gate_status = "STOP"
            gate_reasons.append(
                f"train split has {train_edges} edges, below requested batch/negative scale"
            )
        if not gate_reasons:
            gate_reasons.append("schema, labels, splits, and raw/morph features look sane")

        payload.update(
            {
                "columns": list(df.columns),
                "missing_formatted_columns": missing,
                "missing_edge_feature_columns": edge_missing,
                "total_edges": total_edges,
                "n_nodes": n_nodes,
                "n_positive": total_pos,
                "overall_positive_rate": float(y.float().mean().item()) if y.numel() else 0.0,
                "split_buckets": split_buckets,
                "splits": splits,
                "pattern_metadata": {
                    "path": str(pattern_path),
                    "status": pattern_status,
                    "n_rows": pattern_rows,
                },
                "feature_generation_check": feature_check,
                "submission_gate": {
                    "status": gate_status,
                    "reason": "; ".join(gate_reasons),
                },
            }
        )

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(Path(args.output_md), payload)
    print(json.dumps(payload["submission_gate"], indent=2))


if __name__ == "__main__":
    main()
