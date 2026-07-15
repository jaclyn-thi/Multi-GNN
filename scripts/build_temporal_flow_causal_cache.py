#!/usr/bin/env python3
"""Build dataset-level temporal_flow_causal feature cache (CPU, no training).

Computes the five strictly causal features once per dataset and writes an atomic
cache under ``results/cache/temporal_flow_causal/{dataset}/``.

Cross-split history policy: features are computed on the full CSV in global timestamp
order. Validation rows reflect all prior training transactions; test rows reflect
training + validation history. Labels are not used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataset_specs import get_dataset_spec
from dataset_splits import temporal_edge_split
from morphology.temporal_flow_causal import (
    TEMPORAL_FLOW_CAUSAL_DEFINITIONS,
    TEMPORAL_FLOW_CAUSAL_FEATURE_NAMES,
    TEMPORAL_FLOW_CAUSAL_WINDOW_7D_SEC,
    compute_temporal_flow_causal_features,
    feature_summary_stats,
)
from transaction_knn.features import load_data_config, resolve_amount_column
from util import logger_setup

CACHE_VERSION = "temporal_flow_causal_v1"


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _atomic_write_bytes(final_path: Path, data: bytes) -> None:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(final_path.parent), prefix=f".{final_path.name}.", suffix=".tmp")
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        tmp_path.write_bytes(data)
        os.replace(tmp_path, final_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _atomic_write_json(final_path: Path, payload: Dict[str, Any]) -> None:
    _atomic_write_bytes(final_path, json.dumps(payload, indent=2).encode("utf-8"))


def _atomic_save_npy(final_path: Path, arr: np.ndarray) -> None:
    """Atomically write a .npy file.

    ``np.save(path)`` appends ``.npy`` when missing; write to a temp base path
    then rename the resulting ``base.npy`` file.
    """
    final_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_base = tempfile.mkstemp(
        dir=str(final_path.parent),
        prefix=f".{final_path.stem}.",
        suffix=".tmp",
    )
    os.close(fd)
    os.unlink(tmp_base)
    tmp_npy = Path(f"{tmp_base}.npy")
    try:
        np.save(tmp_base, arr)
        os.replace(tmp_npy, final_path)
    finally:
        if tmp_npy.exists() and tmp_npy.resolve() != final_path.resolve():
            tmp_npy.unlink(missing_ok=True)


def build_cache(
    data: str,
    data_config_path: str,
    cache_root: Path,
    *,
    overwrite: bool = False,
) -> Path:
    spec = get_dataset_spec(data)
    cfg = load_data_config(data_config_path)
    csv_path = Path(cfg["paths"]["aml_data"]) / data / spec.formatted_csv_name()
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    out_dir = cache_root / data
    meta_path = out_dir / "meta.json"
    features_path = out_dir / "features.npy"
    edge_id_path = out_dir / "edge_id.npy"

    if meta_path.is_file() and features_path.is_file() and not overwrite:
        logging.info("Cache already exists at %s (use --overwrite to rebuild)", out_dir)
        return out_dir

    logging.info("Loading %s", csv_path)
    df = pd.read_csv(csv_path)
    df["Timestamp"] = df["Timestamp"] - df["Timestamp"].min()
    amount_col = resolve_amount_column(df)

    y = torch.LongTensor(df[spec.label_col].to_numpy())
    timestamps = torch.Tensor(df["Timestamp"].to_numpy())
    tr_inds, val_inds, te_inds, split_buckets = temporal_edge_split(timestamps, y, spec)

    logging.info("Computing temporal_flow_causal features (%d rows)", len(df))
    features, feature_names = compute_temporal_flow_causal_features(df, amount_col=amount_col)
    if feature_names != list(TEMPORAL_FLOW_CAUSAL_FEATURE_NAMES):
        raise ValueError(f"Feature name mismatch: {feature_names}")

    edge_id = np.arange(len(df), dtype=np.int64)
    split_arrays = {
        "train": tr_inds.numpy().astype(np.int64),
        "val": val_inds.numpy().astype(np.int64),
        "test": te_inds.numpy().astype(np.int64),
    }

    meta: Dict[str, Any] = {
        "cache_version": CACHE_VERSION,
        "dataset": data,
        "feature_group": "temporal_flow_causal",
        "feature_names": list(TEMPORAL_FLOW_CAUSAL_FEATURE_NAMES),
        "feature_definitions": TEMPORAL_FLOW_CAUSAL_DEFINITIONS,
        "n_rows": int(len(df)),
        "n_features": int(features.shape[1]),
        "split_row_counts": {k: int(v.shape[0]) for k, v in split_arrays.items()},
        "split_indices_files": {
            k: f"split_{k}_edge_id.npy" for k in split_arrays
        },
        "timestamp_handling": {
            "policy": "global_timestamp_sort_mergesort",
            "timestamp_ties": (
                "Transactions with equal Timestamp are featurized as a batch using "
                "history strictly before that timestamp; account/pair state is updated "
                "only after the entire tie batch is processed."
            ),
            "timestamp_shift": "Timestamp -= min(Timestamp) over full CSV",
        },
        "causal_history_policy": {
            "past_only": True,
            "uses_labels": False,
            "val_sees_train_history": True,
            "test_sees_train_and_val_history": True,
            "window_7d_sec": TEMPORAL_FLOW_CAUSAL_WINDOW_7D_SEC,
            "no_history_defaults": {
                "log1p_sender_interarrival": 0.0,
                "log1p_receiver_interarrival": 0.0,
                "log1p_sender_past_7d_count": 0.0,
                "log1p_amount_vs_sender_past_mean": 0.0,
                "pair_repeat_indicator": 0.0,
            },
            "normalization_at_cache_time": "none (raw causal values; probe fits StandardScaler on train only)",
            "implementation": "morphology.temporal_flow_causal.compute_temporal_flow_causal_features",
            "richer_features_alignment": (
                "Interarrival, amount-vs-mean, and pair-repeat semantics match "
                "transaction_knn.richer_features with tie-safe batching added."
            ),
        },
        "source_data": {
            "csv_path": str(csv_path.resolve()),
            "csv_sha256": _sha256_file(csv_path),
            "amount_column": amount_col,
            "label_column": spec.label_col,
        },
        "split_buckets": split_buckets,
        "feature_summary": feature_summary_stats(features),
        "prevalence": {
            "positive_rate_full": float(y.float().mean().item()),
            "positive_rate_train": float(y[tr_inds].float().mean().item()),
            "positive_rate_val": float(y[val_inds].float().mean().item()),
            "positive_rate_test": float(y[te_inds].float().mean().item()),
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_metadata": {
            "script": "scripts/build_temporal_flow_causal_cache.py",
            "module": "morphology/temporal_flow_causal.py",
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    _atomic_save_npy(features_path, features.astype(np.float32))
    _atomic_save_npy(edge_id_path, edge_id)
    for split_name, idx in split_arrays.items():
        _atomic_save_npy(out_dir / f"split_{split_name}_edge_id.npy", idx)
    _atomic_write_json(meta_path, meta)

    logging.info("Wrote cache to %s", out_dir)
    return out_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, choices=["Small-HI", "Small-LI"])
    p.add_argument("--data_config", default="data_config.json")
    p.add_argument(
        "--cache_root",
        default="results/cache/temporal_flow_causal",
        help="Root directory for dataset caches",
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logger_setup()
    build_cache(
        args.data,
        args.data_config,
        Path(args.cache_root),
        overwrite=bool(args.overwrite),
    )


if __name__ == "__main__":
    main()
