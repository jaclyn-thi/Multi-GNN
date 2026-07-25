"""Data loading helpers for the standalone txn-node baseline (no Multi-GNN training imports)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from dataset_specs import get_dataset_spec
from dataset_splits import temporal_edge_split
from transaction_knn.features import resolve_amount_column


def load_data_config(path: str = "data_config.json") -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_small_hi_frame(
    data_config_path: str = "data_config.json",
    *,
    max_rows: int = 0,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load Small-HI CSV; return df, tr/val/te index arrays, meta.

    Indices are CSV row positions. For Small-HI, EdgeID aligns with row index.
    """
    cfg = load_data_config(data_config_path)
    spec = get_dataset_spec("Small-HI")
    csv_path = Path(cfg["paths"]["aml_data"]) / "Small-HI" / spec.formatted_csv_name()
    df = pd.read_csv(csv_path)
    if max_rows and max_rows > 0:
        df = df.iloc[: int(max_rows)].copy()
    df["Timestamp"] = df["Timestamp"] - df["Timestamp"].min()
    ts = torch.from_numpy(df["Timestamp"].astype(float).to_numpy())
    y = torch.from_numpy(df[spec.label_col].astype(np.int64).to_numpy())
    tr, va, te, buckets = temporal_edge_split(ts, y, spec)
    meta = {
        "csv_path": str(csv_path),
        "n_rows": int(len(df)),
        "amount_col": resolve_amount_column(df),
        "label_col": spec.label_col,
        "split_buckets": buckets,
        "n_train": int(tr.numel()),
        "n_val": int(va.numel()),
        "n_test": int(te.numel()),
    }
    return df, tr.numpy().astype(np.int64), va.numpy().astype(np.int64), te.numpy().astype(np.int64), meta


def align_knn_to_dataframe_ids(
    knn_node_ids: np.ndarray,
    knn_neighbor_ids: np.ndarray,
    csv_edge_ids_in_cache: Optional[np.ndarray],
    train_row_ids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Map KNN cache ids onto train_row_ids space.

    If cache uses train_split_local 0..N-1 matching ``train_row_ids`` order after
    reset_index of train split, remap neighbors to CSV row ids via train_row_ids.
    """
    # Standard cache: edge_ids = arange(N_train), rows align with temporal train order.
    if knn_node_ids.shape[0] != train_row_ids.shape[0]:
        raise ValueError(
            f"KNN n={knn_node_ids.shape[0]} != train n={train_row_ids.shape[0]}; "
            "refusing silent mismatch"
        )
    # Remap local -> CSV row id
    local_to_csv = train_row_ids
    # Verify cache edge_ids are arange or equal to 0..N-1 permutation of train
    if not np.array_equal(knn_node_ids, np.arange(knn_node_ids.shape[0])):
        # try csv_edge_ids identity with train_row_ids
        if csv_edge_ids_in_cache is not None and np.array_equal(csv_edge_ids_in_cache, train_row_ids):
            return csv_edge_ids_in_cache, knn_neighbor_ids, "csv_edge_ids"
        raise ValueError("Unexpected KNN id space; refusing to guess")
    remapped = local_to_csv[np.clip(knn_neighbor_ids, 0, len(local_to_csv) - 1)].copy()
    remapped[knn_neighbor_ids < 0] = -1
    return local_to_csv, remapped, "train_split_local_via_train_row_ids"
