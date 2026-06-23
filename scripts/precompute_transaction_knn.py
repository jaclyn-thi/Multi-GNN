#!/usr/bin/env python
"""Precompute sparse feature-KNN neighbors for transaction contrastive filtering.

This script builds train-only KNN over label-free transaction features and saves
only top-k neighbor IDs/similarities. It does not create a dense graph view.
Saved ``edge_ids`` and ``neighbor_ids`` are train split-local IDs, matching the
``edge_id`` values used by contrastive training after ``add_arange_ids``.
Original CSV ``EdgeID`` values are also saved for audit/debugging.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset_specs import DEFAULT_EDGE_FEATURE_COLS, get_dataset_spec, spec_summary
from dataset_splits import temporal_edge_split
from morphology.target_registry import morph_target_group


EDGE_NATIVE_COLUMNS = list(DEFAULT_EDGE_FEATURE_COLS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="Small-HI", help="Dataset folder under aml-data")
    parser.add_argument("--data_config", default="data_config.json", help="Path to data_config.json")
    parser.add_argument(
        "--feature_set",
        default="edge_native",
        choices=["edge_native", "degree_fan", "edge_native+degree_fan"],
        help="Label-free feature family for KNN.",
    )
    parser.add_argument("--k", type=int, default=50, help="Top-k neighbors per train transaction")
    parser.add_argument("--output", required=True, help="Output .npz path")
    parser.add_argument(
        "--query_batch_size",
        type=int,
        default=10000,
        help="Rows queried per KNN batch; lowers peak memory for large datasets.",
    )
    parser.add_argument(
        "--metric",
        default="cosine",
        choices=["cosine", "euclidean"],
        help="KNN metric after preprocessing/standardization",
    )
    parser.add_argument("--log_level", default="INFO", help="Python logging level")
    return parser.parse_args()


def load_data_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_log1p(values: pd.Series) -> np.ndarray:
    arr = values.astype(float).to_numpy()
    arr = np.maximum(arr, 0.0)
    return np.log1p(arr)


def edge_native_features(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    cols = [c for c in EDGE_NATIVE_COLUMNS if c in df.columns]
    if len(cols) != len(EDGE_NATIVE_COLUMNS):
        missing = sorted(set(EDGE_NATIVE_COLUMNS) - set(cols))
        raise ValueError(f"formatted_transactions.csv missing edge-native columns: {missing}")
    parts = []
    names = []
    for col in cols:
        if "Amount" in col:
            parts.append(_safe_log1p(df[col]))
            names.append(f"log1p_{col}")
        else:
            parts.append(df[col].astype(float).to_numpy())
            names.append(col)
    return np.column_stack(parts).astype(np.float32), names


def degree_fan_features(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    from_ids = df["from_id"].astype(np.int64).to_numpy()
    to_ids = df["to_id"].astype(np.int64).to_numpy()
    max_node = int(max(from_ids.max(initial=0), to_ids.max(initial=0))) + 1
    out_deg = np.bincount(from_ids, minlength=max_node).astype(np.float32)
    in_deg = np.bincount(to_ids, minlength=max_node).astype(np.float32)
    features = np.column_stack(
        [
            np.log1p(out_deg[from_ids]),
            np.log1p(in_deg[from_ids]),
            np.log1p(out_deg[to_ids]),
            np.log1p(in_deg[to_ids]),
            np.log1p(out_deg[from_ids] + in_deg[from_ids]),
            np.log1p(out_deg[to_ids] + in_deg[to_ids]),
            np.log1p(out_deg[from_ids] + out_deg[to_ids]),
            np.log1p(in_deg[from_ids] + in_deg[to_ids]),
        ]
    ).astype(np.float32)
    names = [
        "log1p_sender_out_degree_train",
        "log1p_sender_in_degree_train",
        "log1p_receiver_out_degree_train",
        "log1p_receiver_in_degree_train",
        "log1p_sender_total_degree_train",
        "log1p_receiver_total_degree_train",
        "log1p_pair_out_degree_sum_train",
        "log1p_pair_in_degree_sum_train",
    ]
    return features, names


def build_features(df_train: pd.DataFrame, feature_set: str) -> Tuple[np.ndarray, List[str]]:
    matrices: List[np.ndarray] = []
    names: List[str] = []
    if "edge_native" in feature_set:
        x, n = edge_native_features(df_train)
        matrices.append(x)
        names.extend(n)
    if "degree_fan" in feature_set:
        x, n = degree_fan_features(df_train)
        matrices.append(x)
        names.extend(n)
    if not matrices:
        raise ValueError(f"Unsupported feature_set={feature_set!r}")
    features = np.concatenate(matrices, axis=1).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError("KNN features contain non-finite values after preprocessing")
    return features, names


def cosine_similarity_from_distance(distance: np.ndarray, metric: str) -> np.ndarray:
    if metric == "cosine":
        return (1.0 - distance).astype(np.float32)
    return (-distance).astype(np.float32)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper()), format="%(asctime)s [%(levelname)s] %(message)s")
    if args.k <= 0:
        raise ValueError("--k must be positive")

    spec = get_dataset_spec(args.data)
    cfg = load_data_config(args.data_config)
    csv_path = Path(cfg["paths"]["aml_data"]) / args.data / spec.formatted_csv_name()
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    logging.info("Loading transactions: %s", csv_path)
    df = pd.read_csv(csv_path)
    df["Timestamp"] = df["Timestamp"] - df["Timestamp"].min()
    y = torch.LongTensor(df[spec.label_col].to_numpy())
    timestamps = torch.Tensor(df["Timestamp"].to_numpy())
    tr_inds, _, _, split = temporal_edge_split(timestamps, y, spec)
    train_pos = tr_inds.numpy()
    df_train = df.iloc[train_pos].reset_index(drop=True)
    logging.info("Train-only KNN rows: %d / %d", len(df_train), len(df))

    features, feature_names = build_features(df_train, args.feature_set)
    if features.shape[0] < 2:
        raise ValueError("Need at least two train transactions to build KNN")
    scaler = StandardScaler()
    features = scaler.fit_transform(features).astype(np.float32)
    k = min(int(args.k), features.shape[0] - 1)
    logging.info("Fitting KNN: feature_set=%s dim=%d k=%d metric=%s", args.feature_set, features.shape[1], k, args.metric)
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric=args.metric, algorithm="auto", n_jobs=-1)
    nbrs.fit(features)
    neighbor_ids = np.full((features.shape[0], k), -1, dtype=np.int64)
    neighbor_sims = np.full((features.shape[0], k), np.nan, dtype=np.float32)
    query_batch_size = max(1, int(args.query_batch_size))
    for start in range(0, features.shape[0], query_batch_size):
        end = min(start + query_batch_size, features.shape[0])
        logging.info("Querying KNN rows %d:%d", start, end)
        distances, indices = nbrs.kneighbors(features[start:end], return_distance=True)
        # Drop self-neighbor when present; fall back to first k non-self entries.
        for local_i in range(end - start):
            row_i = start + local_i
            keep = indices[local_i] != row_i
            idx = indices[local_i][keep][:k]
            dist = distances[local_i][keep][:k]
            neighbor_ids[row_i, : idx.shape[0]] = idx.astype(np.int64)
            neighbor_sims[row_i, : idx.shape[0]] = cosine_similarity_from_distance(dist, args.metric)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "data": args.data,
        "dataset_spec": dict(spec_summary(spec)),
        "feature_set": args.feature_set,
        "feature_names": feature_names,
        "k": k,
        "metric": args.metric,
        "query_batch_size": query_batch_size,
        "id_space": "train_split_local_edge_id",
        "n_train": int(features.shape[0]),
        "split_buckets": split,
        "label_free": True,
    }
    np.savez_compressed(
        output,
        edge_ids=np.arange(features.shape[0], dtype=np.int64),
        csv_edge_ids=df_train["EdgeID"].astype(np.int64).to_numpy(),
        neighbor_ids=neighbor_ids,
        neighbor_sims=neighbor_sims,
        feature_names=np.asarray(feature_names, dtype=object),
        k=np.asarray(k, dtype=np.int64),
        feature_set=np.asarray(args.feature_set),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    logging.info("Wrote sparse KNN cache: %s", output)
    logging.info("Feature groups in cache: %s", sorted({morph_target_group(n) for n in feature_names}))


if __name__ == "__main__":
    main()
