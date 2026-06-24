"""Label-free transaction feature matrices for offline KNN."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler, StandardScaler

from dataset_specs import DEFAULT_EDGE_FEATURE_COLS, get_dataset_spec, spec_summary
from dataset_splits import temporal_edge_split
from transaction_knn.richer_features import (
    compute_causal_edge_stats,
    degree_causal_features,
    flow_rich_features,
    flow_causal_features,
    pair_history_features,
    relative_amount_features,
    resolve_amount_column,
    temporal_causal_features,
    time_bucket_features,
)

EDGE_NATIVE_COLUMNS = list(DEFAULT_EDGE_FEATURE_COLS)
CATEGORICAL_EDGE_COLUMNS = ("Received Currency", "Payment Format")

# Alias → ordered feature groups (concatenated left-to-right).
FEATURE_SET_ALIASES: Dict[str, Tuple[str, ...]] = {
    "richer_v1": (
        "edge_native",
        "time_bucket",
        "degree_fan",
        "degree_causal",
        "flow_rich",
        "relative_amount",
        "temporal_causal",
        "pair_history",
    ),
    "richer_v1_no_pair": (
        "edge_native",
        "time_bucket",
        "degree_fan",
        "degree_causal",
        "flow_rich",
        "relative_amount",
        "temporal_causal",
    ),
    "richer_v1_no_static": (
        "edge_native",
        "time_bucket",
        "degree_causal",
        "flow_causal",
        "relative_amount",
        "temporal_causal",
        "pair_history",
    ),
    "richer_v1_causal_only": (
        "edge_native",
        "time_bucket",
        "degree_causal",
        "flow_causal",
        "relative_amount",
        "temporal_causal",
    ),
}

# Default group weights for cosine KNN (applied after per-group scaling).
DEFAULT_GROUP_WEIGHTS: Dict[str, float] = {
    "edge_native": 1.0,
    "time_bucket": 0.75,
    "degree_fan": 0.5,
    "degree_causal": 1.0,
    "flow_rich": 1.0,
    "relative_amount": 1.25,
    "temporal_causal": 1.0,
    "flow_causal": 1.0,
    "pair_history": 0.75,
    "flow_balance": 1.0,
    "temporal_behavior": 0.75,
}


@dataclass
class FeatureBuildResult:
    features: np.ndarray
    names: List[str]
    groups: List[str]
    group_slices: Dict[str, slice]
    group_dims: Dict[str, int]
    categorical_encoding: str
    feature_set: str
    amount_column: str
    scaling: str
    group_weights: Dict[str, float] = field(default_factory=dict)


def load_data_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_log1p(values: pd.Series) -> np.ndarray:
    arr = values.astype(float).to_numpy()
    arr = np.maximum(arr, 0.0)
    return np.log1p(arr)


def _encode_categorical_column(values: pd.Series, *, one_hot: bool) -> Tuple[np.ndarray, List[str]]:
    col = values.fillna("__missing__").astype(str)
    if not one_hot:
        codes, _ = pd.factorize(col, sort=True)
        return codes.astype(np.float32).reshape(-1, 1), [f"{values.name}_ordinal"]
    dummies = pd.get_dummies(col, prefix=str(values.name), dtype=np.float32)
    return dummies.to_numpy(dtype=np.float32), list(dummies.columns.astype(str))


def edge_native_features(
    df: pd.DataFrame,
    *,
    categorical_encoding: str = "ordinal",
    amount_col: Optional[str] = None,
) -> Tuple[np.ndarray, List[str]]:
    cols = [c for c in EDGE_NATIVE_COLUMNS if c in df.columns]
    if "Timestamp" not in cols or not any("Amount" in c for c in cols):
        missing = sorted(set(EDGE_NATIVE_COLUMNS) - set(cols))
        raise ValueError(f"formatted_transactions.csv missing edge-native columns: {missing}")
    one_hot = categorical_encoding == "one_hot"
    amount_col = amount_col or resolve_amount_column(df)
    parts: List[np.ndarray] = []
    names: List[str] = []
    for col in cols:
        if col in CATEGORICAL_EDGE_COLUMNS and col in df.columns:
            x, n = _encode_categorical_column(df[col], one_hot=one_hot)
            parts.append(x)
            names.extend(n)
        elif "Amount" in col:
            use_col = amount_col if amount_col in df.columns else col
            parts.append(_safe_log1p(df[use_col]).reshape(-1, 1))
            names.append(f"log1p_{use_col}")
        else:
            parts.append(df[col].astype(float).to_numpy().reshape(-1, 1))
            names.append(col)
    return np.concatenate(parts, axis=1).astype(np.float32), names


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


def _flow_balance_features(df: pd.DataFrame, amount_col: str) -> Tuple[np.ndarray, List[str]]:
    from_ids = df["from_id"].astype(np.int64).to_numpy()
    to_ids = df["to_id"].astype(np.int64).to_numpy()
    amounts = np.maximum(df[amount_col].astype(float).to_numpy(), 0.0)
    max_node = int(max(from_ids.max(initial=0), to_ids.max(initial=0))) + 1
    amount_out = np.bincount(from_ids, weights=amounts, minlength=max_node).astype(np.float64)
    amount_in = np.bincount(to_ids, weights=amounts, minlength=max_node).astype(np.float64)
    eps = 1e-8
    s_out = amount_out[from_ids]
    s_in = amount_in[from_ids]
    r_in = amount_in[to_ids]
    r_out = amount_out[to_ids]
    s_ratio = np.clip((s_out - s_in) / (s_out + s_in + eps), -1.0, 1.0)
    r_ratio = np.clip((r_out - r_in) / (r_out + r_in + eps), -1.0, 1.0)
    x = np.column_stack(
        [
            np.log1p(s_out),
            np.log1p(s_in),
            np.log1p(r_in),
            np.log1p(r_out),
            s_ratio,
            r_ratio,
        ]
    ).astype(np.float32)
    names = [
        "log1p_sender_out_amount_train",
        "log1p_sender_in_amount_train",
        "log1p_receiver_in_amount_train",
        "log1p_receiver_out_amount_train",
        "sender_flow_balance_ratio_train",
        "receiver_flow_balance_ratio_train",
    ]
    return x, names


def _temporal_behavior_features(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    ts = df["Timestamp"].astype(float).to_numpy()
    ts_norm = (ts - ts.min()) / max(float(ts.max() - ts.min()), 1.0)
    order = np.argsort(ts)
    inter = np.zeros_like(ts, dtype=np.float32)
    inter[1:] = np.log1p(np.maximum(np.diff(ts[order]), 0.0))
    inv = np.empty_like(order)
    inv[order] = np.arange(order.shape[0])
    inter = inter[inv]
    return np.column_stack([ts_norm, inter]).astype(np.float32), ["timestamp_norm", "log1p_interarrival"]


def _resolve_feature_groups(feature_set: str) -> Tuple[str, Tuple[str, ...]]:
    if feature_set in FEATURE_SET_ALIASES:
        return feature_set, FEATURE_SET_ALIASES[feature_set]
    groups = tuple(g for g in feature_set.split("+") if g)
    return feature_set, groups


def _append_group(
    matrices: List[np.ndarray],
    names: List[str],
    group_slices: Dict[str, slice],
    group: str,
    x: np.ndarray,
    group_names: List[str],
) -> None:
    start = sum(m.shape[1] for m in matrices)
    matrices.append(x)
    names.extend(group_names)
    group_slices[group] = slice(start, start + x.shape[1])


def build_features(
    df_train: pd.DataFrame,
    feature_set: str,
    *,
    categorical_encoding: str = "ordinal",
    include_pair_history: Optional[bool] = None,
) -> Tuple[np.ndarray, List[str]]:
    result = build_features_detailed(
        df_train,
        feature_set,
        categorical_encoding=categorical_encoding,
        include_pair_history=include_pair_history,
    )
    return result.features, result.names


def build_features_detailed(
    df_train: pd.DataFrame,
    feature_set: str,
    *,
    categorical_encoding: str = "ordinal",
    include_pair_history: Optional[bool] = None,
    scaling: str = "none",
    group_weights: Optional[Dict[str, float]] = None,
) -> FeatureBuildResult:
    if categorical_encoding not in {"ordinal", "one_hot"}:
        raise ValueError(f"Unsupported categorical_encoding={categorical_encoding!r}")
    if scaling not in {"none", "standard", "robust"}:
        raise ValueError(f"Unsupported scaling={scaling!r}")

    resolved_name, groups = _resolve_feature_groups(feature_set)
    if include_pair_history is False and "pair_history" in groups:
        groups = tuple(g for g in groups if g != "pair_history")
    if include_pair_history is True and "pair_history" not in groups and resolved_name.startswith("richer_v1"):
        groups = groups + ("pair_history",)

    amount_col = resolve_amount_column(df_train)
    causal: Optional[Dict[str, np.ndarray]] = None
    needs_causal = any(
        g in groups
        for g in ("degree_causal", "flow_rich", "flow_causal", "relative_amount", "temporal_causal", "pair_history")
    )
    if needs_causal:
        causal = compute_causal_edge_stats(df_train, amount_col)

    matrices: List[np.ndarray] = []
    names: List[str] = []
    group_slices: Dict[str, slice] = {}
    active_groups: List[str] = []

    for group in groups:
        if group == "edge_native":
            x, n = edge_native_features(
                df_train, categorical_encoding=categorical_encoding, amount_col=amount_col
            )
        elif group == "time_bucket":
            x, n = time_bucket_features(df_train)
        elif group == "degree_fan":
            x, n = degree_fan_features(df_train)
        elif group == "degree_causal":
            assert causal is not None
            x, n = degree_causal_features(causal)
        elif group == "flow_balance":
            x, n = _flow_balance_features(df_train, amount_col)
        elif group == "flow_rich":
            assert causal is not None
            x, n = flow_rich_features(df_train, amount_col, causal)
        elif group == "flow_causal":
            assert causal is not None
            x, n = flow_causal_features(causal)
        elif group == "relative_amount":
            assert causal is not None
            x, n = relative_amount_features(df_train, amount_col, causal)
        elif group == "temporal_behavior":
            x, n = _temporal_behavior_features(df_train)
        elif group == "temporal_causal":
            assert causal is not None
            x, n = temporal_causal_features(causal)
        elif group == "pair_history":
            assert causal is not None
            x, n = pair_history_features(causal)
        else:
            raise ValueError(f"Unsupported feature group={group!r} in feature_set={feature_set!r}")
        _append_group(matrices, names, group_slices, group, x, n)
        active_groups.append(group)

    if not matrices:
        raise ValueError(f"Unsupported feature_set={feature_set!r}")
    features = np.concatenate(matrices, axis=1).astype(np.float32)
    if not np.isfinite(features).all():
        bad = int((~np.isfinite(features)).sum())
        raise ValueError(f"KNN features contain {bad} non-finite values after preprocessing")

    weights = dict(DEFAULT_GROUP_WEIGHTS)
    if group_weights:
        weights.update(group_weights)

    features_scaled = features
    if scaling != "none":
        features_scaled = prepare_knn_features(
            features,
            group_slices,
            scaling=scaling,
            group_weights=weights if resolved_name.startswith("richer") else None,
        )

    return FeatureBuildResult(
        features=features_scaled,
        names=names,
        groups=active_groups,
        group_slices=group_slices,
        group_dims={g: int(group_slices[g].stop - group_slices[g].start) for g in active_groups},
        categorical_encoding=categorical_encoding,
        feature_set=resolved_name,
        amount_column=amount_col,
        scaling=scaling,
        group_weights={g: float(weights.get(g, 1.0)) for g in active_groups},
    )


def prepare_knn_features(
    features: np.ndarray,
    group_slices: Dict[str, slice],
    *,
    scaling: str = "standard",
    group_weights: Optional[Dict[str, float]] = None,
    l2_normalize: bool = True,
) -> np.ndarray:
    """Per-group scaler + optional weighting; optional row L2 norm (cosine)."""
    if scaling not in {"standard", "robust"}:
        raise ValueError(f"Unsupported scaling={scaling!r}")
    scaler_cls = StandardScaler if scaling == "standard" else RobustScaler
    out = np.empty_like(features, dtype=np.float32)
    for group, sl in group_slices.items():
        block = features[:, sl]
        scaled = scaler_cls().fit_transform(block).astype(np.float32)
        w = float((group_weights or {}).get(group, 1.0))
        if w != 1.0:
            scaled *= np.float32(w)
        out[:, sl] = scaled
    if l2_normalize:
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return (out / norms).astype(np.float32)
    return out.astype(np.float32)


def standardize_features(features: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    return scaler.fit_transform(features).astype(np.float32)


def load_train_frame(
    data: str,
    data_config_path: str,
    *,
    max_rows: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict, object]:
    spec = get_dataset_spec(data)
    cfg = load_data_config(data_config_path)
    csv_path = Path(cfg["paths"]["aml_data"]) / data / spec.formatted_csv_name()
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    df["Timestamp"] = df["Timestamp"] - df["Timestamp"].min()
    y = torch.LongTensor(df[spec.label_col].to_numpy())
    timestamps = torch.Tensor(df["Timestamp"].to_numpy())
    tr_inds, _, _, split = temporal_edge_split(timestamps, y, spec)
    train_pos = tr_inds.numpy()
    df_train = df.iloc[train_pos].reset_index(drop=True)
    if max_rows > 0:
        df_train = df_train.iloc[: int(max_rows)].reset_index(drop=True)
    return df, df_train, split, spec


def feature_set_metadata(result: FeatureBuildResult) -> Dict[str, object]:
    return {
        "feature_set": result.feature_set,
        "feature_names": result.names,
        "feature_groups": result.groups,
        "group_dims": result.group_dims,
        "group_slices": {k: [v.start, v.stop] for k, v in result.group_slices.items()},
        "categorical_encoding": result.categorical_encoding,
        "amount_column": result.amount_column,
        "scaling": result.scaling,
        "group_weights": result.group_weights,
        "label_free": True,
        "leakage_policy": {
            "excluded_columns": [
                "Is Laundering",
                "pattern metadata",
                "fraud labels",
                "val/test statistics",
            ],
            "causal_groups": [
                "degree_causal",
                "temporal_causal",
                "relative_amount",
                "pair_history",
                "flow_rich (causal ratio cols)",
            ],
            "static_train_graph_groups": [
                "degree_fan",
                "flow_rich (train-total amount cols)",
            ],
        },
    }


def dataset_metadata(
    data: str,
    spec,
    feature_set: str,
    feature_names: List[str],
    k: int,
    metric: str,
    query_batch_size: int,
    backend: str,
    n_train: int,
    split: dict,
    **extra,
) -> dict:
    meta = {
        "data": data,
        "dataset_spec": dict(spec_summary(spec)),
        "feature_set": feature_set,
        "feature_names": feature_names,
        "k": k,
        "metric": metric,
        "query_batch_size": query_batch_size,
        "backend": backend,
        "id_space": "train_split_local_edge_id",
        "n_train": int(n_train),
        "split_buckets": split,
        "label_free": True,
    }
    meta.update(extra)
    return meta
