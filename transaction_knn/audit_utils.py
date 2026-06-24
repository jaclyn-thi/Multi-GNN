"""Shared helpers for transaction KNN audit scripts (50k CPU audits)."""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

from transaction_knn.backends import SklearnKNNBackend, _l2_normalize
from transaction_knn.features import (
    build_features_detailed,
    prepare_knn_features,
    standardize_features,
)


def build_audit_matrix(
    df_train,
    *,
    feature_set: str,
    categorical_encoding: str,
    scaling: str,
    metric: str,
    l2_normalize: bool = True,
    apply_group_weights: bool = False,
    include_pair_history: Optional[bool] = None,
) -> Tuple[np.ndarray, object]:
    detail = build_features_detailed(
        df_train,
        feature_set,
        categorical_encoding=categorical_encoding,
        include_pair_history=include_pair_history,
        scaling="none",
    )
    features = detail.features
    if scaling == "legacy_standard":
        features = standardize_features(features)
        if metric == "cosine" and l2_normalize:
            features = _l2_normalize(features)
    elif scaling in {"standard", "robust"}:
        weights = detail.group_weights if apply_group_weights else None
        features = prepare_knn_features(
            features,
            detail.group_slices,
            scaling=scaling,
            group_weights=weights,
            l2_normalize=l2_normalize,
        )
    elif scaling == "none":
        if metric == "cosine" and l2_normalize:
            features = _l2_normalize(features)
    else:
        raise ValueError(f"Unsupported scaling={scaling!r}")
    return features.astype(np.float32), detail


def ordinary_knn(features: np.ndarray, k: int, metric: str) -> Tuple[np.ndarray, np.ndarray]:
    backend = SklearnKNNBackend(metric=metric)
    backend.fit(features, k=k)
    query_idx = np.arange(features.shape[0], dtype=np.int64)
    return backend.query(query_idx, k)


def mutual_knn_neighbors(neighbor_ids: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    n, k = neighbor_ids.shape
    forward = set()
    for i in range(n):
        for j in neighbor_ids[i]:
            if j >= 0:
                forward.add((i, int(j)))
    mutual = np.full((n, k), -1, dtype=np.int64)
    anchors_with_any = 0
    for i in range(n):
        picked = []
        for j in neighbor_ids[i]:
            if j >= 0 and (int(j), i) in forward:
                picked.append(int(j))
        if picked:
            anchors_with_any += 1
        for t, j in enumerate(picked[:k]):
            mutual[i, t] = j
    stats = {
        "mutual_anchors_with_any": float(anchors_with_any),
        "mutual_anchor_coverage": float(anchors_with_any) / max(n, 1),
        "mutual_avg_neighbors_per_anchor": float(np.mean((mutual >= 0).sum(axis=1))),
    }
    return mutual, stats


def hub_filter_neighbors(
    neighbor_ids: np.ndarray,
    freq_fraction: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Drop neighbor ids appearing in more than ``freq_fraction`` of all neighbor slots."""
    n, k = neighbor_ids.shape
    flat = neighbor_ids.reshape(-1)
    valid = flat[flat >= 0]
    counts = Counter(int(x) for x in valid)
    max_slots = max(int(n * k), 1)
    max_count = max(1, int(np.ceil(freq_fraction * max_slots)))
    banned = {nid for nid, c in counts.items() if c > max_count}
    out = np.full_like(neighbor_ids, -1)
    anchors_lost_all = 0
    for i in range(n):
        kept = [int(j) for j in neighbor_ids[i] if j >= 0 and int(j) not in banned]
        if not kept:
            anchors_lost_all += 1
        for t, j in enumerate(kept[:k]):
            out[i, t] = j
    stats = {
        "hub_filter_freq_fraction": float(freq_fraction),
        "hub_filter_max_count": float(max_count),
        "hub_filter_banned_neighbors": float(len(banned)),
        "hub_filter_anchors_lost_all": float(anchors_lost_all),
        "hub_filter_anchors_lost_all_fraction": float(anchors_lost_all) / max(n, 1),
    }
    return out, stats


def endpoint_overlap(df, anchor_idx: np.ndarray, neighbor_idx: np.ndarray) -> Dict[str, float]:
    from_ids = df["from_id"].to_numpy()
    to_ids = df["to_id"].to_numpy()
    a_src = from_ids[anchor_idx]
    a_dst = to_ids[anchor_idx]
    n_src = from_ids[neighbor_idx]
    n_dst = to_ids[neighbor_idx]
    return {
        "same_sender": float((a_src == n_src).mean()) if anchor_idx.size else float("nan"),
        "same_receiver": float((a_dst == n_dst).mean()) if anchor_idx.size else float("nan"),
        "same_pair": float(((a_src == n_src) & (a_dst == n_dst)).mean()) if anchor_idx.size else float("nan"),
    }


def neighbor_diversity(neighbor_ids: np.ndarray) -> Dict[str, float]:
    flat = neighbor_ids.reshape(-1)
    flat = flat[flat >= 0]
    if flat.size == 0:
        return {
            "unique_neighbor_fraction": float("nan"),
            "duplicate_neighbor_rows": 0.0,
            "hubness_top1_neighbor_share": float("nan"),
            "hubness_top10_neighbors_share": float("nan"),
            "hubness_unique_neighbors": 0.0,
        }
    counts = Counter(int(x) for x in flat)
    total = float(flat.size)
    unique_frac = float(len(counts)) / total
    row_dup = float(sum(len(row) != len({int(x) for x in row if x >= 0}) for row in neighbor_ids))
    top_counts = sorted(counts.values(), reverse=True)
    return {
        "unique_neighbor_fraction": unique_frac,
        "duplicate_neighbor_rows": row_dup,
        "hubness_top1_neighbor_share": float(top_counts[0]) / total,
        "hubness_top10_neighbors_share": float(sum(top_counts[:10])) / total,
        "hubness_unique_neighbors": float(len(counts)),
    }


def value_stats(values: np.ndarray, *, prefix: str) -> Dict[str, float]:
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        return {f"{prefix}_{p}": float("nan") for p in ("min", "mean", "p50", "p90", "p95", "p99", "max")}
    return {
        f"{prefix}_min": float(np.min(vals)),
        f"{prefix}_mean": float(np.mean(vals)),
        f"{prefix}_p50": float(np.percentile(vals, 50)),
        f"{prefix}_p90": float(np.percentile(vals, 90)),
        f"{prefix}_p95": float(np.percentile(vals, 95)),
        f"{prefix}_p99": float(np.percentile(vals, 99)),
        f"{prefix}_max": float(np.max(vals)),
    }


def audit_neighbor_set(
    df_train,
    neighbor_ids: np.ndarray,
    neighbor_values: np.ndarray,
    *,
    label_col: str,
    value_prefix: str = "sim",
) -> Dict[str, object]:
    k = neighbor_ids.shape[1]
    query_idx = np.arange(neighbor_ids.shape[0], dtype=np.int64)
    flat_a = np.repeat(query_idx, k)
    flat_n = neighbor_ids.reshape(-1)
    valid = flat_n >= 0
    report: Dict[str, object] = {
        **value_stats(neighbor_values, prefix=value_prefix),
        **neighbor_diversity(neighbor_ids),
    }
    if valid.any():
        report.update(endpoint_overlap(df_train, flat_a[valid], flat_n[valid]))
    if label_col in df_train.columns:
        labels = df_train[label_col].to_numpy()
        report["label_same_fraction"] = float((labels[flat_a[valid]] == labels[flat_n[valid]]).mean())
        report["neighbor_positive_rate"] = float(labels[flat_n[valid]].mean())
        report["label_enrichment_note"] = "analysis only — not used in training"
    return report


def pairwise_jaccard(ids_a: np.ndarray, ids_b: np.ndarray) -> float:
    overlap = []
    for row_a, row_b in zip(ids_a, ids_b):
        sa = {int(x) for x in row_a if int(x) >= 0}
        sb = {int(x) for x in row_b if int(x) >= 0}
        if not sa and not sb:
            overlap.append(1.0)
        elif not sa or not sb:
            overlap.append(0.0)
        else:
            overlap.append(len(sa & sb) / len(sa | sb))
    return float(np.mean(overlap))
