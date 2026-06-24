"""Label-free edge-drop importance scores for contrastive graph augmentations.

Scores are computed on the **train split only** from transaction metadata
(amount, endpoints, degrees). No labels, typology, or val/test statistics.

Caches are keyed by train split-local ``edge_id`` (0…N−1 after ``add_arange_ids``).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from transaction_knn.richer_features import resolve_amount_column

FORBIDDEN_COLUMNS = frozenset(
    {
        "Is Laundering",
        "is_laundering",
        "label",
        "fraud",
        "pattern_type",
        "pattern_detail",
        "laundering_attempt",
    }
)

EDGE_DROP_POLICIES = ("random", "degree_aware", "degree_flow_aware")


def percentile_rank(values: np.ndarray) -> np.ndarray:
    """Return average percentile ranks in [0, 1] (higher = larger value)."""
    arr = np.asarray(values, dtype=np.float64)
    n = arr.shape[0]
    if n <= 1:
        return np.zeros(n, dtype=np.float32)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    return (ranks / float(n - 1)).astype(np.float32)


def _degree_arrays(df_train: pd.DataFrame) -> Dict[str, np.ndarray]:
    from_ids = df_train["from_id"].astype(np.int64).to_numpy()
    to_ids = df_train["to_id"].astype(np.int64).to_numpy()
    max_node = int(max(from_ids.max(initial=0), to_ids.max(initial=0))) + 1
    out_deg = np.bincount(from_ids, minlength=max_node).astype(np.float64)
    in_deg = np.bincount(to_ids, minlength=max_node).astype(np.float64)
    sender_total = out_deg[from_ids] + in_deg[from_ids]
    receiver_total = out_deg[to_ids] + in_deg[to_ids]
    return {
        "from_ids": from_ids,
        "to_ids": to_ids,
        "out_deg": out_deg,
        "in_deg": in_deg,
        "sender_total": sender_total.astype(np.float32),
        "receiver_total": receiver_total.astype(np.float32),
        "max_endpoint": np.maximum(sender_total, receiver_total).astype(np.float32),
        "min_endpoint": np.minimum(sender_total, receiver_total).astype(np.float32),
    }


def compute_degree_aware_routine(df_train: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    deg = _degree_arrays(df_train)
    sender_pct = percentile_rank(deg["sender_total"])
    receiver_pct = percentile_rank(deg["receiver_total"])
    max_pct = percentile_rank(deg["max_endpoint"])
    min_pct = percentile_rank(deg["min_endpoint"])

    routine = 0.40 * max_pct + 0.30 * sender_pct + 0.30 * receiver_pct
    routine = routine - 0.25 * (1.0 - min_pct)
    routine = routine.astype(np.float32)

    aux = {
        "degree_pct": max_pct,
        "sender_degree_pct": sender_pct,
        "receiver_degree_pct": receiver_pct,
        "min_endpoint_degree_pct": min_pct,
    }
    return routine, aux


def compute_degree_flow_aware_routine(df_train: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    routine_deg, aux_deg = compute_degree_aware_routine(df_train)
    amount_col = resolve_amount_column(df_train)
    amounts = np.maximum(df_train[amount_col].astype(np.float64).to_numpy(), 0.0)
    log_amount = np.log1p(amounts).astype(np.float32)

    deg = _degree_arrays(df_train)
    eps = 1e-8
    amounts_f = amounts.astype(np.float64)
    amount_out = np.bincount(deg["from_ids"], weights=amounts_f, minlength=deg["out_deg"].shape[0])
    amount_in = np.bincount(deg["to_ids"], weights=amounts_f, minlength=deg["out_deg"].shape[0])
    s_out = amount_out[deg["from_ids"]]
    s_in = amount_in[deg["from_ids"]]
    r_in = amount_in[deg["to_ids"]]

    rel_sender_out = np.log1p(amounts / (s_out + eps)).astype(np.float32)
    rel_receiver_in = np.log1p(amounts / (r_in + eps)).astype(np.float32)
    flow_imbalance = (np.abs(s_out - s_in) / (s_out + s_in + eps)).astype(np.float32)

    pair_df = pd.DataFrame({"from_id": deg["from_ids"], "to_id": deg["to_ids"]})
    pair_count = pair_df.groupby(["from_id", "to_id"], sort=False).transform("size").to_numpy().astype(np.float32)

    amount_pct = percentile_rank(log_amount)
    rel_sender_out_pct = percentile_rank(rel_sender_out)
    rel_receiver_in_pct = percentile_rank(rel_receiver_in)
    flow_imbalance_pct = percentile_rank(flow_imbalance)
    pair_rarity = 1.0 - percentile_rank(pair_count)

    importance_flow = (
        0.30 * amount_pct
        + 0.25 * rel_sender_out_pct
        + 0.20 * rel_receiver_in_pct
        + 0.15 * flow_imbalance_pct
        + 0.10 * pair_rarity
    )
    routine_flow = (1.0 - importance_flow).astype(np.float32)
    routine = (0.55 * routine_deg + 0.45 * routine_flow).astype(np.float32)

    aux = dict(aux_deg)
    aux.update(
        {
            "amount_pct": amount_pct,
            "rel_sender_out_pct": rel_sender_out_pct,
            "rel_receiver_in_pct": rel_receiver_in_pct,
            "flow_imbalance_pct": flow_imbalance_pct,
            "pair_rarity": pair_rarity,
        }
    )
    return routine, aux


def calibrate_drop_probabilities(
    routine: np.ndarray,
    *,
    target_rate: float,
    alpha: float,
    min_prob: float,
    max_prob: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Map routine scores (higher = more droppable) to per-edge drop probabilities."""
    if not (0.0 <= target_rate <= 1.0):
        raise ValueError(f"target_rate must be in [0, 1], got {target_rate}")
    if not (0.0 <= min_prob <= max_prob <= 1.0):
        raise ValueError(f"invalid clip bounds: min={min_prob}, max={max_prob}")
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}")

    routine = np.asarray(routine, dtype=np.float64)
    z = alpha * (routine - routine.mean()) / (routine.std() + 1e-8)

    lo, hi = -30.0, 30.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        p = 1.0 / (1.0 + np.exp(-(z + mid)))
        if float(p.mean()) < target_rate:
            lo = mid
        else:
            hi = mid
    bias = 0.5 * (lo + hi)
    drop_prob = 1.0 / (1.0 + np.exp(-(z + bias)))
    clipped_low = drop_prob < min_prob
    clipped_high = drop_prob > max_prob
    drop_prob = np.clip(drop_prob, min_prob, max_prob)

  # Light mean correction after clipping (single scale, no dense ops)
    mean_p = float(drop_prob.mean())
    if mean_p > 1e-8 and abs(mean_p - target_rate) > 1e-3:
        scale = np.clip(target_rate / mean_p, 0.5, 2.0)
        drop_prob = np.clip(drop_prob * scale, min_prob, max_prob)

    importance = (1.0 - percentile_rank(routine)).astype(np.float32)
    meta = {
        "target_drop_rate": float(target_rate),
        "realized_drop_rate_preclip": float((1.0 / (1.0 + np.exp(-(z + bias)))).mean()),
        "realized_drop_rate": float(drop_prob.mean()),
        "drop_prob_min": float(drop_prob.min()),
        "drop_prob_mean": float(drop_prob.mean()),
        "drop_prob_max": float(drop_prob.max()),
        "clip_fraction_low": float(clipped_low.mean()),
        "clip_fraction_high": float(clipped_high.mean()),
        "calibration_bias": float(bias),
        "importance_alpha": float(alpha),
    }
    return drop_prob.astype(np.float32), importance, meta


def label_free_train_frame(df_train: pd.DataFrame) -> pd.DataFrame:
    """Return a copy without label / typology columns (may be present in CSV)."""
    forbidden_lower = {c.lower() for c in FORBIDDEN_COLUMNS}
    cols = [c for c in df_train.columns if c.lower() not in forbidden_lower]
    return df_train[cols].copy()


def assert_label_free_inputs(df_train: pd.DataFrame) -> None:
    """Sanity check that scoring frame has no forbidden columns after stripping."""
    forbidden_lower = {c.lower() for c in FORBIDDEN_COLUMNS}
    for col in df_train.columns:
        if col.lower() in forbidden_lower:
            raise ValueError(f"Label leakage guard: forbidden column {col!r} remains in scoring frame")


def compute_edge_drop_scores(
    df_train: pd.DataFrame,
    policy: str,
    *,
    target_rate: float = 0.1,
    alpha: float = 2.0,
    min_prob: float = 0.01,
    max_prob: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, object]]:
    if policy not in EDGE_DROP_POLICIES:
        raise ValueError(f"Unknown edge_drop_policy={policy!r}")
    if policy == "random":
        raise ValueError("compute_edge_drop_scores does not apply to policy='random'")

    df = label_free_train_frame(df_train)
    assert_label_free_inputs(df)

    if policy == "degree_aware":
        routine, aux = compute_degree_aware_routine(df)
    else:
        routine, aux = compute_degree_flow_aware_routine(df)

    drop_prob, importance, cal_meta = calibrate_drop_probabilities(
        routine,
        target_rate=target_rate,
        alpha=alpha,
        min_prob=min_prob,
        max_prob=max_prob,
    )
    metadata = {
        "policy": policy,
        "n_train_edges": int(len(df)),
        "label_columns_used": [],
        "leakage_policy": "train_split_only_static_degrees_and_amounts",
        "calibration": cal_meta,
    }
    return drop_prob, importance, aux, metadata


@dataclass
class EdgeDropScoreCache:
    edge_ids: np.ndarray
    drop_prob: np.ndarray
    importance: np.ndarray
    degree_pct: np.ndarray
    amount_pct: Optional[np.ndarray]
    flow_imbalance_pct: Optional[np.ndarray]
    policy: str
    target_drop_rate: float
    metadata: Dict[str, object] = field(default_factory=dict)

    def lookup_drop_prob(self, edge_ids: torch.Tensor, device: torch.device) -> torch.Tensor:
        ids = edge_ids.detach().long().cpu().numpy()
        out = np.full(ids.shape, self.target_drop_rate, dtype=np.float32)
        valid = (ids >= 0) & (ids < self.drop_prob.shape[0])
        if valid.any():
            out[valid] = self.drop_prob[ids[valid]]
        return torch.from_numpy(out).to(device=device, non_blocking=True)

    def lookup_bucket_values(self, edge_ids: torch.Tensor, field: str) -> Optional[np.ndarray]:
        field_map = {
            "degree_pct": self.degree_pct,
            "amount_pct": self.amount_pct,
            "flow_imbalance_pct": self.flow_imbalance_pct,
        }
        arr = field_map.get(field)
        if arr is None:
            return None
        ids = edge_ids.detach().long().cpu().numpy()
        out = np.full(ids.shape, np.nan, dtype=np.float32)
        valid = (ids >= 0) & (ids < arr.shape[0])
        if valid.any():
            out[valid] = arr[ids[valid]]
        return out


def save_edge_drop_cache(path: str | Path, cache: EdgeDropScoreCache) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        edge_ids=cache.edge_ids.astype(np.int64),
        drop_prob=cache.drop_prob.astype(np.float32),
        importance=cache.importance.astype(np.float32),
        degree_pct=cache.degree_pct.astype(np.float32),
        amount_pct=cache.amount_pct.astype(np.float32) if cache.amount_pct is not None else np.array([], dtype=np.float32),
        flow_imbalance_pct=cache.flow_imbalance_pct.astype(np.float32)
        if cache.flow_imbalance_pct is not None
        else np.array([], dtype=np.float32),
        policy=np.array([cache.policy]),
        target_drop_rate=np.array([cache.target_drop_rate], dtype=np.float32),
        metadata_json=np.array([json.dumps(cache.metadata)]),
    )


def load_edge_drop_cache(path: str | Path) -> EdgeDropScoreCache:
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        amount = z["amount_pct"]
        flow = z["flow_imbalance_pct"]
        meta_raw = z["metadata_json"]
        metadata = json.loads(str(meta_raw[0]) if meta_raw.ndim else str(meta_raw.item()))
        return EdgeDropScoreCache(
            edge_ids=z["edge_ids"].astype(np.int64),
            drop_prob=z["drop_prob"].astype(np.float32),
            importance=z["importance"].astype(np.float32),
            degree_pct=z["degree_pct"].astype(np.float32),
            amount_pct=amount.astype(np.float32) if amount.size else None,
            flow_imbalance_pct=flow.astype(np.float32) if flow.size else None,
            policy=str(z["policy"][0]) if z["policy"].ndim else str(z["policy"].item()),
            target_drop_rate=float(z["target_drop_rate"][0]),
            metadata=metadata,
        )


def build_edge_drop_cache(
    df_train: pd.DataFrame,
    policy: str,
    *,
    target_rate: float = 0.1,
    alpha: float = 2.0,
    min_prob: float = 0.01,
    max_prob: float = 0.95,
) -> EdgeDropScoreCache:
    drop_prob, importance, aux, metadata = compute_edge_drop_scores(
        df_train,
        policy,
        target_rate=target_rate,
        alpha=alpha,
        min_prob=min_prob,
        max_prob=max_prob,
    )
    n = len(df_train)
    return EdgeDropScoreCache(
        edge_ids=np.arange(n, dtype=np.int64),
        drop_prob=drop_prob,
        importance=importance,
        degree_pct=aux["degree_pct"],
        amount_pct=aux.get("amount_pct"),
        flow_imbalance_pct=aux.get("flow_imbalance_pct"),
        policy=policy,
        target_drop_rate=target_rate,
        metadata=metadata,
    )


def load_or_build_edge_drop_cache(args, data_config) -> EdgeDropScoreCache:
    from transaction_knn.features import load_train_frame

    policy = getattr(args, "edge_drop_policy", "random")
    if policy == "random":
        raise ValueError("load_or_build_edge_drop_cache called with policy=random")

    cache_path = getattr(args, "edge_drop_score_cache_path", None)
    if cache_path:
        logging.info("Loading edge-drop score cache from %s", cache_path)
        cache = load_edge_drop_cache(cache_path)
        if cache.policy != policy:
            raise ValueError(f"Cache policy {cache.policy!r} != requested {policy!r}")
        return cache

    logging.info("Building edge-drop score cache in memory (policy=%s)", policy)
    _, df_train, _, _ = load_train_frame(args.data, data_config)
    cache = build_edge_drop_cache(
        df_train,
        policy,
        target_rate=float(getattr(args, "edge_drop_target_rate", 0.1)),
        alpha=float(getattr(args, "edge_drop_importance_alpha", 2.0)),
        min_prob=float(getattr(args, "edge_drop_min_prob", 0.01)),
        max_prob=float(getattr(args, "edge_drop_max_prob", 0.95)),
    )
    if cache_path:
        save_edge_drop_cache(cache_path, cache)
    return cache


def audit_score_distribution(cache: EdgeDropScoreCache) -> Dict[str, object]:
    """Summarize score / probability distributions for pre-training audit."""
    imp = cache.importance
    prob = cache.drop_prob
    corr = float(np.corrcoef(imp, prob)[0, 1]) if imp.size > 1 else float("nan")
    hi = imp >= np.quantile(imp, 0.8)
    lo = imp <= np.quantile(imp, 0.2)
    return {
        "policy": cache.policy,
        "n_edges": int(prob.shape[0]),
        "target_drop_rate": cache.target_drop_rate,
        "drop_prob_mean": float(prob.mean()),
        "drop_prob_std": float(prob.std()),
        "drop_prob_min": float(prob.min()),
        "drop_prob_max": float(prob.max()),
        "importance_mean": float(imp.mean()),
        "importance_importance_prob_corr": corr,
        "mean_drop_prob_top20_importance": float(prob[hi].mean()) if hi.any() else float("nan"),
        "mean_drop_prob_bottom20_importance": float(prob[lo].mean()) if lo.any() else float("nan"),
        "labels_used": [],
        "metadata": cache.metadata,
    }
