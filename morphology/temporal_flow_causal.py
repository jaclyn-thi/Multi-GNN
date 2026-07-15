"""Strictly causal temporal/flow features for downstream probing (temporal_flow_causal group).

Five features (see notes/morphology_temporal_flow_probe_plan.md):

1. log1p_sender_interarrival
2. log1p_receiver_interarrival
3. log1p_sender_past_7d_count
4. log1p_amount_vs_sender_past_mean
5. pair_repeat_indicator

Semantics align with ``transaction_knn.richer_features.compute_causal_edge_stats`` where
applicable, with two deliberate extensions for probe use:

- **Timestamp ties:** transactions sharing the same timestamp are featurized as a batch
  using history strictly before that timestamp; state is updated only after the entire
  batch is processed (same-timestamp edges do not influence one another).
- **Seven-day sender activity count:** ``log1p_sender_past_7d_count`` counts sender
  *outgoing* transactions with t' in (t - W, t), W = 604800 s by default.

Cross-split history: features are computed on the full dataset in global timestamp order.
Validation rows see training history; test rows see training + validation history. This
matches "available at transaction time" deployment and does not use labels.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from transaction_knn.richer_features import resolve_amount_column

TEMPORAL_FLOW_CAUSAL_WINDOW_7D_SEC = 604800.0

TEMPORAL_FLOW_CAUSAL_FEATURE_NAMES: Tuple[str, ...] = (
    "log1p_sender_interarrival",
    "log1p_receiver_interarrival",
    "log1p_sender_past_7d_count",
    "log1p_amount_vs_sender_past_mean",
    "pair_repeat_indicator",
)

TEMPORAL_FLOW_CAUSAL_DEFINITIONS: Dict[str, str] = {
    "log1p_sender_interarrival": (
        "log1p(t - t_prev) where t_prev is the sender's last outgoing/incoming "
        "transaction timestamp strictly before t; 0 if none."
    ),
    "log1p_receiver_interarrival": (
        "log1p(t - t_prev) for the receiver account; 0 if none."
    ),
    "log1p_sender_past_7d_count": (
        "log1p(count) where count is the number of sender outgoing transactions "
        "with timestamp t' in (t - W, t), W=604800 s; 0 if none."
    ),
    "log1p_amount_vs_sender_past_mean": (
        "log1p(a / (mean past sender transaction amount + eps)); 0 if no sender history."
    ),
    "pair_repeat_indicator": (
        "1 if a prior edge with the same ordered (sender, receiver) exists at t' < t; else 0."
    ),
}


@dataclass
class _AccountState:
    last_ts: float = float("-inf")
    tx_count: int = 0
    amount_sum: float = 0.0
    out_timestamps: List[float] = field(default_factory=list)


@dataclass
class _PairState:
    count: int = 0


def _count_in_open_window(sorted_ts: Sequence[float], t: float, window_sec: float) -> int:
    """Count timestamps t' with t - window_sec < t' < t (strictly before t)."""
    if not sorted_ts:
        return 0
    lo = bisect.bisect_right(sorted_ts, t - window_sec)
    hi = bisect.bisect_left(sorted_ts, t)
    return max(hi - lo, 0)


def compute_temporal_flow_causal_features(
    df: pd.DataFrame,
    *,
    amount_col: Optional[str] = None,
    window_7d_sec: float = TEMPORAL_FLOW_CAUSAL_WINDOW_7D_SEC,
) -> Tuple[np.ndarray, List[str]]:
    """Compute the five-feature temporal_flow_causal matrix in CSV row order.

    Returns
    -------
    features : (n_rows, 5) float32
    names : list of feature names (fixed order)
    """
    if amount_col is None:
        amount_col = resolve_amount_column(df)

    n = int(len(df))
    ts = df["Timestamp"].astype(np.float64).to_numpy()
    from_ids = df["from_id"].astype(np.int64).to_numpy()
    to_ids = df["to_id"].astype(np.int64).to_numpy()
    amounts = np.maximum(df[amount_col].astype(np.float64).to_numpy(), 0.0)

    order = np.argsort(ts, kind="mergesort")

    out = np.zeros((n, len(TEMPORAL_FLOW_CAUSAL_FEATURE_NAMES)), dtype=np.float32)
    accounts: Dict[int, _AccountState] = {}
    pairs: Dict[Tuple[int, int], _PairState] = {}
    eps = 1e-8

    pos = 0
    while pos < n:
        t0 = float(ts[order[pos]])
        batch_end = pos + 1
        while batch_end < n and float(ts[order[batch_end]]) == t0:
            batch_end += 1
        batch = order[pos:batch_end]

        for idx in batch:
            s = int(from_ids[idx])
            r = int(to_ids[idx])
            t = float(ts[idx])
            amt = float(amounts[idx])

            s_state = accounts.get(s)
            r_state = accounts.get(r)

            if s_state is not None and s_state.last_ts > float("-inf"):
                out[idx, 0] = np.log1p(max(t - s_state.last_ts, 0.0))
            if r_state is not None and r_state.last_ts > float("-inf"):
                out[idx, 1] = np.log1p(max(t - r_state.last_ts, 0.0))

            if s_state is not None:
                cnt_7d = _count_in_open_window(s_state.out_timestamps, t, window_7d_sec)
                out[idx, 2] = np.log1p(float(cnt_7d))
                if s_state.tx_count > 0:
                    mean_amt = s_state.amount_sum / s_state.tx_count
                    out[idx, 3] = np.log1p(amt / (mean_amt + eps))

            pair_key = (s, r)
            p_state = pairs.get(pair_key)
            out[idx, 4] = 0.0 if p_state is None or p_state.count == 0 else 1.0

        for idx in batch:
            s = int(from_ids[idx])
            r = int(to_ids[idx])
            t = float(ts[idx])
            amt = float(amounts[idx])

            s_state = accounts.get(s)
            if s_state is None:
                s_state = _AccountState()
                accounts[s] = s_state
            r_state = accounts.get(r)
            if r_state is None:
                r_state = _AccountState()
                accounts[r] = r_state

            s_state.last_ts = t
            s_state.tx_count += 1
            s_state.amount_sum += amt
            bisect.insort(s_state.out_timestamps, t)

            r_state.last_ts = t
            r_state.tx_count += 1
            r_state.amount_sum += amt

            pair_key = (s, r)
            p_state = pairs.get(pair_key)
            if p_state is None:
                p_state = _PairState()
                pairs[pair_key] = p_state
            p_state.count += 1

        pos = batch_end

    return out, list(TEMPORAL_FLOW_CAUSAL_FEATURE_NAMES)


def feature_summary_stats(features: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Per-column min/median/max and zero-fraction (useful for cache metadata)."""
    stats: Dict[str, Dict[str, float]] = {}
    for j, name in enumerate(TEMPORAL_FLOW_CAUSAL_FEATURE_NAMES):
        col = features[:, j].astype(np.float64)
        stats[name] = {
            "min": float(np.min(col)),
            "median": float(np.median(col)),
            "max": float(np.max(col)),
            "zero_fraction": float(np.mean(col == 0.0)),
            "nan_count": int(np.isnan(col).sum()),
            "inf_count": int(np.isinf(col).sum()),
        }
    return stats
