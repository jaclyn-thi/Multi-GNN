"""Causal / label-free richer transaction features for offline KNN (richer_v1).

All temporal and history features are computed in a single timestamp sort with
per-account and per-pair running state — O(n log n) time, O(n + u) memory where
u is the number of unique directed pairs observed (no dense N×N structures).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


def resolve_amount_column(df: pd.DataFrame) -> str:
    for col in ("Amount Received", "Amount Sent", "amount"):
        if col in df.columns:
            return col
    raise ValueError("No amount column found (expected Amount Received or Amount Sent)")


def _safe_log1p_arr(arr: np.ndarray) -> np.ndarray:
    return np.log1p(np.maximum(arr.astype(np.float64), 0.0)).astype(np.float32)


@dataclass
class _AccountState:
    last_ts: float = float("-inf")
    tx_count: int = 0
    amount_sum: float = 0.0
    out_count: int = 0
    out_amount: float = 0.0
    in_count: int = 0
    in_amount: float = 0.0
    out_counterparties: Set[int] = field(default_factory=set)
    in_counterparties: Set[int] = field(default_factory=set)


@dataclass
class _PairState:
    count: int = 0
    amount_sum: float = 0.0


def compute_causal_edge_stats(df: pd.DataFrame, amount_col: str) -> Dict[str, np.ndarray]:
    """Past-only account/pair statistics in global timestamp order.

    For each edge, features reflect history strictly before that edge is applied.
    """
    n = int(len(df))
    ts = df["Timestamp"].astype(np.float64).to_numpy()
    from_ids = df["from_id"].astype(np.int64).to_numpy()
    to_ids = df["to_id"].astype(np.int64).to_numpy()
    amounts = np.maximum(df[amount_col].astype(np.float64).to_numpy(), 0.0)

    order = np.argsort(ts, kind="mergesort")

    sender_dt = np.zeros(n, dtype=np.float32)
    receiver_dt = np.zeros(n, dtype=np.float32)
    sender_past_tx = np.zeros(n, dtype=np.float32)
    receiver_past_tx = np.zeros(n, dtype=np.float32)
    sender_past_out_count = np.zeros(n, dtype=np.float32)
    sender_past_in_count = np.zeros(n, dtype=np.float32)
    receiver_past_out_count = np.zeros(n, dtype=np.float32)
    receiver_past_in_count = np.zeros(n, dtype=np.float32)
    sender_past_out_amount = np.zeros(n, dtype=np.float32)
    sender_past_in_amount = np.zeros(n, dtype=np.float32)
    receiver_past_out_amount = np.zeros(n, dtype=np.float32)
    receiver_past_in_amount = np.zeros(n, dtype=np.float32)
    sender_distinct_out = np.zeros(n, dtype=np.float32)
    sender_distinct_in = np.zeros(n, dtype=np.float32)
    receiver_distinct_out = np.zeros(n, dtype=np.float32)
    receiver_distinct_in = np.zeros(n, dtype=np.float32)
    sender_past_mean_amount = np.zeros(n, dtype=np.float32)
    receiver_past_mean_amount = np.zeros(n, dtype=np.float32)
    amount_vs_sender_past_mean = np.zeros(n, dtype=np.float32)
    amount_vs_receiver_past_mean = np.zeros(n, dtype=np.float32)
    amount_vs_sender_past_out = np.zeros(n, dtype=np.float32)
    amount_vs_receiver_past_in = np.zeros(n, dtype=np.float32)
    pair_past_count = np.zeros(n, dtype=np.float32)
    pair_past_amount = np.zeros(n, dtype=np.float32)
    pair_is_new = np.ones(n, dtype=np.float32)

    accounts: Dict[int, _AccountState] = {}
    pairs: Dict[Tuple[int, int], _PairState] = {}
    eps = 1e-8

    for pos in order:
        s = int(from_ids[pos])
        r = int(to_ids[pos])
        t = float(ts[pos])
        amt = float(amounts[pos])

        s_state = accounts.get(s)
        r_state = accounts.get(r)

        if s_state is not None and s_state.last_ts > float("-inf"):
            sender_dt[pos] = np.log1p(max(t - s_state.last_ts, 0.0))
        if r_state is not None and r_state.last_ts > float("-inf"):
            receiver_dt[pos] = np.log1p(max(t - r_state.last_ts, 0.0))

        if s_state is not None:
            sender_past_tx[pos] = float(s_state.tx_count)
            sender_past_out_count[pos] = float(s_state.out_count)
            sender_past_in_count[pos] = float(s_state.in_count)
            sender_past_out_amount[pos] = s_state.out_amount
            sender_past_in_amount[pos] = s_state.in_amount
            sender_distinct_out[pos] = float(len(s_state.out_counterparties))
            sender_distinct_in[pos] = float(len(s_state.in_counterparties))
            if s_state.tx_count > 0:
                sender_past_mean_amount[pos] = s_state.amount_sum / s_state.tx_count
                amount_vs_sender_past_mean[pos] = amt / (sender_past_mean_amount[pos] + eps)
            amount_vs_sender_past_out[pos] = amt / (s_state.out_amount + eps)

        if r_state is not None:
            receiver_past_tx[pos] = float(r_state.tx_count)
            receiver_past_out_count[pos] = float(r_state.out_count)
            receiver_past_in_count[pos] = float(r_state.in_count)
            receiver_past_out_amount[pos] = r_state.out_amount
            receiver_past_in_amount[pos] = r_state.in_amount
            receiver_distinct_out[pos] = float(len(r_state.out_counterparties))
            receiver_distinct_in[pos] = float(len(r_state.in_counterparties))
            if r_state.tx_count > 0:
                receiver_past_mean_amount[pos] = r_state.amount_sum / r_state.tx_count
                amount_vs_receiver_past_mean[pos] = amt / (receiver_past_mean_amount[pos] + eps)
            amount_vs_receiver_past_in[pos] = amt / (r_state.in_amount + eps)

        pair_key = (s, r)
        p_state = pairs.get(pair_key)
        if p_state is not None:
            pair_past_count[pos] = float(p_state.count)
            pair_past_amount[pos] = p_state.amount_sum
            pair_is_new[pos] = 0.0
        else:
            pair_past_count[pos] = 0.0
            pair_past_amount[pos] = 0.0
            pair_is_new[pos] = 1.0

        if s_state is None:
            s_state = _AccountState()
            accounts[s] = s_state
        if r_state is None:
            r_state = _AccountState()
            accounts[r] = r_state

        s_state.last_ts = t
        s_state.tx_count += 1
        s_state.amount_sum += amt
        s_state.out_count += 1
        s_state.out_amount += amt
        s_state.out_counterparties.add(r)

        r_state.last_ts = t
        r_state.tx_count += 1
        r_state.amount_sum += amt
        r_state.in_count += 1
        r_state.in_amount += amt
        r_state.in_counterparties.add(s)

        if p_state is None:
            p_state = _PairState()
            pairs[pair_key] = p_state
        p_state.count += 1
        p_state.amount_sum += amt

    return {
        "sender_dt": sender_dt,
        "receiver_dt": receiver_dt,
        "sender_past_tx": sender_past_tx,
        "receiver_past_tx": receiver_past_tx,
        "sender_past_out_count": sender_past_out_count,
        "sender_past_in_count": sender_past_in_count,
        "receiver_past_out_count": receiver_past_out_count,
        "receiver_past_in_count": receiver_past_in_count,
        "sender_past_out_amount": sender_past_out_amount.astype(np.float32),
        "sender_past_in_amount": sender_past_in_amount.astype(np.float32),
        "receiver_past_out_amount": receiver_past_out_amount.astype(np.float32),
        "receiver_past_in_amount": receiver_past_in_amount.astype(np.float32),
        "sender_distinct_out": sender_distinct_out,
        "sender_distinct_in": sender_distinct_in,
        "receiver_distinct_out": receiver_distinct_out,
        "receiver_distinct_in": receiver_distinct_in,
        "sender_past_mean_amount": sender_past_mean_amount.astype(np.float32),
        "receiver_past_mean_amount": receiver_past_mean_amount.astype(np.float32),
        "amount_vs_sender_past_mean": np.log1p(amount_vs_sender_past_mean).astype(np.float32),
        "amount_vs_receiver_past_mean": np.log1p(amount_vs_receiver_past_mean).astype(np.float32),
        "amount_vs_sender_past_out": np.log1p(amount_vs_sender_past_out).astype(np.float32),
        "amount_vs_receiver_past_in": np.log1p(amount_vs_receiver_past_in).astype(np.float32),
        "pair_past_count": pair_past_count,
        "pair_past_amount": pair_past_amount,
        "pair_is_new": pair_is_new,
    }


def relative_amount_features(
    df: pd.DataFrame,
    amount_col: str,
    causal: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, List[str]]:
    amounts = np.maximum(df[amount_col].astype(np.float64).to_numpy(), 0.0)
    ranks = pd.Series(amounts).rank(method="average").to_numpy(dtype=np.float64)
    amount_rank = (ranks / max(float(len(ranks)), 1.0)).astype(np.float32)
    x = np.column_stack(
        [
            amount_rank,
            causal["amount_vs_sender_past_mean"],
            causal["amount_vs_receiver_past_mean"],
            causal["amount_vs_sender_past_out"],
            causal["amount_vs_receiver_past_in"],
        ]
    ).astype(np.float32)
    names = [
        "amount_train_rank",
        "log1p_amount_vs_sender_past_mean",
        "log1p_amount_vs_receiver_past_mean",
        "log1p_amount_vs_sender_past_out_total",
        "log1p_amount_vs_receiver_past_in_total",
    ]
    return x, names


def flow_rich_features(
    df: pd.DataFrame,
    amount_col: str,
    causal: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, List[str]]:
    from_ids = df["from_id"].astype(np.int64).to_numpy()
    to_ids = df["to_id"].astype(np.int64).to_numpy()
    amounts = np.maximum(df[amount_col].astype(np.float64).to_numpy(), 0.0)
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
            _safe_log1p_arr(s_out),
            _safe_log1p_arr(s_in),
            _safe_log1p_arr(r_in),
            _safe_log1p_arr(r_out),
            s_ratio,
            r_ratio,
            np.abs(s_ratio),
            np.abs(r_ratio),
            causal["amount_vs_sender_past_out"],
            causal["amount_vs_receiver_past_in"],
        ]
    ).astype(np.float32)
    names = [
        "log1p_sender_out_amount_train",
        "log1p_sender_in_amount_train",
        "log1p_receiver_in_amount_train",
        "log1p_receiver_out_amount_train",
        "sender_flow_balance_ratio_train",
        "receiver_flow_balance_ratio_train",
        "abs_sender_flow_balance_ratio_train",
        "abs_receiver_flow_balance_ratio_train",
        "log1p_amount_vs_sender_past_out_causal",
        "log1p_amount_vs_receiver_past_in_causal",
    ]
    return x, names


def temporal_causal_features(causal: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[str]]:
    x = np.column_stack(
        [
            causal["sender_dt"],
            causal["receiver_dt"],
            _safe_log1p_arr(causal["sender_past_tx"]),
            _safe_log1p_arr(causal["receiver_past_tx"]),
            _safe_log1p_arr(causal["sender_past_out_amount"]),
            _safe_log1p_arr(causal["receiver_past_in_amount"]),
        ]
    ).astype(np.float32)
    names = [
        "log1p_sender_interarrival",
        "log1p_receiver_interarrival",
        "log1p_sender_past_tx_count",
        "log1p_receiver_past_tx_count",
        "log1p_sender_past_out_amount_causal",
        "log1p_receiver_past_in_amount_causal",
    ]
    return x, names


def degree_causal_features(causal: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[str]]:
    x = np.column_stack(
        [
            _safe_log1p_arr(causal["sender_past_out_count"]),
            _safe_log1p_arr(causal["sender_past_in_count"]),
            _safe_log1p_arr(causal["receiver_past_out_count"]),
            _safe_log1p_arr(causal["receiver_past_in_count"]),
            _safe_log1p_arr(causal["sender_distinct_out"]),
            _safe_log1p_arr(causal["sender_distinct_in"]),
            _safe_log1p_arr(causal["receiver_distinct_out"]),
            _safe_log1p_arr(causal["receiver_distinct_in"]),
        ]
    ).astype(np.float32)
    names = [
        "log1p_sender_past_out_degree",
        "log1p_sender_past_in_degree",
        "log1p_receiver_past_out_degree",
        "log1p_receiver_past_in_degree",
        "log1p_sender_past_distinct_out_counterparties",
        "log1p_sender_past_distinct_in_counterparties",
        "log1p_receiver_past_distinct_out_counterparties",
        "log1p_receiver_past_distinct_in_counterparties",
    ]
    return x, names


def flow_causal_features(causal: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[str]]:
    x = np.column_stack(
        [
            causal["amount_vs_sender_past_out"],
            causal["amount_vs_receiver_past_in"],
        ]
    ).astype(np.float32)
    names = [
        "log1p_amount_vs_sender_past_out_causal",
        "log1p_amount_vs_receiver_past_in_causal",
    ]
    return x, names


def pair_history_features(causal: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[str]]:
    x = np.column_stack(
        [
            _safe_log1p_arr(causal["pair_past_count"]),
            _safe_log1p_arr(causal["pair_past_amount"]),
            causal["pair_is_new"],
        ]
    ).astype(np.float32)
    names = [
        "log1p_pair_past_tx_count",
        "log1p_pair_past_amount_total",
        "pair_is_new_counterparty",
    ]
    return x, names


def time_bucket_features(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    ts = df["Timestamp"].astype(np.float64).to_numpy()
    span = max(float(ts.max() - ts.min()), 1.0)
    ts_norm = ((ts - ts.min()) / span).astype(np.float32)
    # Generic bucket: hour-of-span and day-of-span (works for AML seconds and PaySim hourly steps).
    hour_bucket = ((ts % 3600.0) / 3600.0).astype(np.float32)
    day_bucket = ((ts % 86400.0) / 86400.0).astype(np.float32)
    x = np.column_stack([ts_norm, hour_bucket, day_bucket]).astype(np.float32)
    return x, ["timestamp_norm_train_span", "hour_of_day_fraction", "day_phase_fraction"]
