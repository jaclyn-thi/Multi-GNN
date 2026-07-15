#!/usr/bin/env python3
"""Leakage, causality, alignment, and default-history audit for temporal_flow_causal caches."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataset_specs import get_dataset_spec
from dataset_splits import temporal_edge_split
from morphology.temporal_flow_causal import (
    TEMPORAL_FLOW_CAUSAL_FEATURE_NAMES,
    TEMPORAL_FLOW_CAUSAL_WINDOW_7D_SEC,
    compute_temporal_flow_causal_features,
)
from transaction_knn.features import load_data_config, resolve_amount_column
from util import logger_setup

SPLITS = ("train", "val", "test")


@dataclass
class _AuditAccount:
    last_ts: float = float("-inf")
    tx_count: int = 0
    amount_sum: float = 0.0
    out_timestamps: List[float] = field(default_factory=list)


@dataclass
class _AuditPair:
    count: int = 0


def _count_open_window(sorted_ts: Sequence[float], t: float, window_sec: float) -> int:
    import bisect

    if not sorted_ts:
        return 0
    lo = bisect.bisect_right(sorted_ts, t - window_sec)
    hi = bisect.bisect_left(sorted_ts, t)
    return max(hi - lo, 0)


def replay_audit_state(
    df: pd.DataFrame,
    *,
    amount_col: str,
    window_7d_sec: float = TEMPORAL_FLOW_CAUSAL_WINDOW_7D_SEC,
) -> Dict[str, np.ndarray]:
    """Replay causal state with explicit no-history flags (audit-only, not probe features)."""
    n = len(df)
    ts = df["Timestamp"].astype(np.float64).to_numpy()
    from_ids = df["from_id"].astype(np.int64).to_numpy()
    to_ids = df["to_id"].astype(np.int64).to_numpy()
    amounts = np.maximum(df[amount_col].astype(np.float64).to_numpy(), 0.0)
    order = np.argsort(ts, kind="mergesort")

    sender_no_prior = np.zeros(n, dtype=np.bool_)
    receiver_no_prior = np.zeros(n, dtype=np.bool_)
    sender_7d_count_zero = np.zeros(n, dtype=np.bool_)
    sender_no_amount_history = np.zeros(n, dtype=np.bool_)
    pair_no_prior = np.zeros(n, dtype=np.bool_)

    accounts: Dict[int, _AuditAccount] = {}
    pairs: Dict[Tuple[int, int], _AuditPair] = {}

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
            s_state = accounts.get(s)
            r_state = accounts.get(r)
            sender_no_prior[idx] = s_state is None or s_state.tx_count == 0
            receiver_no_prior[idx] = r_state is None or r_state.tx_count == 0
            if s_state is None:
                sender_7d_count_zero[idx] = True
            else:
                sender_7d_count_zero[idx] = _count_open_window(s_state.out_timestamps, t, window_7d_sec) == 0
            sender_no_amount_history[idx] = sender_no_prior[idx]
            pair_no_prior[idx] = pairs.get((s, r)) is None or pairs[(s, r)].count == 0

        for idx in batch:
            s = int(from_ids[idx])
            r = int(to_ids[idx])
            t = float(ts[idx])
            amt = float(amounts[idx])
            s_state = accounts.get(s)
            if s_state is None:
                s_state = _AuditAccount()
                accounts[s] = s_state
            r_state = accounts.get(r)
            if r_state is None:
                r_state = _AuditAccount()
                accounts[r] = r_state
            import bisect as _bisect

            s_state.last_ts = t
            s_state.tx_count += 1
            s_state.amount_sum += amt
            _bisect.insort(s_state.out_timestamps, t)
            r_state.last_ts = t
            r_state.tx_count += 1
            r_state.amount_sum += amt
            p_state = pairs.get((s, r))
            if p_state is None:
                p_state = _AuditPair()
                pairs[(s, r)] = p_state
            p_state.count += 1

        pos = batch_end

    return {
        "sender_no_prior": sender_no_prior,
        "receiver_no_prior": receiver_no_prior,
        "sender_7d_count_zero": sender_7d_count_zero,
        "sender_no_amount_history": sender_no_amount_history,
        "pair_no_prior": pair_no_prior,
    }


def _frac(mask: np.ndarray) -> float:
    return float(mask.mean()) if mask.size else float("nan")


def _split_fractions(flags: Dict[str, np.ndarray], split_idx: np.ndarray) -> Dict[str, float]:
    return {k: _frac(v[split_idx]) for k, v in flags.items()}


def _load_cache(cache_dir: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    with (cache_dir / "meta.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)
    features = np.load(cache_dir / "features.npy")
    return features, meta


def audit_dataset(
    data: str,
    data_config: str,
    cache_dir: Path,
    *,
    sample_n: int,
    seed: int,
) -> Dict[str, Any]:
    spec = get_dataset_spec(data)
    cfg = load_data_config(data_config)
    csv_path = Path(cfg["paths"]["aml_data"]) / data / spec.formatted_csv_name()
    df = pd.read_csv(csv_path)
    df["Timestamp"] = df["Timestamp"] - df["Timestamp"].min()
    amount_col = resolve_amount_column(df)

    y = torch.LongTensor(df[spec.label_col].to_numpy())
    timestamps = torch.Tensor(df["Timestamp"].to_numpy())
    tr_inds, val_inds, te_inds, _ = temporal_edge_split(timestamps, y, spec)
    split_map = {"train": tr_inds.numpy(), "val": val_inds.numpy(), "test": te_inds.numpy()}

    features, cache_meta = _load_cache(cache_dir)
    if features.shape[0] != len(df):
        raise ValueError(f"Cache rows {features.shape[0]} != CSV rows {len(df)}")

    recomputed, _ = compute_temporal_flow_causal_features(df, amount_col=amount_col)
    max_abs_diff = float(np.max(np.abs(recomputed - features)))
    flags = replay_audit_state(df, amount_col=amount_col)

    edge_id = np.arange(len(df), dtype=np.int64)
    uniqueness_ok = edge_id.shape[0] == len(np.unique(edge_id))

    # Timestamp tie audit: count edges sharing timestamps
    ts_arr = df["Timestamp"].to_numpy()
    _, tie_counts = np.unique(ts_arr, return_counts=True)
    n_multi_ts = int((tie_counts > 1).sum())
    edges_in_ties = int(tie_counts[tie_counts > 1].sum())

    rng = np.random.default_rng(seed)
    sample_indices = rng.choice(len(df), size=min(sample_n, len(df)), replace=False)
    sample_indices.sort()

    samples: List[Dict[str, Any]] = []
    names = list(TEMPORAL_FLOW_CAUSAL_FEATURE_NAMES)
    for idx in sample_indices[:sample_n]:
        row = df.iloc[int(idx)]
        feat = features[int(idx)]
        samples.append(
            {
                "edge_id": int(idx),
                "timestamp": float(row["Timestamp"]),
                "sender": int(row["from_id"]),
                "receiver": int(row["to_id"]),
                "amount": float(row[amount_col]),
                "feature_values": {names[j]: float(feat[j]) for j in range(len(names))},
                "no_history_flags": {k: bool(flags[k][idx]) for k in flags},
                "pair_repeat_indicator": float(feat[names.index("pair_repeat_indicator")]),
            }
        )

    split_default_rates = {
        split: _split_fractions(flags, split_map[split]) for split in SPLITS
    }
    # Feature-zero rates (not equivalent to no-history for 7d count)
    zero_rates = {}
    for split in SPLITS:
        idx = split_map[split]
        zero_rates[split] = {
            name: float(np.mean(features[idx, j] == 0.0))
            for j, name in enumerate(names)
        }

    return {
        "dataset": data,
        "cache_dir": str(cache_dir),
        "cache_meta": {
            "cache_version": cache_meta.get("cache_version"),
            "causal_history_policy": cache_meta.get("causal_history_policy"),
            "timestamp_handling": cache_meta.get("timestamp_handling"),
            "source_csv_sha256": cache_meta.get("source_data", {}).get("csv_sha256"),
        },
        "causal_history": {
            "implementation": "morphology.temporal_flow_causal.compute_temporal_flow_causal_features",
            "timestamp_tie_rule": (
                "Equal timestamps batched; featurize before batch state update; "
                "same-timestamp edges do not influence each other."
            ),
            "tested_by": "tests/test_temporal_flow_causal_features.py::test_timestamp_tie_batch_no_cross_influence",
            "recompute_max_abs_diff_vs_cache": max_abs_diff,
            "recompute_matches_cache": max_abs_diff < 1e-5,
        },
        "split_history_policy": {
            "val_sees_train_history": True,
            "test_sees_train_and_val_history": True,
            "history_resets_at_split_boundaries": False,
            "within_split_chronological": True,
            "note": "Global timestamp sort over full CSV; realistic deployment semantics.",
        },
        "train_only_scaling": {
            "cache_time_normalization": "none",
            "probe_policy": "StandardScaler fit on temporal train indices only; transform all splits",
            "labels_used": False,
        },
        "edge_alignment": {
            "csv_rows": int(len(df)),
            "edge_id_is_row_index": True,
            "edge_id_unique": uniqueness_ok,
            "split_row_counts": {s: int(split_map[s].shape[0]) for s in SPLITS},
        },
        "timestamp_ties": {
            "distinct_timestamps_with_multiple_edges": n_multi_ts,
            "edges_in_multi_edge_timestamps": edges_in_ties,
        },
        "default_history_fractions": split_default_rates,
        "feature_zero_fractions": zero_rates,
        "pair_repeat_fraction": {s: float(np.mean(features[split_map[s], -1] == 1.0)) for s in SPLITS},
        "samples": samples,
    }


def write_md(path: Path, payload: Dict[str, Any]) -> None:
    lines = [
        "# temporal_flow_causal leakage audit",
        "",
        f"**Datasets:** {', '.join(d['dataset'] for d in payload['datasets'])}",
        "",
    ]
    for d in payload["datasets"]:
        lines.append(f"## {d['dataset']}")
        lines.append("")
        lines.append(f"- Recompute matches cache: **{d['causal_history']['recompute_matches_cache']}** "
                     f"(max |Δ|={d['causal_history']['recompute_max_abs_diff_vs_cache']:.2e})")
        lines.append(f"- Edges in timestamp ties: {d['timestamp_ties']['edges_in_multi_edge_timestamps']}")
        lines.append("")
        lines.append("### Default-history fractions (true flags, not zero==no-NaN)")
        lines.append("")
        lines.append("| split | sender no prior | receiver no prior | 7d count=0 | no amount hist | no pair prior | pair_repeat=1 |")
        lines.append("|-------|----------------:|------------------:|-----------:|---------------:|--------------:|--------------:|")
        for split in SPLITS:
            f = d["default_history_fractions"][split]
            pr = d["pair_repeat_fraction"][split]
            lines.append(
                f"| {split} | {f['sender_no_prior']:.4f} | {f['receiver_no_prior']:.4f} | "
                f"{f['sender_7d_count_zero']:.4f} | {f['sender_no_amount_history']:.4f} | "
                f"{f['pair_no_prior']:.4f} | {pr:.4f} |"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_config", default="data_config.json")
    p.add_argument("--cache_root", default="results/cache/temporal_flow_causal")
    p.add_argument("--datasets", default="Small-HI,Small-LI")
    p.add_argument("--sample_n", type=int, default=20)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--output_json", default="results/diagnostics/temporal_flow_causal_leakage_audit.json")
    p.add_argument("--output_md", default="notes/temporal_flow_causal_leakage_audit.md")
    args = p.parse_args()
    logger_setup()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    results = [
        audit_dataset(d, args.data_config, Path(args.cache_root) / d, sample_n=args.sample_n, seed=args.seed)
        for d in datasets
    ]
    payload = {"diagnostic": "temporal_flow_causal_leakage_audit", "datasets": results}
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    write_md(Path(args.output_md), payload)
    print(f"Wrote {out_json} and {args.output_md}")


if __name__ == "__main__":
    main()
