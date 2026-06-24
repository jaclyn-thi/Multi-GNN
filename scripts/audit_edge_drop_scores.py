#!/usr/bin/env python
"""Audit edge-drop score / probability distributions before training."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from edge_drop_scores import (
    audit_score_distribution,
    build_edge_drop_cache,
    load_edge_drop_cache,
)
from transaction_knn.features import load_train_frame


def _monte_carlo_realized_rate(drop_prob: np.ndarray, *, trials: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    rates = []
    for _ in range(trials):
        keep = rng.random(drop_prob.shape[0]) > drop_prob
        if not keep.any() and drop_prob.shape[0] > 0:
            keep[rng.integers(0, drop_prob.shape[0])] = True
        rates.append(1.0 - float(keep.mean()))
    return float(np.mean(rates))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="Small-HI")
    parser.add_argument("--data_config", default="data_config.json")
    parser.add_argument(
        "--policy",
        required=True,
        choices=["degree_aware", "degree_flow_aware"],
    )
    parser.add_argument("--cache_path", default=None, help="Optional existing .npz (else compute)")
    parser.add_argument("--target_rate", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--min_prob", type=float, default=0.01)
    parser.add_argument("--max_prob", type=float, default=0.95)
    parser.add_argument("--max_rows", type=int, default=50000)
    parser.add_argument("--mc_trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log_level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    if args.cache_path:
        cache = load_edge_drop_cache(args.cache_path)
    else:
        _, df_train, _, _ = load_train_frame(args.data, args.data_config, max_rows=args.max_rows)
        cache = build_edge_drop_cache(
            df_train,
            args.policy,
            target_rate=args.target_rate,
            alpha=args.alpha,
            min_prob=args.min_prob,
            max_prob=args.max_prob,
        )

    summary = audit_score_distribution(cache)
    summary["monte_carlo_realized_drop_rate"] = _monte_carlo_realized_rate(
        cache.drop_prob, trials=args.mc_trials, seed=args.seed
    )
    summary["label_columns_used"] = []
    summary["dense_matrices_built"] = False
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
