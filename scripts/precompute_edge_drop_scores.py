#!/usr/bin/env python
"""Precompute train-split edge-drop probability caches for contrastive augmentation."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from edge_drop_scores import audit_score_distribution, build_edge_drop_cache, save_edge_drop_cache
from transaction_knn.features import load_train_frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="Small-HI")
    parser.add_argument("--data_config", default="data_config.json")
    parser.add_argument(
        "--policy",
        required=True,
        choices=["degree_aware", "degree_flow_aware"],
    )
    parser.add_argument("--output", required=True, help="Output .npz path")
    parser.add_argument("--target_rate", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--min_prob", type=float, default=0.01)
    parser.add_argument("--max_prob", type=float, default=0.95)
    parser.add_argument("--max_rows", type=int, default=0, help="Optional row cap for smoke tests")
    parser.add_argument("--log_level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    _, df_train, _, _ = load_train_frame(args.data, args.data_config, max_rows=args.max_rows)
    cache = build_edge_drop_cache(
        df_train,
        args.policy,
        target_rate=args.target_rate,
        alpha=args.alpha,
        min_prob=args.min_prob,
        max_prob=args.max_prob,
    )
    save_edge_drop_cache(args.output, cache)
    summary = audit_score_distribution(cache)
    logging.info("Wrote %s", args.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
