#!/usr/bin/env python3
"""
Precompute Tier 2 (M3 Phase 0) split-global node morphology — betweenness centrality.

Writes ``{output_dir}/{split}_node_tier2.csv`` with column ``bc`` indexed by node_id.
Use train split tables only for pretrain targets; val/test for eval (no leakage).

Default uses **sampled** Brandes (256 sources) for tractability on Small-HI scale graphs.
Use ``--bc_exact`` for exact BC (slow; small graphs / tests only).

Example::

  python scripts/precompute_morphology_tier2.py \\
    --data Small-HI --output_dir morphology_cache/Small-HI \\
    --reverse_mp --ego --ports

Requires Tier 0 cache optional but recommended in same ``output_dir`` for joint metadata.

**Memory:** Full Small-HI load with ``--reverse_mp --ego --ports`` needs ~128G RAM (same as training data prep). Use ``run_precompute_morphology_tier2.sh`` on Slurm; login nodes may OOM.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data_loading import get_data
from morphology.tier0_global import save_node_table
from morphology.tier2_global import precompute_tier2_for_split
from train_util import add_arange_ids
from util import create_parser, logger_setup, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute Tier 2 node morphology (betweenness centrality) per split.",
    )
    parser.add_argument("--data", type=str, required=True, help="Dataset name (e.g. Small-HI).")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for {train,val,test}_node_tier2.csv",
    )
    parser.add_argument("--config", type=str, default="data_config.json")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--reverse_mp", action="store_true", help="Match hetero training graphs.")
    parser.add_argument("--ego", action="store_true")
    parser.add_argument("--ports", action="store_true")
    parser.add_argument("--tds", action="store_true")
    parser.add_argument(
        "--bc_samples",
        type=int,
        default=256,
        help="Number of BFS sources for approximate BC (default 256). Ignored if --bc_exact.",
    )
    parser.add_argument(
        "--bc_exact",
        action="store_true",
        help="Exact Brandes BC (all nodes as sources). Very slow on large splits.",
    )
    parser.add_argument(
        "--bc_no_normalize",
        action="store_true",
        help="Skip Freeman normalization (raw dependency scores).",
    )
    args = parser.parse_args()

    logger_setup()
    set_seed(args.seed)

    with open(args.config, encoding="utf-8") as f:
        data_config = json.load(f)

    data_args = create_parser().parse_args(
        [
            "--data",
            args.data,
            "--model",
            "gin",
            "--testing",
        ]
        + (["--reverse_mp"] if args.reverse_mp else [])
        + (["--ego"] if args.ego else [])
        + (["--ports"] if args.ports else [])
        + (["--tds"] if args.tds else [])
    )

    tr_data, val_data, te_data, _, _, _ = get_data(data_args, data_config)
    add_arange_ids([tr_data, val_data, te_data])

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    k_samples = None if args.bc_exact else int(args.bc_samples)
    normalized = not args.bc_no_normalize

    meta = {
        "data": args.data,
        "reverse_mp": bool(args.reverse_mp),
        "ports": bool(args.ports),
        "tds": bool(args.tds),
        "tier": 2,
        "metric": "betweenness_centrality_directed",
        "bc_exact": bool(args.bc_exact),
        "bc_samples": k_samples,
        "normalized": normalized,
        "splits": {},
    }

    for split_name, split_data in (
        ("train", tr_data),
        ("val", val_data),
        ("test", te_data),
    ):
        table, split_meta = precompute_tier2_for_split(
            split_data,
            k_samples=k_samples,
            normalized=normalized,
            seed=args.seed,
        )
        path = out_dir / f"{split_name}_node_tier2.csv"
        save_node_table(table, path)
        split_meta["path"] = str(path)
        meta["splits"][split_name] = split_meta
        logging.info("Wrote %s", path)

    meta_path = out_dir / "tier2_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logging.info("Wrote %s", meta_path)


if __name__ == "__main__":
    main()
