#!/usr/bin/env python3
"""
Precompute Tier 0 (split-global) node morphology tables.

Writes ``{output_dir}/{split}_node_morphology.csv`` with columns deg_in, deg_out,
deg_total indexed by node_id. Use only the train table for pretrain targets;
val/test tables are for eval or analysis (no leakage into train targets).

Example::

  python scripts/precompute_morphology_tier0.py \\
    --data Small-HI --output_dir morphology_cache/Small-HI
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Repo root on sys.path when invoked as script
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data_loading import get_data
from morphology import (
    compute_tier0_node_stats,
    get_forward_edge_index,
    get_forward_timestamps,
    get_num_nodes,
    save_node_table,
)
from train_util import add_arange_ids
from util import create_parser, logger_setup, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute Tier 0 node morphology per split.")
    parser.add_argument("--data", type=str, required=True, help="Dataset name (e.g. Small-HI).")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for {train,val,test}_node_morphology.csv",
    )
    parser.add_argument("--config", type=str, default="data_config.json")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--reverse_mp", action="store_true", help="Match hetero training graphs.")
    parser.add_argument("--ego", action="store_true")
    parser.add_argument("--ports", action="store_true")
    parser.add_argument("--tds", action="store_true")
    args = parser.parse_args()

    logger_setup()
    set_seed(args.seed)

    with open(args.config, encoding="utf-8") as f:
        data_config = json.load(f)

    # Minimal args object for get_data (graph flags must match training if hetero).
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

    meta = {
        "data": args.data,
        "reverse_mp": bool(args.reverse_mp),
        "ports": bool(args.ports),
        "tds": bool(args.tds),
        "splits": {},
    }

    for split_name, split_data in (
        ("train", tr_data),
        ("val", val_data),
        ("test", te_data),
    ):
        ei = get_forward_edge_index(split_data)
        n_nodes = get_num_nodes(split_data)
        ts = get_forward_timestamps(split_data)
        table = compute_tier0_node_stats(ei, n_nodes, timestamps=ts)
        path = out_dir / f"{split_name}_node_morphology.csv"
        save_node_table(table, path)
        meta["splits"][split_name] = {
            "path": str(path),
            "num_nodes": int(n_nodes),
            "num_edges": int(ei.shape[1]),
            "deg_in_sum": int(table["deg_in"].sum()),
            "deg_out_sum": int(table["deg_out"].sum()),
        }
        logging.info(
            "Wrote %s: nodes=%d edges=%d",
            path,
            n_nodes,
            ei.shape[1],
        )

    meta_path = out_dir / "tier0_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logging.info("Wrote %s", meta_path)


if __name__ == "__main__":
    main()
