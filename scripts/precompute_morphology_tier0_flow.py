#!/usr/bin/env python3
"""
Precompute Tier 0 flow-balance node amount tables.

Writes ``{output_dir}/{split}_node_flow_balance.csv`` with columns amount_in,
amount_out indexed by node_id. Aggregates Amount Received on the forward split
graph only (label-free).

Example::

  python scripts/precompute_morphology_tier0_flow.py \\
    --data Small-HI --output_dir morphology_cache/Small-HI
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
from morphology import (
    amount_received_from_graph,
    compute_tier0_flow_node_stats,
    get_forward_edge_index,
    get_num_nodes,
    save_node_table,
)
from train_util import add_arange_ids
from util import create_parser, logger_setup, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute Tier 0 flow-balance node tables per split.")
    parser.add_argument("--data", type=str, required=True, help="Dataset name (e.g. Small-HI).")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for {train,val,test}_node_flow_balance.csv",
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
        "amount_column": "Amount Received",
        "splits": {},
    }

    for split_name, split_data in (
        ("train", tr_data),
        ("val", val_data),
        ("test", te_data),
    ):
        ei = get_forward_edge_index(split_data)
        edge_amounts = amount_received_from_graph(split_data)
        n_nodes = get_num_nodes(split_data)
        table = compute_tier0_flow_node_stats(ei, edge_amounts, n_nodes)
        path = out_dir / f"{split_name}_node_flow_balance.csv"
        save_node_table(table, path)
        meta["splits"][split_name] = {
            "path": str(path),
            "num_nodes": int(n_nodes),
            "num_edges": int(ei.shape[1]),
            "amount_in_sum": float(table["amount_in"].sum()),
            "amount_out_sum": float(table["amount_out"].sum()),
        }
        logging.info(
            "Wrote %s: nodes=%d edges=%d amount_in_sum=%.3e amount_out_sum=%.3e",
            path,
            n_nodes,
            ei.shape[1],
            meta["splits"][split_name]["amount_in_sum"],
            meta["splits"][split_name]["amount_out_sum"],
        )

    meta_path = out_dir / "tier0_flow_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logging.info("Wrote %s", meta_path)


if __name__ == "__main__":
    main()
