#!/usr/bin/env python3
"""Precompute Small-HI train-static triangle targets (isolated script).

By default this script does NOT load the full Small-HI dataset. Use
--synthetic-demo to write a tiny deterministic cache for wiring checks, or
--execute-full (NOT RUN in the infrastructure task) with explicit CSV paths.

Never accepts a test split. Never modifies historical checkpoints/embeddings.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from morphology_downstream_probe.cache_io import write_triangle_cache
from morphology_downstream_probe.config import RESULT_ROOT, TRIANGLE_CACHE_REL
from morphology_downstream_probe.distribution_gate import distribution_report
from morphology_downstream_probe.triangles import (
    compute_train_static_triangles,
    edge_log_triangle_targets,
)


def _load_edge_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Accept common column aliases without touching shared loaders.
    colmap = {}
    lower = {c.lower(): c for c in df.columns}
    for want, aliases in {
        "EdgeID": ["edgeid", "edge_id", "txid", "tx_id"],
        "from_id": ["from_id", "sender", "source", "from", "account"],
        "to_id": ["to_id", "receiver", "target", "to"],
    }.items():
        for a in aliases:
            if a in lower:
                colmap[lower[a]] = want
                break
        else:
            if want not in df.columns:
                raise RuntimeError(f"{path}: missing column for {want} (aliases={aliases})")
    df = df.rename(columns=colmap)
    for c in ("EdgeID", "from_id", "to_id"):
        if c not in df.columns:
            raise RuntimeError(f"{path}: missing required column {c}")
    return df[["EdgeID", "from_id", "to_id"]].copy()


def _synthetic_frames():
    # Two triangles sharing edge structure + a val edge to unseen node.
    # Train edges form triangle {0,1,2} and {1,2,3}.
    train = pd.DataFrame(
        {
            "EdgeID": [10, 11, 12, 13, 14, 15, 16],
            "from_id": [0, 1, 2, 1, 2, 3, 0],
            "to_id": [1, 2, 0, 3, 3, 1, 0],  # last is self-loop (ignored)
        }
    )
    # Duplicate parallel edge should not multiply triangles
    train = pd.concat(
        [train, pd.DataFrame({"EdgeID": [17], "from_id": [1], "to_id": [0]})],
        ignore_index=True,
    )
    val = pd.DataFrame(
        {
            "EdgeID": [100, 101, 102],
            "from_id": [0, 9, 1],  # 9 unseen
            "to_id": [2, 1, 9],
        }
    )
    return train, val


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=Path(TRIANGLE_CACHE_REL) / "synthetic_demo")
    p.add_argument("--synthetic-demo", action="store_true")
    p.add_argument("--train-csv", type=Path, default=None)
    p.add_argument("--val-csv", type=Path, default=None)
    p.add_argument(
        "--execute-full",
        action="store_true",
        help="Load provided train/val CSVs and write cache (NOT used in infra task).",
    )
    p.add_argument("--allow-test", action="store_true", help="Forbidden; always refused.")
    args = p.parse_args()

    if args.allow_test:
        raise RuntimeError("Refuse --allow-test: test access is forbidden")

    if args.synthetic_demo:
        train_df, val_df = _synthetic_frames()
    elif args.execute_full:
        if args.train_csv is None or args.val_csv is None:
            raise RuntimeError("--execute-full requires --train-csv and --val-csv")
        train_df = _load_edge_csv(args.train_csv)
        val_df = _load_edge_csv(args.val_csv)
    else:
        print(
            "No work requested. Pass --synthetic-demo for a tiny wiring cache, "
            "or --execute-full with CSV paths for the real Small-HI precompute (NOT RUN in infra task)."
        )
        print(f"Default result root: {RESULT_ROOT}")
        return 0

    counts, prov = compute_train_static_triangles(
        train_df["from_id"].to_numpy(),
        train_df["to_id"].to_numpy(),
        n_train_edges_input=len(train_df),
    )
    y_tr, t_tr, cov_tr = edge_log_triangle_targets(
        train_df["EdgeID"].to_numpy(),
        train_df["from_id"].to_numpy(),
        train_df["to_id"].to_numpy(),
        counts,
        split="train",
    )
    y_va, t_va, cov_va = edge_log_triangle_targets(
        val_df["EdgeID"].to_numpy(),
        val_df["from_id"].to_numpy(),
        val_df["to_id"].to_numpy(),
        counts,
        split="validation",
    )
    dist = distribution_report(
        y_tr,
        y_va,
        t_sum_train=t_tr,
        t_sum_val=t_va,
        train_coverage=cov_tr,
        val_coverage=cov_va,
    )
    paths = write_triangle_cache(
        args.out_dir,
        counts=counts,
        provenance=prov,
        train_edge_ids=train_df["EdgeID"].to_numpy(),
        train_y=y_tr,
        train_t_sum=t_tr,
        train_coverage=cov_tr,
        val_edge_ids=val_df["EdgeID"].to_numpy(),
        val_y=y_va,
        val_t_sum=t_va,
        val_coverage=cov_va,
    )
    summary = {
        "ok": True,
        "mode": "synthetic_demo" if args.synthetic_demo else "execute_full",
        "provenance": prov.to_dict(),
        "distribution_gate": dist,
        "paths": paths,
        "executed_full_smallhi_scan": bool(args.execute_full and not args.synthetic_demo),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "precompute_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"ok": True, "out_dir": str(args.out_dir), "node_sha": prov.node_triangle_sha256}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
