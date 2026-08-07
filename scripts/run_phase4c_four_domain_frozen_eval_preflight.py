#!/usr/bin/env python3
"""Phase-4C four-domain frozen-eval preflight / inventory / plan CLI.

Isolated from the training-manifested legacy extract/probe script.
Real-data modes require SLURM_JOB_ID.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phase4c_four_domain_frozen_eval import (  # noqa: E402
    PREFLIGHT_RESULT_DIR,
    all_plan_cells,
    assert_output_roots_unique,
)
from phase4c_four_domain_frozen_eval.inventory import write_inventory_artifacts  # noqa: E402
from phase4c_four_domain_frozen_eval.plan import write_evaluation_plan  # noqa: E402
from phase4c_four_domain_frozen_eval.preflight import run_preflight  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Phase-4C four-domain frozen-eval preflight infrastructure"
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("inventory", help="Read-only ten-checkpoint inventory (login-safe)")
    sub.add_parser("plan", help="Write 40-cell plan + storage estimate (login-safe)")
    sub.add_parser("preflight", help="Bounded real-data Slurm preflight")
    sub.add_parser("dry-cells", help="Print planned 40 cells (login-safe)")
    args = ap.parse_args()
    assert_output_roots_unique()

    if args.cmd == "inventory":
        out = write_inventory_artifacts(ROOT / PREFLIGHT_RESULT_DIR)
        print(json.dumps({"ok": out["ok"], "n_ok": out["n_ok"], "artifacts": out["artifacts"]}, indent=2))
        return 0 if out["ok"] else 1
    if args.cmd == "plan":
        plan = write_evaluation_plan(ROOT / PREFLIGHT_RESULT_DIR)
        print(
            json.dumps(
                {
                    "n_cells": plan["n_cells"],
                    "artifact": plan["artifact"],
                    "storage_est_gib": plan["storage_estimate"]["total_est_gib"],
                    "free_gib": plan["storage_estimate"]["destination_free_gib"],
                    "sufficient": plan["storage_estimate"]["destination_sufficient"],
                    "submitted": False,
                },
                indent=2,
            )
        )
        return 0
    if args.cmd == "dry-cells":
        print(json.dumps({"n": 40, "cells": all_plan_cells()}, indent=2, default=str))
        return 0
    if args.cmd == "preflight":
        if not os.environ.get("SLURM_JOB_ID"):
            print("Refuse: preflight requires SLURM_JOB_ID", file=sys.stderr)
            return 2
        summary = run_preflight()
        slim = {k: v for k, v in summary.items() if k != "domains"}
        print(json.dumps(slim, indent=2, default=str))
        print(f"end={summary.get('end')}")
        return 0 if summary.get("ok") else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
