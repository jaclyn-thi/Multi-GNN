#!/usr/bin/env python3
"""Run one isolated Phase-4C arm; this script never submits Slurm jobs."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from phase4c_four_domain import ARMS
from phase4c_four_domain.train import train_arm

ap = argparse.ArgumentParser()
ap.add_argument("--arm", choices=tuple(ARMS), required=True)
ap.add_argument("--mode", choices=("smoke", "full", "preflight", "memory_preflight", "dry"), default="smoke")
ap.add_argument("--steps", type=int)
ap.add_argument("--source-manifest-path")
args = ap.parse_args()
if args.mode == "dry":
    print(json.dumps({"ok": True, "dry": True, "arm": args.arm}))
else:
    import os
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit(
            f"Refuse: real-data mode={args.mode!r} requires SLURM_JOB_ID "
            "(login-node get_data/graph loads are forbidden)."
        )
    result = train_arm(args.arm, mode=args.mode, max_steps=args.steps,
                       source_manifest_path=args.source_manifest_path)
    print(json.dumps(result, indent=2))
    if not result["ok"]:
        raise SystemExit(1)
