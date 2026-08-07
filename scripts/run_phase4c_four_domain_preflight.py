#!/usr/bin/env python3
"""Real bounded one-batch/domain Phase-4C preflight (Slurm compute only)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phase4c_four_domain import ARMS
from phase4c_four_domain.train import train_arm


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--arm",
        choices=tuple(ARMS),
        default="FOUR_DOMAIN_PROJECTION_INFONCE_TF_ADAPTIVE_SHORT",
    )
    ap.add_argument(
        "--source-manifest-path",
        default="results/diagnostics/phase4c_four_domain_source_manifest.approved.json",
    )
    ap.add_argument(
        "--result-dir-override",
        default=None,
        help="Attempt-specific diagnostics path (keeps prior failure artifacts intact).",
    )
    ap.add_argument(
        "--ckpt-dir-override",
        default=None,
        help="Attempt-specific checkpoint path for preflight probe ckpts.",
    )
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit(
            "Refuse: real-data Phase-4C preflight requires SLURM_JOB_ID "
            "(submit via slurm/run_phase4c_four_domain_cpu_preflight.sh)."
        )
    result = train_arm(
        args.arm,
        mode="preflight",
        source_manifest_path=args.source_manifest_path,
        result_dir_override=args.result_dir_override,
        ckpt_dir_override=args.ckpt_dir_override,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
