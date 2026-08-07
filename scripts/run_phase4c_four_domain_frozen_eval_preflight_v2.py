#!/usr/bin/env python3
"""Phase-4C v2 full-coverage seed-complete frozen-eval preflight (Slurm only)."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phase4c_four_domain_frozen_eval.paths_v2 import (  # noqa: E402
    LOGICAL_EMB_ROOT,
    PHYSICAL_EMB_ROOT,
    RESULT_ROOT,
    assert_v2_paths_unique,
)
from phase4c_four_domain_frozen_eval.preflight_v2 import (  # noqa: E402
    REASON,
    V2_ARM,
    V2_STEP,
    build_eval_source_manifest_v2,
    run_preflight_v2,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Phase-4C v2 full-coverage seed-complete preflight: "
            f"{V2_ARM}@{V2_STEP} × four domains × full train/val. "
            "Requires SLURM_JOB_ID. Does not submit the evaluation DAG."
        )
    )
    p.add_argument(
        "--login-safe-check",
        action="store_true",
        help="Path uniqueness + eval-source-manifest v2 only (no real data).",
    )
    args = p.parse_args(argv)
    if args.login_safe_check:
        assert_v2_paths_unique()
        man = build_eval_source_manifest_v2()
        print(f"login_safe_ok=1 reason={REASON}")
        print(f"eval_source_manifest_v2_sha256={man['manifest_sha256']}")
        print(f"logical={LOGICAL_EMB_ROOT}")
        print(f"physical={PHYSICAL_EMB_ROOT}")
        print(f"result_root={RESULT_ROOT}")
        return 0
    if not os.environ.get("SLURM_JOB_ID"):
        print(
            "ERROR: SLURM_JOB_ID required for real-data v2 preflight. "
            "Use --login-safe-check on the login node.",
            file=sys.stderr,
        )
        return 2
    summary = run_preflight_v2()
    return 0 if summary.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
