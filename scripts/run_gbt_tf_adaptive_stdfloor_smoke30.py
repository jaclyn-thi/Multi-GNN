#!/usr/bin/env python3
"""30-step smoke / Stage-3 dry-init entry for MIXED_3DOMAIN_GBT_TF_ADAPTIVE_STDFLOOR_1E4.

Delegates to gbt_tf_adaptive_stdfloor_r198.orchestration so unit/integration tests
exercise the same executable path as Slurm Stage 3.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gbt_tf_adaptive_stdfloor_r198 import ARM, SMOKE_MAX_STEPS  # noqa: E402
from gbt_tf_adaptive_stdfloor_r198.integrity import refuse_test_split_access  # noqa: E402
from gbt_tf_adaptive_stdfloor_r198.orchestration import (  # noqa: E402
    OrchestrationConfig,
    assert_api_contracts,
    run_orchestration,
)
from util import logger_setup  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=f"{ARM} smoke / stage3-dry-init")
    p.add_argument("--run-smoke", action="store_true")
    p.add_argument(
        "--stage3-dry-init",
        action="store_true",
        help="Same init path through one no-update step/domain; no full 30-step train",
    )
    p.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic mini-graphs/TF (for dry-init / integration; no financial load)",
    )
    p.add_argument("--max-optimizer-steps", type=int, default=SMOKE_MAX_STEPS)
    p.add_argument("--split", type=str, default="train")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    refuse_test_split_access(args.split)
    logger_setup()

    # Always assert helper signatures before any Stage-3 work.
    assert_api_contracts()

    if args.stage3_dry_init:
        cfg = OrchestrationConfig(
            mode="dry_init",
            total_steps=3,
            do_optimizer_steps=False,
            synthetic=True,
            skip_phase3_init=True,
            refuse_existing_ckpts=False,
            require_seed_match=False,
            require_alpha_unfreeze=False,
            require_bn_divergence=False,
            batch_size_override=32,
            num_neighbors_override=(8, 8),
        )
        summary = run_orchestration(cfg)
        print(json.dumps(summary, indent=2, default=str))
        return 0 if summary.get("gates_ok") and summary.get("status") == "dry_init_ok" else 1

    if not args.run_smoke:
        print(json.dumps({"arm": ARM, "status": "pass --run-smoke or --stage3-dry-init"}, indent=2))
        return 0

    steps = int(args.max_optimizer_steps)
    if steps < 1 or steps > SMOKE_MAX_STEPS:
        raise RuntimeError(f"smoke steps must be in [1,{SMOKE_MAX_STEPS}], got {steps}")

    cfg = OrchestrationConfig(
        mode="smoke",
        total_steps=steps,
        do_optimizer_steps=True,
        synthetic=False,
        skip_phase3_init=False,
        refuse_existing_ckpts=True,
        require_seed_match=True,
        require_alpha_unfreeze=True,
        require_bn_divergence=True,
    )
    summary = run_orchestration(cfg)
    print(json.dumps(summary, indent=2, default=str))
    return 0 if summary.get("status") == "smoke_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
