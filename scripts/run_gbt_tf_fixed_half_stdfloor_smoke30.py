#!/usr/bin/env python3
"""Smoke / Stage-3 dry-init for MIXED_3DOMAIN_GBT_TF_FIXED_HALF_STDFLOOR_1E4.

Reuses gbt_tf_adaptive_stdfloor_r198.orchestration with fixed_half overrides.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gbt_tf_adaptive_stdfloor_r198.integrity import refuse_test_split_access  # noqa: E402
from gbt_tf_adaptive_stdfloor_r198.orchestration import (  # noqa: E402
    OrchestrationConfig,
    assert_api_contracts,
    run_orchestration,
)
from gbt_tf_fixed_half_stdfloor_r198 import (  # noqa: E402
    ARM,
    CKPT_ROOT,
    EXECUTED_STOP_STEP,
    RESULT_ROOT,
    SMOKE_CKPT_ROOT,
    SMOKE_MAX_STEPS,
    SMOKE_RESULT_ROOT,
    orchestration_overrides,
)
from util import logger_setup  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=f"{ARM} smoke / stage3-dry-init")
    p.add_argument("--run-smoke", action="store_true")
    p.add_argument("--stage3-dry-init", action="store_true")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--max-optimizer-steps", type=int, default=SMOKE_MAX_STEPS)
    p.add_argument("--split", type=str, default="train")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    refuse_test_split_access(args.split)
    logger_setup()
    assert_api_contracts()
    overrides = orchestration_overrides()

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
            out_root=ROOT / "results/diagnostics/financial_multidataset_gbt_tf_fixed_half_stdfloor_1e4_stage3_dry_init",
            ckpt_root=ROOT / "results/checkpoints/financial_multidataset_gbt_tf_fixed_half_stdfloor_1e4_stage3_dry_init",
            executed_stop_step=3,
            **overrides,
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
        synthetic=bool(args.synthetic),
        skip_phase3_init=bool(args.synthetic),
        refuse_existing_ckpts=not bool(args.synthetic),
        require_seed_match=not bool(args.synthetic),
        require_alpha_unfreeze=True,
        require_bn_divergence=not bool(args.synthetic),
        out_root=ROOT / SMOKE_RESULT_ROOT,
        ckpt_root=ROOT / SMOKE_CKPT_ROOT,
        executed_stop_step=steps,
        checkpoint_steps=(steps,),
        **overrides,
    )
    summary = run_orchestration(cfg)
    print(json.dumps(summary, indent=2, default=str))
    return 0 if summary.get("gates_ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
