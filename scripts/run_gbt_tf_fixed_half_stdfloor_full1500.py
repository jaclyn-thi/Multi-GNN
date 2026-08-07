#!/usr/bin/env python3
"""1500-step MIXED_3DOMAIN_GBT_TF_FIXED_HALF_STDFLOOR_1E4 training entry.

Uses the adaptive orchestration path with fixed_half overrides. Executes 1500
optimizer steps on the LONG 3000-step LR schedule prefix (not rescaled).
Checkpoints at 750 and 1500 (+ rolling last). No Slurm submit from this script.
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
    CHECKPOINT_STEPS,
    CKPT_ROOT,
    EXECUTED_STOP_STEP,
    RESULT_ROOT,
    SCHEDULE_HORIZON,
    orchestration_overrides,
    resolved_recipe,
)
from util import logger_setup  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=f"{ARM} full {EXECUTED_STOP_STEP}-step train")
    p.add_argument("--run-train", action="store_true")
    p.add_argument("--split", type=str, default="train")
    p.add_argument(
        "--allow-existing-ckpts",
        action="store_true",
        help="Permit overwrite of existing checkpoints (default: refuse)",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    refuse_test_split_access(args.split)
    logger_setup()
    assert_api_contracts()

    if not args.run_train:
        print(
            json.dumps(
                {
                    "arm": ARM,
                    "status": "pass --run-train to execute",
                    "executed_stop_step": EXECUTED_STOP_STEP,
                    "schedule_horizon": SCHEDULE_HORIZON,
                    "recipe": resolved_recipe(),
                },
                indent=2,
                default=str,
            )
        )
        return 0

    overrides = orchestration_overrides()
    # schedule_horizon comes from overrides; keep executed/checkpoint explicit here.
    cfg = OrchestrationConfig(
        mode="smoke",  # same executable orchestration path as adaptive smoke/full helpers
        total_steps=int(EXECUTED_STOP_STEP),
        do_optimizer_steps=True,
        synthetic=False,
        skip_phase3_init=False,
        refuse_existing_ckpts=not bool(args.allow_existing_ckpts),
        require_seed_match=True,
        require_alpha_unfreeze=True,
        require_bn_divergence=True,
        out_root=ROOT / RESULT_ROOT,
        ckpt_root=ROOT / CKPT_ROOT,
        executed_stop_step=int(EXECUTED_STOP_STEP),
        checkpoint_steps=tuple(CHECKPOINT_STEPS),
        **overrides,
    )
    summary = run_orchestration(cfg)
    # Persist a thin train aggregate pointer for later integrity / frozen-eval wiring.
    (ROOT / RESULT_ROOT).mkdir(parents=True, exist_ok=True)
    train_summary = {
        **summary,
        "recipe": resolved_recipe(),
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "schedule_horizon": SCHEDULE_HORIZON,
        "executed_stop_step": EXECUTED_STOP_STEP,
    }
    (ROOT / RESULT_ROOT / "train_summary.json").write_text(
        json.dumps(train_summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, default=str))
    return 0 if summary.get("gates_ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
