#!/usr/bin/env python3
"""Trainer entry for MIXED_3DOMAIN_GBT_TF_FIXED_HALF_STDFLOOR_1E4.

Default path does not train. Explicit flags required for tests / dry / smoke / full.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gbt_tf_adaptive_stdfloor_r198.integrity import refuse_test_split_access  # noqa: E402
from gbt_tf_fixed_half_stdfloor_r198 import (  # noqa: E402
    ARM,
    EXECUTED_STOP_STEP,
    OBJECTIVE_ID,
    SMOKE_MAX_STEPS,
    TF_INPUT_VIEW,
    TF_TARGET_NAMES,
    WEIGHT_MODE,
    resolved_recipe,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=f"{ARM} trainer entry")
    p.add_argument("--dump-recipe", type=str, default="")
    p.add_argument("--focused-tests", action="store_true")
    p.add_argument("--dry-preflight", action="store_true")
    p.add_argument("--run-smoke", action="store_true")
    p.add_argument("--run-train", action="store_true")
    p.add_argument("--stage3-dry-init", action="store_true")
    p.add_argument("--max-optimizer-steps", type=int, default=None)
    p.add_argument("--split", type=str, default="train")
    return p.parse_args(argv)


def _load_script(name: str, path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def run_focused_tests() -> int:
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_gbt_tf_fixed_half_stdfloor_1e4.py",
        "tests/test_direct_r198_tfmoe_weight_modes.py",
        "tests/test_gbt_tf_adaptive_stdfloor_1e4.py",
        "tests/test_gbt_tf_adaptive_smoke_orchestration.py",
        "tests/test_graph_barlow_twins_r198_stdfloor_1e4.py",
        "tests/test_phase4b_objective_ablation.py",
        "--tb=short",
    ]
    print("Running:", " ".join(cmd))
    return int(subprocess.call(cmd, cwd=str(ROOT)))


def main(argv=None) -> int:
    args = parse_args(argv)
    refuse_test_split_access(args.split)

    if args.focused_tests:
        return run_focused_tests()

    if args.dry_preflight:
        mod = _load_script(
            "gbt_tf_fixed_half_dry",
            ROOT / "scripts" / "run_gbt_tf_fixed_half_stdfloor_dry_preflight.py",
        )
        dry_argv = ["--split", args.split]
        if args.run_smoke:
            dry_argv = ["--smoke", *dry_argv]
        return int(mod.main(dry_argv))

    if args.stage3_dry_init:
        mod = _load_script(
            "gbt_tf_fixed_half_smoke",
            ROOT / "scripts" / "run_gbt_tf_fixed_half_stdfloor_smoke30.py",
        )
        return int(mod.main(["--stage3-dry-init", "--synthetic", "--split", args.split]))

    if args.run_smoke:
        steps = SMOKE_MAX_STEPS if args.max_optimizer_steps is None else int(args.max_optimizer_steps)
        if steps > SMOKE_MAX_STEPS:
            raise RuntimeError(
                f"--run-smoke max steps must be <= {SMOKE_MAX_STEPS} (got {steps})"
            )
        mod = _load_script(
            "gbt_tf_fixed_half_smoke",
            ROOT / "scripts" / "run_gbt_tf_fixed_half_stdfloor_smoke30.py",
        )
        return int(mod.main(["--run-smoke", "--max-optimizer-steps", str(steps), "--split", args.split]))

    if args.run_train:
        steps = (
            EXECUTED_STOP_STEP
            if args.max_optimizer_steps is None
            else int(args.max_optimizer_steps)
        )
        if steps != EXECUTED_STOP_STEP:
            raise RuntimeError(
                f"--run-train requires exactly {EXECUTED_STOP_STEP} steps (got {steps})"
            )
        mod = _load_script(
            "gbt_tf_fixed_half_full1500",
            ROOT / "scripts" / "run_gbt_tf_fixed_half_stdfloor_full1500.py",
        )
        return int(mod.main(["--run-train", "--split", args.split]))

    recipe = resolved_recipe()
    if args.dump_recipe:
        out = Path(args.dump_recipe)
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(recipe, indent=2) + "\n", encoding="utf-8")
        print(f"wrote recipe arm={ARM} objective={OBJECTIVE_ID} -> {out}")
        return 0

    print(
        json.dumps(
            {
                "arm": ARM,
                "objective_id": OBJECTIVE_ID,
                "weight_mode": WEIGHT_MODE,
                "status": "config_ok_no_train",
                "tf_input_view": TF_INPUT_VIEW,
                "tf_target_names": list(TF_TARGET_NAMES),
                "executed_stop_step": recipe["executed_stop_step"],
                "schedule_horizon": recipe["schedule_horizon"],
                "result_root": recipe["result_root"],
                "ckpt_root": recipe["ckpt_root"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
