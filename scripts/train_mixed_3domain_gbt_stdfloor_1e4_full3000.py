#!/usr/bin/env python3
"""Trainer entry for MIXED_3DOMAIN_GBT_STDFLOOR_1E4_FULL3000_SEED2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graph_barlow_twins_r198 import (  # noqa: E402
    OBJECTIVE_ID_STDFLOOR_1E4,
    STDFLOOR_FULL3000_ARM,
    TOTAL_STEPS,
    resolved_recipe_stdfloor_1e4_full3000,
)
from graph_barlow_twins_r198.integrity import refuse_test_split_access  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=f"{STDFLOOR_FULL3000_ARM} trainer entry")
    p.add_argument("--dump-recipe", type=str, default="", help="Write recipe JSON and exit")
    p.add_argument(
        "--run-train",
        action="store_true",
        help="Fresh 3000-step stdfloor training from Phase-3 shared init",
    )
    p.add_argument(
        "--max-optimizer-steps",
        type=int,
        default=None,
        help="Must be 3000 with --run-train",
    )
    p.add_argument("--split", type=str, default="train", help="train/val only; test refused")
    p.add_argument(
        "--dry-preflight-only",
        action="store_true",
        help="Config/path/shared-init dry preflight only",
    )
    return p.parse_args(argv)


def _load_script(name: str, path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def main(argv=None) -> int:
    args = parse_args(argv)
    refuse_test_split_access(args.split)

    if args.dry_preflight_only:
        mod = _load_script(
            "gbt_stdfloor_full_dry",
            ROOT / "scripts" / "run_gbt_stdfloor_full3000_dry_preflight.py",
        )
        return int(mod.main())

    if args.run_train:
        steps = TOTAL_STEPS if args.max_optimizer_steps is None else int(args.max_optimizer_steps)
        if steps != TOTAL_STEPS:
            raise RuntimeError(f"--run-train requires exactly {TOTAL_STEPS} steps (got {steps})")
        mod = _load_script(
            "gbt_stdfloor_full3000",
            ROOT / "scripts" / "run_gbt_stdfloor_full3000.py",
        )
        return int(mod.main(["--run-train", "--max-optimizer-steps", str(steps)]))

    recipe = resolved_recipe_stdfloor_1e4_full3000()
    if args.dump_recipe:
        out = Path(args.dump_recipe)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(recipe, indent=2) + "\n", encoding="utf-8")
        print(f"wrote recipe arm={STDFLOOR_FULL3000_ARM} objective={OBJECTIVE_ID_STDFLOOR_1E4} -> {out}")
        return 0

    print(
        json.dumps(
            {
                "arm": STDFLOOR_FULL3000_ARM,
                "objective_id": OBJECTIVE_ID_STDFLOOR_1E4,
                "status": "config_ok_no_train",
                "max_optimizer_steps": recipe["max_optimizer_steps"],
                "gbt_std_floor": recipe["gbt_std_floor"],
                "result_root": recipe["result_root"],
                "ckpt_root": recipe["ckpt_root"],
                "resume_exact_verified": recipe["resume_exact_verified"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
