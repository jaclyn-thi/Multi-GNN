"""Minimal trainer entry for MIXED_3DOMAIN_GRAPH_BARLOW_TWINS_ONLY.

Supports:
  --dump-recipe
  --memory-preflight-only
  --run-smoke --max-optimizer-steps 30
  --run-train   (full 3000-step; authorized after smoke PASS)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graph_barlow_twins_r198 import (  # noqa: E402
    ARM,
    OBJECTIVE_ID,
    SMOKE30_MAX_STEPS,
    TOTAL_STEPS,
    resolved_recipe,
)
from graph_barlow_twins_r198.integrity import refuse_test_split_access  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=f"{ARM} trainer entry")
    p.add_argument("--smoke-steps", type=int, default=None, help="Override steps for recipe dump")
    p.add_argument("--dump-recipe", type=str, default="", help="Write recipe JSON and exit")
    p.add_argument(
        "--memory-preflight-only",
        action="store_true",
        help="Real-data dual-view GPU memory preflight (no optimizer updates)",
    )
    p.add_argument(
        "--run-smoke",
        action="store_true",
        help=f"Run constrained smoke (<= {SMOKE30_MAX_STEPS} optimizer steps)",
    )
    p.add_argument(
        "--max-optimizer-steps",
        type=int,
        default=None,
        help="With --run-smoke: <=30. With --run-train: must be 3000.",
    )
    p.add_argument(
        "--run-train",
        action="store_true",
        help="Full 3000-step training (fresh Phase-3 init; no smoke resume)",
    )
    p.add_argument(
        "--split",
        type=str,
        default="train",
        help="Must remain train/val only; test is refused",
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

    if args.memory_preflight_only:
        mod = _load_script(
            "gbt_mem_preflight", ROOT / "scripts" / "run_gbt_dual_view_memory_preflight.py"
        )
        return int(mod.main(["--memory-preflight-only"]))

    if args.run_smoke and args.run_train:
        raise RuntimeError("pass only one of --run-smoke / --run-train")

    if args.run_smoke:
        steps = SMOKE30_MAX_STEPS if args.max_optimizer_steps is None else int(args.max_optimizer_steps)
        if steps > SMOKE30_MAX_STEPS:
            raise RuntimeError(
                f"--run-smoke hard-refuses max_optimizer_steps>{SMOKE30_MAX_STEPS} (got {steps})"
            )
        mod = _load_script("gbt_smoke30", ROOT / "scripts" / "run_gbt_smoke30.py")
        return int(mod.main(["--run-smoke", "--max-optimizer-steps", str(steps)]))

    if args.run_train:
        steps = TOTAL_STEPS if args.max_optimizer_steps is None else int(args.max_optimizer_steps)
        if steps != TOTAL_STEPS:
            raise RuntimeError(f"--run-train requires exactly {TOTAL_STEPS} steps (got {steps})")
        mod = _load_script("gbt_full3000", ROOT / "scripts" / "run_gbt_full3000.py")
        return int(mod.main(["--run-train", "--max-optimizer-steps", str(steps)]))

    recipe = resolved_recipe(smoke_steps=args.smoke_steps)
    if args.dump_recipe:
        out = Path(args.dump_recipe)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(recipe, indent=2) + "\n", encoding="utf-8")
        print(f"wrote recipe arm={ARM} objective={OBJECTIVE_ID} -> {out}")
        return 0

    print(
        json.dumps(
            {
                "arm": ARM,
                "objective_id": OBJECTIVE_ID,
                "status": "config_ok_no_train",
                "max_optimizer_steps": recipe["max_optimizer_steps"],
                "warmup_steps": recipe["warmup_steps"],
                "linear_decay_steps": recipe["linear_decay_steps"],
                "result_root": recipe["result_root"],
                "ckpt_root": recipe["ckpt_root"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
