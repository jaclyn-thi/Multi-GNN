#!/usr/bin/env python3
"""Dry configuration/checkpoint preflight for GBT stdfloor full3000 (no training)."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graph_barlow_twins_r198 import (  # noqa: E402
    CHECKPOINT_STEPS,
    DECAY_STEPS,
    FULL3000_CKPT_ROOT,
    FULL3000_RESULT_ROOT,
    OBJECTIVE_ID_STDFLOOR_1E4,
    PHASE3_INIT_SHA_PREFIX,
    PHASE3_SHARED_INIT,
    RECOVERY_STDFLOOR_CKPT_ROOT,
    RECOVERY_STDFLOOR_RESULT_ROOT,
    STDFLOOR_FULL3000_CKPT_ROOT,
    STDFLOOR_FULL3000_RESULT_ROOT,
    TOTAL_STEPS,
    WARMUP_STEPS,
    resolved_recipe_stdfloor_1e4_full3000,
)
from mixed_ssl_phase4a.domain_registry import default_smoke_domains  # noqa: E402
from mixed_ssl_phase4a.preflight import preflight_phase4a  # noqa: E402
from phase4b_objective_ablation.matching import load_long_seed_hashes  # noqa: E402


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    recipe = resolved_recipe_stdfloor_1e4_full3000()
    errors = []
    if recipe["max_optimizer_steps"] != TOTAL_STEPS:
        errors.append("max steps != 3000")
    if recipe["warmup_steps"] != WARMUP_STEPS or recipe["linear_decay_steps"] != DECAY_STEPS:
        errors.append("LR schedule not LONG 600/2400")
    if list(recipe["checkpoint_steps"]) != list(CHECKPOINT_STEPS):
        errors.append(f"checkpoint_steps {recipe['checkpoint_steps']} != {CHECKPOINT_STEPS}")
    if recipe.get("is_smoke") or recipe.get("is_recovery_scout"):
        errors.append("recipe must be full stdfloor 3000 (not smoke/recovery)")
    if recipe["objective_id"] != OBJECTIVE_ID_STDFLOOR_1E4:
        errors.append("objective_id mismatch")
    if float(recipe.get("gbt_std_floor") or 0) != 1e-4:
        errors.append("gbt_std_floor must be 1e-4")
    if recipe["result_root"] != STDFLOOR_FULL3000_RESULT_ROOT:
        errors.append("result_root mismatch")
    if recipe["ckpt_root"] != STDFLOOR_FULL3000_CKPT_ROOT:
        errors.append("ckpt_root mismatch")
    if recipe.get("infonce_enabled") or recipe.get("tfmoe_enabled"):
        errors.append("forbidden objectives enabled")
    if recipe.get("contrast_projection_head") or recipe.get("view2_detach"):
        errors.append("projection/detach contract violated")
    if int(recipe.get("loader_num_workers", -1)) != 0:
        errors.append("loader_workers must be 0")
    if recipe.get("resume_exact_verified") is not False:
        errors.append("resume_exact_verified must be false until NeighborLoader fix")

    loss_src = (ROOT / "graph_barlow_twins_r198/loss.py").read_text(encoding="utf-8")
    if "torch.clamp_min(std_a_raw, floor)" not in loss_src:
        errors.append("stdfloor formula missing from loss.py")
    if "(std_a + float(eps))" not in loss_src:
        errors.append("official eps path must remain present")

    init_path = ROOT / PHASE3_SHARED_INIT
    if not init_path.is_file():
        errors.append(f"missing init {init_path}")
        init_sha = None
        file_sha = None
    else:
        import torch

        blob = torch.load(init_path, map_location="cpu", weights_only=False)
        init_sha = str(blob.get("init_sha256", ""))
        file_sha = file_sha256(init_path)
        if not init_sha.startswith(PHASE3_INIT_SHA_PREFIX):
            errors.append(f"init sha prefix {init_sha[:16]} != {PHASE3_INIT_SHA_PREFIX}")

    long_ok = {}
    for d in recipe["domains"]:
        ref = load_long_seed_hashes(ROOT, d, limit=1000)
        long_ok[d] = {"ok": bool(ref.get("ok")), "n_hashes": ref.get("n_hashes"), "path": ref.get("path")}
        if not ref.get("ok") or int(ref.get("n_hashes") or 0) < 1000:
            errors.append(f"LONG seed hashes incomplete for {d}")

    out_dir = ROOT / STDFLOOR_FULL3000_RESULT_ROOT
    ckpt_dir = ROOT / STDFLOOR_FULL3000_CKPT_ROOT
    for p in (out_dir / "logs" / "steps.jsonl", out_dir / "aggregate.json"):
        if p.is_file() and p.stat().st_size > 0:
            errors.append(f"nonempty prior run artifact: {p}")
    if ckpt_dir.is_dir():
        existing = list(ckpt_dir.glob("checkpoint_*.pt")) + list(ckpt_dir.glob("checkpoint_*.tar"))
        if existing:
            errors.append(f"existing checkpoints under {ckpt_dir}: {[x.name for x in existing[:5]]}")

    for hist in (
        ROOT / FULL3000_RESULT_ROOT,
        ROOT / FULL3000_CKPT_ROOT,
        ROOT / RECOVERY_STDFLOOR_RESULT_ROOT,
        ROOT / RECOVERY_STDFLOOR_CKPT_ROOT,
    ):
        if out_dir.resolve() == hist.resolve() or ckpt_dir.resolve() == hist.resolve():
            errors.append(f"output collides with historical tree {hist}")

    if "smoke30" in STDFLOOR_FULL3000_RESULT_ROOT or "recovery" in STDFLOOR_FULL3000_RESULT_ROOT:
        errors.append("full paths collide with smoke/recovery naming")

    pre = preflight_phase4a(root=ROOT, specs=list(default_smoke_domains()))
    if not pre.get("ok"):
        errors.append("phase4a data preflight failed")

    payload = {
        "ok": not errors,
        "errors": errors,
        "objective_id": OBJECTIVE_ID_STDFLOOR_1E4,
        "recipe_head": {
            "max_optimizer_steps": recipe["max_optimizer_steps"],
            "warmup_steps": recipe["warmup_steps"],
            "linear_decay_steps": recipe["linear_decay_steps"],
            "checkpoint_steps": recipe["checkpoint_steps"],
            "result_root": recipe["result_root"],
            "ckpt_root": recipe["ckpt_root"],
            "gbt_std_floor": recipe.get("gbt_std_floor"),
            "loss_definition": recipe.get("loss_definition"),
            "resume_exact_verified": recipe.get("resume_exact_verified"),
        },
        "shared_init_path": str(init_path),
        "shared_init_sha256": init_sha,
        "shared_init_file_sha256": file_sha,
        "long_seed_hashes": long_ok,
        "phase4a_preflight_ok": bool(pre.get("ok")),
    }
    out = ROOT / STDFLOOR_FULL3000_RESULT_ROOT / "dry_preflight.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
