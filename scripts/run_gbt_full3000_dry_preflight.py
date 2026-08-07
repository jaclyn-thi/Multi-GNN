#!/usr/bin/env python3
"""Dry configuration/checkpoint preflight for GBT full3000 (no training)."""

from __future__ import annotations

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
    OBJECTIVE_ID,
    PHASE3_INIT_SHA_PREFIX,
    PHASE3_SHARED_INIT,
    TOTAL_STEPS,
    WARMUP_STEPS,
    resolved_recipe,
)
from phase4b_objective_ablation.matching import load_long_seed_hashes  # noqa: E402
from mixed_ssl_phase4a.preflight import preflight_phase4a  # noqa: E402
from mixed_ssl_phase4a.domain_registry import default_smoke_domains  # noqa: E402
import hashlib


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    recipe = resolved_recipe()
    errors = []
    if recipe["max_optimizer_steps"] != TOTAL_STEPS:
        errors.append("max steps != 3000")
    if recipe["warmup_steps"] != WARMUP_STEPS or recipe["linear_decay_steps"] != DECAY_STEPS:
        errors.append("LR schedule not LONG 600/2400")
    if list(recipe["checkpoint_steps"]) != list(CHECKPOINT_STEPS):
        errors.append(f"checkpoint_steps {recipe['checkpoint_steps']} != {CHECKPOINT_STEPS}")
    if recipe.get("is_smoke"):
        errors.append("recipe marked is_smoke")
    if recipe["result_root"] != FULL3000_RESULT_ROOT:
        errors.append("result_root mismatch")
    if recipe["ckpt_root"] != FULL3000_CKPT_ROOT:
        errors.append("ckpt_root mismatch")
    if recipe.get("infonce_enabled") or recipe.get("tfmoe_enabled"):
        errors.append("forbidden objectives enabled")
    if recipe.get("contrast_projection_head") or recipe.get("view2_detach"):
        errors.append("projection/detach contract violated")
    if int(recipe.get("loader_num_workers", -1)) != 0:
        errors.append("loader_workers must be 0")

    # Formula in loss source
    loss_src = (ROOT / "graph_barlow_twins_r198/loss.py").read_text(encoding="utf-8")
    if "(z_a_f - mean_a) / (std_a + float(eps))" not in loss_src:
        errors.append("loss formula missing")

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

    # LONG seed hashes available (1000/domain)
    long_ok = {}
    for d in recipe["domains"]:
        ref = load_long_seed_hashes(ROOT, d, limit=1000)
        long_ok[d] = {"ok": bool(ref.get("ok")), "n_hashes": ref.get("n_hashes"), "path": ref.get("path")}
        if not ref.get("ok") or int(ref.get("n_hashes") or 0) < 1000:
            errors.append(f"LONG seed hashes incomplete for {d}")

    # Output paths must not contain incompatible run artifacts
    out_dir = ROOT / FULL3000_RESULT_ROOT
    ckpt_dir = ROOT / FULL3000_CKPT_ROOT
    for p in (out_dir / "logs" / "steps.jsonl", out_dir / "aggregate.json"):
        if p.is_file() and p.stat().st_size > 0:
            errors.append(f"nonempty prior run artifact: {p}")
    if ckpt_dir.is_dir():
        existing = list(ckpt_dir.glob("checkpoint_*.pt")) + list(ckpt_dir.glob("checkpoint_*.tar"))
        if existing:
            errors.append(f"existing checkpoints under {ckpt_dir}: {[x.name for x in existing[:5]]}")

    # Must not target smoke paths
    if "smoke30" in FULL3000_RESULT_ROOT or "smoke30" in FULL3000_CKPT_ROOT:
        errors.append("full paths collide with smoke")

    pre = preflight_phase4a(root=ROOT, specs=list(default_smoke_domains()))
    if not pre.get("ok"):
        errors.append("phase4a data preflight failed")

    payload = {
        "ok": not errors,
        "errors": errors,
        "objective_id": OBJECTIVE_ID,
        "recipe_head": {
            "max_optimizer_steps": recipe["max_optimizer_steps"],
            "warmup_steps": recipe["warmup_steps"],
            "linear_decay_steps": recipe["linear_decay_steps"],
            "checkpoint_steps": recipe["checkpoint_steps"],
            "result_root": recipe["result_root"],
            "ckpt_root": recipe["ckpt_root"],
            "loss_definition": recipe.get("loss_definition"),
        },
        "init_sha256": init_sha,
        "init_file_sha256": file_sha,
        "long_seed_hashes": long_ok,
        "phase4a_preflight_ok": bool(pre.get("ok")),
        "resume_from_smoke": False,
    }
    out = ROOT / FULL3000_RESULT_ROOT / "dry_preflight.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
