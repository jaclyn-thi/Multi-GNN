#!/usr/bin/env python3
"""Dry CPU preflight for MIXED_3DOMAIN_GBT_TF_FIXED_HALF_STDFLOOR_1E4.

No graph training, no GPU required, no test access.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from direct_r198 import TF_MOE_TARGET_NAMES  # noqa: E402
from gbt_tf_adaptive_stdfloor_r198.integrity import refuse_test_split_access  # noqa: E402
from gbt_tf_fixed_half_stdfloor_r198 import (  # noqa: E402
    ARM,
    BETA_FREEZE_UNTIL,
    CKPT_ROOT,
    EXECUTED_STOP_STEP,
    FIRST_BETA_UPDATE,
    FIXED_ALPHA,
    FIXED_W_GBT,
    FIXED_W_TF_MASS,
    OBJECTIVE_ID,
    PARENT_GBT_OBJECTIVE_ID,
    RESULT_ROOT,
    SCHEDULE_HORIZON,
    SMOKE_CKPT_ROOT,
    SMOKE_MAX_STEPS,
    SMOKE_RESULT_ROOT,
    TF_CACHE_BY_DOMAIN,
    TF_INPUT_VIEW,
    TF_TARGET_NAMES,
    WEIGHT_MODE,
    resolved_recipe,
)
from graph_barlow_twins_r198 import (  # noqa: E402
    GBT_STD_FLOOR_1E4,
    SMOKE30_CKPT_ROOT,
    SMOKE30_RESULT_ROOT,
    STDFLOOR_FULL3000_CKPT_ROOT,
    STDFLOOR_FULL3000_RESULT_ROOT,
)
from mixed_ssl_phase4a import CALIB_OBS_PER_DOMAIN, CONTRACT_ID, SEED  # noqa: E402
from mixed_ssl_phase4a.domain_registry import default_smoke_domains  # noqa: E402
from mixed_ssl_phase4a.preflight import preflight_phase4a  # noqa: E402
from mixed_ssl_phase4b import (  # noqa: E402
    CANONICAL_DOMAINS,
    MIXED_LONG_LINEAR_DECAY_STEPS,
    MIXED_LONG_WARMUP_STEPS,
    PHASE3_INIT_SHA_PREFIX,
    PHASE3_SHARED_INIT,
)
from phase4b_objective_ablation.matching import load_long_seed_hashes  # noqa: E402

PHASE3_INIT_FULL_SHA = (
    "8821c986c7394caf504393830dc33a9c3c97ba4d5fdd3bcbaa19f70421c7aebc"
)

FORBIDDEN_COLLISION_ROOTS = (
    SMOKE30_RESULT_ROOT,
    SMOKE30_CKPT_ROOT,
    STDFLOOR_FULL3000_RESULT_ROOT,
    STDFLOOR_FULL3000_CKPT_ROOT,
    "results/diagnostics/financial_multidataset_gbt_tf_adaptive_stdfloor_1e4",
    "results/checkpoints/financial_multidataset_gbt_tf_adaptive_stdfloor_1e4_seed2",
    "results/diagnostics/financial_multidataset_gbt_tf_adaptive_stdfloor_1e4_smoke30",
    "results/checkpoints/financial_multidataset_gbt_tf_adaptive_stdfloor_1e4_smoke30_seed2",
    "results/diagnostics/financial_multidataset_shared_core_phase4b_mixed_long_3000",
    "results/checkpoints/financial_multidataset_shared_core_phase4b_mixed_long_3000_seed2",
)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=f"{ARM} dry preflight")
    p.add_argument("--smoke", action="store_true", help="Validate smoke30 recipe/paths")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--out", type=str, default="")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    refuse_test_split_access(args.split)
    recipe = resolved_recipe(smoke_steps=SMOKE_MAX_STEPS if args.smoke else None)
    errors = []

    if recipe["arm"] != ARM:
        errors.append(f"arm={recipe['arm']}")
    if recipe["objective_id"] != OBJECTIVE_ID:
        errors.append("objective_id mismatch")
    if recipe["parent_gbt_objective_id"] != PARENT_GBT_OBJECTIVE_ID:
        errors.append("parent GBT objective mismatch")
    if recipe["contract_id"] != CONTRACT_ID:
        errors.append(f"contract_id={recipe['contract_id']}")
    if list(recipe["domains"]) != list(CANONICAL_DOMAINS):
        errors.append(f"domains={recipe['domains']}")
    if int(recipe["edge_dim"]) != 6:
        errors.append("edge_dim != 6")
    if int(recipe["representation_dim"]) != 198:
        errors.append("R198 dim mismatch")
    if int(recipe["seed"]) != SEED:
        errors.append("seed != 2")
    if float(recipe["gbt_std_floor"]) != float(GBT_STD_FLOOR_1E4):
        errors.append("gbt_std_floor != 1e-4")
    if recipe.get("weight_mode") != WEIGHT_MODE:
        errors.append(f"weight_mode={recipe.get('weight_mode')}")
    if recipe.get("learn_alpha") is not False:
        errors.append("learn_alpha must be False")
    if recipe.get("learn_beta") is not True:
        errors.append("learn_beta must be True")
    if float(recipe.get("fixed_w_gbt", -1)) != float(FIXED_W_GBT):
        errors.append("fixed_w_gbt != 0.5")
    if float(recipe.get("fixed_w_tf_mass", -1)) != float(FIXED_W_TF_MASS):
        errors.append("fixed_w_tf_mass != 0.5")
    if float(recipe.get("fixed_alpha", -1)) != float(FIXED_ALPHA):
        errors.append("fixed_alpha != 0.5")
    if recipe.get("contrast_projection_head") or recipe.get("infonce_enabled"):
        errors.append("projection/InfoNCE must be off")
    if not recipe.get("tfmoe_enabled"):
        errors.append("TF must be on")
    if recipe.get("view2_detach") or not recipe.get("both_views_require_grad"):
        errors.append("both GBT views must be live")
    if recipe.get("amp"):
        errors.append("AMP must be off")
    if int(recipe.get("loader_num_workers", -1)) != 0:
        errors.append("loader_num_workers must be 0")
    if recipe["tf_input_view"] != TF_INPUT_VIEW:
        errors.append("tf_input_view mismatch")
    if list(recipe["tf_target_names"]) != list(TF_TARGET_NAMES):
        errors.append(f"TF targets {recipe['tf_target_names']}")
    if list(TF_TARGET_NAMES) != list(TF_MOE_TARGET_NAMES):
        errors.append("TF_TARGET_NAMES drifted from direct_r198")
    if int(recipe["calib_obs_per_domain"]) != int(CALIB_OBS_PER_DOMAIN):
        errors.append("calib_obs_per_domain != 5")
    if int(recipe["alpha_freeze_until"]) != int(BETA_FREEZE_UNTIL):
        errors.append("beta freeze_until != 15")
    if int(recipe["first_beta_update_step"]) != int(FIRST_BETA_UPDATE):
        errors.append("first_beta_update != 16")
    if int(recipe["warmup_steps"]) != MIXED_LONG_WARMUP_STEPS:
        errors.append("warmup not LONG 600")
    if int(recipe["linear_decay_steps"]) != MIXED_LONG_LINEAR_DECAY_STEPS:
        errors.append("decay not LONG 2400")
    if int(recipe["schedule_horizon"]) != int(SCHEDULE_HORIZON):
        errors.append("schedule_horizon != 3000")
    if recipe.get("skip_test_eval") is not True or recipe.get("forbid_test_split") is not True:
        errors.append("test access not forbidden")

    if args.smoke:
        if int(recipe["max_optimizer_steps"]) != SMOKE_MAX_STEPS:
            errors.append("smoke steps != 30")
        if int(recipe["steps_per_domain"]) != 10:
            errors.append("smoke steps_per_domain != 10")
        if recipe["result_root"] != SMOKE_RESULT_ROOT:
            errors.append("smoke result_root mismatch")
        if recipe["ckpt_root"] != SMOKE_CKPT_ROOT:
            errors.append("smoke ckpt_root mismatch")
        if recipe["lr_schedule_total_planned_steps"] != (
            MIXED_LONG_WARMUP_STEPS + MIXED_LONG_LINEAR_DECAY_STEPS
        ):
            errors.append("smoke LR schedule was rescaled")
        if int(recipe.get("executed_stop_step", -1)) != SMOKE_MAX_STEPS:
            errors.append("smoke executed_stop_step != 30")
    else:
        if int(recipe["max_optimizer_steps"]) != EXECUTED_STOP_STEP:
            errors.append("full executed steps != 1500")
        if int(recipe["steps_per_domain"]) != 500:
            errors.append("full steps_per_domain != 500")
        if list(recipe.get("checkpoint_steps") or []) != [750, 1500]:
            errors.append(f"checkpoint_steps={recipe.get('checkpoint_steps')}")
        if recipe["result_root"] != RESULT_ROOT or recipe["ckpt_root"] != CKPT_ROOT:
            errors.append("full output path mismatch")
        if recipe.get("is_smoke"):
            errors.append("full recipe marked is_smoke")

    loss_src = (ROOT / "graph_barlow_twins_r198/loss.py").read_text(encoding="utf-8")
    if "torch.clamp_min(std_a_raw, floor)" not in loss_src:
        errors.append("stdfloor clamp missing")
    step_src = (ROOT / "gbt_tf_adaptive_stdfloor_r198/step.py").read_text(encoding="utf-8")
    if "edge_aligned_graph_barlow_twins_r198_stdfloor_1e4" not in step_src:
        errors.append("hybrid step missing stdfloor GBT")
    if "tf_moe_mae_losses(z1_seed, seed_id1" not in step_src:
        errors.append("TF path must use z1_seed only")
    if "z2_seed = z2_seed.detach()" in step_src:
        errors.append("z2_seed must not be detached before GBT")
    if 'weight_mode == "fixed_half"' not in (
        ROOT / "direct_r198/__init__.py"
    ).read_text(encoding="utf-8") and "fixed_half" not in (
        ROOT / "direct_r198/__init__.py"
    ).read_text(encoding="utf-8"):
        errors.append("fixed_half weight mode missing from direct_r198")

    init_path = ROOT / PHASE3_SHARED_INIT
    init_sha = None
    file_sha = None
    edge_dim = None
    r198_ok = False
    if not init_path.is_file():
        errors.append(f"missing Phase-3 init {init_path}")
    else:
        import torch

        blob = torch.load(init_path, map_location="cpu", weights_only=False)
        init_sha = str(blob.get("init_sha256", ""))
        file_sha = file_sha256(init_path)
        if not init_sha.startswith(PHASE3_INIT_SHA_PREFIX):
            errors.append(f"init sha prefix {init_sha[:16]} != {PHASE3_INIT_SHA_PREFIX}")
        if init_sha != PHASE3_INIT_FULL_SHA:
            errors.append("Phase-3 init full SHA mismatch")
        ew = blob["model_state_dict"].get("edge_emb.node__to__node.weight")
        if ew is None:
            errors.append("missing edge_emb weight in init")
        else:
            edge_dim = int(ew.shape[-1])
            if edge_dim != 6:
                errors.append(f"init edge_dim={edge_dim}")
        for k, v in blob["model_state_dict"].items():
            if k.endswith("emlps.0.0.node__to__node.weight"):
                r198_ok = int(v.shape[-1]) == 198
                break
        if not r198_ok:
            errors.append("R198 dim not found/verified in Phase-3 init")
        if "moe_state_dict" not in blob or "alpha_beta_state_dict" not in blob:
            errors.append("Phase-3 init missing moe/alpha_beta")
        for bad in ("test_embeddings", "test_edge_ids", "y_test"):
            if bad in blob:
                errors.append(f"init contains forbidden key {bad}")

    pre = preflight_phase4a(root=ROOT, specs=list(default_smoke_domains()))
    if not pre.get("ok"):
        errors.append("phase4a domain/TF preflight failed")
    for d, rel in TF_CACHE_BY_DOMAIN.items():
        cache = ROOT / rel
        if not cache.is_dir():
            errors.append(f"missing TF cache {d}: {cache}")
            continue
        if not (cache / "split_train_edge_id.npy").is_file():
            errors.append(f"{d}: missing train TF split")
        if not (cache / "features.npy").is_file():
            errors.append(f"{d}: missing features.npy")

    long_ok = {}
    for d in CANONICAL_DOMAINS:
        need = 10 if args.smoke else 500
        ref = load_long_seed_hashes(ROOT, d, limit=need)
        long_ok[d] = {
            "ok": int(ref.get("n_hashes") or 0) >= need,
            "n_hashes": ref.get("n_hashes"),
            "path": ref.get("path"),
        }
        if int(ref.get("n_hashes") or 0) < need:
            errors.append(
                f"LONG seed hashes incomplete for {d}: have {ref.get('n_hashes')} need {need}"
            )

    out_root = recipe["result_root"]
    ckpt_root = recipe["ckpt_root"]
    for hist in FORBIDDEN_COLLISION_ROOTS:
        if out_root == hist or ckpt_root == hist:
            errors.append(f"output collides with historical tree {hist}")
    if "gbt_tf_fixed_half_stdfloor_1e4" not in out_root:
        errors.append("result_root must be unique gbt_tf_fixed_half tree")
    if "gbt_tf_fixed_half_stdfloor_1e4" not in ckpt_root:
        errors.append("ckpt_root must be unique gbt_tf_fixed_half tree")

    if args.smoke:
        ckpt_dir = ROOT / SMOKE_CKPT_ROOT
        if ckpt_dir.is_dir():
            existing = list(ckpt_dir.glob("checkpoint_*.pt")) + list(
                ckpt_dir.glob("checkpoint_*.tar")
            )
            if existing:
                errors.append(
                    f"refuse overwrite existing smoke ckpts: {[x.name for x in existing[:5]]}"
                )

    payload = {
        "ok": not errors,
        "errors": errors,
        "arm": ARM,
        "objective_id": OBJECTIVE_ID,
        "mode": "smoke30" if args.smoke else "full1500_config",
        "recipe_head": {
            "max_optimizer_steps": recipe["max_optimizer_steps"],
            "executed_stop_step": recipe.get("executed_stop_step"),
            "schedule_horizon": recipe.get("schedule_horizon"),
            "steps_per_domain": recipe["steps_per_domain"],
            "warmup_steps": recipe["warmup_steps"],
            "linear_decay_steps": recipe["linear_decay_steps"],
            "lr_schedule_total_planned_steps": recipe["lr_schedule_total_planned_steps"],
            "gbt_std_floor": recipe["gbt_std_floor"],
            "weight_mode": recipe["weight_mode"],
            "fixed_w_gbt": recipe.get("fixed_w_gbt"),
            "fixed_w_tf_mass": recipe.get("fixed_w_tf_mass"),
            "learn_alpha": recipe.get("learn_alpha"),
            "learn_beta": recipe.get("learn_beta"),
            "alpha_policy": recipe.get("alpha_policy"),
            "tf_input_view": recipe["tf_input_view"],
            "tf_target_names": recipe["tf_target_names"],
            "calib_obs_per_domain": recipe["calib_obs_per_domain"],
            "alpha_freeze_until": recipe["alpha_freeze_until"],
            "first_beta_update_step": recipe.get("first_beta_update_step"),
            "result_root": recipe["result_root"],
            "ckpt_root": recipe["ckpt_root"],
            "loader_num_workers": recipe["loader_num_workers"],
            "amp": recipe["amp"],
            "contrast_projection_head": recipe["contrast_projection_head"],
            "both_views_require_grad": recipe["both_views_require_grad"],
            "loss_definition": recipe["loss_definition"],
        },
        "shared_init_path": str(init_path),
        "shared_init_sha256": init_sha,
        "shared_init_file_sha256": file_sha,
        "shared_init_edge_dim": edge_dim,
        "shared_init_r198_ok": r198_ok,
        "phase4a_preflight_ok": bool(pre.get("ok")),
        "phase4a_preflight": {k: pre.get(k) for k in ("ok", "domains", "errors") if k in pre},
        "long_seed_hashes": long_ok,
        "tf_caches": dict(TF_CACHE_BY_DOMAIN),
        "forbidden_collision_roots_checked": list(FORBIDDEN_COLLISION_ROOTS),
        "no_test_access": True,
    }
    out = Path(args.out) if args.out else (
        ROOT / (SMOKE_RESULT_ROOT if args.smoke else RESULT_ROOT) / "dry_preflight.json"
    )
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, default=str))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
