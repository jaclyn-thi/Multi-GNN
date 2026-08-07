#!/usr/bin/env python3
"""Paired cross-arm Phase-4C stochastic-stream replay (real data, Slurm only).

Loads the four domains once, then evaluates all four recipes over 40 round-robin
steps on the real orchestration/augmentation path. Writes only versioned replay
artifacts. Does not submit full training, extraction, or probes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from contrastive_projection import ContrastiveProjectionHead  # noqa: E402
from direct_r198 import LearnedAlphaBeta, LossNormState, TFMoEBundle, load_tf_moe_context  # noqa: E402
from direct_r198.lr_scheduler import DirectHWarmupLinearScheduler  # noqa: E402
from mixed_ssl_phase2.bn import apply_bn_, clone_bn_bundle, collect_bn_bundle  # noqa: E402
from mixed_ssl_phase3.hash_util import state_dict_sha256  # noqa: E402
from train_util import AddEgoIds, add_arange_ids  # noqa: E402
from util import set_seed  # noqa: E402

from phase4c_four_domain import (  # noqa: E402
    ALPHA_FREEZE_UNTIL,
    ARMS,
    CALIB_OBS,
    DOMAINS,
    FIRST_AB_UPDATE,
    PHASE3_SHARED_INIT_SHA256,
    PROJECTION_HIDDEN,
    PROJECTION_IN_DIM,
    PROJECTION_OUT,
    SEED,
    arm_unique,
    arm_uses_projection,
    arm_weight_mode,
    resolved_recipe,
)
from phase4c_four_domain.domain_registry import default_domains  # noqa: E402
from phase4c_four_domain.integrity import evaluate_no_test_policy  # noqa: E402
from phase4c_four_domain.rng import (  # noqa: E402
    PROJECTION_INIT_SEED,
    capture_full_rng_state,
    isolated_projection_initialization_rng,
    restore_full_rng_state,
)
from phase4c_four_domain.source_manifest import verify_manifest  # noqa: E402
from phase4c_four_domain.step import phase4c_step  # noqa: E402
from phase4c_four_domain.train import (  # noqa: E402
    ROOT as TRAIN_ROOT,
    _infinite,
    _load_shared_init,
    _model,
    _sha,
    atomic_json,
    build_loader,
    get_data,
    make_ns,
)

ARMS_ORDER = [
    "FOUR_DOMAIN_INFONCE_TF_ADAPTIVE_SHORT",
    "FOUR_DOMAIN_PROJECTION_INFONCE_TF_ADAPTIVE_SHORT",
    "FOUR_DOMAIN_EXPERT_ONLY_SHORT",
    "FOUR_DOMAIN_PROJECTION_INFONCE_TF_ADAPTIVE_LONG",
]
REPLAY_TAG = "cross_arm_rng_replay_v1"
STEPS = 40


def _manifest_file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _optimizer_has_param(optimizer: torch.optim.Optimizer, param: torch.nn.Parameter) -> bool:
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p is param:
                return True
    return False


def _run_arm(
    arm: str,
    *,
    graphs: Dict[str, Any],
    ns_by: Dict[str, Any],
    contexts: Dict[str, Any],
    device: torch.device,
    transform,
) -> Dict[str, Any]:
    recipe = resolved_recipe(arm)
    # Clone shared graphs so AddEgoIds / loader transforms cannot leak across arms.
    graphs = {d: graphs[d].clone() for d in DOMAINS}
    sample_loader = build_loader(graphs[DOMAINS[0]], transform, domain=DOMAINS[0])
    sample = next(iter(sample_loader))
    model = _model(ns_by[DOMAINS[0]], graphs[DOMAINS[0]], sample, device)
    del sample, sample_loader
    moe = TFMoEBundle(198, hidden=64, n_targets=3).to(device)
    alpha_beta = LearnedAlphaBeta(3, init_alpha=0.6).to(device)
    init_sha = _load_shared_init(model, moe, alpha_beta)
    if init_sha != PHASE3_SHARED_INIT_SHA256:
        raise RuntimeError("shared init SHA mismatch in replay")

    canonical_rng = capture_full_rng_state()
    projection = None
    if arm_uses_projection(arm):
        with isolated_projection_initialization_rng(PROJECTION_INIT_SEED):
            projection = ContrastiveProjectionHead(
                PROJECTION_IN_DIM, PROJECTION_HIDDEN, PROJECTION_OUT
            ).to(device)
    restore_full_rng_state(canonical_rng)

    expert_only = arm_weight_mode(arm) == "expert_only"
    if expert_only and projection is not None:
        raise RuntimeError("EXPERT_ONLY constructed a projection head")

    enc_params = list(model.parameters()) + list(moe.parameters())
    if projection is not None:
        enc_params += list(projection.parameters())
    if expert_only:
        alpha_beta.alpha_logit.requires_grad_(False)
        ab_params = [alpha_beta.beta_logits]
    else:
        ab_params = list(alpha_beta.parameters())
    optimizer = torch.optim.Adam(
        [{"params": enc_params, "lr": 0.002}, {"params": ab_params, "lr": 0.001}]
    )
    if expert_only and _optimizer_has_param(optimizer, alpha_beta.alpha_logit):
        raise RuntimeError("EXPERT_ONLY alpha_logit unexpectedly present in optimizer")
    if (not expert_only) and (not _optimizer_has_param(optimizer, alpha_beta.alpha_logit)):
        raise RuntimeError("adaptive arm missing alpha_logit in optimizer")

    scheduler = DirectHWarmupLinearScheduler(
        optimizer,
        warmup_steps=int(recipe["warmup_steps"]),
        linear_steps=int(recipe["linear_decay_steps"]),
        warmup_start=0.1,
        warmup_end=1.0,
        linear_end=0.1,
        steps_per_epoch=int(recipe["max_optimizer_steps"]),
        n_epochs=1,
    )
    loaders = {d: _infinite(build_loader(graphs[d], transform, domain=d)) for d in DOMAINS}
    norms = {d: LossNormState() for d in DOMAINS}
    bn0 = clone_bn_bundle(collect_bn_bundle(model))
    bns = {d: clone_bn_bundle(bn0) for d in DOMAINS}
    counts = {d: 0 for d in DOMAINS}
    calibration = {d: {"n": 0, "contrast": 0.0, "tf": [0.0, 0.0, 0.0]} for d in DOMAINS}
    rows: List[Dict[str, Any]] = []
    alpha_unfrozen_at: Optional[int] = None
    alpha_logit_trace: List[float] = []
    beta_trace: List[List[float]] = []

    for index in range(STEPS):
        domain = DOMAINS[index % len(DOMAINS)]
        apply_bn_(model, bns[domain])
        batch = next(loaders[domain])
        with torch.no_grad():
            alpha_logit_trace.append(float(alpha_beta.alpha_logit.detach().cpu()))
            beta_trace.append(alpha_beta.beta_logits.detach().cpu().tolist())
        stat = phase4c_step(
            arm=arm,
            global_step=index,
            domain=domain,
            model=model,
            moe=moe,
            alpha_beta=alpha_beta,
            loss_norm=norms[domain],
            tf_ctx=contexts[domain],
            optimizer=optimizer,
            batch=batch,
            loader_data=graphs[domain],
            args=ns_by[domain],
            device=device,
            projection=projection,
            seed_ids_sha_fn=_sha,
            do_optimizer_step=True,
        )
        scheduler.step()
        counts[domain] += 1
        completed = index + 1
        if calibration[domain]["n"] < CALIB_OBS and not norms[domain].calibrated:
            calibration[domain]["contrast"] += float(stat["L_contrast_raw"])
            for target in range(3):
                calibration[domain]["tf"][target] += float(stat[f"L_tf_raw_{target}"])
            calibration[domain]["n"] += 1
            if calibration[domain]["n"] == CALIB_OBS:
                norms[domain].contrast_mean = max(
                    calibration[domain]["contrast"] / CALIB_OBS, 1e-12
                )
                norms[domain].tf_means = [
                    max(value / CALIB_OBS, 1e-12) for value in calibration[domain]["tf"]
                ]
                norms[domain].calibrated = True
        all_calibrated = all(norms[d].calibrated for d in DOMAINS)
        if alpha_beta._frozen and all_calibrated and completed >= ALPHA_FREEZE_UNTIL:
            if expert_only:
                alpha_beta.set_learn_flags(learn_alpha=False, learn_beta=True)
            else:
                alpha_beta.set_frozen(False)
            alpha_unfrozen_at = completed
        bns[domain] = clone_bn_bundle(collect_bn_bundle(model))
        stat.update(
            {
                "global_optimizer_step": completed,
                "domain": domain,
                "domain_exposure_count": counts[domain],
                "all_domains_calibrated": all_calibrated,
                "alpha_logit": alpha_logit_trace[-1],
                "beta_logits": beta_trace[-1],
            }
        )
        rows.append(stat)

    # Post-update alpha/beta snapshots after each step for 20/21/22 evidence.
    post_ab = []
    for i, row in enumerate(rows):
        post_ab.append(
            {
                "completed": int(row["global_optimizer_step"]),
                "alpha_logit_pre_step": alpha_logit_trace[i],
                "alpha": float(row.get("alpha", 0.0)),
                "beta": [float(row.get(f"beta_{k}", 0.0)) for k in range(3)],
                "w_contrast": float(row.get("w_contrast", 0.0)),
                "alpha_beta_frozen": bool(row.get("alpha_beta_frozen")),
                "alpha_requires_grad": bool(row.get("alpha_requires_grad")),
                "beta_requires_grad": bool(row.get("beta_requires_grad")),
            }
        )

    expert_audit = {
        "projection_present": projection is not None,
        "projection_init_sha256": (
            state_dict_sha256(projection.state_dict()) if projection is not None else None
        ),
        "alpha_in_optimizer": _optimizer_has_param(optimizer, alpha_beta.alpha_logit),
        "beta_in_optimizer": _optimizer_has_param(optimizer, alpha_beta.beta_logits),
        "contrast_grad_contribution_all_false": all(
            r.get("contrast_grad_contribution") is False for r in rows
        ),
        "w_contrast_all_zero": all(float(r.get("w_contrast", 1.0)) == 0.0 for r in rows),
        "contrastive_loss_in_graph_all_false": all(
            r.get("contrastive_loss_in_graph") is not True for r in rows
        ),
        "alpha_requires_grad_always_false": all(
            r.get("alpha_requires_grad") is not True for r in rows
        ),
        "ab_schedule_release_at_completed": alpha_unfrozen_at,
        "note": (
            "alpha_unfrozen_at_completed metadata means schedule release; "
            "for EXPERT_ONLY it must not imply contrastive α updates."
        ),
    }
    if expert_only:
        fail = []
        if expert_audit["projection_present"]:
            fail.append("projection_present")
        if expert_audit["alpha_in_optimizer"]:
            fail.append("alpha_in_optimizer")
        if not expert_audit["contrast_grad_contribution_all_false"]:
            fail.append("contrast_grad")
        if not expert_audit["w_contrast_all_zero"]:
            fail.append("w_contrast")
        if not expert_audit["contrastive_loss_in_graph_all_false"]:
            fail.append("contrastive_loss_in_graph")
        if not expert_audit["alpha_requires_grad_always_false"]:
            fail.append("alpha_requires_grad")
        if fail:
            raise RuntimeError(f"EXPERT_ONLY fail-closed audit failed: {fail}")

    return {
        "arm": arm,
        "recipe_horizon": int(recipe["max_optimizer_steps"]),
        "warmup_steps": int(recipe["warmup_steps"]),
        "rows": rows,
        "post_ab": post_ab,
        "alpha_unfrozen_at_completed": alpha_unfrozen_at,
        "expert_audit": expert_audit if expert_only else None,
        "projection_init_sha256": (
            state_dict_sha256(projection.state_dict()) if projection is not None else None
        ),
        "projection_present": projection is not None,
        "expert_only": expert_only,
    }


def _compare(arms_out: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    ref = ARMS_ORDER[0]
    mismatches = []
    table = []
    for i in range(STEPS):
        step = i + 1
        domain = DOMAINS[i % len(DOMAINS)]
        seeds = {a: arms_out[a]["rows"][i]["seed_ids_sha256"] for a in ARMS_ORDER}
        v1 = {a: arms_out[a]["rows"][i]["view1_aug_sha256"] for a in ARMS_ORDER}
        v2 = {a: arms_out[a]["rows"][i]["view2_aug_sha256"] for a in ARMS_ORDER}
        seed_ok = len(set(seeds.values())) == 1
        v1_ok = len(set(v1.values())) == 1
        v2_ok = len(set(v2.values())) == 1
        row = {
            "global_optimizer_step": step,
            "domain": domain,
            "seed_ids_match": seed_ok,
            "view1_match": v1_ok,
            "view2_match": v2_ok,
            "seed_ids_sha256": seeds[ref],
            "view1_aug_sha256": v1[ref],
            "view2_aug_sha256": v2[ref],
        }
        table.append(row)
        if not seed_ok:
            mismatches.append({"step": step, "domain": domain, "type": "seed_ids", "values": seeds})
        if not v1_ok:
            mismatches.append({"step": step, "domain": domain, "type": "view1", "values": v1})
        if not v2_ok:
            mismatches.append({"step": step, "domain": domain, "type": "view2", "values": v2})

    # Pairwise invariants requested by the repair plan.
    infonce = "FOUR_DOMAIN_INFONCE_TF_ADAPTIVE_SHORT"
    expert = "FOUR_DOMAIN_EXPERT_ONLY_SHORT"
    proj_s = "FOUR_DOMAIN_PROJECTION_INFONCE_TF_ADAPTIVE_SHORT"
    proj_l = "FOUR_DOMAIN_PROJECTION_INFONCE_TF_ADAPTIVE_LONG"
    pairwise = {
        "infonce_expert_view1_equal": all(
            arms_out[infonce]["rows"][i]["view1_aug_sha256"]
            == arms_out[expert]["rows"][i]["view1_aug_sha256"]
            for i in range(STEPS)
        ),
        "proj_short_long_views_equal": all(
            arms_out[proj_s]["rows"][i]["view1_aug_sha256"]
            == arms_out[proj_l]["rows"][i]["view1_aug_sha256"]
            and arms_out[proj_s]["rows"][i]["view2_aug_sha256"]
            == arms_out[proj_l]["rows"][i]["view2_aug_sha256"]
            for i in range(STEPS)
        ),
        "all_arms_seed_ids_equal": all(r["seed_ids_match"] for r in table),
        "all_arms_view1_equal": all(r["view1_match"] for r in table),
        "all_arms_view2_equal": all(r["view2_match"] for r in table),
    }
    return {
        "steps": STEPS,
        "mismatches": mismatches,
        "mismatch_count": len(mismatches),
        "table": table,
        "pairwise": pairwise,
        "all_match": len(mismatches) == 0,
    }


def _calibration_evidence(arms_out: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for arm, payload in arms_out.items():
        post = {int(x["completed"]): x for x in payload["post_ab"]}
        r20, r21, r22 = post.get(20), post.get(21), post.get(22)
        # First permitted update at completed step 21: requires_grad flips after step 20.
        release = payload["alpha_unfrozen_at_completed"]
        # Numerical change: compare alpha/beta logged on step rows.
        rows = {int(r["global_optimizer_step"]): r for r in payload["rows"]}
        def ab_tuple(step: int):
            r = rows[step]
            return (
                float(r.get("alpha", 0.0)),
                float(r.get("beta_0", 0.0)),
                float(r.get("beta_1", 0.0)),
                float(r.get("beta_2", 0.0)),
            )

        changed_22 = ab_tuple(22) != ab_tuple(21) if 22 in rows and 21 in rows else None
        frozen_through_20 = all(
            bool(rows[s].get("alpha_beta_frozen")) for s in range(1, ALPHA_FREEZE_UNTIL + 1)
        )
        out[arm] = {
            "frozen_through_completed_20": frozen_through_20,
            "ab_schedule_release_at_completed": release,
            "first_permitted_update_step": FIRST_AB_UPDATE,
            "step20": r20,
            "step21": r21,
            "step22": r22,
            "alpha_beta_values_changed_by_step22": changed_22,
            "expert_only": payload["expert_only"],
            "learn_alpha_after_release": (not payload["expert_only"]),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source-manifest-path",
        type=Path,
        default=ROOT / "results/diagnostics/phase4c_four_domain_source_manifest.approved.json",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / f"results/diagnostics/phase4c_four_domain_{REPLAY_TAG}",
    )
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit(
            "Refuse real-data cross-arm RNG replay on a login node (SLURM_JOB_ID absent)."
        )
    os.chdir(ROOT)
    man_path = args.source_manifest_path
    verify_manifest(man_path)
    man_sha = _manifest_file_sha(man_path)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(SEED)
    with (ROOT / "data_config.json").open() as f:
        config = json.load(f)
    specs = default_domains()
    graphs: Dict[str, Any] = {}
    ns_by_arm: Dict[str, Dict[str, Any]] = {arm: {} for arm in ARMS_ORDER}
    for spec in specs:
        # Load each domain once; arm-specific namespaces only differ by weight mode / unique.
        ns_load = make_ns(
            spec.dataset_id, f"phase4c_{REPLAY_TAG}_seed{SEED}", ARMS_ORDER[0], STEPS
        )
        ns_load.direct_r198_tfmoe_cache = str(ROOT / spec.tf_cache_path)
        graph = get_data(ns_load, config)
        add_arange_ids([graph])
        graphs[spec.dataset_id] = graph
        for arm in ARMS_ORDER:
            ns = make_ns(
                spec.dataset_id,
                arm_unique(arm),
                arm,
                int(resolved_recipe(arm)["max_optimizer_steps"]),
            )
            ns.direct_r198_tfmoe_cache = str(ROOT / spec.tf_cache_path)
            ns_by_arm[arm][spec.dataset_id] = ns

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform = AddEgoIds()
    contexts = {s.dataset_id: load_tf_moe_context(ROOT / s.tf_cache_path, device) for s in specs}

    arms_out: Dict[str, Dict[str, Any]] = {}
    for arm in ARMS_ORDER:
        print(f"REPLAY_ARM_BEGIN {arm}", flush=True)
        # Reset global seed before each arm so construction starts identically;
        # isolation still required for projection-on arms.
        set_seed(SEED)
        arms_out[arm] = _run_arm(
            arm,
            graphs=graphs,
            ns_by=ns_by_arm[arm],
            contexts=contexts,
            device=device,
            transform=transform,
        )
        print(f"REPLAY_ARM_END {arm}", flush=True)

    comparison = _compare(arms_out)
    calibration = _calibration_evidence(arms_out)
    no_test = evaluate_no_test_policy(
        {
            "skip_test_eval": True,
            "test_graph_loaded": False,
            "test_metrics_computed": False,
        },
        test_evaluated=False,
    )
    expert_ok = arms_out["FOUR_DOMAIN_EXPERT_ONLY_SHORT"]["expert_audit"]
    authorized = bool(
        comparison["all_match"]
        and no_test["ok"]
        and expert_ok is not None
        and expert_ok["contrast_grad_contribution_all_false"]
        and not expert_ok["projection_present"]
        and not expert_ok["alpha_in_optimizer"]
        and all(calibration[a]["frozen_through_completed_20"] for a in ARMS_ORDER)
        and all(
            calibration[a]["ab_schedule_release_at_completed"] == ALPHA_FREEZE_UNTIL
            for a in ARMS_ORDER
        )
    )
    verdict = "FULL_TRAINING_AUTHORIZED" if authorized else "FULL_TRAINING_BLOCKED"

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "replay_tag": REPLAY_TAG,
        "source_manifest_path": str(man_path),
        "source_manifest_sha256": man_sha,
        "reason": "ISOLATE_PROJECTION_INITIALIZATION_FROM_AUGMENTATION_RNG",
        "steps": STEPS,
        "arms": ARMS_ORDER,
        "comparison": {
            "all_match": comparison["all_match"],
            "mismatch_count": comparison["mismatch_count"],
            "mismatches_head": comparison["mismatches"][:8],
            "pairwise": comparison["pairwise"],
        },
        "calibration": calibration,
        "expert_audit": expert_ok,
        "no_test_policy": no_test,
        "projection_init_sha_by_arm": {
            a: arms_out[a]["projection_init_sha256"] for a in ARMS_ORDER
        },
        "authorization_verdict": verdict,
        "full_training_submitted": False,
        "extraction_submitted": False,
        "probes_submitted": False,
        "test_evaluated": False,
        "end": "SUCCESS" if authorized else "FAILED",
    }
    atomic_json(out_dir / "replay_summary.json", summary)
    atomic_json(out_dir / "per_step_seed_view_table.json", {"table": comparison["table"]})
    atomic_json(
        out_dir / "per_arm_step_hashes.json",
        {
            arm: [
                {
                    "global_optimizer_step": r["global_optimizer_step"],
                    "domain": r["domain"],
                    "seed_ids_sha256": r["seed_ids_sha256"],
                    "view1_aug_sha256": r["view1_aug_sha256"],
                    "view2_aug_sha256": r["view2_aug_sha256"],
                    "w_contrast": r.get("w_contrast"),
                    "contrast_grad_contribution": r.get("contrast_grad_contribution"),
                    "alpha_beta_frozen": r.get("alpha_beta_frozen"),
                    "alpha_requires_grad": r.get("alpha_requires_grad"),
                    "beta_requires_grad": r.get("beta_requires_grad"),
                    "projection_present": r.get("projection_present"),
                }
                for r in arms_out[arm]["rows"]
            ]
            for arm in ARMS_ORDER
        },
    )
    auth = {
        "generated_at_utc": summary["generated_at_utc"],
        "slurm_job_id": summary["slurm_job_id"],
        "old_source_manifest_sha256": None,  # filled by submit wrapper if provided
        "new_source_manifest_sha256": man_sha,
        "reason": "ISOLATE_PROJECTION_INITIALIZATION_FROM_AUGMENTATION_RNG",
        "seed_and_view_match_all_40_steps": comparison["all_match"],
        "authorization_verdict": verdict,
        "full_training_authorized": authorized,
        "full_training_submitted": False,
        "replay_artifact_dir": str(out_dir),
    }
    env_old = os.environ.get("PHASE4C_OLD_MANIFEST_SHA")
    if env_old:
        auth["old_source_manifest_sha256"] = env_old
    atomic_json(out_dir / "cross_arm_replay_authorization.json", auth)
    # Convenience pointer next to diagnostics root (versioned name, does not clobber smokes).
    atomic_json(
        ROOT / "results/diagnostics/phase4c_four_domain_cross_arm_replay_authorization_v1.json",
        auth,
    )
    print(json.dumps({"end": summary["end"], "verdict": verdict, "mismatches": comparison["mismatch_count"]}, indent=2))
    return 0 if authorized else 2


if __name__ == "__main__":
    # Silence unused import lint for TRAIN_ROOT while documenting shared root.
    assert TRAIN_ROOT == ROOT
    raise SystemExit(main())
