#!/usr/bin/env python3
"""Fresh 3000-step MIXED_3DOMAIN_GBT_TF_ADAPTIVE_STDFLOOR_1E4 training.

Fresh Phase-3 shared init (not smoke / GBT recovery / Phase-4B InfoNCE).
No extraction, probes, or test-split access in this job.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_loading import get_data  # noqa: E402
from direct_r198 import LearnedAlphaBeta, LossNormState, TFMoEBundle, load_tf_moe_context  # noqa: E402
from direct_r198.lr_scheduler import DirectHWarmupLinearScheduler  # noqa: E402
from gbt_tf_adaptive_stdfloor_r198 import (  # noqa: E402
    ALPHA_FREEZE_UNTIL,
    ARM,
    CHECKPOINT_STEPS,
    CKPT_ROOT,
    DECAY_STEPS,
    INIT_ALPHA,
    OBJECTIVE_ID,
    PARENT_GBT_OBJECTIVE_ID,
    RESULT_ROOT,
    ROLLING_EVERY,
    SMOKE_CKPT_ROOT,
    SMOKE_RESULT_ROOT,
    STEPS_PER_DOMAIN,
    TF_CACHE_BY_DOMAIN,
    TOTAL_STEPS,
    WARMUP_STEPS,
    resolved_recipe,
)
from gbt_tf_adaptive_stdfloor_r198.calibration import (  # noqa: E402
    fresh_calib_accumulators,
    maybe_unfreeze_alpha_beta,
    observe_calibration,
)
from gbt_tf_adaptive_stdfloor_r198.checkpoint import (  # noqa: E402
    build_checkpoint_payload,
    load_checkpoint,
    save_checkpoint,
)
from gbt_tf_adaptive_stdfloor_r198.integrity import (  # noqa: E402
    c_always_198x198_from_rows,
    filter_stats_for_jsonl,
    refuse_test_split_access,
)
from gbt_tf_adaptive_stdfloor_r198.orchestration import (  # noqa: E402
    assert_api_contracts,
    build_model,
    build_train_loader,
    file_sha256,
    infinite_loader,
    make_ns,
    peak_rss_gib,
    seed_ids_sha,
    verify_phase3_init_compatibility,
    write_json,
)
from gbt_tf_adaptive_stdfloor_r198.step import gbt_tf_adaptive_mixed_step  # noqa: E402
from mixed_ssl_phase2.bn import (  # noqa: E402
    apply_bn_,
    bn_bundle_l1,
    bn_bundles_equal,
    clone_bn_bundle,
    collect_bn_bundle,
)
from mixed_ssl_phase3.hash_util import state_dict_sha256  # noqa: E402
from mixed_ssl_phase4a import ALPHABETA_LR, CALIB_OBS_PER_DOMAIN, ENCODER_LR, SEED  # noqa: E402
from mixed_ssl_phase4a.domain_registry import default_smoke_domains  # noqa: E402
from mixed_ssl_phase4a.preflight import preflight_phase4a  # noqa: E402
from mixed_ssl_phase4a.schedule import restore_rng, round_robin_schedule, snapshot_rng  # noqa: E402
from mixed_ssl_phase4b import (  # noqa: E402
    CANONICAL_DOMAINS,
    PHASE3_INIT_SHA_PREFIX,
    PHASE3_SHARED_INIT,
)
from mixed_ssl_phase4b.matching import init_matching_rng_states  # noqa: E402
from phase4b_objective_ablation.matching import compare_vs_long  # noqa: E402
from train_util import AddEgoIds, FORWARD_EDGE_TYPE, add_arange_ids  # noqa: E402
from util import logger_setup, set_seed  # noqa: E402

GiB = 1024.0 ** 3
NOTE_PATH = ROOT / "notes/financial_multidataset_gbt_tf_adaptive_stdfloor_1e4.md"
TWIN_JSON = ROOT / "results/diagnostics/financial_multidataset_gbt_tf_adaptive_stdfloor_1e4.json"


def refuse_nonempty_run(out_dir: Path, ckpt_dir: Path) -> None:
    blockers: List[str] = []
    for p in (out_dir / "logs" / "steps.jsonl", out_dir / "aggregate.json"):
        if p.is_file() and p.stat().st_size > 0:
            blockers.append(str(p))
    for p in sorted(ckpt_dir.glob("checkpoint_*.pt")) + sorted(ckpt_dir.glob("checkpoint_*.tar")):
        blockers.append(str(p))
    forbidden_sub = (
        "smoke30",
        "phase4b",
        "phase4a",
        "objective_ablation",
        "mixed_long_3000",
        "stdfloor_1e4_recovery",
        "graph_barlow_twins_full3000",
        "graph_barlow_twins_stdfloor_1e4_full3000",
        "synth_orch",
        "stage3_dry_init",
    )
    for d in (out_dir, ckpt_dir):
        s = str(d)
        if any(x in s for x in forbidden_sub):
            blockers.append(f"forbidden_path:{s}")
    if blockers:
        raise RuntimeError(
            "Refuse startup: nonempty/incompatible artifacts:\n  " + "\n  ".join(blockers[:20])
        )


def snapshot_loader_generators(loaders: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for d, loader in loaders.items():
        g = getattr(loader, "generator", None)
        if g is None:
            out[d] = None
        else:
            out[d] = g.get_state().detach().cpu().clone()
    return out


def should_console_log(completed: int) -> bool:
    if completed <= 30:
        return True
    if completed in CHECKPOINT_STEPS:
        return True
    if completed in (300, 600):
        return True
    if completed <= 100 and completed % 10 == 0:
        return True
    if completed % 50 == 0:
        return True
    return False


def cpu_reload_validate(
    ckpt_path: Path,
    *,
    expect_step: int,
    model_ref: nn.Module,
    moe_ref: nn.Module,
    ab_ref: nn.Module,
) -> Dict[str, Any]:
    model_cpu = copy.deepcopy(model_ref).cpu()
    moe_cpu = copy.deepcopy(moe_ref).cpu()
    ab_cpu = copy.deepcopy(ab_ref).cpu()
    with torch.no_grad():
        for p in model_cpu.parameters():
            p.normal_()
        for p in moe_cpu.parameters():
            p.normal_()
    loaded = load_checkpoint(ckpt_path, model=model_cpu, moe=moe_cpu, alpha_beta=ab_cpu)
    ok = (
        loaded.get("objective_id") == OBJECTIVE_ID
        and loaded.get("arm") == ARM
        and int(loaded.get("global_step", -1)) == int(expect_step)
        and bool((loaded.get("recipe") or {}).get("tfmoe_enabled"))
        and not (loaded.get("recipe") or {}).get("infonce_enabled")
        and not (loaded.get("recipe") or {}).get("contrast_projection_head")
        and loaded.get("test_evaluated") is not True
        and bool(loaded.get("bn_bundles"))
        and state_dict_sha256(model_cpu.state_dict())
        == state_dict_sha256({k: v.detach().cpu() for k, v in model_ref.state_dict().items()})
        and state_dict_sha256(moe_cpu.state_dict())
        == state_dict_sha256({k: v.detach().cpu() for k, v in moe_ref.state_dict().items()})
    )
    return {
        "ok": bool(ok),
        "path": str(ckpt_path),
        "sha256": file_sha256(ckpt_path),
        "global_step": loaded.get("global_step"),
        "objective_id": loaded.get("objective_id"),
        "has_bn_bundles": bool(loaded.get("bn_bundles")),
        "has_optimizer": "optimizer_state_dict" in loaded,
        "has_scheduler": "scheduler_state" in loaded,
        "has_loss_norms": bool(loaded.get("loss_norm_states")),
        "test_evaluated": loaded.get("test_evaluated"),
    }


def cuda_snap() -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    free, total = torch.cuda.mem_get_info()
    return {
        "cuda_allocated_gib": torch.cuda.memory_allocated() / GiB,
        "cuda_reserved_gib": torch.cuda.memory_reserved() / GiB,
        "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated() / GiB,
        "cuda_peak_reserved_gib": torch.cuda.max_memory_reserved() / GiB,
        "cuda_free_gib": free / GiB,
        "cuda_total_gib": total / GiB,
    }


def summarize_domain_series(rows: List[Dict[str, Any]], domains: List[str], keys: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for d in domains:
        drows = [r for r in rows if r["domain"] == d]
        out[d] = {}
        for k in keys:
            vals = [float(r[k]) for r in drows if r.get(k) is not None and np.isfinite(float(r[k]))]
            if not vals:
                out[d][k] = None
                continue
            out[d][k] = {
                "first": vals[0],
                "last": vals[-1],
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "mean": float(np.mean(vals)),
            }
    return out


def write_figures(out_dir: Path, rows: List[Dict[str, Any]]) -> List[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        return [f"figures_skipped:{e}"]

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    domains = list(CANONICAL_DOMAINS)

    def series(key: str, domain: Optional[str] = None):
        xs, ys = [], []
        for r in rows:
            if domain is not None and r["domain"] != domain:
                continue
            xs.append(int(r["global_optimizer_step"]))
            ys.append(float(r.get(key, float("nan"))))
        return xs, ys

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for ax, key, title in zip(
        axes,
        ("L_gbt_raw", "L_gbt_norm", "L_total"),
        ("L_gbt_raw", "L_gbt_norm", "L_total"),
    ):
        for d in domains:
            xs, ys = series(key, d)
            ax.plot(xs, ys, label=d, linewidth=1.0)
        ax.set_ylabel(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("global optimizer step")
    fig.tight_layout()
    p = fig_dir / "01_gbt_and_total_loss.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    written.append(str(p))

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for m, ax in enumerate(axes):
        for d in domains:
            xs, ys = series(f"L_tf_raw_{m}", d)
            ax.plot(xs, ys, label=d, linewidth=1.0)
        ax.set_ylabel(f"L_tf_raw_{m}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("step")
    fig.tight_layout()
    p = fig_dir / "02_tf_raw_by_expert.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    written.append(str(p))

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    xs, ys = series("alpha")
    axes[0].plot(xs, ys, label="alpha", linewidth=1.0)
    for m in range(3):
        xs, ys = series(f"beta_{m}")
        axes[0].plot(xs, ys, label=f"beta_{m}", linewidth=1.0)
    xs, ys = series("w_contrast")
    axes[1].plot(xs, ys, label="w_contrast", linewidth=1.0)
    for m in range(3):
        xs, ys = series(f"w_tf_{m}")
        axes[1].plot(xs, ys, label=f"w_tf_{m}", linewidth=1.0)
    axes[0].set_ylabel("alpha/beta")
    axes[1].set_ylabel("effective weights")
    axes[1].set_xlabel("step")
    for ax in axes:
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = fig_dir / "03_alpha_beta_weights.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    written.append(str(p))

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for d in domains:
        xs, ys = series("effective_rank", d)
        axes[0].plot(xs, ys, label=d, linewidth=1.0)
        xs, ys = series("view1_n_floored_dims", d)
        axes[1].plot(xs, ys, label=d, linewidth=1.0)
    axes[0].set_ylabel("effective rank")
    axes[1].set_ylabel("view1 floored dims")
    axes[1].set_xlabel("step")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = fig_dir / "04_rank_and_floor.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    written.append(str(p))

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    xs, ys = series("encoder_grad_norm")
    axes[0].plot(xs, ys, linewidth=0.8)
    axes[0].set_ylabel("encoder grad")
    xs, ys = series("moe_grad_norm")
    axes[1].plot(xs, ys, linewidth=0.8, color="C1")
    axes[1].set_ylabel("moe grad")
    xs, ys = series("encoder_lr")
    axes[2].plot(xs, ys, linewidth=1.0, color="C2")
    axes[2].set_ylabel("encoder LR")
    axes[2].set_xlabel("step")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = fig_dir / "05_grads_and_lr.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    written.append(str(p))
    return written


def run_full3000() -> Dict[str, Any]:
    refuse_test_split_access("train")
    api = assert_api_contracts()
    t_wall0 = time.perf_counter()
    recipe = resolved_recipe()
    if int(recipe["max_optimizer_steps"]) != TOTAL_STEPS:
        raise RuntimeError("full recipe must be 3000 steps")
    if recipe.get("is_smoke"):
        raise RuntimeError("full run must not use smoke recipe")
    if int(recipe["warmup_steps"]) != WARMUP_STEPS or int(recipe["linear_decay_steps"]) != DECAY_STEPS:
        raise RuntimeError("LR schedule must be LONG 600/2400")
    if float(recipe["gbt_std_floor"]) != 1e-4:
        raise RuntimeError("std_floor must remain 1e-4")

    domains = list(CANONICAL_DOMAINS)
    total_steps = TOTAL_STEPS
    steps_per_domain = STEPS_PER_DOMAIN
    ckpt_steps: Set[int] = set(CHECKPOINT_STEPS)

    out_dir = ROOT / RESULT_ROOT
    logs_dir = out_dir / "logs"
    ckpt_dir = ROOT / CKPT_ROOT
    for p in (out_dir, logs_dir, ckpt_dir, NOTE_PATH.parent):
        p.mkdir(parents=True, exist_ok=True)

    for forbidden in (
        ROOT / SMOKE_RESULT_ROOT,
        ROOT / SMOKE_CKPT_ROOT,
        ROOT / "results/diagnostics/financial_multidataset_graph_barlow_twins_stdfloor_1e4_full3000_seed2",
        ROOT / "results/checkpoints/financial_multidataset_graph_barlow_twins_stdfloor_1e4_full3000_seed2",
        ROOT / "results/diagnostics/financial_multidataset_shared_core_phase4b_mixed_long_3000",
        ROOT / "results/checkpoints/financial_multidataset_shared_core_phase4b_mixed_long_3000_seed2",
    ):
        if out_dir.resolve() == forbidden.resolve() or ckpt_dir.resolve() == forbidden.resolve():
            raise RuntimeError(f"full3000 path collides with historical tree: {forbidden}")

    refuse_nonempty_run(out_dir, ckpt_dir)
    write_json(out_dir / "api_contract.json", api)

    logger_setup()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for full3000")
    device = torch.device("cuda:0")
    set_seed(SEED)

    pre = preflight_phase4a(root=ROOT, specs=list(default_smoke_domains()))
    write_json(out_dir / "preflight.json", pre)
    if not pre.get("ok"):
        raise RuntimeError(f"phase4a preflight failed: {pre}")
    write_json(out_dir / "recipe.json", recipe)

    with open(ROOT / "data_config.json") as f:
        data_config = json.load(f)

    unique = f"gbt_tf_adaptive_stdfloor_full3000_seed{SEED}"
    ns_by: Dict[str, Any] = {}
    data_by: Dict[str, Any] = {}
    tf_ctx: Dict[str, Any] = {}
    mem: Dict[str, Any] = {
        "loader_num_workers": 0,
        "rss_gib_after_each_domain_load": {},
        "graph_build_sec": {},
        "test_graph_loaded": False,
        "test_metrics_computed": False,
        "no_frozen_embedding_extraction": True,
        "no_downstream_probe": True,
        "no_test_split_access": True,
        "memory_preflight_job_reused": "19625552",
    }
    transform = AddEgoIds()

    for d in domains:
        logging.info("Loading %s (skip_test_eval; train graph only) ...", d)
        t0 = time.perf_counter()
        ns = make_ns(d, unique=unique, max_steps=total_steps)
        set_seed(SEED)
        tr, va, te, tr_i, va_i, te_i = get_data(ns, data_config)
        del va, te, tr_i, va_i, te_i
        gc.collect()
        if int(tr[FORWARD_EDGE_TYPE].edge_attr.shape[1]) != 6:
            raise RuntimeError(f"{d} edge_dim != 6")
        add_arange_ids([tr])
        ns_by[d] = ns
        data_by[d] = tr
        mem["graph_build_sec"][d] = time.perf_counter() - t0
        mem["rss_gib_after_each_domain_load"][d] = peak_rss_gib()

    for d in domains:
        tf_ctx[d] = load_tf_moe_context(ROOT / TF_CACHE_BY_DOMAIN[d], device)

    set_seed(SEED)
    sample_dom = domains[0]
    sample_loader = build_train_loader(data_by[sample_dom], transform, domain=sample_dom)
    sample = next(iter(sample_loader))
    del sample_loader
    model, emb_dim = build_model(ns_by[sample_dom], data_by[sample_dom], sample, device)
    del sample
    moe = TFMoEBundle(in_dim=int(emb_dim), hidden=64, n_targets=3).to(device)
    alpha_beta = LearnedAlphaBeta(n_tf=3, init_alpha=INIT_ALPHA).to(device)
    alpha_beta.set_frozen(True)

    init_prov = verify_phase3_init_compatibility(
        model, moe, alpha_beta, ROOT / PHASE3_SHARED_INIT
    )
    write_json(out_dir / "shared_init_provenance.json", init_prov)
    init_sha = str(init_prov["init_sha256"])
    enc_sha0 = state_dict_sha256(model.state_dict())
    moe_sha0 = state_dict_sha256(moe.state_dict())

    loaders = {d: build_train_loader(data_by[d], transform, domain=d) for d in domains}
    for d, loader in loaders.items():
        if getattr(loader, "generator", None) is None:
            raise RuntimeError(f"{d}: loader must use dedicated generator")
    iters = {d: infinite_loader(loaders[d]) for d in domains}

    optimizer = torch.optim.Adam(
        [
            {"params": list(model.parameters()) + list(moe.parameters()), "lr": ENCODER_LR},
            {"params": list(alpha_beta.parameters()), "lr": ALPHABETA_LR},
        ]
    )
    scheduler = DirectHWarmupLinearScheduler(
        optimizer,
        warmup_steps=WARMUP_STEPS,
        linear_steps=DECAY_STEPS,
        warmup_start=0.1,
        warmup_end=1.0,
        linear_end=0.1,
        steps_per_epoch=TOTAL_STEPS,
        n_epochs=1,
    )
    if abs(float(ENCODER_LR) * 0.1 - 2e-4) > 1e-12:
        raise RuntimeError("expected final LR floor 2e-4")

    loss_norms = {d: LossNormState() for d in domains}
    calib = fresh_calib_accumulators(domains)
    bn_init = clone_bn_bundle(collect_bn_bundle(model))
    bn_bundles = {d: clone_bn_bundle(bn_init) for d in domains}
    rng_states = init_matching_rng_states(SEED, active_domains=domains)
    schedule = round_robin_schedule(
        domains, total_steps=total_steps, steps_per_domain=steps_per_domain
    )

    seed_hash_log = {d: [] for d in domains}
    seed_first32_log = {d: [] for d in domains}
    view_hash_log = {d: {"view1": [], "view2": []} for d in domains}
    step_counts = {d: 0 for d in domains}

    jsonl_path = logs_dir / "steps.jsonl"
    csv_path = logs_dir / "steps.csv"
    if jsonl_path.exists():
        jsonl_path.unlink()

    rows: List[Dict[str, Any]] = []
    step_times: List[float] = []
    optimizer_step_count = 0
    scheduler_step_count = 0
    checkpoints_meta: Dict[str, Any] = {}
    reload_results: Dict[str, Any] = {}
    alpha_unfrozen_at: Optional[int] = None
    torch.cuda.reset_peak_memory_stats()

    write_json(
        out_dir / "rng_separation_contract.json",
        {
            "ok": True,
            "note": (
                "LinkNeighborLoader uses dedicated torch.Generator per domain. "
                "Augmentation/model use global RNG restored from per-domain snapshots. "
                "View hashes are logged for this arm only; no claim of historical view equality."
            ),
            "loader_generators_independent": True,
            "claim_historical_view_hash_equality": False,
        },
    )

    try:
        with open(jsonl_path, "w", encoding="utf-8") as jsonl:
            for si in range(total_steps):
                t_step0 = time.perf_counter()
                domain = schedule[si]
                restore_rng(rng_states[domain])
                apply_bn_(model, bn_bundles[domain])
                if not bn_bundles_equal(collect_bn_bundle(model), bn_bundles[domain]):
                    raise RuntimeError(f"BN bundle not applied for {domain}")

                batch = next(iters[domain])
                lr_used = float(scheduler.current_lrs()[0])
                opt_before = optimizer_step_count
                sched_before = scheduler.completed_optimizer_steps
                was_frozen = bool(alpha_beta._frozen)
                ab_before = float(alpha_beta.alpha_logit.detach().cpu())

                stats = gbt_tf_adaptive_mixed_step(
                    model=model,
                    moe=moe,
                    alpha_beta=alpha_beta,
                    loss_norm=loss_norms[domain],
                    tf_ctx=tf_ctx[domain],
                    optimizer=optimizer,
                    batch=batch,
                    loader_data=data_by[domain],
                    args=ns_by[domain],
                    device=device,
                    seed_ids_sha_fn=seed_ids_sha,
                    lr=lr_used,
                    domain=domain,
                    split_name="train",
                    do_optimizer_step=True,
                )
                optimizer_step_count += int(stats["optimizer_steps_this_call"])
                if optimizer_step_count != opt_before + 1:
                    raise RuntimeError("expected exactly one optimizer.step")

                for k in (
                    "L_total",
                    "L_gbt_raw",
                    "L_gbt_norm",
                    "L_tf_raw_0",
                    "L_tf_raw_1",
                    "L_tf_raw_2",
                    "encoder_grad_norm",
                    "moe_grad_norm",
                    "view1_repr_grad_norm",
                    "view2_repr_grad_norm",
                    "total_loss_reconstruction_error",
                ):
                    if not np.isfinite(float(stats[k])):
                        raise RuntimeError(f"non-finite {k} at step {si+1}")
                for module in (model, moe, alpha_beta):
                    for p in module.parameters():
                        if not torch.isfinite(p.data).all():
                            raise RuntimeError(f"non-finite parameter at step {si+1}")

                scheduler.step()
                scheduler_step_count += 1
                if scheduler.completed_optimizer_steps != sched_before + 1:
                    raise RuntimeError("expected exactly one scheduler.step")

                completed = si + 1
                ab_after = float(alpha_beta.alpha_logit.detach().cpu())
                if was_frozen and completed <= ALPHA_FREEZE_UNTIL and ab_after != ab_before:
                    raise RuntimeError(f"alpha/beta changed while frozen at step={completed}")

                observe_calibration(
                    domain=domain,
                    stats=stats,
                    calib=calib,
                    loss_norms=loss_norms,
                    calib_obs_per_domain=CALIB_OBS_PER_DOMAIN,
                )
                if maybe_unfreeze_alpha_beta(
                    alpha_beta=alpha_beta,
                    loss_norms=loss_norms,
                    domains=domains,
                    completed_step=completed,
                    freeze_until=ALPHA_FREEZE_UNTIL,
                ):
                    alpha_unfrozen_at = completed
                    if int(completed) != int(ALPHA_FREEZE_UNTIL):
                        raise RuntimeError(
                            f"alpha unfreeze at {completed}, expected {ALPHA_FREEZE_UNTIL}"
                        )

                bn_bundles[domain] = clone_bn_bundle(collect_bn_bundle(model))
                rng_states[domain] = snapshot_rng()
                step_counts[domain] += 1
                if len(seed_hash_log[domain]) < steps_per_domain:
                    seed_hash_log[domain].append(stats["seed_ids_sha256"])
                    seed_first32_log[domain].append(stats["seed_edge_ids_first32"])
                    view_hash_log[domain]["view1"].append(stats["view1_aug_sha256"])
                    view_hash_log[domain]["view2"].append(stats["view2_aug_sha256"])

                if stats.get("infonce_enabled") or stats.get("projection_enabled"):
                    raise RuntimeError("forbidden InfoNCE/projection path active")
                if not stats.get("tfmoe_enabled") or not stats.get("alpha_beta_enabled"):
                    raise RuntimeError("TF/adaptive path missing")

                elapsed = time.perf_counter() - t_step0
                row = {
                    "step": si,
                    "global_optimizer_step": completed,
                    "domain": domain,
                    "domain_exposure_index": step_counts[domain],
                    "domain_exposure_count": step_counts[domain],
                    "encoder_lr": lr_used,
                    "encoder_lr_after_scheduler_step": float(scheduler.current_lrs()[0]),
                    "lr_schedule_phase": scheduler.phase_at(si),
                    "elapsed_step_sec": elapsed,
                    "host_rss_gib": peak_rss_gib(),
                    "bn_domain_selected": domain,
                    "optimizer_steps_total": optimizer_step_count,
                    "scheduler_steps_total": scheduler_step_count,
                    "alpha_beta_frozen": bool(alpha_beta._frozen),
                    "alpha_unfrozen_at": alpha_unfrozen_at,
                    "calibration_complete_domain": bool(loss_norms[domain].calibrated),
                    "all_domains_calibrated": all(loss_norms[d].calibrated for d in domains),
                    **filter_stats_for_jsonl(stats),
                    **cuda_snap(),
                }
                for k, v in list(row.items()):
                    if isinstance(v, (np.floating, np.integer)):
                        row[k] = float(v) if isinstance(v, np.floating) else int(v)
                jsonl.write(json.dumps(row, default=str) + "\n")
                if completed <= 30 or completed % 10 == 0 or completed in ckpt_steps:
                    jsonl.flush()
                rows.append(row)
                step_times.append(elapsed)

                if should_console_log(completed):
                    logging.info(
                        "GBT+TF full step %s/%s domain=%s L=%.4f gbt=%.4f enc_g=%.3f moe_g=%.3f "
                        "alpha=%.3f lr=%.3e frozen=%s",
                        completed,
                        total_steps,
                        domain,
                        float(stats["L_total"]),
                        float(stats["L_gbt_raw"]),
                        float(stats["encoder_grad_norm"]),
                        float(stats["moe_grad_norm"]),
                        float(stats["alpha"]),
                        lr_used,
                        bool(alpha_beta._frozen),
                    )

                if completed in ckpt_steps or completed % ROLLING_EVERY == 0:
                    payload = build_checkpoint_payload(
                        model=model,
                        moe=moe,
                        alpha_beta=alpha_beta,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        bn_bundles=bn_bundles,
                        loss_norms=loss_norms,
                        global_step=completed,
                        recipe=recipe,
                        extra={
                            "seed_hash_log": {d: list(seed_hash_log[d]) for d in domains},
                            "step_counts": dict(step_counts),
                            "init_sha256": init_sha,
                            "mode": "gbt_tf_adaptive_full3000",
                            "alpha_unfrozen_at": alpha_unfrozen_at,
                            "calib": calib,
                            "resume_from_smoke": False,
                            "resume_from_gbt_recovery": False,
                            "test_evaluated": False,
                            "rng_states": {
                                d: {
                                    "python": rng_states[d]["python"],
                                    "numpy": rng_states[d]["numpy"],
                                    "torch": rng_states[d]["torch"],
                                    **(
                                        {"cuda": rng_states[d]["cuda"]}
                                        if "cuda" in rng_states[d]
                                        else {}
                                    ),
                                }
                                for d in rng_states
                            },
                            "loader_generator_states": snapshot_loader_generators(loaders),
                            "claim_historical_view_hash_equality": False,
                        },
                    )
                    if completed in ckpt_steps:
                        p = ckpt_dir / f"checkpoint_step_{completed:04d}.pt"
                        save_checkpoint(p, payload)
                        rel = cpu_reload_validate(
                            p,
                            expect_step=completed,
                            model_ref=model,
                            moe_ref=moe,
                            ab_ref=alpha_beta,
                        )
                        reload_results[f"step_{completed}"] = rel
                        if not rel["ok"]:
                            raise RuntimeError(f"checkpoint reload failed at step {completed}: {rel}")
                        checkpoints_meta[f"step_{completed}"] = {
                            "path": str(p),
                            "sha256": rel["sha256"],
                            "reload_ok": True,
                        }
                    p_last = ckpt_dir / "checkpoint_last.pt"
                    save_checkpoint(p_last, payload)
                    checkpoints_meta["last"] = {
                        "path": str(p_last),
                        "global_step": completed,
                        "sha256": file_sha256(p_last),
                    }

                del batch, stats
                if completed % 50 == 0:
                    gc.collect()
    except Exception as e:
        write_json(
            out_dir / "failure.json",
            {
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(),
                "optimizer_step_count": optimizer_step_count,
                "step_counts": step_counts,
            },
        )
        raise

    csv_keys = [
        "global_optimizer_step",
        "domain",
        "domain_exposure_count",
        "encoder_lr",
        "L_total",
        "L_gbt_raw",
        "L_gbt_norm",
        "weighted_gbt",
        "L_tf_raw_0",
        "L_tf_raw_1",
        "L_tf_raw_2",
        "L_tf_norm_0",
        "L_tf_norm_1",
        "L_tf_norm_2",
        "alpha",
        "beta_0",
        "beta_1",
        "beta_2",
        "w_contrast",
        "w_tf_0",
        "w_tf_1",
        "w_tf_2",
        "view1_n_floored_dims",
        "view2_n_floored_dims",
        "effective_rank",
        "repr_std_mean",
        "encoder_grad_norm",
        "moe_grad_norm",
        "alpha_beta_grad_norm",
        "view1_repr_grad_norm",
        "view2_repr_grad_norm",
        "param_update_norm",
        "batch_size_realized",
        "seed_ids_sha256",
        "view1_aug_sha256",
        "view2_aug_sha256",
        "alpha_beta_frozen",
        "cuda_allocated_gib",
        "host_rss_gib",
        "elapsed_step_sec",
    ]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in csv_keys})

    write_json(
        out_dir / "seed_hash_log.json",
        {
            "hashes": seed_hash_log,
            "first32": {d: seed_first32_log[d][:4] for d in domains},
            "view_hashes": {
                d: {
                    "view1_first4": view_hash_log[d]["view1"][:4],
                    "view2_first4": view_hash_log[d]["view2"][:4],
                    "n": len(view_hash_log[d]["view1"]),
                }
                for d in domains
            },
            "claim_historical_view_hash_equality": False,
        },
    )
    matching_vs_long = {
        d: compare_vs_long(
            seed_hash_log[d],
            ROOT,
            domain=d,
            limit=steps_per_domain,
            label_local=ARM,
        )
        for d in domains
    }
    write_json(out_dir / "seed_stream_vs_long.json", matching_vs_long)
    write_json(out_dir / "checkpoint_reload_integrity.json", reload_results)
    figures = write_figures(out_dir, rows)

    bn_changed = {d: not bn_bundles_equal(bn_bundles[d], bn_init) for d in domains}
    bn_l1 = {d: bn_bundle_l1(bn_bundles[d], bn_init) for d in domains}
    bn_distinct = all(
        not bn_bundles_equal(bn_bundles[d1], bn_bundles[d2])
        for i, d1 in enumerate(domains)
        for d2 in domains[i + 1 :]
    )
    enc_sha_final = state_dict_sha256(model.state_dict())
    moe_sha_final = state_dict_sha256(moe.state_dict())

    traj_keys = [
        "L_gbt_raw",
        "L_gbt_norm",
        "L_tf_raw_0",
        "L_tf_raw_1",
        "L_tf_raw_2",
        "L_tf_norm_0",
        "L_tf_norm_1",
        "L_tf_norm_2",
        "L_total",
        "alpha",
        "w_contrast",
        "weighted_gbt",
        "view1_n_floored_dims",
        "effective_rank",
        "encoder_grad_norm",
        "moe_grad_norm",
    ]
    trajectories = summarize_domain_series(rows, domains, traj_keys)

    mem.update(
        {
            "mean_sec_per_step": float(np.mean(step_times)) if step_times else None,
            "peak_rss_gib": peak_rss_gib(),
            "cuda_peak_allocated_gib": float(torch.cuda.max_memory_allocated() / GiB),
            "cuda_peak_reserved_gib": float(torch.cuda.max_memory_reserved() / GiB),
            "cuda_total_gib": float(torch.cuda.mem_get_info()[1] / GiB),
            "elapsed_train_sec": time.perf_counter() - t_wall0,
        }
    )
    write_json(out_dir / "memory_runtime.json", mem)

    lr_check = {
        "warmup_steps": WARMUP_STEPS,
        "linear_steps": DECAY_STEPS,
        "scheduler_completed": scheduler.completed_optimizer_steps,
        "step1_lr": float(rows[0]["encoder_lr"]) if rows else None,
        "step600_lr": next(
            (float(r["encoder_lr"]) for r in rows if int(r["global_optimizer_step"]) == 600),
            None,
        ),
        "step3000_lr": float(rows[-1]["encoder_lr"]) if rows else None,
        "expected_step1": 2e-4,
        "expected_peak_near_600": 2e-3,
        "expected_final_floor": 2e-4,
    }

    gates = {
        "exact_3000_optimizer_steps": optimizer_step_count == total_steps,
        "exact_1000_updates_per_domain": step_counts == {d: steps_per_domain for d in domains},
        "scheduler_steps_match": scheduler_step_count == total_steps,
        "finite_losses_all_steps": all(np.isfinite(float(r["L_total"])) for r in rows),
        "both_view_grads_nonzero_all_steps": all(
            float(r["view1_repr_grad_norm"]) > 0 and float(r["view2_repr_grad_norm"]) > 0
            for r in rows
        ),
        "moe_grads_nonzero_all_steps": all(float(r["moe_grad_norm"]) > 0 for r in rows),
        "encoder_sha_changed_from_init": enc_sha_final != enc_sha0,
        "moe_sha_changed_from_init": moe_sha_final != moe_sha0,
        "each_domain_bn_changed": all(bn_changed.values()),
        "domain_bn_bundles_distinct": bool(bn_distinct),
        "c_always_198x198": bool(c_always_198x198_from_rows(rows)["ok"]),
        "milestone_checkpoints_reload_ok": all(
            reload_results.get(f"step_{s}", {}).get("ok") for s in sorted(ckpt_steps)
        ),
        "long_seed_hash_match_1000": all(matching_vs_long[d].get("ok") for d in domains),
        "alpha_unfrozen_at_15": alpha_unfrozen_at == ALPHA_FREEZE_UNTIL,
        "all_domains_calibrated": all(loss_norms[d].calibrated for d in domains),
        "no_test_metrics": True,
        "no_extraction": True,
        "no_probe": True,
        "lr_schedule_long_matched": (
            scheduler.warmup_steps == WARMUP_STEPS and scheduler.linear_steps == DECAY_STEPS
        ),
        "init_sha_prefix_ok": init_sha.startswith(PHASE3_INIT_SHA_PREFIX),
        "did_not_resume_smoke": True,
        "did_not_resume_gbt_recovery": True,
        "did_not_resume_phase4b_infonce": True,
        "fresh_phase3_shared_init": True,
        "objective_is_gbt_tf_adaptive_stdfloor": True,
        "projection_amp_off": (not recipe["contrast_projection_head"]) and (not recipe["amp"]),
        "cuda_within_allocation": mem["cuda_peak_reserved_gib"] < mem["cuda_total_gib"],
        "host_rss_under_128g": mem["peak_rss_gib"] < 128.0,
        "view_hashes_logged_without_historical_equality_claim": all(
            r.get("view1_aug_sha256") and r.get("view2_aug_sha256") for r in rows
        ),
        "api_contract_ok": bool(api.get("ok")),
    }
    verdict = "PASS" if all(bool(v) for v in gates.values()) else "FAIL"

    result = {
        "title": "GBT+TF adaptive stdfloor MIXED_3DOMAIN full 3000-step training integrity",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "classification": verdict,
        "ok": verdict == "PASS",
        "arm": ARM,
        "objective_id": OBJECTIVE_ID,
        "parent_gbt_objective_id": PARENT_GBT_OBJECTIVE_ID,
        "mode": "--run-train",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_flags": {
            "partition": os.environ.get("SLURM_JOB_PARTITION") or "mit_normal_gpu",
            "account": os.environ.get("SLURM_JOB_ACCOUNT") or "mit_amf_advanced_gpu",
            "qos": os.environ.get("SLURM_JOB_QOS") or "mit_amf_advanced_gpu",
            "gres": "gpu:1",
            "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK") or "16",
            "mem": "128G",
            "time": "06:00:00",
            "loader_workers": 0,
        },
        "recipe": recipe,
        "step_counts": step_counts,
        "optimizer_step_count": optimizer_step_count,
        "scheduler_step_count": scheduler_step_count,
        "alpha_unfrozen_at": alpha_unfrozen_at,
        "training_integrity_gates": gates,
        "shared_init_provenance": init_prov,
        "encoder_sha_init": enc_sha0,
        "encoder_sha_final": enc_sha_final,
        "moe_sha_init": moe_sha0,
        "moe_sha_final": moe_sha_final,
        "bn_l1_vs_init": bn_l1,
        "bn_changed": bn_changed,
        "bn_distinct": bn_distinct,
        "checkpoints": checkpoints_meta,
        "checkpoint_reload": reload_results,
        "seed_stream_vs_long": matching_vs_long,
        "trajectories_summary": trajectories,
        "lr_check": lr_check,
        "memory_runtime": mem,
        "figures": figures,
        "claim_historical_view_hash_equality": False,
        "test_evaluated": False,
        "result_root": str(out_dir),
        "ckpt_root": str(ckpt_dir),
    }
    write_json(out_dir / "aggregate.json", result)
    write_json(TWIN_JSON, result)

    note_lines = [
        f"# {ARM} full 3000-step training",
        "",
        f"**Job:** `{os.environ.get('SLURM_JOB_ID')}`",
        f"**Verdict:** `{verdict}`",
        f"**Objective:** `{OBJECTIVE_ID}`",
        "",
        "## Locks",
        "- Fresh Phase-3 shared init (not smoke / GBT recovery / Phase-4B InfoNCE)",
        "- 3000 steps / 1000 per domain / LONG LR 600+2400",
        "- std_floor=1e-4 GBT + TF MoE view1 + adaptive α/β",
        "- Projection/AMP off; no test access; no extraction/probes",
        "",
        "## Gates",
        "```json",
        json.dumps(gates, indent=2),
        "```",
        "",
        "## Checkpoints",
        "```json",
        json.dumps(checkpoints_meta, indent=2),
        "```",
        "",
        "Training-integrity analysis only. No extraction/probes/test eval.",
        "",
    ]
    NOTE_PATH.write_text("\n".join(note_lines), encoding="utf-8")

    if verdict != "PASS":
        raise RuntimeError(f"full3000 integrity FAIL: {[k for k,v in gates.items() if not v]}")
    return result


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=f"{ARM} full3000 trainer")
    p.add_argument("--run-train", action="store_true")
    p.add_argument("--max-optimizer-steps", type=int, default=TOTAL_STEPS)
    p.add_argument("--split", type=str, default="train")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    refuse_test_split_access(args.split)
    if not args.run_train:
        print(json.dumps({"arm": ARM, "status": "pass --run-train"}, indent=2))
        return 0
    if int(args.max_optimizer_steps) != TOTAL_STEPS:
        raise RuntimeError(f"--run-train requires exactly {TOTAL_STEPS} steps")
    result = run_full3000()
    print(json.dumps({"ok": result["ok"], "classification": result["classification"], "job_id": result.get("slurm_job_id")}, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
