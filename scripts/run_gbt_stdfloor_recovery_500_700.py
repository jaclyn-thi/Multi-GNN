#!/usr/bin/env python3
"""Isolated GBT std-floor recovery scout: resume @500 → complete @700.

Intervention B only (std floor 1e-4). Does not modify the source checkpoint,
change LR, clip grads meaningfully, rescale loss, extract embeddings, or
touch test data.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import logging
import math
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

from direct_r198.lr_scheduler import DirectHWarmupLinearScheduler  # noqa: E402
from graph_barlow_twins_r198 import (  # noqa: E402
    ARM,
    DECAY_STEPS,
    FULL3000_CKPT_ROOT,
    OBJECTIVE_ID,
    OBJECTIVE_ID_STDFLOOR_1E4,
    RECOVERY_CHECKPOINT_STEPS,
    RECOVERY_END_STEP,
    RECOVERY_SOURCE_CKPT,
    RECOVERY_SOURCE_SHA256,
    RECOVERY_START_STEP,
    RECOVERY_STDFLOOR_CKPT_ROOT,
    RECOVERY_STDFLOOR_ID,
    RECOVERY_STDFLOOR_JSON,
    RECOVERY_STDFLOOR_NOTE,
    RECOVERY_STDFLOOR_RESULT_ROOT,
    STEPS_PER_DOMAIN,
    TOTAL_STEPS,
    WARMUP_STEPS,
    resolved_recipe_stdfloor_1e4_recovery,
)
from graph_barlow_twins_r198.checkpoint import (  # noqa: E402
    assert_source_checkpoint_readonly,
    build_checkpoint_payload,
    load_gbt_checkpoint,
    save_gbt_checkpoint,
)
from graph_barlow_twins_r198.integrity import refuse_test_split_access  # noqa: E402
from graph_barlow_twins_r198.loss import (  # noqa: E402
    edge_aligned_graph_barlow_twins_r198_stdfloor_1e4,
)
from graph_barlow_twins_r198.step import gbt_only_mixed_step  # noqa: E402
from mixed_ssl_phase2.bn import (  # noqa: E402
    apply_bn_,
    bn_bundles_equal,
    clone_bn_bundle,
    collect_bn_bundle,
)
from mixed_ssl_phase3.hash_util import state_dict_sha256  # noqa: E402
from mixed_ssl_phase4a import ENCODER_LR, SEED  # noqa: E402
from mixed_ssl_phase4a.domain_registry import default_smoke_domains  # noqa: E402
from mixed_ssl_phase4a.preflight import preflight_phase4a  # noqa: E402
from mixed_ssl_phase4a.schedule import (  # noqa: E402
    restore_rng,
    round_robin_schedule,
    snapshot_rng,
)
from mixed_ssl_phase4b import CANONICAL_DOMAINS  # noqa: E402
from train_util import AddEgoIds, FORWARD_EDGE_TYPE, add_arange_ids  # noqa: E402
from util import logger_setup, set_seed  # noqa: E402

import importlib.util as _ilu

_smoke_spec = _ilu.spec_from_file_location(
    "gbt_smoke30_helpers", ROOT / "scripts" / "run_gbt_smoke30.py"
)
_smoke = _ilu.module_from_spec(_smoke_spec)
assert _smoke_spec is not None and _smoke_spec.loader is not None
_smoke_spec.loader.exec_module(_smoke)
build_model = _smoke.build_model
build_train_loader = _smoke.build_train_loader
cuda_snap = _smoke.cuda_snap
file_sha256 = _smoke.file_sha256
infinite_loader = _smoke.infinite_loader
make_ns = _smoke.make_ns
peak_rss_gib = _smoke.peak_rss_gib
seed_ids_sha = _smoke.seed_ids_sha
write_json = _smoke.write_json

GRAD_THRESHOLDS = (1e3, 1e6, 1e9, 1e12)
ORIG_STEPS_JSONL = (
    ROOT
    / "results/diagnostics/financial_multidataset_graph_barlow_twins_full3000_seed2/logs/steps.jsonl"
)


def write_json_atomic(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, obj)


def snapshot_loader_generators(loaders: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for d, loader in loaders.items():
        g = getattr(loader, "generator", None)
        if g is None:
            out[d] = None
        else:
            out[d] = g.get_state().detach().cpu().clone()
    return out


def restore_loader_generators(loaders: Dict[str, Any], states: Dict[str, Any]) -> None:
    for d, loader in loaders.items():
        st = states.get(d)
        g = getattr(loader, "generator", None)
        if st is None or g is None:
            raise RuntimeError(f"missing loader generator state for {d}")
        g.set_state(st.to(torch.uint8) if st.dtype != torch.uint8 else st)


def max_view_grad(row: Dict[str, Any]) -> float:
    return max(
        float(row.get("view1_repr_grad_norm") or 0.0),
        float(row.get("view2_repr_grad_norm") or 0.0),
    )


def cpu_reload_validate(
    ckpt_path: Path, *, expect_step: int, model_ref: nn.Module, objective_id: str
) -> Dict[str, Any]:
    model_cpu = copy.deepcopy(model_ref).cpu()
    with torch.no_grad():
        for p in model_cpu.parameters():
            p.normal_()
    loaded = load_gbt_checkpoint(
        ckpt_path, model_cpu, accepted_objective_ids=[objective_id]
    )
    ok = (
        loaded.get("objective_id") == objective_id
        and loaded.get("arm") == ARM
        and int(loaded.get("global_step", -1)) == int(expect_step)
        and not (loaded.get("recipe") or {}).get("infonce_enabled")
        and bool(loaded.get("bn_bundles"))
        and state_dict_sha256(model_cpu.state_dict())
        == state_dict_sha256({k: v.detach().cpu() for k, v in model_ref.state_dict().items()})
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
        "has_rng_extra": bool((loaded.get("extra") or {}).get("rng_states")),
        "has_loader_generators": bool(
            (loaded.get("extra") or {}).get("loader_generator_states")
        ),
    }


def load_original_failed_rows() -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    if not ORIG_STEPS_JSONL.is_file():
        return out
    with ORIG_STEPS_JSONL.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            step = int(row["global_optimizer_step"])
            if 501 <= step <= 587:
                out[step] = row
    return out


def compare_overlap(
    recovery_rows: List[Dict[str, Any]], orig_by_step: Dict[int, Dict[str, Any]]
) -> Dict[str, Any]:
    matched_seed = 0
    matched_aug = 0
    compared = 0
    seed_mismatches = []
    aug_mismatches = []
    for r in recovery_rows:
        step = int(r["global_optimizer_step"])
        if step not in orig_by_step:
            continue
        o = orig_by_step[step]
        compared += 1
        if r.get("seed_ids_sha256") == o.get("seed_ids_sha256"):
            matched_seed += 1
        else:
            if len(seed_mismatches) < 5:
                seed_mismatches.append(step)
        aug_ok = (
            r.get("view1_aug_sha256") == o.get("view1_aug_sha256")
            and r.get("view2_aug_sha256") == o.get("view2_aug_sha256")
        )
        if aug_ok:
            matched_aug += 1
        else:
            if len(aug_mismatches) < 5:
                aug_mismatches.append(step)
        if r.get("domain") != o.get("domain"):
            raise RuntimeError(f"domain mismatch at step {step}")
    return {
        "n_compared_501_587": compared,
        "seed_stream_matches": matched_seed,
        "seed_stream_match_frac": (matched_seed / compared) if compared else None,
        "aug_hash_matches": matched_aug,
        "aug_hash_match_frac": (matched_aug / compared) if compared else None,
        "seed_mismatch_examples": seed_mismatches,
        "aug_mismatch_examples": aug_mismatches,
        "batch_stream_equality_claimed": compared > 0 and matched_seed == compared,
        "augmentation_equality_claimed": compared > 0 and matched_aug == compared,
        "note": (
            "Batch-stream equality requires seed_ids_sha256 match. "
            "Augmentation equality claimed only where view hashes match. "
            "Model trajectory diverges after first modified update."
        ),
    }


def write_figures(out_dir: Path, rows: List[Dict[str, Any]], orig_by_step: Dict[int, Dict[str, Any]]) -> List[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        return [f"figures_skipped:{e}"]

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    written = []
    steps = [int(r["global_optimizer_step"]) for r in rows]
    lr = [float(r["encoder_lr"]) for r in rows]
    vgrad = [max_view_grad(r) for r in rows]
    egrad = [float(r["encoder_grad_norm"]) for r in rows]
    min_std = [
        min(float(r["view1_std_min"]), float(r["view2_std_min"])) for r in rows
    ]
    n_floor = [
        int(r.get("view1_n_floored_dims", 0)) + int(r.get("view2_n_floored_dims", 0))
        for r in rows
    ]

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(steps, lr, color="C2")
    axes[0].axvline(600, color="k", ls="--", alpha=0.4, label="peak-LR region")
    axes[0].set_ylabel("LR")
    axes[0].legend(fontsize=8)
    axes[1].semilogy(steps, [max(v, 1e-30) for v in vgrad], label="max view grad")
    axes[1].semilogy(steps, [max(v, 1e-30) for v in egrad], label="encoder grad")
    axes[1].axhline(1e12, color="r", ls=":", alpha=0.5)
    axes[1].set_ylabel("grad norms")
    axes[1].legend(fontsize=8)
    axes[2].plot(steps, [float(r["L_gbt_total"]) for r in rows], label="L_total")
    axes[2].plot(steps, [float(r["L_invariance"]) for r in rows], label="L_inv")
    axes[2].plot(steps, [float(r["L_redundancy"]) for r in rows], label="L_red")
    axes[2].set_ylabel("loss")
    axes[2].set_xlabel("global step")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = fig_dir / "01_lr_grads_loss.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    written.append(str(p))

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].semilogy(steps, [max(s, 1e-30) for s in min_std], label="min raw std")
    axes[0].axhline(1e-4, color="C1", ls="--", label="floor")
    axes[0].legend(fontsize=8)
    axes[0].set_ylabel("min raw std")
    axes[1].plot(steps, n_floor, color="C3")
    axes[1].set_ylabel("floored dims (v1+v2)")
    axes[1].set_xlabel("global step")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = fig_dir / "02_std_and_floor_activation.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    written.append(str(p))

    # Overlap comparison 501-587
    if orig_by_step:
        osteps = sorted(orig_by_step)
        ov = [max_view_grad(orig_by_step[s]) for s in osteps]
        rv = {
            int(r["global_optimizer_step"]): max_view_grad(r)
            for r in rows
            if 501 <= int(r["global_optimizer_step"]) <= 587
        }
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.semilogy(osteps, [max(v, 1e-30) for v in ov], label="original view-grad", alpha=0.8)
        rs = sorted(rv)
        ax.semilogy(rs, [max(rv[s], 1e-30) for s in rs], label="recovery view-grad", alpha=0.8)
        ax.axhline(1e12, color="r", ls=":", alpha=0.5)
        ax.set_xlabel("global step")
        ax.set_ylabel("max view grad")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p = fig_dir / "03_original_vs_recovery_view_grad_501_587.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        written.append(str(p))

    # Peak LR window 550-620
    win = [r for r in rows if 550 <= int(r["global_optimizer_step"]) <= 620]
    if win:
        ws = [int(r["global_optimizer_step"]) for r in win]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.semilogy(ws, [max(max_view_grad(r), 1e-30) for r in win], label="view")
        ax.semilogy(ws, [max(float(r["encoder_grad_norm"]), 1e-30) for r in win], label="encoder")
        ax.axvline(600, color="k", ls="--", alpha=0.4)
        ax.set_title("grads around peak LR (550–620)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p = fig_dir / "04_peak_lr_window_grads.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        written.append(str(p))

    return written


def classify_verdict(rows: List[Dict[str, Any]], *, completed: int, integrity_ok: bool) -> str:
    if not integrity_ok:
        return "FAIL_INTEGRITY"
    if completed < RECOVERY_END_STEP:
        # nonfinite stop would have raised; treat incomplete as integrity unless failure.json says otherwise
        return "FAIL_INTEGRITY"
    any_nonfinite = False
    spike_1e12 = 0
    view_zero = False
    enc_zero = False
    for r in rows:
        vg = max_view_grad(r)
        eg = float(r["encoder_grad_norm"])
        if not math.isfinite(vg) or not math.isfinite(eg):
            any_nonfinite = True
        if vg > 1e12:
            spike_1e12 += 1
        if vg <= 0:
            view_zero = True
        if eg <= 0:
            enc_zero = True
        for k in ("L_gbt_total", "L_invariance", "L_redundancy"):
            if not math.isfinite(float(r[k])):
                any_nonfinite = True
    if any_nonfinite:
        return "FAIL_NONFINITE"
    if view_zero or enc_zero:
        return "FAIL_INTEGRITY"
    if spike_1e12 > 0:
        return "PASS_FINITE_BUT_SPIKY"
    return "PASS_STABLE"


def run_recovery() -> Dict[str, Any]:
    refuse_test_split_access("train")
    t_wall0 = time.perf_counter()
    recipe = resolved_recipe_stdfloor_1e4_recovery()
    domains = list(CANONICAL_DOMAINS)
    start_step = RECOVERY_START_STEP
    end_step = RECOVERY_END_STEP
    new_steps = end_step - start_step
    ckpt_steps: Set[int] = set(RECOVERY_CHECKPOINT_STEPS)

    out_dir = ROOT / RECOVERY_STDFLOOR_RESULT_ROOT
    logs_dir = out_dir / "logs"
    ckpt_dir = ROOT / RECOVERY_STDFLOOR_CKPT_ROOT
    source_path = ROOT / RECOVERY_SOURCE_CKPT

    # Refuse writing into original full3000 ckpt tree.
    if Path(FULL3000_CKPT_ROOT) in ckpt_dir.parents or ckpt_dir == ROOT / FULL3000_CKPT_ROOT:
        raise RuntimeError("recovery ckpt_dir must not be full3000 tree")
    if "full3000_seed2" in str(ckpt_dir) and "stdfloor" not in str(ckpt_dir):
        raise RuntimeError("refusing full3000 ckpt path")

    for p in (out_dir, logs_dir, ckpt_dir):
        p.mkdir(parents=True, exist_ok=True)

    # Nonempty recovery artifacts → refuse overwrite mid-run.
    if (logs_dir / "steps.jsonl").is_file() and (logs_dir / "steps.jsonl").stat().st_size > 0:
        raise RuntimeError("recovery steps.jsonl already nonempty — refuse overwrite")

    logger_setup()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda:0")

    sha_before = assert_source_checkpoint_readonly(
        source_path,
        required_sha256=RECOVERY_SOURCE_SHA256,
        file_sha256_fn=file_sha256,
    )
    write_json_atomic(out_dir / "source_sha_before.json", sha_before)

    pre = preflight_phase4a(root=ROOT, specs=list(default_smoke_domains()))
    write_json_atomic(out_dir / "preflight.json", pre)
    if not pre.get("ok"):
        raise RuntimeError(f"phase4a preflight failed: {pre}")
    write_json_atomic(out_dir / "recipe.json", recipe)

    with open(ROOT / "data_config.json") as f:
        data_config = json.load(f)
    from data_loading import get_data

    unique = f"gbt_stdfloor_recovery_seed{SEED}"
    ns_by: Dict[str, Any] = {}
    data_by: Dict[str, Any] = {}
    transform = AddEgoIds()
    specs = [s for s in default_smoke_domains() if s.dataset_id in set(domains)]
    for spec in specs:
        d = spec.dataset_id
        logging.info("Loading %s (train only) ...", d)
        ns = make_ns(d, unique=unique, max_steps=TOTAL_STEPS)
        tr, va, te, tr_i, va_i, te_i = get_data(ns, data_config)
        del va, te, tr_i, va_i, te_i
        gc.collect()
        if int(tr[FORWARD_EDGE_TYPE].edge_attr.shape[1]) != 6:
            raise RuntimeError(f"{d} edge_dim != 6")
        add_arange_ids([tr])
        ns_by[d] = ns
        data_by[d] = tr

    set_seed(SEED)
    sample_dom = domains[0]
    sample_loader = build_train_loader(data_by[sample_dom], transform, domain=sample_dom)
    sample = next(iter(sample_loader))
    del sample_loader
    model, emb_dim = build_model(ns_by[sample_dom], data_by[sample_dom], sample, device)
    del sample

    # Load source checkpoint (official objective) into model.
    payload = load_gbt_checkpoint(
        source_path, model, accepted_objective_ids=[OBJECTIVE_ID]
    )
    if int(payload.get("global_step", -1)) != start_step:
        raise RuntimeError(f"source global_step={payload.get('global_step')} != {start_step}")
    if int(payload.get("global_optimizer_step", -1)) != start_step:
        raise RuntimeError("source global_optimizer_step mismatch")
    sched_blob = payload.get("scheduler_state") or {}
    if int(sched_blob.get("completed_optimizer_steps", -1)) != start_step:
        raise RuntimeError("source scheduler completed_optimizer_steps != 500")
    extra = payload.get("extra") or {}
    step_counts = dict(extra.get("step_counts") or {})
    expected_counts = {"Small-HI": 167, "SAML-D": 167, "Small-LI": 166}
    if step_counts != expected_counts:
        raise RuntimeError(f"exposure mismatch: {step_counts} != {expected_counts}")

    # Finite checks on loaded encoder / BN.
    for k, v in model.state_dict().items():
        if torch.is_tensor(v) and v.is_floating_point() and not torch.isfinite(v).all():
            raise RuntimeError(f"non-finite model tensor on load: {k}")
    bn_bundles = {
        d: {kk: vv.detach().cpu().clone() for kk, vv in bundle.items()}
        for d, bundle in (payload.get("bn_bundles") or {}).items()
    }
    if set(bn_bundles) != set(domains):
        raise RuntimeError(f"bn_bundles domains {set(bn_bundles)} != {domains}")
    for d, bundle in bn_bundles.items():
        for k, v in bundle.items():
            if torch.is_tensor(v) and v.is_floating_point() and not torch.isfinite(v).all():
                raise RuntimeError(f"non-finite BN {d}/{k}")

    loaders = {d: build_train_loader(data_by[d], transform, domain=d) for d in domains}
    restore_loader_generators(loaders, extra.get("loader_generator_states") or {})
    iters = {d: infinite_loader(loaders[d]) for d in domains}

    optimizer = torch.optim.Adam([{"params": list(model.parameters()), "lr": ENCODER_LR}])
    if "optimizer_state_dict" not in payload:
        raise RuntimeError("source missing optimizer_state_dict")
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    # Move optimizer state tensors to device if needed.
    for state in optimizer.state.values():
        for sk, sv in list(state.items()):
            if torch.is_tensor(sv):
                state[sk] = sv.to(device)

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
    cfg = sched_blob.get("configuration") or {}
    scheduler.load_state_dict(
        {
            "scheduler_type": "direct_h_warmup_linear",
            "warmup_steps": int(sched_blob.get("warmup_steps", WARMUP_STEPS)),
            "linear_steps": int(sched_blob.get("linear_steps", DECAY_STEPS)),
            "warmup_start": float(cfg.get("warmup_start_mult", 0.1)),
            "warmup_end": float(cfg.get("warmup_end_mult", 1.0)),
            "linear_end": float(cfg.get("linear_end_mult", 0.1)),
            "steps_per_epoch": int(cfg.get("steps_per_epoch", TOTAL_STEPS)),
            "n_epochs": int(cfg.get("n_epochs", 1)),
            "base_lrs": list(cfg.get("base_lrs") or [ENCODER_LR]),
            "completed_optimizer_steps": start_step,
        }
    )
    if scheduler.completed_optimizer_steps != start_step:
        raise RuntimeError("scheduler restore failed")

    rng_states = extra.get("rng_states")
    if not rng_states or set(rng_states) != set(domains):
        raise RuntimeError("source missing per-domain rng_states")

    # Full RR schedule of 3000; we only execute indices start_step .. end_step-1.
    schedule = round_robin_schedule(
        domains, total_steps=TOTAL_STEPS, steps_per_domain=STEPS_PER_DOMAIN
    )

    write_json_atomic(
        out_dir / "resume_integrity.json",
        {
            "source_sha256": sha_before["sha256"],
            "global_step": start_step,
            "scheduler_completed": scheduler.completed_optimizer_steps,
            "step_counts": step_counts,
            "objective_source": payload.get("objective_id"),
            "objective_recovery": OBJECTIVE_ID_STDFLOOR_1E4,
            "lr_at_resume": float(scheduler.current_lrs()[0]),
            "model_finite": True,
            "bn_finite": True,
            "optimizer_restored": True,
            "scheduler_restored": True,
            "loader_generators_restored": True,
            "rng_restored": True,
            "no_optimizer_reset": True,
            "no_scheduler_reset": True,
            "no_new_warmup": True,
        },
    )

    jsonl_path = logs_dir / "steps.jsonl"
    rows: List[Dict[str, Any]] = []
    optimizer_step_count = start_step
    scheduler_step_count = start_step
    checkpoints_meta: Dict[str, Any] = {}
    reload_results: Dict[str, Any] = {}
    grad_thr_counts = {f"view_gt_{t:.0e}": 0 for t in GRAD_THRESHOLDS}
    grad_thr_counts.update({f"enc_gt_{t:.0e}": 0 for t in GRAD_THRESHOLDS})
    torch.cuda.reset_peak_memory_stats()

    fail_error = None
    try:
        with open(jsonl_path, "w", encoding="utf-8") as jsonl:
            for si in range(start_step, end_step):
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

                stats = gbt_only_mixed_step(
                    model=model,
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
                    loss_fn=edge_aligned_graph_barlow_twins_r198_stdfloor_1e4,
                    objective_id_override=OBJECTIVE_ID_STDFLOOR_1E4,
                )
                optimizer_step_count += int(stats["optimizer_steps_this_call"])
                if optimizer_step_count != opt_before + 1:
                    raise RuntimeError("expected exactly one optimizer.step")

                for k in ("L_gbt_total", "L_invariance", "L_redundancy"):
                    if not np.isfinite(float(stats[k])):
                        raise RuntimeError(f"non-finite {k} at step {si+1}")
                if not np.isfinite(float(stats["encoder_grad_norm"])):
                    raise RuntimeError(f"non-finite encoder grad at step {si+1}")
                if not np.isfinite(float(stats["view1_repr_grad_norm"])) or not np.isfinite(
                    float(stats["view2_repr_grad_norm"])
                ):
                    raise RuntimeError(f"non-finite view grad at step {si+1}")
                for p in model.parameters():
                    if not torch.isfinite(p.data).all():
                        raise RuntimeError(f"non-finite parameter at step {si+1}")
                if tuple(stats["C_shape"]) != (198, 198):
                    raise RuntimeError(f"bad C shape {stats['C_shape']}")

                scheduler.step()
                scheduler_step_count += 1
                if scheduler.completed_optimizer_steps != sched_before + 1:
                    raise RuntimeError("expected exactly one scheduler.step")

                bn_bundles[domain] = clone_bn_bundle(collect_bn_bundle(model))
                rng_states[domain] = snapshot_rng()
                step_counts[domain] = int(step_counts.get(domain, 0)) + 1

                completed = si + 1
                vg = max(
                    float(stats["view1_repr_grad_norm"]),
                    float(stats["view2_repr_grad_norm"]),
                )
                eg = float(stats["encoder_grad_norm"])
                for thr in GRAD_THRESHOLDS:
                    if vg > thr:
                        grad_thr_counts[f"view_gt_{thr:.0e}"] += 1
                    if eg > thr:
                        grad_thr_counts[f"enc_gt_{thr:.0e}"] += 1

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
                    "seed": SEED,
                    "values_finite": True,
                    "grads_finite": True,
                    "params_finite": True,
                    "view_grad_gt_1e3": vg > 1e3,
                    "view_grad_gt_1e6": vg > 1e6,
                    "view_grad_gt_1e9": vg > 1e9,
                    "view_grad_gt_1e12": vg > 1e12,
                    "recovery_id": RECOVERY_STDFLOOR_ID,
                    **stats,
                    **cuda_snap(),
                }
                for k, v in list(row.items()):
                    if isinstance(v, (np.floating, np.integer)):
                        row[k] = float(v) if isinstance(v, np.floating) else int(v)
                jsonl.write(json.dumps(row, default=str) + "\n")
                if completed % 10 == 0 or completed in ckpt_steps:
                    jsonl.flush()
                rows.append(row)

                if completed % 10 == 0 or completed in ckpt_steps or completed <= start_step + 5:
                    logging.info(
                        "GBT recovery step %s/%s domain=%s L=%.4f enc_g=%.3f "
                        "view_g=%.3e floor=%s/%s lr=%.3e",
                        completed,
                        end_step,
                        domain,
                        float(stats["L_gbt_total"]),
                        eg,
                        vg,
                        int(stats.get("view1_n_floored_dims", 0)),
                        int(stats.get("view2_n_floored_dims", 0)),
                        lr_used,
                    )

                if completed in ckpt_steps or completed == end_step:
                    ckpt_extra = {
                        "step_counts": dict(step_counts),
                        "recovery_id": RECOVERY_STDFLOOR_ID,
                        "source_checkpoint": str(source_path),
                        "source_sha256": RECOVERY_SOURCE_SHA256,
                        "parent_objective_id": OBJECTIVE_ID,
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
                        "loss_definition": recipe["loss_definition"],
                        "gbt_std_floor": recipe["gbt_std_floor"],
                        "forbid_write_source_checkpoint": True,
                    }
                    payload_out = build_checkpoint_payload(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        bn_bundles=bn_bundles,
                        global_step=completed,
                        recipe=recipe,
                        objective_id=OBJECTIVE_ID_STDFLOOR_1E4,
                        extra=ckpt_extra,
                    )
                    if completed in ckpt_steps:
                        p = ckpt_dir / f"recovery_step_{completed:04d}.pt"
                        # Never write into source dir.
                        if source_path.parent.resolve() == p.parent.resolve():
                            raise RuntimeError("refusing to write recovery ckpt into source dir")
                        save_gbt_checkpoint(p, payload_out)
                        rel = cpu_reload_validate(
                            p,
                            expect_step=completed,
                            model_ref=model,
                            objective_id=OBJECTIVE_ID_STDFLOOR_1E4,
                        )
                        reload_results[f"step_{completed}"] = rel
                        if not rel["ok"]:
                            raise RuntimeError(f"recovery checkpoint reload failed: {rel}")
                        checkpoints_meta[f"step_{completed}"] = {
                            "path": str(p),
                            "sha256": rel["sha256"],
                            "reload_ok": True,
                        }
                    p_last = ckpt_dir / "recovery_last.pt"
                    if source_path.parent.resolve() == p_last.parent.resolve():
                        raise RuntimeError("refusing recovery_last into source dir")
                    save_gbt_checkpoint(p_last, payload_out)
                    checkpoints_meta["last"] = {
                        "path": str(p_last),
                        "global_step": completed,
                        "sha256": file_sha256(p_last),
                    }

                del batch, stats
                if completed % 50 == 0:
                    gc.collect()
    except Exception as e:
        fail_error = f"{type(e).__name__}: {e}"
        logging.exception("GBT stdfloor recovery failed")
        write_json_atomic(
            out_dir / "failure.json",
            {
                "error": fail_error,
                "traceback": traceback.format_exc(),
                "optimizer_step_count": optimizer_step_count,
                "step_counts": step_counts,
                "verdict": "FAIL_NONFINITE"
                if "non-finite" in str(e).lower()
                else "FAIL_INTEGRITY",
            },
        )
        raise

    sha_after = assert_source_checkpoint_readonly(
        source_path,
        required_sha256=RECOVERY_SOURCE_SHA256,
        file_sha256_fn=file_sha256,
    )
    write_json_atomic(out_dir / "source_sha_after.json", sha_after)
    if sha_before["sha256"] != sha_after["sha256"]:
        raise RuntimeError("source checkpoint SHA changed during recovery")

    # CSV
    csv_keys = [
        "global_optimizer_step",
        "domain",
        "encoder_lr",
        "L_gbt_total",
        "L_invariance",
        "L_redundancy",
        "view1_raw_std_min",
        "view1_raw_std_median",
        "view1_raw_std_max",
        "view2_raw_std_min",
        "view2_raw_std_median",
        "view2_raw_std_max",
        "view1_safe_std_min",
        "view1_safe_std_median",
        "view1_safe_std_max",
        "view2_safe_std_min",
        "view2_safe_std_median",
        "view2_safe_std_max",
        "view1_n_floored_dims",
        "view1_frac_floored_dims",
        "view2_n_floored_dims",
        "view2_frac_floored_dims",
        "mean_diag_C",
        "min_diag_C",
        "max_diag_C",
        "off_diagonal_rms",
        "r198_effective_rank",
        "r198_mean_l2_norm",
        "view1_repr_grad_norm",
        "view2_repr_grad_norm",
        "encoder_grad_norm",
        "param_update_norm",
        "batch_size_realized",
        "seed_ids_sha256",
        "view1_aug_sha256",
        "view2_aug_sha256",
        "view_grad_gt_1e12",
        "host_rss_gib",
    ]
    csv_path = logs_dir / "steps.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=csv_keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in csv_keys})

    orig_by_step = load_original_failed_rows()
    overlap = compare_overlap(rows, orig_by_step)
    write_json_atomic(out_dir / "stream_comparison_501_587.json", overlap)

    # Floor activation by domain
    floor_by_domain: Dict[str, Any] = {}
    for d in domains:
        drows = [r for r in rows if r["domain"] == d]
        n = len(drows)
        v1 = sum(int(r.get("view1_n_floored_dims", 0)) for r in drows)
        v2 = sum(int(r.get("view2_n_floored_dims", 0)) for r in drows)
        steps_with = sum(
            1
            for r in drows
            if int(r.get("view1_n_floored_dims", 0)) + int(r.get("view2_n_floored_dims", 0)) > 0
        )
        floor_by_domain[d] = {
            "n_steps": n,
            "sum_view1_floored_dims": v1,
            "sum_view2_floored_dims": v2,
            "steps_with_any_floor": steps_with,
            "frac_steps_with_any_floor": (steps_with / n) if n else None,
            "mean_view1_floored_dims": (v1 / n) if n else None,
            "mean_view2_floored_dims": (v2 / n) if n else None,
        }

    max_view = max(max_view_grad(r) for r in rows) if rows else float("nan")
    max_enc = max(float(r["encoder_grad_norm"]) for r in rows) if rows else float("nan")
    n_spike = sum(1 for r in rows if max_view_grad(r) > 1e12)
    peak_win = [r for r in rows if 550 <= int(r["global_optimizer_step"]) <= 620]
    peak_stats = {
        "n": len(peak_win),
        "max_view_grad": max((max_view_grad(r) for r in peak_win), default=None),
        "max_encoder_grad": max((float(r["encoder_grad_norm"]) for r in peak_win), default=None),
        "median_view_grad": float(
            np.median([max_view_grad(r) for r in peak_win])
        )
        if peak_win
        else None,
        "lr_min": min((float(r["encoder_lr"]) for r in peak_win), default=None),
        "lr_max": max((float(r["encoder_lr"]) for r in peak_win), default=None),
        "n_view_gt_1e12": sum(1 for r in peak_win if max_view_grad(r) > 1e12),
    }

    orig_max_view = max((max_view_grad(o) for o in orig_by_step.values()), default=None)
    orig_max_enc = max(
        (float(o["encoder_grad_norm"]) for o in orig_by_step.values()), default=None
    )
    orig_spikes = sum(1 for o in orig_by_step.values() if max_view_grad(o) > 1e12)

    integrity_ok = (
        optimizer_step_count == end_step
        and scheduler_step_count == end_step
        and all(reload_results.get(f"step_{s}", {}).get("ok") for s in ckpt_steps)
        and sha_after["sha256"] == RECOVERY_SOURCE_SHA256
        and overlap.get("batch_stream_equality_claimed") is True
    )
    verdict = classify_verdict(rows, completed=end_step, integrity_ok=integrity_ok)
    # Stream failure → integrity fail even if finite.
    if not overlap.get("batch_stream_equality_claimed"):
        if verdict.startswith("PASS"):
            verdict = "FAIL_INTEGRITY"

    figures = write_figures(out_dir, rows, orig_by_step)

    # Expected new exposures ~67/66/67
    new_exp = {
        d: int(step_counts[d]) - int(expected_counts[d]) for d in domains
    }
    expected_new = {"Small-HI": 67, "SAML-D": 66, "Small-LI": 67}

    aggregate = {
        "recovery_id": RECOVERY_STDFLOOR_ID,
        "verdict": verdict,
        "objective_id": OBJECTIVE_ID_STDFLOOR_1E4,
        "intervention": "B_std_floor_1e4_only",
        "start_step": start_step,
        "end_step": end_step,
        "n_new_steps": len(rows),
        "expected_new_steps": new_steps,
        "step_counts_final": step_counts,
        "new_exposures": new_exp,
        "expected_new_exposures": expected_new,
        "new_exposures_match": new_exp == expected_new,
        "scheduler_completed": scheduler.completed_optimizer_steps,
        "optimizer_steps": optimizer_step_count,
        "grad_threshold_counts": grad_thr_counts,
        "max_view_grad": max_view,
        "max_encoder_grad": max_enc,
        "n_view_grad_gt_1e12": n_spike,
        "peak_lr_window_550_620": peak_stats,
        "floor_activation_by_domain": floor_by_domain,
        "stream_comparison": overlap,
        "original_overlap_501_587": {
            "max_view_grad": orig_max_view,
            "max_encoder_grad": orig_max_enc,
            "n_view_grad_gt_1e12": orig_spikes,
        },
        "gradient_comparison": {
            "recovery_max_view": max_view,
            "original_max_view_501_587": orig_max_view,
            "recovery_max_encoder": max_enc,
            "original_max_encoder_501_587": orig_max_enc,
            "recovery_spikes_1e12": n_spike,
            "original_spikes_1e12_501_587": orig_spikes,
        },
        "source_sha_before": sha_before["sha256"],
        "source_sha_after": sha_after["sha256"],
        "source_sha_unchanged": sha_before["sha256"] == sha_after["sha256"],
        "checkpoints": checkpoints_meta,
        "checkpoint_reload": reload_results,
        "figures": figures,
        "wall_sec": time.perf_counter() - t_wall0,
        "no_training_beyond_700": True,
        "no_embedding_extraction": True,
        "no_probes": True,
        "no_test_access": True,
        "lr_unchanged": True,
        "no_grad_clip_intervention": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json_atomic(out_dir / "aggregate.json", aggregate)
    write_json_atomic(ROOT / RECOVERY_STDFLOOR_JSON, aggregate)

    # Recommendation
    if verdict == "PASS_STABLE":
        recommendation = (
            "Proceed with a fresh 3000-step std-floor run from shared initialization "
            "(do not continue this recovery checkpoint to 3000)."
        )
        follow_up = "fresh_3000_stdfloor_from_shared_init"
    elif verdict == "PASS_FINITE_BUT_SPIKY":
        recommendation = (
            "Std-floor alone left >1e12 view-grad spikes. Run a second controlled "
            "lower-LR recovery scout (intervention A), still isolated; do not auto-submit."
        )
        follow_up = "second_lower_lr_recovery_scout"
    else:
        recommendation = (
            "Recovery failed. Diagnose failure.json before any follow-up; "
            "do not auto-submit another run."
        )
        follow_up = "diagnose_failure_first"

    note_lines = [
        f"# GBT std-floor recovery scout (`{RECOVERY_STDFLOOR_ID}`)",
        "",
        f"**Verdict:** `{verdict}`",
        f"**Intervention:** B only — `std_safe = clamp_min(std_raw, 1e-4)`",
        f"**Objective:** `{OBJECTIVE_ID_STDFLOOR_1E4}`",
        f"**Steps:** {start_step} → {end_step} ({len(rows)} new optimizer steps)",
        "",
        "## Integrity",
        f"- Source SHA before/after unchanged: `{sha_after['sha256']}`",
        f"- Final exposures: `{json.dumps(step_counts)}` (new `{json.dumps(new_exp)}`)",
        f"- Batch-stream equality 501–587: `{overlap.get('batch_stream_equality_claimed')}`",
        f"- Aug hash equality 501–587: `{overlap.get('augmentation_equality_claimed')}` "
        f"(frac={overlap.get('aug_hash_match_frac')})",
        "",
        "## Gradients",
        f"- Recovery max view / encoder: `{max_view:.6g}` / `{max_enc:.6g}`",
        f"- Recovery >1e12 view spikes: `{n_spike}`",
        f"- Original 501–587 max view / spikes: `{orig_max_view}` / `{orig_spikes}`",
        f"- Peak-LR window 550–620: `{json.dumps(peak_stats)}`",
        "",
        "## Floor activation by domain",
        f"```json\n{json.dumps(floor_by_domain, indent=2)}\n```",
        "",
        "## Recommendation",
        recommendation,
        f"- Follow-up id (not submitted): `{follow_up}`",
        "",
        "## Confirmations",
        "- no LR change / no loss rescale / no real grad clip / no AMP",
        "- source checkpoint not modified",
        "- stopped at 700; no embedding extraction / probes / test access",
        "",
        f"Generated: {aggregate['generated_at_utc']}",
    ]
    (ROOT / RECOVERY_STDFLOOR_NOTE).write_text("\n".join(note_lines) + "\n", encoding="utf-8")
    write_json_atomic(
        out_dir / "recommendation.json",
        {"verdict": verdict, "recommendation": recommendation, "follow_up": follow_up},
    )

    logging.info("Recovery complete verdict=%s", verdict)
    return aggregate


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-recovery", action="store_true")
    p.add_argument("--dry-check-source", action="store_true")
    args = p.parse_args(argv)
    if args.dry_check_source:
        sha = assert_source_checkpoint_readonly(
            ROOT / RECOVERY_SOURCE_CKPT,
            required_sha256=RECOVERY_SOURCE_SHA256,
            file_sha256_fn=file_sha256,
        )
        print(json.dumps(sha, indent=2))
        return 0
    if not args.run_recovery:
        p.error("pass --run-recovery or --dry-check-source")
    run_recovery()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
