#!/usr/bin/env python3
"""Full 3000-step MIXED_3DOMAIN_GRAPH_BARLOW_TWINS_ONLY training.

Fresh from Phase-3 shared init. Does not resume smoke. No extraction/probes/test.
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
from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from direct_r198.lr_scheduler import DirectHWarmupLinearScheduler  # noqa: E402
from graph_barlow_twins_r198 import (  # noqa: E402
    ARM,
    CHECKPOINT_STEPS,
    DECAY_STEPS,
    FULL3000_CKPT_ROOT,
    FULL3000_JSON,
    FULL3000_NOTE,
    FULL3000_RESULT_ROOT,
    OBJECTIVE_ID,
    PHASE3_INIT_SHA_PREFIX,
    PHASE3_SHARED_INIT,
    ROLLING_EVERY,
    STEPS_PER_DOMAIN,
    TOTAL_STEPS,
    WARMUP_STEPS,
    resolved_recipe,
)
from graph_barlow_twins_r198.checkpoint import (  # noqa: E402
    build_checkpoint_payload,
    load_gbt_checkpoint,
    save_gbt_checkpoint,
)
from graph_barlow_twins_r198.integrity import refuse_test_split_access  # noqa: E402
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
from mixed_ssl_phase4b.matching import init_matching_rng_states  # noqa: E402
from phase4b_objective_ablation.matching import compare_vs_long  # noqa: E402
from train_util import AddEgoIds, FORWARD_EDGE_TYPE, add_arange_ids  # noqa: E402
from util import logger_setup, set_seed  # noqa: E402

# Reuse smoke helpers (loaders / model / init / mem) via file import.
import importlib.util as _ilu

_smoke_spec = _ilu.spec_from_file_location(
    "gbt_smoke30_helpers", ROOT / "scripts" / "run_gbt_smoke30.py"
)
_smoke = _ilu.module_from_spec(_smoke_spec)
assert _smoke_spec is not None and _smoke_spec.loader is not None
_smoke_spec.loader.exec_module(_smoke)
GiB = _smoke.GiB
build_model = _smoke.build_model
build_train_loader = _smoke.build_train_loader
cuda_snap = _smoke.cuda_snap
file_sha256 = _smoke.file_sha256
infinite_loader = _smoke.infinite_loader
load_phase3_encoder_only = _smoke.load_phase3_encoder_only
make_ns = _smoke.make_ns
peak_rss_gib = _smoke.peak_rss_gib
seed_ids_sha = _smoke.seed_ids_sha
summarize_domain_losses = _smoke.summarize_domain_losses
write_json = _smoke.write_json


NOTE_PATH = ROOT / FULL3000_NOTE
TWIN_JSON = ROOT / FULL3000_JSON


def refuse_nonempty_run(out_dir: Path, ckpt_dir: Path) -> None:
    blockers: List[str] = []
    for p in (
        out_dir / "logs" / "steps.jsonl",
        out_dir / "aggregate.json",
    ):
        if p.is_file() and p.stat().st_size > 0:
            blockers.append(str(p))
    for p in sorted(ckpt_dir.glob("checkpoint_*.pt")) + sorted(ckpt_dir.glob("checkpoint_*.tar")):
        blockers.append(str(p))
    # Forbid pointing at smoke / ablation / LONG trees.
    forbidden_sub = (
        "smoke30",
        "phase4b",
        "phase4a",
        "phase3",
        "objective_ablation",
        "mixed_long_3000",
    )
    for d in (out_dir, ckpt_dir):
        s = str(d)
        if any(x in s for x in forbidden_sub):
            blockers.append(f"forbidden_path:{s}")
    if blockers:
        raise RuntimeError(
            "Refuse startup: intended output path has nonempty/incompatible artifacts:\n  "
            + "\n  ".join(blockers[:20])
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


def cpu_reload_validate(ckpt_path: Path, *, expect_step: int, model_ref: nn.Module) -> Dict[str, Any]:
    model_cpu = copy.deepcopy(model_ref).cpu()
    with torch.no_grad():
        for p in model_cpu.parameters():
            p.normal_()
    loaded = load_gbt_checkpoint(ckpt_path, model_cpu)
    ok = (
        loaded.get("objective_id") == OBJECTIVE_ID
        and loaded.get("arm") == ARM
        and int(loaded.get("global_step", -1)) == int(expect_step)
        and not (loaded.get("recipe") or {}).get("infonce_enabled")
        and not (loaded.get("recipe") or {}).get("tfmoe_enabled")
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
        "has_loader_generators": bool((loaded.get("extra") or {}).get("loader_generator_states")),
        "forbidden": loaded.get("forbidden"),
    }


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
    steps = [int(r["global_optimizer_step"]) for r in rows]
    domains = sorted({r["domain"] for r in rows})

    def series(key: str, domain: Optional[str] = None):
        xs, ys = [], []
        for r in rows:
            if domain is not None and r["domain"] != domain:
                continue
            xs.append(int(r["global_optimizer_step"]))
            ys.append(float(r.get(key, float("nan"))))
        return xs, ys

    # Loss components
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for ax, key, title in zip(
        axes,
        ("L_gbt_total", "L_invariance", "L_redundancy"),
        ("L_total", "L_invariance", "L_redundancy"),
    ):
        for d in domains:
            xs, ys = series(key, d)
            ax.plot(xs, ys, label=d, linewidth=1.0)
        ax.set_ylabel(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("global optimizer step")
    fig.tight_layout()
    p = fig_dir / "01_loss_components_by_domain.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    written.append(str(p))

    fig, ax = plt.subplots(figsize=(10, 4))
    for d in domains:
        xs, ys = series("mean_diag_C", d)
        ax.plot(xs, ys, label=f"{d} mean_diag_C", linewidth=1.0)
    ax2 = ax.twinx()
    for d in domains:
        xs, ys = series("off_diagonal_rms", d)
        ax2.plot(xs, ys, linestyle="--", label=f"{d} off_rms", linewidth=1.0)
    ax.set_xlabel("step")
    ax.set_ylabel("mean diag C")
    ax2.set_ylabel("off-diagonal RMS")
    ax.legend(loc="upper left", fontsize=7)
    ax2.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = fig_dir / "02_correlation_diag_offrms.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    written.append(str(p))

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for d in domains:
        xs, ys = series("r198_effective_rank", d)
        axes[0].plot(xs, ys, label=d, linewidth=1.0)
        xs, ys = series("view1_std_median", d)
        axes[1].plot(xs, ys, label=d, linewidth=1.0)
    axes[0].set_ylabel("effective rank")
    axes[1].set_ylabel("view1 std median")
    axes[1].set_xlabel("step")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = fig_dir / "03_rank_and_variance.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    written.append(str(p))

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    xs, ys = series("encoder_grad_norm")
    axes[0].plot(xs, ys, linewidth=0.8)
    axes[0].axvline(600, color="k", linestyle=":", alpha=0.5, label="peak-LR region")
    axes[0].set_ylabel("encoder grad")
    xs, ys = series("param_update_norm")
    axes[1].plot(xs, ys, linewidth=0.8, color="C1")
    axes[1].set_ylabel("param update")
    xs, ys = series("encoder_lr")
    axes[2].plot(xs, ys, linewidth=1.0, color="C2")
    axes[2].set_ylabel("encoder LR")
    axes[2].set_xlabel("step")
    axes[0].legend(fontsize=8)
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = fig_dir / "04_grad_update_lr.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    written.append(str(p))
    return written


def run_full3000() -> Dict[str, Any]:
    refuse_test_split_access("train")
    t_wall0 = time.perf_counter()
    recipe = resolved_recipe()
    if int(recipe["max_optimizer_steps"]) != TOTAL_STEPS:
        raise RuntimeError("full recipe must be 3000 steps")
    if int(recipe["warmup_steps"]) != WARMUP_STEPS or int(recipe["linear_decay_steps"]) != DECAY_STEPS:
        raise RuntimeError("LR schedule must be LONG 600/2400")
    if recipe.get("is_smoke"):
        raise RuntimeError("full run must not use smoke recipe")

    domains = list(CANONICAL_DOMAINS)
    total_steps = TOTAL_STEPS
    steps_per_domain = STEPS_PER_DOMAIN
    ckpt_steps: Set[int] = set(CHECKPOINT_STEPS)

    out_dir = ROOT / FULL3000_RESULT_ROOT
    logs_dir = out_dir / "logs"
    ckpt_dir = ROOT / FULL3000_CKPT_ROOT
    for p in (out_dir, logs_dir, ckpt_dir):
        p.mkdir(parents=True, exist_ok=True)
    refuse_nonempty_run(out_dir, ckpt_dir)

    logger_setup()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda:0")
    set_seed(SEED)

    pre = preflight_phase4a(root=ROOT, specs=list(default_smoke_domains()))
    write_json(out_dir / "preflight.json", pre)
    if not pre.get("ok"):
        raise RuntimeError(f"phase4a preflight failed: {pre}")

    write_json(out_dir / "recipe.json", recipe)

    with open(ROOT / "data_config.json") as f:
        data_config = json.load(f)

    from data_loading import get_data  # local import after path setup

    unique = f"gbt_full3000_seed{SEED}"
    ns_by: Dict[str, Any] = {}
    data_by: Dict[str, Any] = {}
    mem: Dict[str, Any] = {
        "loader_num_workers": 0,
        "rss_gib_after_each_domain_load": {},
        "graph_build_sec": {},
        "test_graph_loaded": False,
        "test_metrics_computed": False,
        "no_frozen_embedding_extraction": True,
        "no_downstream_probe": True,
        "no_test_split_access": True,
    }
    transform = AddEgoIds()
    specs = [s for s in default_smoke_domains() if s.dataset_id in set(domains)]

    for spec in specs:
        d = spec.dataset_id
        logging.info("Loading %s (skip_test_eval; train graph only) ...", d)
        t0 = time.perf_counter()
        ns = make_ns(d, unique=unique, max_steps=total_steps)
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

    set_seed(SEED)
    sample_dom = domains[0]
    sample_loader = build_train_loader(data_by[sample_dom], transform, domain=sample_dom)
    sample = next(iter(sample_loader))
    del sample_loader
    model, emb_dim = build_model(ns_by[sample_dom], data_by[sample_dom], sample, device)
    del sample
    init_prov = load_phase3_encoder_only(model, ROOT / PHASE3_SHARED_INIT)
    if not str(init_prov["init_sha256"]).startswith(PHASE3_INIT_SHA_PREFIX):
        raise RuntimeError("shared-init SHA prefix mismatch")
    write_json(out_dir / "shared_init_provenance.json", init_prov)
    init_sha = init_prov["init_sha256"]
    enc_sha0 = init_prov["encoder_state_sha256"]

    # Loader generators are independent of global torch RNG (aug/model).
    loaders = {d: build_train_loader(data_by[d], transform, domain=d) for d in domains}
    for d, loader in loaders.items():
        if getattr(loader, "generator", None) is None:
            raise RuntimeError(f"{d}: loader must use dedicated generator")
    iters = {d: infinite_loader(loaders[d]) for d in domains}

    optimizer = torch.optim.Adam([{"params": list(model.parameters()), "lr": ENCODER_LR}])
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
    # Final floor: peak 2e-3 * 0.1 = 2e-4
    if abs(float(ENCODER_LR) * 0.1 - 2e-4) > 1e-12:
        raise RuntimeError("expected final LR floor 2e-4")

    bn_init = clone_bn_bundle(collect_bn_bundle(model))
    bn_bundles = {d: clone_bn_bundle(bn_init) for d in domains}
    rng_states = init_matching_rng_states(SEED, active_domains=domains)
    schedule = round_robin_schedule(
        domains, total_steps=total_steps, steps_per_domain=steps_per_domain
    )

    seed_hash_log = {d: [] for d in domains}
    seed_first32_log = {d: [] for d in domains}
    step_counts = {d: 0 for d in domains}
    enc_grad_by_domain = {d: [] for d in domains}
    param_upd_by_domain = {d: [] for d in domains}
    model_init_flat = torch.cat(
        [p.detach().float().reshape(-1).cpu() for p in model.parameters()]
    )

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
    torch.cuda.reset_peak_memory_stats()

    write_json(
        out_dir / "rng_separation_contract.json",
        {
            "ok": True,
            "note": (
                "LinkNeighborLoader uses a dedicated torch.Generator per domain "
                "(HI=1,SAML=2,LI=3 offsets). Augmentation/model use the global RNG "
                "restored from per-domain snapshots. Symmetric dual-view computation "
                "therefore cannot alter later seed-batch selection."
            ),
            "loader_generators_independent": True,
        },
    )

    fail_error = None
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
                )
                optimizer_step_count += int(stats["optimizer_steps_this_call"])
                if optimizer_step_count != opt_before + 1:
                    raise RuntimeError("expected exactly one optimizer.step")

                # Fail-fast on non-finite (step already asserts integrity; reinforce).
                for k in ("L_gbt_total", "L_invariance", "L_redundancy"):
                    if not np.isfinite(float(stats[k])):
                        raise RuntimeError(f"non-finite {k} at step {si+1}")
                if not np.isfinite(float(stats["encoder_grad_norm"])):
                    raise RuntimeError(f"non-finite encoder grad at step {si+1}")
                for p in model.parameters():
                    if not torch.isfinite(p.data).all():
                        raise RuntimeError(f"non-finite parameter at step {si+1}")

                scheduler.step()
                scheduler_step_count += 1
                if scheduler.completed_optimizer_steps != sched_before + 1:
                    raise RuntimeError("expected exactly one scheduler.step")

                bn_bundles[domain] = clone_bn_bundle(collect_bn_bundle(model))
                rng_states[domain] = snapshot_rng()
                step_counts[domain] += 1
                if len(seed_hash_log[domain]) < steps_per_domain:
                    seed_hash_log[domain].append(stats["seed_ids_sha256"])
                    seed_first32_log[domain].append(stats["seed_edge_ids_first32"])
                enc_grad_by_domain[domain].append(float(stats["encoder_grad_norm"]))
                param_upd_by_domain[domain].append(float(stats["param_update_norm"]))

                if any(
                    bool(stats.get(k))
                    for k in (
                        "infonce_enabled",
                        "tfmoe_enabled",
                        "projection_enabled",
                        "alpha_beta_enabled",
                        "negatives_enabled",
                    )
                ):
                    raise RuntimeError("forbidden objective path active")

                completed = si + 1
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
                    **stats,
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
                        "GBT full step %s/%s domain=%s L=%.4f enc_g=%.3f lr=%.3e Δθ=%.3e",
                        completed,
                        total_steps,
                        domain,
                        float(stats["L_gbt_total"]),
                        float(stats["encoder_grad_norm"]),
                        lr_used,
                        float(stats["param_update_norm"]),
                    )

                if completed in ckpt_steps or completed % ROLLING_EVERY == 0:
                    payload = build_checkpoint_payload(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        bn_bundles=bn_bundles,
                        global_step=completed,
                        recipe=recipe,
                        extra={
                            "seed_hash_log": {d: list(seed_hash_log[d]) for d in domains},
                            "step_counts": dict(step_counts),
                            "init_sha256": init_sha,
                            "mode": "full3000",
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
                            "gbt_lambda": recipe["gbt_lambda"],
                            "resume_from_smoke": False,
                        },
                    )
                    if completed in ckpt_steps:
                        p = ckpt_dir / f"checkpoint_step_{completed:04d}.pt"
                        # atomic via save then replace already in save_gbt_checkpoint
                        save_gbt_checkpoint(p, payload)
                        rel = cpu_reload_validate(p, expect_step=completed, model_ref=model)
                        reload_results[f"step_{completed}"] = rel
                        if not rel["ok"]:
                            raise RuntimeError(f"checkpoint reload failed at step {completed}: {rel}")
                        checkpoints_meta[f"step_{completed}"] = {
                            "path": str(p),
                            "sha256": rel["sha256"],
                            "reload_ok": True,
                        }
                    p_last = ckpt_dir / "checkpoint_last.pt"
                    save_gbt_checkpoint(p_last, payload)
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
        logging.exception("GBT full3000 failed")
        write_json(
            out_dir / "failure.json",
            {
                "error": fail_error,
                "traceback": traceback.format_exc(),
                "optimizer_step_count": optimizer_step_count,
                "step_counts": step_counts,
            },
        )
        raise

    # CSV (may be large — write compact selected columns + full jsonl retained)
    csv_keys = [
        "global_optimizer_step",
        "domain",
        "domain_exposure_count",
        "encoder_lr",
        "L_gbt_total",
        "L_invariance",
        "L_redundancy",
        "reconstruction_error",
        "mean_diag_C",
        "min_diag_C",
        "max_diag_C",
        "off_diagonal_rms",
        "view1_std_min",
        "view1_std_median",
        "view1_std_max",
        "view2_std_min",
        "view2_std_median",
        "view2_std_max",
        "r198_mean_l2_norm",
        "r198_effective_rank",
        "encoder_grad_norm",
        "view1_repr_grad_norm",
        "view2_repr_grad_norm",
        "param_update_norm",
        "batch_size_realized",
        "seed_ids_sha256",
        "view1_aug_sha256",
        "view2_aug_sha256",
        "cuda_allocated_gib",
        "cuda_reserved_gib",
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
        {"hashes": seed_hash_log, "first32": {d: seed_first32_log[d][:4] for d in domains}},
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
    bn_distinct = all(
        not bn_bundles_equal(bn_bundles[d1], bn_bundles[d2])
        for i, d1 in enumerate(domains)
        for d2 in domains[i + 1 :]
    )
    model_final_flat = torch.cat(
        [p.detach().float().reshape(-1).cpu() for p in model.parameters()]
    )
    encoder_changed = not torch.allclose(model_init_flat, model_final_flat)
    enc_sha_final = state_dict_sha256(model.state_dict())

    loss_by_dom = summarize_domain_losses(rows, domains)
    # Checkpoint-aligned loss snapshots
    loss_at_ckpt = {}
    for s in sorted(ckpt_steps):
        loss_at_ckpt[s] = {
            d: next(
                (
                    {
                        "L_gbt_total": float(r["L_gbt_total"]),
                        "L_invariance": float(r["L_invariance"]),
                        "L_redundancy": float(r["L_redundancy"]),
                    }
                    for r in rows
                    if int(r["global_optimizer_step"]) == s and r["domain"] == d
                ),
                None,
            )
            for d in domains
        }

    ranks = [float(r.get("r198_effective_rank", float("nan"))) for r in rows]
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

    # LR verification samples
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
        "finite_losses_all_steps": all(
            np.isfinite(float(r["L_gbt_total"])) for r in rows
        ),
        "both_view_grads_nonzero_all_steps": all(
            float(r["view1_repr_grad_norm"]) > 0 and float(r["view2_repr_grad_norm"]) > 0
            for r in rows
        ),
        "encoder_params_changed": bool(encoder_changed),
        "encoder_sha_changed_from_init": enc_sha_final != enc_sha0,
        "each_domain_bn_changed": all(bn_changed.values()),
        "domain_bn_bundles_distinct": bool(bn_distinct),
        "c_always_198x198": all(list(r.get("C_shape") or []) == [198, 198] for r in rows),
        "milestone_checkpoints_reload_ok": all(
            reload_results.get(f"step_{s}", {}).get("ok") for s in sorted(ckpt_steps)
        ),
        "long_seed_hash_match_1000": all(matching_vs_long[d].get("ok") for d in domains),
        "no_test_metrics": True,
        "no_extraction": True,
        "no_probe": True,
        "lr_schedule_long_matched": (
            scheduler.warmup_steps == WARMUP_STEPS and scheduler.linear_steps == DECAY_STEPS
        ),
        "init_sha_prefix_ok": init_sha.startswith(PHASE3_INIT_SHA_PREFIX),
        "did_not_resume_smoke": True,
        "cuda_within_allocation": mem["cuda_peak_reserved_gib"] < mem["cuda_total_gib"],
        "host_rss_under_128g": mem["peak_rss_gib"] < 128.0,
        "view_hashes_logged": all(
            r.get("view1_aug_sha256") and r.get("view2_aug_sha256") for r in rows
        ),
    }
    verdict = "PASS" if all(bool(v) for v in gates.values()) else "FAIL"

    result = {
        "title": "GBT MIXED_3DOMAIN full 3000-step training integrity",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "classification": verdict,
        "arm": ARM,
        "objective_id": OBJECTIVE_ID,
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
        "loss_by_domain": loss_by_dom,
        "loss_at_checkpoints": loss_at_ckpt,
        "lr_schedule_verification": lr_check,
        "variance_rank_summary": {
            "r198_effective_rank_first": ranks[0] if ranks else None,
            "r198_effective_rank_last": ranks[-1] if ranks else None,
            "r198_effective_rank_mean": float(np.nanmean(ranks)) if ranks else None,
            "r198_effective_rank_min": float(np.nanmin(ranks)) if ranks else None,
        },
        "gradient_update_evidence": {
            "max_encoder_grad_norm": max(float(r["encoder_grad_norm"]) for r in rows),
            "max_param_update_norm": max(float(r["param_update_norm"]) for r in rows),
            "encoder_changed": encoder_changed,
            "param_updates_by_domain_mean": {
                d: float(np.mean(param_upd_by_domain[d])) for d in domains
            },
        },
        "memory_runtime": mem,
        "checkpoints": checkpoints_meta,
        "checkpoint_reload": reload_results,
        "gates": gates,
        "matching_vs_long": {
            d: {"ok": matching_vs_long[d].get("ok"), "n_compared": matching_vs_long[d].get("n_compared")}
            for d in domains
        },
        "view_hash_note": (
            "View hashes logged every step. LONG lacks historical view hashes — "
            "no cross-arm view equality claimed."
        ),
        "figures": figures,
        "init_provenance": init_prov,
        "bn_changed": bn_changed,
        "bn_distinct": bn_distinct,
        "elapsed_sec": time.perf_counter() - t_wall0,
        "paths": {
            "result_root": str(out_dir),
            "steps_jsonl": str(jsonl_path),
            "steps_csv": str(csv_path),
            "ckpt_root": str(ckpt_dir),
        },
        "no_frozen_embedding_extraction": True,
        "no_downstream_probe": True,
        "no_test_split_access": True,
        "no_additional_objective_arm": True,
        "no_automatic_dependent_dag": True,
        "fail_error": fail_error,
    }
    write_json(out_dir / "aggregate.json", result)
    write_json(TWIN_JSON, result)
    write_json(
        out_dir / "submission_manifest.json",
        {
            "job_id": result["slurm_job_id"],
            "slurm_flags": result["slurm_flags"],
            "arm": ARM,
            "max_optimizer_steps": total_steps,
            "verdict": verdict,
            "ckpt_root": str(ckpt_dir),
            "generated_at_utc": result["generated_at_utc"],
        },
    )

    # Compact note
    lines = [
        "# Graph Barlow Twins full 3000-step training",
        "",
        f"**Verdict:** `{verdict}`",
        f"**Job ID:** `{result.get('slurm_job_id')}`",
        f"**Generated:** {result['generated_at_utc']}",
        "",
        "## Counts",
        f"- optimizer/scheduler: {optimizer_step_count}/{scheduler_step_count}",
        f"- per-domain: `{json.dumps(step_counts)}`",
        f"- LR: step1={lr_check['step1_lr']}, step600={lr_check['step600_lr']}, "
        f"step3000={lr_check['step3000_lr']}",
        "",
        "## Gates",
        "```",
        json.dumps(gates, indent=2),
        "```",
        "",
        "## Memory",
        f"- peak CUDA reserved: {mem.get('cuda_peak_reserved_gib'):.3f} GiB",
        f"- peak host RSS: {mem.get('peak_rss_gib'):.3f} GiB",
        f"- mean sec/step: {mem.get('mean_sec_per_step')}",
        "",
        "## Matching",
        f"- LONG seed-hash match (1000/domain): `{json.dumps(result['matching_vs_long'])}`",
        f"- {result['view_hash_note']}",
        "",
        "Training-integrity analysis only. No extraction/probes/test eval.",
        "",
    ]
    NOTE_PATH.write_text("\n".join(lines), encoding="utf-8")
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-train", action="store_true", required=True)
    p.add_argument("--max-optimizer-steps", type=int, default=TOTAL_STEPS)
    args = p.parse_args(argv)
    if int(args.max_optimizer_steps) != TOTAL_STEPS:
        raise RuntimeError(f"full train requires exactly {TOTAL_STEPS} steps")
    result = run_full3000()
    print(
        json.dumps(
            {
                "classification": result["classification"],
                "job": result.get("slurm_job_id"),
                "optimizer_steps": result.get("optimizer_step_count"),
                "gates_failed": [k for k, v in result["gates"].items() if not v],
            },
            indent=2,
        )
    )
    return 0 if result["classification"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
