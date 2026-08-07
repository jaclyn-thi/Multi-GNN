#!/usr/bin/env python3
"""30-optimizer-step smoke for MIXED_3DOMAIN_GRAPH_BARLOW_TWINS_ONLY.

Uses the locked Phase-4B LONG protocol prefix (round-robin, domain BN, seed
streams) with the full 3000-step LR schedule's first 30 steps (not rescaled).
No embeddings, probes, or test evaluation.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import logging
import os
import resource
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import to_hetero

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_loading import get_data  # noqa: E402
from direct_r198.lr_scheduler import DirectHWarmupLinearScheduler  # noqa: E402
from graph_barlow_twins_r198 import (  # noqa: E402
    ARM,
    OBJECTIVE_ID,
    PHASE3_INIT_SHA_PREFIX,
    PHASE3_SHARED_INIT,
    R198_DIM,
    SMOKE30_CKPT_ROOT,
    SMOKE30_MAX_STEPS,
    SMOKE30_RESULT_ROOT,
    SMOKE30_STEPS_PER_DOMAIN,
    WARMUP_STEPS,
    DECAY_STEPS,
    TOTAL_STEPS,
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
from mixed_ssl_phase4a import (  # noqa: E402
    BATCH_SIZE,
    CONTRACT_ID,
    ENCODER_LR,
    NUM_NEIGHS,
    SEED,
)
from mixed_ssl_phase4a.domain_registry import default_smoke_domains  # noqa: E402
from mixed_ssl_phase4a.preflight import preflight_phase4a  # noqa: E402
from mixed_ssl_phase4a.schedule import (  # noqa: E402
    loader_generator,
    restore_rng,
    round_robin_schedule,
    snapshot_rng,
)
from mixed_ssl_phase4b import CANONICAL_DOMAINS  # noqa: E402
from mixed_ssl_phase4b.matching import init_matching_rng_states  # noqa: E402
from phase4b_objective_ablation.matching import compare_vs_long  # noqa: E402
from train_util import (  # noqa: E402
    AddEgoIds,
    FORWARD_EDGE_TYPE,
    add_arange_ids,
    extract_param,
)
from training import get_model  # noqa: E402
from util import create_parser, logger_setup, set_seed  # noqa: E402

GiB = 1024.0**3
NOTE_PATH = ROOT / "notes/financial_multidataset_graph_barlow_twins_smoke30.md"
TWIN_JSON = ROOT / "results/diagnostics/financial_multidataset_graph_barlow_twins_smoke30.json"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}.{time.time_ns()}")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def seed_ids_sha(t: torch.Tensor) -> str:
    a = t.detach().cpu().contiguous().numpy().astype(np.int64)
    return hashlib.sha256(a.tobytes()).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def peak_rss_gib() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024.0 * 1024.0)


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


def make_ns(data: str, *, unique: str, max_steps: int) -> argparse.Namespace:
    argv = [
        "--data", data, "--model", "gin", "--objective", "contrastive",
        "--unique_name", unique, "--seed", str(SEED),
        "--batch_size", str(BATCH_SIZE),
        "--num_neighs", "100", "100",
        "--loader_num_workers", "0",
        "--reverse_mp", "--ego", "--ports", "--emlps", "--tds",
        "--correct_reverse_edge_features",
        "--feature_contract", CONTRACT_ID,
        "--train_fit_edge_znorm", "--skip_test_eval",
        "--direct_r198_infonce",
        "--contrastive_num_neg_samples", "0",
        "--contrastive_memory_bank_size", "0",
        "--contrastive_accum_steps", "1",
        "--contrastive_temperature", "0.5",
        "--max_optimizer_steps", str(max_steps),
        "--edge_attr_mask_rate", "0.1",
        "--edge_drop_target_rate", "0.1",
    ]
    ns = create_parser().parse_args(argv)
    # GBT isolation — no asymmetric detach / proj / TF / AMP.
    ns.contrastive_asymmetric = False
    ns.contrast_projection_head = False
    ns.direct_r198_tfmoe = False
    ns.preserve_seed_edges = False
    ns.skip_test_eval = True
    ns.loader_num_workers = 0
    if bool(getattr(ns, "amp", False)):
        raise RuntimeError("AMP must be false")
    return ns


def build_train_loader(tr_data: HeteroData, transform, *, domain: str) -> LinkNeighborLoader:
    offsets = {s.dataset_id: s.loader_seed_offset for s in default_smoke_domains()}
    g = loader_generator(
        SEED, domain, domain_order=CANONICAL_DOMAINS, loader_seed_offsets=offsets
    )
    edge_label_index = tr_data[FORWARD_EDGE_TYPE].edge_index
    edge_label = tr_data[FORWARD_EDGE_TYPE].y
    return LinkNeighborLoader(
        tr_data,
        num_neighbors=NUM_NEIGHS,
        edge_label_index=(
            (FORWARD_EDGE_TYPE[0], FORWARD_EDGE_TYPE[1], FORWARD_EDGE_TYPE[2]),
            edge_label_index,
        ),
        edge_label=edge_label,
        batch_size=BATCH_SIZE,
        shuffle=True,
        transform=transform,
        num_workers=0,
        generator=g,
    )


def infinite_loader(loader) -> Iterator[Any]:
    while True:
        for batch in loader:
            yield batch


def build_model(ns, metadata_data: HeteroData, sample_batch, device):
    config = SimpleNamespace(
        model="gin",
        n_hidden=extract_param("n_hidden", ns),
        n_gnn_layers=extract_param("n_gnn_layers", ns),
        n_heads=None,
        dropout=extract_param("dropout", ns),
        final_dropout=extract_param("final_dropout", ns),
    )
    ns.direct_r198_infonce = True
    model = get_model(sample_batch, config, ns)
    emb_dim = int(getattr(model, "embedding_dim", 198))
    if emb_dim != 198:
        raise RuntimeError(f"R198 expected embedding_dim=198, got {emb_dim}")
    model = to_hetero(model, metadata_data.metadata(), aggr="mean").to(device)
    return model, emb_dim


def load_phase3_encoder_only(model: nn.Module, init_path: Path) -> Dict[str, Any]:
    if not init_path.is_file():
        raise RuntimeError(f"Phase-3 shared init missing: {init_path}")
    file_sha = file_sha256(init_path)
    blob = torch.load(init_path, map_location="cpu", weights_only=False)
    init_sha = str(blob.get("init_sha256", ""))
    if not init_sha.startswith(PHASE3_INIT_SHA_PREFIX):
        raise RuntimeError(
            f"Phase-3 init SHA prefix mismatch: {init_sha[:16]} != {PHASE3_INIT_SHA_PREFIX}"
        )
    model_sd = blob["model_state_dict"]
    cur = model.state_dict()
    if set(model_sd.keys()) != set(cur.keys()):
        raise RuntimeError("model key mismatch vs Phase-3 init")
    for k, v in model_sd.items():
        if tuple(v.shape) != tuple(cur[k].shape):
            raise RuntimeError(f"model shape mismatch {k}")
    model.load_state_dict(model_sd, strict=True)
    return {
        "ok": True,
        "path": str(init_path),
        "file_sha256": file_sha,
        "init_sha256": init_sha,
        "encoder_state_sha256": state_dict_sha256(model.state_dict()),
    }


def summarize_domain_losses(rows: List[Dict[str, Any]], domains: Sequence[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for d in domains:
        dr = [r for r in rows if r["domain"] == d]
        if not dr:
            out[d] = {"n": 0}
            continue
        totals = [float(r["L_gbt_total"]) for r in dr]
        inv = [float(r["L_invariance"]) for r in dr]
        red = [float(r["L_redundancy"]) for r in dr]
        out[d] = {
            "n": len(dr),
            "L_gbt_total_first": totals[0],
            "L_gbt_total_last": totals[-1],
            "L_gbt_total_mean": float(np.mean(totals)),
            "L_invariance_first": inv[0],
            "L_invariance_last": inv[-1],
            "L_redundancy_first": red[0],
            "L_redundancy_last": red[-1],
            "any_nonfinite": any(not np.isfinite(x) for x in totals + inv + red),
            "max_abs_step_delta": float(np.max(np.abs(np.diff(totals)))) if len(totals) > 1 else 0.0,
        }
    return out


def run_smoke30(*, max_optimizer_steps: int = SMOKE30_MAX_STEPS) -> Dict[str, Any]:
    if max_optimizer_steps > SMOKE30_MAX_STEPS:
        raise RuntimeError(
            f"--run-smoke hard-refuses max_optimizer_steps>{SMOKE30_MAX_STEPS} "
            f"(got {max_optimizer_steps})"
        )
    if max_optimizer_steps != SMOKE30_MAX_STEPS:
        raise RuntimeError(
            f"smoke requires exactly {SMOKE30_MAX_STEPS} optimizer steps "
            f"(got {max_optimizer_steps})"
        )
    refuse_test_split_access("train")

    t_wall0 = time.perf_counter()
    recipe = resolved_recipe(smoke_steps=max_optimizer_steps)
    # Enforce full LONG LR schedule (not rescaled).
    if int(recipe["warmup_steps"]) != WARMUP_STEPS or int(recipe["linear_decay_steps"]) != DECAY_STEPS:
        raise RuntimeError("smoke recipe must keep LONG warmup/decay (not rescaled)")

    domains = list(CANONICAL_DOMAINS)
    steps_per_domain = SMOKE30_STEPS_PER_DOMAIN
    total_steps = max_optimizer_steps
    if total_steps != len(domains) * steps_per_domain:
        raise RuntimeError("30 steps must equal 3 domains × 10")

    out_dir = ROOT / SMOKE30_RESULT_ROOT
    logs_dir = out_dir / "logs"
    ckpt_dir = ROOT / SMOKE30_CKPT_ROOT
    for p in (out_dir, logs_dir, ckpt_dir):
        p.mkdir(parents=True, exist_ok=True)

    existing = sorted(ckpt_dir.glob("checkpoint_*.tar")) + sorted(ckpt_dir.glob("checkpoint_*.pt"))
    if existing:
        raise RuntimeError(f"refuse overwrite existing smoke ckpts: {[p.name for p in existing[:6]]}")

    logger_setup()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for GBT smoke")
    device = torch.device("cuda:0")
    set_seed(SEED)

    pre = preflight_phase4a(root=ROOT, specs=list(default_smoke_domains()))
    write_json(out_dir / "preflight.json", pre)
    if not pre.get("ok"):
        raise RuntimeError(f"phase4a preflight failed: {pre}")

    with open(ROOT / "data_config.json") as f:
        data_config = json.load(f)

    unique = f"gbt_smoke30_seed{SEED}"
    ns_by: Dict[str, argparse.Namespace] = {}
    data_by: Dict[str, HeteroData] = {}
    mem: Dict[str, Any] = {
        "loader_num_workers": 0,
        "rss_gib_after_each_domain_load": {},
        "graph_build_sec": {},
        "test_graph_loaded": False,
        "test_metrics_computed": False,
    }
    transform = AddEgoIds()
    specs = [s for s in default_smoke_domains() if s.dataset_id in set(domains)]

    for spec in specs:
        d = spec.dataset_id
        logging.info("Loading %s (skip_test_eval) ...", d)
        t0 = time.perf_counter()
        ns = make_ns(d, unique=unique, max_steps=total_steps)
        tr, va, te, tr_i, va_i, te_i = get_data(ns, data_config)
        # Drop val/test immediately — no test metrics.
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
    write_json(out_dir / "shared_init_provenance.json", init_prov)
    init_sha = init_prov["init_sha256"]
    enc_sha0 = init_prov["encoder_state_sha256"]

    loaders = {d: build_train_loader(data_by[d], transform, domain=d) for d in domains}
    iters = {d: infinite_loader(loaders[d]) for d in domains}

    optimizer = torch.optim.Adam([{"params": list(model.parameters()), "lr": ENCODER_LR}])
    # Full LONG LR schedule — smoke only takes the first 30 steps.
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
    if scheduler.warmup_steps != WARMUP_STEPS or scheduler.linear_steps != DECAY_STEPS:
        raise RuntimeError("scheduler must use LONG warmup/decay")

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
    torch.cuda.reset_peak_memory_stats()

    with open(jsonl_path, "w", encoding="utf-8") as jsonl:
        for si in range(total_steps):
            t_step0 = time.perf_counter()
            domain = schedule[si]
            restore_rng(rng_states[domain])
            apply_bn_(model, bn_bundles[domain])
            # Confirm selected BN matches domain bundle before step.
            if not bn_bundles_equal(collect_bn_bundle(model), bn_bundles[domain]):
                raise RuntimeError(f"BN bundle not applied for {domain}")

            batch = next(iters[domain])
            lr_used = float(scheduler.current_lrs()[0])
            opt_steps_before = optimizer_step_count
            sched_steps_before = scheduler.completed_optimizer_steps

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
            if optimizer_step_count != opt_steps_before + 1:
                raise RuntimeError("expected exactly one optimizer.step per global step")

            scheduler.step()
            scheduler_step_count += 1
            if scheduler.completed_optimizer_steps != sched_steps_before + 1:
                raise RuntimeError("expected exactly one scheduler.step per global step")

            bn_bundles[domain] = clone_bn_bundle(collect_bn_bundle(model))
            rng_states[domain] = snapshot_rng()
            step_counts[domain] += 1
            seed_hash_log[domain].append(stats["seed_ids_sha256"])
            seed_first32_log[domain].append(stats["seed_edge_ids_first32"])
            enc_grad_by_domain[domain].append(float(stats["encoder_grad_norm"]))
            param_upd_by_domain[domain].append(float(stats["param_update_norm"]))

            # Every-step runtime asserts already inside step; reinforce isolation.
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

            elapsed = time.perf_counter() - t_step0
            row = {
                "step": si,
                "global_optimizer_step": si + 1,
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
            # JSON-serializable cleanup
            for k, v in list(row.items()):
                if isinstance(v, (np.floating, np.integer)):
                    row[k] = float(v) if isinstance(v, np.floating) else int(v)
                elif isinstance(v, list) and v and isinstance(v[0], (np.integer, int)):
                    row[k] = [int(x) for x in v]
            jsonl.write(json.dumps(row, default=str) + "\n")
            jsonl.flush()
            rows.append(row)
            step_times.append(elapsed)

            logging.info(
                "GBT smoke step %s/%s domain=%s L=%.4f enc_g=%.3f lr=%.3e Δθ=%.3e",
                si + 1,
                total_steps,
                domain,
                float(stats["L_gbt_total"]),
                float(stats["encoder_grad_norm"]),
                lr_used,
                float(stats["param_update_norm"]),
            )
            del batch, stats
            if (si + 1) % 10 == 0:
                gc.collect()

    # CSV
    if rows:
        keys = sorted({k for r in rows for k in r.keys()})
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in keys})

    # Seed hash match vs LONG (first 10 / domain)
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

    # Single versioned smoke checkpoint
    ckpt_payload = build_checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        bn_bundles=bn_bundles,
        global_step=total_steps,
        recipe=recipe,
        extra={
            "seed_hash_log": seed_hash_log,
            "step_counts": step_counts,
            "init_sha256": init_sha,
            "mode": "smoke30",
        },
    )
    ckpt_path = ckpt_dir / f"checkpoint_step_{total_steps:04d}.pt"
    save_gbt_checkpoint(ckpt_path, ckpt_payload)
    ckpt_sha = file_sha256(ckpt_path)

    # Reload integrity on a scrambled clone (do not touch training RNG streams).
    import copy

    model_reload = copy.deepcopy(model).cpu()
    with torch.no_grad():
        for p in model_reload.parameters():
            p.normal_()
    loaded = load_gbt_checkpoint(ckpt_path, model_reload)
    reload_ok = (
        loaded.get("objective_id") == OBJECTIVE_ID
        and loaded.get("arm") == ARM
        and int(loaded.get("global_step", -1)) == total_steps
        and (loaded.get("recipe") or {}).get("objective_id") == OBJECTIVE_ID
        and not (loaded.get("recipe") or {}).get("infonce_enabled")
        and not (loaded.get("recipe") or {}).get("tfmoe_enabled")
        and state_dict_sha256(model_reload.state_dict())
        == state_dict_sha256({k: v.detach().cpu() for k, v in model.state_dict().items()})
    )
    reload_report = {
        "ok": bool(reload_ok),
        "path": str(ckpt_path),
        "sha256": ckpt_sha,
        "objective_id": loaded.get("objective_id"),
        "global_step": loaded.get("global_step"),
        "scheduler_completed": (loaded.get("scheduler_state") or {}).get(
            "completed_optimizer_steps"
        ),
        "has_bn_bundles": bool(loaded.get("bn_bundles")),
        "forbidden": loaded.get("forbidden"),
    }
    write_json(out_dir / "checkpoint_reload_integrity.json", reload_report)

    bn_changed = {d: not bn_bundles_equal(bn_bundles[d], bn_init) for d in domains}
    bn_distinct = True
    for i, d1 in enumerate(domains):
        for d2 in domains[i + 1 :]:
            if bn_bundles_equal(bn_bundles[d1], bn_bundles[d2]):
                bn_distinct = False
    model_final_flat = torch.cat(
        [p.detach().float().reshape(-1).cpu() for p in model.parameters()]
    )
    encoder_changed = not torch.allclose(model_init_flat, model_final_flat)
    enc_sha_final = state_dict_sha256(model.state_dict())

    loss_by_dom = summarize_domain_losses(rows, domains)
    ranks = [float(r.get("r198_effective_rank", float("nan"))) for r in rows]
    stds = [float(r.get("view1_std_median", float("nan"))) for r in rows]
    max_enc_g = max((float(r["encoder_grad_norm"]) for r in rows), default=0.0)
    max_param = max((float(r["param_update_norm"]) for r in rows), default=0.0)

    mem.update(
        {
            "mean_sec_per_step": float(np.mean(step_times)) if step_times else None,
            "peak_rss_gib": peak_rss_gib(),
            "cuda_peak_allocated_gib": float(torch.cuda.max_memory_allocated() / GiB),
            "cuda_peak_reserved_gib": float(torch.cuda.max_memory_reserved() / GiB),
            "cuda_total_gib": float(torch.cuda.mem_get_info()[1] / GiB),
        }
    )
    write_json(out_dir / "memory_runtime.json", mem)

    gates = {
        "exact_30_optimizer_steps": optimizer_step_count == total_steps,
        "exact_10_updates_per_domain": step_counts == {d: steps_per_domain for d in domains},
        "scheduler_steps_match": scheduler_step_count == total_steps,
        "finite_losses_all_steps": all(
            np.isfinite(float(r["L_gbt_total"]))
            and np.isfinite(float(r["L_invariance"]))
            and np.isfinite(float(r["L_redundancy"]))
            for r in rows
        ),
        "both_view_grads_nonzero_all_steps": all(
            float(r["view1_repr_grad_norm"]) > 0 and float(r["view2_repr_grad_norm"]) > 0
            for r in rows
        ),
        "encoder_grads_nonzero_all_domains": all(
            any(g > 0 for g in enc_grad_by_domain[d]) for d in domains
        ),
        "encoder_params_changed": bool(encoder_changed),
        "encoder_sha_changed_from_init": enc_sha_final != enc_sha0,
        "each_domain_bn_changed": all(bn_changed.values()),
        "domain_bn_bundles_distinct": bool(bn_distinct),
        "no_forbidden_objectives": True,
        "c_always_198x198": all(list(r.get("C_shape") or []) == [198, 198] for r in rows),
        "checkpoint_reload_ok": bool(reload_ok),
        "long_seed_hash_match": all(matching_vs_long[d].get("ok") for d in domains),
        "no_test_metrics": True,
        "lr_schedule_not_rescaled": (
            scheduler.warmup_steps == WARMUP_STEPS and scheduler.linear_steps == DECAY_STEPS
        ),
        "init_sha_prefix_ok": init_sha.startswith(PHASE3_INIT_SHA_PREFIX),
        "cuda_within_allocation": mem["cuda_peak_reserved_gib"] < mem["cuda_total_gib"],
        "host_rss_under_128g": mem["peak_rss_gib"] < 128.0,
        "view_hashes_logged": all(
            r.get("view1_aug_sha256") and r.get("view2_aug_sha256") for r in rows
        ),
    }
    # View hashes: report without claiming LONG equality
    view_hash_note = (
        "View augmentation hashes logged every step. Historical MIXED_3DOMAIN_LONG "
        "did not log view hashes — no cross-arm view equality claimed."
    )

    verdict = "PASS" if all(bool(v) for v in gates.values()) else "FAIL"
    proposed_full = (
        "sbatch --partition=mit_normal_gpu --account=mit_amf_advanced_gpu "
        "--qos=mit_amf_advanced_gpu --gres=gpu:1 --cpus-per-task=16 --mem=128G "
        "--time=06:00:00 --job-name=gbt_r198_full3000 "
        "slurm/run_mixed_3domain_graph_barlow_twins_only_full.sh   "
        "# SCRIPT NOT CREATED / NOT SUBMITTED — full training remains blocked"
    )

    result = {
        "title": "GBT MIXED_3DOMAIN 30-step smoke",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "classification": verdict,
        "arm": ARM,
        "objective_id": OBJECTIVE_ID,
        "mode": "--run-smoke",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_flags": {
            "partition": os.environ.get("SLURM_JOB_PARTITION") or "mit_normal_gpu",
            "account": os.environ.get("SLURM_JOB_ACCOUNT") or "mit_amf_advanced_gpu",
            "qos": os.environ.get("SLURM_JOB_QOS") or "mit_amf_advanced_gpu",
            "gres": "gpu:1",
            "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK") or "16",
            "mem": "128G",
            "time": "01:00:00",
            "loader_workers": 0,
        },
        "recipe": recipe,
        "domains": domains,
        "schedule_head": schedule[:12],
        "step_counts": step_counts,
        "optimizer_step_count": optimizer_step_count,
        "scheduler_step_count": scheduler_step_count,
        "loss_by_domain": loss_by_dom,
        "variance_rank_summary": {
            "r198_effective_rank_first": ranks[0] if ranks else None,
            "r198_effective_rank_last": ranks[-1] if ranks else None,
            "r198_effective_rank_mean": float(np.nanmean(ranks)) if ranks else None,
            "view1_std_median_first": stds[0] if stds else None,
            "view1_std_median_last": stds[-1] if stds else None,
            "view1_std_median_mean": float(np.nanmean(stds)) if stds else None,
        },
        "gradient_update_evidence": {
            "max_encoder_grad_norm": max_enc_g,
            "max_param_update_norm": max_param,
            "encoder_changed": encoder_changed,
            "param_updates_by_domain_mean": {
                d: float(np.mean(param_upd_by_domain[d])) for d in domains
            },
            "enc_grad_by_domain_mean": {
                d: float(np.mean(enc_grad_by_domain[d])) for d in domains
            },
        },
        "memory_runtime": mem,
        "checkpoint_reload": reload_report,
        "gates": gates,
        "matching_vs_long": matching_vs_long,
        "view_hash_note": view_hash_note,
        "init_provenance": init_prov,
        "bn_changed": bn_changed,
        "bn_distinct": bn_distinct,
        "elapsed_sec": time.perf_counter() - t_wall0,
        "paths": {
            "result_root": str(out_dir),
            "steps_jsonl": str(jsonl_path),
            "steps_csv": str(csv_path),
            "checkpoint": str(ckpt_path),
            "memory_runtime": str(out_dir / "memory_runtime.json"),
            "checkpoint_reload": str(out_dir / "checkpoint_reload_integrity.json"),
            "seed_stream_vs_long": str(out_dir / "seed_stream_vs_long.json"),
        },
        "proposed_full_run_command": proposed_full,
        "proposed_full_run_submitted": False,
        "no_embeddings_extracted": True,
        "first_step_lr": float(rows[0]["encoder_lr"]) if rows else None,
        "last_step_lr": float(rows[-1]["encoder_lr"]) if rows else None,
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
            "checkpoint": str(ckpt_path),
            "checkpoint_sha256": ckpt_sha,
            "verdict": verdict,
            "generated_at_utc": result["generated_at_utc"],
        },
    )
    return result


def write_note(result: Dict[str, Any]) -> None:
    lines = [
        "# Graph Barlow Twins 30-step smoke",
        "",
        f"**Verdict:** `{result['classification']}`",
        f"**Job ID:** `{result.get('slurm_job_id')}`",
        f"**Generated:** {result['generated_at_utc']}",
        "",
        "## Slurm flags",
        "",
        "```",
        json.dumps(result.get("slurm_flags"), indent=2),
        "```",
        "",
        "## Step / exposure counts",
        "",
        f"- optimizer steps: {result.get('optimizer_step_count')}",
        f"- scheduler steps: {result.get('scheduler_step_count')}",
        f"- per-domain: `{json.dumps(result.get('step_counts'))}`",
        f"- first/last LR (full 3000-step schedule prefix): "
        f"{result.get('first_step_lr')} → {result.get('last_step_lr')}",
        "",
        "## Loss by domain (first vs last)",
        "",
    ]
    for d, s in (result.get("loss_by_domain") or {}).items():
        lines.append(
            f"- **{d}**: L_total {s.get('L_gbt_total_first'):.6g} → {s.get('L_gbt_total_last'):.6g} "
            f"(inv {s.get('L_invariance_first'):.6g}→{s.get('L_invariance_last'):.6g}; "
            f"red {s.get('L_redundancy_first'):.6g}→{s.get('L_redundancy_last'):.6g})"
        )
    vr = result.get("variance_rank_summary") or {}
    gu = result.get("gradient_update_evidence") or {}
    mem = result.get("memory_runtime") or {}
    lines += [
        "",
        "## Variance / effective rank",
        "",
        f"- effective_rank first→last: {vr.get('r198_effective_rank_first')} → "
        f"{vr.get('r198_effective_rank_last')} (mean {vr.get('r198_effective_rank_mean')})",
        f"- view1 std median first→last: {vr.get('view1_std_median_first')} → "
        f"{vr.get('view1_std_median_last')}",
        "",
        "## Gradients / parameter updates",
        "",
        f"- max encoder grad: {gu.get('max_encoder_grad_norm')}",
        f"- max param update: {gu.get('max_param_update_norm')}",
        f"- encoder changed: {gu.get('encoder_changed')}",
        f"- mean Δθ by domain: `{json.dumps(gu.get('param_updates_by_domain_mean'))}`",
        "",
        "## Memory",
        "",
        f"- peak CUDA allocated: {mem.get('cuda_peak_allocated_gib'):.3f} GiB",
        f"- peak CUDA reserved: {mem.get('cuda_peak_reserved_gib'):.3f} GiB",
        f"- peak host RSS: {mem.get('peak_rss_gib'):.3f} GiB",
        f"- mean sec/step: {mem.get('mean_sec_per_step')}",
        "",
        "## Checkpoint reload",
        "",
        f"`{json.dumps(result.get('checkpoint_reload'), indent=2)}`",
        "",
        "## Gates",
        "",
        "```",
        json.dumps(result.get("gates"), indent=2),
        "```",
        "",
        f"**View-hash note:** {result.get('view_hash_note')}",
        "",
        "## Proposed full run (NOT submitted)",
        "",
        "```bash",
        result.get("proposed_full_run_command", ""),
        "```",
        "",
        "Stop after 30-step smoke.",
        "",
    ]
    NOTE_PATH.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="GBT 30-step smoke")
    p.add_argument("--run-smoke", action="store_true", required=True)
    p.add_argument(
        "--max-optimizer-steps",
        type=int,
        default=SMOKE30_MAX_STEPS,
        help=f"Must be <={SMOKE30_MAX_STEPS}; smoke requires exactly {SMOKE30_MAX_STEPS}",
    )
    args = p.parse_args(argv)
    try:
        result = run_smoke30(max_optimizer_steps=int(args.max_optimizer_steps))
        write_note(result)
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
    except Exception:
        err = {
            "classification": "FAIL",
            "error": traceback.format_exc(),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        }
        write_json(TWIN_JSON, err)
        NOTE_PATH.write_text(
            "# GBT smoke30 FAILED\n\n```\n" + err["error"] + "\n```\n", encoding="utf-8"
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
