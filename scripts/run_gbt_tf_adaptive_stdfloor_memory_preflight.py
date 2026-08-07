#!/usr/bin/env python3
"""One-batch/domain no-update memory preflight for GBT+TF adaptive (dual-view + TF).

Measures CUDA/host peak for the combined path. No optimizer.step, no smoke
checkpoints into historical GBT-only trees.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

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
from direct_r198 import LearnedAlphaBeta, LossNormState, TFMoEBundle, load_tf_moe_context  # noqa: E402
from gbt_tf_adaptive_stdfloor_r198 import (  # noqa: E402
    ARM,
    INIT_ALPHA,
    OBJECTIVE_ID,
    TF_CACHE_BY_DOMAIN,
    resolved_recipe,
)
from gbt_tf_adaptive_stdfloor_r198.integrity import refuse_test_split_access  # noqa: E402
from gbt_tf_adaptive_stdfloor_r198.step import gbt_tf_adaptive_mixed_step  # noqa: E402
from mixed_ssl_phase2.bn import apply_bn_, clone_bn_bundle, collect_bn_bundle  # noqa: E402
from mixed_ssl_phase3.hash_util import combined_init_sha  # noqa: E402
from mixed_ssl_phase4a import (  # noqa: E402
    ALPHABETA_LR,
    BATCH_SIZE,
    CONTRACT_ID,
    ENCODER_LR,
    LOADER_NUM_WORKERS,
    NUM_NEIGHS,
    SEED,
)
from mixed_ssl_phase4a.domain_registry import default_smoke_domains  # noqa: E402
from mixed_ssl_phase4a.schedule import loader_generator  # noqa: E402
from mixed_ssl_phase4b import CANONICAL_DOMAINS, PHASE3_INIT_SHA_PREFIX, PHASE3_SHARED_INIT  # noqa: E402
from mixed_ssl_phase4b.matching import init_matching_rng_states  # noqa: E402
from train_util import AddEgoIds, FORWARD_EDGE_TYPE, add_arange_ids, extract_param  # noqa: E402
from training import get_model  # noqa: E402
from util import create_parser, logger_setup, set_seed  # noqa: E402

OUT_DIR = ROOT / (
    "results/diagnostics/financial_multidataset_gbt_tf_adaptive_stdfloor_1e4_memory_preflight"
)
HOST_MEM_LIMIT_GIB = 128.0
HOST_SAFE_RSS_GIB = 120.0
CUDA_SAFE_FRAC = 0.85


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}.{time.time_ns()}")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def peak_rss_gib() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024.0 * 1024.0)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def seed_ids_sha(t: torch.Tensor) -> str:
    a = t.detach().cpu().contiguous().numpy().astype(np.int64)
    return hashlib.sha256(a.tobytes()).hexdigest()


def loader_offsets() -> Dict[str, int]:
    return {s.dataset_id: s.loader_seed_offset for s in default_smoke_domains()}


def build_train_loader(tr_data: HeteroData, transform, *, domain: str) -> LinkNeighborLoader:
    offsets = loader_offsets()
    g = loader_generator(
        SEED, domain, domain_order=CANONICAL_DOMAINS, loader_seed_offsets=offsets
    )
    return LinkNeighborLoader(
        tr_data,
        num_neighbors=NUM_NEIGHS,
        edge_label_index=(
            (FORWARD_EDGE_TYPE[0], FORWARD_EDGE_TYPE[1], FORWARD_EDGE_TYPE[2]),
            tr_data[FORWARD_EDGE_TYPE].edge_index,
        ),
        edge_label=tr_data[FORWARD_EDGE_TYPE].y,
        batch_size=BATCH_SIZE,
        shuffle=True,
        transform=transform,
        num_workers=LOADER_NUM_WORKERS,
        generator=g,
    )


def make_ns(data: str, *, unique: str, max_steps: int = 30):
    """Build util.create_parser namespace; never call parse_args([]) (requires --data/--model)."""
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
    ns.contrastive_asymmetric = False
    ns.contrast_projection_head = False
    ns.direct_r198_tfmoe = False
    ns.preserve_seed_edges = False
    ns.skip_test_eval = True
    ns.loader_num_workers = 0
    return ns


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
    model = to_hetero(model, metadata_data.metadata(), aggr="mean").to(device)
    return model, emb_dim


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--memory-preflight-only", action="store_true")
    p.add_argument("--split", type=str, default="train")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    refuse_test_split_access(args.split)
    if not args.memory_preflight_only:
        print(json.dumps({"status": "pass --memory-preflight-only"}, indent=2))
        return 0

    logger_setup()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("memory preflight requires CUDA")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    recipe = resolved_recipe(smoke_steps=30)
    domains = list(CANONICAL_DOMAINS)
    transform = AddEgoIds()
    data_by: Dict[str, HeteroData] = {}
    ns_by: Dict[str, Any] = {}

    with open(ROOT / "data_config.json") as f:
        data_config = json.load(f)

    for d in domains:
        ns = make_ns(d, unique=f"{ARM}_mem_{d}", max_steps=30)
        set_seed(SEED)
        tr, va, te, tr_i, va_i, te_i = get_data(ns, data_config)
        del va, te, tr_i, va_i, te_i
        if int(tr[FORWARD_EDGE_TYPE].edge_attr.shape[1]) != 6:
            raise RuntimeError(f"{d} edge_dim != 6")
        data_by[d] = tr
        ns_by[d] = ns
        add_arange_ids([tr])

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

    blob = torch.load(ROOT / PHASE3_SHARED_INIT, map_location="cpu", weights_only=False)
    init_sha = str(blob["init_sha256"])
    if not init_sha.startswith(PHASE3_INIT_SHA_PREFIX):
        raise RuntimeError("Phase-3 init SHA prefix mismatch")
    model.load_state_dict(blob["model_state_dict"], strict=True)
    moe.load_state_dict(blob["moe_state_dict"], strict=True)
    alpha_beta.load_state_dict(blob["alpha_beta_state_dict"], strict=True)
    alpha_beta.set_frozen(True)
    if combined_init_sha(model, moe, alpha_beta) != init_sha:
        raise RuntimeError("init sha after load mismatch")

    tf_ctx = {d: load_tf_moe_context(ROOT / TF_CACHE_BY_DOMAIN[d], device) for d in domains}
    loss_norms = {d: LossNormState() for d in domains}
    optimizer = torch.optim.Adam(
        [
            {"params": list(model.parameters()) + list(moe.parameters()), "lr": ENCODER_LR},
            {"params": list(alpha_beta.parameters()), "lr": ALPHABETA_LR},
        ]
    )
    bn_init = clone_bn_bundle(collect_bn_bundle(model))
    bn_bundles = {d: clone_bn_bundle(bn_init) for d in domains}
    rng_states = init_matching_rng_states(SEED, active_domains=domains)

    rows = []
    errors = []
    torch.cuda.reset_peak_memory_stats()
    rss0 = peak_rss_gib()

    for d in domains:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        from mixed_ssl_phase4a.schedule import restore_rng, snapshot_rng

        restore_rng(rng_states[d])
        apply_bn_(model, bn_bundles[d])
        loader = build_train_loader(data_by[d], transform, domain=d)
        batch = next(iter(loader))
        t0 = time.perf_counter()
        stats = gbt_tf_adaptive_mixed_step(
            model=model,
            moe=moe,
            alpha_beta=alpha_beta,
            loss_norm=loss_norms[d],
            tf_ctx=tf_ctx[d],
            optimizer=optimizer,
            batch=batch,
            loader_data=data_by[d],
            args=ns_by[d],
            device=device,
            seed_ids_sha_fn=seed_ids_sha,
            domain=d,
            split_name="train",
            do_optimizer_step=False,  # no update
        )
        elapsed = time.perf_counter() - t0
        peak_alloc = torch.cuda.max_memory_allocated() / (1024**3)
        peak_res = torch.cuda.max_memory_reserved() / (1024**3)
        total_cuda = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        row = {
            "domain": d,
            "elapsed_sec": elapsed,
            "cuda_peak_alloc_gib": peak_alloc,
            "cuda_peak_reserved_gib": peak_res,
            "cuda_total_gib": total_cuda,
            "cuda_alloc_frac": peak_alloc / total_cuda,
            "host_rss_gib": peak_rss_gib(),
            "L_total": stats.get("L_total"),
            "L_gbt_raw": stats.get("L_gbt_raw"),
            "encoder_grad_norm": stats.get("encoder_grad_norm"),
            "moe_grad_norm": stats.get("moe_grad_norm"),
            "view1_repr_grad_norm": stats.get("view1_repr_grad_norm"),
            "view2_repr_grad_norm": stats.get("view2_repr_grad_norm"),
            "C_shape": stats.get("C_shape"),
            "scored_seeds": stats.get("scored_seeds"),
            "do_optimizer_step": False,
        }
        rows.append(row)
        if list(stats.get("C_shape") or []) != [198, 198]:
            errors.append(f"{d}: C_shape={stats.get('C_shape')}")
        if float(stats.get("view1_repr_grad_norm") or 0) <= 0:
            errors.append(f"{d}: view1 grad zero")
        if float(stats.get("view2_repr_grad_norm") or 0) <= 0:
            errors.append(f"{d}: view2 grad zero")
        if float(stats.get("moe_grad_norm") or 0) <= 0:
            errors.append(f"{d}: moe grad zero")
        if not np.isfinite(float(stats.get("L_total") or float("nan"))):
            errors.append(f"{d}: non-finite L_total")
        if peak_alloc / total_cuda > CUDA_SAFE_FRAC:
            errors.append(f"{d}: CUDA peak {peak_alloc:.2f}/{total_cuda:.2f} > {CUDA_SAFE_FRAC}")
        if peak_rss_gib() > HOST_SAFE_RSS_GIB:
            errors.append(f"{d}: host RSS {peak_rss_gib():.2f} > {HOST_SAFE_RSS_GIB}")
        # Do not persist BN updates from no-step path into smoke trees; discard.
        del batch, stats, loader
        rng_states[d] = snapshot_rng()

    max_alloc = max(r["cuda_peak_alloc_gib"] for r in rows)
    max_rss = max(r["host_rss_gib"] for r in rows)
    ok = not errors
    verdict = "PASS_WITH_HEADROOM" if ok else "FAIL"
    payload = {
        "ok": ok,
        "verdict": verdict,
        "errors": errors,
        "arm": ARM,
        "objective_id": OBJECTIVE_ID,
        "recipe_smoke_paths": {
            "result_root": recipe["result_root"],
            "ckpt_root": recipe["ckpt_root"],
        },
        "note": "one batch/domain; do_optimizer_step=False; dual-view GBT + TF(view1)",
        "init_sha256": init_sha,
        "rows": rows,
        "max_cuda_peak_alloc_gib": max_alloc,
        "max_host_rss_gib": max_rss,
        "host_mem_limit_gib": HOST_MEM_LIMIT_GIB,
        "host_safe_rss_gib": HOST_SAFE_RSS_GIB,
        "cuda_safe_frac": CUDA_SAFE_FRAC,
        "rss_before_domains_gib": rss0,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": os.environ.get("SLURM_JOB_ID"),
    }
    write_json(OUT_DIR / "aggregate.json", payload)
    twin = ROOT / "results/diagnostics/financial_multidataset_gbt_tf_adaptive_stdfloor_1e4_memory_preflight.json"
    write_json(twin, payload)
    print(json.dumps({k: payload[k] for k in ("ok", "verdict", "errors", "max_cuda_peak_alloc_gib", "max_host_rss_gib")}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
