#!/usr/bin/env python3
"""Real-data dual-view GPU memory preflight for MIXED_3DOMAIN_GRAPH_BARLOW_TWINS_ONLY.

No optimizer/scheduler updates. No embeddings or training checkpoints written.
Not the 30-step smoke and not full training.
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
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
from direct_r198.seed_readout import align_seed_r198_pair, forward_seed_r198_hetero  # noqa: E402
from financial_multidataset_long_gradient_conflict.core import hash_view_forward  # noqa: E402
from graph_augmentations import (  # noqa: E402
    _hetero_random_edge_drop_view,
    mask_edge_attr,
)
from graph_barlow_twins_r198 import (  # noqa: E402
    ARM,
    OBJECTIVE_ID,
    PHASE3_INIT_SHA_PREFIX,
    PHASE3_SHARED_INIT,
    R198_DIM,
    resolved_recipe,
)
from graph_barlow_twins_r198.integrity import (  # noqa: E402
    assert_gbt_integrity,
    assert_no_forbidden_objectives,
    refuse_test_split_access,
)
from graph_barlow_twins_r198.loss import (  # noqa: E402
    GBT_EPS,
    GBT_LAMBDA,
    GBT_STD_UNBIASED,
    _feature_cross_correlation,
    edge_aligned_graph_barlow_twins_r198,
)
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
    NUM_NEIGHS,
    SEED,
)
from mixed_ssl_phase4a.domain_registry import default_smoke_domains  # noqa: E402
from mixed_ssl_phase4a.preflight import preflight_phase4a  # noqa: E402
from mixed_ssl_phase4a.schedule import loader_generator, restore_rng  # noqa: E402
from mixed_ssl_phase4b import CANONICAL_DOMAINS  # noqa: E402
from mixed_ssl_phase4b.matching import init_matching_rng_states  # noqa: E402
from train_util import (  # noqa: E402
    AddEgoIds,
    FORWARD_EDGE_TYPE,
    REVERSE_EDGE_TYPE,
    add_arange_ids,
    attach_edge_id_from_batch,
    extract_param,
    get_hetero_seed_edge_ids,
)
from training import get_model  # noqa: E402
from util import create_parser, logger_setup, set_seed  # noqa: E402

OUT_DIR = ROOT / "results/diagnostics/financial_multidataset_graph_barlow_twins_memory_preflight"
NOTE_PATH = ROOT / "notes/financial_multidataset_graph_barlow_twins_memory_preflight.md"
JSON_PATH = OUT_DIR / "aggregate.json"
TWIN_JSON = ROOT / "results/diagnostics/financial_multidataset_graph_barlow_twins_memory_preflight.json"
CSV_PATH = OUT_DIR / "per_stage_memory.csv"
GiB = 1024.0**3
HOST_MEM_LIMIT_GIB = 128.0
HOST_SAFE_RSS_GIB = 120.0  # leave margin under 128G Slurm mem

FORMULA_CODE = "(z - mean) / (std_unbiased + 1e-15)"
FORMULA_PROSE_CORRECT = "(Za - mean_0) / (std_0_unbiased + ε)"


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}.{time.time_ns()}")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def peak_rss_gib() -> float:
    # Linux: ru_maxrss is KiB
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


def verify_formula_in_source() -> Dict[str, Any]:
    loss_src = (ROOT / "graph_barlow_twins_r198/loss.py").read_text(encoding="utf-8")
    ok = "(z_a_f - mean_a) / (std_a + float(eps))" in loss_src
    ok = ok and "(z_b_f - mean_b) / (std_b + float(eps))" in loss_src
    bad = "/ (std_a) + " in loss_src or "/ std_a +" in loss_src
    note_src = (
        ROOT / "notes/financial_multidataset_graph_barlow_twins_implementation.md"
    ).read_text(encoding="utf-8")
    note_ok = FORMULA_CODE in note_src and "prose typo" in note_src.lower()
    return {
        "ok": bool(ok and not bad),
        "code_pattern": FORMULA_CODE,
        "prose_correct": FORMULA_PROSE_CORRECT,
        "prose_typo_not_used_in_code": True,
        "implementation_note_clarifies_typo": bool(note_ok),
        "note": "Any chat/summary rendering as (Z-mean)/std + eps is a prose typo only.",
    }


def cuda_mem_snapshot(tag: str, **extra) -> Dict[str, Any]:
    snap: Dict[str, Any] = {
        "stage": tag,
        "host_rss_gib": peak_rss_gib(),
        "cuda_available": bool(torch.cuda.is_available()),
        **extra,
    }
    if not torch.cuda.is_available():
        return snap
    free, total = torch.cuda.mem_get_info()
    snap.update(
        {
            "allocated_gib": torch.cuda.memory_allocated() / GiB,
            "reserved_gib": torch.cuda.memory_reserved() / GiB,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / GiB,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / GiB,
            "cuda_free_gib": free / GiB,
            "cuda_total_gib": total / GiB,
            "cuda_used_gib": (total - free) / GiB,
        }
    )
    return snap


def reset_cuda_peaks() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def clone_param_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def clone_grad_state(model: nn.Module) -> Dict[str, Optional[torch.Tensor]]:
    out: Dict[str, Optional[torch.Tensor]] = {}
    for name, p in model.named_parameters():
        out[name] = None if p.grad is None else p.grad.detach().cpu().clone()
    return out


def restore_param_state(model: nn.Module, state: Dict[str, torch.Tensor]) -> None:
    device = next(model.parameters()).device
    model.load_state_dict({k: v.to(device) for k, v in state.items()}, strict=True)


def restore_grad_state(model: nn.Module, grads: Dict[str, Optional[torch.Tensor]]) -> None:
    for name, p in model.named_parameters():
        g = grads.get(name)
        if g is None:
            p.grad = None
        else:
            p.grad = g.to(p.device)


def grads_equal(
    a: Dict[str, Optional[torch.Tensor]], b: Dict[str, Optional[torch.Tensor]]
) -> bool:
    if set(a) != set(b):
        return False
    for k in a:
        ga, gb = a[k], b[k]
        if ga is None and gb is None:
            continue
        if ga is None or gb is None:
            return False
        if not torch.equal(ga, gb):
            return False
    return True


def make_ns(data: str) -> argparse.Namespace:
    argv = [
        "--data",
        data,
        "--model",
        "gin",
        "--objective",
        "contrastive",
        "--unique_name",
        f"gbt_mem_preflight_{data}",
        "--seed",
        str(SEED),
        "--batch_size",
        str(BATCH_SIZE),
        "--num_neighs",
        "100",
        "100",
        "--loader_num_workers",
        "0",
        "--reverse_mp",
        "--ego",
        "--ports",
        "--emlps",
        "--tds",
        "--correct_reverse_edge_features",
        "--feature_contract",
        CONTRACT_ID,
        "--train_fit_edge_znorm",
        "--skip_test_eval",
        "--direct_r198_infonce",
        "--contrastive_num_neg_samples",
        "0",
        "--contrastive_memory_bank_size",
        "0",
        "--contrastive_accum_steps",
        "1",
        "--contrastive_temperature",
        "0.5",
        "--max_optimizer_steps",
        "1",
        "--edge_attr_mask_rate",
        "0.1",
        "--edge_drop_target_rate",
        "0.1",
    ]
    ns = create_parser().parse_args(argv)
    # GBT isolation: no asymmetric detach, no projection, no TF.
    ns.contrastive_asymmetric = False
    ns.contrast_projection_head = False
    ns.direct_r198_tfmoe = False
    ns.skip_test_eval = True
    ns.loader_num_workers = 0
    ns.preserve_seed_edges = False
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
    local = state_dict_sha256(model.state_dict())
    return {
        "ok": True,
        "path": str(init_path),
        "file_sha256": file_sha,
        "init_sha256": init_sha,
        "encoder_state_sha256": local,
        "edge_dim": 6,
    }


def classify_result(
    *,
    domain_rows: List[Dict[str, Any]],
    oom: bool,
    integrity_ok: bool,
    gpu_total_gib: float,
    max_host_rss_gib: float,
) -> str:
    if oom:
        return "FAIL_OOM"
    if not integrity_ok:
        return "FAIL_INTEGRITY"
    if not domain_rows:
        return "FAIL_INTEGRITY"
    if max_host_rss_gib >= HOST_MEM_LIMIT_GIB:
        return "FAIL_INTEGRITY"
    headroom_ok = True
    tight = False
    if max_host_rss_gib > HOST_SAFE_RSS_GIB:
        headroom_ok = False
        tight = True
    for row in domain_rows:
        peak_reserved = float(row["peak_reserved_gib_after_backward"])
        free_at_peak = float(row["cuda_free_gib_at_peak_stage"])
        rem = gpu_total_gib - peak_reserved
        if peak_reserved > gpu_total_gib + 1e-6:
            return "FAIL_OOM"
        if peak_reserved > 0.85 * gpu_total_gib or rem < 8.0 or free_at_peak < 8.0:
            headroom_ok = False
            tight = True
    if headroom_ok:
        return "PASS_WITH_HEADROOM"
    if tight:
        return "PASS_TIGHT"
    return "FAIL_INTEGRITY"


def _augment_one_view(
    batch,
    *,
    edge_drop_rate: float,
    edge_attr_mask_rate: float,
    seed_edge_ids: torch.Tensor,
) -> HeteroData:
    """Independent single-view augmentation matching generate_views random hetero path."""
    view = _hetero_random_edge_drop_view(
        batch,
        edge_drop_rate,
        seed_edge_ids=seed_edge_ids,
        preserve_seed_edges=False,
    )
    if edge_attr_mask_rate > 0:
        for et in (FORWARD_EDGE_TYPE, REVERSE_EDGE_TYPE):
            store = view[et]
            if store.edge_attr is not None:
                store.edge_attr = mask_edge_attr(
                    store.edge_attr,
                    mask_rate=edge_attr_mask_rate,
                    mask_value=0.0,
                )
    return view


def run_domain_preflight(
    *,
    domain: str,
    model: nn.Module,
    data: HeteroData,
    ns: argparse.Namespace,
    device: torch.device,
    bn_bundle: Dict[str, Any],
    rng_state: Any,
    transform,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    stages: List[Dict[str, Any]] = []
    refuse_test_split_access("train")

    reset_cuda_peaks()
    stages.append(cuda_mem_snapshot("after_model_domain_residency"))

    param_before = clone_param_state(model)
    grad_before = clone_grad_state(model)
    encoder_sha_before = state_dict_sha256(model.state_dict())
    bn_before = clone_bn_bundle(bn_bundle)
    rng_before = deepcopy(rng_state)

    restore_rng(rng_state)
    apply_bn_(model, bn_bundle)
    model.train()

    loader = build_train_loader(data, transform, domain=domain)
    batch = next(iter(loader))
    del loader

    seed_edge_ids = get_hetero_seed_edge_ids(batch, data)
    attach_edge_id_from_batch(batch, data)
    batch = batch.to(device)
    seed_edge_ids = seed_edge_ids.to(device)
    stages.append(
        cuda_mem_snapshot(
            "after_batch_transfer",
            batch_num_nodes=int(batch["node"].num_nodes) if "node" in batch.node_types else None,
            seed_count=int(seed_edge_ids.numel()),
        )
    )

    drop_rate = float(getattr(ns, "edge_drop_target_rate", 0.1))
    mask_rate = float(getattr(ns, "edge_attr_mask_rate", 0.1))

    # View 1: independent aug → live forward
    view1 = _augment_one_view(
        batch,
        edge_drop_rate=drop_rate,
        edge_attr_mask_rate=mask_rate,
        seed_edge_ids=seed_edge_ids,
    )
    stages.append(cuda_mem_snapshot("after_view1_augmentation"))
    v1_hash = hash_view_forward(view1)
    z1_all, id1_all, _ = forward_seed_r198_hetero(model, view1, seed_edge_ids)
    stages.append(
        cuda_mem_snapshot("after_view1_forward", z1_all_shape=list(z1_all.shape))
    )

    # View 2: independent aug → live forward
    view2 = _augment_one_view(
        batch,
        edge_drop_rate=drop_rate,
        edge_attr_mask_rate=mask_rate,
        seed_edge_ids=seed_edge_ids,
    )
    stages.append(cuda_mem_snapshot("after_view2_augmentation"))
    v2_hash = hash_view_forward(view2)
    z2_all, id2_all, _ = forward_seed_r198_hetero(model, view2, seed_edge_ids)
    stages.append(
        cuda_mem_snapshot("after_view2_forward", z2_all_shape=list(z2_all.shape))
    )

    z1, id1, z2, id2 = align_seed_r198_pair(z1_all, id1_all, z2_all, id2_all)
    scored = int(id1.numel())
    if scored < 2:
        raise RuntimeError(f"{domain}: B<2 aligned seeds ({scored})")
    if tuple(z1.shape) != (scored, R198_DIM) or tuple(z2.shape) != (scored, R198_DIM):
        raise RuntimeError(f"{domain}: Za/Zb must be Bx198, got {tuple(z1.shape)}/{tuple(z2.shape)}")
    sid_hash = seed_ids_sha(id1)

    view_grads: Dict[str, Optional[torch.Tensor]] = {"v1": None, "v2": None}

    def h1(g):
        view_grads["v1"] = g.detach()

    def h2(g):
        view_grads["v2"] = g.detach()

    hook1 = z1.register_hook(h1)
    hook2 = z2.register_hook(h2)

    # Staged: construct C, snapshot, then finish loss
    eps = float(GBT_EPS)
    z_a_f = z1.float()
    z_b_f = z2.float()
    mean_a = z_a_f.mean(dim=0)
    mean_b = z_b_f.mean(dim=0)
    std_a = z_a_f.std(dim=0, unbiased=GBT_STD_UNBIASED)
    std_b = z_b_f.std(dim=0, unbiased=GBT_STD_UNBIASED)
    z_a_norm = (z_a_f - mean_a) / (std_a + eps)
    z_b_norm = (z_b_f - mean_b) / (std_b + eps)
    c = _feature_cross_correlation(z_a_norm, z_b_norm)
    c_bytes = int(c.numel() * c.element_size())
    expected_c = R198_DIM * R198_DIM * 4
    if abs(c_bytes - expected_c) > 64:
        raise RuntimeError(f"C bytes {c_bytes} != expected ~{expected_c} (~153 KiB fp32)")
    if tuple(c.shape) != (R198_DIM, R198_DIM):
        raise RuntimeError(f"C shape {tuple(c.shape)} != (198,198)")
    stages.append(
        cuda_mem_snapshot(
            "after_constructing_C",
            C_shape=list(c.shape),
            C_allocated_bytes=c_bytes,
            C_kib=c_bytes / 1024.0,
            z1_shape=list(z1.shape),
            z2_shape=list(z2.shape),
            B=scored,
        )
    )

    off_mask = ~torch.eye(R198_DIM, dtype=torch.bool, device=c.device)
    l_inv = (1.0 - c.diagonal()).pow(2).sum()
    l_red = float(GBT_LAMBDA) * c[off_mask].pow(2).sum()
    total = l_inv + l_red

    # Official path agreement (detached)
    with torch.no_grad():
        ref_loss, ref_diag = edge_aligned_graph_barlow_twins_r198(
            z1.detach(), z2.detach(), id1.detach(), id2.detach()
        )
    if abs(float(total.detach()) - float(ref_loss)) > 1e-4:
        raise RuntimeError(
            f"{domain}: staged loss {float(total.detach())} != official {float(ref_loss)}"
        )

    diag = {
        "L_gbt_total": float(total.detach().item()),
        "C_shape": list(c.shape),
        "C_allocated_bytes": c_bytes,
        "C_kib": c_bytes / 1024.0,
        "infonce_enabled": False,
        "tfmoe_enabled": False,
        "projection_enabled": False,
        "alpha_beta_enabled": False,
        "negatives_enabled": False,
        "mean_diag_C": float(c.diagonal().detach().mean().item()),
        "official_ref_L": float(ref_loss),
    }
    assert_no_forbidden_objectives(diag)
    stages.append(
        cuda_mem_snapshot(
            "after_loss_construction",
            L_gbt_total=diag["L_gbt_total"],
            C_shape=diag["C_shape"],
            C_kib=diag["C_kib"],
            B=scored,
        )
    )

    model.zero_grad(set_to_none=True)
    total.backward()
    hook1.remove()
    hook2.remove()
    stages.append(cuda_mem_snapshot("after_backward"))

    enc_gn = 0.0
    for p in model.parameters():
        if p.grad is not None:
            enc_gn += float(p.grad.detach().float().pow(2).sum())
    enc_gn = float(enc_gn**0.5)
    v1_gn = float(view_grads["v1"].float().norm()) if view_grads["v1"] is not None else 0.0
    v2_gn = float(view_grads["v2"].float().norm()) if view_grads["v2"] is not None else 0.0
    if not (torch.isfinite(torch.tensor(enc_gn)) and enc_gn > 0):
        raise RuntimeError(f"{domain}: encoder grad not finite/nonzero ({enc_gn})")
    if not (v1_gn > 0 and v2_gn > 0 and np.isfinite(v1_gn) and np.isfinite(v2_gn)):
        raise RuntimeError(f"{domain}: view grads bad v1={v1_gn} v2={v2_gn}")

    integ = assert_gbt_integrity(
        encoder_grad_norm=enc_gn,
        view1_repr_grad_norm=v1_gn,
        view2_repr_grad_norm=v2_gn,
        moe_grad_norm=0.0,
        alpha_beta_grad_norm=0.0,
        projection_grad_norm=0.0,
        loss_finite=bool(torch.isfinite(total).item()),
        c_shape=tuple(diag["C_shape"]),
    )
    if not integ["ok"]:
        raise RuntimeError(f"{domain} integrity failed: {integ['errors']}")

    # NO optimizer.step / scheduler.step — restore params, grads, BN, RNG
    restore_param_state(model, param_before)
    restore_grad_state(model, grad_before)
    apply_bn_(model, bn_before)
    for k in list(bn_bundle.keys()):
        bn_bundle[k] = bn_before[k]
    restore_rng(rng_before)

    encoder_sha_after = state_dict_sha256(model.state_dict())
    if encoder_sha_after != encoder_sha_before:
        raise RuntimeError(f"{domain}: encoder hash changed despite restore")
    if not grads_equal(grad_before, clone_grad_state(model)):
        raise RuntimeError(f"{domain}: gradient restore mismatch")
    if not bn_bundles_equal(bn_bundle, bn_before):
        raise RuntimeError(f"{domain}: BN restore mismatch")

    del z1_all, z2_all, z1, z2, id1, id2, id1_all, id2_all
    del view1, view2, batch, total, c, seed_edge_ids, z_a_norm, z_b_norm
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    stages.append(cuda_mem_snapshot("after_cleanup"))

    peak_reserved = max(float(s.get("peak_reserved_gib", 0.0) or 0.0) for s in stages)
    peak_alloc = max(float(s.get("peak_allocated_gib", 0.0) or 0.0) for s in stages)
    # Free memory at the stage that recorded the highest peak_reserved
    peak_stage = max(stages, key=lambda s: float(s.get("peak_reserved_gib", 0.0) or 0.0))
    free_at_peak = float(peak_stage.get("cuda_free_gib", float("nan")))
    max_rss = max(float(s.get("host_rss_gib", 0.0) or 0.0) for s in stages)

    summary = {
        "domain": domain,
        "B": scored,
        "z_shape": [scored, R198_DIM],
        "C_shape": diag["C_shape"],
        "C_allocated_bytes": c_bytes,
        "C_kib": c_bytes / 1024.0,
        "L_gbt_total": diag["L_gbt_total"],
        "encoder_grad_norm": enc_gn,
        "view1_repr_grad_norm": v1_gn,
        "view2_repr_grad_norm": v2_gn,
        "integrity_ok": True,
        "seed_ids_sha256": sid_hash,
        "view1_aug_sha256": v1_hash,
        "view2_aug_sha256": v2_hash,
        "encoder_sha_before": encoder_sha_before,
        "encoder_sha_after_restore": encoder_sha_after,
        "params_restored": True,
        "grads_restored": True,
        "bn_restored": True,
        "rng_restored": True,
        "optimizer_step": False,
        "scheduler_step": False,
        "peak_allocated_gib": peak_alloc,
        "peak_reserved_gib_after_backward": peak_reserved,
        "cuda_free_gib_at_peak_stage": free_at_peak,
        "peak_stage_name": peak_stage.get("stage"),
        "min_cuda_free_during_domain_gib": min(
            float(s.get("cuda_free_gib", 1e9) or 1e9) for s in stages if "cuda_free_gib" in s
        ),
        "max_host_rss_gib": max_rss,
        "infonce_enabled": False,
        "tfmoe_enabled": False,
        "projection_enabled": False,
        "alpha_beta_enabled": False,
        "test_split_loaded": False,
        "stages": stages,
        "rng_restored_state": rng_before,
    }
    return summary, stages


def run_memory_preflight() -> Dict[str, Any]:
    t0 = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    formula = verify_formula_in_source()
    if not formula["ok"]:
        raise RuntimeError("formula verification failed against loss.py")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for dual-view memory preflight")

    device = torch.device("cuda:0")
    logger_setup()
    set_seed(SEED)

    # Confirm test-split gate works, then proceed on train-only.
    try:
        refuse_test_split_access("test")
        raise RuntimeError("test-split gate failed to raise")
    except RuntimeError as e:
        if "Test split access refused" not in str(e):
            raise
    refuse_test_split_access("train")

    pre = preflight_phase4a(root=ROOT, specs=list(default_smoke_domains()))
    if not pre.get("ok"):
        raise RuntimeError(f"phase4a preflight failed: {pre}")

    specs = [s for s in default_smoke_domains() if s.dataset_id in set(CANONICAL_DOMAINS)]
    domains = list(CANONICAL_DOMAINS)
    transform = AddEgoIds()

    ns_by: Dict[str, argparse.Namespace] = {}
    data_by: Dict[str, HeteroData] = {}
    with open(ROOT / "data_config.json") as f:
        data_config = json.load(f)

    load_meta: Dict[str, Any] = {}
    for spec in specs:
        d = spec.dataset_id
        logging.info("Loading %s (train only; skip_test_eval) ...", d)
        ns = make_ns(d)
        tr, va, te, tr_i, va_i, te_i = get_data(ns, data_config)
        del te, te_i, va, va_i, tr_i
        gc.collect()
        add_arange_ids([tr])
        ns_by[d] = ns
        data_by[d] = tr
        load_meta[d] = {"rss_gib": peak_rss_gib()}

    sample_dom = domains[0]
    sample_loader = build_train_loader(data_by[sample_dom], transform, domain=sample_dom)
    sample = next(iter(sample_loader))
    del sample_loader
    model, emb_dim = build_model(ns_by[sample_dom], data_by[sample_dom], sample, device)
    del sample
    init_prov = load_phase3_encoder_only(model, ROOT / PHASE3_SHARED_INIT)
    init_encoder_sha = init_prov["encoder_state_sha256"]
    init_file_sha = init_prov["file_sha256"]

    bn_init = clone_bn_bundle(collect_bn_bundle(model))
    bn_bundles = {d: clone_bn_bundle(bn_init) for d in domains}
    rng_states = init_matching_rng_states(SEED, active_domains=domains)

    residency = cuda_mem_snapshot("after_all_domains_resident_and_model_init")
    gpu_total = float(residency.get("cuda_total_gib") or 0.0)

    domain_summaries: List[Dict[str, Any]] = []
    all_stage_rows: List[Dict[str, Any]] = []
    oom = False
    integrity_ok = True
    fail_stage = None
    fail_error = None

    for d in domains:
        logging.info("=== dual-view memory preflight domain=%s ===", d)
        try:
            summary, stages = run_domain_preflight(
                domain=d,
                model=model,
                data=data_by[d],
                ns=ns_by[d],
                device=device,
                bn_bundle=bn_bundles[d],
                rng_state=rng_states[d],
                transform=transform,
            )
            rng_states[d] = summary.pop("rng_restored_state")
            domain_summaries.append(summary)
            for s in stages:
                all_stage_rows.append({"domain": d, **s})
            if state_dict_sha256(model.state_dict()) != init_encoder_sha:
                integrity_ok = False
                fail_error = f"encoder sha drifted after {d}"
                break
            if file_sha256(ROOT / PHASE3_SHARED_INIT) != init_file_sha:
                integrity_ok = False
                fail_error = f"Phase-3 init file hash changed after {d}"
                break
        except torch.cuda.OutOfMemoryError as e:
            oom = True
            fail_stage = d
            fail_error = f"CUDA OOM: {e}"
            logging.exception("OOM on domain %s — stopping (no retry)", d)
            break
        except Exception as e:
            integrity_ok = False
            fail_stage = d
            fail_error = f"{type(e).__name__}: {e}"
            logging.exception("Integrity/runtime failure on %s", d)
            break

    final_sha = state_dict_sha256(model.state_dict())
    init_unchanged = final_sha == init_encoder_sha and file_sha256(ROOT / PHASE3_SHARED_INIT) == init_file_sha
    max_host_rss = max(
        [float(residency.get("host_rss_gib") or 0.0)]
        + [float(s.get("max_host_rss_gib", 0.0)) for s in domain_summaries]
        + [float(m.get("rss_gib", 0.0)) for m in load_meta.values()],
        default=0.0,
    )

    classification = classify_result(
        domain_rows=domain_summaries,
        oom=oom,
        integrity_ok=integrity_ok and init_unchanged and formula["ok"],
        gpu_total_gib=gpu_total,
        max_host_rss_gib=max_host_rss,
    )

    if all_stage_rows:
        keys = sorted({k for r in all_stage_rows for k in r.keys()})
        with CSV_PATH.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in all_stage_rows:
                w.writerow(r)

    max_peak_reserved = max(
        (float(s.get("peak_reserved_gib_after_backward", 0.0)) for s in domain_summaries),
        default=0.0,
    )
    min_rem = gpu_total - max_peak_reserved if gpu_total else float("nan")
    min_free_at_peak = min(
        (float(s.get("cuda_free_gib_at_peak_stage", 1e9)) for s in domain_summaries),
        default=float("nan"),
    )

    proposed_smoke = (
        "sbatch --partition=mit_normal_gpu "
        "--account=mit_amf_advanced_gpu --qos=mit_amf_advanced_gpu "
        "--gres=gpu:1 --cpus-per-task=16 --mem=128G --time=01:00:00 "
        "slurm/run_mixed_3domain_graph_barlow_twins_only_smoke.sh"
    )

    result = {
        "title": "GBT dual-view GPU memory preflight",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "arm": ARM,
        "objective_id": OBJECTIVE_ID,
        "mode": "--memory-preflight-only",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_flags": {
            "partition": os.environ.get("SLURM_JOB_PARTITION") or "mit_normal_gpu",
            "account": os.environ.get("SLURM_JOB_ACCOUNT") or "mit_amf_advanced_gpu",
            "qos": os.environ.get("SLURM_JOB_QOS") or "mit_amf_advanced_gpu",
            "gres": "gpu:1",
            "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK") or "16",
            "mem": "128G",
            "time": "02:00:00",
            "loader_workers": 0,
        },
        "formula_verification": formula,
        "recipe": resolved_recipe(smoke_steps=30),
        "batch_size": BATCH_SIZE,
        "num_neighbors": list(NUM_NEIGHS),
        "loader_workers": 0,
        "domains": domains,
        "load_meta": load_meta,
        "init_provenance": init_prov,
        "init_encoder_sha_unchanged": init_unchanged,
        "final_encoder_sha256": final_sha,
        "residency": residency,
        "domain_summaries": [
            {k: v for k, v in s.items() if k != "stages"} for s in domain_summaries
        ],
        "domain_stages": {s["domain"]: s.get("stages") for s in domain_summaries},
        "classification": classification,
        "max_peak_reserved_gib": max_peak_reserved,
        "max_peak_allocated_gib": max(
            (float(s.get("peak_allocated_gib", 0.0)) for s in domain_summaries), default=0.0
        ),
        "min_remaining_headroom_gib": min_rem,
        "min_cuda_free_at_peak_gib": min_free_at_peak,
        "max_host_rss_gib": max_host_rss,
        "gpu_total_gib": gpu_total,
        "oom": oom,
        "integrity_ok": integrity_ok and init_unchanged,
        "fail_stage": fail_stage,
        "fail_error": fail_error,
        "no_optimizer_step": True,
        "no_scheduler_step": True,
        "no_embeddings_written": True,
        "no_training_checkpoint": True,
        "no_test_split": True,
        "elapsed_sec": time.perf_counter() - t0,
        "csv_path": str(CSV_PATH),
        "proposed_smoke_command": proposed_smoke,
        "proposed_smoke_submitted": False,
    }
    write_json(JSON_PATH, result)
    write_json(TWIN_JSON, result)

    # Discard resident objects
    del model, data_by, ns_by, bn_bundles, rng_states
    gc.collect()
    torch.cuda.empty_cache()
    return result


def write_note(result: Dict[str, Any]) -> None:
    lines = [
        "# Graph Barlow Twins dual-view GPU memory preflight",
        "",
        f"**Classification:** `{result['classification']}`",
        f"**Job ID:** `{result.get('slurm_job_id')}`",
        f"**Generated:** {result['generated_at_utc']}",
        f"**Mode:** `--memory-preflight-only` (no optimizer/scheduler updates)",
        "",
        "## Slurm flags",
        "",
        "```",
        json.dumps(result.get("slurm_flags"), indent=2),
        "```",
        "",
        "## Formula verification",
        "",
        f"- Code pattern: `{result['formula_verification']['code_pattern']}`",
        f"- OK: `{result['formula_verification']['ok']}`",
        f"- Note: {result['formula_verification']['note']}",
        f"- Implementation note clarifies typo: "
        f"`{result['formula_verification'].get('implementation_note_clarifies_typo')}`",
        "",
        "## Memory by domain",
        "",
        "| Domain | B | peak reserved GiB | free@peak GiB | enc grad | v1/v2 grads | C KiB |",
        "|--------|--:|------------------:|--------------:|---------:|------------|------:|",
    ]
    for s in result.get("domain_summaries") or []:
        lines.append(
            f"| {s['domain']} | {s['B']} | {s['peak_reserved_gib_after_backward']:.3f} | "
            f"{s['cuda_free_gib_at_peak_stage']:.3f} | {s['encoder_grad_norm']:.4g} | "
            f"{s['view1_repr_grad_norm']:.4g}/{s['view2_repr_grad_norm']:.4g} | "
            f"{s['C_kib']:.1f} |"
        )
    lines += [
        "",
        f"**Max peak reserved:** {result.get('max_peak_reserved_gib'):.3f} GiB",
        f"**Max peak allocated:** {result.get('max_peak_allocated_gib'):.3f} GiB",
        f"**Min remaining headroom (total−peak_reserved):** "
        f"{result.get('min_remaining_headroom_gib'):.3f} GiB",
        f"**Min CUDA free at peak stage:** {result.get('min_cuda_free_at_peak_gib'):.3f} GiB",
        f"**Max host RSS:** {result.get('max_host_rss_gib'):.3f} GiB",
        f"**GPU total:** {result.get('gpu_total_gib'):.3f} GiB",
        f"**Init encoder SHA unchanged:** {result.get('init_encoder_sha_unchanged')}",
        "",
        "## Gates",
        "",
        f"- integrity_ok: {result.get('integrity_ok')}",
        f"- oom: {result.get('oom')}",
        f"- fail_stage: {result.get('fail_stage')}",
        f"- fail_error: {result.get('fail_error')}",
        f"- no_optimizer_step / no_scheduler_step: True",
        f"- no_test_split / no embeddings / no training ckpt: True",
        "",
        "## Per-stage CSV",
        "",
        f"`{result.get('csv_path')}`",
        "",
        "## Proposed 30-step smoke (NOT submitted)",
        "",
        "```bash",
        result.get("proposed_smoke_command", ""),
        "```",
        "",
        "Stop after memory preflight. Do not run 30 steps, full training, extraction,",
        "probes, or test evaluation from this job.",
        "",
    ]
    NOTE_PATH.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--memory-preflight-only", action="store_true", required=True)
    args = p.parse_args(argv)
    assert args.memory_preflight_only
    try:
        result = run_memory_preflight()
        write_note(result)
        print(
            json.dumps(
                {
                    "classification": result["classification"],
                    "job": result.get("slurm_job_id"),
                    "max_peak_reserved_gib": result.get("max_peak_reserved_gib"),
                    "min_remaining_headroom_gib": result.get("min_remaining_headroom_gib"),
                },
                indent=2,
            )
        )
        return 0 if result["classification"] in ("PASS_WITH_HEADROOM", "PASS_TIGHT") else 2
    except Exception:
        err = {
            "classification": "FAIL_INTEGRITY",
            "error": traceback.format_exc(),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        }
        write_json(JSON_PATH, err)
        write_json(TWIN_JSON, err)
        NOTE_PATH.write_text(
            "# GBT memory preflight FAILED\n\n```\n" + err["error"] + "\n```\n",
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
