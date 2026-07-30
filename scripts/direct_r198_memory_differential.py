#!/usr/bin/env python3
"""Identical-batch CUDA memory differential: projected (P) vs direct-R198 full vs seed-only.

Records allocated/reserved and key tensor shapes after:
  1 batch_to_gpu  2 views  3-5 GNN/readout  6 seed_select  7 view2  8 InfoNCE  9 backward

No AMP. Writes JSON under results/diagnostics/.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import torch
from torch_geometric.nn import to_hetero

from contrastive_loss import edge_identity_infonce_loss
from contrastive_projection import setup_contrastive_projection
from data_loading import get_data
from direct_r198.seed_readout import (
    align_seed_r198_pair,
    forward_seed_r198_hetero,
    tensor_nbytes,
)
from graph_augmentations import generate_views
from models import GINe
from train_util import (
    AddEgoIds,
    FORWARD_EDGE_TYPE,
    add_arange_ids,
    attach_edge_id_from_batch,
    extract_param,
    get_hetero_seed_edge_ids,
    get_loaders,
    select_shared_seed_edge_embeddings,
)
from util import create_parser, logger_setup, set_seed

ROOT = Path(__file__).resolve().parents[1]


def _mem(device):
    if device.type != "cuda":
        return {"allocated": 0, "reserved": 0}
    if not torch.cuda.is_initialized():
        torch.zeros(1, device=device)
    torch.cuda.synchronize()
    return {
        "allocated": int(torch.cuda.memory_allocated()),
        "reserved": int(torch.cuda.memory_reserved()),
        "max_allocated": int(torch.cuda.max_memory_allocated()),
        "max_reserved": int(torch.cuda.max_memory_reserved()),
    }


def _snap(device, label, extra=None):
    row = {"label": label, **_mem(device)}
    if extra:
        row["extra"] = extra
    return row


def _reset_cuda(device):
    gc.collect()
    if device.type == "cuda":
        # Ensure CUDA context exists before querying/resetting peaks.
        if not torch.cuda.is_initialized():
            torch.zeros(1, device=device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def _build_models(sample_batch, device, mode: str):
    """mode in {P, D198_full, D198_seed}."""
    e_dim = sample_batch[FORWARD_EDGE_TYPE].edge_attr.shape[1] - 1
    n_feats = sample_batch["node"].x.shape[1]
    bypass = mode != "P"
    model = GINe(
        num_features=n_feats,
        num_gnn_layers=int(extract_param("n_gnn_layers", SimpleNamespace(model="gin", n_gnn_layers=None)) or 2)
        if False
        else 2,
        n_classes=2,
        n_hidden=66,
        edge_updates=True,
        edge_dim=e_dim,
        embedding_dim=128 if not bypass else 198,
        supervised_head="embedding",
        bypass_embedding_head=bypass,
    )
    # Use same hidden/layers as training defaults via extract_param after args ready —
    # caller patches n_hidden if needed. Defaults 66 / 2 match Small-HI gin.
    model = to_hetero(model, sample_batch.metadata(), aggr="mean").to(device)
    proj = None
    if mode == "P":
        args = SimpleNamespace(contrast_projection_head=True, contrast_projection_hidden=128, contrast_projection_dim=128)
        proj = setup_contrastive_projection(args, device, embedding_dim=128)
    return model, proj


def _run_arm(mode, batch, device, seed_edge_ids, contrast_kwargs):
    _reset_cuda(device)
    snaps = []
    batch = batch.to(device, non_blocking=False)
    snaps.append(_snap(device, "1_batch_to_gpu", {
        "n_nodes": int(batch["node"].num_nodes),
        "n_fwd": int(batch[FORWARD_EDGE_TYPE].edge_index.shape[1]),
        "n_rev": int(batch["node", "rev_to", "node"].edge_index.shape[1]),
    }))

    # Fresh models each arm so state is isolated
    # Infer edge_dim from stripped later — use batch as-is without ID col for model:
    # Match training: edge_attr[:,1:] is applied in train loop; here batch may still have ID in col0.
    # detach_edge_id already set via attach; training strips before model...
    # Copy and strip ID col for model inputs inside generate_views? generate_views keeps edge_id store.

    e_dim = batch[FORWARD_EDGE_TYPE].edge_attr.shape[1] - 1
    n_feats = batch["node"].x.shape[1] + (1 if False else 0)
    # ego may add feature — use batch x dim
    n_feats = int(batch["node"].x.shape[1])
    bypass = mode != "P"
    homo = GINe(
        num_features=n_feats,
        num_gnn_layers=2,
        n_classes=2,
        n_hidden=66,
        edge_updates=True,
        edge_dim=e_dim,
        embedding_dim=128 if not bypass else 198,
        supervised_head="embedding",
        bypass_embedding_head=bypass,
    )
    # Strip ID for edge_emb dim: training strips before forward on views
    model = to_hetero(homo, batch.metadata(), aggr="mean").to(device)
    proj = None
    if mode == "P":
        proj = setup_contrastive_projection(
            SimpleNamespace(
                contrast_projection_head=True,
                contrast_projection_hidden=128,
                contrast_projection_dim=128,
            ),
            device,
            embedding_dim=128,
        )
        has_emb_head = True
    else:
        has_emb_head = False

    snaps.append(_snap(device, "1b_model_built", {
        "mode": mode,
        "bypass_embedding_head": bypass,
        "has_embedding_head": has_emb_head,
        "has_projection_head": proj is not None,
        "model_embedding_out_dim": 128 if mode == "P" else 198,
    }))

    view1, view2 = generate_views(batch, **contrast_kwargs)
    # Strip edge id channel like training
    for v in (view1, view2):
        for et in (FORWARD_EDGE_TYPE, ("node", "rev_to", "node")):
            ea = v[et].edge_attr
            if ea.size(-1) == e_dim + 1:
                v[et].edge_attr = ea[:, 1:]
    snaps.append(_snap(device, "2_views", {
        "view1_fwd": int(view1[FORWARD_EDGE_TYPE].edge_index.shape[1]),
        "view2_fwd": int(view2[FORWARD_EDGE_TYPE].edge_index.shape[1]),
    }))

    findings = {
        "computes_h128": mode == "P",
        "computes_z128": mode == "P",
        "computes_full_r198_both_edge_types": mode == "D198_full",
        "seed_only_r198": mode == "D198_seed",
        "sample_batch_retained": False,
    }

    if mode == "D198_seed":
        z1_s, id1, st1 = forward_seed_r198_hetero(model, view1, seed_edge_ids)
        snaps.append(_snap(device, "3_5_mp_seed_r198_view1", st1))
        with torch.no_grad():
            z2_s, id2, st2 = forward_seed_r198_hetero(model, view2, seed_edge_ids)
        snaps.append(_snap(device, "7_view2_seed_r198", st2))
        z1_seed, seed_id1, z2_seed, seed_id2 = align_seed_r198_pair(z1_s, id1, z2_s, id2)
        z2_seed = z2_seed.detach().clone()
        del z1_s, z2_s, view1, view2
        snaps.append(_snap(device, "6_seed_select", {
            "z1_seed": list(z1_seed.shape),
            "bytes_z1_seed": tensor_nbytes(z1_seed),
            "n_shared": int(seed_id1.numel()),
        }))
        findings["slice_keeps_full_r198_base"] = False
    else:
        out1 = model(view1.x_dict, view1.edge_index_dict, view1.edge_attr_dict)
        z1 = out1[FORWARD_EDGE_TYPE]
        z_rev = out1.get(("node", "rev_to", "node"))
        snaps.append(_snap(device, "3_5_full_readout_view1", {
            "z1_shape": list(z1.shape),
            "bytes_z1": tensor_nbytes(z1),
            "bytes_z_rev": tensor_nbytes(z_rev) if z_rev is not None else 0,
            "out_keys": [str(k) for k in out1.keys()] if hasattr(out1, "keys") else [],
        }))
        with torch.no_grad():
            out2 = model(view2.x_dict, view2.edge_index_dict, view2.edge_attr_dict)
            z2 = out2[FORWARD_EDGE_TYPE]
        snaps.append(_snap(device, "7_view2_full", {"z2_shape": list(z2.shape), "bytes_z2": tensor_nbytes(z2)}))
        z1_seed, seed_id1, z2_seed, seed_id2 = select_shared_seed_edge_embeddings(
            z1, view1[FORWARD_EDGE_TYPE].edge_id, z2, view2[FORWARD_EDGE_TYPE].edge_id, seed_edge_ids
        )
        # Probe whether seed slice shares storage with full z1
        try:
            same_storage = z1_seed.untyped_storage().data_ptr() == z1.untyped_storage().data_ptr()
        except Exception:
            same_storage = None
        findings["slice_keeps_full_r198_base"] = bool(same_storage) if mode == "D198_full" else False
        findings["slice_same_storage_as_z1"] = same_storage
        z2_seed = z2_seed.detach().clone()
        del out1, out2, z1, z2, view1, view2
        snaps.append(_snap(device, "6_seed_select", {
            "z1_seed": list(z1_seed.shape),
            "bytes_z1_seed": tensor_nbytes(z1_seed),
            "n_shared": int(seed_id1.numel()),
            "slice_same_storage_as_z1": same_storage,
        }))

    if proj is not None:
        z1_c, z2_c = proj(z1_seed), proj(z2_seed)
    else:
        z1_c, z2_c = z1_seed, z2_seed
    loss = edge_identity_infonce_loss(
        z1_c, z2_c, seed_id1, seed_id2, temperature=0.5, num_neg_samples=8192, symmetric=False
    )
    snaps.append(_snap(device, "8_infonce", {"loss": float(loss.detach().cpu())}))
    loss.backward()
    snaps.append(_snap(device, "9_backward"))
    peak = _mem(device)
    # Grad norm sanity
    gsq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            gsq += float(p.grad.detach().float().pow(2).sum().cpu())
    return {
        "mode": mode,
        "snapshots": snaps,
        "peak": peak,
        "findings": findings,
        "encoder_grad_norm": gsq ** 0.5,
        "n_shared_seeds": int(seed_id1.numel()),
        "seed_ids_sha256_first": None,
    }


def main():
    logger_setup()
    parser = create_parser()
    args = parser.parse_args([
        "--data", "Small-HI", "--model", "gin", "--reverse_mp", "--ego", "--ports", "--emlps", "--tds",
        "--correct_reverse_edge_features", "--batch_size", "8192", "--num_neighs", "100", "100",
        "--seed", "2", "--objective", "contrastive", "--testing",
        "--loader_num_workers", "0",
    ])
    set_seed(2)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with open(ROOT / "data_config.json") as f:
        data_config = json.load(f)
    tr, va, te, tr_i, va_i, te_i = get_data(args, data_config)
    transform = AddEgoIds() if args.ego else None
    add_arange_ids([tr, va, te])
    tr_loader, _, _ = get_loaders(tr, va, te, tr_i, va_i, te_i, transform, args)
    batch = next(iter(tr_loader))
    seed_edge_ids = get_hetero_seed_edge_ids(batch, tr_loader.data)
    attach_edge_id_from_batch(batch, tr_loader.data)

    from training import _contrastive_view_kwargs
    contrast_kwargs = _contrastive_view_kwargs(args, {}, seed_edge_ids=seed_edge_ids)

    results = {
        "ok": True,
        "device": str(device),
        "batch": {
            "n_nodes": int(batch["node"].num_nodes),
            "n_fwd": int(batch[FORWARD_EDGE_TYPE].edge_index.shape[1]),
            "n_rev": int(batch["node", "rev_to", "node"].edge_index.shape[1]),
            "n_seeds": int(seed_edge_ids.numel()),
        },
        "amp": False,
        "cache_cleared_between_arms": True,
        "arms": {},
    }
    for mode in ("P", "D198_full", "D198_seed"):
        logging.info("=== memory arm %s ===", mode)
        try:
            results["arms"][mode] = _run_arm(mode, batch.clone(), device, seed_edge_ids, contrast_kwargs)
        except torch.OutOfMemoryError as e:
            _reset_cuda(device)
            results["arms"][mode] = {"mode": mode, "oom": True, "error": str(e), "peak": _mem(device)}
            logging.exception("OOM on %s", mode)
        _reset_cuda(device)

    # Summary deltas
    def peak_alloc(m):
        arm = results["arms"].get(m) or {}
        if arm.get("oom"):
            return None
        return (arm.get("peak") or {}).get("max_allocated")

    results["summary"] = {
        "P_max_allocated": peak_alloc("P"),
        "D198_full_max_allocated": peak_alloc("D198_full"),
        "D198_seed_max_allocated": peak_alloc("D198_seed"),
        "delta_full_minus_P": None
        if peak_alloc("D198_full") is None or peak_alloc("P") is None
        else peak_alloc("D198_full") - peak_alloc("P"),
        "delta_seed_minus_P": None
        if peak_alloc("D198_seed") is None or peak_alloc("P") is None
        else peak_alloc("D198_seed") - peak_alloc("P"),
        "bytes_saved_seed_vs_full": None
        if peak_alloc("D198_full") is None or peak_alloc("D198_seed") is None
        else peak_alloc("D198_full") - peak_alloc("D198_seed"),
    }
    out = ROOT / "results/diagnostics/direct_r198_memory_differential.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n")
    logging.info("Wrote %s summary=%s", out, results["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
