#!/usr/bin/env python3
"""One-batch GPU smoke for edge-centric D+ neighbor-positive transfer.

NOT an exact GCPAL reproduction. Distinct from --enable_knn_soft_positives.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from types import SimpleNamespace

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
from torch_geometric.nn import to_hetero

from contrastive_projection import project_seed_pair, setup_contrastive_projection
from data_loading import get_data
from edge_neighbor_positives import (
    NOT_EXACT_REPRODUCTION,
    build_edge_neighbor_positive_context,
    edge_neighbor_supcon_loss,
    expand_poscomplete_seeds,
)
from edge_neighbor_positives_train import _batch_for_seed_positions, _contrastive_view_kwargs
from graph_augmentations import generate_views
from train_util import (
    FORWARD_EDGE_TYPE,
    AddEgoIds,
    add_arange_ids,
    attach_edge_id_from_batch,
    select_shared_seed_edge_embeddings,
)
from training import get_model
from util import create_parser, set_seed

PEAK_GPU_MIB_LIMIT = 20000.0
KNN_CACHE = "morphology_cache/Small-HI/transaction_knn_edge_native_degree_fan_k15.npz"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    assert NOT_EXACT_REPRODUCTION

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max_total", type=int, default=2048)
    ap.add_argument(
        "--output_json",
        default="results/diagnostics/edge_dplus_neighbor_positive_smoke.json",
    )
    ap.add_argument(
        "--output_md",
        default="notes/edge_dplus_neighbor_positive_smoke.md",
    )
    smoke_args = ap.parse_args()

    parser = create_parser()
    args = parser.parse_args(
        [
            "--data", "Small-HI",
            "--model", "gin",
            "--objective", "contrastive",
            "--reverse_mp", "--ego", "--ports", "--emlps", "--tds",
            "--correct_reverse_edge_features",
            "--preserve_seed_edges",
            "--contrast_projection_head",
            "--contrast_projection_hidden", "128",
            "--contrast_projection_dim", "128",
            "--contrastive_asymmetric",
            "--contrastive_num_neg_samples", "8192",
            "--contrastive_memory_bank_size", "0",
            "--batch_size", "8192",
            "--num_neighs", "100", "100",
            "--loader_num_workers", "0",
            "--seed", "2",
            "--n_epochs", "1",
            "--enable_edge_neighbor_positives",
            "--edge_neighbor_positive_mode", "neighbor",
            "--edge_neighbor_max_total", str(smoke_args.max_total),
            "--edge_neighbor_knn_cache", KNN_CACHE,
            "--testing",
            "--unique_name", "smoke_edge_dplus_neighbor_positives",
        ]
    )

    device = torch.device(smoke_args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    set_seed(int(args.seed))
    with open("data_config.json", encoding="utf-8") as f:
        data_config = json.load(f)

    t_load = time.perf_counter()
    tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(args, data_config)
    del val_data, te_data, val_inds, te_inds
    add_arange_ids([tr_data])
    logging.info("Loaded data in %.1fs", time.perf_counter() - t_load)

    ctx = build_edge_neighbor_positive_context(
        tr_data,
        knn_cache_path=KNN_CACHE,
        max_total_edges=int(args.edge_neighbor_max_total),
        positive_mode="neighbor",
        train_labels=tr_data[FORWARD_EDGE_TYPE].y.detach().cpu().numpy(),
    )
    stream = np.arange(ctx.n_train, dtype=np.int64)
    anchors, seeds, _, exp_stats = expand_poscomplete_seeds(stream, ctx, start=0)
    if anchors.size == 0:
        raise SystemExit("FAIL: no anchors in first poscomplete batch")

    transform = AddEgoIds() if args.ego else None
    t0 = time.perf_counter()
    batch, seed_edge_ids = _batch_for_seed_positions(tr_data, seeds, args, transform)
    # get_model expects the synthetic ID column still present (uses shape[1]-1).
    config = SimpleNamespace(
        epochs=1,
        n_hidden=66.00389566544462,
        n_gnn_layers=2,
        dropout=0.027644401322222123,
        final_dropout=0.0,
        emb_size=64,
        embedding_dim=128,
        w_ce1=1.0,
        w_ce2=1.0,
        lr=0.006213266113989207,
    )
    model = get_model(batch, config, args)
    model = to_hetero(model, tr_data.metadata(), aggr="mean").to(device)
    proj = setup_contrastive_projection(args, device, embedding_dim=128)
    args.contrast_projection_module = proj

    attach_edge_id_from_batch(batch, tr_data)
    batch = batch.to(device)
    seed_edge_ids = seed_edge_ids.to(device)
    edge_drop_stats = {}
    view1, view2 = generate_views(
        batch, **_contrastive_view_kwargs(args, edge_drop_stats, seed_edge_ids=seed_edge_ids)
    )
    out1 = model(view1.x_dict, view1.edge_index_dict, view1.edge_attr_dict)
    z1 = out1[FORWARD_EDGE_TYPE]
    with torch.no_grad():
        out2 = model(view2.x_dict, view2.edge_index_dict, view2.edge_attr_dict)
        z2 = out2[FORWARD_EDGE_TYPE]
    z1_seed, seed_id1, z2_seed, _ = select_shared_seed_edge_embeddings(
        z1, view1[FORWARD_EDGE_TYPE].edge_id, z2, view2[FORWARD_EDGE_TYPE].edge_id, seed_edge_ids
    )
    z1_seed, z2_seed = project_seed_pair(proj, z1_seed, z2_seed.detach())

    loss, stats = edge_neighbor_supcon_loss(
        z1_seed,
        z2_seed,
        seed_id1,
        ctx=ctx,
        anchor_ids=anchors.tolist(),
        temperature=0.5,
        asymmetric=True,
    )
    loss.backward()
    grad_ok = all(
        (p.grad is None) or bool(torch.isfinite(p.grad).all()) for p in model.parameters()
    )
    # Timed steady-state steps (exclude model construction / first-batch compile).
    timed = []
    cur = int(anchors.size)  # continue stream after first expand used start=0
    # Recompute cursor after first expand
    _a, _s, cur, _ = expand_poscomplete_seeds(stream, ctx, start=0)
    for _ in range(3):
        a2, s2, cur, _ = expand_poscomplete_seeds(stream, ctx, start=cur)
        if a2.size == 0:
            break
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_step = time.perf_counter()
        b2, sid2 = _batch_for_seed_positions(tr_data, s2, args, transform)
        attach_edge_id_from_batch(b2, tr_data)
        b2 = b2.to(device)
        sid2 = sid2.to(device)
        v1, v2 = generate_views(
            b2, **_contrastive_view_kwargs(args, {}, seed_edge_ids=sid2)
        )
        o1 = model(v1.x_dict, v1.edge_index_dict, v1.edge_attr_dict)
        z1b = o1[FORWARD_EDGE_TYPE]
        with torch.no_grad():
            o2 = model(v2.x_dict, v2.edge_index_dict, v2.edge_attr_dict)
            z2b = o2[FORWARD_EDGE_TYPE]
        z1s, ids, z2s, _ = select_shared_seed_edge_embeddings(
            z1b, v1[FORWARD_EDGE_TYPE].edge_id, z2b, v2[FORWARD_EDGE_TYPE].edge_id, sid2
        )
        z1s, z2s = project_seed_pair(proj, z1s, z2s.detach())
        loss2, _ = edge_neighbor_supcon_loss(
            z1s, z2s, ids, ctx=ctx, anchor_ids=a2.tolist(), temperature=0.5, asymmetric=True
        )
        loss2.backward()
        model.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        timed.append(time.perf_counter() - t_step)
        del b2, v1, v2, o1, z1b, z2b

    step_s = float(np.median(timed)) if timed else (time.perf_counter() - t0)
    peak_alloc = float(torch.cuda.max_memory_allocated() / (1024**2)) if device.type == "cuda" else 0.0
    peak_reserved = (
        float(torch.cuda.max_memory_reserved() / (1024**2)) if device.type == "cuda" else 0.0
    )

    # D+-matched microbatch budget (not full-stream coverage).
    batches_per_epoch = int(np.ceil(ctx.n_train / float(args.batch_size)))
    full_stream_batches = int(np.ceil(ctx.n_train / max(float(anchors.size), 1.0)))
    sec_per_epoch = batches_per_epoch * step_s
    proj_10h = sec_per_epoch * 10 / 3600.0
    proj_40h = sec_per_epoch * 40 / 3600.0
    envelope_ok = proj_10h < 5.5

    finite_loss = bool(torch.isfinite(loss).item())
    positives_ok = float(stats.get("n_pos_total_mean", 0)) > 1.01
    passed = (
        finite_loss
        and grad_ok
        and positives_ok
        and int(stats["n_anchor_rows"]) > 0
        and peak_alloc < PEAK_GPU_MIB_LIMIT
        and envelope_ok
        and len(timed) >= 1
    )

    payload = {
        "title": "edge_dplus_neighbor_positive_smoke",
        "not_exact_gcpal_reproduction": True,
        "distinct_from_enable_knn_soft_positives": True,
        "passed": passed,
        "finite_loss": finite_loss,
        "finite_gradients": grad_ok,
        "loss": float(loss.detach().item()),
        "positive_stats": stats,
        "expand_stats": {
            k: float(v) if isinstance(v, (int, float, np.floating)) else v
            for k, v in exp_stats.items()
        },
        "requested_anchors": int(anchors.size),
        "realized_anchors": int(stats["n_anchor_rows"]),
        "retrieved_positives": int(max(seeds.size - anchors.size, 0)),
        "total_transaction_edges": int(seeds.size),
        "mp_edges": int(batch[FORWARD_EDGE_TYPE].edge_index.shape[1]),
        "peak_alloc_mib": peak_alloc,
        "peak_reserved_mib": peak_reserved,
        "step_seconds_first_incl_setup": time.perf_counter() - t0,  # rough
        "step_seconds_median_steady": step_s,
        "timed_steps": timed,
        "batches_per_epoch_dplus_matched": batches_per_epoch,
        "batches_per_epoch_full_stream": full_stream_batches,
        "epoch_coverage_note": (
            "Training uses D+-matched microbatch count per epoch "
            "(ceil(n_train/batch_size)); full positive-complete stream coverage "
            "is infeasible under the 6h envelope."
        ),
        "sec_per_epoch_estimate": sec_per_epoch,
        "approx_10ep_hours": proj_10h,
        "approx_40ep_hours": proj_40h,
        "six_hour_feasible_for_10ep": envelope_ok,
        "leakage_audit": ctx.leakage_audit,
        "reference_dplus_job": "18514684",
        "reference_fullstack_job": "18678029",
    }
    Path(smoke_args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(smoke_args.output_json).write_text(json.dumps(payload, indent=2) + "\n")
    md = "\n".join(
        [
            "# Edge D+ neighbor-positive smoke",
            "",
            f"**Passed:** {passed}",
            "",
            f"- loss finite: {finite_loss} ({payload['loss']:.4f})",
            f"- grads finite: {grad_ok}",
            f"- anchors requested/realized: {anchors.size}/{int(stats['n_anchor_rows'])}",
            f"- mean positives/anchor: {stats.get('n_pos_total_mean')}",
            f"- structural/knn additions: {stats.get('n_structural')}/{stats.get('n_knn')}",
            f"- peak alloc/reserved MiB: {peak_alloc:.1f}/{peak_reserved:.1f}",
            f"- steady step median: {step_s:.3f}s (n={len(timed)})",
            f"- batches/epoch (D+-matched): {batches_per_epoch}; full-stream would be ~{full_stream_batches}",
            f"- ~10ep hours @ matched budget: {proj_10h:.2f}; ~40ep hours: {proj_40h:.2f}",
            f"- 6h envelope OK for 10ep: {envelope_ok}",
            "",
            "NOT an exact GCPAL reproduction. Distinct from `--enable_knn_soft_positives`.",
            "",
        ]
    )
    Path(smoke_args.output_md).write_text(md + "\n")
    logging.info("Smoke passed=%s wrote %s", passed, smoke_args.output_json)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
