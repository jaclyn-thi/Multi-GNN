#!/usr/bin/env python3
"""Stage 6: one-batch Small-HI smoke for the GCPAL-style txn-node baseline."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch

from gcpal_txn_node.adjacency import adjacency_list_from_edge_index, build_directed_flow_adjacency
from gcpal_txn_node.data import load_small_hi_frame
from gcpal_txn_node.features import fit_feature_preprocessor
from gcpal_txn_node.knn_adapter import load_train_knn_cache
from gcpal_txn_node.model import SharedTxnNodeEncoder
from gcpal_txn_node.spec import (
    DEFAULT_KNN_CACHE,
    DocumentedAssumptions,
    LAMBDA_MIX,
    NOT_EXACT_REPRODUCTION,
    TEMPERATURE,
)
from gcpal_txn_node.train_step import StepConfig, run_contrastive_step


def _peak_gpu_mib():
    if not torch.cuda.is_available():
        return None
    return float(torch.cuda.max_memory_allocated() / (1024**2))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_config", default="data_config.json")
    p.add_argument("--knn_cache", default=DEFAULT_KNN_CACHE)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--adjacency_policy", default="immediate_next")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max_train_rows", type=int, default=0, help="Optional cap for faster debug")
    p.add_argument("--output_json", default="results/diagnostics/gcpal_txn_node_smoke.json")
    p.add_argument("--output_md", default="notes/gcpal_txn_node_smoke.md")
    args = p.parse_args()

    assert NOT_EXACT_REPRODUCTION
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    df, tr, va, te, meta = load_small_hi_frame(args.data_config, max_rows=args.max_train_rows)
    # Train-local frame (aligns with KNN cache id space)
    df_train = df.iloc[tr].reset_index(drop=True).copy()
    logging.info("Loaded Small-HI train n=%d val=%d test=%d", len(df_train), len(va), len(te))

    prep = fit_feature_preprocessor(df_train, amount_col=meta["amount_col"])
    x_all = prep.transform(df_train)

    flow_ei, flow_stats = build_directed_flow_adjacency(
        df_train["from_id"].to_numpy(),
        df_train["to_id"].to_numpy(),
        df_train["Timestamp"].astype(float).to_numpy(),
        policy=args.adjacency_policy,
    )
    flow_adj = adjacency_list_from_edge_index(flow_ei, len(df_train))
    logging.info("Flow graph: %s", flow_stats)

    knn = load_train_knn_cache(args.knn_cache, expected_k=15)
    if knn.node_ids.shape[0] != len(df_train):
        raise SystemExit(
            f"KNN n={knn.node_ids.shape[0]} != train n={len(df_train)}; refusing mismatch"
        )
    knn_adj = knn.adjacency_lists()
    assumptions = DocumentedAssumptions(
        adjacency_policy=flow_stats.policy,
        knn_cache_feature_set=knn.feature_set,
    )

    rng = np.random.RandomState(args.seed)
    b = min(int(args.batch_size), len(df_train))
    anchors = rng.choice(len(df_train), size=b, replace=False).astype(np.int64)

    encoder = SharedTxnNodeEncoder(in_dim=x_all.shape[1], emb_dim=128).to(device)
    opt = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    cfg = StepConfig(
        edge_drop=0.1,
        feature_drop=0.1,
        lambda_mix=LAMBDA_MIX,
        temperature=TEMPERATURE,
        include_identity=True,
        use_knn_contrast=True,
    )

    t_step = time.perf_counter()
    out = run_contrastive_step(
        encoder=encoder,
        x_all=x_all,
        flow_edge_index=flow_ei,
        flow_adj=flow_adj,
        knn_adj=knn_adj,
        knn_graph_edge_fn=knn.edge_index_for_nodes,
        anchor_ids=anchors,
        device=device,
        cfg=cfg,
        seed=args.seed,
    )
    loss = out["loss"]
    opt.zero_grad(set_to_none=True)
    loss.backward()
    grad_ok = True
    grad_norm = 0.0
    for p_ in encoder.parameters():
        if p_.grad is None:
            continue
        if not torch.isfinite(p_.grad).all():
            grad_ok = False
        grad_norm += float(p_.grad.detach().norm().cpu())
    opt.step()
    step_s = time.perf_counter() - t_step

    n_batches_per_epoch = int(np.ceil(len(df_train) / float(b)))
    projected_epoch_s = step_s * n_batches_per_epoch
    projected_5ep_h = projected_epoch_s * 5 / 3600.0
    fits_6h = projected_5ep_h < 6.0

    payload = {
        "not_exact_reproduction": True,
        "assumptions": assumptions.to_dict(),
        "knn_deviation_notes": knn.deviation_notes,
        "knn_meta": knn.meta,
        "flow_graph_stats": {
            "n_nodes": flow_stats.n_nodes,
            "n_edges": flow_stats.n_edges,
            "policy": flow_stats.policy,
            "mean_out_degree": flow_stats.mean_out_degree,
            "max_out_degree": flow_stats.max_out_degree,
            "fraction_nodes_with_out_edge": flow_stats.fraction_nodes_with_out_edge,
            "note": flow_stats.note,
        },
        "smoke": {
            "device": str(device),
            "n_anchors": out["n_anchors"],
            "n_sampled_nodes": out["n_sampled_nodes"],
            "n_edges_view1": out["n_edges_view1"],
            "n_edges_view2": out["n_edges_view2"],
            "n_edges_knn": out["n_edges_knn"],
            "positive_stats": out["positive_stats"],
            "growth_stats": out["growth_stats"],
            "unique_negatives_mean": out["unique_negatives_mean"],
            "loss": float(loss.detach().cpu()),
            "loss_random_random": float(out["loss_random_random"].detach().cpu()),
            "loss_random_knn": float(out["loss_random_knn"].detach().cpu()),
            "lambda_mix": out["lambda_mix"],
            "finite_loss": bool(torch.isfinite(loss).item()),
            "finite_gradients": grad_ok,
            "grad_norm_l2_sum": grad_norm,
            "step_runtime_seconds": step_s,
            "peak_gpu_mib": _peak_gpu_mib(),
            "n_batches_per_epoch": n_batches_per_epoch,
            "projected_epoch_seconds": projected_epoch_s,
            "projected_5ep_hours": projected_5ep_h,
            "fits_advanced_gpu_6h_for_5ep": fits_6h,
            "total_setup_seconds": time.perf_counter() - t0,
        },
        "feature_dim": int(x_all.shape[1]),
        "feature_names_head": prep.feature_names[:8],
        "data_meta": meta,
        "stop_for_full_training": not fits_6h,
    }

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n")

    s = payload["smoke"]
    lines = [
        "# GCPAL-style txn-node one-batch smoke (Small-HI)",
        "",
        "**Not an exact GCPAL reproduction.**",
        "",
        f"- Anchors: {s['n_anchors']}",
        f"- Sampled nodes: {s['n_sampled_nodes']}",
        f"- Edges view1/view2/knn: {s['n_edges_view1']}/{s['n_edges_view2']}/{s['n_edges_knn']}",
        f"- Positives/anchor: {s['positive_stats']}",
        f"- Unique negatives (mean): {s['unique_negatives_mean']:.2f}",
        f"- Loss total / rr / r-knn: {s['loss']:.4f} / {s['loss_random_random']:.4f} / {s['loss_random_knn']:.4f}",
        f"- Finite grads: {s['finite_gradients']}",
        f"- Step runtime: {s['step_runtime_seconds']:.2f}s",
        f"- Peak GPU MiB: {s['peak_gpu_mib']}",
        f"- Projected epoch: {s['projected_epoch_seconds']:.1f}s; 5ep ≈ {s['projected_5ep_hours']:.2f}h",
        f"- Fits 6h Advanced GPU for 5ep? **{s['fits_advanced_gpu_6h_for_5ep']}**",
        "",
        "## Flow graph assumption",
        "",
        f"```json\n{json.dumps(payload['flow_graph_stats'], indent=2)}\n```",
        "",
        "## KNN deviations",
        "",
        f"```json\n{json.dumps(payload['knn_deviation_notes'], indent=2)}\n```",
        "",
    ]
    if payload["stop_for_full_training"]:
        lines.append("**STOP:** projected full training does not fit the 6h limit; do not submit scouts.")
    else:
        lines.append("Smoke OK to proceed to Stage 7 five-epoch matched scouts.")
    out_md.write_text("\n".join(lines))
    logging.info("Wrote %s", out_json)
    logging.info("Wrote %s", out_md)
    if payload["stop_for_full_training"]:
        raise SystemExit("Projected runtime exceeds 6h budget; stopping before scout submit.")
    print(out_json)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
