#!/usr/bin/env python3
"""One-batch GPU smoke for positive-complete GCPAL-style txn-node batching."""

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
from gcpal_txn_node.train_step import StepConfig, run_positive_complete_step

# Established Advanced GPU envelope (A100-class ~40–80GiB); keep a large safety margin.
PEAK_GPU_MIB_LIMIT = 20000.0


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_config", default="data_config.json")
    p.add_argument("--knn_cache", default=DEFAULT_KNN_CACHE)
    p.add_argument("--max_total_nodes", type=int, default=2048)
    p.add_argument("--adjacency_policy", default="immediate_next")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output_json", default="results/diagnostics/gcpal_txn_node_poscomplete_smoke.json")
    p.add_argument("--output_md", default="notes/gcpal_txn_node_poscomplete_smoke.md")
    args = p.parse_args()

    assert NOT_EXACT_REPRODUCTION
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    df, tr, va, te, meta = load_small_hi_frame(args.data_config)
    df_train = df.iloc[tr].reset_index(drop=True).copy()
    n_train = len(df_train)
    allowed = set(range(n_train))
    labels = df_train[meta["label_col"]].to_numpy().astype(np.int64)
    logging.info("Loaded Small-HI train n=%d val=%d test=%d", n_train, len(va), len(te))

    prep = fit_feature_preprocessor(df_train, amount_col=meta["amount_col"])
    x_all = prep.transform(df_train)
    flow_ei, flow_stats = build_directed_flow_adjacency(
        df_train["from_id"].to_numpy(),
        df_train["to_id"].to_numpy(),
        df_train["Timestamp"].astype(float).to_numpy(),
        policy=args.adjacency_policy,
    )
    flow_adj = adjacency_list_from_edge_index(flow_ei, n_train)
    knn = load_train_knn_cache(args.knn_cache, expected_k=15)
    if knn.node_ids.shape[0] != n_train:
        raise SystemExit(f"KNN n={knn.node_ids.shape[0]} != train n={n_train}")
    knn_adj = knn.adjacency_lists()
    assumptions = DocumentedAssumptions(
        adjacency_policy=flow_stats.policy,
        knn_cache_feature_set=knn.feature_set,
        batching="positive_complete_capped_2048",
    )

    rng = np.random.RandomState(args.seed)
    stream = rng.permutation(n_train).astype(np.int64)
    encoder = SharedTxnNodeEncoder(in_dim=x_all.shape[1], emb_dim=128).to(device)
    opt = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    cfg = StepConfig(
        edge_drop=0.1,
        feature_drop=0.1,
        lambda_mix=LAMBDA_MIX,
        temperature=TEMPERATURE,
        include_identity=True,
        use_knn_contrast=True,
        max_total_nodes=int(args.max_total_nodes),
        batching_mode="positive_complete",
        knn_k=15,
    )

    out = run_positive_complete_step(
        encoder=encoder,
        x_all=x_all,
        flow_edge_index=flow_ei,
        flow_adj=flow_adj,
        knn_adj=knn_adj,
        knn_graph_edge_fn=knn.edge_index_for_nodes,
        candidate_stream=stream,
        stream_start=0,
        device=device,
        cfg=cfg,
        seed=args.seed,
        allowed_ids=allowed,
        labels=labels,
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

    g = out["growth_stats"]
    pos = out["positive_stats"]
    peak = out["peak_cuda_mib"]
    frac_ge1 = float(g["frac_anchors_knn_ge1"])
    frac_all = float(g["frac_anchors_knn_all_available"])
    finite_loss = bool(torch.isfinite(loss).item())
    no_leak = True  # sampler raises otherwise
    aligned = bool(out["views_aligned"])

    # Cap responsibility: by construction accepted anchors should have all available neighbors.
    cap_responsible_for_miss = False
    if frac_all < 0.90:
        cap_responsible_for_miss = bool(g.get("cap_stopped", 0.0) > 0.0)

    gate = {
        "frac_anchors_knn_ge1_ge_0.95": frac_ge1 >= 0.95,
        "frac_anchors_knn_all_available_ge_0.90_or_cap": frac_all >= 0.90 or cap_responsible_for_miss,
        "finite_loss": finite_loss,
        "finite_gradients": grad_ok,
        "peak_gpu_safe": peak is None or float(peak) < PEAK_GPU_MIB_LIMIT,
        "no_leakage": no_leak,
        "views_aligned": aligned,
        "n_nodes_le_cap": int(out["n_sampled_nodes"]) <= int(args.max_total_nodes),
    }
    gate["pass"] = all(gate.values())

    # Full-train coverage with ~124 anchors/batch is too many steps for the 6h envelope.
    # Scouts match the legacy optimizer-step count: ceil(n_train / 2048) batches/epoch.
    step_s = float(out["step_time_seconds"])
    n_anchors = max(int(out["n_anchors"]), 1)
    steps_full_epoch = int(np.ceil(n_train / float(n_anchors)))
    steps_matched_epoch = int(np.ceil(n_train / 2048.0))
    projected_5ep_h_full = step_s * steps_full_epoch * 5 / 3600.0
    projected_5ep_h = step_s * steps_matched_epoch * 5 / 3600.0
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
            "batching_mode": "positive_complete",
            "max_total_nodes": int(args.max_total_nodes),
            "requested_n_anchors_stream_tail": int(g["requested_n_anchors"]),
            "realized_n_anchors": int(out["n_anchors"]),
            "n_sampled_nodes": int(out["n_sampled_nodes"]),
            "growth_stats": g,
            "positive_stats": pos,
            "unique_negatives_mean": out["unique_negatives_mean"],
            "loss": float(loss.detach().cpu()),
            "loss_random_random": float(out["loss_random_random"].detach().cpu()),
            "loss_random_knn": float(out["loss_random_knn"].detach().cpu()),
            "lambda_mix": out["lambda_mix"],
            "finite_loss": finite_loss,
            "finite_gradients": grad_ok,
            "grad_norm_l2_sum": grad_norm,
            "step_runtime_seconds": step_s,
            "peak_gpu_mib": peak,
            "peak_gpu_mib_limit": PEAK_GPU_MIB_LIMIT,
            "steps_per_epoch_full_coverage_est": steps_full_epoch,
            "steps_per_epoch_matched_legacy_opt": steps_matched_epoch,
            "projected_5ep_hours_full_coverage": projected_5ep_h_full,
            "projected_5ep_hours": projected_5ep_h,
            "fits_advanced_gpu_6h_for_5ep": fits_6h,
            "total_setup_seconds": time.perf_counter() - t0,
            "gate": gate,
        },
        # Correctness gate is required; runtime uses matched step budget (not full coverage).
        "stop_for_scouts": not (gate["pass"] and fits_6h),
    }

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n")

    s = payload["smoke"]
    lines = [
        "# Positive-complete txn-node smoke (Small-HI)",
        "",
        "**Not an exact GCPAL reproduction.**",
        "",
        f"- Realized anchors: {s['realized_n_anchors']}",
        f"- Unique transactions: {s['n_sampled_nodes']} (cap {args.max_total_nodes})",
        f"- frac KNN≥1: {g['frac_anchors_knn_ge1']:.4f}",
        f"- frac all available KNN present: {g['frac_anchors_knn_all_available']:.4f}",
        f"- frac structural≥1: {g['frac_anchors_struct_ge1']:.4f}",
        f"- mean/median KNN pos: {g['mean_knn_pos']:.2f}/{g['median_knn_pos']:.2f}",
        f"- pos/neg mask density: {pos.get('positive_mask_density')}/{pos.get('negative_mask_density')}",
        f"- Loss: {s['loss']:.4f} (finite={s['finite_loss']}, grads={s['finite_gradients']})",
        f"- Step time: {s['step_runtime_seconds']:.2f}s; peak GPU MiB: {s['peak_gpu_mib']}",
        f"- Matched steps/epoch (legacy opt count): {s['steps_per_epoch_matched_legacy_opt']}",
        f"- Projected 5ep (matched) ≈ {s['projected_5ep_hours']:.2f}h; full-coverage would be ≈ {s['projected_5ep_hours_full_coverage']:.1f}h",
        f"- Fits 6h with matched steps? **{s['fits_advanced_gpu_6h_for_5ep']}**",
        f"- Gate pass? **{gate['pass']}** → {json.dumps(gate)}",
        "",
    ]
    if payload["stop_for_scouts"]:
        lines.append("**STOP:** smoke gate failed or runtime budget exceeded; do not submit scouts.")
    else:
        lines.append("Smoke OK to submit matched positive-complete 5-epoch scouts A/B.")
    out_md.write_text("\n".join(lines) + "\n")
    logging.info("Wrote %s", out_json)
    logging.info("Wrote %s", out_md)
    if payload["stop_for_scouts"]:
        raise SystemExit("Positive-complete smoke gate failed; refusing scout submit.")
    print(out_json)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
