#!/usr/bin/env python3
"""Matched 5-epoch positive-complete scouts (A=identity control, B=GCPAL-style)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit

from gcpal_txn_node.adjacency import (
    adjacency_list_from_edge_index,
    build_directed_flow_adjacency,
    induce_edge_index,
)
from gcpal_txn_node.data import load_small_hi_frame
from gcpal_txn_node.eval_mlp import train_eval_mlp_suite
from gcpal_txn_node.features import fit_feature_preprocessor
from gcpal_txn_node.knn_adapter import load_train_knn_cache
from gcpal_txn_node.model import SharedTxnNodeEncoder
from gcpal_txn_node.spec import DEFAULT_KNN_CACHE, LAMBDA_MIX, NOT_EXACT_REPRODUCTION, TEMPERATURE
from gcpal_txn_node.loss import DEFAULT_POSITIVE_AGGREGATION, POSITIVE_AGGREGATION_MODES
from gcpal_txn_node.train_step import StepConfig, run_positive_complete_step


@torch.no_grad()
def encode_nodes_induced(
    encoder: SharedTxnNodeEncoder,
    x_all: np.ndarray,
    flow_ei: np.ndarray,
    node_ids: np.ndarray,
    device: torch.device,
    chunk: int = 4096,
) -> np.ndarray:
    encoder.eval()
    out = np.zeros((node_ids.shape[0], encoder.gin.out_dim), dtype=np.float32)
    for start in range(0, node_ids.shape[0], chunk):
        ids = node_ids[start : start + chunk]
        x = torch.from_numpy(x_all[ids]).to(device)
        ei = torch.from_numpy(induce_edge_index(flow_ei, ids)).to(device)
        h, _ = encoder(x, ei)
        out[start : start + chunk] = h.detach().cpu().numpy()
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["A_identity", "B_gcpal"], required=True)
    p.add_argument("--data_config", default="data_config.json")
    p.add_argument("--knn_cache", default=DEFAULT_KNN_CACHE)
    p.add_argument("--max_total_nodes", type=int, default=2048)
    p.add_argument("--n_epochs", type=int, default=5)
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output_json", required=True)
    p.add_argument("--output_md", required=True)
    p.add_argument(
        "--positive_aggregation",
        choices=list(POSITIVE_AGGREGATION_MODES),
        default=DEFAULT_POSITIVE_AGGREGATION,
        help="Positive aggregation (default sum_logsumexp preserves historical behavior).",
    )
    args = p.parse_args()
    assert NOT_EXACT_REPRODUCTION
    if int(args.n_epochs) > 5:
        raise SystemExit("Refusing n_epochs>5")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    df, tr, va, te, meta = load_small_hi_frame(args.data_config)
    df_train = df.iloc[tr].reset_index(drop=True).copy()
    n = len(df_train)
    allowed = set(range(n))
    labels = df_train[meta["label_col"]].to_numpy().astype(np.int64)
    prep = fit_feature_preprocessor(df_train, amount_col=meta["amount_col"])
    x_train = prep.transform(df_train)

    flow_ei, flow_stats = build_directed_flow_adjacency(
        df_train["from_id"].to_numpy(),
        df_train["to_id"].to_numpy(),
        df_train["Timestamp"].astype(float).to_numpy(),
        policy="immediate_next",
    )
    flow_adj = adjacency_list_from_edge_index(flow_ei, n)
    knn = load_train_knn_cache(args.knn_cache, expected_k=15)
    if knn.node_ids.shape[0] != n:
        raise SystemExit("KNN/train size mismatch")
    knn_adj = knn.adjacency_lists()
    empty_adj = [np.zeros(0, dtype=np.int64) for _ in range(n)]

    # Matched batching for A and B: identical growth. Mode only changes positives / KNN contrast.
    if args.mode == "A_identity":
        cfg = StepConfig(
            edge_drop=0.1,
            feature_drop=0.1,
            lambda_mix=1.0,
            temperature=TEMPERATURE,
            include_identity=True,
            use_knn_contrast=False,
            max_total_nodes=int(args.max_total_nodes),
            batching_mode="positive_complete",
            knn_k=15,
            positive_aggregation=str(args.positive_aggregation),
        )
        pos_flow, pos_knn = empty_adj, empty_adj
    else:
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
            positive_aggregation=str(args.positive_aggregation),
        )
        pos_flow, pos_knn = flow_adj, knn_adj

    encoder = SharedTxnNodeEncoder(in_dim=x_train.shape[1], emb_dim=128).to(device)
    opt = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    rng = np.random.RandomState(args.seed)
    history = []
    batch_diags = []
    t0 = time.perf_counter()
    h_train_acc = np.zeros((n, 128), dtype=np.float32)
    seen = np.zeros(n, dtype=bool)
    total_opt_steps = 0
    total_anchor_exposures = 0

    # Match legacy scout optimizer-step count so A/B fit the 6h Advanced GPU envelope.
    # Full positive-complete coverage of train would be ~ceil(n / n_anchors_realized) ≫ this.
    steps_per_epoch = int(np.ceil(n / 2048.0))
    logging.info(
        "Positive-complete scout: steps_per_epoch=%d (matched to legacy bs=2048); cap=%d",
        steps_per_epoch,
        int(args.max_total_nodes),
    )

    for epoch in range(int(args.n_epochs)):
        perm = rng.permutation(n).astype(np.int64)
        cursor = 0
        losses = []
        for step_i in range(steps_per_epoch):
            if cursor >= n:
                # Reshuffle remainder if the stream is exhausted early (should be rare).
                perm = rng.permutation(n).astype(np.int64)
                cursor = 0
            out = run_positive_complete_step(
                encoder=encoder,
                x_all=x_train,
                flow_edge_index=flow_ei,
                flow_adj=flow_adj,
                knn_adj=knn_adj,
                knn_graph_edge_fn=knn.edge_index_for_nodes,
                candidate_stream=perm,
                stream_start=cursor,
                device=device,
                cfg=cfg,
                seed=args.seed * 10007 + epoch * 97 + step_i,
                pos_flow_adj=pos_flow,
                pos_knn_adj=pos_knn,
                allowed_ids=allowed,
                labels=labels,
            )
            loss = out["loss"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_opt_steps += 1
            total_anchor_exposures += int(out["n_anchors"])
            losses.append(float(loss.detach().cpu()))
            if len(batch_diags) < 8:
                batch_diags.append(
                    {
                        "epoch": epoch + 1,
                        "realized_n_anchors": int(out["n_anchors"]),
                        "n_nodes": int(out["n_sampled_nodes"]),
                        "growth_stats": out["growth_stats"],
                        "positive_stats": out["positive_stats"],
                        "step_time_seconds": out["step_time_seconds"],
                        "peak_cuda_mib": out["peak_cuda_mib"],
                    }
                )
            cursor = int(out["stream_next"])
            if epoch == int(args.n_epochs) - 1:
                anchors = out["anchor_ids"]
                h_train_acc[anchors] = out["h_anchors"].cpu().numpy()
                seen[anchors] = True
        history.append({"epoch": epoch + 1, "loss_mean": float(np.mean(losses)), "n_steps": len(losses)})
        logging.info(
            "epoch %s loss=%.4f steps=%d",
            epoch + 1,
            history[-1]["loss_mean"],
            history[-1]["n_steps"],
        )

    train_s = time.perf_counter() - t0
    if not bool(seen.all()):
        missing = np.where(~seen)[0]
        h_train_acc[missing] = encode_nodes_induced(
            encoder, x_train, flow_ei, missing.astype(np.int64), device
        )

    flow_full, _ = build_directed_flow_adjacency(
        df["from_id"].to_numpy(),
        df["to_id"].to_numpy(),
        df["Timestamp"].astype(float).to_numpy(),
        policy="immediate_next",
    )
    x_full = np.zeros((len(df), x_train.shape[1]), dtype=np.float32)
    x_full[tr] = x_train
    x_full[va] = prep.transform(df.iloc[va])
    x_full[te] = prep.transform(df.iloc[te])
    y_all = df[meta["label_col"]].to_numpy().astype(np.int64)

    h_full = np.zeros((len(df), 128), dtype=np.float32)
    h_full[tr] = h_train_acc
    h_full[va] = encode_nodes_induced(encoder, x_full, flow_full, va.astype(np.int64), device)
    h_full[te] = encode_nodes_induced(encoder, x_full, flow_full, te.astype(np.int64), device)

    temporal = train_eval_mlp_suite(
        h_full[tr],
        x_full[tr],
        y_all[tr],
        h_full[te],
        x_full[te],
        y_all[te],
        h_val=h_full[va],
        x_val=x_full[va],
        y_val=y_all[va],
        seed=args.seed,
        device=device,
    )
    sss = StratifiedShuffleSplit(n_splits=1, train_size=0.4, random_state=args.seed)
    tr_r, te_r = next(sss.split(np.arange(len(df)), y_all))
    # Hold out 20% of the 40% train as val for threshold selection within diagnostic split.
    sss_inner = StratifiedShuffleSplit(n_splits=1, train_size=0.75, random_state=args.seed + 1)
    tr_r2, va_r = next(sss_inner.split(tr_r, y_all[tr_r]))
    tr_idx = tr_r[tr_r2]
    va_idx = tr_r[va_r]
    random40 = train_eval_mlp_suite(
        h_full[tr_idx],
        x_full[tr_idx],
        y_all[tr_idx],
        h_full[te_r],
        x_full[te_r],
        y_all[te_r],
        h_val=h_full[va_idx],
        x_val=x_full[va_idx],
        y_val=y_all[va_idx],
        seed=args.seed,
        device=device,
    )

    payload = {
        "not_exact_reproduction": True,
        "mode": args.mode,
        "positive_aggregation": str(args.positive_aggregation),
        "batching_mode": "positive_complete",
        "max_total_nodes": int(args.max_total_nodes),
        "n_epochs": int(args.n_epochs),
        "seed": args.seed,
        "history": history,
        "train_seconds": train_s,
        "steps_per_epoch": steps_per_epoch,
        "total_opt_steps": total_opt_steps,
        "total_anchor_exposures": total_anchor_exposures,
        "step_budget_note": (
            "Optimizer steps/epoch matched to legacy ceil(n_train/2048); "
            "batches remain positive-complete under the 2048-node cap."
        ),
        "batch_diagnostics_head": batch_diags,
        "flow_stats": {
            "n_nodes": flow_stats.n_nodes,
            "n_edges": flow_stats.n_edges,
            "policy": flow_stats.policy,
            "note": flow_stats.note,
        },
        "knn_deviations": knn.deviation_notes,
        "temporal_primary": temporal,
        "random40_diagnostic": random40,
        "eval_notes": [
            "Temporal protocol is primary; stratified random-40 is diagnostic only.",
            "Reports X / H / H||X MLP with fixed 0.5 and validation-selected thresholds.",
            "A and B share positive-complete batch construction (capacity 2048).",
            "Do not extend automatically beyond 5 epochs.",
        ],
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(payload, indent=2) + "\n")

    def _line(proto: str, block: dict) -> list[str]:
        lines = [f"### {proto}", ""]
        for rep in ("X", "H", "HxX"):
            lines.append(f"- **{rep}** @0.5: {json.dumps(block[rep]['threshold_0.5'])}")
            lines.append(
                f"- **{rep}** @val-thr: {json.dumps(block[rep]['threshold_val_selected'])}"
            )
        lines.append("")
        return lines

    md = [
        f"# Positive-complete txn-node scout (`{args.mode}`)",
        "",
        "**Not an exact GCPAL reproduction.**",
        "",
        f"- epochs={args.n_epochs} cap={args.max_total_nodes} seed={args.seed}",
        f"- opt_steps={total_opt_steps} anchor_exposures={total_anchor_exposures}",
        f"- train_seconds={train_s:.1f}",
        "",
        "## Temporal (primary)",
        "",
        *_line("temporal", temporal),
        "## Random-40 diagnostic",
        "",
        *_line("random40", random40),
    ]
    Path(args.output_md).write_text("\n".join(md))
    print(args.output_json)


if __name__ == "__main__":
    main()
