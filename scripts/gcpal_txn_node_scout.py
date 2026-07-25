#!/usr/bin/env python3
"""Stage 7: tiny matched five-epoch scouts (control vs GCPAL-style)."""

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
from gcpal_txn_node.eval_mlp import train_eval_mlp
from gcpal_txn_node.features import fit_feature_preprocessor
from gcpal_txn_node.knn_adapter import load_train_knn_cache
from gcpal_txn_node.model import SharedTxnNodeEncoder
from gcpal_txn_node.spec import DEFAULT_KNN_CACHE, LAMBDA_MIX, NOT_EXACT_REPRODUCTION, TEMPERATURE
from gcpal_txn_node.train_step import StepConfig, run_contrastive_step


@torch.no_grad()
def encode_nodes_induced(
    encoder: SharedTxnNodeEncoder,
    x_all: np.ndarray,
    flow_ei: np.ndarray,
    node_ids: np.ndarray,
    device: torch.device,
    chunk: int = 4096,
) -> np.ndarray:
    """Encode nodes via induced flow subgraphs in chunks (no full-graph materialization)."""
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
    p.add_argument("--mode", choices=["control", "gcpal"], required=True)
    p.add_argument("--data_config", default="data_config.json")
    p.add_argument("--knn_cache", default=DEFAULT_KNN_CACHE)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--n_epochs", type=int, default=5)
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output_json", required=True)
    p.add_argument("--output_md", required=True)
    args = p.parse_args()
    assert NOT_EXACT_REPRODUCTION
    if int(args.n_epochs) > 5:
        raise SystemExit("Refusing n_epochs>5")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    df, tr, va, te, meta = load_small_hi_frame(args.data_config)
    df_train = df.iloc[tr].reset_index(drop=True).copy()
    prep = fit_feature_preprocessor(df_train, amount_col=meta["amount_col"])
    x_train = prep.transform(df_train)

    flow_ei, flow_stats = build_directed_flow_adjacency(
        df_train["from_id"].to_numpy(),
        df_train["to_id"].to_numpy(),
        df_train["Timestamp"].astype(float).to_numpy(),
        policy="immediate_next",
    )
    flow_adj = adjacency_list_from_edge_index(flow_ei, len(df_train))
    knn = load_train_knn_cache(args.knn_cache, expected_k=15)
    if knn.node_ids.shape[0] != len(df_train):
        raise SystemExit("KNN/train size mismatch")
    knn_adj = knn.adjacency_lists()
    empty_adj = [np.zeros(0, dtype=np.int64) for _ in range(len(df_train))]

    if args.mode == "control":
        cfg = StepConfig(
            edge_drop=0.1,
            feature_drop=0.1,
            lambda_mix=1.0,
            temperature=TEMPERATURE,
            include_identity=True,
            use_knn_contrast=False,
            max_struct_extras=0,
            max_knn_extras=0,
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
        )
        pos_flow, pos_knn = flow_adj, knn_adj

    encoder = SharedTxnNodeEncoder(in_dim=x_train.shape[1], emb_dim=128).to(device)
    opt = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    n = len(df_train)
    bs = min(int(args.batch_size), n)
    rng = np.random.RandomState(args.seed)
    history = []
    t0 = time.perf_counter()
    h_train_acc = np.zeros((n, 128), dtype=np.float32)
    seen = np.zeros(n, dtype=bool)

    for epoch in range(int(args.n_epochs)):
        perm = rng.permutation(n)
        losses = []
        for start in range(0, n, bs):
            anchors = perm[start : start + bs].astype(np.int64)
            out = run_contrastive_step(
                encoder=encoder,
                x_all=x_train,
                flow_edge_index=flow_ei,
                flow_adj=flow_adj,
                knn_adj=knn_adj,
                knn_graph_edge_fn=knn.edge_index_for_nodes,
                anchor_ids=anchors,
                device=device,
                cfg=cfg,
                seed=args.seed * 10007 + epoch * 97 + start,
                pos_flow_adj=pos_flow,
                pos_knn_adj=pos_knn,
            )
            loss = out["loss"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
            if epoch == int(args.n_epochs) - 1:
                h_train_acc[anchors] = out["h_anchors"].cpu().numpy()
                seen[anchors] = True
        history.append({"epoch": epoch + 1, "loss_mean": float(np.mean(losses))})
        logging.info("epoch %s loss=%.4f", epoch + 1, history[-1]["loss_mean"])

    train_s = time.perf_counter() - t0
    if not bool(seen.all()):
        missing = np.where(~seen)[0]
        h_train_acc[missing] = encode_nodes_induced(
            encoder, x_train, flow_ei, missing.astype(np.int64), device
        )

    # Temporal eval: train embeddings from last epoch; val/test via induced full-df graph chunks
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

    temporal_metrics = train_eval_mlp(
        h_full[tr], x_full[tr], y_all[tr], h_full[te], x_full[te], y_all[te], seed=args.seed, device=device
    )
    sss = StratifiedShuffleSplit(n_splits=1, train_size=0.4, random_state=args.seed)
    tr_r, te_r = next(sss.split(np.arange(len(df)), y_all))
    random_metrics = train_eval_mlp(
        h_full[tr_r],
        x_full[tr_r],
        y_all[tr_r],
        h_full[te_r],
        x_full[te_r],
        y_all[te_r],
        seed=args.seed,
        device=device,
    )

    payload = {
        "not_exact_reproduction": True,
        "mode": args.mode,
        "n_epochs": int(args.n_epochs),
        "batch_size": bs,
        "seed": args.seed,
        "history": history,
        "train_seconds": train_s,
        "flow_stats": {
            "n_nodes": flow_stats.n_nodes,
            "n_edges": flow_stats.n_edges,
            "policy": flow_stats.policy,
            "note": flow_stats.note,
        },
        "knn_deviations": knn.deviation_notes,
        "temporal_mlp_HxX_fixed0.5": temporal_metrics,
        "random40_mlp_HxX_fixed0.5_diagnostic": random_metrics,
        "eval_notes": [
            "Temporal protocol is primary; random 40/60 is diagnostic only.",
            "Train H taken from last-epoch contrastive batches; val/test via induced subgraph chunks.",
            "Feature preprocessor fit on temporal train only.",
        ],
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(payload, indent=2) + "\n")
    Path(args.output_md).write_text(
        "\n".join(
            [
                f"# GCPAL-style txn-node scout (`{args.mode}`)",
                "",
                "**Not an exact GCPAL reproduction.**",
                "",
                f"- epochs={args.n_epochs} batch={bs} seed={args.seed}",
                f"- temporal MLP H||X @0.5: {json.dumps(temporal_metrics)}",
                f"- random40 diagnostic: {json.dumps(random_metrics)}",
                f"- train_seconds={train_s:.1f}",
                "",
            ]
        )
    )
    print(args.output_json)


if __name__ == "__main__":
    main()
