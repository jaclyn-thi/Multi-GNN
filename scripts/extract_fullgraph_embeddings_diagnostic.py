#!/usr/bin/env python3
"""Full-graph embedding extraction for random_transductive_diagnostic protocols.

Uses the saved GNN checkpoint under --unique_name. Seeds = all forward edges on the
full (test) graph so every transaction is encoded under one consistent graph context.
Does not retrain.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import to_hetero

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data_loading import get_data
from train_util import (
    AddEgoIds,
    FORWARD_EDGE_TYPE,
    add_arange_ids,
    checkpoint_path,
    expected_seed_edge_ids,
    extract_param,
    extract_seed_embeddings_hetero,
    get_loaders,
    infer_pre_embedding_dim,
    load_checkpoint_weights,
    log_seed_coverage,
    resolve_embedding_head_linear,
    save_embedding_split_npz,
)
from training import get_model
from util import create_parser, logger_setup, set_seed


class _MaxBatchLoader:
    """Thin wrapper that preserves ``.data`` but yields at most ``max_batches`` batches."""

    def __init__(self, loader, max_batches: int):
        self._loader = loader
        self._max_batches = int(max_batches)
        self.data = loader.data

    def __iter__(self):
        for i, batch in enumerate(self._loader):
            if i >= self._max_batches:
                break
            yield batch


def main() -> None:
    logger_setup()
    parser = create_parser()
    parser.add_argument("--output_unique_name", required=True)
    parser.add_argument(
        "--representation_source",
        default="post_embedding",
        choices=["post_embedding", "pre_embedding_3h"],
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--max_batches",
        type=int,
        default=0,
        help="If >0, stop extraction after this many loader batches (preflight only).",
    )
    # Force required graph flags for D-compatible extraction if caller uses create_parser only.
    args, _unknown = parser.parse_known_args()
    # Re-parse with explicit defaults for this diagnostic.
    if not args.unique_name:
        raise SystemExit("--unique_name (checkpoint) is required")

    # Ensure D-compatible flags when not passed (slurm passes them via embedding_extraction style).
    # This script is invoked with graph flags from the shell for D.
    with open("data_config.json", "r", encoding="utf-8") as f:
        data_config = json.load(f)

    set_seed(int(args.seed))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logging.info(
        "Full-graph diagnostic extraction: ckpt=%s out=%s source=%s device=%s "
        "correct_reverse=%s preserve_seed=%s ports=%s tds=%s",
        args.unique_name,
        args.output_unique_name,
        args.representation_source,
        device,
        getattr(args, "correct_reverse_edge_features", False),
        getattr(args, "preserve_seed_edges", False),
        getattr(args, "ports", False),
        getattr(args, "tds", False),
    )

    tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(args, data_config)
    assert isinstance(te_data, HeteroData)
    transform = AddEgoIds() if args.ego else None
    add_arange_ids([tr_data, val_data, te_data])

    n_edges = int(te_data[FORWARD_EDGE_TYPE].edge_index.shape[1])
    all_inds = torch.arange(n_edges, dtype=torch.long)
    logging.info("Full-graph seed edges: %d (label=random_transductive_diagnostic)", n_edges)

    # Loaders: reuse train/val for model sample batch; te loader seeds = all edges.
    tr_loader, val_loader, te_loader = get_loaders(
        tr_data,
        val_data,
        te_data,
        tr_inds,
        val_inds,
        all_inds,
        transform,
        args,
        train_shuffle=False,
    )

    sample_batch = next(iter(tr_loader))
    config = SimpleNamespace(
        model=args.model,
        n_hidden=extract_param("n_hidden", args),
        n_gnn_layers=extract_param("n_gnn_layers", args),
        n_heads=extract_param("n_heads", args) if args.model == "gat" else None,
        dropout=extract_param("dropout", args),
        final_dropout=extract_param("final_dropout", args),
    )
    model = get_model(sample_batch, config, args)
    # Match embedding_extraction.py: resolve head on homogeneous model before to_hetero.
    emb_dim = int(getattr(model, "embedding_dim", 128))
    actual_n_hidden = int(getattr(model, "n_hidden", round(float(config.n_hidden))))
    head_spec = None
    if args.representation_source == "pre_embedding_3h":
        head_spec = resolve_embedding_head_linear(model, emb_dim)
        pre_dim = head_spec.in_features
        logging.info(
            "Resolved pre_embedding_3h head: module=%s in_features=%d out_features=%d "
            "(model n_hidden=%d, requested n_hidden=%s)",
            head_spec.module_name,
            head_spec.in_features,
            head_spec.out_features,
            actual_n_hidden,
            config.n_hidden,
        )
    else:
        # post_embedding path does not need a live head; keep legacy 3*n_hidden for meta only.
        pre_dim = infer_pre_embedding_dim(model, emb_dim)
    if args.reverse_mp:
        model = to_hetero(model, te_data.metadata(), aggr="mean")
    load_checkpoint_weights(model, device, args, data_config)
    model.eval()

    extract_loader = te_loader
    if int(getattr(args, "max_batches", 0) or 0) > 0:
        logging.warning(
            "PREFLIGHT: limiting extraction to max_batches=%d (not a full-graph artifact)",
            int(args.max_batches),
        )
        extract_loader = _MaxBatchLoader(te_loader, int(args.max_batches))

    t0 = time.perf_counter()
    edge_ids, z, y = extract_seed_embeddings_hetero(
        extract_loader,
        all_inds,
        model,
        te_data,
        device,
        args,
        representation_source=args.representation_source,
        pre_dim=pre_dim,
        emb_dim=emb_dim,
        head_spec=head_spec,
    )
    logging.info(
        "Extracted full-graph embeddings in %.1fs: n=%d dim=%d",
        time.perf_counter() - t0,
        int(z.shape[0]),
        int(z.shape[1]),
    )
    expected = expected_seed_edge_ids(te_data, all_inds, hetero=True)

    out_root = Path(data_config["paths"].get("embeddings", "embeddings")) / args.output_unique_name
    if args.representation_source != "post_embedding":
        out_dir = out_root / args.representation_source
    else:
        out_dir = out_root
    out_dir.mkdir(parents=True, exist_ok=True)
    # Persist before coverage logging so a logging API mismatch cannot discard a finished extract.
    save_embedding_split_npz(out_dir / "all.npz", z, y, edge_ids)
    log_seed_coverage(edge_ids, expected, split_name="fullgraph_all")

    meta = {
        "label": "random_transductive_diagnostic",
        "note": (
            "All transaction embeddings inferred under one consistent full-graph context "
            "(te_data / all forward edges). Not a thesis-primary temporal protocol artifact."
        ),
        "source_unique_name": args.unique_name,
        "output_unique_name": args.output_unique_name,
        "representation_source": args.representation_source,
        "representation_dim": int(z.shape[1]),
        "n_rows": int(z.shape[0]),
        "n_positives": int(y.sum().item()) if hasattr(y, "sum") else int(np.asarray(y).sum()),
        "checkpoint_path": str(checkpoint_path(data_config, args.unique_name)),
        "correct_reverse_edge_features": bool(
            getattr(args, "correct_reverse_edge_features", False)
        ),
        "ports": bool(getattr(args, "ports", False)),
        "tds": bool(getattr(args, "tds", False)),
        "ego": bool(getattr(args, "ego", False)),
        "emlps": bool(getattr(args, "emlps", False)),
        "reverse_mp": bool(getattr(args, "reverse_mp", False)),
        "seed": int(args.seed),
        "max_batches": int(getattr(args, "max_batches", 0) or 0),
        "preflight_partial": bool(int(getattr(args, "max_batches", 0) or 0) > 0),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    # also root meta for post
    if args.representation_source == "post_embedding":
        (out_root / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(out_dir / "all.npz")


if __name__ == "__main__":
    main()
