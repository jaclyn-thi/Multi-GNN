#!/usr/bin/env python3
"""GPU smoke: one contrastive batch/step with corrected reverse + preserve_seed_edges."""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import to_hetero

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data_loading import get_data
from graph_augmentations import generate_views
from models import GINe
from train_util import (
    AddEgoIds,
    FORWARD_EDGE_TYPE,
    REVERSE_EDGE_TYPE,
    add_arange_ids,
    attach_edge_id_from_batch,
    extract_param,
    get_hetero_seed_edge_ids,
    get_loaders,
    select_shared_seed_edge_embeddings,
)
from util import create_parser, logger_setup, set_seed


def main() -> None:
    logger_setup()
    parser = create_parser()
    args = parser.parse_args(
        [
            "--data",
            "Small-HI",
            "--model",
            "gin",
            "--objective",
            "contrastive",
            "--reverse_mp",
            "--ego",
            "--ports",
            "--emlps",
            "--tds",
            "--correct_reverse_edge_features",
            "--preserve_seed_edges",
            "--batch_size",
            "2048",
            "--num_neighs",
            "100",
            "100",
            "--loader_num_workers",
            "0",
            "--seed",
            "2",
            "--unique_name",
            "smoke_correct_reverse_preserve_seed",
            "--testing",
        ]
    )
    with open("data_config.json", "r", encoding="utf-8") as f:
        data_config = json.load(f)

    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info("device=%s", device)

    tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(args, data_config)
    assert isinstance(tr_data, HeteroData)
    assert getattr(tr_data, "reverse_edge_feature_semantics", None) == "corrected"
    schema = getattr(tr_data, "edge_feature_schema", {})
    fwd_dim = int(tr_data[FORWARD_EDGE_TYPE].edge_attr.shape[1])
    assert fwd_dim == 8, f"expected edge_dim=8 before IDs, got {fwd_dim}"
    assert schema.get("indices", {}).get("in_port") == 4
    assert schema.get("indices", {}).get("in_td") == 6
    assert (
        tr_data[FORWARD_EDGE_TYPE].edge_attr.untyped_storage().data_ptr()
        != tr_data[REVERSE_EDGE_TYPE].edge_attr.untyped_storage().data_ptr()
    )

    transform = AddEgoIds()
    add_arange_ids([tr_data, val_data, te_data])
    tr_loader, _, _ = get_loaders(
        tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform, args
    )

    sample = next(iter(tr_loader))
    n_feats = sample["node"].x.shape[1]
    # After add_arange_ids, model edge_dim excludes the prepended id column.
    e_dim = sample[FORWARD_EDGE_TYPE].edge_attr.shape[1] - 1
    assert e_dim == 8, f"model edge_dim expected 8, got {e_dim}"

    config_n_hidden = int(round(float(extract_param("n_hidden", args))))
    config_layers = int(round(float(extract_param("n_gnn_layers", args))))
    dropout = float(extract_param("dropout", args))
    model = GINe(
        num_features=n_feats,
        num_gnn_layers=config_layers,
        n_classes=2,
        n_hidden=config_n_hidden,
        edge_updates=True,
        edge_dim=e_dim,
        dropout=dropout,
        final_dropout=float(extract_param("final_dropout", args)),
    )
    model = to_hetero(model, te_data.metadata(), aggr="mean").to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(extract_param("lr", args)))

    batch = next(iter(tr_loader))
    seed_ids = get_hetero_seed_edge_ids(batch, tr_data)
    attach_edge_id_from_batch(batch, tr_data)
    batch = batch.to(device)
    seed_ids = seed_ids.to(device)
    requested = int(seed_ids.numel())

    # After attach_edge_id_from_batch, synthetic ID column is stripped; model edge_dim=8.
    assert batch[FORWARD_EDGE_TYPE].edge_attr.shape[1] == 8
    assert batch[REVERSE_EDGE_TYPE].edge_attr.shape[1] == 8

    t0 = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    view_kwargs = {
        "edge_attr_mask_rate": 0.1,
        "edge_drop_rate": 0.1,
        "mask_value": 0.0,
        "exclude_last_column": False,
        "preserve_seed_edges": True,
        "seed_edge_ids": seed_ids.detach().cpu(),
    }
    v1, v2 = generate_views(batch, **view_kwargs)

    model.train()
    opt.zero_grad(set_to_none=True)
    out1 = model(v1.x_dict, v1.edge_index_dict, v1.edge_attr_dict)[FORWARD_EDGE_TYPE]
    out2 = model(v2.x_dict, v2.edge_index_dict, v2.edge_attr_dict)[FORWARD_EDGE_TYPE]
    z1, id1, z2, id2 = select_shared_seed_edge_embeddings(
        out1,
        v1[FORWARD_EDGE_TYPE].edge_id,
        out2,
        v2[FORWARD_EDGE_TYPE].edge_id,
        seed_ids,
    )
    shared = int(id1.numel())
    # Dense-all-edges guard: contrastive anchors are seed-intersection sized, not MP-edge sized.
    mp_edges = int(v1[FORWARD_EDGE_TYPE].edge_index.shape[1])
    assert shared <= requested
    assert shared < mp_edges or mp_edges < 1000  # smoke batches have many context edges
    loss = -(torch.nn.functional.cosine_similarity(z1, z2, dim=-1).mean())
    assert torch.isfinite(loss).item()
    loss.backward()
    grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += float(p.grad.detach().float().norm().item() ** 2)
    grad_norm = grad_norm ** 0.5
    assert grad_norm > 0.0 and (grad_norm == grad_norm)
    opt.step()
    elapsed = time.perf_counter() - t0
    peak_mb = (
        torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else float("nan")
    )

    payload = {
        "ok": True,
        "device": str(device),
        "edge_dim_model": e_dim,
        "edge_feature_schema": schema,
        "reverse_edge_feature_semantics": "corrected",
        "correct_reverse_edge_features": True,
        "preserve_seed_edges": True,
        "requested_seed_edges": requested,
        "shared_seed_edges": shared,
        "shared_frac": shared / max(requested, 1),
        "forward_mp_edges_view1": mp_edges,
        "loss": float(loss.detach().cpu()),
        "grad_norm": grad_norm,
        "elapsed_s": elapsed,
        "peak_cuda_mem_mb": peak_mb,
        "no_alias_forward_reverse": True,
    }
    out = Path("results/diagnostics/smoke_correct_reverse_preserve_seed.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    logging.info("SMOKE_OK %s", json.dumps(payload))
    print(out)


if __name__ == "__main__":
    main()
