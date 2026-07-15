#!/usr/bin/env python3
"""GPU smoke test for width-aligned PNA SSL scout before Slurm submission."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_util import extract_param
from training import get_model

try:
    from torch_geometric.data import HeteroData
    from torch_geometric.nn import to_hetero
except ImportError as exc:
    raise SystemExit(f"torch_geometric required: {exc}")


def _synthetic_hetero_batch(num_nodes=256, num_edges=1024, edge_dim=8, num_features=2):
    data = HeteroData()
    data["node"].x = torch.randn(num_nodes, num_features)
    src = torch.randint(0, num_nodes, (num_edges,))
    dst = torch.randint(0, num_nodes, (num_edges,))
    data["node", "to", "node"].edge_index = torch.stack([src, dst], dim=0)
    data["node", "rev_to", "node"].edge_index = torch.stack([dst, src], dim=0)
    data["node", "to", "node"].edge_attr = torch.randn(num_edges, edge_dim)
    data["node", "rev_to", "node"].edge_attr = torch.randn(num_edges, edge_dim)
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    gin_args = SimpleNamespace(model="gin")
    pna_args = SimpleNamespace(
        model="pna",
        emlps=True,
        reverse_mp=True,
        override_n_hidden=66,
        override_lr=extract_param("lr", gin_args),
        override_dropout=extract_param("dropout", gin_args),
        override_final_dropout=extract_param("final_dropout", gin_args),
    )
    nh_req = int(round(float(extract_param("n_hidden", pna_args))))
    nh = int((nh_req // 5) * 5)
    config = SimpleNamespace(
        model="pna",
        n_hidden=nh,
        n_gnn_layers=int(extract_param("n_gnn_layers", pna_args)),
        n_heads=None,
        dropout=float(extract_param("dropout", pna_args)),
        final_dropout=float(extract_param("final_dropout", pna_args)),
    )

    batch = _synthetic_hetero_batch()
    metadata = batch.metadata()
    model = get_model(batch, config, pna_args)
    model = to_hetero(model, metadata, aggr="mean")
    model.to(args.device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"actual_n_hidden={nh}")
    print(f"pre_embedding_3h_dim={3 * nh}")
    print(f"post_embedding_dim=128")
    print(f"total_params={n_params}")

    # projection head (contrastive)
    proj = torch.nn.Sequential(
        torch.nn.Linear(128, 128), torch.nn.ReLU(), torch.nn.Linear(128, 128)
    ).to(args.device)
    opt = torch.optim.Adam(
        list(model.parameters()) + list(proj.parameters()),
        lr=float(extract_param("lr", pna_args)),
    )

    batch = batch.to(args.device)
    x_dict = batch.x_dict
    edge_index_dict = batch.edge_index_dict
    edge_attr_dict = {k: v[:, 1:] if v.shape[1] > 1 else v for k, v in batch.edge_attr_dict.items()}

    model.train()
    opt.zero_grad()
    z = model(x_dict, edge_index_dict, edge_attr_dict)[("node", "to", "node")]
    n = min(args.batch_size, z.shape[0])
    z = z[:n]
    z_proj = proj(z)
    loss = (z_proj * 0.01).sum()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()), "no encoder grad"
    assert any(p.grad is not None for p in proj.parameters()), "no projection grad"
    if args.device.startswith("cuda"):
        print(f"peak_gpu_mem_mb={torch.cuda.max_memory_allocated() / 1e6:.1f}")
    print("SMOKE_OK")


if __name__ == "__main__":
    main()
