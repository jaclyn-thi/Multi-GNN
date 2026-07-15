#!/usr/bin/env python3
"""Audit PNA degree histogram sources: first loader batch vs full graph vs extraction loader.

Inherited from upstream IBM Multi-GNN (minibatch ``deg`` in ``get_model``). Classify differences
as comparability concerns unless they materially change model initialization.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from types import SimpleNamespace

import torch
from torch_geometric.data import HeteroData
from torch_geometric.utils import degree

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_loading import get_data
from train_util import AddEgoIds, add_arange_ids, get_loaders
from training import get_model
from util import create_parser, logger_setup, set_seed

logging.basicConfig(level=logging.INFO, format="%(message)s")


def _hist_stats(deg: torch.Tensor) -> dict:
    deg = deg.detach().cpu().float()
    nonzero = int((deg > 0).sum().item())
    total = float(deg.sum().item())
    n = int(deg.numel())
    mean = float(deg.mean().item()) if n else float("nan")
    mx = float(deg.max().item()) if n else float("nan")
    return {
        "num_bins": n,
        "total_count": total,
        "nonzero_bins": nonzero,
        "max_degree": mx,
        "mean_degree": mean,
    }


def _hetero_degree_histogram(data: HeteroData) -> torch.Tensor:
    index = torch.cat(
        (data["node", "to", "node"].edge_index[1], data["node", "rev_to", "node"].edge_index[1]),
        0,
    )
    d = degree(index, dtype=torch.long)
    return torch.bincount(d, minlength=1)


def _deg_from_loader_batch(batch) -> torch.Tensor:
    if isinstance(batch, HeteroData):
        index = torch.cat(
            (
                batch["node", "to", "node"].edge_index[1],
                batch["node", "rev_to", "node"].edge_index[1],
            ),
            0,
        )
    else:
        index = batch.edge_index[1]
    d = degree(index, dtype=torch.long)
    return torch.bincount(d, minlength=1)


def _compare(a: torch.Tensor, b: torch.Tensor) -> dict:
    same_shape = tuple(a.shape) == tuple(b.shape)
    if not same_shape:
        return {"identical": False, "same_shape": False, "max_abs_diff": None}
    diff = (a.float() - b.float()).abs()
    return {
        "identical": bool(torch.all(diff == 0)),
        "same_shape": True,
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
    }


def audit_real_data(args, data_config) -> dict:
    tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(args, data_config)
    transform = AddEgoIds() if args.ego else None
    add_arange_ids([tr_data, val_data, te_data])

    tr_loader, val_loader, te_loader = get_loaders(
        tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform, args, train_shuffle=False
    )
    ext_loader, _, _ = get_loaders(
        tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform, args, train_shuffle=False
    )

    sample_batch = next(iter(tr_loader))
    config = SimpleNamespace(
        model="pna",
        n_hidden=20,
        n_gnn_layers=2,
        n_heads=None,
        dropout=0.0,
        final_dropout=0.1,
    )
    model_train = get_model(sample_batch, config, args)
    deg_train_init = None
    for conv in model_train.convs:
        if hasattr(conv, "deg"):
            deg_train_init = conv.deg.clone()

    sample_ext = next(iter(ext_loader))
    model_ext = get_model(sample_ext, config, args)
    deg_ext_init = model_ext.convs[0].deg.clone()

    full_deg = _hetero_degree_histogram(tr_data)
    batch_deg = _deg_from_loader_batch(sample_batch)

    return {
        "data": args.data,
        "full_train_graph": _hist_stats(full_deg),
        "first_train_minibatch": _hist_stats(batch_deg),
        "train_vs_full_batch": _compare(full_deg, batch_deg),
        "train_init_vs_extraction_init": _compare(deg_train_init, deg_ext_init),
        "classification": (
            "comparability_concern"
            if not _compare(full_deg, batch_deg)["identical"]
            else "minibatch_matches_full_train_graph"
        ),
        "note": (
            "Upstream and current code both compute deg from the first LinkNeighborLoader "
            "minibatch in get_model(), not the full graph. Differences vs the full-graph "
            "histogram are inherited behavior."
        ),
    }


def main() -> None:
    parser = create_parser()
    parser.add_argument("--output_md", default="notes/pna_degree_histogram_audit.md")
    parser.add_argument("--output_json", default="results/diagnostics/pna_degree_histogram_audit.json")
    args = parser.parse_args()
    if not args.model:
        args.model = "pna"
    if not args.data:
        parser.error("--data is required (e.g. Small-HI)")

    logger_setup()
    set_seed(args.seed)

    with open(ROOT / "data_config.json", encoding="utf-8") as f:
        data_config = json.load(f)
    payload = audit_real_data(args, data_config)
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md = [
        "# PNA degree histogram audit",
        "",
        f"**Dataset:** {payload['data']}",
        "",
        "## Summary",
        "",
        f"- Full train graph: `{payload['full_train_graph']}`",
        f"- First train minibatch: `{payload['first_train_minibatch']}`",
        f"- Train vs full batch identical: `{payload['train_vs_full_batch']['identical']}` (max abs diff {payload['train_vs_full_batch'].get('max_abs_diff')})",
        f"- Train init vs extraction init identical: `{payload['train_init_vs_extraction_init']['identical']}`",
        f"- Classification: **{payload['classification']}**",
        "",
        payload["note"],
        "",
        f"JSON: `{out_json}`",
    ]
    out_md = Path(args.output_md)
    out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {out_json} and {out_md}")


if __name__ == "__main__":
    main()
