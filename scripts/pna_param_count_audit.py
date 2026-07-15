#!/usr/bin/env python3
"""PNA width / parameter-count audit vs GIN reference (local, no training)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch_geometric.nn import to_hetero
from torch_geometric.utils import degree

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import GINe, PNA
from train_util import extract_param

GIN_UNIQUE = "hi_contrastive_gin_emlps_tds_proj_asym_8192neg_queue0_accum4_20ep"


def _count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def _synthetic_deg(n_hidden: int = 20) -> torch.Tensor:
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
    d = degree(edge_index[1], num_nodes=4, dtype=torch.long)
    return torch.bincount(d, minlength=1)


def _pna_row(requested_n_hidden: int, *, emlps: bool = True) -> dict:
    nh = int((requested_n_hidden // 5) * 5)
    deg = _synthetic_deg()
    model = PNA(
        num_features=2,
        num_gnn_layers=2,
        n_hidden=requested_n_hidden,
        edge_updates=emlps,
        edge_dim=8,
        deg=deg,
        embedding_dim=128,
        supervised_head="embedding",
    )
    # contrastive projection not in encoder model
    encoder_params = _count_params(model)
    emb_head = _count_params(model.embedding_head) + _count_params(model.classifier)
    return {
        "requested_n_hidden": requested_n_hidden,
        "actual_n_hidden": nh,
        "pre_embedding_3h_dim": 3 * nh,
        "post_embedding_dim": 128,
        "encoder_params": encoder_params,
        "embedding_head_plus_classifier_params": emb_head,
        "total_params": encoder_params,
    }


def _gin_reference() -> dict:
    args = SimpleNamespace(model="gin")
    lr = extract_param("lr", args)
    dropout = extract_param("dropout", args)
    final_dropout = extract_param("final_dropout", args)
    n_hidden = int(round(extract_param("n_hidden", args)))
    ckpt_path = ROOT / "saved-models" / f"checkpoint_{GIN_UNIQUE}.tar"
    ckpt_params = None
    if ckpt_path.is_file():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        ckpt_params = sum(v.numel() for v in ckpt["model_state_dict"].values())
    model = GINe(
        num_features=2, num_gnn_layers=2, n_hidden=n_hidden, edge_updates=True,
        edge_dim=8, embedding_dim=128, supervised_head="embedding",
    )
    return {
        "unique_name": GIN_UNIQUE,
        "n_hidden": n_hidden,
        "pre_embedding_3h_dim": 3 * n_hidden,
        "post_embedding_dim": 128,
        "lr": lr,
        "dropout": dropout,
        "final_dropout": final_dropout,
        "homogeneous_model_params": _count_params(model),
        "checkpoint_total_params": ckpt_params,
    }


def main() -> None:
    widths = [20, 30, 40, 50, 55, 60, 65, 66]
    rows = [_pna_row(w) for w in widths]
    gin = _gin_reference()
    gin_total = gin["checkpoint_total_params"] or gin["homogeneous_model_params"]
    closest = min(rows, key=lambda r: abs(r["total_params"] - gin_total))
    payload = {
        "gin_reference": gin,
        "pna_width_sweep": rows,
        "planned_scout": {
            "requested_n_hidden": 66,
            "expected_actual_n_hidden": 65,
            "pre_embedding_3h_dim": 195,
            "label": "width-matched to GIN pre-3h (198 vs 195); parameter-matched only if total params within ~10%",
        },
        "closest_param_count_among_valid_widths": closest,
        "gin_total_params_reference": gin_total,
    }
    out = ROOT / "results/diagnostics/pna_width_param_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
