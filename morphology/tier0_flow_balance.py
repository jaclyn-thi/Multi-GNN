"""Tier 0 flow balance: split-global amount aggregates and endpoint lift.

Computes per-account total amount received (in) and sent (out) on a forward
split graph, then lifts finance-motivated flow imbalance features to seed edges.

Convention (label-free, split-local only):

- ``amount_in[v]``  = sum of transaction amounts on edges with receiver ``v``.
- ``amount_out[v]`` = sum of transaction amounts on edges with sender ``v``.
- ``flow_balance_ratio`` = ``(out - in) / (out + in + eps)``, clipped to [-1, 1].
- Heavy-tailed amount scalars use ``log1p`` before expert loss (suffix ``_log``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch

from dataset_specs import DEFAULT_EDGE_FEATURE_COLS
from morphology.graph_access import get_forward_edge_attr, get_forward_edge_index, get_num_nodes
from morphology.tier0_global import load_node_table, save_node_table

FLOW_BALANCE_EPS = 1e-8
FLOW_BALANCE_RATIO_CLIP = 1.0

TIER0_FLOW_NODE_COLUMNS = ("amount_in", "amount_out")

FLOW_BALANCE_FEATURE_NAMES: List[str] = [
    "sender_in_amount_log",
    "sender_out_amount_log",
    "receiver_in_amount_log",
    "receiver_out_amount_log",
    "sender_flow_balance_ratio",
    "receiver_flow_balance_ratio",
    "sender_abs_flow_imbalance_log",
    "receiver_abs_flow_imbalance_log",
    "edge_to_sender_out_ratio_log",
    "edge_to_receiver_in_ratio_log",
]

FLOW_BALANCE_LIFT_DIM = len(FLOW_BALANCE_FEATURE_NAMES)


@dataclass
class MorphTier0FlowContext:
    """Split-global amount lookup tensors for flow-balance endpoint lift."""

    edge_index: torch.Tensor
    amount_in: torch.Tensor
    amount_out: torch.Tensor
    edge_amounts: torch.Tensor

    @property
    def num_nodes(self) -> int:
        return int(self.amount_in.shape[0])

    @property
    def device(self) -> torch.device:
        return self.edge_index.device


def amount_received_feature_index() -> int:
    """Column index of Amount Received in forward ``edge_attr`` **with** synthetic EdgeID."""

    if "Amount Received" not in DEFAULT_EDGE_FEATURE_COLS:
        raise KeyError("DEFAULT_EDGE_FEATURE_COLS missing Amount Received")
    return 1 + list(DEFAULT_EDGE_FEATURE_COLS).index("Amount Received")


def amount_received_from_graph(data) -> torch.Tensor:
    """Per-forward-edge Amount Received (non-negative float32), shape ``[E]``."""

    attr = get_forward_edge_attr(data)
    if attr is None or attr.shape[1] <= amount_received_feature_index():
        raise ValueError("Forward edge_attr missing Amount Received column")
    return attr[:, amount_received_feature_index()].float().clamp(min=0.0)


def compute_tier0_flow_node_stats(
    edge_index: torch.Tensor,
    edge_amounts: torch.Tensor,
    num_nodes: int,
) -> pd.DataFrame:
    """Aggregate split-global in/out amount totals per node."""

    if edge_index.numel() == 0:
        df = pd.DataFrame(
            {
                "amount_in": np.zeros(num_nodes, dtype=np.float64),
                "amount_out": np.zeros(num_nodes, dtype=np.float64),
            },
            index=np.arange(num_nodes, dtype=np.int64),
        )
        df.index.name = "node_id"
        return df

    ei = edge_index.cpu()
    amounts = edge_amounts.detach().cpu().float().view(-1)
    if amounts.shape[0] != ei.shape[1]:
        raise ValueError("edge_amounts length must match edge_index.num_edges")

    amount_out = torch.zeros(num_nodes, dtype=torch.float64)
    amount_in = torch.zeros(num_nodes, dtype=torch.float64)
    amount_out.scatter_add_(0, ei[0], amounts.double())
    amount_in.scatter_add_(0, ei[1], amounts.double())

    df = pd.DataFrame(
        {
            "amount_in": amount_in.numpy(),
            "amount_out": amount_out.numpy(),
        },
        index=np.arange(num_nodes, dtype=np.int64),
    )
    df.index.name = "node_id"
    return df


def _flow_balance_ratio(out_amount: torch.Tensor, in_amount: torch.Tensor) -> torch.Tensor:
    denom = out_amount + in_amount + FLOW_BALANCE_EPS
    ratio = (out_amount - in_amount) / denom
    return ratio.clamp(-FLOW_BALANCE_RATIO_CLIP, FLOW_BALANCE_RATIO_CLIP)


def _context_from_table_and_graph(
    edge_index: torch.Tensor,
    edge_amounts: torch.Tensor,
    node_table: pd.DataFrame,
    device: torch.device,
) -> MorphTier0FlowContext:
    for col in TIER0_FLOW_NODE_COLUMNS:
        if col not in node_table.columns:
            raise KeyError(f"node_table missing column {col!r}")
    amount_in = torch.as_tensor(node_table["amount_in"].to_numpy(), device=device, dtype=torch.float32)
    amount_out = torch.as_tensor(node_table["amount_out"].to_numpy(), device=device, dtype=torch.float32)
    return MorphTier0FlowContext(
        edge_index=edge_index,
        amount_in=amount_in,
        amount_out=amount_out,
        edge_amounts=edge_amounts.to(device=device, dtype=torch.float32),
    )


def tier0_flow_context_from_graph(split_data, device: torch.device) -> MorphTier0FlowContext:
    """Build flow-balance lookup from a split graph."""

    ei = get_forward_edge_index(split_data).to(device)
    edge_amounts = amount_received_from_graph(split_data).to(device)
    n_nodes = get_num_nodes(split_data)
    table = compute_tier0_flow_node_stats(ei.cpu(), edge_amounts.cpu(), n_nodes)
    return _context_from_table_and_graph(ei, edge_amounts, table, device)


def tier0_flow_context_from_cache(
    path: Union[str, Path],
    split_data,
    device: torch.device,
) -> MorphTier0FlowContext:
    """Load precomputed node flow table and pair with split forward graph amounts."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Tier 0 flow cache not found: {path}")
    table = load_node_table(path)
    n_nodes = get_num_nodes(split_data)
    if len(table) != n_nodes:
        raise ValueError(
            f"Tier 0 flow table rows ({len(table)}) != split num_nodes ({n_nodes}) for {path}"
        )
    ei = get_forward_edge_index(split_data).to(device)
    edge_amounts = amount_received_from_graph(split_data).to(device)
    return _context_from_table_and_graph(ei, edge_amounts, table, device)


def lift_flow_balance_to_seed_edges_torch(
    seed_edge_ids: torch.Tensor,
    flow_ctx: MorphTier0FlowContext,
) -> torch.Tensor:
    """
    Lift split-global flow-balance features to seed edges.

    Returns float32 tensor ``(n_seeds, len(FLOW_BALANCE_FEATURE_NAMES))``.
    """

    device = flow_ctx.device
    seed = seed_edge_ids.long().view(-1).to(device)
    ei = flow_ctx.edge_index
    if seed.numel() == 0:
        return torch.empty((0, FLOW_BALANCE_LIFT_DIM), device=device, dtype=torch.float32)
    if seed.max().item() >= ei.shape[1] or seed.min().item() < 0:
        raise ValueError(
            f"seed edge_ids out of range for split edge_index E={ei.shape[1]}: "
            f"min={seed.min().item()} max={seed.max().item()}"
        )

    senders = ei[0, seed]
    receivers = ei[1, seed]
    edge_amt = flow_ctx.edge_amounts[seed]

    s_in = flow_ctx.amount_in[senders]
    s_out = flow_ctx.amount_out[senders]
    r_in = flow_ctx.amount_in[receivers]
    r_out = flow_ctx.amount_out[receivers]

    s_ratio = _flow_balance_ratio(s_out, s_in)
    r_ratio = _flow_balance_ratio(r_out, r_in)

    return torch.stack(
        [
            torch.log1p(s_in.clamp(min=0.0)),
            torch.log1p(s_out.clamp(min=0.0)),
            torch.log1p(r_in.clamp(min=0.0)),
            torch.log1p(r_out.clamp(min=0.0)),
            s_ratio,
            r_ratio,
            torch.log1p(torch.abs(s_out - s_in).clamp(min=0.0)),
            torch.log1p(torch.abs(r_out - r_in).clamp(min=0.0)),
            torch.log1p(edge_amt / (s_out + FLOW_BALANCE_EPS)),
            torch.log1p(edge_amt / (r_in + FLOW_BALANCE_EPS)),
        ],
        dim=1,
    )


def lift_flow_balance_to_seed_edges_np(
    seed_edge_ids: Union[torch.Tensor, np.ndarray, Sequence[int]],
    edge_index: torch.Tensor,
    node_table: pd.DataFrame,
    edge_amounts: torch.Tensor,
) -> np.ndarray:
    """NumPy lift for tests and offline audits."""

    ctx = _context_from_table_and_graph(
        edge_index,
        edge_amounts,
        node_table,
        torch.device("cpu"),
    )
    seed = torch.as_tensor(seed_edge_ids, dtype=torch.long).view(-1)
    return lift_flow_balance_to_seed_edges_torch(seed, ctx).numpy()


def setup_morph_tier0_flow_contexts(args, tr_data, val_data, device: torch.device) -> None:
    """
    Attach ``args.morph_tier0_flow_train`` and ``args.morph_tier0_flow_val`` when
    ``--morph_flow_balance`` is enabled.
    """

    if not bool(getattr(args, "morph_flow_balance", False)):
        return

    cache_dir = getattr(args, "morph_tier0_flow_cache", None) or getattr(args, "morph_tier0_cache", None)
    if cache_dir:
        cache = Path(cache_dir)
        train_path = cache / "train_node_flow_balance.csv"
        val_path = cache / "val_node_flow_balance.csv"
        if train_path.is_file() and val_path.is_file():
            logging.info("Loading Tier 0 flow balance from cache dir %s", cache)
            args.morph_tier0_flow_train = tier0_flow_context_from_cache(train_path, tr_data, device)
            args.morph_tier0_flow_val = tier0_flow_context_from_cache(val_path, val_data, device)
        else:
            logging.info(
                "Flow-balance node cache missing under %s; computing from split graphs",
                cache,
            )
            args.morph_tier0_flow_train = tier0_flow_context_from_graph(tr_data, device)
            args.morph_tier0_flow_val = tier0_flow_context_from_graph(val_data, device)
    else:
        logging.info(
            "Computing Tier 0 flow balance from split graphs "
            "(set --morph_tier0_flow_cache to reuse tables)"
        )
        args.morph_tier0_flow_train = tier0_flow_context_from_graph(tr_data, device)
        args.morph_tier0_flow_val = tier0_flow_context_from_graph(val_data, device)

    logging.info(
        "Tier 0 flow train: nodes=%d edges=%d | val: nodes=%d edges=%d",
        args.morph_tier0_flow_train.num_nodes,
        int(args.morph_tier0_flow_train.edge_index.shape[1]),
        args.morph_tier0_flow_val.num_nodes,
        int(args.morph_tier0_flow_val.edge_index.shape[1]),
    )


__all__ = [
    "FLOW_BALANCE_EPS",
    "FLOW_BALANCE_FEATURE_NAMES",
    "FLOW_BALANCE_LIFT_DIM",
    "FLOW_BALANCE_RATIO_CLIP",
    "MorphTier0FlowContext",
    "TIER0_FLOW_NODE_COLUMNS",
    "amount_received_feature_index",
    "amount_received_from_graph",
    "compute_tier0_flow_node_stats",
    "lift_flow_balance_to_seed_edges_np",
    "lift_flow_balance_to_seed_edges_torch",
    "setup_morph_tier0_flow_contexts",
    "tier0_flow_context_from_cache",
    "tier0_flow_context_from_graph",
]
