"""Graph accessors for morphology (homogeneous and hetero forward edges)."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
from torch_geometric.data import Data, HeteroData

from train_util import FORWARD_EDGE_TYPE

GraphData = Union[Data, HeteroData]


def get_forward_edge_index(data: GraphData) -> torch.Tensor:
    """Forward transaction ``edge_index`` ``[2, E]`` (sender row 0, receiver row 1)."""
    if isinstance(data, HeteroData):
        return data[FORWARD_EDGE_TYPE].edge_index
    return data.edge_index


def get_forward_edge_attr(data: GraphData) -> torch.Tensor:
    """Forward ``edge_attr`` (includes synthetic id column 0 after ``add_arange_ids``)."""
    if isinstance(data, HeteroData):
        return data[FORWARD_EDGE_TYPE].edge_attr
    return data.edge_attr


def get_forward_timestamps(data: GraphData) -> Optional[torch.Tensor]:
    if isinstance(data, HeteroData):
        store = data[FORWARD_EDGE_TYPE]
        return getattr(store, "timestamps", None)
    return getattr(data, "timestamps", None)


def get_num_nodes(data: GraphData) -> int:
    if isinstance(data, HeteroData):
        return int(data["node"].x.shape[0])
    return int(data.num_nodes)


def get_edge_ids_for_positions(
    data: GraphData,
    positions: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return ``EdgeID`` (synthetic id column) for each forward edge position."""
    attr = get_forward_edge_attr(data)
    if attr is None or attr.shape[1] < 1:
        raise ValueError("Forward edge_attr missing id column; run add_arange_ids first.")
    ids = attr[:, 0].long()
    if positions is None:
        return ids
    return ids[positions.long().view(-1)]


def seed_endpoints_from_edge_ids(
    edge_ids: torch.Tensor,
    edge_index: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
  Map seed ``EdgeID`` values to ``(sender, receiver)`` node ids.

  Assumes ``edge_id`` equals the column index in ``edge_index`` on this graph.
  """
    eid = edge_ids.long().view(-1).cpu()
    ei = edge_index.cpu()
    if eid.max().item() >= ei.shape[1] or eid.min().item() < 0:
        raise ValueError(
            f"edge_ids out of range for edge_index with E={ei.shape[1]}: "
            f"min={eid.min().item()} max={eid.max().item()}"
        )
    senders = ei[0, eid]
    receivers = ei[1, eid]
    return senders, receivers
