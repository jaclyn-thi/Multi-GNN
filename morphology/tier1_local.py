"""Tier 1: batch-local morphology on a forward subgraph (matches loader support)."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

LOCAL_FEATURE_NAMES: List[str] = [
    "n_edges_sub",
    "n_nodes_sub",
    "sender_deg_out_local",
    "sender_deg_in_local",
    "receiver_deg_out_local",
    "receiver_deg_in_local",
    "deg_sum_out_local",
    "deg_sum_in_local",
]

# Indices within LOCAL_FEATURE_NAMES that are count-like (apply log1p before MSE).
LOCAL_COUNT_FEATURE_INDICES: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7)


def _degree_vectors(
    edge_index: torch.Tensor,
    num_nodes: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    e = edge_index.shape[1]
    ones = torch.ones(e, dtype=torch.long, device=device)
    deg_out = torch.zeros(num_nodes, dtype=torch.long, device=device)
    deg_in = torch.zeros(num_nodes, dtype=torch.long, device=device)
    if e > 0:
        deg_out.scatter_add_(0, edge_index[0], ones)
        deg_in.scatter_add_(0, edge_index[1], ones)
    return deg_out, deg_in


def _seed_positions_in_subgraph(
    seed_edge_ids: torch.Tensor,
    subgraph_edge_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
  Map each seed EdgeID to a column index in ``subgraph_edge_ids``.

  Returns ``(positions, valid_seed_mask)``. Positions are -1 where missing.
  Uses sorted lookup (O(E log E + S)) instead of a dense S×E compare matrix.
  """
    seed = seed_edge_ids.long().view(-1)
    sub_ids = subgraph_edge_ids.long().view(-1)
    if seed.numel() == 0 or sub_ids.numel() == 0:
        empty = seed.new_empty((0,), dtype=torch.long)
        return empty, seed.new_zeros((0,), dtype=torch.bool)

    sorted_sub, sort_idx = torch.sort(sub_ids)
    idx = torch.searchsorted(sorted_sub, seed)
    n_sub = sorted_sub.numel()
    idx_safe = idx.clamp(max=max(n_sub - 1, 0))
    valid = (idx < n_sub) & (sorted_sub[idx_safe] == seed)
    positions = sort_idx[idx_safe]
    positions = torch.where(valid, positions, torch.full_like(positions, -1))
    return positions, valid


def compute_local_morphology_torch(
    edge_index: torch.Tensor,
    subgraph_edge_ids: torch.Tensor,
    seed_edge_ids: torch.Tensor,
    num_nodes: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
  Batch-local morphology for seeds present in the subgraph.

  Parameters
  ----------
  edge_index :
      Forward ``[2, E_sub]`` (view1 subgraph).
  subgraph_edge_ids :
      ``edge_id`` per column in ``edge_index``, shape ``[E_sub]``.
  seed_edge_ids :
      Seed ``EdgeID`` values (e.g. ``seed_id1`` aligned with ``z1_seed``).
  num_nodes :
      Node count for degree vectors (``batch.num_nodes``).

  Returns
  -------
  Tensor ``(n_valid, len(LOCAL_FEATURE_NAMES))`` float32 on ``device``, rows in
  the same order as ``seed_edge_ids[valid]``. If some seeds are missing from the
  subgraph, they are dropped (caller should align embeddings separately).
  """
    device = device or edge_index.device
    edge_index = edge_index.to(device)
    positions, valid = _seed_positions_in_subgraph(seed_edge_ids, subgraph_edge_ids)
    if not valid.any():
        return torch.empty((0, len(LOCAL_FEATURE_NAMES)), device=device, dtype=torch.float32)

    pos = positions[valid]
    senders = edge_index[0, pos]
    receivers = edge_index[1, pos]
    deg_out, deg_in = _degree_vectors(edge_index, num_nodes, device)

    s_out = deg_out[senders].float()
    s_in = deg_in[senders].float()
    r_out = deg_out[receivers].float()
    r_in = deg_in[receivers].float()

    n_edges = float(edge_index.shape[1])
    if edge_index.numel() == 0:
        n_nodes = 0.0
    else:
        n_nodes = float(torch.unique(edge_index).numel())

    n = pos.shape[0]
    feats = torch.stack(
        [
            torch.full((n,), n_edges, device=device),
            torch.full((n,), n_nodes, device=device),
            s_out,
            s_in,
            r_out,
            r_in,
            s_out + r_out,
            s_in + r_in,
        ],
        dim=1,
    )
    return feats


def align_seed_embeddings_with_morph(
    z_seed: torch.Tensor,
    seed_edge_ids: torch.Tensor,
    subgraph_edge_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
  Drop seed rows whose EdgeID is not present in the subgraph (for target build).
  """
    positions, valid = _seed_positions_in_subgraph(seed_edge_ids, subgraph_edge_ids)
    if valid.all():
        return z_seed, seed_edge_ids
    return z_seed[valid], seed_edge_ids[valid]


def gather_seed_forward_edge_attr(
    edge_attr: torch.Tensor,
    subgraph_edge_ids: torch.Tensor,
    seed_edge_ids: torch.Tensor,
) -> torch.Tensor:
    """``edge_attr`` rows for each seed EdgeID, shape ``(n_valid, F)``."""
    positions, valid = _seed_positions_in_subgraph(seed_edge_ids, subgraph_edge_ids)
    if not valid.any():
        return edge_attr.new_empty((0, edge_attr.shape[1]))
    return edge_attr[positions[valid]]


def transform_morph_targets(
    local_feats: torch.Tensor,
    edge_native: Optional[torch.Tensor] = None,
    global_feats: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """log1p on count features; concat global and edge-native columns if provided."""
    from morphology.tier0_global import GLOBAL_COUNT_FEATURE_INDICES

    out = local_feats.clone()
    for idx in LOCAL_COUNT_FEATURE_INDICES:
        out[:, idx] = torch.log1p(out[:, idx].clamp(min=0))
    parts = [out]
    if global_feats is not None and global_feats.numel() > 0:
        g = global_feats.clone()
        for idx in GLOBAL_COUNT_FEATURE_INDICES:
            g[:, idx] = torch.log1p(g[:, idx].clamp(min=0))
        parts.append(g)
    if edge_native is not None and edge_native.numel() > 0:
        parts.append(edge_native.float())
    return torch.cat(parts, dim=1)


def compute_local_morphology(
    edge_index: torch.Tensor,
    seed_edge_ids: Union[torch.Tensor, np.ndarray, Sequence[int]],
    num_nodes: int,
    feature_names: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
  NumPy local morphology (tests / offline). Requires ``edge_id == column index``.

  See ``compute_local_morphology_torch`` for production batch subgraphs.
  """
    from morphology.graph_access import seed_endpoints_from_edge_ids

    names = list(feature_names or LOCAL_FEATURE_NAMES)
    eid = torch.as_tensor(seed_edge_ids, dtype=torch.long).view(-1)
    senders, receivers = seed_endpoints_from_edge_ids(eid, edge_index)
    deg_out, deg_in = _degree_vectors(edge_index, num_nodes, torch.device("cpu"))

    s_out = deg_out[senders].float().numpy()
    s_in = deg_in[senders].float().numpy()
    r_out = deg_out[receivers].float().numpy()
    r_in = deg_in[receivers].float().numpy()

    n_edges = float(edge_index.shape[1])
    active = torch.unique(edge_index.cpu()).numel() if edge_index.numel() else 0
    n_nodes = float(active)

    features = np.stack(
        [
            np.full(eid.shape[0], n_edges, dtype=np.float32),
            np.full(eid.shape[0], n_nodes, dtype=np.float32),
            s_out,
            s_in,
            r_out,
            r_in,
            s_out + r_out,
            s_in + r_in,
        ],
        axis=1,
    )
    if features.shape[1] != len(names):
        raise RuntimeError(f"feature dim {features.shape[1]} != len(names) {len(names)}")
    return features, names


def resolve_seed_positions_in_subgraph(
    seed_edge_ids: torch.Tensor,
    subgraph_edge_ids: torch.Tensor,
) -> torch.Tensor:
    """Map global seed EdgeIDs to column indices; raises if any seed is missing."""
    positions, valid = _seed_positions_in_subgraph(seed_edge_ids, subgraph_edge_ids)
    if not valid.all():
        missing = int((~valid).sum().item())
        raise ValueError(
            f"{missing} seed edge ids not in subgraph "
            f"(subgraph has {subgraph_edge_ids.numel()} edges)"
        )
    return positions[valid]
