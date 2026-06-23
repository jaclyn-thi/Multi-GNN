"""Tier 1: batch-local morphology on a forward subgraph (matches loader support).

These metrics are computed on the **view1** subgraph visible in the current
``LinkNeighborLoader`` batch. They differ from Tier 0 global degrees: a hub node
may have low *local* degree if the sampled neighborhood is small.

Used by:
- M1 expert targets (``build_morph_targets``) — 14 local dims (degree/ego + clustering + triangles)
- M2 contrast binning (``build_morph_features_for_contrast``) — optional ``local_clustering`` / ``local_triangles`` groups
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import torch

# Ego + degree block (first 8); clustering (3); triangle counts (3) — Jun 2026 Tier-1 extensions.
LOCAL_DEGREE_FEATURE_END = 8
LOCAL_CLUSTERING_FEATURE_END = 11

LOCAL_FEATURE_NAMES: List[str] = [
    "n_edges_sub",
    "n_nodes_sub",
    "sender_deg_out_local",
    "sender_deg_in_local",
    "receiver_deg_out_local",
    "receiver_deg_in_local",
    "deg_sum_out_local",
    "deg_sum_in_local",
    "sender_clustering_local",
    "receiver_clustering_local",
    "mean_clustering_local",
    "sender_triangles_local",
    "receiver_triangles_local",
    "mean_triangles_local",
]

LOCAL_CLUSTERING_FEATURE_NAMES: Tuple[str, ...] = (
    "sender_clustering_local",
    "receiver_clustering_local",
    "mean_clustering_local",
)

LOCAL_TRIANGLES_FEATURE_NAMES: Tuple[str, ...] = (
    "sender_triangles_local",
    "receiver_triangles_local",
    "mean_triangles_local",
)

# Indices within LOCAL_FEATURE_NAMES that are count-like (apply log1p before MSE).
LOCAL_COUNT_FEATURE_INDICES: Tuple[int, ...] = tuple(range(LOCAL_DEGREE_FEATURE_END)) + tuple(
    range(LOCAL_CLUSTERING_FEATURE_END, len(LOCAL_FEATURE_NAMES))
)

# M2 contrast groups slice into these index ranges (degree block excludes ego cols 0–1).
LOCAL_DEGREE_INDICES: Tuple[int, ...] = tuple(range(2, LOCAL_DEGREE_FEATURE_END))
LOCAL_CLUSTERING_INDICES: Tuple[int, ...] = tuple(
    range(LOCAL_DEGREE_FEATURE_END, LOCAL_CLUSTERING_FEATURE_END)
)
LOCAL_TRIANGLES_INDICES: Tuple[int, ...] = tuple(
    range(LOCAL_CLUSTERING_FEATURE_END, len(LOCAL_FEATURE_NAMES))
)

VALID_MORPH_LOCAL_SUBSETS = frozenset({"all", "degree", "clustering", "triangles"})


def local_feature_indices_for_subset(subset: str) -> Tuple[int, ...]:
    """
    Column indices into the full ``LOCAL_FEATURE_NAMES`` vector for an expert subset.

    ``degree`` = ego + local degrees (8). ``clustering`` / ``triangles`` add 3 cols each
    on top of degree (11 total). ``all`` = full 14-dim vector.
    """
    key = str(subset).lower()
    if key not in VALID_MORPH_LOCAL_SUBSETS:
        raise ValueError(
            f"morph local subset {subset!r} invalid; use one of {sorted(VALID_MORPH_LOCAL_SUBSETS)}."
        )
    if key == "degree":
        return tuple(range(LOCAL_DEGREE_FEATURE_END))
    if key == "clustering":
        return tuple(range(LOCAL_DEGREE_FEATURE_END)) + LOCAL_CLUSTERING_INDICES
    if key == "triangles":
        return tuple(range(LOCAL_DEGREE_FEATURE_END)) + LOCAL_TRIANGLES_INDICES
    return tuple(range(len(LOCAL_FEATURE_NAMES)))


def local_feature_dim_for_subset(subset: str) -> int:
    return len(local_feature_indices_for_subset(subset))


def local_feature_names_for_subset(subset: str) -> List[str]:
    idx = local_feature_indices_for_subset(subset)
    return [LOCAL_FEATURE_NAMES[i] for i in idx]


def slice_local_morph_features(local_feats: torch.Tensor, subset: str) -> torch.Tensor:
    """Select expert local columns from the full Tier-1 computation."""
    idx = local_feature_indices_for_subset(subset)
    return local_feats[:, idx]


def local_count_indices_in_subset(subset: str) -> Tuple[int, ...]:
    """Indices within a subset-sliced local tensor that receive ``log1p`` before expert loss."""
    full_idx = local_feature_indices_for_subset(subset)
    return tuple(
        out_i
        for out_i, full_i in enumerate(full_idx)
        if full_i in LOCAL_COUNT_FEATURE_INDICES
    )


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


def _build_undirected_adjacency(edge_index: torch.Tensor) -> Dict[int, Set[int]]:
    """Undirected neighbor sets from a forward ``edge_index`` (view1 subgraph)."""
    adj: Dict[int, Set[int]] = {}
    if edge_index.numel() == 0:
        return adj
    ei = edge_index.cpu()
    for u, v in zip(ei[0].tolist(), ei[1].tolist()):
        if u == v:
            continue
        adj.setdefault(int(u), set()).add(int(v))
        adj.setdefault(int(v), set()).add(int(u))
    return adj


def _count_undirected_neighbor_links(adj: Dict[int, Set[int]], node: int) -> int:
    """
    Undirected edges between distinct neighbor pairs of ``node``.

    Each triangle through ``node`` contributes exactly one such link; this count
    equals the local triangle count at ``node``.
    """
    nbrs = adj.get(node)
    if not nbrs or len(nbrs) < 2:
        return 0
    nbr_list = list(nbrs)
    k = len(nbr_list)
    links = 0
    for i in range(k):
        nbr_i_set = adj[nbr_list[i]]
        for j in range(i + 1, k):
            if nbr_list[j] in nbr_i_set:
                links += 1
    return links


def _undirected_clustering_coefficient(adj: Dict[int, Set[int]], node: int) -> float:
    """
    Local clustering on the induced undirected subgraph.

    For node ``v`` with ``k`` neighbors, count undirected edges between neighbor
    pairs and divide by ``k * (k - 1) / 2``. Returns ``0`` when ``k < 2``.
    """
    nbrs = adj.get(node)
    if not nbrs or len(nbrs) < 2:
        return 0.0
    k = len(nbrs)
    links = _count_undirected_neighbor_links(adj, node)
    return float((2.0 * links) / (k * (k - 1)))


def _undirected_triangle_count(adj: Dict[int, Set[int]], node: int) -> float:
    """Number of undirected triangles incident on ``node`` in the batch subgraph."""
    return float(_count_undirected_neighbor_links(adj, node))


def _local_clustering_map(edge_index: torch.Tensor) -> Dict[int, float]:
    """Clustering coefficient for every node incident on ``edge_index``."""
    adj = _build_undirected_adjacency(edge_index)
    if not adj:
        return {}
    active = edge_index.unique().cpu().tolist()
    return {int(n): _undirected_clustering_coefficient(adj, int(n)) for n in active}


def _local_triangle_map(edge_index: torch.Tensor) -> Dict[int, float]:
    """Triangle count for every node incident on ``edge_index``."""
    adj = _build_undirected_adjacency(edge_index)
    if not adj:
        return {}
    active = edge_index.unique().cpu().tolist()
    return {int(n): _undirected_triangle_count(adj, int(n)) for n in active}


def _lookup_clustering(
    clust_map: Dict[int, float],
    node_ids: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    vals = [clust_map.get(int(n.item()), 0.0) for n in node_ids.cpu()]
    return torch.tensor(vals, device=device, dtype=torch.float32)


def _clustering_triplet_for_endpoints(
    edge_index: torch.Tensor,
    senders: torch.Tensor,
    receivers: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    clust_map = _local_clustering_map(edge_index)
    s_clust = _lookup_clustering(clust_map, senders, device)
    r_clust = _lookup_clustering(clust_map, receivers, device)
    mean_clust = 0.5 * (s_clust + r_clust)
    return s_clust, r_clust, mean_clust


def _triangle_triplet_for_endpoints(
    edge_index: torch.Tensor,
    senders: torch.Tensor,
    receivers: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tri_map = _local_triangle_map(edge_index)
    s_tri = _lookup_clustering(tri_map, senders, device)
    r_tri = _lookup_clustering(tri_map, receivers, device)
    mean_tri = 0.5 * (s_tri + r_tri)
    return s_tri, r_tri, mean_tri


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

    s_clust, r_clust, mean_clust = _clustering_triplet_for_endpoints(
        edge_index, senders, receivers, device
    )
    s_tri, r_tri, mean_tri = _triangle_triplet_for_endpoints(
        edge_index, senders, receivers, device
    )

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
            s_clust,
            r_clust,
            mean_clust,
            s_tri,
            r_tri,
            mean_tri,
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
    Drop seed rows whose EdgeID is not present in the subgraph.

    Returns aligned ``(z_seed, seed_edge_ids)`` for expert target construction.
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
    flow_feats: Optional[torch.Tensor] = None,
    tier2_feats: Optional[torch.Tensor] = None,
    local_count_indices: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    """
    Concatenate local / global / flow / tier2 / edge-native blocks into the expert target vector.

    Applies log1p to count-like columns (local, global degree, Tier 2 BC lift).
    Flow-balance and edge-native blocks are expected pre-transformed at lift time.
    """
    from morphology.tier0_global import GLOBAL_COUNT_FEATURE_INDICES

    count_idx = (
        tuple(local_count_indices)
        if local_count_indices is not None
        else LOCAL_COUNT_FEATURE_INDICES
    )
    out = local_feats.clone()
    for idx in count_idx:
        out[:, idx] = torch.log1p(out[:, idx].clamp(min=0))
    parts = [out]
    if global_feats is not None and global_feats.numel() > 0:
        g = global_feats.clone()
        for idx in GLOBAL_COUNT_FEATURE_INDICES:
            g[:, idx] = torch.log1p(g[:, idx].clamp(min=0))
        parts.append(g)
    if flow_feats is not None and flow_feats.numel() > 0:
        parts.append(flow_feats.float())
    if tier2_feats is not None and tier2_feats.numel() > 0:
        t2 = tier2_feats.clone()
        for idx in range(t2.shape[1]):
            t2[:, idx] = torch.log1p(t2[:, idx].clamp(min=0))
        parts.append(t2)
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

    senders_t = torch.as_tensor(senders, dtype=torch.long)
    receivers_t = torch.as_tensor(receivers, dtype=torch.long)
    s_clust, r_clust, mean_clust = _clustering_triplet_for_endpoints(
        edge_index, senders_t, receivers_t, torch.device("cpu")
    )
    s_tri, r_tri, mean_tri = _triangle_triplet_for_endpoints(
        edge_index, senders_t, receivers_t, torch.device("cpu")
    )
    s_clust_np = s_clust.numpy()
    r_clust_np = r_clust.numpy()
    mean_clust_np = mean_clust.numpy()
    s_tri_np = s_tri.numpy()
    r_tri_np = r_tri.numpy()
    mean_tri_np = mean_tri.numpy()

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
            s_clust_np,
            r_clust_np,
            mean_clust_np,
            s_tri_np,
            r_tri_np,
            mean_tri_np,
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
