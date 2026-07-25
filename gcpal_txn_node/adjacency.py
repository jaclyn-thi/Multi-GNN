"""Directed transaction-flow adjacency (txn-as-node)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class FlowGraphStats:
    n_nodes: int
    n_edges: int
    policy: str
    mean_out_degree: float
    max_out_degree: int
    fraction_nodes_with_out_edge: float
    note: str


def build_directed_flow_adjacency(
    from_id: np.ndarray,
    to_id: np.ndarray,
    timestamp: np.ndarray,
    *,
    policy: str = "immediate_next",
    capped_k: int = 3,
) -> Tuple[np.ndarray, FlowGraphStats]:
    """Build sparse directed flow edges between transactions.

    Edge i -> j when receiver(i) == sender(j) and timestamp[j] > timestamp[i].

    Policies
    --------
    immediate_next :
        Only the nearest subsequent outgoing transaction from the receiver account.
    capped_next_k :
        Up to ``capped_k`` subsequent outgoing transactions (still not all-pairs).
    """
    if policy not in ("immediate_next", "capped_next_k"):
        raise ValueError(f"Unknown adjacency policy: {policy}")
    n = int(from_id.shape[0])
    if to_id.shape[0] != n or timestamp.shape[0] != n:
        raise ValueError("from_id, to_id, timestamp length mismatch")

    outgoing: Dict[int, List[Tuple[float, int]]] = defaultdict(list)
    for i in range(n):
        outgoing[int(from_id[i])].append((float(timestamp[i]), i))
    for acc in outgoing:
        outgoing[acc].sort(key=lambda x: (x[0], x[1]))

    src: List[int] = []
    dst: List[int] = []
    max_take = 1 if policy == "immediate_next" else max(1, int(capped_k))

    for i in range(n):
        recv = int(to_id[i])
        key = (float(timestamp[i]), i)
        cand = outgoing.get(recv)
        if not cand:
            continue
        lo, hi = 0, len(cand)
        while lo < hi:
            mid = (lo + hi) // 2
            if cand[mid] > key:
                hi = mid
            else:
                lo = mid + 1
        taken = 0
        for jpos in range(lo, len(cand)):
            _t, j = cand[jpos]
            if j == i:
                continue
            src.append(i)
            dst.append(j)
            taken += 1
            if taken >= max_take:
                break

    if src:
        edge_index = np.vstack([np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64)])
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)

    out_deg = np.bincount(edge_index[0], minlength=n) if edge_index.shape[1] else np.zeros(n, dtype=np.int64)
    stats = FlowGraphStats(
        n_nodes=n,
        n_edges=int(edge_index.shape[1]),
        policy=policy if policy == "immediate_next" else f"{policy}_k{max_take}",
        mean_out_degree=float(out_deg.mean()) if n else 0.0,
        max_out_degree=int(out_deg.max()) if n else 0,
        fraction_nodes_with_out_edge=float((out_deg > 0).mean()) if n else 0.0,
        note=(
            "Implementation assumption: directed receiver→next-sender flow; "
            "not all pairwise same-account connections."
        ),
    )
    return edge_index, stats


def adjacency_list_from_edge_index(edge_index: np.ndarray, n_nodes: int) -> List[np.ndarray]:
    """Outgoing neighbor arrays per node (int64)."""
    lists: List[List[int]] = [[] for _ in range(n_nodes)]
    for s, d in zip(edge_index[0].tolist(), edge_index[1].tolist()):
        lists[int(s)].append(int(d))
    return [np.asarray(v, dtype=np.int64) for v in lists]


def induce_edge_index(edge_index: np.ndarray, node_ids: np.ndarray) -> np.ndarray:
    """Keep edges with both endpoints in ``node_ids``; reindex to 0..len-1."""
    if node_ids.size == 0:
        return np.zeros((2, 0), dtype=np.int64)
    mapping = -np.ones(int(node_ids.max()) + 1, dtype=np.int64)
    mapping[node_ids] = np.arange(node_ids.shape[0], dtype=np.int64)
    src = edge_index[0]
    dst = edge_index[1]
    keep = (src <= node_ids.max()) & (dst <= node_ids.max())
    if keep.any():
        src = src[keep]
        dst = dst[keep]
        msrc = mapping[src]
        mdst = mapping[dst]
        ok = (msrc >= 0) & (mdst >= 0)
        if ok.any():
            return np.vstack([msrc[ok], mdst[ok]]).astype(np.int64)
    return np.zeros((2, 0), dtype=np.int64)
