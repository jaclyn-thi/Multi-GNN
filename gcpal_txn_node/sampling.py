"""Positive-aware and positive-complete batch construction for txn-node contrastive training."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch


def _neighbor_list(
    adj: Sequence[np.ndarray],
    node: int,
    *,
    exclude_self: bool = True,
) -> np.ndarray:
    if node < 0 or node >= len(adj):
        return np.zeros(0, dtype=np.int64)
    nbs = np.asarray(adj[node], dtype=np.int64)
    if nbs.size == 0:
        return nbs
    if exclude_self:
        nbs = nbs[nbs != int(node)]
    # Deduplicate while preserving order
    if nbs.size:
        _, idx = np.unique(nbs, return_index=True)
        nbs = nbs[np.sort(idx)]
    return nbs


def sample_positive_aware_batch(
    anchor_ids: np.ndarray,
    flow_adj: Sequence[np.ndarray],
    knn_adj: Sequence[np.ndarray],
    *,
    max_struct_extras: int = 4,
    max_knn_extras: int = 8,
    max_total_nodes: int = 4096,
    rng: Optional[np.random.RandomState] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Legacy capped expansion (previous txn-node mode). Unchanged semantics."""
    rng = rng or np.random.RandomState(0)
    nodes = set(int(a) for a in anchor_ids.tolist())
    struct_added = 0
    knn_added = 0
    for a in anchor_ids.tolist():
        a = int(a)
        outs = _neighbor_list(flow_adj, a)
        if outs.size:
            take = outs[:max_struct_extras]
            for j in take.tolist():
                if j not in nodes and len(nodes) < max_total_nodes:
                    nodes.add(int(j))
                    struct_added += 1
        kn = _neighbor_list(knn_adj, a)
        if kn.size:
            order = np.arange(kn.size)
            rng.shuffle(order)
            for j in kn[order][:max_knn_extras].tolist():
                if int(j) == a:
                    continue
                if int(j) not in nodes and len(nodes) < max_total_nodes:
                    nodes.add(int(j))
                    knn_added += 1
        if len(nodes) >= max_total_nodes:
            break
    node_ids = np.asarray(sorted(nodes), dtype=np.int64)
    stats = {
        "n_anchors": float(anchor_ids.size),
        "n_nodes": float(node_ids.size),
        "struct_extras_added": float(struct_added),
        "knn_extras_added": float(knn_added),
        "growth_ratio": float(node_ids.size / max(anchor_ids.size, 1)),
        "batching_mode": 0.0,  # legacy marker for JSON-friendly floats
    }
    return node_ids, stats


def _closure_for_anchor(
    anchor: int,
    flow_adj: Sequence[np.ndarray],
    knn_adj: Sequence[np.ndarray],
) -> Tuple[Set[int], np.ndarray, np.ndarray]:
    """Return ({anchor}∪struct∪knn, struct_nbs, knn_nbs) with self excluded from neighbor lists."""
    struct = _neighbor_list(flow_adj, anchor)
    knn = _neighbor_list(knn_adj, anchor)
    closure = {int(anchor)}
    closure.update(int(x) for x in struct.tolist())
    closure.update(int(x) for x in knn.tolist())
    return closure, struct, knn


def sample_positive_complete_batch(
    candidate_stream: np.ndarray,
    flow_adj: Sequence[np.ndarray],
    knn_adj: Sequence[np.ndarray],
    *,
    max_total_nodes: int = 2048,
    start: int = 0,
    allowed_ids: Optional[Set[int]] = None,
    labels: Optional[np.ndarray] = None,
    knn_k: int = 15,
) -> Tuple[np.ndarray, np.ndarray, int, Dict[str, float]]:
    """Greedily accept anchors with full global KNN + structural closures under a node cap.

    Parameters
    ----------
    candidate_stream :
        Ordered train-local transaction ids (typically a permutation). Sampling is
        label-agnostic; ``labels`` is diagnostics-only.
    allowed_ids :
        If provided, every accepted node must be in this set (split-leakage guard).
    labels :
        Optional binary labels aligned to train-local ids; used only for minority counts.

    Returns
    -------
    anchor_ids, node_ids (sorted unique), next_stream_index, diagnostics
    """
    if start < 0 or start > int(candidate_stream.size):
        raise ValueError(f"invalid start={start}")
    anchors: List[int] = []
    node_set: Set[int] = set()
    struct_req = 0
    knn_req = 0
    struct_unique_added = 0
    knn_unique_added = 0
    self_excluded = 0
    collision_already_present = 0
    rejected_not_allowed = 0
    cap_stopped = False
    j = int(start)

    while j < int(candidate_stream.size):
        a = int(candidate_stream[j])
        if allowed_ids is not None and a not in allowed_ids:
            raise ValueError(f"candidate anchor {a} not in allowed train id set")
        closure, struct, knn = _closure_for_anchor(a, flow_adj, knn_adj)
        # Count self hits in raw adj (defensive)
        if a < len(flow_adj):
            raw_f = np.asarray(flow_adj[a], dtype=np.int64)
            self_excluded += int(np.sum(raw_f == a))
        if a < len(knn_adj):
            raw_k = np.asarray(knn_adj[a], dtype=np.int64)
            self_excluded += int(np.sum(raw_k == a))

        if allowed_ids is not None:
            bad = [n for n in closure if n not in allowed_ids]
            if bad:
                rejected_not_allowed += len(bad)
                raise ValueError(
                    f"split leakage: neighbors of {a} outside allowed train ids: {bad[:5]}"
                )

        to_add = [n for n in closure if n not in node_set]
        if anchors and (len(node_set) + len(to_add) > int(max_total_nodes)):
            cap_stopped = True
            break

        # First anchor must always fit (k=15 + struct << 2048); force-add if empty.
        if not anchors and (len(to_add) > int(max_total_nodes)):
            raise RuntimeError(
                f"single-anchor closure size {len(to_add)} exceeds max_total_nodes={max_total_nodes}"
            )

        for n in to_add:
            if n in node_set:
                collision_already_present += 1
            else:
                if n != a and n in set(int(x) for x in struct.tolist()):
                    struct_unique_added += 1
                if n != a and n in set(int(x) for x in knn.tolist()):
                    knn_unique_added += 1
                node_set.add(n)
        # Nodes already present still "collide" for requested neighbors
        for n in closure:
            if n != a and n in node_set and n not in to_add:
                collision_already_present += 1

        struct_req += int(struct.size)
        knn_req += int(knn.size)
        anchors.append(a)
        j += 1
        if len(node_set) >= int(max_total_nodes):
            # Batch full; next stream index is j (may still have candidates).
            cap_stopped = True
            break

    if not anchors:
        raise RuntimeError("positive-complete sampler produced zero anchors")

    anchor_ids = np.asarray(anchors, dtype=np.int64)
    node_ids = np.asarray(sorted(node_set), dtype=np.int64)
    local = {int(g): i for i, g in enumerate(node_ids.tolist())}

    # Coverage among accepted anchors (should be complete by construction).
    knn_ge1 = 0
    knn_all = 0
    knn_available_all = 0
    struct_ge1 = 0
    struct_all = 0
    knn_pos_counts: List[int] = []
    struct_pos_counts: List[int] = []
    identity_counts: List[int] = []
    for a in anchors:
        struct = _neighbor_list(flow_adj, a)
        knn = _neighbor_list(knn_adj, a)
        knn_present = sum(1 for n in knn.tolist() if int(n) in local)
        struct_present = sum(1 for n in struct.tolist() if int(n) in local)
        knn_pos_counts.append(knn_present)
        struct_pos_counts.append(struct_present)
        identity_counts.append(1)
        if knn_present >= 1:
            knn_ge1 += 1
        if knn.size == 0 or knn_present == int(knn.size):
            knn_available_all += 1
        if int(knn.size) >= int(knn_k) and knn_present >= int(knn_k):
            knn_all += 1
        elif int(knn.size) < int(knn_k) and knn_present == int(knn.size):
            # Fewer than k cached — treat as "all available present"
            knn_all += 1
        if struct_present >= 1:
            struct_ge1 += 1
        if struct.size == 0 or struct_present == int(struct.size):
            struct_all += 1

    minority = 0
    if labels is not None:
        for a in anchors:
            if int(labels[int(a)]) == 1:
                minority += 1

    def _mean(xs: List[int]) -> float:
        return float(np.mean(xs)) if xs else 0.0

    def _median(xs: List[int]) -> float:
        return float(np.median(xs)) if xs else 0.0

    n_a = float(len(anchors))
    stats: Dict[str, float] = {
        "batching_mode": 1.0,  # positive_complete
        "requested_n_anchors": float(max(0, int(candidate_stream.size) - int(start))),
        "realized_n_anchors": n_a,
        "n_anchors": n_a,
        "n_nodes": float(node_ids.size),
        "max_total_nodes": float(max_total_nodes),
        "frac_anchors_knn_ge1": knn_ge1 / n_a,
        "frac_anchors_knn_all_k": knn_all / n_a,
        "frac_anchors_knn_all_available": knn_available_all / n_a,
        "frac_anchors_struct_ge1": struct_ge1 / n_a,
        "frac_anchors_struct_all_available": struct_all / n_a,
        "mean_identity_pos": _mean(identity_counts),
        "median_identity_pos": _median(identity_counts),
        "mean_structural_pos": _mean(struct_pos_counts),
        "median_structural_pos": _median(struct_pos_counts),
        "mean_knn_pos": _mean(knn_pos_counts),
        "median_knn_pos": _median(knn_pos_counts),
        "struct_neighbors_requested": float(struct_req),
        "knn_neighbors_requested": float(knn_req),
        "struct_unique_added": float(struct_unique_added),
        "knn_unique_added": float(knn_unique_added),
        "collision_already_present": float(collision_already_present),
        "self_excluded": float(self_excluded),
        "rejected_not_allowed": float(rejected_not_allowed),
        "cap_stopped": float(1.0 if cap_stopped else 0.0),
        "stream_start": float(start),
        "stream_next": float(j),
        "minority_anchor_count": float(minority),
        "growth_ratio": float(node_ids.size / max(len(anchors), 1)),
    }
    return anchor_ids, node_ids, j, stats


def local_pairs_from_adj(
    adj: Sequence[np.ndarray],
    node_ids: np.ndarray,
    *,
    sources: Optional[np.ndarray] = None,
) -> torch.Tensor:
    """Return [2, E] local-index pairs for edges with both ends in ``node_ids``.

    If ``sources`` is provided, only emit edges whose global source is in ``sources``
    (used so context nodes are not treated as independent anchors).
    """
    local = {int(g): i for i, g in enumerate(node_ids.tolist())}
    src_filter: Optional[Set[int]] = None
    if sources is not None:
        src_filter = {int(s) for s in sources.tolist()}
    src: List[int] = []
    dst: List[int] = []
    iterate = sources if sources is not None else node_ids
    for g in iterate.tolist():
        g = int(g)
        if src_filter is not None and g not in src_filter:
            continue
        if g >= len(adj):
            continue
        for nb in _neighbor_list(adj, g).tolist():
            nb = int(nb)
            if nb in local and g in local:
                src.append(local[g])
                dst.append(local[nb])
    if not src:
        return torch.zeros((2, 0), dtype=torch.long)
    return torch.tensor([src, dst], dtype=torch.long)


def assert_batch_subset_of_allowed(node_ids: np.ndarray, allowed_ids: Set[int]) -> None:
    for n in node_ids.tolist():
        if int(n) not in allowed_ids:
            raise ValueError(f"split leakage: node {n} not in allowed train ids")
