"""Edge-centric GCPAL-inspired neighbor positives (NOT an exact GCPAL reproduction).

Distinct from ``--enable_knn_soft_positives`` (low-weight soft InfoNCE).

Positives for an anchor transaction edge across two random views are the boolean
union of:

1. Identity (same edge_id in the other view)
2. Directed-flow structural neighbors (receiver→next-sender, immediate_next)
3. Feature-KNN neighbors from the train-split global cache (k=15)

Positive-complete batching retrieves those edges into the seed set before view
construction. Loss uses ``supcon_mean_logprob`` averaged over original anchors only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

from gcpal_txn_node.adjacency import (
    adjacency_list_from_edge_index,
    build_directed_flow_adjacency,
)
from gcpal_txn_node.knn_adapter import SparseKNNGraph, load_train_knn_cache
from gcpal_txn_node.loss import (
    POSITIVE_AGGREGATION_SUPCON,
    multipositive_infonce,
    validate_positive_aggregation,
)
from gcpal_txn_node.sampling import sample_positive_complete_batch
from gcpal_txn_node.spec import DEFAULT_KNN_CACHE

NOT_EXACT_REPRODUCTION = True
DEFAULT_MAX_TOTAL_EDGES = 2048
DEFAULT_FLOW_POLICY = "immediate_next"
DEFAULT_KNN_K = 15
CHECKPOINT_EPOCHS_DEFAULT = (1, 3, 5, 10)


@dataclass
class EdgeNeighborPositiveContext:
    """Train-split-only structures for neighbor-positive SSL (no labels in construction)."""

    n_train: int
    flow_adj: List[np.ndarray]
    knn_adj: List[np.ndarray]
    knn: SparseKNNGraph
    flow_stats: Any
    train_labels: Optional[np.ndarray]  # diagnostics only
    csv_edge_ids: Optional[np.ndarray]
    allowed_ids: Set[int]
    knn_cache_path: str
    flow_policy: str
    knn_k: int
    max_total_edges: int
    positive_mode: str  # "neighbor" | "identity"
    positive_aggregation: str
    leakage_audit: Dict[str, Any]


def _endpoints_and_times_from_hetero(tr_data) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    store = tr_data["node", "to", "node"]
    ei = store.edge_index.detach().cpu().numpy()
    from_id = ei[0].astype(np.int64)
    to_id = ei[1].astype(np.int64)
    ts = store.timestamps.detach().cpu().numpy().astype(np.float64).reshape(-1)
    if ts.shape[0] != from_id.shape[0]:
        raise ValueError(
            f"timestamp/edge length mismatch: ts={ts.shape[0]} edges={from_id.shape[0]}"
        )
    return from_id, to_id, ts


def build_edge_neighbor_positive_context(
    tr_data,
    *,
    knn_cache_path: str = DEFAULT_KNN_CACHE,
    flow_policy: str = DEFAULT_FLOW_POLICY,
    knn_k: int = DEFAULT_KNN_K,
    max_total_edges: int = DEFAULT_MAX_TOTAL_EDGES,
    positive_mode: str = "neighbor",
    positive_aggregation: str = POSITIVE_AGGREGATION_SUPCON,
    train_labels: Optional[np.ndarray] = None,
) -> EdgeNeighborPositiveContext:
    """Build train-only flow + KNN adjacency in train-local edge-id space."""
    if positive_mode not in ("neighbor", "identity"):
        raise ValueError(f"positive_mode must be neighbor|identity, got {positive_mode!r}")
    agg = validate_positive_aggregation(positive_aggregation)
    if agg != POSITIVE_AGGREGATION_SUPCON:
        logging.warning(
            "edge neighbor positives: expected %s (val-selected in txn-node ablation); got %s",
            POSITIVE_AGGREGATION_SUPCON,
            agg,
        )

    from_id, to_id, ts = _endpoints_and_times_from_hetero(tr_data)
    n_train = int(from_id.shape[0])
    flow_ei, flow_stats = build_directed_flow_adjacency(
        from_id, to_id, ts, policy=flow_policy
    )
    flow_adj = adjacency_list_from_edge_index(flow_ei, n_train)

    knn = load_train_knn_cache(knn_cache_path, expected_k=int(knn_k))
    if int(knn.node_ids.shape[0]) != n_train:
        raise ValueError(
            f"KNN cache n={knn.node_ids.shape[0]} != train edges n={n_train}; "
            "refuse silent misalignment"
        )
    if not np.array_equal(knn.node_ids, np.arange(n_train, dtype=np.int64)):
        raise ValueError(
            "KNN cache edge_ids must be train-local arange(0..n_train-1) to match "
            "add_arange_ids on tr_data"
        )
    knn_adj = knn.adjacency_lists()

    csv_edge_ids = None
    data = np.load(knn_cache_path, allow_pickle=True)
    if "csv_edge_ids" in data.files:
        csv_edge_ids = np.asarray(data["csv_edge_ids"], dtype=np.int64)
        if csv_edge_ids.shape[0] != n_train:
            raise ValueError("csv_edge_ids length mismatch vs train")

    allowed = set(range(n_train))
    # Leakage audit: every neighbor must be in train-local allowed set
    bad_flow = 0
    bad_knn = 0
    for i in range(n_train):
        for j in flow_adj[i].tolist():
            if int(j) not in allowed:
                bad_flow += 1
        for j in knn_adj[i].tolist():
            if int(j) not in allowed:
                bad_knn += 1
    if bad_flow or bad_knn:
        raise RuntimeError(
            f"Split leakage in adjacency: bad_flow={bad_flow} bad_knn={bad_knn}"
        )

    leakage_audit = {
        "n_train": n_train,
        "id_space": "train_local_arange_matches_add_arange_ids",
        "knn_cache_path": str(knn_cache_path),
        "knn_feature_set": knn.feature_set,
        "knn_deviations": list(knn.deviation_notes),
        "flow_policy": flow_stats.policy,
        "flow_n_edges": int(flow_stats.n_edges),
        "flow_mean_out_degree": float(flow_stats.mean_out_degree),
        "bad_flow_outside_train": 0,
        "bad_knn_outside_train": 0,
        "labels_used_in_construction": False,
        "not_exact_gcpal_reproduction": True,
        "distinct_from_enable_knn_soft_positives": True,
    }
    logging.info(
        "Edge neighbor-positive context: n_train=%s flow_edges=%s knn_k=%s mode=%s agg=%s",
        n_train,
        flow_stats.n_edges,
        knn_k,
        positive_mode,
        agg,
    )
    return EdgeNeighborPositiveContext(
        n_train=n_train,
        flow_adj=flow_adj,
        knn_adj=knn_adj,
        knn=knn,
        flow_stats=flow_stats,
        train_labels=None if train_labels is None else np.asarray(train_labels, dtype=np.int64),
        csv_edge_ids=csv_edge_ids,
        allowed_ids=allowed,
        knn_cache_path=str(knn_cache_path),
        flow_policy=str(flow_policy),
        knn_k=int(knn_k),
        max_total_edges=int(max_total_edges),
        positive_mode=str(positive_mode),
        positive_aggregation=agg,
        leakage_audit=leakage_audit,
    )


def expand_poscomplete_seeds(
    candidate_stream: np.ndarray,
    ctx: EdgeNeighborPositiveContext,
    *,
    start: int = 0,
) -> Tuple[np.ndarray, np.ndarray, int, Dict[str, float]]:
    """Greedy positive-complete expansion; returns anchors, all_seed_ids, next_idx, stats."""
    anchors, nodes, nxt, stats = sample_positive_complete_batch(
        candidate_stream,
        ctx.flow_adj,
        ctx.knn_adj,
        max_total_nodes=int(ctx.max_total_edges),
        start=int(start),
        allowed_ids=ctx.allowed_ids,
        labels=ctx.train_labels,
        knn_k=int(ctx.knn_k),
    )
    # Post-hoc minority diagnostics only
    if ctx.train_labels is not None and anchors.size:
        lab = ctx.train_labels[anchors]
        stats["anchor_minority_frac"] = float(lab.mean()) if lab.size else 0.0
    else:
        stats["anchor_minority_frac"] = float("nan")
    stats["labels_used_in_construction"] = 0.0
    return anchors, nodes, nxt, stats


def build_edge_neighbor_positive_mask(
    shared_edge_ids: torch.Tensor,
    *,
    ctx: EdgeNeighborPositiveContext,
    anchor_ids: Optional[Sequence[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Boolean (B,B) positive mask over aligned shared seed rows + anchor row indices.

    Identity is always included. Structural/KNN added only when ``ctx.positive_mode=='neighbor'``.
    Duplicates collapse via boolean OR. ``anchor_row_indices`` selects original anchors
    that survived in ``shared_edge_ids`` (empty tensor if none).
    """
    ids = shared_edge_ids.detach().long().view(-1).cpu()
    b = int(ids.numel())
    device = shared_edge_ids.device
    mask = torch.zeros((b, b), dtype=torch.bool, device=device)
    if b == 0:
        empty = torch.zeros((0,), dtype=torch.long, device=device)
        return mask, empty, {
            "n_shared": 0.0,
            "n_anchor_rows": 0.0,
            "n_identity": 0.0,
            "n_structural": 0.0,
            "n_knn": 0.0,
            "n_pos_total_mean": 0.0,
            "n_neg_mean": 0.0,
        }

    # Identity
    mask.fill_diagonal_(True)
    id_to_row = {int(e): i for i, e in enumerate(ids.tolist())}

    n_struct = 0
    n_knn = 0
    if ctx.positive_mode == "neighbor":
        for i, eid in enumerate(ids.tolist()):
            eid = int(eid)
            for j in ctx.flow_adj[eid].tolist():
                r = id_to_row.get(int(j))
                if r is not None and r != i:
                    if not bool(mask[i, r]):
                        n_struct += 1
                    mask[i, r] = True
            for j in ctx.knn_adj[eid].tolist():
                r = id_to_row.get(int(j))
                if r is not None and r != i:
                    if not bool(mask[i, r]):
                        n_knn += 1
                    mask[i, r] = True

    if anchor_ids is None:
        anchor_rows = torch.arange(b, device=device, dtype=torch.long)
    else:
        want = {int(a) for a in anchor_ids}
        rows = [id_to_row[e] for e in ids.tolist() if int(e) in want]
        anchor_rows = torch.tensor(rows, dtype=torch.long, device=device)

    n_pos = mask.float().sum(dim=1)
    if anchor_rows.numel():
        n_pos_mean = float(n_pos[anchor_rows].mean().item())
        n_neg_mean = float((~mask)[anchor_rows].float().sum(dim=1).mean().item())
    else:
        n_pos_mean = 0.0
        n_neg_mean = 0.0

    stats = {
        "n_shared": float(b),
        "n_anchor_rows": float(anchor_rows.numel()),
        "n_identity": float(b),  # one per row
        "n_structural": float(n_struct),
        "n_knn": float(n_knn),
        "n_pos_total_mean": n_pos_mean,
        "n_neg_mean": n_neg_mean,
        "positive_mode": 1.0 if ctx.positive_mode == "neighbor" else 0.0,
    }
    return mask, anchor_rows, stats


def edge_neighbor_supcon_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    shared_edge_ids: torch.Tensor,
    *,
    ctx: EdgeNeighborPositiveContext,
    anchor_ids: Sequence[int],
    temperature: float = 0.5,
    asymmetric: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """SupCon multipositive loss; averages over surviving original anchors only."""
    mask, row_indices, stats = build_edge_neighbor_positive_mask(
        shared_edge_ids, ctx=ctx, anchor_ids=anchor_ids
    )
    if row_indices.numel() == 0:
        raise RuntimeError(
            "No original anchors survived both views; cannot form neighbor-positive loss"
        )
    fwd = multipositive_infonce(
        z1,
        z2,
        mask,
        temperature=float(temperature),
        row_indices=row_indices,
        positive_aggregation=ctx.positive_aggregation,
    )
    if asymmetric:
        loss = fwd["loss"]
    else:
        mask_sym = mask | mask.T
        mask_sym.fill_diagonal_(True)
        rev = multipositive_infonce(
            z2,
            z1,
            mask_sym,
            temperature=float(temperature),
            row_indices=row_indices,
            positive_aggregation=ctx.positive_aggregation,
        )
        loss = 0.5 * (fwd["loss"] + rev["loss"])
    stats = dict(stats)
    stats["loss"] = float(loss.detach().item())
    stats["n_pos_mean_loss"] = float(fwd["n_pos_mean"].detach().item())
    stats["n_unique_negatives_mean"] = float(fwd["n_unique_negatives_mean"].detach().item())
    return loss, stats


def assert_no_labels_in_construction(ctx: EdgeNeighborPositiveContext) -> None:
    if ctx.leakage_audit.get("labels_used_in_construction"):
        raise AssertionError("labels were used in positive construction")
