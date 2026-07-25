"""One training step / short-run helpers for the txn-node baseline."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Set

import numpy as np
import torch

from gcpal_txn_node.adjacency import induce_edge_index
from gcpal_txn_node.features import GraphView, make_knn_view, make_random_structural_view
from gcpal_txn_node.loss import (
    DEFAULT_POSITIVE_AGGREGATION,
    build_positive_mask,
    mixed_gcpal_loss,
    similarity_type_diagnostics,
    validate_positive_aggregation,
)
from gcpal_txn_node.model import SharedTxnNodeEncoder
from gcpal_txn_node.sampling import (
    assert_batch_subset_of_allowed,
    local_pairs_from_adj,
    sample_positive_aware_batch,
    sample_positive_complete_batch,
)


@dataclass
class StepConfig:
    edge_drop: float = 0.1
    feature_drop: float = 0.1
    lambda_mix: float = 0.3
    temperature: float = 0.5
    include_identity: bool = True
    use_knn_contrast: bool = True
    max_struct_extras: int = 4
    max_knn_extras: int = 8
    max_total_nodes: int = 4096
    # "legacy_capped" preserves prior txn-node behavior; "positive_complete" is the new mode.
    batching_mode: str = "legacy_capped"
    knn_k: int = 15
    # Default preserves historical sum_logsumexp behavior when omitted.
    positive_aggregation: str = DEFAULT_POSITIVE_AGGREGATION


def _verify_knn_endpoints_in_batch(knn_pairs: torch.Tensor, n_nodes: int) -> None:
    if knn_pairs.numel() == 0:
        return
    if int(knn_pairs.min()) < 0 or int(knn_pairs.max()) >= n_nodes:
        raise AssertionError("KNN positive pair index out of batch range")


def _peak_mib(device: torch.device) -> Optional[float]:
    if device.type == "cuda" and torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated(device) / (1024**2))
    return None


def run_contrastive_step(
    *,
    encoder: SharedTxnNodeEncoder,
    x_all: np.ndarray,
    flow_edge_index: np.ndarray,
    flow_adj: Sequence[np.ndarray],
    knn_adj: Sequence[np.ndarray],
    knn_graph_edge_fn,
    anchor_ids: np.ndarray,
    device: torch.device,
    cfg: StepConfig,
    seed: int = 0,
    pos_flow_adj: Optional[Sequence[np.ndarray]] = None,
    pos_knn_adj: Optional[Sequence[np.ndarray]] = None,
) -> Dict[str, Any]:
    """Legacy / default step: capped positive-aware growth; contrast among anchors only.

    Preserves previous txn-node scout/smoke behavior when ``batching_mode='legacy_capped'``.
    For positive-complete minibatches use :func:`run_positive_complete_step`.
    """
    if cfg.batching_mode == "positive_complete":
        raise ValueError("Use run_positive_complete_step for batching_mode='positive_complete'")

    pos_flow_adj = flow_adj if pos_flow_adj is None else pos_flow_adj
    pos_knn_adj = knn_adj if pos_knn_adj is None else pos_knn_adj
    rng = np.random.RandomState(seed)
    t0 = time.perf_counter()
    node_ids, grow_stats = sample_positive_aware_batch(
        anchor_ids,
        pos_flow_adj,
        pos_knn_adj,
        max_struct_extras=cfg.max_struct_extras,
        max_knn_extras=cfg.max_knn_extras,
        max_total_nodes=cfg.max_total_nodes,
        rng=rng,
    )
    x = torch.from_numpy(x_all[node_ids]).to(device)
    flow_local = torch.from_numpy(induce_edge_index(flow_edge_index, node_ids)).to(device)
    knn_ei_np = knn_graph_edge_fn(node_ids)
    knn_local = torch.from_numpy(knn_ei_np).to(device)
    node_ids_t = torch.from_numpy(node_ids).to(device)

    torch.manual_seed(seed)
    v1 = make_random_structural_view(
        x, flow_local, node_ids_t, edge_drop=cfg.edge_drop, feature_drop=cfg.feature_drop, name="random1"
    )
    torch.manual_seed(seed + 1)
    v2 = make_random_structural_view(
        x, flow_local, node_ids_t, edge_drop=cfg.edge_drop, feature_drop=cfg.feature_drop, name="random2"
    )
    assert torch.equal(v1.node_ids, v2.node_ids)
    assert v1.x.shape[0] == x.shape[0]

    h1, z1 = encoder(v1.x, v1.edge_index)
    h2, z2 = encoder(v2.x, v2.edge_index)
    z_knn = None
    v_knn: Optional[GraphView] = None
    if cfg.use_knn_contrast:
        v_knn = make_knn_view(x, knn_local, node_ids_t, feature_drop=0.0)
        assert torch.equal(v_knn.node_ids, v1.node_ids)
        _hk, z_knn = encoder(v_knn.x, v_knn.edge_index)

    local_map = {int(g): i for i, g in enumerate(node_ids.tolist())}
    anchor_local = torch.tensor(
        [local_map[int(a)] for a in anchor_ids.tolist()], dtype=torch.long, device=device
    )

    z1_a = z1[anchor_local]
    z2_a = z2[anchor_local]
    z_knn_a = z_knn[anchor_local] if z_knn is not None else None

    struct_pairs = local_pairs_from_adj(pos_flow_adj, anchor_ids).to(device)
    knn_pairs = (
        local_pairs_from_adj(pos_knn_adj, anchor_ids).to(device)
        if cfg.use_knn_contrast
        else torch.zeros((2, 0), dtype=torch.long, device=device)
    )

    pos_mask, pos_stats = build_positive_mask(
        int(anchor_ids.size),
        identity=cfg.include_identity,
        structural_pairs=struct_pairs,
        knn_pairs=knn_pairs,
        device=device,
    )

    agg = validate_positive_aggregation(cfg.positive_aggregation)
    loss_out = mixed_gcpal_loss(
        z1_a,
        z2_a,
        z_knn_a,
        pos_mask,
        lambda_mix=cfg.lambda_mix,
        temperature=cfg.temperature,
        use_knn=cfg.use_knn_contrast,
        positive_aggregation=agg,
    )
    pos_stats_clean = {k: v for k, v in pos_stats.items() if not str(k).startswith("_")}

    return {
        "loss": loss_out["loss"],
        "loss_random_random": loss_out["loss_random_random"],
        "loss_random_knn": loss_out["loss_random_knn"],
        "lambda_mix": float(loss_out["lambda_mix"]),
        "positive_aggregation": agg,
        "n_anchors": int(anchor_ids.size),
        "n_sampled_nodes": int(node_ids.size),
        "n_edges_view1": int(v1.edge_index.shape[1]),
        "n_edges_view2": int(v2.edge_index.shape[1]),
        "n_edges_knn": int(v_knn.edge_index.shape[1]) if v_knn is not None else 0,
        "unique_negatives_mean": float(loss_out["n_unique_negatives_mean"].detach().cpu()),
        "positive_stats": pos_stats_clean,
        "growth_stats": grow_stats,
        "h_anchors": h1[anchor_local].detach(),
        "anchor_ids": anchor_ids,
        "node_ids": node_ids,
        "views_aligned": True,
        "step_time_seconds": time.perf_counter() - t0,
        "peak_cuda_mib": _peak_mib(device),
        "batching_mode": "legacy_capped",
    }


def run_positive_complete_step(
    *,
    encoder: SharedTxnNodeEncoder,
    x_all: np.ndarray,
    flow_edge_index: np.ndarray,
    flow_adj: Sequence[np.ndarray],
    knn_adj: Sequence[np.ndarray],
    knn_graph_edge_fn,
    candidate_stream: np.ndarray,
    stream_start: int,
    device: torch.device,
    cfg: StepConfig,
    seed: int = 0,
    pos_flow_adj: Optional[Sequence[np.ndarray]] = None,
    pos_knn_adj: Optional[Sequence[np.ndarray]] = None,
    allowed_ids: Optional[Set[int]] = None,
    labels: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Positive-complete minibatch: full global KNN+structural closures; anchor-only loss rows.

    Batch membership growth always uses ``flow_adj`` / ``knn_adj`` so control and GCPAL
    modes can share identical batch construction. Positive overrides only affect the mask.
    """
    # Growth uses full structural+KNN regardless of positive override (matched A/B batches).
    anchor_ids, node_ids, next_stream, grow_stats = sample_positive_complete_batch(
        candidate_stream,
        flow_adj,
        knn_adj,
        max_total_nodes=int(cfg.max_total_nodes),
        start=int(stream_start),
        allowed_ids=allowed_ids,
        labels=labels,
        knn_k=int(cfg.knn_k),
    )
    pos_flow_adj = flow_adj if pos_flow_adj is None else pos_flow_adj
    pos_knn_adj = knn_adj if pos_knn_adj is None else pos_knn_adj

    if allowed_ids is not None:
        assert_batch_subset_of_allowed(node_ids, allowed_ids)
        assert_batch_subset_of_allowed(anchor_ids, allowed_ids)

    t0 = time.perf_counter()
    x = torch.from_numpy(x_all[node_ids]).to(device)
    flow_local = torch.from_numpy(induce_edge_index(flow_edge_index, node_ids)).to(device)
    knn_local = torch.from_numpy(knn_graph_edge_fn(node_ids)).to(device)
    node_ids_t = torch.from_numpy(node_ids).to(device)

    torch.manual_seed(seed)
    v1 = make_random_structural_view(
        x, flow_local, node_ids_t, edge_drop=cfg.edge_drop, feature_drop=cfg.feature_drop, name="random1"
    )
    torch.manual_seed(seed + 1)
    v2 = make_random_structural_view(
        x, flow_local, node_ids_t, edge_drop=cfg.edge_drop, feature_drop=cfg.feature_drop, name="random2"
    )
    assert torch.equal(v1.node_ids, v2.node_ids)
    assert int(v1.x.shape[0]) == int(node_ids.size)

    h1, z1 = encoder(v1.x, v1.edge_index)
    h2, z2 = encoder(v2.x, v2.edge_index)
    z_knn = None
    v_knn: Optional[GraphView] = None
    if cfg.use_knn_contrast:
        v_knn = make_knn_view(x, knn_local, node_ids_t, feature_drop=0.0)
        assert torch.equal(v_knn.node_ids, v1.node_ids)
        _hk, z_knn = encoder(v_knn.x, v_knn.edge_index)

    local_map = {int(g): i for i, g in enumerate(node_ids.tolist())}
    for a in anchor_ids.tolist():
        if int(a) not in local_map:
            raise AssertionError(f"anchor {a} missing from batch node_ids")
    anchor_local = torch.tensor(
        [local_map[int(a)] for a in anchor_ids.tolist()], dtype=torch.long, device=device
    )
    struct_pairs = local_pairs_from_adj(pos_flow_adj, node_ids, sources=anchor_ids).to(device)
    knn_pairs = local_pairs_from_adj(pos_knn_adj, node_ids, sources=anchor_ids).to(device)
    _verify_knn_endpoints_in_batch(knn_pairs, int(node_ids.size))

    # Every claimed KNN positive endpoint must sit in the batch (construction invariant).
    if knn_pairs.numel():
        assert int(knn_pairs.min()) >= 0 and int(knn_pairs.max()) < int(node_ids.size)

    pos_mask, pos_stats = build_positive_mask(
        int(node_ids.size),
        identity=cfg.include_identity,
        structural_pairs=struct_pairs,
        knn_pairs=knn_pairs,
        device=device,
        anchor_local=anchor_local,
    )
    # No self-neighbor positives beyond identity on anchor rows.
    eye = torch.eye(int(node_ids.size), dtype=torch.bool, device=device)
    if knn_pairs.numel():
        knn_only = torch.zeros_like(pos_mask)
        knn_only[knn_pairs[0], knn_pairs[1]] = True
        assert not bool((knn_only & eye).any()), "KNN positives must exclude self"
    if struct_pairs.numel():
        st_only = torch.zeros_like(pos_mask)
        st_only[struct_pairs[0], struct_pairs[1]] = True
        assert not bool((st_only & eye).any()), "structural positives must exclude self"

    agg = validate_positive_aggregation(cfg.positive_aggregation)
    loss_out = mixed_gcpal_loss(
        z1,
        z2,
        z_knn,
        pos_mask,
        lambda_mix=cfg.lambda_mix,
        temperature=cfg.temperature,
        use_knn=cfg.use_knn_contrast,
        row_indices=anchor_local,
        positive_aggregation=agg,
    )
    step_s = time.perf_counter() - t0

    id_mask = pos_stats.pop("_identity_mask")
    st_mask = pos_stats.pop("_structural_mask")
    kn_mask = pos_stats.pop("_knn_mask")
    sim_diag = similarity_type_diagnostics(
        loss_out["logits_fwd"],
        positive_mask=pos_mask,
        identity_mask=id_mask,
        structural_mask=st_mask,
        knn_mask=kn_mask,
        row_indices=anchor_local,
    )
    n_pos_anchor = loss_out["n_pos_fwd"][anchor_local].detach().cpu()
    log_num_a = loss_out["log_num_fwd"][anchor_local].detach().cpu()
    log_den_a = loss_out["log_denom_fwd"][anchor_local].detach().cpu()
    z_anchor = z1[anchor_local].detach()
    h_anchor = h1[anchor_local].detach()
    emb_diag = {
        "z_norm_mean": float(z_anchor.norm(dim=1).mean().cpu()),
        "z_norm_std": float(z_anchor.norm(dim=1).std(unbiased=False).cpu()),
        "h_norm_mean": float(h_anchor.norm(dim=1).mean().cpu()),
        "h_norm_std": float(h_anchor.norm(dim=1).std(unbiased=False).cpu()),
        "h_var_mean": float(h_anchor.var(dim=0, unbiased=False).mean().cpu()),
        "z_var_mean": float(z_anchor.var(dim=0, unbiased=False).mean().cpu()),
        # Collapse proxy: mean pairwise cosine among anchors (lower diversity → higher).
        "anchor_mean_pairwise_cosine": float(
            (
                torch.nn.functional.normalize(z_anchor, dim=1)
                @ torch.nn.functional.normalize(z_anchor, dim=1).T
            )
            .fill_diagonal_(0.0)
            .mean()
            .cpu()
        ),
    }

    return {
        "loss": loss_out["loss"],
        "loss_random_random": loss_out["loss_random_random"],
        "loss_random_knn": loss_out["loss_random_knn"],
        "lambda_mix": float(loss_out["lambda_mix"]),
        "positive_aggregation": agg,
        "n_anchors": int(anchor_ids.size),
        "n_sampled_nodes": int(node_ids.size),
        "loss_pool_n": int(node_ids.size),
        "n_edges_view1": int(v1.edge_index.shape[1]),
        "n_edges_view2": int(v2.edge_index.shape[1]),
        "n_edges_knn": int(v_knn.edge_index.shape[1]) if v_knn is not None else 0,
        "unique_negatives_mean": float(loss_out["n_unique_negatives_mean"].detach().cpu()),
        "positive_stats": pos_stats,
        "growth_stats": grow_stats,
        "similarity_diagnostics": sim_diag,
        "embedding_diagnostics": emb_diag,
        "n_pos_per_anchor": n_pos_anchor,
        "log_num_per_anchor": log_num_a,
        "log_denom_per_anchor": log_den_a,
        "loss_vec_per_anchor": loss_out["loss_vec_fwd"][anchor_local].detach().cpu(),
        "h_anchors": h_anchor,
        "anchor_ids": anchor_ids,
        "node_ids": node_ids,
        "stream_next": int(next_stream),
        "views_aligned": True,
        "step_time_seconds": step_s,
        "peak_cuda_mib": _peak_mib(device),
        "batching_mode": "positive_complete",
    }
