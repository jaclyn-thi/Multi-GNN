"""Phase M2: morphology-aware contrast (binned soft positives merged into InfoNCE).

Workflow
--------
1. At startup, sample morphology features from the **train** loader and estimate
   per-dimension quantile edges (``setup_morph_contrast_bin_edges``).
2. Each train step, compute detached features on **view1** for shared seeds and
   map each seed to an integer bin id (``build_morph_bin_ids_for_seeds``).
3. ``contrastive_loss.edge_identity_infonce_loss`` treats same-bin cross-view
   pairs as **additional positives** in the InfoNCE numerator (no dense B×B mask).

Feature groups (``--morph_contrast_features``) select subsets of the same target
vector used by the morphology expert (local / global lift / edge-native).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import torch

from morphology.expert import MorphExpertConfig, build_morph_targets
from morphology.tier0_global import MorphTier0Context
from morphology.tier1_local import (
    LOCAL_CLUSTERING_INDICES,
    LOCAL_DEGREE_INDICES,
    LOCAL_FEATURE_NAMES,
    LOCAL_TRIANGLES_INDICES,
)

# Feature groups for binning (toggle via CLI).
MORPH_CONTRAST_FEATURE_GROUPS = (
    "local_ego",
    "local_degree",
    "local_clustering",
    "local_triangles",
    "global_degree",
    "edge_native",
)

_LOCAL_EGO_INDICES = (0, 1)  # n_edges_sub, n_nodes_sub


@dataclass
class MorphContrastConfig:
    """
    Configuration for morphology-bin soft positives in edge InfoNCE.

    Attributes
    ----------
    feature_groups :
        Subset of ``MORPH_CONTRAST_FEATURE_GROUPS`` to bin on.
    include_global :
        Whether Tier 0 endpoint lift features are available for binning.
    include_edge_native :
        Whether forward ``edge_attr`` columns (excl. EdgeID) are binned.
    bins_per_dim :
        Number of quantile buckets per selected feature dimension.
    bin_edges :
        Train-split quantile thresholds, shape ``(n_dims, bins_per_dim - 1)``.
        Set at startup; must be present before training steps.
    max_soft_positives :
        Cap on same-bin positives per anchor in the InfoNCE numerator (0 in CLI
        means no cap; config stores a positive default).
    """

    feature_groups: Tuple[str, ...] = ("local_ego", "local_degree")
    include_global: bool = False
    include_edge_native: bool = False
    bins_per_dim: int = 5
    # (n_dims, bins_per_dim) train-split quantile edges; set at startup.
    bin_edges: Optional[torch.Tensor] = None
    max_soft_positives: int = 256  # cap same-bin positives per anchor (numerator only)


def parse_morph_contrast_features(spec: str) -> Tuple[str, ...]:
    """
    Parse ``--morph_contrast_features`` comma-separated group names.

    Parameters
    ----------
    spec :
        e.g. ``"local_ego,local_degree"``.

    Returns
    -------
    tuple of str
        Lowercase group names, validated against ``MORPH_CONTRAST_FEATURE_GROUPS``.
    """
    raw = [s.strip().lower() for s in spec.split(",") if s.strip()]
    if not raw:
        raise ValueError("morph_contrast_features must list at least one group.")
    unknown = set(raw) - set(MORPH_CONTRAST_FEATURE_GROUPS)
    if unknown:
        raise ValueError(
            f"Unknown morph_contrast_features {unknown}; "
            f"choose from {MORPH_CONTRAST_FEATURE_GROUPS}."
        )
    return tuple(raw)


def _morph_contrast_scope(args) -> str:
    return str(getattr(args, "morph_contrast_scope", "local")).lower()


def morph_contrast_config_from_args(args) -> MorphContrastConfig:
    groups = parse_morph_contrast_features(getattr(args, "morph_contrast_features", "local_ego,local_degree"))
    scope = _morph_contrast_scope(args)
    if scope not in ("local", "local+global"):
        raise ValueError(f"--morph_contrast_scope {scope!r} invalid; use 'local' or 'local+global'.")
    include_global = scope == "local+global" or "global_degree" in groups
    if "global_degree" in groups and scope == "local":
        logging.warning(
            "morph_contrast_features includes global_degree but scope=local; "
            "enable --morph_contrast_scope local+global or drop global_degree."
        )
    return MorphContrastConfig(
        feature_groups=groups,
        include_global=include_global,
        include_edge_native="edge_native" in groups,
        bins_per_dim=max(2, int(getattr(args, "morph_contrast_bins", 5))),
        max_soft_positives=max(1, int(getattr(args, "morph_contrast_max_soft_positives", 256))),
    )


def setup_morphology_contrast(args, device: torch.device) -> Optional[MorphContrastConfig]:
    """
    Build ``MorphContrastConfig`` when ``--morph_contrast`` is set.

    Does not estimate bin edges; call ``setup_morph_contrast_bin_edges`` after
    the train loader exists.

    Returns
    -------
    MorphContrastConfig or None
    """
    if not getattr(args, "morph_contrast", False):
        return None
    if str(getattr(args, "objective", "contrastive")) != "contrastive":
        raise ValueError("--morph_contrast requires --objective contrastive.")
    cfg = morph_contrast_config_from_args(args)
    edges = getattr(args, "morph_contrast_bin_edges", None)
    if edges is not None:
        cfg.bin_edges = edges.to(device)
    logging.info(
        "Morphology contrast (M2): groups=%s scope_bins=%d bins_per_dim=%d include_global=%s edge_native=%s",
        cfg.feature_groups,
        0 if cfg.bin_edges is None else cfg.bin_edges.shape[0],
        cfg.bins_per_dim,
        cfg.include_global,
        cfg.include_edge_native,
    )
    return cfg


def _column_indices_for_groups(cfg: MorphContrastConfig, edge_native_dim: int = 0) -> List[int]:
    """Indices into the full morph target vector used for binning."""
    cols: List[int] = []
    offset = 0
    n_local = len(LOCAL_FEATURE_NAMES)

    if "local_ego" in cfg.feature_groups:
        cols.extend(offset + i for i in _LOCAL_EGO_INDICES)
    if "local_degree" in cfg.feature_groups:
        cols.extend(offset + i for i in LOCAL_DEGREE_INDICES)
    if "local_clustering" in cfg.feature_groups:
        cols.extend(offset + i for i in LOCAL_CLUSTERING_INDICES)
    if "local_triangles" in cfg.feature_groups:
        cols.extend(offset + i for i in LOCAL_TRIANGLES_INDICES)
    offset += n_local

    if cfg.include_global and "global_degree" in cfg.feature_groups:
        from morphology.tier0_global import DEFAULT_LIFT_FEATURE_NAMES

        cols.extend(offset + i for i in range(len(DEFAULT_LIFT_FEATURE_NAMES)))
        offset += len(DEFAULT_LIFT_FEATURE_NAMES)

    if cfg.include_edge_native and "edge_native" in cfg.feature_groups:
        if edge_native_dim <= 0:
            raise ValueError("edge_native contrast features require edge_native_dim > 0.")
        cols.extend(offset + i for i in range(edge_native_dim))

    if not cols:
        raise ValueError("No morphology contrast columns selected.")
    return cols


def build_morph_features_for_contrast(
    edge_index: torch.Tensor,
    subgraph_edge_ids: torch.Tensor,
    seed_edge_ids: torch.Tensor,
    seed_edge_attr: Optional[torch.Tensor],
    num_nodes: int,
    cfg: MorphContrastConfig,
    tier0_ctx: Optional[MorphTier0Context] = None,
    edge_native_dim: int = 0,
) -> torch.Tensor:
    """Detached (n_seeds, F_sel) features for binning (aligned with seed_edge_ids order)."""
    expert_cfg = MorphExpertConfig(
        include_edge_native=cfg.include_edge_native,
        include_global=cfg.include_global,
    )
    full = build_morph_targets(
        edge_index,
        subgraph_edge_ids,
        seed_edge_ids,
        seed_edge_attr,
        num_nodes,
        expert_cfg,
        tier0_ctx=tier0_ctx,
    )
    col_idx = _column_indices_for_groups(cfg, edge_native_dim=edge_native_dim)
    return full[:, col_idx].detach()


@torch.no_grad()
def assign_morph_bin_ids(
    features: torch.Tensor,
    cfg: MorphContrastConfig,
) -> torch.Tensor:
    """
    Map each seed row to an integer bin id (mixed-radix over per-dim buckets).

    Uses train-split ``cfg.bin_edges`` from ``torch.quantile`` (shape ``(D, K-1)``).
    """
    if features.numel() == 0:
        return features.new_empty((0,), dtype=torch.long)
    if cfg.bin_edges is None:
        raise ValueError("MorphContrastConfig.bin_edges must be set before training.")

    features = features.contiguous()
    edges = cfg.bin_edges.to(features.device, features.dtype)
    if edges.dim() != 2 or edges.shape[0] != features.shape[1]:
        raise ValueError(
            f"bin_edges shape {tuple(edges.shape)} incompatible with features {tuple(features.shape)}"
        )
    n_bins = cfg.bins_per_dim
    multiplier = 1
    out = torch.zeros(features.shape[0], device=features.device, dtype=torch.long)
    for d in range(features.shape[1]):
        digit = torch.bucketize(features[:, d], edges[d], right=False).clamp(0, n_bins - 1)
        out = out + digit.long() * multiplier
        multiplier *= n_bins
    return out


@torch.no_grad()
def estimate_morph_bin_edges(
    feature_batches: Sequence[torch.Tensor],
    bins_per_dim: int,
) -> torch.Tensor:
    """Train-split quantile edges per feature column (shape ``(D, bins_per_dim - 1)``)."""
    if not feature_batches:
        raise ValueError("Need at least one feature batch to estimate bin edges.")
    stacked = torch.cat([f.detach().float().cpu() for f in feature_batches if f.numel() > 0], dim=0)
    if stacked.shape[0] < 2:
        raise ValueError("Too few morphology contrast samples to estimate bin edges.")
    n_bins = max(2, int(bins_per_dim))
    qs = torch.linspace(0.0, 1.0, n_bins + 1, device=stacked.device)[1:-1]
    edges = torch.quantile(stacked, qs, dim=0).T.contiguous()
    # Strictly increasing edges for bucketize stability.
    for d in range(edges.shape[0]):
        col = edges[d]
        eps = torch.tensor(1e-6, dtype=col.dtype)
        for i in range(1, col.numel()):
            if col[i] <= col[i - 1]:
                col[i] = col[i - 1] + eps
        edges[d] = col
    return edges


@torch.no_grad()
def collect_morph_contrast_feature_samples(
    tr_loader,
    model: torch.nn.Module,
    cfg: MorphContrastConfig,
    device: torch.device,
    args,
    *,
    max_batches: int = 32,
    is_hetero: bool,
) -> List[torch.Tensor]:
    """Sample morphology contrast features from train loader (view1, no grad)."""
    from graph_augmentations import generate_views
    from morphology.graph_access import get_forward_edge_attr
    from train_util import (
        FORWARD_EDGE_TYPE,
        attach_edge_id_from_batch,
        get_hetero_seed_edge_ids,
        get_homo_seed_edge_ids,
        select_shared_seed_edge_embeddings,
    )

    tier0_ctx = getattr(args, "morph_tier0_train", None)
    if cfg.include_global and tier0_ctx is None:
        raise ValueError("morph_tier0_train required for global morphology contrast scope.")

    edge_native_dim = 0
    if cfg.include_edge_native:
        data = tr_loader.data
        attr = get_forward_edge_attr(data) if is_hetero else data.edge_attr
        if attr is None:
            raise ValueError("edge_attr required for edge_native morphology contrast.")
        edge_native_dim = int(attr.shape[1]) - 1
        cfg._edge_native_dim = edge_native_dim  # type: ignore[attr-defined]

    batches: List[torch.Tensor] = []
    model.eval()
    n = 0
    for batch in tr_loader:
        if n >= max_batches:
            break
        if is_hetero:
            seed_edge_ids = get_hetero_seed_edge_ids(batch, tr_loader.data)
            attach_edge_id_from_batch(batch, tr_loader.data)
            batch = batch.to(device)
            seed_edge_ids = seed_edge_ids.to(device)
            view1, _ = generate_views(
                batch,
                edge_attr_mask_rate=0.1,
                edge_drop_rate=0.1,
                mask_value=0.0,
                mask_cols=None,
                exclude_last_column=(args.model == "rgcn"),
            )
            store = view1[FORWARD_EDGE_TYPE]
            feats = build_morph_features_for_contrast(
                store.edge_index,
                store.edge_id,
                seed_edge_ids,
                store.edge_attr,
                int(batch["node"].num_nodes),
                cfg,
                tier0_ctx=tier0_ctx,
                edge_native_dim=edge_native_dim,
            )
        else:
            seed_edge_ids = get_homo_seed_edge_ids(batch, tr_loader.data)
            attach_edge_id_from_batch(batch)
            batch = batch.to(device)
            seed_edge_ids = seed_edge_ids.to(device)
            view1, _ = generate_views(
                batch,
                edge_attr_mask_rate=0.1,
                edge_drop_rate=0.1,
                mask_value=0.0,
                mask_cols=None,
                exclude_last_column=(args.model == "rgcn"),
            )
            n_nodes = int(batch.num_nodes) if getattr(batch, "num_nodes", None) is not None else int(batch.x.shape[0])
            feats = build_morph_features_for_contrast(
                view1.edge_index,
                view1.edge_id,
                seed_edge_ids,
                view1.edge_attr,
                n_nodes,
                cfg,
                tier0_ctx=tier0_ctx,
                edge_native_dim=edge_native_dim,
            )
        if feats.numel() > 0:
            batches.append(feats)
        n += 1
    model.train()
    return batches


def setup_morph_contrast_bin_edges(
    args,
    tr_loader,
    model: torch.nn.Module,
    device: torch.device,
    is_hetero: bool,
) -> None:
    """Estimate train-split quantile bin edges and store on args + config."""
    cfg: Optional[MorphContrastConfig] = getattr(args, "morph_contrast_cfg", None)
    if cfg is None:
        return
    if cfg.include_global and not getattr(args, "morph_tier0_train", None):
        from morphology.expert import setup_morph_tier0_contexts

        setup_morph_tier0_contexts(args, tr_loader.data, getattr(args, "_val_data_for_morph", None), device)

    samples = collect_morph_contrast_feature_samples(
        tr_loader,
        model,
        cfg,
        device,
        args,
        max_batches=int(getattr(args, "morph_contrast_calib_batches", 32)),
        is_hetero=is_hetero,
    )
    edges = estimate_morph_bin_edges(samples, cfg.bins_per_dim)
    cfg.bin_edges = edges.to(device)
    args.morph_contrast_bin_edges = cfg.bin_edges
    logging.info(
        "Morphology contrast bin edges: shape=%s (from %d calibration batches)",
        tuple(edges.shape),
        min(int(getattr(args, "morph_contrast_calib_batches", 32)), len(samples)),
    )


def build_morph_bin_ids_for_seeds(
    edge_index: torch.Tensor,
    subgraph_edge_ids: torch.Tensor,
    seed_edge_ids: torch.Tensor,
    seed_edge_attr: Optional[torch.Tensor],
    num_nodes: int,
    cfg: MorphContrastConfig,
    tier0_ctx: Optional[MorphTier0Context] = None,
    edge_native_dim: int = 0,
) -> torch.Tensor:
    """
    Integer bin id per seed edge for M2 InfoNCE soft positives.

    Seeds missing from the batch subgraph are dropped; callers must align
    embeddings with the returned bin tensor row order (shared-seed path only).

    Parameters
    ----------
    edge_index, subgraph_edge_ids, seed_edge_ids, seed_edge_attr, num_nodes :
        View1 forward subgraph and seed identifiers (same contract as expert targets).
    cfg :
        Must have ``bin_edges`` populated.
    tier0_ctx :
        Train- or val-split Tier 0 context when global features are enabled.
    edge_native_dim :
        Number of edge_attr columns excluding synthetic EdgeID.

    Returns
    -------
    Tensor
        Long tensor ``(n_seeds,)`` with mixed-radix bin ids (>= 0).
    """
    from morphology.tier1_local import _seed_positions_in_subgraph

    positions, valid = _seed_positions_in_subgraph(seed_edge_ids, subgraph_edge_ids)
    if not valid.any():
        return seed_edge_ids.new_empty((0,), dtype=torch.long)
    ids_use = seed_edge_ids[valid]
    feats = build_morph_features_for_contrast(
        edge_index,
        subgraph_edge_ids,
        ids_use,
        seed_edge_attr,
        num_nodes,
        cfg,
        tier0_ctx=tier0_ctx,
        edge_native_dim=edge_native_dim,
    )
    bins = assign_morph_bin_ids(feats, cfg)
    if valid.all():
        return bins
    out = seed_edge_ids.new_full((seed_edge_ids.shape[0],), -1, dtype=torch.long)
    out[valid] = bins
    if (out < 0).any():
        raise ValueError("Some seed edges lack morphology contrast bins in the batch subgraph.")
    return out


__all__ = [
    "MORPH_CONTRAST_FEATURE_GROUPS",
    "MorphContrastConfig",
    "assign_morph_bin_ids",
    "build_morph_bin_ids_for_seeds",
    "build_morph_features_for_contrast",
    "estimate_morph_bin_edges",
    "morph_contrast_config_from_args",
    "parse_morph_contrast_features",
    "setup_morph_contrast_bin_edges",
    "setup_morphology_contrast",
]
