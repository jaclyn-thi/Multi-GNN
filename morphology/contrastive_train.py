"""
Morphology integration for contrastive training loops (M1 / M1b / M2).

This module is the **glue** between ``training.py`` and the morphology package.
It does not define metrics or losses itself; it orchestrates:

- Per-step expert MSE (``morph_expert_loss_*`` → ``morphology.expert``)
- Per-step contrast bin ids (``morph_contrast_bin_ids_*`` → ``morphology.contrast``)
- End-of-epoch val metrics on the **val split** loader (no AML labels)
- Throttling via ``--morph_val_every`` / ``--morph_val_max_batches``

Val passes mirror the train step: two augmented views, GNN forward(s), then either
expert MSE or merged InfoNCE with morphology soft positives. Each val batch is
expensive (similar cost to a train batch without backward).
"""

from __future__ import annotations

from typing import Optional

import torch
from torch_geometric.data import Data, HeteroData

from morphology.contrast import MorphContrastConfig, build_morph_bin_ids_for_seeds
from morphology.expert import MorphExpertConfig, MorphExpertHead, morphology_expert_step
from morphology.tier0_global import MorphTier0Context
from morphology.tier2_global import MorphTier2Context
from train_util import FORWARD_EDGE_TYPE


def should_run_morph_val(epoch: int, n_epochs: int, args) -> bool:
    """
    Whether to run ``morph/expert_val`` and ``morph/contrast_val`` at epoch end.

    Parameters
    ----------
    epoch :
        Zero-based epoch index inside the training loop.
    n_epochs :
        Total epochs (``config.epochs``).
    args :
        Parsed CLI namespace; reads ``morph_val_every`` (default 1 = every epoch).

    Returns
    -------
    bool
        True if morphology val should run. Always True when ``morph_val_every <= 1``,
        on the final epoch, or when ``epoch % morph_val_every == 0``.
    """
    every = int(getattr(args, "morph_val_every", 1))
    if every <= 1:
        return True
    if epoch >= n_epochs - 1:
        return True
    return (epoch % every) == 0


def morph_val_max_batches(args) -> Optional[int]:
    """
    Cap on val-loader batches for morphology val passes.

    Parameters
    ----------
    args :
        Parsed CLI namespace; reads ``morph_val_max_batches`` (0 = no cap).

    Returns
    -------
    int or None
        Maximum batches to iterate, or None for a full val pass.
    """
    n = int(getattr(args, "morph_val_max_batches", 0))
    return n if n > 0 else None


def _edge_native_dim_for_contrast(args, tr_data, is_hetero: bool) -> int:
    """Number of edge_attr columns excluding synthetic EdgeID (column 0)."""
    if not getattr(args, "morph_contrast_cfg", None):
        return 0
    cfg: MorphContrastConfig = args.morph_contrast_cfg
    if not cfg.include_edge_native:
        return 0
    from morphology.graph_access import get_forward_edge_attr

    attr = get_forward_edge_attr(tr_data) if is_hetero else tr_data.edge_attr
    if attr is None:
        return 0
    return int(attr.shape[1]) - 1


def morph_contrast_bin_ids_hetero(
    view1: HeteroData,
    seed_edge_ids: torch.Tensor,
    batch: HeteroData,
    cfg: MorphContrastConfig,
    tier0_ctx: Optional[MorphTier0Context],
    edge_native_dim: int,
) -> torch.Tensor:
    """
    Assign morphology contrast bin ids for shared seed edges (hetero path).

    Bins are computed from **view1** forward-subgraph features (detached).
    Used as ``morph_bin_ids`` in ``edge_identity_infonce_loss``.

    Returns
    -------
    Tensor
        Long tensor of shape ``(n_shared_seeds,)``, one bin id per aligned seed.
    """
    store = view1[FORWARD_EDGE_TYPE]
    return build_morph_bin_ids_for_seeds(
        store.edge_index,
        store.edge_id,
        seed_edge_ids,
        store.edge_attr,
        int(batch["node"].num_nodes),
        cfg,
        tier0_ctx=tier0_ctx,
        edge_native_dim=edge_native_dim,
    )


def morph_contrast_bin_ids_homo(
    view1: Data,
    seed_edge_ids: torch.Tensor,
    batch: Data,
    cfg: MorphContrastConfig,
    tier0_ctx: Optional[MorphTier0Context],
    edge_native_dim: int,
) -> torch.Tensor:
    """Homogeneous-graph variant of ``morph_contrast_bin_ids_hetero``."""
    n_nodes = int(batch.num_nodes) if getattr(batch, "num_nodes", None) is not None else int(batch.x.shape[0])
    return build_morph_bin_ids_for_seeds(
        view1.edge_index,
        view1.edge_id,
        seed_edge_ids,
        view1.edge_attr,
        n_nodes,
        cfg,
        tier0_ctx=tier0_ctx,
        edge_native_dim=edge_native_dim,
    )


@torch.no_grad()
def eval_morph_expert_val_hetero(
    val_loader,
    model: torch.nn.Module,
    expert_head: MorphExpertHead,
    cfg: MorphExpertConfig,
    device: torch.device,
    args,
) -> float:
    """
    Mean morphology expert MSE over the val loader (hetero).

    Uses val-split Tier 0 context when ``cfg.include_global`` (no train leakage).
    Respects ``morph_val_max_batches`` via ``args``.

    Returns
    -------
    float
        Average expert loss per val batch (unweighted MSE scale; already includes
        ``cfg.loss_weight`` from ``morphology_expert_step``).
    """
    from train_util import attach_edge_id_from_batch, get_hetero_seed_edge_ids, select_shared_seed_edge_embeddings
    from graph_augmentations import generate_views

    tier0_ctx: Optional[MorphTier0Context] = getattr(args, "morph_tier0_val", None)
    if cfg.include_global and tier0_ctx is None:
        raise ValueError("morph_tier0_val missing for local+global morphology eval")
    tier2_ctx: Optional[MorphTier2Context] = getattr(args, "morph_tier2_val", None)
    if cfg.include_tier2 and tier2_ctx is None:
        raise ValueError("morph_tier2_val missing for local+global+tier2 morphology eval")

    model.eval()
    expert_head.eval()
    loss_sum = torch.zeros((), device=device)
    n_batches = 0
    max_batches = morph_val_max_batches(args)
    for batch in val_loader:
        seed_edge_ids = get_hetero_seed_edge_ids(batch, val_loader.data)
        attach_edge_id_from_batch(batch, val_loader.data)
        batch = batch.to(device)
        seed_edge_ids = seed_edge_ids.to(device)
        view1, view2 = generate_views(
            batch,
            edge_attr_mask_rate=0.1,
            edge_drop_rate=0.1,
            mask_value=0.0,
            mask_cols=None,
            exclude_last_column=(args.model == "rgcn"),
        )
        with torch.no_grad():
            out1 = model(view1.x_dict, view1.edge_index_dict, view1.edge_attr_dict)
            z1 = out1[FORWARD_EDGE_TYPE]
            out2 = model(view2.x_dict, view2.edge_index_dict, view2.edge_attr_dict)
            z2 = out2[FORWARD_EDGE_TYPE]
        z1_seed, seed_id1, _, _ = select_shared_seed_edge_embeddings(
            z1, view1[FORWARD_EDGE_TYPE].edge_id, z2, view2[FORWARD_EDGE_TYPE].edge_id, seed_edge_ids
        )
        store = view1[FORWARD_EDGE_TYPE]
        loss, _ = morphology_expert_step(
            z1_seed,
            seed_id1,
            store.edge_index,
            store.edge_id,
            store.edge_attr,
            int(batch["node"].num_nodes),
            expert_head,
            cfg,
            tier0_ctx=tier0_ctx,
            tier2_ctx=tier2_ctx,
        )
        loss_sum = loss_sum + loss.detach()
        n_batches += 1
        if max_batches is not None and n_batches >= max_batches:
            break
    model.train()
    expert_head.train()
    if n_batches == 0:
        return 0.0
    return float((loss_sum / n_batches).cpu())


@torch.no_grad()
def eval_morph_expert_val_homo(
    val_loader,
    model: torch.nn.Module,
    expert_head: MorphExpertHead,
    cfg: MorphExpertConfig,
    device: torch.device,
    args,
) -> float:
    """Homogeneous-graph variant of ``eval_morph_expert_val_hetero``."""
    from train_util import attach_edge_id_from_batch, get_homo_seed_edge_ids, select_shared_seed_edge_embeddings
    from graph_augmentations import generate_views

    tier0_ctx: Optional[MorphTier0Context] = getattr(args, "morph_tier0_val", None)
    if cfg.include_global and tier0_ctx is None:
        raise ValueError("morph_tier0_val missing for local+global morphology eval")
    tier2_ctx: Optional[MorphTier2Context] = getattr(args, "morph_tier2_val", None)
    if cfg.include_tier2 and tier2_ctx is None:
        raise ValueError("morph_tier2_val missing for local+global+tier2 morphology eval")

    model.eval()
    expert_head.eval()
    loss_sum = torch.zeros((), device=device)
    n_batches = 0
    max_batches = morph_val_max_batches(args)
    for batch in val_loader:
        seed_edge_ids = get_homo_seed_edge_ids(batch, val_loader.data)
        attach_edge_id_from_batch(batch)
        batch = batch.to(device)
        seed_edge_ids = seed_edge_ids.to(device)
        view1, view2 = generate_views(
            batch,
            edge_attr_mask_rate=0.1,
            edge_drop_rate=0.1,
            mask_value=0.0,
            mask_cols=None,
            exclude_last_column=(args.model == "rgcn"),
        )
        with torch.no_grad():
            z1 = model(view1.x, view1.edge_index, view1.edge_attr)
            z2 = model(view2.x, view2.edge_index, view2.edge_attr)
        z1_seed, seed_id1, _, _ = select_shared_seed_edge_embeddings(
            z1, view1.edge_id, z2, view2.edge_id, seed_edge_ids
        )
        n_nodes = int(batch.num_nodes) if getattr(batch, "num_nodes", None) is not None else int(batch.x.shape[0])
        loss, _ = morphology_expert_step(
            z1_seed,
            seed_id1,
            view1.edge_index,
            view1.edge_id,
            view1.edge_attr,
            n_nodes,
            expert_head,
            cfg,
            tier0_ctx=tier0_ctx,
            tier2_ctx=tier2_ctx,
        )
        loss_sum = loss_sum + loss.detach()
        n_batches += 1
        if max_batches is not None and n_batches >= max_batches:
            break
    model.train()
    expert_head.train()
    if n_batches == 0:
        return 0.0
    return float((loss_sum / n_batches).cpu())


def morph_expert_loss_hetero_step(
    view1: HeteroData,
    z1_seed: torch.Tensor,
    seed_id1: torch.Tensor,
    batch: HeteroData,
    expert_head: MorphExpertHead,
    cfg: MorphExpertConfig,
    tier0_ctx: Optional[MorphTier0Context] = None,
    tier2_ctx: Optional[MorphTier2Context] = None,
) -> torch.Tensor:
    """
    Single training-step morphology expert loss (hetero).

    Targets are built from **view1** subgraph features (detached). Uses train-split
    ``tier0_ctx`` / ``tier2_ctx`` when global / Tier 2 targets are enabled.
    """
    store = view1[FORWARD_EDGE_TYPE]
    loss, _ = morphology_expert_step(
        z1_seed,
        seed_id1,
        store.edge_index,
        store.edge_id,
        store.edge_attr,
        int(batch["node"].num_nodes),
        expert_head,
        cfg,
        tier0_ctx=tier0_ctx,
        tier2_ctx=tier2_ctx,
    )
    return loss


def morph_expert_loss_homo_step(
    view1: Data,
    z1_seed: torch.Tensor,
    seed_id1: torch.Tensor,
    batch: Data,
    expert_head: MorphExpertHead,
    cfg: MorphExpertConfig,
    tier0_ctx: Optional[MorphTier0Context] = None,
    tier2_ctx: Optional[MorphTier2Context] = None,
) -> torch.Tensor:
    """Homogeneous-graph variant of ``morph_expert_loss_hetero_step``."""
    n_nodes = int(batch.num_nodes) if getattr(batch, "num_nodes", None) is not None else int(batch.x.shape[0])
    loss, _ = morphology_expert_step(
        z1_seed,
        seed_id1,
        view1.edge_index,
        view1.edge_id,
        view1.edge_attr,
        n_nodes,
        expert_head,
        cfg,
        tier0_ctx=tier0_ctx,
        tier2_ctx=tier2_ctx,
    )
    return loss


@torch.no_grad()
def eval_morph_contrast_val_hetero(
    val_loader,
    model: torch.nn.Module,
    cfg: MorphContrastConfig,
    device: torch.device,
    args,
) -> float:
    """
    Mean merged InfoNCE loss (identity + morphology soft positives) on val (hetero).

    Mirrors train contrastive settings: asymmetric flag, neg subsample, soft-positive cap.
    Logged as ``morph/contrast_val`` in W&B.
    """
    from contrastive_loss import edge_identity_infonce_loss
    from contrastive_projection import project_seed_pair
    from graph_augmentations import generate_views
    from train_util import attach_edge_id_from_batch, get_hetero_seed_edge_ids, select_shared_seed_edge_embeddings

    tier0_ctx: Optional[MorphTier0Context] = getattr(args, "morph_tier0_val", None)
    if cfg.include_global and tier0_ctx is None:
        raise ValueError("morph_tier0_val missing for local+global morphology contrast eval")

    edge_native_dim = int(getattr(args, "_morph_contrast_edge_native_dim", 0))
    neg_kw = int(getattr(args, "contrastive_num_neg_samples", 0))
    num_neg_samples = neg_kw if neg_kw > 0 else None
    contrastive_symmetric = not bool(getattr(args, "contrastive_asymmetric", False))
    max_soft = int(cfg.max_soft_positives)
    if max_soft <= 0:
        max_soft = None

    model.eval()
    proj_head = getattr(args, "contrast_projection_module", None)
    if proj_head is not None:
        proj_head.eval()
    loss_sum = torch.zeros((), device=device)
    n_batches = 0
    max_batches = morph_val_max_batches(args)
    for batch in val_loader:
        seed_edge_ids = get_hetero_seed_edge_ids(batch, val_loader.data)
        attach_edge_id_from_batch(batch, val_loader.data)
        batch = batch.to(device)
        seed_edge_ids = seed_edge_ids.to(device)
        view1, view2 = generate_views(
            batch,
            edge_attr_mask_rate=0.1,
            edge_drop_rate=0.1,
            mask_value=0.0,
            mask_cols=None,
            exclude_last_column=(args.model == "rgcn"),
        )
        with torch.no_grad():
            out1 = model(view1.x_dict, view1.edge_index_dict, view1.edge_attr_dict)
            z1 = out1[FORWARD_EDGE_TYPE]
            out2 = model(view2.x_dict, view2.edge_index_dict, view2.edge_attr_dict)
            z2 = out2[FORWARD_EDGE_TYPE]
        z1_seed, seed_id1, z2_seed, seed_id2 = select_shared_seed_edge_embeddings(
            z1, view1[FORWARD_EDGE_TYPE].edge_id, z2, view2[FORWARD_EDGE_TYPE].edge_id, seed_edge_ids
        )
        morph_bins = morph_contrast_bin_ids_hetero(
            view1, seed_id1, batch, cfg, tier0_ctx, edge_native_dim
        )
        proj_head = getattr(args, "contrast_projection_module", None)
        z1_con, z2_con = project_seed_pair(proj_head, z1_seed, z2_seed)
        loss = edge_identity_infonce_loss(
            z1_con,
            z2_con,
            seed_id1,
            seed_id2,
            temperature=0.5,
            num_neg_samples=num_neg_samples,
            symmetric=contrastive_symmetric,
            morph_bin_ids=morph_bins,
            max_soft_positives=max_soft,
        )
        loss_sum = loss_sum + loss.detach()
        n_batches += 1
        if max_batches is not None and n_batches >= max_batches:
            break
    model.train()
    if proj_head is not None:
        proj_head.train()
    if n_batches == 0:
        return 0.0
    return float((loss_sum / n_batches).cpu())


@torch.no_grad()
def eval_morph_contrast_val_homo(
    val_loader,
    model: torch.nn.Module,
    cfg: MorphContrastConfig,
    device: torch.device,
    args,
) -> float:
    """Homogeneous-graph variant of ``eval_morph_contrast_val_hetero``."""
    from contrastive_loss import edge_identity_infonce_loss
    from contrastive_projection import project_seed_pair
    from graph_augmentations import generate_views
    from train_util import attach_edge_id_from_batch, get_homo_seed_edge_ids, select_shared_seed_edge_embeddings

    tier0_ctx: Optional[MorphTier0Context] = getattr(args, "morph_tier0_val", None)
    if cfg.include_global and tier0_ctx is None:
        raise ValueError("morph_tier0_val missing for local+global morphology contrast eval")

    edge_native_dim = int(getattr(args, "_morph_contrast_edge_native_dim", 0))
    neg_kw = int(getattr(args, "contrastive_num_neg_samples", 0))
    num_neg_samples = neg_kw if neg_kw > 0 else None
    contrastive_symmetric = not bool(getattr(args, "contrastive_asymmetric", False))
    max_soft = int(cfg.max_soft_positives)
    if max_soft <= 0:
        max_soft = None

    model.eval()
    proj_head = getattr(args, "contrast_projection_module", None)
    if proj_head is not None:
        proj_head.eval()
    loss_sum = torch.zeros((), device=device)
    n_batches = 0
    max_batches = morph_val_max_batches(args)
    for batch in val_loader:
        seed_edge_ids = get_homo_seed_edge_ids(batch, val_loader.data)
        attach_edge_id_from_batch(batch)
        batch = batch.to(device)
        seed_edge_ids = seed_edge_ids.to(device)
        view1, view2 = generate_views(
            batch,
            edge_attr_mask_rate=0.1,
            edge_drop_rate=0.1,
            mask_value=0.0,
            mask_cols=None,
            exclude_last_column=(args.model == "rgcn"),
        )
        with torch.no_grad():
            z1 = model(view1.x, view1.edge_index, view1.edge_attr)
            z2 = model(view2.x, view2.edge_index, view2.edge_attr)
        z1_seed, seed_id1, z2_seed, seed_id2 = select_shared_seed_edge_embeddings(
            z1, view1.edge_id, z2, view2.edge_id, seed_edge_ids
        )
        morph_bins = morph_contrast_bin_ids_homo(
            view1, seed_id1, batch, cfg, tier0_ctx, edge_native_dim
        )
        proj_head = getattr(args, "contrast_projection_module", None)
        z1_con, z2_con = project_seed_pair(proj_head, z1_seed, z2_seed)
        loss = edge_identity_infonce_loss(
            z1_con,
            z2_con,
            seed_id1,
            seed_id2,
            temperature=0.5,
            num_neg_samples=num_neg_samples,
            symmetric=contrastive_symmetric,
            morph_bin_ids=morph_bins,
            max_soft_positives=max_soft,
        )
        loss_sum = loss_sum + loss.detach()
        n_batches += 1
        if max_batches is not None and n_batches >= max_batches:
            break
    model.train()
    if proj_head is not None:
        proj_head.train()
    if n_batches == 0:
        return 0.0
    return float((loss_sum / n_batches).cpu())
