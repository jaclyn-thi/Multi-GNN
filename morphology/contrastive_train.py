"""Morphology expert integration for contrastive training loops."""

from __future__ import annotations

from typing import Optional

import torch
from torch_geometric.data import Data, HeteroData

from morphology.expert import MorphExpertConfig, MorphologyExpertHead, morphology_expert_step
from morphology.tier0_global import MorphTier0Context
from train_util import FORWARD_EDGE_TYPE


@torch.no_grad()
def eval_morph_expert_val_hetero(
    val_loader,
    model: torch.nn.Module,
    expert_head: MorphologyExpertHead,
    cfg: MorphExpertConfig,
    device: torch.device,
    args,
) -> float:
    from train_util import attach_edge_id_from_batch, get_hetero_seed_edge_ids, select_shared_seed_edge_embeddings
    from graph_augmentations import generate_views

    tier0_ctx: Optional[MorphTier0Context] = getattr(args, "morph_tier0_val", None)
    if cfg.include_global and tier0_ctx is None:
        raise ValueError("morph_tier0_val missing for local+global morphology eval")

    model.eval()
    expert_head.eval()
    loss_sum = torch.zeros((), device=device)
    n_batches = 0
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
        )
        loss_sum = loss_sum + loss.detach()
        n_batches += 1
    model.train()
    expert_head.train()
    if n_batches == 0:
        return 0.0
    return float((loss_sum / n_batches).cpu())


@torch.no_grad()
def eval_morph_expert_val_homo(
    val_loader,
    model: torch.nn.Module,
    expert_head: MorphologyExpertHead,
    cfg: MorphExpertConfig,
    device: torch.device,
    args,
) -> float:
    from train_util import attach_edge_id_from_batch, get_homo_seed_edge_ids, select_shared_seed_edge_embeddings
    from graph_augmentations import generate_views

    tier0_ctx: Optional[MorphTier0Context] = getattr(args, "morph_tier0_val", None)
    if cfg.include_global and tier0_ctx is None:
        raise ValueError("morph_tier0_val missing for local+global morphology eval")

    model.eval()
    expert_head.eval()
    loss_sum = torch.zeros((), device=device)
    n_batches = 0
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
        )
        loss_sum = loss_sum + loss.detach()
        n_batches += 1
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
    expert_head: MorphologyExpertHead,
    cfg: MorphExpertConfig,
    tier0_ctx: Optional[MorphTier0Context] = None,
) -> torch.Tensor:
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
    )
    return loss


def morph_expert_loss_homo_step(
    view1: Data,
    z1_seed: torch.Tensor,
    seed_id1: torch.Tensor,
    batch: Data,
    expert_head: MorphologyExpertHead,
    cfg: MorphExpertConfig,
    tier0_ctx: Optional[MorphTier0Context] = None,
) -> torch.Tensor:
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
    )
    return loss
