"""Morphology expert head and loss (Phase M1 / M1b)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from morphology.tier0_global import (
    DEFAULT_LIFT_FEATURE_NAMES,
    MorphTier0Context,
    lift_global_to_seed_edges_torch,
    setup_morph_tier0_contexts,
)
from morphology.tier1_local import (
    LOCAL_FEATURE_NAMES,
    align_seed_embeddings_with_morph,
    compute_local_morphology_torch,
    gather_seed_forward_edge_attr,
    transform_morph_targets,
)


@dataclass
class MorphExpertConfig:
    embedding_dim: int = 128
    hidden_dim: int = 64
    include_edge_native: bool = True
    include_global: bool = False
    loss_weight: float = 1.0


class MorphologyExpertHead(nn.Module):
    """Predict morphology targets from seed-edge embeddings (model-agnostic)."""

    def __init__(self, embedding_dim: int, target_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.target_dim = int(target_dim)
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.target_dim),
        )

    def forward(self, z_seed: torch.Tensor) -> torch.Tensor:
        return self.net(z_seed)


def morph_expert_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.numel() == 0:
        return pred.new_zeros(())
    return F.mse_loss(pred, target)


def build_morph_targets(
    edge_index: torch.Tensor,
    subgraph_edge_ids: torch.Tensor,
    seed_edge_ids: torch.Tensor,
    seed_edge_attr: Optional[torch.Tensor],
    num_nodes: int,
    cfg: MorphExpertConfig,
    tier0_ctx: Optional[MorphTier0Context] = None,
) -> torch.Tensor:
    """Detached morphology targets for seeds present in the view1 subgraph."""
    local = compute_local_morphology_torch(
        edge_index,
        subgraph_edge_ids,
        seed_edge_ids,
        num_nodes,
        device=edge_index.device,
    )
    global_feats = None
    if cfg.include_global:
        if tier0_ctx is None:
            raise ValueError("include_global=True requires tier0_ctx")
        global_feats = lift_global_to_seed_edges_torch(seed_edge_ids, tier0_ctx)
    edge_native = None
    if cfg.include_edge_native and seed_edge_attr is not None and seed_edge_attr.numel() > 0:
        edge_native = gather_seed_forward_edge_attr(
            seed_edge_attr, subgraph_edge_ids, seed_edge_ids
        )
    return transform_morph_targets(
        local,
        edge_native=edge_native,
        global_feats=global_feats,
    ).detach()


def morphology_expert_step(
    z_seed: torch.Tensor,
    seed_edge_ids: torch.Tensor,
    edge_index: torch.Tensor,
    subgraph_edge_ids: torch.Tensor,
    edge_attr: Optional[torch.Tensor],
    num_nodes: int,
    expert_head: MorphologyExpertHead,
    cfg: MorphExpertConfig,
    tier0_ctx: Optional[MorphTier0Context] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
  Compute morphology expert loss for one batch of shared seeds.

  Returns ``(loss, targets)`` where ``loss`` is scaled by ``cfg.loss_weight``.
  """
    z_use, ids_use = align_seed_embeddings_with_morph(
        z_seed, seed_edge_ids, subgraph_edge_ids
    )
    if z_use.numel() == 0:
        return z_seed.new_zeros(()), z_seed.new_empty((0, expert_head.target_dim))

    targets = build_morph_targets(
        edge_index,
        subgraph_edge_ids,
        ids_use,
        edge_attr,
        num_nodes,
        cfg,
        tier0_ctx=tier0_ctx,
    )
    if z_use.shape[0] != targets.shape[0]:
        min_n = min(z_use.shape[0], targets.shape[0])
        z_use = z_use[:min_n]
        targets = targets[:min_n]

    pred = expert_head(z_use)
    loss = morph_expert_mse_loss(pred, targets) * float(cfg.loss_weight)
    return loss, targets


def target_dim_for_config(
    edge_attr_dim: int,
    cfg: MorphExpertConfig,
) -> int:
    n = len(LOCAL_FEATURE_NAMES)
    if cfg.include_global:
        n += len(DEFAULT_LIFT_FEATURE_NAMES)
    if cfg.include_edge_native:
        n += int(edge_attr_dim)
    return n


def create_morph_expert_bundle(
    edge_attr_dim: int,
    cfg: MorphExpertConfig,
    device: torch.device,
) -> Tuple[MorphologyExpertHead, MorphExpertConfig]:
    dim = target_dim_for_config(edge_attr_dim, cfg)
    head = MorphologyExpertHead(cfg.embedding_dim, dim, hidden_dim=cfg.hidden_dim).to(device)
    return head, cfg


def _morph_targets_mode(args) -> str:
    return str(getattr(args, "morph_targets", "local")).lower()


def setup_morphology_expert(args, tr_data, device, is_hetero: bool):
    """
  Build expert head + config when ``--morph_expert`` is set (contrastive M1 / M1b).

  Returns ``(head, cfg)`` or ``(None, None)``.
  """
    if not getattr(args, "morph_expert", False):
        return None, None
    targets = _morph_targets_mode(args)
    if targets not in ("local", "local+global"):
        raise ValueError(
            f"--morph_targets {targets!r} is invalid; use 'local' or 'local+global'."
        )
    from morphology.graph_access import get_forward_edge_attr

    if is_hetero:
        attr = get_forward_edge_attr(tr_data)
    else:
        attr = tr_data.edge_attr
    if attr is None:
        raise ValueError("Cannot infer edge_attr dim for morphology expert (missing edge_attr).")
    edge_attr_dim = int(attr.shape[1]) - 1  # column 0 is synthetic EdgeID from add_arange_ids

    cfg = MorphExpertConfig(
        embedding_dim=128,
        hidden_dim=int(getattr(args, "morph_expert_hidden", 64)),
        include_edge_native=not bool(getattr(args, "no_morph_edge_native", False)),
        include_global=(targets == "local+global"),
        loss_weight=float(getattr(args, "morph_expert_weight", 1.0)),
    )
    head, cfg = create_morph_expert_bundle(edge_attr_dim, cfg, device)
    extra = ""
    if cfg.include_global:
        extra += f" + global={len(DEFAULT_LIFT_FEATURE_NAMES)}"
    if cfg.include_edge_native:
        extra += f" + edge_native={edge_attr_dim}"
    logging.info(
        "Morphology expert head: target_dim=%d (local=%d%s) weight=%.4f morph_targets=%s",
        head.target_dim,
        len(LOCAL_FEATURE_NAMES),
        extra,
        cfg.loss_weight,
        targets,
    )
    return head, cfg


__all__ = [
    "MorphExpertConfig",
    "MorphologyExpertHead",
    "build_morph_targets",
    "create_morph_expert_bundle",
    "morph_expert_mse_loss",
    "morphology_expert_step",
    "setup_morphology_expert",
    "setup_morph_tier0_contexts",
    "target_dim_for_config",
]
