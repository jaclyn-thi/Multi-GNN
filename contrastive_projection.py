"""
GraphCL-style projection head for edge InfoNCE (contrastive pretrain only).

The GNN readout ``z`` (128-d transaction embedding) is unchanged for morphology
expert loss, morphology val, and ``embedding_extraction.py``. When
``--contrast_projection_head`` is set, seed embeddings pass through this MLP
before ``edge_identity_infonce_loss`` and the optional memory queue.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn


class ContrastiveProjectionHead(nn.Module):
    """Two-layer MLP with BatchNorm (GraphCL default)."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.numel() == 0:
            return z
        return self.net(z)


def setup_contrastive_projection(
    args,
    device: torch.device,
    *,
    embedding_dim: int = 128,
) -> Optional[ContrastiveProjectionHead]:
    """Build projection head when ``--contrast_projection_head`` is set."""
    if not bool(getattr(args, "contrast_projection_head", False)):
        return None
    hidden = int(getattr(args, "contrast_projection_hidden", 128))
    out_dim = int(getattr(args, "contrast_projection_dim", embedding_dim))
    head = ContrastiveProjectionHead(embedding_dim, hidden, out_dim).to(device)
    logging.info(
        "Contrastive projection head: %d → %d → %d (InfoNCE only; extract/probe use encoder z)",
        embedding_dim,
        hidden,
        out_dim,
    )
    return head


def project_seed_pair(
    head: Optional[nn.Module],
    z1_seed: torch.Tensor,
    z2_seed: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply projection to aligned seed embeddings for contrastive loss."""
    if head is None:
        return z1_seed, z2_seed
    return head(z1_seed), head(z2_seed)


def project_seeds(head: Optional[nn.Module], z_seed: torch.Tensor) -> torch.Tensor:
    """Apply projection to a single seed embedding tensor."""
    if head is None:
        return z_seed
    return head(z_seed)
