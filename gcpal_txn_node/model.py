"""Shared vanilla GIN (node-level) + projection MLP."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv


class VanillaGINEncoder(nn.Module):
    """Two-layer GINConv encoder producing 128-d node representations."""

    def __init__(self, in_dim: int, hidden_dim: int = 128, out_dim: int = 128, dropout: float = 0.0):
        super().__init__()
        self.dropout = float(dropout)

        def mlp(d_in: int, d_out: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(d_in, d_out),
                nn.ReLU(),
                nn.Linear(d_out, d_out),
            )

        self.conv1 = GINConv(mlp(in_dim, hidden_dim), train_eps=True)
        self.conv2 = GINConv(mlp(hidden_dim, out_dim), train_eps=True)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        return h


class ProjectionMLP(nn.Module):
    def __init__(self, in_dim: int = 128, hidden_dim: int = 128, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class SharedTxnNodeEncoder(nn.Module):
    """Shared GIN + projection; same parameters encode all views."""

    def __init__(self, in_dim: int, emb_dim: int = 128):
        super().__init__()
        self.gin = VanillaGINEncoder(in_dim, hidden_dim=emb_dim, out_dim=emb_dim)
        self.proj = ProjectionMLP(emb_dim, emb_dim, emb_dim)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.gin(x, edge_index)

    def project(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(h)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encode(x, edge_index)
        z = self.project(h)
        return h, z
