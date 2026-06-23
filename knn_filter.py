"""Sparse transaction KNN cache support for contrastive negative exclusion.

The cache is intentionally offline and sparse: it maps train split-local
``edge_id`` values to a small list of neighboring train split-local edge ids.
During InfoNCE, those neighbors can be excluded from the negative candidate pool
without constructing a dense KNN graph view.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch


@dataclass
class TransactionKNNFilter:
    """In-memory sparse KNN neighbor lookup keyed by split-local edge ids."""

    edge_ids: torch.Tensor
    neighbor_ids: torch.Tensor
    neighbor_sims: Optional[torch.Tensor] = None
    max_k: int = 0

    @classmethod
    def from_npz(
        cls,
        path: str,
        *,
        filter_k: int = 0,
        device: Optional[torch.device] = None,
    ) -> "TransactionKNNFilter":
        cache_path = Path(path)
        if not cache_path.is_file():
            raise FileNotFoundError(f"KNN cache not found: {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        if "edge_ids" not in data or "neighbor_ids" not in data:
            raise ValueError(f"KNN cache {cache_path} must contain edge_ids and neighbor_ids arrays")
        edge_ids_np = np.asarray(data["edge_ids"], dtype=np.int64).reshape(-1)
        neighbor_ids_np = np.asarray(data["neighbor_ids"], dtype=np.int64)
        if neighbor_ids_np.ndim != 2:
            raise ValueError("KNN cache neighbor_ids must be a 2D array")
        if edge_ids_np.shape[0] != neighbor_ids_np.shape[0]:
            raise ValueError("KNN cache edge_ids and neighbor_ids row counts differ")

        k_available = int(neighbor_ids_np.shape[1])
        k_use = k_available if int(filter_k) <= 0 else min(int(filter_k), k_available)
        neighbor_ids_np = neighbor_ids_np[:, :k_use]
        neighbor_sims = None
        if "neighbor_sims" in data:
            sims_np = np.asarray(data["neighbor_sims"], dtype=np.float32)
            if sims_np.shape[:2] == (edge_ids_np.shape[0], k_available):
                neighbor_sims = torch.as_tensor(sims_np[:, :k_use], dtype=torch.float32, device=device)

        edge_ids = torch.as_tensor(edge_ids_np, dtype=torch.long, device=device)
        neighbor_ids = torch.as_tensor(neighbor_ids_np, dtype=torch.long, device=device)
        order = torch.argsort(edge_ids)
        edge_ids = edge_ids[order]
        neighbor_ids = neighbor_ids[order]
        if neighbor_sims is not None:
            neighbor_sims = neighbor_sims[order]

        out = cls(edge_ids=edge_ids, neighbor_ids=neighbor_ids, neighbor_sims=neighbor_sims, max_k=k_use)
        logging.info(
            "Loaded KNN negative filter cache from %s: anchors=%d k=%d",
            cache_path,
            int(edge_ids.numel()),
            k_use,
        )
        if "metadata_json" in data:
            try:
                logging.info("KNN cache metadata: %s", str(data["metadata_json"].item()))
            except Exception:
                pass
        return out

    def to(self, device: torch.device) -> "TransactionKNNFilter":
        self.edge_ids = self.edge_ids.to(device)
        self.neighbor_ids = self.neighbor_ids.to(device)
        if self.neighbor_sims is not None:
            self.neighbor_sims = self.neighbor_sims.to(device)
        return self

    def allowed_mask(
        self,
        anchor_ids: torch.Tensor,
        candidate_ids: torch.Tensor,
        *,
        stats: Optional[Dict[str, float]] = None,
    ) -> torch.Tensor:
        """Return ``True`` for candidates allowed as negatives."""

        anchor_ids = anchor_ids.long().view(-1)
        candidate_ids = candidate_ids.long().view(-1)
        allowed = torch.ones(
            (anchor_ids.numel(), candidate_ids.numel()),
            dtype=torch.bool,
            device=anchor_ids.device,
        )
        if anchor_ids.numel() == 0 or candidate_ids.numel() == 0 or self.max_k <= 0 or self.edge_ids.numel() == 0:
            return allowed

        edge_ids = self.edge_ids.to(anchor_ids.device)
        neighbor_ids = self.neighbor_ids.to(anchor_ids.device)
        pos = torch.searchsorted(edge_ids, anchor_ids)
        in_range = pos < edge_ids.numel()
        found = in_range & (edge_ids[pos.clamp(max=edge_ids.numel() - 1)] == anchor_ids)
        if not found.any():
            if stats is not None:
                stats["rows"] = stats.get("rows", 0.0) + float(anchor_ids.numel())
            return allowed

        rows = torch.nonzero(found, as_tuple=False).view(-1)
        neigh = neighbor_ids[pos[rows]]
        present = neigh >= 0
        if present.any():
            exclude = (candidate_ids.view(1, 1, -1) == neigh.unsqueeze(-1)) & present.unsqueeze(-1)
            exclude = exclude.any(dim=1)
            allowed[rows] = allowed[rows] & ~exclude
            removed = int(exclude.sum().item())
            rows_with_present = int(exclude.any(dim=1).sum().item())
        else:
            removed = 0
            rows_with_present = 0

        if stats is not None:
            stats["rows"] = stats.get("rows", 0.0) + float(anchor_ids.numel())
            stats["rows_with_cache"] = stats.get("rows_with_cache", 0.0) + float(rows.numel())
            stats["rows_with_knn_in_pool"] = stats.get("rows_with_knn_in_pool", 0.0) + float(rows_with_present)
            stats["candidate_before"] = stats.get("candidate_before", 0.0) + float(allowed.numel())
            stats["candidate_after"] = stats.get("candidate_after", 0.0) + float(allowed.sum().item())
            stats["knn_removed"] = stats.get("knn_removed", 0.0) + float(removed)
        return allowed


def load_transaction_knn_filter(
    path: Optional[str],
    *,
    enabled: bool,
    filter_k: int,
    device: torch.device,
) -> Optional[TransactionKNNFilter]:
    """Load an optional transaction KNN filter from disk."""

    if not enabled:
        return None
    if not path:
        raise ValueError("--enable_knn_negative_filter requires --knn_cache_path")
    return TransactionKNNFilter.from_npz(path, filter_k=filter_k, device=device)


__all__ = ["TransactionKNNFilter", "load_transaction_knn_filter"]
