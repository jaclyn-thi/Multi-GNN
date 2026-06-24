"""KNN soft positives for transaction-level contrastive pretraining (GCPAL-style).

Selected train-split KNN neighbors are injected via an auxiliary LinkNeighborLoader
forward pass on explicit seed edge ids, so positives are available even when random
8192-negative sampling would never overlap cached neighbors.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.cuda.amp import autocast

from graph_augmentations import generate_views
from knn_filter import TransactionKNNFilter
from train_util import (
    FORWARD_EDGE_TYPE,
    attach_edge_id_from_batch,
    get_hetero_seed_edge_ids,
    get_homo_seed_edge_ids,
    select_shared_seed_edge_embeddings,
)


@dataclass
class KnnSoftPositiveSample:
    """Per-anchor KNN positive ids, similarities, and weights for one training step."""

    pos_ids: torch.Tensor
    pos_sims: torch.Tensor
    pos_weights: torch.Tensor
    pos_valid: torch.Tensor
    pos_z2: torch.Tensor


class KnnSoftPositiveCache:
    """Offline KNN cache + deterministic positive sampling."""

    def __init__(
        self,
        path: str,
        *,
        source_k: int,
        pos_m: int,
        total_weight: float,
        weight_mode: str = "uniform",
        min_sim: Optional[float] = None,
        base_seed: int = 0,
        device: torch.device,
    ) -> None:
        del device
        self.source_k = max(1, int(source_k))
        self.pos_m = max(1, int(pos_m))
        self.total_weight = float(total_weight)
        self.weight_mode = str(weight_mode)
        self.min_sim = None if min_sim is None else float(min_sim)
        self.base_seed = int(base_seed)
        if self.weight_mode not in {"uniform", "similarity"}:
            raise ValueError(f"Unsupported knn_pos_weight_mode={weight_mode!r}")
        self._filter = TransactionKNNFilter.from_npz(
            path,
            filter_k=self.source_k,
            device=torch.device("cpu"),
        )
        logging.info(
            "Loaded KNN soft-positive cache from %s: anchors=%d source_k=%d pos_m=%d total_weight=%.4f mode=%s",
            path,
            int(self._filter.edge_ids.numel()),
            self.source_k,
            self.pos_m,
            self.total_weight,
            self.weight_mode,
        )

    def _lookup_row(self, anchor_id: int) -> Tuple[np.ndarray, np.ndarray]:
        edge_ids = self._filter.edge_ids.numpy()
        pos = int(np.searchsorted(edge_ids, anchor_id))
        if pos >= edge_ids.shape[0] or int(edge_ids[pos]) != int(anchor_id):
            return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32)
        neigh = self._filter.neighbor_ids[pos].numpy()
        sims = (
            self._filter.neighbor_sims[pos].numpy()
            if self._filter.neighbor_sims is not None
            else np.ones(neigh.shape[0], dtype=np.float32)
        )
        valid = neigh >= 0
        if self.min_sim is not None:
            valid &= sims >= float(self.min_sim)
        valid &= neigh != int(anchor_id)
        return neigh[valid], sims[valid]

    def sample(
        self,
        anchor_ids: torch.Tensor,
        *,
        step: int,
        epoch: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
        anchors = anchor_ids.detach().cpu().long().view(-1)
        n = int(anchors.numel())
        m = self.pos_m
        pos_ids = np.full((n, m), -1, dtype=np.int64)
        pos_sims = np.zeros((n, m), dtype=np.float32)
        pos_weights = np.zeros((n, m), dtype=np.float32)

        stats: Dict[str, float] = {
            "anchors": float(n),
            "requested_positives": float(n * m),
            "usable_positives": 0.0,
            "anchors_with_any_pos": 0.0,
            "dropped_self_or_invalid": 0.0,
            "dropped_insufficient_candidates": 0.0,
            "sim_min": float("inf"),
            "sim_max": float("-inf"),
            "sim_sum": 0.0,
            "sim_count": 0.0,
        }

        seed = self.base_seed + int(epoch) * 1_000_003 + int(step) * 9973
        rng = np.random.default_rng(seed)

        for i, aid in enumerate(anchors.numpy()):
            cand_ids, cand_sims = self._lookup_row(int(aid))
            if cand_ids.size == 0:
                stats["dropped_insufficient_candidates"] += float(m)
                continue
            take = min(m, int(cand_ids.shape[0]))
            if take < m:
                stats["dropped_insufficient_candidates"] += float(m - take)
            pick = rng.choice(cand_ids.shape[0], size=take, replace=False)
            chosen_ids = cand_ids[pick]
            chosen_sims = cand_sims[pick]
            pos_ids[i, :take] = chosen_ids
            pos_sims[i, :take] = chosen_sims
            stats["usable_positives"] += float(take)
            stats["anchors_with_any_pos"] += 1.0
            stats["sim_min"] = min(stats["sim_min"], float(chosen_sims.min()))
            stats["sim_max"] = max(stats["sim_max"], float(chosen_sims.max()))
            stats["sim_sum"] += float(chosen_sims.sum())
            stats["sim_count"] += float(take)
            if self.weight_mode == "uniform":
                w = self.total_weight / float(m)
                pos_weights[i, :take] = w
            else:
                sims = np.clip(chosen_sims.astype(np.float64), 1e-8, None)
                frac = sims / sims.sum()
                pos_weights[i, :take] = (frac * self.total_weight).astype(np.float32)

        if stats["sim_count"] <= 0:
            stats["sim_min"] = float("nan")
            stats["sim_max"] = float("nan")

        valid = pos_ids >= 0
        return (
            torch.as_tensor(pos_ids, dtype=torch.long, device=anchor_ids.device),
            torch.as_tensor(pos_sims, dtype=torch.float32, device=anchor_ids.device),
            torch.as_tensor(pos_weights, dtype=torch.float32, device=anchor_ids.device),
            torch.as_tensor(valid, dtype=torch.bool, device=anchor_ids.device),
            stats,
        )


def _hetero_forward(model, batch, use_amp: bool) -> torch.Tensor:
    with autocast(enabled=use_amp):
        return model(
            batch.x_dict,
            batch.edge_index_dict,
            batch.edge_attr_dict,
        )[FORWARD_EDGE_TYPE]


def _homo_forward(model, batch, use_amp: bool) -> torch.Tensor:
    with autocast(enabled=use_amp):
        return model(batch.x, batch.edge_index, batch.edge_attr)


@torch.no_grad()
def forward_view2_embeddings_for_edge_ids(
    tr_loader,
    model,
    edge_ids: torch.Tensor,
    *,
    device: torch.device,
    use_amp: bool,
    contrastive_symmetric: bool,
    chunk_size: int = 4096,
    exclude_last_column: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run view-2 forward passes with explicit seed edge ids; return (ids, z2)."""
    if edge_ids.numel() == 0:
        dim = 128
        return (
            edge_ids.new_empty((0,), dtype=torch.long),
            edge_ids.new_empty((0, dim), dtype=edge_ids.dtype),
        )

    loader_data = tr_loader.data
    hetero = hasattr(loader_data, "node_types")
    edge_ids = torch.unique(edge_ids.long().cpu())
    edge_ids = edge_ids[edge_ids >= 0]
    id_chunks = []
    z_chunks = []
    grad_ctx = torch.enable_grad() if contrastive_symmetric else torch.no_grad()

    with grad_ctx:
        for start in range(0, int(edge_ids.numel()), int(chunk_size)):
            chunk = edge_ids[start : start + int(chunk_size)]
            batch = tr_loader(chunk)
            attach_edge_id_from_batch(batch, loader_data)
            batch = batch.to(device, non_blocking=True)
            _, view2 = generate_views(
                batch,
                edge_attr_mask_rate=0.1,
                edge_drop_rate=0.1,
                mask_value=0.0,
                mask_cols=None,
                exclude_last_column=exclude_last_column,
            )
            if hetero:
                z2 = _hetero_forward(model, view2, use_amp)
                edge_id2 = view2[FORWARD_EDGE_TYPE].edge_id
                seed_ids = get_hetero_seed_edge_ids(batch, loader_data).to(device)
            else:
                z2 = _homo_forward(model, view2, use_amp)
                edge_id2 = view2.edge_id
                seed_ids = get_homo_seed_edge_ids(batch, loader_data).to(device)

            z2_seed, out_ids, _, _ = select_shared_seed_edge_embeddings(
                z2,
                edge_id2,
                z2,
                edge_id2,
                seed_ids,
            )
            id_chunks.append(out_ids.detach().cpu())
            z_chunks.append(z2_seed.detach().cpu())

    if not id_chunks:
        return edge_ids.new_empty((0,), dtype=torch.long), edge_ids.new_empty((0, 128))

    out_ids = torch.cat(id_chunks, dim=0).long()
    out_z = torch.cat(z_chunks, dim=0)
    order = torch.argsort(out_ids)
    out_ids = out_ids[order]
    out_z = out_z[order]
    keep = torch.ones(out_ids.shape[0], dtype=torch.bool)
    keep[1:] = out_ids[1:] != out_ids[:-1]
    return out_ids[keep], out_z[keep]


def gather_knn_positive_embeddings(
    pos_ids: torch.Tensor,
    pos_valid: torch.Tensor,
    unique_ids: torch.Tensor,
    unique_z2: torch.Tensor,
    stats: Optional[Dict[str, float]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Map sampled positive ids to view-2 embeddings; mark missing as invalid."""
    embed_dim = int(unique_z2.shape[1]) if unique_z2.numel() > 0 else 128
    if unique_ids.numel() == 0:
        out = pos_ids.new_zeros((pos_ids.shape[0], pos_ids.shape[1], embed_dim))
        return out, pos_valid.clone()

    pos_ids_cpu = pos_ids.detach().cpu()
    valid_cpu = pos_valid.detach().cpu()
    unique_cpu = unique_ids.detach().cpu()
    z_cpu = unique_z2.detach().cpu()
    pos = torch.searchsorted(unique_cpu, pos_ids_cpu.clamp(min=0))
    in_range = (pos < unique_cpu.numel()) & (pos_ids_cpu >= 0)
    found = in_range & (unique_cpu[pos.clamp(max=max(int(unique_cpu.numel()) - 1, 0))] == pos_ids_cpu.clamp(min=0))
    gathered = torch.zeros((pos_ids_cpu.shape[0], pos_ids_cpu.shape[1], z_cpu.shape[1]), dtype=z_cpu.dtype)
    usable = valid_cpu & found
    if usable.any():
        rows, cols = torch.nonzero(usable, as_tuple=True)
        gathered[rows, cols] = z_cpu[pos[rows, cols]]
    if stats is not None:
        stats["positives_missing_embedding"] = stats.get("positives_missing_embedding", 0.0) + float(
            (valid_cpu & ~found).sum().item()
        )
        stats["unique_pos_ids_requested"] = stats.get("unique_pos_ids_requested", 0.0) + float(
            torch.unique(pos_ids_cpu[valid_cpu]).numel()
        )
        stats["unique_pos_ids_resolved"] = stats.get("unique_pos_ids_resolved", 0.0) + float(unique_cpu.numel())
    return gathered.to(pos_ids.device), usable.to(pos_ids.device)


def update_knn_endpoint_overlap_stats(
    anchor_ids: torch.Tensor,
    pos_ids: torch.Tensor,
    pos_valid: torch.Tensor,
    edge_index: torch.Tensor,
    stats: Dict[str, float],
) -> None:
    """Label-only overlap diagnostics between anchors and selected KNN positives."""
    if pos_ids.numel() == 0:
        return
    edge_index = edge_index.long().cpu()
    anchors = anchor_ids.long().cpu().view(-1)
    pos = pos_ids.long().cpu()
    valid = pos_valid.cpu()
    a_src = edge_index[0, anchors.clamp(max=edge_index.shape[1] - 1)]
    a_dst = edge_index[1, anchors.clamp(max=edge_index.shape[1] - 1)]
    flat_pos = pos[valid]
    if flat_pos.numel() == 0:
        return
    p_src = edge_index[0, flat_pos.clamp(max=edge_index.shape[1] - 1)]
    p_dst = edge_index[1, flat_pos.clamp(max=edge_index.shape[1] - 1)]
    anchor_rep_src = a_src.unsqueeze(1).expand(-1, pos.shape[1])[valid]
    anchor_rep_dst = a_dst.unsqueeze(1).expand(-1, pos.shape[1])[valid]
    stats["knn_pos_same_sender"] = stats.get("knn_pos_same_sender", 0.0) + float((p_src == anchor_rep_src).sum().item())
    stats["knn_pos_same_receiver"] = stats.get("knn_pos_same_receiver", 0.0) + float((p_dst == anchor_rep_dst).sum().item())
    stats["knn_pos_same_pair"] = stats.get("knn_pos_same_pair", 0.0) + float(
        ((p_src == anchor_rep_src) & (p_dst == anchor_rep_dst)).sum().item()
    )


def load_knn_soft_positive_cache(
    path: Optional[str],
    *,
    enabled: bool,
    source_k: int,
    pos_m: int,
    total_weight: float,
    weight_mode: str,
    min_sim: Optional[float],
    base_seed: int,
    device: torch.device,
) -> Optional[KnnSoftPositiveCache]:
    if not enabled:
        return None
    if not path:
        raise ValueError("--enable_knn_soft_positives requires --knn_cache_path")
    return KnnSoftPositiveCache(
        path,
        source_k=source_k,
        pos_m=pos_m,
        total_weight=total_weight,
        weight_mode=weight_mode,
        min_sim=min_sim,
        base_seed=base_seed,
        device=device,
    )


__all__ = [
    "KnnSoftPositiveCache",
    "KnnSoftPositiveSample",
    "forward_view2_embeddings_for_edge_ids",
    "gather_knn_positive_embeddings",
    "load_knn_soft_positive_cache",
    "update_knn_endpoint_overlap_stats",
]
