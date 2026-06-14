"""
Edge-level contrastive losses for transaction-graph self-supervised learning.

Primary API
-----------
``edge_identity_infonce_loss`` — InfoNCE on aligned seed-edge embeddings across
two augmented views. Optional extensions:

- ``num_neg_samples`` — GPU subsampled negatives (memory-friendly at large batch)
- ``memory_queue`` — detached FIFO queue of past seed embeddings as extra negatives
- ``morph_bin_ids`` — Phase M2: same morphology bin → additional positives in the
  numerator via **bin-grouped** indexing (no dense ``(B, B)`` mask)

Supporting utilities align shared ``edge_id`` rows across views before loss
computation. See ``morphology/contrast.py`` and ``morphology/contrastive_train.py``.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

# important: inputs must be projected embeddings
# shapes must be aligned: (N,D)
# And A and A_knn must be node aligned (same node ordering across views)


def positive_mask_identity(edge_id1: torch.Tensor, edge_id2: torch.Tensor) -> torch.Tensor:
    """
    Identity tier: ``pos[i, j]`` iff the same stable edge / transaction id appears
    in both views.
    """
    return edge_id1.unsqueeze(1) == edge_id2.unsqueeze(0)


def merge_positive_tiers(*masks: torch.Tensor) -> torch.Tensor:
    """Combine positive tiers with logical OR."""
    if not masks:
        raise ValueError("merge_positive_tiers requires at least one mask.")
    out = masks[0]
    for m in masks[1:]:
        out = out | m
    return out


class EdgeMemoryQueue:
    """
    Detached FIFO queue of prior seed-edge embeddings used as extra negatives.

    The queue never contributes positives; matching ids are filtered out from the
    negative set at loss time.
    """

    def __init__(self, capacity: int, device: torch.device):
        self.capacity = max(0, int(capacity))
        self.device = device
        self.embeddings: Optional[torch.Tensor] = None
        self.edge_ids: Optional[torch.Tensor] = None

    @property
    def size(self) -> int:
        if self.edge_ids is None:
            return 0
        return int(self.edge_ids.numel())

    def get(
        self,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out_device = device if device is not None else self.device
        if self.size == 0 or self.embeddings is None or self.edge_ids is None:
            empty_z = torch.empty((0, 0), device=out_device, dtype=dtype or torch.float32)
            empty_ids = torch.empty((0,), device=out_device, dtype=torch.long)
            return empty_z, empty_ids
        out_z = self.embeddings
        if device is not None or dtype is not None:
            out_z = out_z.to(device=out_device, dtype=dtype or out_z.dtype)
        out_ids = self.edge_ids if out_device == self.edge_ids.device else self.edge_ids.to(out_device)
        return out_z, out_ids

    @torch.no_grad()
    def enqueue(self, embeddings: torch.Tensor, edge_ids: torch.Tensor) -> None:
        if self.capacity <= 0 or embeddings.numel() == 0:
            return
        z = F.normalize(embeddings.detach().float(), dim=1).to(self.device)
        ids = edge_ids.detach().long().view(-1).to(self.device)
        if z.shape[0] != ids.shape[0]:
            raise ValueError("Queued embeddings and edge ids must have matching lengths.")
        if z.shape[0] > self.capacity:
            z = z[-self.capacity :]
            ids = ids[-self.capacity :]
        if self.embeddings is None or self.edge_ids is None:
            self.embeddings = z
            self.edge_ids = ids
            return
        self.embeddings = torch.cat([self.embeddings, z], dim=0)[-self.capacity :]
        self.edge_ids = torch.cat([self.edge_ids, ids], dim=0)[-self.capacity :]


def _normalize_embeddings(z: torch.Tensor) -> torch.Tensor:
    return F.normalize(z.float(), dim=1)


def _align_morph_bin_ids_to_shared(
    morph_bin_ids: torch.Tensor,
    edge_id1: torch.Tensor,
    edge_id2: torch.Tensor,
) -> torch.Tensor:
    """Reorder morphology bins to match ``_align_shared_edge_pairs`` row order."""
    edge_id1 = edge_id1.long().view(-1)
    edge_id2 = edge_id2.long().view(-1)
    if morph_bin_ids.shape[0] != edge_id1.shape[0]:
        raise ValueError("morph_bin_ids must align with edge_id1 rows before pairing.")
    shared_ids = edge_id1[torch.isin(edge_id1, edge_id2)]
    if shared_ids.numel() == 0:
        return morph_bin_ids.new_empty((0,), dtype=torch.long)
    shared_ids = torch.unique(shared_ids, sorted=True)
    keep1 = torch.isin(edge_id1, shared_ids)
    bins = morph_bin_ids[keep1]
    ids = edge_id1[keep1]
    order = torch.argsort(ids)
    return bins[order]


def _align_shared_edge_pairs(
    z1: torch.Tensor,
    edge_id1: torch.Tensor,
    z2: torch.Tensor,
    edge_id2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    edge_id1 = edge_id1.long().view(-1)
    edge_id2 = edge_id2.long().view(-1)
    if edge_id1.shape[0] != z1.shape[0] or edge_id2.shape[0] != z2.shape[0]:
        raise ValueError("edge_id lengths must match embedding row counts.")
    if edge_id1.numel() == 0 or edge_id2.numel() == 0:
        return (
            z1.new_empty((0, z1.shape[1])),
            edge_id1.new_empty((0,), dtype=torch.long),
            z2.new_empty((0, z2.shape[1])),
            edge_id2.new_empty((0,), dtype=torch.long),
        )

    shared_ids = edge_id1[torch.isin(edge_id1, edge_id2)]
    if shared_ids.numel() == 0:
        return (
            z1.new_empty((0, z1.shape[1])),
            edge_id1.new_empty((0,), dtype=torch.long),
            z2.new_empty((0, z2.shape[1])),
            edge_id2.new_empty((0,), dtype=torch.long),
        )

    shared_ids = torch.unique(shared_ids, sorted=True)
    keep1 = torch.isin(edge_id1, shared_ids)
    keep2 = torch.isin(edge_id2, shared_ids)

    z1 = z1[keep1]
    z2 = z2[keep2]
    edge_id1 = edge_id1[keep1]
    edge_id2 = edge_id2[keep2]

    order1 = torch.argsort(edge_id1)
    order2 = torch.argsort(edge_id2)
    z1 = z1[order1]
    z2 = z2[order2]
    edge_id1 = edge_id1[order1]
    edge_id2 = edge_id2[order2]

    if edge_id1.shape[0] != edge_id2.shape[0] or not torch.equal(edge_id1, edge_id2):
        raise ValueError("Failed to align shared positive pairs by edge_id.")
    return z1, edge_id1, z2, edge_id2


def _sample_negative_indices_batched_gpu(
    candidate_ids: torch.Tensor,
    anchor_ids: torch.Tensor,
    k: int,
    *,
    oversample_factor: int = 4,
    max_rounds: int = 6,
) -> torch.Tensor:
    """
    Batched GPU negative sampling with replacement.

    Returns an index tensor of shape ``(B, k)`` with ``-1`` marking rows/slots that
    could not be filled (e.g. no valid negatives available).
    """
    b = int(anchor_ids.numel())
    if k <= 0 or b == 0 or candidate_ids.numel() == 0:
        return torch.empty((b, 0), device=anchor_ids.device, dtype=torch.long)

    neg_idx = torch.full((b, int(k)), -1, device=anchor_ids.device, dtype=torch.long)
    filled = torch.zeros((b,), device=anchor_ids.device, dtype=torch.long)
    sample_width = max(int(k) * int(max(1, oversample_factor)), 32)
    n_candidates = int(candidate_ids.numel())

    for _ in range(max_rounds):
        remaining = torch.nonzero(filled < k, as_tuple=False).view(-1)
        if remaining.numel() == 0:
            break
        base_filled = filled[remaining]
        cand = torch.randint(0, n_candidates, (remaining.numel(), sample_width), device=anchor_ids.device)
        cand_ids = candidate_ids[cand]
        valid = cand_ids != anchor_ids[remaining].unsqueeze(1)
        if not valid.any():
            continue

        rank = torch.cumsum(valid.to(torch.int64), dim=1) - 1
        take_mask = valid & (rank < (k - base_filled).unsqueeze(1))
        if not take_mask.any():
            continue

        local_rows = torch.arange(remaining.numel(), device=anchor_ids.device).unsqueeze(1).expand_as(cand)
        selected_rows_local = local_rows[take_mask]
        selected_rows = remaining[selected_rows_local]
        selected_cols = base_filled[selected_rows_local] + rank[take_mask]
        neg_idx[selected_rows, selected_cols] = cand[take_mask]
        filled[remaining] = base_filled + take_mask.sum(dim=1).to(base_filled.dtype)

    return neg_idx


def _logsumexp_over_queue(
    z_b: torch.Tensor,
    anchor_ids: torch.Tensor,
    queue_z: torch.Tensor,
    queue_ids: torch.Tensor,
    temperature: float,
    col_chunk: int,
) -> Optional[torch.Tensor]:
    if queue_z.numel() == 0 or queue_ids.numel() == 0:
        return None
    log_queue: Optional[torch.Tensor] = None
    for j0 in range(0, queue_z.shape[0], col_chunk):
        j1 = min(j0 + col_chunk, queue_z.shape[0])
        chunk_z = queue_z[j0:j1]
        chunk_ids = queue_ids[j0:j1]
        logits = (z_b @ chunk_z.T) / temperature
        valid = chunk_ids.unsqueeze(0) != anchor_ids.unsqueeze(1)
        logits = torch.where(valid, logits, torch.full_like(logits, float("-inf")))
        ls = logits.logsumexp(dim=1)
        log_queue = ls if log_queue is None else torch.logaddexp(log_queue, ls)
    return log_queue


def morphology_soft_positive_mask(morph_bin_ids: torch.Tensor) -> torch.Tensor:
    """``(B, B)`` bool: soft positives share the same morphology bin (debug / tests only)."""
    return morph_bin_ids.unsqueeze(0) == morph_bin_ids.unsqueeze(1)


def _bin_positive_index_segments(morph_bin_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sort seed rows by morphology bin for group-wise positive lookup.

    Returns ``(inv, group_starts, sorted_row_indices)`` where positives for
    anchor row ``i`` lie in ``sorted_row_indices[group_starts[inv[i]]:group_starts[inv[i+1]]]``.

    Avoids materializing a ``(B, B)`` soft-positive mask at large batch sizes.
    """
    if morph_bin_ids.numel() == 0:
        empty = morph_bin_ids.new_empty((0,), dtype=torch.long)
        return empty, empty, empty
    _, inv = torch.unique(morph_bin_ids, return_inverse=True)
    order = torch.argsort(inv)
    sorted_idx = order
    sorted_inv = inv[order]
    n_groups = int(inv.max().item()) + 1
    counts = torch.bincount(inv, minlength=n_groups)
    starts = torch.zeros(n_groups + 1, device=inv.device, dtype=torch.long)
    starts[1:] = torch.cumsum(counts, dim=0)
    return inv, starts, sorted_idx


def _cap_positive_row_indices(
    pos_idx: torch.Tensor,
    anchor_row: int,
    max_positives: int,
) -> torch.Tensor:
    if pos_idx.numel() <= max_positives:
        return pos_idx
    keep = pos_idx[pos_idx == anchor_row]
    if keep.numel() == 0:
        keep = pos_idx[:1]
    others = pos_idx[pos_idx != anchor_row]
    n_other = max(0, max_positives - int(keep.numel()))
    if others.numel() > n_other:
        pick = torch.randperm(others.numel(), device=others.device)[:n_other]
        others = others[pick]
    return torch.cat([keep, others], dim=0)


def _logsumexp_positive_logits_bin_grouped(
    z_row: torch.Tensor,
    z_other: torch.Tensor,
    pos_idx: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if pos_idx.numel() == 0:
        return z_row.new_tensor(float("-inf"))
    if pos_idx.numel() == 1:
        return (z_row * z_other[pos_idx[0]]).sum() / temperature
    logits = (z_row @ z_other[pos_idx].T) / temperature
    return torch.logsumexp(logits.reshape(-1), dim=0)


def _sample_negative_indices_by_bin(
    morph_bin_ids: torch.Tensor,
    candidate_ids: torch.Tensor,
    anchor_ids: torch.Tensor,
    anchor_bins: torch.Tensor,
    k: int,
    *,
    oversample_factor: int = 4,
    max_rounds: int = 6,
) -> torch.Tensor:
    """
    Sample negatives without a dense mask.

    Batch columns ``j < B`` are negatives when ``morph_bin_ids[j] != anchor_bin``.
    Queue columns ``j >= B`` are negatives when ``candidate_ids[j] != anchor_id``.
    """
    b = int(anchor_ids.numel())
    n_batch = int(morph_bin_ids.numel())
    if k <= 0 or b == 0 or candidate_ids.numel() == 0:
        return torch.empty((b, 0), device=anchor_ids.device, dtype=torch.long)

    neg_idx = torch.full((b, int(k)), -1, device=anchor_ids.device, dtype=torch.long)
    filled = torch.zeros((b,), device=anchor_ids.device, dtype=torch.long)
    n_candidates = int(candidate_ids.numel())
    sample_width = max(int(k) * int(max(1, oversample_factor)), 32)

    for _ in range(max_rounds):
        remaining = torch.nonzero(filled < k, as_tuple=False).view(-1)
        if remaining.numel() == 0:
            break
        base_filled = filled[remaining]
        cand = torch.randint(0, n_candidates, (remaining.numel(), sample_width), device=anchor_ids.device)
        is_queue = cand >= n_batch
        in_batch = cand < n_batch
        bin_mismatch = torch.zeros_like(is_queue)
        if n_batch > 0:
            bin_mismatch = in_batch & (
                morph_bin_ids[cand.clamp(max=n_batch - 1)]
                != anchor_bins[remaining].unsqueeze(1)
            )
        valid = (candidate_ids[cand] != anchor_ids[remaining].unsqueeze(1)) & (is_queue | bin_mismatch)
        if not valid.any():
            continue

        rank = torch.cumsum(valid.to(torch.int64), dim=1) - 1
        take_mask = valid & (rank < (k - base_filled).unsqueeze(1))
        if not take_mask.any():
            continue

        local_rows = torch.arange(remaining.numel(), device=anchor_ids.device).unsqueeze(1).expand_as(cand)
        selected_rows_local = local_rows[take_mask]
        selected_rows = remaining[selected_rows_local]
        selected_cols = base_filled[selected_rows_local] + rank[take_mask]
        neg_idx[selected_rows, selected_cols] = cand[take_mask]
        filled[remaining] = base_filled + take_mask.sum(dim=1).to(base_filled.dtype)

    return neg_idx


def _infonce_row_chunk_for_neg_subsample(
    row_chunk: int,
    k_neg: int,
    embed_dim: int,
    *,
    max_neg_tensor_bytes: int = 256 * 1024 * 1024,
) -> int:
    """Shrink row batches so ``neg_z`` stays under ~256 MiB (rows × k_neg × D × 4)."""
    if k_neg <= 0 or embed_dim <= 0:
        return row_chunk
    bytes_per_row = k_neg * embed_dim * 4
    max_rows = max(1, max_neg_tensor_bytes // bytes_per_row)
    return min(row_chunk, max_rows)


def _neg_k_chunk_for_subsample(
    row_chunk: int,
    embed_dim: int,
    *,
    max_neg_tensor_bytes: int = 32 * 1024 * 1024,
) -> int:
    """Chunk size along the negative axis (rows × k_chunk × D × 4 ≤ ~32 MiB)."""
    if row_chunk <= 0 or embed_dim <= 0:
        return 128
    k_chunk = max_neg_tensor_bytes // (row_chunk * embed_dim * 4)
    return max(1, int(k_chunk))


def _neg_logsumexp_subsampled(
    z_b: torch.Tensor,
    candidate_z: torch.Tensor,
    neg_idx: torch.Tensor,
    valid: torch.Tensor,
    temperature: float,
    *,
    k_chunk: int,
) -> torch.Tensor:
    """Logsumexp over subsampled negatives without materializing (rows, k_neg, D)."""
    k_neg = int(neg_idx.shape[1])
    log_neg = None
    for k0 in range(0, k_neg, k_chunk):
        k1 = min(k0 + k_chunk, k_neg)
        idx_slice = neg_idx[:, k0:k1].clamp(min=0)
        valid_slice = valid[:, k0:k1]
        neg_z = candidate_z[idx_slice]
        logits = (z_b.unsqueeze(1) * neg_z).sum(dim=-1) / temperature
        logits = torch.where(valid_slice, logits, torch.full_like(logits, float("-inf")))
        ls = logits.logsumexp(dim=1)
        log_neg = ls if log_neg is None else torch.logaddexp(log_neg, ls)
    if log_neg is None:
        return z_b.new_full((z_b.shape[0],), float("-inf"))
    return log_neg


def _directional_aligned_infonce(
    z_anchor: torch.Tensor,
    z_other: torch.Tensor,
    edge_ids: torch.Tensor,
    temperature: float,
    *,
    row_chunk: int = 512,
    col_chunk: int = 1024,
    num_neg_samples: Optional[int] = None,
    memory_queue: Optional[EdgeMemoryQueue] = None,
    morph_bin_ids: Optional[torch.Tensor] = None,
    max_soft_positives: Optional[int] = None,
) -> torch.Tensor:
    if z_anchor.shape != z_other.shape:
        raise ValueError("Aligned directional InfoNCE expects z_anchor and z_other to share shape.")
    if edge_ids.shape[0] != z_anchor.shape[0]:
        raise ValueError("edge_ids length must match aligned embedding rows.")
    if z_anchor.shape[0] == 0:
        return z_anchor.new_zeros(())

    queue_z, queue_ids = (
        memory_queue.get(device=z_anchor.device, dtype=z_anchor.dtype)
        if memory_queue is not None
        else (
            z_anchor.new_empty((0, z_anchor.shape[1])),
            edge_ids.new_empty((0,), dtype=torch.long),
        )
    )
    losses = []
    use_neg_subsample = num_neg_samples is not None and int(num_neg_samples) > 0
    k_neg = int(num_neg_samples) if use_neg_subsample else 0
    embed_dim = int(z_anchor.shape[1])
    k_chunk = 1
    if use_neg_subsample:
        row_chunk = _infonce_row_chunk_for_neg_subsample(
            row_chunk, k_neg, embed_dim
        )
        k_chunk = _neg_k_chunk_for_subsample(row_chunk, embed_dim)

    candidate_ids = edge_ids
    candidate_z = z_other
    if queue_ids.numel() > 0:
        candidate_ids = torch.cat([candidate_ids, queue_ids], dim=0)
        candidate_z = torch.cat([candidate_z, queue_z], dim=0)

    use_morph = morph_bin_ids is not None
    bin_inv: Optional[torch.Tensor] = None
    bin_starts: Optional[torch.Tensor] = None
    bin_sorted_idx: Optional[torch.Tensor] = None
    if use_morph:
        bin_inv, bin_starts, bin_sorted_idx = _bin_positive_index_segments(morph_bin_ids)

    for i0 in range(0, z_anchor.shape[0], row_chunk):
        i1 = min(i0 + row_chunk, z_anchor.shape[0])
        z_b = z_anchor[i0:i1]
        ids_b = edge_ids[i0:i1]
        chunk_rows = int(z_b.shape[0])
        log_num = z_b.new_empty((chunk_rows,))

        if use_morph and bin_inv is not None and bin_starts is not None and bin_sorted_idx is not None:
            bins_b = morph_bin_ids[i0:i1]
            for ii in range(chunk_rows):
                i = i0 + ii
                g = int(bin_inv[i].item())
                s, e = int(bin_starts[g].item()), int(bin_starts[g + 1].item())
                pos_idx = bin_sorted_idx[s:e]
                if max_soft_positives is not None and max_soft_positives > 0:
                    pos_idx = _cap_positive_row_indices(pos_idx, i, int(max_soft_positives))
                log_num[ii] = _logsumexp_positive_logits_bin_grouped(
                    z_b[ii], z_other, pos_idx, temperature
                )
        else:
            z_pos = z_other[i0:i1]
            log_num = ((z_b * z_pos).sum(dim=1) / temperature)

        if use_neg_subsample:
            if use_morph and morph_bin_ids is not None:
                neg_idx = _sample_negative_indices_by_bin(
                    morph_bin_ids,
                    candidate_ids,
                    ids_b,
                    bins_b,
                    k_neg,
                )
            else:
                neg_idx = _sample_negative_indices_batched_gpu(candidate_ids, ids_b, k_neg)
            if neg_idx.numel() == 0:
                log_denom = log_num
            else:
                valid = neg_idx >= 0
                log_neg = _neg_logsumexp_subsampled(
                    z_b,
                    candidate_z,
                    neg_idx,
                    valid,
                    temperature,
                    k_chunk=k_chunk,
                )
                log_denom = torch.where(
                    torch.isfinite(log_neg),
                    torch.logaddexp(log_num, log_neg),
                    log_num,
                )
        else:
            log_denom: Optional[torch.Tensor] = None
            for j0 in range(0, z_other.shape[0], col_chunk):
                j1 = min(j0 + col_chunk, z_other.shape[0])
                logits = (z_b @ z_other[j0:j1].T) / temperature
                ls = logits.logsumexp(dim=1)
                log_denom = ls if log_denom is None else torch.logaddexp(log_denom, ls)

            if log_denom is None:
                continue
            log_queue = _logsumexp_over_queue(
                z_b,
                ids_b,
                queue_z,
                queue_ids,
                temperature,
                col_chunk,
            )
            if log_queue is not None:
                log_denom = torch.logaddexp(log_denom, log_queue)

        losses.append(-(log_num - log_denom))

    if not losses:
        return z_anchor.new_zeros(())
    return torch.cat(losses, dim=0).mean()


def build_edge_positive_mask(
    edge_id1: torch.Tensor,
    edge_id2: torch.Tensor,
    *,
    extra_tiers: Tuple[torch.Tensor, ...] = (),
) -> torch.Tensor:
    """
    Construct the full positive mask for edge–edge contrastive loss.

    Currently only the **identity** tier is implemented. Pass optional boolean
    ``extra_tiers`` of the same shape (E1, E2) when adding future positives
    (e.g. shared-account, time-aware) without changing the loss driver API.
    """
    identity = positive_mask_identity(edge_id1, edge_id2)
    if not extra_tiers:
        return identity
    return merge_positive_tiers(identity, *extra_tiers)


def _merged_positive_mask(
    ids1: torch.Tensor,
    ids2: torch.Tensor,
    morph_bin_ids: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """Identity positives OR same morphology bin (debug/tests; training uses bin groups)."""
    if morph_bin_ids is None:
        return None
    if morph_bin_ids.shape[0] != ids1.numel():
        raise ValueError("morph_bin_ids length must match aligned seed count.")
    identity = positive_mask_identity(ids1, ids2)
    morph = morphology_soft_positive_mask(morph_bin_ids)
    return merge_positive_tiers(identity, morph)


def edge_identity_infonce_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    edge_id1: torch.Tensor,
    edge_id2: torch.Tensor,
    temperature: float = 0.5,
    eps: float = 1e-8,
    debug: bool = False,
    num_neg_samples: Optional[int] = None,
    symmetric: bool = True,
    memory_queue: Optional[EdgeMemoryQueue] = None,
    morph_bin_ids: Optional[torch.Tensor] = None,
    max_soft_positives: Optional[int] = 256,
):
    """
    InfoNCE-style edge contrastive loss with identity positives, optionally merged
    with morphology-bin soft positives (same bin across views, Phase M2).

    Inputs may arrive in arbitrary row order; the loss first aligns shared edge ids
    across the two views, then treats those aligned rows as positives. This matches
    the seed-edge-only training path, where large sampled subgraphs are still used
    for message passing but the contrastive objective scales with the surviving seed
    edges rather than every message-passing edge in the subgraph.

    If ``num_neg_samples`` is None or <= 0, the denominator uses all current aligned
    candidates (plus optional queue negatives) via chunked ``logsumexp``. If > 0,
    negatives are sampled on GPU in row batches from the current aligned batch plus
    the optional queue.

    Parameters
    ----------
    morph_bin_ids :
        Optional long tensor aligned with shared seeds after ``_align_shared_edge_pairs``.
        When set, seeds in the same bin are soft positives (cross-view) in addition
        to identity matches. Capped by ``max_soft_positives`` per anchor.
    max_soft_positives :
        Max same-bin positives in the numerator (default 256). ``None`` disables cap.

    Returns
    -------
    Tensor or dict
        Scalar loss, or debug dict when ``debug=True``.
    """
    del eps  # retained for backward compatibility with older callers

    if z1.dim() != 2 or z2.dim() != 2:
        raise ValueError("z1 and z2 must be 2D (E, D).")

    z1 = _normalize_embeddings(z1)
    z2 = _normalize_embeddings(z2)
    z1, ids1, z2, ids2 = _align_shared_edge_pairs(z1, edge_id1, z2, edge_id2)
    if morph_bin_ids is not None:
        morph_bin_ids = _align_morph_bin_ids_to_shared(morph_bin_ids, edge_id1, edge_id2)

    n_pairs = int(ids1.numel())
    if n_pairs == 0:
        loss = (z1.sum() + z2.sum()) * 0.0
        if debug:
            return {
                "loss": loss,
                "n_pairs": 0,
                "queue_size": 0 if memory_queue is None else memory_queue.size,
            }
        return loss

    loss_12 = _directional_aligned_infonce(
        z1,
        z2,
        ids1,
        temperature,
        num_neg_samples=num_neg_samples,
        memory_queue=memory_queue,
        morph_bin_ids=morph_bin_ids,
        max_soft_positives=max_soft_positives,
    )
    if symmetric:
        loss_21 = _directional_aligned_infonce(
            z2,
            z1,
            ids2,
            temperature,
            num_neg_samples=num_neg_samples,
            memory_queue=memory_queue,
            morph_bin_ids=morph_bin_ids,
            max_soft_positives=max_soft_positives,
        )
        loss = 0.5 * (loss_12 + loss_21)
    else:
        loss = loss_12

    if debug:
        out = {
            "loss": loss,
            "n_pairs": n_pairs,
            "queue_size": 0 if memory_queue is None else memory_queue.size,
        }
        max_elems = 2_000_000
        if z1.shape[0] * z2.shape[0] <= max_elems:
            pm = _merged_positive_mask(ids1, ids2, morph_bin_ids)
            out["pos_mask"] = pm if pm is not None else build_edge_positive_mask(ids1, ids2)
            out["logits_12"] = (z1 @ z2.T) / temperature
        else:
            out["pos_mask"] = None
            out["logits_12"] = None
        return out

    return loss


def contrastive_loss(
    z1,
    z2,
    A=None,
    temperature=0.5,
    eps=1e-8,
    debug=False,
):
    """
    Contrastive loss between two graph views.

    Args:
        z1, z2: (N, D) node embeddings (projected)
        A: (N, N) adjacency matrix (forward edges only)
        temperature: float

    Returns:
        scalar loss (or debug dict)
    """

    if z1.shape != z2.shape:
        raise ValueError(f"Shape mismatch: {z1.shape} vs {z2.shape}")

    N = z1.shape[0]
    device = z1.device

    if A is not None:
        assert A.shape[0] == N, "Adjacency must match embedding size"

    # --- Normalize ---
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    # --- Positive mask: A ∪ I ---
    if A is not None:
        A_pos = (A > 0).to(device=device, dtype=torch.bool)
    else:
        A_pos = torch.zeros((N, N), device=device, dtype=torch.bool)

    eye = torch.eye(N, device=device, dtype=torch.bool)

    pos_mask = A_pos | eye  # (N, N)

    # --- Similarity ---
    logits = (z1 @ z2.T) / temperature

    neg_inf = torch.tensor(float("-inf"), device=device, dtype=logits.dtype)

    def directional_loss(logits, pos_mask):
        # denominator: all nodes
        log_denom = torch.logsumexp(logits, dim=1)

        # numerator: positives only
        masked_logits = torch.where(pos_mask, logits, neg_inf)
        log_num = torch.logsumexp(masked_logits, dim=1)

        # safety (should not happen due to identity)
        log_num = torch.where(torch.isfinite(log_num), log_num, torch.zeros_like(log_num))

        loss_vec = -(log_num - log_denom)
        return loss_vec.mean(), log_num, log_denom

    # forward
    loss_12, log_num_12, log_denom_12 = directional_loss(logits, pos_mask)

    # reverse
    loss_21, log_num_21, log_denom_21 = directional_loss(logits.T, pos_mask)

    loss = 0.5 * (loss_12 + loss_21)

    if debug:
        return {
            "loss": loss,
            "logits": logits,
            "pos_mask": pos_mask,
            "log_num_12": log_num_12,
            "log_denom_12": log_denom_12,
            "log_num_21": log_num_21,
            "log_denom_21": log_denom_21,
        }

    return loss
