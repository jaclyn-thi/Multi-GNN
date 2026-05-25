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

    Args:
        edge_id1: (E1,) long
        edge_id2: (E2,) long

    Returns:
        Boolean tensor of shape (E1, E2).
    """
    return edge_id1.unsqueeze(1) == edge_id2.unsqueeze(0)


def merge_positive_tiers(*masks: torch.Tensor) -> torch.Tensor:
    """
    Combine positive tiers with logical OR (e.g. identity | shared_account later).

    Args:
        masks: one or more boolean (E1, E2) tensors.

    Returns:
        Union of all masks.
    """
    if not masks:
        raise ValueError("merge_positive_tiers requires at least one mask.")
    out = masks[0]
    for m in masks[1:]:
        out = out | m
    return out


def _sorted_edge_ids(edge_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """``sorted_ids = edge_ids[sort_idx]`` in non-decreasing order for :func:`torch.searchsorted`."""
    sort_idx = torch.argsort(edge_ids)
    return edge_ids[sort_idx], sort_idx


def _sample_negative_edge_indices(
    e_other: int,
    pos_js: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """
    Sample up to ``k`` edge indices in ``[0, e_other)`` that are not in ``pos_js``.

    Runs on **CPU** and returns a small ``LongTensor`` on ``pos_js.device`` so we never
    allocate large ``randperm`` / ``cat`` pools on GPU when VRAM is already full from
    the GNN forwards.
    """
    out_device = pos_js.device
    pos_u = torch.unique(pos_js.long().detach().cpu())
    n_pos = int(pos_u.numel())
    if e_other <= n_pos:
        return torch.empty(0, dtype=torch.long, device=out_device)
    kk = min(int(k), e_other - n_pos)
    if kk <= 0:
        return torch.empty(0, dtype=torch.long, device=out_device)

    pool = torch.empty(0, dtype=torch.long)
    chunk = min(max(kk * 16, 512), 65536)
    for _ in range(24):
        if pool.numel() >= kk + max(kk // 4, 8):
            break
        cand = torch.randint(0, e_other, (chunk,), dtype=torch.long)
        cand = cand[~torch.isin(cand, pos_u)]
        if cand.numel() == 0:
            continue
        pool = torch.cat([pool, cand])
        if pool.numel() > 131072:
            sub = torch.randint(0, pool.numel(), (65536,), dtype=torch.long)
            pool = torch.unique(pool[sub])

    pool = torch.unique(pool)
    pool = pool[~torch.isin(pool, pos_u)]
    if pool.numel() == 0:
        return pool.to(out_device)
    kk_eff = min(kk, int(pool.numel()))
    if pool.numel() <= kk_eff:
        return pool.to(out_device)

    if pool.numel() <= 32768:
        idx = torch.randperm(pool.numel(), dtype=torch.long)[:kk_eff]
        return pool[idx].to(out_device)

    idx = torch.randint(0, pool.numel(), (kk_eff * 3,), dtype=torch.long)
    chosen = torch.unique(pool[idx])
    for _ in range(12):
        if chosen.numel() >= kk_eff:
            break
        idx2 = torch.randint(0, pool.numel(), (kk_eff * 3,), dtype=torch.long)
        chosen = torch.unique(torch.cat([chosen, pool[idx2]]))
    if chosen.numel() > kk_eff:
        idx3 = torch.randint(0, chosen.numel(), (kk_eff * 2,), dtype=torch.long)
        chosen = torch.unique(chosen[idx3])
    return chosen[:kk_eff].to(out_device)


def _logsumexp_neg_logits_row(
    row_z: torch.Tensor,
    z_other: torch.Tensor,
    neg_js: torch.Tensor,
    temperature: float,
    neg_col_chunk: int = 512,
) -> torch.Tensor:
    """``logsumexp`` over similarities to ``z_other[neg_js]`` in column chunks."""
    log_acc: Optional[torch.Tensor] = None
    for s0 in range(0, neg_js.numel(), neg_col_chunk):
        s1 = min(s0 + neg_col_chunk, neg_js.numel())
        sl = neg_js[s0:s1]
        part = (row_z @ z_other[sl].T) / temperature
        lp = part.logsumexp(dim=1).squeeze(0)
        log_acc = lp if log_acc is None else torch.logaddexp(log_acc, lp)
    assert log_acc is not None
    return log_acc


def _directional_identity_infonce_chunked(
    z_anchor: torch.Tensor,
    z_other: torch.Tensor,
    id_anchor: torch.Tensor,
    sorted_ids_other: torch.Tensor,
    sort_idx_other: torch.Tensor,
    temperature: float,
    row_chunk: int = 512,
    col_chunk: int = 1024,
    num_neg_samples: Optional[int] = None,
) -> torch.Tensor:
    """
    InfoNCE-style directional loss without ``E_anchor × E_other`` logits.

    If ``num_neg_samples`` is None or <= 0, the denominator is a full (chunked)
    ``logsumexp`` over all edges in ``z_other``. If > 0, the denominator uses only
    positives plus ``num_neg_samples`` uniformly sampled negatives (biased but cheap).
    """
    e_anchor = z_anchor.shape[0]
    e_other = z_other.shape[0]
    losses = []
    use_neg_subsample = num_neg_samples is not None and int(num_neg_samples) > 0
    k_neg = int(num_neg_samples) if use_neg_subsample else 0

    for i0 in range(0, e_anchor, row_chunk):
        i1 = min(i0 + row_chunk, e_anchor)
        z_b = z_anchor[i0:i1]
        b = z_b.shape[0]
        ids_b = id_anchor[i0:i1]
        lo = torch.searchsorted(sorted_ids_other, ids_b, right=False)
        hi = torch.searchsorted(sorted_ids_other, ids_b, right=True)
        has_pos = lo < hi
        if not has_pos.any():
            continue

        if use_neg_subsample:
            block_vals = []
            for r in range(b):
                if not has_pos[r]:
                    continue
                row_z = z_b[r : r + 1]
                js = sort_idx_other[lo[r] : hi[r]]
                logits_pos = (row_z @ z_other[js].T) / temperature
                log_num = logits_pos.logsumexp(dim=1).squeeze(0)
                log_num = torch.where(
                    torch.isfinite(log_num),
                    log_num,
                    torch.zeros((), device=log_num.device, dtype=log_num.dtype),
                )
                neg_js = _sample_negative_edge_indices(e_other, js, k_neg)
                if neg_js.numel() == 0:
                    log_denom = log_num
                else:
                    log_neg = _logsumexp_neg_logits_row(
                        row_z, z_other, neg_js, temperature, neg_col_chunk=512
                    )
                    log_denom = torch.logaddexp(log_num, log_neg)
                block_vals.append(-(log_num - log_denom))
            if block_vals:
                losses.append(torch.stack(block_vals).mean())
            continue

        log_denom: Optional[torch.Tensor] = None
        for j0 in range(0, e_other, col_chunk):
            j1 = min(j0 + col_chunk, e_other)
            part = (z_b @ z_other[j0:j1].T) / temperature
            ls = part.logsumexp(dim=1)
            log_denom = ls if log_denom is None else torch.logaddexp(log_denom, ls)

        if log_denom is None:
            continue

        block_vals = []
        for r in range(b):
            if not has_pos[r]:
                continue
            row_z = z_b[r : r + 1]
            js = sort_idx_other[lo[r] : hi[r]]
            logits_pos = (row_z @ z_other[js].T) / temperature
            log_num = logits_pos.logsumexp(dim=1).squeeze(0)
            log_num = torch.where(
                torch.isfinite(log_num),
                log_num,
                torch.zeros((), device=log_num.device, dtype=log_num.dtype),
            )
            block_vals.append(-(log_num - log_denom[r]))
        if block_vals:
            losses.append(torch.stack(block_vals).mean())

    if not losses:
        return z_anchor.new_zeros(())
    return torch.stack(losses).mean()


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
):
    """
    InfoNCE-style edge contrastive loss with identity positives only.

    Rows of ``z1`` / ``z2`` need not align; positives are pairs (i, j) with
    ``edge_id1[i] == edge_id2[j]``. Anchors with no positive in the other view
    are skipped.

    If ``num_neg_samples`` is None or <= 0, negatives are **all** other edges in
    the opposite view (denominator via chunked ``logsumexp``). If > 0, each anchor
    uses that many **uniformly sampled** negatives for the denominator (memory
    friendly; biased vs. full softmax).

    Symmetric: mean of directional losses 1→2 and 2→1 over valid anchors.
    If ``symmetric`` is False, only the 1→2 direction is used (e.g. with view-2
    computed under ``torch.no_grad()`` to save memory).

    Without negative subsampling, uses row/column-chunked logits (peak
    ``row_chunk × col_chunk``) instead of dense ``E1×E2`` tensors.
    """
    if z1.dim() != 2 or z2.dim() != 2:
        raise ValueError("z1 and z2 must be 2D (E, D).")

    edge_id1 = edge_id1.long().view(-1)
    edge_id2 = edge_id2.long().view(-1)

    if edge_id1.shape[0] != z1.shape[0] or edge_id2.shape[0] != z2.shape[0]:
        raise ValueError("edge_id lengths must match embedding row counts.")

    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    sorted_ids2, sort_idx2 = _sorted_edge_ids(edge_id2)
    sorted_ids1, sort_idx1 = _sorted_edge_ids(edge_id1)

    lo_12 = torch.searchsorted(sorted_ids2, edge_id1, right=False)
    hi_12 = torch.searchsorted(sorted_ids2, edge_id1, right=True)
    n12 = (lo_12 < hi_12).sum()

    lo_21 = torch.searchsorted(sorted_ids1, edge_id2, right=False)
    hi_21 = torch.searchsorted(sorted_ids1, edge_id2, right=True)
    n21 = (lo_21 < hi_21).sum()

    loss_12 = _directional_identity_infonce_chunked(
        z1, z2, edge_id1, sorted_ids2, sort_idx2, temperature, num_neg_samples=num_neg_samples
    )

    if not symmetric:
        if n12 == 0:
            loss = (z1.sum() + z2.sum()) * 0.0
        else:
            loss = loss_12
    else:
        loss_21 = _directional_identity_infonce_chunked(
            z2, z1, edge_id2, sorted_ids1, sort_idx1, temperature, num_neg_samples=num_neg_samples
        )

        # If one side has no anchors with positives, ignore that direction (0 contribution).
        if n12 == 0 and n21 == 0:
            loss = (z1.sum() + z2.sum()) * 0.0
        elif n12 == 0:
            loss = loss_21
        elif n21 == 0:
            loss = loss_12
        else:
            loss = 0.5 * (loss_12 + loss_21)

    if debug:
        out = {
            "loss": loss,
            "n_anchors_12": int(n12),
            "n_anchors_21": int(n21),
        }
        # Dense debug tensors are O(E1·E2); omit when too large for memory.
        max_elems = 2_000_000
        if z1.shape[0] * z2.shape[0] <= max_elems:
            pos_mask = build_edge_positive_mask(edge_id1, edge_id2)
            out["pos_mask"] = pos_mask
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
