"""Symmetric multi-positive InfoNCE for the txn-node baseline.

Positive aggregation modes (``positive_aggregation``):

- ``sum_logsumexp`` (default, backward-compatible):
  ``loss_i = logsumexp(all) - logsumexp(positives)``
- ``logmeanexp_count_normalized``:
  ``loss_i = logsumexp(all) - (logsumexp(positives) - log(|P_i|))``
- ``supcon_mean_logprob``:
  ``loss_i = mean_{j in P_i}[ logsumexp(all) - logit_ij ]``
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

POSITIVE_AGGREGATION_SUM = "sum_logsumexp"
POSITIVE_AGGREGATION_LOGMEANEXP = "logmeanexp_count_normalized"
POSITIVE_AGGREGATION_SUPCON = "supcon_mean_logprob"
POSITIVE_AGGREGATION_MODES = (
    POSITIVE_AGGREGATION_SUM,
    POSITIVE_AGGREGATION_LOGMEANEXP,
    POSITIVE_AGGREGATION_SUPCON,
)
DEFAULT_POSITIVE_AGGREGATION = POSITIVE_AGGREGATION_SUM


def validate_positive_aggregation(mode: str) -> str:
    m = str(mode)
    if m not in POSITIVE_AGGREGATION_MODES:
        raise ValueError(
            f"Unknown positive_aggregation={mode!r}; expected one of {POSITIVE_AGGREGATION_MODES}"
        )
    return m


def _assert_nonempty_positives(
    positive_mask: torch.Tensor,
    row_indices: Optional[torch.Tensor],
) -> torch.Tensor:
    """Return per-row positive counts; fail if any selected row has zero positives."""
    n_pos = positive_mask.float().sum(dim=1)
    if row_indices is None:
        bad = n_pos <= 0
        if bool(bad.any()):
            idxs = bad.nonzero(as_tuple=False).view(-1)[:8].tolist()
            raise RuntimeError(
                f"Empty positive row(s) in InfoNCE mask (examples local rows={idxs}). "
                "Every anchor must retain at least its identity positive."
            )
    else:
        if row_indices.numel() == 0:
            raise ValueError("row_indices must be non-empty")
        n_sel = n_pos[row_indices]
        if bool((n_sel <= 0).any()):
            raise RuntimeError(
                "Empty positive row among anchor rows; identity positive required."
            )
    return n_pos


def multipositive_infonce(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    positive_mask: torch.Tensor,
    *,
    temperature: float = 0.5,
    row_indices: Optional[torch.Tensor] = None,
    positive_aggregation: str = DEFAULT_POSITIVE_AGGREGATION,
) -> Dict[str, torch.Tensor]:
    """Multi-positive InfoNCE; full Softmax denominator includes positives.

    ``positive_mask[i,j]`` marks j as a positive for row i (typically includes identity).
    If ``row_indices`` is set, the scalar loss averages only those rows (anchor-only).
    """
    mode = validate_positive_aggregation(positive_aggregation)
    if z_a.ndim != 2 or z_b.ndim != 2:
        raise ValueError("z_a/z_b must be (B, D)")
    b = z_a.shape[0]
    if z_b.shape[0] != b or positive_mask.shape != (b, b):
        raise ValueError("batch size mismatch")
    za = F.normalize(z_a, dim=1)
    zb = F.normalize(z_b, dim=1)
    logits = (za @ zb.T) / float(temperature)
    n_pos = _assert_nonempty_positives(positive_mask, row_indices)

    pos_logits = logits.masked_fill(~positive_mask, float("-inf"))
    log_num = torch.logsumexp(pos_logits, dim=1)
    log_denom = torch.logsumexp(logits, dim=1)

    if mode == POSITIVE_AGGREGATION_SUM:
        # loss = logsumexp(all) - logsumexp(positives)
        loss_vec = log_denom - log_num
    elif mode == POSITIVE_AGGREGATION_LOGMEANEXP:
        # loss = logsumexp(all) - (logsumexp(pos) - log(|P|))
        loss_vec = log_denom - (log_num - torch.log(n_pos.clamp(min=1.0)))
    elif mode == POSITIVE_AGGREGATION_SUPCON:
        # mean_j [log_denom - logit_j] over positives = log_denom - mean_j logit_j
        pos_sum = (logits * positive_mask.float()).sum(dim=1)
        mean_pos_logit = pos_sum / n_pos.clamp(min=1.0)
        loss_vec = log_denom - mean_pos_logit
    else:  # pragma: no cover
        raise ValueError(mode)

    if row_indices is None:
        loss = loss_vec.mean()
        neg_counts = (~positive_mask).float().sum(dim=1).mean()
        n_pos_mean = n_pos.mean()
    else:
        loss = loss_vec[row_indices].mean()
        neg_counts = (~positive_mask)[row_indices].float().sum(dim=1).mean()
        n_pos_mean = n_pos[row_indices].mean()
    return {
        "loss": loss,
        "loss_vec": loss_vec,
        "logits": logits,
        "log_num": log_num,
        "log_denom": log_denom,
        "n_pos": n_pos,
        "n_pos_mean": n_pos_mean,
        "neg_mask": ~positive_mask,
        "n_unique_negatives_mean": neg_counts,
        "positive_aggregation": mode,
    }


def symmetric_multipositive_infonce(
    z1: torch.Tensor,
    z2: torch.Tensor,
    positive_mask: torch.Tensor,
    *,
    temperature: float = 0.5,
    row_indices: Optional[torch.Tensor] = None,
    positive_aggregation: str = DEFAULT_POSITIVE_AGGREGATION,
) -> Dict[str, torch.Tensor]:
    """Average of both directions (symmetric; no detached branch)."""
    mode = validate_positive_aggregation(positive_aggregation)
    fwd = multipositive_infonce(
        z1,
        z2,
        positive_mask,
        temperature=temperature,
        row_indices=row_indices,
        positive_aggregation=mode,
    )
    # Reverse: OR with transpose so directed edges contribute both ways.
    mask_sym = positive_mask | positive_mask.T
    mask_sym.fill_diagonal_(True)
    rev = multipositive_infonce(
        z2,
        z1,
        mask_sym,
        temperature=temperature,
        row_indices=row_indices,
        positive_aggregation=mode,
    )
    loss = 0.5 * (fwd["loss"] + rev["loss"])
    return {
        "loss": loss,
        "loss_fwd": fwd["loss"],
        "loss_rev": rev["loss"],
        "loss_vec_fwd": fwd["loss_vec"],
        "n_unique_negatives_mean": fwd["n_unique_negatives_mean"],
        "positive_mask_sym": mask_sym,
        "log_num_fwd": fwd["log_num"],
        "log_denom_fwd": fwd["log_denom"],
        "n_pos_fwd": fwd["n_pos"],
        "logits_fwd": fwd["logits"],
        "positive_aggregation": mode,
    }


def mixed_gcpal_loss(
    z_r1: torch.Tensor,
    z_r2: torch.Tensor,
    z_knn: Optional[torch.Tensor],
    positive_mask: torch.Tensor,
    *,
    lambda_mix: float = 0.3,
    temperature: float = 0.5,
    use_knn: bool = True,
    row_indices: Optional[torch.Tensor] = None,
    positive_aggregation: str = DEFAULT_POSITIVE_AGGREGATION,
) -> Dict[str, torch.Tensor]:
    """L = λ L(r1,r2) + (1-λ) L(r2, KNN). Control mode: KNN term omitted (λ ignored → L_rr only)."""
    mode = validate_positive_aggregation(positive_aggregation)
    rr = symmetric_multipositive_infonce(
        z_r1,
        z_r2,
        positive_mask,
        temperature=temperature,
        row_indices=row_indices,
        positive_aggregation=mode,
    )
    out: Dict[str, torch.Tensor] = {
        "loss_random_random": rr["loss"],
        "n_unique_negatives_mean": rr["n_unique_negatives_mean"],
        "positive_aggregation": mode,
        "log_num_fwd": rr["log_num_fwd"],
        "log_denom_fwd": rr["log_denom_fwd"],
        "n_pos_fwd": rr["n_pos_fwd"],
        "logits_fwd": rr["logits_fwd"],
        "loss_fwd": rr["loss_fwd"],
        "loss_rev": rr["loss_rev"],
        "loss_vec_fwd": rr["loss_vec_fwd"],
    }
    if use_knn:
        if z_knn is None:
            raise ValueError("use_knn=True requires z_knn")
        rk = symmetric_multipositive_infonce(
            z_r2,
            z_knn,
            positive_mask,
            temperature=temperature,
            row_indices=row_indices,
            positive_aggregation=mode,
        )
        out["loss_random_knn"] = rk["loss"]
        out["loss"] = float(lambda_mix) * rr["loss"] + (1.0 - float(lambda_mix)) * rk["loss"]
        out["lambda_mix"] = torch.tensor(float(lambda_mix), device=z_r1.device)
        out["logits_rk_fwd"] = rk["logits_fwd"]
        out["log_num_rk_fwd"] = rk["log_num_fwd"]
        out["log_denom_rk_fwd"] = rk["log_denom_fwd"]
    else:
        out["loss_random_knn"] = torch.zeros((), device=z_r1.device)
        out["loss"] = rr["loss"]
        out["lambda_mix"] = torch.tensor(1.0, device=z_r1.device)
    return out


def similarity_type_diagnostics(
    logits: torch.Tensor,
    *,
    positive_mask: torch.Tensor,
    identity_mask: torch.Tensor,
    structural_mask: torch.Tensor,
    knn_mask: torch.Tensor,
    row_indices: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Mean logits (sim/τ) for identity / structural / KNN / non-positive (anchor rows)."""
    rows = slice(None) if row_indices is None else row_indices

    def _mean_where(mask: torch.Tensor) -> float:
        m = mask[rows]
        if not bool(m.any()):
            return float("nan")
        return float(logits[rows][m].mean().item())

    return {
        "mean_sim_identity": _mean_where(identity_mask),
        "mean_sim_structural": _mean_where(structural_mask),
        "mean_sim_knn": _mean_where(knn_mask),
        "mean_sim_positive_any": _mean_where(positive_mask),
        "mean_sim_nonpositive": _mean_where(~positive_mask),
    }


def build_positive_mask(
    batch_size: int,
    *,
    identity: bool,
    structural_pairs: torch.Tensor,
    knn_pairs: torch.Tensor,
    device: torch.device,
    anchor_local: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Build BxB bool mask from local-index pairs (src, dst) within the batch.

    ``structural_pairs`` / ``knn_pairs`` are shape [2, E] local indices.
    If ``anchor_local`` is provided, diagnostic means are restricted to those rows
    (context nodes may appear as positives/negatives but are not primary anchors).
    """
    mask = torch.zeros((batch_size, batch_size), dtype=torch.bool, device=device)
    id_mask = torch.zeros_like(mask)
    if identity:
        if anchor_local is None:
            mask.fill_diagonal_(True)
            id_mask.fill_diagonal_(True)
        else:
            mask[anchor_local, anchor_local] = True
            id_mask[anchor_local, anchor_local] = True
    if structural_pairs.numel():
        mask[structural_pairs[0], structural_pairs[1]] = True
    if knn_pairs.numel():
        mask[knn_pairs[0], knn_pairs[1]] = True
    if identity and anchor_local is None:
        mask.fill_diagonal_(True)
        id_mask.fill_diagonal_(True)
    elif identity and anchor_local is not None:
        mask[anchor_local, anchor_local] = True
        id_mask[anchor_local, anchor_local] = True

    eye = torch.eye(batch_size, dtype=torch.bool, device=device)
    struct_only = torch.zeros_like(mask)
    if structural_pairs.numel():
        struct_only[structural_pairs[0], structural_pairs[1]] = True
        struct_only &= ~eye
    knn_only = torch.zeros_like(mask)
    if knn_pairs.numel():
        knn_only[knn_pairs[0], knn_pairs[1]] = True
        knn_only &= ~eye

    if anchor_local is None:
        rows = slice(None)
        n_rows = float(batch_size)
        n_id = float(mask.diagonal().sum().item())
    else:
        rows = anchor_local
        n_rows = float(anchor_local.numel())
        n_id = float(mask[anchor_local, anchor_local].sum().item()) if n_rows else 0.0

    pos_rows = mask[rows].float()
    struct_rows = struct_only[rows].float()
    knn_rows = knn_only[rows].float()
    neg_rows = (~mask)[rows].float()
    stats: Dict[str, float] = {
        "mean_identity": (1.0 if identity else 0.0),
        "mean_structural_pos": float(struct_rows.sum(dim=1).mean().item()) if n_rows else 0.0,
        "mean_knn_pos": float(knn_rows.sum(dim=1).mean().item()) if n_rows else 0.0,
        "mean_total_pos": float(pos_rows.sum(dim=1).mean().item()) if n_rows else 0.0,
        "median_structural_pos": float(struct_rows.sum(dim=1).median().item()) if n_rows else 0.0,
        "median_knn_pos": float(knn_rows.sum(dim=1).median().item()) if n_rows else 0.0,
        "frac_with_structural": float((struct_rows.sum(dim=1) > 0).float().mean().item()) if n_rows else 0.0,
        "frac_with_knn": float((knn_rows.sum(dim=1) > 0).float().mean().item()) if n_rows else 0.0,
        "n_identity_diag": n_id,
        "positive_mask_density": float(pos_rows.mean().item()) if n_rows else 0.0,
        "negative_mask_density": float(neg_rows.mean().item()) if n_rows else 0.0,
        "n_anchor_rows": n_rows,
    }
    # Type masks for optional step diagnostics (callers must strip before JSON dump).
    stats["_identity_mask"] = id_mask  # type: ignore[assignment]
    stats["_structural_mask"] = struct_only  # type: ignore[assignment]
    stats["_knn_mask"] = knn_only  # type: ignore[assignment]
    return mask, stats
