"""Shared GCPAL diagnostic helpers (no torch_geometric dependency)."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def multipositive_infonce_reference(
    z1: torch.Tensor,
    z2: torch.Tensor,
    positive_mask: torch.Tensor,
    *,
    temperature: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """
    Simple dense reference for multi-positive InfoNCE over a toy BxB pool.

    ``positive_mask[i,j]`` marks j as a positive for anchor i (includes identity).
    Positives are excluded from the negative set. Not used in production training.
    """
    if z1.ndim != 2 or z2.ndim != 2:
        raise ValueError("z1/z2 must be (B, D)")
    b = z1.shape[0]
    if z2.shape[0] != b or positive_mask.shape != (b, b):
        raise ValueError("z1, z2, positive_mask must share batch size B")
    if not bool(positive_mask.diagonal().all()):
        raise ValueError("identity must be included on the positive diagonal")

    z1n = F.normalize(z1, dim=1)
    z2n = F.normalize(z2, dim=1)
    logits = (z1n @ z2n.T) / float(temperature)
    pos_logits = logits.masked_fill(~positive_mask, float("-inf"))
    log_num = torch.logsumexp(pos_logits, dim=1)
    log_denom = torch.logsumexp(logits, dim=1)
    loss = -(log_num - log_denom)
    neg_mask = ~positive_mask
    return {
        "logits": logits,
        "log_num": log_num,
        "log_denom": log_denom,
        "loss": loss,
        "neg_mask": neg_mask,
        "positive_mask": positive_mask,
    }


def merge_positive_masks(*masks: torch.Tensor) -> torch.Tensor:
    out = masks[0].clone()
    for m in masks[1:]:
        out |= m
    out.fill_diagonal_(True)
    return out
