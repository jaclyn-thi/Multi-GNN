"""One-logit weighted CE mathematically matched to two-logit CrossEntropyLoss.

For logits [0, z] and class weights [w0, w1]:

  NLL(y=0) = softplus(z)
  NLL(y=1) = softplus(-z)

  L = sum_i w[y_i] * NLL_i / sum_i w[y_i]

which equals CrossEntropyLoss(weight=[w0,w1], reduction='mean') on [0,z].
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


ArrayLike = Union[np.ndarray, torch.Tensor]


def load_class_weights_from_checkpoint(ckpt_path: str) -> Tuple[float, float, str]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config") or {}
    if "w_ce1" not in cfg or "w_ce2" not in cfg:
        raise KeyError(f"w_ce1/w_ce2 missing in checkpoint config: {ckpt_path}")
    return float(cfg["w_ce1"]), float(cfg["w_ce2"]), "checkpoint['config']"


def matched_weighted_ce_from_one_logit(
    logits: torch.Tensor,
    y: torch.Tensor,
    weight: Sequence[float],
) -> torch.Tensor:
    """Scalar loss matching CE(weight, mean) on equivalent two-logit [0, z]."""
    z = logits.reshape(-1)
    y_long = y.reshape(-1).long()
    if z.shape[0] != y_long.shape[0]:
        raise ValueError(f"logits/y length mismatch: {z.shape[0]} vs {y_long.shape[0]}")
    w = torch.as_tensor(list(weight), dtype=z.dtype, device=z.device)
    if w.numel() != 2:
        raise ValueError("weight must be length-2 [w0, w1]")
    # y=1 -> w1 * softplus(-z); y=0 -> w0 * softplus(z)
    nll = torch.where(y_long == 1, F.softplus(-z), F.softplus(z))
    wy = w[y_long]
    denom = wy.sum()
    if float(denom.detach().cpu()) <= 0.0:
        raise RuntimeError("sum(weight[y]) is non-positive")
    return (wy * nll).sum() / denom


def two_logit_from_one(z: torch.Tensor) -> torch.Tensor:
    z = z.reshape(-1)
    zeros = torch.zeros_like(z)
    return torch.stack([zeros, z], dim=-1)


def matched_weighted_ce_numpy(
    logit: np.ndarray,
    y: np.ndarray,
    weight: Sequence[float],
) -> float:
    z = torch.as_tensor(np.asarray(logit).reshape(-1), dtype=torch.float64)
    yy = torch.as_tensor(np.asarray(y).reshape(-1), dtype=torch.int64)
    return float(matched_weighted_ce_from_one_logit(z, yy, weight).item())


def unweighted_binary_ce_numpy(logit: np.ndarray, y: np.ndarray) -> float:
    z = torch.as_tensor(np.asarray(logit).reshape(-1), dtype=torch.float64)
    yy = torch.as_tensor(np.asarray(y).reshape(-1), dtype=torch.float64)
    return float(F.binary_cross_entropy_with_logits(z, yy, reduction="mean").item())


def unweighted_binary_ce_from_proba(y: np.ndarray, p: np.ndarray, eps: float = 1e-12) -> float:
    y64 = np.asarray(y, dtype=np.float64).reshape(-1)
    p64 = np.clip(np.asarray(p, dtype=np.float64).reshape(-1), eps, 1.0 - eps)
    return float(np.mean(-y64 * np.log(p64) - (1.0 - y64) * np.log(1.0 - p64)))


def supervised_weighted_ce_two_logit(
    y: np.ndarray,
    logits_2: np.ndarray,
    weight: Sequence[float],
) -> float:
    y_t = torch.as_tensor(np.asarray(y).reshape(-1), dtype=torch.int64)
    z = torch.as_tensor(np.asarray(logits_2), dtype=torch.float64)
    if z.ndim != 2 or z.shape[1] != 2:
        raise ValueError(f"expected [N,2] logits, got {tuple(z.shape)}")
    w = torch.as_tensor(list(weight), dtype=torch.float64)
    return float(F.cross_entropy(z, y_t, weight=w, reduction="mean").item())


def class_weight_summary(w0: float, w1: float, source: str) -> Dict[str, object]:
    return {
        "w0": float(w0),
        "w1": float(w1),
        "source": source,
        "formula": (
            "per_example = w1*softplus(-z) if y=1 else w0*softplus(z); "
            "loss = sum(per_example) / sum(weight[y])"
        ),
        "equivalent_to": "CrossEntropyLoss(weight=[w0,w1], reduction='mean') on logits [0,z]",
    }
