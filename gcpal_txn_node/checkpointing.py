"""Checkpoint helpers for positive-complete txn-node resume runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


def save_training_checkpoint(
    path: Path,
    *,
    epoch: int,
    encoder: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    rng: np.random.RandomState,
    history: list,
    total_opt_steps: int,
    total_anchor_exposures: int,
    meta: Dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "model_state_dict": encoder.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "numpy_rng_state": rng.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "history": history,
        "total_opt_steps": int(total_opt_steps),
        "total_anchor_exposures": int(total_anchor_exposures),
        "meta": meta,
        # No LR scheduler was used in the original 5ep scouts (constant Adam lr).
        "scheduler_state_dict": None,
        "scheduler_note": "original_scouts_used_constant_adam_lr_1e-3_no_scheduler",
    }
    torch.save(payload, path)


def load_training_checkpoint(
    path: Path,
    *,
    encoder: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    map_location: Optional[str] = None,
) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=map_location or "cpu")
    encoder.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt


def restore_rng_from_checkpoint(ckpt: Dict[str, Any]) -> np.random.RandomState:
    rng = np.random.RandomState()
    rng.set_state(ckpt["numpy_rng_state"])
    torch.set_rng_state(ckpt["torch_rng_state"])
    if ckpt.get("cuda_rng_state_all") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(ckpt["cuda_rng_state_all"])
    return rng
