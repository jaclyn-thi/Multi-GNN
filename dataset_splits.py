"""
Temporal train / val / test index splits for edge-level datasets.

AMLWorld uses calendar-day buckets; PaySim uses hourly steps (Timestamp = step
* 3600 from the formatter). Both preserve temporal order and target ~60/20/20
edge counts via the same bucket-and-partition search used in the original
``data_loading.py``.
"""

from __future__ import annotations

import itertools
import logging
from typing import List, Sequence, Tuple

import torch

from dataset_specs import EdgeDatasetSpec

_SECONDS_PER_BUCKET = {
    "calendar_day": 24 * 3600,
    "hourly_step": 3600,
}


def _bucket_seconds(split_mode: str) -> int:
    if split_mode not in _SECONDS_PER_BUCKET:
        raise ValueError(
            f"Unsupported split_mode {split_mode!r}; "
            f"expected one of {sorted(_SECONDS_PER_BUCKET)}"
        )
    return _SECONDS_PER_BUCKET[split_mode]


def temporal_edge_split(
    timestamps: torch.Tensor,
    y: torch.Tensor,
    spec: EdgeDatasetSpec,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[List[int]]]:
    """
    Compute ``tr_inds``, ``val_inds``, ``te_inds`` from edge timestamps.

    Returns
    -------
    tr_inds, val_inds, te_inds :
        Index tensors into the full edge list (chronological CSV order).
    split :
        Bucket ids per partition ``[train_buckets, val_buckets, test_buckets]``.
    """
    split_per = list(spec.split_fractions)
    if abs(sum(split_per) - 1.0) > 1e-6:
        raise ValueError(f"split_fractions must sum to 1.0 (got {split_per})")

    bucket_sec = _bucket_seconds(spec.split_mode)
    n_buckets = int(timestamps.max().item() / bucket_sec + 1)
    n_samples = int(y.shape[0])

    daily_inds: List[torch.Tensor] = []
    daily_trans: List[int] = []
    for bucket in range(n_buckets):
        left = bucket * bucket_sec
        right = (bucket + 1) * bucket_sec
        inds = torch.where((timestamps >= left) & (timestamps < right))[0]
        daily_inds.append(inds)
        daily_trans.append(int(inds.shape[0]))

    daily_totals = daily_trans
    d_ts = daily_totals
    bucket_ids = list(range(len(d_ts)))
    split_scores = {}
    for i, j in itertools.combinations(bucket_ids, 2):
        if j < i:
            continue
        split_totals = [sum(d_ts[:i]), sum(d_ts[i:j]), sum(d_ts[j:])]
        split_totals_sum = sum(split_totals)
        split_props = [v / split_totals_sum for v in split_totals]
        split_error = [abs(v - t) / t for v, t in zip(split_props, split_per)]
        split_scores[(i, j)] = max(split_error)

    i, j = min(split_scores, key=split_scores.get)
    split = [list(range(i)), list(range(i, j)), list(range(j, len(daily_totals)))]

    unit = "day" if spec.split_mode == "calendar_day" else "step"
    logging.info(
        "Temporal split (%s, mode=%s): train_%ss=%s val_%ss=%s test_%ss=%s",
        spec.name,
        spec.split_mode,
        unit,
        split[0][:5],
        unit,
        split[1][:5],
        unit,
        split[2][:5],
    )

    split_inds: dict = {k: [] for k in range(3)}
    for part in range(3):
        for bucket in split[part]:
            split_inds[part].append(daily_inds[bucket])

    tr_inds = torch.cat(split_inds[0])
    val_inds = torch.cat(split_inds[1])
    te_inds = torch.cat(split_inds[2])
    return tr_inds, val_inds, te_inds, split


def log_split_label_stats(
    y: torch.Tensor,
    tr_inds: torch.Tensor,
    val_inds: torch.Tensor,
    te_inds: torch.Tensor,
    *,
    label_col: str,
    split_mode: str,
) -> None:
    n = max(int(y.shape[0]), 1)
    unit = "days" if split_mode == "calendar_day" else "steps"
    logging.info(
        "Total train samples: %.2f%% || positive rate: %.4f%% || (%s)",
        tr_inds.shape[0] / n * 100,
        y[tr_inds].float().mean() * 100,
        label_col,
    )
    logging.info(
        "Total val samples: %.2f%% || positive rate: %.4f%%",
        val_inds.shape[0] / n * 100,
        y[val_inds].float().mean() * 100,
    )
    logging.info(
        "Total test samples: %.2f%% || positive rate: %.4f%%",
        te_inds.shape[0] / n * 100,
        y[te_inds].float().mean() * 100,
    )
    logging.info("Split buckets use %s (%s)", unit, split_mode)
