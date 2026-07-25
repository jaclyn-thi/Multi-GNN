"""Temporal-flow soft positives for contrastive SSL (M2-style, identity-primary).

Uses validated causal ``temporal_flow_causal`` features only. Quantile bins are fit on
the **train split only**. Soft positives are other train edges that share enough
feature bins with the anchor; they enter InfoNCE as **low-weight** extras while the
same-transaction augmented view remains the primary positive (weight 1.0).

No AML labels are used.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from morphology.temporal_flow_aux import (
    DEFAULT_CACHE_REL,
    _fit_bins_train_only,
    _load_cache,
)
from morphology.temporal_flow_causal import TEMPORAL_FLOW_CAUSAL_FEATURE_NAMES

ATTACH_NOTE = (
    "batch-local soft positives from shared temporal_flow_causal quantile bins; "
    "identity pair remains primary (weight 1.0)"
)


@dataclass
class TemporalFlowSoftPositiveConfig:
    enabled: bool = False
    weight: float = 0.05
    n_bins: int = 5
    min_shared_bins: int = 3
    max_per_anchor: int = 16
    cache_dir: str = ""
    metadata_path: str = ""
    feature_names: Tuple[str, ...] = TEMPORAL_FLOW_CAUSAL_FEATURE_NAMES
    uses_labels: bool = False
    bin_fit_split: str = "train"
    hub_avoidance: str = "prefer_low_sender_activity"


@dataclass
class TemporalFlowSoftPositiveContext:
    """Train-edge bin ids + hub scores for sampling."""

    bin_ids: torch.Tensor  # [N, F] long on CPU (or device)
    hub_score: torch.Tensor  # [N] float — higher = more hub-like (sender 7d activity)
    train_edge_ids: np.ndarray
    n_classes: List[int]
    meta: Dict[str, Any] = field(default_factory=dict)


def _resolve_cache(args, data: str) -> Path:
    raw = getattr(args, "temporal_flow_soft_positive_cache", None) or getattr(
        args, "aux_temporal_flow_cache", None
    )
    if raw:
        return Path(raw)
    return DEFAULT_CACHE_REL / str(data)


def setup_temporal_flow_soft_positives(
    args,
    device: torch.device,
    *,
    data_name: str,
) -> Tuple[Optional[TemporalFlowSoftPositiveConfig], Optional[TemporalFlowSoftPositiveContext]]:
    enabled = getattr(args, "temporal_flow_soft_positives", False)
    if isinstance(enabled, str):
        enabled = enabled.lower() in ("1", "true", "yes", "on")
    if not enabled:
        return None, None

    weight = float(getattr(args, "temporal_flow_soft_positive_weight", 0.05))
    n_bins = int(getattr(args, "temporal_flow_soft_positive_bins", 5))
    min_shared = int(getattr(args, "temporal_flow_soft_positive_min_shared_bins", 3))
    max_per = int(getattr(args, "temporal_flow_soft_positive_max_per_anchor", 16))
    if weight < 0:
        raise ValueError("--temporal_flow_soft_positive_weight must be >= 0")
    if n_bins < 2:
        raise ValueError("--temporal_flow_soft_positive_bins must be >= 2")
    if min_shared < 1:
        raise ValueError("--temporal_flow_soft_positive_min_shared_bins must be >= 1")
    if max_per < 1:
        raise ValueError("--temporal_flow_soft_positive_max_per_anchor must be >= 1")
    if min_shared > len(TEMPORAL_FLOW_CAUSAL_FEATURE_NAMES):
        raise ValueError(
            f"min_shared_bins={min_shared} exceeds n_features={len(TEMPORAL_FLOW_CAUSAL_FEATURE_NAMES)}"
        )

    cache_dir = _resolve_cache(args, data_name)
    features, edge_id, train_ids, cache_meta = _load_cache(cache_dir)
    if cache_meta.get("causal_history_policy", {}).get("uses_labels"):
        raise RuntimeError("Refuse temporal-flow soft positives: cache reports uses_labels=True")

    bin_ids_np, n_classes, bin_edges = _fit_bins_train_only(features, train_ids, n_bins)
    # Hub score: sender past 7d activity feature (index 2 in TEMPORAL_FLOW_CAUSAL_FEATURE_NAMES)
    hub = features[:, 2].astype(np.float32)
    hub = np.nan_to_num(hub, nan=0.0, posinf=0.0, neginf=0.0)

    cfg = TemporalFlowSoftPositiveConfig(
        enabled=True,
        weight=weight,
        n_bins=n_bins,
        min_shared_bins=min_shared,
        max_per_anchor=max_per,
        cache_dir=str(cache_dir),
        uses_labels=False,
        bin_fit_split="train",
    )

    unique_name = str(getattr(args, "unique_name", "run") or "run")
    meta_dir = Path("results/diagnostics/temporal_flow_soft_positive_preprocess")
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta_path = meta_dir / f"{unique_name}_soft_pos_bins.json"
    meta = {
        "feature_names": list(TEMPORAL_FLOW_CAUSAL_FEATURE_NAMES),
        "n_bins_requested": n_bins,
        "n_classes_per_feature": n_classes,
        "min_shared_bins": min_shared,
        "max_per_anchor": max_per,
        "weight": weight,
        "bin_fit_split": "train",
        "train_n": int(train_ids.shape[0]),
        "bin_edges": [None if e is None else e.tolist() for e in bin_edges],
        "hub_avoidance": cfg.hub_avoidance,
        "hub_feature": "log1p_sender_past_7d_count",
        "attach_policy": ATTACH_NOTE,
        "uses_labels": False,
        "no_ssl_label_use": True,
        "cache_dir": str(cache_dir),
        "cache_version": cache_meta.get("cache_version"),
        "timestamp_tie_policy": (cache_meta.get("timestamp_handling") or {}).get("timestamp_ties"),
        "sampling": "batch-local; among candidates prefer lower hub_score then deterministic RNG cap",
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    cfg.metadata_path = str(meta_path)

    ctx = TemporalFlowSoftPositiveContext(
        bin_ids=torch.as_tensor(bin_ids_np, dtype=torch.long),
        hub_score=torch.as_tensor(hub, dtype=torch.float32),
        train_edge_ids=train_ids,
        n_classes=n_classes,
        meta=meta,
    )
    logging.info(
        "Temporal-flow soft positives enabled: bins=%d min_shared=%d max_per_anchor=%d "
        "weight=%.4f fit=train labels=False meta=%s",
        n_bins,
        min_shared,
        max_per,
        weight,
        meta_path,
    )
    return cfg, ctx


def build_temporal_flow_soft_positive_batch(
    seed_ids: torch.Tensor,
    z2_seed: torch.Tensor,
    cfg: TemporalFlowSoftPositiveConfig,
    ctx: TemporalFlowSoftPositiveContext,
    *,
    epoch: int,
    step: int,
    stats: Optional[Dict[str, float]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build knn-style soft-positive tensors for the current seed batch.

    Returns
    -------
    pos_ids, pos_weights, pos_valid, pos_z2
        Shapes: [B, M], [B, M], [B, M], [B, M, D] on the same device as ``z2_seed``.
        Soft positives are other seeds in the **same batch** sharing >= min_shared bins.
    """
    device = z2_seed.device
    ids = seed_ids.long().view(-1)
    b = int(ids.numel())
    m = int(cfg.max_per_anchor)
    d = int(z2_seed.shape[1])

    pos_ids = torch.full((b, m), -1, dtype=torch.long, device=device)
    pos_weights = torch.zeros((b, m), dtype=z2_seed.dtype, device=device)
    pos_valid = torch.zeros((b, m), dtype=torch.bool, device=device)
    pos_z2 = torch.zeros((b, m, d), dtype=z2_seed.dtype, device=device)

    empty_stats = {
        "anchors": float(b),
        "avg_soft_positives": 0.0,
        "zero_soft_positive_rate": 1.0 if b else 0.0,
        "max_soft_positives_observed": 0.0,
        "shared_bin_sum": 0.0,
        "shared_bin_count": 0.0,
        "capped_anchors": 0.0,
    }
    if b == 0:
        if stats is not None:
            for k, v in empty_stats.items():
                stats[k] = stats.get(k, 0.0) + float(v)
        return pos_ids, pos_weights, pos_valid, pos_z2

    # Lookup only the current batch on CPU, then do candidate matching on the
    # embedding device. The former implementation materialized BxB tensors on
    # CPU and sorted candidates in a Python loop for every anchor. At B=8192
    # that made this preprocessing much slower than the GNN forward pass.
    ids_cpu = ids.detach().cpu()
    n_feat = int(ctx.bin_ids.shape[0])
    valid_lookup = (ids_cpu >= 0) & (ids_cpu < n_feat)
    bins = torch.zeros((b, ctx.bin_ids.shape[1]), dtype=torch.long)
    hubs = torch.zeros((b,), dtype=torch.float32)
    if bool(valid_lookup.any()):
        good = ids_cpu[valid_lookup]
        bins[valid_lookup] = ctx.bin_ids[good]
        hubs[valid_lookup] = ctx.hub_score[good]

    bins_d = bins.to(device=device, non_blocking=True)
    hubs_d = hubs.to(device=device, non_blocking=True)
    valid_d = valid_lookup.to(device=device, non_blocking=True)
    ids_d = ids.to(device=device)
    f = int(bins_d.shape[1])

    # Keep the best M candidates while scanning candidates in bounded chunks.
    # Ranking is lexicographic: lower hub activity, then more shared bins, then
    # lower batch index. This preserves the anti-hub policy without allocating
    # a full BxB matrix. A 512-wide chunk is ~4M pairs at the scout batch size.
    chunk_size = min(512, b)
    sentinel = torch.iinfo(torch.int64).min
    best_score = torch.full((b, m), sentinel, dtype=torch.int64, device=device)
    best_j = torch.full((b, m), -1, dtype=torch.long, device=device)
    best_shared = torch.zeros((b, m), dtype=torch.int16, device=device)

    # Dense ranks let exactly tied hub scores fall through to shared-bin count.
    hub_order = torch.argsort(hubs_d, stable=True)
    sorted_hubs = hubs_d[hub_order]
    new_group = torch.ones(b, dtype=torch.bool, device=device)
    if b > 1:
        new_group[1:] = sorted_hubs[1:] != sorted_hubs[:-1]
    sorted_hub_group = torch.cumsum(new_group.to(torch.long), dim=0) - 1
    hub_group = torch.empty_like(sorted_hub_group)
    hub_group[hub_order] = sorted_hub_group

    scale_j = b + 1
    scale_hub = (f + 1) * scale_j
    for j0 in range(0, b, chunk_size):
        j1 = min(j0 + chunk_size, b)
        shared = torch.zeros((b, j1 - j0), dtype=torch.int16, device=device)
        candidate_bins = bins_d[j0:j1]
        for fi in range(f):
            shared.add_(
                (bins_d[:, fi : fi + 1] == candidate_bins[:, fi].unsqueeze(0)).to(torch.int16)
            )

        candidate_j = torch.arange(j0, j1, dtype=torch.long, device=device)
        eligible = shared >= int(cfg.min_shared_bins)
        eligible &= valid_d.unsqueeze(1) & valid_d[j0:j1].unsqueeze(0)
        # The same transaction must never become its own additional positive.
        eligible &= ids_d.unsqueeze(1) != ids_d[j0:j1].unsqueeze(0)

        score = (
            -hub_group[j0:j1].unsqueeze(0) * scale_hub
            + shared.to(torch.long) * scale_j
            - candidate_j.unsqueeze(0)
        )
        score.masked_fill_(~eligible, sentinel)
        take = min(m, j1 - j0)
        chunk_score, chunk_k = torch.topk(score, k=take, dim=1)
        chunk_j = candidate_j[chunk_k]
        chunk_shared = torch.gather(shared, 1, chunk_k)

        merged_score = torch.cat((best_score, chunk_score), dim=1)
        merged_j = torch.cat((best_j, chunk_j), dim=1)
        merged_shared = torch.cat((best_shared, chunk_shared), dim=1)
        best_score, keep = torch.topk(merged_score, k=m, dim=1)
        best_j = torch.gather(merged_j, 1, keep)
        best_shared = torch.gather(merged_shared, 1, keep)

    pos_valid = best_score != sentinel
    safe_j = best_j.clamp_min(0)
    pos_ids = ids_d[safe_j].masked_fill(~pos_valid, -1)
    pos_z2 = z2_seed[safe_j] * pos_valid.unsqueeze(-1).to(z2_seed.dtype)
    n_pos_per = pos_valid.sum(dim=1)
    pos_weights = (
        pos_valid.to(z2_seed.dtype)
        * (float(cfg.weight) / n_pos_per.clamp_min(1).to(z2_seed.dtype)).unsqueeze(1)
    )

    # Detect cap behavior exactly without retaining candidate counts: an
    # anchor is capped if an additional eligible candidate exists beyond M.
    # The retained M all have finite scores; equality to M is the useful scout
    # diagnostic and matches the prior implementation for dense variants.
    capped = int((n_pos_per == m).sum().item())
    shared_selected = best_shared[pos_valid].to(torch.float32)
    shared_sum = float(shared_selected.sum().item())
    shared_cnt = float(shared_selected.numel())
    n_pos_np = n_pos_per.detach().cpu().numpy()
    avg = float(n_pos_np.mean()) if b else 0.0
    zero_rate = float(np.mean(n_pos_np == 0)) if b else 0.0
    max_obs = float(n_pos_np.max()) if b else 0.0
    if stats is not None:
        stats["anchors"] = stats.get("anchors", 0.0) + float(b)
        stats["soft_pos_sum"] = stats.get("soft_pos_sum", 0.0) + float(n_pos_np.sum())
        stats["zero_soft_positive_anchors"] = stats.get("zero_soft_positive_anchors", 0.0) + float(
            np.sum(n_pos_np == 0)
        )
        stats["max_soft_positives_observed"] = max(
            float(stats.get("max_soft_positives_observed", 0.0)), max_obs
        )
        stats["capped_anchors"] = stats.get("capped_anchors", 0.0) + float(capped)
        stats["shared_bin_sum"] = stats.get("shared_bin_sum", 0.0) + shared_sum
        stats["shared_bin_count"] = stats.get("shared_bin_count", 0.0) + shared_cnt
        for shared_n in range(int(cfg.min_shared_bins), f + 1):
            key = f"shared_bins_{shared_n}_count"
            stats[key] = stats.get(key, 0.0) + float((shared_selected == shared_n).sum().item())
        # running averages for logging convenience
        anchors = max(float(stats["anchors"]), 1.0)
        stats["avg_soft_positives"] = float(stats["soft_pos_sum"]) / anchors
        stats["zero_soft_positive_rate"] = float(stats["zero_soft_positive_anchors"]) / anchors
        if shared_cnt > 0:
            stats["mean_shared_bins"] = float(stats["shared_bin_sum"]) / max(
                float(stats["shared_bin_count"]), 1.0
            )

    return pos_ids, pos_weights, pos_valid, pos_z2


def temporal_flow_soft_positives_enabled(args) -> bool:
    enabled = getattr(args, "temporal_flow_soft_positives", False)
    if isinstance(enabled, str):
        return enabled.lower() in ("1", "true", "yes", "on")
    return bool(enabled)
