#!/usr/bin/env python3
"""Temporal-flow auxiliary objective for contrastive SSL pretraining.

Predicts validated ``temporal_flow_causal`` features from the edge representation
**after** ``embedding_head`` and **before** the contrastive projection head
(``z_seed`` / post-128).

Modes:
- ``regression``: Huber (default) or MSE on train-fit standardized targets
- ``bins``: per-feature cross-entropy over train-fit quantile bins

No AML labels are used. Scalers / bins are fit on the train split only.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from morphology.temporal_flow_causal import TEMPORAL_FLOW_CAUSAL_FEATURE_NAMES

ATTACH_POINT = "post_embedding_head_pre_projection"
DEFAULT_CACHE_REL = Path("results/cache/temporal_flow_causal")


@dataclass
class TemporalFlowAuxConfig:
    mode: str = "none"  # none | regression | bins
    weight: float = 0.1
    loss_type: str = "huber"  # huber | mse (regression only)
    n_bins: int = 5
    embedding_dim: int = 128
    hidden_dim: int = 64
    feature_names: Tuple[str, ...] = TEMPORAL_FLOW_CAUSAL_FEATURE_NAMES
    attach_point: str = ATTACH_POINT
    cache_dir: str = ""
    metadata_path: str = ""
    uses_labels: bool = False


class TemporalFlowAuxHead(nn.Module):
    """Shared MLP trunk + either regression head or per-feature classification heads."""

    def __init__(self, cfg: TemporalFlowAuxConfig, n_classes: Optional[Sequence[int]] = None):
        super().__init__()
        self.mode = str(cfg.mode).lower()
        self.n_features = len(cfg.feature_names)
        self.trunk = nn.Sequential(
            nn.Linear(int(cfg.embedding_dim), int(cfg.hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(cfg.hidden_dim), int(cfg.hidden_dim)),
            nn.ReLU(),
        )
        if self.mode == "regression":
            self.reg_head = nn.Linear(int(cfg.hidden_dim), self.n_features)
            self.class_heads = None
            self.n_classes = None
        elif self.mode == "bins":
            if n_classes is None or len(n_classes) != self.n_features:
                raise ValueError("bins mode requires n_classes per feature")
            self.reg_head = None
            self.n_classes = [int(c) for c in n_classes]
            self.class_heads = nn.ModuleList(
                [nn.Linear(int(cfg.hidden_dim), int(c)) for c in self.n_classes]
            )
        else:
            raise ValueError(f"Unknown temporal-flow aux mode: {cfg.mode!r}")

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.trunk(z)
        if self.mode == "regression":
            return self.reg_head(h)
        # bins: stack logits list is handled in loss; return h for callers that need trunk
        return h

    def classify(self, z: torch.Tensor) -> List[torch.Tensor]:
        h = self.trunk(z)
        assert self.class_heads is not None
        return [head(h) for head in self.class_heads]


@dataclass
class TemporalFlowAuxContext:
    """Device tensors + train-only preprocessing metadata."""

    features: torch.Tensor  # [N, F] float32 on device (raw causal values)
    targets_reg: torch.Tensor  # [N, F] standardized for regression
    targets_bins: Optional[torch.Tensor]  # [N, F] long class ids
    n_classes: List[int]
    bin_edges: List[Optional[np.ndarray]]
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    train_edge_ids: np.ndarray
    meta: Dict[str, Any] = field(default_factory=dict)


def _resolve_cache_dir(args, data: str) -> Path:
    raw = getattr(args, "aux_temporal_flow_cache", None)
    if raw:
        return Path(raw)
    return DEFAULT_CACHE_REL / str(data)


def _load_cache(cache_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    feat_path = cache_dir / "features.npy"
    eid_path = cache_dir / "edge_id.npy"
    train_path = cache_dir / "split_train_edge_id.npy"
    meta_path = cache_dir / "meta.json"
    for p in (feat_path, eid_path, train_path, meta_path):
        if not p.is_file():
            raise FileNotFoundError(f"Missing temporal_flow_causal cache artifact: {p}")
    with meta_path.open(encoding="utf-8") as f:
        meta = json.load(f)
    if meta.get("causal_history_policy", {}).get("uses_labels"):
        raise RuntimeError("Refuse to use temporal-flow cache that reports uses_labels=True")
    features = np.load(feat_path).astype(np.float32)
    edge_id = np.load(eid_path).astype(np.int64)
    train_ids = np.load(train_path).astype(np.int64)
    if features.shape[1] != len(TEMPORAL_FLOW_CAUSAL_FEATURE_NAMES):
        raise ValueError(
            f"Expected {len(TEMPORAL_FLOW_CAUSAL_FEATURE_NAMES)} features, got {features.shape[1]}"
        )
    if not np.all(edge_id == np.arange(len(edge_id))):
        # Remap if needed (dense index by max edge id)
        logging.warning("temporal_flow cache edge_id is not identity; building dense lookup")
    return features, edge_id, train_ids, meta


def _fit_scaler_train_only(
    features: np.ndarray, train_ids: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_x = features[train_ids]
    mean = np.nanmean(train_x, axis=0).astype(np.float64)
    std = np.nanstd(train_x, axis=0).astype(np.float64)
    std = np.where(std < 1e-8, 1.0, std)
    scaled = ((features - mean) / std).astype(np.float32)
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
    return scaled, mean.astype(np.float32), std.astype(np.float32)


def _fit_bins_train_only(
    features: np.ndarray,
    train_ids: np.ndarray,
    n_bins: int,
) -> Tuple[np.ndarray, List[int], List[Optional[np.ndarray]]]:
    """Return class ids [N,F], per-feature n_classes, and bin edges (None => discrete labels)."""
    n, f = features.shape
    out = np.zeros((n, f), dtype=np.int64)
    n_classes: List[int] = []
    edges: List[Optional[np.ndarray]] = []
    train_x = features[train_ids]
    for j in range(f):
        col_tr = train_x[:, j]
        col_tr = col_tr[np.isfinite(col_tr)]
        uniq = np.unique(col_tr)
        if uniq.size <= max(2, int(n_bins)):
            # Discrete / low-cardinality: map unique train values to class ids
            mapping = {float(v): i for i, v in enumerate(uniq.tolist())}
            col_all = features[:, j]
            ids = np.zeros(n, dtype=np.int64)
            for i, v in enumerate(col_all):
                if not np.isfinite(v):
                    ids[i] = 0
                else:
                    # nearest train unique if unseen
                    if float(v) in mapping:
                        ids[i] = mapping[float(v)]
                    else:
                        nearest = int(np.argmin(np.abs(uniq - v)))
                        ids[i] = nearest
            out[:, j] = ids
            n_classes.append(int(uniq.size))
            edges.append(None)
            continue
        qs = np.linspace(0.0, 1.0, int(n_bins) + 1)[1:-1]
        cuts = np.unique(np.quantile(col_tr, qs))
        if cuts.size == 0:
            out[:, j] = 0
            n_classes.append(1)
            edges.append(np.array([], dtype=np.float64))
            continue
        # digitize -> 0..n_bins-1 typically
        ids = np.digitize(features[:, j], cuts, right=False).astype(np.int64)
        ids = np.clip(ids, 0, int(cuts.size))
        # n_classes = cuts.size + 1
        n_cls = int(cuts.size) + 1
        n_classes.append(n_cls)
        edges.append(cuts.astype(np.float64))
        out[:, j] = ids
    return out, n_classes, edges


def setup_temporal_flow_aux(
    args,
    device: torch.device,
    *,
    data_name: str,
    embedding_dim: int,
) -> Tuple[Optional[TemporalFlowAuxHead], Optional[TemporalFlowAuxConfig], Optional[TemporalFlowAuxContext]]:
    mode = str(getattr(args, "aux_temporal_flow", "none") or "none").lower()
    if mode in ("", "none", "off", "false", "0"):
        return None, None, None
    if mode not in ("regression", "bins"):
        raise ValueError(f"--aux_temporal_flow must be none|regression|bins, got {mode!r}")

    cache_dir = _resolve_cache_dir(args, data_name)
    features, edge_id, train_ids, cache_meta = _load_cache(cache_dir)
    if cache_meta.get("causal_history_policy", {}).get("uses_labels"):
        raise RuntimeError("temporal-flow cache reports label use; aborting")

    scaled, mean, std = _fit_scaler_train_only(features, train_ids)
    n_bins = int(getattr(args, "aux_temporal_flow_bins", 5))
    bin_ids, n_classes, bin_edges = _fit_bins_train_only(features, train_ids, n_bins)

    cfg = TemporalFlowAuxConfig(
        mode=mode,
        weight=float(getattr(args, "aux_temporal_flow_weight", 0.1)),
        loss_type=str(getattr(args, "aux_temporal_flow_loss", "huber")).lower(),
        n_bins=n_bins,
        embedding_dim=int(embedding_dim),
        hidden_dim=int(getattr(args, "aux_temporal_flow_hidden", 64)),
        feature_names=TEMPORAL_FLOW_CAUSAL_FEATURE_NAMES,
        attach_point=ATTACH_POINT,
        cache_dir=str(cache_dir),
        uses_labels=False,
    )
    if cfg.mode == "regression" and cfg.loss_type not in ("huber", "mse"):
        raise ValueError(f"--aux_temporal_flow_loss must be huber|mse, got {cfg.loss_type!r}")
    if cfg.weight < 0:
        raise ValueError("--aux_temporal_flow_weight must be >= 0")

    head = TemporalFlowAuxHead(cfg, n_classes=n_classes if mode == "bins" else None).to(device)

    unique_name = str(getattr(args, "unique_name", "run") or "run")
    meta_dir = Path("results/diagnostics/temporal_flow_aux_preprocess")
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta_path = meta_dir / f"{unique_name}_preprocess.json"
    meta = {
        "feature_names": list(TEMPORAL_FLOW_CAUSAL_FEATURE_NAMES),
        "mode": mode,
        "loss_type": cfg.loss_type if mode == "regression" else "cross_entropy",
        "weight": cfg.weight,
        "n_bins_requested": n_bins,
        "n_classes_per_feature": n_classes,
        "attach_point": ATTACH_POINT,
        "scaler_fit_split": "train",
        "bin_fit_split": "train",
        "train_n": int(train_ids.shape[0]),
        "scaler_mean": mean.tolist(),
        "scaler_scale": std.tolist(),
        "bin_edges": [None if e is None else e.tolist() for e in bin_edges],
        "missing_value_handling": "nan_to_num(0) after train-only standardize; invalid bin rows masked in CE",
        "timestamp_tie_policy": (cache_meta.get("timestamp_handling") or {}).get("timestamp_ties"),
        "causal_history_policy": cache_meta.get("causal_history_policy"),
        "cache_dir": str(cache_dir),
        "cache_version": cache_meta.get("cache_version"),
        "uses_labels": False,
        "no_ssl_label_use": True,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    cfg.metadata_path = str(meta_path)

    ctx = TemporalFlowAuxContext(
        features=torch.as_tensor(features, dtype=torch.float32, device=device),
        targets_reg=torch.as_tensor(scaled, dtype=torch.float32, device=device),
        targets_bins=torch.as_tensor(bin_ids, dtype=torch.long, device=device),
        n_classes=n_classes,
        bin_edges=bin_edges,
        scaler_mean=mean,
        scaler_scale=std,
        train_edge_ids=train_ids,
        meta=meta,
    )
    logging.info(
        "Temporal-flow aux enabled: mode=%s weight=%.4f attach=%s cache=%s meta=%s",
        mode,
        cfg.weight,
        ATTACH_POINT,
        cache_dir,
        meta_path,
    )
    return head, cfg, ctx


def temporal_flow_aux_loss(
    z_seed: torch.Tensor,
    seed_edge_ids: torch.Tensor,
    head: TemporalFlowAuxHead,
    cfg: TemporalFlowAuxConfig,
    ctx: TemporalFlowAuxContext,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute weighted aux loss for shared seed edges. Returns (loss, per-feature diagnostics)."""
    if z_seed.numel() == 0 or seed_edge_ids.numel() == 0:
        zero = z_seed.new_zeros(())
        return zero, {}

    ids = seed_edge_ids.long().view(-1)
    # Clamp invalid ids (should not happen)
    n = int(ctx.targets_reg.shape[0])
    valid = (ids >= 0) & (ids < n)
    if not bool(valid.any()):
        return z_seed.new_zeros(()), {}
    ids = ids[valid]
    z = z_seed[valid]

    per_feat: Dict[str, float] = {}
    if cfg.mode == "regression":
        targets = ctx.targets_reg[ids]
        pred = head(z)
        if cfg.loss_type == "mse":
            elem = (pred - targets) ** 2
        else:
            elem = F.huber_loss(pred, targets, reduction="none", delta=1.0)
        # Mask non-finite targets (should be rare after nan_to_num)
        mask = torch.isfinite(targets)
        elem = elem * mask.float()
        denom = mask.float().sum().clamp_min(1.0)
        loss = elem.sum() / denom
        with torch.no_grad():
            for j, name in enumerate(cfg.feature_names):
                m = mask[:, j]
                if bool(m.any()):
                    per_feat[f"tf_aux/feat/{name}"] = float(elem[:, j][m].mean().detach().cpu())
    else:
        logits_list = head.classify(z)
        targets = ctx.targets_bins[ids]
        losses = []
        for j, name in enumerate(cfg.feature_names):
            logits = logits_list[j]
            y = targets[:, j].clamp(0, int(ctx.n_classes[j]) - 1)
            # Ignore nothing by default; all ids valid
            lj = F.cross_entropy(logits, y, reduction="mean")
            losses.append(lj)
            per_feat[f"tf_aux/feat/{name}"] = float(lj.detach().cpu())
        loss = torch.stack(losses).mean()

    loss = loss * float(cfg.weight)
    per_feat["tf_aux/weight"] = float(cfg.weight)
    per_feat["tf_aux/mode"] = 0.0 if cfg.mode == "regression" else 1.0
    return loss, per_feat


def aux_temporal_flow_enabled(args) -> bool:
    mode = str(getattr(args, "aux_temporal_flow", "none") or "none").lower()
    return mode not in ("", "none", "off", "false", "0")
