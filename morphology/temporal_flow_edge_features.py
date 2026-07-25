"""Append label-free temporal_flow_causal features to GNN edge_attr (encoder input).

Gated by ``--include_temporal_flow_edge_features`` (default false).
Uses the existing causal cache; train-only z-scaling; no AML labels.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch_geometric.data import HeteroData

from morphology.temporal_flow_causal import TEMPORAL_FLOW_CAUSAL_FEATURE_NAMES

DEFAULT_CACHE_REL = Path("results/cache/temporal_flow_causal")
FORWARD_EDGE_TYPE = ("node", "to", "node")
REVERSE_EDGE_TYPE = ("node", "rev_to", "node")


@dataclass(frozen=True)
class TemporalFlowEdgeFeaturesMeta:
    enabled: bool
    cache_dir: str
    feature_names: Tuple[str, ...]
    n_features: int
    edge_dim_before: int
    edge_dim_after: int
    scaler_fit_on: str
    scaler_mean: Tuple[float, ...]
    scaler_scale: Tuple[float, ...]
    uses_labels: bool
    past_only: bool
    n_rows: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "cache_dir": self.cache_dir,
            "feature_names": list(self.feature_names),
            "n_features": self.n_features,
            "edge_dim_before": self.edge_dim_before,
            "edge_dim_after": self.edge_dim_after,
            "scaler_fit_on": self.scaler_fit_on,
            "scaler_mean": list(self.scaler_mean),
            "scaler_scale": list(self.scaler_scale),
            "uses_labels": self.uses_labels,
            "past_only": self.past_only,
            "n_rows": self.n_rows,
        }


def resolve_temporal_flow_edge_features_cache(args, data: str) -> Path:
    raw = getattr(args, "temporal_flow_edge_features_cache", None) or getattr(
        args, "aux_temporal_flow_cache", None
    )
    if raw:
        return Path(str(raw))
    return DEFAULT_CACHE_REL / str(data)


def load_temporal_flow_edge_feature_cache(
    cache_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load features / edge_id / train ids / meta; refuse labeled caches."""
    feat_path = cache_dir / "features.npy"
    eid_path = cache_dir / "edge_id.npy"
    train_path = cache_dir / "split_train_edge_id.npy"
    meta_path = cache_dir / "meta.json"
    missing = [p for p in (feat_path, eid_path, train_path, meta_path) if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing temporal_flow_causal cache for encoder edge features. "
            f"Expected under {cache_dir}. Missing: "
            + ", ".join(str(p.name) for p in missing)
            + ". Build with scripts/build_temporal_flow_causal_cache.py."
        )
    with meta_path.open(encoding="utf-8") as f:
        meta = json.load(f)
    policy = meta.get("causal_history_policy") or {}
    if policy.get("uses_labels") is True:
        raise RuntimeError(
            f"Refuse temporal_flow cache at {cache_dir}: causal_history_policy.uses_labels=true. "
            "Encoder-input TF features must be label-free."
        )
    if policy.get("past_only") is not True:
        raise RuntimeError(
            f"Refuse temporal_flow cache at {cache_dir}: causal_history_policy.past_only must be true "
            f"(got {policy.get('past_only')!r})."
        )
    features = np.load(feat_path).astype(np.float32)
    edge_id = np.load(eid_path).astype(np.int64)
    train_ids = np.load(train_path).astype(np.int64)
    if features.ndim != 2 or features.shape[1] != len(TEMPORAL_FLOW_CAUSAL_FEATURE_NAMES):
        raise ValueError(
            f"Expected features shape [N, {len(TEMPORAL_FLOW_CAUSAL_FEATURE_NAMES)}], "
            f"got {features.shape} in {feat_path}"
        )
    if features.shape[0] != edge_id.shape[0]:
        raise ValueError(
            f"features/edge_id length mismatch in {cache_dir}: "
            f"{features.shape[0]} vs {edge_id.shape[0]}"
        )
    return features, edge_id, train_ids, meta


def fit_train_only_scaler(
    features: np.ndarray, train_ids: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score all rows using mean/std fit on train_ids only."""
    train_x = features[train_ids]
    mean = np.nanmean(train_x, axis=0).astype(np.float64)
    std = np.nanstd(train_x, axis=0).astype(np.float64)
    std = np.where(std < 1e-8, 1.0, std)
    scaled = ((features - mean) / std).astype(np.float32)
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
    return scaled, mean.astype(np.float32), std.astype(np.float32)


def _edge_attr_tensor(data) -> torch.Tensor:
    if isinstance(data, HeteroData):
        return data[FORWARD_EDGE_TYPE].edge_attr
    return data.edge_attr


def _append_cols(edge_attr: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
    if cols.device != edge_attr.device:
        cols = cols.to(edge_attr.device)
    if cols.dtype != edge_attr.dtype:
        cols = cols.to(dtype=edge_attr.dtype)
    if cols.shape[0] != edge_attr.shape[0]:
        raise ValueError(
            f"TF append row mismatch: edge_attr has {edge_attr.shape[0]} rows, "
            f"tf has {cols.shape[0]}"
        )
    return torch.cat([edge_attr, cols], dim=1)


def append_scaled_features_to_graph(data, scaled_rows: np.ndarray) -> None:
    """Append [E, F] features to homo edge_attr or both hetero forward/reverse stores."""
    cols = torch.from_numpy(np.ascontiguousarray(scaled_rows))
    if isinstance(data, HeteroData):
        fwd = data[FORWARD_EDGE_TYPE]
        rev = data[REVERSE_EDGE_TYPE]
        fwd.edge_attr = _append_cols(fwd.edge_attr, cols)
        # Same causal TF on reverse (do not port-swap TF columns).
        rev.edge_attr = _append_cols(rev.edge_attr, cols)
    else:
        data.edge_attr = _append_cols(data.edge_attr, cols)


def maybe_append_temporal_flow_edge_features(
    tr_data,
    val_data,
    te_data,
    *,
    e_tr: np.ndarray,
    e_val: np.ndarray,
    args,
    data_name: str,
    n_edges_full: int,
) -> Optional[TemporalFlowEdgeFeaturesMeta]:
    """
    If ``args.include_temporal_flow_edge_features``, append train-scaled TF cols
    after base z_norm / hetero construction. Mutates the three graphs in place.
    """
    if not bool(getattr(args, "include_temporal_flow_edge_features", False)):
        return None

    cache_dir = resolve_temporal_flow_edge_features_cache(args, data_name)
    features, edge_id, train_ids, meta = load_temporal_flow_edge_feature_cache(cache_dir)
    if int(features.shape[0]) != int(n_edges_full):
        raise ValueError(
            f"temporal_flow cache n_rows={features.shape[0]} != graph n_edges={n_edges_full} "
            f"({cache_dir})"
        )
    if not np.array_equal(edge_id, np.arange(len(edge_id), dtype=np.int64)):
        dense = np.zeros_like(features)
        if edge_id.min() < 0 or int(edge_id.max()) >= len(edge_id):
            raise ValueError(f"Invalid edge_id range in {cache_dir}")
        dense[edge_id] = features
        features = dense

    train_ids = np.asarray(train_ids, dtype=np.int64)
    e_tr = np.asarray(e_tr, dtype=np.int64)
    e_val = np.asarray(e_val, dtype=np.int64)
    if set(train_ids.tolist()) != set(e_tr.tolist()):
        logging.warning(
            "temporal_flow train split ids differ from get_data e_tr "
            "(cache=%d, e_tr=%d); using cache train ids for scaler fit.",
            len(train_ids),
            len(e_tr),
        )

    scaled, mean, scale = fit_train_only_scaler(features, train_ids)
    edge_dim_before = int(_edge_attr_tensor(tr_data).shape[1])

    append_scaled_features_to_graph(tr_data, scaled[e_tr])
    append_scaled_features_to_graph(val_data, scaled[e_val])
    append_scaled_features_to_graph(te_data, scaled[np.arange(n_edges_full, dtype=np.int64)])

    edge_dim_after = int(_edge_attr_tensor(tr_data).shape[1])
    n_feat = len(TEMPORAL_FLOW_CAUSAL_FEATURE_NAMES)
    if edge_dim_after != edge_dim_before + n_feat:
        raise RuntimeError(
            f"Expected edge_dim {edge_dim_before}+{n_feat}={edge_dim_before + n_feat}, "
            f"got {edge_dim_after}"
        )

    policy = meta.get("causal_history_policy") or {}
    out = TemporalFlowEdgeFeaturesMeta(
        enabled=True,
        cache_dir=str(cache_dir),
        feature_names=tuple(TEMPORAL_FLOW_CAUSAL_FEATURE_NAMES),
        n_features=n_feat,
        edge_dim_before=edge_dim_before,
        edge_dim_after=edge_dim_after,
        scaler_fit_on="temporal_flow_cache split_train_edge_id (train-only)",
        scaler_mean=tuple(float(x) for x in mean.tolist()),
        scaler_scale=tuple(float(x) for x in scale.tolist()),
        uses_labels=False,
        past_only=bool(policy.get("past_only")),
        n_rows=int(features.shape[0]),
    )
    logging.info(
        "Appended temporal_flow_causal encoder edge features: edge_dim %d -> %d "
        "(cache=%s, uses_labels=false, past_only=true, train-only scaler)",
        edge_dim_before,
        edge_dim_after,
        cache_dir,
    )
    for data in (tr_data, val_data, te_data):
        data.temporal_flow_edge_features_meta = out.to_dict()
    return out


def assert_checkpoint_tf_edge_features_flag(
    checkpoint: Dict[str, Any],
    args,
    *,
    path: Union[str, Path],
) -> None:
    """Fail loudly if extraction/train flag disagrees with checkpoint metadata."""
    ckpt_flag = bool(checkpoint.get("include_temporal_flow_edge_features", False))
    req_flag = bool(getattr(args, "include_temporal_flow_edge_features", False))
    path = Path(path)
    if ckpt_flag == req_flag:
        return
    if ckpt_flag and not req_flag:
        raise ValueError(
            f"include_temporal_flow_edge_features mismatch: checkpoint {path.name} was trained "
            "WITH temporal-flow encoder edge features, but the current run does not pass "
            "--include_temporal_flow_edge_features. Re-run extraction/training with that flag "
            "so edge_dim matches the checkpoint."
        )
    raise ValueError(
        f"include_temporal_flow_edge_features mismatch: checkpoint {path.name} was trained "
        "WITHOUT temporal-flow encoder edge features, but the current run passes "
        "--include_temporal_flow_edge_features. Omit the flag to match this checkpoint."
    )
