#!/usr/bin/env python3
"""Build dataset-level temporal_flow_causal feature cache (CPU, no training).

Computes the five strictly causal features once per dataset and writes an atomic
cache. AMLWorld datasets use ``results/cache/temporal_flow_causal/{dataset}/``
with version ``temporal_flow_causal_v1``. PaySim uses ``temporal_flow_cache/PaySim/``
with version ``temporal_flow_causal_paysim_v1`` (does not overwrite AML caches).
SAML-D shared-core Phase-1 uses
``results/cache/temporal_flow_causal_samld_shared_core_v1/SAML-D/`` with version
``temporal_flow_causal_samld_shared_core_v1`` (train/val only; no test cache).

Cross-split history policy: features are computed in global timestamp order.
Validation rows reflect all prior training transactions; when test rows are
retained, they reflect training + validation history. Labels are not used for
features. SAML-D Phase-1 retains train∪val rows only.

Timestamp ties (canonical AML policy B): equal timestamps are featurized as a batch
using history strictly before that timestamp; state updates after the batch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataset_specs import get_dataset_spec
from dataset_splits import temporal_edge_split
from morphology.temporal_flow_causal import (
    TEMPORAL_FLOW_CAUSAL_DEFINITIONS,
    TEMPORAL_FLOW_CAUSAL_FEATURE_NAMES,
    TEMPORAL_FLOW_CAUSAL_WINDOW_7D_SEC,
    compute_temporal_flow_causal_features,
    feature_summary_stats,
)
from transaction_knn.features import load_data_config, resolve_amount_column
from util import logger_setup

CACHE_VERSION_AML = "temporal_flow_causal_v1"
CACHE_VERSION_PAYSIM = "temporal_flow_causal_paysim_v1"
CACHE_VERSION_SAMLD_SHARED_CORE = "temporal_flow_causal_samld_shared_core_v1"
DEFAULT_CACHE_ROOT_AML = "results/cache/temporal_flow_causal"
DEFAULT_CACHE_ROOT_PAYSIM = "temporal_flow_cache"
DEFAULT_CACHE_ROOT_SAMLD_SHARED_CORE = (
    "results/cache/temporal_flow_causal_samld_shared_core_v1"
)
SUPPORTED_DATASETS = ("Small-HI", "Small-LI", "PaySim", "SAML-D")

# MoE targets (subset of the five causal features); standardization is train-only.
MOE_TARGET_NAMES = (
    "log1p_sender_interarrival",
    "log1p_sender_past_7d_count",
    "log1p_amount_vs_sender_past_mean",
)

TIE_POLICY_ID = "B_simultaneous_batch_strictly_earlier_timestamps"
TIE_POLICY_DESCRIPTION = (
    "Transactions with equal Timestamp are featurized as a batch using history "
    "strictly before that timestamp; account/pair state is updated only after the "
    "entire tie batch is processed (same-timestamp edges do not influence one another). "
    "Does not use CSV row order for causality within a tie batch."
)


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _code_hashes() -> Dict[str, str]:
    paths = [
        _ROOT / "morphology" / "temporal_flow_causal.py",
        _ROOT / "scripts" / "build_temporal_flow_causal_cache.py",
        _ROOT / "dataset_specs.py",
        _ROOT / "dataset_splits.py",
    ]
    out: Dict[str, str] = {}
    for p in paths:
        if p.is_file():
            out[str(p.relative_to(_ROOT))] = _sha256_file(p)
    return out


def _atomic_write_bytes(final_path: Path, data: bytes) -> None:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(final_path.parent), prefix=f".{final_path.name}.", suffix=".tmp")
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        tmp_path.write_bytes(data)
        os.replace(tmp_path, final_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _atomic_write_json(final_path: Path, payload: Dict[str, Any]) -> None:
    _atomic_write_bytes(final_path, json.dumps(payload, indent=2).encode("utf-8"))


def _atomic_save_npy(final_path: Path, arr: np.ndarray) -> None:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_base = tempfile.mkstemp(
        dir=str(final_path.parent),
        prefix=f".{final_path.stem}.",
        suffix=".tmp",
    )
    os.close(fd)
    os.unlink(tmp_base)
    tmp_npy = Path(f"{tmp_base}.npy")
    try:
        np.save(tmp_base, arr)
        os.replace(tmp_npy, final_path)
    finally:
        if tmp_npy.exists() and tmp_npy.resolve() != final_path.resolve():
            tmp_npy.unlink(missing_ok=True)


def cache_version_for(data: str) -> str:
    if data == "PaySim":
        return CACHE_VERSION_PAYSIM
    if data == "SAML-D":
        return CACHE_VERSION_SAMLD_SHARED_CORE
    return CACHE_VERSION_AML


def default_cache_root_for(data: str) -> str:
    if data == "PaySim":
        return DEFAULT_CACHE_ROOT_PAYSIM
    if data == "SAML-D":
        return DEFAULT_CACHE_ROOT_SAMLD_SHARED_CORE
    return DEFAULT_CACHE_ROOT_AML


def _moe_train_scaler(
    features: np.ndarray,
    train_edge_ids: np.ndarray,
    edge_id: np.ndarray,
    feature_names: Sequence[str],
) -> Dict[str, Any]:
    """Train-only mean/std for the three MoE targets (dataset-specific)."""
    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    cols_idx = [name_to_idx[n] for n in MOE_TARGET_NAMES]
    # Map train EdgeIDs → rows in features/edge_id arrays
    id_to_row = {int(e): i for i, e in enumerate(edge_id.tolist())}
    try:
        rows = np.array([id_to_row[int(e)] for e in train_edge_ids.tolist()], dtype=np.int64)
    except KeyError as exc:
        raise RuntimeError(f"Train EdgeID missing from cache rows: {exc}") from exc
    tr = features[rows][:, cols_idx].astype(np.float64)
    tr = np.nan_to_num(tr, nan=0.0, posinf=0.0, neginf=0.0)
    mean = tr.mean(axis=0)
    scale = tr.std(axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    payload = np.concatenate([mean, scale]).tobytes()
    return {
        "target_names": list(MOE_TARGET_NAMES),
        "target_indices_in_features": cols_idx,
        "train_only": True,
        "n_train": int(rows.shape[0]),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "scaler_sha256": hashlib.sha256(payload).hexdigest(),
        "reuses_small_hi_statistics": False,
    }


def timestamp_multiplicity_stats(timestamps: np.ndarray) -> Dict[str, Any]:
    ts = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    n = int(ts.shape[0])
    if n == 0:
        return {
            "n_rows": 0,
            "n_distinct_timestamps": 0,
            "n_timestamps_with_multiple_edges": 0,
            "fraction_edges_sharing_timestamp": float("nan"),
            "median_edges_per_timestamp": float("nan"),
            "p95_edges_per_timestamp": float("nan"),
            "max_edges_per_timestamp": 0,
        }
    _, counts = np.unique(ts, return_counts=True)
    multi = counts[counts > 1]
    edges_in_multi = int(multi.sum()) if multi.size else 0
    return {
        "n_rows": n,
        "n_distinct_timestamps": int(counts.shape[0]),
        "n_timestamps_with_multiple_edges": int(multi.shape[0]),
        "fraction_edges_sharing_timestamp": float(edges_in_multi / n),
        "median_edges_per_timestamp": float(np.median(counts)),
        "p95_edges_per_timestamp": float(np.percentile(counts, 95)),
        "max_edges_per_timestamp": int(counts.max()),
    }


def chronological_edge_fraction_split(
    n_rows: int,
    fractions: Sequence[float] = (0.6, 0.2, 0.2),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Smoke-only split: chronological row thirds by edge count.

    Used when ``max_timestamps`` subsets are too small for ``temporal_edge_split``
    (which can assign an empty train partition on few hourly steps). Production
    full-dataset caches must continue to use ``temporal_edge_split``.
    """
    if n_rows < 3:
        raise ValueError(f"chronological smoke split needs >=3 rows, got {n_rows}")
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError(f"fractions must sum to 1.0, got {fractions}")
    n_tr = max(1, int(round(float(fractions[0]) * n_rows)))
    n_va = max(1, int(round(float(fractions[1]) * n_rows)))
    if n_tr + n_va >= n_rows:
        n_tr = max(1, n_rows - 2)
        n_va = 1
    n_te = n_rows - n_tr - n_va
    if n_te < 1:
        raise ValueError(f"chronological smoke split produced empty test (n={n_rows})")
    tr = np.arange(0, n_tr, dtype=np.int64)
    va = np.arange(n_tr, n_tr + n_va, dtype=np.int64)
    te = np.arange(n_tr + n_va, n_rows, dtype=np.int64)
    meta = {
        "policy": "chronological_edge_fraction_smoke_only",
        "fractions": list(fractions),
        "counts": {"train": int(n_tr), "val": int(n_va), "test": int(n_te)},
        "not_production_temporal_split": True,
    }
    return tr, va, te, meta


def resolve_amount_for_dataset(df: pd.DataFrame, data: str) -> str:
    """Prefer Amount Received for shared AML/SAML-D and PaySim formatted schemas."""
    if data in ("SAML-D", "Small-HI", "Small-LI") and "Amount Received" in df.columns:
        return "Amount Received"
    col = resolve_amount_column(df)
    if data == "PaySim" and "Amount Received" in df.columns and col != "Amount Received":
        # Prefer the canonical AML schema channel when present.
        if "amount" in df.columns and col == "amount":
            return "Amount Received"
    return col


def build_cache(
    data: str,
    data_config_path: str,
    cache_root: Path,
    *,
    overwrite: bool = False,
    max_timestamps: Optional[int] = None,
    write_test_split_files: bool = True,
    train_val_only: Optional[bool] = None,
) -> Path:
    if data not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset {data!r}; choose from {SUPPORTED_DATASETS}")

    samld_shared = data == "SAML-D"
    if train_val_only is None:
        train_val_only = bool(samld_shared)
    if samld_shared:
        # Phase-1 SAML-D shared-core protocol: no test cache materialization.
        write_test_split_files = False
        train_val_only = True

    spec = get_dataset_spec(data)
    cfg = load_data_config(data_config_path)
    csv_path = Path(cfg["paths"]["aml_data"]) / data / spec.formatted_csv_name()
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    out_dir = cache_root / data
    meta_path = out_dir / "meta.json"
    features_path = out_dir / "features.npy"
    edge_id_path = out_dir / "edge_id.npy"
    expected_version = cache_version_for(data)

    # Never allow PaySim writes under the AML cache root accidentally.
    if data == "PaySim" and "temporal_flow_causal" in str(cache_root) and "temporal_flow_cache" not in str(cache_root):
        raise SystemExit(
            f"Refusing to write PaySim TF under AML cache root {cache_root}; "
            f"use --cache_root {DEFAULT_CACHE_ROOT_PAYSIM}"
        )
    if samld_shared and "samld_shared_core" not in str(cache_root):
        raise SystemExit(
            f"Refusing SAML-D TF write under non-unique root {cache_root}; "
            f"use --cache_root {DEFAULT_CACHE_ROOT_SAMLD_SHARED_CORE}"
        )

    if meta_path.is_file() and features_path.is_file():
        prev = json.loads(meta_path.read_text(encoding="utf-8"))
        prev_ver = str(prev.get("cache_version"))
        if prev_ver != expected_version and not overwrite:
            raise RuntimeError(
                f"Refusing nonmatching cache at {out_dir}: "
                f"existing={prev_ver!r} expected={expected_version!r} "
                f"(pass --overwrite only after intentional rebuild)"
            )
        if not overwrite:
            logging.info("Cache already exists at %s (use --overwrite to rebuild)", out_dir)
            return out_dir

    logging.info("Loading %s", csv_path)
    df = pd.read_csv(csv_path)
    required = ["Timestamp", "from_id", "to_id", spec.label_col]
    if samld_shared:
        required.append("Amount Received")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{data} CSV missing columns: {missing}")

    # Stable edge identity: prefer EdgeID column when unique.
    # SAML-D formatter assigns EdgeID then sorts by Timestamp, so EdgeID is unique
    # in [0, n) but not equal to post-sort CSV row index. Retain EdgeID for joins.
    edge_id_equals_row_index = True
    if "EdgeID" in df.columns:
        edge_id_col = df["EdgeID"].to_numpy(dtype=np.int64)
        if int(edge_id_col.shape[0]) != len(df):
            raise ValueError(f"{data}: EdgeID length mismatch")
        if len(set(edge_id_col.tolist())) != int(edge_id_col.shape[0]):
            raise ValueError(f"{data}: duplicate EdgeIDs; refusing cache build")
        edge_id_equals_row_index = bool(
            np.array_equal(edge_id_col, np.arange(len(df), dtype=np.int64))
        )
        if not edge_id_equals_row_index and data not in ("SAML-D",):
            raise ValueError(
                f"{data}: EdgeID is not equal to row index; refusing silent CSV-order join"
            )
        edge_id = edge_id_col.copy()
    else:
        edge_id = np.arange(len(df), dtype=np.int64)

    df = df.copy()
    df["Timestamp"] = df["Timestamp"] - df["Timestamp"].min()
    amount_col = resolve_amount_for_dataset(df, data)
    multiplicity = timestamp_multiplicity_stats(df["Timestamp"].to_numpy())

    # Smoke / subset: keep earliest max_timestamps distinct timestamps (causal prefix).
    if max_timestamps is not None:
        uniq = np.sort(df["Timestamp"].unique())
        if max_timestamps < 1:
            raise ValueError("max_timestamps must be >= 1")
        keep_ts = set(uniq[: int(max_timestamps)].tolist())
        mask = df["Timestamp"].isin(keep_ts).to_numpy()
        df = df.loc[mask].reset_index(drop=True)
        edge_id = edge_id[mask]
        logging.info(
            "Smoke subset: max_timestamps=%d -> %d rows (original edge_ids preserved)",
            max_timestamps,
            len(df),
        )

    y = torch.LongTensor(df[spec.label_col].to_numpy())
    timestamps = torch.Tensor(df["Timestamp"].to_numpy())
    smoke_split_meta: Optional[Dict[str, Any]] = None
    if max_timestamps is not None and not samld_shared:
        # Tiny prefixes can make temporal_edge_split assign an empty train set
        # (observed on PaySim with 8 hourly steps). Smoke only needs non-empty
        # partitions for metadata; TF features themselves are label/split-free.
        tr_np, va_np, te_np, smoke_split_meta = chronological_edge_fraction_split(
            len(df), fractions=tuple(spec.split_fractions)
        )
        tr_inds = torch.from_numpy(tr_np)
        val_inds = torch.from_numpy(va_np)
        te_inds = torch.from_numpy(te_np)
        split_buckets = [[], [], []]
        logging.info(
            "Smoke split (chronological edge fractions, not production temporal split): "
            "train=%d val=%d test=%d",
            int(tr_np.shape[0]),
            int(va_np.shape[0]),
            int(te_np.shape[0]),
        )
    else:
        tr_inds, val_inds, te_inds, split_buckets = temporal_edge_split(timestamps, y, spec)
        if tr_inds.numel() == 0 or val_inds.numel() == 0:
            raise RuntimeError(
                f"temporal_edge_split returned empty train/val for {data}: "
                f"train={int(tr_inds.numel())} val={int(val_inds.numel())}"
            )
        if write_test_split_files and te_inds.numel() == 0:
            raise RuntimeError(
                f"temporal_edge_split returned empty test for {data}"
            )

    if train_val_only:
        # Drop test rows before feature construction (no test cache / inspection).
        keep_pos = np.unique(
            np.concatenate(
                [tr_inds.numpy().astype(np.int64), val_inds.numpy().astype(np.int64)]
            )
        )
        keep_pos.sort()
        df = df.iloc[keep_pos].reset_index(drop=True)
        edge_id = edge_id[keep_pos]
        y = y[keep_pos]
        old_to_new = {int(old): new for new, old in enumerate(keep_pos.tolist())}
        tr_inds = torch.tensor(
            [old_to_new[int(i)] for i in tr_inds.tolist()], dtype=torch.long
        )
        val_inds = torch.tensor(
            [old_to_new[int(i)] for i in val_inds.tolist()], dtype=torch.long
        )
        te_inds = torch.tensor([], dtype=torch.long)
        logging.info(
            "train_val_only: retained %d rows (train=%d val=%d; test excluded)",
            len(df),
            int(tr_inds.numel()),
            int(val_inds.numel()),
        )

    logging.info("Computing temporal_flow_causal features (%d rows)", len(df))
    features, feature_names = compute_temporal_flow_causal_features(df, amount_col=amount_col)
    if feature_names != list(TEMPORAL_FLOW_CAUSAL_FEATURE_NAMES):
        raise ValueError(f"Feature name mismatch: {feature_names}")
    if features.shape[1] != 5:
        raise ValueError(f"Expected 5 TF columns, got {features.shape[1]}")
    if not np.isfinite(features).all():
        raise ValueError("Non-finite TF values after construction")

    split_arrays = {
        "train": tr_inds.numpy().astype(np.int64),
        "val": val_inds.numpy().astype(np.int64),
    }
    if write_test_split_files:
        split_arrays["test"] = te_inds.numpy().astype(np.int64)

    # Map split indices: when subsetting, split returns positions in subset df
    # but edge_id holds original CSV indices. Remap split files to original edge_ids.
    split_edge_ids = {k: edge_id[v] for k, v in split_arrays.items()}
    if set(split_edge_ids["train"].tolist()) & set(split_edge_ids["val"].tolist()):
        raise RuntimeError("train/val EdgeID overlap")
    if len(set(edge_id.tolist())) != int(edge_id.size):
        raise RuntimeError("Duplicate EdgeIDs in cache rows")

    moe_scaler = _moe_train_scaler(
        features, split_edge_ids["train"], edge_id, feature_names
    )
    cache_version = expected_version
    meta: Dict[str, Any] = {
        "cache_version": cache_version,
        "dataset": data,
        "feature_contract_id": (
            "smallhi_samld_shared_core_v1" if samld_shared else None
        ),
        "feature_group": "temporal_flow_causal",
        "feature_names": list(TEMPORAL_FLOW_CAUSAL_FEATURE_NAMES),
        "feature_definitions": TEMPORAL_FLOW_CAUSAL_DEFINITIONS,
        "moe_targets": list(MOE_TARGET_NAMES),
        "moe_target_train_scaler": moe_scaler,
        "n_rows": int(len(df)),
        "n_features": int(features.shape[1]),
        "row_order": (
            "features[i] corresponds to edge_id[i] (EdgeID of CSV row i after load)"
        ),
        "edge_id_policy": (
            "formatted CSV EdgeID column (unique); may differ from post-sort row index "
            "on SAML-D"
            if not edge_id_equals_row_index
            else "CSV row index (== EdgeID when present and contiguous)"
        ),
        "edge_id_equals_row_index": bool(edge_id_equals_row_index),
        "test_split_written": bool(write_test_split_files),
        "train_val_only": bool(train_val_only),
        "split_row_counts": {k: int(v.shape[0]) for k, v in split_edge_ids.items()},
        "split_indices_files": {k: f"split_{k}_edge_id.npy" for k in split_edge_ids},
        "timestamp_handling": {
            "policy": "global_timestamp_sort_mergesort",
            "tie_policy_id": TIE_POLICY_ID,
            "timestamp_ties": TIE_POLICY_DESCRIPTION,
            "timestamp_units": "seconds",
            "timestamp_shift": "Timestamp -= min(Timestamp) over retained CSV rows",
            "does_not_rely_on_csv_row_order_within_ties": True,
            "alternative_rejected": (
                "A_stable_Timestamp_edge_id_order would create arbitrary within-step "
                "ordering on PaySim where nearly all edges share coarse step timestamps"
            ),
        },
        "timestamp_multiplicity": multiplicity,
        "causal_history_policy": {
            "past_only": True,
            "uses_labels": False,
            "val_sees_train_history": True,
            "test_sees_train_and_val_history": bool(write_test_split_files and not train_val_only),
            "window_7d_sec": TEMPORAL_FLOW_CAUSAL_WINDOW_7D_SEC,
            "window_note_paysim": (
                "W=604800s with Timestamp=step*3600 => 168 PaySim steps; valid past window"
                if data == "PaySim"
                else None
            ),
            "no_history_defaults": {
                "log1p_sender_interarrival": 0.0,
                "log1p_receiver_interarrival": 0.0,
                "log1p_sender_past_7d_count": 0.0,
                "log1p_amount_vs_sender_past_mean": 0.0,
                "pair_repeat_indicator": 0.0,
            },
            "normalization_at_cache_time": "none (raw causal values)",
            "scaler_policy": (
                "moe_targets_train_fit_standardization_in_moe_target_train_scaler"
                if samld_shared
                else "deferred_to_probe_StandardScaler_fit_train_only"
            ),
            "implementation": "morphology.temporal_flow_causal.compute_temporal_flow_causal_features",
        },
        "field_mapping": {
            "sender": "from_id",
            "receiver": "to_id",
            "time": "Timestamp",
            "amount": amount_col,
            "label_column_unused_for_features": spec.label_col,
            "join_key": "edge_id",
        },
        "source_data": {
            "csv_path": str(csv_path.resolve()),
            "csv_sha256": _sha256_file(csv_path) if max_timestamps is None else None,
            "amount_column": amount_col,
            "label_column": spec.label_col,
            "smoke_max_timestamps": max_timestamps,
        },
        "split_buckets": split_buckets,
        "smoke_split": smoke_split_meta,
        "feature_summary": feature_summary_stats(features),
        "coverage": {
            "n_feature_rows": int(features.shape[0]),
            "n_train": int(split_edge_ids["train"].shape[0]),
            "n_val": int(split_edge_ids["val"].shape[0]),
            "moe_finite_fraction": float(
                np.isfinite(
                    features[
                        :,
                        [
                            list(TEMPORAL_FLOW_CAUSAL_FEATURE_NAMES).index(n)
                            for n in MOE_TARGET_NAMES
                        ],
                    ]
                ).mean()
            ),
        },
        "prevalence": {
            "n_train": int(tr_inds.numel()),
            "n_val": int(val_inds.numel()),
            "n_pos_train": int(y[tr_inds].sum().item()),
            "n_pos_val": int(y[val_inds].sum().item()),
            "positive_rate_train": float(y[tr_inds].float().mean().item()),
            "positive_rate_val": float(y[val_inds].float().mean().item()),
            "positive_rate_test": (
                float(y[te_inds].float().mean().item())
                if write_test_split_files and te_inds.numel() > 0
                else None
            ),
            "note": "Label counts from split metadata only; unused in TF construction",
        },
        "labels_used_in_feature_construction": False,
        "reuses_small_hi_target_statistics": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_metadata": {
            "script": "scripts/build_temporal_flow_causal_cache.py",
            "module": "morphology/temporal_flow_causal.py",
            "source_code_sha256": _code_hashes(),
            "features_sha256": _sha256_bytes(features.astype(np.float32).tobytes()),
            "edge_id_sha256": _sha256_bytes(edge_id.tobytes()),
            "moe_scaler_sha256": moe_scaler["scaler_sha256"],
        },
        "phase1_note": (
            "SAML-D shared-core Phase-1: train/val TF only; MoE target scaler is "
            "dataset-specific (never reuses Small-HI statistics)."
            if samld_shared
            else (
                "Full-row TF matrix may include test-row features for expanding-window "
                "parity; Phase-1 PaySim downstream evaluation must not use test metrics."
                if data == "PaySim"
                else None
            )
        ),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    test_split_path = out_dir / "split_test_edge_id.npy"
    if not write_test_split_files and test_split_path.is_file():
        test_split_path.unlink()
    _atomic_save_npy(features_path, features.astype(np.float32))
    _atomic_save_npy(edge_id_path, edge_id.astype(np.int64))
    for split_name, idx in split_edge_ids.items():
        _atomic_save_npy(out_dir / f"split_{split_name}_edge_id.npy", idx.astype(np.int64))
    _atomic_write_json(meta_path, meta)
    if samld_shared:
        _atomic_write_json(out_dir / "moe_target_train_scaler.json", moe_scaler)

    logging.info("Wrote cache to %s (version=%s)", out_dir, cache_version)
    return out_dir


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, choices=list(SUPPORTED_DATASETS))
    p.add_argument("--data_config", default="data_config.json")
    p.add_argument(
        "--cache_root",
        default=None,
        help=(
            "Root directory for dataset caches. Defaults: "
            f"{DEFAULT_CACHE_ROOT_AML} for AMLWorld, {DEFAULT_CACHE_ROOT_PAYSIM} for PaySim, "
            f"{DEFAULT_CACHE_ROOT_SAMLD_SHARED_CORE} for SAML-D."
        ),
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--max_timestamps",
        type=int,
        default=None,
        help="Optional smoke subset: keep earliest K distinct timestamps only.",
    )
    p.add_argument(
        "--no_test_split_files",
        action="store_true",
        help="Omit split_test_edge_id.npy (forced for SAML-D).",
    )
    p.add_argument(
        "--train_val_only",
        action="store_true",
        help="Retain train∪val rows only (forced for SAML-D).",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    logger_setup()
    root = Path(args.cache_root) if args.cache_root else Path(default_cache_root_for(args.data))
    build_cache(
        args.data,
        args.data_config,
        root,
        overwrite=bool(args.overwrite),
        max_timestamps=args.max_timestamps,
        write_test_split_files=not bool(args.no_test_split_files),
        train_val_only=True if args.train_val_only else None,
    )


if __name__ == "__main__":
    main()
