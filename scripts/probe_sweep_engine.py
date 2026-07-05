#!/usr/bin/env python3
"""Shared probe sweep engine: feature-matrix cache + checkpoint/resume."""

from __future__ import annotations

import hashlib
import json
import logging
import time
import traceback
from argparse import Namespace
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from linear_probe import (
    evaluate_probe,
    fit_logistic_probe,
    load_embedding_npz,
    resolve_class_weight,
    serialize_class_weight,
    tune_threshold_max_f1,
)
from scripts.probe_feature_ablation import (
    assemble_split_matrix,
    build_full_feature_matrix,
    load_dataset_frames,
    resolve_mode_groups,
)
from dataset_specs import get_dataset_spec

CACHE_VERSION = "v1"
ENGINE_VERSION = "probe_sweep_engine_v1"
ALERT_BUDGET_KS = (100, 500, 1000)


def parse_class_weight_policy(policy: str) -> Tuple[str, Optional[float]]:
    """Map sweep policy labels to linear_probe class-weight args."""
    if policy.startswith("pos_"):
        return "explicit", float(policy.split("_", 1)[1])
    return policy, None


@dataclass(frozen=True)
class SweepCellSpec:
    run_name: str
    run_label: str
    embedding_dir: str
    feature_mode: str
    class_weight_policy: str
    probe_C: float

    @property
    def cell_id(self) -> str:
        return (
            f"{self.run_name}|{self.feature_mode}|{self.class_weight_policy}|{self.probe_C:g}"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_name": self.run_name,
            "run_label": self.run_label,
            "embedding_dir": self.embedding_dir,
            "feature_mode": self.feature_mode,
            "class_weight_policy": self.class_weight_policy,
            "probe_C": self.probe_C,
            "cell_id": self.cell_id,
        }


@dataclass
class FeatureBundle:
    x_train: np.ndarray
    x_val: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_meta: Dict[str, Any]
    cache_source: str  # "memory" | "disk" | "built"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def cell_key(cell: SweepCellSpec) -> str:
    return cell.cell_id


def make_cell_specs(
    *,
    run_spec: Dict[str, str],
    feature_modes: Sequence[str],
    class_weights: Sequence[str],
    c_grid: Sequence[float],
) -> List[SweepCellSpec]:
    specs: List[SweepCellSpec] = []
    for feature_mode in feature_modes:
        for class_weight_policy in class_weights:
            for probe_c in c_grid:
                specs.append(
                    SweepCellSpec(
                        run_name=run_spec["run_name"],
                        run_label=run_spec["run_label"],
                        embedding_dir=run_spec["embedding_dir"],
                        feature_mode=feature_mode,
                        class_weight_policy=class_weight_policy,
                        probe_C=float(probe_c),
                    )
                )
    return specs


def _cache_dir(cache_root: Path, run_name: str, feature_mode: str) -> Path:
    return cache_root / run_name / feature_mode.replace("+", "_")


def _cache_meta_path(cache_dir: Path) -> Path:
    return cache_dir / "meta.json"


def _bundle_paths(cache_dir: Path) -> Dict[str, Path]:
    return {
        "train": cache_dir / "x_train.npy",
        "val": cache_dir / "x_val.npy",
        "test": cache_dir / "x_test.npy",
        "y_train": cache_dir / "y_train.npy",
        "y_val": cache_dir / "y_val.npy",
        "y_test": cache_dir / "y_test.npy",
    }


def _cache_fingerprint(
    *,
    embedding_dir: str,
    feature_mode: str,
    data: str,
    categorical_encoding: str,
) -> str:
    payload = "|".join(
        [CACHE_VERSION, embedding_dir, feature_mode, data, categorical_encoding]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _validate_labels(df, split_arrays, data: str, use_embedding: bool, x_full, groups) -> None:
    spec = get_dataset_spec(data)
    for split_name in ("train", "val", "test"):
        z, y, edge_ids = split_arrays[split_name]
        x = assemble_split_matrix(z, edge_ids, x_full, use_embedding=use_embedding)
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"{split_name}: row mismatch x={x.shape[0]} y={y.shape[0]}")
        labels_from_df = df.iloc[edge_ids][spec.label_col].to_numpy()
        if not np.array_equal(labels_from_df.astype(np.int64), y.astype(np.int64)):
            raise ValueError(f"{split_name}: labels disagree between embeddings and dataframe")


def build_feature_bundle(
    *,
    run_spec: Dict[str, str],
    feature_mode: str,
    df,
    df_train,
    tr_np: np.ndarray,
    args: Namespace,
    cache_root: Path,
    memory_cache: Dict[Tuple[str, str], FeatureBundle],
    use_disk_cache: bool = True,
) -> FeatureBundle:
    run_name = run_spec["run_name"]
    mem_key = (run_name, feature_mode)
    if mem_key in memory_cache:
        bundle = memory_cache[mem_key]
        logging.info(
            "feature_cache HIT memory run=%s mode=%s shape=%s",
            run_name,
            feature_mode,
            bundle.x_train.shape,
        )
        return bundle

    embedding_dir = Path(run_spec["embedding_dir"])
    fingerprint = _cache_fingerprint(
        embedding_dir=str(embedding_dir),
        feature_mode=feature_mode,
        data=args.data,
        categorical_encoding=args.categorical_encoding,
    )
    cache_dir = _cache_dir(cache_root, run_name, feature_mode)
    meta_path = _cache_meta_path(cache_dir)

    if use_disk_cache and meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("fingerprint") == fingerprint and meta.get("cache_version") == CACHE_VERSION:
            paths = _bundle_paths(cache_dir)
            if all(p.is_file() for p in paths.values()):
                bundle = FeatureBundle(
                    x_train=np.load(paths["train"]),
                    x_val=np.load(paths["val"]),
                    x_test=np.load(paths["test"]),
                    y_train=np.load(paths["y_train"]),
                    y_val=np.load(paths["y_val"]),
                    y_test=np.load(paths["y_test"]),
                    feature_meta=meta.get("feature_meta", {}),
                    cache_source="disk",
                )
                memory_cache[mem_key] = bundle
                logging.info(
                    "feature_cache HIT disk run=%s mode=%s dir=%s shape=%s",
                    run_name,
                    feature_mode,
                    cache_dir,
                    bundle.x_train.shape,
                )
                return bundle

    use_embedding, groups = resolve_mode_groups(feature_mode)
    split_paths = {
        "train": embedding_dir / "train.npz",
        "val": embedding_dir / "val.npz",
        "test": embedding_dir / "test.npz",
    }
    split_arrays = {k: load_embedding_npz(v) for k, v in split_paths.items()}

    x_full = None
    feature_meta: Dict[str, Any] = {
        "feature_mode": feature_mode,
        "feature_groups_included": list(groups),
        "uses_embedding": use_embedding,
        "embedding_dim": int(split_arrays["train"][0].shape[1]) if use_embedding else 0,
        "categorical_encoding": args.categorical_encoding,
        "embedding_dir": str(embedding_dir),
        "cache_version": CACHE_VERSION,
        "fingerprint": fingerprint,
    }

    if groups:
        x_raw, _, group_slices, group_meta = build_full_feature_matrix(
            df,
            df_train,
            groups,
            categorical_encoding=args.categorical_encoding,
        )
        from scripts.probe_feature_ablation import GroupwiseScaler

        scaler = GroupwiseScaler(group_slices=group_slices)
        scaler.fit(x_raw[tr_np])
        x_full = scaler.transform(x_raw)
        feature_meta.update(group_meta)
        feature_meta["scaling"] = "standard"
        feature_meta["feature_dim_non_embedding"] = int(x_full.shape[1])
    else:
        feature_meta["feature_dim_non_embedding"] = 0

    _validate_labels(df, split_arrays, args.data, use_embedding, x_full, groups)

    mats = {}
    ys = {}
    for split_name in ("train", "val", "test"):
        z, y, edge_ids = split_arrays[split_name]
        mats[split_name] = assemble_split_matrix(z, edge_ids, x_full, use_embedding=use_embedding)
        ys[split_name] = y

    feature_meta["feature_dim_total"] = int(mats["train"].shape[1])

    bundle = FeatureBundle(
        x_train=mats["train"].astype(np.float32, copy=False),
        x_val=mats["val"].astype(np.float32, copy=False),
        x_test=mats["test"].astype(np.float32, copy=False),
        y_train=ys["train"].astype(np.int64, copy=False),
        y_val=ys["val"].astype(np.int64, copy=False),
        y_test=ys["test"].astype(np.int64, copy=False),
        feature_meta=feature_meta,
        cache_source="built",
    )

    if use_disk_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        paths = _bundle_paths(cache_dir)
        np.save(paths["train"], bundle.x_train)
        np.save(paths["val"], bundle.x_val)
        np.save(paths["test"], bundle.x_test)
        np.save(paths["y_train"], bundle.y_train)
        np.save(paths["y_val"], bundle.y_val)
        np.save(paths["y_test"], bundle.y_test)
        meta_path.write_text(
            json.dumps(
                {
                    "cache_version": CACHE_VERSION,
                    "fingerprint": fingerprint,
                    "run_name": run_name,
                    "feature_mode": feature_mode,
                    "embedding_dir": str(embedding_dir),
                    "data": args.data,
                    "feature_meta": feature_meta,
                    "written_at": _utc_now(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        logging.info(
            "feature_cache WRITE disk run=%s mode=%s dir=%s shape=%s",
            run_name,
            feature_mode,
            cache_dir,
            bundle.x_train.shape,
        )

    memory_cache[mem_key] = bundle
    logging.info(
        "feature_cache BUILT run=%s mode=%s shape=%s",
        run_name,
        feature_mode,
        bundle.x_train.shape,
    )
    return bundle


def fit_probe_cell(
    *,
    bundle: FeatureBundle,
    cell: SweepCellSpec,
    args: Namespace,
) -> Dict[str, Any]:
    class_weight_mode, class_weight_pos = parse_class_weight_policy(cell.class_weight_policy)
    probe_args = Namespace(
        data=args.data,
        class_weight=class_weight_mode,
        class_weight_pos=class_weight_pos,
        model=getattr(args, "model", "gin"),
        probe_C=cell.probe_C,
        probe_max_iter=args.probe_max_iter,
        probe_n_jobs=args.probe_n_jobs,
        seed=args.seed,
    )
    class_weight = resolve_class_weight(probe_args)
    clf = fit_logistic_probe(
        bundle.x_train,
        bundle.y_train,
        class_weight=class_weight,
        max_iter=int(args.probe_max_iter),
        seed=int(args.seed),
        n_jobs=int(args.probe_n_jobs),
        C=float(cell.probe_C),
    )

    val_proba = clf.predict_proba(bundle.x_val)[:, 1]
    selected_threshold, val_f1_at_selection = tune_threshold_max_f1(bundle.y_val, val_proba)

    def _alert_budget_metrics(proba: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        y = y.astype(np.int64)
        n = int(y.shape[0])
        positives = int(y.sum())
        prevalence = float(y.mean()) if n else float("nan")
        out: Dict[str, float] = {}
        if n == 0:
            return out
        order = np.argsort(-proba)
        for k in ALERT_BUDGET_KS:
            kk = min(int(k), n)
            top = order[:kk]
            tp = int(y[top].sum())
            precision = float(tp / kk) if kk else float("nan")
            recall = float(tp / positives) if positives else float("nan")
            lift = float(precision / prevalence) if prevalence > 0 else float("nan")
            out[f"precision_at_{k}"] = precision
            out[f"recall_at_{k}"] = recall
            out[f"lift_at_{k}"] = lift
        return out

    def _split_metrics(x, y, split_name: str) -> Dict[str, Any]:
        proba = clf.predict_proba(x)[:, 1]
        metrics = evaluate_probe(clf, x, y, split_name, threshold=selected_threshold)
        metrics_default = evaluate_probe(clf, x, y, split_name, threshold=0.5)
        out = {
            "n": metrics["n"],
            "positive_rate": metrics["positive_rate"],
            "auroc": metrics["auroc"],
            "auprc": metrics["auprc"],
            "f1": metrics["f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "threshold": float(selected_threshold),
            "f1_at_0_5": metrics_default["f1"],
        }
        out.update(_alert_budget_metrics(proba, y))
        return out

    return {
        **cell.to_dict(),
        "class_weight": serialize_class_weight(class_weight),
        "feature_dim": int(bundle.x_train.shape[1]),
        "feature_cache_source": bundle.cache_source,
        "threshold": float(selected_threshold),
        "val_f1_at_selected_threshold": float(val_f1_at_selection),
        "train": _split_metrics(bundle.x_train, bundle.y_train, "train"),
        "val": _split_metrics(bundle.x_val, bundle.y_val, "val"),
        "test": _split_metrics(bundle.x_test, bundle.y_test, "test"),
        "probe_config": {
            "probe_max_iter": int(args.probe_max_iter),
            "probe_n_jobs": int(args.probe_n_jobs),
            "seed": int(args.seed),
            "solver": "lbfgs",
            "threshold_tuning": "max_f1_on_val",
        },
    }


def load_checkpoint(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {
            "engine_version": ENGINE_VERSION,
            "cells": [],
            "cells_by_id": {},
        }
    data = json.loads(path.read_text(encoding="utf-8"))
    cells = data.get("cells", [])
    cells_by_id = {c["cell_id"]: c for c in cells if "cell_id" in c}
    data["cells_by_id"] = cells_by_id
    return data


def save_checkpoint(path: Path, payload: Dict[str, Any]) -> None:
    out = dict(payload)
    out["updated_at"] = _utc_now()
    out.pop("cells_by_id", None)
    _atomic_write_json(path, out)


def run_checkpointed_sweep(
    *,
    cell_specs: Sequence[SweepCellSpec],
    run_specs_by_name: Dict[str, Dict[str, str]],
    partial_path: Path,
    final_path: Path,
    protocol: Dict[str, Any],
    args: Namespace,
    cache_root: Path,
    force: bool = False,
) -> Dict[str, Any]:
    df, df_train, tr_np, _, _, _ = load_dataset_frames(args.data, args.data_config)
    checkpoint = load_checkpoint(partial_path)
    checkpoint.setdefault("protocol", protocol)
    checkpoint.setdefault("engine_version", ENGINE_VERSION)
    checkpoint["expected_cells"] = len(cell_specs)

    cells_by_id: Dict[str, Dict[str, Any]] = checkpoint.get("cells_by_id", {})
    memory_cache: Dict[Tuple[str, str], FeatureBundle] = {}

    completed = skipped = failed = 0

    for cell in cell_specs:
        cid = cell.cell_id
        prev = cells_by_id.get(cid)
        if prev and prev.get("status") == "completed" and not force:
            skipped += 1
            logging.info("probe_sweep SKIP completed %s", cid)
            continue

        run_spec = run_specs_by_name[cell.run_name]
        t0 = time.perf_counter()
        record: Dict[str, Any] = {
            **cell.to_dict(),
            "status": "running",
            "started_at": _utc_now(),
        }
        try:
            bundle = build_feature_bundle(
                run_spec=run_spec,
                feature_mode=cell.feature_mode,
                df=df,
                df_train=df_train,
                tr_np=tr_np,
                args=args,
                cache_root=cache_root,
                memory_cache=memory_cache,
            )
            result = fit_probe_cell(bundle=bundle, cell=cell, args=args)
            runtime = time.perf_counter() - t0
            record.update(result)
            record["status"] = "completed"
            record["runtime_seconds"] = float(runtime)
            record["finished_at"] = _utc_now()
            completed += 1
            logging.info(
                "probe_sweep DONE %s | test F1=%.4f AUPRC=%.4f runtime=%.1fs cache=%s",
                cid,
                record["test"]["f1"],
                record["test"]["auprc"],
                runtime,
                bundle.cache_source,
            )
        except Exception as exc:  # noqa: BLE001 — record and continue sweep
            runtime = time.perf_counter() - t0
            record["status"] = "failed"
            record["runtime_seconds"] = float(runtime)
            record["finished_at"] = _utc_now()
            record["error"] = str(exc)
            record["traceback"] = traceback.format_exc()
            failed += 1
            logging.exception("probe_sweep FAILED %s", cid)

        cells_by_id[cid] = record
        checkpoint["cells"] = list(cells_by_id.values())
        checkpoint["cells_by_id"] = cells_by_id
        checkpoint["completed_cells"] = sum(
            1 for c in cells_by_id.values() if c.get("status") == "completed"
        )
        checkpoint["failed_cells"] = sum(
            1 for c in cells_by_id.values() if c.get("status") == "failed"
        )
        save_checkpoint(partial_path, checkpoint)

    payload = {
        "protocol": protocol,
        "engine_version": ENGINE_VERSION,
        "partial_source": str(partial_path),
        "expected_cells": len(cell_specs),
        "cells": [cells_by_id[c.cell_id] for c in cell_specs if c.cell_id in cells_by_id],
        "summary": {
            "completed": completed,
            "skipped": skipped,
            "failed": failed,
            "total_recorded": len(cells_by_id),
        },
    }

    all_done = all(
        cells_by_id.get(c.cell_id, {}).get("status") == "completed" for c in cell_specs
    )
    if all_done:
        _atomic_write_json(final_path, payload)
        logging.info("Wrote final sweep JSON %s", final_path)

    return payload
