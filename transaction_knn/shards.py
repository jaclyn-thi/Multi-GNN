"""Shard write/merge/validate for sparse transaction KNN caches."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def shard_path(shard_dir: Path, start: int, end: int) -> Path:
    return shard_dir / f"shard_{start:08d}_{end:08d}.npz"


def manifest_path(shard_dir: Path) -> Path:
    return shard_dir / "manifest.json"


def load_manifest(shard_dir: Path) -> dict:
    path = manifest_path(shard_dir)
    if not path.is_file():
        return {"shards": []}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(shard_dir: Path, manifest: dict) -> None:
    shard_dir.mkdir(parents=True, exist_ok=True)
    with manifest_path(shard_dir).open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def write_shard(
    shard_dir: Path,
    start: int,
    end: int,
    *,
    edge_ids: np.ndarray,
    csv_edge_ids: np.ndarray,
    neighbor_ids: np.ndarray,
    neighbor_sims: np.ndarray,
    metadata: dict,
) -> Path:
    shard_dir.mkdir(parents=True, exist_ok=True)
    out = shard_path(shard_dir, start, end)
    np.savez_compressed(
        out,
        edge_ids=edge_ids.astype(np.int64),
        csv_edge_ids=csv_edge_ids.astype(np.int64),
        neighbor_ids=neighbor_ids.astype(np.int64),
        neighbor_sims=neighbor_sims.astype(np.float32),
        row_start=np.asarray(start, dtype=np.int64),
        row_end=np.asarray(end, dtype=np.int64),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    manifest = load_manifest(shard_dir)
    shards = [s for s in manifest.get("shards", []) if not (s["start"] == start and s["end"] == end)]
    shards.append({"start": int(start), "end": int(end), "path": out.name})
    shards.sort(key=lambda s: (s["start"], s["end"]))
    manifest["shards"] = shards
    manifest["metadata"] = metadata
    save_manifest(shard_dir, manifest)
    logging.info("Wrote shard %s", out)
    return out


def merge_shards(shard_dir: Path, output: Path) -> Path:
    manifest = load_manifest(shard_dir)
    shards = manifest.get("shards", [])
    if not shards:
        raise FileNotFoundError(f"No shards found in {shard_dir}")

    edge_parts: List[np.ndarray] = []
    csv_parts: List[np.ndarray] = []
    neigh_parts: List[np.ndarray] = []
    sim_parts: List[np.ndarray] = []
    metadata = dict(manifest.get("metadata", {}))
    for entry in shards:
        path = shard_dir / entry["path"]
        data = np.load(path, allow_pickle=True)
        edge_parts.append(np.asarray(data["edge_ids"], dtype=np.int64))
        csv_parts.append(np.asarray(data["csv_edge_ids"], dtype=np.int64))
        neigh_parts.append(np.asarray(data["neighbor_ids"], dtype=np.int64))
        sim_parts.append(np.asarray(data["neighbor_sims"], dtype=np.float32))

    edge_ids = np.concatenate(edge_parts)
    csv_edge_ids = np.concatenate(csv_parts)
    neighbor_ids = np.concatenate(neigh_parts, axis=0)
    neighbor_sims = np.concatenate(sim_parts, axis=0)
    feature_names = metadata.get("feature_names", [])
    k = int(metadata.get("k", neighbor_ids.shape[1]))

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        edge_ids=edge_ids,
        csv_edge_ids=csv_edge_ids,
        neighbor_ids=neighbor_ids,
        neighbor_sims=neighbor_sims,
        feature_names=np.asarray(feature_names, dtype=object),
        k=np.asarray(k, dtype=np.int64),
        feature_set=np.asarray(metadata.get("feature_set", "")),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    logging.info("Merged %d shards -> %s", len(shards), output)
    return output


def validate_cache(path: Path) -> Dict[str, object]:
    data = np.load(path, allow_pickle=True)
    edge_ids = np.asarray(data["edge_ids"], dtype=np.int64).reshape(-1)
    neighbor_ids = np.asarray(data["neighbor_ids"], dtype=np.int64)
    neighbor_sims = np.asarray(data["neighbor_sims"], dtype=np.float32)
    feature_names = [str(x) for x in data["feature_names"].tolist()]
    metadata = json.loads(str(data["metadata_json"].item()))
    k = int(metadata.get("k", neighbor_ids.shape[1]))

    self_neighbor_rows = int((neighbor_ids == edge_ids[:, None]).any(axis=1).sum())
    finite_sims = neighbor_sims[np.isfinite(neighbor_sims)]
    report = {
        "path": str(path),
        "n_rows": int(edge_ids.shape[0]),
        "neighbor_shape": tuple(neighbor_ids.shape),
        "feature_names": feature_names,
        "k": k,
        "self_neighbor_rows": self_neighbor_rows,
        "sim_min": float(finite_sims.min()) if finite_sims.size else float("nan"),
        "sim_mean": float(finite_sims.mean()) if finite_sims.size else float("nan"),
        "sim_max": float(finite_sims.max()) if finite_sims.size else float("nan"),
        "metadata": metadata,
    }
    if self_neighbor_rows:
        raise ValueError(f"found self-neighbors in {self_neighbor_rows} rows")
    if neighbor_ids.shape[1] != k:
        raise ValueError("neighbor array width does not match metadata k")
    return report


def print_sanity_report(report: Dict[str, object]) -> None:
    print("=== KNN cache sanity ===")
    print(f"output_path={report['path']}")
    print(f"feature_names={','.join(report['feature_names'])}")
    print(f"n_rows={report['n_rows']}")
    print(f"query_batch_size={report['metadata'].get('query_batch_size')}")
    print(f"backend={report['metadata'].get('backend')}")
    print(f"neighbor_ids_shape={report['neighbor_shape']}")
    print(f"self_neighbor_rows={report['self_neighbor_rows']}")
    print(
        "neighbor_sims_min/mean/max="
        f"{report['sim_min']:.6f}/{report['sim_mean']:.6f}/{report['sim_max']:.6f}"
    )
    print(f"k={report['k']}")
    if report["metadata"].get("shard_dir"):
        print(f"shard_dir={report['metadata']['shard_dir']}")
