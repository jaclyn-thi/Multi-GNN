"""Sparse KNN graph adapter for the standalone txn-node baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np


@dataclass
class SparseKNNGraph:
    """Train-split sparse KNN in a chosen node id space."""

    neighbor_ids: np.ndarray  # [N, k] in node id space; -1 pad
    neighbor_sims: np.ndarray
    node_ids: np.ndarray  # length N ids matching neighbor_ids rows
    k: int
    feature_set: str
    meta: Dict[str, Any]
    deviation_notes: List[str]

    def adjacency_lists(self) -> List[np.ndarray]:
        out: List[np.ndarray] = []
        for i in range(self.neighbor_ids.shape[0]):
            row = self.neighbor_ids[i]
            valid = row[row >= 0]
            nid = int(self.node_ids[i])
            valid = valid[valid != nid]
            if valid.size > self.k:
                valid = valid[: self.k]
            out.append(valid.astype(np.int64))
        return out

    def edge_index_for_nodes(self, node_ids: np.ndarray) -> np.ndarray:
        """Induce KNN edges among ``node_ids`` (global ids), reindexed 0..B-1."""
        if node_ids.size == 0:
            return np.zeros((2, 0), dtype=np.int64)
        id_to_row = {int(e): i for i, e in enumerate(self.node_ids.tolist())}
        local = {int(e): i for i, e in enumerate(node_ids.tolist())}
        src: List[int] = []
        dst: List[int] = []
        for gid in node_ids.tolist():
            row = id_to_row.get(int(gid))
            if row is None:
                continue
            for nb in self.neighbor_ids[row].tolist():
                if nb < 0 or int(nb) == int(gid):
                    continue
                if int(nb) in local:
                    src.append(local[int(gid)])
                    dst.append(local[int(nb)])
        if not src:
            return np.zeros((2, 0), dtype=np.int64)
        return np.vstack([np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64)])


def load_train_knn_cache(
    path: Union[str, Path],
    *,
    expected_k: int = 15,
) -> SparseKNNGraph:
    """Load existing sparse global train KNN cache.

    Uses train_split_local ids from the cache. Records degree_fan as a deviation
    when present. Never materializes XX^T.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    edge_ids = np.asarray(data["edge_ids"], dtype=np.int64)
    neighbor_ids = np.asarray(data["neighbor_ids"], dtype=np.int64).copy()
    neighbor_sims = np.asarray(data["neighbor_sims"], dtype=np.float32)
    k = int(np.asarray(data["k"]).reshape(-1)[0]) if "k" in data.files else neighbor_ids.shape[1]
    feature_set = str(data["feature_set"]) if "feature_set" in data.files else "unknown"
    deviations = []
    if "degree_fan" in feature_set or "degree" in feature_set:
        deviations.append(
            "KNN cache feature_set includes degree_fan (train-graph degrees) beyond raw AML columns."
        )
    if k != expected_k:
        deviations.append(f"cache k={k} != expected_k={expected_k}")
    for i in range(neighbor_ids.shape[0]):
        neighbor_ids[i, neighbor_ids[i] == edge_ids[i]] = -1
    if neighbor_ids.shape[1] > expected_k:
        neighbor_ids = neighbor_ids[:, :expected_k]
        neighbor_sims = neighbor_sims[:, :expected_k]
        k = expected_k
    meta = {
        "path": str(path),
        "n": int(edge_ids.shape[0]),
        "k": k,
        "feature_set": feature_set,
        "id_space": "train_split_local_edge_id",
        "has_csv_edge_ids": "csv_edge_ids" in data.files,
    }
    return SparseKNNGraph(
        neighbor_ids=neighbor_ids,
        neighbor_sims=neighbor_sims,
        node_ids=edge_ids,
        k=k,
        feature_set=feature_set,
        meta=meta,
        deviation_notes=deviations,
    )


def assert_sparse_knn_bounds(graph: SparseKNNGraph) -> None:
    for i in range(graph.neighbor_ids.shape[0]):
        row = graph.neighbor_ids[i]
        valid = row[row >= 0]
        assert valid.size <= graph.k
        assert int(graph.node_ids[i]) not in set(valid.tolist())
