"""GPU/CPU KNN backends for transaction feature precompute."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore


def list_backends() -> List[str]:
    return ["auto", "cpu", "faiss_gpu", "faiss_cpu", "torch_gpu", "faiss_ivf"]


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(norms, eps)).astype(np.float32)


def _distance_to_similarity(distances: np.ndarray, metric: str) -> np.ndarray:
    if metric == "cosine":
        return (1.0 - distances).astype(np.float32)
    return (-distances).astype(np.float32)


def _similarity_to_distance(similarities: np.ndarray, metric: str) -> np.ndarray:
    if metric == "cosine":
        return (1.0 - similarities).astype(np.float32)
    return (-similarities).astype(np.float32)


def _drop_self_neighbors(
    query_indices: np.ndarray,
    neighbor_indices: np.ndarray,
    neighbor_values: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n_queries = neighbor_indices.shape[0]
    out_idx = np.full((n_queries, k), -1, dtype=np.int64)
    out_val = np.full((n_queries, k), np.nan, dtype=np.float32)
    for local_i, global_i in enumerate(query_indices):
        keep = neighbor_indices[local_i] != int(global_i)
        idx = neighbor_indices[local_i][keep][:k]
        val = neighbor_values[local_i][keep][:k]
        out_idx[local_i, : idx.shape[0]] = idx.astype(np.int64)
        out_val[local_i, : idx.shape[0]] = val.astype(np.float32)
    return out_idx, out_val


class KNNBackend(ABC):
    name: str

    def __init__(self, metric: str = "cosine") -> None:
        if metric not in ("cosine", "euclidean"):
            raise ValueError(f"Unsupported metric {metric!r}")
        self.metric = metric
        self.backend_id: str = self.name

    @abstractmethod
    def fit(self, features: np.ndarray, *, k: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def query(self, query_indices: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return neighbor indices and similarities for global row ids in query_indices."""

    def recall_at_k(
        self,
        query_indices: np.ndarray,
        k: int,
        other: "KNNBackend",
    ) -> float:
        approx_idx, _ = self.query(query_indices, k)
        exact_idx, _ = other.query(query_indices, k)
        hits = 0
        total = 0
        for row in range(query_indices.shape[0]):
            a = set(int(x) for x in approx_idx[row] if x >= 0)
            e = set(int(x) for x in exact_idx[row] if x >= 0)
            if not e:
                continue
            hits += len(a & e)
            total += len(e)
        return float(hits / total) if total else float("nan")


class SklearnKNNBackend(KNNBackend):
    name = "cpu"

    def __init__(self, metric: str = "cosine") -> None:
        super().__init__(metric=metric)
        self.backend_id = "cpu"
        self._nbrs = None
        self._features: Optional[np.ndarray] = None
        self._fit_neighbors = 1

    def fit(self, features: np.ndarray, *, k: int) -> None:
        from sklearn.neighbors import NearestNeighbors

        self._features = np.asarray(features, dtype=np.float32)
        self._fit_neighbors = min(int(k) + 1, self._features.shape[0])
        self._nbrs = NearestNeighbors(
            n_neighbors=self._fit_neighbors,
            metric=self.metric,
            algorithm="auto",
            n_jobs=-1,
        )
        self._nbrs.fit(self._features)

    def query(self, query_indices: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self._features is None or self._nbrs is None:
            raise RuntimeError("fit() must be called before query()")
        queries = self._features[query_indices]
        n_query = min(int(k) + 1, self._fit_neighbors)
        distances, indices = self._nbrs.kneighbors(queries, n_neighbors=n_query, return_distance=True)
        sims = _distance_to_similarity(distances, self.metric)
        return _drop_self_neighbors(query_indices, indices.astype(np.int64), sims, k)


def _torch_merge_topk(
    best_vals: torch.Tensor,
    best_idx: torch.Tensor,
    cand_vals: torch.Tensor,
    cand_idx: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    merged_vals = torch.cat([best_vals, cand_vals], dim=1)
    merged_idx = torch.cat([best_idx, cand_idx], dim=1)
    new_vals, order = torch.topk(merged_vals, k=k, dim=1)
    new_idx = torch.gather(merged_idx, 1, order)
    return new_vals, new_idx


class TorchGpuKNNBackend(KNNBackend):
    name = "torch_gpu"

    def __init__(
        self,
        metric: str = "cosine",
        device: Optional[str] = None,
        db_chunk_size: int = 65_536,
    ) -> None:
        super().__init__(metric=metric)
        self.backend_id = "torch_gpu"
        if torch is None or not torch.cuda.is_available():
            raise RuntimeError("torch_gpu backend requires PyTorch with CUDA")
        self.device = torch.device(device or "cuda:0")
        self.db_chunk_size = max(1, int(db_chunk_size))
        self._features: Optional[torch.Tensor] = None

    def fit(self, features: np.ndarray, *, k: int) -> None:
        x = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        if self.metric == "cosine":
            x = torch.nn.functional.normalize(x, p=2, dim=1)
        self._features = x

    def query(self, query_indices: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self._features is None:
            raise RuntimeError("fit() must be called before query()")
        queries = self._features[query_indices]
        k_query = int(k) + 1
        n_db = int(self._features.shape[0])

        if self.metric == "cosine":
            best_vals: Optional[torch.Tensor] = None
            best_idx: Optional[torch.Tensor] = None
            for db_start in range(0, n_db, self.db_chunk_size):
                db_end = min(db_start + self.db_chunk_size, n_db)
                db_chunk = self._features[db_start:db_end]
                sims = queries @ db_chunk.T
                chunk_k = min(k_query, sims.shape[1])
                local = torch.topk(sims, k=chunk_k, dim=1)
                global_idx = local.indices + int(db_start)
                if best_vals is None:
                    best_vals = local.values
                    best_idx = global_idx
                else:
                    best_vals, best_idx = _torch_merge_topk(
                        best_vals, best_idx, local.values, global_idx, k_query
                    )
            assert best_vals is not None and best_idx is not None
            indices = best_idx.detach().cpu().numpy()
            values = best_vals.detach().cpu().numpy()
        else:
            indices_list = []
            values_list = []
            q_chunk = 256
            for q_start in range(0, queries.shape[0], q_chunk):
                q_end = min(q_start + q_chunk, queries.shape[0])
                q = queries[q_start:q_end]
                chunk_best_vals: Optional[torch.Tensor] = None
                chunk_best_idx: Optional[torch.Tensor] = None
                for db_start in range(0, n_db, self.db_chunk_size):
                    db_end = min(db_start + self.db_chunk_size, n_db)
                    dist = torch.cdist(q, self._features[db_start:db_end], p=2)
                    chunk_k = min(k_query, dist.shape[1])
                    local = torch.topk(dist, k=chunk_k, largest=False, dim=1)
                    global_idx = local.indices + int(db_start)
                    if chunk_best_vals is None:
                        chunk_best_vals = -local.values
                        chunk_best_idx = global_idx
                    else:
                        chunk_best_vals, chunk_best_idx = _torch_merge_topk(
                            chunk_best_vals,
                            chunk_best_idx,
                            -local.values,
                            global_idx,
                            k_query,
                        )
                assert chunk_best_vals is not None and chunk_best_idx is not None
                indices_list.append(chunk_best_idx.detach().cpu().numpy())
                values_list.append(chunk_best_vals.detach().cpu().numpy())
            indices = np.concatenate(indices_list, axis=0)
            values = np.concatenate(values_list, axis=0)
        return _drop_self_neighbors(query_indices, indices.astype(np.int64), values.astype(np.float32), k)


class FaissExactBackend(KNNBackend):
    name = "faiss"

    def __init__(self, metric: str = "cosine", use_gpu: bool = True, gpu_id: int = 0) -> None:
        super().__init__(metric=metric)
        try:
            import faiss  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("faiss backend requires faiss-cpu or faiss-gpu") from exc
        self.faiss = faiss
        self.use_gpu = bool(use_gpu)
        self.gpu_id = int(gpu_id)
        self.backend_id = "faiss_gpu" if self.use_gpu else "faiss_cpu"
        self._index = None
        self._features: Optional[np.ndarray] = None

    def fit(self, features: np.ndarray, *, k: int) -> None:
        faiss = self.faiss
        x = np.ascontiguousarray(features.astype(np.float32))
        dim = x.shape[1]
        if self.metric == "cosine":
            x = _l2_normalize(x)
            index = faiss.IndexFlatIP(dim)
        else:
            index = faiss.IndexFlatL2(dim)
        if self.use_gpu:
            if not hasattr(faiss, "StandardGpuResources"):
                raise RuntimeError("faiss GPU support not available in this install")
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, self.gpu_id, index)
        index.add(x)
        self._index = index
        self._features = x

    def query(self, query_indices: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self._index is None or self._features is None:
            raise RuntimeError("fit() must be called before query()")
        queries = np.ascontiguousarray(self._features[query_indices])
        distances, indices = self._index.search(queries, k + 1)
        if self.metric == "cosine":
            sims = distances.astype(np.float32)
        else:
            sims = (-distances).astype(np.float32)
        return _drop_self_neighbors(query_indices, indices.astype(np.int64), sims, k)


class FaissIVFBackend(FaissExactBackend):
    name = "faiss_ivf"

    def __init__(
        self,
        metric: str = "cosine",
        use_gpu: bool = True,
        gpu_id: int = 0,
        nlist: int = 4096,
        nprobe: int = 64,
        train_size: int = 200_000,
    ) -> None:
        super().__init__(metric=metric, use_gpu=use_gpu, gpu_id=gpu_id)
        self.backend_id = "faiss_ivf"
        self.nlist = int(nlist)
        self.nprobe = int(nprobe)
        self.train_size = int(train_size)

    def fit(self, features: np.ndarray, *, k: int) -> None:
        faiss = self.faiss
        x = np.ascontiguousarray(features.astype(np.float32))
        dim = x.shape[1]
        if self.metric == "cosine":
            x = _l2_normalize(x)
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, self.nlist, faiss.METRIC_INNER_PRODUCT)
        else:
            quantizer = faiss.IndexFlatL2(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, self.nlist, faiss.METRIC_L2)
        train_n = min(self.train_size, x.shape[0])
        rng = np.random.default_rng(0)
        train_idx = rng.choice(x.shape[0], size=train_n, replace=False)
        index.train(x[train_idx])
        if self.use_gpu:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, self.gpu_id, index)
        index.nprobe = self.nprobe
        index.add(x)
        self._index = index
        self._features = x


def _faiss_gpu_available() -> bool:
    try:
        import faiss  # type: ignore

        return hasattr(faiss, "StandardGpuResources")
    except ImportError:
        return False


def _faiss_cpu_available() -> bool:
    try:
        import faiss  # type: ignore

        return True
    except ImportError:
        return False


def build_backend(
    backend: str,
    metric: str = "cosine",
    *,
    faiss_nlist: int = 4096,
    faiss_nprobe: int = 64,
    faiss_train_size: int = 200_000,
    torch_device: Optional[str] = None,
) -> KNNBackend:
    choice = str(backend).lower()
    if choice == "auto":
        if _faiss_gpu_available():
            choice = "faiss_gpu"
        elif torch is not None and torch.cuda.is_available():
            choice = "torch_gpu"
        else:
            choice = "cpu"
        logging.info("Resolved backend=auto -> %s", choice)

    backend: KNNBackend
    if choice == "cpu":
        backend = SklearnKNNBackend(metric=metric)
    elif choice == "faiss_gpu":
        backend = FaissExactBackend(metric=metric, use_gpu=True)
    elif choice == "faiss_cpu":
        backend = FaissExactBackend(metric=metric, use_gpu=False)
    elif choice == "torch_gpu":
        backend = TorchGpuKNNBackend(metric=metric, device=torch_device)
    elif choice == "faiss_ivf":
        backend = FaissIVFBackend(
            metric=metric,
            use_gpu=_faiss_gpu_available(),
            nlist=faiss_nlist,
            nprobe=faiss_nprobe,
            train_size=faiss_train_size,
        )
    else:
        raise ValueError(f"Unsupported backend {backend!r}; choose from {list_backends()}")
    return backend


def build_exact_reference_backend(metric: str) -> KNNBackend:
    if _faiss_gpu_available():
        return FaissExactBackend(metric=metric, use_gpu=True)
    if torch is not None and torch.cuda.is_available():
        return TorchGpuKNNBackend(metric=metric)
    return SklearnKNNBackend(metric=metric)
