"""Canonical / sensitivity frozen extraction for txn-node checkpoints.

Modes
-----
``frozen_checkpoint_induce_per_temporal_split_v1`` (sensitivity)
    Full-split induce isolation: each temporal split encoded on its own induced
    subgraph only.

``frozen_checkpoint_temporal_expanding_window_v1`` (thesis-primary candidate)
    Train: train nodes + train→train edges, output train.
    Val: train∪val nodes + all edges in that set, output val only.
    Test: all nodes + full flow graph, output test only.

``frozen_checkpoint_joint_full_graph_random40_v1`` (diagnostic)
    Encode all transactions jointly on the full flow graph; random-40 probe
    split applied after extraction.

``legacy_chunked_induce_4096_v0`` (historical / noncanonical)
    Graph-destructive 4096 chunking — do not use for new results.

Graph-destructive output-chunk induction is forbidden for non-legacy modes.
NOT an exact GCPAL reproduction.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from gcpal_txn_node.adjacency import induce_edge_index
from gcpal_txn_node.model import SharedTxnNodeEncoder
from gcpal_txn_node.spec import NOT_EXACT_REPRODUCTION

# Sensitivity (isolation)
PER_TEMPORAL_SPLIT_V1 = "frozen_checkpoint_induce_per_temporal_split_v1"
# Thesis-primary candidate
TEMPORAL_EXPANDING_WINDOW_V1 = "frozen_checkpoint_temporal_expanding_window_v1"
# Random-40 diagnostic
JOINT_FULL_GRAPH_RANDOM40_V1 = "frozen_checkpoint_joint_full_graph_random40_v1"
# Historical
LEGACY_CHUNKED_EXTRACTION_MODE = "legacy_chunked_induce_4096_v0"
# Back-compat alias used by earlier lock docs/tests
CANONICAL_EXTRACTION_MODE = PER_TEMPORAL_SPLIT_V1

LEGACY_CHUNK_SIZE = 4096

# Measured Small-HI immediate_next scopes (CSV row graph)
EXPECTED_EDGE_COUNTS = {
    "train": 1_614_187,
    "train_val": 2_086_777,  # 1614187 + 125991 + 346599
    "full": 2_605_952,
}
EXPECTED_N_NODES_FULL = 5_078_345


class ChunkPolicy(str, Enum):
    FULL_SPLIT = "full_split"
    LEGACY_FIXED_4096 = "legacy_fixed_4096"
    # Non-legacy modes must use ONE_SHOT_SCOPE (no graph-destructive chunking).
    ONE_SHOT_SCOPE = "one_shot_scope"


@dataclass(frozen=True)
class ExtractionConfig:
    extraction_mode: str
    chunk_policy: ChunkPolicy
    adjacency_policy: str = "immediate_next"
    graph_scope: str = "unspecified"
    uses_training_h_anchors: bool = False
    uses_augmentation: bool = False
    uses_labels_in_extraction: bool = False
    deterministic_flag: bool = True
    seed: int = 0
    protocol_role: str = "unspecified"  # sensitivity | thesis_primary_candidate | diagnostic

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["chunk_policy"] = self.chunk_policy.value
        d["not_exact_reproduction"] = bool(NOT_EXACT_REPRODUCTION)
        return d


def per_temporal_split_config(*, seed: int = 0) -> ExtractionConfig:
    return ExtractionConfig(
        extraction_mode=PER_TEMPORAL_SPLIT_V1,
        chunk_policy=ChunkPolicy.ONE_SHOT_SCOPE,
        graph_scope="induce_per_temporal_split_full",
        seed=int(seed),
        protocol_role="sensitivity",
    )


def temporal_expanding_window_config(*, seed: int = 0) -> ExtractionConfig:
    return ExtractionConfig(
        extraction_mode=TEMPORAL_EXPANDING_WINDOW_V1,
        chunk_policy=ChunkPolicy.ONE_SHOT_SCOPE,
        graph_scope="temporal_expanding_window",
        seed=int(seed),
        protocol_role="thesis_primary_candidate",
    )


def joint_full_graph_random40_config(*, seed: int = 0) -> ExtractionConfig:
    return ExtractionConfig(
        extraction_mode=JOINT_FULL_GRAPH_RANDOM40_V1,
        chunk_policy=ChunkPolicy.ONE_SHOT_SCOPE,
        graph_scope="joint_full_graph",
        seed=int(seed),
        protocol_role="diagnostic_random40",
    )


def canonical_extraction_config(*, seed: int = 0) -> ExtractionConfig:
    """Back-compat name → per-split sensitivity mode (not expanding-window)."""
    return per_temporal_split_config(seed=seed)


def legacy_chunked_extraction_config(*, seed: int = 0) -> ExtractionConfig:
    return ExtractionConfig(
        extraction_mode=LEGACY_CHUNKED_EXTRACTION_MODE,
        chunk_policy=ChunkPolicy.LEGACY_FIXED_4096,
        graph_scope="legacy_chunked_induce",
        seed=int(seed),
        protocol_role="historical_noncanonical",
    )


def sha256_file(path: Path, *, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_int64_array(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(arr.astype(np.int64, copy=False))
    return sha256_bytes(a.tobytes())


def sha256_json(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return sha256_bytes(payload.encode("utf-8"))


def configure_extraction_determinism(seed: int) -> None:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def _gpu_mem_stats(device: torch.device) -> Dict[str, Optional[float]]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {"allocated_mib": None, "reserved_mib": None, "max_allocated_mib": None}
    return {
        "allocated_mib": float(torch.cuda.memory_allocated(device) / (1024**2)),
        "reserved_mib": float(torch.cuda.memory_reserved(device) / (1024**2)),
        "max_allocated_mib": float(torch.cuda.max_memory_allocated(device) / (1024**2)),
    }


def _peak_reset(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()


@torch.no_grad()
def encode_on_scope(
    encoder: SharedTxnNodeEncoder,
    x_all: np.ndarray,
    flow_ei: np.ndarray,
    graph_node_ids: np.ndarray,
    output_node_ids: np.ndarray,
    device: torch.device,
    *,
    forbid_destructive_chunking: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """One no-grad forward on the induced subgraph of ``graph_node_ids``.

    Returns embeddings aligned to ``output_node_ids`` (must be a subset of
    ``graph_node_ids``). Never induces separate graphs per output chunk.
    """
    if encoder.training:
        encoder.eval()
    graph_node_ids = np.asarray(graph_node_ids, dtype=np.int64)
    output_node_ids = np.asarray(output_node_ids, dtype=np.int64)
    if forbid_destructive_chunking is False:
        raise ValueError("Destructive chunking is not supported in encode_on_scope")

    # Verify subset
    graph_set = set(graph_node_ids.tolist())
    missing = [int(i) for i in output_node_ids.tolist() if int(i) not in graph_set]
    if missing:
        raise AssertionError(
            f"output_node_ids not subset of graph_node_ids (e.g. {missing[:5]})"
        )

    ei_local = induce_edge_index(flow_ei, graph_node_ids)
    n_graph = int(graph_node_ids.shape[0])
    n_edges = int(ei_local.shape[1])
    # Full-graph retained fraction relative to edges among graph_node_ids in flow_ei
    # (induce should keep 100% of those)
    diagnostics = {
        "n_graph_nodes": n_graph,
        "n_output_nodes": int(output_node_ids.shape[0]),
        "n_edges_passed_to_forward": n_edges,
        "retained_edge_fraction_of_induced_scope": 1.0,
        "graph_node_id_hash": sha256_int64_array(graph_node_ids),
        "output_node_id_hash": sha256_int64_array(output_node_ids),
    }

    _peak_reset(device)
    t0 = time.perf_counter()
    try:
        x = torch.from_numpy(np.ascontiguousarray(x_all[graph_node_ids])).to(device)
        ei = torch.from_numpy(ei_local).to(device)
        h_all, _z = encoder(x, ei)
        h_np = h_all.detach().cpu().numpy()
    except RuntimeError as e:
        diagnostics["oom_or_runtime_error"] = str(e)
        diagnostics["peak_gpu"] = _gpu_mem_stats(device)
        diagnostics["wall_seconds"] = time.perf_counter() - t0
        raise

    # Map output global IDs → local rows
    mapping = -np.ones(int(graph_node_ids.max()) + 1, dtype=np.int64)
    mapping[graph_node_ids] = np.arange(n_graph, dtype=np.int64)
    local = mapping[output_node_ids]
    if (local < 0).any():
        raise AssertionError("Failed to map output IDs into graph scope")
    out = h_np[local]
    diagnostics["wall_seconds"] = float(time.perf_counter() - t0)
    diagnostics["peak_gpu"] = _gpu_mem_stats(device)
    diagnostics["embedding_dim"] = int(out.shape[1])
    diagnostics["all_finite"] = bool(np.isfinite(out).all())
    diagnostics["duplicate_output_ids"] = int(
        output_node_ids.shape[0] - np.unique(output_node_ids).shape[0]
    )
    return out, diagnostics


@torch.no_grad()
def encode_nodes_induced(
    encoder: SharedTxnNodeEncoder,
    x_all: np.ndarray,
    flow_ei: np.ndarray,
    node_ids: np.ndarray,
    device: torch.device,
    *,
    chunk_policy: ChunkPolicy = ChunkPolicy.ONE_SHOT_SCOPE,
    legacy_chunk_size: int = LEGACY_CHUNK_SIZE,
) -> np.ndarray:
    """Frozen induced encode.

    Non-legacy policies use a single induced subgraph for all ``node_ids``.
    Legacy 4096 chunking is retained only for forensic comparison.
    """
    if encoder.training:
        encoder.eval()
    node_ids = np.asarray(node_ids, dtype=np.int64)
    out = np.zeros((node_ids.shape[0], encoder.gin.out_dim), dtype=np.float32)
    if node_ids.size == 0:
        return out

    if chunk_policy in (ChunkPolicy.FULL_SPLIT, ChunkPolicy.ONE_SHOT_SCOPE):
        h, _ = encode_on_scope(
            encoder,
            x_all,
            flow_ei,
            graph_node_ids=node_ids,
            output_node_ids=node_ids,
            device=device,
        )
        return h

    if chunk_policy is ChunkPolicy.LEGACY_FIXED_4096:
        cs = int(legacy_chunk_size)
        for start in range(0, node_ids.shape[0], cs):
            end = min(start + cs, node_ids.shape[0])
            ids = node_ids[start:end]
            x = torch.from_numpy(np.ascontiguousarray(x_all[ids])).to(device)
            ei = torch.from_numpy(induce_edge_index(flow_ei, ids)).to(device)
            h, _z = encoder(x, ei)
            out[start:end] = h.detach().cpu().numpy()
        return out

    raise ValueError(f"Unknown chunk_policy: {chunk_policy}")


def load_encoder_from_checkpoint(
    ckpt_path: Path,
    *,
    in_dim: int,
    emb_dim: int = 128,
    map_location: Optional[str] = None,
) -> Tuple[SharedTxnNodeEncoder, Dict[str, Any]]:
    """Load encoder weights only (frozen extract). Ignores optimizer/RNG."""
    map_location = map_location or ("cuda" if torch.cuda.is_available() else "cpu")
    try:
        blob = torch.load(str(ckpt_path), map_location=map_location, weights_only=False)
    except TypeError:
        blob = torch.load(str(ckpt_path), map_location=map_location)
    encoder = SharedTxnNodeEncoder(in_dim=in_dim, emb_dim=emb_dim)
    state = blob.get("model_state_dict") or blob.get("encoder_state_dict")
    if state is None:
        raise KeyError(f"No model_state_dict in checkpoint {ckpt_path}")
    encoder.load_state_dict(state)
    encoder.eval()
    meta = {
        "epoch": blob.get("epoch"),
        "total_opt_steps": blob.get("total_opt_steps"),
        "checkpoint_meta": blob.get("meta"),
        "has_optimizer_state": "optimizer_state_dict" in blob,
    }
    return encoder, meta


def verify_scope_edge_count(
    flow_ei: np.ndarray,
    graph_node_ids: np.ndarray,
    *,
    expected: Optional[int] = None,
    name: str = "scope",
) -> Dict[str, Any]:
    ei = induce_edge_index(flow_ei, np.asarray(graph_node_ids, dtype=np.int64))
    n_edges = int(ei.shape[1])
    out = {"scope": name, "n_nodes": int(len(graph_node_ids)), "n_edges": n_edges}
    if expected is not None:
        out["expected_n_edges"] = int(expected)
        out["matches_expected"] = n_edges == int(expected)
        if n_edges != int(expected):
            raise AssertionError(
                f"{name}: induced edges {n_edges} != expected {expected}"
            )
    return out


def extract_temporal_expanding_window(
    *,
    encoder: SharedTxnNodeEncoder,
    x_all: np.ndarray,
    flow_ei: np.ndarray,
    tr: np.ndarray,
    va: np.ndarray,
    te: np.ndarray,
    device: torch.device,
    config: Optional[ExtractionConfig] = None,
    checkpoint_path: Optional[Path] = None,
    verify_expected_edges: bool = True,
) -> Dict[str, Any]:
    """Thesis-primary-candidate temporal expanding-window extraction."""
    config = config or temporal_expanding_window_config()
    if config.extraction_mode != TEMPORAL_EXPANDING_WINDOW_V1:
        raise ValueError(f"Expected mode {TEMPORAL_EXPANDING_WINDOW_V1}")
    if config.uses_training_h_anchors or config.uses_augmentation:
        raise RuntimeError("Expanding-window extraction forbids h_anchors/augmentation")
    if config.uses_labels_in_extraction:
        raise RuntimeError("Extraction must be label-free")
    if config.deterministic_flag:
        configure_extraction_determinism(config.seed)

    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    tr = np.asarray(tr, dtype=np.int64)
    va = np.asarray(va, dtype=np.int64)
    te = np.asarray(te, dtype=np.int64)
    train_val = np.concatenate([tr, va])

    scope_checks = []
    if verify_expected_edges:
        scope_checks.append(
            verify_scope_edge_count(flow_ei, tr, expected=EXPECTED_EDGE_COUNTS["train"], name="train")
        )
        scope_checks.append(
            verify_scope_edge_count(
                flow_ei, train_val, expected=EXPECTED_EDGE_COUNTS["train_val"], name="train_val"
            )
        )
        all_ids = np.concatenate([tr, va, te])
        # Full graph may include only these nodes if df is complete
        scope_checks.append(
            verify_scope_edge_count(
                flow_ei, all_ids, expected=EXPECTED_EDGE_COUNTS["full"], name="full"
            )
        )

    # Assert train graph has no val/test endpoints
    ei_tr = induce_edge_index(flow_ei, tr)
    if ei_tr.shape[1]:
        gsrc = tr[ei_tr[0]]
        gdst = tr[ei_tr[1]]
        bad = set(gsrc.tolist()) | set(gdst.tolist())
        if bad & set(va.tolist()) or bad & set(te.tolist()):
            raise AssertionError("Train induce leaked val/test endpoints")

    forwards = {}
    h_train, d_tr = encode_on_scope(encoder, x_all, flow_ei, tr, tr, device)
    forwards["train"] = d_tr
    h_val, d_va = encode_on_scope(encoder, x_all, flow_ei, train_val, va, device)
    forwards["validation"] = d_va
    all_ids = np.concatenate([tr, va, te])
    h_test, d_te = encode_on_scope(encoder, x_all, flow_ei, all_ids, te, device)
    forwards["test"] = d_te

    ckpt_hash = sha256_file(checkpoint_path) if checkpoint_path and checkpoint_path.is_file() else None
    return {
        "not_exact_reproduction": bool(NOT_EXACT_REPRODUCTION),
        "extraction_mode": config.extraction_mode,
        "config": config.to_dict(),
        "config_hash": sha256_json(config.to_dict()),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_hash_sha256": ckpt_hash,
        "embeddings": {"train": h_train, "val": h_val, "test": h_test},
        "split_node_ids": {"train": tr, "val": va, "test": te},
        "row_id_hashes_sha256": {
            "train": sha256_int64_array(tr),
            "val": sha256_int64_array(va),
            "test": sha256_int64_array(te),
        },
        "coverage": {
            k: {
                "n_requested": int(v.shape[0]),
                "n_written": int(forwards[k if k != "val" else "validation"]["n_output_nodes"]),
                "dim": int(v.shape[1]),
                "coverage": 1.0,
                "all_finite": bool(np.isfinite(v).all()),
            }
            for k, v in [("train", h_train), ("val", h_val), ("test", h_test)]
        },
        "forward_diagnostics": forwards,
        "scope_edge_checks": scope_checks,
        "graph_scope_note": (
            "Expanding window: train on train→train; val on train∪val; test on full graph; "
            "output rows are split-specific."
        ),
    }


def extract_joint_full_graph(
    *,
    encoder: SharedTxnNodeEncoder,
    x_all: np.ndarray,
    flow_ei: np.ndarray,
    all_node_ids: np.ndarray,
    device: torch.device,
    config: Optional[ExtractionConfig] = None,
    checkpoint_path: Optional[Path] = None,
    verify_expected_edges: bool = True,
) -> Dict[str, Any]:
    """Joint full-graph encode (for random-40 diagnostic probe after extraction)."""
    config = config or joint_full_graph_random40_config()
    if config.extraction_mode != JOINT_FULL_GRAPH_RANDOM40_V1:
        raise ValueError(f"Expected mode {JOINT_FULL_GRAPH_RANDOM40_V1}")
    if config.uses_training_h_anchors or config.uses_augmentation:
        raise RuntimeError("Joint full-graph extraction forbids h_anchors/augmentation")
    if config.uses_labels_in_extraction:
        raise RuntimeError("Extraction must be label-free")
    if config.deterministic_flag:
        configure_extraction_determinism(config.seed)

    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    all_node_ids = np.asarray(all_node_ids, dtype=np.int64)
    scope_checks = []
    if verify_expected_edges:
        scope_checks.append(
            verify_scope_edge_count(
                flow_ei, all_node_ids, expected=EXPECTED_EDGE_COUNTS["full"], name="full"
            )
        )

    h_all, diag = encode_on_scope(
        encoder, x_all, flow_ei, all_node_ids, all_node_ids, device
    )
    ckpt_hash = sha256_file(checkpoint_path) if checkpoint_path and checkpoint_path.is_file() else None
    return {
        "not_exact_reproduction": bool(NOT_EXACT_REPRODUCTION),
        "extraction_mode": config.extraction_mode,
        "config": config.to_dict(),
        "config_hash": sha256_json(config.to_dict()),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_hash_sha256": ckpt_hash,
        "embeddings": {"all": h_all},
        "split_node_ids": {"all": all_node_ids},
        "row_id_hashes_sha256": {"all": sha256_int64_array(all_node_ids)},
        "coverage": {
            "all": {
                "n_requested": int(all_node_ids.shape[0]),
                "n_written": int(h_all.shape[0]),
                "dim": int(h_all.shape[1]),
                "coverage": 1.0,
                "all_finite": bool(np.isfinite(h_all).all()),
            }
        },
        "forward_diagnostics": {"joint_full": diag},
        "scope_edge_checks": scope_checks,
        "label": [
            "random-40",
            "transductive",
            "diagnostic-only",
            "not thesis-primary",
        ],
        "graph_scope_note": (
            "All transactions encoded jointly on the full flow graph; "
            "random-40 label/probe split must be applied after extraction."
        ),
    }


def extract_split_embeddings(
    *,
    encoder: SharedTxnNodeEncoder,
    x_all: np.ndarray,
    flow_ei: np.ndarray,
    split_node_ids: Mapping[str, np.ndarray],
    device: torch.device,
    config: ExtractionConfig,
    checkpoint_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Per-split isolation (sensitivity) or legacy chunked extract."""
    if config.uses_training_h_anchors:
        raise RuntimeError("Canonical extraction forbids training h_anchors caches")
    if config.uses_augmentation:
        raise RuntimeError("Canonical extraction forbids training-time augmentation")
    if config.uses_labels_in_extraction:
        raise RuntimeError("Extraction must be label-free")
    if config.deterministic_flag:
        configure_extraction_determinism(config.seed)

    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    embeddings: Dict[str, np.ndarray] = {}
    coverage: Dict[str, Any] = {}
    row_id_hashes: Dict[str, str] = {}
    forward_diagnostics: Dict[str, Any] = {}
    for name, ids in split_node_ids.items():
        ids_arr = np.asarray(ids, dtype=np.int64)
        if config.chunk_policy is ChunkPolicy.LEGACY_FIXED_4096:
            h = encode_nodes_induced(
                encoder,
                x_all,
                flow_ei,
                ids_arr,
                device,
                chunk_policy=config.chunk_policy,
            )
            forward_diagnostics[name] = {
                "n_graph_nodes": int(ids_arr.shape[0]),
                "n_edges_passed_to_forward": "per_chunk_variable",
                "legacy_chunked": True,
            }
        else:
            h, diag = encode_on_scope(
                encoder, x_all, flow_ei, ids_arr, ids_arr, device
            )
            forward_diagnostics[name] = diag
        embeddings[name] = h
        coverage[name] = {
            "n_requested": int(ids_arr.shape[0]),
            "n_written": int(h.shape[0]),
            "dim": int(h.shape[1]) if h.ndim == 2 else None,
            "coverage": 1.0 if ids_arr.shape[0] == h.shape[0] else float(h.shape[0]) / max(ids_arr.shape[0], 1),
            "all_finite": bool(np.isfinite(h).all()),
        }
        row_id_hashes[name] = sha256_int64_array(ids_arr)

    ckpt_hash = sha256_file(checkpoint_path) if checkpoint_path and checkpoint_path.is_file() else None
    config_hash = sha256_json(config.to_dict())
    return {
        "not_exact_reproduction": bool(NOT_EXACT_REPRODUCTION),
        "extraction_mode": config.extraction_mode,
        "config": config.to_dict(),
        "config_hash": config_hash,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_hash_sha256": ckpt_hash,
        "embeddings": embeddings,
        "split_node_ids": {k: np.asarray(v, dtype=np.int64) for k, v in split_node_ids.items()},
        "row_id_hashes_sha256": row_id_hashes,
        "coverage": coverage,
        "forward_diagnostics": forward_diagnostics,
        "graph_scope_note": (
            "Per-temporal-split isolation (sensitivity) or legacy chunked mode."
        ),
    }


def assert_row_id_alignment(
    node_ids: np.ndarray,
    embeddings: np.ndarray,
    *,
    expected_ids: Optional[Sequence[int]] = None,
) -> None:
    if embeddings.shape[0] != node_ids.shape[0]:
        raise AssertionError(
            f"Row count mismatch: embeddings={embeddings.shape[0]} ids={node_ids.shape[0]}"
        )
    if expected_ids is not None:
        exp = np.asarray(expected_ids, dtype=np.int64)
        if not np.array_equal(node_ids, exp):
            raise AssertionError("Transaction ID order does not match expected IDs")


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
