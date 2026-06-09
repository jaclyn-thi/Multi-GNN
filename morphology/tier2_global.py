"""Tier 2: split-global expensive node statistics (M3 Phase 0 — betweenness centrality).

Computed offline per split on the forward directed transaction graph. Phase 0 provides
precompute + endpoint lift plumbing only; expert/contrast training integration is deferred.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch

from morphology.graph_access import get_forward_edge_index, get_num_nodes, seed_endpoints_from_edge_ids
from morphology.tier0_global import load_node_table, save_node_table

TIER2_NODE_COLUMNS = ("bc",)

# Endpoint lift for BC (log1p applied at expert target assembly when wired in M3).
TIER2_BC_LIFT_FEATURE_NAMES: List[str] = [
    "sender_bc",
    "receiver_bc",
    "bc_sum_global",
    "bc_max_global",
]

TIER2_BC_LIFT_MODE_FULL = "full"
TIER2_BC_LIFT_MODE_MAX = "max"
TIER2_BC_LIFT_MODES = (TIER2_BC_LIFT_MODE_FULL, TIER2_BC_LIFT_MODE_MAX)

# All Tier 2 lift columns are non-negative scalars; apply log1p before expert MSE (full mode).
TIER2_BC_COUNT_FEATURE_INDICES: tuple[int, ...] = (0, 1, 2, 3)


def tier2_bc_lift_feature_names(lift_mode: str = TIER2_BC_LIFT_MODE_FULL) -> List[str]:
    """Feature names for the selected Tier 2 BC lift mode."""
    mode = str(lift_mode).lower()
    if mode == TIER2_BC_LIFT_MODE_MAX:
        return ["bc_max_global"]
    if mode == TIER2_BC_LIFT_MODE_FULL:
        return list(TIER2_BC_LIFT_FEATURE_NAMES)
    raise ValueError(f"Unknown tier2 BC lift mode {lift_mode!r}; use {TIER2_BC_LIFT_MODES}.")


def tier2_bc_lift_dim(lift_mode: str = TIER2_BC_LIFT_MODE_FULL) -> int:
    return len(tier2_bc_lift_feature_names(lift_mode))


@dataclass
class MorphTier2Context:
    """
    Split-global Tier 2 lookup for betweenness centrality endpoint lift.

    ``bc`` is indexed by ``node_id`` (float32, non-negative).
    """

    edge_index: torch.Tensor
    bc: torch.Tensor

    @property
    def num_nodes(self) -> int:
        return int(self.bc.shape[0])

    @property
    def device(self) -> torch.device:
        return self.edge_index.device


def _adjacency_lists(edge_index: torch.Tensor, num_nodes: int) -> List[List[int]]:
    """Forward adjacency lists from ``edge_index`` ``[2, E]`` (sender -> receiver)."""
    ei = edge_index.cpu().numpy()
    adj: List[List[int]] = [[] for _ in range(num_nodes)]
    for u, v in zip(ei[0], ei[1]):
        u_i = int(u)
        v_i = int(v)
        if u_i < 0 or u_i >= num_nodes or v_i < 0 or v_i >= num_nodes:
            raise ValueError(f"edge_index out of range for num_nodes={num_nodes}: ({u_i}, {v_i})")
        adj[u_i].append(v_i)
    return adj


def _brandes_betweenness_directed(
    adj: Sequence[Sequence[int]],
    sources: Sequence[int],
    normalized: bool,
) -> np.ndarray:
    """
    Brandes betweenness for a directed graph, accumulating only from ``sources``.

    When ``sources`` is all nodes, this is exact BC. When ``sources`` is a sample,
    returns an unscaled partial sum; caller should scale by ``n / len(sources)``.
    """
    n = len(adj)
    bc = np.zeros(n, dtype=np.float64)
    for s in sources:
        stack: List[int] = []
        pred: List[List[int]] = [[] for _ in range(n)]
        sigma = np.zeros(n, dtype=np.float64)
        dist = np.full(n, -1, dtype=np.int64)
        sigma[s] = 1.0
        dist[s] = 0
        queue: List[int] = [s]

        while queue:
            v = queue.pop(0)
            stack.append(v)
            for w in adj[v]:
                if dist[w] < 0:
                    queue.append(w)
                    dist[w] = dist[v] + 1
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)

        delta = np.zeros(n, dtype=np.float64)
        while stack:
            w = stack.pop()
            for v in pred[w]:
                if sigma[w] > 0:
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != s:
                bc[w] += delta[w]

    if normalized and n > 1:
        # Freeman normalization for directed graphs.
        bc *= 1.0 / ((n - 1) * (n - 2))
    return bc


def compute_betweenness_centrality_directed(
    edge_index: torch.Tensor,
    num_nodes: int,
    *,
    k_samples: Optional[int] = None,
    normalized: bool = True,
    seed: int = 1,
) -> np.ndarray:
    """
    Betweenness centrality on the forward directed split graph.

    Parameters
    ----------
    edge_index :
        ``[2, E]`` forward edges (sender, receiver).
    num_nodes :
        Number of nodes in the split.
    k_samples :
        If ``None``, exact BC (all nodes as BFS sources). If ``< num_nodes``,
        approximate by sampling that many source nodes without replacement.
    normalized :
        Apply standard (n-1)(n-2) normalization for directed graphs.
    seed :
        RNG seed for source sampling.

    Returns
    -------
    float64 array ``(num_nodes,)`` of BC scores (non-negative).
    """
    if num_nodes <= 0:
        return np.zeros(0, dtype=np.float64)
    if num_nodes == 1:
        return np.zeros(1, dtype=np.float64)

    adj = _adjacency_lists(edge_index, num_nodes)
    if k_samples is None or k_samples >= num_nodes:
        sources = list(range(num_nodes))
        scale = 1.0
    else:
        k = max(1, int(k_samples))
        rng = np.random.default_rng(seed)
        sources = rng.choice(num_nodes, size=k, replace=False).tolist()
        scale = float(num_nodes) / float(len(sources))

    bc = _brandes_betweenness_directed(adj, sources, normalized=False)
    bc *= scale
    if normalized and num_nodes > 2:
        bc *= 1.0 / ((num_nodes - 1) * (num_nodes - 2))
    bc = np.maximum(bc, 0.0)
    return bc


def compute_tier2_node_stats(
    edge_index: torch.Tensor,
    num_nodes: int,
    *,
    k_samples: Optional[int] = None,
    normalized: bool = True,
    seed: int = 1,
) -> pd.DataFrame:
    """
    Build a Tier 2 node table (currently ``bc`` only) indexed by ``node_id``.
    """
    bc = compute_betweenness_centrality_directed(
        edge_index,
        num_nodes,
        k_samples=k_samples,
        normalized=normalized,
        seed=seed,
    )
    df = pd.DataFrame({"bc": bc.astype(np.float64)}, index=np.arange(num_nodes, dtype=np.int64))
    df.index.name = "node_id"
    return df


def _context_from_table(
    edge_index: torch.Tensor,
    node_table: pd.DataFrame,
    device: torch.device,
) -> MorphTier2Context:
    if "bc" not in node_table.columns:
        raise KeyError("node_table missing column 'bc'")
    bc = torch.as_tensor(node_table["bc"].to_numpy(), device=device, dtype=torch.float32)
    return MorphTier2Context(edge_index=edge_index, bc=bc)


def tier2_context_from_cache(
    path: Union[str, Path],
    split_data,
    device: torch.device,
) -> MorphTier2Context:
    """Load ``{split}_node_tier2.csv`` and pair with the split forward ``edge_index``."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Tier 2 cache not found: {path}")
    table = load_node_table(path)
    n_nodes = get_num_nodes(split_data)
    if len(table) != n_nodes:
        raise ValueError(
            f"Tier 2 table rows ({len(table)}) != split num_nodes ({n_nodes}) for {path}"
        )
    ei = get_forward_edge_index(split_data).to(device)
    return _context_from_table(ei, table, device)


def lift_tier2_bc_to_seed_edges_torch(
    seed_edge_ids: torch.Tensor,
    tier2_ctx: MorphTier2Context,
    *,
    lift_mode: str = TIER2_BC_LIFT_MODE_FULL,
) -> torch.Tensor:
    """
    Lift BC to seed edges via sender/receiver endpoints.

    ``lift_mode``:
      - ``full``: sender, receiver, sum, max (4 cols)
      - ``max``: ``bc_max_global`` only (1 col)
    """
    mode = str(lift_mode).lower()
    n_cols = tier2_bc_lift_dim(mode)
    device = tier2_ctx.device
    seed = seed_edge_ids.long().view(-1).to(device)
    ei = tier2_ctx.edge_index
    if seed.numel() == 0:
        return torch.empty((0, n_cols), device=device, dtype=torch.float32)
    if seed.max().item() >= ei.shape[1] or seed.min().item() < 0:
        raise ValueError(
            f"seed edge_ids out of range for split edge_index E={ei.shape[1]}: "
            f"min={seed.min().item()} max={seed.max().item()}"
        )
    senders = ei[0, seed]
    receivers = ei[1, seed]
    s_bc = tier2_ctx.bc[senders]
    r_bc = tier2_ctx.bc[receivers]
    if mode == TIER2_BC_LIFT_MODE_MAX:
        return torch.maximum(s_bc, r_bc).unsqueeze(1)
    if mode == TIER2_BC_LIFT_MODE_FULL:
        return torch.stack(
            [
                s_bc,
                r_bc,
                s_bc + r_bc,
                torch.maximum(s_bc, r_bc),
            ],
            dim=1,
        )
    raise ValueError(f"Unknown tier2 BC lift mode {lift_mode!r}; use {TIER2_BC_LIFT_MODES}.")


def lift_tier2_bc_to_seed_edges_np(
    seed_edge_ids: Union[torch.Tensor, np.ndarray, Sequence[int]],
    edge_index: torch.Tensor,
    node_table: pd.DataFrame,
) -> np.ndarray:
    """NumPy lift for tests and offline analysis."""
    eid = torch.as_tensor(seed_edge_ids, dtype=torch.long).view(-1)
    senders, receivers = seed_endpoints_from_edge_ids(eid, edge_index)
    s = node_table.loc[senders.numpy(), "bc"].to_numpy(dtype=np.float32)
    t = node_table.loc[receivers.numpy(), "bc"].to_numpy(dtype=np.float32)
    return np.stack([s, t, s + t, np.maximum(s, t)], axis=1).astype(np.float32)


def precompute_tier2_for_split(
    split_data,
    *,
    k_samples: Optional[int] = 256,
    normalized: bool = True,
    seed: int = 1,
) -> tuple[pd.DataFrame, dict]:
    """
    Compute Tier 2 node stats for one split; returns table and timing metadata.
    """
    t0 = time.perf_counter()
    ei = get_forward_edge_index(split_data)
    n_nodes = get_num_nodes(split_data)
    table = compute_tier2_node_stats(
        ei,
        n_nodes,
        k_samples=k_samples,
        normalized=normalized,
        seed=seed,
    )
    elapsed = time.perf_counter() - t0
    meta = {
        "num_nodes": int(n_nodes),
        "num_edges": int(ei.shape[1]),
        "k_samples": k_samples if k_samples is not None else int(n_nodes),
        "normalized": bool(normalized),
        "wall_seconds": float(elapsed),
        "bc_min": float(table["bc"].min()),
        "bc_max": float(table["bc"].max()),
        "bc_mean": float(table["bc"].mean()),
    }
    logging.info(
        "Tier 2 BC: nodes=%d edges=%d k_samples=%s wall=%.1fs bc_max=%.6f",
        meta["num_nodes"],
        meta["num_edges"],
        meta["k_samples"],
        meta["wall_seconds"],
        meta["bc_max"],
    )
    return table, meta


def tier2_context_from_graph(
    split_data,
    device: torch.device,
    *,
    k_samples: Optional[int] = 256,
    normalized: bool = True,
    seed: int = 1,
) -> MorphTier2Context:
    """Build Tier 2 BC lookup from a split graph (slow on large splits; prefer cache)."""
    ei = get_forward_edge_index(split_data)
    n_nodes = get_num_nodes(split_data)
    table = compute_tier2_node_stats(
        ei,
        n_nodes,
        k_samples=k_samples,
        normalized=normalized,
        seed=seed,
    )
    return _context_from_table(ei.to(device), table, device)


def morph_targets_includes_tier2(targets: str) -> bool:
    """True when expert targets include Tier 2 BC lift (``local+tier2`` or ``local+global+tier2``)."""
    t = str(targets).lower()
    return t in ("local+tier2", "local+global+tier2")


def setup_morph_tier2_contexts(args, tr_data, val_data, device: torch.device) -> None:
    """
    Attach ``args.morph_tier2_train`` and ``args.morph_tier2_val`` for M3 expert targets.

    Uses ``--morph_tier2_cache`` when set; otherwise computes BC from split graphs
    at startup (slow on Small-HI — precompute with ``scripts/precompute_morphology_tier2.py``).
    """
    targets = str(getattr(args, "morph_targets", "local")).lower()
    if not morph_targets_includes_tier2(targets):
        return

    cache_dir = getattr(args, "morph_tier2_cache", None)
    k_samples = getattr(args, "morph_tier2_bc_samples", None)
    if k_samples is not None:
        k_samples = int(k_samples)
    bc_exact = bool(getattr(args, "morph_tier2_bc_exact", False))
    if bc_exact:
        k_samples = None

    if cache_dir:
        cache = Path(cache_dir)
        train_path = cache / "train_node_tier2.csv"
        val_path = cache / "val_node_tier2.csv"
        logging.info("Loading Tier 2 BC from cache dir %s", cache)
        args.morph_tier2_train = tier2_context_from_cache(train_path, tr_data, device)
        args.morph_tier2_val = tier2_context_from_cache(val_path, val_data, device)
    else:
        logging.warning(
            "Computing Tier 2 BC from split graphs at startup (set --morph_tier2_cache to reuse tables)"
        )
        seed = int(getattr(args, "seed", 1))
        args.morph_tier2_train = tier2_context_from_graph(
            tr_data, device, k_samples=k_samples, seed=seed
        )
        args.morph_tier2_val = tier2_context_from_graph(
            val_data, device, k_samples=k_samples, seed=seed
        )
    logging.info(
        "Tier 2 train: nodes=%d | val: nodes=%d",
        args.morph_tier2_train.num_nodes,
        args.morph_tier2_val.num_nodes,
    )
