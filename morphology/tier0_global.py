"""Tier 0: split-global node statistics and endpoint lift to seed edges."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch

from morphology.graph_access import get_forward_edge_index, get_num_nodes, seed_endpoints_from_edge_ids

# Default columns stored per node in offline tables.
TIER0_NODE_COLUMNS = ("deg_in", "deg_out", "deg_total")

# Default lifted features per seed edge (global endpoint stats).
DEFAULT_LIFT_FEATURE_NAMES: List[str] = [
    "sender_deg_in",
    "sender_deg_out",
    "sender_deg_total",
    "receiver_deg_in",
    "receiver_deg_out",
    "receiver_deg_total",
    "deg_sum_out_global",
    "deg_sum_in_global",
    "deg_sum_total_global",
]

GLOBAL_LIFT_FEATURE_NAMES = DEFAULT_LIFT_FEATURE_NAMES

# All global lift features are count-like (log1p before MSE).
GLOBAL_COUNT_FEATURE_INDICES: tuple[int, ...] = tuple(range(len(DEFAULT_LIFT_FEATURE_NAMES)))


@dataclass
class MorphTier0Context:
    """Split-global Tier 0 data for endpoint lift at training time."""

    edge_index: torch.Tensor
    deg_in: torch.Tensor
    deg_out: torch.Tensor
    deg_total: torch.Tensor

    @property
    def num_nodes(self) -> int:
        return int(self.deg_in.shape[0])

    @property
    def device(self) -> torch.device:
        return self.edge_index.device


def compute_tier0_node_stats(
    edge_index: torch.Tensor,
    num_nodes: int,
    timestamps: Optional[torch.Tensor] = None,
) -> pd.DataFrame:
    """
  Compute split-global in/out degree per node on a **forward-only** graph.

  Parameters
  ----------
  edge_index :
      ``[2, E]`` with row 0 = sender, row 1 = receiver.
  num_nodes :
      Number of nodes (holdings) in this split graph.
  timestamps :
      Optional per-edge times (not used in v0; reserved for time-since-previous).

  Returns
  -------
  DataFrame indexed by ``node_id`` with columns ``deg_in``, ``deg_out``, ``deg_total``.
  """
    del timestamps  # reserved for future Tier 0 time features
    if edge_index.numel() == 0:
        df = pd.DataFrame(
            {
                "deg_in": np.zeros(num_nodes, dtype=np.int64),
                "deg_out": np.zeros(num_nodes, dtype=np.int64),
            }
        )
        df.index.name = "node_id"
        df["deg_total"] = df["deg_in"] + df["deg_out"]
        return df

    ei = edge_index.cpu()
    e = ei.shape[1]
    ones = torch.ones(e, dtype=torch.long)
    deg_out = torch.zeros(num_nodes, dtype=torch.long)
    deg_in = torch.zeros(num_nodes, dtype=torch.long)
    deg_out.scatter_add_(0, ei[0], ones)
    deg_in.scatter_add_(0, ei[1], ones)

    df = pd.DataFrame(
        {
            "deg_in": deg_in.numpy(),
            "deg_out": deg_out.numpy(),
        },
        index=np.arange(num_nodes, dtype=np.int64),
    )
    df.index.name = "node_id"
    df["deg_total"] = df["deg_in"] + df["deg_out"]
    return df


def lift_node_to_seed_edges(
    seed_edge_ids: Union[torch.Tensor, np.ndarray, Sequence[int]],
    edge_index: torch.Tensor,
    node_table: pd.DataFrame,
    node_columns: Optional[Sequence[str]] = None,
    feature_names: Optional[List[str]] = None,
) -> np.ndarray:
    """
  Lift per-node Tier 0 stats to seed transactions via sender/receiver endpoints.

  Parameters
  ----------
  seed_edge_ids :
      ``EdgeID`` values (must index columns of ``edge_index`` on this graph).
  edge_index :
      Forward ``[2, E]`` for the **same** split graph used to build ``node_table``.
  node_table :
      Indexed by ``node_id``; must contain ``node_columns``.
  node_columns :
      Node-level columns to lift (default: deg_in, deg_out, deg_total).
  feature_names :
      If provided, must match output width; otherwise ``DEFAULT_LIFT_FEATURE_NAMES``.

  Returns
  -------
  float32 array of shape ``(n_seeds, d_morph)``.
  """
    node_columns = list(node_columns or TIER0_NODE_COLUMNS)
    for col in node_columns:
        if col not in node_table.columns:
            raise KeyError(f"node_table missing column {col!r}")

    eid = torch.as_tensor(seed_edge_ids, dtype=torch.long).view(-1)
    senders, receivers = seed_endpoints_from_edge_ids(eid, edge_index)
    senders_np = senders.numpy()
    receivers_np = receivers.numpy()

    s = node_table.loc[senders_np, node_columns].to_numpy(dtype=np.float32)
    t = node_table.loc[receivers_np, node_columns].to_numpy(dtype=np.float32)

    # Derived globals (order matches node_columns: in, out, total)
    if len(node_columns) >= 2:
        s_out = s[:, 1]
        t_out = t[:, 1]
        s_in = s[:, 0]
        t_in = t[:, 0]
        s_tot = s[:, 2] if len(node_columns) > 2 else s_in + s_out
        t_tot = t[:, 2] if len(node_columns) > 2 else t_in + t_out
    else:
        raise ValueError("node_columns must include at least deg_in and deg_out")

    derived = np.stack(
        [
            s_out + t_out,
            s_in + t_in,
            s_tot + t_tot,
        ],
        axis=1,
    )
    out = np.concatenate([s, t, derived], axis=1).astype(np.float32)

    expected_names = feature_names or DEFAULT_LIFT_FEATURE_NAMES
    expected_dim = 2 * len(node_columns) + 3
    if out.shape[1] != expected_dim:
        raise RuntimeError(f"lift dim {out.shape[1]} != expected {expected_dim}")
    if len(expected_names) != expected_dim:
        raise ValueError(
            f"feature_names length {len(expected_names)} != output dim {expected_dim}"
        )
    return out


def _node_table_to_context(
    edge_index: torch.Tensor,
    node_table: pd.DataFrame,
    device: torch.device,
) -> MorphTier0Context:
    cols = list(TIER0_NODE_COLUMNS)
    for col in cols:
        if col not in node_table.columns:
            raise KeyError(f"node_table missing column {col!r}")
    deg_in = torch.as_tensor(node_table["deg_in"].to_numpy(), device=device, dtype=torch.float32)
    deg_out = torch.as_tensor(node_table["deg_out"].to_numpy(), device=device, dtype=torch.float32)
    deg_total = torch.as_tensor(node_table["deg_total"].to_numpy(), device=device, dtype=torch.float32)
    return MorphTier0Context(
        edge_index=edge_index,
        deg_in=deg_in,
        deg_out=deg_out,
        deg_total=deg_total,
    )


def tier0_context_from_graph(split_data, device: torch.device) -> MorphTier0Context:
    """Build Tier 0 lookup tensors from a split graph (train or val only for targets)."""
    ei = get_forward_edge_index(split_data)
    n_nodes = get_num_nodes(split_data)
    table = compute_tier0_node_stats(ei, n_nodes)
    return _node_table_to_context(ei.to(device), table, device)


def tier0_context_from_cache(
    path: Union[str, Path],
    split_data,
    device: torch.device,
) -> MorphTier0Context:
    """Load a precomputed node table and pair it with the split forward ``edge_index``."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Tier 0 cache not found: {path}")
    table = load_node_table(path)
    n_nodes = get_num_nodes(split_data)
    if len(table) != n_nodes:
        raise ValueError(
            f"Tier 0 table rows ({len(table)}) != split num_nodes ({n_nodes}) for {path}"
        )
    ei = get_forward_edge_index(split_data).to(device)
    return _node_table_to_context(ei, table, device)


def lift_global_to_seed_edges_torch(
    seed_edge_ids: torch.Tensor,
    tier0_ctx: MorphTier0Context,
) -> torch.Tensor:
    """
  Lift split-global endpoint degrees to seed edges (torch, on ``tier0_ctx.device``).

  Returns float32 tensor ``(n_seeds, len(DEFAULT_LIFT_FEATURE_NAMES))``.
  """
    device = tier0_ctx.device
    seed = seed_edge_ids.long().view(-1).to(device)
    ei = tier0_ctx.edge_index
    if seed.numel() == 0:
        return torch.empty((0, len(DEFAULT_LIFT_FEATURE_NAMES)), device=device, dtype=torch.float32)
    if seed.max().item() >= ei.shape[1] or seed.min().item() < 0:
        raise ValueError(
            f"seed edge_ids out of range for split edge_index E={ei.shape[1]}: "
            f"min={seed.min().item()} max={seed.max().item()}"
        )
    senders = ei[0, seed]
    receivers = ei[1, seed]
    s_in = tier0_ctx.deg_in[senders]
    s_out = tier0_ctx.deg_out[senders]
    s_tot = tier0_ctx.deg_total[senders]
    r_in = tier0_ctx.deg_in[receivers]
    r_out = tier0_ctx.deg_out[receivers]
    r_tot = tier0_ctx.deg_total[receivers]
    return torch.stack(
        [
            s_in,
            s_out,
            s_tot,
            r_in,
            r_out,
            r_tot,
            s_out + r_out,
            s_in + r_in,
            s_tot + r_tot,
        ],
        dim=1,
    )


def setup_morph_tier0_contexts(args, tr_data, val_data, device: torch.device) -> None:
    """
  Attach ``args.morph_tier0_train`` and ``args.morph_tier0_val`` when ``--morph_targets local+global``.

  Uses ``--morph_tier0_cache`` if set; otherwise computes from split graphs at startup.
  """
    targets = str(getattr(args, "morph_targets", "local")).lower()
    if targets != "local+global":
        return
    cache_dir = getattr(args, "morph_tier0_cache", None)
    if cache_dir:
        cache = Path(cache_dir)
        train_path = cache / "train_node_morphology.csv"
        val_path = cache / "val_node_morphology.csv"
        logging.info("Loading Tier 0 morphology from cache dir %s", cache)
        args.morph_tier0_train = tier0_context_from_cache(train_path, tr_data, device)
        args.morph_tier0_val = tier0_context_from_cache(val_path, val_data, device)
    else:
        logging.info(
            "Computing Tier 0 morphology from split graphs (set --morph_tier0_cache to reuse tables)"
        )
        args.morph_tier0_train = tier0_context_from_graph(tr_data, device)
        args.morph_tier0_val = tier0_context_from_graph(val_data, device)
    logging.info(
        "Tier 0 train: nodes=%d edges=%d | val: nodes=%d edges=%d",
        args.morph_tier0_train.num_nodes,
        int(args.morph_tier0_train.edge_index.shape[1]),
        args.morph_tier0_val.num_nodes,
        int(args.morph_tier0_val.edge_index.shape[1]),
    )


def get_default_lift_feature_names(
    node_columns: Optional[Sequence[str]] = None,
) -> List[str]:
    """Names for ``lift_node_to_seed_edges`` with default ``node_columns``."""
    node_columns = list(node_columns or TIER0_NODE_COLUMNS)
    if node_columns == list(TIER0_NODE_COLUMNS):
        return list(DEFAULT_LIFT_FEATURE_NAMES)
    names: List[str] = []
    for side in ("sender", "receiver"):
        for col in node_columns:
            names.append(f"{side}_{col}")
    names.extend(
        [
            "deg_sum_out_global",
            "deg_sum_in_global",
            "deg_sum_total_global",
        ]
    )
    return names


def save_node_table(node_table: pd.DataFrame, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        node_table.to_parquet(path)
    else:
        node_table.to_csv(path)


def load_node_table(path: Union[str, Path]) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    df = pd.read_csv(path)
    if "node_id" in df.columns:
        df = df.set_index("node_id")
    return df
