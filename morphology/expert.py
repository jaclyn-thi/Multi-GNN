"""Morphology expert head and loss (Phase M1 / M1b).

The expert head is a small MLP that predicts **detached** morphology target vectors
from seed-edge embeddings ``z_seed``. Targets combine:

- Tier 1 local subgraph stats (14 dims: degree/ego + clustering + triangle counts on view1)
- Tier 0 global endpoint lift (M1b, ``--morph_targets local+global``)
- Tier 2 betweenness centrality lift (M3: ``local+tier2`` or ``local+global+tier2``)
- Forward edge-native attributes (default on; disable with ``--no_morph_edge_native``)

Loss is weighted MSE or MAE (``--morph_expert_loss``, default MSE; Papagei uses MAE).
With ``--morph_expert_layout grouped``, separate block MLPs and per-block loss
(``--morph_expert_group_weight_tier2`` for Tier 2).
The head is checkpointed for resume but **not** used during embedding extraction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset_specs import DEFAULT_EDGE_FEATURE_COLS
from morphology.target_registry import MORPH_TARGET_GROUPS, morph_target_group, safe_target_log_name
from morphology.tier0_global import (
    DEFAULT_LIFT_FEATURE_NAMES,
    MorphTier0Context,
    lift_global_to_seed_edges_torch,
    morph_targets_includes_global,
    setup_morph_tier0_contexts,
)
from morphology.tier2_global import (
    MorphTier2Context,
    TIER2_BC_LIFT_FEATURE_NAMES,
    lift_tier2_bc_to_seed_edges_torch,
    morph_targets_includes_tier2,
    setup_morph_tier2_contexts,
    tier2_bc_lift_dim,
    tier2_bc_lift_feature_names,
)
from morphology.tier1_local import (
    LOCAL_FEATURE_NAMES,
    align_seed_embeddings_with_morph,
    compute_local_morphology_torch,
    gather_seed_forward_edge_attr,
    local_count_indices_in_subset,
    local_feature_dim_for_subset,
    local_feature_names_for_subset,
    slice_local_morph_features,
    transform_morph_targets,
)


@dataclass
class MorphExpertConfig:
    """
    Expert head hyperparameters and target composition flags.

    Attributes
    ----------
    embedding_dim :
        Input dim (128-d transaction embedding from GNN readout).
    hidden_dim :
        MLP hidden size (``--morph_expert_hidden``).
    include_edge_native :
        Append forward edge_attr columns to targets.
    include_global :
        Append Tier 0 endpoint lift (9 dims by default).
    include_tier2 :
        Append Tier 2 BC endpoint lift after global block (width set by ``tier2_lift_mode``).
    tier2_lift_mode :
        ``full`` (4 BC lift cols) or ``max`` (``bc_max_global`` only).
    loss_type :
        ``mse`` or ``mae`` (``--morph_expert_loss``).
    loss_weight :
        Scale on expert regression loss relative to InfoNCE (``--morph_expert_weight``).
    layout :
        ``shared`` = single MLP; ``grouped`` = one MLP per target block (M5a).
    edge_attr_dim :
        Forward edge_attr width (excl. EdgeID); used for block slicing when grouped.
    group_weight_tier2 :
        Per-block MSE scale for Tier 2 when ``layout=grouped`` (sweep for M1b+BC test).
    local_subset :
        Tier-1 expert columns: ``all`` (14), ``clustering`` (11, no triangles),
        ``triangles`` (11, no clustering), or ``degree`` (8).
    target_groups :
        Optional semantic target groups to keep. Empty means all groups.
    """

    embedding_dim: int = 128
    hidden_dim: int = 64
    include_edge_native: bool = True
    include_global: bool = False
    include_tier2: bool = False
    tier2_lift_mode: str = "full"
    loss_weight: float = 1.0
    loss_type: str = "mse"
    layout: str = "shared"
    edge_attr_dim: int = 0
    group_weight_local: float = 1.0
    group_weight_global: float = 1.0
    group_weight_tier2: float = 1.0
    group_weight_edge_native: float = 1.0
    local_subset: str = "all"
    target_groups: Tuple[str, ...] = ()


class MorphologyExpertHead(nn.Module):
    """Predict morphology targets from seed-edge embeddings (model-agnostic)."""

    def __init__(self, embedding_dim: int, target_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.target_dim = int(target_dim)
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.target_dim),
        )

    def forward(self, z_seed: torch.Tensor) -> torch.Tensor:
        return self.net(z_seed)


# Target block order matches ``transform_morph_targets`` concatenation.
MORPH_TARGET_BLOCK_ORDER: Tuple[str, ...] = ("local", "global", "tier2", "edge_native")


class MorphologyGroupedExpertHead(nn.Module):
    """One small MLP per morphology block (local / global / tier2 / edge_native)."""

    def __init__(
        self,
        embedding_dim: int,
        block_dims: Dict[str, int],
        hidden_dim: int = 64,
        block_order: Tuple[str, ...] = MORPH_TARGET_BLOCK_ORDER,
    ):
        super().__init__()
        self.block_order = tuple(name for name in block_order if name in block_dims)
        if not self.block_order:
            raise ValueError("block_dims must include at least one morphology block")
        self.block_dims = {name: int(block_dims[name]) for name in self.block_order}
        self.target_dim = sum(self.block_dims.values())
        self.heads = nn.ModuleDict(
            {
                name: MorphologyExpertHead(embedding_dim, dim, hidden_dim=hidden_dim)
                for name, dim in self.block_dims.items()
            }
        )

    def forward(self, z_seed: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.heads[name](z_seed) for name in self.block_order], dim=1)


MorphExpertHead = Union[MorphologyExpertHead, MorphologyGroupedExpertHead]


def morph_target_blocks(cfg: MorphExpertConfig) -> List[Tuple[str, slice]]:
    """Return ``(block_name, column_slice)`` pairs in expert target column order."""
    start = 0
    blocks: List[Tuple[str, slice]] = []
    n_local = local_feature_dim_for_subset(cfg.local_subset)
    blocks.append(("local", slice(start, start + n_local)))
    start += n_local
    if cfg.include_global:
        n = len(DEFAULT_LIFT_FEATURE_NAMES)
        blocks.append(("global", slice(start, start + n)))
        start += n
    if cfg.include_tier2:
        n = tier2_bc_lift_dim(cfg.tier2_lift_mode)
        blocks.append(("tier2", slice(start, start + n)))
        start += n
    if cfg.include_edge_native:
        n = int(cfg.edge_attr_dim)
        blocks.append(("edge_native", slice(start, start + n)))
    return blocks


def _edge_native_target_names(edge_attr_dim: int) -> List[str]:
    base = list(DEFAULT_EDGE_FEATURE_COLS)
    n = int(edge_attr_dim)
    if n <= len(base):
        return base[:n]
    return base + [f"edge_native_{i}" for i in range(len(base), n)]


def _all_morph_target_names(cfg: MorphExpertConfig) -> List[str]:
    """Return unfiltered scalar morphology target names in target column order."""

    names = list(local_feature_names_for_subset(cfg.local_subset))
    if cfg.include_global:
        names.extend(DEFAULT_LIFT_FEATURE_NAMES)
    if cfg.include_tier2:
        names.extend(tier2_bc_lift_feature_names(cfg.tier2_lift_mode))
    if cfg.include_edge_native:
        names.extend(_edge_native_target_names(cfg.edge_attr_dim))
    return names


def morph_target_indices(cfg: MorphExpertConfig) -> Tuple[int, ...]:
    """Return selected target column indices after semantic group filtering."""

    names = _all_morph_target_names(cfg)
    groups = tuple(getattr(cfg, "target_groups", ()) or ())
    if not groups:
        return tuple(range(len(names)))
    selected = set(groups)
    return tuple(i for i, name in enumerate(names) if morph_target_group(name) in selected)


def morph_target_names(cfg: MorphExpertConfig) -> List[str]:
    """Return selected scalar morphology target names in expert target column order."""

    names = _all_morph_target_names(cfg)
    idx = morph_target_indices(cfg)
    return [names[i] for i in idx]


def _group_weight_for_block(block_name: str, cfg: MorphExpertConfig) -> float:
    return {
        "local": float(cfg.group_weight_local),
        "global": float(cfg.group_weight_global),
        "tier2": float(cfg.group_weight_tier2),
        "edge_native": float(cfg.group_weight_edge_native),
    }.get(block_name, 1.0)


VALID_MORPH_EXPERT_LOSS_TYPES = frozenset({"mse", "mae"})


def _morph_expert_elementwise_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_type: str,
) -> torch.Tensor:
    loss_type = str(loss_type).lower()
    if loss_type == "mse":
        return F.mse_loss(pred, target)
    if loss_type == "mae":
        return F.l1_loss(pred, target)
    raise ValueError(
        f"morph expert loss_type {loss_type!r} is invalid; use 'mse' or 'mae'."
    )


def morph_expert_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Unweighted MSE; returns zero scalar when ``pred`` is empty."""
    if pred.numel() == 0:
        return pred.new_zeros(())
    return _morph_expert_elementwise_loss(pred, target, "mse")


def morph_expert_mae_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Unweighted MAE (L1); returns zero scalar when ``pred`` is empty."""
    if pred.numel() == 0:
        return pred.new_zeros(())
    return _morph_expert_elementwise_loss(pred, target, "mae")


def morph_expert_regression_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_type: str,
) -> torch.Tensor:
    """Unweighted MSE or MAE on the full target vector."""
    if pred.numel() == 0:
        return pred.new_zeros(())
    return _morph_expert_elementwise_loss(pred, target, loss_type)


def morph_expert_mse_diagnostics(
    pred: torch.Tensor,
    target: torch.Tensor,
    cfg: MorphExpertConfig,
) -> Dict[str, Any]:
    """
    Detached per-group and per-target MSE diagnostics.

    This helper does not participate in gradient computation and does not change
    the expert loss. Non-finite target/prediction cells are ignored. Empty groups
    are omitted rather than logged as NaN.
    """

    out: Dict[str, Any] = {
        "total_sum": pred.new_zeros(()).detach(),
        "total_count": 0,
        "groups": {},
        "targets": {},
    }
    if pred.numel() == 0 or target.numel() == 0:
        return out
    if pred.shape != target.shape:
        raise ValueError("pred and target must have matching shapes for morphology diagnostics")

    names = morph_target_names(cfg)
    if len(names) < pred.shape[1]:
        names = names + [f"target_{i}" for i in range(len(names), pred.shape[1])]
    elif len(names) > pred.shape[1]:
        names = names[: pred.shape[1]]

    with torch.no_grad():
        finite = torch.isfinite(pred) & torch.isfinite(target)
        sq_err = (pred.detach() - target.detach()).pow(2)
        total_mask = finite
        total_count = int(total_mask.sum().item())
        if total_count > 0:
            out["total_sum"] = sq_err[total_mask].sum().detach()
            out["total_count"] = total_count

        for col_idx, target_name in enumerate(names):
            col_mask = finite[:, col_idx]
            count = int(col_mask.sum().item())
            if count == 0:
                continue
            loss_sum = sq_err[:, col_idx][col_mask].sum().detach()
            safe_name = safe_target_log_name(target_name)
            out["targets"][safe_name] = {
                "sum": loss_sum,
                "count": count,
                "raw_name": target_name,
            }
            group_name = morph_target_group(target_name)
            group_entry = out["groups"].setdefault(
                group_name,
                {"sum": pred.new_zeros(()).detach(), "count": 0},
            )
            group_entry["sum"] = group_entry["sum"] + loss_sum
            group_entry["count"] += count
    return out


def update_morph_expert_diagnostics_accumulator(
    accumulator: Dict[str, Any],
    pred: torch.Tensor,
    target: torch.Tensor,
    cfg: MorphExpertConfig,
) -> None:
    """Accumulate detached morphology MSE diagnostics into ``accumulator``."""

    diag = morph_expert_mse_diagnostics(pred, target, cfg)
    accumulator["total_sum"] = accumulator.get("total_sum", pred.new_zeros(()).detach()) + diag["total_sum"]
    accumulator["total_count"] = int(accumulator.get("total_count", 0)) + int(diag["total_count"])
    groups = accumulator.setdefault("groups", {})
    for group_name, values in diag["groups"].items():
        entry = groups.setdefault(
            group_name,
            {"sum": pred.new_zeros(()).detach(), "count": 0},
        )
        entry["sum"] = entry["sum"] + values["sum"]
        entry["count"] += int(values["count"])
    targets = accumulator.setdefault("targets", {})
    for target_name, values in diag["targets"].items():
        entry = targets.setdefault(
            target_name,
            {
                "sum": pred.new_zeros(()).detach(),
                "count": 0,
                "raw_name": values.get("raw_name", target_name),
            },
        )
        entry["sum"] = entry["sum"] + values["sum"]
        entry["count"] += int(values["count"])


def finalize_morph_expert_diagnostics(
    accumulator: Optional[Dict[str, Any]],
) -> Dict[str, float]:
    """Convert accumulated diagnostics into W&B/log-payload metric keys."""

    if not accumulator:
        return {}
    metrics: Dict[str, float] = {}
    total_count = int(accumulator.get("total_count", 0))
    if total_count > 0:
        metrics["morphology/loss_mse_total"] = float(
            (accumulator["total_sum"] / total_count).detach().cpu()
        )
    for group_name, values in sorted(accumulator.get("groups", {}).items()):
        count = int(values.get("count", 0))
        if count <= 0:
            continue
        metrics[f"morphology/loss_group/{group_name}"] = float(
            (values["sum"] / count).detach().cpu()
        )
    for target_name, values in sorted(accumulator.get("targets", {}).items()):
        count = int(values.get("count", 0))
        if count <= 0:
            continue
        metrics[f"morphology/loss_target/{target_name}"] = float(
            (values["sum"] / count).detach().cpu()
        )
    return metrics


def morph_expert_grouped_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    cfg: MorphExpertConfig,
) -> torch.Tensor:
    """Sum of per-block MSE terms with optional block weights (grouped layout)."""
    return morph_expert_grouped_regression_loss(pred, target, cfg, "mse")


def morph_expert_grouped_mae_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    cfg: MorphExpertConfig,
) -> torch.Tensor:
    """Sum of per-block MAE terms with optional block weights (grouped layout)."""
    return morph_expert_grouped_regression_loss(pred, target, cfg, "mae")


def morph_expert_grouped_regression_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    cfg: MorphExpertConfig,
    loss_type: str,
) -> torch.Tensor:
    """Sum of per-block MSE/MAE terms with optional block weights (grouped layout)."""
    if pred.numel() == 0:
        return pred.new_zeros(())
    loss = pred.new_zeros(())
    for block_name, sl in morph_target_blocks(cfg):
        w = _group_weight_for_block(block_name, cfg)
        if w == 0.0:
            continue
        block_loss = _morph_expert_elementwise_loss(pred[:, sl], target[:, sl], loss_type)
        loss = loss + w * block_loss
    return loss


def build_morph_targets(
    edge_index: torch.Tensor,
    subgraph_edge_ids: torch.Tensor,
    seed_edge_ids: torch.Tensor,
    seed_edge_attr: Optional[torch.Tensor],
    num_nodes: int,
    cfg: MorphExpertConfig,
    tier0_ctx: Optional[MorphTier0Context] = None,
    tier2_ctx: Optional[MorphTier2Context] = None,
) -> torch.Tensor:
    """
    Assemble detached morphology target matrix for seeds in the view1 subgraph.

    Returns
    -------
    Tensor
        Float tensor ``(n_seeds, target_dim)`` with log1p applied to count-like
        columns via ``transform_morph_targets``.
    """
    local_full = compute_local_morphology_torch(
        edge_index,
        subgraph_edge_ids,
        seed_edge_ids,
        num_nodes,
        device=edge_index.device,
    )
    local = slice_local_morph_features(local_full, cfg.local_subset)
    global_feats = None
    if cfg.include_global:
        if tier0_ctx is None:
            raise ValueError("include_global=True requires tier0_ctx")
        global_feats = lift_global_to_seed_edges_torch(seed_edge_ids, tier0_ctx)
    tier2_feats = None
    if cfg.include_tier2:
        if tier2_ctx is None:
            raise ValueError("include_tier2=True requires tier2_ctx")
        tier2_feats = lift_tier2_bc_to_seed_edges_torch(
            seed_edge_ids, tier2_ctx, lift_mode=cfg.tier2_lift_mode
        )
    edge_native = None
    if cfg.include_edge_native and seed_edge_attr is not None and seed_edge_attr.numel() > 0:
        edge_native = gather_seed_forward_edge_attr(
            seed_edge_attr, subgraph_edge_ids, seed_edge_ids
        )
    targets = transform_morph_targets(
        local,
        edge_native=edge_native,
        global_feats=global_feats,
        tier2_feats=tier2_feats,
        local_count_indices=local_count_indices_in_subset(cfg.local_subset),
    ).detach()
    idx = morph_target_indices(cfg)
    if len(idx) == targets.shape[1]:
        return targets
    if not idx:
        raise ValueError(
            "morphology target group filter selected zero columns; "
            f"requested groups={getattr(cfg, 'target_groups', ())!r}"
        )
    return targets[:, list(idx)]


def morphology_expert_step(
    z_seed: torch.Tensor,
    seed_edge_ids: torch.Tensor,
    edge_index: torch.Tensor,
    subgraph_edge_ids: torch.Tensor,
    edge_attr: Optional[torch.Tensor],
    num_nodes: int,
    expert_head: MorphExpertHead,
    cfg: MorphExpertConfig,
    tier0_ctx: Optional[MorphTier0Context] = None,
    tier2_ctx: Optional[MorphTier2Context] = None,
    diagnostics_accumulator: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute morphology expert loss for one batch of shared seeds.

    Returns ``(loss, targets)`` where ``loss`` is scaled by ``cfg.loss_weight``.
    """
    z_use, ids_use = align_seed_embeddings_with_morph(
        z_seed, seed_edge_ids, subgraph_edge_ids
    )
    if z_use.numel() == 0:
        return z_seed.new_zeros(()), z_seed.new_empty((0, expert_head.target_dim))

    targets = build_morph_targets(
        edge_index,
        subgraph_edge_ids,
        ids_use,
        edge_attr,
        num_nodes,
        cfg,
        tier0_ctx=tier0_ctx,
        tier2_ctx=tier2_ctx,
    )
    if z_use.shape[0] != targets.shape[0]:
        min_n = min(z_use.shape[0], targets.shape[0])
        z_use = z_use[:min_n]
        targets = targets[:min_n]

    pred = expert_head(z_use)
    if diagnostics_accumulator is not None:
        update_morph_expert_diagnostics_accumulator(
            diagnostics_accumulator,
            pred,
            targets,
            cfg,
        )
    loss_type = str(cfg.loss_type).lower()
    if cfg.layout == "grouped":
        loss = morph_expert_grouped_regression_loss(pred, targets, cfg, loss_type)
    else:
        loss = morph_expert_regression_loss(pred, targets, loss_type)
    loss = loss * float(cfg.loss_weight)
    return loss, targets


def target_dim_for_config(
    edge_attr_dim: int,
    cfg: MorphExpertConfig,
) -> int:
    cfg.edge_attr_dim = int(edge_attr_dim)
    return len(morph_target_names(cfg))


def _block_dims_for_config(edge_attr_dim: int, cfg: MorphExpertConfig) -> Dict[str, int]:
    blocks: Dict[str, int] = {"local": local_feature_dim_for_subset(cfg.local_subset)}
    if cfg.include_global:
        blocks["global"] = len(DEFAULT_LIFT_FEATURE_NAMES)
    if cfg.include_tier2:
        blocks["tier2"] = tier2_bc_lift_dim(cfg.tier2_lift_mode)
    if cfg.include_edge_native:
        blocks["edge_native"] = int(edge_attr_dim)
    return blocks


def create_morph_expert_bundle(
    edge_attr_dim: int,
    cfg: MorphExpertConfig,
    device: torch.device,
) -> Tuple[MorphExpertHead, MorphExpertConfig]:
    cfg.edge_attr_dim = int(edge_attr_dim)
    if cfg.layout == "grouped":
        head = MorphologyGroupedExpertHead(
            cfg.embedding_dim,
            _block_dims_for_config(edge_attr_dim, cfg),
            hidden_dim=cfg.hidden_dim,
        ).to(device)
    else:
        dim = target_dim_for_config(edge_attr_dim, cfg)
        head = MorphologyExpertHead(cfg.embedding_dim, dim, hidden_dim=cfg.hidden_dim).to(device)
    return head, cfg


def _morph_targets_mode(args) -> str:
    return str(getattr(args, "morph_targets", "local")).lower()


VALID_MORPH_TARGETS = frozenset({"local", "local+global", "local+tier2", "local+global+tier2"})


def _parse_morph_target_groups(raw: str) -> Tuple[str, ...]:
    value = str(raw or "all").strip().lower()
    if value in ("", "all"):
        return ()
    groups = tuple(part.strip() for part in value.split(",") if part.strip())
    invalid = sorted(set(groups) - set(MORPH_TARGET_GROUPS))
    if invalid:
        raise ValueError(
            f"--morph_target_groups contains invalid groups {invalid}; "
            f"use 'all' or a comma-separated subset of {list(MORPH_TARGET_GROUPS)}."
        )
    return groups


def setup_morphology_expert(args, tr_data, device, is_hetero: bool):
    """
    Build expert head + config when ``--morph_expert`` is set (contrastive M1 / M1b).

    Parameters
    ----------
    args :
        CLI namespace; reads ``morph_targets``, ``morph_expert_weight``, etc.
    tr_data :
        Training split graph (for edge_attr dim inference).
    device :
        Torch device for the head.
    is_hetero :
        Whether ``tr_data`` is ``HeteroData``.

    Returns
    -------
    tuple
        ``(MorphologyExpertHead, MorphExpertConfig)`` or ``(None, None)`` if disabled.
    """
    if not getattr(args, "morph_expert", False):
        return None, None
    targets = _morph_targets_mode(args)
    if targets not in VALID_MORPH_TARGETS:
        raise ValueError(
            f"--morph_targets {targets!r} is invalid; "
            f"use one of {sorted(VALID_MORPH_TARGETS)}."
        )
    from morphology.graph_access import get_forward_edge_attr

    if is_hetero:
        attr = get_forward_edge_attr(tr_data)
    else:
        attr = tr_data.edge_attr
    if attr is None:
        raise ValueError("Cannot infer edge_attr dim for morphology expert (missing edge_attr).")
    edge_attr_dim = int(attr.shape[1]) - 1  # column 0 is synthetic EdgeID from add_arange_ids

    layout = str(getattr(args, "morph_expert_layout", "shared")).lower()
    if layout not in ("shared", "grouped"):
        raise ValueError(
            f"--morph_expert_layout {layout!r} is invalid; use 'shared' or 'grouped'."
        )
    target_groups = _parse_morph_target_groups(getattr(args, "morph_target_groups", "all"))
    if target_groups and layout == "grouped":
        raise ValueError(
            "--morph_target_groups currently supports the shared expert head only; "
            "use --morph_expert_layout shared."
        )
    loss_type = str(getattr(args, "morph_expert_loss", "mse")).lower()
    if loss_type not in VALID_MORPH_EXPERT_LOSS_TYPES:
        raise ValueError(
            f"--morph_expert_loss {loss_type!r} is invalid; use 'mse' or 'mae'."
        )
    local_subset = str(getattr(args, "morph_local_subset", "all")).lower()
    from morphology.tier1_local import VALID_MORPH_LOCAL_SUBSETS

    if local_subset not in VALID_MORPH_LOCAL_SUBSETS:
        raise ValueError(
            f"--morph_local_subset {local_subset!r} invalid; "
            f"use one of {sorted(VALID_MORPH_LOCAL_SUBSETS)}."
        )
    cfg = MorphExpertConfig(
        embedding_dim=128,
        hidden_dim=int(getattr(args, "morph_expert_hidden", 64)),
        include_edge_native=not bool(getattr(args, "no_morph_edge_native", False)),
        include_global=morph_targets_includes_global(targets),
        include_tier2=morph_targets_includes_tier2(targets),
        tier2_lift_mode=str(getattr(args, "morph_tier2_lift", "full")).lower(),
        loss_weight=float(getattr(args, "morph_expert_weight", 1.0)),
        loss_type=loss_type,
        layout=layout,
        group_weight_tier2=float(getattr(args, "morph_expert_group_weight_tier2", 1.0)),
        local_subset=local_subset,
        target_groups=target_groups,
    )
    head, cfg = create_morph_expert_bundle(edge_attr_dim, cfg, device)
    extra = ""
    if cfg.include_global:
        extra += f" + global={len(DEFAULT_LIFT_FEATURE_NAMES)}"
    if cfg.include_tier2:
        extra += f" + tier2_bc={tier2_bc_lift_dim(cfg.tier2_lift_mode)}({cfg.tier2_lift_mode})"
    if cfg.include_edge_native:
        extra += f" + edge_native={edge_attr_dim}"
    layout_extra = ""
    if cfg.layout == "grouped":
        layout_extra = f" layout=grouped w_tier2={cfg.group_weight_tier2:g}"
    group_extra = ""
    if cfg.target_groups:
        group_extra = f" target_groups={','.join(cfg.target_groups)}"
    logging.info(
        "Morphology expert head: target_dim=%d (local=%d subset=%s%s) loss=%s weight=%.4f morph_targets=%s%s%s",
        head.target_dim,
        local_feature_dim_for_subset(cfg.local_subset),
        cfg.local_subset,
        extra,
        cfg.loss_type,
        cfg.loss_weight,
        targets,
        layout_extra,
        group_extra,
    )
    return head, cfg


__all__ = [
    "MORPH_TARGET_BLOCK_ORDER",
    "MorphExpertConfig",
    "MorphExpertHead",
    "MorphologyExpertHead",
    "MorphologyGroupedExpertHead",
    "build_morph_targets",
    "create_morph_expert_bundle",
    "finalize_morph_expert_diagnostics",
    "morph_expert_grouped_mae_loss",
    "morph_expert_grouped_mse_loss",
    "morph_expert_grouped_regression_loss",
    "morph_expert_mae_loss",
    "morph_expert_mse_diagnostics",
    "morph_expert_mse_loss",
    "morph_expert_regression_loss",
    "morph_target_blocks",
    "morph_target_names",
    "morphology_expert_step",
    "setup_morphology_expert",
    "setup_morph_tier0_contexts",
    "setup_morph_tier2_contexts",
    "target_dim_for_config",
    "update_morph_expert_diagnostics_accumulator",
]
