"""
Graph augmentations for contrastive learning on edge-level embeddings.

Supports independent random edge dropping per view (different edge counts /
ordering). Identity contrastive pairs are matched via ``edge_id``, not row index.

When ``edge_drop_policy`` is ``degree_aware`` or ``degree_flow_aware``, per-edge
drop probabilities come from a precomputed train-split cache (see
``edge_drop_scores``). Default ``random`` preserves the legacy uniform policy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional

import torch
from torch_geometric.data import Data, HeteroData

from train_util import FORWARD_EDGE_TYPE, REVERSE_EDGE_TYPE

if TYPE_CHECKING:
    from edge_drop_scores import EdgeDropScoreCache


def _edge_keep_mask(num_edges: int, drop_rate: float, device: torch.device) -> torch.Tensor:
    """Bernoulli keep mask of shape (E,); at least one edge kept when E > 0."""
    if num_edges == 0:
        return torch.ones(0, dtype=torch.bool, device=device)

    if not (0.0 <= drop_rate <= 1.0):
        raise ValueError(f"drop_rate must be in [0, 1], got {drop_rate}")

    keep = torch.rand(num_edges, device=device) > drop_rate
    if not keep.any():
        keep[torch.randint(0, num_edges, (1,), device=device)] = True
    return keep


def _force_keep_seed_edges(
    keep: torch.Tensor,
    edge_ids: torch.Tensor,
    seed_edge_ids: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Opt-in: force-keep edges whose stable ``edge_id`` is in ``seed_edge_ids``.

    Approximates seed retention by retaining the seed *relation* in the
    message-passing graph (not a separated query-only readout).
    """
    if seed_edge_ids is None or seed_edge_ids.numel() == 0:
        return keep
    seed_ids = seed_edge_ids.detach().to(device=edge_ids.device, dtype=torch.long).view(-1)
    is_seed = torch.isin(edge_ids.long().view(-1), seed_ids)
    return keep | is_seed


def _policy_edge_keep_mask(drop_probs: torch.Tensor) -> torch.Tensor:
    """Bernoulli keep mask from per-edge drop probabilities."""
    num_edges = int(drop_probs.numel())
    if num_edges == 0:
        return torch.ones(0, dtype=torch.bool, device=drop_probs.device)
    drop_probs = drop_probs.clamp(0.0, 1.0)
    keep = torch.rand(num_edges, device=drop_probs.device) > drop_probs
    if not keep.any():
        keep[torch.randint(0, num_edges, (1,), device=drop_probs.device)] = True
    return keep


def _accumulate_edge_drop_stats(
    stats: Optional[Dict[str, float]],
    *,
    edges_before: int,
    keep_mask: torch.Tensor,
    drop_probs: torch.Tensor,
    edge_ids: Optional[torch.Tensor],
    edge_drop_cache: Optional["EdgeDropScoreCache"],
    view_tag: str,
) -> None:
    if stats is None:
        return
    kept = int(keep_mask.sum().item())
    dropped = edges_before - kept
    stats["edges_before"] = stats.get("edges_before", 0.0) + float(edges_before)
    stats[f"edges_kept_{view_tag}"] = stats.get(f"edges_kept_{view_tag}", 0.0) + float(kept)
    stats[f"edges_dropped_{view_tag}"] = stats.get(f"edges_dropped_{view_tag}", 0.0) + float(dropped)
    stats["drop_prob_sum"] = stats.get("drop_prob_sum", 0.0) + float(drop_probs.sum().item())
    stats["drop_prob_count"] = stats.get("drop_prob_count", 0.0) + float(drop_probs.numel())
    stats["drop_prob_min"] = min(float(stats.get("drop_prob_min", float("inf"))), float(drop_probs.min().item()))
    stats["drop_prob_max"] = max(float(stats.get("drop_prob_max", float("-inf"))), float(drop_probs.max().item()))

    if edge_ids is None or edge_drop_cache is None:
        return

    ids_cpu = edge_ids.detach().long().cpu().numpy()
    deg_pct = edge_drop_cache.lookup_bucket_values(edge_ids, "degree_pct")
    if deg_pct is not None:
        for lo, hi, label in ((0.0, 0.2, "p0_20"), (0.2, 0.4, "p20_40"), (0.4, 0.6, "p40_60"), (0.6, 0.8, "p60_80"), (0.8, 1.01, "p80_100")):
            m = (deg_pct >= lo) & (deg_pct < hi)
            if m.any():
                drop_rate = float((~keep_mask.detach().cpu().numpy()[m]).mean())
                key = f"drop_rate_degree_{label}_{view_tag}"
                stats[key] = stats.get(key, 0.0) + drop_rate
                cnt_key = f"drop_rate_degree_{label}_{view_tag}_count"
                stats[cnt_key] = stats.get(cnt_key, 0.0) + 1.0

    if edge_drop_cache.amount_pct is not None:
        amt_pct = edge_drop_cache.lookup_bucket_values(edge_ids, "amount_pct")
        if amt_pct is not None:
            for lo, hi, label in ((0.0, 0.2, "p0_20"), (0.2, 0.4, "p20_40"), (0.4, 0.6, "p40_60"), (0.6, 0.8, "p60_80"), (0.8, 1.01, "p80_100")):
                m = (amt_pct >= lo) & (amt_pct < hi)
                if m.any():
                    drop_rate = float((~keep_mask.detach().cpu().numpy()[m]).mean())
                    key = f"drop_rate_amount_{label}_{view_tag}"
                    stats[key] = stats.get(key, 0.0) + drop_rate
                    cnt_key = f"drop_rate_amount_{label}_{view_tag}_count"
                    stats[cnt_key] = stats.get(cnt_key, 0.0) + 1.0

    if edge_drop_cache.flow_imbalance_pct is not None:
        flow_pct = edge_drop_cache.lookup_bucket_values(edge_ids, "flow_imbalance_pct")
        if flow_pct is not None:
            for lo, hi, label in ((0.0, 0.2, "p0_20"), (0.2, 0.4, "p20_40"), (0.4, 0.6, "p40_60"), (0.6, 0.8, "p60_80"), (0.8, 1.01, "p80_100")):
                m = (flow_pct >= lo) & (flow_pct < hi)
                if m.any():
                    drop_rate = float((~keep_mask.detach().cpu().numpy()[m]).mean())
                    key = f"drop_rate_flow_{label}_{view_tag}"
                    stats[key] = stats.get(key, 0.0) + drop_rate
                    cnt_key = f"drop_rate_flow_{label}_{view_tag}_count"
                    stats[cnt_key] = stats.get(cnt_key, 0.0) + 1.0

    del ids_cpu


def random_edge_drop(edge_index, edge_attr, drop_rate):
    """
    Randomly drops edges from a directed multigraph.

    Args:
        edge_index: (2, E)
        edge_attr: (E, D)
        drop_rate: float in [0,1]

    Returns:
        new_edge_index, new_edge_attr
    """
    E = edge_index.size(1)
    if E == 0:
        return edge_index, edge_attr

    keep = _edge_keep_mask(E, drop_rate, edge_index.device)
    return edge_index[:, keep], edge_attr[keep]


def _slice_edge_aligned_fields(data: Data, keep: torch.Tensor) -> None:
    """Apply ``keep`` to edge-aligned tensors on a Data object (mutates ``data``)."""
    E = keep.numel()
    data.edge_index = data.edge_index[:, keep].clone()
    if getattr(data, "edge_attr", None) is not None:
        data.edge_attr = data.edge_attr[keep].clone()

    eid = getattr(data, "edge_id", None)
    if eid is not None:
        if eid.shape[0] != E:
            raise ValueError("edge_id length must match edge_index before masking.")
        data.edge_id = eid[keep].clone()

    y = getattr(data, "y", None)
    if y is not None and y.dim() > 0 and y.shape[0] == E:
        data.y = y[keep].clone()

    ts = getattr(data, "timestamps", None)
    if ts is not None and ts.dim() > 0 and ts.shape[0] == E:
        data.timestamps = ts[keep].clone()


def _slice_hetero_edge_store(store, keep: torch.Tensor) -> None:
    """Apply ``keep`` to one heterogeneous edge store (mutates ``store``)."""
    _slice_edge_aligned_fields(store, keep)


def _random_edge_drop_view(
    data: Data,
    drop_rate: float,
    *,
    seed_edge_ids: Optional[torch.Tensor] = None,
    preserve_seed_edges: bool = False,
) -> Data:
    """
    Independent random edge subset; does not mutate the input ``data``.

    Requires ``edge_id`` when ``drop_rate > 0`` so contrastive pairing stays defined.
    """
    out = data.clone()
    E = out.edge_index.size(1)
    if drop_rate <= 0.0:
        return out

    if getattr(out, "edge_id", None) is None:
        raise ValueError(
            "edge_id is required on the batch when edge_drop_rate > 0; "
            "use train_util.attach_edge_id_from_batch before augmentations."
        )

    keep = _edge_keep_mask(E, drop_rate, out.edge_index.device)
    if preserve_seed_edges:
        keep = _force_keep_seed_edges(keep, out.edge_id, seed_edge_ids)
    _slice_edge_aligned_fields(out, keep)
    return out


def _hetero_random_edge_drop_view(
    batch: HeteroData,
    drop_rate: float,
    forward_et=FORWARD_EDGE_TYPE,
    reverse_et=REVERSE_EDGE_TYPE,
    *,
    seed_edge_ids: Optional[torch.Tensor] = None,
    preserve_seed_edges: bool = False,
) -> HeteroData:
    """
    Independent random edge drop on forward transactions; reverse edges with the same
    transaction ``edge_id`` are dropped together (synchronized augmentation).
    """
    out = batch.clone()
    fwd = out[forward_et]
    E = fwd.edge_index.size(1)
    if drop_rate <= 0.0:
        return out

    if getattr(fwd, "edge_id", None) is None:
        raise ValueError(
            "edge_id is required on forward edges when edge_drop_rate > 0; "
            "use train_util.attach_edge_id_from_batch before augmentations."
        )

    keep_fwd = _edge_keep_mask(E, drop_rate, fwd.edge_index.device)
    if preserve_seed_edges:
        keep_fwd = _force_keep_seed_edges(keep_fwd, fwd.edge_id, seed_edge_ids)
    _slice_hetero_edge_store(fwd, keep_fwd)
    kept_txn = fwd.edge_id

    rev = out[reverse_et]
    if getattr(rev, "edge_id", None) is None:
        raise ValueError("edge_id is required on reverse edges for synchronized hetero augmentation.")
    rev_keep = torch.isin(rev.edge_id, kept_txn)
    _slice_hetero_edge_store(rev, rev_keep)
    return out


def _hetero_policy_edge_drop_view(
    batch: HeteroData,
    edge_drop_cache: "EdgeDropScoreCache",
    forward_et=FORWARD_EDGE_TYPE,
    reverse_et=REVERSE_EDGE_TYPE,
    edge_drop_stats: Optional[Dict[str, float]] = None,
    view_tag: str = "v1",
    *,
    seed_edge_ids: Optional[torch.Tensor] = None,
    preserve_seed_edges: bool = False,
) -> HeteroData:
    """Independent policy-weighted edge drop on forward transactions."""
    out = batch.clone()
    fwd = out[forward_et]
    E = fwd.edge_index.size(1)
    if E == 0:
        return out

    if getattr(fwd, "edge_id", None) is None:
        raise ValueError(
            "edge_id is required on forward edges for policy edge drop; "
            "use train_util.attach_edge_id_from_batch before augmentations."
        )

    drop_probs = edge_drop_cache.lookup_drop_prob(fwd.edge_id, fwd.edge_index.device)
    keep_fwd = _policy_edge_keep_mask(drop_probs)
    if preserve_seed_edges:
        keep_fwd = _force_keep_seed_edges(keep_fwd, fwd.edge_id, seed_edge_ids)
    _accumulate_edge_drop_stats(
        edge_drop_stats,
        edges_before=E,
        keep_mask=keep_fwd,
        drop_probs=drop_probs,
        edge_ids=fwd.edge_id,
        edge_drop_cache=edge_drop_cache,
        view_tag=view_tag,
    )
    _slice_hetero_edge_store(fwd, keep_fwd)
    kept_txn = fwd.edge_id

    rev = out[reverse_et]
    if getattr(rev, "edge_id", None) is None:
        raise ValueError("edge_id is required on reverse edges for synchronized hetero augmentation.")
    rev_keep = torch.isin(rev.edge_id, kept_txn)
    _slice_hetero_edge_store(rev, rev_keep)
    return out


def _policy_edge_drop_view(
    data: Data,
    edge_drop_cache: "EdgeDropScoreCache",
    edge_drop_stats: Optional[Dict[str, float]] = None,
    view_tag: str = "v1",
    *,
    seed_edge_ids: Optional[torch.Tensor] = None,
    preserve_seed_edges: bool = False,
) -> Data:
    out = data.clone()
    E = out.edge_index.size(1)
    if E == 0:
        return out

    if getattr(out, "edge_id", None) is None:
        raise ValueError(
            "edge_id is required on the batch for policy edge drop; "
            "use train_util.attach_edge_id_from_batch before augmentations."
        )

    drop_probs = edge_drop_cache.lookup_drop_prob(out.edge_id, out.edge_index.device)
    keep = _policy_edge_keep_mask(drop_probs)
    if preserve_seed_edges:
        keep = _force_keep_seed_edges(keep, out.edge_id, seed_edge_ids)
    _accumulate_edge_drop_stats(
        edge_drop_stats,
        edges_before=E,
        keep_mask=keep,
        drop_probs=drop_probs,
        edge_ids=out.edge_id,
        edge_drop_cache=edge_drop_cache,
        view_tag=view_tag,
    )
    _slice_edge_aligned_fields(out, keep)
    return out


def mask_edge_attr(
    edge_attr,
    mask_rate=0.1,
    mask_value=0.0,
    mask_cols=None,
    exclude_last_column=False,
):
    """
    Randomly mask entries in edge_attr. Shape is unchanged; input is not
    mutated.

    Args:
        edge_attr: (E, D) tensor.
        mask_rate: fraction in [0, 1] of eligible entries to mask (Bernoulli).
        mask_value: value written where masked.
        mask_cols: If None, all columns are eligible (subject to
            exclude_last_column). If an iterable of column indices, only those
            columns receive Bernoulli masks.
        exclude_last_column: If True, never mask the last column (e.g. RGCN
            relation / edge type in the final feature dim).

    Returns:
        New tensor (E, D) with same shape and dtype as edge_attr.
    """
    if not (0.0 <= mask_rate <= 1.0):
        raise ValueError(f"mask_rate must be in [0, 1], got {mask_rate}")

    out = edge_attr.clone()
    if out.numel() == 0:
        return out

    E, D = out.shape
    device = edge_attr.device
    eligible = torch.zeros((E, D), dtype=torch.bool, device=device)

    if mask_cols is None:
        eligible.fill_(True)
    else:
        cols = torch.as_tensor(list(mask_cols), device=device, dtype=torch.long)
        if cols.numel() > 0:
            eligible[:, cols] = True

    if exclude_last_column and D > 0:
        eligible[:, -1] = False

    if not eligible.any():
        return out

    rand = torch.rand((E, D), device=device)
    apply_mask = (rand < mask_rate) & eligible
    out[apply_mask] = mask_value
    return out


# Semantic base-slot indices within the ports/TDS schema (no synthetic ID column).
# Layout after attach_edge_id_from_batch strips the arange id:
#   [Timestamp, Amount, Currency, PaymentFormat] + [in_port, out_port]? + [in_td, out_td]?
SEMANTIC_CURRENCY_BASE_INDEX = 2
SEMANTIC_PAYMENT_FORMAT_BASE_INDEX = 3


def resolve_semantic_categorical_indices(
    edge_dim: int,
    *,
    ports: bool,
    tds: bool,
    id_column_present: bool = False,
) -> Dict[str, int]:
    """Map currency / payment-format slots using authoritative layout metadata.

    Does **not** use trailing-column heuristics (ports/TDS follow the base slots).
    """
    from data_util import resolve_directional_edge_feature_schema

    offset = 1 if id_column_present else 0
    feature_dim = int(edge_dim) - offset
    schema = resolve_directional_edge_feature_schema(feature_dim, ports=ports, tds=tds)
    if schema["base_dim"] < 4:
        raise ValueError(
            f"semantic group mask requires base_dim>=4, got {schema['base_dim']} "
            f"(edge_dim={edge_dim}, id_column_present={id_column_present})"
        )
    return {
        "currency": offset + SEMANTIC_CURRENCY_BASE_INDEX,
        "payment_format": offset + SEMANTIC_PAYMENT_FORMAT_BASE_INDEX,
        "schema": schema,
        "offset": offset,
    }


def sample_semantic_group_mask_state(
    categorical_group_mask_prob: float,
    *,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, bool]:
    """One Bernoulli decision per categorical group for a whole view/batch."""
    if not (0.0 <= float(categorical_group_mask_prob) <= 1.0):
        raise ValueError(
            f"categorical_group_mask_prob must be in [0, 1], got {categorical_group_mask_prob}"
        )
    p = float(categorical_group_mask_prob)
    if p <= 0.0:
        return {"mask_currency": False, "mask_payment_format": False}
    # Sample two independent uniforms; keep device optional for CPU unit tests.
    if generator is None:
        u = torch.rand(2, device=device)
    else:
        u = torch.rand(2, generator=generator, device=device)
    return {
        "mask_currency": bool(u[0].item() < p),
        "mask_payment_format": bool(u[1].item() < p),
    }


def apply_semantic_group_mask(
    edge_attr: torch.Tensor,
    state: Dict[str, bool],
    *,
    ports: bool,
    tds: bool,
    id_column_present: bool = False,
    mask_value: float = 0.0,
) -> torch.Tensor:
    """Zero selected semantic columns for **all** edges (schema-level group mask).

    Currency and/or payment-format columns only. Timestamp, Amount, ports, and TDS
    are never modified by this mechanism.
    """
    if edge_attr is None or edge_attr.numel() == 0:
        return edge_attr
    if not state.get("mask_currency", False) and not state.get("mask_payment_format", False):
        return edge_attr

    idx = resolve_semantic_categorical_indices(
        int(edge_attr.shape[1]),
        ports=ports,
        tds=tds,
        id_column_present=id_column_present,
    )
    out = edge_attr.clone()
    if state.get("mask_currency", False):
        out[:, idx["currency"]] = mask_value
    if state.get("mask_payment_format", False):
        out[:, idx["payment_format"]] = mask_value
    return out


def _accumulate_semantic_mask_stats(
    stats: Optional[Dict[str, float]],
    *,
    view_tag: str,
    state: Dict[str, bool],
) -> None:
    if stats is None:
        return
    stats[f"semantic_mask_currency_{view_tag}"] = stats.get(
        f"semantic_mask_currency_{view_tag}", 0.0
    ) + float(bool(state.get("mask_currency", False)))
    stats[f"semantic_mask_payment_format_{view_tag}"] = stats.get(
        f"semantic_mask_payment_format_{view_tag}", 0.0
    ) + float(bool(state.get("mask_payment_format", False)))


def generate_views(
    data,
    edge_attr_mask_rate=0.1,
    edge_drop_rate=0.1,
    mask_value=0.0,
    mask_cols=None,
    exclude_last_column=False,
    edge_drop_policy: str = "random",
    edge_drop_cache: Optional["EdgeDropScoreCache"] = None,
    edge_drop_stats: Optional[Dict[str, float]] = None,
    *,
    seed_edge_ids: Optional[torch.Tensor] = None,
    preserve_seed_edges: bool = False,
    semantic_group_mask: bool = False,
    categorical_group_mask_prob: float = 0.0,
    semantic_mask_ports: bool = True,
    semantic_mask_tds: bool = True,
    semantic_mask_generator: Optional[torch.Generator] = None,
):
    """
    Two augmented views for edge-level contrastive learning.

    - If ``edge_drop_rate > 0``: independent edge dropping per view
      (same ``keep_mask`` applied to ``edge_index``, ``edge_attr``, ``edge_id``,
      ``y``, and ``timestamps`` when their first dim matches the edge count).
      Default ``edge_drop_policy='random'`` uses uniform ``edge_drop_rate``.
      ``degree_aware`` / ``degree_flow_aware`` use calibrated per-edge
      probabilities from ``edge_drop_cache``.
    - If ``edge_attr_mask_rate > 0``: independent edge-attribute masking on the
      surviving edges of each view (GraphCL-style per-cell Bernoulli).
    - If ``semantic_group_mask`` and ``categorical_group_mask_prob > 0``: once per
      view, independently decide whether to zero the currency and/or payment-format
      **entire columns** (all edges). Forward/reverse copies of a view share the
      same schema state; view1 and view2 sample independently. Ports/TDS and
      Timestamp/Amount are never masked by this mechanism.
    - If ``preserve_seed_edges`` is True, edges whose ``edge_id`` is in
      ``seed_edge_ids`` are force-kept in both views (opt-in). This retains the
      seed *relation* in the message-passing graph; it is an approximation of
      seed-as-query retention, not a separated readout path. Default False
      preserves legacy behavior (seeds may be dropped).

    Does not mutate ``data`` (including ``data.edge_attr`` / ``data.edge_id``).

    For ``HeteroData``, edge drops are synchronized across forward and reverse edge
    types by transaction ``edge_id``; GraphCL attribute masking is applied
    independently per view on both edge types; semantic group masks are shared
    across forward/reverse within each view.
    """
    if preserve_seed_edges and (seed_edge_ids is None or seed_edge_ids.numel() == 0):
        raise ValueError("preserve_seed_edges=True requires non-empty seed_edge_ids")

    use_policy = edge_drop_policy != "random"
    if use_policy and edge_drop_cache is None:
        raise ValueError(f"edge_drop_policy={edge_drop_policy!r} requires edge_drop_cache")

    if edge_drop_stats is not None:
        edge_drop_stats["target_drop_rate"] = float(
            edge_drop_cache.target_drop_rate if edge_drop_cache is not None else edge_drop_rate
        )
        edge_drop_stats["edge_drop_policy"] = edge_drop_policy
        edge_drop_stats["preserve_seed_edges"] = bool(preserve_seed_edges)
        edge_drop_stats["semantic_group_mask"] = bool(semantic_group_mask)
        edge_drop_stats["categorical_group_mask_prob"] = float(categorical_group_mask_prob)

    drop_kw = {
        "seed_edge_ids": seed_edge_ids,
        "preserve_seed_edges": bool(preserve_seed_edges),
    }

    apply_semantic = bool(semantic_group_mask) and float(categorical_group_mask_prob) > 0.0

    if isinstance(data, HeteroData):
        if use_policy:
            view1 = _hetero_policy_edge_drop_view(
                data, edge_drop_cache, edge_drop_stats=edge_drop_stats, view_tag="v1", **drop_kw
            )
            view2 = _hetero_policy_edge_drop_view(
                data, edge_drop_cache, edge_drop_stats=edge_drop_stats, view_tag="v2", **drop_kw
            )
        else:
            view1 = _hetero_random_edge_drop_view(data, edge_drop_rate, **drop_kw)
            view2 = _hetero_random_edge_drop_view(data, edge_drop_rate, **drop_kw)
        if edge_drop_stats is not None and edge_drop_rate > 0 or use_policy:
            eid1 = view1[FORWARD_EDGE_TYPE].edge_id
            eid2 = view2[FORWARD_EDGE_TYPE].edge_id
            overlap = int(torch.isin(eid1, eid2).sum().item())
            edge_drop_stats["two_view_edge_overlap"] = edge_drop_stats.get("two_view_edge_overlap", 0.0) + float(overlap)
            edge_drop_stats["two_view_overlap_batches"] = edge_drop_stats.get("two_view_overlap_batches", 0.0) + 1.0
        if edge_attr_mask_rate > 0:
            for et in (FORWARD_EDGE_TYPE, REVERSE_EDGE_TYPE):
                store1, store2 = view1[et], view2[et]
                if store1.edge_attr is not None:
                    store1.edge_attr = mask_edge_attr(
                        store1.edge_attr,
                        mask_rate=edge_attr_mask_rate,
                        mask_value=mask_value,
                        mask_cols=mask_cols,
                        exclude_last_column=exclude_last_column,
                    )
                    store2.edge_attr = mask_edge_attr(
                        store2.edge_attr,
                        mask_rate=edge_attr_mask_rate,
                        mask_value=mask_value,
                        mask_cols=mask_cols,
                        exclude_last_column=exclude_last_column,
                    )
        if apply_semantic:
            device = view1[FORWARD_EDGE_TYPE].edge_attr.device
            state1 = sample_semantic_group_mask_state(
                categorical_group_mask_prob, generator=semantic_mask_generator, device=device
            )
            state2 = sample_semantic_group_mask_state(
                categorical_group_mask_prob, generator=semantic_mask_generator, device=device
            )
            for et in (FORWARD_EDGE_TYPE, REVERSE_EDGE_TYPE):
                if view1[et].edge_attr is not None:
                    view1[et].edge_attr = apply_semantic_group_mask(
                        view1[et].edge_attr,
                        state1,
                        ports=semantic_mask_ports,
                        tds=semantic_mask_tds,
                        mask_value=mask_value,
                    )
                if view2[et].edge_attr is not None:
                    view2[et].edge_attr = apply_semantic_group_mask(
                        view2[et].edge_attr,
                        state2,
                        ports=semantic_mask_ports,
                        tds=semantic_mask_tds,
                        mask_value=mask_value,
                    )
            _accumulate_semantic_mask_stats(edge_drop_stats, view_tag="v1", state=state1)
            _accumulate_semantic_mask_stats(edge_drop_stats, view_tag="v2", state=state2)
            if edge_drop_stats is not None:
                edge_drop_stats["semantic_mask_batches"] = (
                    edge_drop_stats.get("semantic_mask_batches", 0.0) + 1.0
                )
                edge_drop_stats["last_semantic_state_v1"] = dict(state1)
                edge_drop_stats["last_semantic_state_v2"] = dict(state2)
        return view1, view2

    if use_policy:
        view1 = _policy_edge_drop_view(
            data, edge_drop_cache, edge_drop_stats=edge_drop_stats, view_tag="v1", **drop_kw
        )
        view2 = _policy_edge_drop_view(
            data, edge_drop_cache, edge_drop_stats=edge_drop_stats, view_tag="v2", **drop_kw
        )
    else:
        view1 = _random_edge_drop_view(data, edge_drop_rate, **drop_kw)
        view2 = _random_edge_drop_view(data, edge_drop_rate, **drop_kw)

    if edge_drop_stats is not None and (edge_drop_rate > 0 or use_policy):
        eid1 = view1.edge_id
        eid2 = view2.edge_id
        overlap = int(torch.isin(eid1, eid2).sum().item())
        edge_drop_stats["two_view_edge_overlap"] = edge_drop_stats.get("two_view_edge_overlap", 0.0) + float(overlap)
        edge_drop_stats["two_view_overlap_batches"] = edge_drop_stats.get("two_view_overlap_batches", 0.0) + 1.0

    if edge_attr_mask_rate > 0 and view1.edge_attr is not None:
        view1.edge_attr = mask_edge_attr(
            view1.edge_attr,
            mask_rate=edge_attr_mask_rate,
            mask_value=mask_value,
            mask_cols=mask_cols,
            exclude_last_column=exclude_last_column,
        )
        view2.edge_attr = mask_edge_attr(
            view2.edge_attr,
            mask_rate=edge_attr_mask_rate,
            mask_value=mask_value,
            mask_cols=mask_cols,
            exclude_last_column=exclude_last_column,
        )
    if apply_semantic and view1.edge_attr is not None:
        device = view1.edge_attr.device
        state1 = sample_semantic_group_mask_state(
            categorical_group_mask_prob, generator=semantic_mask_generator, device=device
        )
        state2 = sample_semantic_group_mask_state(
            categorical_group_mask_prob, generator=semantic_mask_generator, device=device
        )
        view1.edge_attr = apply_semantic_group_mask(
            view1.edge_attr,
            state1,
            ports=semantic_mask_ports,
            tds=semantic_mask_tds,
            mask_value=mask_value,
        )
        view2.edge_attr = apply_semantic_group_mask(
            view2.edge_attr,
            state2,
            ports=semantic_mask_ports,
            tds=semantic_mask_tds,
            mask_value=mask_value,
        )
        _accumulate_semantic_mask_stats(edge_drop_stats, view_tag="v1", state=state1)
        _accumulate_semantic_mask_stats(edge_drop_stats, view_tag="v2", state=state2)
        if edge_drop_stats is not None:
            edge_drop_stats["semantic_mask_batches"] = (
                edge_drop_stats.get("semantic_mask_batches", 0.0) + 1.0
            )
            edge_drop_stats["last_semantic_state_v1"] = dict(state1)
            edge_drop_stats["last_semantic_state_v2"] = dict(state2)
    return view1, view2


def assert_shape_preserving_views(data, view1, view2, edge_attr_before, mask_rate):
    """
    Lightweight checks for the edge_drop_rate=0 path (same topology as ``data``).
    """
    assert type(view1) is type(data)
    assert type(view2) is type(data)
    assert view1.edge_index.shape == data.edge_index.shape
    assert view2.edge_index.shape == data.edge_index.shape
    assert torch.equal(view1.edge_index, data.edge_index)
    assert torch.equal(view2.edge_index, data.edge_index)
    assert view1.edge_attr.shape == data.edge_attr.shape
    assert view2.edge_attr.shape == data.edge_attr.shape
    assert torch.equal(data.edge_attr, edge_attr_before)
    if mask_rate > 0 and data.edge_attr.numel() > 0:
        assert not torch.equal(view1.edge_attr, view2.edge_attr)


if __name__ == "__main__":
    x = torch.ones(6, 1)
    ei = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]], dtype=torch.long)
    ea = torch.randn(6, 4)
    d = Data(x=x, edge_index=ei, edge_attr=ea.clone())
    d.edge_id = torch.arange(6, dtype=torch.long)
    d.y = torch.zeros(6, dtype=torch.long)
    d.timestamps = torch.randn(6)
    ea_snap = d.edge_attr.clone()
    v1, v2 = generate_views(
        d,
        edge_drop_rate=0.0,
        edge_attr_mask_rate=0.5,
        exclude_last_column=True,
    )
    assert_shape_preserving_views(d, v1, v2, ea_snap, mask_rate=0.5)
    print("graph_augmentations: assert_shape_preserving_views OK")

    v1d, v2d = generate_views(d, edge_drop_rate=0.4, edge_attr_mask_rate=0.0)
    assert v1d.edge_id.shape[0] == v1d.edge_index.size(1)
    assert v2d.edge_id.shape[0] == v2d.edge_index.size(1)
    assert torch.equal(d.edge_attr, ea_snap)
    print("graph_augmentations: independent edge drop OK")

    from torch_geometric.data import HeteroData as HD

    h = HD()
    h["node"].x = x
    h["node", "to", "node"].edge_index = ei
    h["node", "rev_to", "node"].edge_index = ei.flip(0)
    h["node", "to", "node"].edge_attr = ea.clone()
    h["node", "rev_to", "node"].edge_attr = ea.clone()
    h["node", "to", "node"].edge_id = torch.arange(6, dtype=torch.long)
    h["node", "rev_to", "node"].edge_id = torch.arange(6, dtype=torch.long)
    hv1, hv2 = generate_views(h, edge_drop_rate=0.5, edge_attr_mask_rate=0.0)
    assert hv1["node", "to", "node"].edge_id.shape[0] == hv1["node", "to", "node"].edge_index.size(1)
    assert torch.equal(
        torch.sort(hv1["node", "to", "node"].edge_id).values,
        torch.sort(hv1["node", "rev_to", "node"].edge_id).values,
    )
    print("graph_augmentations: hetero synchronized edge drop OK")
