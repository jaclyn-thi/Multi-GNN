"""Semantic registry for morphology expert targets.

The registry is diagnostic-only: it maps scalar target names to coarse groups
used for per-group loss logging and ``--morph_target_groups`` filtering.
Unknown targets intentionally fall back to ``other`` so new features can be
added without breaking training.

Legacy group aliases (``local_motif``, ``centrality``, ``temporal``) are
accepted on the CLI for backward compatibility and expand to the finer groups
below.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Mapping, Tuple

MORPH_TARGET_GROUPS: Tuple[str, ...] = (
    "degree_fan",
    "flow_balance",
    "volume_activity",
    "temporal_behavior",
    "motif_participation",
    "local_density",
    "local_context_size",
    "global_role",
    "other",
)

# CLI-only aliases → expanded semantic groups (diagnostics / filtering).
MORPH_TARGET_GROUP_ALIASES: Dict[str, Tuple[str, ...]] = {
    "local_motif": (
        "motif_participation",
        "local_density",
        "local_context_size",
    ),
    "centrality": ("global_role",),
    "temporal": ("temporal_behavior",),
}

# Keys are ``normalize_target_name`` outputs.
_EXPLICIT_TARGET_GROUPS: Dict[str, str] = {
    # Tier 1 — local context size (batch subgraph ego extent).
    "n_edges_sub": "local_context_size",
    "n_nodes_sub": "local_context_size",
    # Tier 1 — endpoint degrees on the sampled subgraph.
    "sender_deg_out_local": "degree_fan",
    "sender_deg_in_local": "degree_fan",
    "receiver_deg_out_local": "degree_fan",
    "receiver_deg_in_local": "degree_fan",
    "deg_sum_out_local": "degree_fan",
    "deg_sum_in_local": "degree_fan",
    # Tier 1 — clustering / density on the sampled subgraph.
    "sender_clustering_local": "local_density",
    "receiver_clustering_local": "local_density",
    "mean_clustering_local": "local_density",
    # Tier 1 — triangle / motif participation on the sampled subgraph.
    "sender_triangles_local": "motif_participation",
    "receiver_triangles_local": "motif_participation",
    "mean_triangles_local": "motif_participation",
    # Tier 0 — split-global endpoint degree lift.
    "sender_deg_in": "degree_fan",
    "sender_deg_out": "degree_fan",
    "sender_deg_total": "degree_fan",
    "receiver_deg_in": "degree_fan",
    "receiver_deg_out": "degree_fan",
    "receiver_deg_total": "degree_fan",
    "deg_sum_out_global": "degree_fan",
    "deg_sum_in_global": "degree_fan",
    "deg_sum_total_global": "degree_fan",
    # Tier 2 — betweenness centrality endpoint lift.
    "sender_bc": "global_role",
    "receiver_bc": "global_role",
    "bc_sum_global": "global_role",
    "bc_max_global": "global_role",
    # Tier 0 — split-global amount flow balance lift.
    "sender_in_amount_log": "flow_balance",
    "sender_out_amount_log": "flow_balance",
    "receiver_in_amount_log": "flow_balance",
    "receiver_out_amount_log": "flow_balance",
    "sender_flow_balance_ratio": "flow_balance",
    "receiver_flow_balance_ratio": "flow_balance",
    "sender_abs_flow_imbalance_log": "flow_balance",
    "receiver_abs_flow_imbalance_log": "flow_balance",
    "edge_to_sender_out_ratio_log": "flow_balance",
    "edge_to_receiver_in_ratio_log": "flow_balance",
    # Edge-native AML schema (transaction-level).
    "timestamp": "temporal_behavior",
    "amount_sent": "volume_activity",
    "amount_received": "volume_activity",
    "sent_currency": "other",
    "received_currency": "other",
    "payment_format": "other",
}


def normalize_target_name(name: str) -> str:
    """Lowercase target name with punctuation/whitespace normalized to underscores."""

    normalized = re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip().lower())
    return normalized.strip("_")


def safe_target_log_name(name: str) -> str:
    """Stable target name for metric keys."""

    return normalize_target_name(name) or "unnamed_target"


def morph_target_group(name: str) -> str:
    """Return semantic group for a morphology target name."""

    key = normalize_target_name(name)
    if key in _EXPLICIT_TARGET_GROUPS:
        return _EXPLICIT_TARGET_GROUPS[key]

    if any(token in key for token in ("timestamp", "time", "temporal", "burst", "interarrival", "recency")):
        return "temporal_behavior"
    if any(
        token in key
        for token in (
            "betweenness",
            "centrality",
            "bc_max",
            "bc_sum",
            "sender_bc",
            "receiver_bc",
            "_bc",
        )
    ):
        return "global_role"
    if any(token in key for token in ("triangle", "wedge", "cycle", "motif")):
        return "motif_participation"
    if "clustering" in key or "local_density" in key or "ego_density" in key:
        return "local_density"
    if any(token in key for token in ("n_edges_sub", "n_nodes_sub", "ego_size", "ego_edge", "neighborhood_size")):
        return "local_context_size"
    if any(token in key for token in ("degree", "deg", "fan_in", "fan_out", "fan")):
        return "degree_fan"
    if any(
        token in key
        for token in (
            "flow_balance_ratio",
            "abs_flow_imbalance",
            "edge_to_sender_out_ratio",
            "edge_to_receiver_in_ratio",
            "in_amount_log",
            "out_amount_log",
        )
    ):
        return "flow_balance"
    if any(
        token in key
        for token in (
            "balance",
            "net_amount",
            "net_flow",
            "amount_in",
            "amount_out",
            "in_amount",
            "out_amount",
            "flow_asym",
            "imbalance",
        )
    ):
        return "flow_balance"
    if any(token in key for token in ("amount", "volume", "activity", "transaction_count", "tx_count", "count")):
        return "volume_activity"
    return "other"


def expand_morph_target_groups(groups: Iterable[str]) -> Tuple[str, ...]:
    """Expand legacy CLI aliases to canonical semantic groups (deduped, stable order)."""

    expanded: List[str] = []
    seen = set()
    for raw in groups:
        key = str(raw).strip().lower()
        if not key:
            continue
        resolved = MORPH_TARGET_GROUP_ALIASES.get(key, (key,))
        for group in resolved:
            if group not in seen:
                seen.add(group)
                expanded.append(group)
    return tuple(expanded)


def morph_target_group_counts(names: Iterable[str]) -> Dict[str, int]:
    """Count targets per semantic group."""

    counts = {group: 0 for group in MORPH_TARGET_GROUPS}
    for name in names:
        group = morph_target_group(name)
        counts[group] = counts.get(group, 0) + 1
    return {group: count for group, count in counts.items() if count > 0}


def morph_target_names_by_group(names: Iterable[str]) -> Dict[str, List[str]]:
    """Map semantic groups to the target names assigned to each."""

    out: Dict[str, List[str]] = {group: [] for group in MORPH_TARGET_GROUPS}
    for name in names:
        out[morph_target_group(name)].append(str(name))
    return {group: vals for group, vals in out.items() if vals}


def format_morph_target_group_summary(names: Iterable[str]) -> str:
    """Compact human-readable per-group target counts for logging."""

    by_group = morph_target_names_by_group(names)
    parts = []
    for group in MORPH_TARGET_GROUPS:
        targets = by_group.get(group)
        if targets:
            parts.append(f"{group}={len(targets)}")
    return ", ".join(parts) if parts else "none"


def format_morph_target_group_details(names: Iterable[str]) -> List[str]:
    """One log line per non-empty semantic group listing member targets."""

    by_group = morph_target_names_by_group(names)
    lines: List[str] = []
    for group in MORPH_TARGET_GROUPS:
        targets = by_group.get(group)
        if targets:
            lines.append(f"{group} ({len(targets)}): {', '.join(targets)}")
    return lines


__all__ = [
    "MORPH_TARGET_GROUPS",
    "MORPH_TARGET_GROUP_ALIASES",
    "expand_morph_target_groups",
    "format_morph_target_group_details",
    "format_morph_target_group_summary",
    "morph_target_group",
    "morph_target_group_counts",
    "morph_target_names_by_group",
    "normalize_target_name",
    "safe_target_log_name",
]
