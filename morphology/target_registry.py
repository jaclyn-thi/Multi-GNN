"""Semantic registry for morphology expert targets.

The registry is diagnostic-only: it maps scalar target names to coarse groups
used for per-group loss logging. Unknown targets intentionally fall back to
``other`` so new features can be added without breaking training.
"""

from __future__ import annotations

import re
from typing import Dict

MORPH_TARGET_GROUPS = (
    "degree_fan",
    "local_motif",
    "centrality",
    "flow_balance",
    "volume_activity",
    "temporal",
    "other",
)

_EXPLICIT_TARGET_GROUPS: Dict[str, str] = {
    # Tier 1 local ego / motif structure.
    "n_edges_sub": "local_motif",
    "n_nodes_sub": "local_motif",
    "sender_clustering_local": "local_motif",
    "receiver_clustering_local": "local_motif",
    "mean_clustering_local": "local_motif",
    "sender_triangles_local": "local_motif",
    "receiver_triangles_local": "local_motif",
    "mean_triangles_local": "local_motif",
    # Edge-native AML schema.
    "timestamp": "temporal",
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

    if any(token in key for token in ("timestamp", "time", "temporal", "burst", "interarrival")):
        return "temporal"
    if any(token in key for token in ("betweenness", "centrality", "bc_")):
        return "centrality"
    if any(token in key for token in ("triangle", "clustering", "motif", "ego")):
        return "local_motif"
    if any(token in key for token in ("degree", "deg", "fan_in", "fan_out", "fan")):
        return "degree_fan"
    if any(token in key for token in ("balance", "net_amount", "amount_in", "amount_out", "in_amount", "out_amount")):
        return "flow_balance"
    if any(token in key for token in ("amount", "volume", "activity", "transaction_count", "tx_count", "count")):
        return "volume_activity"
    return "other"


__all__ = [
    "MORPH_TARGET_GROUPS",
    "morph_target_group",
    "normalize_target_name",
    "safe_target_log_name",
]
