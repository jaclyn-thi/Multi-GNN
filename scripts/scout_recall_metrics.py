#!/usr/bin/env python3
"""Shared metric key list for scout summarize scripts (TF aux / morph / soft-positive).

Import this so future scout summaries automatically surface recall-oriented fields
when present in probe JSONs.
"""

from __future__ import annotations

from typing import Any, Dict, List

RECALL_ORIENTED_METRIC_KEYS: List[str] = [
    "precision_at_100",
    "recall_at_100",
    "lift_at_100",
    "precision_at_500",
    "recall_at_500",
    "lift_at_500",
    "precision_at_1000",
    "recall_at_1000",
    "lift_at_1000",
    "recall_at_precision_ge_0.95",
    "recall_at_precision_ge_0.90",
    "recall_at_precision_ge_0.80",
    "recall_at_precision_ge_0.70",
    "threshold_at_precision_ge_0.95",
    "threshold_at_precision_ge_0.90",
    "threshold_at_precision_ge_0.80",
    "threshold_at_precision_ge_0.70",
    "n_alerts_at_precision_ge_0.95",
    "n_alerts_at_precision_ge_0.90",
    "n_alerts_at_precision_ge_0.80",
    "n_alerts_at_precision_ge_0.70",
    "precision_achieved_at_precision_ge_0.95",
    "precision_achieved_at_precision_ge_0.90",
    "precision_achieved_at_precision_ge_0.80",
    "precision_achieved_at_precision_ge_0.70",
]


def extract_recall_oriented(test_block: Dict[str, Any]) -> Dict[str, Any]:
    """Pull recall-oriented metrics from a probe test block if present."""
    return {k: test_block[k] for k in RECALL_ORIENTED_METRIC_KEYS if k in test_block}
