"""Ranking / recall-oriented metrics for frozen probes and supervised eval.

Computes alert-budget (P@K / R@K / Lift@K) and precision-constrained recall
from binary labels and prediction scores. Scores-only; no validation thresholding.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.metrics import precision_recall_curve

ALERT_BUDGET_KS: Tuple[int, ...] = (100, 500, 1000)
PRECISION_TARGETS: Tuple[float, ...] = (0.95, 0.90, 0.80, 0.70)


def _as_1d(y: np.ndarray) -> np.ndarray:
    return np.asarray(y).reshape(-1)


def alert_budget_metrics(
    y: np.ndarray,
    scores: np.ndarray,
    ks: Sequence[int] = ALERT_BUDGET_KS,
) -> Dict[str, float]:
    """P@K, R@K, Lift@K for each K in ``ks``.

    Ranking is by descending score. Ties follow ``numpy.argsort(-scores)``
    (stable mergesort is not used; ties keep relative order from argsort).
    When K exceeds n, K is clipped to n.
    """
    y = _as_1d(y).astype(np.int64)
    scores = _as_1d(scores).astype(np.float64)
    if y.shape[0] != scores.shape[0]:
        raise ValueError(f"y/scores length mismatch: {y.shape[0]} vs {scores.shape[0]}")
    n = int(y.shape[0])
    positives = int(y.sum())
    prevalence = float(y.mean()) if n else float("nan")
    out: Dict[str, float] = {}
    if n == 0:
        for k in ks:
            out[f"precision_at_{k}"] = float("nan")
            out[f"recall_at_{k}"] = float("nan")
            out[f"lift_at_{k}"] = float("nan")
        return out
    # Stable sort for deterministic ties: mergesort is stable; negate scores for desc.
    order = np.argsort(-scores, kind="mergesort")
    for k in ks:
        kk = min(int(k), n)
        top = order[:kk]
        tp = int(y[top].sum())
        precision = float(tp / kk) if kk else float("nan")
        recall = float(tp / positives) if positives else float("nan")
        lift = float(precision / prevalence) if prevalence > 0 else float("nan")
        out[f"precision_at_{k}"] = precision
        out[f"recall_at_{k}"] = recall
        out[f"lift_at_{k}"] = lift
    return out


def precision_constrained_recall(
    y: np.ndarray,
    scores: np.ndarray,
    targets: Sequence[float] = PRECISION_TARGETS,
) -> Dict[str, float]:
    """Maximum recall attainable at precision >= each target (test-set PR curve).

    For each target t, among PR-curve operating points with precision >= t,
    select the one with maximum recall. Also records the corresponding
    threshold (score cutoff), number of alerts (predictions above threshold),
    and achieved precision.

    Keys (example for t=0.90):
      recall_at_precision_ge_0.90
      threshold_at_precision_ge_0.90
      n_alerts_at_precision_ge_0.90
      precision_achieved_at_precision_ge_0.90
    """
    y = _as_1d(y).astype(np.int64)
    scores = _as_1d(scores).astype(np.float64)
    if y.shape[0] != scores.shape[0]:
        raise ValueError(f"y/scores length mismatch: {y.shape[0]} vs {scores.shape[0]}")
    out: Dict[str, float] = {}
    n = int(y.shape[0])
    for t in targets:
        key = f"{float(t):.2f}"
        out[f"recall_at_precision_ge_{key}"] = float("nan")
        out[f"threshold_at_precision_ge_{key}"] = float("nan")
        out[f"n_alerts_at_precision_ge_{key}"] = float("nan")
        out[f"precision_achieved_at_precision_ge_{key}"] = float("nan")

    if n == 0 or int(y.sum()) == 0 or len(np.unique(y)) < 2:
        return out

    precision, recall, thresholds = precision_recall_curve(y, scores)
    # sklearn: precision/recall length = len(thresholds)+1; last point is recall=0
    # Use points that have an associated threshold (all but the final artificial point).
    if thresholds.size == 0:
        return out
    prec = precision[:-1]
    rec = recall[:-1]
    thr = thresholds

    for t in targets:
        key = f"{float(t):.2f}"
        mask = prec >= float(t)
        if not bool(np.any(mask)):
            continue
        # Maximize recall among feasible points; break ties by higher precision then lower threshold
        cand_idx = np.flatnonzero(mask)
        best_local = int(np.argmax(rec[cand_idx]))
        i = int(cand_idx[best_local])
        thr_i = float(thr[i])
        n_alerts = int(np.sum(scores >= thr_i))
        out[f"recall_at_precision_ge_{key}"] = float(rec[i])
        out[f"threshold_at_precision_ge_{key}"] = thr_i
        out[f"n_alerts_at_precision_ge_{key}"] = float(n_alerts)
        out[f"precision_achieved_at_precision_ge_{key}"] = float(prec[i])
    return out


def ranking_metrics(
    y: np.ndarray,
    scores: np.ndarray,
    *,
    ks: Sequence[int] = ALERT_BUDGET_KS,
    precision_targets: Sequence[float] = PRECISION_TARGETS,
) -> Dict[str, float]:
    """Combined alert-budget + precision-constrained recall metrics."""
    out = alert_budget_metrics(y, scores, ks=ks)
    out.update(precision_constrained_recall(y, scores, targets=precision_targets))
    return out


def ranking_metric_keys(
    ks: Sequence[int] = ALERT_BUDGET_KS,
    precision_targets: Sequence[float] = PRECISION_TARGETS,
) -> List[str]:
    keys: List[str] = []
    for k in ks:
        keys.extend([f"precision_at_{k}", f"recall_at_{k}", f"lift_at_{k}"])
    for t in precision_targets:
        key = f"{float(t):.2f}"
        keys.extend(
            [
                f"recall_at_precision_ge_{key}",
                f"threshold_at_precision_ge_{key}",
                f"n_alerts_at_precision_ge_{key}",
                f"precision_achieved_at_precision_ge_{key}",
            ]
        )
    return keys


def merge_ranking_into_test_block(
    test_block: Mapping[str, object],
    y: np.ndarray,
    scores: np.ndarray,
) -> Dict[str, object]:
    """Return a copy of ``test_block`` with ranking metrics filled/updated."""
    out = dict(test_block)
    out.update(ranking_metrics(y, scores))
    return out
