"""
Post-hoc laundering typology diagnostics on test predictions.

Consumes ``te_data.csv_edge_ids`` and ``te_data.pattern_metadata_by_edge_id`` only
after predictions are produced. Does not affect training, inference, splits, or labels.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from pattern_metadata import LaunderingPatternMetadata, lookup_pattern_metadata


def evaluate_binary_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: Optional[np.ndarray] = None,
    *,
    split_name: str = "test",
) -> Dict[str, float]:
    """Overall binary metrics using the same conventions as ``linear_probe.evaluate_probe``."""
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)

    if y_true.shape[0] == 0:
        logging.warning("pattern diagnostics %s: empty split", split_name)
        return {
            "auroc": float("nan"),
            "f1": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "n": 0.0,
            "positive_rate": float("nan"),
        }

    metrics: Dict[str, float] = {
        "n": float(y_true.shape[0]),
        "positive_rate": float(y_true.mean()),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }
    if y_scores is None:
        metrics["auroc"] = float("nan")
    elif len(np.unique(y_true)) < 2:
        logging.warning(
            "pattern diagnostics %s: only one class present; AUROC undefined", split_name
        )
        metrics["auroc"] = float("nan")
    else:
        metrics["auroc"] = float(roc_auc_score(y_true, np.asarray(y_scores).reshape(-1)))
    return metrics


def _recall_bucket(true_positives: int, false_negatives: int) -> Dict[str, Any]:
    support = true_positives + false_negatives
    recall = float(true_positives / support) if support > 0 else float("nan")
    return {
        "support": int(support),
        "true_positives": int(true_positives),
        "false_negatives": int(false_negatives),
        "recall": recall,
    }


def _one_vs_rest_auroc(
    positive_mask: np.ndarray,
    y_scores: Optional[np.ndarray],
    *,
    context: str,
) -> float:
    """
    AUROC for detecting ``positive_mask`` edges vs all other rows (one-vs-rest).

    Uses the same probe scores as overall test AUROC; positives are typically
    laundering edges in a given pattern bucket, negatives are all other test edges.
    """
    if y_scores is None:
        return float("nan")
    positive_mask = np.asarray(positive_mask, dtype=bool).reshape(-1)
    scores = np.asarray(y_scores, dtype=np.float64).reshape(-1)
    labels = positive_mask.astype(np.int64)
    if labels.shape[0] != scores.shape[0]:
        raise ValueError(
            f"positive_mask length {labels.shape[0]} != y_scores length {scores.shape[0]}"
        )
    if len(np.unique(labels)) < 2:
        logging.warning(
            "pattern diagnostics %s: only one class present; AUROC undefined", context
        )
        return float("nan")
    return float(roc_auc_score(labels, scores))


def _pattern_bucket(
    true_positives: int,
    false_negatives: int,
    positive_mask: np.ndarray,
    y_scores: Optional[np.ndarray],
    *,
    auroc_context: str,
) -> Dict[str, Any]:
    bucket = _recall_bucket(true_positives, false_negatives)
    bucket["auroc"] = _one_vs_rest_auroc(
        positive_mask, y_scores, context=auroc_context
    )
    return bucket


def _resolve_metadata_for_rows(
    graph_edge_indices: np.ndarray,
    te_data: Any,
) -> List[Optional[LaunderingPatternMetadata]]:
    metadata_by_edge_id = getattr(te_data, "pattern_metadata_by_edge_id", None) or {}
    csv_edge_ids = getattr(te_data, "csv_edge_ids", None)
    if csv_edge_ids is None:
        raise ValueError(
            "te_data is missing csv_edge_ids; call get_data() with pattern metadata available."
        )

    csv_edge_ids_np = np.asarray(csv_edge_ids.cpu().numpy()).reshape(-1)
    graph_edge_indices = np.asarray(graph_edge_indices, dtype=np.int64).reshape(-1)

    resolved: List[Optional[LaunderingPatternMetadata]] = []
    for graph_idx in graph_edge_indices:
        if graph_idx < 0 or graph_idx >= csv_edge_ids_np.shape[0]:
            raise IndexError(
                f"graph_edge_index {int(graph_idx)} out of range for csv_edge_ids "
                f"(n={csv_edge_ids_np.shape[0]})"
            )
        csv_edge_id = int(csv_edge_ids_np[int(graph_idx)])
        resolved.append(lookup_pattern_metadata(csv_edge_id, metadata_by_edge_id))
    return resolved


def evaluate_pattern_typology_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    te_data: Any,
    graph_edge_indices: np.ndarray,
    *,
    y_scores: Optional[np.ndarray] = None,
    include_attempt_id: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate test predictions by known laundering typology.

    Parameters
    ----------
    y_true, y_pred, y_scores
        Aligned per test row (e.g. from ``test.npz`` and a fitted linear probe).
    te_data
        Full test graph from ``get_data()`` with ``csv_edge_ids`` and
        ``pattern_metadata_by_edge_id`` attached.
    graph_edge_indices
        Graph row index per test row (``edge_id`` column from ``test.npz``).

    Per-pattern metrics
    -------------------
    * **recall** — among laundering test edges with known metadata in the bucket,
      fraction predicted positive at the probe threshold.
    * **auroc** — one-vs-rest on the full test split: positives are laundering edges
      in the bucket; negatives are all other test rows (other patterns, unlabeled
      laundering, and legitimate edges).
    * **by_pattern_detail** — same metrics grouped by ``pattern_detail`` (e.g.
      fan-out degree / cycle length variants).
    """
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    graph_edge_indices = np.asarray(graph_edge_indices, dtype=np.int64).reshape(-1)
    n_rows = y_true.shape[0]

    if y_scores is not None:
        y_scores = np.asarray(y_scores, dtype=np.float64).reshape(-1)
        if y_scores.shape[0] != n_rows:
            raise ValueError(
                f"y_scores length {y_scores.shape[0]} != y_true length {n_rows}"
            )

    if not (y_true.shape[0] == y_pred.shape[0] == graph_edge_indices.shape[0]):
        raise ValueError(
            "y_true, y_pred, and graph_edge_indices must have the same length "
            f"({y_true.shape[0]}, {y_pred.shape[0]}, {graph_edge_indices.shape[0]})"
        )

    overall = evaluate_binary_predictions(
        y_true, y_pred, y_scores=y_scores, split_name="test"
    )

    metadata_rows = _resolve_metadata_for_rows(graph_edge_indices, te_data)
    laundering_mask = y_true == 1
    laundering_total = int(laundering_mask.sum())

    known_mask = np.array(
        [meta is not None for meta in metadata_rows], dtype=bool
    )
    laundering_known = int((laundering_mask & known_mask).sum())
    laundering_unknown = int((laundering_mask & ~known_mask).sum())

    type_tp: Dict[str, int] = defaultdict(int)
    type_fn: Dict[str, int] = defaultdict(int)
    type_support: Dict[str, int] = defaultdict(int)
    type_positive_mask: Dict[str, np.ndarray] = {}

    detail_tp: Dict[str, int] = defaultdict(int)
    detail_fn: Dict[str, int] = defaultdict(int)
    detail_support: Dict[str, int] = defaultdict(int)
    detail_pattern_type: Dict[str, str] = {}
    detail_positive_mask: Dict[str, np.ndarray] = {}

    attempt_tp: Dict[int, int] = defaultdict(int)
    attempt_fn: Dict[int, int] = defaultdict(int)
    attempt_support: Dict[int, int] = defaultdict(int)

    laundering_known_indices = np.where(laundering_mask & known_mask)[0]
    for idx in laundering_known_indices:
        meta = metadata_rows[int(idx)]
        assert meta is not None
        pattern_type = meta.pattern_type
        type_support[pattern_type] += 1
        if y_pred[int(idx)] == 1:
            type_tp[pattern_type] += 1
        else:
            type_fn[pattern_type] += 1

        pattern_detail = meta.pattern_detail or ""
        detail_support[pattern_detail] += 1
        detail_pattern_type.setdefault(pattern_detail, pattern_type)
        if y_pred[int(idx)] == 1:
            detail_tp[pattern_detail] += 1
        else:
            detail_fn[pattern_detail] += 1

        if include_attempt_id:
            attempt_support[meta.attempt_id] += 1
            if y_pred[int(idx)] == 1:
                attempt_tp[meta.attempt_id] += 1
            else:
                attempt_fn[meta.attempt_id] += 1

    for pattern_type in sorted(type_support):
        positive_mask = np.zeros(n_rows, dtype=bool)
        for idx in laundering_known_indices:
            meta = metadata_rows[int(idx)]
            assert meta is not None
            if meta.pattern_type == pattern_type:
                positive_mask[int(idx)] = True
        type_positive_mask[pattern_type] = positive_mask

    for pattern_detail in sorted(detail_support):
        positive_mask = np.zeros(n_rows, dtype=bool)
        for idx in laundering_known_indices:
            meta = metadata_rows[int(idx)]
            assert meta is not None
            if (meta.pattern_detail or "") == pattern_detail:
                positive_mask[int(idx)] = True
        detail_positive_mask[pattern_detail] = positive_mask

    by_pattern_type: Dict[str, Dict[str, Any]] = {}
    for pattern_type in sorted(type_support):
        by_pattern_type[pattern_type] = _pattern_bucket(
            type_tp[pattern_type],
            type_fn[pattern_type],
            type_positive_mask[pattern_type],
            y_scores,
            auroc_context=f"pattern_type={pattern_type}",
        )

    by_pattern_detail: Dict[str, Dict[str, Any]] = {}
    for pattern_detail in sorted(detail_support):
        bucket = _pattern_bucket(
            detail_tp[pattern_detail],
            detail_fn[pattern_detail],
            detail_positive_mask[pattern_detail],
            y_scores,
            auroc_context=f"pattern_detail={pattern_detail!r}",
        )
        bucket["pattern_type"] = detail_pattern_type[pattern_detail]
        by_pattern_detail[pattern_detail] = bucket

    result: Dict[str, Any] = {
        "overall_test": overall,
        "laundering_test_edges": {
            "total": laundering_total,
            "with_known_pattern_metadata": laundering_known,
            "without_known_pattern_metadata": laundering_unknown,
        },
        "by_pattern_type": by_pattern_type,
        "by_pattern_detail": by_pattern_detail,
    }

    if include_attempt_id:
        attempt_pattern_type: Dict[int, str] = {}
        for idx in np.where(laundering_mask & known_mask)[0]:
            meta = metadata_rows[int(idx)]
            assert meta is not None
            attempt_pattern_type.setdefault(meta.attempt_id, meta.pattern_type)

        by_attempt_id: Dict[str, Dict[str, Any]] = {}
        for attempt_id in sorted(attempt_support):
            bucket = _recall_bucket(attempt_tp[attempt_id], attempt_fn[attempt_id])
            bucket["pattern_type"] = attempt_pattern_type[attempt_id]
            by_attempt_id[str(attempt_id)] = bucket
        result["by_attempt_id"] = by_attempt_id

    return result


def pattern_type_rows_for_csv(diagnostics: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for pattern_type, stats in diagnostics.get("by_pattern_type", {}).items():
        rows.append(
            {
                "pattern_type": pattern_type,
                "support": stats["support"],
                "true_positives": stats["true_positives"],
                "false_negatives": stats["false_negatives"],
                "recall": stats["recall"],
                "auroc": stats.get("auroc", float("nan")),
            }
        )
    return rows


def pattern_detail_rows_for_csv(diagnostics: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for pattern_detail, stats in diagnostics.get("by_pattern_detail", {}).items():
        rows.append(
            {
                "pattern_detail": pattern_detail,
                "pattern_type": stats.get("pattern_type"),
                "support": stats["support"],
                "true_positives": stats["true_positives"],
                "false_negatives": stats["false_negatives"],
                "recall": stats["recall"],
                "auroc": stats.get("auroc", float("nan")),
            }
        )
    return rows


def attempt_id_rows_for_csv(diagnostics: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for attempt_id, stats in diagnostics.get("by_attempt_id", {}).items():
        rows.append(
            {
                "attempt_id": int(attempt_id),
                "pattern_type": stats.get("pattern_type"),
                "support": stats["support"],
                "true_positives": stats["true_positives"],
                "false_negatives": stats["false_negatives"],
                "recall": stats["recall"],
            }
        )
    return sorted(rows, key=lambda row: row["attempt_id"])


def write_pattern_typology_diagnostics(
    diagnostics: Mapping[str, Any],
    output_dir: Path,
    *,
    write_attempt_csv: bool = False,
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}

    json_path = output_dir / "pattern_typology_test.json"
    json_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    written["json"] = json_path

    type_csv_path = output_dir / "pattern_typology_by_type.csv"
    _write_csv(type_csv_path, pattern_type_rows_for_csv(diagnostics))
    written["by_type_csv"] = type_csv_path

    detail_csv_path = output_dir / "pattern_typology_by_detail.csv"
    _write_csv(detail_csv_path, pattern_detail_rows_for_csv(diagnostics))
    written["by_detail_csv"] = detail_csv_path

    if write_attempt_csv and "by_attempt_id" in diagnostics:
        attempt_csv_path = output_dir / "pattern_typology_by_attempt.csv"
        _write_csv(attempt_csv_path, attempt_id_rows_for_csv(diagnostics))
        written["by_attempt_csv"] = attempt_csv_path

    return written


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import csv

    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def log_pattern_typology_diagnostics(diagnostics: Mapping[str, Any]) -> None:
    overall = diagnostics["overall_test"]
    laundering = diagnostics["laundering_test_edges"]
    logging.info(
        "Pattern typology diagnostics — test overall: AUROC=%.4f F1=%.4f "
        "precision=%.4f recall=%.4f (n=%d)",
        overall.get("auroc", float("nan")),
        overall["f1"],
        overall["precision"],
        overall["recall"],
        int(overall["n"]),
    )
    logging.info(
        "Laundering test edges: total=%d with_known_pattern_metadata=%d "
        "without_known_pattern_metadata=%d",
        laundering["total"],
        laundering["with_known_pattern_metadata"],
        laundering["without_known_pattern_metadata"],
    )
    logging.info(
        "Per-pattern_type recall + one-vs-rest AUROC (laundering edges with known metadata):"
    )
    for pattern_type, stats in sorted(diagnostics.get("by_pattern_type", {}).items()):
        logging.info(
            "  %s: support=%d TP=%d FN=%d recall=%.4f auroc=%.4f",
            pattern_type,
            stats["support"],
            stats["true_positives"],
            stats["false_negatives"],
            stats["recall"],
            stats.get("auroc", float("nan")),
        )
    by_detail = diagnostics.get("by_pattern_detail", {})
    if by_detail:
        logging.info("Per-pattern_detail recall + one-vs-rest AUROC:")
        for pattern_detail, stats in sorted(
            by_detail.items(),
            key=lambda item: (item[1].get("pattern_type", ""), item[0]),
        ):
            logging.info(
                "  %s (%s): support=%d TP=%d FN=%d recall=%.4f auroc=%.4f",
                pattern_detail,
                stats.get("pattern_type"),
                stats["support"],
                stats["true_positives"],
                stats["false_negatives"],
                stats["recall"],
                stats.get("auroc", float("nan")),
            )
