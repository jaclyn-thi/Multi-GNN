"""
Phase 5b: Papagei-style linear probing on frozen embeddings (Phase 5a ``.npz`` files).

Fits sklearn logistic regression on train embeddings and reports AUROC and AUPRC
(threshold-free) plus F1 / precision / recall at a validation-selected threshold
(max F1 on val by default).
When the selected threshold is not 0.5, also writes ``splits_at_threshold_0.5`` for comparison.

``probe_results.json`` structure:
  - ``classification_threshold``: how the flagging threshold was chosen
  - ``splits_at_selected_threshold``: primary task metrics (train / val / test)
  - ``splits_at_threshold_0.5``: optional baseline at fixed 0.5 (omitted if selected is 0.5)

Example (after ``embedding_extraction.py``):

  python linear_probe.py --unique_name my_pretrain --testing
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import wandb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from train_util import extract_param
from util import logger_setup, set_seed


def load_embedding_npz(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing embedding file: {path}")
    data = np.load(path)
    z = np.asarray(data["Z"], dtype=np.float32)
    y = np.asarray(data["y"]).reshape(-1).astype(np.int64)
    edge_id = np.asarray(data["edge_id"]).reshape(-1)
    if z.shape[0] != y.shape[0] or z.shape[0] != edge_id.shape[0]:
        raise ValueError(
            f"Row count mismatch in {path}: Z={z.shape[0]} y={y.shape[0]} edge_id={edge_id.shape[0]}"
        )
    return z, y, edge_id


def resolve_class_weight(args) -> Optional[Any]:
    mode = str(args.class_weight).lower()
    if mode == "none":
        return None
    if mode == "balanced":
        return "balanced"
    if mode == "model":
        if not args.model:
            raise ValueError("--class_weight model requires --model (e.g. gin).")
        w0 = float(extract_param("w_ce1", args))
        w1 = float(extract_param("w_ce2", args))
        return {0: w0, 1: w1}
    if mode == "explicit":
        pos = getattr(args, "class_weight_pos", None)
        if pos is None:
            raise ValueError("--class_weight explicit requires --class_weight_pos.")
        return {0: 1.0, 1: float(pos)}
    raise ValueError(f"Unsupported --class_weight {args.class_weight!r}")


def serialize_class_weight(class_weight: Optional[Any]) -> Any:
    """JSON-safe representation of sklearn ``class_weight``."""
    if class_weight is None:
        return None
    if isinstance(class_weight, str):
        return class_weight
    return {int(k): float(v) for k, v in class_weight.items()}


def tune_threshold_max_f1(y: np.ndarray, proba: np.ndarray) -> Tuple[float, float]:
    """
    Pick the score threshold on validation that maximizes F1.

    Returns ``(threshold, val_f1)``. Falls back to 0.5 if val has a single class.
    """
    y = y.astype(np.int64)
    if len(np.unique(y)) < 2:
        logging.warning("val threshold tuning: only one class present; using 0.5")
        pred = (proba >= 0.5).astype(np.int64)
        return 0.5, float(f1_score(y, pred, zero_division=0))

    precisions, recalls, thresholds = precision_recall_curve(y, proba)
    if thresholds.size == 0:
        pred = (proba >= 0.5).astype(np.int64)
        return 0.5, float(f1_score(y, pred, zero_division=0))

    f1_scores = (2 * precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-12)
    best_i = int(np.argmax(f1_scores))
    threshold = float(thresholds[best_i])
    return threshold, float(f1_scores[best_i])


def fit_logistic_probe(
    z_train: np.ndarray,
    y_train: np.ndarray,
    class_weight: Optional[Any],
    max_iter: int,
    seed: int,
    n_jobs: int = -1,
    C: float = 1.0,
) -> LogisticRegression:
    clf = LogisticRegression(
        class_weight=class_weight,
        max_iter=max_iter,
        random_state=seed,
        solver="lbfgs",
        n_jobs=n_jobs,
        C=float(C),
    )
    clf.fit(z_train, y_train)
    return clf


def evaluate_probe(
    clf: LogisticRegression,
    z: np.ndarray,
    y: np.ndarray,
    split_name: str,
    threshold: float = 0.5,
) -> Dict[str, float]:
    if z.shape[0] == 0:
        logging.warning("linear probe %s: empty split", split_name)
        return {
            "auroc": float("nan"),
            "auprc": float("nan"),
            "f1": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "n": 0.0,
            "positive_rate": float("nan"),
        }

    proba = clf.predict_proba(z)[:, 1]
    pred = (proba >= threshold).astype(np.int64)
    y = y.astype(np.int64)

    metrics: Dict[str, float] = {
        "n": float(z.shape[0]),
        "positive_rate": float(y.mean()),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
    }
    if len(np.unique(y)) < 2:
        logging.warning(
            "linear probe %s: only one class present; AUROC/AUPRC undefined", split_name
        )
        metrics["auroc"] = float("nan")
        metrics["auprc"] = float("nan")
    else:
        metrics["auroc"] = float(roc_auc_score(y, proba))
        metrics["auprc"] = float(average_precision_score(y, proba))
    return metrics


def run_linear_probe(embeddings_root: Path, args) -> Dict[str, Any]:
    split_paths = {
        "train": embeddings_root / "train.npz",
        "val": embeddings_root / "val.npz",
        "test": embeddings_root / "test.npz",
    }
    z_train, y_train, _ = load_embedding_npz(split_paths["train"])

    class_weight = resolve_class_weight(args)
    logging.info(
        "Fitting logistic regression on train Z=%s (class_weight=%s)",
        z_train.shape,
        class_weight,
    )
    clf = fit_logistic_probe(
        z_train,
        y_train,
        class_weight=class_weight,
        max_iter=int(args.probe_max_iter),
        seed=int(args.seed),
    )

    tuning_mode = str(getattr(args, "threshold_tuning", "max_f1_val"))
    if tuning_mode == "max_f1_val":
        z_val, y_val, _ = load_embedding_npz(split_paths["val"])
        val_proba = clf.predict_proba(z_val)[:, 1]
        selected_threshold, val_f1_at_selection = tune_threshold_max_f1(y_val, val_proba)
        classification_threshold = {
            "method": "max_f1_on_val",
            "value": selected_threshold,
            "selected_on": "val",
        }
        logging.info(
            "Classification threshold selected on val: %.6f (val F1=%.4f, method=max_f1_on_val)",
            selected_threshold,
            val_f1_at_selection,
        )
    else:
        selected_threshold = 0.5
        classification_threshold = {
            "method": "fixed_0.5",
            "value": 0.5,
            "selected_on": None,
        }

    results: Dict[str, Any] = {
        "unique_name": args.unique_name,
        "embeddings_dir": str(embeddings_root),
        "class_weight": serialize_class_weight(class_weight),
        "probe_max_iter": int(args.probe_max_iter),
        "threshold_tuning": tuning_mode,
        "classification_threshold": classification_threshold,
        "splits_at_selected_threshold": {},
    }
    include_threshold_0_5_baseline = abs(selected_threshold - 0.5) > 1e-9
    if include_threshold_0_5_baseline:
        results["splits_at_threshold_0.5"] = {}

    meta_path = embeddings_root / "meta.json"
    if meta_path.is_file():
        with meta_path.open("r", encoding="utf-8") as f:
            results["extraction_meta"] = json.load(f)

    for split_name, path in split_paths.items():
        z, y, _ = load_embedding_npz(path)
        metrics_selected = evaluate_probe(
            clf, z, y, split_name, threshold=selected_threshold
        )
        results["splits_at_selected_threshold"][split_name] = metrics_selected
        logging.info(
            "aml_probe/linear/%s @ selected threshold %.4f: AUROC=%.4f AUPRC=%.4f F1=%.4f precision=%.4f recall=%.4f (n=%d)",
            split_name,
            selected_threshold,
            metrics_selected["auroc"],
            metrics_selected["auprc"],
            metrics_selected["f1"],
            metrics_selected["precision"],
            metrics_selected["recall"],
            int(metrics_selected["n"]),
        )
        if include_threshold_0_5_baseline:
            metrics_default = evaluate_probe(clf, z, y, split_name, threshold=0.5)
            results["splits_at_threshold_0.5"][split_name] = metrics_default
            logging.info(
                "aml_probe/linear/%s @ threshold 0.5000: F1=%.4f precision=%.4f recall=%.4f",
                split_name,
                metrics_default["f1"],
                metrics_default["precision"],
                metrics_default["recall"],
            )

    probe_output = getattr(args, "probe_output", None)
    out_path = Path(probe_output) if probe_output else embeddings_root / "probe_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logging.info("Wrote probe results to %s", out_path)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Linear probe on frozen GNN embeddings.")
    parser.add_argument(
        "--unique_name",
        type=str,
        required=True,
        help="Subfolder under --embeddings_dir (same as extraction / checkpoint name).",
    )
    parser.add_argument(
        "--embeddings_dir",
        type=str,
        default="embeddings",
        help="Root directory written by embedding_extraction.py.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Required when --class_weight model (reads w_ce1/w_ce2 from model_settings.json).",
    )
    parser.add_argument(
        "--class_weight",
        type=str,
        default="balanced",
        choices=["balanced", "none", "model", "explicit"],
        help="Logistic regression class weights (default balanced, Papagei-style).",
    )
    parser.add_argument(
        "--class_weight_pos",
        type=float,
        default=None,
        help="Positive class weight for --class_weight explicit; resolves to {0: 1.0, 1: value}.",
    )
    parser.add_argument("--probe_max_iter", type=int, default=1000)
    parser.add_argument(
        "--threshold_tuning",
        type=str,
        default="max_f1_val",
        choices=["max_f1_val", "fixed_0.5"],
        help="Set classification threshold for F1/precision/recall: tune on val (default) or fixed 0.5.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--probe_output",
        type=str,
        default=None,
        help="Optional path for probe_results.json (default: embeddings/{unique_name}/probe_results.json).",
    )
    parser.add_argument("--testing", action="store_true", help="Disable wandb.")
    args = parser.parse_args()

    logger_setup()
    set_seed(args.seed)

    embeddings_root = Path(args.embeddings_dir) / args.unique_name
    if not embeddings_root.is_dir():
        raise FileNotFoundError(
            f"Embeddings directory not found: {embeddings_root}. "
            "Run embedding_extraction.py first."
        )

    wandb.init(
        mode="disabled" if args.testing else "online",
        project="multi-gnn",
        config={
            "task": "aml_linear_probe",
            "unique_name": args.unique_name,
            "embeddings_dir": args.embeddings_dir,
            "class_weight": args.class_weight,
            "model": args.model,
            "probe_max_iter": args.probe_max_iter,
            "threshold_tuning": args.threshold_tuning,
        },
    )

    results = run_linear_probe(embeddings_root, args)

    log_payload = {}
    for split_name, metrics in results["splits_at_selected_threshold"].items():
        for key, value in metrics.items():
            if key == "n":
                continue
            log_payload[f"aml_probe/linear/{split_name}/{key}"] = value
    log_payload["aml_probe/classification_threshold"] = results["classification_threshold"]["value"]
    wandb.log(log_payload)
    wandb.finish()


if __name__ == "__main__":
    main()
