"""
Phase 5b: Papagei-style linear probing on frozen embeddings (Phase 5a ``.npz`` files).

Fits sklearn logistic regression on train embeddings and reports AUROC + F1 on
train / val / test.

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
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

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
    raise ValueError(f"Unsupported --class_weight {args.class_weight!r}")


def fit_logistic_probe(
    z_train: np.ndarray,
    y_train: np.ndarray,
    class_weight: Optional[Any],
    max_iter: int,
    seed: int,
) -> LogisticRegression:
    clf = LogisticRegression(
        class_weight=class_weight,
        max_iter=max_iter,
        random_state=seed,
        solver="lbfgs",
        n_jobs=-1,
    )
    clf.fit(z_train, y_train)
    return clf


def evaluate_probe(
    clf: LogisticRegression,
    z: np.ndarray,
    y: np.ndarray,
    split_name: str,
) -> Dict[str, float]:
    if z.shape[0] == 0:
        logging.warning("linear probe %s: empty split", split_name)
        return {
            "auroc": float("nan"),
            "f1": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "n": 0.0,
            "positive_rate": float("nan"),
        }

    proba = clf.predict_proba(z)[:, 1]
    pred = (proba >= 0.5).astype(np.int64)
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
            "linear probe %s: only one class present; AUROC undefined", split_name
        )
        metrics["auroc"] = float("nan")
    else:
        metrics["auroc"] = float(roc_auc_score(y, proba))
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

    results: Dict[str, Any] = {
        "unique_name": args.unique_name,
        "embeddings_dir": str(embeddings_root),
        "class_weight": class_weight if isinstance(class_weight, str) else dict(class_weight),
        "probe_max_iter": int(args.probe_max_iter),
        "splits": {},
    }

    meta_path = embeddings_root / "meta.json"
    if meta_path.is_file():
        with meta_path.open("r", encoding="utf-8") as f:
            results["extraction_meta"] = json.load(f)

    for split_name, path in split_paths.items():
        z, y, _ = load_embedding_npz(path)
        metrics = evaluate_probe(clf, z, y, split_name)
        results["splits"][split_name] = metrics
        logging.info(
            "aml_probe/linear/%s: AUROC=%.4f F1=%.4f precision=%.4f recall=%.4f (n=%d)",
            split_name,
            metrics["auroc"],
            metrics["f1"],
            metrics["precision"],
            metrics["recall"],
            int(metrics["n"]),
        )

    val_f1 = results["splits"]["val"]["f1"]
    te_f1 = results["splits"]["test"]["f1"]
    results["best_test_f1_at_val"] = {
        "note": "sklearn probe has no epoch loop; reports test F1 for this single fit",
        "val_f1": val_f1,
        "test_f1": te_f1,
    }

    out_path = embeddings_root / "probe_results.json"
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
        choices=["balanced", "none", "model"],
        help="Logistic regression class weights (default balanced, Papagei-style).",
    )
    parser.add_argument("--probe_max_iter", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
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
        },
    )

    results = run_linear_probe(embeddings_root, args)

    log_payload = {}
    for split_name, metrics in results["splits"].items():
        for key, value in metrics.items():
            if key == "n":
                continue
            log_payload[f"aml_probe/linear/{split_name}/{key}"] = value
    wandb.log(log_payload)
    wandb.finish()


if __name__ == "__main__":
    main()
