#!/usr/bin/env python3
"""
Label-efficiency linear probes on frozen embeddings (GCPAL / Papagei-style).

Fits logistic regression on stratified subsets of the **train** split, tunes the
classification threshold on the **full val** split (unchanged), and reports val/test
metrics. Compare curves across ``--unique_names`` without retraining the encoder.

Example (batch):

  python scripts/label_efficiency_probe.py \\
    --unique_names hi_morphology_global_20ep hi_morph_global_contrast_10ep \\
    --class_weight model --model gin --testing

Example (single run; merges into existing ``label_efficiency_summary.json``):

  python scripts/label_efficiency_probe.py \\
    --unique_name hi_morphology_global_clustering_20ep \\
    --class_weight model --model gin --testing
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import wandb
from sklearn.model_selection import train_test_split

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from linear_probe import (  # noqa: E402
    evaluate_probe,
    fit_logistic_probe,
    load_embedding_npz,
    resolve_class_weight,
    serialize_class_weight,
    tune_threshold_max_f1,
)
from util import logger_setup, set_seed


def subsample_train_indices(
    y_train: np.ndarray,
    fraction: float,
    seed: int,
) -> np.ndarray:
    """
    Stratified subset of train row indices.

    ``fraction >= 1.0`` returns all indices. For ``fraction < 1.0``, uses
    ``train_test_split(..., stratify=y)`` so both classes remain when possible.
    """
    n = int(y_train.shape[0])
    indices = np.arange(n, dtype=np.int64)
    if fraction >= 1.0:
        return indices

    y = y_train.reshape(-1).astype(np.int64)
    if len(np.unique(y)) < 2:
        raise ValueError("Train split has a single class; cannot stratify subsample.")

    n_keep = max(2, int(round(n * fraction)))
    n_keep = min(n_keep, n)
    if n_keep >= n:
        return indices

    selected, _ = train_test_split(
        indices,
        train_size=n_keep,
        stratify=y,
        random_state=seed,
    )
    return np.sort(np.asarray(selected, dtype=np.int64))


def run_label_efficiency_for_run(
    embeddings_root: Path,
    args,
    train_fractions: Sequence[float],
) -> Dict[str, Any]:
    split_paths = {
        "train": embeddings_root / "train.npz",
        "val": embeddings_root / "val.npz",
        "test": embeddings_root / "test.npz",
    }
    z_train_full, y_train_full, _ = load_embedding_npz(split_paths["train"])
    z_val, y_val, _ = load_embedding_npz(split_paths["val"])

    class_weight = resolve_class_weight(args)
    tuning_mode = str(getattr(args, "threshold_tuning", "max_f1_val"))
    seed = int(args.seed)
    max_iter = int(args.probe_max_iter)

    results: Dict[str, Any] = {
        "unique_name": args.unique_name,
        "embeddings_dir": str(embeddings_root),
        "class_weight": serialize_class_weight(class_weight),
        "probe_max_iter": max_iter,
        "threshold_tuning": tuning_mode,
        "seed": seed,
        "train_fractions": [float(f) for f in train_fractions],
        "runs": [],
    }

    meta_path = embeddings_root / "meta.json"
    if meta_path.is_file():
        with meta_path.open("r", encoding="utf-8") as f:
            results["extraction_meta"] = json.load(f)

    for fraction in train_fractions:
        frac = float(fraction)
        train_idx = subsample_train_indices(y_train_full, frac, seed=seed)
        z_sub = z_train_full[train_idx]
        y_sub = y_train_full[train_idx]

        logging.info(
            "Label-efficiency %s: fraction=%.4f train_n=%d positives=%d",
            args.unique_name,
            frac,
            z_sub.shape[0],
            int(y_sub.sum()),
        )

        clf = fit_logistic_probe(
            z_sub,
            y_sub,
            class_weight=class_weight,
            max_iter=max_iter,
            seed=seed,
            n_jobs=int(getattr(args, "probe_n_jobs", 1)),
        )

        if tuning_mode == "max_f1_val":
            val_proba = clf.predict_proba(z_val)[:, 1]
            selected_threshold, val_f1_at_selection = tune_threshold_max_f1(y_val, val_proba)
            classification_threshold = {
                "method": "max_f1_on_val",
                "value": selected_threshold,
                "selected_on": "val",
                "val_f1_at_selection": val_f1_at_selection,
            }
        else:
            selected_threshold = 0.5
            classification_threshold = {
                "method": "fixed_0.5",
                "value": 0.5,
                "selected_on": None,
            }

        run_record: Dict[str, Any] = {
            "train_fraction": frac,
            "train_labeled_n": int(z_sub.shape[0]),
            "train_positive_n": int(y_sub.sum()),
            "train_negative_n": int((y_sub == 0).sum()),
            "classification_threshold": classification_threshold,
            "val": evaluate_probe(clf, z_val, y_val, "val", threshold=selected_threshold),
        }

        z_test, y_test, _ = load_embedding_npz(split_paths["test"])
        run_record["test"] = evaluate_probe(
            clf, z_test, y_test, "test", threshold=selected_threshold
        )

        logging.info(
            "  fraction=%.4f test AUROC=%.4f F1=%.4f (threshold=%.4f)",
            frac,
            run_record["test"]["auroc"],
            run_record["test"]["f1"],
            selected_threshold,
        )
        results["runs"].append(run_record)

    out_path = embeddings_root / "label_efficiency_results.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logging.info("Wrote label-efficiency results to %s", out_path)
    return results


def load_summary_runs_by_name(summary_path: Path) -> Dict[str, Any]:
    """Load prior ``runs_by_name`` entries so incremental runs do not wipe the summary."""
    if not summary_path.is_file():
        return {}
    with summary_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    runs = payload.get("runs_by_name")
    if not isinstance(runs, dict):
        logging.warning("Ignoring malformed summary at %s", summary_path)
        return {}
    return runs


def parse_fractions(raw: str) -> List[float]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("Expected at least one train fraction.")
    fracs = sorted(set(float(p) for p in parts))
    for f in fracs:
        if f <= 0.0 or f > 1.0:
            raise ValueError(f"Train fraction must be in (0, 1]; got {f}")
    return fracs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Label-efficiency linear probe on frozen GNN embeddings.",
    )
    parser.add_argument(
        "--unique_name",
        type=str,
        default=None,
        help="Single run folder under --embeddings_dir.",
    )
    parser.add_argument(
        "--unique_names",
        type=str,
        nargs="*",
        default=None,
        help="Multiple runs to probe in one invocation (compare curves).",
    )
    parser.add_argument(
        "--embeddings_dir",
        type=str,
        default="embeddings",
        help="Root directory written by embedding_extraction.py.",
    )
    parser.add_argument(
        "--train_fractions",
        type=str,
        default="0.1,0.25,0.5,1.0",
        help="Comma-separated fractions of stratified train labels (default: 0.1,0.25,0.5,1.0).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Required when --class_weight model.",
    )
    parser.add_argument(
        "--class_weight",
        type=str,
        default="model",
        choices=["balanced", "none", "model"],
    )
    parser.add_argument("--probe_max_iter", type=int, default=5000)
    parser.add_argument(
        "--probe_n_jobs",
        type=int,
        default=1,
        help="sklearn LogisticRegression n_jobs (default 1 to avoid OOM on large train.npz).",
    )
    parser.add_argument(
        "--threshold_tuning",
        type=str,
        default="max_f1_val",
        choices=["max_f1_val", "fixed_0.5"],
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--testing", action="store_true", help="Disable wandb.")
    args = parser.parse_args()

    names: List[str] = []
    if args.unique_names:
        names.extend(args.unique_names)
    if args.unique_name:
        names.append(args.unique_name)
    if not names:
        parser.error("Provide --unique_name or --unique_names.")

    logger_setup()
    set_seed(args.seed)
    train_fractions = parse_fractions(args.train_fractions)

    wandb.init(
        mode="disabled" if args.testing else "online",
        project="multi-gnn",
        config={
            "task": "aml_label_efficiency_probe",
            "unique_names": names,
            "train_fractions": train_fractions,
            "class_weight": args.class_weight,
            "model": args.model,
            "probe_max_iter": args.probe_max_iter,
            "threshold_tuning": args.threshold_tuning,
        },
    )

    summary_path = Path(args.embeddings_dir) / "label_efficiency_summary.json"
    all_results: Dict[str, Any] = {
        "runs_by_name": load_summary_runs_by_name(summary_path),
    }
    for name in names:
        args.unique_name = name
        embeddings_root = Path(args.embeddings_dir) / name
        if not embeddings_root.is_dir():
            raise FileNotFoundError(
                f"Embeddings directory not found: {embeddings_root}. "
                "Run embedding_extraction.py first."
            )
        results = run_label_efficiency_for_run(embeddings_root, args, train_fractions)
        all_results["runs_by_name"][name] = results

        for run in results["runs"]:
            frac = run["train_fraction"]
            for split in ("val", "test"):
                metrics = run[split]
                prefix = f"aml_label_eff/{name}/frac_{frac:.4f}/{split}"
                for key, value in metrics.items():
                    if key == "n":
                        continue
                    wandb.log({f"{prefix}/{key}": value})

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    logging.info(
        "Wrote combined summary to %s (%d encoder(s) this run, %d total)",
        summary_path,
        len(names),
        len(all_results["runs_by_name"]),
    )
    wandb.finish()


if __name__ == "__main__":
    main()
