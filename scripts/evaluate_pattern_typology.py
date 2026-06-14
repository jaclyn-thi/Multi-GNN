#!/usr/bin/env python3
"""
Post-hoc laundering typology diagnostics on test-set probe predictions.

Fits the same linear probe convention as ``linear_probe.py``, then evaluates
recall by ``pattern_type`` using auxiliary metadata on ``te_data`` only.

Does not modify training, inference, splits, labels, or graph construction.

Example (cluster: ``module load miniforge && conda activate multignn``)::

  python scripts/evaluate_pattern_typology.py \\
    --unique_name my_pretrain \\
    --data Small-HI \\
    --model gin \\
    --testing \\
    --by_attempt

  # Fair cross-run comparison: shared fixed threshold (matches linear_probe --threshold_tuning fixed_0.5)
  python scripts/evaluate_pattern_typology.py \\
    --unique_name my_pretrain \\
    --data Small-HI \\
    --model gin \\
    --testing \\
    --by_attempt \\
    --threshold_tuning fixed_0.5 \\
    --output_dir results/diagnostics/my_pretrain_thr0.5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data_loading import get_data  # noqa: E402
from linear_probe import (  # noqa: E402
    fit_logistic_probe,
    load_embedding_npz,
    resolve_class_weight,
    tune_threshold_max_f1,
)
from pattern_diagnostics import (  # noqa: E402
    evaluate_pattern_typology_diagnostics,
    log_pattern_typology_diagnostics,
    write_pattern_typology_diagnostics,
)
from util import create_parser, logger_setup, set_seed  # noqa: E402


def _fit_test_predictions(embeddings_root: Path, args) -> tuple:
    z_train, y_train, _ = load_embedding_npz(embeddings_root / "train.npz")
    z_val, y_val, _ = load_embedding_npz(embeddings_root / "val.npz")
    z_test, y_test, edge_id_test = load_embedding_npz(embeddings_root / "test.npz")

    class_weight = resolve_class_weight(args)
    clf = fit_logistic_probe(
        z_train,
        y_train,
        class_weight=class_weight,
        max_iter=int(args.probe_max_iter),
        seed=int(args.seed),
    )

    tuning_mode = str(getattr(args, "threshold_tuning", "max_f1_val"))
    if tuning_mode == "max_f1_val":
        val_proba = clf.predict_proba(z_val)[:, 1]
        selected_threshold, _ = tune_threshold_max_f1(y_val, val_proba)
    else:
        selected_threshold = 0.5

    test_proba = clf.predict_proba(z_test)[:, 1]
    test_pred = (test_proba >= selected_threshold).astype("int64")
    return y_test, test_pred, test_proba, edge_id_test, selected_threshold


def main() -> None:
    base_parser = create_parser()
    parser = argparse.ArgumentParser(
        parents=[base_parser],
        description="Laundering typology diagnostics on test probe predictions.",
        conflict_handler="resolve",
    )
    parser.add_argument(
        "--unique_name",
        type=str,
        required=True,
        help="Embedding subfolder under --embeddings_dir (same as linear_probe).",
    )
    parser.add_argument(
        "--embeddings_dir",
        type=str,
        default="embeddings",
        help="Root directory written by embedding_extraction.py.",
    )
    parser.add_argument(
        "--class_weight",
        type=str,
        default="balanced",
        choices=["balanced", "none", "model"],
        help="Logistic regression class weights (default balanced).",
    )
    parser.add_argument("--probe_max_iter", type=int, default=1000)
    parser.add_argument(
        "--threshold_tuning",
        type=str,
        default="max_f1_val",
        choices=["max_f1_val", "fixed_0.5"],
        help="Classification threshold selection (same as linear_probe.py).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Diagnostics output directory (default: results/diagnostics/{unique_name}).",
    )
    parser.add_argument(
        "--by_attempt",
        action="store_true",
        help="Also compute per-attempt_id support and recall.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="data_config.json",
        help="Path to data_config.json.",
    )
    args = parser.parse_args()

    logger_setup()
    set_seed(args.seed)

    embeddings_root = Path(args.embeddings_dir) / args.unique_name
    if not embeddings_root.is_dir():
        raise FileNotFoundError(
            f"Embeddings directory not found: {embeddings_root}. "
            "Run embedding_extraction.py first."
        )

    y_test, test_pred, test_proba, edge_id_test, threshold = _fit_test_predictions(
        embeddings_root, args
    )
    logging.info("Using classification threshold %.6f for typology diagnostics", threshold)

    with open(args.config, encoding="utf-8") as handle:
        data_config = json.load(handle)

    data_args = SimpleNamespace(
        data=args.data,
        model=args.model,
        reverse_mp=bool(args.reverse_mp),
        ports=bool(args.ports),
        tds=bool(args.tds),
        ego=bool(args.ego),
        load_pattern_metadata=True,
        pattern_metadata=None,
    )
    _, _, te_data, _, _, _ = get_data(data_args, data_config)

    if not getattr(te_data, "pattern_metadata_by_edge_id", None):
        raise RuntimeError(
            "Pattern metadata is empty on te_data. Ensure "
            "aml-data/{data}/laundering_attempt_metadata.csv exists or pass "
            "--load_pattern_metadata."
        )

    diagnostics = evaluate_pattern_typology_diagnostics(
        y_test,
        test_pred,
        te_data,
        edge_id_test,
        y_scores=test_proba,
        include_attempt_id=bool(args.by_attempt),
    )
    diagnostics["threshold_tuning"] = str(args.threshold_tuning)
    diagnostics["classification_threshold"] = float(threshold)
    diagnostics["unique_name"] = args.unique_name
    diagnostics["data"] = args.data
    diagnostics["embeddings_dir"] = str(embeddings_root)

    log_pattern_typology_diagnostics(diagnostics)

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("results") / "diagnostics" / args.unique_name
    )
    written = write_pattern_typology_diagnostics(
        diagnostics,
        output_dir,
        write_attempt_csv=bool(args.by_attempt),
    )
    for label, path in written.items():
        logging.info("Wrote %s: %s", label, path)


if __name__ == "__main__":
    main()
