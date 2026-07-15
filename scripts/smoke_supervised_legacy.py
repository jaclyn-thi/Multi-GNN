#!/usr/bin/env python3
"""Tiny CPU smoke test for the supervised legacy/embedding heads.

Generates a small synthetic AMLWorld-format dataset, runs a 2-epoch supervised GINe run
through the real get_data -> train_gnn path, and asserts that:
  - both checkpoints (checkpoint_last.tar, checkpoint_best_val_f1.tar) are written,
  - the flat compatibility checkpoint is written,
  - the per-epoch history JSON is written with the expected fields,
  - the best checkpoint metadata is correct,
  - the best checkpoint reloads into a freshly built model.

Usage:
  python scripts/smoke_supervised_legacy.py --supervised_head legacy
  python scripts/smoke_supervised_legacy.py --supervised_head embedding
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data_loading import get_data  # noqa: E402
from training import get_model, train_gnn  # noqa: E402
from train_util import (  # noqa: E402
    add_arange_ids,
    get_loaders,
    supervised_epoch_history_path,
    supervised_summary_json_path,
    supervised_summary_md_path,
)
from util import create_parser, logger_setup, set_seed  # noqa: E402


def _write_synthetic_csv(path: Path, *, seed: int = 0, n_nodes: int = 40,
                         n_days: int = 6, edges_per_day: int = 120) -> None:
    rng = np.random.default_rng(seed)
    rows = []
    for day in range(n_days):
        for _ in range(edges_per_day):
            src, dst = rng.integers(0, n_nodes, size=2)
            ts = day * 86400 + int(rng.integers(0, 86400))
            amount = float(rng.uniform(1.0, 10000.0))
            # ~10% positives, guaranteed in every day bucket.
            label = int(rng.random() < 0.10)
            rows.append(
                {
                    "from_id": int(src),
                    "to_id": int(dst),
                    "Timestamp": ts,
                    "Amount Sent": amount,
                    "Sent Currency": int(rng.integers(0, 3)),
                    "Amount Received": amount,
                    "Received Currency": int(rng.integers(0, 3)),
                    "Payment Format": int(rng.integers(0, 4)),
                    "Is Laundering": label,
                }
            )
        # ensure at least a couple positives per day
        rows[-1]["Is Laundering"] = 1
        rows[-2]["Is Laundering"] = 1
    df = pd.DataFrame(rows).sort_values("Timestamp").reset_index(drop=True)
    df.insert(0, "EdgeID", np.arange(len(df), dtype=np.int64))
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _build_args(data_root: Path, supervised_head: str, run_name: str) -> argparse.Namespace:
    parser = create_parser()
    return parser.parse_args(
        [
            "--data", "SmokeTiny",
            "--model", "gin",
            "--objective", "supervised",
            "--supervised_head", supervised_head,
            "--n_epochs", "2",
            "--batch_size", "512",
            "--num_neighs", "50", "50",
            "--loader_num_workers", "0",
            "--seed", "1",
            "--testing",
            "--save_model",
            "--unique_name", run_name,
        ]
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--supervised_head", choices=["legacy", "embedding"], default="legacy")
    ap.add_argument("--keep_artifacts", action="store_true",
                    help="Keep the results/notes summary + history artifacts (default: clean up).")
    opts = ap.parse_args()

    logger_setup()
    run_name = f"smoke_{opts.supervised_head}_gine_tiny"
    tmp = Path(tempfile.mkdtemp(prefix="smoke_supervised_"))
    aml_root = tmp / "aml-data"
    saved_models = tmp / "saved-models"
    saved_models.mkdir(parents=True, exist_ok=True)
    _write_synthetic_csv(aml_root / "SmokeTiny" / "formatted_transactions.csv")

    data_config = {
        "paths": {
            "aml_data": str(aml_root),
            "model_to_load": str(saved_models),
            "model_to_save": str(saved_models),
        }
    }

    args = _build_args(aml_root, opts.supervised_head, run_name)
    set_seed(args.seed)

    tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(args, data_config)
    train_gnn(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, args, data_config)

    # --- assertions -------------------------------------------------------
    run_dir = saved_models / run_name
    last_path = run_dir / "checkpoint_last.tar"
    best_path = run_dir / "checkpoint_best_val_f1.tar"
    flat_path = saved_models / f"checkpoint_{run_name}.tar"
    history_path = supervised_epoch_history_path(args)
    summary_json = supervised_summary_json_path(args)
    summary_md = supervised_summary_md_path(args)

    problems = []
    for label, p in (
        ("checkpoint_last", last_path),
        ("checkpoint_best_val_f1", best_path),
        ("flat_compat_checkpoint", flat_path),
        ("epoch_history", history_path),
        ("summary_json", summary_json),
        ("summary_md", summary_md),
    ):
        if not p.is_file():
            problems.append(f"missing {label}: {p}")

    if not problems:
        history = json.loads(history_path.read_text())
        required_fields = {
            "epoch", "train_loss", "validation_minority_f1_argmax", "test_minority_f1_argmax",
            "validation_precision_argmax", "validation_recall_argmax", "test_precision_argmax",
            "test_recall_argmax", "validation_auroc", "validation_auprc", "test_auroc",
            "test_auprc", "learning_rate",
        }
        if len(history["epochs"]) != 2:
            problems.append(f"expected 2 epochs, got {len(history['epochs'])}")
        else:
            missing = required_fields - set(history["epochs"][0])
            if missing:
                problems.append(f"history missing fields: {sorted(missing)}")
        if history.get("decision_rule") != "argmax over two-class logits":
            problems.append("history decision_rule mismatch")

        best = torch.load(best_path, map_location="cpu", weights_only=False)
        for key in ("model_state_dict", "optimizer_state_dict", "seed", "args", "config",
                    "selected_epoch", "best_validation_f1", "test_f1_at_selected_epoch",
                    "selection_metric", "decision_rule", "supervised_head"):
            if key not in best:
                problems.append(f"best checkpoint missing key: {key}")
        if best.get("supervised_head") != opts.supervised_head:
            problems.append("best checkpoint supervised_head mismatch")
        if best.get("selection_metric") != "validation_minority_f1":
            problems.append("best checkpoint selection_metric mismatch")

        # reload best checkpoint into a fresh model
        sample = next(iter(get_loaders(
            tr_data, val_data, te_data, tr_inds, val_inds, te_inds, None, args, train_shuffle=False
        )[0]))
        from types import SimpleNamespace
        cfg = SimpleNamespace(**best["config"])
        cfg.n_gnn_layers = int(round(best["config"]["n_gnn_layers"]))
        fresh = get_model(sample, cfg, args)
        fresh.load_state_dict(best["model_state_dict"])
        print(f"Reloaded best checkpoint into fresh {opts.supervised_head} model OK.")

    if not opts.keep_artifacts:
        for p in (history_path, summary_json, summary_md):
            if p.is_file():
                p.unlink()
    shutil.rmtree(tmp, ignore_errors=True)

    if problems:
        print("SMOKE TEST FAILED:")
        for pr in problems:
            print("  -", pr)
        sys.exit(1)
    print(f"SMOKE TEST PASSED ({opts.supervised_head} head).")


if __name__ == "__main__":
    main()
