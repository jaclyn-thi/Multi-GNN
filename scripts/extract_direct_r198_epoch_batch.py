#!/usr/bin/env python3
"""Batch frozen R198 extract for DIRECT_H / DIRECT_H_TFMOE epoch checkpoints.

Loads Small-HI once, then extracts train+val only for each (run, epoch) cell.
Never touches test. Never trains. MoE discarded (encoder-only forward).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
import torch

from data_loading import get_data
from embedding_extraction import run_embedding_extraction
from util import create_parser, logger_setup, set_seed


EPOCHS = (1, 3, 5, 10)
RUNS = (
    "direct_h_infonce_10ep_seed2_sched",
    "direct_h_tfmoe_learned_alpha_10ep_seed2_sched",
)


def _base_args() -> argparse.Namespace:
    parser = create_parser()
    # embedding_extraction extras
    parser.add_argument("--embeddings_dir", type=str, default="embeddings")
    parser.add_argument("--random_init", action="store_true")
    parser.add_argument("--checkpoint_suffix", type=str, default="")
    parser.add_argument("--embeddings_subdir", type=str, default=None)
    parser.add_argument(
        "--representation_source",
        type=str,
        default="pre_embedding_3h",
        choices=["post_embedding", "pre_embedding_3h"],
    )
    parser.add_argument("--extract_splits", type=str, default="train,val")
    # Mimic formal train graph flags
    argv = [
        "--data",
        "Small-HI",
        "--model",
        "gin",
        "--batch_size",
        "8192",
        "--num_neighs",
        "100",
        "100",
        "--loader_num_workers",
        "16",
        "--reverse_mp",
        "--ego",
        "--ports",
        "--emlps",
        "--tds",
        "--correct_reverse_edge_features",
        "--seed",
        "2",
        "--tqdm",
        "--testing",
        "--direct_r198_infonce",
        "--representation_source",
        "pre_embedding_3h",
        "--extract_splits",
        "train,val",
        "--objective",
        "contrastive",
    ]
    args = parser.parse_args(argv)
    return args


def main() -> int:
    logger_setup()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--runs",
        type=str,
        default=",".join(RUNS),
        help="Comma-separated unique_name list",
    )
    ap.add_argument(
        "--epochs",
        type=str,
        default=",".join(str(e) for e in EPOCHS),
    )
    ap.add_argument("--embeddings_dir", type=str, default="embeddings")
    cli = ap.parse_args()
    runs = [r.strip() for r in cli.runs.split(",") if r.strip()]
    epochs = [int(x) for x in cli.epochs.split(",") if x.strip()]

    with open("data_config.json", "r", encoding="utf-8") as f:
        data_config = json.load(f)

    args = _base_args()
    args.embeddings_dir = cli.embeddings_dir
    set_seed(int(args.seed))

    logging.info("Retrieving data once for batch R198 extract (train+val only)")
    t0 = time.perf_counter()
    tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(args, data_config)
    logging.info("Retrieved data in %.2fs", time.perf_counter() - t0)

    # Integrity: refuse if test would be selected
    if "test" in {s.strip() for s in args.extract_splits.split(",")}:
        raise RuntimeError("Refusing test extraction for DIRECT_H locked val analysis")

    manifest = []
    for run in runs:
        for ep in epochs:
            suffix = f"_epoch{ep:02d}"
            ckpt = Path(data_config["paths"]["model_to_save"]) / f"checkpoint_{run}{suffix}.tar"
            if not ckpt.is_file():
                raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
            sub = f"{run}_epoch{ep:02d}"
            args.unique_name = run
            args.checkpoint_suffix = suffix
            args.embeddings_subdir = sub
            logging.info("=== Extract R198 run=%s epoch=%s -> embeddings/%s/pre_embedding_3h ===", run, ep, sub)
            out = run_embedding_extraction(
                tr_data, val_data, te_data, tr_inds, val_inds, te_inds, args, data_config
            )
            meta_path = Path(out) / "meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
            import numpy as np

            tr = np.load(Path(out) / "train.npz")
            va = np.load(Path(out) / "val.npz")
            zdim = int(tr["Z"].shape[1])
            assert zdim == 198, f"Expected R198, got dim={zdim} keys={list(tr.files)}"
            assert not (Path(out) / "test.npz").exists(), "test.npz must not exist"
            cell = {
                "run": run,
                "epoch": ep,
                "checkpoint": str(ckpt),
                "embedding_dir": str(out),
                "dim": zdim,
                "n_train": int(tr["Z"].shape[0]),
                "n_val": int(va["Z"].shape[0]),
                "representation_source": "pre_embedding_3h",
                "direct_r198_infonce": True,
                "test_extracted": False,
                "meta_checkpoint_path": meta.get("checkpoint_path"),
            }
            manifest.append(cell)
            logging.info("OK cell %s", cell)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    out_manifest = Path("results/diagnostics/direct_h_tfmoe_scheduled_val_analysis/extract_manifest.json")
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text(json.dumps({"cells": manifest, "test_evaluated": False}, indent=2) + "\n")
    logging.info("Wrote %s (%d cells)", out_manifest, len(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
