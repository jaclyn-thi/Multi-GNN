#!/usr/bin/env python3
"""Compare full-readout-selected seed R198 vs seed-only R198 on identical batches.

Requires max_abs_diff <= 1e-5, aligned IDs, dim=198, finite, same checkpoint hash,
model.eval() + torch.inference_mode().
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch_geometric.nn import to_hetero

from data_loading import get_data
from direct_r198.seed_readout import forward_seed_r198_hetero, full_fwd_r198_for_compare
from embedding_extraction import _build_model_config
from training import get_model
from train_util import (
    FORWARD_EDGE_TYPE,
    AddEgoIds,
    add_arange_ids,
    attach_edge_id_from_batch,
    checkpoint_path,
    get_hetero_seed_edge_ids,
    get_loaders,
    load_checkpoint_weights,
)
from util import create_parser, logger_setup, set_seed

ROOT = Path(__file__).resolve().parents[1]
FWD = FORWARD_EDGE_TYPE
TOL = 1e-5


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_args(run: str, suffix: str, seed: int = 2):
    parser = create_parser()
    args = parser.parse_args(
        [
            "--data", "Small-HI", "--model", "gin",
            "--batch_size", "1024",
            "--num_neighs", "100", "100",
            "--loader_num_workers", "0",
            "--reverse_mp", "--ego", "--ports", "--emlps", "--tds",
            "--correct_reverse_edge_features",
            "--seed", str(seed), "--tqdm", "--testing",
            "--skip_test_eval",
            "--direct_r198_infonce",
            "--objective", "contrastive",
            "--unique_name", run,
        ]
    )
    args.checkpoint_suffix = suffix
    return args


@torch.inference_mode()
def compare_batches(model, loader, device, n_batches: int) -> dict:
    model.eval()
    max_abs = 0.0
    n_rows = 0
    n_checked = 0
    for bi, batch in enumerate(loader):
        if bi >= n_batches:
            break
        seed_ids = get_hetero_seed_edge_ids(batch, loader.data)
        attach_edge_id_from_batch(batch, loader.data)
        batch = batch.to(device)
        seed_ids = seed_ids.to(device)

        z_seed, id_seed, stats = forward_seed_r198_hetero(model, batch, seed_ids)
        z_full, eid_full = full_fwd_r198_for_compare(model, batch)
        # Select full-readout rows for the same seed IDs (sorted unique)
        keep = torch.isin(eid_full, id_seed)
        z_ref = z_full[keep]
        id_ref = eid_full[keep]
        order = torch.argsort(id_ref)
        z_ref = z_ref[order]
        id_ref = id_ref[order]

        if not torch.equal(id_seed, id_ref):
            raise RuntimeError(
                f"batch {bi}: ID misalignment seed={id_seed[:5].tolist()} ref={id_ref[:5].tolist()}"
            )
        if z_seed.shape[1] != 198 or z_ref.shape[1] != 198:
            raise RuntimeError(f"dim mismatch {z_seed.shape} vs {z_ref.shape}")
        if not torch.isfinite(z_seed).all() or not torch.isfinite(z_ref).all():
            raise RuntimeError("non-finite values")
        diff = (z_seed.float() - z_ref.float()).abs().max().item()
        max_abs = max(max_abs, float(diff))
        n_rows += int(z_seed.shape[0])
        n_checked += 1
        logging.info(
            "batch=%s seeds=%s max_abs_diff=%.3e full_fwd=%s seed_r198_bytes=%s",
            bi,
            int(z_seed.shape[0]),
            diff,
            stats.get("n_fwd_edges"),
            stats.get("bytes_r198_seed"),
        )
        del batch, z_seed, z_full, z_ref
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {
        "n_batches": n_checked,
        "n_seed_rows": n_rows,
        "max_abs_diff": max_abs,
        "tol": TOL,
        "pass": bool(max_abs <= TOL),
    }


def main() -> int:
    logger_setup()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="direct_h_infonce_10ep_seed2_sched")
    ap.add_argument("--epoch", type=int, default=1)
    ap.add_argument("--n_batches", type=int, default=3)
    ap.add_argument("--out", type=str, default="")
    cli = ap.parse_args()

    suffix = f"_epoch{cli.epoch:02d}"
    with open(ROOT / "data_config.json", encoding="utf-8") as f:
        data_config = json.load(f)
    ckpt = Path(checkpoint_path(data_config, cli.run, finetuned=False, suffix=suffix))
    ckpt_sha = _sha256(ckpt)
    args = build_args(cli.run, suffix)
    set_seed(2)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(args, data_config)
    add_arange_ids([tr_data, val_data, te_data])
    transform = AddEgoIds() if args.ego else None
    config = _build_model_config(args)
    sample_args = SimpleNamespace(**vars(args))
    sample_args.loader_num_workers = 0
    sample_loader, _, _ = get_loaders(
        tr_data, val_data, te_data, tr_inds, val_inds, te_inds,
        transform, sample_args, train_shuffle=False,
    )
    sample_batch = next(iter(sample_loader))
    del sample_loader
    model = get_model(sample_batch, config, args)
    del sample_batch
    model = to_hetero(model, tr_data.metadata(), aggr="mean")
    model.bypass_embedding_head = True
    load_checkpoint_weights(model, device, args, data_config)
    model.to(device)
    model.eval()

    tr_loader, _, _ = get_loaders(
        tr_data, val_data, te_data, tr_inds, val_inds, te_inds,
        transform, args, train_shuffle=False,
    )
    cmp = compare_batches(model, tr_loader, device, cli.n_batches)
    report = {
        "ok": cmp["pass"],
        "checkpoint": str(ckpt),
        "checkpoint_sha256": ckpt_sha,
        "model_eval": True,
        "inference_mode": True,
        "dim": 198,
        "tolerance": TOL,
        "tolerance_justification": (
            "1e-5 matches existing tests/test_direct_r198_seed_readout.py "
            "torch.allclose(..., atol=1e-5) for seed-only vs indexed full readout."
        ),
        **cmp,
    }
    out = Path(cli.out) if cli.out else (
        ROOT / "results/diagnostics/direct_h_tfmoe_scheduled_val_analysis/seed_only_equivalence.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    logging.info("Wrote %s pass=%s max_abs_diff=%.3e", out, report["ok"], report["max_abs_diff"])
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
