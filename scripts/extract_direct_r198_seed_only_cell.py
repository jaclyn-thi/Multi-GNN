#!/usr/bin/env python3
"""One-checkpoint seed-only R198 extract (DIRECT_H / DIRECT_H_TFMOE).

Host-memory safe:
  - single (run, epoch, splits) cell per process
  - loader_num_workers=0
  - --skip_test_eval (no test graph materialization / te_inds emptied)
  - seed-only R198 via forward_seed_r198_hetero (no full-subgraph R198)
  - streaming write into memmap; no retained chunk lists of graph tensors
  - atomic rename into final .npz

Never trains. Never reads/writes test.npz.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
import resource
import shutil
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Set

import numpy as np
import torch
import tqdm
from torch_geometric.nn import to_hetero

from data_loading import get_data
from direct_r198.seed_readout import forward_seed_r198_hetero
from embedding_extraction import _build_model_config
from training import get_model
from train_util import (
    FORWARD_EDGE_TYPE,
    AddEgoIds,
    add_arange_ids,
    attach_edge_id_from_batch,
    checkpoint_path,
    expected_seed_edge_ids,
    get_loaders,
    load_checkpoint_weights,
    log_seed_coverage,
)
from util import create_parser, logger_setup, set_seed

ROOT = Path(__file__).resolve().parents[1]
FWD = FORWARD_EDGE_TYPE


def _rss_gb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024.0 * 1024.0)


def _gpu_peak_gb() -> Optional[float]:
    if not torch.cuda.is_available():
        return None
    return float(torch.cuda.max_memory_allocated()) / (1024.0 ** 3)


def _file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _atomic_savez(path: Path, *, Z: np.ndarray, y: np.ndarray, edge_id: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial.npz")
    if tmp.exists():
        tmp.unlink()
    np.savez_compressed(tmp, Z=Z.astype(np.float32, copy=False), y=y, edge_id=edge_id)
    written = tmp if tmp.is_file() else Path(str(tmp) + ".npz")
    os.replace(written, path)


def _release() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_args(*, unique_name: str, checkpoint_suffix: str, batch_size: int, seed: int):
    parser = create_parser()
    argv = [
        "--data", "Small-HI", "--model", "gin",
        "--batch_size", str(batch_size),
        "--num_neighs", "100", "100",
        "--loader_num_workers", "0",
        "--reverse_mp", "--ego", "--ports", "--emlps", "--tds",
        "--correct_reverse_edge_features",
        "--seed", str(seed), "--tqdm", "--testing",
        "--skip_test_eval",
        "--direct_r198_infonce",
        "--objective", "contrastive",
        "--unique_name", unique_name,
    ]
    args = parser.parse_args(argv)
    args.checkpoint_suffix = checkpoint_suffix
    return args


@torch.inference_mode()
def extract_split_seed_only(
    *,
    loader,
    split_inds: torch.Tensor,
    model: torch.nn.Module,
    device: torch.device,
    args,
    out_npz: Path,
    expected_n: int,
    split_name: str,
) -> Dict[str, Any]:
    out_npz = Path(out_npz)
    work = out_npz.parent / f".work_{out_npz.stem}"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)

    cap = int(max(expected_n, 1) + 8192)
    z_path = work / "Z.npy"
    y_path = work / "y.npy"
    id_path = work / "edge_id.npy"
    z_mm = np.lib.format.open_memmap(str(z_path), mode="w+", dtype=np.float32, shape=(cap, 198))
    y_mm = np.lib.format.open_memmap(str(y_path), mode="w+", dtype=np.int64, shape=(cap,))
    id_mm = np.lib.format.open_memmap(str(id_path), mode="w+", dtype=np.int64, shape=(cap,))

    seen: Set[int] = set()
    n_written = 0
    n_dup = 0
    n_batches = 0
    first_stats: Optional[Dict[str, Any]] = None

    model.eval()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    split_inds_cpu = split_inds.detach().cpu().long().view(-1)
    for batch in tqdm.tqdm(loader, disable=not args.tqdm, desc=f"seed-only {split_name}"):
        n_batches += 1
        # Resolve like full extract: input_id indexes the seed list (split_inds),
        # not the global edge store. Using raw input_id as a graph index maps val
        # seeds onto the train portion of train∪val (INVALID val metrics).
        input_id = batch[FWD].input_id.long().view(-1).cpu()
        batch_edge_inds = split_inds_cpu[input_id]
        seed_edge_ids = (
            loader.data[FWD].edge_attr.detach().cpu()[batch_edge_inds, 0].long().clone()
        )
        y_seed = loader.data[FWD].y[batch_edge_inds].long().cpu().numpy()
        id_seed = seed_edge_ids.detach().cpu().numpy().astype(np.int64)

        attach_edge_id_from_batch(batch, loader.data)
        batch = batch.to(device)
        seed_edge_ids_dev = seed_edge_ids.to(device)

        z_seed, ids_out, stats = forward_seed_r198_hetero(model, batch, seed_edge_ids_dev)
        if first_stats is None:
            first_stats = {k: (int(v) if isinstance(v, (int, np.integer)) else v) for k, v in stats.items()}

        z_cpu = z_seed.detach().cpu().contiguous().numpy().astype(np.float32, copy=False)
        ids_cpu = ids_out.detach().cpu().numpy().astype(np.int64)
        id_to_y = {int(i): int(y) for i, y in zip(id_seed.tolist(), y_seed.tolist())}
        y_aligned = np.fromiter((id_to_y[int(i)] for i in ids_cpu.tolist()), dtype=np.int64)

        if z_cpu.shape[1] != 198:
            raise RuntimeError(f"expected dim 198, got {z_cpu.shape}")
        if not np.isfinite(z_cpu).all():
            raise RuntimeError("non-finite R198 values")

        for row in range(z_cpu.shape[0]):
            eid = int(ids_cpu[row])
            if eid in seen:
                n_dup += 1
                continue
            if n_written >= cap:
                raise RuntimeError(f"memmap capacity {cap} exceeded")
            seen.add(eid)
            z_mm[n_written] = z_cpu[row]
            y_mm[n_written] = int(y_aligned[row])
            id_mm[n_written] = eid
            n_written += 1

        del batch, z_seed, ids_out, z_cpu, ids_cpu, y_aligned, seed_edge_ids, seed_edge_ids_dev
        _release()

    z_mm.flush()
    y_mm.flush()
    id_mm.flush()
    Z = np.asarray(z_mm[:n_written], dtype=np.float32)
    y = np.asarray(y_mm[:n_written], dtype=np.int64)
    edge_id = np.asarray(id_mm[:n_written], dtype=np.int64)
    del z_mm, y_mm, id_mm
    _release()

    if edge_id.size != np.unique(edge_id).size:
        raise RuntimeError("duplicate edge_id after streaming dedupe")
    if Z.shape != (n_written, 198):
        raise RuntimeError(f"bad Z shape {Z.shape}")
    if not np.isfinite(Z).all():
        raise RuntimeError("non-finite Z before save")

    expected = expected_seed_edge_ids(loader.data, split_inds, hetero=True)
    log_seed_coverage(torch.from_numpy(edge_id), expected, split_name)

    _atomic_savez(out_npz, Z=Z, y=y, edge_id=edge_id)
    shutil.rmtree(work, ignore_errors=True)
    del Z, y, edge_id
    _release()

    return {
        "n_rows": int(n_written),
        "n_dup_skipped": int(n_dup),
        "n_batches": int(n_batches),
        "first_batch_stats": first_stats,
        "rss_gb_max": _rss_gb(),
        "gpu_peak_gb": _gpu_peak_gb(),
        "dim": 198,
        "path": str(out_npz),
    }


def main() -> int:
    logger_setup()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--epoch", type=int, required=True)
    ap.add_argument("--splits", type=str, default="train,val")
    ap.add_argument("--batch_size", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--embeddings_subdir", type=str, default="")
    ap.add_argument("--force", action="store_true")
    cli = ap.parse_args()

    splits = [s.strip().lower() for s in cli.splits.split(",") if s.strip()]
    if not splits or any(s not in ("train", "val") for s in splits):
        raise SystemExit("Only train/val allowed; test is forbidden")

    suffix = f"_epoch{int(cli.epoch):02d}"
    sub = cli.embeddings_subdir or f"{cli.run}_epoch{int(cli.epoch):02d}"
    out_dir = ROOT / "embeddings" / sub / "pre_embedding_3h"
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "test.npz").exists():
        raise RuntimeError(f"Refusing: test.npz present at {out_dir}")

    with open(ROOT / "data_config.json", encoding="utf-8") as f:
        data_config = json.load(f)
    ckpt = Path(checkpoint_path(data_config, cli.run, finetuned=False, suffix=suffix))
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)
    ckpt_sha = _file_sha256(ckpt)

    # Skip splits that already look complete unless --force
    pending = []
    for sp in splits:
        p = out_dir / f"{sp}.npz"
        if p.is_file() and not cli.force:
            logging.info("SKIP existing %s (use --force to overwrite)", p)
        else:
            pending.append(sp)
    if not pending:
        logging.info("Nothing to extract; writing/updating meta only if missing")
        meta_path = out_dir / "meta.json"
        if not meta_path.is_file():
            meta_path.write_text(
                json.dumps(
                    {
                        "unique_name": sub,
                        "source_unique_name": cli.run,
                        "checkpoint_path": str(ckpt),
                        "checkpoint_sha256": ckpt_sha,
                        "checkpoint_epoch": int(cli.epoch),
                        "representation_source": "pre_embedding_3h",
                        "seed_only_r198": True,
                        "dim": 198,
                        "test_extracted": False,
                        "skip_test_eval": True,
                        "loader_num_workers": 0,
                    },
                    indent=2,
                )
                + "\n"
            )
        print(json.dumps({"ok": True, "skipped_all": True, "out_dir": str(out_dir)}))
        return 0

    args = build_args(
        unique_name=cli.run,
        checkpoint_suffix=suffix,
        batch_size=cli.batch_size,
        seed=cli.seed,
    )
    set_seed(int(args.seed))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logging.info(
        "=== seed-only R198 cell run=%s epoch=%s pending=%s rss=%.2fGiB ===",
        cli.run,
        cli.epoch,
        pending,
        _rss_gb(),
    )
    logging.info("checkpoint=%s sha256=%s", ckpt, ckpt_sha)

    t0 = time.perf_counter()
    tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(args, data_config)
    if int(te_inds.numel()) != 0:
        raise RuntimeError("te_inds non-empty despite skip_test_eval")
    logging.info("Retrieved data in %.2fs rss=%.2fGiB", time.perf_counter() - t0, _rss_gb())

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
    _release()

    if not bool(getattr(model, "bypass_embedding_head", False)):
        raise RuntimeError("expected bypass_embedding_head on DIRECT_H model")
    model = to_hetero(model, tr_data.metadata(), aggr="mean")
    model.bypass_embedding_head = True
    ckpt_epoch = load_checkpoint_weights(model, device, args, data_config)
    model.to(device)
    model.eval()
    logging.info("Loaded checkpoint epoch=%s model.eval inference_mode extract", ckpt_epoch)

    tr_loader, val_loader, _te_loader = get_loaders(
        tr_data, val_data, te_data, tr_inds, val_inds, te_inds,
        transform, args, train_shuffle=False,
    )
    del _te_loader

    split_map = {
        "train": (tr_loader, tr_inds),
        "val": (val_loader, val_inds),
    }
    results = {}
    for sp in pending:
        loader, inds = split_map[sp]
        out_npz = out_dir / f"{sp}.npz"
        logging.info("Extracting %s -> %s", sp, out_npz)
        stats = extract_split_seed_only(
            loader=loader,
            split_inds=inds,
            model=model,
            device=device,
            args=args,
            out_npz=out_npz,
            expected_n=int(inds.numel()),
            split_name=sp,
        )
        results[sp] = stats
        logging.info(
            "%s done n=%s rss_max=%.2fGiB gpu_peak=%.2fGiB first_stats=%s",
            sp,
            stats["n_rows"],
            stats["rss_gb_max"],
            stats["gpu_peak_gb"] or -1.0,
            stats["first_batch_stats"],
        )

    # Meta only after requested artifacts exist
    have_train = (out_dir / "train.npz").is_file()
    have_val = (out_dir / "val.npz").is_file()
    meta = {
        "unique_name": sub,
        "source_unique_name": cli.run,
        "checkpoint_path": str(ckpt),
        "checkpoint_sha256": ckpt_sha,
        "checkpoint_epoch": int(ckpt_epoch) if ckpt_epoch is not None else int(cli.epoch),
        "representation_source": "pre_embedding_3h",
        "seed_only_r198": True,
        "dim": 198,
        "test_extracted": False,
        "skip_test_eval": True,
        "loader_num_workers": 0,
        "direct_r198_infonce": True,
        "extract_results": results,
        "rss_gb_max": _rss_gb(),
        "gpu_peak_gb": _gpu_peak_gb(),
        "have_train": have_train,
        "have_val": have_val,
    }
    meta_tmp = out_dir / "meta.json.partial"
    meta_tmp.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    os.replace(meta_tmp, out_dir / "meta.json")

    # Free graphs
    del model, tr_loader, val_loader, tr_data, val_data, te_data
    _release()
    logging.info("DONE cell rss_max=%.2fGiB", _rss_gb())
    print(json.dumps({"ok": True, "out_dir": str(out_dir), "results": results, "ckpt_sha256": ckpt_sha}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
