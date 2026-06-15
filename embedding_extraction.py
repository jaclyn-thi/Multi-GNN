"""
Phase 5a: extract frozen seed-edge embeddings from a pretrained checkpoint.

Writes per-split ``.npz`` files (Z, y, edge_id) for downstream linear probing (Phase 5b).

Example (from repo root):

  python embedding_extraction.py --data Small_HI --model gin --unique_name my_pretrain \\
      --reverse_mp --tqdm

Requires ``checkpoint_{unique_name}.tar`` under ``data_config.json`` → ``paths.model_to_load``.

Next step: ``python linear_probe.py --unique_name <same> --testing``
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import to_hetero

from data_loading import get_data
from train_util import (
    AddEgoIds,
    add_arange_ids,
    checkpoint_path,
    expected_seed_edge_ids,
    extract_param,
    extract_seed_embeddings_hetero,
    extract_seed_embeddings_homo,
    get_loaders,
    load_checkpoint_weights,
    log_seed_coverage,
    save_embedding_split_npz,
)
from training import get_model
from util import create_parser, logger_setup, set_seed


def _build_model_config(args) -> SimpleNamespace:
    return SimpleNamespace(
        model=args.model,
        n_hidden=extract_param("n_hidden", args),
        n_gnn_layers=extract_param("n_gnn_layers", args),
        n_heads=extract_param("n_heads", args) if args.model == "gat" else None,
        dropout=extract_param("dropout", args),
        final_dropout=extract_param("final_dropout", args),
    )


def run_embedding_extraction(
    tr_data,
    val_data,
    te_data,
    tr_inds,
    val_inds,
    te_inds,
    args,
    data_config,
) -> Path:
    if not args.unique_name:
        raise ValueError("--unique_name is required to locate checkpoint_{unique_name}.tar")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    hetero = bool(args.reverse_mp)
    config = _build_model_config(args)

    transform = AddEgoIds() if args.ego else None
    add_arange_ids([tr_data, val_data, te_data])

    tr_loader, val_loader, te_loader = get_loaders(
        tr_data,
        val_data,
        te_data,
        tr_inds,
        val_inds,
        te_inds,
        transform,
        args,
        train_shuffle=False,
    )

    sample_batch = next(iter(tr_loader))
    model = get_model(sample_batch, config, args)
    if hetero:
        model = to_hetero(model, te_data.metadata(), aggr="mean")

    finetuned = bool(getattr(args, "finetune", False))
    random_init = bool(getattr(args, "random_init", False))
    if random_init and finetuned:
        raise ValueError("--random_init cannot be combined with --finetune.")

    if random_init:
        model.to(device)
        ckpt_epoch = None
        ckpt_path = None
        logging.info(
            "Random-init extraction: skipping checkpoint load (--unique_name=%s labels output only).",
            args.unique_name,
        )
    else:
        ckpt_epoch = load_checkpoint_weights(model, device, args, data_config)
        ckpt_path = checkpoint_path(data_config, args.unique_name, finetuned=finetuned)
        ckpt_label = f"checkpoint_{args.unique_name}{'_finetuned' if finetuned else ''}.tar"
        logging.info("Loaded %s (epoch=%s)", ckpt_label, ckpt_epoch)

    model.eval()

    embed_name = f"{args.unique_name}_finetuned" if finetuned else args.unique_name
    out_dir = Path(args.embeddings_dir) / embed_name
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = (
        ("train", tr_loader, tr_inds, tr_data),
        ("val", val_loader, val_inds, val_data),
        ("test", te_loader, te_inds, te_data),
    )

    embedding_dim: int | None = None
    for split_name, loader, split_inds, graph_data in splits:
        expected = expected_seed_edge_ids(loader.data, split_inds, hetero=hetero)
        if hetero:
            edge_ids, z, y = extract_seed_embeddings_hetero(
                loader, split_inds, model, graph_data, device, args
            )
        else:
            edge_ids, z, y = extract_seed_embeddings_homo(
                loader, split_inds, model, graph_data, device, args
            )

        log_seed_coverage(edge_ids, expected, split_name)
        save_embedding_split_npz(out_dir / f"{split_name}.npz", z, y, edge_ids)
        embedding_dim = int(z.shape[1])
        logging.info(
            "Wrote %s: Z=%s y=%s path=%s",
            split_name,
            tuple(z.shape),
            tuple(y.shape),
            out_dir / f"{split_name}.npz",
        )

    meta = {
        "unique_name": embed_name,
        "source_unique_name": args.unique_name,
        "finetuned": finetuned,
        "random_init": random_init,
        "checkpoint_epoch": ckpt_epoch,
        "data": args.data,
        "model": args.model,
        "reverse_mp": hetero,
        "embedding_dim": embedding_dim,
        "batch_size": int(args.batch_size),
        "num_neighs": list(args.num_neighs),
        "ports": bool(args.ports),
        "tds": bool(args.tds),
        "ego": bool(args.ego),
        "emlps": bool(args.emlps),
        "checkpoint_path": str(ckpt_path) if ckpt_path is not None else None,
    }
    dataset_spec = getattr(te_data, "dataset_spec_summary", None)
    if dataset_spec is not None:
        meta["dataset_spec"] = dataset_spec
    meta_path = out_dir / "meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logging.info("Wrote metadata to %s", meta_path)
    return out_dir


def main() -> None:
    parser = create_parser()
    parser.add_argument(
        "--embeddings_dir",
        type=str,
        default="embeddings",
        help="Root directory for extracted embedding .npz files (subfolder per --unique_name).",
    )
    parser.add_argument(
        "--random_init",
        action="store_true",
        help="Skip checkpoint load; extract with randomly initialized encoder weights "
        "(transfer baseline). --unique_name names the output folder only.",
    )
    args = parser.parse_args()

    if args.inference:
        parser.error("embedding_extraction.py does not support --inference.")

    with open("data_config.json", "r", encoding="utf-8") as f:
        data_config = json.load(f)

    logger_setup()
    set_seed(args.seed)

    logging.info("Retrieving data")
    t0 = time.perf_counter()
    tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(args, data_config)
    logging.info("Retrieved data in %.2fs", time.perf_counter() - t0)

    out_dir = run_embedding_extraction(
        tr_data, val_data, te_data, tr_inds, val_inds, te_inds, args, data_config
    )
    logging.info("Embedding extraction complete: %s", out_dir)


if __name__ == "__main__":
    main()
