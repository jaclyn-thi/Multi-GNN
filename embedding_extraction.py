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
    infer_pre_embedding_dim,
    load_checkpoint_weights,
    log_seed_coverage,
    resolve_embedding_head_linear,
    save_embedding_split_npz,
    REPRESENTATION_SOURCES,
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

    representation_source = getattr(args, "representation_source", "post_embedding")
    # Locate the embedding head on the homogeneous model (before to_hetero replicates it).
    emb_dim = int(getattr(model, "embedding_dim", 128))
    actual_n_hidden = int(getattr(model, "n_hidden", round(float(config.n_hidden))))
    head_spec = None
    if representation_source == "pre_embedding_3h":
        head_spec = resolve_embedding_head_linear(model, emb_dim)
        pre_dim = head_spec.in_features
        logging.info(
            "Resolved pre_embedding_3h head: module=%s in_features=%d out_features=%d "
            "(model n_hidden=%d, requested n_hidden=%s)",
            head_spec.module_name,
            head_spec.in_features,
            head_spec.out_features,
            actual_n_hidden,
            config.n_hidden,
        )
    else:
        pre_dim = 3 * actual_n_hidden

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
        suffix = str(getattr(args, "checkpoint_suffix", "") or "")
        ckpt_path = checkpoint_path(data_config, args.unique_name, finetuned=finetuned, suffix=suffix)
        ckpt_label = (
            f"checkpoint_{args.unique_name}{'_finetuned' if finetuned else ''}{suffix}.tar"
        )
        logging.info("Loaded %s (epoch=%s)", ckpt_label, ckpt_epoch)

    model.eval()

    embed_name = getattr(args, "embeddings_subdir", None) or args.unique_name
    if finetuned and embed_name == args.unique_name:
        embed_name = f"{args.unique_name}_finetuned"
    out_dir = Path(args.embeddings_dir) / embed_name
    # Keep post_embedding flat (unchanged); route non-default sources to a labeled subdir so
    # the current 128-d embeddings are never overwritten.
    if representation_source != "post_embedding":
        out_dir = out_dir / representation_source
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = (
        ("train", tr_loader, tr_inds, tr_data),
        ("val", val_loader, val_inds, val_data),
        ("test", te_loader, te_inds, te_data),
    )

    embedding_dim: int | None = None
    split_checksums: dict = {}
    for split_name, loader, split_inds, graph_data in splits:
        expected = expected_seed_edge_ids(loader.data, split_inds, hetero=hetero)
        if hetero:
            edge_ids, z, y = extract_seed_embeddings_hetero(
                loader, split_inds, model, graph_data, device, args,
                representation_source=representation_source, pre_dim=pre_dim, emb_dim=emb_dim,
                head_spec=head_spec,
            )
        else:
            edge_ids, z, y = extract_seed_embeddings_homo(
                loader, split_inds, model, graph_data, device, args,
                representation_source=representation_source, pre_dim=pre_dim, emb_dim=emb_dim,
                head_spec=head_spec,
            )

        log_seed_coverage(edge_ids, expected, split_name)
        save_embedding_split_npz(out_dir / f"{split_name}.npz", z, y, edge_ids)
        embedding_dim = int(z.shape[1])
        # Edge-order/identity checksum so a paired comparison can verify both representations
        # cover an identical, identically-ordered set of seed transactions.
        eid_np = edge_ids.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        split_checksums[split_name] = {
            "num_rows": int(eid_np.shape[0]),
            "num_positives": int(y_np.sum()),
            "positive_rate": float(y_np.mean()) if eid_np.shape[0] else float("nan"),
            "edge_id_sum": int(eid_np.astype("int64").sum()),
            "edge_id_first": int(eid_np[0]) if eid_np.shape[0] else None,
            "edge_id_last": int(eid_np[-1]) if eid_np.shape[0] else None,
        }
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
        "representation_source": representation_source,
        "representation_dim": embedding_dim,
        "n_hidden": actual_n_hidden,
        "requested_n_hidden": int(round(float(config.n_hidden))),
        "expected_pre_embedding_3h_dim": pre_dim,
        "pre_embedding_dim": pre_dim if representation_source == "pre_embedding_3h" else None,
        "embedding_head_module": head_spec.module_name if head_spec is not None else None,
        "seed": int(args.seed),
        "split_checksums": split_checksums,
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
    parser.add_argument(
        "--checkpoint_suffix",
        type=str,
        default="",
        help="Checkpoint filename suffix after unique_name (e.g. _last for checkpoint_{name}_last.tar).",
    )
    parser.add_argument(
        "--embeddings_subdir",
        type=str,
        default=None,
        help="Override output folder name under --embeddings_dir (default: --unique_name).",
    )
    parser.add_argument(
        "--representation_source",
        type=str,
        default="post_embedding",
        choices=list(REPRESENTATION_SOURCES),
        help=(
            "Which frozen representation to export. 'post_embedding' (default) exports the "
            "embedding_head output (current 128-d z; unchanged behavior, flat output dir). "
            "'pre_embedding_3h' exports the tensor fed INTO embedding_head "
            "(cat(src_node, dst_node, edge_attr) = 3*n_hidden) into a 'pre_embedding_3h/' subdir. "
            "Neither uses the contrastive projection-head output."
        ),
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
