"""
One training batch from real AML data: contrastive forward, loss, single optimizer step.

Homogeneous LinkNeighborLoader path only (same as train_homo_contrastive). Run from repo root, e.g.:

  python debug_contrastive_one_batch.py --data Small_HI --model gin --testing

The GNN forward on a full ``LinkNeighborLoader`` batch can consume tens of GB; this
script **caps** ``--batch_size`` and ``--num_neighs`` by default so forward + chunked
loss fit on a single GPU. Pass ``--debug_full_batch`` to use your CLI values (same as
training; may OOM).

Requires data_config.json paths (e.g. aml-data/) and model_settings.json hyperparameters.
"""

import json
import logging
import sys
from typing import Optional, Tuple

import torch
import wandb
from torch.cuda.amp import GradScaler, autocast
from torch_geometric.data import HeteroData

from contrastive_loss import EdgeMemoryQueue, edge_identity_infonce_loss
from data_loading import get_data
from graph_augmentations import generate_views
from train_util import (
    AddEgoIds,
    add_arange_ids,
    attach_edge_id_from_batch,
    extract_param,
    get_homo_seed_edge_ids,
    get_loaders,
    resolve_training_setup,
    select_shared_seed_edge_embeddings,
    validate_training_setup,
)
from training import get_model
from util import create_parser, logger_setup, set_seed


def _first_nonzero_grad_norm(model: torch.nn.Module) -> Tuple[str, Optional[float]]:
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        n = p.grad.detach().norm().item()
        if n > 0.0:
            return name, n
    return "(none)", None


def main() -> None:
    parser = create_parser()
    parser.add_argument(
        "--debug_full_batch",
        action="store_true",
        help="Do not cap batch_size / num_neighs (matches training; needs enough VRAM).",
    )
    args = parser.parse_args()
    args.testing = True  # wandb offline / no network

    with open("data_config.json", "r") as f:
        data_config = json.load(f)

    logger_setup()
    set_seed(args.seed)

    setup = resolve_training_setup(args)
    try:
        validate_training_setup(setup)
    except NotImplementedError as exc:
        logging.error("%s", exc)
        sys.exit(1)

    if not setup.is_contrastive:
        logging.error(
            "debug_contrastive_one_batch.py requires --objective contrastive (default)."
        )
        sys.exit(1)

    if setup.is_hetero:
        logging.error(
            "debug_contrastive_one_batch.py supports the homogeneous path only. "
            "Run without --reverse_mp."
        )
        sys.exit(1)

    if not args.debug_full_batch:
        orig_bs, orig_neigh = args.batch_size, list(args.num_neighs)
        args.batch_size = min(int(args.batch_size), 256)
        args.num_neighs = [min(int(n), 6) for n in args.num_neighs]
        logging.info(
            "debug: VRAM-safe loader caps (batch_size %s -> %s, num_neighs %s -> %s). "
            "Use --debug_full_batch for full CLI settings.",
            orig_bs,
            args.batch_size,
            orig_neigh,
            args.num_neighs,
        )

    logging.info("Loading AML data via get_data(...)")
    tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(args, data_config)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info("Device: %s", device)

    wandb.init(
        mode="disabled",
        project="multi-gnn",
        config={
            "epochs": args.n_epochs,
            "batch_size": args.batch_size,
            "model": args.model,
            "data": args.data,
            "num_neighbors": args.num_neighs,
            "lr": extract_param("lr", args),
            "n_hidden": extract_param("n_hidden", args),
            "n_gnn_layers": extract_param("n_gnn_layers", args),
            "w_ce1": extract_param("w_ce1", args),
            "w_ce2": extract_param("w_ce2", args),
            "dropout": extract_param("dropout", args),
            "final_dropout": extract_param("final_dropout", args),
            "n_heads": extract_param("n_heads", args) if args.model == "gat" else None,
            "amp": bool(getattr(args, "amp", False)),
            "gradient_checkpointing": bool(getattr(args, "gradient_checkpointing", False)),
            "contrastive_num_neg_samples": int(getattr(args, "contrastive_num_neg_samples", 0)),
            "contrastive_asymmetric": bool(getattr(args, "contrastive_asymmetric", False)),
            "contrastive_memory_bank_size": int(getattr(args, "contrastive_memory_bank_size", 0)),
            "loader_num_workers": int(getattr(args, "loader_num_workers", 10)),
        },
    )
    config = wandb.config

    transform = AddEgoIds() if args.ego else None
    add_arange_ids([tr_data, val_data, te_data])
    tr_loader, _, _ = get_loaders(
        tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform, args
    )

    batch = next(iter(tr_loader))
    if isinstance(batch, HeteroData):
        logging.error("Expected a homogeneous batch; got HeteroData.")
        sys.exit(1)

    model = get_model(batch, config, args)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    seed_edge_ids = get_homo_seed_edge_ids(batch, tr_loader.data)
    batch.to(device)
    seed_edge_ids = seed_edge_ids.to(device)
    ea_before = batch.edge_attr.clone()
    logging.info(
        "pre-attach: edge_attr shape=%s, edge_id=%s, e_id=%s",
        tuple(ea_before.shape),
        getattr(batch, "edge_id", None) is not None,
        getattr(batch, "e_id", None) is not None,
    )
    attach_edge_id_from_batch(batch)

    expected_features = ea_before[:, 1:]
    if batch.edge_attr.shape != expected_features.shape:
        raise AssertionError(
            f"attach_edge_id_from_batch: expected edge_attr shape {tuple(expected_features.shape)} "
            f"(strip synthetic id column), got {tuple(batch.edge_attr.shape)}. "
            "If batch.edge_id or batch.e_id was set by the loader without stripping edge_attr, "
            "align attach_edge_id_from_batch with your batch layout."
        )
    if not torch.allclose(batch.edge_attr, expected_features, rtol=1e-5, atol=1e-6):
        md = (batch.edge_attr - expected_features).abs().max().item()
        raise AssertionError(
            f"attach: edge features after id strip mismatch max_abs_diff={md} "
            "(check column 0 is the add_arange_ids index)."
        )

    features_after_attach = batch.edge_attr.clone()

    edge_drop_rate = 0.12
    view1, view2 = generate_views(
        batch,
        edge_attr_mask_rate=0.1,
        edge_drop_rate=edge_drop_rate,
        mask_value=0.0,
        mask_cols=None,
        exclude_last_column=(args.model == "rgcn"),
    )

    if not torch.allclose(batch.edge_attr, features_after_attach, rtol=1e-5, atol=1e-6):
        md = (batch.edge_attr - features_after_attach).abs().max().item()
        raise AssertionError(
            f"generate_views mutated input batch.edge_attr (max_abs_diff={md}); expected no in-place change."
        )
    assert hasattr(view1, "edge_id") and hasattr(view2, "edge_id")
    assert view1.edge_id.shape[0] == view1.edge_index.shape[1]
    assert view2.edge_id.shape[0] == view2.edge_index.shape[1]
    assert view1.edge_index.shape[1] > 0 and view2.edge_index.shape[1] > 0

    model.train()
    use_amp = bool(getattr(args, "amp", False)) and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)
    neg_kw = int(getattr(args, "contrastive_num_neg_samples", 8192))
    num_neg_samples = neg_kw if neg_kw > 0 else None
    contrastive_symmetric = not bool(getattr(args, "contrastive_asymmetric", False))
    memory_bank_size = max(0, int(getattr(args, "contrastive_memory_bank_size", 0)))
    memory_queue = EdgeMemoryQueue(memory_bank_size, device=device) if memory_bank_size > 0 else None

    optimizer.zero_grad(set_to_none=True)

    with autocast(enabled=use_amp):
        z1 = model(
            view1.x, view1.edge_index, view1.edge_attr
        )
        if contrastive_symmetric:
            z2 = model(
                view2.x, view2.edge_index, view2.edge_attr
            )
        else:
            with torch.no_grad():
                z2 = model(
                    view2.x, view2.edge_index, view2.edge_attr
                )
    z1_seed, seed_id1, z2_seed, seed_id2 = select_shared_seed_edge_embeddings(
        z1,
        view1.edge_id,
        z2,
        view2.edge_id,
        seed_edge_ids,
    )
    with autocast(enabled=False):
        loss = edge_identity_infonce_loss(
            z1_seed,
            z2_seed,
            seed_id1,
            seed_id2,
            temperature=0.5,
            num_neg_samples=num_neg_samples,
            symmetric=contrastive_symmetric,
            memory_queue=memory_queue,
        )

    if use_amp:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()

    loss_ok = bool(torch.isfinite(loss.detach()).item())
    gname, gnorm = _first_nonzero_grad_norm(model)

    print("--- debug_contrastive_one_batch ---")
    print(f"batch.edge_index shape: {tuple(batch.edge_index.shape)}")
    print(f"batch.edge_attr shape:  {tuple(batch.edge_attr.shape)}")
    print(f"view1 edges: {view1.edge_index.shape[1]}, view2 edges: {view2.edge_index.shape[1]}")
    print(f"seed edges requested: {seed_edge_ids.numel()}")
    print(f"shared seed edges kept: {seed_id1.numel()}")
    print(f"z1 seed shape: {tuple(z1_seed.shape)}")
    print(f"z2 seed shape: {tuple(z2_seed.shape)}")
    print(f"loss: {loss.item():.6f}")
    print(
        f"amp: {bool(getattr(args, 'amp', False))}, num_neg_samples: {num_neg_samples}, "
        f"symmetric: {contrastive_symmetric}, queue_size: {memory_bank_size}"
    )
    print(f"loss finite: {loss_ok}")
    print(f"first nonzero grad: {gname} -> {gnorm}")
    wandb.finish()


if __name__ == "__main__":
    main()
