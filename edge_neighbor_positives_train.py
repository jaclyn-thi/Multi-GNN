"""Hetero contrastive training with edge-centric neighbor-positive poscomplete batching.

GCPAL-inspired transfer into the D+ edge-centric recipe — NOT an exact GCPAL
reproduction. Distinct from ``--enable_knn_soft_positives``.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import tqdm
from torch.cuda.amp import GradScaler, autocast
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader

from contrastive_projection import project_seed_pair
from edge_neighbor_positives import (
    CHECKPOINT_EPOCHS_DEFAULT,
    NOT_EXACT_REPRODUCTION,
    build_edge_neighbor_positive_context,
    edge_neighbor_supcon_loss,
    expand_poscomplete_seeds,
)
from graph_augmentations import generate_views
from train_util import (
    FORWARD_EDGE_TYPE,
    AddEgoIds,
    CheckpointTracker,
    attach_edge_id_from_batch,
    save_model,
    select_shared_seed_edge_embeddings,
)


def _contrastive_view_kwargs(args, edge_drop_stats, seed_edge_ids=None):
    return {
        "edge_attr_mask_rate": float(getattr(args, "edge_attr_mask_rate", 0.1)),
        "edge_drop_rate": float(getattr(args, "edge_drop_target_rate", 0.1)),
        "mask_value": 0.0,
        "mask_cols": None,
        "exclude_last_column": (args.model == "rgcn"),
        "edge_drop_policy": getattr(args, "edge_drop_policy", "random"),
        "edge_drop_cache": getattr(args, "edge_drop_cache", None),
        "edge_drop_stats": edge_drop_stats,
        "seed_edge_ids": seed_edge_ids,
        "preserve_seed_edges": bool(getattr(args, "preserve_seed_edges", False)),
    }


def _link_loader_kwargs(args) -> dict:
    # Poscomplete rebuilds a one-shot loader every microbatch; worker processes
    # would dominate runtime. Force single-process sampling here.
    del args
    return {"num_workers": 0}


def _batch_for_seed_positions(
    tr_data: HeteroData,
    seed_positions: np.ndarray,
    args,
    transform,
) -> Tuple[HeteroData, torch.Tensor]:
    """Build one LinkNeighborLoader batch for train-local seed positions.

    Returns ``(batch, seed_edge_ids)`` where ``seed_edge_ids`` are the train-local
    arange ids (equal to ``seed_positions``). Do **not** use
    ``get_hetero_seed_edge_ids`` here: with a subset ``edge_label_index``, PyG
    ``input_id`` indexes the subset, not the full ``tr_data`` edge table.
    """
    seed_positions = np.asarray(seed_positions, dtype=np.int64)
    if seed_positions.size == 0:
        raise ValueError("empty seed_positions")
    ei = tr_data[FORWARD_EDGE_TYPE].edge_index[:, torch.as_tensor(seed_positions, dtype=torch.long)]
    loader = LinkNeighborLoader(
        tr_data,
        num_neighbors=list(args.num_neighs),
        edge_label_index=(FORWARD_EDGE_TYPE, ei),
        batch_size=int(seed_positions.shape[0]),
        shuffle=False,
        transform=transform,
        **_link_loader_kwargs(args),
    )
    batch = next(iter(loader))
    seed_edge_ids = torch.as_tensor(seed_positions, dtype=torch.long)
    return batch, seed_edge_ids


def train_hetero_edge_neighbor_positives(
    tr_loader,
    val_loader,
    te_loader,
    tr_inds,
    val_inds,
    te_inds,
    model,
    optimizer,
    loss_fn,
    args,
    config,
    device,
    val_data,
    te_data,
    data_config,
):
    """Poscomplete neighbor-positive (or matched identity) edge-centric SSL."""
    del te_loader, tr_inds, val_inds, te_inds, loss_fn, te_data, val_loader, val_data
    assert NOT_EXACT_REPRODUCTION
    if bool(getattr(args, "enable_knn_soft_positives", False)):
        raise ValueError(
            "Refuse combining --enable_edge_neighbor_positives with --enable_knn_soft_positives"
        )
    if str(getattr(args, "multi_positive_mode", "none")) != "none":
        raise ValueError("Refuse combining edge neighbor positives with --multi_positive_mode")
    if not bool(getattr(args, "contrastive_asymmetric", False)):
        raise ValueError("Edge neighbor positives require --contrastive_asymmetric (D+ recipe)")
    if not bool(getattr(args, "preserve_seed_edges", False)):
        raise ValueError("Edge neighbor positives require --preserve_seed_edges (D+ recipe)")

    tr_data = tr_loader.data
    labels = None
    y = getattr(tr_data[FORWARD_EDGE_TYPE], "y", None)
    if y is not None:
        labels = y.detach().cpu().numpy().astype(np.int64)

    ctx = build_edge_neighbor_positive_context(
        tr_data,
        knn_cache_path=str(
            getattr(args, "edge_neighbor_knn_cache", None)
            or getattr(args, "knn_cache_path", None)
            or "morphology_cache/Small-HI/transaction_knn_edge_native_degree_fan_k15.npz"
        ),
        flow_policy=str(getattr(args, "edge_neighbor_flow_policy", "immediate_next")),
        knn_k=int(getattr(args, "edge_neighbor_knn_k", 15)),
        max_total_edges=int(getattr(args, "edge_neighbor_max_total", 2048)),
        positive_mode=str(getattr(args, "edge_neighbor_positive_mode", "neighbor")),
        positive_aggregation=str(
            getattr(args, "edge_neighbor_positive_aggregation", "supcon_mean_logprob")
        ),
        train_labels=labels,
    )
    args._edge_neighbor_ctx = ctx
    logging.info(
        "Edge neighbor-positive training: mode=%s agg=%s max_total=%s "
        "(poscomplete changes batch membership → matched identity control required when comparing)",
        ctx.positive_mode,
        ctx.positive_aggregation,
        ctx.max_total_edges,
    )

    use_amp = bool(getattr(args, "amp", False)) and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)
    accum_steps = max(1, int(getattr(args, "contrastive_accum_steps", 1)))
    contrastive_temperature = float(getattr(args, "contrastive_temperature", 0.5))
    proj_head = getattr(args, "contrast_projection_module", None)
    transform = AddEgoIds() if bool(getattr(args, "ego", False)) else None

    ckpt_epochs = set(
        int(x)
        for x in (
            getattr(args, "edge_neighbor_checkpoint_epochs", None) or CHECKPOINT_EPOCHS_DEFAULT
        )
    )
    ckpt_tracker = CheckpointTracker(getattr(args, "checkpoint_policy", "last"))

    n_train = ctx.n_train
    dplus_batch = max(1, int(getattr(args, "batch_size", 8192)))
    default_batches = int(np.ceil(n_train / float(dplus_batch)))
    max_batches = getattr(args, "edge_neighbor_max_batches_per_epoch", None)
    max_batches = int(max_batches) if max_batches is not None else default_batches
    max_batches = max(1, max_batches)
    logging.info(
        "Edge neighbor-positive epoch budget: max_batches=%s "
        "(D+ matched default ceil(n_train/batch_size)=%s); accum=%s → ≈%s optimizer steps/epoch. "
        "Full stream coverage would need ~n_train/anchors_per_batch steps and exceeds 6h; "
        "documenting partial-epoch streaming under matched microbatch count.",
        max_batches,
        default_batches,
        accum_steps,
        max(1, max_batches // accum_steps),
    )

    for epoch in range(config.epochs):
        model.train()
        t0 = time.perf_counter()
        loss_sum = 0.0
        n_anchor_rows = 0
        n_batches = 0
        optimizer_steps = 0
        requested_anchors = 0
        realized_anchors = 0
        retrieved_pos = 0
        total_tx_edges = 0
        total_mp_edges = 0
        pos_stats_acc: Dict[str, float] = {}
        edge_drop_stats: Dict[str, float] = {}
        rng = np.random.RandomState(int(getattr(args, "seed", 0)) * 10007 + epoch)
        stream = rng.permutation(n_train).astype(np.int64)
        cursor = 0
        optimizer.zero_grad(set_to_none=True)
        micro = 0

        pbar = tqdm.tqdm(total=max_batches, disable=not args.tqdm, desc=f"ep{epoch+1}")
        while cursor < n_train and n_batches < max_batches:
            anchors, seeds, cursor, exp_stats = expand_poscomplete_seeds(
                stream, ctx, start=cursor
            )
            if anchors.size == 0:
                break
            batch, seed_edge_ids = _batch_for_seed_positions(tr_data, seeds, args, transform)
            attach_edge_id_from_batch(batch, tr_data)
            batch = batch.to(device, non_blocking=True)
            seed_edge_ids = seed_edge_ids.to(device, non_blocking=True)

            requested_anchors += int(anchors.size)
            retrieved_pos += int(max(seeds.size - anchors.size, 0))
            total_tx_edges += int(seeds.size)
            total_mp_edges += int(batch[FORWARD_EDGE_TYPE].edge_index.shape[1])

            view1, view2 = generate_views(
                batch,
                **_contrastive_view_kwargs(args, edge_drop_stats, seed_edge_ids=seed_edge_ids),
            )
            with autocast(enabled=use_amp):
                out1 = model(view1.x_dict, view1.edge_index_dict, view1.edge_attr_dict)
                z1 = out1[FORWARD_EDGE_TYPE]
                with torch.no_grad():
                    out2 = model(view2.x_dict, view2.edge_index_dict, view2.edge_attr_dict)
                    z2 = out2[FORWARD_EDGE_TYPE]

            edge_id1 = view1[FORWARD_EDGE_TYPE].edge_id
            edge_id2 = view2[FORWARD_EDGE_TYPE].edge_id
            z1_seed, seed_id1, z2_seed, seed_id2 = select_shared_seed_edge_embeddings(
                z1, edge_id1, z2, edge_id2, seed_edge_ids
            )
            z2_seed = z2_seed.detach()
            z1_con, z2_con = project_seed_pair(proj_head, z1_seed, z2_seed)

            with autocast(enabled=False):
                loss_raw, step_stats = edge_neighbor_supcon_loss(
                    z1_con,
                    z2_con,
                    seed_id1,
                    ctx=ctx,
                    anchor_ids=anchors.tolist(),
                    temperature=contrastive_temperature,
                    asymmetric=True,
                )
            realized_anchors += int(step_stats["n_anchor_rows"])
            n_anchor_rows += int(step_stats["n_anchor_rows"])
            for k, v in step_stats.items():
                if isinstance(v, (int, float)):
                    pos_stats_acc[k] = pos_stats_acc.get(k, 0.0) + float(v)
            for k, v in exp_stats.items():
                if isinstance(v, (int, float)):
                    pos_stats_acc[f"expand/{k}"] = pos_stats_acc.get(f"expand/{k}", 0.0) + float(v)

            loss = loss_raw / float(accum_steps)
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            loss_sum += float(loss_raw.detach().item()) * max(int(step_stats["n_anchor_rows"]), 1)
            n_batches += 1
            micro += 1
            pbar.update(1)

            if micro % accum_steps == 0:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1

            del batch, view1, view2, out1, z1, z2, z1_seed, z2_seed
            if device.type == "cuda":
                torch.cuda.empty_cache()

        pbar.close()
        if micro % accum_steps != 0:
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1

        avg_loss = loss_sum / max(n_anchor_rows, 1)
        peak_alloc = (
            float(torch.cuda.max_memory_allocated() / (1024**2)) if device.type == "cuda" else 0.0
        )
        peak_reserved = (
            float(torch.cuda.max_memory_reserved() / (1024**2)) if device.type == "cuda" else 0.0
        )
        elapsed = time.perf_counter() - t0
        log_payload = {
            "loss/train": avg_loss,
            "neighbor_pos/mode": ctx.positive_mode,
            "neighbor_pos/n_batches": n_batches,
            "neighbor_pos/optimizer_steps": optimizer_steps,
            "neighbor_pos/requested_anchors": requested_anchors,
            "neighbor_pos/realized_anchors": realized_anchors,
            "neighbor_pos/retrieved_positives": retrieved_pos,
            "neighbor_pos/total_transaction_edges": total_tx_edges,
            "neighbor_pos/total_mp_edges": total_mp_edges,
            "neighbor_pos/peak_alloc_mib": peak_alloc,
            "neighbor_pos/peak_reserved_mib": peak_reserved,
            "neighbor_pos/epoch_seconds": elapsed,
        }
        for k, v in pos_stats_acc.items():
            log_payload[f"neighbor_pos/{k}"] = v / max(n_batches, 1)

        logging.info(
            "Epoch %s/%s loss=%.4f batches=%s opt_steps=%s realized_anchors=%s/%s "
            "retrieved_pos=%s peak_alloc_MiB=%.1f time=%.1fs",
            epoch + 1,
            config.epochs,
            avg_loss,
            n_batches,
            optimizer_steps,
            realized_anchors,
            requested_anchors,
            retrieved_pos,
            peak_alloc,
            elapsed,
        )

        ckpt_tracker.on_epoch_end(epoch, log_payload, model, optimizer, args, data_config)
        if (epoch + 1) in ckpt_epochs:
            save_model(
                model,
                optimizer,
                epoch,
                args,
                data_config,
                suffix=f"_ep{epoch+1:02d}",
            )
            logging.info("Saved epoch checkpoint suffix=_ep%02d", epoch + 1)

    ckpt_tracker.finalize(config.epochs - 1, model, optimizer, args, data_config)
    return model
