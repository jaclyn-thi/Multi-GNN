"""
Training loops for contrastive and supervised Multi-GNN objectives.

Contrastive paths (``train_homo_contrastive``, ``train_hetero_contrastive``):
  - Sample k-hop subgraphs, build two augmented views, InfoNCE on shared seed edges
  - Optional morphology expert MSE (``--morph_expert``) and M2 soft positives (``--morph_contrast``)
  - End-of-epoch morphology val metrics (throttled via ``--morph_val_every``)

Supervised paths retain original in-graph CE + F1 evaluation.
"""

import torch
import tqdm
from typing import Dict, Optional
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import f1_score
from train_util import (
    AddEgoIds,
    extract_param,
    add_arange_ids,
    FORWARD_EDGE_TYPE,
    attach_edge_id_from_batch,
    get_hetero_seed_edge_ids,
    get_homo_seed_edge_ids,
    get_loaders,
    evaluate_homo,
    evaluate_hetero,
    edge_classifier_logits,
    log_training_setup,
    resolve_training_setup,
    save_model,
    CheckpointTracker,
    select_shared_seed_edge_embeddings,
    load_model,
    load_checkpoint_auxiliary_modules,
    validate_training_setup,
    validate_masked_edge_args,
)
from models import GINe, PNA, GATe, RGCN
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import to_hetero, summary
from torch_geometric.utils import degree
from pytorch_metric_learning.losses import NTXentLoss
from graph_augmentations import generate_views
from edge_drop_scores import load_or_build_edge_drop_cache
from contrastive_loss import EdgeMemoryQueue, edge_identity_infonce_loss
from contrastive_projection import project_seed_pair, project_seeds, setup_contrastive_projection
from knn_filter import load_transaction_knn_filter
from knn_soft_positives import (
    forward_view2_embeddings_for_edge_ids,
    gather_knn_positive_embeddings,
    load_knn_soft_positive_cache,
    update_knn_endpoint_overlap_stats,
)
from morphology.contrast import setup_morph_contrast_bin_edges, setup_morphology_contrast
from morphology.contrastive_train import (
    eval_morph_contrast_val_hetero,
    eval_morph_contrast_val_homo,
    eval_morph_expert_val_hetero,
    eval_morph_expert_val_homo,
    should_run_morph_val,
    morph_contrast_bin_ids_hetero,
    morph_contrast_bin_ids_homo,
    morph_expert_loss_hetero_step,
    morph_expert_loss_homo_step,
)
from morphology.expert import (
    finalize_morph_expert_diagnostics,
    setup_morph_tier0_contexts,
    setup_morph_tier0_flow_contexts,
    setup_morphology_expert,
)
from masked_edge import (
    build_masked_edge_spec,
    compute_masked_edge_loss,
    prepare_masked_edge_batch,
    setup_masked_edge_decoder,
)
import wandb
import logging


def _edge_endpoints_for_ids(edge_index: torch.Tensor, edge_ids: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Return ``(src, dst)`` endpoints for split-local edge ids."""
    if edge_ids.numel() == 0:
        return torch.empty((0, 2), device=device, dtype=torch.long)
    endpoints = edge_index[:, edge_ids.detach().long().cpu()].T.contiguous()
    return endpoints.to(device=device, non_blocking=True).long()


def _log_false_neg_filter_stats(prefix: str, stats: dict, mode: str) -> None:
    before = float(stats.get("candidate_before", 0.0))
    after = float(stats.get("candidate_after", 0.0))
    if before <= 0:
        return
    removed = max(0.0, before - after)
    frac = removed / before
    fallback_rows = int(stats.get("fallback_rows", 0.0))
    rows = int(stats.get("rows", 0.0))
    logging.info(
        "%s false-negative filter (%s): candidates before=%d after=%d removed=%d (%.4f), fallback_rows=%d/%d",
        prefix,
        mode,
        int(before),
        int(after),
        int(removed),
        frac,
        fallback_rows,
        rows,
    )
    if fallback_rows > 0:
        logging.warning(
            "%s false-negative filter (%s): fallback used for %d/%d anchor rows with too few negatives.",
            prefix,
            mode,
            fallback_rows,
            rows,
        )


def _log_multi_positive_stats(prefix: str, stats: dict, mode: str, weight: float) -> None:
    anchors = float(stats.get("anchors", 0.0))
    if mode == "none" or anchors <= 0:
        return
    identity = float(stats.get("identity_positives", 0.0))
    weak = float(stats.get("weak_positives", 0.0))
    total_pos = float(stats.get("total_positives", identity + weak))
    anchors_without_weak = float(stats.get("anchors_without_weak", 0.0))
    logging.info(
        "%s multi-positive mode=%s weight=%.4f: identity_pos=%d weak_pos=%d avg_pos_per_anchor=%.4f anchors_without_weak=%.4f",
        prefix,
        mode,
        float(weight),
        int(identity),
        int(weak),
        total_pos / anchors,
        anchors_without_weak / anchors,
    )


def _log_knn_filter_stats(prefix: str, stats: dict) -> None:
    before = float(stats.get("candidate_before", 0.0))
    rows = float(stats.get("rows", 0.0))
    if before <= 0 or rows <= 0:
        return
    removed = float(stats.get("knn_removed", max(0.0, before - float(stats.get("candidate_after", before)))))
    rows_with_cache = float(stats.get("rows_with_cache", 0.0))
    rows_with_knn = float(stats.get("rows_with_knn_in_pool", 0.0))
    fallback_rows = float(stats.get("fallback_rows", 0.0))
    overlap = float(stats.get("overlap_with_endpoint_removed", 0.0))
    logging.info(
        "%s KNN negative filter: candidates=%d removed=%d (%.4f), anchors_with_cache=%.4f, anchors_with_knn_in_pool=%.4f, endpoint_overlap_removed=%d, fallback_rows=%d",
        prefix,
        int(before),
        int(removed),
        removed / before,
        rows_with_cache / rows,
        rows_with_knn / rows,
        int(overlap),
        int(fallback_rows),
    )


def _log_knn_soft_pos_stats(prefix: str, stats: dict) -> None:
    anchors = float(stats.get("anchors", 0.0))
    if anchors <= 0:
        return
    sim_count = float(stats.get("sim_count", 0.0))
    sim_mean = float(stats.get("sim_sum", 0.0)) / sim_count if sim_count > 0 else float("nan")
    logging.info(
        "%s KNN soft positives: anchors_with_pos=%.4f usable_pos=%.0f requested=%.0f missing_emb=%.0f unique_resolved=%.0f sim_min/mean/max=%.4f/%.4f/%.4f same_sender/receiver/pair=%.0f/%.0f/%.0f identity_num=%.2e knn_num=%.2e",
        prefix,
        float(stats.get("anchors_with_any_pos", 0.0)) / anchors,
        float(stats.get("usable_positives", 0.0)),
        float(stats.get("requested_positives", 0.0)),
        float(stats.get("positives_missing_embedding", 0.0)),
        float(stats.get("unique_pos_ids_resolved", 0.0)),
        float(stats.get("sim_min", float("nan"))),
        sim_mean,
        float(stats.get("sim_max", float("nan"))),
        float(stats.get("knn_pos_same_sender", 0.0)),
        float(stats.get("knn_pos_same_receiver", 0.0)),
        float(stats.get("knn_pos_same_pair", 0.0)),
        float(stats.get("identity_num_contrib", 0.0)),
        float(stats.get("knn_num_contrib", 0.0)),
    )


def _contrastive_view_kwargs(args, edge_drop_stats: Optional[dict] = None) -> dict:
    return {
        "edge_attr_mask_rate": 0.1,
        "edge_drop_rate": float(getattr(args, "edge_drop_target_rate", 0.1)),
        "mask_value": 0.0,
        "mask_cols": None,
        "exclude_last_column": (args.model == "rgcn"),
        "edge_drop_policy": getattr(args, "edge_drop_policy", "random"),
        "edge_drop_cache": getattr(args, "edge_drop_cache", None),
        "edge_drop_stats": edge_drop_stats,
    }


def _log_edge_drop_stats(prefix: str, stats: dict) -> None:
    v1_kept = float(stats.get("edges_kept_v1", 0.0))
    v2_kept = float(stats.get("edges_kept_v2", 0.0))
    v1_dropped = float(stats.get("edges_dropped_v1", 0.0))
    v2_dropped = float(stats.get("edges_dropped_v2", 0.0))
    v1_total = v1_kept + v1_dropped
    v2_total = v2_kept + v2_dropped
    if v1_total <= 0 and v2_total <= 0:
        return
    target = float(stats.get("target_drop_rate", float("nan")))
    policy = stats.get("edge_drop_policy", "random")
    prob_count = float(stats.get("drop_prob_count", 0.0))
    prob_mean = float(stats.get("drop_prob_sum", 0.0)) / prob_count if prob_count > 0 else float("nan")
    overlap_batches = float(stats.get("two_view_overlap_batches", 0.0))
    overlap = float(stats.get("two_view_edge_overlap", 0.0)) / overlap_batches if overlap_batches > 0 else float("nan")
    realized_v1 = v1_dropped / v1_total if v1_total > 0 else float("nan")
    realized_v2 = v2_dropped / v2_total if v2_total > 0 else float("nan")
    logging.info(
        "%s edge drop (%s): target=%.4f realized_v1=%.4f realized_v2=%.4f overlap=%.1f "
        "drop_prob_min/mean/max=%.4f/%.4f/%.4f labels_used=[]",
        prefix,
        policy,
        target,
        realized_v1,
        realized_v2,
        overlap,
        float(stats.get("drop_prob_min", float("nan"))),
        prob_mean,
        float(stats.get("drop_prob_max", float("nan"))),
    )
    for bucket_prefix in ("drop_rate_degree", "drop_rate_amount", "drop_rate_flow"):
        for view_tag in ("v1", "v2"):
            for label in ("p0_20", "p20_40", "p40_60", "p60_80", "p80_100"):
                key = f"{bucket_prefix}_{label}_{view_tag}"
                cnt_key = f"{key}_count"
                cnt = float(stats.get(cnt_key, 0.0))
                if cnt > 0:
                    logging.info(
                        "%s %s %s: mean_drop_rate=%.4f (batches=%.0f)",
                        prefix,
                        bucket_prefix,
                        f"{label}_{view_tag}",
                        float(stats.get(key, 0.0)) / cnt,
                        cnt,
                    )


def _prepare_knn_soft_positives(
    *,
    knn_soft_cache,
    seed_ids: torch.Tensor,
    tr_loader,
    model,
    proj_head,
    args,
    device,
    use_amp: bool,
    contrastive_symmetric: bool,
    train_edge_index: torch.Tensor,
    epoch: int,
    step: int,
    stats: dict,
):
    if knn_soft_cache is None:
        return None, None, None, None
    pos_ids, pos_sims, pos_weights, pos_valid, sample_stats = knn_soft_cache.sample(
        seed_ids,
        step=step,
        epoch=epoch,
    )
    stats.update(sample_stats)
    unique_ids = torch.unique(pos_ids[pos_valid])
    unique_z2 = seed_ids.new_empty((0, 128), dtype=torch.float32)
    if unique_ids.numel() > 0:
        resolved_ids, z2_raw = forward_view2_embeddings_for_edge_ids(
            tr_loader,
            model,
            unique_ids,
            device=device,
            use_amp=use_amp,
            contrastive_symmetric=contrastive_symmetric,
            chunk_size=int(getattr(args, "knn_pos_loader_batch_size", 4096)),
            exclude_last_column=(args.model == "rgcn"),
        )
        z2_proj = project_seeds(proj_head, z2_raw.to(device))
        unique_ids = resolved_ids.to(device)
        unique_z2 = z2_proj
    pos_z2, pos_valid = gather_knn_positive_embeddings(
        pos_ids,
        pos_valid,
        unique_ids,
        unique_z2,
        stats=stats,
    )
    update_knn_endpoint_overlap_stats(seed_ids, pos_ids, pos_valid, train_edge_index, stats)
    del pos_sims
    return pos_ids, pos_weights, pos_valid, pos_z2


def _log_morphology_group_losses(prefix: str, metrics: dict) -> None:
    """Print compact morphology group diagnostics for Slurm/stdout inspection."""
    if not metrics:
        return
    parts = []
    total = metrics.get("morphology/loss_total")
    if total is not None:
        parts.append(f"total={float(total):.4f}")
    for key in sorted(metrics):
        if key.startswith("morphology/loss_group/"):
            group = key.rsplit("/", 1)[-1]
            parts.append(f"{group}={float(metrics[key]):.4f}")
    if parts:
        logging.info("%s morphology losses: %s", prefix, " ".join(parts))


def train_homo_contrastive(tr_loader, val_loader, te_loader, tr_inds, val_inds, te_inds, model, optimizer, loss_fn, args, config, device, val_data, te_data, data_config):
    """
    Homogeneous-graph contrastive pretraining with optional morphology losses.

    Morphology (when enabled) uses train-split Tier 0 context on each step and
    runs separate val-loader passes for ``morph/expert_val`` / ``morph/contrast_val``.
    """
    # Contrastive pretraining on homogeneous graphs.
    # NOTE: The following arguments are unused in contrastive pretraining:
    # tr_inds, te_loader, te_inds, loss_fn, te_data
    # best_val_f1 = 0
    use_amp = bool(getattr(args, "amp", False)) and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)
    neg_kw = int(getattr(args, "contrastive_num_neg_samples", 0))
    num_neg_samples = neg_kw if neg_kw > 0 else None
    contrastive_symmetric = not bool(getattr(args, "contrastive_asymmetric", False))
    accum_steps = max(1, int(getattr(args, "contrastive_accum_steps", 1)))
    memory_bank_size = max(0, int(getattr(args, "contrastive_memory_bank_size", 0)))
    memory_queue = EdgeMemoryQueue(memory_bank_size, device=device) if memory_bank_size > 0 else None
    false_neg_filter_mode = str(getattr(args, "false_neg_filter_mode", "none"))
    false_neg_filter_min_negatives = max(0, int(getattr(args, "false_neg_filter_min_negatives", 1)))
    multi_positive_mode = str(getattr(args, "multi_positive_mode", "none"))
    multi_positive_weight = float(getattr(args, "multi_positive_weight", 0.1))
    contrastive_temperature = float(getattr(args, "contrastive_temperature", 0.5))
    if contrastive_temperature <= 0.0:
        raise ValueError("--contrastive_temperature must be > 0.")
    logging.info("Contrastive temperature: %.4f", contrastive_temperature)
    knn_filter = load_transaction_knn_filter(
        getattr(args, "knn_cache_path", None),
        enabled=bool(getattr(args, "enable_knn_negative_filter", False)),
        filter_k=int(getattr(args, "knn_filter_k", 0)),
        device=device,
    )
    train_edge_index = tr_loader.data.edge_index.detach().cpu()
    try:
        n_train_batches = len(tr_loader)
    except TypeError:
        n_train_batches = None
    if accum_steps > 1 and n_train_batches is None:
        logging.warning(
            "contrastive_accum_steps=%s ignored: training loader has no len(); use accum_steps=1.",
            accum_steps,
        )
        accum_steps = 1

    morph_head = getattr(args, "morph_expert_head", None)
    morph_cfg = getattr(args, "morph_expert_cfg", None)
    morph_contrast_cfg = getattr(args, "morph_contrast_cfg", None)
    proj_head = getattr(args, "contrast_projection_module", None)
    ckpt_tracker = CheckpointTracker(getattr(args, "checkpoint_policy", "last"))
    if ckpt_tracker.policy == "best":
        logging.info(
            "Checkpoint policy: best (lowest morph/expert_val + morph/contrast_val, else loss/train)"
        )

    for epoch in range(config.epochs):
        total_examples = 0
        loss_sum = torch.zeros((), device=device)
        morph_loss_sum = torch.zeros((), device=device)
        morph_diag_accumulator = {} if morph_head is not None else None
        false_neg_stats = dict()
        multi_pos_stats = dict()
        knn_filter_stats = dict()
        edge_drop_stats = dict()
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(tqdm.tqdm(tr_loader, disable=not args.tqdm)):
            seed_edge_ids = get_homo_seed_edge_ids(batch, tr_loader.data)
            attach_edge_id_from_batch(batch)

            batch.to(device, non_blocking=True)
            seed_edge_ids = seed_edge_ids.to(device, non_blocking=True)
            if epoch == 0 and step == 0:
                n_sub = int(batch.num_nodes) if getattr(batch, "num_nodes", None) is not None else int(batch.x.shape[0])
                e_mp = int(batch.edge_index.shape[1])
                logging.info(
                    "Contrastive first batch: subgraph_nodes=%s message_passing_edges=%s. "
                    "Peak VRAM is dominated by this subgraph (GNN activations), not the loss matmul. "
                    "If you OOM, reduce --batch_size and/or --num_neighs; use --contrastive_accum_steps>1 "
                    "with a smaller batch to average gradients over more steps.",
                    n_sub,
                    e_mp,
                )

            # (old) forward pass
            # out = model(batch.x, batch.edge_index, batch.edge_attr)
            # pred = out[mask]

            # Two views: independent edge drops (pair by batch.edge_id) + optional attr mask
            view1, view2 = generate_views(
                batch,
                **_contrastive_view_kwargs(args, edge_drop_stats),
            )

            with autocast(enabled=use_amp):
                z1 = model(view1.x, view1.edge_index, view1.edge_attr)
                if contrastive_symmetric:
                    z2 = model(view2.x, view2.edge_index, view2.edge_attr)
                else:
                    with torch.no_grad():
                        z2 = model(view2.x, view2.edge_index, view2.edge_attr)
            z1_seed, seed_id1, z2_seed, seed_id2 = select_shared_seed_edge_embeddings(
                z1,
                view1.edge_id,
                z2,
                view2.edge_id,
                seed_edge_ids,
            )
            if epoch == 0 and step == 0:
                logging.info(
                    "Contrastive seed-edge filtering: requested_seed_edges=%s shared_seed_edges=%s queue_size=%s",
                    int(seed_edge_ids.numel()),
                    int(seed_id1.numel()),
                    0 if memory_queue is None else memory_queue.size,
                )
            with autocast(enabled=False):
                # M2: bin ids from view1 detached features → soft positives in InfoNCE
                morph_bins = None
                if morph_contrast_cfg is not None:
                    morph_bins = morph_contrast_bin_ids_homo(
                        view1,
                        seed_id1,
                        batch,
                        morph_contrast_cfg,
                        getattr(args, "morph_tier0_train", None),
                        int(getattr(args, "_morph_contrast_edge_native_dim", 0)),
                    )
                max_soft = (
                    int(morph_contrast_cfg.max_soft_positives)
                    if morph_contrast_cfg is not None
                    else 256
                )
                if max_soft <= 0:
                    max_soft = None
                z1_con, z2_con = project_seed_pair(proj_head, z1_seed, z2_seed)
                seed_endpoints = (
                    _edge_endpoints_for_ids(train_edge_index, seed_id1, device)
                    if false_neg_filter_mode != "none" or multi_positive_mode != "none"
                    else None
                )
                loss_raw = edge_identity_infonce_loss(
                    z1_con,
                    z2_con,
                    seed_id1,
                    seed_id2,
                    temperature=contrastive_temperature,
                    num_neg_samples=num_neg_samples,
                    symmetric=contrastive_symmetric,
                    memory_queue=memory_queue,
                    morph_bin_ids=morph_bins,
                    max_soft_positives=max_soft,
                    edge_endpoints=seed_endpoints,
                    false_neg_filter_mode=false_neg_filter_mode,
                    false_neg_filter_min_negatives=false_neg_filter_min_negatives,
                    false_neg_filter_stats=false_neg_stats,
                    multi_positive_mode=multi_positive_mode,
                    multi_positive_weight=multi_positive_weight,
                    multi_positive_stats=multi_pos_stats,
                    knn_filter=knn_filter,
                    knn_filter_stats=knn_filter_stats,
                )
                # M1/M1b: predict detached morphology targets from z_seed (view1)
                if morph_head is not None and morph_cfg is not None:
                    morph_loss = morph_expert_loss_homo_step(
                        view1,
                        z1_seed,
                        seed_id1,
                        batch,
                        morph_head,
                        morph_cfg,
                        tier0_ctx=getattr(args, "morph_tier0_train", None),
                        flow_ctx=getattr(args, "morph_tier0_flow_train", None),
                        tier2_ctx=getattr(args, "morph_tier2_train", None),
                        diagnostics_accumulator=morph_diag_accumulator,
                    )
                    loss_raw = loss_raw + morph_loss
                    morph_loss_sum = morph_loss_sum + morph_loss.detach()
            loss = loss_raw / float(accum_steps)

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            step_one_indexed = step + 1
            should_step = (step_one_indexed % accum_steps == 0) or (
                n_train_batches is not None and step_one_indexed == n_train_batches
            )
            if should_step:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if memory_queue is not None and seed_id2.numel() > 0:
                z2_queue = project_seeds(proj_head, z2_seed)
                queue_endpoints = (
                    _edge_endpoints_for_ids(train_edge_index, seed_id2, device)
                    if false_neg_filter_mode != "none" or multi_positive_mode != "none"
                    else None
                )
                memory_queue.enqueue(z2_queue, seed_id2, queue_endpoints)

            loss_sum = loss_sum + loss_raw.detach()
            total_examples += 1

        avg_loss = float((loss_sum / max(total_examples, 1)).cpu())
        log_payload = {"loss/train": avg_loss}
        if morph_head is not None:
            avg_morph = float((morph_loss_sum / max(total_examples, 1)).cpu())
            log_payload["morph/expert_train"] = avg_morph
            log_payload["morphology/loss_total"] = avg_morph
            morph_diag_metrics = finalize_morph_expert_diagnostics(morph_diag_accumulator)
            log_payload.update(morph_diag_metrics)
            if val_loader is not None and should_run_morph_val(epoch, config.epochs, args):
                log_payload["morph/expert_val"] = eval_morph_expert_val_homo(
                    val_loader, model, morph_head, morph_cfg, device, args
                )
            logging.info(f"Train Loss: {avg_loss:.4f} | morph/expert_train: {avg_morph:.4f}")
            _log_morphology_group_losses("homo/train", log_payload)
        else:
            logging.info(f"Train Loss: {avg_loss:.4f}")
        _log_false_neg_filter_stats("homo/train", false_neg_stats, false_neg_filter_mode)
        _log_multi_positive_stats("homo/train", multi_pos_stats, multi_positive_mode, multi_positive_weight)
        _log_knn_filter_stats("homo/train", knn_filter_stats)
        _log_edge_drop_stats("homo/train", edge_drop_stats)
        if (
            morph_contrast_cfg is not None
            and val_loader is not None
            and should_run_morph_val(epoch, config.epochs, args)
        ):
            log_payload["morph/contrast_val"] = eval_morph_contrast_val_homo(
                val_loader, model, morph_contrast_cfg, device, args
            )
        ckpt_tracker.on_epoch_end(epoch, log_payload, model, optimizer, args, data_config)
        if ckpt_tracker.policy == "best":
            log_payload["checkpoint/best_epoch"] = ckpt_tracker.best_epoch
            log_payload["checkpoint/best_score"] = (
                ckpt_tracker.best_score if ckpt_tracker.best_epoch >= 0 else float("nan")
            )
        wandb.log(log_payload, step=epoch)

        # old F1 computation block
        # pred = torch.cat(preds, dim=0).detach().cpu().numpy()
        # ground_truth = torch.cat(ground_truths, dim=0).detach().cpu().numpy()
        # f1 = f1_score(ground_truth, pred)
        # wandb.log({"f1/train": f1}, step=epoch)
        # logging.info(f'Train F1: {f1:.4f}')

        # (old) evaluate
        # val_f1 = evaluate_homo(val_loader, val_inds, model, val_data, device, args)
        # te_f1 = evaluate_homo(te_loader, te_inds, model, te_data, device, args)

        # wandb.log({"f1/validation": val_f1}, step=epoch)
        # wandb.log({"f1/test": te_f1}, step=epoch)
        # logging.info(f'Validation F1: {val_f1:.4f}')
        # logging.info(f'Test F1: {te_f1:.4f}')

        # if epoch == 0:
        #     wandb.log({"best_test_f1": te_f1}, step=epoch)
        # elif val_f1 > best_val_f1:
        #     best_val_f1 = val_f1
        #     wandb.log({"best_test_f1": te_f1}, step=epoch)
        #     if args.save_model:
        #         save_model(model, optimizer, epoch, args, data_config)

    ckpt_tracker.finalize(config.epochs - 1, model, optimizer, args, data_config)
    return model


def train_homo_supervised(tr_loader, val_loader, te_loader, tr_inds, val_inds, te_inds, model, optimizer, loss_fn, args, config, device, val_data, te_data, data_config):
    """Supervised AML edge classification on homogeneous graphs."""
    best_val_f1 = 0
    for epoch in range(config.epochs):
        total_loss = total_examples = 0
        preds = []
        ground_truths = []
        for batch in tqdm.tqdm(tr_loader, disable=not args.tqdm):
            optimizer.zero_grad()
            inds = tr_inds.detach().cpu()
            batch_edge_inds = inds[batch.input_id.detach().cpu()]
            batch_edge_ids = tr_loader.data.edge_attr.detach().cpu()[batch_edge_inds, 0]
            mask = torch.isin(batch.edge_attr[:, 0].detach().cpu(), batch_edge_ids)

            batch.edge_attr = batch.edge_attr[:, 1:]

            batch.to(device)
            mask = mask.to(device, non_blocking=True)
            z = model(batch.x, batch.edge_index, batch.edge_attr)
            pred = model.classifier(z)[mask]
            ground_truth = batch.y[mask]
            preds.append(pred.argmax(dim=-1))
            ground_truths.append(ground_truth)
            loss = loss_fn(pred, ground_truth)

            loss.backward()
            optimizer.step()

            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()

        pred = torch.cat(preds, dim=0).detach().cpu().numpy()
        ground_truth = torch.cat(ground_truths, dim=0).detach().cpu().numpy()
        f1 = f1_score(ground_truth, pred)
        wandb.log({"f1/train": f1}, step=epoch)
        logging.info(f"Train F1: {f1:.4f}")

        val_f1 = evaluate_homo(val_loader, val_inds, model, val_data, device, args)
        te_f1 = evaluate_homo(te_loader, te_inds, model, te_data, device, args)

        wandb.log({"f1/validation": val_f1}, step=epoch)
        wandb.log({"f1/test": te_f1}, step=epoch)
        logging.info(f"Validation F1: {val_f1:.4f}")
        logging.info(f"Test F1: {te_f1:.4f}")

        if epoch == 0:
            wandb.log({"best_test_f1": te_f1}, step=epoch)
        elif val_f1 > best_val_f1:
            best_val_f1 = val_f1
            wandb.log({"best_test_f1": te_f1}, step=epoch)
            if args.save_model:
                save_model(model, optimizer, epoch, args, data_config)

    if args.save_model:
        save_model(model, optimizer, epoch, args, data_config)
        logging.info("Saved final-epoch checkpoint for %s", args.unique_name)

    return model


def train_hetero_supervised(tr_loader, val_loader, te_loader, tr_inds, val_inds, te_inds, model, optimizer, loss_fn, args, config, device, val_data, te_data, data_config):
    """Supervised AML edge classification on heterogeneous (reverse MP) graphs."""
    best_val_f1 = 0
    for epoch in range(config.epochs):
        total_loss = total_examples = 0
        preds = []
        ground_truths = []
        for batch in tqdm.tqdm(tr_loader, disable=not args.tqdm):
            optimizer.zero_grad()
            #select the seed edges from which the batch was created
            inds = tr_inds.detach().cpu()
            batch_edge_inds = inds[batch['node', 'to', 'node'].input_id.detach().cpu()]
            batch_edge_ids = tr_loader.data['node', 'to', 'node'].edge_attr.detach().cpu()[batch_edge_inds, 0]
            mask = torch.isin(batch['node', 'to', 'node'].edge_attr[:, 0].detach().cpu(), batch_edge_ids)

            #remove the unique edge id from the edge features, as it's no longer needed
            batch['node', 'to', 'node'].edge_attr = batch['node', 'to', 'node'].edge_attr[:, 1:]
            batch['node', 'rev_to', 'node'].edge_attr = batch['node', 'rev_to', 'node'].edge_attr[:, 1:]

            batch.to(device)
            mask = mask.to(device, non_blocking=True)
            z = model(
                batch.x_dict,
                batch.edge_index_dict,
                batch.edge_attr_dict,
            )[('node', 'to', 'node')]
            pred = edge_classifier_logits(model, z)[mask]
            ground_truth = batch['node', 'to', 'node'].y[mask]
            preds.append(pred.argmax(dim=-1))
            ground_truths.append(batch['node', 'to', 'node'].y[mask])
            loss = loss_fn(pred, ground_truth)

            loss.backward()
            optimizer.step()

            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()

        pred = torch.cat(preds, dim=0).detach().cpu().numpy()
        ground_truth = torch.cat(ground_truths, dim=0).detach().cpu().numpy()
        f1 = f1_score(ground_truth, pred)
        wandb.log({"f1/train": f1}, step=epoch)
        logging.info(f'Train F1: {f1:.4f}')

        #evaluate
        val_f1 = evaluate_hetero(val_loader, val_inds, model, val_data, device, args)
        te_f1 = evaluate_hetero(te_loader, te_inds, model, te_data, device, args)

        wandb.log({"f1/validation": val_f1}, step=epoch)
        wandb.log({"f1/test": te_f1}, step=epoch)
        logging.info(f'Validation F1: {val_f1:.4f}')
        logging.info(f'Test F1: {te_f1:.4f}')

        if epoch == 0:
            wandb.log({"best_test_f1": te_f1}, step=epoch)
        elif val_f1 > best_val_f1:
            best_val_f1 = val_f1
            wandb.log({"best_test_f1": te_f1}, step=epoch)
            if args.save_model:
                save_model(model, optimizer, epoch, args, data_config)

    if args.save_model:
        save_model(model, optimizer, epoch, args, data_config)
        logging.info("Saved final-epoch checkpoint for %s", args.unique_name)

    return model


def train_hetero_contrastive(tr_loader, val_loader, te_loader, tr_inds, val_inds, te_inds, model, optimizer, loss_fn, args, config, device, val_data, te_data, data_config):
    """
    Heterogeneous-graph contrastive pretraining (production path for Small-HI).

    Same morphology integration as ``train_homo_contrastive`` but on forward
    ``('node', 'to', 'node')`` edge embeddings from ``to_hetero`` models.
    """
    """Contrastive pretraining on heterogeneous graphs (forward transactions as anchors)."""
    del te_loader, tr_inds, val_inds, te_inds, loss_fn, te_data
    use_amp = bool(getattr(args, "amp", False)) and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)
    neg_kw = int(getattr(args, "contrastive_num_neg_samples", 0))
    num_neg_samples = neg_kw if neg_kw > 0 else None
    contrastive_symmetric = not bool(getattr(args, "contrastive_asymmetric", False))
    accum_steps = max(1, int(getattr(args, "contrastive_accum_steps", 1)))
    memory_bank_size = max(0, int(getattr(args, "contrastive_memory_bank_size", 0)))
    memory_queue = EdgeMemoryQueue(memory_bank_size, device=device) if memory_bank_size > 0 else None
    false_neg_filter_mode = str(getattr(args, "false_neg_filter_mode", "none"))
    false_neg_filter_min_negatives = max(0, int(getattr(args, "false_neg_filter_min_negatives", 1)))
    multi_positive_mode = str(getattr(args, "multi_positive_mode", "none"))
    multi_positive_weight = float(getattr(args, "multi_positive_weight", 0.1))
    contrastive_temperature = float(getattr(args, "contrastive_temperature", 0.5))
    if contrastive_temperature <= 0.0:
        raise ValueError("--contrastive_temperature must be > 0.")
    logging.info("Contrastive temperature: %.4f", contrastive_temperature)
    knn_filter = load_transaction_knn_filter(
        getattr(args, "knn_cache_path", None),
        enabled=bool(getattr(args, "enable_knn_negative_filter", False)),
        filter_k=int(getattr(args, "knn_filter_k", 0)),
        device=device,
    )
    knn_soft_cache = load_knn_soft_positive_cache(
        getattr(args, "knn_cache_path", None),
        enabled=bool(getattr(args, "enable_knn_soft_positives", False)),
        source_k=int(getattr(args, "knn_pos_source_k", 15)),
        pos_m=int(getattr(args, "knn_pos_m", 1)),
        total_weight=float(getattr(args, "knn_pos_weight", 0.025)),
        weight_mode=str(getattr(args, "knn_pos_weight_mode", "uniform")),
        min_sim=getattr(args, "knn_pos_min_sim", None),
        base_seed=int(getattr(args, "knn_pos_seed", 0)),
        device=device,
    )
    train_edge_index = tr_loader.data[FORWARD_EDGE_TYPE].edge_index.detach().cpu()
    try:
        n_train_batches = len(tr_loader)
    except TypeError:
        n_train_batches = None
    if accum_steps > 1 and n_train_batches is None:
        logging.warning(
            "contrastive_accum_steps=%s ignored: training loader has no len(); use accum_steps=1.",
            accum_steps,
        )
        accum_steps = 1

    morph_head = getattr(args, "morph_expert_head", None)
    morph_cfg = getattr(args, "morph_expert_cfg", None)
    morph_contrast_cfg = getattr(args, "morph_contrast_cfg", None)
    if knn_soft_cache is not None and contrastive_symmetric:
        raise ValueError("KNN soft positives require asymmetric contrastive training (--contrastive_asymmetric).")
    if knn_soft_cache is not None and morph_contrast_cfg is not None:
        raise ValueError("KNN soft positives cannot be combined with --morph_contrast.")
    if knn_soft_cache is not None and multi_positive_mode != "none":
        raise ValueError("KNN soft positives cannot be combined with --multi_positive_mode.")
    proj_head = getattr(args, "contrast_projection_module", None)
    ckpt_tracker = CheckpointTracker(getattr(args, "checkpoint_policy", "last"))
    if ckpt_tracker.policy == "best":
        logging.info(
            "Checkpoint policy: best (lowest morph/expert_val + morph/contrast_val, else loss/train)"
        )

    for epoch in range(config.epochs):
        total_examples = 0
        loss_sum = torch.zeros((), device=device)
        morph_loss_sum = torch.zeros((), device=device)
        morph_diag_accumulator = {} if morph_head is not None else None
        false_neg_stats = dict()
        multi_pos_stats = dict()
        knn_filter_stats = dict()
        knn_soft_pos_stats = dict()
        edge_drop_stats = dict()
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(tqdm.tqdm(tr_loader, disable=not args.tqdm)):
            seed_edge_ids = get_hetero_seed_edge_ids(batch, tr_loader.data)
            attach_edge_id_from_batch(batch, tr_loader.data)

            batch.to(device, non_blocking=True)
            seed_edge_ids = seed_edge_ids.to(device, non_blocking=True)
            if epoch == 0 and step == 0:
                n_sub = int(batch["node"].num_nodes)
                e_fwd = int(batch[FORWARD_EDGE_TYPE].edge_index.shape[1])
                e_rev = int(batch["node", "rev_to", "node"].edge_index.shape[1])
                logging.info(
                    "Hetero contrastive first batch: subgraph_nodes=%s forward_edges=%s reverse_edges=%s. "
                    "Contrastive loss uses forward-edge embeddings only.",
                    n_sub,
                    e_fwd,
                    e_rev,
                )

            view1, view2 = generate_views(
                batch,
                **_contrastive_view_kwargs(args, edge_drop_stats),
            )

            with autocast(enabled=use_amp):
                out1 = model(
                    view1.x_dict,
                    view1.edge_index_dict,
                    view1.edge_attr_dict,
                )
                z1 = out1[FORWARD_EDGE_TYPE]
                if contrastive_symmetric:
                    out2 = model(
                        view2.x_dict,
                        view2.edge_index_dict,
                        view2.edge_attr_dict,
                    )
                    z2 = out2[FORWARD_EDGE_TYPE]
                else:
                    with torch.no_grad():
                        out2 = model(
                            view2.x_dict,
                            view2.edge_index_dict,
                            view2.edge_attr_dict,
                        )
                        z2 = out2[FORWARD_EDGE_TYPE]

            edge_id1 = view1[FORWARD_EDGE_TYPE].edge_id
            edge_id2 = view2[FORWARD_EDGE_TYPE].edge_id
            z1_seed, seed_id1, z2_seed, seed_id2 = select_shared_seed_edge_embeddings(
                z1,
                edge_id1,
                z2,
                edge_id2,
                seed_edge_ids,
            )
            if not contrastive_symmetric:
                z2_seed = z2_seed.detach().clone()
                del out2, z2, view2
            if epoch == 0 and step == 0:
                logging.info(
                    "Hetero contrastive seed-edge filtering: requested_seed_edges=%s shared_seed_edges=%s queue_size=%s",
                    int(seed_edge_ids.numel()),
                    int(seed_id1.numel()),
                    0 if memory_queue is None else memory_queue.size,
                )

            with autocast(enabled=False):
                # M2: morphology bin ids on view1 → merged InfoNCE positives
                morph_bins = None
                if morph_contrast_cfg is not None:
                    morph_bins = morph_contrast_bin_ids_hetero(
                        view1,
                        seed_id1,
                        batch,
                        morph_contrast_cfg,
                        getattr(args, "morph_tier0_train", None),
                        int(getattr(args, "_morph_contrast_edge_native_dim", 0)),
                    )
                max_soft = (
                    int(morph_contrast_cfg.max_soft_positives)
                    if morph_contrast_cfg is not None
                    else 256
                )
                if max_soft <= 0:
                    max_soft = None
                z1_con, z2_con = project_seed_pair(proj_head, z1_seed, z2_seed)
                knn_pos_ids, knn_pos_weights, knn_pos_valid, knn_pos_z2 = _prepare_knn_soft_positives(
                    knn_soft_cache=knn_soft_cache,
                    seed_ids=seed_id1,
                    tr_loader=tr_loader,
                    model=model,
                    proj_head=proj_head,
                    args=args,
                    device=device,
                    use_amp=use_amp,
                    contrastive_symmetric=contrastive_symmetric,
                    train_edge_index=train_edge_index,
                    epoch=epoch,
                    step=step,
                    stats=knn_soft_pos_stats,
                )
                seed_endpoints = (
                    _edge_endpoints_for_ids(train_edge_index, seed_id1, device)
                    if false_neg_filter_mode != "none" or multi_positive_mode != "none"
                    else None
                )
                loss_raw = edge_identity_infonce_loss(
                    z1_con,
                    z2_con,
                    seed_id1,
                    seed_id2,
                    temperature=contrastive_temperature,
                    num_neg_samples=num_neg_samples,
                    symmetric=contrastive_symmetric,
                    memory_queue=memory_queue,
                    morph_bin_ids=morph_bins,
                    max_soft_positives=max_soft,
                    edge_endpoints=seed_endpoints,
                    false_neg_filter_mode=false_neg_filter_mode,
                    false_neg_filter_min_negatives=false_neg_filter_min_negatives,
                    false_neg_filter_stats=false_neg_stats,
                    multi_positive_mode=multi_positive_mode,
                    multi_positive_weight=multi_positive_weight,
                    multi_positive_stats=multi_pos_stats,
                    knn_filter=knn_filter,
                    knn_filter_stats=knn_filter_stats,
                    knn_pos_ids=knn_pos_ids,
                    knn_pos_weights=knn_pos_weights,
                    knn_pos_valid=knn_pos_valid,
                    knn_pos_z2=knn_pos_z2,
                    knn_soft_pos_stats=knn_soft_pos_stats,
                )
                # M1/M1b: auxiliary morphology MSE on shared seed embeddings
                if morph_head is not None and morph_cfg is not None:
                    morph_loss = morph_expert_loss_hetero_step(
                        view1,
                        z1_seed,
                        seed_id1,
                        batch,
                        morph_head,
                        morph_cfg,
                        tier0_ctx=getattr(args, "morph_tier0_train", None),
                        flow_ctx=getattr(args, "morph_tier0_flow_train", None),
                        tier2_ctx=getattr(args, "morph_tier2_train", None),
                        diagnostics_accumulator=morph_diag_accumulator,
                    )
                    loss_raw = loss_raw + morph_loss
                    morph_loss_sum = morph_loss_sum + morph_loss.detach()
                aux_weight = float(getattr(args, "masked_edge_aux_weight", 0.0))
                masked_decoder = getattr(args, "masked_edge_decoder", None)
                masked_spec = getattr(args, "masked_edge_spec", None)
                if aux_weight > 0.0 and masked_decoder is not None and masked_spec is not None:
                    aux_batch = batch.clone()
                    aux_gen = _masked_edge_generator(
                        device, int(getattr(args, "masked_edge_seed", 1)), epoch, step
                    )
                    aux_batch, aux_state = prepare_masked_edge_batch(
                        aux_batch,
                        spec=masked_spec,
                        seed_edge_ids=seed_edge_ids,
                        is_hetero=True,
                        generator=aux_gen,
                        loader_data=tr_loader.data,
                    )
                    attach_edge_id_from_batch(aux_batch, tr_loader.data)
                    with autocast(enabled=use_amp):
                        aux_out = model(
                            aux_batch.x_dict,
                            aux_batch.edge_index_dict,
                            aux_batch.edge_attr_dict,
                        )
                        z_aux = aux_out[FORWARD_EDGE_TYPE]
                        z_aux_seed = z_aux[aux_state.seed_mask_fwd]
                    with autocast(enabled=False):
                        aux_loss, _ = compute_masked_edge_loss(
                            z_aux_seed, aux_state, masked_decoder, masked_spec
                        )
                    loss_raw = loss_raw + aux_weight * aux_loss
            loss = loss_raw / float(accum_steps)

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            step_one_indexed = step + 1
            should_step = (step_one_indexed % accum_steps == 0) or (
                n_train_batches is not None and step_one_indexed == n_train_batches
            )
            if should_step:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if memory_queue is not None and seed_id2.numel() > 0:
                z2_queue = project_seeds(proj_head, z2_seed)
                queue_endpoints = (
                    _edge_endpoints_for_ids(train_edge_index, seed_id2, device)
                    if false_neg_filter_mode != "none" or multi_positive_mode != "none"
                    else None
                )
                memory_queue.enqueue(z2_queue, seed_id2, queue_endpoints)

            loss_sum = loss_sum + loss_raw.detach()
            total_examples += 1

        avg_loss = float((loss_sum / max(total_examples, 1)).cpu())
        log_payload = {"loss/train": avg_loss}
        if morph_head is not None:
            avg_morph = float((morph_loss_sum / max(total_examples, 1)).cpu())
            log_payload["morph/expert_train"] = avg_morph
            log_payload["morphology/loss_total"] = avg_morph
            morph_diag_metrics = finalize_morph_expert_diagnostics(morph_diag_accumulator)
            log_payload.update(morph_diag_metrics)
            if val_loader is not None and should_run_morph_val(epoch, config.epochs, args):
                log_payload["morph/expert_val"] = eval_morph_expert_val_hetero(
                    val_loader, model, morph_head, morph_cfg, device, args
                )
            logging.info(f"Train Loss: {avg_loss:.4f} | morph/expert_train: {avg_morph:.4f}")
            _log_morphology_group_losses("hetero/train", log_payload)
        else:
            logging.info(f"Train Loss: {avg_loss:.4f}")
        _log_false_neg_filter_stats("hetero/train", false_neg_stats, false_neg_filter_mode)
        _log_multi_positive_stats("hetero/train", multi_pos_stats, multi_positive_mode, multi_positive_weight)
        _log_knn_filter_stats("hetero/train", knn_filter_stats)
        _log_knn_soft_pos_stats("hetero/train", knn_soft_pos_stats)
        _log_edge_drop_stats("hetero/train", edge_drop_stats)
        if (
            morph_contrast_cfg is not None
            and val_loader is not None
            and should_run_morph_val(epoch, config.epochs, args)
        ):
            log_payload["morph/contrast_val"] = eval_morph_contrast_val_hetero(
                val_loader, model, morph_contrast_cfg, device, args
            )
        ckpt_tracker.on_epoch_end(epoch, log_payload, model, optimizer, args, data_config)
        if ckpt_tracker.policy == "best":
            log_payload["checkpoint/best_epoch"] = ckpt_tracker.best_epoch
            log_payload["checkpoint/best_score"] = (
                ckpt_tracker.best_score if ckpt_tracker.best_epoch >= 0 else float("nan")
            )
        wandb.log(log_payload, step=epoch)

    ckpt_tracker.finalize(config.epochs - 1, model, optimizer, args, data_config)
    return model


def _masked_edge_generator(device: torch.device, base_seed: int, epoch: int, step: int) -> torch.Generator:
    gen = torch.Generator(device=device)
    gen.manual_seed(int(base_seed) + epoch * 1_000_003 + step * 9_173)
    return gen


def _accumulate_masked_edge_logs(
    accum: Dict[str, float],
    loss_logs: Dict[str, float],
    weight: int,
) -> None:
    for key, val in loss_logs.items():
        accum[key] = accum.get(key, 0.0) + float(val) * weight


def train_hetero_masked_edge(
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
    """Masked edge-attribute reconstruction on heterogeneous transaction graphs."""
    del te_loader, tr_inds, val_inds, te_inds, loss_fn, te_data, val_loader, val_data
    use_amp = bool(getattr(args, "amp", False)) and device.type == "cuda"
    spec = args.masked_edge_spec
    decoder = args.masked_edge_decoder
    mask_seed = int(getattr(args, "masked_edge_seed", 1))
    ckpt_tracker = CheckpointTracker(getattr(args, "checkpoint_policy", "last"))
    if ckpt_tracker.policy == "best":
        logging.info("Checkpoint policy: best (lowest loss/train)")

    for epoch in range(config.epochs):
        loss_sum = 0.0
        total_seed_edges = 0
        mask_rate_sum = {field: 0.0 for field in spec.fields}
        loss_accum: Dict[str, float] = {}
        n_batches = 0
        for step, batch in enumerate(tqdm.tqdm(tr_loader, disable=not args.tqdm)):
            optimizer.zero_grad(set_to_none=True)
            seed_edge_ids = get_hetero_seed_edge_ids(batch, tr_loader.data)
            batch.to(device, non_blocking=True)
            gen = _masked_edge_generator(device, mask_seed, epoch, step)
            batch, mstate = prepare_masked_edge_batch(
                batch,
                spec=spec,
                seed_edge_ids=seed_edge_ids,
                is_hetero=True,
                generator=gen,
                loader_data=tr_loader.data,
            )
            attach_edge_id_from_batch(batch, tr_loader.data)

            with autocast(enabled=use_amp):
                out = model(
                    batch.x_dict,
                    batch.edge_index_dict,
                    batch.edge_attr_dict,
                )
                z = out[FORWARD_EDGE_TYPE]
                z_seed = z[mstate.seed_mask_fwd]

            with autocast(enabled=False):
                loss, loss_logs = compute_masked_edge_loss(z_seed, mstate, decoder, spec)

            loss.backward()
            optimizer.step()

            n_seed = int(mstate.seed_mask_fwd.sum().item())
            if n_seed > 0:
                loss_sum += float(loss.detach()) * n_seed
                total_seed_edges += n_seed
                _accumulate_masked_edge_logs(loss_accum, loss_logs, n_seed)
            for field in spec.fields:
                mask_rate_sum[field] += mstate.stats.get(f"mask_rate_{field}", 0.0)
            n_batches += 1

        avg_loss = loss_sum / max(total_seed_edges, 1)
        log_payload = {
            "loss/train": avg_loss,
            "masked_edge/total": loss_accum.get("masked_edge/total", 0.0) / max(total_seed_edges, 1),
        }
        for field in spec.fields:
            log_payload[f"masked_edge/{field}"] = (
                loss_accum.get(f"masked_edge/{field}", 0.0) / max(total_seed_edges, 1)
            )
            log_payload[f"masked_edge/mask_rate_{field}"] = mask_rate_sum[field] / max(n_batches, 1)
        logging.info(
            "Epoch %s masked_edge loss/train=%.4f seed_edges=%s",
            epoch + 1,
            avg_loss,
            total_seed_edges,
        )
        ckpt_tracker.on_epoch_end(epoch, log_payload, model, optimizer, args, data_config)
        if ckpt_tracker.policy == "best":
            log_payload["checkpoint/best_epoch"] = ckpt_tracker.best_epoch
            log_payload["checkpoint/best_score"] = (
                ckpt_tracker.best_score if ckpt_tracker.best_epoch >= 0 else float("nan")
            )
        wandb.log(log_payload, step=epoch)

    ckpt_tracker.finalize(config.epochs - 1, model, optimizer, args, data_config)
    return model


def train_homo_masked_edge(
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
    """Masked edge-attribute reconstruction on homogeneous graphs."""
    del te_loader, tr_inds, val_inds, te_inds, loss_fn, te_data, val_loader, val_data
    use_amp = bool(getattr(args, "amp", False)) and device.type == "cuda"
    spec = args.masked_edge_spec
    decoder = args.masked_edge_decoder
    mask_seed = int(getattr(args, "masked_edge_seed", 1))
    ckpt_tracker = CheckpointTracker(getattr(args, "checkpoint_policy", "last"))

    for epoch in range(config.epochs):
        loss_sum = 0.0
        total_seed_edges = 0
        mask_rate_sum = {field: 0.0 for field in spec.fields}
        loss_accum: Dict[str, float] = {}
        n_batches = 0
        for step, batch in enumerate(tqdm.tqdm(tr_loader, disable=not args.tqdm)):
            optimizer.zero_grad(set_to_none=True)
            seed_edge_ids = get_homo_seed_edge_ids(batch, tr_loader.data)
            batch.to(device, non_blocking=True)
            gen = _masked_edge_generator(device, mask_seed, epoch, step)
            batch, mstate = prepare_masked_edge_batch(
                batch,
                spec=spec,
                seed_edge_ids=seed_edge_ids,
                is_hetero=False,
                generator=gen,
            )
            attach_edge_id_from_batch(batch)
            with autocast(enabled=use_amp):
                z = model(batch.x, batch.edge_index, batch.edge_attr)
                z_seed = z[mstate.seed_mask_fwd]
            with autocast(enabled=False):
                loss, loss_logs = compute_masked_edge_loss(z_seed, mstate, decoder, spec)
            loss.backward()
            optimizer.step()

            n_seed = int(mstate.seed_mask_fwd.sum().item())
            if n_seed > 0:
                loss_sum += float(loss.detach()) * n_seed
                total_seed_edges += n_seed
                _accumulate_masked_edge_logs(loss_accum, loss_logs, n_seed)
            for field in spec.fields:
                mask_rate_sum[field] += mstate.stats.get(f"mask_rate_{field}", 0.0)
            n_batches += 1

        avg_loss = loss_sum / max(total_seed_edges, 1)
        log_payload = {"loss/train": avg_loss}
        for field in spec.fields:
            log_payload[f"masked_edge/mask_rate_{field}"] = mask_rate_sum[field] / max(n_batches, 1)
        ckpt_tracker.on_epoch_end(epoch, log_payload, model, optimizer, args, data_config)
        wandb.log(log_payload, step=epoch)

    ckpt_tracker.finalize(config.epochs - 1, model, optimizer, args, data_config)
    return model


def build_edge_adjacency(edge_index):
    """
    Builds edge-level adjacency matrix:
    A[i, j] = 1 if edge i and edge j share a node (touch the same node somewhere)

    Args:
        edge_index: (2, E)

    Returns:
        A_edge: (E, E) boolean tensor
    """
    src = edge_index[0]
    dst = edge_index[1]

    # edges share a node if:
    # src_i == src_j OR src_i == dst_j OR dst_i == src_j OR dst_i == dst_j
    same_src = src.unsqueeze(1) == src.unsqueeze(0)
    src_dst = src.unsqueeze(1) == dst.unsqueeze(0)
    dst_src = dst.unsqueeze(1) == src.unsqueeze(0)
    same_dst = dst.unsqueeze(1) == dst.unsqueeze(0)

    A_edge = same_src | src_dst | dst_src | same_dst

    return A_edge


def get_model(sample_batch, config, args):
    n_feats = sample_batch.x.shape[1] if not isinstance(sample_batch, HeteroData) else sample_batch['node'].x.shape[1]
    e_dim = (sample_batch.edge_attr.shape[1] - 1) if not isinstance(sample_batch, HeteroData) else (sample_batch['node', 'to', 'node'].edge_attr.shape[1] - 1)

    if args.model == "gin":
        model = GINe(
                num_features=n_feats, num_gnn_layers=config.n_gnn_layers, n_classes=2,
                n_hidden=round(config.n_hidden), residual=False, edge_updates=args.emlps, edge_dim=e_dim,
                dropout=config.dropout, final_dropout=config.final_dropout,
                use_gradient_checkpointing=getattr(args, "gradient_checkpointing", False),
                )
    elif args.model == "gat":
        model = GATe(
                num_features=n_feats, num_gnn_layers=config.n_gnn_layers, n_classes=2,
                n_hidden=round(config.n_hidden), n_heads=round(config.n_heads),
                edge_updates=args.emlps, edge_dim=e_dim,
                dropout=config.dropout, final_dropout=config.final_dropout
                )
    elif args.model == "pna":
        if not isinstance(sample_batch, HeteroData):
            d = degree(sample_batch.edge_index[1], dtype=torch.long)
        else:
            index = torch.cat((sample_batch['node', 'to', 'node'].edge_index[1], sample_batch['node', 'rev_to', 'node'].edge_index[1]), 0)
            d = degree(index, dtype=torch.long)
        deg = torch.bincount(d, minlength=1)
        model = PNA(
            num_features=n_feats, num_gnn_layers=config.n_gnn_layers, n_classes=2,
            n_hidden=round(config.n_hidden), edge_updates=args.emlps, edge_dim=e_dim,
            dropout=config.dropout, deg=deg, final_dropout=config.final_dropout
            )
    elif args.model == "rgcn":
        num_relations = 2 if args.reverse_mp else 1
        model = RGCN(
            num_features=n_feats, edge_dim=e_dim, num_relations=num_relations,
            num_gnn_layers=round(config.n_gnn_layers),
            n_classes=2, n_hidden=round(config.n_hidden),
            edge_update=args.emlps, dropout=config.dropout, final_dropout=config.final_dropout, n_bases=None
        )

    return model


def train_gnn(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, args, data_config):
    setup = resolve_training_setup(args)
    validate_training_setup(setup)
    validate_masked_edge_args(args, setup)
    log_training_setup(setup, args)

    #set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #define a model config dictionary and wandb logging at the same time
    wandb.init(
        mode="disabled" if args.testing else "online",
        project="multi-gnn", #replace this with your wandb project name if you want to use wandb logging

        config={
            "epochs": args.n_epochs,
            "batch_size": args.batch_size,
            "model": args.model,
            "data": args.data,
            "graph_form": setup.graph_form,
            "objective": setup.objective,
            "reverse_mp": bool(args.reverse_mp),
            "finetune": bool(args.finetune),
            "num_neighbors": args.num_neighs,
            "lr": extract_param("lr", args),
            "n_hidden": extract_param("n_hidden", args),
            "n_gnn_layers": extract_param("n_gnn_layers", args),
            "loss": (
                "infonce"
                if setup.is_contrastive
                else ("masked_edge" if setup.is_masked_edge else "ce")
            ),
            "w_ce1": extract_param("w_ce1", args),
            "w_ce2": extract_param("w_ce2", args),
            "dropout": extract_param("dropout", args),
            "final_dropout": extract_param("final_dropout", args),
            "n_heads": extract_param("n_heads", args) if args.model == 'gat' else None,
            "amp": bool(getattr(args, "amp", False)),
            "gradient_checkpointing": bool(getattr(args, "gradient_checkpointing", False)),
            "contrastive_num_neg_samples": int(getattr(args, "contrastive_num_neg_samples", 0)),
            "contrastive_asymmetric": bool(getattr(args, "contrastive_asymmetric", False)),
            "contrastive_accum_steps": int(getattr(args, "contrastive_accum_steps", 1)),
            "contrastive_memory_bank_size": int(getattr(args, "contrastive_memory_bank_size", 0)),
            "enable_knn_negative_filter": bool(getattr(args, "enable_knn_negative_filter", False)),
            "knn_cache_path": getattr(args, "knn_cache_path", None),
            "knn_filter_k": int(getattr(args, "knn_filter_k", 0)),
            "enable_knn_soft_positives": bool(getattr(args, "enable_knn_soft_positives", False)),
            "knn_pos_source_k": int(getattr(args, "knn_pos_source_k", 15)),
            "knn_pos_m": int(getattr(args, "knn_pos_m", 1)),
            "knn_pos_weight": float(getattr(args, "knn_pos_weight", 0.025)),
            "knn_pos_weight_mode": getattr(args, "knn_pos_weight_mode", "uniform"),
            "knn_pos_min_sim": getattr(args, "knn_pos_min_sim", None),
            "knn_pos_seed": int(getattr(args, "knn_pos_seed", 0)),
            "loader_num_workers": int(getattr(args, "loader_num_workers", 10)),
            "morph_expert": bool(getattr(args, "morph_expert", False)),
            "morph_targets": getattr(args, "morph_targets", "local"),
            "morph_tier0_cache": getattr(args, "morph_tier0_cache", None),
            "morph_tier2_cache": getattr(args, "morph_tier2_cache", None),
            "morph_tier2_lift": getattr(args, "morph_tier2_lift", "full"),
            "morph_expert_weight": float(getattr(args, "morph_expert_weight", 1.0)),
            "morph_expert_layout": getattr(args, "morph_expert_layout", "shared"),
            "morph_expert_group_weight_tier2": float(
                getattr(args, "morph_expert_group_weight_tier2", 1.0)
            ),
            "morph_contrast": bool(getattr(args, "morph_contrast", False)),
            "morph_contrast_features": getattr(args, "morph_contrast_features", "local_ego,local_degree"),
            "morph_contrast_scope": getattr(args, "morph_contrast_scope", "local"),
            "morph_contrast_bins": int(getattr(args, "morph_contrast_bins", 5)),
            "morph_contrast_max_soft_positives": int(
                getattr(args, "morph_contrast_max_soft_positives", 256)
            ),
            "morph_val_every": int(getattr(args, "morph_val_every", 1)),
            "morph_val_max_batches": int(getattr(args, "morph_val_max_batches", 0)),
            "mask_edge_attr_rate": float(getattr(args, "mask_edge_attr_rate", 0.15)),
            "mask_edge_attr_fields": getattr(args, "mask_edge_attr_fields", "amount,currency,payment_format"),
            "mask_edge_attr_token_strategy": getattr(args, "mask_edge_attr_token_strategy", "zero"),
            "masked_edge_decoder_hidden_dim": int(getattr(args, "masked_edge_decoder_hidden_dim", 128)),
            "masked_edge_seed": int(getattr(args, "masked_edge_seed", 1)),
            "masked_edge_aux_weight": float(getattr(args, "masked_edge_aux_weight", 0.0)),
        }
    )

    config = wandb.config

    #set the transform if ego ids should be used
    if args.ego:
        transform = AddEgoIds()
    else:
        transform = None

    #add the unique ids to later find the seed edges
    add_arange_ids([tr_data, val_data, te_data])

    tr_loader, val_loader, te_loader = get_loaders(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform, args)
    logging.info("LinkNeighborLoader num_workers=%s", int(getattr(args, "loader_num_workers", 10)))

    #get the model
    sample_batch = next(iter(tr_loader))
    model = get_model(sample_batch, config, args)

    if args.reverse_mp:
        model = to_hetero(model, te_data.metadata(), aggr='mean')

    morph_head, morph_cfg = setup_morphology_expert(args, tr_data, device, setup.is_hetero)
    args.morph_expert_head = morph_head
    args.morph_expert_cfg = morph_cfg
    morph_contrast_cfg = setup_morphology_contrast(args, device)
    args.morph_contrast_cfg = morph_contrast_cfg
    proj_head = setup_contrastive_projection(args, device)
    args.contrast_projection_module = proj_head
    args._val_data_for_morph = val_data

    args.masked_edge_spec = None
    args.masked_edge_decoder = None
    masked_edge_aux_weight = float(getattr(args, "masked_edge_aux_weight", 0.0))
    if setup.is_masked_edge or masked_edge_aux_weight > 0.0:
        spec = build_masked_edge_spec(tr_data, args, device=device)
        args.masked_edge_spec = spec
        embed_dim = int(getattr(model, "embedding_dim", 128))
        decoder = setup_masked_edge_decoder(args, spec, device, embed_dim=embed_dim)
        args.masked_edge_decoder = decoder

    need_tier0 = (
        (morph_cfg is not None and morph_cfg.include_global)
        or (morph_contrast_cfg is not None and morph_contrast_cfg.include_global)
    )
    if need_tier0:
        setup_morph_tier0_contexts(args, tr_data, val_data, device)

    if morph_cfg is not None and morph_cfg.include_flow_balance:
        setup_morph_tier0_flow_contexts(args, tr_data, val_data, device)

    need_tier2 = morph_cfg is not None and morph_cfg.include_tier2
    if need_tier2:
        from morphology.tier2_global import setup_morph_tier2_contexts

        setup_morph_tier2_contexts(args, tr_data, val_data, device)

    if morph_contrast_cfg is not None and morph_contrast_cfg.include_edge_native:
        from morphology.contrastive_train import _edge_native_dim_for_contrast

        args._morph_contrast_edge_native_dim = _edge_native_dim_for_contrast(
            args, tr_data, setup.is_hetero
        )
    else:
        args._morph_contrast_edge_native_dim = 0

    if args.finetune:
        model, optimizer = load_model(model, device, args, config, data_config)
        load_checkpoint_auxiliary_modules(args, data_config, device)
        opt_params = list(model.parameters())
        if morph_head is not None:
            opt_params += list(morph_head.parameters())
        if proj_head is not None:
            opt_params += list(proj_head.parameters())
        masked_decoder = getattr(args, "masked_edge_decoder", None)
        if masked_decoder is not None:
            opt_params += list(masked_decoder.parameters())
        masked_spec = getattr(args, "masked_edge_spec", None)
        if masked_spec is not None and masked_spec.learned_mask_tokens is not None:
            opt_params += list(masked_spec.learned_mask_tokens.parameters())
        optimizer = torch.optim.Adam(opt_params, lr=config.lr)
    else:
        opt_params = list(model.parameters())
        if morph_head is not None:
            opt_params += list(morph_head.parameters())
        if proj_head is not None:
            opt_params += list(proj_head.parameters())
        masked_decoder = getattr(args, "masked_edge_decoder", None)
        if masked_decoder is not None:
            opt_params += list(masked_decoder.parameters())
        masked_spec = getattr(args, "masked_edge_spec", None)
        if masked_spec is not None and masked_spec.learned_mask_tokens is not None:
            opt_params += list(masked_spec.learned_mask_tokens.parameters())
        optimizer = torch.optim.Adam(opt_params, lr=config.lr)

    model.to(device)
    if morph_contrast_cfg is not None and setup.is_contrastive:
        setup_morph_contrast_bin_edges(args, tr_loader, model, device, setup.is_hetero)

    args.edge_drop_cache = None
    if setup.is_contrastive and getattr(args, "edge_drop_policy", "random") != "random":
        args.edge_drop_cache = load_or_build_edge_drop_cache(args, "data_config.json")
        logging.info(
            "Edge-drop policy=%s target_rate=%.4f cache_edges=%d",
            args.edge_drop_policy,
            float(args.edge_drop_target_rate),
            int(args.edge_drop_cache.drop_prob.shape[0]),
        )

    sample_batch.to(device)
    sample_x = sample_batch.x if not isinstance(sample_batch, HeteroData) else sample_batch.x_dict
    sample_edge_index = sample_batch.edge_index if not isinstance(sample_batch, HeteroData) else sample_batch.edge_index_dict
    if isinstance(sample_batch, HeteroData):
        sample_batch['node', 'to', 'node'].edge_attr = sample_batch['node', 'to', 'node'].edge_attr[:, 1:]
        sample_batch['node', 'rev_to', 'node'].edge_attr = sample_batch['node', 'rev_to', 'node'].edge_attr[:, 1:]
    else:
        sample_batch.edge_attr = sample_batch.edge_attr[:, 1:]
    sample_edge_attr = sample_batch.edge_attr if not isinstance(sample_batch, HeteroData) else sample_batch.edge_attr_dict
    logging.info(summary(model, sample_x, sample_edge_index, sample_edge_attr))

    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([config.w_ce1, config.w_ce2]).to(device))

    train_kwargs = (
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
    )

    if setup.is_hetero and setup.is_contrastive:
        model = train_hetero_contrastive(*train_kwargs)
    elif setup.is_hetero and setup.is_masked_edge:
        model = train_hetero_masked_edge(*train_kwargs)
    elif setup.is_hetero:
        model = train_hetero_supervised(*train_kwargs)
    elif setup.is_contrastive:
        model = train_homo_contrastive(*train_kwargs)
    elif setup.is_masked_edge:
        model = train_homo_masked_edge(*train_kwargs)
    else:
        model = train_homo_supervised(*train_kwargs)

    wandb.finish()
