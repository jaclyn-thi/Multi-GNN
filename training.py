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
)
from models import GINe, PNA, GATe, RGCN
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import to_hetero, summary
from torch_geometric.utils import degree
from pytorch_metric_learning.losses import NTXentLoss
from graph_augmentations import generate_views
from contrastive_loss import EdgeMemoryQueue, edge_identity_infonce_loss
from contrastive_projection import project_seed_pair, project_seeds, setup_contrastive_projection
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
from morphology.expert import setup_morphology_expert, setup_morph_tier0_contexts
import wandb
import logging


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
                edge_attr_mask_rate=0.1,
                edge_drop_rate=0.1,
                mask_value=0.0,
                mask_cols=None,
                exclude_last_column=(args.model == "rgcn"),
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
                loss_raw = edge_identity_infonce_loss(
                    z1_con,
                    z2_con,
                    seed_id1,
                    seed_id2,
                    temperature=0.5,
                    num_neg_samples=num_neg_samples,
                    symmetric=contrastive_symmetric,
                    memory_queue=memory_queue,
                    morph_bin_ids=morph_bins,
                    max_soft_positives=max_soft,
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
                        tier2_ctx=getattr(args, "morph_tier2_train", None),
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
                memory_queue.enqueue(z2_queue, seed_id2)

            loss_sum = loss_sum + loss_raw.detach()
            total_examples += 1

        avg_loss = float((loss_sum / max(total_examples, 1)).cpu())
        log_payload = {"loss/train": avg_loss}
        if morph_head is not None:
            avg_morph = float((morph_loss_sum / max(total_examples, 1)).cpu())
            log_payload["morph/expert_train"] = avg_morph
            if val_loader is not None and should_run_morph_val(epoch, config.epochs, args):
                log_payload["morph/expert_val"] = eval_morph_expert_val_homo(
                    val_loader, model, morph_head, morph_cfg, device, args
                )
            logging.info(f"Train Loss: {avg_loss:.4f} | morph/expert_train: {avg_morph:.4f}")
        else:
            logging.info(f"Train Loss: {avg_loss:.4f}")
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
                edge_attr_mask_rate=0.1,
                edge_drop_rate=0.1,
                mask_value=0.0,
                mask_cols=None,
                exclude_last_column=(args.model == "rgcn"),
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
                loss_raw = edge_identity_infonce_loss(
                    z1_con,
                    z2_con,
                    seed_id1,
                    seed_id2,
                    temperature=0.5,
                    num_neg_samples=num_neg_samples,
                    symmetric=contrastive_symmetric,
                    memory_queue=memory_queue,
                    morph_bin_ids=morph_bins,
                    max_soft_positives=max_soft,
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
                        tier2_ctx=getattr(args, "morph_tier2_train", None),
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
                memory_queue.enqueue(z2_queue, seed_id2)

            loss_sum = loss_sum + loss_raw.detach()
            total_examples += 1

        avg_loss = float((loss_sum / max(total_examples, 1)).cpu())
        log_payload = {"loss/train": avg_loss}
        if morph_head is not None:
            avg_morph = float((morph_loss_sum / max(total_examples, 1)).cpu())
            log_payload["morph/expert_train"] = avg_morph
            if val_loader is not None and should_run_morph_val(epoch, config.epochs, args):
                log_payload["morph/expert_val"] = eval_morph_expert_val_hetero(
                    val_loader, model, morph_head, morph_cfg, device, args
                )
            logging.info(f"Train Loss: {avg_loss:.4f} | morph/expert_train: {avg_morph:.4f}")
        else:
            logging.info(f"Train Loss: {avg_loss:.4f}")
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
    elif config.model == "rgcn":
        model = RGCN(
            num_features=n_feats, edge_dim=e_dim, num_relations=8, num_gnn_layers=round(config.n_gnn_layers),
            n_classes=2, n_hidden=round(config.n_hidden),
            edge_update=args.emlps, dropout=config.dropout, final_dropout=config.final_dropout, n_bases=None #(maybe)
        )

    return model


def train_gnn(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, args, data_config):
    setup = resolve_training_setup(args)
    validate_training_setup(setup)
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
            "loss": "infonce" if setup.is_contrastive else "ce",
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

    need_tier0 = (
        (morph_cfg is not None and morph_cfg.include_global)
        or (morph_contrast_cfg is not None and morph_contrast_cfg.include_global)
    )
    if need_tier0:
        setup_morph_tier0_contexts(args, tr_data, val_data, device)

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
        optimizer = torch.optim.Adam(opt_params, lr=config.lr)
    else:
        opt_params = list(model.parameters())
        if morph_head is not None:
            opt_params += list(morph_head.parameters())
        if proj_head is not None:
            opt_params += list(proj_head.parameters())
        optimizer = torch.optim.Adam(opt_params, lr=config.lr)

    model.to(device)
    if morph_contrast_cfg is not None and setup.is_contrastive:
        setup_morph_contrast_bin_edges(args, tr_loader, model, device, setup.is_hetero)

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
    elif setup.is_hetero:
        model = train_hetero_supervised(*train_kwargs)
    elif setup.is_contrastive:
        model = train_homo_contrastive(*train_kwargs)
    else:
        model = train_homo_supervised(*train_kwargs)

    wandb.finish()
