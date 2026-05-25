import torch
import tqdm
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import f1_score
from train_util import (
    AddEgoIds,
    extract_param,
    add_arange_ids,
    attach_edge_id_from_batch,
    get_loaders,
    evaluate_homo,
    evaluate_hetero,
    save_model,
    load_model,
)
from models import GINe, PNA, GATe, RGCN
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import to_hetero, summary
from torch_geometric.utils import degree
from pytorch_metric_learning.losses import NTXentLoss
from graph_augmentations import generate_views
from contrastive_loss import edge_identity_infonce_loss
import wandb
import logging


def train_homo(tr_loader, val_loader, te_loader, tr_inds, val_inds, te_inds, model, optimizer, loss_fn, args, config, device, val_data, te_data, data_config):
    #training
    # NOTE: The following arguments are unused in contrastive pretraining:
    # tr_inds, val_loader, te_loader, val_inds, te_inds, loss_fn, val_data, te_data
    # best_val_f1 = 0
    use_amp = bool(getattr(args, "amp", False)) and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)
    neg_kw = int(getattr(args, "contrastive_num_neg_samples", 0))
    num_neg_samples = neg_kw if neg_kw > 0 else None
    contrastive_symmetric = not bool(getattr(args, "contrastive_asymmetric", False))
    accum_steps = max(1, int(getattr(args, "contrastive_accum_steps", 1)))
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

    for epoch in range(config.epochs):
        total_loss = total_examples = 0
        # preds = []
        # ground_truths = []
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(tqdm.tqdm(tr_loader, disable=not args.tqdm)):
            # (old) seed edges and masking
            #select the seed edges from which the batch was created
            # inds = tr_inds.detach().cpu()
            # batch_edge_inds = inds[batch.input_id.detach().cpu()]
            # batch_edge_ids = tr_loader.data.edge_attr.detach().cpu()[batch_edge_inds, 0]
            # mask = torch.isin(batch.edge_attr[:, 0].detach().cpu(), batch_edge_ids)

            attach_edge_id_from_batch(batch)

            batch.to(device)
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
                z1 = model(view1.x, view1.edge_index, view1.edge_attr, return_embeddings=True)
                if contrastive_symmetric:
                    z2 = model(view2.x, view2.edge_index, view2.edge_attr, return_embeddings=True)
                else:
                    with torch.no_grad():
                        z2 = model(view2.x, view2.edge_index, view2.edge_attr, return_embeddings=True)
            with autocast(enabled=False):
                loss_raw = edge_identity_infonce_loss(
                    z1,
                    z2,
                    view1.edge_id,
                    view2.edge_id,
                    temperature=0.5,
                    num_neg_samples=num_neg_samples,
                    symmetric=contrastive_symmetric,
                )
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

            # (old) loss scaling
            # total_loss += float(loss) * pred.numel()
            # total_examples += pred.numel()

            total_loss += float(loss_raw.detach().item())
            total_examples += 1

        avg_loss = total_loss / total_examples
        wandb.log({"loss/train": avg_loss}, step=epoch)
        logging.info(f"Train Loss: {avg_loss:.4f}")

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

    return model


def train_hetero(tr_loader, val_loader, te_loader, tr_inds, val_inds, te_inds, model, optimizer, loss_fn, args, config, device, val_data, te_data, data_config):
    #training
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
            out = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
            out = out[('node', 'to', 'node')]
            pred = out[mask]
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
            "num_neighbors": args.num_neighs,
            "lr": extract_param("lr", args),
            "n_hidden": extract_param("n_hidden", args),
            "n_gnn_layers": extract_param("n_gnn_layers", args),
            "loss": "ce",
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
            "loader_num_workers": int(getattr(args, "loader_num_workers", 10)),
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

    if args.finetune:
        model, optimizer = load_model(model, device, args, config, data_config)
    else:
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

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

    # TODO switch to contrastive loss. keep class weights?
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([config.w_ce1, config.w_ce2]).to(device))

    if args.reverse_mp:
        model = train_hetero(tr_loader, val_loader, te_loader, tr_inds, val_inds, te_inds, model, optimizer, loss_fn, args, config, device, val_data, te_data, data_config)
    else:
        model = train_homo(tr_loader, val_loader, te_loader, tr_inds, val_inds, te_inds, model, optimizer, loss_fn, args, config, device, val_data, te_data, data_config)

    wandb.finish()
