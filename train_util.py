import torch
import tqdm
from torch_geometric.transforms import BaseTransform
from typing import Tuple, Union
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import LinkNeighborLoader
from sklearn.metrics import f1_score
import json


class AddEgoIds(BaseTransform):
    r"""Add IDs to the centre nodes of the batch.
    """
    def __init__(self):
        pass

    def __call__(self, data: Union[Data, HeteroData]):
        x = data.x if not isinstance(data, HeteroData) else data['node'].x
        device = x.device
        ids = torch.zeros((x.shape[0], 1), device=device)
        if not isinstance(data, HeteroData):
            nodes = torch.unique(data.edge_label_index.view(-1)).to(device)
        else:
            nodes = torch.unique(data['node', 'to', 'node'].edge_label_index.view(-1)).to(device)
        ids[nodes] = 1
        if not isinstance(data, HeteroData):
            data.x = torch.cat([x, ids], dim=1)
        else:
            data['node'].x = torch.cat([x, ids], dim=1)

        return data

def extract_param(parameter_name: str, args) -> float:
    """
    Extract the value of the specified parameter for the given model.

    Args:
    - parameter_name (str): Name of the parameter (e.g., "lr").
    - args (argparser): Arguments given to this specific run.

    Returns:
    - float: Value of the specified parameter.
    """
    file_path = './model_settings.json'
    with open(file_path, "r") as file:
        data = json.load(file)

    return data.get(args.model, {}).get("params", {}).get(parameter_name, None)

def _strip_synthetic_edge_attr_id_column(batch: Data) -> None:
    """If column 0 of ``edge_attr`` duplicates ``batch.edge_id`` (from :func:`add_arange_ids`), drop it."""
    ea = batch.edge_attr
    if ea is None or ea.shape[1] < 2:
        return
    if torch.equal(batch.edge_id, ea[:, 0].long().view(-1)):
        batch.edge_attr = ea[:, 1:].clone()


def add_arange_ids(data_list):
    '''
    Add the index as an id to the edge features to find seed edges in training, validation and testing.

    Args:
    - data_list (str): List of tr_data, val_data and te_data.
    '''
    for data in data_list:
        if isinstance(data, HeteroData):
            data['node', 'to', 'node'].edge_attr = torch.cat([torch.arange(data['node', 'to', 'node'].edge_attr.shape[0]).view(-1, 1), data['node', 'to', 'node'].edge_attr], dim=1)
            offset = data['node', 'to', 'node'].edge_attr.shape[0]
            data['node', 'rev_to', 'node'].edge_attr = torch.cat([torch.arange(offset, data['node', 'rev_to', 'node'].edge_attr.shape[0] + offset).view(-1, 1), data['node', 'rev_to', 'node'].edge_attr], dim=1)
        else:
            data.edge_attr = torch.cat([torch.arange(data.edge_attr.shape[0]).view(-1, 1), data.edge_attr], dim=1)


def attach_edge_id_from_batch(batch: Union[Data, HeteroData]) -> None:
    """
    Populate ``batch.edge_id`` (long, one per message-passing edge) for contrastive pairing.

    Prefer an existing stable id from the batch (``edge_id``, PyG ``e_id``). Otherwise use
    column 0 of ``edge_attr`` from :func:`add_arange_ids`, then remove that column so the
    GNN does not see ids as features.

    When ``edge_id``/``e_id`` comes from the loader but ``edge_attr`` still has a leading
    id column from :func:`add_arange_ids` (matching ``batch.edge_id``), that column is
    removed as well.

    Mutates ``batch`` in place (same pattern as training/eval stripping of the id column).
    """
    if isinstance(batch, HeteroData):
        raise TypeError("attach_edge_id_from_batch supports homogeneous Data batches only.")

    if getattr(batch, "edge_id", None) is not None:
        if batch.edge_id.shape[0] != batch.edge_index.shape[1]:
            raise ValueError("batch.edge_id length must match number of edges.")
        _strip_synthetic_edge_attr_id_column(batch)
        return

    eid = getattr(batch, "e_id", None)
    if eid is not None:
        eid = eid.long().view(-1).clone()
        if eid.shape[0] != batch.edge_index.shape[1]:
            raise ValueError(
                f"e_id length {eid.shape[0]} must match edge_index ({batch.edge_index.shape[1]})."
            )
        batch.edge_id = eid
        _strip_synthetic_edge_attr_id_column(batch)
        return

    if batch.edge_attr is None or batch.edge_attr.shape[1] < 1:
        raise ValueError(
            "No edge_id/e_id and no edge_attr id column; run add_arange_ids on graph data "
            "before the loader, or set batch.edge_id."
        )
    batch.edge_id = batch.edge_attr[:, 0].long().clone()
    batch.edge_attr = batch.edge_attr[:, 1:].clone()


def get_homo_seed_edge_ids(batch: Data, loader_data: Data) -> torch.Tensor:
    """
    Resolve the stable edge ids for the *seed* edges used to form a homogeneous
    ``LinkNeighborLoader`` batch.

    ``add_arange_ids()`` stores the stable id in column 0 of ``loader_data.edge_attr``.
    PyG exposes the seed-edge positions as ``batch.input_id``.
    """
    if isinstance(batch, HeteroData):
        raise TypeError("get_homo_seed_edge_ids supports homogeneous Data batches only.")
    input_id = getattr(batch, "input_id", None)
    if input_id is None:
        raise ValueError("Batch is missing input_id; cannot resolve seed edge ids.")
    if getattr(loader_data, "edge_attr", None) is None or loader_data.edge_attr.shape[1] < 1:
        raise ValueError("Loader data is missing edge_attr ids from add_arange_ids().")
    seed_edge_ids = loader_data.edge_attr[input_id.long().view(-1).cpu(), 0]
    return seed_edge_ids.long().clone()


def select_shared_seed_edge_embeddings(
    z1: torch.Tensor,
    edge_id1: torch.Tensor,
    z2: torch.Tensor,
    edge_id2: torch.Tensor,
    seed_edge_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Keep only seed edges that survive in *both* augmented views and align them by
    stable ``edge_id`` so the contrastive loss scales with seed edges, not the full
    sampled subgraph.
    """
    edge_id1 = edge_id1.long().view(-1)
    edge_id2 = edge_id2.long().view(-1)
    seed_edge_ids = seed_edge_ids.to(edge_id1.device).long().view(-1)

    if seed_edge_ids.numel() == 0 or edge_id1.numel() == 0 or edge_id2.numel() == 0:
        empty1 = z1.new_empty((0, z1.shape[1]))
        empty2 = z2.new_empty((0, z2.shape[1]))
        empty_ids = edge_id1.new_empty((0,), dtype=torch.long)
        return empty1, empty_ids, empty2, empty_ids.clone()

    shared_ids = seed_edge_ids[
        torch.isin(seed_edge_ids, edge_id1) & torch.isin(seed_edge_ids, edge_id2)
    ]
    if shared_ids.numel() == 0:
        empty1 = z1.new_empty((0, z1.shape[1]))
        empty2 = z2.new_empty((0, z2.shape[1]))
        empty_ids = edge_id1.new_empty((0,), dtype=torch.long)
        return empty1, empty_ids, empty2, empty_ids.clone()

    shared_ids = torch.unique(shared_ids, sorted=True)

    keep1 = torch.isin(edge_id1, shared_ids)
    keep2 = torch.isin(edge_id2, shared_ids)

    z1_seed = z1[keep1]
    z2_seed = z2[keep2]
    ids1 = edge_id1[keep1]
    ids2 = edge_id2[keep2]

    order1 = torch.argsort(ids1)
    order2 = torch.argsort(ids2)
    z1_seed = z1_seed[order1]
    z2_seed = z2_seed[order2]
    ids1 = ids1[order1]
    ids2 = ids2[order2]

    if ids1.numel() != ids2.numel() or not torch.equal(ids1, ids2):
        raise ValueError("Failed to align shared seed-edge embeddings across views.")

    return z1_seed, ids1, z2_seed, ids2

def _link_neighbor_loader_kwargs(args) -> dict:
    """``num_workers`` + ``persistent_workers`` for PyG ``LinkNeighborLoader`` (same as DataLoader)."""
    n = max(0, int(getattr(args, "loader_num_workers", 10)))
    kw = {"num_workers": n}
    if n > 0:
        kw["persistent_workers"] = True
    return kw


def get_loaders(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform, args):
    lw = _link_neighbor_loader_kwargs(args)
    if isinstance(tr_data, HeteroData):
        tr_edge_label_index = tr_data['node', 'to', 'node'].edge_index
        tr_edge_label = tr_data['node', 'to', 'node'].y


        tr_loader =  LinkNeighborLoader(tr_data, num_neighbors=args.num_neighs,
                                    edge_label_index=(('node', 'to', 'node'), tr_edge_label_index),
                                    edge_label=tr_edge_label, batch_size=args.batch_size, shuffle=True, transform=transform, **lw)

        val_edge_label_index = val_data['node', 'to', 'node'].edge_index[:,val_inds]
        val_edge_label = val_data['node', 'to', 'node'].y[val_inds]


        val_loader =  LinkNeighborLoader(val_data, num_neighbors=args.num_neighs,
                                    edge_label_index=(('node', 'to', 'node'), val_edge_label_index),
                                    edge_label=val_edge_label, batch_size=args.batch_size, shuffle=False, transform=transform, **lw)

        te_edge_label_index = te_data['node', 'to', 'node'].edge_index[:,te_inds]
        te_edge_label = te_data['node', 'to', 'node'].y[te_inds]


        te_loader =  LinkNeighborLoader(te_data, num_neighbors=args.num_neighs,
                                    edge_label_index=(('node', 'to', 'node'), te_edge_label_index),
                                    edge_label=te_edge_label, batch_size=args.batch_size, shuffle=False, transform=transform, **lw)
    else:
        tr_loader =  LinkNeighborLoader(tr_data, num_neighbors=args.num_neighs, batch_size=args.batch_size, shuffle=True, transform=transform, **lw)
        val_loader = LinkNeighborLoader(val_data,num_neighbors=args.num_neighs, edge_label_index=val_data.edge_index[:, val_inds],
                                        edge_label=val_data.y[val_inds], batch_size=args.batch_size, shuffle=False, transform=transform, **lw)
        te_loader =  LinkNeighborLoader(te_data,num_neighbors=args.num_neighs, edge_label_index=te_data.edge_index[:, te_inds],
                                edge_label=te_data.y[te_inds], batch_size=args.batch_size, shuffle=False, transform=transform, **lw)

    return tr_loader, val_loader, te_loader

@torch.no_grad()
def evaluate_homo(loader, inds, model, data, device, args):
    '''Evaluates the model performane for homogenous graph data.'''
    preds = []
    ground_truths = []
    for batch in tqdm.tqdm(loader, disable=not args.tqdm):
        #select the seed edges from which the batch was created
        inds = inds.detach().cpu()
        batch_edge_inds = inds[batch.input_id.detach().cpu()]
        batch_edge_ids = loader.data.edge_attr.detach().cpu()[batch_edge_inds, 0]
        mask = torch.isin(batch.edge_attr[:, 0].detach().cpu(), batch_edge_ids)

        #add the seed edges that have not been sampled to the batch
        missing = ~torch.isin(batch_edge_ids, batch.edge_attr[:, 0].detach().cpu())

        if missing.sum() != 0 and (args.data == 'Small_J' or args.data == 'Small_Q'):
            missing_ids = batch_edge_ids[missing].int()
            n_ids = batch.n_id
            add_edge_index = data.edge_index[:, missing_ids].detach().clone()
            node_mapping = {value.item(): idx for idx, value in enumerate(n_ids)}
            add_edge_index = torch.tensor([[node_mapping[val.item()] for val in row] for row in add_edge_index])
            add_edge_attr = data.edge_attr[missing_ids, :].detach().clone()
            add_y = data.y[missing_ids].detach().clone()

            batch.edge_index = torch.cat((batch.edge_index, add_edge_index), 1)
            batch.edge_attr = torch.cat((batch.edge_attr, add_edge_attr), 0)
            batch.y = torch.cat((batch.y, add_y), 0)

            mask = torch.cat((mask, torch.ones(add_y.shape[0], dtype=torch.bool)))

        #remove the unique edge id from the edge features, as it's no longer needed
        batch.edge_attr = batch.edge_attr[:, 1:]

        with torch.no_grad():
            batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr)
            out = out[mask]
            pred = out.argmax(dim=-1)
            preds.append(pred)
            ground_truths.append(batch.y[mask])
    pred = torch.cat(preds, dim=0).cpu().numpy()
    ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
    f1 = f1_score(ground_truth, pred)

    return f1

@torch.no_grad()
def evaluate_hetero(loader, inds, model, data, device, args):
    '''Evaluates the model performane for heterogenous graph data.'''
    preds = []
    ground_truths = []
    for batch in tqdm.tqdm(loader, disable=not args.tqdm):
        #select the seed edges from which the batch was created
        inds = inds.detach().cpu()
        batch_edge_inds = inds[batch['node', 'to', 'node'].input_id.detach().cpu()]
        batch_edge_ids = loader.data['node', 'to', 'node'].edge_attr.detach().cpu()[batch_edge_inds, 0]
        mask = torch.isin(batch['node', 'to', 'node'].edge_attr[:, 0].detach().cpu(), batch_edge_ids)

        #add the seed edges that have not been sampled to the batch
        missing = ~torch.isin(batch_edge_ids, batch['node', 'to', 'node'].edge_attr[:, 0].detach().cpu())

        if missing.sum() != 0 and (args.data == 'Small_J' or args.data == 'Small_Q'):
            missing_ids = batch_edge_ids[missing].int()
            n_ids = batch['node'].n_id
            add_edge_index = data['node', 'to', 'node'].edge_index[:, missing_ids].detach().clone()
            node_mapping = {value.item(): idx for idx, value in enumerate(n_ids)}
            add_edge_index = torch.tensor([[node_mapping[val.item()] for val in row] for row in add_edge_index])
            add_edge_attr = data['node', 'to', 'node'].edge_attr[missing_ids, :].detach().clone()
            add_y = data['node', 'to', 'node'].y[missing_ids].detach().clone()

            batch['node', 'to', 'node'].edge_index = torch.cat((batch['node', 'to', 'node'].edge_index, add_edge_index), 1)
            batch['node', 'to', 'node'].edge_attr = torch.cat((batch['node', 'to', 'node'].edge_attr, add_edge_attr), 0)
            batch['node', 'to', 'node'].y = torch.cat((batch['node', 'to', 'node'].y, add_y), 0)

            mask = torch.cat((mask, torch.ones(add_y.shape[0], dtype=torch.bool)))

        #remove the unique edge id from the edge features, as it's no longer needed
        batch['node', 'to', 'node'].edge_attr = batch['node', 'to', 'node'].edge_attr[:, 1:]
        batch['node', 'rev_to', 'node'].edge_attr = batch['node', 'rev_to', 'node'].edge_attr[:, 1:]

        with torch.no_grad():
            batch.to(device)
            out = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
            out = out[('node', 'to', 'node')]
            out = out[mask]
            pred = out.argmax(dim=-1)
            preds.append(pred)
            ground_truths.append(batch['node', 'to', 'node'].y[mask])
    pred = torch.cat(preds, dim=0).cpu().numpy()
    ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
    f1 = f1_score(ground_truth, pred)

    return f1

def save_model(model, optimizer, epoch, args, data_config):
    # Save the model in a dictionary
    torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, f'{data_config["paths"]["model_to_save"]}/checkpoint_{args.unique_name}{"" if not args.finetune else "_finetuned"}.tar')

def load_model(model, device, args, config, data_config):
    checkpoint = torch.load(f'{data_config["paths"]["model_to_load"]}/checkpoint_{args.unique_name}.tar')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer
