import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import tqdm
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import LinkNeighborLoader
from sklearn.metrics import f1_score

# Heterogeneous graph edge types (reverse MP).
FORWARD_EDGE_TYPE = ("node", "to", "node")
REVERSE_EDGE_TYPE = ("node", "rev_to", "node")


def edge_classifier_logits(model: torch.nn.Module, z: torch.Tensor) -> torch.Tensor:
    """Edge-level classifier logits (Sequential on homo; flattened ModuleList after to_hetero)."""
    clf = model.classifier
    if isinstance(clf, torch.nn.Sequential):
        return clf(z)
    if isinstance(clf, torch.nn.ModuleList):
        out = z
        for layer in clf:
            out = layer(out)
        return out
    return clf(z)


@dataclass(frozen=True)
class TrainingSetup:
    """Resolved graph form and training objective (independent of each other)."""

    graph_form: str
    objective: str

    @property
    def is_hetero(self) -> bool:
        return self.graph_form == "hetero"

    @property
    def is_contrastive(self) -> bool:
        return self.objective == "contrastive"


def resolve_training_setup(args) -> TrainingSetup:
    objective = str(getattr(args, "objective", "contrastive")).lower()
    if objective not in ("contrastive", "supervised"):
        raise ValueError(f"Unsupported --objective {objective!r}; use 'contrastive' or 'supervised'.")
    graph_form = "hetero" if bool(getattr(args, "reverse_mp", False)) else "homo"
    return TrainingSetup(graph_form=graph_form, objective=objective)


def validate_training_setup(setup: TrainingSetup) -> None:
    """Reserved for future incompatible flag combinations."""
    del setup


def log_training_setup(setup: TrainingSetup, args) -> None:
    logging.info(
        "Training setup: graph_form=%s objective=%s reverse_mp=%s finetune=%s",
        setup.graph_form,
        setup.objective,
        bool(getattr(args, "reverse_mp", False)),
        bool(getattr(args, "finetune", False)),
    )


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

def _strip_synthetic_edge_attr_id_column(store) -> None:
    """If column 0 of ``edge_attr`` duplicates ``edge_id`` (from :func:`add_arange_ids`), drop it."""
    ea = store.edge_attr
    if ea is None or ea.shape[1] < 2:
        return
    if torch.equal(store.edge_id, ea[:, 0].long().view(-1)):
        store.edge_attr = ea[:, 1:].clone()


def _attach_edge_id_edge_store(store) -> None:
    """Populate ``store.edge_id`` on one homogeneous edge store (Data or Hetero edge type)."""
    if getattr(store, "edge_id", None) is not None:
        if store.edge_id.shape[0] != store.edge_index.shape[1]:
            raise ValueError("edge_id length must match number of edges.")
        _strip_synthetic_edge_attr_id_column(store)
        return

    eid = getattr(store, "e_id", None)
    if eid is not None:
        eid = eid.long().view(-1).clone()
        if eid.shape[0] != store.edge_index.shape[1]:
            raise ValueError(
                f"e_id length {eid.shape[0]} must match edge_index ({store.edge_index.shape[1]})."
            )
        store.edge_id = eid
        _strip_synthetic_edge_attr_id_column(store)
        return

    if store.edge_attr is None or store.edge_attr.shape[1] < 1:
        raise ValueError(
            "No edge_id/e_id and no edge_attr id column; run add_arange_ids on graph data "
            "before the loader, or set edge_id."
        )
    store.edge_id = store.edge_attr[:, 0].long().clone()
    store.edge_attr = store.edge_attr[:, 1:].clone()


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


def attach_edge_id_from_batch(
    batch: Union[Data, HeteroData],
    loader_data: Union[Data, HeteroData, None] = None,
) -> None:
    """
    Populate ``edge_id`` for contrastive pairing (transaction identity on forward edges).

    Homogeneous batches: sets ``batch.edge_id``.

    Heterogeneous batches (requires ``loader_data``): sets forward ``edge_id`` to the
    transaction id and reverse ``edge_id`` to the same id (reverse ``add_arange_ids`` cols
    are offset by the global forward edge count). Strips synthetic id columns from both
    edge types before the GNN forward.

    Mutates ``batch`` in place.
    """
    if isinstance(batch, HeteroData):
        if loader_data is None or not isinstance(loader_data, HeteroData):
            raise TypeError("attach_edge_id_from_batch on HeteroData requires loader_data (full graph).")
        n_fwd_global = loader_data[FORWARD_EDGE_TYPE].edge_attr.shape[0]
        _attach_edge_id_edge_store(batch[FORWARD_EDGE_TYPE])
        rev = batch[REVERSE_EDGE_TYPE]
        if getattr(rev, "edge_id", None) is not None:
            if rev.edge_id.shape[0] != rev.edge_index.shape[1]:
                raise ValueError("reverse edge_id length must match reverse edge_index.")
            _strip_synthetic_edge_attr_id_column(rev)
            return
        if rev.edge_attr is None or rev.edge_attr.shape[1] < 1:
            raise ValueError("Reverse edges missing edge_attr id column from add_arange_ids().")
        rev.edge_id = rev.edge_attr[:, 0].long().clone() - int(n_fwd_global)
        rev.edge_attr = rev.edge_attr[:, 1:].clone()
        return

    if loader_data is not None:
        del loader_data
    _attach_edge_id_edge_store(batch)


def get_hetero_seed_edge_ids(batch: HeteroData, loader_data: HeteroData) -> torch.Tensor:
    """
    Stable transaction ids for seed edges in a heterogeneous ``LinkNeighborLoader`` batch.

    Seed edges are always forward transactions (``FORWARD_EDGE_TYPE``); contrastive loss
    uses only their embeddings, not reverse synthetic edges.
    """
    if not isinstance(batch, HeteroData) or not isinstance(loader_data, HeteroData):
        raise TypeError("get_hetero_seed_edge_ids requires HeteroData batch and loader_data.")
    store = batch[FORWARD_EDGE_TYPE]
    input_id = getattr(store, "input_id", None)
    if input_id is None:
        raise ValueError("Forward edge store is missing input_id; cannot resolve seed edge ids.")
    fwd_attr = loader_data[FORWARD_EDGE_TYPE].edge_attr
    if fwd_attr is None or fwd_attr.shape[1] < 1:
        raise ValueError("Loader data is missing forward edge_attr ids from add_arange_ids().")
    return fwd_attr[input_id.long().view(-1).cpu(), 0].long().clone()


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


def get_loaders(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform, args, train_shuffle=True):
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
        tr_loader = LinkNeighborLoader(
            tr_data,
            num_neighbors=args.num_neighs,
            batch_size=args.batch_size,
            shuffle=train_shuffle,
            transform=transform,
            **lw,
        )
        val_loader = LinkNeighborLoader(val_data,num_neighbors=args.num_neighs, edge_label_index=val_data.edge_index[:, val_inds],
                                        edge_label=val_data.y[val_inds], batch_size=args.batch_size, shuffle=False, transform=transform, **lw)
        te_loader =  LinkNeighborLoader(te_data,num_neighbors=args.num_neighs, edge_label_index=te_data.edge_index[:, te_inds],
                                edge_label=te_data.y[te_inds], batch_size=args.batch_size, shuffle=False, transform=transform, **lw)

    return tr_loader, val_loader, te_loader


def expected_seed_edge_ids(loader_data, split_inds, hetero: bool) -> torch.Tensor:
    """Global transaction ids (column 0 of ``edge_attr``) for all seed edges in a split."""
    if hetero:
        attr = loader_data[FORWARD_EDGE_TYPE].edge_attr
    else:
        attr = loader_data.edge_attr
    return attr[split_inds.long(), 0].long().clone()


def checkpoint_path(data_config, unique_name: str, *, finetuned: bool = False) -> Path:
    """Path to ``checkpoint_{unique_name}[ _finetuned].tar`` under ``model_to_load``."""
    suffix = "_finetuned" if finetuned else ""
    return Path(data_config["paths"]["model_to_load"]) / f"checkpoint_{unique_name}{suffix}.tar"


def load_checkpoint_weights(model, device, args, data_config) -> int:
    """Load ``model_state_dict`` from a pretrain/adapt checkpoint (no optimizer)."""
    finetuned = bool(getattr(args, "finetune", False))
    path = checkpoint_path(data_config, args.unique_name, finetuned=finetuned)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return int(checkpoint.get("epoch", -1))


def dedupe_seed_embeddings(
    edge_ids: torch.Tensor,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Keep one row per ``edge_id`` (first occurrence). Returns deduped tensors and duplicate count.
    """
    if edge_ids.numel() == 0:
        empty_z = embeddings.new_empty((0, embeddings.shape[1] if embeddings.dim() > 1 else 0))
        return (
            edge_ids.new_empty((0,), dtype=torch.long),
            empty_z,
            labels.new_empty((0,), dtype=torch.long),
            0,
        )

    edge_ids = edge_ids.long().view(-1)
    seen: Dict[int, int] = {}
    keep_indices: List[int] = []
    duplicates = 0
    for i, eid in enumerate(edge_ids.tolist()):
        if eid in seen:
            duplicates += 1
            continue
        seen[eid] = i
        keep_indices.append(i)

    idx = torch.tensor(keep_indices, dtype=torch.long, device=edge_ids.device)
    return edge_ids[idx], embeddings[idx], labels[idx], duplicates


def log_seed_coverage(
    found_edge_ids: torch.Tensor,
    expected_edge_ids: torch.Tensor,
    split_name: str,
) -> None:
    found = set(found_edge_ids.detach().cpu().tolist())
    expected = set(expected_edge_ids.detach().cpu().tolist())
    n_expected = len(expected)
    n_hit = len(expected & found)
    pct = 100.0 * n_hit / max(n_expected, 1)
    logging.info(
        "Embedding extraction %s: covered %d / %d seed edges (%.2f%%)",
        split_name,
        n_hit,
        n_expected,
        pct,
    )
    missing = len(expected - found)
    if missing > 0:
        logging.warning(
            "Embedding extraction %s: %d seed edges missing from loader pass",
            split_name,
            missing,
        )


def save_embedding_split_npz(
    path: Path,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    edge_ids: torch.Tensor,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        Z=embeddings.detach().cpu().numpy().astype(np.float32),
        y=labels.detach().cpu().numpy(),
        edge_id=edge_ids.detach().cpu().numpy(),
    )


@torch.no_grad()
def extract_seed_embeddings_homo(
    loader,
    split_inds,
    model,
    data,
    device,
    args,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collect frozen edge embeddings and labels for seed edges (homogeneous graphs)."""
    edge_id_chunks: List[torch.Tensor] = []
    z_chunks: List[torch.Tensor] = []
    y_chunks: List[torch.Tensor] = []

    split_inds_cpu = split_inds.detach().cpu()
    for batch in tqdm.tqdm(loader, disable=not args.tqdm, desc="extract homo"):
        batch_edge_inds = split_inds_cpu[batch.input_id.detach().cpu()]
        batch_edge_ids = loader.data.edge_attr.detach().cpu()[batch_edge_inds, 0]
        mask = torch.isin(batch.edge_attr[:, 0].detach().cpu(), batch_edge_ids)

        missing = ~torch.isin(batch_edge_ids, batch.edge_attr[:, 0].detach().cpu())
        if missing.sum() != 0 and (args.data == "Small_J" or args.data == "Small_Q"):
            missing_ids = batch_edge_ids[missing].int()
            n_ids = batch.n_id
            add_edge_index = data.edge_index[:, missing_ids].detach().clone()
            node_mapping = {value.item(): idx for idx, value in enumerate(n_ids)}
            add_edge_index = torch.tensor(
                [[node_mapping[val.item()] for val in row] for row in add_edge_index]
            )
            add_edge_attr = data.edge_attr[missing_ids, :].detach().clone()
            add_y = data.y[missing_ids].detach().clone()

            batch.edge_index = torch.cat((batch.edge_index, add_edge_index), 1)
            batch.edge_attr = torch.cat((batch.edge_attr, add_edge_attr), 0)
            batch.y = torch.cat((batch.y, add_y), 0)
            mask = torch.cat((mask, torch.ones(add_y.shape[0], dtype=torch.bool)))

        edge_ids = batch.edge_attr[:, 0].long().clone()
        batch.edge_attr = batch.edge_attr[:, 1:]

        batch.to(device)
        z = model(batch.x, batch.edge_index, batch.edge_attr)
        mask_dev = mask.to(device, non_blocking=True)
        edge_id_chunks.append(edge_ids[mask].detach().cpu())
        z_chunks.append(z[mask_dev].detach().cpu())
        y_chunks.append(batch.y[mask_dev].detach().cpu().long())

    edge_ids = torch.cat(edge_id_chunks, dim=0)
    embeddings = torch.cat(z_chunks, dim=0)
    labels = torch.cat(y_chunks, dim=0)
    edge_ids, embeddings, labels, n_dup = dedupe_seed_embeddings(edge_ids, embeddings, labels)
    if n_dup > 0:
        logging.info("extract homo: dropped %d duplicate seed rows", n_dup)
    return edge_ids, embeddings, labels


@torch.no_grad()
def extract_seed_embeddings_hetero(
    loader,
    split_inds,
    model,
    data,
    device,
    args,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collect frozen forward-edge embeddings for seed transactions (heterogeneous graphs)."""
    edge_id_chunks: List[torch.Tensor] = []
    z_chunks: List[torch.Tensor] = []
    y_chunks: List[torch.Tensor] = []
    store = FORWARD_EDGE_TYPE

    split_inds_cpu = split_inds.detach().cpu()
    for batch in tqdm.tqdm(loader, disable=not args.tqdm, desc="extract hetero"):
        fwd = batch[store]
        batch_edge_inds = split_inds_cpu[fwd.input_id.detach().cpu()]
        batch_edge_ids = loader.data[store].edge_attr.detach().cpu()[batch_edge_inds, 0]
        mask = torch.isin(fwd.edge_attr[:, 0].detach().cpu(), batch_edge_ids)

        missing = ~torch.isin(batch_edge_ids, fwd.edge_attr[:, 0].detach().cpu())
        if missing.sum() != 0 and (args.data == "Small_J" or args.data == "Small_Q"):
            missing_ids = batch_edge_ids[missing].int()
            n_ids = batch["node"].n_id
            add_edge_index = data[store].edge_index[:, missing_ids].detach().clone()
            node_mapping = {value.item(): idx for idx, value in enumerate(n_ids)}
            add_edge_index = torch.tensor(
                [[node_mapping[val.item()] for val in row] for row in add_edge_index]
            )
            add_edge_attr = data[store].edge_attr[missing_ids, :].detach().clone()
            add_y = data[store].y[missing_ids].detach().clone()

            fwd.edge_index = torch.cat((fwd.edge_index, add_edge_index), 1)
            fwd.edge_attr = torch.cat((fwd.edge_attr, add_edge_attr), 0)
            fwd.y = torch.cat((fwd.y, add_y), 0)
            mask = torch.cat((mask, torch.ones(add_y.shape[0], dtype=torch.bool)))

        edge_ids = fwd.edge_attr[:, 0].long().clone()
        fwd.edge_attr = fwd.edge_attr[:, 1:]
        batch[REVERSE_EDGE_TYPE].edge_attr = batch[REVERSE_EDGE_TYPE].edge_attr[:, 1:]

        batch.to(device)
        out = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
        z = out[store]
        mask_dev = mask.to(device, non_blocking=True)
        edge_id_chunks.append(edge_ids[mask].detach().cpu())
        z_chunks.append(z[mask_dev].detach().cpu())
        y_chunks.append(fwd.y[mask_dev].detach().cpu().long())

    edge_ids = torch.cat(edge_id_chunks, dim=0)
    embeddings = torch.cat(z_chunks, dim=0)
    labels = torch.cat(y_chunks, dim=0)
    edge_ids, embeddings, labels, n_dup = dedupe_seed_embeddings(edge_ids, embeddings, labels)
    if n_dup > 0:
        logging.info("extract hetero: dropped %d duplicate seed rows", n_dup)
    return edge_ids, embeddings, labels


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
            z = model(batch.x, batch.edge_index, batch.edge_attr)
            pred = model.classifier(z)[mask].argmax(dim=-1)
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
            z = model(
                batch.x_dict,
                batch.edge_index_dict,
                batch.edge_attr_dict,
            )[('node', 'to', 'node')]
            pred = edge_classifier_logits(model, z)[mask].argmax(dim=-1)
            preds.append(pred)
            ground_truths.append(batch['node', 'to', 'node'].y[mask])
    pred = torch.cat(preds, dim=0).cpu().numpy()
    ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
    f1 = f1_score(ground_truth, pred)

    return f1

def save_model(model, optimizer, epoch, args, data_config):
    # Save the model in a dictionary
    payload = {
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    morph_head = getattr(args, "morph_expert_head", None)
    if morph_head is not None:
        payload["morph_expert_state_dict"] = morph_head.state_dict()
    torch.save(
        payload,
        f'{data_config["paths"]["model_to_save"]}/checkpoint_{args.unique_name}{"" if not args.finetune else "_finetuned"}.tar',
    )

def load_model(model, device, args, config, data_config):
    checkpoint = torch.load(f'{data_config["paths"]["model_to_load"]}/checkpoint_{args.unique_name}.tar')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer
