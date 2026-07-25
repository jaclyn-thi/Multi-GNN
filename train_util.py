import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import tqdm
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import LinkNeighborLoader
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

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

    @property
    def is_masked_edge(self) -> bool:
        return self.objective == "masked_edge"


def resolve_training_setup(args) -> TrainingSetup:
    objective = str(getattr(args, "objective", "contrastive")).lower()
    if objective not in ("contrastive", "supervised", "masked_edge"):
        raise ValueError(
            f"Unsupported --objective {objective!r}; use 'contrastive', 'supervised', or 'masked_edge'."
        )
    graph_form = "hetero" if bool(getattr(args, "reverse_mp", False)) else "homo"
    return TrainingSetup(graph_form=graph_form, objective=objective)


def validate_training_setup(setup: TrainingSetup) -> None:
    """Reserved for incompatible flag combinations."""
    del setup


def validate_masked_edge_args(args, setup: TrainingSetup) -> None:
    if not setup.is_masked_edge:
        return
    blocked = []
    if bool(getattr(args, "morph_expert", False)):
        blocked.append("--morph_expert")
    if bool(getattr(args, "morph_contrast", False)):
        blocked.append("--morph_contrast")
    if bool(getattr(args, "contrast_projection_head", False)):
        blocked.append("--contrast_projection_head")
    if bool(getattr(args, "enable_knn_negative_filter", False)):
        blocked.append("--enable_knn_negative_filter")
    if bool(getattr(args, "enable_knn_soft_positives", False)):
        blocked.append("--enable_knn_soft_positives")
    if getattr(args, "false_neg_filter_mode", "none") != "none":
        blocked.append("--false_neg_filter_mode")
    if blocked:
        raise ValueError(
            "masked_edge objective is incompatible with: " + ", ".join(blocked)
        )


def log_training_setup(setup: TrainingSetup, args) -> None:
    logging.info(
        "Training setup: graph_form=%s objective=%s reverse_mp=%s finetune=%s "
        "correct_reverse_edge_features=%s reverse_edge_feature_semantics=%s "
        "preserve_seed_edges=%s",
        setup.graph_form,
        setup.objective,
        bool(getattr(args, "reverse_mp", False)),
        bool(getattr(args, "finetune", False)),
        bool(getattr(args, "correct_reverse_edge_features", False)),
        getattr(args, "reverse_edge_feature_semantics", None)
        or (
            "corrected"
            if bool(getattr(args, "correct_reverse_edge_features", False))
            else "inherited_legacy"
        ),
        bool(getattr(args, "preserve_seed_edges", False)),
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

    Per-run CLI overrides (when set) take precedence over model_settings.json:
    ``--override_lr``, ``--override_n_hidden``, ``--override_dropout``, ``--override_final_dropout``.

    Args:
    - parameter_name (str): Name of the parameter (e.g., "lr").
    - args (argparser): Arguments given to this specific run.

    Returns:
    - float: Value of the specified parameter.
    """
    if parameter_name == "lr":
        override = getattr(args, "override_lr", None)
        if override is not None:
            return float(override)
    if parameter_name == "n_hidden":
        override = getattr(args, "override_n_hidden", None)
        if override is not None:
            return float(override)
    if parameter_name == "dropout":
        override = getattr(args, "override_dropout", None)
        if override is not None:
            return float(override)
    if parameter_name == "final_dropout":
        override = getattr(args, "override_final_dropout", None)
        if override is not None:
            return float(override)

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

    Idempotent: skips graphs that already had ids prepended (safe for multi-checkpoint
    extraction on a shared in-memory graph).
    '''
    for data in data_list:
        if bool(getattr(data, "_arange_ids_added", False)):
            continue
        if isinstance(data, HeteroData):
            data['node', 'to', 'node'].edge_attr = torch.cat([torch.arange(data['node', 'to', 'node'].edge_attr.shape[0]).view(-1, 1), data['node', 'to', 'node'].edge_attr], dim=1)
            offset = data['node', 'to', 'node'].edge_attr.shape[0]
            data['node', 'rev_to', 'node'].edge_attr = torch.cat([torch.arange(offset, data['node', 'rev_to', 'node'].edge_attr.shape[0] + offset).view(-1, 1), data['node', 'rev_to', 'node'].edge_attr], dim=1)
        else:
            data.edge_attr = torch.cat([torch.arange(data.edge_attr.shape[0]).view(-1, 1), data.edge_attr], dim=1)
        try:
            data._arange_ids_added = True
        except Exception:
            pass


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


def checkpoint_path(
    data_config,
    unique_name: str,
    *,
    finetuned: bool = False,
    suffix: str = "",
) -> Path:
    """Path to ``checkpoint_{unique_name}[ _finetuned][ suffix].tar`` under ``model_to_load``."""
    finetune_suffix = "_finetuned" if finetuned else ""
    return Path(data_config["paths"]["model_to_load"]) / (
        f"checkpoint_{unique_name}{finetune_suffix}{suffix}.tar"
    )


def checkpoint_selection_score(log_payload: dict, args) -> Optional[float]:
    """
    Composite SSL validation score for best-checkpoint selection. Lower is better.

    When morphology val metrics are logged this epoch, sum all available val losses.
    When no morphology is enabled, use ``loss/train``. Returns ``None`` on morph runs
    when val metrics were skipped this epoch (e.g. ``--morph_val_every > 1``).
    """
    morph_head = getattr(args, "morph_expert_head", None)
    morph_contrast_cfg = getattr(args, "morph_contrast_cfg", None)
    score = 0.0
    n = 0
    if morph_head is not None and "morph/expert_val" in log_payload:
        score += float(log_payload["morph/expert_val"])
        n += 1
    if morph_contrast_cfg is not None and "morph/contrast_val" in log_payload:
        score += float(log_payload["morph/contrast_val"])
        n += 1
    if n > 0:
        return score
    if morph_head is None and morph_contrast_cfg is None:
        return float(log_payload["loss/train"])
    return None


@dataclass
class CheckpointTracker:
    """
    Selects and saves checkpoints for contrastive training (M4).

    ``last``: overwrite ``checkpoint_{unique_name}.tar`` every epoch (legacy).
    ``best``: keep the lowest ``checkpoint_selection_score`` in the main checkpoint;
    after training, write the final epoch to ``checkpoint_{unique_name}_last.tar``.
    """

    policy: str
    best_score: float = float("inf")
    best_epoch: int = -1

    def on_epoch_end(
        self,
        epoch: int,
        log_payload: dict,
        model,
        optimizer,
        args,
        data_config,
    ) -> None:
        if not getattr(args, "save_model", False):
            return

        if self.policy == "last":
            save_model(model, optimizer, epoch, args, data_config)
            return

        score = checkpoint_selection_score(log_payload, args)
        if score is not None:
            log_payload["checkpoint/score"] = score
        if score is not None and score < self.best_score:
            self.best_score = score
            self.best_epoch = epoch + 1
            save_model(model, optimizer, epoch, args, data_config)
            logging.info(
                "New best checkpoint (epoch %s, score=%.4f, policy=best)",
                self.best_epoch,
                score,
            )

    def finalize(
        self,
        last_epoch: int,
        model,
        optimizer,
        args,
        data_config,
    ) -> None:
        if not getattr(args, "save_model", False) or self.policy == "last":
            return

        unique = args.unique_name
        if self.best_epoch < 0:
            logging.warning(
                "No best checkpoint selected; saving final epoch to checkpoint_%s.tar",
                unique,
            )
            save_model(model, optimizer, last_epoch, args, data_config)
            return

        if last_epoch + 1 != self.best_epoch:
            save_model(model, optimizer, last_epoch, args, data_config, suffix="_last")
        logging.info(
            "Best checkpoint: epoch=%s score=%.4f → checkpoint_%s.tar%s",
            self.best_epoch,
            self.best_score,
            unique,
            (
                f" | last epoch={last_epoch + 1} → checkpoint_{unique}_last.tar"
                if last_epoch + 1 != self.best_epoch
                else " (same as final epoch)"
            ),
        )


def save_model(model, optimizer, epoch, args, data_config, *, suffix: str = ""):
    # Save the model in a dictionary
    payload = {
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "embedding_dim": int(getattr(args, "embedding_dim", 128)),
        "include_temporal_flow_edge_features": bool(
            getattr(args, "include_temporal_flow_edge_features", False)
        ),
        "correct_reverse_edge_features": bool(
            getattr(args, "correct_reverse_edge_features", False)
        ),
        "reverse_edge_feature_semantics": (
            getattr(args, "reverse_edge_feature_semantics", None)
            or (
                "corrected"
                if bool(getattr(args, "correct_reverse_edge_features", False))
                else "inherited_legacy"
            )
        ),
        "preserve_seed_edges": bool(getattr(args, "preserve_seed_edges", False)),
        "ports": bool(getattr(args, "ports", False)),
        "tds": bool(getattr(args, "tds", False)),
    }
    schema = getattr(args, "edge_feature_schema", None)
    if schema is not None:
        payload["edge_feature_schema"] = schema
    # Record edge_dim when available (GINe/GATe/PNA/RGCN expose edge_emb).
    edge_emb = getattr(model, "edge_emb", None)
    if edge_emb is not None and hasattr(edge_emb, "in_features"):
        payload["edge_dim"] = int(edge_emb.in_features)
    tf_meta = getattr(args, "temporal_flow_edge_features_meta", None)
    if tf_meta is not None:
        payload["temporal_flow_edge_features_meta"] = tf_meta
    morph_head = getattr(args, "morph_expert_head", None)
    if morph_head is not None:
        payload["morph_expert_state_dict"] = morph_head.state_dict()
    proj_head = getattr(args, "contrast_projection_module", None)
    if proj_head is not None:
        payload["contrast_projection_state_dict"] = proj_head.state_dict()
    tf_head = getattr(args, "temporal_flow_aux_head", None)
    if tf_head is not None:
        payload["temporal_flow_aux_state_dict"] = tf_head.state_dict()
        tf_cfg = getattr(args, "temporal_flow_aux_cfg", None)
        if tf_cfg is not None:
            payload["temporal_flow_aux_meta"] = {
                "mode": tf_cfg.mode,
                "weight": tf_cfg.weight,
                "loss_type": tf_cfg.loss_type,
                "n_bins": tf_cfg.n_bins,
                "attach_point": tf_cfg.attach_point,
                "feature_names": list(tf_cfg.feature_names),
                "metadata_path": tf_cfg.metadata_path,
                "uses_labels": False,
            }
    masked_decoder = getattr(args, "masked_edge_decoder", None)
    if masked_decoder is not None:
        payload["masked_edge_decoder_state_dict"] = masked_decoder.state_dict()
    finetune_suffix = "_finetuned" if getattr(args, "finetune", False) else ""
    path = (
        Path(data_config["paths"]["model_to_save"])
        / f"checkpoint_{args.unique_name}{finetune_suffix}{suffix}.tar"
    )
    torch.save(payload, path)


def load_checkpoint_weights(model, device, args, data_config) -> int:
    """Load ``model_state_dict`` from a pretrain/adapt checkpoint (no optimizer)."""
    finetuned = bool(getattr(args, "finetune", False))
    suffix = str(getattr(args, "checkpoint_suffix", "") or "")
    path = checkpoint_path(data_config, args.unique_name, finetuned=finetuned, suffix=suffix)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device)
    # Fail clearly on an embedding-dim mismatch (e.g. loading an emb198 checkpoint with the default
    # 128, or vice versa) before the lower-level state_dict shape error.
    ckpt_emb_dim = checkpoint.get("embedding_dim")
    req_emb_dim = int(getattr(args, "embedding_dim", 128))
    if ckpt_emb_dim is not None and int(ckpt_emb_dim) != req_emb_dim:
        raise ValueError(
            f"embedding_dim mismatch: checkpoint {path.name} was trained with embedding_dim="
            f"{int(ckpt_emb_dim)} but the current run uses --embedding_dim {req_emb_dim}. "
            f"Pass --embedding_dim {int(ckpt_emb_dim)} to match this checkpoint."
        )
    from morphology.temporal_flow_edge_features import assert_checkpoint_tf_edge_features_flag

    assert_checkpoint_tf_edge_features_flag(checkpoint, args, path=path)
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
    """Atomic write: ``path.tmp.npz`` then ``os.replace`` into ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(
        tmp,
        Z=embeddings.detach().cpu().numpy().astype(np.float32),
        y=labels.detach().cpu().numpy(),
        edge_id=edge_ids.detach().cpu().numpy(),
    )
    # np.savez may append .npz when the destination does not already end in .npz.
    written = tmp if tmp.is_file() else Path(str(tmp) + ".npz")
    if written != path:
        os.replace(written, path)
    if tmp.is_file() and tmp != path:
        tmp.unlink(missing_ok=True)


# Representation-source lever for extraction diagnostics.
#   post_embedding  -> output of embedding_head (current 128-d z; default, unchanged behavior)
#   pre_embedding_3h -> tensor fed INTO embedding_head (cat(src_node, dst_node, edge_attr) = 3*n_hidden)
REPRESENTATION_SOURCES = ("post_embedding", "pre_embedding_3h")


@dataclass(frozen=True)
class EmbeddingHeadSpec:
    """Resolved embedding-head Linear used for pre_embedding_3h capture."""

    module: torch.nn.Linear
    module_name: str
    in_features: int
    out_features: int


def _linear_module_path(root: torch.nn.Module, target: torch.nn.Module) -> str:
    for name, module in root.named_modules():
        if module is target:
            return name or "<root>"
    return repr(target)


def resolve_embedding_head_linear(
    model: torch.nn.Module,
    emb_dim: int,
    pre_dim: Optional[int] = None,
) -> EmbeddingHeadSpec:
    """Locate the embedding-head Linear that maps pre-edge representation -> exported embedding.

    Prefers the named ``embedding_head`` when it is a matching ``nn.Linear``. Otherwise searches
    for ``nn.Linear`` layers with ``out_features == emb_dim`` and ``in_features % 3 == 0``,
    excluding obvious projection/classifier heads (``in_features == emb_dim``).

    Raises ``ValueError`` when no candidate is found or when multiple candidates remain ambiguous.
    """
    emb_dim = int(emb_dim)
    named_head = getattr(model, "embedding_head", None)
    if isinstance(named_head, torch.nn.Linear):
        if named_head.out_features != emb_dim:
            raise ValueError(
                "model.embedding_head out_features={0} != embedding_dim={1}".format(
                    named_head.out_features, emb_dim
                )
            )
        if pre_dim is not None and named_head.in_features != int(pre_dim):
            raise ValueError(
                "model.embedding_head in_features={0} != expected pre_dim={1}".format(
                    named_head.in_features, pre_dim
                )
            )
        if named_head.in_features % 3 != 0:
            raise ValueError(
                "model.embedding_head in_features={0} is not divisible by 3".format(
                    named_head.in_features
                )
            )
        return EmbeddingHeadSpec(
            module=named_head,
            module_name="embedding_head",
            in_features=int(named_head.in_features),
            out_features=int(named_head.out_features),
        )

    candidates: List[Tuple[str, torch.nn.Linear]] = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if module.out_features != emb_dim:
            continue
        if module.in_features == emb_dim:
            # Projection/classifier-style head (e.g. Linear(128, 128)); not pre-edge representation.
            continue
        if module.in_features % 3 != 0:
            continue
        if pre_dim is not None and module.in_features != int(pre_dim):
            continue
        candidates.append((name or "<unnamed>", module))

    if not candidates:
        raise ValueError(
            "pre_embedding_3h requested but no embedding-head Linear "
            f"(out_features={emb_dim}, in_features%3==0"
            + (f", in_features={int(pre_dim)}" if pre_dim is not None else "")
            + ") was found. This checkpoint is incompatible with pre_embedding_3h extraction "
            "(e.g. a legacy/supervised head, or a model with a different embedding_dim)."
        )
    if len(candidates) > 1:
        desc = ", ".join(
            "{0}(in={1}, out={2})".format(name, lin.in_features, lin.out_features)
            for name, lin in candidates
        )
        raise ValueError(
            "pre_embedding_3h requested but multiple embedding-head candidates were found: "
            f"{desc}. Resolve ambiguity explicitly (prefer model.embedding_head)."
        )

    name, module = candidates[0]
    return EmbeddingHeadSpec(
        module=module,
        module_name=name,
        in_features=int(module.in_features),
        out_features=int(module.out_features),
    )


def infer_pre_embedding_dim(model: torch.nn.Module, emb_dim: int) -> int:
    """Return ``3 * n_hidden`` from the resolved embedding head (handles PNA width rounding)."""
    return resolve_embedding_head_linear(model, emb_dim).in_features


class PreEmbeddingCapture:
    """Capture the tensor fed into ``embedding_head`` via a forward hook (no forward changes).

    Uses :func:`resolve_embedding_head_linear` to locate the head on homogeneous or
    ``to_hetero``-wrapped models. The captured input is keyed by the ``id`` of the produced
    output tensor so the caller looks up the exact head call that produced the forward-edge
    embedding.

    Raises ``ValueError`` if no matching head is found (e.g. a legacy/supervised checkpoint or a
    different embedding_dim), satisfying the "fail clearly on incompatible checkpoint" requirement.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        pre_dim: Optional[int] = None,
        emb_dim: int = 128,
        head_spec: Optional[EmbeddingHeadSpec] = None,
    ):
        self.emb_dim = int(emb_dim)
        self.head_spec = head_spec or resolve_embedding_head_linear(model, self.emb_dim, pre_dim)
        self.pre_dim = int(self.head_spec.in_features)
        if pre_dim is not None and self.pre_dim != int(pre_dim):
            raise ValueError(
                f"Resolved embedding-head in_features={self.pre_dim} != requested pre_dim={pre_dim}"
            )
        logging.info(
            "pre_embedding_3h capture: module=%s in_features=%d out_features=%d",
            self.head_spec.module_name,
            self.head_spec.in_features,
            self.head_spec.out_features,
        )
        self.captured: Dict[int, torch.Tensor] = {}
        self.handles: List[Any] = []
        matched = 0
        for module in model.modules():
            if (
                isinstance(module, torch.nn.Linear)
                and module.in_features == self.pre_dim
                and module.out_features == self.emb_dim
            ):
                self.handles.append(module.register_forward_hook(self._hook))
                matched += 1
        if matched == 0:
            raise ValueError(
                "pre_embedding_3h requested but no embedding-head Linear "
                f"(in_features={self.pre_dim}, out_features={self.emb_dim}) was found in the "
                f"execution model. Resolved head {self.head_spec.module_name!r} on the "
                "homogeneous template may not have been replicated after to_hetero."
            )
        self.matched = matched

    def _hook(self, module, inputs, output):
        self.captured[id(output)] = inputs[0].detach()

    def clear(self) -> None:
        self.captured.clear()

    def get(self, output: torch.Tensor) -> torch.Tensor:
        pre = self.captured.get(id(output))
        if pre is None:
            raise RuntimeError(
                "pre_embedding_3h capture failed: the forward output was not matched to an "
                "embedding-head input. The model forward may not route through embedding_head."
            )
        if pre.shape[-1] != self.pre_dim:
            raise RuntimeError(
                "pre_embedding_3h capture dimension mismatch: captured last dim "
                f"{pre.shape[-1]} != embedding-head in_features={self.pre_dim} "
                f"(module={self.head_spec.module_name})"
            )
        return pre

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _validate_representation_source(representation_source: str) -> str:
    src = str(representation_source or "post_embedding")
    if src not in REPRESENTATION_SOURCES:
        raise ValueError(
            f"Unsupported representation_source {src!r}; use one of {REPRESENTATION_SOURCES}."
        )
    return src


@torch.no_grad()
def extract_seed_embeddings_homo(
    loader,
    split_inds,
    model,
    data,
    device,
    args,
    representation_source: str = "post_embedding",
    pre_dim: Optional[int] = None,
    emb_dim: Optional[int] = None,
    head_spec: Optional[EmbeddingHeadSpec] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collect frozen edge representations and labels for seed edges (homogeneous graphs).

    ``representation_source='post_embedding'`` (default) returns the embedding-head output
    (unchanged behavior). ``'pre_embedding_3h'`` returns the tensor fed into embedding_head
    (``3*n_hidden``) captured via a forward hook; ``pre_dim``/``emb_dim`` locate the head.
    """
    representation_source = _validate_representation_source(representation_source)
    capture = (
        PreEmbeddingCapture(model, pre_dim=pre_dim, emb_dim=emb_dim or 128, head_spec=head_spec)
        if representation_source == "pre_embedding_3h"
        else None
    )
    try:
        return _extract_seed_embeddings_homo_impl(
            loader, split_inds, model, data, device, args, capture
        )
    finally:
        if capture is not None:
            capture.remove()


@torch.no_grad()
def _extract_seed_embeddings_homo_impl(
    loader,
    split_inds,
    model,
    data,
    device,
    args,
    capture: Optional[PreEmbeddingCapture],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        if capture is not None:
            capture.clear()
        z = model(batch.x, batch.edge_index, batch.edge_attr)
        if capture is not None:
            z = capture.get(z)
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
    representation_source: str = "post_embedding",
    pre_dim: Optional[int] = None,
    emb_dim: Optional[int] = None,
    head_spec: Optional[EmbeddingHeadSpec] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collect frozen forward-edge representations for seed transactions (hetero graphs).

    See :func:`extract_seed_embeddings_homo` for the ``representation_source`` semantics.
    """
    representation_source = _validate_representation_source(representation_source)
    capture = (
        PreEmbeddingCapture(model, pre_dim=pre_dim, emb_dim=emb_dim or 128, head_spec=head_spec)
        if representation_source == "pre_embedding_3h"
        else None
    )
    try:
        edge_ids, z_post, y, z_pre = _extract_seed_embeddings_hetero_impl(
            loader, split_inds, model, data, device, args, capture, dual=False
        )
        if representation_source == "pre_embedding_3h":
            return edge_ids, z_pre, y
        return edge_ids, z_post, y
    finally:
        if capture is not None:
            capture.remove()


@torch.no_grad()
def extract_seed_embeddings_hetero_dual(
    loader,
    split_inds,
    model,
    data,
    device,
    args,
    pre_dim: Optional[int] = None,
    emb_dim: Optional[int] = None,
    head_spec: Optional[EmbeddingHeadSpec] = None,
    max_batches: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One forward: return ``(edge_ids, z_pre3h, z_post128, y)`` with identical row order."""
    capture = PreEmbeddingCapture(
        model, pre_dim=pre_dim, emb_dim=emb_dim or 128, head_spec=head_spec
    )
    try:
        return _extract_seed_embeddings_hetero_impl(
            loader,
            split_inds,
            model,
            data,
            device,
            args,
            capture,
            dual=True,
            max_batches=max_batches,
        )
    finally:
        capture.remove()


@torch.no_grad()
def _extract_seed_embeddings_hetero_impl(
    loader,
    split_inds,
    model,
    data,
    device,
    args,
    capture: Optional[PreEmbeddingCapture],
    dual: bool = False,
    max_batches: Optional[int] = None,
):
    edge_id_chunks: List[torch.Tensor] = []
    z_post_chunks: List[torch.Tensor] = []
    z_pre_chunks: List[torch.Tensor] = []
    y_chunks: List[torch.Tensor] = []
    store = FORWARD_EDGE_TYPE

    split_inds_cpu = split_inds.detach().cpu()
    for batch_i, batch in enumerate(tqdm.tqdm(loader, disable=not args.tqdm, desc="extract hetero")):
        if max_batches is not None and batch_i >= int(max_batches):
            break
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
        if capture is not None:
            capture.clear()
        out = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
        z_post = out[store]
        z_pre = capture.get(z_post) if capture is not None else None
        mask_dev = mask.to(device, non_blocking=True)
        edge_id_chunks.append(edge_ids[mask].detach().cpu())
        z_post_chunks.append(z_post[mask_dev].detach().cpu())
        if z_pre is not None:
            z_pre_chunks.append(z_pre[mask_dev].detach().cpu())
        y_chunks.append(fwd.y[mask_dev].detach().cpu().long())

    edge_ids = torch.cat(edge_id_chunks, dim=0)
    z_post = torch.cat(z_post_chunks, dim=0)
    y = torch.cat(y_chunks, dim=0)
    if capture is not None:
        z_pre = torch.cat(z_pre_chunks, dim=0)
        # Dedupe on shared order: apply same keep mask from edge_ids
        edge_ids_d, z_post_d, y_d, n_dup = dedupe_seed_embeddings(edge_ids, z_post, y)
        # Rebuild keep indices via first-occurrence of edge_ids
        seen = {}
        keep = []
        for i, eid in enumerate(edge_ids.tolist()):
            if eid in seen:
                continue
            seen[eid] = i
            keep.append(i)
        idx = torch.tensor(keep, dtype=torch.long)
        z_pre_d = z_pre[idx]
        if n_dup > 0:
            logging.info("extract hetero: dropped %d duplicate seed rows", n_dup)
        if dual:
            return edge_ids_d, z_pre_d, z_post_d, y_d
        return edge_ids_d, z_post_d, y_d, z_pre_d
    edge_ids, z_post, y, n_dup = dedupe_seed_embeddings(edge_ids, z_post, y)
    if n_dup > 0:
        logging.info("extract hetero: dropped %d duplicate seed rows", n_dup)
    if dual:
        raise RuntimeError("dual extract requires pre_embedding capture")
    return edge_ids, z_post, y, None


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


# ---------------------------------------------------------------------------
# Paper-compatible supervised evaluation, per-epoch history, dual checkpointing
# and run summaries (legacy IBM Multi-GNN / Egressy et al. reproduction protocol).
#
# Primary decision rule for the reproduction metric: argmax over two-class logits.
# Checkpoint-selection metric: validation minority-class (label 1) F1 only.
# Per-epoch metrics are computed in the upstream train-mode regime (model.eval() is
# NOT toggled), matching the fork-point (commit fc751e8) evaluation. Deterministic
# richer evaluation is produced post-hoc from the best-val checkpoint by
# scripts/evaluate_supervised_gnn.py (which does call model.eval()).
# ---------------------------------------------------------------------------

DECISION_RULE_ARGMAX = "argmax over two-class logits"
SELECTION_METRIC_VAL_F1 = "validation_minority_f1"
CHECKPOINT_SELECTION_RULE = (
    "best validation minority-class F1 (strict improvement; earliest epoch kept on tie)"
)
SUPERVISED_ARG_KEYS = (
    "model", "data", "supervised_head", "objective", "reverse_mp", "emlps",
    "ports", "tds", "ego", "seed", "n_epochs", "batch_size", "num_neighs",
    "override_lr", "override_n_hidden", "override_final_dropout", "finetune",
    "unique_name", "correct_reverse_edge_features", "preserve_seed_edges",
)


def supervised_run_name(args) -> str:
    return str(getattr(args, "unique_name", None) or f"{getattr(args, 'model', 'model')}_supervised")


def supervised_diagnostics_dir() -> Path:
    return Path("results") / "diagnostics"


def supervised_epoch_history_path(args) -> Path:
    return supervised_diagnostics_dir() / (
        f"supervised_{getattr(args, 'data', 'data')}_{supervised_run_name(args)}_epoch_history.json"
    )


def supervised_summary_json_path(args) -> Path:
    return supervised_diagnostics_dir() / (
        f"supervised_{getattr(args, 'data', 'data')}_{supervised_run_name(args)}_summary.json"
    )


def supervised_summary_md_path(args) -> Path:
    return Path("notes") / (
        f"supervised_{getattr(args, 'data', 'data')}_{supervised_run_name(args)}_summary.md"
    )


def supervised_run_dir(data_config, run_name: str) -> Path:
    return Path(data_config["paths"]["model_to_save"]) / str(run_name)


def _atomic_json_dump(payload: Dict[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(tmp, path)


def _atomic_torch_save(payload: Dict[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _supervised_split_metrics(
    y_true: np.ndarray, y_pred_argmax: np.ndarray, y_proba: np.ndarray
) -> Dict[str, float]:
    metrics = {
        "f1_argmax": float(f1_score(y_true, y_pred_argmax, zero_division=0)),
        "precision_argmax": float(precision_score(y_true, y_pred_argmax, zero_division=0)),
        "recall_argmax": float(recall_score(y_true, y_pred_argmax, zero_division=0)),
    }
    if np.unique(y_true).size < 2:
        metrics["auroc"] = float("nan")
        metrics["auprc"] = float("nan")
    else:
        metrics["auroc"] = float(roc_auc_score(y_true, y_proba))
        metrics["auprc"] = float(average_precision_score(y_true, y_proba))
    return metrics


@torch.no_grad()
def _collect_supervised_predictions_homo(loader, inds, model, data, device, args):
    preds: List[torch.Tensor] = []
    grounds: List[torch.Tensor] = []
    probas: List[torch.Tensor] = []
    for batch in tqdm.tqdm(loader, disable=not args.tqdm):
        inds_cpu = inds.detach().cpu()
        batch_edge_inds = inds_cpu[batch.input_id.detach().cpu()]
        batch_edge_ids = loader.data.edge_attr.detach().cpu()[batch_edge_inds, 0]
        mask = torch.isin(batch.edge_attr[:, 0].detach().cpu(), batch_edge_ids)

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

        batch.edge_attr = batch.edge_attr[:, 1:]
        batch.to(device)
        z = model(batch.x, batch.edge_index, batch.edge_attr)
        logits = model.classifier(z)[mask]
        probas.append(torch.softmax(logits, dim=-1)[:, 1].cpu())
        preds.append(logits.argmax(dim=-1).cpu())
        grounds.append(batch.y[mask].cpu())
    y = torch.cat(grounds).numpy()
    pred = torch.cat(preds).numpy()
    proba = torch.cat(probas).numpy()
    return y, pred, proba


@torch.no_grad()
def _collect_supervised_predictions_hetero(loader, inds, model, data, device, args):
    preds: List[torch.Tensor] = []
    grounds: List[torch.Tensor] = []
    probas: List[torch.Tensor] = []
    for batch in tqdm.tqdm(loader, disable=not args.tqdm):
        inds_cpu = inds.detach().cpu()
        batch_edge_inds = inds_cpu[batch['node', 'to', 'node'].input_id.detach().cpu()]
        batch_edge_ids = loader.data['node', 'to', 'node'].edge_attr.detach().cpu()[batch_edge_inds, 0]
        mask = torch.isin(batch['node', 'to', 'node'].edge_attr[:, 0].detach().cpu(), batch_edge_ids)

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

        batch['node', 'to', 'node'].edge_attr = batch['node', 'to', 'node'].edge_attr[:, 1:]
        batch['node', 'rev_to', 'node'].edge_attr = batch['node', 'rev_to', 'node'].edge_attr[:, 1:]
        batch.to(device)
        z = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)[('node', 'to', 'node')]
        logits = edge_classifier_logits(model, z)[mask]
        probas.append(torch.softmax(logits, dim=-1)[:, 1].cpu())
        preds.append(logits.argmax(dim=-1).cpu())
        grounds.append(batch['node', 'to', 'node'].y[mask].cpu())
    y = torch.cat(grounds).numpy()
    pred = torch.cat(preds).numpy()
    proba = torch.cat(probas).numpy()
    return y, pred, proba


def evaluate_supervised_split(loader, inds, model, data, device, args) -> Dict[str, float]:
    """One-pass supervised metrics for a split (argmax F1/precision/recall + AUROC/AUPRC).

    The paper-compatible fields (``*_argmax``) use argmax over two-class logits, matching the
    upstream evaluation. ``model.eval()`` is deliberately NOT toggled here, reproducing the
    fork-point train-mode evaluation regime; the caller controls the model's mode.
    """
    if isinstance(data, HeteroData):
        y, pred, proba = _collect_supervised_predictions_hetero(loader, inds, model, data, device, args)
    else:
        y, pred, proba = _collect_supervised_predictions_homo(loader, inds, model, data, device, args)
    return _supervised_split_metrics(y, pred, proba)


def supervised_run_metadata(args, config) -> Dict[str, Any]:
    return {
        "run_name": supervised_run_name(args),
        "dataset": getattr(args, "data", None),
        "seed": int(getattr(args, "seed", -1)),
        "model_architecture": getattr(args, "model", None),
        "supervised_head": getattr(args, "supervised_head", "embedding"),
        "graph_flags": {
            "emlps": bool(getattr(args, "emlps", False)),
            "reverse_mp": bool(getattr(args, "reverse_mp", False)),
            "ports": bool(getattr(args, "ports", False)),
            "tds": bool(getattr(args, "tds", False)),
            "ego": bool(getattr(args, "ego", False)),
            "correct_reverse_edge_features": bool(
                getattr(args, "correct_reverse_edge_features", False)
            ),
            "preserve_seed_edges": bool(getattr(args, "preserve_seed_edges", False)),
            "reverse_edge_feature_semantics": (
                getattr(args, "reverse_edge_feature_semantics", None)
                or (
                    "corrected"
                    if bool(getattr(args, "correct_reverse_edge_features", False))
                    else "inherited_legacy"
                )
            ),
        },
        "class_weights": {
            "0": float(getattr(config, "w_ce1", float("nan"))),
            "1": float(getattr(config, "w_ce2", float("nan"))),
        },
        "optimizer": "adam",
        "n_epochs": int(getattr(config, "epochs", getattr(args, "n_epochs", -1))),
        "checkpoint_selection_rule": CHECKPOINT_SELECTION_RULE,
        "selection_metric": SELECTION_METRIC_VAL_F1,
        "decision_rule": DECISION_RULE_ARGMAX,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }


class SupervisedHistoryRecorder:
    """Incrementally writes per-epoch supervised history with an atomic replace each epoch."""

    def __init__(self, path: Path, metadata: Dict[str, Any]):
        self.path = Path(path)
        self.metadata = metadata
        self.epochs: List[Dict[str, Any]] = []

    def record_epoch(self, row: Dict[str, Any]) -> None:
        self.epochs.append(row)
        payload = dict(self.metadata)
        payload["epochs"] = self.epochs
        try:
            _atomic_json_dump(payload, self.path)
        except OSError as exc:  # keep long training runs alive if disk write fails
            logging.warning("Failed to write supervised epoch history %s: %s", self.path, exc)


class SupervisedCheckpointer:
    """Dual supervised checkpointing on the Egressy-style best-validation protocol.

    - ``checkpoint_last.tar``        : written every epoch (resume state).
    - ``checkpoint_best_val_f1.tar`` : written only when validation minority-class F1 strictly
                                       improves (ties keep the earliest epoch).
    - flat ``checkpoint_{unique_name}.tar`` : retained (= last epoch) for backward-compatible
                                       tooling. Reproduction evaluation MUST use the best-val file.

    Best-validation selection state is tracked regardless of ``--save_model``; files are only
    written when ``args.save_model`` is set.
    """

    def __init__(self, args, config, data_config):
        self.args = args
        self.config = config
        self.data_config = data_config
        self.save = bool(getattr(args, "save_model", False))
        self.run_name = supervised_run_name(args)
        self.run_dir = supervised_run_dir(data_config, self.run_name)
        self.flat_path = Path(data_config["paths"]["model_to_save"]) / f"checkpoint_{self.run_name}.tar"
        self.best_val_f1: Optional[float] = None
        self.selected_epoch: Optional[int] = None
        self.test_f1_at_selected: Optional[float] = None
        if self.save:
            self.run_dir.mkdir(parents=True, exist_ok=True)

    def _base_payload(self, epoch: int, model, optimizer, scheduler) -> Dict[str, Any]:
        return {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "seed": int(getattr(self.args, "seed", -1)),
            "supervised_head": getattr(self.args, "supervised_head", "embedding"),
            "args": {k: getattr(self.args, k, None) for k in SUPERVISED_ARG_KEYS},
            "config": {
                "lr": float(getattr(self.config, "lr", float("nan"))),
                "n_hidden": float(getattr(self.config, "n_hidden", float("nan"))),
                "n_gnn_layers": int(round(float(getattr(self.config, "n_gnn_layers", 0) or 0))),
                "dropout": float(getattr(self.config, "dropout", float("nan"))),
                "final_dropout": float(getattr(self.config, "final_dropout", float("nan"))),
                "w_ce1": float(getattr(self.config, "w_ce1", float("nan"))),
                "w_ce2": float(getattr(self.config, "w_ce2", float("nan"))),
                "epochs": int(getattr(self.config, "epochs", -1)),
            },
            "best_validation_f1": self.best_val_f1,
            "selected_epoch": self.selected_epoch,
            "test_f1_at_selected_epoch": self.test_f1_at_selected,
        }

    def update(self, epoch: int, model, optimizer, scheduler, val_f1: float, test_f1: float) -> bool:
        """Update selection state and (when saving) write last + best checkpoints.

        Selection depends ONLY on ``val_f1`` (validation minority-class F1); ``test_f1`` is
        recorded alongside the selected checkpoint but never influences selection.
        """
        improved = self.best_val_f1 is None or val_f1 > self.best_val_f1
        if improved:
            self.best_val_f1 = float(val_f1)
            self.selected_epoch = epoch + 1
            self.test_f1_at_selected = float(test_f1)

        if self.save:
            payload = self._base_payload(epoch, model, optimizer, scheduler)
            _atomic_torch_save(payload, self.run_dir / "checkpoint_last.tar")
            # Flat compatibility checkpoint tracks the LAST epoch for legacy tooling.
            _atomic_torch_save(payload, self.flat_path)
            if improved:
                best_payload = dict(payload)
                best_payload.update({
                    "selected_epoch": self.selected_epoch,
                    "best_validation_f1": self.best_val_f1,
                    "test_f1_at_selected_epoch": self.test_f1_at_selected,
                    "selection_metric": SELECTION_METRIC_VAL_F1,
                    "decision_rule": "argmax",
                })
                _atomic_torch_save(best_payload, self.run_dir / "checkpoint_best_val_f1.tar")
        return improved


def prepare_supervised_resume(
    args,
    model,
    optimizer,
    recorder: SupervisedHistoryRecorder,
    checkpointer: SupervisedCheckpointer,
    device,
) -> int:
    """Load ``checkpoint_last.tar`` when ``--resume_supervised`` is set.

    Restores model/optimizer state, best-validation selection, and prior epoch history.
    Returns the 0-based epoch index to continue from (``checkpoint['epoch']`` = completed count).
    """
    if not getattr(args, "resume_supervised", False):
        return 0
    last_path = checkpointer.run_dir / "checkpoint_last.tar"
    if not last_path.is_file():
        logging.warning(
            "--resume_supervised set but %s is missing; starting from epoch 0.", last_path
        )
        return 0
    checkpoint = torch.load(last_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    checkpointer.best_val_f1 = checkpoint.get("best_validation_f1")
    checkpointer.selected_epoch = checkpoint.get("selected_epoch")
    checkpointer.test_f1_at_selected = checkpoint.get("test_f1_at_selected_epoch")
    hist_path = supervised_epoch_history_path(args)
    if hist_path.is_file():
        with hist_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        recorder.epochs = list(payload.get("epochs", []))
    start_epoch = int(checkpoint.get("epoch", 0))
    logging.info(
        "Resuming supervised training from epoch %d (loaded %s; best val F1 %.4f @ ep %s).",
        start_epoch,
        last_path,
        checkpointer.best_val_f1 if checkpointer.best_val_f1 is not None else float("nan"),
        checkpointer.selected_epoch,
    )
    return start_epoch


def _is_configured_to_reproduce_egressy(args) -> bool:
    """Conservative gate for reproduction wording (constraint 8).

    Only GINe with the legacy head is a numerically validated legacy restoration. A ``--testing``
    (dev/scout) run is never labeled reproduction-configured. Full reproduction still requires
    manual confirmation of data split, hyperparameters, class weights, optimizer, epoch count,
    selection rule, and decision rule against the upstream row.
    """
    return (
        getattr(args, "supervised_head", "embedding") == "legacy"
        and getattr(args, "model", None) == "gin"
        and not bool(getattr(args, "testing", False))
    )


def write_supervised_summary(args, config, recorder: SupervisedHistoryRecorder,
                             checkpointer: SupervisedCheckpointer) -> Optional[Path]:
    """Write the Phase 6 run summary JSON + Markdown note. Never raises during training."""
    try:
        epochs = recorder.epochs
        if not epochs:
            return None
        final_row = epochs[-1]
        final_test_f1 = float(final_row.get("test_minority_f1_argmax", float("nan")))
        selected_epoch = checkpointer.selected_epoch
        selected_row = None
        if selected_epoch is not None:
            for row in epochs:
                if row.get("epoch") == selected_epoch:
                    selected_row = row
                    break
        best_val_f1 = checkpointer.best_val_f1
        test_f1_at_best = checkpointer.test_f1_at_selected
        best_final_delta = (
            abs(final_test_f1 - test_f1_at_best)
            if test_f1_at_best is not None and not np.isnan(final_test_f1)
            else float("nan")
        )
        differ_substantially = bool(best_final_delta > 0.02) if not np.isnan(best_final_delta) else False
        head = getattr(args, "supervised_head", "embedding")
        reproduction = _is_configured_to_reproduce_egressy(args)
        is_scout = bool(getattr(args, "testing", False))
        run_kind = "scout/dev (--testing)" if is_scout else "standard"
        if head == "legacy" and getattr(args, "model", None) == "gin":
            if reproduction:
                caveat = (
                    "Legacy GINe head is a numerically validated restoration; 'configured to "
                    "reproduce the corresponding Egressy et al. setup' still requires manual "
                    "confirmation of data split, hyperparameters, class weights, optimizer, epoch "
                    "count, selection rule, and decision rule."
                )
            else:
                caveat = (
                    "Legacy GINe head (numerically validated), but this is a scout/dev run "
                    "(--testing) and/or not the full upstream setup: NOT paper-comparable. Run a "
                    "non-testing, upstream epoch-count job before comparing to the Egressy et al. table."
                )
        elif head == "legacy":
            caveat = (
                f"Legacy head on a non-GINe architecture ({getattr(args, 'model', None)}) is "
                "restored-but-unvalidated; not paper-comparable."
            )
        else:
            caveat = (
                "This is the current embedding-head supervised control; it is NOT the "
                "Egressy/Multi-GNN baseline."
            )
        metadata = recorder.metadata

        summary = {
            **metadata,
            "supervised_mode": head,
            "run_kind": run_kind,
            "paper_comparable": reproduction,
            "best_validation_epoch": selected_epoch,
            "validation_minority_f1_argmax_at_best": best_val_f1,
            "test_minority_f1_argmax_at_best": test_f1_at_best,
            "final_epoch_test_minority_f1_argmax": final_test_f1,
            "best_vs_final_test_f1_abs_delta": best_final_delta,
            "best_and_final_differ_substantially": differ_substantially,
            "primary_reproduction_metric": "test_minority_f1_argmax_at_best (decision rule: argmax over two-class logits)",
            "richer_ranking_metrics_at_best": {
                "validation_auroc": selected_row.get("validation_auroc") if selected_row else None,
                "validation_auprc": selected_row.get("validation_auprc") if selected_row else None,
                "test_auroc": selected_row.get("test_auroc") if selected_row else None,
                "test_auprc": selected_row.get("test_auprc") if selected_row else None,
            },
            "configured_to_reproduce_egressy_setup": reproduction,
            "reproduction_caveat": caveat,
            "epoch_history_path": str(recorder.path),
            "best_val_checkpoint_path": str(checkpointer.run_dir / "checkpoint_best_val_f1.tar"),
            "last_checkpoint_path": str(checkpointer.run_dir / "checkpoint_last.tar"),
            "flat_compat_checkpoint_path": str(checkpointer.flat_path),
        }
        json_path = supervised_summary_json_path(args)
        _atomic_json_dump(summary, json_path)
        _write_supervised_summary_md(supervised_summary_md_path(args), summary)
        logging.info("Wrote supervised run summary to %s", json_path)
        return json_path
    except Exception as exc:  # noqa: BLE001 - summary must never break training
        logging.warning("Failed to write supervised run summary: %s", exc)
        return None


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return "n/a" if np.isnan(value) else f"{value:.4f}"
    return str(value)


def _write_supervised_summary_md(path: Path, summary: Dict[str, Any]) -> None:
    head = summary.get("supervised_head", "embedding")
    run_kind = summary.get("run_kind", "standard")
    if head == "legacy":
        scout_tag = " scout/integration run (NOT the full reproduction)" if "scout" in run_kind else ""
        mode_line = f"legacy supervised reproduction head (IBM Multi-GNN / Egressy et al.){scout_tag}"
    else:
        mode_line = "current embedding-head supervised control (NOT the Egressy/Multi-GNN baseline)"
    flags = summary.get("graph_flags", {})
    flag_str = ", ".join(f"{k}={v}" for k, v in flags.items())
    ranking = summary.get("richer_ranking_metrics_at_best", {})
    lines = [
        f"# Supervised run summary: {summary.get('run_name')}",
        "",
        f"- **Supervised mode:** {mode_line}",
        f"- **Run kind:** {run_kind}  |  **Paper-comparable:** {summary.get('paper_comparable')}",
        f"- **Model architecture:** {summary.get('model_architecture')} (supervised_head={head})",
        f"- **Dataset:** {summary.get('dataset')}  |  **Seed:** {summary.get('seed')}",
        f"- **Graph flags:** {flag_str}",
        f"- **Class weights (0,1):** ({_fmt(summary.get('class_weights', {}).get('0'))}, "
        f"{_fmt(summary.get('class_weights', {}).get('1'))})",
        f"- **Optimizer / epochs:** {summary.get('optimizer')} / {summary.get('n_epochs')}",
        f"- **Selection metric:** {summary.get('selection_metric')}  |  "
        f"**Decision rule:** {summary.get('decision_rule')}",
        "",
        "## Best-validation selection",
        "",
        f"- **Best validation epoch:** {_fmt(summary.get('best_validation_epoch'))}",
        f"- **Validation minority F1 (argmax) at best:** {_fmt(summary.get('validation_minority_f1_argmax_at_best'))}",
        f"- **Test minority F1 (argmax) at best epoch:** {_fmt(summary.get('test_minority_f1_argmax_at_best'))}  "
        "(primary reproduction metric)",
        f"- **Final-epoch test minority F1 (argmax):** {_fmt(summary.get('final_epoch_test_minority_f1_argmax'))}",
        f"- **Best vs final test F1 |delta|:** {_fmt(summary.get('best_vs_final_test_f1_abs_delta'))}  "
        f"(differ substantially: {summary.get('best_and_final_differ_substantially')})",
        "",
        "## Richer ranking metrics at best epoch (train-mode, diagnostic)",
        "",
        f"- validation AUROC: {_fmt(ranking.get('validation_auroc'))}  |  "
        f"validation AUPRC: {_fmt(ranking.get('validation_auprc'))}",
        f"- test AUROC: {_fmt(ranking.get('test_auroc'))}  |  "
        f"test AUPRC: {_fmt(ranking.get('test_auprc'))}",
        "",
        "## Reproduction comparability",
        "",
        f"- **Configured to reproduce Egressy et al. setup:** {summary.get('configured_to_reproduce_egressy_setup')}",
        f"- {summary.get('reproduction_caveat')}",
        "",
        "## Artifacts",
        "",
        f"- Epoch history: `{summary.get('epoch_history_path')}`",
        f"- Best-val checkpoint (use for reproduction eval): `{summary.get('best_val_checkpoint_path')}`",
        f"- Last checkpoint: `{summary.get('last_checkpoint_path')}`",
        f"- Flat compatibility checkpoint (= last epoch): `{summary.get('flat_compat_checkpoint_path')}`",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def load_model(model, device, args, config, data_config):
    path = Path(data_config["paths"]["model_to_load"]) / f"checkpoint_{args.unique_name}.tar"
    checkpoint = torch.load(path, map_location=device)
    from morphology.temporal_flow_edge_features import assert_checkpoint_tf_edge_features_flag

    assert_checkpoint_tf_edge_features_flag(checkpoint, args, path=path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer


def load_checkpoint_auxiliary_modules(args, data_config, device) -> None:
    """Load morphology expert and/or contrastive projection from a pretrain checkpoint."""
    path = Path(data_config["paths"]["model_to_load"]) / f"checkpoint_{args.unique_name}.tar"
    if not path.is_file():
        return
    checkpoint = torch.load(path, map_location=device)
    morph_head = getattr(args, "morph_expert_head", None)
    if morph_head is not None and "morph_expert_state_dict" in checkpoint:
        morph_head.load_state_dict(checkpoint["morph_expert_state_dict"])
    proj_head = getattr(args, "contrast_projection_module", None)
    if proj_head is not None and "contrast_projection_state_dict" in checkpoint:
        proj_head.load_state_dict(checkpoint["contrast_projection_state_dict"])
    tf_head = getattr(args, "temporal_flow_aux_head", None)
    if tf_head is not None and "temporal_flow_aux_state_dict" in checkpoint:
        tf_head.load_state_dict(checkpoint["temporal_flow_aux_state_dict"])
    masked_decoder = getattr(args, "masked_edge_decoder", None)
    if masked_decoder is not None and "masked_edge_decoder_state_dict" in checkpoint:
        masked_decoder.load_state_dict(checkpoint["masked_edge_decoder_state_dict"])
