import pandas as pd
import numpy as np
import torch
import logging
from pathlib import Path

from data_util import GraphData, HeteroData, z_norm, create_hetero_obj
from dataset_specs import get_dataset_spec, spec_summary
from dataset_splits import log_split_label_stats, temporal_edge_split
from pattern_metadata import (
    load_laundering_pattern_metadata,
    resolve_pattern_metadata_path,
)


def get_data(args, data_config):
    '''Loads edge-classification transaction graph data.

    1. Resolve dataset spec (AMLWorld default or registered adapter e.g. PaySim).
    2. Load formatted_transactions.csv and build tensors.
    3. Temporal train/val/test split per spec.
    4. PyG Data objects for each split.
    '''

    spec = get_dataset_spec(args.data)
    spec.validate_edge_feature_cols()
    logging.info("Dataset spec: %s", spec_summary(spec))

    aml_root = Path(str(data_config["paths"]["aml_data"]))
    transaction_file = aml_root / args.data / spec.formatted_csv_name()
    if not transaction_file.is_file():
        raise FileNotFoundError(
            f"Missing {transaction_file}. "
            f"For PaySim run: python format_paysim.py <PS_*.csv> "
            f"and copy to aml-data/PaySim/formatted_transactions.csv"
        )

    df_edges = pd.read_csv(transaction_file)

    logging.info(f'Available Edge Features: {df_edges.columns.tolist()}')

    pattern_metadata_by_edge_id = {}
    load_pattern = bool(getattr(args, "load_pattern_metadata", False))
    if spec.supports_pattern_metadata or load_pattern:
        metadata_path = resolve_pattern_metadata_path(
            data_config,
            args.data,
            explicit_path=getattr(args, "pattern_metadata", None),
            load_requested=load_pattern,
        )
        if metadata_path is not None:
            pattern_metadata_by_edge_id = load_laundering_pattern_metadata(
                metadata_path, df_edges
            )
    elif getattr(args, "pattern_metadata", None):
        logging.warning(
            "pattern_metadata path set but dataset %s does not support pattern metadata",
            args.data,
        )

    if spec.label_col not in df_edges.columns:
        raise ValueError(
            f"{transaction_file} missing label column {spec.label_col!r}. "
            f"Columns: {df_edges.columns.tolist()}"
        )

    df_edges["Timestamp"] = df_edges["Timestamp"] - df_edges["Timestamp"].min()

    max_n_id = df_edges.loc[:, ["from_id", "to_id"]].to_numpy().max() + 1
    df_nodes = pd.DataFrame({"NodeID": np.arange(max_n_id), "Feature": np.ones(max_n_id)})
    timestamps = torch.Tensor(df_edges["Timestamp"].to_numpy())
    y = torch.LongTensor(df_edges[spec.label_col].to_numpy())

    logging.info(
        "Positive edge ratio (%s) = %d / %d = %.4f%%",
        spec.label_col,
        int(y.sum()),
        len(y),
        float(y.float().mean() * 100),
    )
    logging.info(f"Number of nodes (holdings doing transcations) = {df_nodes.shape[0]}")
    logging.info(f"Number of transactions = {df_edges.shape[0]}")

    edge_features = list(spec.edge_feature_cols)
    node_features = ["Feature"]

    logging.info(f"Edge features being used: {edge_features}")
    logging.info(f'Node features being used: {node_features} ("Feature" is a placeholder feature of all 1s)')

    x = torch.tensor(df_nodes.loc[:, node_features].to_numpy()).float()
    edge_index = torch.LongTensor(df_edges.loc[:, ["from_id", "to_id"]].to_numpy().T)
    edge_attr = torch.tensor(df_edges.loc[:, edge_features].to_numpy()).float()

    bucket_sec = 24 * 3600 if spec.split_mode == "calendar_day" else 3600
    n_buckets = int(timestamps.max() / bucket_sec + 1)
    logging.info(
        "Temporal buckets (%s): %d buckets, %d transactions",
        spec.split_mode,
        n_buckets,
        y.shape[0],
    )

    tr_inds, val_inds, te_inds, _split = temporal_edge_split(timestamps, y, spec)
    log_split_label_stats(
        y, tr_inds, val_inds, te_inds, label_col=spec.label_col, split_mode=spec.split_mode
    )

    #Creating the final data objects
    tr_x, val_x, te_x = x, x, x
    e_tr = tr_inds.numpy()
    e_val = np.concatenate([tr_inds, val_inds])

    tr_edge_index, tr_edge_attr, tr_y, tr_edge_times = (
        edge_index[:, e_tr],
        edge_attr[e_tr],
        y[e_tr],
        timestamps[e_tr],
    )
    val_edge_index, val_edge_attr, val_y, val_edge_times = (
        edge_index[:, e_val],
        edge_attr[e_val],
        y[e_val],
        timestamps[e_val],
    )
    te_edge_index, te_edge_attr, te_y, te_edge_times = (
        edge_index,
        edge_attr,
        y,
        timestamps,
    )

    tr_data = GraphData(
        x=tr_x, y=tr_y, edge_index=tr_edge_index, edge_attr=tr_edge_attr, timestamps=tr_edge_times
    )
    val_data = GraphData(
        x=val_x,
        y=val_y,
        edge_index=val_edge_index,
        edge_attr=val_edge_attr,
        timestamps=val_edge_times,
    )
    te_data = GraphData(
        x=te_x, y=te_y, edge_index=te_edge_index, edge_attr=te_edge_attr, timestamps=te_edge_times
    )

    #Adding ports and time-deltas if applicable
    if args.ports:
        logging.info(f"Start: adding ports")
        tr_data.add_ports()
        val_data.add_ports()
        te_data.add_ports()
        logging.info(f"Done: adding ports")
    if args.tds:
        logging.info(f"Start: adding time-deltas")
        tr_data.add_time_deltas()
        val_data.add_time_deltas()
        te_data.add_time_deltas()
        logging.info(f"Done: adding time-deltas")

    #Normalize data
    tr_data.x = z_norm(tr_data.x)
    val_data.x = tr_data.x.clone()
    te_data.x = tr_data.x.clone()

    if not args.model == "rgcn":
        tr_data.edge_attr, val_data.edge_attr, te_data.edge_attr = (
            z_norm(tr_data.edge_attr),
            z_norm(val_data.edge_attr),
            z_norm(te_data.edge_attr),
        )
    else:
        tr_data.edge_attr[:, :-1], val_data.edge_attr[:, :-1], te_data.edge_attr[:, :-1] = (
            z_norm(tr_data.edge_attr[:, :-1]),
            z_norm(val_data.edge_attr[:, :-1]),
            z_norm(te_data.edge_attr[:, :-1]),
        )

    if args.reverse_mp:
        tr_data = create_hetero_obj(
            tr_data.x, tr_data.y, tr_data.edge_index, tr_data.edge_attr, tr_data.timestamps, args
        )
        val_data = create_hetero_obj(
            val_data.x,
            val_data.y,
            val_data.edge_index,
            val_data.edge_attr,
            val_data.timestamps,
            args,
        )
        te_data = create_hetero_obj(
            te_data.x, te_data.y, te_data.edge_index, te_data.edge_attr, te_data.timestamps, args
        )

    logging.info(f"train data object: {tr_data}")
    logging.info(f"validation data object: {val_data}")
    logging.info(f"test data object: {te_data}")

    tr_data.is_contrastive = True
    val_data.is_contrastive = True
    te_data.is_contrastive = True

    te_data.csv_edge_ids = torch.LongTensor(df_edges["EdgeID"].astype(np.int64).to_numpy())
    te_data.pattern_metadata_by_edge_id = pattern_metadata_by_edge_id
    te_data.dataset_spec_summary = spec_summary(spec)

    return tr_data, val_data, te_data, tr_inds, val_inds, te_inds


def get_forward_edge_index(data):
    """
    Returns the correct edge_index for adjacency construction.

    Supports:
    - homogeneous GraphData
    - heterogeneous graphs (reverse_mp)

    For hetero graphs, uses only forward edges.
    """

    # Heterogeneous graph
    if hasattr(data, "edge_types"):
        return data["node", "to", "node"].edge_index

    # Homogeneous graph
    return data.edge_index
