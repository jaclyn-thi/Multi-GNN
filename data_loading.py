import pandas as pd
import numpy as np
import torch
import logging
import time
from pathlib import Path

from data_util import (
    GraphData,
    HeteroData,
    append_rgcn_relation_type,
    create_hetero_obj,
    z_norm,
)
from dataset_specs import get_dataset_spec, spec_summary
from dataset_splits import log_split_label_stats, temporal_edge_split
from morphology.temporal_flow_edge_features import maybe_append_temporal_flow_edge_features
from pattern_metadata import (
    load_laundering_pattern_metadata,
    resolve_pattern_metadata_path,
)


def _stage_tick(args, name: str, t0: float) -> float:
    """Record elapsed seconds for ``name`` on ``args._stage_timings`` if present."""
    timings = getattr(args, "_stage_timings", None)
    if isinstance(timings, dict):
        timings[name] = float(time.perf_counter() - t0)
    return time.perf_counter()


def get_data(args, data_config):
    '''Loads edge-classification transaction graph data.

    1. Resolve dataset spec (AMLWorld default or registered adapter e.g. PaySim).
    2. Load formatted_transactions.csv and build tensors.
    3. Temporal train/val/test split per spec.
    4. PyG Data objects for each split.
    '''

    if getattr(args, "record_stage_timings", False) and not hasattr(args, "_stage_timings"):
        args._stage_timings = {}

    t_all = time.perf_counter()
    spec = get_dataset_spec(args.data)
    spec.validate_edge_feature_cols()
    logging.info("Dataset spec: %s", spec_summary(spec))

    aml_root = Path(str(data_config["paths"]["aml_data"]))
    transaction_file = aml_root / args.data / spec.formatted_csv_name()
    t_csv = time.perf_counter()
    if not transaction_file.is_file():
        raise FileNotFoundError(
            f"Missing {transaction_file}. "
            f"For PaySim run: python format_paysim.py <PS_*.csv> "
            f"and copy to aml-data/PaySim/formatted_transactions.csv"
        )

    df_edges = pd.read_csv(transaction_file)
    t_csv = _stage_tick(args, "data_loading_csv_sec", t_csv)

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
        t_ports = time.perf_counter()
        tr_data.add_ports()
        val_data.add_ports()
        te_data.add_ports()
        _stage_tick(args, "ports_construction_sec", t_ports)
        logging.info(f"Done: adding ports")
    if args.tds:
        logging.info(f"Start: adding time-deltas")
        t_tds = time.perf_counter()
        tr_data.add_time_deltas()
        val_data.add_time_deltas()
        te_data.add_time_deltas()
        _stage_tick(args, "tds_construction_sec", t_tds)
        logging.info(f"Done: adding time-deltas")

    #Normalize data
    # Node x: always train-fit (clone to val/test) — inductive for nodes.
    tr_data.x = z_norm(tr_data.x)
    val_data.x = tr_data.x.clone()
    te_data.x = tr_data.x.clone()

    # Edge attr z-norm:
    # - Default (legacy): independent per-graph z_norm on train / train∪val / all.
    #   Transductive w.r.t. split-graph edge statistics; must NOT be used for
    #   inductive transfer claims (e.g. AMLWorld→PaySim frozen D+).
    # - --train_fit_edge_znorm: fit mean/std on train edge_attr only; apply to val/test.
    train_fit_edge = bool(getattr(args, "train_fit_edge_znorm", False))
    if train_fit_edge:
        logging.info(
            "Edge attr z-norm: train-fit inductive "
            "(fit mean/std on train edge_attr only; apply to val/test). "
            "Independent per-graph z_norm is the legacy/transductive path."
        )
        mean = tr_data.edge_attr.mean(0).unsqueeze(0)
        std = tr_data.edge_attr.std(0).unsqueeze(0)
        std = torch.where(std == 0, torch.tensor(1, dtype=torch.float32).cpu(), std)
        tr_data.edge_attr = (tr_data.edge_attr - mean) / std
        val_data.edge_attr = (val_data.edge_attr - mean) / std
        te_data.edge_attr = (te_data.edge_attr - mean) / std
    else:
        logging.info(
            "Edge attr z-norm: independent per-graph (legacy/transductive). "
            "For inductive PaySim transfer use --train_fit_edge_znorm."
        )
        tr_data.edge_attr, val_data.edge_attr, te_data.edge_attr = (
            z_norm(tr_data.edge_attr),
            z_norm(val_data.edge_attr),
            z_norm(te_data.edge_attr),
        )

    if args.reverse_mp:
        correct_rev = bool(getattr(args, "correct_reverse_edge_features", False))
        logging.info(
            "Hetero reverse-edge feature semantics: mode=%s "
            "(correct_reverse_edge_features=%s ports=%s tds=%s edge_dim=%d)",
            "corrected" if correct_rev else "inherited_legacy",
            correct_rev,
            bool(getattr(args, "ports", False)),
            bool(getattr(args, "tds", False)),
            int(tr_data.edge_attr.shape[1]),
        )
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
        schema = getattr(tr_data, "edge_feature_schema", None)
        if schema is not None:
            logging.info(
                "Edge feature schema (pre-ID): names=%s indices=%s swap_pairs=%s",
                schema.get("names"),
                schema.get("indices"),
                schema.get("swap_pairs"),
            )
            # Persist on args for checkpoint / summary metadata.
            args.edge_feature_schema = schema
            args.reverse_edge_feature_semantics = getattr(
                tr_data, "reverse_edge_feature_semantics", None
            )

    # Append causal temporal-flow encoder inputs after base z_norm / hetero port swap
    # so TF columns are not mistaken for ports in create_hetero_obj.
    maybe_append_temporal_flow_edge_features(
        tr_data,
        val_data,
        te_data,
        e_tr=e_tr,
        e_val=e_val,
        args=args,
        data_name=args.data,
        n_edges_full=int(edge_attr.shape[0]),
    )

    if args.model == "rgcn":
        append_rgcn_relation_type(tr_data, args.reverse_mp)
        append_rgcn_relation_type(val_data, args.reverse_mp)
        append_rgcn_relation_type(te_data, args.reverse_mp)

    logging.info(f"train data object: {tr_data}")
    logging.info(f"validation data object: {val_data}")
    logging.info(f"test data object: {te_data}")

    tr_data.is_contrastive = True
    val_data.is_contrastive = True
    te_data.is_contrastive = True

    te_data.csv_edge_ids = torch.LongTensor(df_edges["EdgeID"].astype(np.int64).to_numpy())
    te_data.pattern_metadata_by_edge_id = pattern_metadata_by_edge_id
    te_data.dataset_spec_summary = spec_summary(spec)

    _stage_tick(args, "get_data_total_sec", t_all)
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
