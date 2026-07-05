#!/usr/bin/env python3
"""
CPU downstream probe comparing raw / morphology / SSL embedding feature sets.

Uses the same logistic-regression probe protocol as ``linear_probe.py``:
class weights, val-tuned max-F1 threshold, fixed 0.5 threshold baseline, AUROC and AUPRC.

Non-embedding features are label-free and scaled with train-split statistics only.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataset_specs import DEFAULT_EDGE_FEATURE_COLS, get_dataset_spec
from dataset_splits import temporal_edge_split
from linear_probe import (
    evaluate_probe,
    fit_logistic_probe,
    load_embedding_npz,
    resolve_class_weight,
    serialize_class_weight,
    tune_threshold_max_f1,
)
from transaction_knn.features import (
    CATEGORICAL_EDGE_COLUMNS,
    load_data_config,
    resolve_amount_column,
)
from util import logger_setup, set_seed

FEATURE_MODES = (
    "raw",
    "morph",
    "raw+morph",
    "embedding",
    "embedding+raw",
    "embedding+raw+morph",
)
ALERT_BUDGET_KS = (100, 500, 1000)

RAW_GROUPS = ("edge_native",)
MORPH_GROUPS = ("degree_fan", "flow_balance", "temporal_behavior")


@dataclass
class CategoricalEncoderState:
    encoding: str
    maps: Dict[str, Dict[str, int]] = field(default_factory=dict)
    one_hot_columns: Dict[str, List[str]] = field(default_factory=dict)

    def fit_transform_column(self, train: pd.Series, full: pd.Series) -> Tuple[np.ndarray, List[str]]:
        col = str(train.name)
        train_vals = train.fillna("__missing__").astype(str)
        full_vals = full.fillna("__missing__").astype(str)
        if self.encoding == "one_hot":
            categories = sorted(train_vals.unique())
            columns = [f"{col}_{cat}" for cat in categories]
            self.one_hot_columns[col] = columns
            out = np.zeros((len(full_vals), len(columns)), dtype=np.float32)
            cat_to_j = {cat: j for j, cat in enumerate(categories)}
            for i, val in enumerate(full_vals.to_numpy()):
                j = cat_to_j.get(val)
                if j is not None:
                    out[i, j] = 1.0
            return out, columns
        uniques = sorted(train_vals.unique())
        mapping = {cat: idx for idx, cat in enumerate(uniques)}
        self.maps[col] = mapping
        codes = full_vals.map(mapping).fillna(-1).astype(np.float32).to_numpy().reshape(-1, 1)
        return codes, [f"{col}_ordinal"]

    def metadata(self) -> Dict[str, Any]:
        return {
            "encoding": self.encoding,
            "ordinal_maps": {k: len(v) for k, v in self.maps.items()},
            "one_hot_columns": {k: len(v) for k, v in self.one_hot_columns.items()},
        }


@dataclass
class GroupwiseScaler:
    group_slices: Dict[str, slice]
    scalers: Dict[str, StandardScaler] = field(default_factory=dict)

    def fit(self, x_train: np.ndarray) -> None:
        for group, sl in self.group_slices.items():
            scaler = StandardScaler()
            scaler.fit(x_train[:, sl])
            self.scalers[group] = scaler

    def transform(self, x: np.ndarray) -> np.ndarray:
        out = np.empty_like(x, dtype=np.float32)
        for group, sl in self.group_slices.items():
            out[:, sl] = self.scalers[group].transform(x[:, sl]).astype(np.float32)
        return out


def _safe_log1p(values: np.ndarray) -> np.ndarray:
    return np.log1p(np.maximum(values.astype(np.float64), 0.0)).astype(np.float32)


def degree_fan_features_train_static(
    df: pd.DataFrame, df_train: pd.DataFrame
) -> Tuple[np.ndarray, List[str]]:
    train_from = df_train["from_id"].astype(np.int64).to_numpy()
    train_to = df_train["to_id"].astype(np.int64).to_numpy()
    from_ids = df["from_id"].astype(np.int64).to_numpy()
    to_ids = df["to_id"].astype(np.int64).to_numpy()
    max_node = (
        int(
            max(
                from_ids.max(initial=0),
                to_ids.max(initial=0),
                train_from.max(initial=0),
                train_to.max(initial=0),
            )
        )
        + 1
    )
    out_deg = np.bincount(train_from, minlength=max_node).astype(np.float32)
    in_deg = np.bincount(train_to, minlength=max_node).astype(np.float32)
    features = np.column_stack(
        [
            _safe_log1p(out_deg[from_ids]),
            _safe_log1p(in_deg[from_ids]),
            _safe_log1p(out_deg[to_ids]),
            _safe_log1p(in_deg[to_ids]),
            _safe_log1p(out_deg[from_ids] + in_deg[from_ids]),
            _safe_log1p(out_deg[to_ids] + in_deg[to_ids]),
            _safe_log1p(out_deg[from_ids] + out_deg[to_ids]),
            _safe_log1p(in_deg[from_ids] + in_deg[to_ids]),
        ]
    ).astype(np.float32)
    names = [
        "log1p_sender_out_degree_train",
        "log1p_sender_in_degree_train",
        "log1p_receiver_out_degree_train",
        "log1p_receiver_in_degree_train",
        "log1p_sender_total_degree_train",
        "log1p_receiver_total_degree_train",
        "log1p_pair_out_degree_sum_train",
        "log1p_pair_in_degree_sum_train",
    ]
    return features, names


def flow_balance_features_train_static(
    df: pd.DataFrame, df_train: pd.DataFrame, amount_col: str
) -> Tuple[np.ndarray, List[str]]:
    train_from = df_train["from_id"].astype(np.int64).to_numpy()
    train_to = df_train["to_id"].astype(np.int64).to_numpy()
    train_amounts = np.maximum(df_train[amount_col].astype(np.float64).to_numpy(), 0.0)
    from_ids = df["from_id"].astype(np.int64).to_numpy()
    to_ids = df["to_id"].astype(np.int64).to_numpy()
    max_node = (
        int(
            max(
                from_ids.max(initial=0),
                to_ids.max(initial=0),
                train_from.max(initial=0),
                train_to.max(initial=0),
            )
        )
        + 1
    )
    amount_out = np.bincount(train_from, weights=train_amounts, minlength=max_node).astype(np.float64)
    amount_in = np.bincount(train_to, weights=train_amounts, minlength=max_node).astype(np.float64)
    eps = 1e-8
    s_out = amount_out[from_ids]
    s_in = amount_in[from_ids]
    r_in = amount_in[to_ids]
    r_out = amount_out[to_ids]
    s_ratio = np.clip((s_out - s_in) / (s_out + s_in + eps), -1.0, 1.0)
    r_ratio = np.clip((r_out - r_in) / (r_out + r_in + eps), -1.0, 1.0)
    x = np.column_stack(
        [
            _safe_log1p(s_out),
            _safe_log1p(s_in),
            _safe_log1p(r_in),
            _safe_log1p(r_out),
            s_ratio,
            r_ratio,
        ]
    ).astype(np.float32)
    names = [
        "log1p_sender_out_amount_train",
        "log1p_sender_in_amount_train",
        "log1p_receiver_in_amount_train",
        "log1p_receiver_out_amount_train",
        "sender_flow_balance_ratio_train",
        "receiver_flow_balance_ratio_train",
    ]
    return x, names


def temporal_behavior_features(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    ts = df["Timestamp"].astype(float).to_numpy()
    ts_norm = (ts - ts.min()) / max(float(ts.max() - ts.min()), 1.0)
    order = np.argsort(ts)
    inter = np.zeros_like(ts, dtype=np.float32)
    inter[1:] = np.log1p(np.maximum(np.diff(ts[order]), 0.0))
    inv = np.empty_like(order)
    inv[order] = np.arange(order.shape[0])
    inter = inter[inv]
    return np.column_stack([ts_norm, inter]).astype(np.float32), [
        "timestamp_norm",
        "log1p_interarrival",
    ]


def edge_native_features_train_fit(
    df: pd.DataFrame,
    df_train: pd.DataFrame,
    *,
    categorical_encoding: str,
    amount_col: str,
) -> Tuple[np.ndarray, List[str], CategoricalEncoderState]:
    encoder = CategoricalEncoderState(encoding=categorical_encoding)
    parts: List[np.ndarray] = []
    names: List[str] = []
    for col in DEFAULT_EDGE_FEATURE_COLS:
        if col not in df.columns:
            continue
        if col in CATEGORICAL_EDGE_COLUMNS:
            x, n = encoder.fit_transform_column(df_train[col], df[col])
            parts.append(x)
            names.extend(n)
        elif "Amount" in col:
            use_col = amount_col if amount_col in df.columns else col
            parts.append(_safe_log1p(df[use_col].astype(float).to_numpy()).reshape(-1, 1))
            names.append(f"log1p_{use_col}")
        else:
            parts.append(df[col].astype(float).to_numpy().reshape(-1, 1))
            names.append(col)
    return np.concatenate(parts, axis=1).astype(np.float32), names, encoder


def resolve_mode_groups(features: str) -> Tuple[bool, Tuple[str, ...]]:
    if features == "raw":
        return False, RAW_GROUPS
    if features == "morph":
        return False, MORPH_GROUPS
    if features == "raw+morph":
        return False, RAW_GROUPS + MORPH_GROUPS
    if features == "embedding":
        return True, ()
    if features == "embedding+raw":
        return True, RAW_GROUPS
    if features == "embedding+raw+morph":
        return True, RAW_GROUPS + MORPH_GROUPS
    raise ValueError(f"Unsupported --features {features!r}")


def build_full_feature_matrix(
    df: pd.DataFrame,
    df_train: pd.DataFrame,
    groups: Sequence[str],
    *,
    categorical_encoding: str,
) -> Tuple[np.ndarray, List[str], Dict[str, slice], Dict[str, Any]]:
    amount_col = resolve_amount_column(df)
    matrices: List[np.ndarray] = []
    names: List[str] = []
    group_slices: Dict[str, slice] = {}
    meta: Dict[str, Any] = {"groups": list(groups), "categorical_encoding": categorical_encoding}

    for group in groups:
        start = sum(m.shape[1] for m in matrices)
        if group == "edge_native":
            x, n, enc = edge_native_features_train_fit(
                df,
                df_train,
                categorical_encoding=categorical_encoding,
                amount_col=amount_col,
            )
            meta["edge_native_encoder"] = enc.metadata()
        elif group == "degree_fan":
            x, n = degree_fan_features_train_static(df, df_train)
        elif group == "flow_balance":
            x, n = flow_balance_features_train_static(df, df_train, amount_col)
        elif group == "temporal_behavior":
            x, n = temporal_behavior_features(df)
        else:
            raise ValueError(f"Unsupported feature group {group!r}")
        matrices.append(x)
        names.extend(n)
        group_slices[group] = slice(start, start + x.shape[1])

    if not matrices:
        raise ValueError("No feature groups requested")
    features = np.concatenate(matrices, axis=1).astype(np.float32)
    meta["feature_names"] = names
    meta["group_dims"] = {g: int(group_slices[g].stop - group_slices[g].start) for g in groups}
    return features, names, group_slices, meta


def load_dataset_frames(data: str, data_config_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, Any]:
    spec = get_dataset_spec(data)
    cfg = load_data_config(data_config_path)
    csv_path = Path(cfg["paths"]["aml_data"]) / data / spec.formatted_csv_name()
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    df["Timestamp"] = df["Timestamp"] - df["Timestamp"].min()
    y = torch.LongTensor(df[spec.label_col].to_numpy())
    timestamps = torch.Tensor(df["Timestamp"].to_numpy())
    tr_inds, val_inds, te_inds, _ = temporal_edge_split(timestamps, y, spec)
    tr_np = tr_inds.numpy()
    df_train = df.iloc[tr_np].reset_index(drop=True)
    return df, df_train, tr_np, val_inds.numpy(), te_inds.numpy(), spec


def assemble_split_matrix(
    z: np.ndarray,
    edge_ids: np.ndarray,
    x_full: Optional[np.ndarray],
    *,
    use_embedding: bool,
) -> np.ndarray:
    blocks: List[np.ndarray] = []
    if use_embedding:
        blocks.append(z.astype(np.float32))
    if x_full is not None:
        blocks.append(x_full[edge_ids])
    if not blocks:
        raise ValueError("Empty feature matrix")
    return np.concatenate(blocks, axis=1).astype(np.float32)


def run_probe_for_mode(
    *,
    features: str,
    embedding_dir: Path,
    df: pd.DataFrame,
    df_train: pd.DataFrame,
    tr_np: np.ndarray,
    args,
) -> Dict[str, Any]:
    use_embedding, groups = resolve_mode_groups(features)
    split_paths = {
        "train": embedding_dir / "train.npz",
        "val": embedding_dir / "val.npz",
        "test": embedding_dir / "test.npz",
    }
    split_arrays: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for split_name, path in split_paths.items():
        split_arrays[split_name] = load_embedding_npz(path)

    x_full: Optional[np.ndarray] = None
    feature_meta: Dict[str, Any] = {
        "feature_mode": features,
        "feature_groups_included": list(groups),
        "uses_embedding": use_embedding,
        "embedding_dim": int(split_arrays["train"][0].shape[1]) if use_embedding else 0,
        "categorical_encoding": args.categorical_encoding,
        "scaling": "none",
        "scaling_policy": "train-fit StandardScaler on non-embedding groups only; embeddings unscaled",
    }
    scaler_meta: Optional[Dict[str, Any]] = None

    if groups:
        x_raw, feat_names, group_slices, group_meta = build_full_feature_matrix(
            df,
            df_train,
            groups,
            categorical_encoding=args.categorical_encoding,
        )
        scaler = GroupwiseScaler(group_slices=group_slices)
        scaler.fit(x_raw[tr_np])
        x_full = scaler.transform(x_raw)
        feature_meta.update(group_meta)
        feature_meta["scaling"] = "standard"
        feature_meta["feature_dim_non_embedding"] = int(x_full.shape[1])
        scaler_meta = {
            "groups_scaled": list(group_slices.keys()),
            "fit_on": "train_split_rows",
        }
    else:
        feature_meta["feature_dim_non_embedding"] = 0

    x_train = assemble_split_matrix(
        split_arrays["train"][0],
        split_arrays["train"][2],
        x_full,
        use_embedding=use_embedding,
    )
    y_train = split_arrays["train"][1]
    for split_name in ("train", "val", "test"):
        z, y, edge_ids = split_arrays[split_name]
        x = assemble_split_matrix(z, edge_ids, x_full, use_embedding=use_embedding)
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"{split_name}: row mismatch x={x.shape[0]} y={y.shape[0]}")
        labels_from_df = df.iloc[edge_ids][get_dataset_spec(args.data).label_col].to_numpy()
        if not np.array_equal(labels_from_df.astype(np.int64), y.astype(np.int64)):
            raise ValueError(f"{split_name}: labels disagree between embeddings and dataframe")

    feature_meta["feature_dim_total"] = int(x_train.shape[1])
    feature_meta["groupwise_scaler"] = scaler_meta

    class_weight = resolve_class_weight(args)
    probe_c = float(getattr(args, "probe_C", 1.0))
    clf = fit_logistic_probe(
        x_train,
        y_train,
        class_weight=class_weight,
        max_iter=int(args.probe_max_iter),
        seed=int(args.seed),
        n_jobs=int(args.probe_n_jobs),
        C=probe_c,
    )

    z_val, y_val, _ = split_arrays["val"]
    x_val = assemble_split_matrix(z_val, split_arrays["val"][2], x_full, use_embedding=use_embedding)
    val_proba = clf.predict_proba(x_val)[:, 1]
    selected_threshold, val_f1_at_selection = tune_threshold_max_f1(y_val, val_proba)

    report_all_splits = bool(getattr(args, "report_all_splits", False))

    def _alert_budget_metrics(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        proba = clf.predict_proba(x)[:, 1]
        y = y.astype(np.int64)
        n = int(y.shape[0])
        positives = int(y.sum())
        prevalence = float(y.mean()) if n else float("nan")
        out: Dict[str, float] = {}
        if n == 0:
            return out
        order = np.argsort(-proba)
        for k in ALERT_BUDGET_KS:
            kk = min(k, n)
            top = order[:kk]
            tp = int(y[top].sum())
            precision = float(tp / kk) if kk else float("nan")
            recall = float(tp / positives) if positives else float("nan")
            lift = float(precision / prevalence) if prevalence > 0 else float("nan")
            out[f"precision_at_{k}"] = precision
            out[f"recall_at_{k}"] = recall
            out[f"lift_at_{k}"] = lift
        return out

    result: Dict[str, Any] = {
        "features": features,
        "embedding_dir": str(embedding_dir),
        "data": args.data,
        "class_weight": serialize_class_weight(class_weight),
        "class_weight_mode": str(args.class_weight),
        "probe_C": probe_c,
        "probe_max_iter": int(args.probe_max_iter),
        "seed": int(args.seed),
        "classification_threshold": {
            "method": "max_f1_on_val",
            "value": float(selected_threshold),
            "selected_on": "val",
        },
        "val_f1_at_selected_threshold": float(val_f1_at_selection),
        "feature_meta": feature_meta,
        "train": {},
        "val": {},
        "test": {},
        "test_at_threshold_0.5": {},
    }

    def _split_matrix(split_name: str) -> np.ndarray:
        z, _, edge_ids = split_arrays[split_name]
        return assemble_split_matrix(z, edge_ids, x_full, use_embedding=use_embedding)

    if report_all_splits:
        for split_name in ("train", "val"):
            metrics = evaluate_probe(
                clf,
                _split_matrix(split_name),
                split_arrays[split_name][1],
                split_name,
                threshold=selected_threshold,
            )
            metrics_default = evaluate_probe(
                clf,
                _split_matrix(split_name),
                split_arrays[split_name][1],
                split_name,
                threshold=0.5,
            )
            result[split_name] = {
                "n": metrics["n"],
                "positive_rate": metrics["positive_rate"],
                "auroc": metrics["auroc"],
                "auprc": metrics["auprc"],
                "f1": metrics["f1"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "threshold": float(selected_threshold),
                "f1_at_0_5": metrics_default["f1"],
            }
            result[split_name].update(_alert_budget_metrics(_split_matrix(split_name), split_arrays[split_name][1]))

    test_metrics = evaluate_probe(
        clf,
        assemble_split_matrix(
            split_arrays["test"][0],
            split_arrays["test"][2],
            x_full,
            use_embedding=use_embedding,
        ),
        split_arrays["test"][1],
        "test",
        threshold=selected_threshold,
    )
    test_default = evaluate_probe(
        clf,
        assemble_split_matrix(
            split_arrays["test"][0],
            split_arrays["test"][2],
            x_full,
            use_embedding=use_embedding,
        ),
        split_arrays["test"][1],
        "test",
        threshold=0.5,
    )
    result["test"] = {
        "n": test_metrics["n"],
        "positive_rate": test_metrics["positive_rate"],
        "auroc": test_metrics["auroc"],
        "auprc": test_metrics["auprc"],
        "f1": test_metrics["f1"],
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "threshold": float(selected_threshold),
        "f1_at_0_5": test_default["f1"],
        "val_f1": float(val_f1_at_selection),
        "feature_dim": int(x_train.shape[1]),
        "feature_groups_included": (
            (["embedding"] if use_embedding else [])
            + list(groups)
        ),
    }
    result["test"].update(
        _alert_budget_metrics(
            assemble_split_matrix(
                split_arrays["test"][0],
                split_arrays["test"][2],
                x_full,
                use_embedding=use_embedding,
            ),
            split_arrays["test"][1],
        )
    )
    result["test_at_threshold_0.5"] = {
        "f1": test_default["f1"],
        "precision": test_default["precision"],
        "recall": test_default["recall"],
    }
    logging.info(
        "probe_feature_ablation %s: AUROC=%.4f AUPRC=%.4f F1=%.4f P=%.4f R=%.4f dim=%d groups=%s",
        features,
        result["test"]["auroc"],
        result["test"]["auprc"],
        result["test"]["f1"],
        result["test"]["precision"],
        result["test"]["recall"],
        result["test"]["feature_dim"],
        result["test"]["feature_groups_included"],
    )
    return result


def write_markdown(path: Path, payload: Dict[str, Any]) -> None:
    lines = [
        "# Probe feature ablation",
        "",
        f"- **data:** {payload.get('data')}",
        f"- **embedding_dir:** `{payload.get('embedding_dir')}`",
        f"- **categorical_encoding:** {payload.get('categorical_encoding')}",
        "",
        "| features | dim | groups | AUROC | AUPRC | F1 | Prec | Recall | thr | val F1 | F1@0.5 |",
        "|----------|-----|--------|-------|-------|-----|------|--------|-----|--------|--------|",
    ]
    for row in payload.get("runs", []):
        test = row["test"]
        t05 = row["test_at_threshold_0.5"]
        groups = ", ".join(test["feature_groups_included"]) or "—"
        lines.append(
            f"| `{row['features']}` | {test['feature_dim']} | {groups} | "
            f"{test['auroc']:.3f} | {test['auprc']:.3f} | {test['f1']:.3f} | {test['precision']:.3f} | "
            f"{test['recall']:.3f} | {test['threshold']:.3f} | {test['val_f1']:.3f} | "
            f"{t05['f1']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Scaling: non-embedding groups use `StandardScaler` fit on **train split rows only**; "
            "SSL embeddings are passed through unscaled.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="Small-HI")
    parser.add_argument("--data_config", default="data_config.json")
    parser.add_argument(
        "--embedding_dir",
        required=True,
        help="Directory with train/val/test.npz from embedding_extraction.py",
    )
    parser.add_argument(
        "--features",
        required=True,
        choices=[*FEATURE_MODES, "all"],
        help="Feature set for the probe, or 'all' to run every mode sequentially.",
    )
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    parser.add_argument(
        "--categorical_encoding",
        default="ordinal",
        choices=["ordinal", "one_hot"],
        help="Encoding for Received Currency and Payment Format (fit on train only).",
    )
    parser.add_argument("--class_weight", default="model", choices=["balanced", "none", "model", "explicit"])
    parser.add_argument(
        "--class_weight_pos",
        type=float,
        default=None,
        help="Positive class weight for --class_weight explicit; resolves to {0: 1.0, 1: value}.",
    )
    parser.add_argument("--model", default="gin")
    parser.add_argument(
        "--probe_C",
        type=float,
        default=1.0,
        help="LogisticRegression inverse regularization strength (sklearn C; default 1.0).",
    )
    parser.add_argument(
        "--feature_modes",
        type=str,
        default=None,
        help="Comma-separated feature modes (overrides --features when set).",
    )
    parser.add_argument(
        "--report_all_splits",
        action="store_true",
        help="Include train/val metrics in each run row (test always included).",
    )
    parser.add_argument("--probe_max_iter", type=int, default=1000)
    parser.add_argument("--probe_n_jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--testing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger_setup()
    set_seed(args.seed)

    embedding_dir = Path(args.embedding_dir)
    if not embedding_dir.is_dir():
        raise FileNotFoundError(f"embedding_dir not found: {embedding_dir}")
    for split in ("train", "val", "test"):
        if not (embedding_dir / f"{split}.npz").is_file():
            raise FileNotFoundError(f"Missing {embedding_dir / f'{split}.npz'}")

    df, df_train, tr_np, _, _, _ = load_dataset_frames(args.data, args.data_config)
    if args.feature_modes:
        modes = [m.strip() for m in args.feature_modes.split(",") if m.strip()]
        for mode in modes:
            if mode not in FEATURE_MODES:
                raise ValueError(f"Unknown feature mode {mode!r}; expected one of {FEATURE_MODES}")
    elif args.features == "all":
        modes = list(FEATURE_MODES)
    else:
        modes = [args.features]

    runs: List[Dict[str, Any]] = []
    for mode in modes:
        runs.append(
            run_probe_for_mode(
                features=mode,
                embedding_dir=embedding_dir,
                df=df,
                df_train=df_train,
                tr_np=tr_np,
                args=args,
            )
        )

    payload = {
        "data": args.data,
        "embedding_dir": str(embedding_dir),
        "categorical_encoding": args.categorical_encoding,
        "protocol": {
            "probe": "sklearn LogisticRegression (lbfgs)",
            "threshold_tuning": "max_f1_on_val",
            "fixed_threshold_baseline": 0.5,
            "scaling": "StandardScaler fit on train split rows for non-embedding features only",
            "label_free_feature_policy": {
                "raw": list(RAW_GROUPS),
                "morph": list(MORPH_GROUPS),
                "degree_fan_flow_balance": "train-split static graph statistics only",
            },
        },
        "runs": runs,
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logging.info("Wrote %s", out_json)

    write_markdown(Path(args.output_md), payload)
    logging.info("Wrote %s", args.output_md)


if __name__ == "__main__":
    main()
