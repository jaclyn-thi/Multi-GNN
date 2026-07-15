#!/usr/bin/env python3
"""Four-arm paired downstream probe: pre_embedding_3h ± raw ± temporal_flow_causal.

Arms (primary baseline is B):
  A. pre_embedding_3h
  B. pre_embedding_3h + raw
  C. pre_embedding_3h + temporal_flow_causal
  D. pre_embedding_3h + raw + temporal_flow_causal

Uses frozen pre-3h embeddings only — no SSL/GNN retraining.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from linear_probe import (
    evaluate_probe,
    fit_logistic_probe,
    load_embedding_npz,
    resolve_class_weight,
    serialize_class_weight,
    tune_threshold_max_f1,
)
from morphology.temporal_flow_causal import TEMPORAL_FLOW_CAUSAL_FEATURE_NAMES
from util import logger_setup, set_seed

ALERT_BUDGET_KS = (100, 500, 1000)
SPLITS = ("train", "val", "test")
ARMS = ("A_embedding", "B_embedding_raw", "C_embedding_temporal_flow", "D_embedding_raw_temporal_flow")


def _alert_budget_metrics(clf, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
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
        out[f"precision_at_{k}"] = float(tp / kk) if kk else float("nan")
        out[f"recall_at_{k}"] = float(tp / positives) if positives else float("nan")
        out[f"lift_at_{k}"] = (
            float(out[f"precision_at_{k}"] / prevalence) if prevalence > 0 else float("nan")
        )
    return out


def _load_temporal_flow_cache(cache_dir: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    meta_path = cache_dir / "meta.json"
    features_path = cache_dir / "features.npy"
    if not meta_path.is_file() or not features_path.is_file():
        raise FileNotFoundError(f"Missing temporal_flow cache under {cache_dir}")
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    features = np.load(features_path).astype(np.float32)
    return features, meta


def _load_splits(embedding_dir: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    out: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for split in SPLITS:
        out[split] = load_embedding_npz(embedding_dir / f"{split}.npz")
    return out


def _align_split(
    z: np.ndarray,
    y: np.ndarray,
    edge_id: np.ndarray,
    x_raw: Optional[np.ndarray],
    x_tf: Optional[np.ndarray],
    split: str,
) -> Dict[str, Any]:
    """Use embedding edge_ids as reference; verify auxiliary features align."""
    n = int(edge_id.shape[0])
    if z.shape[0] != n or y.shape[0] != n:
        raise ValueError(f"{split}: embedding row mismatch")
    if x_raw is not None and x_raw.shape[0] != n:
        raise ValueError(f"{split}: raw feature row mismatch")
    if x_tf is not None and x_tf.shape[0] != n:
        raise ValueError(f"{split}: temporal_flow row mismatch")
    return {
        "z": z.astype(np.float32),
        "y": y.astype(np.int64),
        "edge_id": edge_id.astype(np.int64),
        "x_raw": None if x_raw is None else x_raw.astype(np.float32),
        "x_tf": None if x_tf is None else x_tf.astype(np.float32),
        "coverage": {
            "rows": n,
            "positives": int(y.sum()),
            "positive_rate": float(y.mean()) if n else float("nan"),
            "joined_fraction": 1.0,
        },
    }


def _build_arm_matrix(
    aligned: Dict[str, Dict[str, Any]],
    split: str,
    arm: str,
) -> np.ndarray:
    d = aligned[split]
    blocks: List[np.ndarray] = [d["z"]]
    if arm in ("B_embedding_raw", "D_embedding_raw_temporal_flow"):
        if d["x_raw"] is None:
            raise ValueError(f"raw features required for arm {arm}")
        blocks.append(d["x_raw"])
    if arm in ("C_embedding_temporal_flow", "D_embedding_raw_temporal_flow"):
        if d["x_tf"] is None:
            raise ValueError(f"temporal_flow features required for arm {arm}")
        blocks.append(d["x_tf"])
    return np.concatenate(blocks, axis=1).astype(np.float32)


def _arm_feature_groups(arm: str) -> List[str]:
    groups = ["pre_embedding_3h"]
    if arm in ("B_embedding_raw", "D_embedding_raw_temporal_flow"):
        groups.append("raw")
    if arm in ("C_embedding_temporal_flow", "D_embedding_raw_temporal_flow"):
        groups.append("temporal_flow_causal")
    return groups


def _shuffle_temporal_within_splits(
    aligned: Dict[str, Dict[str, Any]], *, seed: int
) -> None:
    """Permute temporal_flow rows independently within each split (marginal control)."""
    rng = np.random.default_rng(int(seed))
    for split in SPLITS:
        x_tf = aligned[split]["x_tf"]
        if x_tf is None:
            raise ValueError(f"Cannot shuffle: missing x_tf for split {split}")
        perm = rng.permutation(x_tf.shape[0])
        aligned[split]["x_tf"] = x_tf[perm].copy()


def _convergence_info(clf, max_iter: int) -> Dict[str, Any]:
    n_iter = int(clf.n_iter_[0]) if getattr(clf, "n_iter_", None) is not None else None
    converged = n_iter is not None and n_iter < int(max_iter)
    return {
        "max_iter": int(max_iter),
        "n_iter": n_iter,
        "converged": converged,
        "status": "converged" if converged else "max_iter_reached",
    }


def _probe_arm(
    aligned: Dict[str, Dict[str, Any]],
    arm: str,
    *,
    class_weight: Any,
    seed: int,
    max_iter: int,
    probe_c: float,
    n_jobs: int,
) -> Dict[str, Any]:
    x_train = _build_arm_matrix(aligned, "train", arm)
    y_train = aligned["train"]["y"]
    x_val = _build_arm_matrix(aligned, "val", arm)
    y_val = aligned["val"]["y"]
    x_test = _build_arm_matrix(aligned, "test", arm)
    y_test = aligned["test"]["y"]

    clf = fit_logistic_probe(
        x_train,
        y_train,
        class_weight=class_weight,
        max_iter=max_iter,
        seed=seed,
        n_jobs=n_jobs,
        C=probe_c,
    )
    val_proba = clf.predict_proba(x_val)[:, 1]
    selected_threshold, val_f1_at_selection = tune_threshold_max_f1(y_val, val_proba)

    test_sel = evaluate_probe(clf, x_test, y_test, "test", threshold=selected_threshold)
    test_05 = evaluate_probe(clf, x_test, y_test, "test", threshold=0.5)

    coef = clf.coef_.reshape(-1)
    feature_groups = _arm_feature_groups(arm)
    result: Dict[str, Any] = {
        "arm": arm,
        "feature_groups": feature_groups,
        "feature_dim": int(x_train.shape[1]),
        "selected_threshold": float(selected_threshold),
        "val_f1_at_selected_threshold": float(val_f1_at_selection),
        "convergence": _convergence_info(clf, max_iter),
        "coefficients": {
            "values": coef.tolist(),
            "note": "Standardized inputs where scaler applied; not causal importance.",
        },
        "test": {
            "n": test_sel["n"],
            "positive_rate": test_sel["positive_rate"],
            "auroc": test_sel["auroc"],
            "auprc": test_sel["auprc"],
            "f1_at_selected_threshold": test_sel["f1"],
            "precision_at_selected_threshold": test_sel["precision"],
            "recall_at_selected_threshold": test_sel["recall"],
            "f1_at_threshold_0.5": test_05["f1"],
            "precision_at_threshold_0.5": test_05["precision"],
            "recall_at_threshold_0.5": test_05["recall"],
        },
    }
    result["test"].update(_alert_budget_metrics(clf, x_test, y_test))
    return result


def _univariate_auprc(x: np.ndarray, y: np.ndarray) -> List[Dict[str, Any]]:
    y = y.astype(np.int64)
    rows: List[Dict[str, Any]] = []
    for j, name in enumerate(TEMPORAL_FLOW_CAUSAL_FEATURE_NAMES):
        col = x[:, j]
        if len(np.unique(y)) < 2:
            auprc = float("nan")
        else:
            auprc = float(average_precision_score(y, col))
        rows.append({"feature": name, "univariate_auprc": auprc})
    return rows


def _feature_diagnostics(
    x_tf_train: np.ndarray,
    x_tf_all_splits: np.ndarray,
    y_train: np.ndarray,
) -> Dict[str, Any]:
    corr = np.corrcoef(x_tf_all_splits.T).astype(np.float64)
    no_hist = {name: float(np.mean(x_tf_train[:, j] == 0.0)) for j, name in enumerate(TEMPORAL_FLOW_CAUSAL_FEATURE_NAMES)}
    return {
        "correlation_matrix": corr.tolist(),
        "feature_names": list(TEMPORAL_FLOW_CAUSAL_FEATURE_NAMES),
        "no_history_fraction_train": no_hist,
        "univariate_auprc_train": _univariate_auprc(x_tf_train, y_train),
    }


def _deltas(arms: Dict[str, Dict[str, Any]], arm_a: str, arm_b: str) -> Dict[str, float]:
    ta = arms[arm_a]["test"]
    tb = arms[arm_b]["test"]
    keys = [
        "auroc",
        "auprc",
        "f1_at_selected_threshold",
        "precision_at_selected_threshold",
        "recall_at_selected_threshold",
        "precision_at_100",
        "recall_at_100",
        "lift_at_100",
        "precision_at_500",
        "recall_at_500",
        "lift_at_500",
        "precision_at_1000",
        "recall_at_1000",
        "lift_at_1000",
    ]
    return {k: float(tb.get(k, float("nan")) - ta.get(k, float("nan"))) for k in keys}


def run_ablation(args) -> Dict[str, Any]:
    embedding_dir = Path(args.embedding_dir)
    cache_dir = Path(args.temporal_flow_cache_dir)
    for split in SPLITS:
        if not (embedding_dir / f"{split}.npz").is_file():
            raise FileNotFoundError(f"Missing {embedding_dir / f'{split}.npz'}")

    splits = _load_splits(embedding_dir)
    tf_full, tf_meta = _load_temporal_flow_cache(cache_dir)

    from scripts.probe_feature_ablation import (  # noqa: WPS433
        GroupwiseScaler,
        RAW_GROUPS,
        build_full_feature_matrix,
        load_dataset_frames,
    )

    df, df_train, tr_np, _, _, spec = load_dataset_frames(args.data, args.data_config)
    x_raw_full, _, raw_slices, raw_meta = build_full_feature_matrix(
        df, df_train, RAW_GROUPS, categorical_encoding=args.categorical_encoding
    )
    raw_scaler = GroupwiseScaler(group_slices=raw_slices)
    raw_scaler.fit(x_raw_full[tr_np])
    x_raw_scaled = raw_scaler.transform(x_raw_full)

    tf_scaler = StandardScaler()
    tf_scaler.fit(tf_full[tr_np])
    tf_scaled = tf_scaler.transform(tf_full).astype(np.float32)

    embedding_train_edge_ids = splits["train"][2].astype(np.int64)
    temporal_train_edge_ids = tr_np.astype(np.int64)
    scaler_audit = {
        "temporal_split_train_rows": int(temporal_train_edge_ids.shape[0]),
        "embedding_train_rows": int(embedding_train_edge_ids.shape[0]),
        "train_edge_id_intersection": int(
            np.intersect1d(temporal_train_edge_ids, embedding_train_edge_ids).shape[0]
        ),
        "temporal_only_train_edges": int(
            np.setdiff1d(temporal_train_edge_ids, embedding_train_edge_ids).shape[0]
        ),
        "embedding_only_train_edges": int(
            np.setdiff1d(embedding_train_edge_ids, temporal_train_edge_ids).shape[0]
        ),
        "raw_scaler_fit_on": "temporal_split_train_indices (tr_np)",
        "temporal_flow_scaler_fit_on": "temporal_split_train_indices (tr_np)",
        "probe_fit_on": "embedding_train_split_rows (aligned train)",
    }

    aligned: Dict[str, Dict[str, Any]] = {}
    for split in SPLITS:
        z, y, edge_id = splits[split]
        x_raw = x_raw_scaled[edge_id]
        x_tf = tf_scaled[edge_id]
        aligned[split] = _align_split(z, y, edge_id, x_raw, x_tf, split)

    shuffle_applied = bool(getattr(args, "shuffle_temporal_features_within_split", False))
    if shuffle_applied:
        _shuffle_temporal_within_splits(aligned, seed=int(getattr(args, "shuffle_seed", 1)))

    min_cov = min(a["coverage"]["joined_fraction"] for a in aligned.values())
    if min_cov < float(args.min_pairing_coverage):
        raise RuntimeError(
            f"Pairing coverage {min_cov:.4f} below minimum {args.min_pairing_coverage}; aborting."
        )

    class_weight = resolve_class_weight(args)
    arms_to_run = _resolve_arms(args)
    arms: Dict[str, Dict[str, Any]] = {}
    for arm in arms_to_run:
        arms[arm] = _probe_arm(
            aligned,
            arm,
            class_weight=class_weight,
            seed=int(args.seed),
            max_iter=int(args.probe_max_iter),
            probe_c=float(args.probe_C),
            n_jobs=int(args.probe_n_jobs),
        )
        t = arms[arm]["test"]
        logging.info(
            "%s dim=%d AUPRC=%.4f F1=%.4f P@100=%.4f",
            arm,
            arms[arm]["feature_dim"],
            t["auprc"],
            t["f1_at_selected_threshold"],
            t.get("precision_at_100", float("nan")),
        )

    tf_diag = None
    if not shuffle_applied and any(
        k in arms for k in ("C_embedding_temporal_flow", "D_embedding_raw_temporal_flow")
    ):
        tf_diag = _feature_diagnostics(
            aligned["train"]["x_tf"],
            np.concatenate([aligned[s]["x_tf"] for s in SPLITS], axis=0),
            aligned["train"]["y"],
        )

    extraction_meta = None
    meta_path = embedding_dir / "meta.json"
    if meta_path.is_file():
        with meta_path.open("r", encoding="utf-8") as f:
            extraction_meta = json.load(f)

    diagnostic_tag = str(getattr(args, "diagnostic_tag", "temporal_flow_causal_ablation"))
    payload: Dict[str, Any] = {
        "diagnostic": diagnostic_tag,
        "no_ssl_retraining": True,
        "paired": True,
        "pairing": "embedding edge_id reference; raw and temporal_flow indexed by same edge_id (100% when shapes match)",
        "data": args.data,
        "run_name": args.run_name,
        "embedding_dir": str(embedding_dir),
        "temporal_flow_cache_dir": str(cache_dir),
        "representation": str(getattr(args, "representation_source", "pre_embedding_3h")),
        "representation_dim": int(aligned["train"]["z"].shape[1]),
        "primary_baseline_arm": "B_embedding_raw",
        "primary_comparison": "D_embedding_raw_temporal_flow vs B_embedding_raw",
        "probe": {
            "impl": "sklearn LogisticRegression (lbfgs)",
            "class_weight_mode": str(args.class_weight),
            "class_weight": serialize_class_weight(class_weight),
            "probe_C": float(args.probe_C),
            "probe_max_iter": int(args.probe_max_iter),
            "seed": int(args.seed),
            "threshold_tuning": "max_f1_on_val",
            "categorical_encoding": args.categorical_encoding,
            "scaling": {
                "embeddings": "none",
                "raw": "GroupwiseScaler fit on train split (probe_feature_ablation RAW_GROUPS)",
                "temporal_flow_causal": "StandardScaler fit on train split rows only",
            },
            "protocol_source": args.protocol_source,
            "alert_budget_ks": list(ALERT_BUDGET_KS),
        },
        "shuffle_control": {
            "applied": shuffle_applied,
            "shuffle_seed": int(getattr(args, "shuffle_seed", 1)) if shuffle_applied else None,
            "note": "temporal_flow_causal rows permuted independently within each split",
        },
        "scaler_audit": scaler_audit,
        "split_pairing": {s: aligned[s]["coverage"] for s in SPLITS},
        "arms_run": arms_to_run,
        "arms": arms,
        "deltas": _compute_deltas(arms),
        "temporal_flow_cache_meta": {
            "cache_version": tf_meta.get("cache_version"),
            "causal_history_policy": tf_meta.get("causal_history_policy"),
            "timestamp_handling": tf_meta.get("timestamp_handling"),
        },
        "raw_feature_meta": raw_meta,
        "extraction_meta": extraction_meta,
        "feature_diagnostics": tf_diag,
    }
    return payload


def _compute_deltas(arms: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if "B_embedding_raw" in arms and "D_embedding_raw_temporal_flow" in arms:
        out["D_minus_B_primary"] = _deltas(arms, "B_embedding_raw", "D_embedding_raw_temporal_flow")
    if "A_embedding" in arms and "C_embedding_temporal_flow" in arms:
        out["C_minus_A"] = _deltas(arms, "A_embedding", "C_embedding_temporal_flow")
    if "A_embedding" in arms and "B_embedding_raw" in arms:
        out["B_minus_A"] = _deltas(arms, "A_embedding", "B_embedding_raw")
    return out


def _resolve_arms(args) -> List[str]:
    raw = getattr(args, "arms", None)
    if not raw:
        return list(ARMS)
    selected = [a.strip() for a in str(raw).split(",") if a.strip()]
    for arm in selected:
        if arm not in ARMS:
            raise ValueError(f"Unknown arm {arm!r}; expected one of {ARMS}")
    return selected


def write_markdown(path: Path, payload: Dict[str, Any]) -> None:
    arms = payload["arms"]
    d_primary = payload.get("deltas", {}).get("D_minus_B_primary", {})
    lines = [
        f"# Temporal flow causal ablation — {payload['data']} (`{payload['run_name']}`)",
        "",
        f"- **embedding:** `{payload['embedding_dir']}` ({payload['representation_dim']}-d pre-3h)",
        f"- **cache:** `{payload['temporal_flow_cache_dir']}`",
        f"- **primary comparison:** Arm D vs Arm B (ΔAUPRC = {d_primary.get('auprc', float('nan')):+.4f})",
        f"- **no SSL retraining:** {payload['no_ssl_retraining']}",
        "",
        "## Four-arm test metrics",
        "",
        "| arm | groups | dim | AUROC | AUPRC | F1 | P@100 | R@100 | lift@100 |",
        "|-----|--------|----:|------:|------:|---:|------:|------:|---------:|",
    ]
    for key in payload.get("arms_run", ARMS):
        if key not in arms:
            continue
        arm = arms[key]
        t = arm["test"]
        groups = ", ".join(arm["feature_groups"])
        lines.append(
            f"| {key} | {groups} | {arm['feature_dim']} | {t['auroc']:.4f} | {t['auprc']:.4f} | "
            f"{t['f1_at_selected_threshold']:.4f} | {t.get('precision_at_100', float('nan')):.4f} | "
            f"{t.get('recall_at_100', float('nan')):.4f} | {t.get('lift_at_100', float('nan')):.2f} |"
        )
    lines.extend(
        [
            "",
            "## Primary deltas (D − B)",
            "",
            f"- ΔAUPRC: **{d_primary.get('auprc', float('nan')):+.4f}**",
            f"- ΔF1: {d_primary.get('f1_at_selected_threshold', float('nan')):+.4f}",
            f"- ΔP@100: {d_primary.get('precision_at_100', float('nan')):+.4f}",
            f"- ΔR@100: {d_primary.get('recall_at_100', float('nan')):+.4f}",
            f"- Δlift@100: {d_primary.get('lift_at_100', float('nan')):+.2f}",
            "",
            "Conservative read: single checkpoint; downstream probe only.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--data_config", default="data_config.json")
    p.add_argument("--run-name", required=True)
    p.add_argument("--embedding_dir", required=True, help="pre_embedding_3h directory")
    p.add_argument(
        "--temporal_flow_cache_dir",
        required=True,
        help="Dataset cache from build_temporal_flow_causal_cache.py",
    )
    p.add_argument("--output_json", required=True)
    p.add_argument("--output_md", required=True)
    p.add_argument("--class_weight", default="model", choices=["balanced", "none", "model", "explicit"])
    p.add_argument("--class_weight_pos", type=float, default=None)
    p.add_argument("--model", default="gin", help="Model key for --class_weight model (w_ce1/w_ce2 lookup).")
    p.add_argument("--probe_C", type=float, default=1.0)
    p.add_argument("--probe_max_iter", type=int, default=1000)
    p.add_argument(
        "--max_iter",
        type=int,
        default=None,
        help="Alias for --probe_max_iter (backward-compatible override).",
    )
    p.add_argument("--probe_n_jobs", type=int, default=16)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--categorical_encoding", default="ordinal")
    p.add_argument("--min_pairing_coverage", type=float, default=0.999)
    p.add_argument(
        "--protocol_source",
        default="compare_representation_source.py / pre3h strong-run probes (cw=model, C=1.0, seed=1)",
    )
    p.add_argument(
        "--arms",
        default=None,
        help=f"Comma-separated subset of arms to run (default: all). Choices: {','.join(ARMS)}",
    )
    p.add_argument(
        "--shuffle_temporal_features_within_split",
        action="store_true",
        help="Permute temporal_flow rows independently within train/val/test (control).",
    )
    p.add_argument("--shuffle_seed", type=int, default=1)
    p.add_argument(
        "--diagnostic_tag",
        default="temporal_flow_causal_ablation",
        help="Label stored in output JSON diagnostic field.",
    )
    p.add_argument(
        "--representation_source",
        default="pre_embedding_3h",
        help="Representation label stored in output JSON (e.g. pre_embedding_3h, post_embedding_128).",
    )
    p.add_argument("--testing", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_iter is not None:
        args.probe_max_iter = int(args.max_iter)
    logger_setup()
    set_seed(args.seed)
    payload = run_ablation(args)
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    write_markdown(Path(args.output_md), payload)
    logging.info("Wrote %s and %s", out_json, args.output_md)


if __name__ == "__main__":
    main()
