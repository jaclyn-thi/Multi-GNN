#!/usr/bin/env python3
"""Label-scarcity probe on frozen pre-3h ± raw ± temporal_flow_causal features.

Subsample only the labeled training set for the logistic probe; val/test unchanged.
No SSL/GNN retraining and no embedding regeneration.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from linear_probe import (  # noqa: E402
    evaluate_probe,
    fit_logistic_probe,
    resolve_class_weight,
    serialize_class_weight,
    tune_threshold_max_f1,
)
from scripts.probe_temporal_flow_ablation import (  # noqa: E402
    ARMS,
    _alert_budget_metrics,
    _align_split,
    _arm_feature_groups,
    _build_arm_matrix,
    _convergence_info,
    _load_splits,
    _load_temporal_flow_cache,
)
from util import logger_setup, set_seed  # noqa: E402

DEFAULT_FRACTIONS = (0.01, 0.05, 0.10, 0.25, 0.50, 1.0)
PRIMARY_ARMS = ("B_embedding_raw", "D_embedding_raw_temporal_flow")
SPLITS = ("train", "val", "test")


def stratified_train_indices(
    y: np.ndarray,
    fraction: float,
    scarcity_seed: int,
    *,
    max_retries: int = 8,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Stratified subsample of train indices; retries if zero positives."""
    y = np.asarray(y).reshape(-1).astype(np.int64)
    n = int(y.shape[0])
    fraction = float(fraction)
    if fraction <= 0 or fraction > 1:
        raise ValueError(f"label fraction must be in (0, 1], got {fraction}")
    if fraction >= 1.0 - 1e-12:
        idx = np.arange(n, dtype=np.int64)
        return idx, {
            "fraction": 1.0,
            "n_labeled": n,
            "n_positive_labeled": int(y.sum()),
            "scarcity_seed_used": int(scarcity_seed),
            "resample_attempts": 0,
            "status": "full_train",
        }

    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    if pos.size == 0:
        raise RuntimeError("Training split has zero positives; cannot run scarcity probe.")

    last_err = "unknown"
    for attempt in range(int(max_retries)):
        seed = int(scarcity_seed) + attempt
        rng = np.random.default_rng(seed)
        n_pos = max(1, int(round(pos.size * fraction)))
        n_neg = max(0, int(round(neg.size * fraction)))
        n_pos = min(n_pos, pos.size)
        n_neg = min(n_neg, neg.size)
        if n_pos < 1:
            last_err = "rounded positive count was zero"
            continue
        chosen_pos = rng.choice(pos, size=n_pos, replace=False)
        chosen_neg = rng.choice(neg, size=n_neg, replace=False) if n_neg else np.array([], dtype=np.int64)
        idx = np.sort(np.concatenate([chosen_pos, chosen_neg]).astype(np.int64))
        n_pos_lab = int(y[idx].sum())
        if n_pos_lab < 1:
            last_err = "sampled subset had zero positives"
            continue
        return idx, {
            "fraction": fraction,
            "n_labeled": int(idx.shape[0]),
            "n_positive_labeled": n_pos_lab,
            "n_negative_labeled": int(idx.shape[0] - n_pos_lab),
            "scarcity_seed_used": seed,
            "resample_attempts": attempt,
            "status": "ok",
        }
    raise RuntimeError(
        f"Failed to sample fraction={fraction} with >=1 positive after {max_retries} attempts "
        f"(scarcity_seed={scarcity_seed}): {last_err}"
    )


def _probe_arm_with_train_idx(
    aligned: Dict[str, Dict[str, Any]],
    arm: str,
    train_idx: np.ndarray,
    *,
    class_weight: Any,
    seed: int,
    max_iter: int,
    probe_c: float,
    n_jobs: int,
) -> Dict[str, Any]:
    x_train_full = _build_arm_matrix(aligned, "train", arm)
    y_train_full = aligned["train"]["y"]
    x_train = x_train_full[train_idx]
    y_train = y_train_full[train_idx]
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

    result: Dict[str, Any] = {
        "arm": arm,
        "feature_groups": _arm_feature_groups(arm),
        "feature_dim": int(x_train.shape[1]),
        "selected_threshold": float(selected_threshold),
        "val_f1_at_selected_threshold": float(val_f1_at_selection),
        "convergence": _convergence_info(clf, max_iter),
        "train_subset": {
            "n_labeled": int(train_idx.shape[0]),
            "n_positive_labeled": int(y_train.sum()),
            "positive_rate_labeled": float(y_train.mean()) if train_idx.size else float("nan"),
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


def _parse_fractions(raw: str) -> List[float]:
    vals = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    if not vals:
        raise ValueError("empty --label_fractions")
    for v in vals:
        if v <= 0 or v > 1:
            raise ValueError(f"invalid fraction {v}")
    return vals


def _resolve_arms(raw: Optional[str]) -> List[str]:
    if not raw:
        return list(PRIMARY_ARMS)
    selected = [a.strip() for a in str(raw).split(",") if a.strip()]
    for arm in selected:
        if arm not in ARMS:
            raise ValueError(f"Unknown arm {arm!r}; expected one of {ARMS}")
    return selected


def load_aligned(args) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
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

    df, df_train, tr_np, _, _, _spec = load_dataset_frames(args.data, args.data_config)
    x_raw_full, _, raw_slices, raw_meta = build_full_feature_matrix(
        df, df_train, RAW_GROUPS, categorical_encoding=args.categorical_encoding
    )
    raw_scaler = GroupwiseScaler(group_slices=raw_slices)
    raw_scaler.fit(x_raw_full[tr_np])
    x_raw_scaled = raw_scaler.transform(x_raw_full)

    from sklearn.preprocessing import StandardScaler

    tf_scaler = StandardScaler()
    tf_scaler.fit(tf_full[tr_np])
    tf_scaled = tf_scaler.transform(tf_full).astype(np.float32)

    aligned: Dict[str, Dict[str, Any]] = {}
    for split in SPLITS:
        z, y, edge_id = splits[split]
        aligned[split] = _align_split(
            z, y, edge_id, x_raw_scaled[edge_id], tf_scaled[edge_id], split
        )
    return aligned, tf_meta, raw_meta


def run_scarcity(args) -> Dict[str, Any]:
    fractions = _parse_fractions(args.label_fractions)
    arms_to_run = _resolve_arms(args.arms)
    aligned, tf_meta, raw_meta = load_aligned(args)
    class_weight = resolve_class_weight(args)

    fraction_results: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    for frac in fractions:
        try:
            train_idx, sample_meta = stratified_train_indices(
                aligned["train"]["y"],
                frac,
                int(args.scarcity_seed),
            )
        except RuntimeError as exc:
            logging.warning("Skipping fraction=%.4f: %s", frac, exc)
            skipped.append({"fraction": frac, "reason": str(exc)})
            continue

        logging.info(
            "fraction=%.4f n_labeled=%d n_pos=%d seed_used=%s",
            frac,
            sample_meta["n_labeled"],
            sample_meta["n_positive_labeled"],
            sample_meta["scarcity_seed_used"],
        )
        arms: Dict[str, Any] = {}
        for arm in arms_to_run:
            arms[arm] = _probe_arm_with_train_idx(
                aligned,
                arm,
                train_idx,
                class_weight=class_weight,
                seed=int(args.seed),
                max_iter=int(args.probe_max_iter),
                probe_c=float(args.probe_C),
                n_jobs=int(args.probe_n_jobs),
            )
            t = arms[arm]["test"]
            logging.info(
                "  %s AUPRC=%.4f F1=%.4f P@100=%.4f conv=%s n_iter=%s",
                arm,
                t["auprc"],
                t["f1_at_selected_threshold"],
                t.get("precision_at_100", float("nan")),
                arms[arm]["convergence"].get("status"),
                arms[arm]["convergence"].get("n_iter"),
            )

        deltas = {}
        if "B_embedding_raw" in arms and "D_embedding_raw_temporal_flow" in arms:
            tb = arms["B_embedding_raw"]["test"]
            td = arms["D_embedding_raw_temporal_flow"]["test"]
            for key in (
                "auroc",
                "auprc",
                "f1_at_selected_threshold",
                "precision_at_100",
                "recall_at_100",
                "lift_at_100",
            ):
                deltas[f"D_minus_B_{key}"] = float(td.get(key, float("nan")) - tb.get(key, float("nan")))

        fraction_results.append({
            "fraction": frac,
            "sample": sample_meta,
            "arms": arms,
            "deltas": deltas,
        })

    max_iter = int(args.probe_max_iter if args.max_iter is None else args.max_iter)
    payload: Dict[str, Any] = {
        "diagnostic": "label_scarcity_temporal_flow_probe",
        "no_ssl_retraining": True,
        "no_embedding_regeneration": True,
        "data": args.data,
        "run_name": args.run_name,
        "embedding_dir": str(args.embedding_dir),
        "temporal_flow_cache_dir": str(args.temporal_flow_cache_dir),
        "representation": "pre_embedding_3h",
        "representation_dim": int(aligned["train"]["z"].shape[1]),
        "scarcity_seed": int(args.scarcity_seed),
        "label_fractions": fractions,
        "arms_run": arms_to_run,
        "probe": {
            "impl": "sklearn LogisticRegression (lbfgs)",
            "class_weight_mode": str(args.class_weight),
            "class_weight": serialize_class_weight(class_weight),
            "probe_C": float(args.probe_C),
            "probe_max_iter": max_iter,
            "seed": int(args.seed),
            "threshold_tuning": "max_f1_on_val",
            "note": "Only train labels subsampled; val/test unchanged.",
        },
        "split_pairing": {s: aligned[s]["coverage"] for s in SPLITS},
        "fraction_results": fraction_results,
        "skipped_fractions": skipped,
        "temporal_flow_cache_meta": {
            "cache_version": tf_meta.get("cache_version"),
            "causal_history_policy": tf_meta.get("causal_history_policy"),
        },
        "raw_feature_meta": raw_meta,
        "primary_comparison": "D_embedding_raw_temporal_flow vs B_embedding_raw",
    }
    return payload


def write_markdown(path: Path, payload: Dict[str, Any]) -> None:
    lines = [
        f"# Label-scarcity temporal-flow probe — {payload['data']} (`{payload['run_name']}`)",
        "",
        f"- scarcity_seed={payload['scarcity_seed']}",
        f"- embedding: `{payload['embedding_dir']}`",
        f"- primary: pre-3h+raw+temporal-flow vs pre-3h+raw",
        f"- no SSL retraining / no embedding regeneration",
        "",
        "| frac | n_lab | n_pos | B AUPRC | D AUPRC | ΔAUPRC | D F1 | D P@100 |",
        "|-----:|------:|------:|-------:|-------:|-------:|-----:|--------:|",
    ]
    for fr in payload.get("fraction_results") or []:
        b = (fr.get("arms") or {}).get("B_embedding_raw", {}).get("test", {})
        d = (fr.get("arms") or {}).get("D_embedding_raw_temporal_flow", {}).get("test", {})
        sample = fr.get("sample") or {}
        delta = (fr.get("deltas") or {}).get("D_minus_B_auprc", float("nan"))
        lines.append(
            "| {frac:.2f} | {n_lab} | {n_pos} | {b:.4f} | {d:.4f} | {delta:+.4f} | {f1:.4f} | {p100:.4f} |".format(
                frac=float(fr["fraction"]),
                n_lab=sample.get("n_labeled", "—"),
                n_pos=sample.get("n_positive_labeled", "—"),
                b=float(b.get("auprc", float("nan"))),
                d=float(d.get("auprc", float("nan"))),
                delta=float(delta),
                f1=float(d.get("f1_at_selected_threshold", float("nan"))),
                p100=float(d.get("precision_at_100", float("nan"))),
            )
        )
    if payload.get("skipped_fractions"):
        lines.extend(["", "## Skipped fractions", ""])
        for sk in payload["skipped_fractions"]:
            lines.append(f"- {sk.get('fraction')}: {sk.get('reason')}")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--data_config", default="data_config.json")
    p.add_argument("--run-name", required=True)
    p.add_argument("--embedding_dir", required=True)
    p.add_argument("--temporal_flow_cache_dir", required=True)
    p.add_argument("--output_json", required=True)
    p.add_argument("--output_md", required=True)
    p.add_argument(
        "--label_fractions",
        default=",".join(str(x) for x in DEFAULT_FRACTIONS),
        help="Comma-separated train label fractions in (0,1].",
    )
    p.add_argument("--scarcity_seed", type=int, default=1)
    p.add_argument("--arms", default=",".join(PRIMARY_ARMS))
    p.add_argument("--class_weight", default="model", choices=["balanced", "none", "model", "explicit"])
    p.add_argument("--class_weight_pos", type=float, default=None)
    p.add_argument("--model", default="gin")
    p.add_argument("--probe_C", type=float, default=1.0)
    p.add_argument("--probe_max_iter", type=int, default=5000)
    p.add_argument("--max_iter", type=int, default=None, help="Alias for --probe_max_iter.")
    p.add_argument("--probe_n_jobs", type=int, default=16)
    p.add_argument("--seed", type=int, default=1, help="Probe/sklearn seed (not scarcity seed).")
    p.add_argument("--categorical_encoding", default="ordinal")
    p.add_argument("--testing", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.max_iter is not None:
        args.probe_max_iter = int(args.max_iter)
    logger_setup()
    set_seed(args.seed)
    payload = run_scarcity(args)
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    write_markdown(Path(args.output_md), payload)
    logging.info("Wrote %s and %s", out_json, args.output_md)


if __name__ == "__main__":
    main()
