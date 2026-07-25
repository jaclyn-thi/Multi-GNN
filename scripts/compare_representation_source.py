#!/usr/bin/env python3
"""Paired linear-probe comparison: pre_embedding_3h vs post_embedding_128.

Both representations come from the SAME frozen contrastively trained checkpoint. This script
runs the identical frozen linear-probe pipeline on each and reports a paired metric table.

Pairing is enforced by an inner-join on ``edge_id`` per split, so both probes are fit and
evaluated on exactly the same transactions, in the same order, with the same labels, split
assignment, class weights, regularization, threshold-selection procedure, and seed. Only the
representation (128-d embedding_head output vs 3*n_hidden pre-embedding) differs.

Primary comparison is embedding-only. An optional ``--with_raw`` secondary comparison appends
the same train-fit raw edge features to both representations.

Example:
  python scripts/compare_representation_source.py \\
    --data Small-HI \\
    --post_dir embeddings/gin_emlps_tds_asym_proj_8192neg_queue0_40ep_seed1 \\
    --pre_dir embeddings/gin_emlps_tds_asym_proj_8192neg_queue0_40ep_seed1/pre_embedding_3h \\
    --run_name gin_emlps_tds_40ep_seed1 \\
    --class_weight model --model gin --seed 1 --with_raw \\
    --output_json results/diagnostics/pre_embedding_3h_vs_post_embedding_small_hi.json \\
    --output_md notes/pre_embedding_3h_vs_post_embedding_small_hi.md
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

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
from util import logger_setup, set_seed
from ranking_metrics import ALERT_BUDGET_KS, ranking_metrics

SPLITS = ("train", "val", "test")
REPRESENTATIONS = ("post_embedding_128", "pre_embedding_3h")


def _load_split(embedding_dir: Path, split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return load_embedding_npz(embedding_dir / f"{split}.npz")


def _align_by_edge_id(
    post: Tuple[np.ndarray, np.ndarray, np.ndarray],
    pre: Tuple[np.ndarray, np.ndarray, np.ndarray],
    split: str,
) -> Dict[str, Any]:
    """Inner-join post/pre on edge_id (ascending), verifying labels agree.

    Returns aligned arrays plus coverage diagnostics so the pairing is auditable.
    """
    zp, yp, ep = post
    zq, yq, eq = pre
    common = np.intersect1d(ep, eq)
    post_pos = {int(e): i for i, e in enumerate(ep)}
    pre_pos = {int(e): i for i, e in enumerate(eq)}
    idx_p = np.array([post_pos[int(e)] for e in common], dtype=np.int64)
    idx_q = np.array([pre_pos[int(e)] for e in common], dtype=np.int64)
    yp_c = yp[idx_p].astype(np.int64)
    yq_c = yq[idx_q].astype(np.int64)
    if not np.array_equal(yp_c, yq_c):
        raise ValueError(f"{split}: labels disagree between post/pre for joined edge_ids")
    return {
        "z_post": zp[idx_p].astype(np.float32),
        "z_pre": zq[idx_q].astype(np.float32),
        "y": yp_c,
        "edge_id": common.astype(np.int64),
        "coverage": {
            "post_rows": int(ep.shape[0]),
            "pre_rows": int(eq.shape[0]),
            "joined_rows": int(common.shape[0]),
            "post_only": int(ep.shape[0] - common.shape[0]),
            "pre_only": int(eq.shape[0] - common.shape[0]),
            "joined_fraction_of_post": float(common.shape[0] / ep.shape[0]) if ep.shape[0] else float("nan"),
            "positives": int(yp_c.sum()),
            "positive_rate": float(yp_c.mean()) if yp_c.shape[0] else float("nan"),
        },
    }


def _alert_budget_metrics(clf, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    proba = clf.predict_proba(x)[:, 1]
    return ranking_metrics(y, proba)


def _build_group_features(
    data: str,
    data_config: str,
    edge_ids_by_split: Dict[str, np.ndarray],
    groups: Tuple[str, ...],
) -> Dict[str, np.ndarray]:
    """Train-fit engineered edge features for ``groups`` aligned to joined edge_ids per split.

    Reuses probe_feature_ablation's feature builder + train-split GroupwiseScaler so the
    +raw / +raw+morph secondaries use the identical feature protocol as the established probe
    sweeps. ``groups`` is e.g. RAW_GROUPS or RAW_GROUPS + MORPH_GROUPS.
    """
    from scripts.probe_feature_ablation import (  # noqa: WPS433 (local import; heavy deps)
        GroupwiseScaler,
        build_full_feature_matrix,
        load_dataset_frames,
    )

    df, df_train, tr_np, _, _, _ = load_dataset_frames(data, data_config)
    x_grp, _, group_slices, _ = build_full_feature_matrix(
        df, df_train, groups, categorical_encoding="ordinal"
    )
    scaler = GroupwiseScaler(group_slices=group_slices)
    scaler.fit(x_grp[tr_np])
    x_full = scaler.transform(x_grp)  # indexed by global edge_id
    return {split: x_full[edge_ids] for split, edge_ids in edge_ids_by_split.items()}


def _probe_one_representation(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    class_weight: Any,
    seed: int,
    max_iter: int,
    probe_c: float,
    n_jobs: int,
) -> Dict[str, Any]:
    clf = fit_logistic_probe(
        x_train, y_train, class_weight=class_weight,
        max_iter=max_iter, seed=seed, n_jobs=n_jobs, C=probe_c,
    )
    val_proba = clf.predict_proba(x_val)[:, 1]
    selected_threshold, val_f1_at_selection = tune_threshold_max_f1(y_val, val_proba)

    test_sel = evaluate_probe(clf, x_test, y_test, "test", threshold=selected_threshold)
    test_05 = evaluate_probe(clf, x_test, y_test, "test", threshold=0.5)
    val_sel = evaluate_probe(clf, x_val, y_val, "val", threshold=selected_threshold)

    result: Dict[str, Any] = {
        "feature_dim": int(x_train.shape[1]),
        "selected_threshold": float(selected_threshold),
        "val_f1_at_selected_threshold": float(val_f1_at_selection),
        "test": {
            "auroc": test_sel["auroc"],
            "auprc": test_sel["auprc"],
            "f1_at_selected_threshold": test_sel["f1"],
            "precision_at_selected_threshold": test_sel["precision"],
            "recall_at_selected_threshold": test_sel["recall"],
            "f1_at_threshold_0.5": test_05["f1"],
            "precision_at_threshold_0.5": test_05["precision"],
            "recall_at_threshold_0.5": test_05["recall"],
            "n": test_sel["n"],
            "positive_rate": test_sel["positive_rate"],
        },
        "val": {
            "auroc": val_sel["auroc"],
            "auprc": val_sel["auprc"],
            "f1_at_selected_threshold": val_sel["f1"],
        },
    }
    result["test"].update(_alert_budget_metrics(clf, x_test, y_test))
    return result


def _winner(post_val: float, pre_val: float, higher_is_better: bool = True) -> str:
    if np.isnan(post_val) and np.isnan(pre_val):
        return "n/a"
    if np.isnan(post_val):
        return "pre_embedding_3h"
    if np.isnan(pre_val):
        return "post_embedding_128"
    if abs(post_val - pre_val) < 1e-9:
        return "tie"
    post_better = post_val > pre_val if higher_is_better else post_val < pre_val
    return "post_embedding_128" if post_better else "pre_embedding_3h"


def run_comparison(args) -> Dict[str, Any]:
    post_dir = Path(args.post_dir)
    pre_dir = Path(args.pre_dir)
    for d in (post_dir, pre_dir):
        for split in SPLITS:
            if not (d / f"{split}.npz").is_file():
                raise FileNotFoundError(f"Missing {d / f'{split}.npz'}")

    class_weight = resolve_class_weight(args)

    aligned: Dict[str, Dict[str, Any]] = {}
    edge_ids_by_split: Dict[str, np.ndarray] = {}
    for split in SPLITS:
        aligned[split] = _align_by_edge_id(
            _load_split(post_dir, split), _load_split(pre_dir, split), split
        )
        edge_ids_by_split[split] = aligned[split]["edge_id"]
        logging.info("split=%s pairing: %s", split, aligned[split]["coverage"])

    def _feature_comparison(feature_mode: str, raw_by_split: Optional[Dict[str, np.ndarray]]) -> Dict[str, Any]:
        def _matrix(split: str, key: str) -> np.ndarray:
            z = aligned[split][key]
            if raw_by_split is None:
                return z
            return np.concatenate([z, raw_by_split[split]], axis=1).astype(np.float32)

        reps: Dict[str, Any] = {}
        for rep_key, z_key in (("post_embedding_128", "z_post"), ("pre_embedding_3h", "z_pre")):
            reps[rep_key] = _probe_one_representation(
                x_train=_matrix("train", z_key),
                y_train=aligned["train"]["y"],
                x_val=_matrix("val", z_key),
                y_val=aligned["val"]["y"],
                x_test=_matrix("test", z_key),
                y_test=aligned["test"]["y"],
                class_weight=class_weight,
                seed=int(args.seed),
                max_iter=int(args.probe_max_iter),
                probe_c=float(args.probe_C),
                n_jobs=int(args.probe_n_jobs),
            )
            t = reps[rep_key]["test"]
            logging.info(
                "[%s] %s: dim=%d AUROC=%.4f AUPRC=%.4f F1=%.4f P=%.4f R=%.4f",
                feature_mode, rep_key, reps[rep_key]["feature_dim"],
                t["auroc"], t["auprc"], t["f1_at_selected_threshold"],
                t["precision_at_selected_threshold"], t["recall_at_selected_threshold"],
            )

        post_t = reps["post_embedding_128"]["test"]
        pre_t = reps["pre_embedding_3h"]["test"]
        winners = {
            "auprc": _winner(post_t["auprc"], pre_t["auprc"]),
            "auroc": _winner(post_t["auroc"], pre_t["auroc"]),
            "f1_at_selected_threshold": _winner(
                post_t["f1_at_selected_threshold"], pre_t["f1_at_selected_threshold"]
            ),
            "recall_at_100": _winner(post_t.get("recall_at_100", float("nan")), pre_t.get("recall_at_100", float("nan"))),
            "recall_at_500": _winner(post_t.get("recall_at_500", float("nan")), pre_t.get("recall_at_500", float("nan"))),
            "recall_at_1000": _winner(post_t.get("recall_at_1000", float("nan")), pre_t.get("recall_at_1000", float("nan"))),
        }
        deltas = {
            "auprc_pre_minus_post": float(pre_t["auprc"] - post_t["auprc"]),
            "auroc_pre_minus_post": float(pre_t["auroc"] - post_t["auroc"]),
            "f1_pre_minus_post": float(
                pre_t["f1_at_selected_threshold"] - post_t["f1_at_selected_threshold"]
            ),
        }
        return {"feature_mode": feature_mode, "representations": reps, "winners": winners, "deltas": deltas}

    from scripts.probe_feature_ablation import MORPH_GROUPS, RAW_GROUPS  # noqa: WPS433

    comparisons: Dict[str, Any] = {"embedding_only": _feature_comparison("embedding_only", None)}
    if args.with_raw or args.with_morph:
        raw_by_split = _build_group_features(
            args.data, args.data_config, edge_ids_by_split, RAW_GROUPS
        )
        comparisons["embedding_plus_raw"] = _feature_comparison("embedding_plus_raw", raw_by_split)
    if args.with_morph:
        raw_morph_by_split = _build_group_features(
            args.data, args.data_config, edge_ids_by_split, RAW_GROUPS + MORPH_GROUPS
        )
        comparisons["embedding_plus_raw_morph"] = _feature_comparison(
            "embedding_plus_raw_morph", raw_morph_by_split
        )

    payload: Dict[str, Any] = {
        "diagnostic": "pre_embedding_3h_vs_post_embedding_128",
        "no_ssl_retraining": True,
        "paired": True,
        "pairing": "inner-join on edge_id per split; identical rows/labels/order for both representations",
        "data": args.data,
        "run_name": args.run_name,
        "post_embedding_dir": str(post_dir),
        "pre_embedding_dir": str(pre_dir),
        "representation_dims": {
            "post_embedding_128": int(aligned["train"]["z_post"].shape[1]),
            "pre_embedding_3h": int(aligned["train"]["z_pre"].shape[1]),
        },
        "probe": {
            "impl": "sklearn LogisticRegression (lbfgs)",
            "class_weight_mode": str(args.class_weight),
            "class_weight": serialize_class_weight(class_weight),
            "probe_C": float(args.probe_C),
            "probe_max_iter": int(args.probe_max_iter),
            "seed": int(args.seed),
            "threshold_tuning": "max_f1_on_val",
            "alert_budget_ks": list(ALERT_BUDGET_KS),
        },
        "split_pairing": {split: aligned[split]["coverage"] for split in SPLITS},
        "comparisons": comparisons,
    }

    # Attach extraction metadata for provenance if available.
    for tag, d in (("post", post_dir), ("pre", pre_dir)):
        meta_path = d / "meta.json"
        if meta_path.is_file():
            with meta_path.open("r", encoding="utf-8") as f:
                payload.setdefault("extraction_meta", {})[tag] = json.load(f)
    return payload


def write_markdown(path: Path, payload: Dict[str, Any]) -> None:
    dims = payload["representation_dims"]
    lines = [
        f"# pre_embedding_3h vs post_embedding_128 — {payload['data']} ({payload['run_name']})",
        "",
        f"- **checkpoint dirs:** post=`{payload['post_embedding_dir']}`, pre=`{payload['pre_embedding_dir']}`",
        f"- **representation dims:** post_embedding_128 = {dims['post_embedding_128']}, "
        f"pre_embedding_3h = {dims['pre_embedding_3h']}",
        f"- **no SSL retraining:** {payload['no_ssl_retraining']}  |  **paired:** {payload['paired']} "
        f"({payload['pairing']})",
        f"- **probe:** {payload['probe']['impl']}, class_weight={payload['probe']['class_weight_mode']}"
        f"={payload['probe']['class_weight']}, C={payload['probe']['probe_C']}, "
        f"threshold={payload['probe']['threshold_tuning']}, seed={payload['probe']['seed']}",
        "",
    ]
    for mode, comp in payload["comparisons"].items():
        lines.append(f"## {mode}")
        lines.append("")
        lines.append(
            "| metric (test) | post_embedding_128 | pre_embedding_3h | winner |"
        )
        lines.append("|---|---|---|---|")
        post = comp["representations"]["post_embedding_128"]["test"]
        pre = comp["representations"]["pre_embedding_3h"]["test"]
        rows = [
            ("AUROC", "auroc", "auroc"),
            ("AUPRC", "auprc", "auprc"),
            ("F1 @ val-thr", "f1_at_selected_threshold", "f1_at_selected_threshold"),
            ("precision @ val-thr", "precision_at_selected_threshold", None),
            ("recall @ val-thr", "recall_at_selected_threshold", None),
            ("F1 @ 0.5", "f1_at_threshold_0.5", None),
            ("recall@100", "recall_at_100", "recall_at_100"),
            ("recall@500", "recall_at_500", "recall_at_500"),
            ("recall@1000", "recall_at_1000", "recall_at_1000"),
            ("precision@100", "precision_at_100", None),
            ("precision@500", "precision_at_500", None),
            ("precision@1000", "precision_at_1000", None),
            ("lift@100", "lift_at_100", None),
            ("lift@1000", "lift_at_1000", None),
        ]
        winners = comp["winners"]
        for label, key, win_key in rows:
            pv = post.get(key, float("nan"))
            qv = pre.get(key, float("nan"))
            w = winners.get(win_key, "") if win_key else ""
            lines.append(f"| {label} | {pv:.4f} | {qv:.4f} | {w} |")
        sel_post = comp["representations"]["post_embedding_128"]["selected_threshold"]
        sel_pre = comp["representations"]["pre_embedding_3h"]["selected_threshold"]
        lines.append(
            f"| selected val threshold | {sel_post:.4f} | {sel_pre:.4f} |  |"
        )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="Small-HI")
    p.add_argument("--data_config", default="data_config.json")
    p.add_argument("--post_dir", required=True, help="Directory with post_embedding (128-d) train/val/test.npz")
    p.add_argument("--pre_dir", required=True, help="Directory with pre_embedding_3h train/val/test.npz")
    p.add_argument("--run_name", required=True, help="Human-readable run label for reports.")
    p.add_argument("--model", default="gin")
    p.add_argument("--class_weight", default="model", choices=["balanced", "none", "model", "explicit"])
    p.add_argument("--class_weight_pos", type=float, default=None)
    p.add_argument("--probe_C", type=float, default=1.0)
    p.add_argument("--probe_max_iter", type=int, default=1000)
    p.add_argument("--probe_n_jobs", type=int, default=-1)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--with_raw", action="store_true", help="Add secondary embedding+raw comparison.")
    p.add_argument(
        "--with_morph",
        action="store_true",
        help="Add embedding+raw+morph comparison (implies +raw; morph = degree_fan, flow_balance, "
        "temporal_behavior groups).",
    )
    p.add_argument("--output_json", required=True)
    p.add_argument("--output_md", required=True)
    p.add_argument("--testing", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logger_setup()
    set_seed(args.seed)

    payload = run_comparison(args)

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logging.info("Wrote %s", out_json)

    write_markdown(Path(args.output_md), payload)
    logging.info("Wrote %s", args.output_md)


if __name__ == "__main__":
    main()
