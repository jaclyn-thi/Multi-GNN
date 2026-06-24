#!/usr/bin/env python3
"""Audit transaction feature-KNN neighborhoods across baseline and richer_v1 sets."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from transaction_knn.backends import SklearnKNNBackend
from transaction_knn.features import (
    build_features_detailed,
    feature_set_metadata,
    load_train_frame,
    standardize_features,
)


def _endpoint_overlap(df, anchor_idx: np.ndarray, neighbor_idx: np.ndarray) -> Dict[str, float]:
    from_ids = df["from_id"].to_numpy()
    to_ids = df["to_id"].to_numpy()
    a_src = from_ids[anchor_idx]
    a_dst = to_ids[anchor_idx]
    n_src = from_ids[neighbor_idx]
    n_dst = to_ids[neighbor_idx]
    return {
        "same_sender": float((a_src == n_src).mean()),
        "same_receiver": float((a_dst == n_dst).mean()),
        "same_pair": float(((a_src == n_src) & (a_dst == n_dst)).mean()),
    }


def _neighbor_diversity(neighbor_ids: np.ndarray) -> Dict[str, float]:
    flat = neighbor_ids.reshape(-1)
    flat = flat[flat >= 0]
    if flat.size == 0:
        return {
            "unique_neighbor_fraction": float("nan"),
            "duplicate_neighbor_rows": 0.0,
            "hubness_top1_neighbor_share": float("nan"),
            "hubness_top10_neighbors_share": float("nan"),
            "hubness_unique_neighbors": 0.0,
        }
    counts = Counter(int(x) for x in flat)
    total = float(flat.size)
    unique_frac = float(len(counts)) / total
    row_dup = float(sum(len(row) != len(np.unique(row[row >= 0])) for row in neighbor_ids))
    top_counts = sorted(counts.values(), reverse=True)
    top1_share = float(top_counts[0]) / total if top_counts else float("nan")
    top10_share = float(sum(top_counts[:10])) / total if top_counts else float("nan")
    return {
        "unique_neighbor_fraction": unique_frac,
        "duplicate_neighbor_rows": row_dup,
        "hubness_top1_neighbor_share": top1_share,
        "hubness_top10_neighbors_share": top10_share,
        "hubness_unique_neighbors": float(len(counts)),
    }


def _sim_stats(sims: np.ndarray) -> Dict[str, float]:
    vals = sims[np.isfinite(sims)]
    if vals.size == 0:
        return {k: float("nan") for k in [
            "sim_min", "sim_mean", "sim_p50", "sim_p75", "sim_p90", "sim_p95", "sim_p99", "sim_max"
        ]}
    return {
        "sim_min": float(np.min(vals)),
        "sim_mean": float(np.mean(vals)),
        "sim_p50": float(np.percentile(vals, 50)),
        "sim_p75": float(np.percentile(vals, 75)),
        "sim_p90": float(np.percentile(vals, 90)),
        "sim_p95": float(np.percentile(vals, 95)),
        "sim_p99": float(np.percentile(vals, 99)),
        "sim_max": float(np.max(vals)),
    }


def _finite_checks(features: np.ndarray) -> Dict[str, float]:
    return {
        "nan_count": float(np.isnan(features).sum()),
        "inf_count": float(np.isinf(features).sum()),
        "finite_fraction": float(np.isfinite(features).mean()),
    }


def _pairwise_neighbor_overlap(ids_a: np.ndarray, ids_b: np.ndarray) -> float:
    overlap = []
    for row_a, row_b in zip(ids_a, ids_b):
        sa = {int(x) for x in row_a if int(x) >= 0}
        sb = {int(x) for x in row_b if int(x) >= 0}
        if not sa and not sb:
            overlap.append(1.0)
        elif not sa or not sb:
            overlap.append(0.0)
        else:
            overlap.append(len(sa & sb) / len(sa | sb))
    return float(np.mean(overlap))


def _report_key(feature_set: str, categorical_encoding: str, scaling: str) -> str:
    return f"{feature_set}|{categorical_encoding}|{scaling}"


def audit_feature_set(
    df_train,
    *,
    feature_set: str,
    k: int,
    categorical_encoding: str,
    scaling: str,
    label_col: str,
    include_pair_history: Optional[bool],
) -> Tuple[Dict[str, object], np.ndarray]:
    detail = build_features_detailed(
        df_train,
        feature_set,
        categorical_encoding=categorical_encoding,
        include_pair_history=include_pair_history,
        scaling=scaling if scaling != "legacy_standard" else "none",
    )
    if scaling == "legacy_standard":
        x = standardize_features(detail.features)
    elif scaling in {"standard", "robust"}:
        x = detail.features
    else:
        raise ValueError(f"Unsupported scaling={scaling!r}")

    backend = SklearnKNNBackend(metric="cosine")
    backend.fit(x, k=k)
    query_idx = np.arange(x.shape[0], dtype=np.int64)
    nbr_ids, nbr_sims = backend.query(query_idx, k)

    meta = feature_set_metadata(detail)
    meta["scaling"] = scaling
    report: Dict[str, object] = {
        "feature_set": detail.feature_set,
        "categorical_encoding": categorical_encoding,
        "feature_names": detail.names,
        "feature_groups": detail.groups,
        "group_dims": detail.group_dims,
        "group_weights": detail.group_weights,
        "n_features": int(detail.features.shape[1]),
        "n_rows": int(detail.features.shape[0]),
        "k": int(k),
        "amount_column": detail.amount_column,
        **_finite_checks(x),
        **_sim_stats(nbr_sims),
        **_neighbor_diversity(nbr_ids),
        **meta,
    }
    flat_a = np.repeat(query_idx, k)
    flat_n = nbr_ids.reshape(-1)
    valid = flat_n >= 0
    report.update(_endpoint_overlap(df_train, flat_a[valid], flat_n[valid]))

    if label_col in df_train.columns:
        labels = df_train[label_col].to_numpy()
        report["label_same_fraction"] = float((labels[flat_a[valid]] == labels[flat_n[valid]]).mean())
        report["neighbor_positive_rate"] = float(labels[flat_n[valid]].mean())
        report["anchor_positive_rate"] = float(labels[flat_a[valid]].mean())
        report["label_enrichment_note"] = "analysis only — not used in training"

    return report, nbr_ids


def _write_markdown(
    path: Path,
    *,
    data: str,
    k: int,
    max_rows: int,
    reports: List[Dict[str, object]],
    jaccard_pairs: List[Dict[str, object]],
    baseline_key: Optional[str],
) -> None:
    lines = [
        "# Transaction KNN feature audit",
        "",
        f"- **Dataset:** {data}",
        f"- **Rows audited:** {max_rows if max_rows > 0 else 'full train'}",
        f"- **k:** {k}",
        "",
        "Label-enrichment fields are **analysis only** and must not be used in training.",
        "",
        "## Per feature set",
        "",
    ]
    for rep in reports:
        key = _report_key(rep["feature_set"], rep["categorical_encoding"], rep["scaling"])
        lines.extend(
            [
                f"### `{rep['feature_set']}` ({rep['categorical_encoding']}, scaling={rep['scaling']})",
                "",
                f"- **Dimensions:** {rep['n_features']} across groups {rep.get('group_dims', {})}",
                f"- **Finite values:** {rep['finite_fraction']:.6f} (nan={rep['nan_count']:.0f}, inf={rep['inf_count']:.0f})",
                f"- **Similarity:** min={rep['sim_min']:.6f}, mean={rep['sim_mean']:.6f}, "
                f"p50={rep['sim_p50']:.6f}, p90={rep['sim_p90']:.6f}, p95={rep['sim_p95']:.6f}, "
                f"p99={rep['sim_p99']:.6f}, max={rep['sim_max']:.6f}",
                f"- **Diversity:** unique_neighbor_fraction={rep['unique_neighbor_fraction']:.6f}, "
                f"hubness_top1={rep['hubness_top1_neighbor_share']:.4f}, "
                f"hubness_top10={rep['hubness_top10_neighbors_share']:.4f}",
                f"- **Endpoint overlap:** same_sender={rep['same_sender']:.4f}, "
                f"same_receiver={rep['same_receiver']:.4f}, same_pair={rep['same_pair']:.4f}",
            ]
        )
        if baseline_key and key != baseline_key:
            for pair in jaccard_pairs:
                if pair["b"] == key and pair["a"] == baseline_key:
                    lines.append(f"- **Jaccard vs baseline:** {pair['mean_jaccard']:.4f}")
        if "label_same_fraction" in rep:
            lines.append(
                f"- **Label enrichment (analysis only):** label_same_fraction={rep['label_same_fraction']:.4f}"
            )
        names = rep["feature_names"]
        preview = ", ".join(str(n) for n in names[:10])
        if len(names) > 10:
            preview += f", ... (+{len(names) - 10} more)"
        lines.extend(["", f"Features ({len(names)}): {preview}", ""])

    lines.extend(["## Pairwise top-k Jaccard overlap", ""])
    if not jaccard_pairs:
        lines.append("_No pairwise comparisons._")
    else:
        lines.append("| A | B | mean Jaccard |")
        lines.append("|---|---|--------------|")
        for pair in jaccard_pairs:
            lines.append(f"| `{pair['a']}` | `{pair['b']}` | {pair['mean_jaccard']:.4f} |")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _default_audit_configs() -> List[Dict[str, object]]:
    return [
        {"feature_set": "edge_native+degree_fan", "categorical_encoding": "ordinal", "scaling": "legacy_standard"},
        {"feature_set": "edge_native+degree_fan", "categorical_encoding": "one_hot", "scaling": "legacy_standard"},
        {"feature_set": "richer_v1", "categorical_encoding": "one_hot", "scaling": "robust", "include_pair_history": True},
        {"feature_set": "richer_v1_no_pair", "categorical_encoding": "one_hot", "scaling": "robust", "include_pair_history": False},
        {"feature_set": "richer_v1", "categorical_encoding": "ordinal", "scaling": "robust", "include_pair_history": True},
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="Small-HI")
    parser.add_argument("--data_config", default="data_config.json")
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--max_rows", type=int, default=50000)
    parser.add_argument(
        "--audit_profile",
        choices=["richer_v1", "legacy"],
        default="richer_v1",
        help="richer_v1 compares baseline vs richer_v1 variants; legacy runs older flow_balance ablations.",
    )
    parser.add_argument("--output_json", default="logs/knn_feature_audit_richer_v1_50k.json")
    parser.add_argument("--output_md", default="notes/knn_feature_audit_richer_v1_50k.md")
    parser.add_argument(
        "--baseline_key",
        default="edge_native+degree_fan|ordinal|legacy_standard",
        help="Report key for Jaccard baseline comparisons in markdown.",
    )
    args = parser.parse_args()

    _, df_train, _, spec = load_train_frame(args.data, args.data_config, max_rows=args.max_rows)
    configs = _default_audit_configs() if args.audit_profile == "richer_v1" else []

    reports: List[Dict[str, object]] = []
    neighbor_cache: Dict[str, np.ndarray] = {}

    for cfg in configs:
        fs = str(cfg["feature_set"])
        enc = str(cfg["categorical_encoding"])
        scaling = str(cfg["scaling"])
        include_pair = cfg.get("include_pair_history")
        rep, nbr_ids = audit_feature_set(
            df_train,
            feature_set=fs,
            k=args.k,
            categorical_encoding=enc,
            scaling=scaling,
            label_col=spec.label_col,
            include_pair_history=include_pair if include_pair is None else bool(include_pair),
        )
        reports.append(rep)
        neighbor_cache[_report_key(fs, enc, scaling)] = nbr_ids

    jaccard_pairs: List[Dict[str, object]] = []
    keys = list(neighbor_cache.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            jacc = _pairwise_neighbor_overlap(neighbor_cache[keys[i]], neighbor_cache[keys[j]])
            jaccard_pairs.append({"a": keys[i], "b": keys[j], "mean_jaccard": jacc})

    print("=== transaction KNN feature audit ===")
    print(f"data={args.data} max_rows={args.max_rows} k={args.k} n_train={len(df_train)} profile={args.audit_profile}")
    for rep in reports:
        print(json.dumps({k: rep[k] for k in rep if k != "feature_names"}, indent=2, sort_keys=True))
        print()
    for pair in jaccard_pairs:
        print(f"neighbor_jaccard {pair['a']} vs {pair['b']}: {pair['mean_jaccard']:.4f}")

    payload = {
        "data": args.data,
        "max_rows": args.max_rows,
        "k": args.k,
        "n_train": len(df_train),
        "audit_profile": args.audit_profile,
        "reports": reports,
        "neighbor_jaccard": jaccard_pairs,
    }
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Wrote {out}")
    if args.output_md:
        _write_markdown(
            Path(args.output_md),
            data=args.data,
            k=args.k,
            max_rows=args.max_rows,
            reports=reports,
            jaccard_pairs=jaccard_pairs,
            baseline_key=args.baseline_key,
        )
        print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
