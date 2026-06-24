#!/usr/bin/env python3
"""Metric and scaling audit on 50k train rows (CPU, audit-only)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from transaction_knn.audit_utils import (
    audit_neighbor_set,
    build_audit_matrix,
    ordinary_knn,
    pairwise_jaccard,
)
from transaction_knn.features import feature_set_metadata, load_train_frame


VARIANTS = [
    {
        "name": "baseline_cosine",
        "feature_set": "edge_native+degree_fan",
        "categorical_encoding": "ordinal",
        "scaling": "legacy_standard",
        "metric": "cosine",
        "l2_normalize": True,
        "apply_group_weights": False,
    },
    {
        "name": "richer_v1_cosine_robust",
        "feature_set": "richer_v1",
        "categorical_encoding": "one_hot",
        "scaling": "robust",
        "metric": "cosine",
        "l2_normalize": True,
        "apply_group_weights": True,
    },
    {
        "name": "richer_v1_euclidean_robust",
        "feature_set": "richer_v1",
        "categorical_encoding": "one_hot",
        "scaling": "robust",
        "metric": "euclidean",
        "l2_normalize": False,
        "apply_group_weights": False,
    },
    {
        "name": "richer_v1_euclidean_robust_weighted",
        "feature_set": "richer_v1",
        "categorical_encoding": "one_hot",
        "scaling": "robust",
        "metric": "euclidean",
        "l2_normalize": False,
        "apply_group_weights": True,
    },
    {
        "name": "richer_v1_cosine_no_static",
        "feature_set": "richer_v1_no_static",
        "categorical_encoding": "one_hot",
        "scaling": "robust",
        "metric": "cosine",
        "l2_normalize": True,
        "apply_group_weights": True,
    },
    {
        "name": "richer_v1_causal_only_cosine",
        "feature_set": "richer_v1_causal_only",
        "categorical_encoding": "one_hot",
        "scaling": "robust",
        "metric": "cosine",
        "l2_normalize": True,
        "apply_group_weights": True,
    },
]


def _value_prefix(metric: str) -> str:
    return "sim" if metric == "cosine" else "dist"


def _audit_variant(df_train, label_col: str, cfg: Dict, k: int) -> Dict[str, object]:
    features, detail = build_audit_matrix(
        df_train,
        feature_set=cfg["feature_set"],
        categorical_encoding=cfg["categorical_encoding"],
        scaling=cfg["scaling"],
        metric=cfg["metric"],
        l2_normalize=bool(cfg.get("l2_normalize", True)),
        apply_group_weights=bool(cfg.get("apply_group_weights", False)),
    )
    nbr_ids, nbr_vals = ordinary_knn(features, k, cfg["metric"])
    prefix = _value_prefix(cfg["metric"])
    rep = audit_neighbor_set(df_train, nbr_ids, nbr_vals, label_col=label_col, value_prefix=prefix)
    rep["variant_name"] = cfg["name"]
    rep["metric"] = cfg["metric"]
    rep["scaling"] = cfg["scaling"]
    rep["l2_normalize"] = bool(cfg.get("l2_normalize", True))
    rep["apply_group_weights"] = bool(cfg.get("apply_group_weights", False))
    rep["n_features"] = int(features.shape[1])
    rep.update(feature_set_metadata(detail))
    return {"config": cfg, "report": rep, "neighbor_ids": nbr_ids}


def _write_md(path: Path, payload: dict, baseline_key: str) -> None:
    lines = [
        "# KNN metric / scaling audit (50k train rows)",
        "",
        f"- **Dataset:** {payload['data']}",
        f"- **Rows:** {payload['n_train']}",
        f"- **k:** {payload['k']}",
        f"- **Baseline for Jaccard:** {baseline_key}",
        "",
    ]
    jacc = {p["b"]: p["mean_jaccard"] for p in payload["neighbor_jaccard"] if p["a"] == baseline_key}
    for item in payload["variants"]:
        rep = item["report"]
        name = rep["variant_name"]
        prefix = "sim" if rep["metric"] == "cosine" else "dist"
        lines.extend(
            [
                f"## {name}",
                "",
                f"- feature_set=`{rep['feature_set']}` dims={rep['n_features']} metric={rep['metric']} "
                f"scaling={rep['scaling']} l2={rep['l2_normalize']} group_weights={rep['apply_group_weights']}",
                f"- {prefix}_mean={rep.get(f'{prefix}_mean', float('nan')):.4f}, "
                f"{prefix}_p50={rep.get(f'{prefix}_p50', float('nan')):.4f}, "
                f"{prefix}_p90={rep.get(f'{prefix}_p90', float('nan')):.4f}",
                f"- unique_neighbor_fraction={rep.get('unique_neighbor_fraction', float('nan')):.4f}, "
                f"hub1={rep.get('hubness_top1_neighbor_share', float('nan')):.4f}, "
                f"hub10={rep.get('hubness_top10_neighbors_share', float('nan')):.4f}",
                f"- Jaccard vs baseline={jacc.get(name, float('nan')):.4f}",
                f"- endpoint sender={rep.get('same_sender', float('nan')):.4f}, "
                f"pair={rep.get('same_pair', float('nan')):.4f}",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="Small-HI")
    parser.add_argument("--data_config", default="data_config.json")
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--max_rows", type=int, default=50000)
    parser.add_argument("--output_json", default="logs/knn_metric_scaling_audit_50k.json")
    parser.add_argument("--output_md", default="notes/knn_metric_scaling_audit_50k.md")
    parser.add_argument("--baseline_name", default="baseline_cosine")
    args = parser.parse_args()

    _, df_train, _, spec = load_train_frame(args.data, args.data_config, max_rows=args.max_rows)
    audited = [_audit_variant(df_train, spec.label_col, cfg, args.k) for cfg in VARIANTS]
    variants_out = [{"config": a["config"], "report": a["report"]} for a in audited]

    baseline_ids: Optional[object] = None
    neighbor_jaccard: List[Dict[str, object]] = []
    for a in audited:
        name = a["config"]["name"]
        if name == args.baseline_name:
            baseline_ids = a["neighbor_ids"]
    if baseline_ids is not None:
        for a in audited:
            name = a["config"]["name"]
            if name == args.baseline_name:
                continue
            neighbor_jaccard.append(
                {
                    "a": args.baseline_name,
                    "b": name,
                    "mean_jaccard": pairwise_jaccard(baseline_ids, a["neighbor_ids"]),
                }
            )

    payload = {
        "data": args.data,
        "max_rows": args.max_rows,
        "k": args.k,
        "n_train": len(df_train),
        "baseline_name": args.baseline_name,
        "variants": variants_out,
        "neighbor_jaccard": neighbor_jaccard,
    }

    print("=== metric / scaling KNN audit ===")
    for item in variants_out:
        rep = item["report"]
        print(json.dumps({k: rep[k] for k in rep if k not in ("feature_names", "leakage_policy")}, indent=2))
        print()
    for pair in neighbor_jaccard:
        print(f"jaccard {pair['a']} vs {pair['b']}: {pair['mean_jaccard']:.4f}")

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {out_json}")
    if args.output_md:
        _write_md(Path(args.output_md), payload, args.baseline_name)
        print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
