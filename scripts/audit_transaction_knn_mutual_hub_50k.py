#!/usr/bin/env python3
"""Mutual-KNN and hub-filter audit on 50k train rows (CPU, audit-only)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from transaction_knn.audit_utils import (
    audit_neighbor_set,
    build_audit_matrix,
    hub_filter_neighbors,
    mutual_knn_neighbors,
    ordinary_knn,
)
from transaction_knn.features import feature_set_metadata, load_train_frame


FEATURE_CONFIGS = [
    {
        "name": "baseline_edge_native_degree_fan",
        "feature_set": "edge_native+degree_fan",
        "categorical_encoding": "ordinal",
        "scaling": "legacy_standard",
        "metric": "cosine",
    },
    {
        "name": "richer_v1_one_hot",
        "feature_set": "richer_v1",
        "categorical_encoding": "one_hot",
        "scaling": "robust",
        "metric": "cosine",
        "apply_group_weights": True,
    },
    {
        "name": "richer_v1_no_pair",
        "feature_set": "richer_v1_no_pair",
        "categorical_encoding": "one_hot",
        "scaling": "robust",
        "metric": "cosine",
        "apply_group_weights": True,
    },
]

HUB_FILTER_FRACTIONS = (0.001, 0.005, 0.01)


def _audit_one_feature_set(df_train, label_col: str, cfg: Dict, k: int) -> Dict[str, object]:
    features, detail = build_audit_matrix(
        df_train,
        feature_set=cfg["feature_set"],
        categorical_encoding=cfg["categorical_encoding"],
        scaling=cfg["scaling"],
        metric=cfg["metric"],
        l2_normalize=True,
        apply_group_weights=bool(cfg.get("apply_group_weights", False)),
    )
    nbr_ids, nbr_vals = ordinary_knn(features, k, cfg["metric"])
    ordinary = audit_neighbor_set(df_train, nbr_ids, nbr_vals, label_col=label_col, value_prefix="sim")
    ordinary["selection_mode"] = "ordinary_topk"
    ordinary["config_name"] = cfg["name"]
    ordinary["feature_set"] = detail.feature_set
    ordinary["n_features"] = int(features.shape[1])
    ordinary.update(feature_set_metadata(detail))

    mutual_ids, mutual_meta = mutual_knn_neighbors(nbr_ids)
    mutual_vals = nbr_vals.copy()
    mutual_vals[mutual_ids < 0] = float("nan")
    mutual = audit_neighbor_set(df_train, mutual_ids, mutual_vals, label_col=label_col, value_prefix="sim")
    mutual["selection_mode"] = "mutual_knn"
    mutual["config_name"] = cfg["name"]
    mutual["feature_set"] = detail.feature_set
    mutual.update(mutual_meta)

    hub_reports = []
    for frac in HUB_FILTER_FRACTIONS:
        filtered_ids, filter_meta = hub_filter_neighbors(nbr_ids, frac)
        filtered_vals = nbr_vals.copy()
        filtered_vals[filtered_ids < 0] = float("nan")
        rep = audit_neighbor_set(
            df_train, filtered_ids, filtered_vals, label_col=label_col, value_prefix="sim"
        )
        rep["selection_mode"] = f"hub_filter_{frac}"
        rep["config_name"] = cfg["name"]
        rep["feature_set"] = detail.feature_set
        rep.update(filter_meta)
        hub_reports.append(rep)

    return {
        "config": cfg,
        "ordinary_topk": ordinary,
        "mutual_knn": mutual,
        "hub_filters": hub_reports,
    }


def _write_md(path: Path, payload: dict) -> None:
    lines = [
        "# KNN mutual / hub-filter audit (50k train rows)",
        "",
        f"- **Dataset:** {payload['data']}",
        f"- **Rows:** {payload['n_train']}",
        f"- **k:** {payload['k']}",
        "",
        "Label enrichment is analysis-only.",
        "",
    ]
    for block in payload["results"]:
        cfg = block["config"]
        lines.append(f"## {cfg['name']} (`{cfg['feature_set']}`)")
        lines.append("")
        for key in ("ordinary_topk", "mutual_knn"):
            rep = block[key]
            lines.append(
                f"### {rep['selection_mode']}: sim_mean={rep.get('sim_mean', float('nan')):.4f}, "
                f"uniq={rep.get('unique_neighbor_fraction', float('nan')):.4f}, "
                f"hub1={rep.get('hubness_top1_neighbor_share', float('nan')):.4f}"
            )
            if key == "mutual_knn":
                lines.append(
                    f"- mutual coverage={rep.get('mutual_anchor_coverage', float('nan')):.4f}, "
                    f"avg mutual neighbors={rep.get('mutual_avg_neighbors_per_anchor', float('nan')):.2f}"
                )
            lines.append(
                f"- endpoint: sender={rep.get('same_sender', float('nan')):.4f}, "
                f"pair={rep.get('same_pair', float('nan')):.4f}"
            )
            lines.append("")
        lines.append("### Hub filters (vs ordinary)")
        for rep in block["hub_filters"]:
            lines.append(
                f"- `{rep['selection_mode']}`: uniq={rep.get('unique_neighbor_fraction', float('nan')):.4f}, "
                f"hub1={rep.get('hubness_top1_neighbor_share', float('nan')):.4f}, "
                f"lost_all={rep.get('hub_filter_anchors_lost_all_fraction', float('nan')):.4f}, "
                f"banned={rep.get('hub_filter_banned_neighbors', 0):.0f}"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="Small-HI")
    parser.add_argument("--data_config", default="data_config.json")
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--max_rows", type=int, default=50000)
    parser.add_argument("--output_json", default="logs/knn_mutual_hub_audit_50k.json")
    parser.add_argument("--output_md", default="notes/knn_mutual_hub_audit_50k.md")
    args = parser.parse_args()

    _, df_train, _, spec = load_train_frame(args.data, args.data_config, max_rows=args.max_rows)
    results = [_audit_one_feature_set(df_train, spec.label_col, cfg, args.k) for cfg in FEATURE_CONFIGS]
    payload = {
        "data": args.data,
        "max_rows": args.max_rows,
        "k": args.k,
        "n_train": len(df_train),
        "hub_filter_fractions": list(HUB_FILTER_FRACTIONS),
        "results": results,
    }

    print("=== mutual / hub-filter KNN audit ===")
    print(json.dumps({k: v for k, v in payload.items() if k != "results"}, indent=2))
    for block in results:
        print(f"\n## {block['config']['name']}")
        print(json.dumps(block["ordinary_topk"], indent=2, sort_keys=True))
        print(json.dumps(block["mutual_knn"], indent=2, sort_keys=True))

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {out_json}")
    if args.output_md:
        _write_md(Path(args.output_md), payload)
        print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
