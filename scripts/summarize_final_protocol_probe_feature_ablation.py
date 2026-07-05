#!/usr/bin/env python3
"""Consolidate final-protocol probe feature ablation JSONs into one comparison table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]

COMPARE_MODES = ("embedding", "embedding+raw", "embedding+raw+morph")

RUNS = (
    {
        "label": "emlps+tds baseline",
        "json": "results/diagnostics/probe_feature_ablation_hi_contrastive_gin_emlps_tds_embedding_raw.json",
    },
    {
        "label": "FNF + emlps+tds",
        "json": "results/diagnostics/probe_feature_ablation_same_pair_fnf_emlps_tds.json",
    },
    {
        "label": "degree-aware + emlps+tds",
        "json": "results/diagnostics/probe_feature_ablation_degree_aware_edgedrop_emlps_tds.json",
    },
)


def _row(run_label: str, mode: str, entry: Dict[str, Any]) -> Dict[str, Any]:
    test = entry["test"]
    t05 = entry.get("test_at_threshold_0.5", {})
    return {
        "run_label": run_label,
        "features": mode,
        "auroc": test.get("auroc"),
        "auprc": test.get("auprc"),
        "f1": test.get("f1"),
        "precision": test.get("precision"),
        "recall": test.get("recall"),
        "f1_at_0_5": t05.get("f1"),
        "threshold": test.get("threshold"),
        "val_f1": test.get("val_f1"),
        "feature_dim": test.get("feature_dim"),
        "embedding_dir": entry.get("embedding_dir"),
    }


def _load_rows(path: Path, run_label: str) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    by_mode = {r["features"]: r for r in payload["runs"]}
    rows = []
    for mode in COMPARE_MODES:
        if mode not in by_mode:
            raise KeyError(f"{path}: missing features={mode!r}")
        rows.append(_row(run_label, mode, by_mode[mode]))
    return rows


def _write_md(path: Path, rows: List[Dict[str, Any]]) -> None:
    lines = [
        "# Final-protocol probe feature ablation comparison",
        "",
        "Frozen embeddings; logistic regression; val max-F1 threshold.",
        "",
        "| Run | Features | AUROC | AUPRC | F1 | Prec | Recall | F1@0.5 | Thr |",
        "|-----|----------|------:|------:|---:|-----:|-------:|-------:|----:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['run_label']} | `{r['features']}` | "
            f"{r['auroc']:.4f} | {r['auprc']:.4f} | {r['f1']:.4f} | "
            f"{r['precision']:.4f} | {r['recall']:.4f} | "
            f"{(r['f1_at_0_5'] if r['f1_at_0_5'] is not None else float('nan')):.4f} | "
            f"{r['threshold']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_json",
        default="results/diagnostics/probe_feature_ablation_final_protocol_comparison.json",
    )
    parser.add_argument(
        "--output_md",
        default="notes/archive/probe_feature_ablation_final_protocol_comparison.md",
    )
    args = parser.parse_args()

    all_rows: List[Dict[str, Any]] = []
    sources = []
    for spec in RUNS:
        path = ROOT / spec["json"]
        all_rows.extend(_load_rows(path, spec["label"]))
        sources.append({"label": spec["label"], "json": str(path.relative_to(ROOT))})

    out = {
        "protocol": {
            "feature_modes": list(COMPARE_MODES),
            "probe": "sklearn LogisticRegression (lbfgs)",
            "threshold_tuning": "max_f1_on_val",
        },
        "sources": sources,
        "rows": all_rows,
    }
    json_path = ROOT / args.output_json
    md_path = ROOT / args.output_md
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    _write_md(md_path, all_rows)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
