#!/usr/bin/env python3
"""Summarize the Small-LI current-protocol probe sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
FEATURE_MODES = ("raw+morph", "embedding", "embedding+raw", "embedding+raw+morph")

DEFAULT_RESULTS = {
    "raw+morph": {"auroc": 0.857628, "auprc": 0.016125, "f1": 0.056649, "f1_at_0_5": 0.049725},
    "embedding": {"auroc": 0.898733, "auprc": 0.016010, "f1": 0.052497, "f1_at_0_5": 0.049180},
    "embedding+raw": {"auroc": 0.909303, "auprc": 0.027209, "f1": 0.075668, "f1_at_0_5": 0.080882},
    "embedding+raw+morph": {"auroc": 0.925030, "auprc": 0.039119, "f1": 0.055511, "f1_at_0_5": 0.072663},
}

SMALL_HI_REFERENCES = [
    {"label": "Small-HI GINe emlps+tds 20ep seed1 embedding", "auroc": 0.944, "auprc": 0.213, "f1": 0.259, "f1_at_0_5": 0.257},
    {"label": "Small-HI GINe emlps+tds 20ep seed1 embedding+raw+morph", "auroc": 0.945, "auprc": 0.276, "f1": 0.298, "f1_at_0_5": 0.327},
    {"label": "Small-HI FNF seed1 embedding+raw+morph", "auroc": 0.959, "auprc": 0.276, "f1": 0.319, "f1_at_0_5": 0.303},
    {"label": "Small-HI raw+morph only", "auroc": 0.905, "auprc": 0.066, "f1": 0.136, "f1_at_0_5": 0.132},
]


def load_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [c for c in payload.get("cells", []) if c.get("status") == "completed"]


def metric(row: Dict[str, Any], key: str) -> float:
    return float(row["test"][key])


def top_n(rows: List[Dict[str, Any]], key: str, n: int = 10) -> List[Dict[str, Any]]:
    return sorted(rows, key=lambda r: metric(r, key), reverse=True)[:n]


def best_per_feature(rows: List[Dict[str, Any]], key: str = "f1") -> List[Dict[str, Any]]:
    out = []
    for feature in FEATURE_MODES:
        cand = [r for r in rows if r.get("feature_mode") == feature]
        if cand:
            out.append(max(cand, key=lambda r: metric(r, key)))
    return out


def best_feature(rows: List[Dict[str, Any]], feature: str, key: str = "f1") -> Optional[Dict[str, Any]]:
    cand = [r for r in rows if r.get("feature_mode") == feature]
    return max(cand, key=lambda r: metric(r, key)) if cand else None


def row_line(row: Dict[str, Any]) -> str:
    test = row["test"]
    return (
        f"| `{row['feature_mode']}` | {row['class_weight_policy']} | {row['probe_C']} | "
        f"{test['auroc']:.4f} | {test['auprc']:.4f} | {test['f1']:.4f} | "
        f"{test['f1_at_0_5']:.4f} | {test.get('precision_at_500', float('nan')):.4f} | "
        f"{test.get('recall_at_500', float('nan')):.4f} | {test.get('lift_at_500', float('nan')):.1f} | "
        f"{test['threshold']:.4f} |"
    )


def write_markdown(path: Path, payload: Dict[str, Any]) -> None:
    header = (
        "| Features | cw | C | AUROC | AUPRC | F1 | F1@0.5 | P@500 | R@500 | lift@500 | Thr |"
        "\n|----------|----|---|------:|------:|---:|-------:|------:|------:|---------:|----:|"
    )
    raw_best = payload["best_by_feature_f1"].get("raw+morph")
    emb_best = payload["best_by_feature_f1"].get("embedding")
    emb_raw_best = payload["best_by_feature_f1"].get("embedding+raw")
    emb_morph_best = payload["best_by_feature_f1"].get("embedding+raw+morph")

    lines = [
        "# Small-LI current-protocol probe sweep",
        "",
        "Checkpointed CPU sweep on frozen Small-LI SSL embeddings. This is a Small-LI scout, not a frozen benchmark.",
        "",
        f"Completed cells: **{payload['cells_completed']}** / 48",
        "",
        "## Key Answers",
        "",
    ]
    if emb_raw_best and emb_morph_best:
        better = "`embedding+raw`" if metric(emb_raw_best, "f1") >= metric(emb_morph_best, "f1") else "`embedding+raw+morph`"
        lines.append(
            f"- Best F1 after tuning is {better}: "
            f"`embedding+raw` {metric(emb_raw_best, 'f1'):.4f} vs "
            f"`embedding+raw+morph` {metric(emb_morph_best, 'f1'):.4f}."
        )
    if raw_best and emb_best:
        lines.append(
            f"- Embedding-only best F1 {metric(emb_best, 'f1'):.4f}; raw+morph-only best F1 {metric(raw_best, 'f1'):.4f}."
        )
    if emb_raw_best and raw_best:
        lines.append(
            f"- Best embedding+raw F1 improves over raw+morph by {metric(emb_raw_best, 'f1') - metric(raw_best, 'f1'):+.4f}."
        )
    lines.append("- `balanced` class weights are exploratory; inspect thresholds and F1@0.5 before trusting val-tuned F1.")

    sections = [
        ("Top 10 by test F1", payload["top10_test_f1"]),
        ("Top 10 by test AUPRC", payload["top10_test_auprc"]),
        ("Top 10 by F1@0.5", payload["top10_test_f1_at_0_5"]),
        ("Best setting per feature mode (F1)", list(payload["best_by_feature_f1"].values())),
    ]
    for title, rows in sections:
        lines.extend(["", f"## {title}", "", header])
        for row in rows:
            lines.append(row_line(row))

    lines.extend(["", "## Original Default Small-LI Result", ""])
    lines.append("| Features | AUROC | AUPRC | F1 | F1@0.5 |")
    lines.append("|----------|------:|------:|---:|-------:|")
    for feature, vals in DEFAULT_RESULTS.items():
        lines.append(
            f"| `{feature}` | {vals['auroc']:.4f} | {vals['auprc']:.4f} | "
            f"{vals['f1']:.4f} | {vals['f1_at_0_5']:.4f} |"
        )

    lines.extend(["", "## Small-HI References", ""])
    lines.append("| Reference | AUROC | AUPRC | F1 | F1@0.5 |")
    lines.append("|-----------|------:|------:|---:|-------:|")
    for row in SMALL_HI_REFERENCES:
        lines.append(
            f"| {row['label']} | {row['auroc']:.3f} | {row['auprc']:.3f} | "
            f"{row['f1']:.3f} | {row['f1_at_0_5']:.3f} |"
        )

    lines.extend(
        [
            "",
            "Artifacts:",
            "",
            "- Partial JSON: `results/diagnostics/probe_sweep_small_li_current_protocol_partial.json`",
            "- Final JSON: `results/diagnostics/probe_sweep_small_li_current_protocol.json`",
            "- Feature cache: `results/cache/probe_features_small_li_current_protocol/`",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def summarize(input_path: Path, out_json: Path, out_md: Path, notes_md: Path) -> None:
    rows = load_rows(input_path)
    best_f1 = {r["feature_mode"]: r for r in best_per_feature(rows, "f1")}
    payload = {
        "description": "Small-LI current-protocol probe sweep summary.",
        "cells_completed": len(rows),
        "rows": rows,
        "top10_test_f1": top_n(rows, "f1"),
        "top10_test_auprc": top_n(rows, "auprc"),
        "top10_test_f1_at_0_5": top_n(rows, "f1_at_0_5"),
        "best_by_feature_f1": best_f1,
        "best_by_feature_auprc": {r["feature_mode"]: r for r in best_per_feature(rows, "auprc")},
        "default_results": DEFAULT_RESULTS,
        "small_hi_references": SMALL_HI_REFERENCES,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(out_md, payload)
    # notes/probe_sweep_small_li_current_protocol.md is curated by hand; only the raw
    # diagnostics markdown is regenerated here to avoid clobbering the interpreted note.
    _ = notes_md


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=str(ROOT / "results/diagnostics/probe_sweep_small_li_current_protocol.json"),
    )
    parser.add_argument(
        "--out_json",
        default=str(ROOT / "results/diagnostics/probe_sweep_small_li_current_protocol_summary.json"),
    )
    parser.add_argument(
        "--out_md",
        default=str(ROOT / "results/diagnostics/probe_sweep_small_li_current_protocol.md"),
    )
    parser.add_argument(
        "--notes_md",
        default=str(ROOT / "notes/probe_sweep_small_li_current_protocol.md"),
    )
    args = parser.parse_args()
    summarize(Path(args.input), Path(args.out_json), Path(args.out_md), Path(args.notes_md))
    print(args.out_md)


if __name__ == "__main__":
    main()
