#!/usr/bin/env python3
"""Summarize explicit positive-weight current-protocol probe sweeps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_rows(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [c for c in payload.get("cells", []) if c.get("status") == "completed"]


def metric(row: Dict[str, Any], key: str) -> float:
    return float(row["test"][key])


def top(rows: List[Dict[str, Any]], key: str, n: int = 10) -> List[Dict[str, Any]]:
    return sorted(rows, key=lambda r: metric(r, key), reverse=True)[:n]


def best_by_group(rows: List[Dict[str, Any]], keys: tuple[str, ...], metric_key: str = "f1") -> List[Dict[str, Any]]:
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row.get(k) for k in keys), []).append(row)
    return [max(vals, key=lambda r: metric(r, metric_key)) for vals in groups.values()]


def row_line(row: Dict[str, Any]) -> str:
    test = row["test"]
    return (
        f"| {row.get('run_label', row.get('run_name'))} | `{row['feature_mode']}` | "
        f"{row['class_weight_policy']} | {row['probe_C']} | {test['auroc']:.4f} | "
        f"{test['auprc']:.4f} | {test['f1']:.4f} | {test['f1_at_0_5']:.4f} | "
        f"{test.get('precision_at_500', float('nan')):.4f} | {test.get('recall_at_500', float('nan')):.4f} | "
        f"{test.get('lift_at_500', float('nan')):.1f} | {test['threshold']:.4f} |"
    )


def write_markdown(path: Path, payload: Dict[str, Any]) -> None:
    header = (
        "| Run | Features | weight | C | AUROC | AUPRC | F1 | F1@0.5 | P@500 | R@500 | lift@500 | Thr |"
        "\n|-----|----------|--------|---|------:|------:|---:|-------:|------:|------:|---------:|----:|"
    )
    rows = payload["rows"]
    best_f1 = payload["top10_test_f1"][0] if payload["top10_test_f1"] else None
    best_auprc = payload["top10_test_auprc"][0] if payload["top10_test_auprc"] else None
    lines = [
        f"# {payload['description']}",
        "",
        f"Completed cells: **{payload['cells_completed']}**",
        "",
        "## Key Answers",
        "",
    ]
    if best_f1:
        lines.append(
            f"- Best F1 uses `{best_f1['feature_mode']}` on {best_f1.get('run_label', best_f1.get('run_name'))} "
            f"with {best_f1['class_weight_policy']} (C={best_f1['probe_C']}): **{best_f1['test']['f1']:.4f}**."
        )
    if best_auprc:
        lines.append(
            f"- Best AUPRC uses `{best_auprc['feature_mode']}` on {best_auprc.get('run_label', best_auprc.get('run_name'))} "
            f"with {best_auprc['class_weight_policy']} (C={best_auprc['probe_C']}): **{best_auprc['test']['auprc']:.4f}**."
        )
    close_to_model = [
        r for r in rows if r.get("class_weight_policy") == "pos_6.275"
    ]
    if close_to_model:
        best_modelish = max(close_to_model, key=lambda r: metric(r, "f1"))
        lines.append(
            f"- Best explicit `pos_6.275` F1 is **{best_modelish['test']['f1']:.4f}** "
            f"(`{best_modelish['feature_mode']}`, {best_modelish.get('run_label', best_modelish.get('run_name'))})."
        )
    lines.append("- Explicit positive weights are evaluation-only; they do not alter SSL pretraining.")

    sections = [
        ("Top 10 by test F1", payload["top10_test_f1"]),
        ("Top 10 by test AUPRC", payload["top10_test_auprc"]),
        ("Top 10 by F1@0.5", payload["top10_test_f1_at_0_5"]),
        ("Best setting per run/feature by F1", payload["best_by_run_feature_f1"]),
    ]
    for title, section_rows in sections:
        lines.extend(["", f"## {title}", "", header])
        for row in section_rows:
            lines.append(row_line(row))
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def summarize(input_path: Path, out_json: Path, out_md: Path, notes_md: Path, description: str) -> None:
    rows = load_rows(input_path)
    payload = {
        "description": description,
        "cells_completed": len(rows),
        "rows": rows,
        "top10_test_f1": top(rows, "f1"),
        "top10_test_auprc": top(rows, "auprc"),
        "top10_test_f1_at_0_5": top(rows, "f1_at_0_5"),
        "best_by_run_feature_f1": best_by_group(rows, ("run_name", "feature_mode"), "f1"),
        "best_by_run_feature_auprc": best_by_group(rows, ("run_name", "feature_mode"), "auprc"),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(out_md, payload)
    write_markdown(notes_md, payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--out_md", required=True)
    parser.add_argument("--notes_md", required=True)
    parser.add_argument("--description", required=True)
    args = parser.parse_args()
    summarize(
        Path(args.input),
        Path(args.out_json),
        Path(args.out_md),
        Path(args.notes_md),
        args.description,
    )
    print(args.notes_md)


if __name__ == "__main__":
    main()
