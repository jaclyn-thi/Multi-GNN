#!/usr/bin/env python3
"""Summarize current-protocol probe sweep JSON into comparison tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]

FOCUS_FEATURES = ("embedding+raw", "embedding+raw+morph")
GIN_40EP_RUNS = ("gin_40ep_seed1", "gin_40ep_seed2")
REFERENCE_CW = "model"
REFERENCE_C = 1.0


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _filter(rows: List[Dict[str, Any]], **kwargs: Any) -> List[Dict[str, Any]]:
    out = rows
    for key, val in kwargs.items():
        if val is None:
            continue
        out = [r for r in out if r.get(key) == val]
    return out


def _best(rows: List[Dict[str, Any]], metric: str) -> Optional[Dict[str, Any]]:
    valid = [r for r in rows if r.get(metric) is not None]
    if not valid:
        return None
    return max(valid, key=lambda r: float(r[metric]))


def _write_md(path: Path, title: str, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([f"# {title}", ""] + lines) + "\n", encoding="utf-8")


def _row_line(r: Dict[str, Any]) -> str:
    return (
        f"| {r['run_label']} | `{r['features']}` | {r['class_weight_mode']} | {r['probe_C']} | "
        f"{r['test_auroc']:.4f} | {r['test_auprc']:.4f} | {r['test_f1']:.4f} | "
        f"{r['test_f1_at_0_5']:.4f} | {r['threshold']:.4f} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_json",
        default=str(ROOT / "results/diagnostics/probe_sweep_current_protocol.json"),
    )
    parser.add_argument(
        "--output_md",
        default=str(ROOT / "results/diagnostics/probe_sweep_current_protocol_summary.md"),
    )
    args = parser.parse_args()

    payload = _load(Path(args.input_json))
    rows: List[Dict[str, Any]] = payload["rows"]

    header = (
        "| Run | Features | cw | C | AUROC | AUPRC | F1 | F1@0.5 | Thr |"
        "\n|-----|----------|----|---|------:|------:|---:|-------:|----:|"
    )

    # Q1: embedding+raw vs embedding+raw+morph for 40ep @ reference settings
    q1_lines = [header]
    for run_name in GIN_40EP_RUNS:
        for feat in FOCUS_FEATURES:
            match = _filter(
                rows,
                run_name=run_name,
                features=feat,
                class_weight_mode=REFERENCE_CW,
                probe_C=REFERENCE_C,
            )
            if match:
                q1_lines.append(_row_line(match[0]))
    q1_lines.extend(
        [
            "",
            "**Q1 note:** Compare F1/AUPRC across the two feature stacks at shared GIN weights, C=1.0.",
            "",
        ]
    )

    # Q2: 40ep seed2 embedding+raw across cw/C
    q2_lines = [header]
    seed2_raw = _filter(rows, run_name="gin_40ep_seed2", features="embedding+raw")
    seed2_raw.sort(key=lambda r: (-float(r["test_f1"]), -float(r["test_auprc"])))
    for r in seed2_raw[:12]:
        q2_lines.append(_row_line(r))
    q2_lines.extend(
        [
            "",
            "**Q2 note:** Top 12 settings for GIN 40ep seed2 `embedding+raw` (sorted by test F1).",
            "",
        ]
    )

    # Q3: Is 0.347 robust? Show all seed2 embedding+raw with F1 >= 0.32 or top 5
    q3_lines = [header]
    high_f1 = [r for r in seed2_raw if float(r["test_f1"]) >= 0.32]
    if not high_f1:
        high_f1 = seed2_raw[:5]
    for r in high_f1:
        q3_lines.append(_row_line(r))
    ref347 = _filter(
        rows,
        run_name="gin_40ep_seed2",
        features="embedding+raw",
        class_weight_mode=REFERENCE_CW,
        probe_C=REFERENCE_C,
    )
    q3_lines.extend(["", "**Q3 reference (prior ablation, cw=model C=1.0):**"])
    if ref347:
        q3_lines.append(_row_line(ref347[0]))
    q3_lines.append("")

    # Q4: Best AUPRC / F1 / F1@0.5 overall (SSL modes only)
    ssl_rows = [r for r in rows if r["features"] in ("embedding", "embedding+raw", "embedding+raw+morph")]
    best_f1 = _best(ssl_rows, "test_f1")
    best_auprc = _best(ssl_rows, "test_auprc")
    best_f1_05 = _best(ssl_rows, "test_f1_at_0_5")

    q4_lines = [
        "## Best overall (SSL feature modes)",
        "",
        header,
    ]
    for label, row in (
        ("Best test F1", best_f1),
        ("Best test AUPRC", best_auprc),
        ("Best test F1@0.5", best_f1_05),
    ):
        if row:
            q4_lines.append(_row_line(row))
            q4_lines.append(f"<!-- {label} -->")
    q4_lines.append("")

    # 40ep: embedding+raw better than full stack? Count at each cw/C
    q1b_lines = [
        "## Q1b: `embedding+raw` vs `embedding+raw+morph` win rate (40ep, all cw×C)",
        "",
        "| Run | raw wins F1 | morph wins F1 | ties | raw wins AUPRC |",
        "|-----|------------:|--------------:|-----:|-----------------:|",
    ]
    for run_name in GIN_40EP_RUNS:
        f1_raw_wins = morph_wins = tie = 0
        auprc_raw = 0
        for cw in ("model", "none", "balanced"):
            for c in (0.01, 0.1, 1.0, 10.0):
                raw_row = _filter(
                    rows, run_name=run_name, features="embedding+raw", class_weight_mode=cw, probe_C=c
                )
                morph_row = _filter(
                    rows,
                    run_name=run_name,
                    features="embedding+raw+morph",
                    class_weight_mode=cw,
                    probe_C=c,
                )
                if not raw_row or not morph_row:
                    continue
                rf, mf = float(raw_row[0]["test_f1"]), float(morph_row[0]["test_f1"])
                if rf > mf:
                    f1_raw_wins += 1
                elif mf > rf:
                    morph_wins += 1
                else:
                    tie += 1
                if float(raw_row[0]["test_auprc"]) > float(morph_row[0]["test_auprc"]):
                    auprc_raw += 1
        label = next(r["run_label"] for r in rows if r["run_name"] == run_name)
        q1b_lines.append(
            f"| {label} | {f1_raw_wins} | {morph_wins} | {tie} | {auprc_raw} |"
        )
    q1b_lines.append("")

    all_lines = (
        ["## Q1 — 40ep stack comparison @ cw=model, C=1.0"]
        + q1_lines[1:]
        + ["## Q2 — seed2 `embedding+raw` across probe settings"]
        + q2_lines[1:]
        + ["## Q3 — high-F1 seed2 `embedding+raw` settings"]
        + q3_lines[1:]
        + q4_lines
        + q1b_lines
    )

    out_md = Path(args.output_md)
    _write_md(out_md, "Current-protocol probe sweep summary", all_lines)
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
