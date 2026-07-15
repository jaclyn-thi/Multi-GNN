#!/usr/bin/env python3
"""Consolidated summary: Small-HI seed2 + Small-LI multiseed temporal_flow ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _load(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _answer_questions(
    hi: Dict[str, Any],
    li_ms: Dict[str, Any],
) -> Dict[str, str]:
    hi_d = hi["deltas"]["D_minus_B_primary"]
    hi_c = hi["deltas"]["C_minus_A"]
    li_agg = li_ms["aggregate"]["D_minus_B"]["metrics"]
    li_auprc_mean = li_agg["auprc"]["mean"]
    li_wins = li_ms["D_beats_B_auprc_count"]
    li_n = li_ms["n_seeds"]
    consistent = li_ms["conclusions"]["direction_consistent"]

    def _verdict(delta: float, eps: float = 0.005) -> str:
        if delta > eps:
            return "modest improvement"
        if delta < -eps:
            return "regression"
        return "no clear change (within noise floor)"

    return {
        "1_temporal_flow_vs_pre3h_alone": (
            f"HI ΔAUPRC(C−A)={hi_c['auprc']:+.4f} ({_verdict(hi_c['auprc'])}); "
            f"LI mean ΔAUPRC(C−A) from per-seed payloads — see multiseed JSON."
        ),
        "2_temporal_flow_vs_pre3h_plus_raw_primary": (
            f"HI ΔAUPRC(D−B)={hi_d['auprc']:+.4f} ({_verdict(hi_d['auprc'])}); "
            f"LI mean ΔAUPRC(D−B)={li_auprc_mean:+.4f} ({_verdict(li_auprc_mean)}); "
            f"D beats B on {li_wins}/{li_n} LI seeds."
        ),
        "3_holds_on_small_hi": _verdict(hi_d["auprc"]),
        "4_li_direction_consistent": "yes" if consistent else "mixed/no",
        "5_feature_level": "See per-run feature_diagnostics (coefficients, correlation, univariate AUPRC on train).",
        "6_gain_type": (
            "Compare ΔAUPRC (ranking), ΔF1 (thresholded), and Δlift@100 (alert budget) in per-run JSONs."
        ),
        "7_redundancy_with_raw": (
            "Inspect temporal_flow correlation matrix and whether D−B ≪ C−A (suggests raw overlap)."
        ),
        "8_thesis_placement": (
            "Include in main method only if D beats B on HI and ≥2/3 LI seeds without F1 regression >0.01; "
            "otherwise appendix or future work."
        ),
    }


def write_md(path: Path, summary: Dict[str, Any]) -> None:
    lines = [
        "# Temporal flow causal ablation — consolidated summary",
        "",
        "## Answers",
        "",
    ]
    for k, v in summary["answers"].items():
        lines.append(f"{k}. {v}")
    lines.extend(["", "## Sources", ""])
    for src in summary["sources"]:
        lines.append(f"- `{src}`")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hi_json", default="results/diagnostics/temporal_flow_ablation_small_hi_40ep_seed2.json")
    p.add_argument(
        "--li_multiseed_json",
        default="results/diagnostics/temporal_flow_ablation_small_li_multiseed.json",
    )
    p.add_argument("--output_json", default="results/diagnostics/temporal_flow_causal_ablation_summary.json")
    p.add_argument("--output_md", default="notes/temporal_flow_causal_ablation_summary.md")
    args = p.parse_args()

    hi = _load(Path(args.hi_json))
    li_ms = _load(Path(args.li_multiseed_json))
    summary = {
        "diagnostic": "temporal_flow_causal_ablation_summary",
        "sources": [args.hi_json, args.li_multiseed_json],
        "small_hi": {
            "run_name": hi.get("run_name"),
            "D_minus_B": hi["deltas"]["D_minus_B_primary"],
            "C_minus_A": hi["deltas"]["C_minus_A"],
        },
        "small_li_multiseed": li_ms.get("aggregate"),
        "answers": _answer_questions(hi, li_ms),
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_md(Path(args.output_md), summary)
    print(f"Wrote {out} and {args.output_md}")


if __name__ == "__main__":
    main()
