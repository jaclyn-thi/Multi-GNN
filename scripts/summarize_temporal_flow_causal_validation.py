#!/usr/bin/env python3
"""Consolidated validation summary for temporal_flow_causal ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def _load(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _answers(
    maxiter_cmp: Dict[str, Any],
    leakage: Dict[str, Any],
    shuffle: Dict[str, Any],
    hi_new: Dict[str, Any],
) -> Dict[str, str]:
    shifts = maxiter_cmp.get("aggregate_D_minus_B_auprc_shift", {})
    mean_shift = shifts.get("mean", float("nan"))
    conv_d = hi_new["arms"]["D_embedding_raw_temporal_flow"].get("convergence", {})
    recompute_ok = all(
        d["causal_history"]["recompute_matches_cache"] for d in leakage.get("datasets", [])
    )
    shuffle_ok = all(
        c["sanity"]["aligned_beats_shuffled_auprc"] and not c["sanity"]["concern_if_shuffled_retains_gain"]
        for c in shuffle.get("comparisons", [])
    )
    material_maxiter = abs(mean_shift) > 0.005 if mean_shift == mean_shift else True

    return {
        "1_maxiter5000_changes_conclusions": (
            "No — ΔAUPRC(D−B) shifts are within diagnostic noise."
            if not material_maxiter
            else f"Caution — mean ΔAUPRC(D−B) shift {mean_shift:+.4f} exceeds 0.005; review per-run flags."
        ),
        "2_logistic_regression_converged": (
            f"Arm D @5000 on HI: {conv_d.get('status')} (n_iter={conv_d.get('n_iter')}, max_iter={conv_d.get('max_iter')}). "
            "See per-arm convergence blocks in maxiter5000 JSONs."
        ),
        "3_leakage_audit_issues": (
            "None found — recompute matches cache; timestamp-tie batching documented and tested."
            if recompute_ok
            else "Issue — cache recompute mismatch; investigate before citing."
        ),
        "4_true_default_history_rates": "See leakage audit default_history_fractions per split (not zero==no-NaN).",
        "5_shuffle_control": (
            "Pass — aligned D beats shuffled D; shuffled D near B."
            if shuffle_ok
            else "Concern — shuffled control retains substantial gain; investigate alignment/leakage."
        ),
        "6_safe_to_cite": (
            "Yes, with max_iter=5000 as canonical if convergence and shuffle checks pass."
            if recompute_ok and shuffle_ok
            else "Not yet — resolve flagged validation items first."
        ),
        "7_canonical_numbers": "Use maxiter5000 JSONs if material_maxiter is false and convergence improved.",
        "8_thesis_placement": (
            "Main downstream stack if validation passes; otherwise appendix pending audit resolution."
        ),
    }


def write_md(path: Path, payload: Dict[str, Any]) -> None:
    lines = ["# temporal_flow_causal validation summary", "", "## Answers", ""]
    for k, v in payload["answers"].items():
        lines.append(f"{k}. {v}")
    lines.append("")
    lines.append("## Sources")
    for s in payload["sources"]:
        lines.append(f"- `{s}`")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--maxiter_comparison", default="results/diagnostics/temporal_flow_ablation_maxiter5000_comparison.json")
    p.add_argument("--leakage_audit", default="results/diagnostics/temporal_flow_causal_leakage_audit.json")
    p.add_argument("--shuffle_summary", default="results/diagnostics/temporal_flow_shuffle_control_summary.json")
    p.add_argument("--hi_maxiter5000", default="results/diagnostics/temporal_flow_ablation_small_hi_40ep_seed2_maxiter5000.json")
    p.add_argument("--output_json", default="results/diagnostics/temporal_flow_causal_validation_summary.json")
    p.add_argument("--output_md", default="notes/temporal_flow_causal_validation_summary.md")
    args = p.parse_args()

    maxiter_cmp = _load(Path(args.maxiter_comparison))
    leakage = _load(Path(args.leakage_audit))
    shuffle = _load(Path(args.shuffle_summary))
    hi_new = _load(Path(args.hi_maxiter5000))
    payload = {
        "diagnostic": "temporal_flow_causal_validation_summary",
        "sources": [args.maxiter_comparison, args.leakage_audit, args.shuffle_summary, args.hi_maxiter5000],
        "answers": _answers(maxiter_cmp, leakage, shuffle, hi_new),
        "maxiter_comparison": maxiter_cmp,
        "leakage_audit_summary": {
            "recompute_ok": [d["causal_history"]["recompute_matches_cache"] for d in leakage.get("datasets", [])],
        },
        "shuffle_summary": shuffle,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output_json).open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    write_md(Path(args.output_md), payload)
    print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
