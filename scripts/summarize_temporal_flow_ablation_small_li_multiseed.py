#!/usr/bin/env python3
"""Multiseed summary for Small-LI temporal_flow_causal ablation (seeds 1–3)."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PRIMARY_DELTA = "D_minus_B_primary"
METRICS = (
    "auprc",
    "f1_at_selected_threshold",
    "precision_at_100",
    "recall_at_100",
    "lift_at_100",
)


def _load(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _sample_std(vals: List[float]) -> float:
    clean = [float(v) for v in vals if isinstance(v, (int, float)) and not math.isnan(v)]
    if len(clean) < 2:
        return float("nan")
    m = sum(clean) / len(clean)
    return math.sqrt(sum((x - m) ** 2 for x in clean) / (len(clean) - 1))


def _mean(vals: List[float]) -> float:
    clean = [float(v) for v in vals if isinstance(v, (int, float)) and not math.isnan(v)]
    return float("nan") if not clean else sum(clean) / len(clean)


def _fmt(x: Any, nd: int = 4) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if math.isnan(v) else f"{v:.{nd}f}"


def build_summary(seed_payloads: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    per_seed: List[Dict[str, Any]] = []
    for seed in sorted(seed_payloads):
        p = seed_payloads[seed]
        d_primary = p["deltas"][PRIMARY_DELTA]
        d_c_a = p["deltas"]["C_minus_A"]
        per_seed.append(
            {
                "seed": seed,
                "run_name": p.get("run_name"),
                "D_minus_B": d_primary,
                "C_minus_A": d_c_a,
                "arms": {
                    k: v["test"] for k, v in p.get("arms", {}).items()
                },
            }
        )

    aggregate: Dict[str, Any] = {}
    for label, key in (("D_minus_B", PRIMARY_DELTA), ("C_minus_A", "C_minus_A")):
        agg: Dict[str, Any] = {"n": len(per_seed), "metrics": {}}
        for metric in METRICS:
            vals = [s[label if label != "D_minus_B" else "D_minus_B"][metric] for s in per_seed]
            agg["metrics"][metric] = {
                "mean": _mean(vals),
                "sample_std_ddof1": _sample_std(vals),
                "per_seed": vals,
            }
        aggregate[label] = agg

    wins = sum(1 for s in per_seed if s["D_minus_B"]["auprc"] > 0)
    return {
        "diagnostic": "temporal_flow_ablation_small_li_multiseed",
        "data": "Small-LI",
        "n_seeds": len(per_seed),
        "seeds": sorted(seed_payloads.keys()),
        "per_seed": per_seed,
        "aggregate": aggregate,
        "D_beats_B_auprc_count": wins,
        "conclusions": {
            "direction_consistent": wins == len(per_seed),
            "conservative_read": (
                f"{len(per_seed)} seeds; downstream probe only; treat |ΔAUPRC| < 0.005 as noise."
            ),
        },
    }


def write_md(path: Path, summary: Dict[str, Any]) -> None:
    agg = summary["aggregate"]["D_minus_B"]["metrics"]
    lines = [
        "# Temporal flow causal ablation — Small-LI multiseed",
        "",
        f"**Seeds:** {summary['seeds']} (n={summary['n_seeds']})",
        "",
        "## Primary Δ (Arm D − Arm B) — mean ± sample SD (ddof=1)",
        "",
        "| metric | mean | sample SD | per-seed |",
        "|--------|-----:|----------:|----------|",
    ]
    for metric in METRICS:
        m = agg[metric]
        per = ", ".join(f"{v:+.4f}" if metric != "lift_at_100" else f"{v:+.2f}" for v in m["per_seed"])
        lines.append(
            f"| {metric} | {_fmt(m['mean'])} | {_fmt(m['sample_std_ddof1'])} | {per} |"
        )
    lines.append("")
    lines.append(f"- D beats B on AUPRC in **{summary['D_beats_B_auprc_count']}/{summary['n_seeds']}** seeds")
    lines.append("")
    for s in summary["per_seed"]:
        lines.append(f"## Seed {s['seed']} — `{s['run_name']}`")
        lines.append("")
        lines.append("| arm | AUPRC | F1 | P@100 | lift@100 |")
        lines.append("|-----|------:|---:|------:|---------:|")
        for arm, t in s["arms"].items():
            lines.append(
                f"| {arm} | {t['auprc']:.4f} | {t['f1_at_selected_threshold']:.4f} | "
                f"{t.get('precision_at_100', float('nan')):.4f} | {t.get('lift_at_100', float('nan')):.2f} |"
            )
        d = s["D_minus_B"]
        lines.append("")
        lines.append(
            f"ΔAUPRC(D−B)={d['auprc']:+.4f}; ΔF1={d['f1_at_selected_threshold']:+.4f}; "
            f"Δlift@100={d.get('lift_at_100', float('nan')):+.2f}"
        )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed1_json", default="results/diagnostics/temporal_flow_ablation_small_li_seed1.json")
    p.add_argument("--seed2_json", default="results/diagnostics/temporal_flow_ablation_small_li_seed2.json")
    p.add_argument("--seed3_json", default="results/diagnostics/temporal_flow_ablation_small_li_seed3.json")
    p.add_argument("--output_json", default="results/diagnostics/temporal_flow_ablation_small_li_multiseed.json")
    p.add_argument("--output_md", default="notes/temporal_flow_ablation_small_li_multiseed.md")
    args = p.parse_args()

    seed_payloads: Dict[int, Dict[str, Any]] = {}
    for seed, path_str in ((1, args.seed1_json), (2, args.seed2_json), (3, args.seed3_json)):
        payload = _load(Path(path_str))
        if payload is not None:
            seed_payloads[seed] = payload

    if len(seed_payloads) < 3:
        missing = [s for s in (1, 2, 3) if s not in seed_payloads]
        raise FileNotFoundError(f"Missing seed JSONs: {missing}")

    summary = build_summary(seed_payloads)
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_md(Path(args.output_md), summary)
    print(f"Wrote {out} and {args.output_md}")


if __name__ == "__main__":
    main()
