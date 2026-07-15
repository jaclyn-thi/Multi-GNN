#!/usr/bin/env python3
"""Compare temporal_flow ablation results: max_iter=1000 vs max_iter=5000."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ARMS = ("A_embedding", "B_embedding_raw", "C_embedding_temporal_flow", "D_embedding_raw_temporal_flow")
METRICS = (
    "auroc",
    "auprc",
    "f1_at_selected_threshold",
    "precision_at_100",
    "recall_at_100",
    "lift_at_100",
    "precision_at_500",
    "recall_at_500",
    "lift_at_500",
    "precision_at_1000",
    "recall_at_1000",
    "lift_at_1000",
)
THRESHOLDS = {"auprc": 0.005, "f1_at_selected_threshold": 0.01, "precision_at_100": 0.02}


def _load(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _compare_run(old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"run_name": new.get("run_name"), "arms": {}, "deltas": {}}
    for arm in ARMS:
        if arm not in old.get("arms", {}) or arm not in new.get("arms", {}):
            continue
        oa, na = old["arms"][arm], new["arms"][arm]
        arm_cmp: Dict[str, Any] = {
            "threshold_old": oa.get("selected_threshold"),
            "threshold_new": na.get("selected_threshold"),
            "convergence_old": oa.get("convergence"),
            "convergence_new": na.get("convergence"),
            "metric_deltas": {},
        }
        for m in METRICS:
            ov = oa["test"].get(m, float("nan"))
            nv = na["test"].get(m, float("nan"))
            delta = float(nv - ov) if not (math.isnan(ov) or math.isnan(nv)) else float("nan")
            flagged = m in THRESHOLDS and not math.isnan(delta) and abs(delta) > THRESHOLDS[m]
            arm_cmp["metric_deltas"][m] = {"old": ov, "new": nv, "delta": delta, "flagged": flagged}
        out["arms"][arm] = arm_cmp
    for key in ("D_minus_B_primary", "C_minus_A"):
        if key in old.get("deltas", {}) and key in new.get("deltas", {}):
            d: Dict[str, Any] = {}
            for m in METRICS:
                ov = old["deltas"][key].get(m, float("nan"))
                nv = new["deltas"][key].get(m, float("nan"))
                delta = float(nv - ov) if not (math.isnan(ov) or math.isnan(nv)) else float("nan")
                d[m] = {"old": ov, "new": nv, "delta": delta}
            out["deltas"][key] = d
    return out


def _sample_std(vals: List[float]) -> float:
    clean = [float(v) for v in vals if not math.isnan(v)]
    if len(clean) < 2:
        return float("nan")
    m = sum(clean) / len(clean)
    return math.sqrt(sum((x - m) ** 2 for x in clean) / (len(clean) - 1))


def build_summary(pairs: List[Tuple[str, Dict[str, Any], Dict[str, Any]]]) -> Dict[str, Any]:
    comparisons = [{"label": label, "comparison": _compare_run(old, new)} for label, old, new in pairs]
    d_deltas = [
        c["comparison"]["deltas"]["D_minus_B_primary"]["auprc"]["delta"]
        for c in comparisons
        if "D_minus_B_primary" in c["comparison"].get("deltas", {})
    ]
    return {
        "diagnostic": "temporal_flow_maxiter5000_comparison",
        "comparisons": comparisons,
        "aggregate_D_minus_B_auprc_shift": {
            "mean": sum(d_deltas) / len(d_deltas) if d_deltas else float("nan"),
            "sample_std_ddof1": _sample_std(d_deltas),
            "per_run": d_deltas,
        },
        "thresholds": THRESHOLDS,
    }


def write_md(path: Path, summary: Dict[str, Any]) -> None:
    lines = ["# max_iter=5000 vs 1000 comparison", ""]
    for c in summary["comparisons"]:
        lines.append(f"## {c['label']}")
        lines.append("")
        if "D_minus_B_primary" in c["comparison"].get("deltas", {}):
            d = c["comparison"]["deltas"]["D_minus_B_primary"]["auprc"]
            lines.append(f"- ΔAUPRC(D−B) shift: {d['old']:.4f} → {d['new']:.4f} (Δ={d['delta']:+.4f})")
        for arm in ("D_embedding_raw_temporal_flow",):
            if arm not in c["comparison"]["arms"]:
                continue
            conv = c["comparison"]["arms"][arm]["convergence_new"]
            lines.append(f"- Arm D convergence @5000: {conv}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hi_old", default="results/diagnostics/temporal_flow_ablation_small_hi_40ep_seed2.json")
    p.add_argument("--hi_new", default="results/diagnostics/temporal_flow_ablation_small_hi_40ep_seed2_maxiter5000.json")
    p.add_argument("--li_old_prefix", default="results/diagnostics/temporal_flow_ablation_small_li_seed")
    p.add_argument("--li_new_suffix", default="_maxiter5000.json")
    p.add_argument("--output_json", default="results/diagnostics/temporal_flow_ablation_maxiter5000_comparison.json")
    p.add_argument("--output_md", default="notes/temporal_flow_ablation_maxiter5000_comparison.md")
    args = p.parse_args()

    pairs: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = [
        ("Small-HI seed2", _load(Path(args.hi_old)), _load(Path(args.hi_new))),
    ]
    for seed in (1, 2, 3):
        pairs.append(
            (
                f"Small-LI seed{seed}",
                _load(Path(f"{args.li_old_prefix}{seed}.json")),
                _load(Path(f"{args.li_old_prefix}{seed}{args.li_new_suffix}")),
            )
        )
    summary = build_summary(pairs)
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output_json).open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_md(Path(args.output_md), summary)
    print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
