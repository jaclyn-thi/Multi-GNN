#!/usr/bin/env python3
"""Summarize shuffle-control temporal_flow validation probes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

METRICS = ("auprc", "f1_at_selected_threshold", "precision_at_100", "lift_at_100")


def _load(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _arm_test(payload: Dict[str, Any], arm: str) -> Dict[str, float]:
    return payload["arms"][arm]["test"]


def _compare_one(
    aligned_path: Path,
    shuffled_path: Path,
    *,
    label: str,
) -> Dict[str, Any]:
    aligned = _load(aligned_path)
    shuffled = _load(shuffled_path)
    b = _arm_test(aligned, "B_embedding_raw")
    d_aligned = _arm_test(aligned, "D_embedding_raw_temporal_flow")
    d_shuf = _arm_test(shuffled, "D_embedding_raw_temporal_flow")
    out: Dict[str, Any] = {"label": label, "metrics": {}}
    for m in METRICS:
        out["metrics"][m] = {
            "B": b.get(m, float("nan")),
            "D_aligned": d_aligned.get(m, float("nan")),
            "D_shuffled": d_shuf.get(m, float("nan")),
            "aligned_minus_B": float(d_aligned.get(m, float("nan")) - b.get(m, float("nan"))),
            "shuffled_minus_B": float(d_shuf.get(m, float("nan")) - b.get(m, float("nan"))),
            "aligned_minus_shuffled": float(d_aligned.get(m, float("nan")) - d_shuf.get(m, float("nan"))),
        }
    sm = out["metrics"]["auprc"]["shuffled_minus_B"]
    am = out["metrics"]["auprc"]["aligned_minus_B"]
    out["sanity"] = {
        "aligned_beats_shuffled_auprc": out["metrics"]["auprc"]["aligned_minus_shuffled"] > 0,
        "shuffled_near_B_auprc": abs(sm) < max(abs(am) * 0.25, 0.01),
        "concern_if_shuffled_retains_gain": (sm > 0.5 * am) if am > 0 else (sm > 0.01),
    }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hi_aligned", default="results/diagnostics/temporal_flow_ablation_small_hi_40ep_seed2_maxiter5000.json")
    p.add_argument("--hi_shuffled", default="results/diagnostics/temporal_flow_shuffle_control_small_hi_seed2.json")
    p.add_argument("--li_aligned_prefix", default="results/diagnostics/temporal_flow_ablation_small_li_seed")
    p.add_argument("--li_aligned_suffix", default="_maxiter5000.json")
    p.add_argument("--li_shuffled_prefix", default="results/diagnostics/temporal_flow_shuffle_control_small_li_seed")
    p.add_argument("--output_json", default="results/diagnostics/temporal_flow_shuffle_control_summary.json")
    p.add_argument("--output_md", default="notes/temporal_flow_shuffle_control_summary.md")
    args = p.parse_args()

    rows = [
        _compare_one(Path(args.hi_aligned), Path(args.hi_shuffled), label="Small-HI seed2"),
    ]
    for seed in (1, 2, 3):
        rows.append(
            _compare_one(
                Path(f"{args.li_aligned_prefix}{seed}{args.li_aligned_suffix}"),
                Path(f"{args.li_shuffled_prefix}{seed}.json"),
                label=f"Small-LI seed{seed}",
            )
        )
    payload = {"diagnostic": "temporal_flow_shuffle_control_summary", "comparisons": rows}
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output_json).open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    lines = ["# Shuffle control summary", ""]
    for r in rows:
        m = r["metrics"]["auprc"]
        lines.append(f"## {r['label']}")
        lines.append(f"- B: {m['B']:.4f}; D aligned: {m['D_aligned']:.4f}; D shuffled: {m['D_shuffled']:.4f}")
        lines.append(f"- aligned−shuffled AUPRC: {m['aligned_minus_shuffled']:+.4f}")
        lines.append(f"- sanity: {r['sanity']}")
        lines.append("")
    Path(args.output_md).write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
