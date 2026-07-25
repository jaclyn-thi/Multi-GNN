#!/usr/bin/env python3
"""Quickcheck: edge_drop_0.05 seed1 vs matched seed1 baseline (pre-3h A/B primary)."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.scout_recall_metrics import extract_recall_oriented

BASELINE_JSON = "results/diagnostics/morph_obj_baseline_pre3h_seed1.json"
SCOUT_JSON = "results/diagnostics/edge_drop_0.05_seed1_pre3h.json"
RESOLVED_JSON = "results/diagnostics/edge_drop_0.05_seed1_resolved_run.json"
SEED2_SCOUT = "results/diagnostics/contrastive_objective_resource_scout.json"


def _load(p: Path) -> Optional[Dict[str, Any]]:
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _arm(payload: Dict[str, Any], arm: str) -> Dict[str, Any]:
    block = ((payload.get("arms") or {}).get(arm) or {})
    t = block.get("test") or {}
    out = {
        "auroc": t.get("auroc"),
        "auprc": t.get("auprc"),
        "f1": t.get("f1_at_selected_threshold"),
    }
    out.update(extract_recall_oriented(t))
    return out


def _fmt(v: Any, nd: int = 4) -> str:
    if v is None:
        return "—"
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    if x != x:
        return "—"
    return f"{x:.{nd}f}"


def _delta(a: Any, b: Any) -> Optional[float]:
    if a is None or b is None:
        return None
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError):
        return None
    if fa != fa or fb != fb:
        return None
    return fa - fb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output_json",
        default="results/diagnostics/edge_drop_0.05_seed1_quickcheck.json",
    )
    ap.add_argument(
        "--output_md",
        default="notes/edge_drop_0.05_seed1_quickcheck.md",
    )
    args = ap.parse_args()

    out_json = ROOT / args.output_json
    out_md = ROOT / args.output_md
    if out_json.is_file():
        print(f"ABORT: refusing overwrite of {out_json}", file=sys.stderr)
        return 1

    base = _load(ROOT / BASELINE_JSON)
    scout = _load(ROOT / SCOUT_JSON)
    resolved = _load(ROOT / RESOLVED_JSON) or {}
    if base is None or scout is None:
        print("ABORT: missing baseline or scout probe JSON", file=sys.stderr)
        return 1

    arms_primary = ("A_embedding", "B_embedding_raw")
    base_arms = {a: _arm(base, a) for a in arms_primary + ("D_embedding_raw_temporal_flow",)}
    scout_arms = {a: _arm(scout, a) for a in arms_primary + ("D_embedding_raw_temporal_flow",)}

    deltas = {
        "A_delta_auprc": _delta(scout_arms["A_embedding"].get("auprc"), base_arms["A_embedding"].get("auprc")),
        "A_delta_p100": _delta(
            scout_arms["A_embedding"].get("precision_at_100"),
            base_arms["A_embedding"].get("precision_at_100"),
        ),
        "A_delta_r_p80": _delta(
            scout_arms["A_embedding"].get("recall_at_precision_ge_0.80"),
            base_arms["A_embedding"].get("recall_at_precision_ge_0.80"),
        ),
        "A_delta_r_p90": _delta(
            scout_arms["A_embedding"].get("recall_at_precision_ge_0.90"),
            base_arms["A_embedding"].get("recall_at_precision_ge_0.90"),
        ),
        "B_delta_auprc": _delta(scout_arms["B_embedding_raw"].get("auprc"), base_arms["B_embedding_raw"].get("auprc")),
        "B_delta_p100": _delta(
            scout_arms["B_embedding_raw"].get("precision_at_100"),
            base_arms["B_embedding_raw"].get("precision_at_100"),
        ),
        "D_delta_auprc": _delta(
            scout_arms["D_embedding_raw_temporal_flow"].get("auprc"),
            base_arms["D_embedding_raw_temporal_flow"].get("auprc"),
        ),
    }

    a_up = deltas["A_delta_auprc"] is not None and deltas["A_delta_auprc"] > 0
    b_up = deltas["B_delta_auprc"] is not None and deltas["B_delta_auprc"] > 0
    primary_success = a_up or b_up
    p100_collapse = (
        deltas["A_delta_p100"] is not None and deltas["A_delta_p100"] < -0.25
    ) or (
        scout_arms["A_embedding"].get("precision_at_100") is not None
        and float(scout_arms["A_embedding"]["precision_at_100"]) < 0.25
    )

    # Seed2 prior result from resource scout
    seed2 = _load(ROOT / SEED2_SCOUT) or {}
    seed2_helped = seed2.get("lower_edge_drop_helped")
    seed2_delta = ((seed2.get("paired_deltas_vs_seed2_baseline") or {}).get("edge_drop") or {})

    if primary_success and not p100_collapse and seed2_helped:
        recommendation = "keep_diagnostic_promising"
    elif primary_success and not p100_collapse:
        recommendation = "keep_diagnostic_seed1_only_mixed"
    else:
        recommendation = "stop_edge_drop_experiments"

    ep = (scout.get("extraction_meta") or {}).get("checkpoint_epoch")

    payload = {
        "scout": "edge_drop_0.05_seed1_quickcheck",
        "thesis_role": "diagnostic_or_scout",
        "validation_status": "diagnostic_only",
        "table_eligible": False,
        "seed": 1,
        "run_name": scout.get("run_name") or "hi_contrastive_edge_drop_0.05_seed1",
        "baseline_run": base.get("run_name"),
        "baseline_json": BASELINE_JSON,
        "scout_json": SCOUT_JSON,
        "ssl_labels_used": False,
        "selected_checkpoint_epoch": ep,
        "resolved": resolved,
        "baseline_pre3h": base_arms,
        "edge_drop_pre3h": scout_arms,
        "paired_deltas_vs_seed1_baseline": deltas,
        "primary_success": primary_success,
        "a_auprc_improved": a_up,
        "b_auprc_improved": b_up,
        "precision_collapse": p100_collapse,
        "seed2_prior": {
            "helped": seed2_helped,
            "deltas": seed2_delta,
            "source": SEED2_SCOUT,
        },
        "recommendation": recommendation,
        "next": (
            "do_not_train_seed3_unless_explicitly_requested"
            if recommendation.startswith("keep")
            else "stop_edge_drop; skip_fanout_and_edge_drop_0.00"
        ),
    }

    def _sanitize(obj: Any) -> Any:
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        return obj

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(_sanitize(payload), indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Edge-drop 0.05 seed1 quickcheck",
        "",
        "**Thesis role:** diagnostic_or_scout · **table_eligible:** false",
        "",
        f"Run: `{payload['run_name']}` · Baseline: `{payload['baseline_run']}`",
        f"Selected checkpoint epoch: **{ep}**",
        "",
        f"## Recommendation: `{recommendation}`",
        "",
        f"- Seed1 primary A/B success: **{primary_success}** (A up={a_up}, B up={b_up})",
        f"- Seed1 P@100 collapse: **{p100_collapse}**",
        f"- Seed2 prior edge_drop helped: **{seed2_helped}**",
        f"- Next: {payload['next']}",
        "",
        "## Training diagnostics",
        "",
        f"- edge_drop_target_rate: `{resolved.get('edge_drop_target_rate', 0.05)}`",
        f"- peak GPU MiB: {_fmt(resolved.get('peak_gpu_mem_mib'), 0)}",
        f"- shared_seed line: `{((resolved.get('log_snippets') or {}).get('shared_seed_line')) or '—'}`",
        f"- edge_drop line: `{((resolved.get('log_snippets') or {}).get('edge_drop_line')) or '—'}`",
        "",
        "## Pre-3h metrics vs matched seed1 baseline",
        "",
        "| Variant | Arm | AUROC | AUPRC | F1 | P@100 | R@100 | Lift@100 | P@500 | R@500 | Lift@500 | P@1000 | R@1000 | Lift@1000 | R@P≥0.90 | R@P≥0.80 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, arms in (("baseline", base_arms), ("edge_drop_0.05", scout_arms)):
        for arm in arms_primary + ("D_embedding_raw_temporal_flow",):
            m = arms[arm]
            # Only emphasize D if primary succeeded; still show numbers
            lines.append(
                f"| {label} | {arm} | {_fmt(m.get('auroc'))} | {_fmt(m.get('auprc'))} | "
                f"{_fmt(m.get('f1'))} | {_fmt(m.get('precision_at_100'))} | "
                f"{_fmt(m.get('recall_at_100'))} | {_fmt(m.get('lift_at_100'))} | "
                f"{_fmt(m.get('precision_at_500'))} | {_fmt(m.get('recall_at_500'))} | "
                f"{_fmt(m.get('lift_at_500'))} | {_fmt(m.get('precision_at_1000'))} | "
                f"{_fmt(m.get('recall_at_1000'))} | {_fmt(m.get('lift_at_1000'))} | "
                f"{_fmt(m.get('recall_at_precision_ge_0.90'))} | "
                f"{_fmt(m.get('recall_at_precision_ge_0.80'))} |"
            )

    lines.extend(
        [
            "",
            "## Paired deltas (edge_drop − seed1 baseline)",
            "",
            f"| ΔA AUPRC | ΔA P@100 | ΔA R@P≥0.80 | ΔA R@P≥0.90 | ΔB AUPRC | ΔB P@100 | ΔD AUPRC |",
            f"|---:|---:|---:|---:|---:|---:|---:|",
            f"| {_fmt(deltas['A_delta_auprc'])} | {_fmt(deltas['A_delta_p100'])} | "
            f"{_fmt(deltas['A_delta_r_p80'])} | {_fmt(deltas['A_delta_r_p90'])} | "
            f"{_fmt(deltas['B_delta_auprc'])} | {_fmt(deltas['B_delta_p100'])} | "
            f"{_fmt(deltas['D_delta_auprc'])} |",
            "",
            "## Notes",
            "",
            "- Primary decision uses pre-3h A/B only.",
            "- D is reported for completeness; do not count D-only gains.",
            "- Post-128 was not extracted/probed.",
            "- Seed3 not trained.",
            "",
        ]
    )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    print(f"recommendation={recommendation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
