#!/usr/bin/env python3
"""Quickcheck: edge_attr_mask_rate=0.05 seed2 vs matched seed2 baseline (pre-3h A/B)."""

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

BASELINE_JSON = "results/diagnostics/morph_obj_baseline_pre3h_seed2.json"
SCOUT_JSON = "results/diagnostics/edge_attr_mask_0.05_seed2_pre3h.json"
RESOLVED_JSON = "results/diagnostics/edge_attr_mask_0.05_seed2_resolved_run.json"


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
        default="results/diagnostics/edge_attr_mask_0.05_seed2_quickcheck.json",
    )
    ap.add_argument(
        "--output_md",
        default="notes/edge_attr_mask_0.05_seed2_quickcheck.md",
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
    all_arms = arms_primary + ("D_embedding_raw_temporal_flow",)
    base_arms = {a: _arm(base, a) for a in all_arms}
    scout_arms = {a: _arm(scout, a) for a in all_arms}

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

    if (not primary_success) or p100_collapse:
        recommendation = "stop_attr_mask_branch"
        next_step = "stop; do not replicate seed1; do not lower mask further without new design"
    elif a_up and b_up and not p100_collapse:
        recommendation = "consider_seed1_replication"
        next_step = "optional one seed1 replication of edge_attr_mask_rate=0.05"
    else:
        recommendation = "keep_diagnostic_promising"
        next_step = "keep diagnostic; B-only or mixed — do not promote; no seed3"

    ep = (scout.get("extraction_meta") or {}).get("checkpoint_epoch")
    payload = {
        "scout": "edge_attr_mask_0.05_seed2_quickcheck",
        "thesis_role": "diagnostic_or_scout",
        "validation_status": "diagnostic_only",
        "table_eligible": False,
        "seed": 2,
        "run_name": scout.get("run_name") or "hi_contrastive_edge_attr_mask_0.05_seed2",
        "baseline_run": base.get("run_name"),
        "baseline_json": BASELINE_JSON,
        "scout_json": SCOUT_JSON,
        "edge_attr_mask_rate": resolved.get("edge_attr_mask_rate", 0.05),
        "edge_drop_target_rate": resolved.get("edge_drop_target_rate", 0.1),
        "ssl_labels_used": False,
        "selected_checkpoint_epoch": ep,
        "resolved": resolved,
        "baseline_pre3h": base_arms,
        "scout_pre3h": scout_arms,
        "paired_deltas_vs_seed2_baseline": deltas,
        "primary_success": primary_success,
        "a_auprc_improved": a_up,
        "b_auprc_improved": b_up,
        "precision_collapse": p100_collapse,
        "recommendation": recommendation,
        "next": next_step,
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
        "# Edge-attr mask 0.05 seed2 quickcheck",
        "",
        "**Thesis role:** diagnostic_or_scout · **validation_status:** diagnostic_only · **table_eligible:** false",
        "",
        f"Run: `{payload['run_name']}` · Baseline: `{payload['baseline_run']}`",
        f"Selected checkpoint epoch: **{ep}**",
        f"edge_attr_mask_rate: **{payload['edge_attr_mask_rate']}** (baseline hardcoded/default 0.1)",
        f"edge_drop_target_rate: **{payload['edge_drop_target_rate']}** (unchanged default)",
        "",
        f"## Recommendation: `{recommendation}`",
        "",
        f"- Primary A/B success: **{primary_success}** (A up={a_up}, B up={b_up})",
        f"- P@100 collapse: **{p100_collapse}**",
        f"- Next: {next_step}",
        "",
        "## Training diagnostics",
        "",
        f"- peak GPU MiB: {_fmt(resolved.get('peak_gpu_mem_mib'), 0)}",
        f"- shared_seed: `{((resolved.get('log_snippets') or {}).get('shared_seed_line')) or '—'}`",
        f"- view_aug log: `{((resolved.get('log_snippets') or {}).get('view_aug_line')) or '—'}`",
        "",
        "## Pre-3h metrics vs matched seed2 baseline",
        "",
        "| Variant | Arm | AUROC | AUPRC | F1 | P@100 | R@100 | Lift@100 | P@500 | R@500 | Lift@500 | P@1000 | R@1000 | Lift@1000 | R@P≥0.90 | R@P≥0.80 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, arms in (("baseline", base_arms), ("attr_mask_0.05", scout_arms)):
        for arm in all_arms:
            m = arms[arm]
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
            "## Paired deltas (attr_mask_0.05 − seed2 baseline)",
            "",
            "| ΔA AUPRC | ΔA P@100 | ΔA R@P≥0.80 | ΔA R@P≥0.90 | ΔB AUPRC | ΔB P@100 | ΔD AUPRC |",
            "|---:|---:|---:|---:|---:|---:|---:|",
            f"| {_fmt(deltas['A_delta_auprc'])} | {_fmt(deltas['A_delta_p100'])} | "
            f"{_fmt(deltas['A_delta_r_p80'])} | {_fmt(deltas['A_delta_r_p90'])} | "
            f"{_fmt(deltas['B_delta_auprc'])} | {_fmt(deltas['B_delta_p100'])} | "
            f"{_fmt(deltas['D_delta_auprc'])} |",
            "",
            "## Notes",
            "",
            "- Primary decision uses pre-3h A/B only.",
            "- D reported for completeness only if present; do not count D-only gains.",
            "- Post-128 not extracted/probed.",
            "- Not table-eligible.",
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
