#!/usr/bin/env python3
"""Summarize temporal-flow soft-positive scout (pre-3h primary + recall metrics)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.scout_recall_metrics import RECALL_ORIENTED_METRIC_KEYS, extract_recall_oriented

VARIANTS = (
    "tf_soft_bins5_min3_cap16_w0.05",
    "tf_soft_bins5_min4_cap16_w0.10",
    "tf_soft_bins10_min4_cap32_w0.05",
    "tf_soft_strict_bins10_min5_cap4_w0.01",
)

# Approximate seed-1 pre-3h baselines from the same GIN recipe family
# (40ep strong-run comparison; soft-positive scout is 20ep — used only as diagnostic reference).
BASELINE_PRE3H_A_AUPRC = 0.2244
BASELINE_PRE3H_B_AUPRC = 0.2737
PRIMARY_ARMS = ("A_embedding", "B_embedding_raw", "D_embedding_raw_temporal_flow")


def _load(p: Path) -> Optional[Dict[str, Any]]:
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _arm_test(payload: Dict[str, Any], arm: str) -> Dict[str, Any]:
    t = ((payload.get("arms") or {}).get(arm) or {}).get("test") or {}
    out = {
        "auroc": t.get("auroc"),
        "auprc": t.get("auprc"),
        "f1": t.get("f1_at_selected_threshold"),
        "precision": t.get("precision_at_selected_threshold"),
        "recall": t.get("recall_at_selected_threshold"),
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument(
        "--output_json",
        default="results/diagnostics/temporal_flow_soft_positive_scout.json",
    )
    ap.add_argument(
        "--output_md",
        default="notes/temporal_flow_soft_positive_scout.md",
    )
    ap.add_argument(
        "--output_table_md",
        default="tables/temporal_flow_soft_positive_scout.md",
    )
    ap.add_argument(
        "--output_table_tex",
        default="tables/temporal_flow_soft_positive_scout.tex",
    )
    args = ap.parse_args()
    seed = int(args.seed)

    rows: List[Dict[str, Any]] = []
    missing: List[str] = []
    for variant in VARIANTS:
        run = (
            f"hi_tf_soft_{variant}_optv2_gin_emlps_tds_asym_proj_"
            f"8192neg_queue0_accum4_20ep_seed{seed}"
        )
        pre_p = ROOT / f"results/diagnostics/tf_soft_{variant}_pre3h_seed{seed}.json"
        post_p = ROOT / f"results/diagnostics/tf_soft_{variant}_post128_seed{seed}.json"
        pre = _load(pre_p)
        post = _load(post_p)
        if pre is None:
            missing.append(str(pre_p.relative_to(ROOT)))
            rows.append({"variant": variant, "run_name": run, "status": "pending"})
            continue
        entry: Dict[str, Any] = {
            "variant": variant,
            "run_name": run,
            "status": "complete",
            "ssl_labels_used": False,
            "pre_embedding_3h": {
                "source_json": str(pre_p.relative_to(ROOT)),
                "arms": {a: _arm_test(pre, a) for a in PRIMARY_ARMS if a in (pre.get("arms") or {})},
            },
            "post_embedding_128_diagnostic": None,
        }
        if post is not None:
            entry["post_embedding_128_diagnostic"] = {
                "source_json": str(post_p.relative_to(ROOT)),
                "arms": {
                    a: _arm_test(post, a)
                    for a in ("A_embedding", "B_embedding_raw")
                    if a in (post.get("arms") or {})
                },
            }
        # soft-pos preprocess meta if present
        meta_p = ROOT / f"results/diagnostics/temporal_flow_soft_positive_preprocess/{run}_soft_pos_bins.json"
        if meta_p.is_file():
            entry["soft_positive_meta"] = json.loads(meta_p.read_text(encoding="utf-8"))
        rows.append(entry)

    baseline = _load(
        ROOT / "results/diagnostics/enriched/baseline_hi_20ep_seed1_post128_recall_metrics.json"
    )
    # Prefer a pre-3h baseline if/when available from validated TF run
    pre_base = _load(
        ROOT / "results/diagnostics/enriched/tf_validated_hi_40ep_seed2_pre3h_recall_metrics.json"
    )

    def pre_auprc(variant: str, arm: str) -> float:
        for r in rows:
            if r.get("variant") != variant or r.get("status") != "complete":
                continue
            v = (((r.get("pre_embedding_3h") or {}).get("arms") or {}).get(arm) or {}).get("auprc")
            return float(v) if v is not None else float("-inf")
        return float("-inf")

    complete = [r["variant"] for r in rows if r.get("status") == "complete"]
    rankings = {
        "by_pre3h_A_auprc": sorted(complete, key=lambda v: pre_auprc(v, "A_embedding"), reverse=True),
        "by_pre3h_B_auprc": sorted(complete, key=lambda v: pre_auprc(v, "B_embedding_raw"), reverse=True),
        "by_pre3h_D_auprc": sorted(
            complete, key=lambda v: pre_auprc(v, "D_embedding_raw_temporal_flow"), reverse=True
        ),
        "by_pre3h_A_R500": sorted(
            complete,
            key=lambda v: float(
                ((((next(r for r in rows if r["variant"] == v).get("pre_embedding_3h") or {}).get("arms") or {}).get("A_embedding") or {}).get("recall_at_500")
                or float("-inf"))
            ),
            reverse=True,
        ),
        "by_pre3h_A_R_at_P90": sorted(
            complete,
            key=lambda v: float(
                ((((next(r for r in rows if r["variant"] == v).get("pre_embedding_3h") or {}).get("arms") or {}).get("A_embedding") or {}).get("recall_at_precision_ge_0.90")
                or float("-inf"))
            ),
            reverse=True,
        ),
    }

    # Diagnostic verdict vs approximate pre-3h baseline (same recipe family).
    main_variants = [v for v in complete if not v.startswith("tf_soft_strict_")]
    any_a_beat = any(pre_auprc(v, "A_embedding") > BASELINE_PRE3H_A_AUPRC for v in main_variants)
    any_b_beat = any(pre_auprc(v, "B_embedding_raw") > BASELINE_PRE3H_B_AUPRC for v in main_variants)
    verdict = {
        "thesis_role": "negative_result",
        "validation_status": "diagnostic_only",
        "table_eligible": False,
        "not_in_main_thesis_tables": True,
        "main_scout_passed": False,
        "reasons": [
            "pre-3h embedding-only AUPRC below baseline (~0.224) for every main scout variant",
            "pre-3h + raw did not beat baseline (~0.274)",
            "P@100 collapsed vs baseline (~0.76)",
            "recall at precision >= 0.90 never achieved; R@P>=0.80 only trivial",
            "soft-positive caps saturated (avg ≈ max_per_anchor) — positives too broad/low-quality",
        ],
        "baseline_pre3h_a_auprc_ref": BASELINE_PRE3H_A_AUPRC,
        "baseline_pre3h_b_auprc_ref": BASELINE_PRE3H_B_AUPRC,
        "any_main_variant_beat_baseline_A": any_a_beat,
        "any_main_variant_beat_baseline_B": any_b_beat,
        "stop_rule": (
            "If the optional strict scarce-positive sanity variant still saturates "
            "or underperforms, stop soft-positive experiments."
        ),
    }

    out = {
        "diagnostic_only": True,
        "not_in_main_thesis_tables": True,
        "thesis_role": "negative_result",
        "validation_status": "diagnostic_only",
        "table_eligible": False,
        "primary_representation": "pre_embedding_3h",
        "post_128_is_diagnostic_only": True,
        "seed": seed,
        "ssl_no_label_use": True,
        "required_recall_metric_keys": RECALL_ORIENTED_METRIC_KEYS,
        "variants": rows,
        "missing": missing,
        "rankings": rankings,
        "best_pre3h_embedding_only": rankings["by_pre3h_A_auprc"][:1],
        "best_pre3h_plus_raw": rankings["by_pre3h_B_auprc"][:1],
        "best_final_stack": rankings["by_pre3h_D_auprc"][:1],
        "best_recall_oriented": rankings["by_pre3h_A_R_at_P90"][:1],
        "verdict": verdict,
        "baseline_post128_enriched": (
            str(Path("results/diagnostics/enriched/baseline_hi_20ep_seed1_post128_recall_metrics.json"))
            if baseline
            else None
        ),
        "context_pre3h_validated_40ep_seed2": (
            str(Path("results/diagnostics/enriched/tf_validated_hi_40ep_seed2_pre3h_recall_metrics.json"))
            if pre_base
            else None
        ),
        "success_criterion": (
            "Primary: improve pre-3h A and/or B AUPRC; improve R@500/R@1000 or recall@P>=0.80/0.90 "
            "without severe P@100 collapse; do not count D-only gains as SSL success."
        ),
    }

    out_json = ROOT / args.output_json
    out_md = ROOT / args.output_md
    out_tbl = ROOT / args.output_table_md
    out_tex = ROOT / args.output_table_tex
    for p in (out_json, out_md, out_tbl, out_tex):
        p.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Temporal-flow soft-positive scout",
        "",
        "## Verdict",
        "",
        "**Negative result** (`thesis_role=negative_result`, `validation_status=diagnostic_only`).",
        "Do **not** insert into main thesis tables.",
        "",
        "Main scout (A/B/C) failed primary criteria:",
        "",
        "- pre-3h embedding-only AUPRC below baseline (~0.224) for every variant",
        "- pre-3h + raw did not beat baseline (~0.274)",
        "- P@100 collapsed vs baseline (~0.76)",
        "- recall at precision ≥ 0.90 never achieved; R@P≥0.80 only trivial",
        "- soft-positive caps saturated (avg ≈ max_per_anchor) — positives too broad/low-quality",
        "",
        "Optional single scarce-positive sanity: `tf_soft_strict_bins10_min5_cap4_w0.01` "
        "(min_shared=5 = all 5 TF features, cap=4, w=0.01). "
        "If it still saturates or underperforms, stop soft-positive experiments.",
        "",
        "- Primary representation: **pre-3h** (post-128 is diagnostic only)",
        "- SSL soft positives use causal `temporal_flow_causal` bins; **no labels**",
        "- Identity pair remains primary; TF soft positives are low-weight extras",
        "",
        f"- Best pre-3h embedding-only: **{(rankings['by_pre3h_A_auprc'] or ['—'])[0]}**",
        f"- Best pre-3h + raw: **{(rankings['by_pre3h_B_auprc'] or ['—'])[0]}**",
        f"- Best final stack (D): **{(rankings['by_pre3h_D_auprc'] or ['—'])[0]}**",
        f"- Best recall-oriented (A R@P≥0.90): **{(rankings['by_pre3h_A_R_at_P90'] or ['—'])[0]}**",
        "",
        "## Pre-3h primary metrics",
        "",
        "| variant | arm | AUPRC | P@100 | R@500 | R@1000 | R@P≥0.90 | R@P≥0.80 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        if r.get("status") != "complete":
            lines.append(f"| {r['variant']} | — | pending | | | | | |")
            continue
        for arm in PRIMARY_ARMS:
            m = ((r.get("pre_embedding_3h") or {}).get("arms") or {}).get(arm) or {}
            lines.append(
                f"| {r['variant']} | {arm} | {_fmt(m.get('auprc'))} | {_fmt(m.get('precision_at_100'))} | "
                f"{_fmt(m.get('recall_at_500'))} | {_fmt(m.get('recall_at_1000'))} | "
                f"{_fmt(m.get('recall_at_precision_ge_0.90'))} | {_fmt(m.get('recall_at_precision_ge_0.80'))} |"
            )
    if missing:
        lines += ["", f"Missing: {', '.join(missing)}", ""]
    lines += ["", f"Full JSON: `{out_json.relative_to(ROOT)}`", ""]
    text = "\n".join(lines) + "\n"
    out_md.write_text(text, encoding="utf-8")
    out_tbl.write_text(text, encoding="utf-8")

    tex = [
        r"% Diagnostic only — soft-positive scout; pre-3h primary.",
        r"\begin{tabular}{llrrrrrr}",
        r"\hline",
        r"Variant & Arm & AUPRC & P@100 & R@500 & R@1000 & R@P$\ge$0.90 & R@P$\ge$0.80 \\",
        r"\hline",
    ]
    for r in rows:
        if r.get("status") != "complete":
            continue
        for arm in ("A_embedding", "B_embedding_raw"):
            m = ((r.get("pre_embedding_3h") or {}).get("arms") or {}).get(arm) or {}
            variant_tex = str(r["variant"]).replace("_", r"\_")
            arm_tex = arm.replace("_", r"\_")
            tex.append(
                f"{variant_tex} & {arm_tex} & "
                f"{_fmt(m.get('auprc'))} & {_fmt(m.get('precision_at_100'))} & "
                f"{_fmt(m.get('recall_at_500'))} & {_fmt(m.get('recall_at_1000'))} & "
                f"{_fmt(m.get('recall_at_precision_ge_0.90'))} & {_fmt(m.get('recall_at_precision_ge_0.80'))} \\\\"
            )
    tex += [r"\hline", r"\end{tabular}", ""]
    out_tex.write_text("\n".join(tex), encoding="utf-8")

    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    main_missing = [m for m in missing if "strict" not in m]
    if missing:
        print(f"MISSING: {missing}")
    if main_missing:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
