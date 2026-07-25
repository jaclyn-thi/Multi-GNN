#!/usr/bin/env python3
"""Summarize temporal-flow aux scout with recall-oriented metrics (diagnostic table)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VARIANTS = ("tf_reg_w0.10", "tf_reg_w0.05", "tf_bins5_w0.10", "tf_bins10_w0.10")
ARMS = ("A_embedding", "B_embedding_raw", "D_embedding_raw_temporal_flow")
REPS = (("post128", "post_embedding_128"), ("pre3h", "pre_embedding_3h"))


def _load(p: Path) -> Optional[Dict[str, Any]]:
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _test(arm: Dict[str, Any]) -> Dict[str, Any]:
    return arm.get("test") or {}


def _fmt(v: Any, nd: int = 4) -> str:
    if v is None:
        return "—"
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    if x != x:  # nan
        return "—"
    return f"{x:.{nd}f}"


def _pick(payload: Dict[str, Any], arm: str) -> Dict[str, Any]:
    return _test((payload.get("arms") or {}).get(arm) or {})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument(
        "--output_json",
        default="results/diagnostics/temporal_flow_aux_objective_scout_recall_metrics.json",
    )
    ap.add_argument(
        "--output_md",
        default="notes/temporal_flow_aux_objective_scout_recall_metrics.md",
    )
    ap.add_argument(
        "--output_table_md",
        default="tables/temporal_flow_aux_objective_scout_recall_metrics.md",
    )
    ap.add_argument(
        "--output_table_tex",
        default="tables/temporal_flow_aux_objective_scout_recall_metrics.tex",
    )
    args = ap.parse_args()
    seed = int(args.seed)

    rows: List[Dict[str, Any]] = []
    missing: List[str] = []
    for variant in VARIANTS:
        run = f"hi_tf_aux_{variant}_gin_emlps_tds_asym_proj_8192neg_queue0_accum4_20ep_seed{seed}"
        entry: Dict[str, Any] = {"variant": variant, "run_name": run, "reps": {}}
        for tag, rep_name in REPS:
            p = ROOT / f"results/diagnostics/enriched/tf_aux_{variant}_{tag}_recall_metrics.json"
            payload = _load(p)
            if payload is None:
                missing.append(str(p.relative_to(ROOT)))
                entry["reps"][rep_name] = None
                continue
            arms = {}
            for arm in ARMS:
                t = _pick(payload, arm)
                arms[arm] = {
                    "auroc": t.get("auroc"),
                    "auprc": t.get("auprc"),
                    "precision_at_100": t.get("precision_at_100"),
                    "recall_at_100": t.get("recall_at_100"),
                    "lift_at_100": t.get("lift_at_100"),
                    "precision_at_500": t.get("precision_at_500"),
                    "recall_at_500": t.get("recall_at_500"),
                    "lift_at_500": t.get("lift_at_500"),
                    "precision_at_1000": t.get("precision_at_1000"),
                    "recall_at_1000": t.get("recall_at_1000"),
                    "lift_at_1000": t.get("lift_at_1000"),
                    "recall_at_precision_ge_0.95": t.get("recall_at_precision_ge_0.95"),
                    "recall_at_precision_ge_0.90": t.get("recall_at_precision_ge_0.90"),
                    "recall_at_precision_ge_0.80": t.get("recall_at_precision_ge_0.80"),
                    "recall_at_precision_ge_0.70": t.get("recall_at_precision_ge_0.70"),
                    "precision_achieved_at_precision_ge_0.90": t.get(
                        "precision_achieved_at_precision_ge_0.90"
                    ),
                    "precision_achieved_at_precision_ge_0.80": t.get(
                        "precision_achieved_at_precision_ge_0.80"
                    ),
                    "n_alerts_at_precision_ge_0.90": t.get("n_alerts_at_precision_ge_0.90"),
                    "n_alerts_at_precision_ge_0.80": t.get("n_alerts_at_precision_ge_0.80"),
                    "threshold_at_precision_ge_0.90": t.get("threshold_at_precision_ge_0.90"),
                    "threshold_at_precision_ge_0.80": t.get("threshold_at_precision_ge_0.80"),
                }
            entry["reps"][rep_name] = {"source_json": str(p.relative_to(ROOT)), "arms": arms}
        rows.append(entry)

    baseline = _load(
        ROOT / "results/diagnostics/enriched/baseline_hi_20ep_seed1_post128_recall_metrics.json"
    )
    baseline_block = None
    if baseline:
        baseline_block = {
            "source_json": "results/diagnostics/enriched/baseline_hi_20ep_seed1_post128_recall_metrics.json",
            "arms": {arm: _pick(baseline, arm) for arm in ARMS if arm in (baseline.get("arms") or {})},
        }

    context_tf = _load(
        ROOT / "results/diagnostics/enriched/tf_validated_hi_40ep_seed2_pre3h_recall_metrics.json"
    )
    context_block = None
    if context_tf:
        context_block = {
            "source_json": "results/diagnostics/enriched/tf_validated_hi_40ep_seed2_pre3h_recall_metrics.json",
            "note": "Previous best-stack context (40ep seed2 pre-3h); not a matched seed1 baseline.",
            "arms": {
                arm: _pick(context_tf, arm)
                for arm in ("A_embedding", "B_embedding_raw", "D_embedding_raw_temporal_flow")
                if arm in (context_tf.get("arms") or {})
            },
        }

    # Rankings among complete variants (pre-3h primary)
    def _auprc(variant: str, rep: str, arm: str) -> float:
        for r in rows:
            if r["variant"] != variant:
                continue
            block = (r["reps"] or {}).get(rep) or {}
            arms = block.get("arms") or {}
            v = (arms.get(arm) or {}).get("auprc")
            return float(v) if v is not None else float("-inf")
        return float("-inf")

    complete = [r["variant"] for r in rows if r["reps"].get("pre_embedding_3h")]
    rankings = {
        "by_pre3h_A_auprc": sorted(complete, key=lambda v: _auprc(v, "pre_embedding_3h", "A_embedding"), reverse=True),
        "by_pre3h_B_auprc": sorted(complete, key=lambda v: _auprc(v, "pre_embedding_3h", "B_embedding_raw"), reverse=True),
        "by_pre3h_A_R500": sorted(
            complete,
            key=lambda v: float(
                ((((next(r for r in rows if r["variant"] == v)["reps"].get("pre_embedding_3h") or {}).get("arms") or {}).get("A_embedding") or {}).get("recall_at_500")
                or float("-inf"))
            ),
            reverse=True,
        ),
    }

    promising = []
    base_a = None
    base_b = None
    if baseline_block:
        base_a = (baseline_block["arms"].get("A_embedding") or {}).get("auprc")
        base_b = (baseline_block["arms"].get("B_embedding_raw") or {}).get("auprc")
    for r in rows:
        post = (r["reps"].get("post_embedding_128") or {}).get("arms") or {}
        a = post.get("A_embedding") or {}
        b = post.get("B_embedding_raw") or {}
        if not a:
            continue
        auprc_ok = True
        if base_a is not None:
            auprc_ok = float(a.get("auprc") or 0) > float(base_a) + 0.005
        recall_ok = (
            float(a.get("recall_at_500") or 0) > 0
            or float(a.get("recall_at_precision_ge_0.80") or 0) > 0
        )
        p100 = float(a.get("precision_at_100") or 0)
        p100_ok = p100 >= 0.5  # do not collapse severely vs typical ~0.7+
        if auprc_ok and recall_ok and p100_ok:
            promising.append(r["variant"])

    out = {
        "diagnostic_only": True,
        "not_in_main_thesis_tables": True,
        "seed": seed,
        "variants": rows,
        "baseline_post128": baseline_block,
        "context_pre3h_tf_validated_40ep_seed2": context_block,
        "rankings": rankings,
        "promising_variants": promising,
        "missing_enriched_json": missing,
        "interpretation_note": (
            "Promising if AUPRC improves and R@500/R@1000 or recall@P>=0.80/0.90 improves "
            "without severe P@100 collapse. Do not rank by recall alone."
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
        "# Temporal-flow aux scout — recall-oriented metrics",
        "",
        "Diagnostic/scout table only — **not** inserted into main thesis tables.",
        "",
        f"Promising variants (heuristic): {', '.join(promising) if promising else '(pending enriched probes)'}",
        "",
        "## Pre-3h embedding-only (A)",
        "",
        "| variant | AUPRC | P@100 | R@500 | R@1000 | R@P≥0.90 | R@P≥0.80 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        a = (((r["reps"].get("pre_embedding_3h") or {}).get("arms") or {}).get("A_embedding") or {})
        lines.append(
            f"| {r['variant']} | {_fmt(a.get('auprc'))} | {_fmt(a.get('precision_at_100'))} | "
            f"{_fmt(a.get('recall_at_500'))} | {_fmt(a.get('recall_at_1000'))} | "
            f"{_fmt(a.get('recall_at_precision_ge_0.90'))} | {_fmt(a.get('recall_at_precision_ge_0.80'))} |"
        )
    lines += [
        "",
        "## Post-128 embedding-only (A) vs baseline",
        "",
        "| variant | AUPRC | P@100 | R@500 | R@1000 | R@P≥0.90 | R@P≥0.80 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    if baseline_block:
        a = baseline_block["arms"].get("A_embedding") or {}
        lines.append(
            f"| baseline (no aux) | {_fmt(a.get('auprc'))} | {_fmt(a.get('precision_at_100'))} | "
            f"{_fmt(a.get('recall_at_500'))} | {_fmt(a.get('recall_at_1000'))} | "
            f"{_fmt(a.get('recall_at_precision_ge_0.90'))} | {_fmt(a.get('recall_at_precision_ge_0.80'))} |"
        )
    for r in rows:
        a = (((r["reps"].get("post_embedding_128") or {}).get("arms") or {}).get("A_embedding") or {})
        lines.append(
            f"| {r['variant']} | {_fmt(a.get('auprc'))} | {_fmt(a.get('precision_at_100'))} | "
            f"{_fmt(a.get('recall_at_500'))} | {_fmt(a.get('recall_at_1000'))} | "
            f"{_fmt(a.get('recall_at_precision_ge_0.90'))} | {_fmt(a.get('recall_at_precision_ge_0.80'))} |"
        )
    if missing:
        lines += ["", f"Missing enriched JSONs: {', '.join(missing)}", ""]
    lines += ["", f"Full JSON: `{out_json.relative_to(ROOT)}`", ""]
    text = "\n".join(lines) + "\n"
    out_md.write_text(text, encoding="utf-8")
    out_tbl.write_text(text, encoding="utf-8")

    # Minimal TeX table (pre-3h A)
    tex = [
        r"% Diagnostic only — not for main thesis tables.",
        r"\begin{tabular}{lrrrrrr}",
        r"\hline",
        r"Variant & AUPRC & P@100 & R@500 & R@1000 & R@P$\ge$0.90 & R@P$\ge$0.80 \\",
        r"\hline",
    ]
    for r in rows:
        a = (((r["reps"].get("pre_embedding_3h") or {}).get("arms") or {}).get("A_embedding") or {})
        tex.append(
            f"{r['variant'].replace('_', r'\_')} & {_fmt(a.get('auprc'))} & {_fmt(a.get('precision_at_100'))} & "
            f"{_fmt(a.get('recall_at_500'))} & {_fmt(a.get('recall_at_1000'))} & "
            f"{_fmt(a.get('recall_at_precision_ge_0.90'))} & {_fmt(a.get('recall_at_precision_ge_0.80'))} \\\\"
        )
    tex += [r"\hline", r"\end{tabular}", ""]
    out_tex.write_text("\n".join(tex), encoding="utf-8")

    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    print(f"Wrote {out_tbl}")
    print(f"Wrote {out_tex}")
    if missing:
        print(f"MISSING: {missing}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
