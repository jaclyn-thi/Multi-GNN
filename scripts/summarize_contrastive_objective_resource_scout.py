#!/usr/bin/env python3
"""Summarize contrastive objective resource scout (seed2 large_bs + edge_drop).

Diagnostic only; not table-eligible for main thesis tables.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.scout_recall_metrics import extract_recall_oriented

PRIMARY_ARMS = ("A_embedding", "B_embedding_raw", "D_embedding_raw_temporal_flow")
TABLE_GROUP = "contrastive_objective_resource_scout"
BASELINE_RUN = "hi_contrastive_gin_emlps_tds_seed2_proj_asym_8192neg_queue0_accum4_20ep"
VARIANTS = ("large_bs", "edge_drop")


def _load(p: Path) -> Optional[Dict[str, Any]]:
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _arm_test(payload: Dict[str, Any], arm: str) -> Dict[str, Any]:
    arm_block = ((payload.get("arms") or {}).get(arm) or {})
    t = arm_block.get("test") or {}
    out: Dict[str, Any] = {
        "auroc": t.get("auroc"),
        "auprc": t.get("auprc"),
        "f1": t.get("f1_at_selected_threshold"),
        "precision": t.get("precision_at_selected_threshold"),
        "recall": t.get("recall_at_selected_threshold"),
        "selected_threshold": arm_block.get("selected_threshold"),
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


def _ckpt_epoch(payload: Optional[Dict[str, Any]]) -> Any:
    if not payload:
        return None
    return (payload.get("extraction_meta") or {}).get("checkpoint_epoch")


def _get_arm(row: Dict[str, Any], arm: str, key: str) -> Any:
    return (((row.get("pre_embedding_3h") or {}).get("arms") or {}).get(arm) or {}).get(key)


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


def _parse_train_time(log_path: Optional[str]) -> Optional[float]:
    if not log_path:
        return None
    p = ROOT / log_path
    if not p.is_file():
        return None
    text = p.read_text(encoding="utf-8", errors="replace")
    # Prefer wall clock from start/end ISO if present in slurm out; else epoch timings
    starts = re.findall(r"start=([0-9T:\-+.]+)", text)
    ends = re.findall(r"end=([0-9T:\-+.]+)", text)
    # Fallback: look for "Epoch .* took"
    times = [float(x) for x in re.findall(r"took\s+([0-9.]+)\s*s", text, flags=re.I)]
    if times:
        return sum(times)
    return None


def _row_from_probes(
    variant: str,
    run_name: str,
    pre: Dict[str, Any],
    post: Optional[Dict[str, Any]],
    resolved: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "variant": variant,
        "seed": 2,
        "run_name": run_name,
        "status": "complete",
        "ssl_labels_used": False,
        "thesis_role": "diagnostic_or_scout",
        "validation_status": "diagnostic_only",
        "table_eligible": False,
        "table_group": TABLE_GROUP,
        "selected_checkpoint_epoch": _ckpt_epoch(pre),
        "resolved": resolved,
        "train_time_sec_approx": _parse_train_time((resolved or {}).get("train_log")),
        "peak_gpu_mem_mib": (resolved or {}).get("peak_gpu_mem_mib"),
        "pre_embedding_3h": {
            "source_json": f"results/diagnostics/ctr_res_{variant}_pre3h_seed2.json"
            if variant != "baseline"
            else "results/diagnostics/morph_obj_baseline_pre3h_seed2.json",
            "arms": {
                a: _arm_test(pre, a)
                for a in PRIMARY_ARMS
                if a in (pre.get("arms") or {})
            },
        },
        "post_embedding_128_diagnostic": {
            "source_json": f"results/diagnostics/ctr_res_{variant}_post128_seed2.json"
            if variant != "baseline"
            else "results/diagnostics/morph_obj_baseline_post128_seed2.json",
            "arms": {
                a: _arm_test(post, a)
                for a in PRIMARY_ARMS
                if post is not None and a in (post.get("arms") or {})
            },
        }
        if post is not None
        else None,
    }


def _judge(variant: str, deltas: Dict[str, Any], row: Dict[str, Any]) -> Dict[str, Any]:
    a_d = deltas.get("A_delta_auprc")
    b_d = deltas.get("B_delta_auprc")
    p100_d = deltas.get("A_delta_p100")
    d_a = deltas.get("D_delta_auprc")
    d_p100 = deltas.get("D_delta_p100")

    a_up = a_d is not None and a_d > 0
    b_up = b_d is not None and b_d > 0
    primary = a_up or b_up

    a_p100 = _get_arm(row, "A_embedding", "precision_at_100")
    collapse = False
    if p100_d is not None and p100_d < -0.25:
        collapse = True
    if a_p100 is not None and float(a_p100) < 0.25:
        collapse = True

    a_down_hard = a_d is not None and a_d < -0.05
    b_down_hard = b_d is not None and b_d < -0.05
    d_collapse = d_p100 is not None and d_p100 < -0.15

    ep = row.get("selected_checkpoint_epoch")
    early_suspicious = ep is not None and int(ep) <= 2

    if collapse or (a_down_hard and b_down_hard) or d_collapse or early_suspicious:
        verdict = "stop"
    elif primary:
        verdict = "replicate_seeds_1_3"
    else:
        verdict = "stop_or_no_gain"

    return {
        "primary_success": primary,
        "a_auprc_improved": a_up,
        "b_auprc_improved": b_up,
        "precision_collapse": collapse,
        "d_auprc_delta": d_a,
        "early_checkpoint_suspicious": early_suspicious,
        "verdict": verdict,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output_json",
        default="results/diagnostics/contrastive_objective_resource_scout.json",
    )
    ap.add_argument(
        "--output_md",
        default="notes/contrastive_objective_resource_scout.md",
    )
    ap.add_argument(
        "--output_table_md",
        default="tables/contrastive_objective_resource_scout.md",
    )
    ap.add_argument(
        "--output_table_tex",
        default="tables/contrastive_objective_resource_scout.tex",
    )
    args = ap.parse_args()

    out_json = ROOT / args.output_json
    if out_json.is_file():
        print(f"ABORT: refusing overwrite of {out_json}", file=sys.stderr)
        return 1

    base_pre = _load(ROOT / "results/diagnostics/morph_obj_baseline_pre3h_seed2.json")
    base_post = _load(ROOT / "results/diagnostics/morph_obj_baseline_post128_seed2.json")
    if base_pre is None:
        print("ABORT: missing seed2 baseline pre3h probe", file=sys.stderr)
        return 1

    rows: List[Dict[str, Any]] = []
    rows.append(
        _row_from_probes(
            "baseline",
            BASELINE_RUN,
            base_pre,
            base_post,
            {
                "batch_size": 8192,
                "contrastive_accum_steps": 4,
                "edge_drop_target_rate": 0.1,
                "contrastive_num_neg_samples": 8192,
                "contrastive_memory_bank_size": 0,
                "oom_fallback_used": False,
                "note": "matched plain-contrastive seed2; not retrained",
            },
        )
    )
    base = rows[0]

    missing = []
    for variant in VARIANTS:
        resolved = _load(
            ROOT / f"results/diagnostics/contrastive_resource_{variant}_resolved_run_seed2.json"
        )
        pre = _load(ROOT / f"results/diagnostics/ctr_res_{variant}_pre3h_seed2.json")
        post = _load(ROOT / f"results/diagnostics/ctr_res_{variant}_post128_seed2.json")
        if pre is None:
            missing.append(f"results/diagnostics/ctr_res_{variant}_pre3h_seed2.json")
            continue
        run_name = (resolved or {}).get("actual_run_name") or pre.get("run_name")
        rows.append(_row_from_probes(variant, run_name, pre, post, resolved))

    if missing:
        print("Missing required probe JSONs:", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        return 1

    paired: Dict[str, Any] = {}
    judgments: Dict[str, Any] = {}
    for r in rows:
        if r["variant"] == "baseline":
            continue
        dlt = {
            "A_delta_auprc": _delta(
                _get_arm(r, "A_embedding", "auprc"),
                _get_arm(base, "A_embedding", "auprc"),
            ),
            "A_delta_p100": _delta(
                _get_arm(r, "A_embedding", "precision_at_100"),
                _get_arm(base, "A_embedding", "precision_at_100"),
            ),
            "A_delta_r_p80": _delta(
                _get_arm(r, "A_embedding", "recall_at_precision_ge_0.80"),
                _get_arm(base, "A_embedding", "recall_at_precision_ge_0.80"),
            ),
            "A_delta_r_p90": _delta(
                _get_arm(r, "A_embedding", "recall_at_precision_ge_0.90"),
                _get_arm(base, "A_embedding", "recall_at_precision_ge_0.90"),
            ),
            "B_delta_auprc": _delta(
                _get_arm(r, "B_embedding_raw", "auprc"),
                _get_arm(base, "B_embedding_raw", "auprc"),
            ),
            "B_delta_p500": _delta(
                _get_arm(r, "B_embedding_raw", "precision_at_500"),
                _get_arm(base, "B_embedding_raw", "precision_at_500"),
            ),
            "B_delta_r500": _delta(
                _get_arm(r, "B_embedding_raw", "recall_at_500"),
                _get_arm(base, "B_embedding_raw", "recall_at_500"),
            ),
            "D_delta_auprc": _delta(
                _get_arm(r, "D_embedding_raw_temporal_flow", "auprc"),
                _get_arm(base, "D_embedding_raw_temporal_flow", "auprc"),
            ),
            "D_delta_p100": _delta(
                _get_arm(r, "D_embedding_raw_temporal_flow", "precision_at_100"),
                _get_arm(base, "D_embedding_raw_temporal_flow", "precision_at_100"),
            ),
            "D_delta_r_p90": _delta(
                _get_arm(r, "D_embedding_raw_temporal_flow", "recall_at_precision_ge_0.90"),
                _get_arm(base, "D_embedding_raw_temporal_flow", "recall_at_precision_ge_0.90"),
            ),
        }
        paired[r["variant"]] = dlt
        judgments[r["variant"]] = _judge(r["variant"], dlt, r)

    large_helped = judgments.get("large_bs", {}).get("primary_success")
    drop_helped = judgments.get("edge_drop", {}).get("primary_success")

    next_steps = []
    if large_helped:
        next_steps.append("replicate_large_bs_on_seeds_1_and_3")
    if drop_helped:
        next_steps.append("replicate_edge_drop_0.05_on_seeds_1_and_3")
        next_steps.append("consider_edge_drop_0.00_followup")
    if large_helped or drop_helped:
        next_steps.append("defer_fanout_200_until_after_replication")
    else:
        next_steps.append("stop_resource_scouts")
        next_steps.append("skip_fanout_200_for_now")
        next_steps.append("skip_edge_drop_0.00")

    if large_helped and drop_helped:
        overall = "replicate_both"
    elif large_helped:
        overall = "replicate_large_bs_only"
    elif drop_helped:
        overall = "replicate_edge_drop_only"
    else:
        overall = "stop"

    payload = {
        "scout": "contrastive_objective_resource_scout",
        "thesis_role": "diagnostic_or_scout",
        "validation_status": "diagnostic_only",
        "table_eligible": False,
        "table_group": TABLE_GROUP,
        "seed": 2,
        "baseline_run": BASELINE_RUN,
        "baseline_retrained": False,
        "excluded": [
            "fanout_200",
            "edge_drop_0.00",
            "soft_positives",
            "morphology",
            "temporal_flow_bins",
            "hard_negatives",
            "masked_feature_reconstruction",
            "broad_sweep",
        ],
        "audit_finding": (
            "InfoNCE negatives are per microbatch only; accum enlarges optimizer batch, "
            "not the contrastive denominator."
        ),
        "variants": rows,
        "paired_deltas_vs_seed2_baseline": paired,
        "judgments": judgments,
        "true_larger_batch_helped": large_helped,
        "lower_edge_drop_helped": drop_helped,
        "recommendation": overall,
        "next_steps": next_steps,
    }

    def _sanitize(obj: Any) -> Any:
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        return obj

    out_md = ROOT / args.output_md
    out_tbl = ROOT / args.output_table_md
    out_tex = ROOT / args.output_table_tex
    for p in (out_json.parent, out_md.parent, out_tbl.parent, out_tex.parent):
        p.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(_sanitize(payload), indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Contrastive objective resource scout (seed 2)",
        "",
        f"**Thesis role:** diagnostic_or_scout · **validation_status:** diagnostic_only · "
        f"**table_eligible:** false · **table_group:** `{TABLE_GROUP}`",
        "",
        "Matched baseline (not retrained): "
        f"`{BASELINE_RUN}`",
        "",
        "Audit reminder: InfoNCE negatives are **per microbatch only**. "
        "`accum` does not enlarge the contrastive denominator.",
        "",
        "## Recommendation",
        "",
        f"- **Overall: `{overall}`**",
        f"- True larger batch helped: **{large_helped}**",
        f"- Lower edge drop helped: **{drop_helped}**",
        f"- Next: {', '.join(next_steps)}",
        "",
        "## Training / resource diagnostics",
        "",
        "| Variant | Run | bs | accum | edge_drop | OOM fallback | peak GPU MiB | ckpt ep |",
        "|---|---|---:|---:|---:|---|---:|---:|",
    ]
    for r in rows:
        res = r.get("resolved") or {}
        lines.append(
            f"| {r['variant']} | `{r['run_name']}` | {res.get('batch_size', '—')} | "
            f"{res.get('contrastive_accum_steps', '—')} | {res.get('edge_drop_target_rate', '—')} | "
            f"{res.get('oom_fallback_used', False)} | {_fmt(r.get('peak_gpu_mem_mib'), 0)} | "
            f"{r.get('selected_checkpoint_epoch', '—')} |"
        )

    lines.extend(
        [
            "",
            "## Pre-3h metrics (primary)",
            "",
            "| Variant | Arm | AUROC | AUPRC | F1 | P@100 | R@100 | P@500 | R@500 | "
            "P@1000 | R@1000 | R@P≥0.95 | R@P≥0.90 | R@P≥0.80 | R@P≥0.70 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for r in rows:
        for arm in PRIMARY_ARMS:
            m = ((r.get("pre_embedding_3h") or {}).get("arms") or {}).get(arm) or {}
            if not m:
                continue
            lines.append(
                f"| {r['variant']} | {arm} | {_fmt(m.get('auroc'))} | {_fmt(m.get('auprc'))} | "
                f"{_fmt(m.get('f1'))} | {_fmt(m.get('precision_at_100'))} | "
                f"{_fmt(m.get('recall_at_100'))} | {_fmt(m.get('precision_at_500'))} | "
                f"{_fmt(m.get('recall_at_500'))} | {_fmt(m.get('precision_at_1000'))} | "
                f"{_fmt(m.get('recall_at_1000'))} | "
                f"{_fmt(m.get('recall_at_precision_ge_0.95'))} | "
                f"{_fmt(m.get('recall_at_precision_ge_0.90'))} | "
                f"{_fmt(m.get('recall_at_precision_ge_0.80'))} | "
                f"{_fmt(m.get('recall_at_precision_ge_0.70'))} |"
            )

    lines.extend(
        [
            "",
            "## Paired deltas vs seed2 baseline (pre-3h)",
            "",
            "| Variant | ΔA AUPRC | ΔA P@100 | ΔA R@P≥0.80 | ΔB AUPRC | ΔD AUPRC | ΔD P@100 | ΔD R@P≥0.90 | Verdict |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for v in VARIANTS:
        d = paired.get(v, {})
        j = judgments.get(v, {})
        lines.append(
            f"| {v} | {_fmt(d.get('A_delta_auprc'))} | {_fmt(d.get('A_delta_p100'))} | "
            f"{_fmt(d.get('A_delta_r_p80'))} | {_fmt(d.get('B_delta_auprc'))} | "
            f"{_fmt(d.get('D_delta_auprc'))} | {_fmt(d.get('D_delta_p100'))} | "
            f"{_fmt(d.get('D_delta_r_p90'))} | {j.get('verdict', '—')} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Do not count D-only gains as representation improvement.",
            "- Fanout_200 and edge_drop_0.00 were **not** launched in this batch.",
            "- Do not insert into main thesis tables yet.",
            "",
        ]
    )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    tbl = [
        "# Contrastive objective resource scout (pre-3h, seed2)",
        "",
        f"table_group=`{TABLE_GROUP}` · diagnostic_only · recommendation=`{overall}`",
        "",
        "| Variant | A AUPRC | B AUPRC | D AUPRC | A P@100 | A R@P≥0.80 | ΔA AUPRC | ΔB AUPRC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        a = ((r.get("pre_embedding_3h") or {}).get("arms") or {}).get("A_embedding") or {}
        b = ((r.get("pre_embedding_3h") or {}).get("arms") or {}).get("B_embedding_raw") or {}
        darm = ((r.get("pre_embedding_3h") or {}).get("arms") or {}).get(
            "D_embedding_raw_temporal_flow"
        ) or {}
        dlt = paired.get(r["variant"], {})
        tbl.append(
            f"| {r['variant']} | {_fmt(a.get('auprc'))} | {_fmt(b.get('auprc'))} | "
            f"{_fmt(darm.get('auprc'))} | {_fmt(a.get('precision_at_100'))} | "
            f"{_fmt(a.get('recall_at_precision_ge_0.80'))} | "
            f"{_fmt(dlt.get('A_delta_auprc'))} | {_fmt(dlt.get('B_delta_auprc'))} |"
        )
    out_tbl.write_text("\n".join(tbl) + "\n", encoding="utf-8")

    tex = [
        r"% Contrastive objective resource scout (diagnostic only)",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Variant & A AUPRC & B AUPRC & D AUPRC & A P@100 & $\Delta$A & $\Delta$B \\",
        r"\midrule",
    ]
    for r in rows:
        a = ((r.get("pre_embedding_3h") or {}).get("arms") or {}).get("A_embedding") or {}
        b = ((r.get("pre_embedding_3h") or {}).get("arms") or {}).get("B_embedding_raw") or {}
        darm = ((r.get("pre_embedding_3h") or {}).get("arms") or {}).get(
            "D_embedding_raw_temporal_flow"
        ) or {}
        dlt = paired.get(r["variant"], {})
        vtex = str(r["variant"]).replace("_", r"\_")
        tex.append(
            f"{vtex} & {_fmt(a.get('auprc'))} & {_fmt(b.get('auprc'))} & "
            f"{_fmt(darm.get('auprc'))} & {_fmt(a.get('precision_at_100'))} & "
            f"{_fmt(dlt.get('A_delta_auprc'))} & {_fmt(dlt.get('B_delta_auprc'))} \\\\"
        )
    tex.extend([r"\bottomrule", r"\end{tabular}", ""])
    out_tex.write_text("\n".join(tex), encoding="utf-8")

    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    print(f"Wrote {out_tbl}")
    print(f"Wrote {out_tex}")
    print(f"recommendation={overall}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
