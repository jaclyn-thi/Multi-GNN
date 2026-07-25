#!/usr/bin/env python3
"""Summarize morphology-objective recall scout (pre-3h primary + recall metrics)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.scout_recall_metrics import extract_recall_oriented

# ARM tags → (display name, run_name template, required?)
ARMS = (
    ("baseline", "baseline re-probe (plain contrastive+proj)", True),
    ("degflow", "degree_fan + flow_balance expert", True),
    ("clustering", "clustering local+global expert", True),
    ("degflow_tfreg", "degflow + TF regression aux (optional)", False),
)

PRIMARY_ARMS = ("A_embedding", "B_embedding_raw", "D_embedding_raw_temporal_flow")


def _run_name(tag: str, seed: int) -> str:
    if tag == "baseline":
        return "hi_contrastive_gin_emlps_tds_proj_asym_8192neg_queue0_accum4_20ep"
    if tag == "degflow":
        return (
            f"hi_morph_obj_degflow_gin_emlps_tds_asym_proj_"
            f"8192neg_queue0_accum4_20ep_seed{seed}"
        )
    if tag == "clustering":
        return (
            f"hi_morph_obj_clustering_gin_emlps_tds_asym_proj_"
            f"8192neg_queue0_accum4_20ep_seed{seed}"
        )
    if tag == "degflow_tfreg":
        return (
            f"hi_morph_obj_degflow_tfreg_w0.05_gin_emlps_tds_asym_proj_"
            f"8192neg_queue0_accum4_20ep_seed{seed}"
        )
    raise ValueError(tag)


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


def _get(row: Dict[str, Any], arm: str, key: str) -> Any:
    return (((row.get("pre_embedding_3h") or {}).get("arms") or {}).get(arm) or {}).get(key)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument(
        "--output_json",
        default="results/diagnostics/morphology_objective_recall_scout.json",
    )
    ap.add_argument(
        "--output_md",
        default="notes/morphology_objective_recall_scout.md",
    )
    ap.add_argument(
        "--output_table_md",
        default="tables/morphology_objective_recall_scout.md",
    )
    ap.add_argument(
        "--output_table_tex",
        default="tables/morphology_objective_recall_scout.tex",
    )
    args = ap.parse_args()
    seed = int(args.seed)

    rows: List[Dict[str, Any]] = []
    missing_required: List[str] = []
    for tag, label, required in ARMS:
        run = _run_name(tag, seed)
        pre_p = ROOT / f"results/diagnostics/morph_obj_{tag}_pre3h_seed{seed}.json"
        post_p = ROOT / f"results/diagnostics/morph_obj_{tag}_post128_seed{seed}.json"
        pre = _load(pre_p)
        post = _load(post_p)
        if pre is None:
            if required:
                missing_required.append(str(pre_p.relative_to(ROOT)))
            rows.append(
                {
                    "variant": tag,
                    "label": label,
                    "run_name": run,
                    "status": "pending",
                    "required": required,
                }
            )
            continue
        entry: Dict[str, Any] = {
            "variant": tag,
            "label": label,
            "run_name": run,
            "status": "complete",
            "required": required,
            "ssl_labels_used": False,
            "thesis_role": "diagnostic_or_scout",
            "validation_status": "diagnostic_only",
            "table_eligible": False,
            "table_group": "morphology_objective_recall_scout",
            "pre_embedding_3h": {
                "source_json": str(pre_p.relative_to(ROOT)),
                "arms": {
                    a: _arm_test(pre, a)
                    for a in PRIMARY_ARMS
                    if a in (pre.get("arms") or {})
                },
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
        rows.append(entry)

    if missing_required:
        print("Missing required probe JSONs:", file=sys.stderr)
        for m in missing_required:
            print(f"  {m}", file=sys.stderr)
        return 1

    complete = [r for r in rows if r.get("status") == "complete"]
    baseline = next((r for r in complete if r["variant"] == "baseline"), None)
    base_a = _get(baseline, "A_embedding", "auprc") if baseline else None
    base_b = _get(baseline, "B_embedding_raw", "auprc") if baseline else None

    def _best(arm: str, key: str = "auprc"):
        scored = []
        for r in complete:
            v = _get(r, arm, key)
            if v is not None:
                scored.append((float(v), r))
        return max(scored, key=lambda x: x[0]) if scored else (None, None)

    best_a_val, best_a_row = _best("A_embedding")
    best_b_val, best_b_row = _best("B_embedding_raw")
    best_d_val, best_d_row = _best("D_embedding_raw_temporal_flow")
    best_r90_val, best_r90_row = _best("A_embedding", "recall_at_precision_ge_0.90")
    best_r80_val, best_r80_row = _best("A_embedding", "recall_at_precision_ge_0.80")

    # Precision-collapse heuristics vs baseline A
    precision_collapse: List[str] = []
    recall_ok: List[str] = []
    if baseline is not None:
        bp100 = _get(baseline, "A_embedding", "precision_at_100")
        for r in complete:
            if r["variant"] == "baseline":
                continue
            a = _get(r, "A_embedding", "auprc")
            p100 = _get(r, "A_embedding", "precision_at_100")
            r500 = _get(r, "A_embedding", "recall_at_500")
            br500 = _get(baseline, "A_embedding", "recall_at_500")
            r90 = _get(r, "A_embedding", "recall_at_precision_ge_0.90")
            br90 = _get(baseline, "A_embedding", "recall_at_precision_ge_0.90")
            improved_a = a is not None and base_a is not None and float(a) > float(base_a)
            improved_recall = (
                (r500 is not None and br500 is not None and float(r500) > float(br500))
                or (r90 is not None and br90 is not None and float(r90) > float(br90))
            )
            if (
                p100 is not None
                and bp100 is not None
                and float(bp100) > 0
                and float(p100) < 0.5 * float(bp100)
            ):
                precision_collapse.append(r["variant"])
            if improved_a or improved_recall:
                if r["variant"] not in precision_collapse:
                    recall_ok.append(r["variant"])

    recommendation = "stop"
    if any(r["variant"] in ("degflow", "clustering") and r["variant"] in recall_ok for r in complete):
        recommendation = "scale"
    if any(r["variant"] == "degflow_tfreg" and r.get("status") == "complete" for r in complete):
        if "degflow_tfreg" in recall_ok:
            recommendation = "combine_with_temporal_flow_aux"
        elif recommendation == "scale":
            recommendation = "scale_morph_only_skip_combo"
    if not recall_ok and complete:
        recommendation = "stop"

    payload = {
        "scout": "morphology_objective_recall",
        "thesis_role": "diagnostic_or_scout",
        "validation_status": "diagnostic_only",
        "table_eligible": False,
        "table_group": "morphology_objective_recall_scout",
        "primary_representation": "pre_embedding_3h",
        "post_128_is_diagnostic_only": True,
        "ssl_labels_used": False,
        "excluded": [
            "morph_contrast_m2",
            "temporal_flow_soft_positives",
            "tier2_betweenness",
        ],
        "seed": seed,
        "baseline_pre3h_a_auprc": base_a,
        "baseline_pre3h_b_auprc": base_b,
        "variants": rows,
        "best_pre3h_embedding_only": {
            "variant": None if best_a_row is None else best_a_row["variant"],
            "auprc": best_a_val,
        },
        "best_pre3h_plus_raw": {
            "variant": None if best_b_row is None else best_b_row["variant"],
            "auprc": best_b_val,
        },
        "best_final_d_stack": {
            "variant": None if best_d_row is None else best_d_row["variant"],
            "auprc": best_d_val,
        },
        "best_recall_oriented": {
            "by_recall_at_precision_ge_0.90": {
                "variant": None if best_r90_row is None else best_r90_row["variant"],
                "value": best_r90_val,
            },
            "by_recall_at_precision_ge_0.80": {
                "variant": None if best_r80_row is None else best_r80_row["variant"],
                "value": best_r80_val,
            },
        },
        "recall_improved_at_acceptable_precision": recall_ok,
        "precision_collapse_variants": precision_collapse,
        "recommendation": recommendation,
    }

    out_json = ROOT / args.output_json
    out_md = ROOT / args.output_md
    out_tbl = ROOT / args.output_table_md
    out_tex = ROOT / args.output_table_tex
    for p in (out_json.parent, out_md.parent, out_tbl.parent, out_tex.parent):
        p.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Morphology-objective recall scout",
        "",
        f"**Thesis role:** diagnostic_or_scout · **validation_status:** diagnostic_only · "
        f"**table_eligible:** false · **table_group:** morphology_objective_recall_scout",
        "",
        "Primary representation: **pre_embedding_3h**. Post-128 is diagnostic only.",
        "SSL: morphology expert **regression** only (no M2 contrast, no TF soft positives, "
        "no tier2/betweenness, **no labels**).",
        "",
        "## Verdict / recommendation",
        "",
        f"- Recommendation: **`{recommendation}`**",
        f"- Recall improved at acceptable precision: `{recall_ok or 'none'}`",
        f"- Precision-collapse variants (P@100 < 50% of baseline): `{precision_collapse or 'none'}`",
        f"- Best pre-3h embedding-only (A AUPRC): "
        f"**{payload['best_pre3h_embedding_only']['variant']}** "
        f"({_fmt(best_a_val)})",
        f"- Best pre-3h + raw (B AUPRC): "
        f"**{payload['best_pre3h_plus_raw']['variant']}** ({_fmt(best_b_val)})",
        f"- Best final D stack AUPRC: "
        f"**{payload['best_final_d_stack']['variant']}** ({_fmt(best_d_val)})",
        "",
        "## Pre-3h primary metrics",
        "",
        "| Variant | Arm | AUROC | AUPRC | F1 | P@100 | R@500 | R@1000 | "
        "R@P≥0.90 | R@P≥0.80 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in complete:
        for arm in PRIMARY_ARMS:
            m = ((r.get("pre_embedding_3h") or {}).get("arms") or {}).get(arm) or {}
            if not m:
                continue
            lines.append(
                f"| {r['variant']} | {arm} | {_fmt(m.get('auroc'))} | {_fmt(m.get('auprc'))} | "
                f"{_fmt(m.get('f1'))} | {_fmt(m.get('precision_at_100'))} | "
                f"{_fmt(m.get('recall_at_500'))} | {_fmt(m.get('recall_at_1000'))} | "
                f"{_fmt(m.get('recall_at_precision_ge_0.90'))} | "
                f"{_fmt(m.get('recall_at_precision_ge_0.80'))} |"
            )
    lines.extend(
        [
            "",
            "## Baseline re-probe",
            "",
            f"- Run: `{_run_name('baseline', seed)}`",
            f"- Pre-3h A AUPRC: {_fmt(base_a)}",
            f"- Pre-3h B AUPRC: {_fmt(base_b)}",
            "",
            "## Notes",
            "",
            "- ARM 1 (`degflow`): `--morph_expert --morph_targets local+global "
            "--morph_flow_balance --morph_target_groups degree_fan,flow_balance "
            "--morph_expert_weight 1.0`",
            "- ARM 2 (`clustering`): `--morph_expert --morph_targets local+global "
            "--morph_local_subset clustering --morph_expert_weight 1.0` "
            "(matches best-known 11-dim clustering+proj style; excludes triangles)",
            "- Optional ARM 3 (`degflow_tfreg`): InfoNCE + λ_morph=0.05 morph MSE + "
            "λ_tf=0.05 TF Huber regression",
            "- Do not insert into main thesis tables yet.",
            "",
        ]
    )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    tbl = [
        "# Morphology-objective recall scout (pre-3h)",
        "",
        "| Variant | A AUPRC | B AUPRC | D AUPRC | A R@500 | A R@P≥0.90 | A P@100 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in complete:
        a = ((r.get("pre_embedding_3h") or {}).get("arms") or {}).get("A_embedding") or {}
        b = ((r.get("pre_embedding_3h") or {}).get("arms") or {}).get("B_embedding_raw") or {}
        d = ((r.get("pre_embedding_3h") or {}).get("arms") or {}).get(
            "D_embedding_raw_temporal_flow"
        ) or {}
        tbl.append(
            f"| {r['variant']} | {_fmt(a.get('auprc'))} | {_fmt(b.get('auprc'))} | "
            f"{_fmt(d.get('auprc'))} | {_fmt(a.get('recall_at_500'))} | "
            f"{_fmt(a.get('recall_at_precision_ge_0.90'))} | "
            f"{_fmt(a.get('precision_at_100'))} |"
        )
    out_tbl.write_text("\n".join(tbl) + "\n", encoding="utf-8")

    tex = [
        r"% Morphology-objective recall scout (diagnostic only; not for main tables)",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Variant & A AUPRC & B AUPRC & D AUPRC & A R@500 & A R@P$\geq$0.90 & A P@100 \\",
        r"\midrule",
    ]
    for r in complete:
        a = ((r.get("pre_embedding_3h") or {}).get("arms") or {}).get("A_embedding") or {}
        b = ((r.get("pre_embedding_3h") or {}).get("arms") or {}).get("B_embedding_raw") or {}
        d = ((r.get("pre_embedding_3h") or {}).get("arms") or {}).get(
            "D_embedding_raw_temporal_flow"
        ) or {}
        vtex = r["variant"].replace("_", r"\_")
        tex.append(
            f"{vtex} & {_fmt(a.get('auprc'))} & {_fmt(b.get('auprc'))} & "
            f"{_fmt(d.get('auprc'))} & {_fmt(a.get('recall_at_500'))} & "
            f"{_fmt(a.get('recall_at_precision_ge_0.90'))} & "
            f"{_fmt(a.get('precision_at_100'))} \\\\"
        )
    tex.extend([r"\bottomrule", r"\end{tabular}", ""])
    out_tex.write_text("\n".join(tex), encoding="utf-8")

    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    print(f"Wrote {out_tbl}")
    print(f"Wrote {out_tex}")
    print(f"recommendation={recommendation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
