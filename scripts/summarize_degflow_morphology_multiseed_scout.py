#!/usr/bin/env python3
"""Summarize degflow morphology multiseed scout (seeds 1–3).

Separates two claims:
  Claim 1 — Representation improvement (pre-3h A / B before TF features)
  Claim 2 — Final-stack D tradeoff (AUPRC vs high-precision operating points)

Diagnostic only; not table-eligible for main thesis tables.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.scout_recall_metrics import extract_recall_oriented

PRIMARY_ARMS = ("A_embedding", "B_embedding_raw", "D_embedding_raw_temporal_flow")
TABLE_GROUP = "degflow_morphology_multiseed_scout"

DEGFLOW_FLAGS = (
    "--morph_expert --morph_targets local+global --morph_flow_balance "
    "--morph_target_groups degree_fan,flow_balance --morph_expert_weight 1.0 "
    "--morph_expert_hidden 64"
)


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
        "val_f1_at_selected_threshold": arm_block.get("val_f1_at_selected_threshold"),
        "convergence": arm_block.get("convergence"),
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
    if x != x:  # NaN
        return "—"
    return f"{x:.{nd}f}"


def _mean_sd(vals: Sequence[float]) -> Dict[str, Any]:
    xs = [float(v) for v in vals if v is not None and v == v]
    if not xs:
        return {"n": 0, "mean": None, "sd": None, "values": []}
    mean = sum(xs) / len(xs)
    if len(xs) < 2:
        return {"n": len(xs), "mean": mean, "sd": None, "values": xs}
    var = sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)
    return {"n": len(xs), "mean": mean, "sd": math.sqrt(var), "values": xs}


def _get_arm(row: Dict[str, Any], arm: str, key: str) -> Any:
    return (((row.get("pre_embedding_3h") or {}).get("arms") or {}).get(arm) or {}).get(key)


def _ckpt_epoch(payload: Optional[Dict[str, Any]]) -> Any:
    if not payload:
        return None
    return (payload.get("extraction_meta") or {}).get("checkpoint_epoch")


def _variant_row(
    tag: str,
    seed: int,
    run_name: str,
    pre: Optional[Dict[str, Any]],
    post: Optional[Dict[str, Any]],
    required: bool,
) -> Dict[str, Any]:
    if pre is None:
        return {
            "variant": tag,
            "seed": seed,
            "run_name": run_name,
            "status": "pending" if required else "missing_optional",
            "required": required,
        }
    entry: Dict[str, Any] = {
        "variant": tag,
        "seed": seed,
        "run_name": run_name,
        "status": "complete",
        "required": required,
        "ssl_labels_used": False,
        "thesis_role": "diagnostic_or_scout",
        "validation_status": "diagnostic_only",
        "table_eligible": False,
        "table_group": TABLE_GROUP,
        "selected_checkpoint_epoch": _ckpt_epoch(pre),
        "pre_embedding_3h": {
            "source_json": f"results/diagnostics/morph_obj_{tag}_pre3h_seed{seed}.json",
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
            "source_json": f"results/diagnostics/morph_obj_{tag}_post128_seed{seed}.json",
            "arms": {
                a: _arm_test(post, a)
                for a in ("A_embedding", "B_embedding_raw")
                if a in (post.get("arms") or {})
            },
        }
    return entry


def _degflow_run(seed: int) -> str:
    return (
        f"hi_morph_obj_degflow_gin_emlps_tds_asym_proj_"
        f"8192neg_queue0_accum4_20ep_seed{seed}"
    )


def _baseline_run(seed: int) -> Optional[str]:
    if seed == 1:
        return "hi_contrastive_gin_emlps_tds_proj_asym_8192neg_queue0_accum4_20ep"
    if seed == 2:
        return "hi_contrastive_gin_emlps_tds_seed2_proj_asym_8192neg_queue0_accum4_20ep"
    return None


def _paired_delta(
    deg: Dict[str, Any], base: Dict[str, Any], arm: str, key: str
) -> Optional[float]:
    a = _get_arm(deg, arm, key)
    b = _get_arm(base, arm, key)
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
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument(
        "--require_degflow_seeds",
        type=int,
        nargs="*",
        default=[2, 3],
        help="Abort if these degflow seed probe JSONs are missing",
    )
    ap.add_argument(
        "--output_json",
        default="results/diagnostics/degflow_morphology_multiseed_scout.json",
    )
    ap.add_argument(
        "--output_md",
        default="notes/degflow_morphology_multiseed_scout.md",
    )
    ap.add_argument(
        "--output_table_md",
        default="tables/degflow_morphology_multiseed_scout.md",
    )
    ap.add_argument(
        "--output_table_tex",
        default="tables/degflow_morphology_multiseed_scout.tex",
    )
    args = ap.parse_args()

    out_json = ROOT / args.output_json
    if out_json.is_file():
        print(f"ABORT: refusing overwrite of {out_json}", file=sys.stderr)
        return 1

    seeds = list(args.seeds)
    require = set(args.require_degflow_seeds or [])

    rows: List[Dict[str, Any]] = []
    missing_required: List[str] = []

    for seed in seeds:
        # baseline (optional except seed1 which already exists)
        base_run = _baseline_run(seed)
        if base_run is not None:
            pre_p = ROOT / f"results/diagnostics/morph_obj_baseline_pre3h_seed{seed}.json"
            post_p = ROOT / f"results/diagnostics/morph_obj_baseline_post128_seed{seed}.json"
            pre = _load(pre_p)
            post = _load(post_p)
            required_base = seed == 1
            if pre is None and required_base:
                missing_required.append(str(pre_p.relative_to(ROOT)))
            rows.append(
                _variant_row("baseline", seed, base_run, pre, post, required_base)
            )
        else:
            rows.append(
                {
                    "variant": "baseline",
                    "seed": seed,
                    "run_name": None,
                    "status": "unavailable",
                    "required": False,
                    "note": "no matched 20ep plain-contrastive checkpoint; not retrained",
                }
            )

        # degflow (required for listed seeds)
        deg_run = _degflow_run(seed)
        pre_p = ROOT / f"results/diagnostics/morph_obj_degflow_pre3h_seed{seed}.json"
        post_p = ROOT / f"results/diagnostics/morph_obj_degflow_post128_seed{seed}.json"
        pre = _load(pre_p)
        post = _load(post_p)
        required = seed in require or seed == 1
        if pre is None and required:
            missing_required.append(str(pre_p.relative_to(ROOT)))
        rows.append(_variant_row("degflow", seed, deg_run, pre, post, required))

    if missing_required:
        print("Missing required probe JSONs:", file=sys.stderr)
        for m in missing_required:
            print(f"  {m}", file=sys.stderr)
        return 1

    deg_complete = [
        r for r in rows if r.get("variant") == "degflow" and r.get("status") == "complete"
    ]
    base_complete = [
        r
        for r in rows
        if r.get("variant") == "baseline" and r.get("status") == "complete"
    ]
    base_by_seed = {int(r["seed"]): r for r in base_complete}
    deg_by_seed = {int(r["seed"]): r for r in deg_complete}

    # Aggregate degflow metrics across seeds
    def _agg(arm: str, key: str) -> Dict[str, Any]:
        vals = []
        for r in deg_complete:
            v = _get_arm(r, arm, key)
            if v is not None:
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                if fv == fv:
                    vals.append(fv)
        return _mean_sd(vals)

    agg_keys = [
        "auprc",
        "auroc",
        "precision_at_100",
        "recall_at_100",
        "lift_at_100",
        "precision_at_500",
        "recall_at_500",
        "lift_at_500",
        "precision_at_1000",
        "recall_at_1000",
        "lift_at_1000",
        "recall_at_precision_ge_0.95",
        "recall_at_precision_ge_0.90",
        "recall_at_precision_ge_0.80",
        "recall_at_precision_ge_0.70",
    ]
    aggregates: Dict[str, Any] = {}
    for arm in PRIMARY_ARMS:
        aggregates[arm] = {k: _agg(arm, k) for k in agg_keys}

    # Paired seed deltas where matched baseline exists
    paired_deltas: Dict[str, Any] = {}
    for seed, deg in deg_by_seed.items():
        base = base_by_seed.get(seed)
        if base is None:
            paired_deltas[str(seed)] = {"status": "no_matched_baseline"}
            continue
        paired_deltas[str(seed)] = {
            "status": "ok",
            "baseline_run": base.get("run_name"),
            "degflow_run": deg.get("run_name"),
            "claim1": {
                "A_delta_auprc": _paired_delta(deg, base, "A_embedding", "auprc"),
                "A_delta_p100": _paired_delta(
                    deg, base, "A_embedding", "precision_at_100"
                ),
                "A_delta_r_p80": _paired_delta(
                    deg, base, "A_embedding", "recall_at_precision_ge_0.80"
                ),
                "A_delta_r_p90": _paired_delta(
                    deg, base, "A_embedding", "recall_at_precision_ge_0.90"
                ),
                "B_delta_auprc": _paired_delta(deg, base, "B_embedding_raw", "auprc"),
                "B_delta_p100": _paired_delta(
                    deg, base, "B_embedding_raw", "precision_at_100"
                ),
            },
            "claim2_D": {
                "delta_auprc": _paired_delta(
                    deg, base, "D_embedding_raw_temporal_flow", "auprc"
                ),
                "delta_p100": _paired_delta(
                    deg, base, "D_embedding_raw_temporal_flow", "precision_at_100"
                ),
                "delta_r100": _paired_delta(
                    deg, base, "D_embedding_raw_temporal_flow", "recall_at_100"
                ),
                "delta_p500": _paired_delta(
                    deg, base, "D_embedding_raw_temporal_flow", "precision_at_500"
                ),
                "delta_r500": _paired_delta(
                    deg, base, "D_embedding_raw_temporal_flow", "recall_at_500"
                ),
                "delta_p1000": _paired_delta(
                    deg, base, "D_embedding_raw_temporal_flow", "precision_at_1000"
                ),
                "delta_r1000": _paired_delta(
                    deg, base, "D_embedding_raw_temporal_flow", "recall_at_1000"
                ),
                "delta_r_p90": _paired_delta(
                    deg,
                    base,
                    "D_embedding_raw_temporal_flow",
                    "recall_at_precision_ge_0.90",
                ),
                "delta_r_p80": _paired_delta(
                    deg,
                    base,
                    "D_embedding_raw_temporal_flow",
                    "recall_at_precision_ge_0.80",
                ),
            },
        }

    # Claim 1 consistency
    claim1_a_improve = []
    claim1_b_improve = []
    claim1_r80_improve = []
    precision_collapse_seeds = []
    for seed_s, dlt in paired_deltas.items():
        if dlt.get("status") != "ok":
            continue
        c1 = dlt["claim1"]
        if c1["A_delta_auprc"] is not None:
            claim1_a_improve.append(c1["A_delta_auprc"] > 0)
        if c1["B_delta_auprc"] is not None:
            claim1_b_improve.append(c1["B_delta_auprc"] > 0)
        if c1["A_delta_r_p80"] is not None:
            claim1_r80_improve.append(c1["A_delta_r_p80"] > 0)
        # precision collapse: A P@100 < 50% of matched baseline
        base = base_by_seed[int(seed_s)]
        deg = deg_by_seed[int(seed_s)]
        bp = _get_arm(base, "A_embedding", "precision_at_100")
        dp = _get_arm(deg, "A_embedding", "precision_at_100")
        if bp is not None and dp is not None and float(bp) > 0 and float(dp) < 0.5 * float(bp):
            precision_collapse_seeds.append(int(seed_s))

    # Best-of identifiers across all complete degflow+baseline rows (pre-3h)
    candidates = [
        r
        for r in rows
        if r.get("status") == "complete"
        and r.get("variant") in ("baseline", "degflow")
    ]

    def _best_by(arm: str, key: str) -> Tuple[Optional[str], Optional[float]]:
        scored = []
        for r in candidates:
            v = _get_arm(r, arm, key)
            if v is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if fv != fv:
                continue
            scored.append((fv, f"{r['variant']}_seed{r['seed']}"))
        if not scored:
            return None, None
        best = max(scored, key=lambda x: x[0])
        return best[1], best[0]

    best_rep_a, best_rep_a_v = _best_by("A_embedding", "auprc")
    best_raw_b, best_raw_b_v = _best_by("B_embedding_raw", "auprc")
    best_d_auprc, best_d_auprc_v = _best_by("D_embedding_raw_temporal_flow", "auprc")
    best_d_r90, best_d_r90_v = _best_by(
        "D_embedding_raw_temporal_flow", "recall_at_precision_ge_0.90"
    )
    best_d_r80, best_d_r80_v = _best_by(
        "D_embedding_raw_temporal_flow", "recall_at_precision_ge_0.80"
    )
    best_d_r1000, best_d_r1000_v = _best_by(
        "D_embedding_raw_temporal_flow", "recall_at_1000"
    )
    best_d_p100, best_d_p100_v = _best_by(
        "D_embedding_raw_temporal_flow", "precision_at_100"
    )

    # Recommendation logic
    n_paired = sum(1 for d in paired_deltas.values() if d.get("status") == "ok")
    a_all = bool(claim1_a_improve) and all(claim1_a_improve)
    b_all = bool(claim1_b_improve) and all(claim1_b_improve)
    r80_any = any(claim1_r80_improve) if claim1_r80_improve else False
    no_collapse = len(precision_collapse_seeds) == 0

    # Claim 2: do not promote as overall best final method unless D improves AUPRC
    # AND high-precision points across paired seeds.
    d_auprc_improve = []
    d_r90_improve = []
    d_p100_improve = []
    for dlt in paired_deltas.values():
        if dlt.get("status") != "ok":
            continue
        c2 = dlt["claim2_D"]
        if c2["delta_auprc"] is not None:
            d_auprc_improve.append(c2["delta_auprc"] > 0)
        if c2["delta_r_p90"] is not None:
            d_r90_improve.append(c2["delta_r_p90"] > 0)
        if c2["delta_p100"] is not None:
            d_p100_improve.append(c2["delta_p100"] > 0)

    claim2_overall_best = (
        bool(d_auprc_improve)
        and all(d_auprc_improve)
        and bool(d_r90_improve)
        and all(d_r90_improve)
        and bool(d_p100_improve)
        and all(d_p100_improve)
    )

    if n_paired == 0 and len(deg_complete) < 3:
        recommendation = "pending"
    elif a_all and b_all and no_collapse and r80_any:
        recommendation = "promote_as_representation_objective"
        if claim2_overall_best:
            recommendation = "promote_and_consider_40ep_scaleup"
        elif d_auprc_improve and d_p100_improve and (
            any(d_auprc_improve) and not all(d_p100_improve)
        ):
            recommendation = (
                "promote_representation_keep_baseline_D_for_strict_precision"
            )
    elif a_all and no_collapse:
        recommendation = "replicate_more_or_scale_cautiously"
    else:
        recommendation = "stop"

    promote_as_rep = a_all and b_all and no_collapse
    scale_40ep = promote_as_rep and len(deg_complete) >= 3 and (
        recommendation
        in (
            "promote_as_representation_objective",
            "promote_and_consider_40ep_scaleup",
            "promote_representation_keep_baseline_D_for_strict_precision",
        )
    )

    payload = {
        "scout": "degflow_morphology_multiseed",
        "thesis_role": "diagnostic_or_scout",
        "validation_status": "diagnostic_only",
        "table_eligible": False,
        "table_group": TABLE_GROUP,
        "primary_representation": "pre_embedding_3h",
        "post_128_is_diagnostic_only": True,
        "ssl_labels_used": False,
        "excluded": [
            "clustering",
            "degflow_tfreg",
            "morph_contrast_m2",
            "temporal_flow_soft_positives",
            "tier2_betweenness",
        ],
        "degflow_flags": DEGFLOW_FLAGS,
        "seeds": seeds,
        "variants": rows,
        "aggregates_degflow_mean_sd": aggregates,
        "paired_deltas_vs_matched_baseline": paired_deltas,
        "claim1_representation_improvement": {
            "primary_metrics": [
                "pre3h_A_auprc",
                "pre3h_A_P@K",
                "pre3h_A_recall@P>=0.90/0.80",
                "pre3h_B_auprc",
                "pre3h_B_P@K/R@K",
            ],
            "a_auprc_improved_all_paired_seeds": a_all if claim1_a_improve else None,
            "b_auprc_improved_all_paired_seeds": b_all if claim1_b_improve else None,
            "a_recall_p80_improved_any_paired": r80_any,
            "precision_collapse_seeds": precision_collapse_seeds,
            "n_paired_seeds": n_paired,
            "best_representation_only": {"id": best_rep_a, "auprc": best_rep_a_v},
            "best_plus_raw": {"id": best_raw_b, "auprc": best_raw_b_v},
        },
        "claim2_final_stack_tradeoff": {
            "note": (
                "Seed-1 scout: degflow D improved AUPRC and R@1000 but reduced P@100 "
                "and high-precision recall vs baseline D. Treat as precision/recall "
                "tradeoff, not a simple win/loss."
            ),
            "d_auprc_improved_all_paired": (
                all(d_auprc_improve) if d_auprc_improve else None
            ),
            "d_r90_improved_all_paired": all(d_r90_improve) if d_r90_improve else None,
            "d_p100_improved_all_paired": (
                all(d_p100_improve) if d_p100_improve else None
            ),
            "overall_best_final_method": claim2_overall_best,
            "best_final_d_by_auprc": {"id": best_d_auprc, "auprc": best_d_auprc_v},
            "best_final_d_by_strict_high_precision_r90": {
                "id": best_d_r90,
                "recall_at_precision_ge_0.90": best_d_r90_v,
            },
            "best_final_d_by_r80": {
                "id": best_d_r80,
                "recall_at_precision_ge_0.80": best_d_r80_v,
            },
            "best_final_d_by_broader_recall_r1000": {
                "id": best_d_r1000,
                "recall_at_1000": best_d_r1000_v,
            },
            "best_final_d_by_p100": {
                "id": best_d_p100,
                "precision_at_100": best_d_p100_v,
            },
        },
        "recommendation": recommendation,
        "promote_from_diagnostic": promote_as_rep and claim2_overall_best,
        "promote_as_representation_objective_even_if_baseline_D_better_at_strict_precision": promote_as_rep,
        "scale_to_40ep": bool(scale_40ep),
    }

    out_md = ROOT / args.output_md
    out_tbl = ROOT / args.output_table_md
    out_tex = ROOT / args.output_table_tex
    for p in (out_json.parent, out_md.parent, out_tbl.parent, out_tex.parent):
        p.mkdir(parents=True, exist_ok=True)

    # JSON with NaN -> null
    def _sanitize(obj: Any) -> Any:
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        return obj

    out_json.write_text(
        json.dumps(_sanitize(payload), indent=2) + "\n", encoding="utf-8"
    )

    def _msd(arm: str, key: str) -> str:
        st = aggregates[arm][key]
        if st["mean"] is None:
            return "—"
        if st["sd"] is None:
            return f"{_fmt(st['mean'])} (n={st['n']})"
        return f"{_fmt(st['mean'])}±{_fmt(st['sd'])} (n={st['n']})"

    lines = [
        "# Degflow morphology multiseed scout",
        "",
        f"**Thesis role:** diagnostic_or_scout · **validation_status:** diagnostic_only · "
        f"**table_eligible:** false · **table_group:** `{TABLE_GROUP}`",
        "",
        "Primary representation: **pre_embedding_3h**. Post-128 diagnostic only.",
        "SSL: InfoNCE + morphology expert **regression** only "
        "(degree_fan + flow_balance). **No labels.**",
        "",
        f"Exact flags: `{DEGFLOW_FLAGS}`",
        "",
        "Excluded: clustering, degflow_tfreg, M2/bin contrast, TF soft positives, "
        "tier2/betweenness.",
        "",
        "## Recommendation",
        "",
        f"- **`{recommendation}`**",
        f"- Promote as representation objective (even if baseline D wins strict "
        f"precision): **{promote_as_rep}**",
        f"- Promote as overall best final method: **{claim2_overall_best}**",
        f"- Scale to 40ep: **{bool(scale_40ep)}**",
        f"- Precision-collapse seeds (A P@100 < 50% of matched baseline): "
        f"`{precision_collapse_seeds or 'none'}`",
        f"- Paired seeds with matched baseline: **{n_paired}**",
        "",
        "## Claim 1 — Representation improvement",
        "",
        "Evaluate whether degflow improves the learned pre-3h representation "
        "**before** downstream temporal-flow features are added.",
        "",
        f"- Best representation-only (A AUPRC): **{best_rep_a}** ({_fmt(best_rep_a_v)})",
        f"- Best +raw (B AUPRC): **{best_raw_b}** ({_fmt(best_raw_b_v)})",
        f"- Degflow A AUPRC mean±SD: {_msd('A_embedding', 'auprc')}",
        f"- Degflow B AUPRC mean±SD: {_msd('B_embedding_raw', 'auprc')}",
        f"- Degflow A P@100 mean±SD: {_msd('A_embedding', 'precision_at_100')}",
        f"- Degflow A R@P≥0.80 mean±SD: {_msd('A_embedding', 'recall_at_precision_ge_0.80')}",
        f"- Degflow A R@P≥0.90 mean±SD: {_msd('A_embedding', 'recall_at_precision_ge_0.90')}",
        f"- A AUPRC improved on all paired seeds: **{a_all if claim1_a_improve else 'n/a'}**",
        f"- B AUPRC improved on all paired seeds: **{b_all if claim1_b_improve else 'n/a'}**",
        "",
        "## Claim 2 — Final-stack (D) tradeoff",
        "",
        "Do **not** rank D by AUPRC alone. Seed-1 showed degflow D can raise AUPRC / "
        "R@1000 while lowering P@100 and high-precision recall vs baseline D.",
        "",
        f"- Best final D by AUPRC: **{best_d_auprc}** ({_fmt(best_d_auprc_v)})",
        f"- Best final D by strict high-precision (R@P≥0.90): **{best_d_r90}** "
        f"({_fmt(best_d_r90_v)})",
        f"- Best final D by R@P≥0.80: **{best_d_r80}** ({_fmt(best_d_r80_v)})",
        f"- Best final D by broader recall (R@1000): **{best_d_r1000}** "
        f"({_fmt(best_d_r1000_v)})",
        f"- Best final D by P@100: **{best_d_p100}** ({_fmt(best_d_p100_v)})",
        f"- Degflow D AUPRC mean±SD: {_msd('D_embedding_raw_temporal_flow', 'auprc')}",
        f"- Degflow D P@100 mean±SD: {_msd('D_embedding_raw_temporal_flow', 'precision_at_100')}",
        f"- Degflow D R@1000 mean±SD: {_msd('D_embedding_raw_temporal_flow', 'recall_at_1000')}",
        f"- Degflow D R@P≥0.90 mean±SD: "
        f"{_msd('D_embedding_raw_temporal_flow', 'recall_at_precision_ge_0.90')}",
        "",
        "## Per-seed pre-3h metrics",
        "",
        "| Seed | Variant | ckpt ep | Arm | AUROC | AUPRC | F1 | P@100 | R@500 | "
        "R@1000 | R@P≥0.90 | R@P≥0.80 |",
        "|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for r in rows:
        if r.get("status") != "complete":
            continue
        ep = r.get("selected_checkpoint_epoch")
        for arm in PRIMARY_ARMS:
            m = ((r.get("pre_embedding_3h") or {}).get("arms") or {}).get(arm) or {}
            if not m:
                continue
            lines.append(
                f"| {r['seed']} | {r['variant']} | {ep if ep is not None else '—'} | "
                f"{arm} | {_fmt(m.get('auroc'))} | {_fmt(m.get('auprc'))} | "
                f"{_fmt(m.get('f1'))} | {_fmt(m.get('precision_at_100'))} | "
                f"{_fmt(m.get('recall_at_500'))} | {_fmt(m.get('recall_at_1000'))} | "
                f"{_fmt(m.get('recall_at_precision_ge_0.90'))} | "
                f"{_fmt(m.get('recall_at_precision_ge_0.80'))} |"
            )

    lines.extend(
        [
            "",
            "## Paired deltas (degflow − matched baseline)",
            "",
            "| Seed | ΔA AUPRC | ΔA P@100 | ΔA R@P≥0.80 | ΔB AUPRC | ΔD AUPRC | "
            "ΔD P@100 | ΔD R@1000 | ΔD R@P≥0.90 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for seed in seeds:
        dlt = paired_deltas.get(str(seed), {})
        if dlt.get("status") != "ok":
            lines.append(
                f"| {seed} | — | — | — | — | — | — | — | — |"
            )
            continue
        c1, c2 = dlt["claim1"], dlt["claim2_D"]
        lines.append(
            f"| {seed} | {_fmt(c1['A_delta_auprc'])} | {_fmt(c1['A_delta_p100'])} | "
            f"{_fmt(c1['A_delta_r_p80'])} | {_fmt(c1['B_delta_auprc'])} | "
            f"{_fmt(c2['delta_auprc'])} | {_fmt(c2['delta_p100'])} | "
            f"{_fmt(c2['delta_r1000'])} | {_fmt(c2['delta_r_p90'])} |"
        )

    lines.extend(
        [
            "",
            "## Baseline availability",
            "",
            "- Seed 1: `hi_contrastive_gin_emlps_tds_proj_asym_8192neg_queue0_accum4_20ep` "
            "(already probed).",
            "- Seed 2: `hi_contrastive_gin_emlps_tds_seed2_proj_asym_8192neg_queue0_accum4_20ep` "
            "(extract+probe only; no retrain).",
            "- Seed 3: no matched 20ep plain-contrastive checkpoint; **not retrained**.",
            "",
            "## Notes",
            "",
            "- Rank primarily by pre-3h A/B AUPRC and recall@P / P@K — not AUROC or F1 alone.",
            "- Val-tuned F1 can be degenerate (flag-everything thresholds); treat with caution.",
            "- Do not insert into main thesis tables yet.",
            "",
        ]
    )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Compact tables
    tbl = [
        "# Degflow morphology multiseed scout (pre-3h)",
        "",
        f"table_group=`{TABLE_GROUP}` · diagnostic_only · not main-table eligible",
        "",
        "## Degflow mean ± sample SD",
        "",
        "| Arm | AUPRC | P@100 | R@500 | R@1000 | R@P≥0.90 | R@P≥0.80 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, label in (
        ("A_embedding", "A"),
        ("B_embedding_raw", "B"),
        ("D_embedding_raw_temporal_flow", "D"),
    ):
        tbl.append(
            f"| {label} | {_msd(arm, 'auprc')} | {_msd(arm, 'precision_at_100')} | "
            f"{_msd(arm, 'recall_at_500')} | {_msd(arm, 'recall_at_1000')} | "
            f"{_msd(arm, 'recall_at_precision_ge_0.90')} | "
            f"{_msd(arm, 'recall_at_precision_ge_0.80')} |"
        )
    tbl.extend(
        [
            "",
            "## Per-seed degflow A/B/D AUPRC",
            "",
            "| Seed | A AUPRC | B AUPRC | D AUPRC | A P@100 | A R@P≥0.80 | D P@100 | "
            "D R@P≥0.90 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for r in deg_complete:
        a = ((r.get("pre_embedding_3h") or {}).get("arms") or {}).get("A_embedding") or {}
        b = ((r.get("pre_embedding_3h") or {}).get("arms") or {}).get("B_embedding_raw") or {}
        d = ((r.get("pre_embedding_3h") or {}).get("arms") or {}).get(
            "D_embedding_raw_temporal_flow"
        ) or {}
        tbl.append(
            f"| {r['seed']} | {_fmt(a.get('auprc'))} | {_fmt(b.get('auprc'))} | "
            f"{_fmt(d.get('auprc'))} | {_fmt(a.get('precision_at_100'))} | "
            f"{_fmt(a.get('recall_at_precision_ge_0.80'))} | "
            f"{_fmt(d.get('precision_at_100'))} | "
            f"{_fmt(d.get('recall_at_precision_ge_0.90'))} |"
        )
    out_tbl.write_text("\n".join(tbl) + "\n", encoding="utf-8")

    tex = [
        r"% Degflow morphology multiseed scout (diagnostic only; not for main tables)",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Arm & AUPRC & P@100 & R@500 & R@1000 & R@P$\geq$0.90 & R@P$\geq$0.80 \\",
        r"\midrule",
    ]
    for arm, label in (
        ("A_embedding", "A"),
        ("B_embedding_raw", "B"),
        ("D_embedding_raw_temporal_flow", "D"),
    ):
        tex.append(
            f"{label} & {_msd(arm, 'auprc')} & {_msd(arm, 'precision_at_100')} & "
            f"{_msd(arm, 'recall_at_500')} & {_msd(arm, 'recall_at_1000')} & "
            f"{_msd(arm, 'recall_at_precision_ge_0.90')} & "
            f"{_msd(arm, 'recall_at_precision_ge_0.80')} \\\\"
        )
    tex.extend([r"\bottomrule", r"\end{tabular}", ""])
    out_tex.write_text("\n".join(tex), encoding="utf-8")

    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    print(f"Wrote {out_tbl}")
    print(f"Wrote {out_tex}")
    print(f"recommendation={recommendation}")
    print(f"n_degflow_complete={len(deg_complete)} n_paired={n_paired}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
