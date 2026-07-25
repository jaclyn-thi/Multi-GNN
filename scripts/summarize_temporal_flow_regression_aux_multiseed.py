#!/usr/bin/env python3
"""Summarize temporal-flow regression aux multiseed confirmation (seeds 1–3).

Separates two claims:
  Claim 1 — Representation improvement (pre-3h A / B before TF features)
  Claim 2 — Final-stack D tradeoff (AUPRC vs high-precision operating points)

Diagnostic only; not table-eligible for main thesis tables.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.scout_recall_metrics import extract_recall_oriented

PRIMARY_ARMS = ("A_embedding", "B_embedding_raw", "D_embedding_raw_temporal_flow")
TABLE_GROUP = "temporal_flow_regression_aux_multiseed"
ATTACH_POINT = "post_embedding_head_pre_projection"
VARIANTS = ("tf_reg_w0.10", "tf_reg_w0.05")

VARIANT_FLAGS = {
    "tf_reg_w0.10": (
        "--aux_temporal_flow regression --aux_temporal_flow_weight 0.10 "
        "--aux_temporal_flow_loss huber"
    ),
    "tf_reg_w0.05": (
        "--aux_temporal_flow regression --aux_temporal_flow_weight 0.05 "
        "--aux_temporal_flow_loss huber"
    ),
}


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
    if x != x:
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


def _tf_reg_run(variant: str, seed: int) -> str:
    return (
        f"hi_tf_aux_{variant}_gin_emlps_tds_asym_proj_"
        f"8192neg_queue0_accum4_20ep_seed{seed}"
    )


def _baseline_run(seed: int) -> Optional[str]:
    if seed == 1:
        return "hi_contrastive_gin_emlps_tds_proj_asym_8192neg_queue0_accum4_20ep"
    if seed == 2:
        return "hi_contrastive_gin_emlps_tds_seed2_proj_asym_8192neg_queue0_accum4_20ep"
    return None


def _resolve_tf_probe(variant: str, rep: str, seed: int) -> Tuple[Optional[Path], str]:
    """Prefer enriched recall-metrics for seed1; else standard probe JSON."""
    enriched = (
        ROOT
        / "results/diagnostics/enriched"
        / f"tf_aux_{variant}_{rep}_recall_metrics.json"
    )
    standard = ROOT / f"results/diagnostics/tf_aux_{variant}_{rep}_seed{seed}.json"
    if seed == 1 and enriched.is_file():
        return enriched, str(enriched.relative_to(ROOT))
    if standard.is_file():
        return standard, str(standard.relative_to(ROOT))
    if enriched.is_file():
        return enriched, str(enriched.relative_to(ROOT))
    return None, str(standard.relative_to(ROOT))


def _parse_aux_diagnostics(run_name: str) -> Dict[str, Any]:
    log = ROOT / "logs" / f"{run_name}_train.log"
    out: Dict[str, Any] = {"train_log": str(log.relative_to(ROOT)) if log.is_file() else None}
    if not log.is_file():
        return out
    text = log.read_text(encoding="utf-8", errors="replace")
    modes = re.findall(r"tf_aux mode=(\w+)", text)
    weights = re.findall(r"weight=([0-9.]+)", text)
    attach = re.findall(r"attach=([^\s]+)", text)
    aux_losses = [
        float(x) for x in re.findall(r"loss/temporal_flow_aux[=:\s]+([0-9.eE+-]+)", text)
    ]
    if modes:
        out["tf_aux_mode"] = modes[-1]
    if weights:
        out["tf_aux_weight_logged"] = float(weights[0])
    if attach:
        out["attach_point_logged"] = attach[0]
    if aux_losses:
        out["aux_loss_last"] = aux_losses[-1]
        out["aux_loss_first"] = aux_losses[0]
        out["aux_loss_n_logged"] = len(aux_losses)
    out["labels_in_ssl"] = bool(re.search(r"--objective\s+supervised|ssl_labels_used.?=.?true", text, re.I))
    return out


def _variant_row(
    tag: str,
    seed: int,
    run_name: Optional[str],
    pre: Optional[Dict[str, Any]],
    post: Optional[Dict[str, Any]],
    *,
    pre_src: Optional[str],
    post_src: Optional[str],
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
        "attach_point": ATTACH_POINT,
        "thesis_role": "diagnostic_or_scout",
        "validation_status": "diagnostic_only",
        "table_eligible": False,
        "table_group": TABLE_GROUP,
        "selected_checkpoint_epoch": _ckpt_epoch(pre),
        "aux_diagnostics": _parse_aux_diagnostics(run_name) if run_name else {},
        "pre_embedding_3h": {
            "source_json": pre_src,
            "arms": {
                a: _arm_test(pre, a)
                for a in PRIMARY_ARMS
                if a in (pre.get("arms") or {})
            },
        },
        "post_embedding_128_diagnostic": None,
    }
    if post is not None:
        post_arms = ("A_embedding", "B_embedding_raw", "D_embedding_raw_temporal_flow")
        entry["post_embedding_128_diagnostic"] = {
            "source_json": post_src,
            "arms": {
                a: _arm_test(post, a)
                for a in post_arms
                if a in (post.get("arms") or {})
            },
        }
    return entry


def _paired_delta(
    var: Dict[str, Any], base: Dict[str, Any], arm: str, key: str
) -> Optional[float]:
    a = _get_arm(var, arm, key)
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


def _majority(flags: Sequence[bool]) -> Optional[bool]:
    if not flags:
        return None
    return sum(1 for f in flags if f) > (len(flags) / 2.0)


def _claim_for_variant(
    variant: str,
    rows: List[Dict[str, Any]],
    base_by_seed: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    var_complete = [
        r for r in rows if r.get("variant") == variant and r.get("status") == "complete"
    ]
    var_by_seed = {int(r["seed"]): r for r in var_complete}

    def _agg(arm: str, key: str) -> Dict[str, Any]:
        vals = []
        for r in var_complete:
            v = _get_arm(r, arm, key)
            if v is None:
                continue
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
        "f1",
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
        "n_alerts_at_precision_ge_0.95",
        "n_alerts_at_precision_ge_0.90",
        "n_alerts_at_precision_ge_0.80",
        "n_alerts_at_precision_ge_0.70",
        "precision_achieved_at_precision_ge_0.95",
        "precision_achieved_at_precision_ge_0.90",
        "precision_achieved_at_precision_ge_0.80",
        "precision_achieved_at_precision_ge_0.70",
    ]
    aggregates = {arm: {k: _agg(arm, k) for k in agg_keys} for arm in PRIMARY_ARMS}

    paired_deltas: Dict[str, Any] = {}
    for seed, var in var_by_seed.items():
        base = base_by_seed.get(seed)
        if base is None:
            paired_deltas[str(seed)] = {"status": "no_matched_baseline"}
            continue
        paired_deltas[str(seed)] = {
            "status": "ok",
            "baseline_run": base.get("run_name"),
            "variant_run": var.get("run_name"),
            "claim1": {
                "A_delta_auprc": _paired_delta(var, base, "A_embedding", "auprc"),
                "A_delta_p100": _paired_delta(
                    var, base, "A_embedding", "precision_at_100"
                ),
                "A_delta_r_p80": _paired_delta(
                    var, base, "A_embedding", "recall_at_precision_ge_0.80"
                ),
                "A_delta_r_p90": _paired_delta(
                    var, base, "A_embedding", "recall_at_precision_ge_0.90"
                ),
                "B_delta_auprc": _paired_delta(var, base, "B_embedding_raw", "auprc"),
                "B_delta_p100": _paired_delta(
                    var, base, "B_embedding_raw", "precision_at_100"
                ),
            },
            "claim2_D": {
                "delta_auprc": _paired_delta(
                    var, base, "D_embedding_raw_temporal_flow", "auprc"
                ),
                "delta_p100": _paired_delta(
                    var, base, "D_embedding_raw_temporal_flow", "precision_at_100"
                ),
                "delta_r_p90": _paired_delta(
                    var,
                    base,
                    "D_embedding_raw_temporal_flow",
                    "recall_at_precision_ge_0.90",
                ),
                "delta_r_p80": _paired_delta(
                    var,
                    base,
                    "D_embedding_raw_temporal_flow",
                    "recall_at_precision_ge_0.80",
                ),
            },
        }

    claim1_a_improve: List[bool] = []
    claim1_b_improve: List[bool] = []
    a_deltas: List[float] = []
    b_deltas: List[float] = []
    precision_collapse_seeds: List[int] = []
    for seed_s, dlt in paired_deltas.items():
        if dlt.get("status") != "ok":
            continue
        c1 = dlt["claim1"]
        if c1["A_delta_auprc"] is not None:
            claim1_a_improve.append(c1["A_delta_auprc"] > 0)
            a_deltas.append(float(c1["A_delta_auprc"]))
        if c1["B_delta_auprc"] is not None:
            claim1_b_improve.append(c1["B_delta_auprc"] > 0)
            b_deltas.append(float(c1["B_delta_auprc"]))
        base = base_by_seed[int(seed_s)]
        var = var_by_seed[int(seed_s)]
        bp = _get_arm(base, "A_embedding", "precision_at_100")
        vp = _get_arm(var, "A_embedding", "precision_at_100")
        if bp is not None and vp is not None and float(bp) > 0 and float(vp) < 0.5 * float(bp):
            precision_collapse_seeds.append(int(seed_s))

    n_paired = sum(1 for d in paired_deltas.values() if d.get("status") == "ok")
    a_most = _majority(claim1_a_improve)
    b_most = _majority(claim1_b_improve)
    mean_a = _mean_sd(a_deltas)
    mean_b = _mean_sd(b_deltas)
    mean_a_pos = mean_a["mean"] is not None and mean_a["mean"] > 0
    mean_b_pos = mean_b["mean"] is not None and mean_b["mean"] > 0
    no_collapse = len(precision_collapse_seeds) == 0

    # Only-seed1 improvement pattern
    only_seed1 = False
    if n_paired >= 2 and claim1_a_improve:
        seed_flags = {
            int(s): (d["claim1"]["A_delta_auprc"] or 0) > 0
            for s, d in paired_deltas.items()
            if d.get("status") == "ok" and d["claim1"]["A_delta_auprc"] is not None
        }
        if seed_flags.get(1) and not any(v for s, v in seed_flags.items() if s != 1):
            only_seed1 = True

    claim1_pass = (
        n_paired >= 1
        and no_collapse
        and not only_seed1
        and ((a_most and mean_a_pos) or (b_most and mean_b_pos))
    )

    d_auprc_improve: List[bool] = []
    d_r90_ok: List[bool] = []
    d_r80_ok: List[bool] = []
    d_p100_ok: List[bool] = []
    for dlt in paired_deltas.values():
        if dlt.get("status") != "ok":
            continue
        c2 = dlt["claim2_D"]
        if c2["delta_auprc"] is not None:
            d_auprc_improve.append(c2["delta_auprc"] > 0)
        # "unacceptable tradeoff": large drop in P@100 or high-precision recall
        if c2["delta_p100"] is not None:
            d_p100_ok.append(c2["delta_p100"] >= -0.10)
        if c2["delta_r_p90"] is not None:
            d_r90_ok.append(c2["delta_r_p90"] >= -0.02)
        if c2["delta_r_p80"] is not None:
            d_r80_ok.append(c2["delta_r_p80"] >= -0.02)

    claim2_pass = (
        bool(d_auprc_improve)
        and _majority(d_auprc_improve) is True
        and (not d_p100_ok or all(d_p100_ok))
        and (not d_r90_ok or all(d_r90_ok))
        and (not d_r80_ok or all(d_r80_ok))
    )

    # Absolute weak seeds (collapse without baseline): A P@100 < 0.20 or A AUPRC < 0.10
    collapse_absolute = []
    for r in var_complete:
        a = _get_arm(r, "A_embedding", "auprc")
        p = _get_arm(r, "A_embedding", "precision_at_100")
        if (a is not None and float(a) < 0.10) or (p is not None and float(p) < 0.20):
            collapse_absolute.append(int(r["seed"]))

    if only_seed1 or (len(collapse_absolute) >= 2 and 1 not in collapse_absolute):
        recommendation = "stop"
    elif claim1_pass and claim2_pass:
        recommendation = "promote"
    elif claim1_pass:
        recommendation = "keep_diagnostic"
    elif n_paired == 0:
        recommendation = "pending"
    else:
        recommendation = "stop"

    return {
        "variant": variant,
        "flags": VARIANT_FLAGS[variant],
        "n_complete": len(var_complete),
        "aggregates_mean_sd": aggregates,
        "paired_deltas_vs_matched_baseline": paired_deltas,
        "claim1_representation_improvement": {
            "pass": claim1_pass,
            "a_auprc_improved_most_paired": a_most,
            "b_auprc_improved_most_paired": b_most,
            "mean_paired_delta_a_auprc": mean_a,
            "mean_paired_delta_b_auprc": mean_b,
            "precision_collapse_seeds": precision_collapse_seeds,
            "only_seed1_improves": only_seed1,
            "n_paired_seeds": n_paired,
            "absolute_collapse_seeds": collapse_absolute,
        },
        "claim2_final_stack_tradeoff": {
            "pass": claim2_pass,
            "d_auprc_improved_most_paired": _majority(d_auprc_improve),
            "d_p100_no_large_drop_all_paired": all(d_p100_ok) if d_p100_ok else None,
            "d_r90_no_large_drop_all_paired": all(d_r90_ok) if d_r90_ok else None,
            "d_r80_no_large_drop_all_paired": all(d_r80_ok) if d_r80_ok else None,
        },
        "recommendation": recommendation,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument(
        "--require_seeds",
        type=int,
        nargs="*",
        default=[2, 3],
        help="Abort if these seed probe JSONs are missing for both variants",
    )
    ap.add_argument(
        "--output_json",
        default="results/diagnostics/temporal_flow_regression_aux_multiseed.json",
    )
    ap.add_argument(
        "--output_md",
        default="notes/temporal_flow_regression_aux_multiseed.md",
    )
    ap.add_argument(
        "--output_table_md",
        default="tables/temporal_flow_regression_aux_multiseed.md",
    )
    ap.add_argument(
        "--output_table_tex",
        default="tables/temporal_flow_regression_aux_multiseed.tex",
    )
    args = ap.parse_args()

    out_json = ROOT / args.output_json
    if out_json.is_file():
        print(f"ABORT: refusing overwrite of {out_json}", file=sys.stderr)
        return 1

    seeds = list(args.seeds)
    require = set(args.require_seeds or [])
    rows: List[Dict[str, Any]] = []
    missing_required: List[str] = []

    for seed in seeds:
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
                _variant_row(
                    "baseline",
                    seed,
                    base_run,
                    pre,
                    post,
                    pre_src=str(pre_p.relative_to(ROOT)) if pre else None,
                    post_src=str(post_p.relative_to(ROOT)) if post else None,
                    required=required_base,
                )
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

        for variant in VARIANTS:
            run = _tf_reg_run(variant, seed)
            pre_path, pre_src = _resolve_tf_probe(variant, "pre3h", seed)
            post_path, post_src = _resolve_tf_probe(variant, "post128", seed)
            pre = _load(pre_path) if pre_path else None
            post = _load(post_path) if post_path else None
            required = seed in require or seed == 1
            if pre is None and required:
                missing_required.append(pre_src)
            rows.append(
                _variant_row(
                    variant,
                    seed,
                    run,
                    pre,
                    post,
                    pre_src=pre_src if pre else None,
                    post_src=post_src if post else None,
                    required=required,
                )
            )

    if missing_required:
        print("Missing required probe JSONs:", file=sys.stderr)
        for m in missing_required:
            print(f"  {m}", file=sys.stderr)
        return 1

    base_complete = [
        r for r in rows if r.get("variant") == "baseline" and r.get("status") == "complete"
    ]
    base_by_seed = {int(r["seed"]): r for r in base_complete}

    per_variant = {
        v: _claim_for_variant(v, rows, base_by_seed) for v in VARIANTS
    }

    candidates = [r for r in rows if r.get("status") == "complete"]

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

    # Stability: lower SD of pre-3h A AUPRC across seeds, preferring claim1 pass
    def _stability(v: str) -> Tuple[float, float]:
        st = per_variant[v]["aggregates_mean_sd"]["A_embedding"]["auprc"]
        mean = st["mean"] if st["mean"] is not None else float("-inf")
        sd = st["sd"] if st["sd"] is not None else float("inf")
        return mean, -sd

    more_stable = max(VARIANTS, key=_stability)

    recs = {v: per_variant[v]["recommendation"] for v in VARIANTS}
    if any(r == "promote" for r in recs.values()):
        overall = "promote"
        overall_variant = next(v for v in VARIANTS if recs[v] == "promote")
    elif any(r == "keep_diagnostic" for r in recs.values()):
        overall = "keep_diagnostic"
        overall_variant = next(v for v in VARIANTS if recs[v] == "keep_diagnostic")
    elif all(r == "pending" for r in recs.values()):
        overall = "pending"
        overall_variant = None
    else:
        overall = "stop"
        overall_variant = None

    # Seed1 stage0 snapshot (for report)
    stage0: Dict[str, Any] = {"note": "seed1 reused; recall@P from enriched when available"}
    for variant in VARIANTS:
        r = next(
            (
                x
                for x in rows
                if x.get("variant") == variant
                and int(x.get("seed", -1)) == 1
                and x.get("status") == "complete"
            ),
            None,
        )
        if not r:
            continue
        stage0[variant] = {
            "selected_checkpoint_epoch": r.get("selected_checkpoint_epoch"),
            "pre_embedding_3h": (r.get("pre_embedding_3h") or {}).get("arms"),
            "post_embedding_128_diagnostic": (
                (r.get("post_embedding_128_diagnostic") or {}).get("arms")
            ),
            "source_pre": (r.get("pre_embedding_3h") or {}).get("source_json"),
            "source_post": (r.get("post_embedding_128_diagnostic") or {}).get(
                "source_json"
            ),
        }

    payload = {
        "scout": "temporal_flow_regression_aux_multiseed",
        "thesis_role": "diagnostic_or_scout",
        "validation_status": "diagnostic_only",
        "table_eligible": False,
        "table_group": TABLE_GROUP,
        "primary_representation": "pre_embedding_3h",
        "post_128_is_diagnostic_only": True,
        "ssl_labels_used": False,
        "attach_point": ATTACH_POINT,
        "excluded": [
            "temporal_flow_bins",
            "temporal_flow_soft_positives",
            "morphology_objectives",
            "degflow",
            "clustering",
            "betweenness_centrality",
        ],
        "recipe": {
            "projection": "asym",
            "negatives": 8192,
            "queue": 0,
            "temperature": 0.5,
            "reverse_mp": True,
            "ego": True,
            "ports": True,
            "emlps": True,
            "tds": True,
            "batch_size": 8192,
            "accum": 4,
            "n_epochs": 20,
            "checkpoint_policy": "best",
        },
        "variant_flags": VARIANT_FLAGS,
        "seeds": seeds,
        "variants": rows,
        "stage0_seed1": stage0,
        "per_variant": per_variant,
        "best_pre3h_embedding_only": {"id": best_rep_a, "auprc": best_rep_a_v},
        "best_pre3h_plus_raw": {"id": best_raw_b, "auprc": best_raw_b_v},
        "best_final_d_by_auprc": {"id": best_d_auprc, "auprc": best_d_auprc_v},
        "best_final_d_by_high_precision_r90": {
            "id": best_d_r90,
            "recall_at_precision_ge_0.90": best_d_r90_v,
        },
        "best_final_d_by_r80": {
            "id": best_d_r80,
            "recall_at_precision_ge_0.80": best_d_r80_v,
        },
        "more_stable_weight": more_stable,
        "recommendation": overall,
        "recommendation_variant": overall_variant,
        "baseline_availability": {
            "seed1": "morph_obj_baseline_*_seed1.json (matched)",
            "seed2": "morph_obj_baseline_*_seed2.json (matched)",
            "seed3": "unavailable; absolute metrics only; not retrained",
        },
    }

    out_md = ROOT / args.output_md
    out_tbl = ROOT / args.output_table_md
    out_tex = ROOT / args.output_table_tex
    for p in (out_json.parent, out_md.parent, out_tbl.parent, out_tex.parent):
        p.mkdir(parents=True, exist_ok=True)

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

    def _msd(variant: str, arm: str, key: str) -> str:
        st = per_variant[variant]["aggregates_mean_sd"][arm][key]
        if st["mean"] is None:
            return "—"
        if st["sd"] is None:
            return f"{_fmt(st['mean'])} (n={st['n']})"
        return f"{_fmt(st['mean'])}±{_fmt(st['sd'])} (n={st['n']})"

    lines = [
        "# Temporal-flow regression aux multiseed confirmation",
        "",
        f"**Thesis role:** diagnostic_or_scout · **validation_status:** diagnostic_only · "
        f"**table_eligible:** false · **table_group:** `{TABLE_GROUP}`",
        "",
        "Primary representation: **pre_embedding_3h**. Post-128 diagnostic only.",
        "SSL: InfoNCE + temporal_flow_causal **regression** (Huber). **No labels.**",
        f"Attach point: `{ATTACH_POINT}` (fixed; no new attach-point code).",
        "",
        "Excluded: bins, soft positives, morphology/degflow/clustering/betweenness.",
        "",
        "## Recommendation",
        "",
        f"- **Overall: `{overall}`**"
        + (f" (best claim via `{overall_variant}`)" if overall_variant else ""),
        f"- More stable weight (higher mean A AUPRC, lower SD): **{more_stable}**",
        f"- Best pre-3h embedding-only (A): **{best_rep_a}** ({_fmt(best_rep_a_v)})",
        f"- Best pre-3h + raw (B): **{best_raw_b}** ({_fmt(best_raw_b_v)})",
        f"- Best final D by AUPRC: **{best_d_auprc}** ({_fmt(best_d_auprc_v)})",
        f"- Best final D by R@P≥0.90: **{best_d_r90}** ({_fmt(best_d_r90_v)})",
        f"- Best final D by R@P≥0.80: **{best_d_r80}** ({_fmt(best_d_r80_v)})",
        "",
        "## Per-variant claims",
        "",
    ]
    for v in VARIANTS:
        c = per_variant[v]
        c1 = c["claim1_representation_improvement"]
        c2 = c["claim2_final_stack_tradeoff"]
        lines.extend(
            [
                f"### `{v}`",
                "",
                f"- Flags: `{c['flags']}`",
                f"- Recommendation: **{c['recommendation']}**",
                f"- Claim 1 (representation): **{c1['pass']}** "
                f"(A most={c1['a_auprc_improved_most_paired']}, "
                f"B most={c1['b_auprc_improved_most_paired']}, "
                f"mean ΔA={_fmt((c1['mean_paired_delta_a_auprc'] or {}).get('mean'))}, "
                f"collapse={c1['precision_collapse_seeds'] or 'none'})",
                f"- Claim 2 (final D): **{c2['pass']}** "
                f"(D AUPRC most={c2['d_auprc_improved_most_paired']})",
                f"- Pre-3h A AUPRC mean±SD: {_msd(v, 'A_embedding', 'auprc')}",
                f"- Pre-3h B AUPRC mean±SD: {_msd(v, 'B_embedding_raw', 'auprc')}",
                f"- Pre-3h D AUPRC mean±SD: {_msd(v, 'D_embedding_raw_temporal_flow', 'auprc')}",
                "",
            ]
        )

    lines.extend(
        [
            "## Per-seed pre-3h metrics",
            "",
            "| Seed | Variant | ckpt ep | Arm | AUROC | AUPRC | F1 | P@100 | R@100 | "
            "P@500 | R@500 | P@1000 | R@1000 | R@P≥0.95 | R@P≥0.90 | R@P≥0.80 | R@P≥0.70 |",
            "|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
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
            "## Post-128 diagnostic (A/B/D)",
            "",
            "| Seed | Variant | Arm | AUPRC | P@100 | R@P≥0.90 | R@P≥0.80 |",
            "|---:|---|---|---:|---:|---:|---:|",
        ]
    )
    for r in rows:
        if r.get("status") != "complete":
            continue
        post = r.get("post_embedding_128_diagnostic") or {}
        for arm, m in (post.get("arms") or {}).items():
            lines.append(
                f"| {r['seed']} | {r['variant']} | {arm} | {_fmt(m.get('auprc'))} | "
                f"{_fmt(m.get('precision_at_100'))} | "
                f"{_fmt(m.get('recall_at_precision_ge_0.90'))} | "
                f"{_fmt(m.get('recall_at_precision_ge_0.80'))} |"
            )

    lines.extend(
        [
            "",
            "## Paired deltas (variant − matched baseline, pre-3h)",
            "",
            "| Variant | Seed | ΔA AUPRC | ΔA P@100 | ΔB AUPRC | ΔD AUPRC | "
            "ΔD P@100 | ΔD R@P≥0.90 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for v in VARIANTS:
        for seed in seeds:
            dlt = per_variant[v]["paired_deltas_vs_matched_baseline"].get(str(seed), {})
            if dlt.get("status") != "ok":
                lines.append(f"| {v} | {seed} | — | — | — | — | — | — |")
                continue
            c1, c2 = dlt["claim1"], dlt["claim2_D"]
            lines.append(
                f"| {v} | {seed} | {_fmt(c1['A_delta_auprc'])} | {_fmt(c1['A_delta_p100'])} | "
                f"{_fmt(c1['B_delta_auprc'])} | {_fmt(c2['delta_auprc'])} | "
                f"{_fmt(c2['delta_p100'])} | {_fmt(c2['delta_r_p90'])} |"
            )

    lines.extend(
        [
            "",
            "## Baseline availability",
            "",
            "- Seed 1: matched morph_obj_baseline probes (reuse).",
            "- Seed 2: matched morph_obj_baseline probes (reuse; no retrain).",
            "- Seed 3: no matched baseline; absolute metrics only; **not retrained**.",
            "",
            "## Notes",
            "",
            "- Do not count D-only gains as representation improvement.",
            "- Do not promote on post-128-only gains.",
            "- Do not insert into main thesis tables yet.",
            "",
        ]
    )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    tbl = [
        "# Temporal-flow regression aux multiseed (pre-3h)",
        "",
        f"table_group=`{TABLE_GROUP}` · diagnostic_only · not main-table eligible",
        "",
        f"Overall recommendation: **{overall}** · more stable: **{more_stable}**",
        "",
    ]
    for v in VARIANTS:
        tbl.extend(
            [
                f"## `{v}` mean ± sample SD",
                "",
                "| Arm | AUPRC | P@100 | R@500 | R@1000 | R@P≥0.90 | R@P≥0.80 |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for arm, label in (
            ("A_embedding", "A"),
            ("B_embedding_raw", "B"),
            ("D_embedding_raw_temporal_flow", "D"),
        ):
            tbl.append(
                f"| {label} | {_msd(v, arm, 'auprc')} | {_msd(v, arm, 'precision_at_100')} | "
                f"{_msd(v, arm, 'recall_at_500')} | {_msd(v, arm, 'recall_at_1000')} | "
                f"{_msd(v, arm, 'recall_at_precision_ge_0.90')} | "
                f"{_msd(v, arm, 'recall_at_precision_ge_0.80')} |"
            )
        tbl.append("")
    out_tbl.write_text("\n".join(tbl) + "\n", encoding="utf-8")

    tex = [
        r"% Temporal-flow regression aux multiseed (diagnostic only; not for main tables)",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Variant & Arm & AUPRC & P@100 & R@500 & R@1000 & R@P$\geq$0.90 & R@P$\geq$0.80 \\",
        r"\midrule",
    ]
    for v in VARIANTS:
        v_tex = v.replace("_", r"\_")
        for arm, label in (
            ("A_embedding", "A"),
            ("B_embedding_raw", "B"),
            ("D_embedding_raw_temporal_flow", "D"),
        ):
            tex.append(
                f"{v_tex} & {label} & {_msd(v, arm, 'auprc')} & "
                f"{_msd(v, arm, 'precision_at_100')} & {_msd(v, arm, 'recall_at_500')} & "
                f"{_msd(v, arm, 'recall_at_1000')} & "
                f"{_msd(v, arm, 'recall_at_precision_ge_0.90')} & "
                f"{_msd(v, arm, 'recall_at_precision_ge_0.80')} \\\\"
            )
    tex.extend([r"\bottomrule", r"\end{tabular}", ""])
    out_tex.write_text("\n".join(tex), encoding="utf-8")

    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    print(f"Wrote {out_tbl}")
    print(f"Wrote {out_tex}")
    print(f"recommendation={overall} more_stable={more_stable}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
