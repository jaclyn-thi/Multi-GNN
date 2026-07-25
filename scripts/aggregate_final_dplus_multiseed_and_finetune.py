#!/usr/bin/env python3
"""Aggregate frozen D+ multiseed + secondary FT test into final thesis artifacts."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]


def _mean_std(xs: List[float]) -> Dict[str, float]:
    xs = [float(x) for x in xs]
    mean = float(statistics.mean(xs))
    std = float(statistics.stdev(xs)) if len(xs) > 1 else 0.0
    med = float(statistics.median(xs))
    return {"mean": mean, "sample_std": std, "median": med, "n": float(len(xs)), "values": xs}


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed1_json", required=True)
    p.add_argument("--seed2_json", required=True, help="Reuse 18678029-compatible frozen seed2 report")
    p.add_argument("--seed3_json", required=True)
    p.add_argument("--ft_json", required=True)
    p.add_argument(
        "--output_json",
        default="results/diagnostics/final_dplus_multiseed_and_finetune_analysis.json",
    )
    p.add_argument(
        "--output_md",
        default="notes/final_dplus_multiseed_and_finetune_analysis.md",
    )
    p.add_argument(
        "--table_md",
        default="tables/final_dplus_frozen_multiseed_primary.md",
    )
    p.add_argument(
        "--table_tex",
        default="tables/final_dplus_frozen_multiseed_primary.tex",
    )
    p.add_argument(
        "--ft_table_md",
        default="tables/final_dplus_partial_finetune_secondary.md",
    )
    p.add_argument(
        "--ft_table_tex",
        default="tables/final_dplus_partial_finetune_secondary.tex",
    )
    args = p.parse_args()

    s1 = load_json(Path(args.seed1_json))
    s2 = load_json(Path(args.seed2_json))
    s3 = load_json(Path(args.seed3_json))
    ft = load_json(Path(args.ft_json))
    seeds = [s1, s2, s3]

    def get(seed_rep, *keys, default=None):
        cur = seed_rep
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    rows = []
    for s in seeds:
        rows.append({
            "encoder_seed": s.get("encoder_seed"),
            "checkpoint_epoch": s.get("checkpoint_epoch"),
            "checkpoint_sha256": s.get("checkpoint_sha256"),
            "unique_name": s.get("unique_name"),
            "val_auprc": s.get("val_auprc"),
            "val_f1": s.get("val_f1_at_selected"),
            "test_auroc": s.get("test_auroc"),
            "test_auprc": s.get("test_auprc"),
            "test_f1_0.5": s.get("test_f1_fixed_0.5"),
            "test_precision_0.5": get(s, "metrics", "threshold_0.5", "precision"),
            "test_recall_0.5": get(s, "metrics", "threshold_0.5", "recall"),
            "test_f1_val_thr": s.get("test_f1_val_threshold"),
            "test_precision_val_thr": get(s, "metrics", "threshold_val_selected", "precision"),
            "test_recall_val_thr": get(s, "metrics", "threshold_val_selected", "recall"),
            "threshold": get(s, "metrics", "validation_selected_threshold")
            or get(s, "metrics", "threshold_val_selected", "validation_selected_threshold"),
            "P100": get(s, "metrics", "threshold_0.5", "precision_at_100"),
            "P500": get(s, "metrics", "threshold_0.5", "precision_at_500"),
            "P1000": get(s, "metrics", "threshold_0.5", "precision_at_1000"),
            "pos_rate": get(s, "metrics", "threshold_0.5", "positive_prediction_rate"),
            "tp": get(s, "metrics", "threshold_0.5", "tp"),
            "fp": get(s, "metrics", "threshold_0.5", "fp"),
            "tn": get(s, "metrics", "threshold_0.5", "tn"),
            "fn": get(s, "metrics", "threshold_0.5", "fn"),
            "coverage": s.get("coverage"),
        })

    agg = {
        "val_auprc": _mean_std([r["val_auprc"] for r in rows]),
        "val_f1": _mean_std([r["val_f1"] for r in rows]),
        "test_auroc": _mean_std([r["test_auroc"] for r in rows]),
        "test_auprc": _mean_std([r["test_auprc"] for r in rows]),
        "test_f1_0.5": _mean_std([r["test_f1_0.5"] for r in rows]),
        "test_f1_val_thr": _mean_std([r["test_f1_val_thr"] for r in rows]),
        "P100": _mean_std([r["P100"] for r in rows]),
        "P500": _mean_std([r["P500"] for r in rows]),
        "P1000": _mean_std([r["P1000"] for r in rows]),
    }

    pub_gin = 0.6479
    pub_gin_sd = 0.0122
    pub_pna = 0.6816
    pub_pna_sd = 0.0265
    our_sup = 0.660
    our_sup_sd = 0.060

    f1_mean = agg["test_f1_0.5"]["mean"]
    ft_f1 = ft["test_metrics_threshold_0.5"]["f1"]
    ft_auprc = ft["test_metrics_threshold_0.5"]["auprc"]

    answers = {
        "1_frozen_multiseed_mean_pm_sd": {
            "test_f1_0.5": f"{agg['test_f1_0.5']['mean']:.4f} ± {agg['test_f1_0.5']['sample_std']:.4f}",
            "test_auprc": f"{agg['test_auprc']['mean']:.4f} ± {agg['test_auprc']['sample_std']:.4f}",
            "test_auroc": f"{agg['test_auroc']['mean']:.4f} ± {agg['test_auroc']['sample_std']:.4f}",
            "val_auprc": f"{agg['val_auprc']['mean']:.4f} ± {agg['val_auprc']['sample_std']:.4f}",
        },
        "2_frozen_mean_numerically_exceeds_multigin_eu": bool(f1_mean > pub_gin),
        "2_detail": {
            "frozen_mean_f1_0.5": f1_mean,
            "published_multigin_eu": pub_gin,
            "delta": f1_mean - pub_gin,
        },
        "3_locked_finetuned_seed2_test": {
            "test_auprc": ft_auprc,
            "test_auroc": ft["test_metrics_threshold_0.5"]["auroc"],
            "test_f1_0.5": ft_f1,
            "test_f1_val_thr": ft["test_metrics_val_threshold"]["f1"],
            "threshold": ft["stored_validation_selected_threshold"],
            "best_epoch": ft["best_epoch"],
            "stored_val_auprc": ft["stored_best_val_auprc"],
        },
        "4_finetuning_improves_over_frozen_seed2": {
            "val_auprc": bool(ft["stored_best_val_auprc"] > 0.550),
            "test_auprc": bool(ft_auprc > 0.674),
            "test_f1_0.5": bool(ft_f1 > 0.656),
            "deltas": ft.get("comparison_to_frozen_seed2"),
        },
        "5_finetuning_numerically_exceeds_multipna_eu": bool(ft_f1 > pub_pna),
        "5_detail": {"ft_f1_0.5": ft_f1, "published_multipna_eu": pub_pna, "delta": ft_f1 - pub_pna},
        "6_abstract_conclusion_result": "frozen_dplus_multiseed_primary",
        "7_finetune_placement": "secondary_sensitivity_or_appendix",
        "8_limitations": [
            "Downstream MLP uses AML labels; pipeline is not wholly unsupervised.",
            "Feature stack / learner / downstream seed locked on seed-2 validation (18678029).",
            "Partial FT updates final encoder block with AML labels; not the primary claim.",
            "Paper baselines differ in protocol details; prefer 'numerically exceeds reported mean'.",
            "n=3 encoder seeds; sample SD is descriptive, not a formal superiority test.",
        ],
        "9_test_metrics_did_not_influence_selection": True,
        "10_no_training_or_followup_jobs_in_this_eval": True,
    }

    report = {
        "title": "final_dplus_multiseed_and_finetune_analysis",
        "hierarchy": {
            "PRIMARY": "frozen D+ multiseed (SSL encoder + frozen + supervised downstream MLP on H+X+TF)",
            "SECONDARY": "SSL-pretrained D+ with supervised partial fine-tuning (seed 2)",
        },
        "provenance": {
            "seed1_job": 18801429,
            "seed2_job": 18514684,
            "seed3_job": 18802579,
            "ft_job": 18801435,
            "frozen_seed2_eval_job": 18678029,
        },
        "per_seed": rows,
        "aggregate": agg,
        "finetune_secondary": ft,
        "comparators": {
            "published_multigin_eu_f1": pub_gin,
            "published_multigin_eu_sd": pub_gin_sd,
            "published_multipna_eu_f1": pub_pna,
            "published_multipna_eu_sd": pub_pna_sd,
            "our_supervised_multigin_eu_f1": our_sup,
            "our_supervised_multigin_eu_sd": our_sup_sd,
        },
        "answers": answers,
        "claim_language": {
            "primary": (
                "A self-supervised contrastive Multi-GIN encoder (D+: corrected reverse-edge "
                "semantics and preserve_seed_edges), evaluated with the encoder frozen and a "
                "supervised downstream MLP on pre-3h H+X+TF under a temporal split, achieves "
                f"test F1@0.5 of {agg['test_f1_0.5']['mean']:.3f} ± {agg['test_f1_0.5']['sample_std']:.3f} "
                f"over three encoder seeds."
            ),
            "secondary": (
                "As a secondary sensitivity analysis, SSL-pretrained D+ with supervised partial "
                f"fine-tuning of the final GNN block reaches test F1@0.5 of {ft_f1:.3f} on seed 2; "
                "AML labels update both the classifier and the unfrozen encoder block."
            ),
        },
    }

    out_j = ROOT / args.output_json
    out_m = ROOT / args.output_md
    out_j.parent.mkdir(parents=True, exist_ok=True)
    out_j.write_text(json.dumps(report, indent=2) + "\n")

    def fmt_pm(stat):
        return f"{stat['mean']:.4f} ± {stat['sample_std']:.4f}"

    md = []
    md += [
        "# Final D+ multiseed + fine-tune analysis",
        "",
        "## Thesis-result hierarchy (locked)",
        "",
        "1. **PRIMARY:** Self-supervised contrastive encoder evaluated using a supervised "
        "downstream classifier, with the encoder frozen (pre-3h H+X+TF MLP).",
        "2. **SECONDARY:** SSL-pretrained D+ with supervised partial fine-tuning (seed 2).",
        "",
        "## Provenance",
        "",
        "| Encoder | Job | Checkpoint epoch | sha256 |",
        "|---------|-----|------------------|--------|",
    ]
    for r in rows:
        md.append(
            f"| seed {r['encoder_seed']} | "
            f"{report['provenance']['seed1_job'] if r['encoder_seed']==1 else report['provenance']['seed2_job'] if r['encoder_seed']==2 else report['provenance']['seed3_job']} | "
            f"{r['checkpoint_epoch']} | `{r['checkpoint_sha256'][:16]}…` |"
        )
    md += [
        f"| partial FT seed2 | {report['provenance']['ft_job']} | {ft['best_epoch']} | `{ft['checkpoint_sha256'][:16]}…` |",
        "",
        "## PRIMARY — frozen D+ multiseed (H+X+TF MLP)",
        "",
        "| Seed | val AUPRC | val F1 | test AUROC | test AUPRC | F1@0.5 | F1@val-thr | P@100 | P@500 | P@1000 |",
        "|-----:|----------:|-------:|-----------:|-----------:|-------:|-----------:|------:|------:|-------:|",
    ]
    for r in rows:
        md.append(
            f"| {r['encoder_seed']} | {r['val_auprc']:.4f} | {r['val_f1']:.4f} | {r['test_auroc']:.4f} | "
            f"{r['test_auprc']:.4f} | {r['test_f1_0.5']:.4f} | {r['test_f1_val_thr']:.4f} | "
            f"{r['P100']:.3f} | {r['P500']:.3f} | {r['P1000']:.3f} |"
        )
    md += [
        f"| **mean±sd** | {fmt_pm(agg['val_auprc'])} | {fmt_pm(agg['val_f1'])} | "
        f"{fmt_pm(agg['test_auroc'])} | {fmt_pm(agg['test_auprc'])} | "
        f"**{fmt_pm(agg['test_f1_0.5'])}** | {fmt_pm(agg['test_f1_val_thr'])} | "
        f"{fmt_pm(agg['P100'])} | {fmt_pm(agg['P500'])} | {fmt_pm(agg['P1000'])} |",
        "",
        "### Recommended primary claim",
        "",
        report["claim_language"]["primary"],
        "",
        "## SECONDARY — partial fine-tune seed 2",
        "",
        f"- Stored val AUPRC @ ep {ft['best_epoch']}: **{ft['stored_best_val_auprc']:.4f}**",
        f"- Test AUPRC: **{ft_auprc:.4f}**",
        f"- Test F1@0.5: **{ft_f1:.4f}**",
        f"- Test F1@val-thr ({ft['stored_validation_selected_threshold']}): "
        f"**{ft['test_metrics_val_threshold']['f1']:.4f}**",
        f"- vs frozen seed2 (val 0.550 / test AUPRC 0.674 / F1@0.5 0.656): "
        f"ΔvalA={ft['comparison_to_frozen_seed2']['delta_stored_val_auprc']:+.4f}, "
        f"ΔtestA={ft['comparison_to_frozen_seed2']['delta_test_auprc_vs_ref']:+.4f}, "
        f"ΔF1={ft['comparison_to_frozen_seed2']['delta_test_f1_0.5_vs_ref']:+.4f}",
        "",
        report["claim_language"]["secondary"],
        "",
        "## Cautious published comparisons (fixed-0.5 F1)",
        "",
        f"- Published Multi-GIN+EU: {pub_gin:.4f} ± {pub_gin_sd:.4f}",
        f"- Published Multi-PNA+EU: {pub_pna:.4f} ± {pub_pna_sd:.4f}",
        f"- Our supervised Multi-GIN+EU: {our_sup:.3f} ± {our_sup_sd:.3f}",
        f"- Frozen D+ mean: **{f1_mean:.4f}** "
        f"({'numerically exceeds' if f1_mean > pub_gin else 'does not exceed'} Multi-GIN+EU mean)",
        f"- Partial FT seed2: **{ft_f1:.4f}** "
        f"({'numerically exceeds' if ft_f1 > pub_pna else 'does not exceed'} Multi-PNA+EU mean; "
        "encoder partially updated with AML labels)",
        "",
        "## Final answers",
        "",
    ]
    for k, v in answers.items():
        md.append(f"- **{k}:** `{json.dumps(v) if not isinstance(v, str) else v}`")
    md += [
        "",
        "## Confirmations",
        "",
        "- Test metrics did **not** influence model/checkpoint/feature/learner/threshold selection.",
        "- No GNN retraining and no automatic follow-up training jobs in this evaluation stage.",
        "",
    ]
    out_m.write_text("\n".join(md) + "\n")

    # Compact primary table
    tmd = [
        "# Primary: frozen D+ multiseed (pre-3h H+X+TF MLP)",
        "",
        "Self-supervised contrastive encoder; encoder frozen; supervised downstream MLP.",
        "",
        "| Seed | val AUPRC | test AUPRC | test F1@0.5 |",
        "|-----:|----------:|-----------:|------------:|",
    ]
    for r in rows:
        tmd.append(f"| {r['encoder_seed']} | {r['val_auprc']:.4f} | {r['test_auprc']:.4f} | {r['test_f1_0.5']:.4f} |")
    tmd.append(
        f"| mean±sd | {fmt_pm(agg['val_auprc'])} | {fmt_pm(agg['test_auprc'])} | {fmt_pm(agg['test_f1_0.5'])} |"
    )
    tmd.append("")
    Path(args.table_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.table_md).write_text("\n".join(tmd) + "\n")

    tex = [
        r"% Primary: frozen D+ multiseed",
        r"\begin{tabular}{rccc}",
        r"\toprule",
        r"Seed & val AUPRC & test AUPRC & test F1@0.5 \\",
        r"\midrule",
    ]
    for r in rows:
        tex.append(
            f"{r['encoder_seed']} & {r['val_auprc']:.4f} & {r['test_auprc']:.4f} & {r['test_f1_0.5']:.4f} \\\\"
        )
    tex += [
        r"\midrule",
        f"mean$\\pm$sd & {agg['val_auprc']['mean']:.4f}$\\pm${agg['val_auprc']['sample_std']:.4f} & "
        f"{agg['test_auprc']['mean']:.4f}$\\pm${agg['test_auprc']['sample_std']:.4f} & "
        f"{agg['test_f1_0.5']['mean']:.4f}$\\pm${agg['test_f1_0.5']['sample_std']:.4f} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        "",
    ]
    Path(args.table_tex).write_text("\n".join(tex) + "\n")

    ft_md = [
        "# Secondary: SSL-pretrained D+ with supervised partial fine-tuning (seed 2)",
        "",
        "| Metric | Frozen seed-2 ref | Partial FT best (ep 18) |",
        "|--------|------------------:|------------------------:|",
        f"| val AUPRC | 0.550 | {ft['stored_best_val_auprc']:.4f} |",
        f"| test AUPRC | 0.674 | {ft_auprc:.4f} |",
        f"| test F1@0.5 | 0.656 | {ft_f1:.4f} |",
        "",
        "AML labels update the classifier and final encoder block (`convs.1`/`emlps.1`/`batch_norms.1`).",
        "",
    ]
    Path(args.ft_table_md).write_text("\n".join(ft_md) + "\n")
    ft_tex = [
        r"% Secondary: partial fine-tune",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Metric & Frozen seed-2 & Partial FT (ep 18) \\",
        r"\midrule",
        f"val AUPRC & 0.550 & {ft['stored_best_val_auprc']:.4f} \\\\",
        f"test AUPRC & 0.674 & {ft_auprc:.4f} \\\\",
        f"test F1@0.5 & 0.656 & {ft_f1:.4f} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        "",
    ]
    Path(args.ft_table_tex).write_text("\n".join(ft_tex) + "\n")
    print(json.dumps(answers, indent=2))


if __name__ == "__main__":
    main()
