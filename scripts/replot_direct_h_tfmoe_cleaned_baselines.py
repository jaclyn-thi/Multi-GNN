#!/usr/bin/env python3
"""Offline replot: drop projected baselines; keep clear supervised refs.

Does not re-probe or change DIRECT_H / DIRECT_H_TFMOE metric values.
Avoids importing torch (analyze_direct_h_tfmoe_scheduled_val.py pulls CUDA).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results/diagnostics/direct_h_tfmoe_scheduled_val_analysis"
FIG_DIR = OUT_DIR / "figures"
JSON_OUT = ROOT / "results/diagnostics/direct_h_tfmoe_scheduled_val_analysis.json"
MD_OUT = ROOT / "notes/direct_h_tfmoe_scheduled_val_analysis.md"
EPOCHS = (1, 3, 5, 10)

REF_SUPERVISED = {
    "run": "small_hi_legacy_supervised_gin_emlps_ports_50ep_seed2",
    "horizon_epochs": 50,
    "seed": 2,
    "tds": False,
    "note": "Paper-faithful Multi-GIN+EU (ports; TDS off). DIRECT_H recipe uses TDS on.",
    "source": "notes/supervised_Small-HI_small_hi_legacy_supervised_gin_emlps_ports_50ep_seed2_summary.md",
    "validation_f1": 0.6101,
    "validation_f1_decision_rule": "argmax over two-class logits",
    "validation_f1_checkpoint": "best validation minority F1 (epoch 43)",
    "validation_auprc": 0.5509,
    "best_epoch": 43,
    "plot_label_f1": "Supervised Multi-GIN+EU, validation F1 (argmax), seed 2",
    "plot_label_auprc": "Supervised Multi-GIN+EU, validation AUPRC, seed 2",
}

OMIT_REASON = (
    "Removed from DIRECT_H figure/table package: projected F1@opt≈0.571096 "
    "(and related projected AUPRC/F1@0.5 refs) lacked unambiguous "
    "threshold/split/seed presentation for these analyses."
)


def _answers(report: Dict[str, Any]) -> Dict[str, str]:
    sel = report["selection"]
    h = sel["DIRECT_H"]
    t = sel["DIRECT_H_TFMOE"]
    d_a = t["validation_auprc"] - h["validation_auprc"]
    d_f = t["f1_opt"] - h["f1_opt"]
    tm = report["cells"]["DIRECT_H_TFMOE"][str(t["epoch"])]["tfmoe"]
    ew = tm["effective_weights"]
    meaningful = []
    for i, name in enumerate(tm["target_names"]):
        if ew[f"w_tf_{i}"] >= 0.15:
            meaningful.append(f"{name} (w={ew[f'w_tf_{i}']:.3f})")
    gaps = [
        tm["val"]["mae"][name] - tm["train"]["mae"][name] for name in tm["target_names"]
    ]
    mean_gap = float(np.mean(gaps))
    if mean_gap < 0.05 and all(tm["val"]["mae"][n] < 0.5 for n in tm["target_names"]):
        gen = "a) validation-generalizing prediction (val MAE tracks train; no large gap)"
    elif mean_gap > 0.15:
        gen = "b) auxiliary overfitting (val MAE substantially worse than train)"
    else:
        gen = (
            "c) mixed / possible target misalignment: TF aux improves geometry or weights "
            f"but mean val−train MAE gap={mean_gap:.3f}; downstream ΔAUPRC={d_a:+.4f}"
        )
    h_er = report["cells"]["DIRECT_H"][str(h["epoch"])]["repr_val"]["effective_rank"]
    t_er = report["cells"]["DIRECT_H_TFMOE"][str(t["epoch"])]["repr_val"]["effective_rank"]
    h_n = report["cells"]["DIRECT_H"][str(h["epoch"])]["repr_val"]["mean_l2_norm"]
    t_n = report["cells"]["DIRECT_H_TFMOE"][str(t["epoch"])]["repr_val"]["mean_l2_norm"]
    geom = (
        f"TFMOE keeps smaller val L2 norm ({t_n:.2f} vs {h_n:.2f}) and "
        f"effective rank {t_er:.1f} vs {h_er:.1f} at each arm's selected epoch"
    )
    approach = (
        f"DIRECT_H selected val F1@val-threshold={h['f1_opt']:.4f} vs "
        f"supervised Multi-GIN+EU seed2 validation F1 (argmax)={REF_SUPERVISED['validation_f1']:.4f}; "
        f"gap={REF_SUPERVISED['validation_f1']-h['f1_opt']:+.4f}. "
        f"Does not approach supervised F1 at 10ep."
        if h["f1_opt"] < REF_SUPERVISED["validation_f1"] - 0.02
        else (
            f"Within ~2pp of supervised validation F1 "
            f"({h['f1_opt']:.4f} vs {REF_SUPERVISED['validation_f1']:.4f})."
        )
    )
    improve = (
        f"Yes (ΔAUPRC={d_a:+.4f}, ΔF1@val-threshold={d_f:+.4f} at val-selected epochs)."
        if (d_a >= 0.003 or d_f >= 0.01)
        else (
            f"No clear improvement under locked gates "
            f"(ΔAUPRC={d_a:+.4f}, ΔF1@val-threshold={d_f:+.4f}; "
            f"gates ≥0.003 AUPRC or ≥0.01 F1)."
        )
    )
    longer = (
        "Optional: longer SSL and/or BCE+MoE fallback if TF aux continues to "
        "dominate weight away from contrast without lifting primary AUPRC."
        if d_a < 0.003
        else "Optional follow-up only if TF continues to help geometry without AUPRC."
    )
    return {
        "direct_r198_approaches_supervised_val_f1": approach,
        "tfmoe_improves_over_direct_h": improve,
        "tf_objectives_with_meaningful_weight": (
            ", ".join(meaningful) if meaningful else "none above 0.15 effective weight"
        ),
        "tf_experts_generalize_or_overfit": gen,
        "tfmoe_improved_representation_geometry": geom,
        "longer_run_or_bce_moe_justified": longer,
        "no_encoder_retrain_no_test": (
            "Confirmed: frozen checkpoint extract only; extract_splits=train,val; "
            "test_evaluated=false everywhere; no test.npz written or read."
        ),
    }


def _plot_downstream(report: Dict[str, Any]) -> None:
    """Regenerate only figs 01/02 (downstream baselines). Leave other figs untouched."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    epochs = list(EPOCHS)

    def series(arm: str, stack: str, key_path: List[str]):
        ys = []
        for ep in epochs:
            node = report["cells"][arm][str(ep)][stack]
            for k in key_path:
                node = node[k]
            ys.append(float(node))
        return ys

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(epochs, series("DIRECT_H", "primary", ["validation_auprc"]), "o-", label="DIRECT_H primary")
    ax.plot(epochs, series("DIRECT_H_TFMOE", "primary", ["validation_auprc"]), "s-", label="TFMOE primary")
    ax.plot(epochs, series("DIRECT_H", "diagnostic", ["validation_auprc"]), "o--", alpha=0.7, label="DIRECT_H R198-only")
    ax.plot(epochs, series("DIRECT_H_TFMOE", "diagnostic", ["validation_auprc"]), "s--", alpha=0.7, label="TFMOE R198-only")
    ax.axhline(
        REF_SUPERVISED["validation_auprc"],
        color="black",
        ls=":",
        label=REF_SUPERVISED["plot_label_auprc"],
    )
    ax.set_xlabel("SSL epoch")
    ax.set_ylabel("Validation AUPRC")
    ax.set_title("Downstream validation AUPRC vs epoch")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_val_auprc_vs_epoch.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for arm, short, marker in (
        ("DIRECT_H", "DIRECT_H", "o"),
        ("DIRECT_H_TFMOE", "DIRECT_H_TFMOE", "s"),
    ):
        ax.plot(
            epochs,
            series(arm, "primary", ["validation_metrics_at_val_optimal_f1", "f1"]),
            f"{marker}-",
            label=f"{short} F1@val-threshold",
        )
        ax.plot(
            epochs,
            series(arm, "primary", ["validation_metrics_at_0.5", "f1"]),
            f"{marker}--",
            alpha=0.7,
            label=f"{short} F1@0.5",
        )
    ax.axhline(
        REF_SUPERVISED["validation_f1"],
        color="black",
        ls=":",
        label=REF_SUPERVISED["plot_label_f1"],
    )
    ax.set_xlabel("SSL epoch")
    ax.set_ylabel("Validation F1")
    ax.set_title("Downstream validation F1 vs SSL epoch")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_val_f1_vs_epoch.png", dpi=160)
    plt.close(fig)


def _write_md(report: Dict[str, Any]) -> None:
    cells = report["cells"]
    sel = report["selection"]
    deltas = report["deltas_tfmoe_minus_direct_h"]
    lines = []
    lines.append("# DIRECT_H / DIRECT_H_TFMOE scheduled validation analysis")
    lines.append("")
    lines.append("Locked validation-only extract + PaperStyleMLP probe for scheduled warmup+cosine runs.")
    lines.append("**No encoder retraining. No test evaluation.**")
    lines.append("")
    lines.append("## Jobs")
    lines.append("")
    lines.append("| Arm | Job | Run |")
    lines.append("|-----|-----|-----|")
    lines.append("| DIRECT_H | 19251526 (`dir_h_sched`) | `direct_h_infonce_10ep_seed2_sched` |")
    lines.append("| DIRECT_H_TFMOE | 19251528 (`dir_h_tfmoe_s`) | `direct_h_tfmoe_learned_alpha_10ep_seed2_sched` |")
    lines.append("")
    lines.append("## Primary downstream (R198+X+TF → PaperStyleMLP)")
    lines.append("")
    lines.append(
        "| Epoch | H AUPRC | H F1@0.5 | H F1@val-threshold | TF AUPRC | TF F1@0.5 | "
        "TF F1@val-threshold | ΔAUPRC | ΔF1@val-threshold |"
    )
    lines.append(
        "|------:|--------:|---------:|-------------------:|---------:|----------:"
        "|--------------------:|-------:|------------------:|"
    )
    for ep in EPOCHS:
        h = cells["DIRECT_H"][str(ep)]["primary"]
        t = cells["DIRECT_H_TFMOE"][str(ep)]["primary"]
        d = deltas[str(ep)]["primary"]
        lines.append(
            f"| {ep} | {h['validation_auprc']:.4f} | {h['validation_metrics_at_0.5']['f1']:.4f} | "
            f"{h['validation_metrics_at_val_optimal_f1']['f1']:.4f} | {t['validation_auprc']:.4f} | "
            f"{t['validation_metrics_at_0.5']['f1']:.4f} | {t['validation_metrics_at_val_optimal_f1']['f1']:.4f} | "
            f"{d['auprc']:+.4f} | {d['f1_opt']:+.4f} |"
        )
    lines.append("")
    lines.append("### Val-selected checkpoints (by primary AUPRC)")
    lines.append("")
    lines.append(
        f"- DIRECT_H: epoch **{sel['DIRECT_H']['epoch']}** "
        f"(AUPRC={sel['DIRECT_H']['validation_auprc']:.4f}, "
        f"F1@val-threshold={sel['DIRECT_H']['f1_opt']:.4f}, F1@0.5={sel['DIRECT_H']['f1_0.5']:.4f})"
    )
    lines.append(
        f"- DIRECT_H_TFMOE: epoch **{sel['DIRECT_H_TFMOE']['epoch']}** "
        f"(AUPRC={sel['DIRECT_H_TFMOE']['validation_auprc']:.4f}, "
        f"F1@val-threshold={sel['DIRECT_H_TFMOE']['f1_opt']:.4f}, "
        f"F1@0.5={sel['DIRECT_H_TFMOE']['f1_0.5']:.4f})"
    )
    lines.append("")
    lines.append("## References (validation only)")
    lines.append("")
    lines.append(
        f"- {REF_SUPERVISED['plot_label_f1']}: "
        f"**{REF_SUPERVISED['validation_f1']:.4f}** "
        f"(run `{REF_SUPERVISED['run']}`, best epoch {REF_SUPERVISED['best_epoch']}, "
        f"decision rule: {REF_SUPERVISED['validation_f1_decision_rule']}; "
        f"val AUPRC={REF_SUPERVISED['validation_auprc']:.4f}; "
        f"[source]({REF_SUPERVISED['source']}))."
    )
    lines.append(
        "- Projected-encoder baselines (including former “projected F1@opt ≈ 0.571”) "
        "are **omitted** from figures and this table: threshold/split/seed provenance "
        "was not unambiguous enough for these DIRECT_H analyses."
    )
    lines.append("")
    lines.append("## TF MoE diagnostics")
    lines.append("")
    lines.append("| Epoch | α | w_c | w_tf0 | w_tf1 | w_tf2 | train MAE | val MAE |")
    lines.append("|------:|--:|----:|------:|------:|------:|----------:|--------:|")
    for ep in EPOCHS:
        tm = cells["DIRECT_H_TFMOE"][str(ep)]["tfmoe"]
        names = tm["target_names"]
        tr = [tm["train"]["mae"][n] for n in names]
        va = [tm["val"]["mae"][n] for n in names]
        ew = tm["effective_weights"]
        lines.append(
            f"| {ep} | {tm['alpha']:.3f} | {ew['w_contrast']:.3f} | {ew['w_tf_0']:.3f} | "
            f"{ew['w_tf_1']:.3f} | {ew['w_tf_2']:.3f} | "
            f"{tr[0]:.3f}/{tr[1]:.3f}/{tr[2]:.3f} | {va[0]:.3f}/{va[1]:.3f}/{va[2]:.3f} |"
        )
    lines.append("")
    lines.append(f"Expert generalization verdict: **{report['answers']['tf_experts_generalize_or_overfit']}**")
    lines.append("")
    lines.append("## Integrity")
    lines.append("")
    integ = report["integrity"]
    lines.append(f"- First-32 seed-edge hashes match across arms: **{integ['arms_match']}**")
    lines.append(f"- unique_negs_per_anchor NaN: {integ['unique_negs_per_anchor_nan_reason']}")
    lines.append("")
    lines.append("## Figures")
    lines.append("")
    for i, name in enumerate(
        [
            "01_val_auprc_vs_epoch.png",
            "02_val_f1_vs_epoch.png",
            "03_tf_mae_train_val_vs_epoch.png",
            "04_alpha_effective_weights.png",
            "05_raw_contrastive_loss.png",
            "06_repr_scale_effective_rank.png",
            "07_lr_dual_axis.png",
        ],
        1,
    ):
        lines.append(f"{i}. `results/diagnostics/direct_h_tfmoe_scheduled_val_analysis/figures/{name}`")
    lines.append("")
    lines.append("## Answers")
    lines.append("")
    for i, (k, v) in enumerate(report["answers"].items(), 1):
        lines.append(f"{i}. **{k}:** {v}")
    lines.append("")
    MD_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    report = json.loads(JSON_OUT.read_text(encoding="utf-8"))
    # Preserve all cell/selection/ssl metrics; only refresh refs + answers text.
    report["references"] = {
        "projected_40ep": None,
        "projected_40ep_omitted_reason": OMIT_REASON,
        "supervised_multigin": REF_SUPERVISED,
    }
    report["answers"] = _answers(report)
    _plot_downstream(report)
    _write_md(report)
    text = json.dumps(report, indent=2) + "\n"
    JSON_OUT.write_text(text, encoding="utf-8")
    (OUT_DIR / "report.json").write_text(text, encoding="utf-8")

    md = MD_OUT.read_text(encoding="utf-8")
    primary = md.split("## Primary")[1].split("## References")[0]
    assert "F1@val-threshold" in primary
    assert "projected F1@opt" not in primary.lower()
    assert "F1@opt" not in primary
    # Table must not include a projected column/row
    assert "0.571096" not in md
    assert "0.5711" not in primary
    print(
        json.dumps(
            {
                "updated_json": str(JSON_OUT),
                "updated_md": str(MD_OUT),
                "figs": [
                    str(FIG_DIR / "01_val_auprc_vs_epoch.png"),
                    str(FIG_DIR / "02_val_f1_vs_epoch.png"),
                ],
                "projected_removed": 0.571096,
                "supervised_retained_f1": REF_SUPERVISED["validation_f1"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
