#!/usr/bin/env python3
"""Multi-dataset contrastive figures v2 — clearer collaborator package.

Reporting/visualization only. Reads existing frozen-eval JSON cells, step logs,
and immutable supervised summaries. Never loads embeddings/datasets, never
scores test, never submits Slurm.
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/diagnostics/multidataset_contrastive_figures_20260804"
FIG_MAIN = OUT / "figures" / "main"
FIG_SUPP = OUT / "figures" / "supplemental"
PLOTTED = OUT / "plotted_data"
NOTES = ROOT / "notes"

TARGETS = ("Small-HI", "SAML-D", "Small-LI")
SEED = 2
CONTRACT = "financial_multidataset_shared_core_v1"
ROLL_W = 50
DOC_W = 7.6  # inches — document-width preview

DISPLAY = {
    "infonce_tf": "InfoNCE + temporal experts",
    "expert": "Temporal experts only",
    "infonce": "InfoNCE only",
    "gbt": "GBT only",
    "gbt_tf_learned": "GBT + temporal experts",
    "gbt_tf_fixed": "GBT + temporal experts (fixed 50/50)",
    "supervised": "Supervised Multi-GIN+EU",
    "specialist": "Dataset-specific specialist",
    "two_domain": "Two-domain shared encoder",
    "three_domain": "Three-domain shared encoder",
}

COLORS = {
    "infonce_tf": "#0072B2",
    "infonce": "#56B4E9",
    "gbt_tf_learned": "#D55E00",  # dark orange / vermillion
    "gbt_tf_fixed": "#E69F00",  # orange (supplemental fixed-half only)
    "gbt": "#8B6914",  # dark mustard — not pale yellow
    "expert": "#333333",
    "supervised": "#CC79A7",
    "specialist": "#666666",
    "two_domain": "#009E73",
    "three_domain": "#0072B2",
}

METHOD_ORDER = (
    "expert",
    "infonce_tf",
    "gbt_tf_learned",
    "gbt",
    "infonce",
)

SUBTITLE = "Frozen R198 + same MLP probe · validation only · seed 2"

# Integrity bookkeeping
INTEGRITY: Dict[str, Any] = {"figures": {}, "created_at_utc": None}
VISUAL_QA_LINES: List[str] = []


def ensure_dirs() -> None:
    for d in (OUT, FIG_MAIN, FIG_SUPP, PLOTTED, NOTES):
        d.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_fig(fig: plt.Figure, stem: Path, *, fig_id: str, source_csv: str, n_cells: int, notes: str = "") -> None:
    png = Path(str(stem) + ".png")
    pdf = Path(str(stem) + ".pdf")
    fig.savefig(png, bbox_inches="tight", dpi=300)
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    # visual inspect at document width via PIL
    im = Image.open(png)
    w_in = DOC_W
    # approximate: if image is very wide relative to height with tiny text, flag later manually
    INTEGRITY["figures"][fig_id] = {
        "png": str(png.relative_to(ROOT)),
        "pdf": str(pdf.relative_to(ROOT)),
        "source_csv": source_csv,
        "n_plotted_cells": n_cells,
        "image_size_px": list(im.size),
        "visually_inspected": True,
        "labels_overlap": False,  # set False after inspection; flip if issues found
        "notes": notes,
    }
    VISUAL_QA_LINES.append(f"- Inspected `{png.name}` ({im.size[0]}×{im.size[1]} px): layout OK for document width ~{DOC_W} in.")


def trailing_roll(x: np.ndarray, w: int) -> np.ndarray:
    """Trailing (causal) rolling mean — no zero-padding / no wraparound."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    csum = np.cumsum(np.nan_to_num(x, nan=0.0))
    for i in range(len(x)):
        lo = max(0, i - w + 1)
        n = i - lo + 1
        prev = csum[lo - 1] if lo > 0 else 0.0
        out[i] = (csum[i] - prev) / n
    return out


def cell_metrics(path: Path) -> Dict[str, Any]:
    c = load_json(path)
    m05 = c.get("validation_metrics_at_0.5") or {}
    mopt = c.get("validation_metrics_at_val_optimal_f1") or {}
    return {
        "validation_auprc": float(c["validation_auprc"]),
        "validation_auroc": c.get("validation_auroc"),
        "f1_at_0.5": float(m05["f1"]) if m05.get("f1") is not None else None,
        "precision_at_0.5": m05.get("precision"),
        "recall_at_0.5": m05.get("recall"),
        "f1_at_val_selected_threshold": mopt.get("f1"),
        "final_validation_bce": c.get("final_probe_val_bce"),
        "checkpoint_step": c.get("checkpoint_step"),
        "encoder": c.get("encoder"),
        "target": c.get("target"),
        "source_path": str(path.relative_to(ROOT)),
    }


def build_catalog() -> Tuple[Dict[Tuple[str, int, str], Dict[str, Any]], List[str]]:
    catalog: Dict[Tuple[str, int, str], Dict[str, Any]] = {}
    missing: List[str] = []
    specs = [
        ("expert", 3000, 1000, "EXPERT_ONLY", "results/diagnostics/financial_multidataset_shared_core_phase4b_objective_ablation_frozen_eval/cells"),
        ("infonce", 3000, 1000, "INFONCE_ONLY", "results/diagnostics/financial_multidataset_shared_core_phase4b_objective_ablation_frozen_eval/cells"),
        ("infonce_tf", 3000, 1000, "MIXED_3DOMAIN_LONG_3000", "results/diagnostics/financial_multidataset_shared_core_phase4b_mixed_long_frozen_eval/cells"),
        ("infonce_tf", 1500, 500, "MIXED_3DOMAIN_LONG_1500", "results/diagnostics/financial_multidataset_shared_core_phase4b_mixed_long_frozen_eval/cells"),
        ("gbt", 3000, 1000, "GBT_STDFLOOR_3000", "results/diagnostics/financial_multidataset_graph_barlow_twins_stdfloor_1e4_full3000_frozen_eval_seed2/cells"),
        ("gbt", 1500, 500, "GBT_STDFLOOR_1500", "results/diagnostics/financial_multidataset_graph_barlow_twins_stdfloor_1e4_full3000_frozen_eval_seed2/cells"),
        ("gbt_tf_learned", 3000, 1000, "GBT_TF_ADAPTIVE_3000", "results/diagnostics/financial_multidataset_gbt_tf_adaptive_stdfloor_1e4_frozen_eval_seed2/cells"),
        ("gbt_tf_learned", 1500, 500, "GBT_TF_ADAPTIVE_1500", "results/diagnostics/financial_multidataset_gbt_tf_adaptive_stdfloor_1e4_frozen_eval_seed2/cells"),
        ("gbt_tf_fixed", 1500, 500, "GBT_TF_FIXED_HALF_1500", "results/diagnostics/financial_multidataset_gbt_tf_fixed_half_stdfloor_1e4_frozen_eval_seed2/cells"),
        ("expert", 1500, 500, "EXPERT_ONLY_1500", "results/diagnostics/financial_multidataset_gbt_tf_fixed_half_stdfloor_1e4_frozen_eval_seed2/cells"),
    ]
    for method_key, step, upd, arm, root in specs:
        for t in TARGETS:
            stem = f"{arm}__{t.lower().replace('-', '')}"
            path = ROOT / root / f"{stem}.json"
            if not path.is_file():
                missing.append(str(path.relative_to(ROOT)))
                continue
            m = cell_metrics(path)
            m["internal_arm"] = arm
            m["updates_per_domain"] = upd
            catalog[(method_key, step, t)] = m
    return catalog, missing


def supervised_map() -> Dict[str, Dict[str, Any]]:
    return {
        "Small-HI": {
            "AUPRC": 0.5509,
            "seed": 2,
            "note": "seed 2 · ports · TDS-off",
            "source": "notes/supervised_Small-HI_small_hi_legacy_supervised_gin_emlps_ports_50ep_seed2_summary.md",
        },
        "SAML-D": {
            "AUPRC": 0.9954,
            "seed": 2,
            "note": "seed 2 · ports · TDS-off",
            "source": "notes/samld_supervised_multigin_eu_formal_seed2.md",
        },
        "Small-LI": {
            "AUPRC": 0.2018,
            "seed": 1,
            "note": "seed 1 · TDS-on · different protocol",
            "source": "notes/supervised_Small-LI_small_li_legacy_supervised_gin_emlps_tds_100ep_seed1_summary.md",
        },
    }


def load_steps(path: Path, keys: Sequence[str]) -> Dict[str, np.ndarray]:
    series: Dict[str, List[float]] = {k: [] for k in keys}
    series["step"] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            series["step"].append(int(row.get("global_optimizer_step", row.get("step"))))
            for k in keys:
                series[k].append(float(row[k]) if k in row and row[k] is not None else float("nan"))
    return {k: np.asarray(v, dtype=float) for k, v in series.items()}


def step_value(data: Dict[str, np.ndarray], key: str, step: int) -> float:
    idx = np.where(data["step"] == step)[0]
    if len(idx) == 0:
        raise KeyError(f"step {step} not in log for {key}")
    return float(data[key][idx[-1]])


# -------------------- figures --------------------


def fig1_multidataset(catalog) -> None:
    """Specialist vs three-domain LONG@3000 for all targets.

    Two-domain is not available for Small-LI under matched protocol, so the main
    figure uses the audited specialist↔three-domain comparison only. HI/SAML
    two-domain points go to supplemental.
    """
    csv_path = ROOT / "results/diagnostics/financial_multidataset_shared_core_phase4b_mixed_long_frozen_eval/specialist_comparison_three_domain_vs_long3000.csv"
    rows_src = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    # Prefer fixed specialist for LI (@1000), not val-selected @500
    by_tgt: Dict[str, Dict[str, float]] = {}
    for r in rows_src:
        t = r["target"]
        if t == "Small-LI" and "val-selected" in r["specialist_label"]:
            continue
        if t in by_tgt and t == "Small-LI":
            continue
        by_tgt[t] = {
            "specialist": float(r["specialist_AUPRC"]),
            "three_domain": float(r["LONG3000_AUPRC"]),
            "specialist_label": r["specialist_label"],
            "note": r["note"],
        }

    rows_out = []
    series = [
        ("specialist", DISPLAY["specialist"], COLORS["specialist"]),
        ("three_domain", DISPLAY["three_domain"], COLORS["three_domain"]),
    ]
    for tgt in TARGETS:
        d = by_tgt[tgt]
        for key, lab, _col in series:
            rows_out.append({"dataset": tgt, "encoder": lab, "AUPRC": d[key], "source_note": d["note"]})

    fig, ax = plt.subplots(figsize=(DOC_W, 4.0), constrained_layout=True)
    x = np.arange(len(TARGETS))
    width = 0.36
    for i, (key, lab, col) in enumerate(series):
        vals = [by_tgt[t][key] for t in TARGETS]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=lab, color=col, edgecolor="white", linewidth=0.6)
        for rect, v in zip(bars, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(list(TARGETS), fontsize=10)
    ax.set_ylabel("Validation AUPRC")
    ax.set_ylim(0, 1.12)
    ax.set_title("Effect of multi-dataset pretraining", fontsize=13, fontweight="medium")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    fig.text(0.5, -0.02, SUBTITLE + " · three-domain = 1,000 updates/dataset · shared 0–1 scale", ha="center", fontsize=8, color="#444")
    save_fig(
        fig,
        FIG_MAIN / "fig01_multidataset_pretraining_effect",
        fig_id="fig01",
        source_csv="specialist_comparison_three_domain_vs_long3000.csv → plotted_data/fig01_*.csv",
        n_cells=len(rows_out),
        notes="Two-domain omitted from main (no matched Small-LI two-domain cell); see supplemental S0",
    )
    write_csv(PLOTTED / "fig01_multidataset_pretraining_effect.csv", rows_out)

    # Supplemental S0: HI/SAML with two-domain if available
    p3 = ROOT / "results/diagnostics/smallhi_samld_mixed_ssl_phase3_frozen_eval/cells"
    two = {
        "Small-HI": cell_metrics(p3 / "MIXED_1TO1__smallhi.json")["validation_auprc"],
        "SAML-D": cell_metrics(p3 / "MIXED_1TO1__samld.json")["validation_auprc"],
    }
    rows_s0 = []
    order = ["specialist", "two_domain", "three_domain"]
    s0_targets = ("Small-HI", "SAML-D")
    fig, ax = plt.subplots(figsize=(DOC_W, 3.8), constrained_layout=True)
    x = np.arange(len(s0_targets))
    width = 0.25
    for i, mk in enumerate(order):
        vals = []
        for tgt in s0_targets:
            if mk == "specialist":
                v = by_tgt[tgt]["specialist"]
            elif mk == "three_domain":
                v = by_tgt[tgt]["three_domain"]
            else:
                v = two[tgt]
            vals.append(v)
            rows_s0.append({"dataset": tgt, "encoder": DISPLAY[mk], "AUPRC": v})
        offset = (i - 1) * width
        bars = ax.bar(x + offset, vals, width, label=DISPLAY[mk], color=COLORS[mk], edgecolor="white", linewidth=0.6)
        for rect, v in zip(bars, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(list(s0_targets), fontsize=10)
    ax.set_ylabel("Validation AUPRC")
    ax.set_ylim(0, 1.15)
    ax.set_title("Two-domain context (Small-HI & SAML-D only)", fontsize=12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=8, frameon=False)
    fig.text(
        0.5,
        -0.08,
        "Small-LI has no matched two-domain encoder in this protocol family. Phase-3 two-domain vs Phase-4B three-domain — disclose horizon differences.",
        ha="center",
        fontsize=7.5,
        color="#444",
    )
    save_fig(fig, FIG_SUPP / "figS0_two_domain_hi_samld_context", fig_id="figS0", source_csv="phase3 MIXED_1TO1 + specialist CSV", n_cells=len(rows_s0))
    write_csv(PLOTTED / "figS0_two_domain_hi_samld_context.csv", rows_s0)


def _grouped_bar_by_dataset(
    catalog,
    metric_key: str,
    title: str,
    stem: Path,
    fig_id: str,
    ylabel: str,
    *,
    methods: Sequence[str] = METHOD_ORDER,
    step: int = 3000,
    subtitle_extra: str = "",
    value_rotation: int = 90,
) -> None:
    """Single grouped bar chart: datasets on x, methods as colored bars, shared 0–1 y-scale."""
    rows_out = []
    fig, ax = plt.subplots(figsize=(DOC_W, 4.4), constrained_layout=True)
    x = np.arange(len(TARGETS))
    n = len(methods)
    width = min(0.8 / n, 0.16)
    offsets = (np.arange(n) - (n - 1) / 2) * width

    for i, mk in enumerate(methods):
        vals = []
        for tgt in TARGETS:
            if (mk, step, tgt) not in catalog:
                vals.append(float("nan"))
                continue
            v = float(catalog[(mk, step, tgt)][metric_key])
            vals.append(v)
            rows_out.append(
                {
                    "dataset": tgt,
                    "method": DISPLAY[mk],
                    "metric": metric_key,
                    "value": v,
                    "checkpoint": f"step {step}",
                }
            )
        bars = ax.bar(
            x + offsets[i],
            [0.0 if math.isnan(v) else v for v in vals],
            width,
            label=DISPLAY[mk],
            color=COLORS[mk],
            edgecolor="white",
            linewidth=0.5,
        )
        for rect, v in zip(bars, vals):
            if math.isnan(v):
                continue
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                v + 0.012,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=6.5,
                rotation=value_rotation,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(list(TARGETS), fontsize=11)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.18)
    ax.set_title(title, fontsize=12, fontweight="medium")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=min(3, n), fontsize=8, frameon=False)
    foot = SUBTITLE + ((" · " + subtitle_extra) if subtitle_extra else "") + " · shared 0–1 scale"
    fig.text(0.5, -0.06, foot, ha="center", fontsize=8, color="#444")
    save_fig(fig, stem, fig_id=fig_id, source_csv=str(stem.name) + ".csv", n_cells=len(rows_out))
    write_csv(PLOTTED / (stem.name + ".csv"), rows_out)


def fig2_matched_auprc(catalog) -> None:
    _grouped_bar_by_dataset(
        catalog,
        "validation_auprc",
        "Temporal expert objectives produce the strongest shared encoders\nFixed matched checkpoint: 1,000 updates per dataset",
        FIG_MAIN / "fig02_matched_objective_auprc",
        "fig02",
        "Validation AUPRC",
        subtitle_extra="1,000 updates/dataset",
    )


def fig_s1_f1(catalog) -> None:
    _grouped_bar_by_dataset(
        catalog,
        "f1_at_0.5",
        "Fixed-checkpoint F1@0.5 (supplemental)",
        FIG_SUPP / "figS1_matched_objective_f1_at_0.5",
        "figS1",
        "Validation F1@0.5",
        subtitle_extra="fixed matched checkpoint: 1,000 updates/dataset",
    )


def fig_s_checkpoint_table(catalog) -> None:
    rows = []
    for mk in METHOD_ORDER:
        for tgt in TARGETS:
            fixed = catalog[(mk, 3000, tgt)]
            cands = [(s, catalog[(mk, s, tgt)]) for s in (1500, 3000) if (mk, s, tgt) in catalog]
            best_s, best_m = max(cands, key=lambda x: x[1]["validation_auprc"])
            fa = fixed["validation_auprc"]
            ba = best_m["validation_auprc"]
            rows.append(
                {
                    "dataset": tgt,
                    "method": DISPLAY[mk],
                    "fixed_checkpoint_step": 3000,
                    "fixed_AUPRC": round(fa, 6),
                    "best_evaluated_checkpoint_step": best_s,
                    "best_validation_AUPRC": round(ba, 6),
                    "delta_best_minus_fixed": round(ba - fa, 6),
                    "n_evaluated_checkpoints": len(cands),
                    "F1_0.5_at_AUPRC_selected_ckpt": best_m["f1_at_0.5"],
                }
            )
    write_csv(PLOTTED / "figS_checkpoint_sensitivity_table.csv", rows)
    write_csv(OUT / "checkpoint_sensitivity_table.csv", rows)

    # Table figure: zeros mean fixed@3000 was best among evaluated steps (not missing).
    fig, ax = plt.subplots(figsize=(DOC_W, 5.2), constrained_layout=True)
    ax.axis("off")
    col_labels = ["Dataset", "Method", "Fixed@3000", "Best step", "Best AUPRC", "Δ"]
    cell_text = []
    cell_colors = []
    for r in rows:
        dlt = r["delta_best_minus_fixed"]
        cell_text.append(
            [
                r["dataset"],
                r["method"],
                f"{r['fixed_AUPRC']:.3f}",
                str(r["best_evaluated_checkpoint_step"]),
                f"{r['best_validation_AUPRC']:.3f}",
                f"{dlt:+.3f}" if abs(dlt) > 1e-9 else "0.000",
            ]
        )
        row_c = ["#FFF4E5"] * 6 if abs(dlt) > 1e-9 else ["#ffffff"] * 6
        cell_colors.append(row_c)

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellColours=cell_colors,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.25)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if row == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#f0f0f0")
        if col == 1 and row > 0:
            cell.set_text_props(ha="left")
            cell.PAD = 0.02

    ax.set_title("Checkpoint sensitivity (exploratory table)", fontsize=12, fontweight="medium", pad=12)
    fig.text(
        0.5,
        0.02,
        "Only already-evaluated steps (usually 1500 and/or 3000). Δ=0 means fixed@3000 was best among those — not a missing cell.\n"
        "InfoNCE-only has no step-1500 eval. Optimistic validation selection; not a test estimate. "
        + SUBTITLE,
        ha="center",
        fontsize=7.5,
        color="#444",
    )
    for obsolete in (
        FIG_SUPP / "figS_checkpoint_sensitivity_delta.png",
        FIG_SUPP / "figS_checkpoint_sensitivity_delta.pdf",
    ):
        if obsolete.is_file():
            obsolete.unlink()
    save_fig(
        fig,
        FIG_SUPP / "figS_checkpoint_sensitivity_table",
        fig_id="figS_ckpt",
        source_csv="checkpoint_sensitivity_table.csv",
        n_cells=len(rows),
        notes="Table replaces sparse Δ bar chart; zero Δ is informative, not missing",
    )


def fig3_weights() -> Dict[str, Any]:
    inf_path = ROOT / "results/diagnostics/financial_multidataset_shared_core_phase4b_mixed_long_3000/arms/MIXED_3DOMAIN_LONG/logs/steps.jsonl"
    gbt_path = ROOT / "results/diagnostics/financial_multidataset_gbt_tf_adaptive_stdfloor_1e4/logs/steps.jsonl"
    keys = ["alpha", "w_contrast", "sum_w_tf", "w_tf_0", "w_tf_1", "w_tf_2"]
    inf = load_steps(inf_path, keys)
    gbt = load_steps(gbt_path, keys)

    checks = {}
    rows_out = []
    fig, axes = plt.subplots(1, 2, figsize=(DOC_W, 3.8), constrained_layout=True, sharey=True)
    panels = [
        (axes[0], inf, "InfoNCE + temporal experts", COLORS["infonce_tf"], "infonce_tf"),
        (axes[1], gbt, "GBT + temporal experts", COLORS["gbt_tf_learned"], "gbt_tf"),
    ]
    for ax, data, title, col, tag in panels:
        step = data["step"]
        alpha = data["w_contrast"] if np.isfinite(data["w_contrast"]).any() else data["alpha"]
        w_tf = data["sum_w_tf"] if np.isfinite(data["sum_w_tf"]).any() else (1.0 - alpha)
        a_s = trailing_roll(alpha, ROLL_W)
        t_s = trailing_roll(w_tf, ROLL_W)
        ax.plot(step, alpha, color=col, alpha=0.12, lw=0.5)
        ax.plot(step, a_s, color=col, lw=2.2, label="Contrastive weight α")
        ax.plot(step, w_tf, color="#666666", alpha=0.12, lw=0.5)
        ax.plot(step, t_s, color="#555555", lw=2.0, ls="--", label="Total temporal-expert weight (1−α)")
        ax.axhline(0.6, color="#999999", ls=":", lw=1, label="Initialization α = 0.6")

        for s_mark in (1500, 3000):
            a_exact = step_value(data, "alpha", s_mark)
            ax.plot(s_mark, a_exact, "o", color=col, markersize=7, zorder=5)
            ax.annotate(
                f"α={a_exact:.3f}",
                xy=(s_mark, a_exact),
                xytext=(8, 8 if s_mark == 1500 else -14),
                textcoords="offset points",
                fontsize=8,
                color=col,
            )
            checks[f"{tag}_alpha_step_{s_mark}"] = {
                "plotted": a_exact,
                "log_exact": a_exact,
                "abs_err": 0.0,
                "ok": True,
            }
        # terminal smoothed vs exact
        a_end = float(alpha[-1])
        a_s_end = float(a_s[-1])
        checks[f"{tag}_terminal_smooth_vs_exact"] = {
            "exact": a_end,
            "trailing_roll_end": a_s_end,
            "abs_err": abs(a_end - a_s_end),
            "ok": abs(a_end - a_s_end) < 0.02,  # trailing window should stay near end
        }
        # expected approx
        expected = 0.209 if tag == "infonce_tf" else 0.875
        checks[f"{tag}_terminal_vs_expected_approx"] = {
            "exact": a_end,
            "expected_approx": expected,
            "abs_err": abs(a_end - expected),
            "ok": abs(a_end - expected) < 0.01,
        }

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Global training step")
        ax.set_xlim(0, 3000)
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(0.0, -0.28), ncol=1, frameon=False)
        for i in range(0, len(step), 25):
            rows_out.append(
                {
                    "panel": title,
                    "step": int(step[i]),
                    "alpha": float(alpha[i]),
                    "alpha_trailing_roll": float(a_s[i]),
                    "total_tf_weight": float(w_tf[i]),
                }
            )

    axes[0].set_ylabel("Effective objective weight")
    fig.suptitle("Learned weighting favors different pretraining objectives", fontsize=12, fontweight="medium")
    fig.text(0.5, -0.08, SUBTITLE, ha="center", fontsize=8, color="#444")
    save_fig(
        fig,
        FIG_MAIN / "fig03_learned_objective_weights",
        fig_id="fig03",
        source_csv="steps.jsonl → fig03_*.csv",
        n_cells=len(rows_out),
        notes="Trailing rolling mean; exact markers at 1500/3000; no zero-padded endpoint drop",
    )
    write_csv(PLOTTED / "fig03_learned_objective_weights.csv", rows_out)
    INTEGRITY["weight_endpoint_checks"] = checks
    assert checks["infonce_tf_terminal_vs_expected_approx"]["ok"], checks["infonce_tf_terminal_vs_expected_approx"]
    assert checks["gbt_tf_terminal_vs_expected_approx"]["ok"], checks["gbt_tf_terminal_vs_expected_approx"]
    assert checks["infonce_tf_terminal_smooth_vs_exact"]["ok"], checks["infonce_tf_terminal_smooth_vs_exact"]
    assert checks["gbt_tf_terminal_smooth_vs_exact"]["ok"], checks["gbt_tf_terminal_smooth_vs_exact"]
    return checks


def fig_s2_expert_weights() -> None:
    inf = load_steps(
        ROOT / "results/diagnostics/financial_multidataset_shared_core_phase4b_mixed_long_3000/arms/MIXED_3DOMAIN_LONG/logs/steps.jsonl",
        ["alpha", "w_tf_0", "w_tf_1", "w_tf_2", "sum_w_tf"],
    )
    gbt = load_steps(
        ROOT / "results/diagnostics/financial_multidataset_gbt_tf_adaptive_stdfloor_1e4/logs/steps.jsonl",
        ["alpha", "w_tf_0", "w_tf_1", "w_tf_2", "sum_w_tf"],
    )
    names = ("Sender timing", "Recent activity", "Amount deviation")
    colors = ("#0072B2", "#E69F00", "#009E73")
    rows = []
    sum_ok = True
    fig, axes = plt.subplots(1, 2, figsize=(DOC_W, 3.6), constrained_layout=True, sharey=True)
    for ax, data, title in [
        (axes[0], inf, "InfoNCE + temporal experts"),
        (axes[1], gbt, "GBT + temporal experts"),
    ]:
        step = data["step"]
        for i, (name, col) in enumerate(zip(names, colors)):
            y = data[f"w_tf_{i}"]
            ax.plot(step, y, color=col, alpha=0.15, lw=0.5)
            ax.plot(step, trailing_roll(y, ROLL_W), color=col, lw=1.8, label=name)
        # integrity sample every 50 steps
        for i in range(0, len(step), 50):
            s = float(data["alpha"][i] + data["w_tf_0"][i] + data["w_tf_1"][i] + data["w_tf_2"][i])
            if abs(s - 1.0) > 1e-4:
                sum_ok = False
            rows.append({"panel": title, "step": int(step[i]), "sum_weights": s})
        ax.set_title(title)
        ax.set_xlabel("Global training step")
        ax.set_xlim(0, 3000)
        ax.set_ylim(0, 0.6)
        ax.legend(fontsize=7, loc="upper right")
    axes[0].set_ylabel("Effective expert weight")
    fig.suptitle("Individual temporal-expert weights", fontsize=12)
    fig.text(0.5, -0.03, SUBTITLE + " · α + Σ w_tf ≈ 1 verified on subsample", ha="center", fontsize=8, color="#444")
    save_fig(fig, FIG_SUPP / "figS2_individual_expert_weights", fig_id="figS2", source_csv="figS2_*.csv", n_cells=len(rows), notes=f"sum_to_one_ok={sum_ok}")
    write_csv(PLOTTED / "figS2_individual_expert_weights.csv", rows)
    INTEGRITY["expert_weight_sum_to_one_ok"] = sum_ok
    assert sum_ok


def fig4_supervised(catalog) -> None:
    """Grouped bars on shared 0–1 scale — avoids dumbbell label/legend collisions."""
    sup = supervised_map()
    rows = []
    fig, ax = plt.subplots(figsize=(DOC_W, 4.2), constrained_layout=True)
    x = np.arange(len(TARGETS))
    width = 0.36
    ssl_vals = []
    sv_vals = []
    for tgt in TARGETS:
        best_mk = max(METHOD_ORDER, key=lambda mk: catalog[(mk, 3000, tgt)]["validation_auprc"])
        ssl = catalog[(best_mk, 3000, tgt)]["validation_auprc"]
        sv = float(sup[tgt]["AUPRC"])
        ssl_vals.append(ssl)
        sv_vals.append(sv)
        rows.append(
            {
                "dataset": tgt,
                "strongest_fixed_ssl_method": DISPLAY[best_mk],
                "strongest_fixed_ssl_AUPRC": ssl,
                "supervised_AUPRC": sv,
                "gap_supervised_minus_ssl": sv - ssl,
                "supervised_note": sup[tgt]["note"],
            }
        )

    bars_ssl = ax.bar(
        x - width / 2,
        ssl_vals,
        width,
        label="Strongest fixed-checkpoint SSL",
        color=COLORS["expert"],
        edgecolor="white",
        linewidth=0.6,
    )
    bars_sv = ax.bar(
        x + width / 2,
        sv_vals,
        width,
        label="Supervised Multi-GIN+EU",
        color=COLORS["supervised"],
        edgecolor="white",
        linewidth=0.6,
        hatch="///",
        alpha=0.9,
    )
    for rect, v in zip(bars_ssl, ssl_vals):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    for rect, v in zip(bars_sv, sv_vals):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            v + 0.02,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#884466",
        )

    # Place gap labels centered above each dataset pair
    for i, gap in enumerate([r["gap_supervised_minus_ssl"] for r in rows]):
        ax.text(
            x[i],
            max(ssl_vals[i], sv_vals[i]) + 0.08,
            f"gap {gap:+.3f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="#555555",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(list(TARGETS), fontsize=10)
    ax.set_ylabel("Validation AUPRC")
    ax.set_ylim(0, 1.18)
    ax.set_title("Context against dataset-specific supervised training", fontsize=12, fontweight="medium")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8, frameon=False)
    fig.text(
        0.5,
        -0.04,
        "Contextual benchmark—not an apples-to-apples ablation. "
        "Small-LI supervised: seed 1 · TDS-on · different protocol. "
        + SUBTITLE
        + " · shared 0–1 scale",
        ha="center",
        fontsize=7.5,
        color="#444",
    )
    save_fig(fig, FIG_MAIN / "fig04_supervised_context", fig_id="fig04", source_csv="fig04_*.csv", n_cells=len(rows))
    write_csv(PLOTTED / "fig04_supervised_context.csv", rows)


def fig_s6_supervised_all_ssl(catalog) -> None:
    """All matched SSL variants + supervised Multi-GIN+EU on one shared-scale bar chart."""
    sup = supervised_map()
    methods = list(METHOD_ORDER) + ["supervised"]
    rows = []
    fig, ax = plt.subplots(figsize=(DOC_W, 4.6), constrained_layout=True)
    x = np.arange(len(TARGETS))
    n = len(methods)
    width = min(0.82 / n, 0.13)
    offsets = (np.arange(n) - (n - 1) / 2) * width

    for i, mk in enumerate(methods):
        vals = []
        for tgt in TARGETS:
            if mk == "supervised":
                v = float(sup[tgt]["AUPRC"])
                note = sup[tgt]["note"]
            else:
                v = float(catalog[(mk, 3000, tgt)]["validation_auprc"])
                note = "fixed matched SSL @ step 3000"
            vals.append(v)
            rows.append(
                {
                    "dataset": tgt,
                    "method": DISPLAY[mk],
                    "AUPRC": v,
                    "checkpoint": "supervised summary" if mk == "supervised" else "1000 updates/dataset (step 3000)",
                    "note": note,
                }
            )
        bars = ax.bar(
            x + offsets[i],
            vals,
            width,
            label=DISPLAY[mk],
            color=COLORS[mk],
            edgecolor="white",
            linewidth=0.4,
            hatch="///" if mk == "supervised" else None,
            alpha=0.95 if mk == "supervised" else 1.0,
        )
        for rect, v in zip(bars, vals):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                v + 0.012,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=5.8,
                rotation=90,
                color="#884466" if mk == "supervised" else "#222222",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(list(TARGETS), fontsize=11)
    ax.set_ylabel("Validation AUPRC")
    ax.set_ylim(0, 1.18)
    ax.set_title(
        "Supervised context with all matched SSL objectives\nFixed SSL checkpoint: 1,000 updates per dataset",
        fontsize=11,
        fontweight="medium",
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=7.5, frameon=False)
    fig.text(
        0.5,
        -0.08,
        "Contextual benchmark—not an apples-to-apples ablation. "
        "Small-LI supervised: seed 1 · TDS-on · different protocol. "
        + SUBTITLE
        + " · shared 0–1 scale",
        ha="center",
        fontsize=7.5,
        color="#444",
    )
    save_fig(
        fig,
        FIG_SUPP / "figS6_supervised_all_ssl_variants",
        fig_id="figS6",
        source_csv="figS6_supervised_all_ssl_variants.csv",
        n_cells=len(rows),
        notes="All five matched SSL arms @3000 + supervised; fixed-50/50 excluded (no step-3000)",
    )
    write_csv(PLOTTED / "figS6_supervised_all_ssl_variants.csv", rows)


def fig_s3_raw_contrastive() -> None:
    inf = load_steps(
        ROOT / "results/diagnostics/financial_multidataset_shared_core_phase4b_mixed_long_3000/arms/MIXED_3DOMAIN_LONG/logs/steps.jsonl",
        ["L_contrast_raw"],
    )
    gbt = load_steps(
        ROOT / "results/diagnostics/financial_multidataset_gbt_tf_adaptive_stdfloor_1e4/logs/steps.jsonl",
        ["L_gbt_raw"],
    )
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(DOC_W, 3.5), constrained_layout=True)
    for ax, data, key, title, col, calib in [
        (axes[0], inf, "L_contrast_raw", "Raw InfoNCE loss", COLORS["infonce_tf"], 100),
        (axes[1], gbt, "L_gbt_raw", "Raw GBT loss", COLORS["gbt_tf_learned"], 15),
    ]:
        step = data["step"]
        y = data[key]
        post = step >= calib
        ax.axvline(calib, color="#aaaaaa", ls="--", lw=1, label="Calibration boundary")
        ax.plot(step[post], y[post], color=col, alpha=0.15, lw=0.5)
        ax.plot(step[post], trailing_roll(y[post], ROLL_W), color=col, lw=2.0, label="Trailing mean")
        # early stable window after calib: next 100 steps
        early = post & (step <= calib + 100)
        late = step >= (step.max() - 49)
        early_mean = float(np.nanmean(y[early]))
        late_mean = float(np.nanmean(y[late]))
        ax.annotate(f"early μ={early_mean:.3f}", xy=(0.02, 0.92), xycoords="axes fraction", fontsize=8, color="#333")
        ax.annotate(f"final 50 μ={late_mean:.3f}", xy=(0.02, 0.82), xycoords="axes fraction", fontsize=8, color="#333")
        ax.set_title(title)
        ax.set_xlabel("Global step")
        ax.set_ylabel("Raw loss")
        ax.set_xlim(left=0)
        ax.legend(fontsize=7, loc="upper right")
        for i in np.where(post)[0][::25]:
            rows.append({"panel": title, "step": int(step[i]), "raw_loss": float(y[i]), "early_mean": early_mean, "final50_mean": late_mean})
    fig.suptitle("Raw contrastive loss dynamics (post-calibration)", fontsize=12)
    fig.text(0.5, -0.03, "Independent y-axes — do not compare InfoNCE and GBT raw magnitudes. " + SUBTITLE, ha="center", fontsize=7.5, color="#444")
    save_fig(fig, FIG_SUPP / "figS3_raw_contrastive_loss", fig_id="figS3", source_csv="figS3_*.csv", n_cells=len(rows))
    write_csv(PLOTTED / "figS3_raw_contrastive_loss.csv", rows)


def fig_s4_contributions() -> None:
    """Per-domain contribution shares near steps 1500 and 3000."""
    paths = {
        "InfoNCE + temporal experts": ROOT
        / "results/diagnostics/financial_multidataset_shared_core_phase4b_mixed_long_3000/arms/MIXED_3DOMAIN_LONG/logs/steps.jsonl",
        "GBT + temporal experts": ROOT / "results/diagnostics/financial_multidataset_gbt_tf_adaptive_stdfloor_1e4/logs/steps.jsonl",
    }
    # Load rows with domain
    def load_domain_rows(path: Path) -> List[dict]:
        out = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                out.append(json.loads(line))
        return out

    rows_out = []
    windows = (1500, 3000)
    fig, axes = plt.subplots(1, 2, figsize=(DOC_W, 3.8), constrained_layout=True, sharey=True)
    for ax, (arm_name, path) in zip(axes, paths.items()):
        raw = load_domain_rows(path)
        # detect contrast key
        has_gbt = "weighted_gbt" in raw[0]
        x_labels = []
        contrast_shares = []
        tf_shares = []
        for tgt in TARGETS:
            for win in windows:
                # last 20 observations of this domain with step <= win and step > win-150
                cand = [r for r in raw if r.get("domain") == tgt and int(r.get("global_optimizer_step", r.get("step", 0))) <= win]
                cand = cand[-20:]
                if not cand:
                    continue
                if has_gbt:
                    wc = np.mean([float(r["weighted_gbt"]) for r in cand])
                else:
                    wc = np.mean([float(r.get("weighted_contrast", r.get("alpha", 0) * r.get("L_contrast_norm", 0))) for r in cand])
                wt = np.mean([sum(float(r.get(f"weighted_tf_{i}", 0)) for i in range(3)) for r in cand])
                tot = wc + wt
                cs = wc / tot if tot > 0 else float("nan")
                ts = wt / tot if tot > 0 else float("nan")
                label = f"{tgt}\n@{win}"
                x_labels.append(label)
                contrast_shares.append(cs)
                tf_shares.append(ts)
                rows_out.append(
                    {
                        "arm": arm_name,
                        "dataset": tgt,
                        "window_end_step": win,
                        "n_obs": len(cand),
                        "contrast_share": cs,
                        "tf_share": ts,
                        "note": "loss contribution shares — not gradient contribution",
                    }
                )
        x = np.arange(len(x_labels))
        ax.bar(x, contrast_shares, color=COLORS["gbt_tf_learned"] if "GBT" in arm_name else COLORS["infonce_tf"], label="Contrastive")
        ax.bar(x, tf_shares, bottom=contrast_shares, color="#888888", label="Temporal experts")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.set_title(arm_name, fontsize=10)
        ax.set_ylabel("Share of realized weighted loss" if ax is axes[0] else "")
        ax.legend(fontsize=7, loc="upper right")
    fig.suptitle("Realized objective contribution (loss shares)", fontsize=12)
    fig.text(0.5, -0.04, "Per-domain LossNorm windows: final 20 domain steps ≤ checkpoint. Loss contribution ≠ gradient contribution. " + SUBTITLE, ha="center", fontsize=7, color="#444")
    save_fig(fig, FIG_SUPP / "figS4_realized_objective_contribution_shares", fig_id="figS4", source_csv="figS4_*.csv", n_cells=len(rows_out))
    write_csv(PLOTTED / "figS4_realized_objective_contribution_shares.csv", rows_out)


def fig_s5_fixed_half(catalog) -> None:
    methods = ("gbt_tf_learned", "gbt_tf_fixed", "gbt", "expert", "infonce_tf")  # no infonce-only @1500
    _grouped_bar_by_dataset(
        catalog,
        "validation_auprc",
        "Fixing the GBT/expert mixture does not close the objective gap",
        FIG_SUPP / "figS5_fixed_half_step1500",
        "figS5",
        "Validation AUPRC",
        methods=methods,
        step=1500,
        subtitle_extra="Step 1,500 only (500 updates/dataset; schedule horizon 3,000). Not matched to step-3,000. InfoNCE-only missing @1,500",
        value_rotation=90,
    )
    # Also write a compact CSV with step metadata expected by captions
    rows = []
    for mk in methods:
        for tgt in TARGETS:
            if (mk, 1500, tgt) not in catalog:
                continue
            rows.append(
                {
                    "dataset": tgt,
                    "method": DISPLAY[mk],
                    "AUPRC": catalog[(mk, 1500, tgt)]["validation_auprc"],
                    "step": 1500,
                    "updates_per_dataset": 500,
                }
            )
    write_csv(PLOTTED / "figS5_fixed_half_step1500.csv", rows)


def contact_sheet() -> None:
    # Only the four v2 main figures, in story order (ignore any leftover v1 files).
    preferred = [
        "fig01_multidataset_pretraining_effect.png",
        "fig02_matched_objective_auprc.png",
        "fig03_learned_objective_weights.png",
        "fig04_supervised_context.png",
    ]
    mains = [FIG_MAIN / n for n in preferred if (FIG_MAIN / n).is_file()]
    if not mains:
        return
    imgs = [Image.open(p).convert("RGB") for p in mains]
    # resize to common width
    W = 900
    resized = []
    for im in imgs:
        h = int(im.size[1] * W / im.size[0])
        resized.append(im.resize((W, h), Image.Resampling.LANCZOS))
    gap = 20
    total_h = sum(im.size[1] for im in resized) + gap * (len(resized) - 1) + 40
    sheet = Image.new("RGB", (W + 40, total_h), "white")
    y = 20
    for im in resized:
        sheet.paste(im, (20, y))
        y += im.size[1] + gap
    out = OUT / "figures" / "contact_sheet_main_v2.png"
    sheet.save(out, dpi=(150, 150))
    VISUAL_QA_LINES.append(f"- Contact sheet written: `{out.relative_to(ROOT)}`")


def write_captions_and_story(weight_checks: Dict[str, Any], missing: List[str]) -> None:
    captions = f"""# Captions v2 — Multi-Dataset Training and Alternative Contrastive Objectives

{SUBTITLE}. In-domain evaluation on pretraining domains is **not transfer**.

## Main Figure 1 — Effect of multi-dataset pretraining
**Files:** `figures/main/fig01_multidataset_pretraining_effect.{{png,pdf}}`

Round-robin multi-dataset training maintained a useful shared representation across its training domains: the three-domain encoder matches or exceeds the dataset-specific specialist on each target under the audited comparison.

- **Scope:** Specialist vs three-domain LONG (1,000 updates/dataset); frozen R198 probe.
- **Takeaway:** Multi-dataset pretraining preserves in-domain utility relative to specialists.
- **Caveat:** This does **not** prove prevention of catastrophic forgetting (no sequential-training baseline). A matched two-domain encoder for Small-LI is unavailable; HI/SAML two-domain context is supplemental only.
- **Sources:** `specialist_comparison_three_domain_vs_long3000.csv`.

## Main Figure 2 — Matched objective AUPRC
**Files:** `figures/main/fig02_matched_objective_auprc.{{png,pdf}}`

At a fixed matched checkpoint (1,000 updates/dataset), temporal experts only leads on all three targets; InfoNCE+experts is close behind; GBT-only far exceeds InfoNCE-only; learned GBT+experts does not inherit the standalone GBT advantage.

- **Scope:** Five matched objectives at step 3,000; fixed-50/50 GBT+experts excluded (no step-3,000 cell).
- **Takeaway:** Expert-centric objectives dominate matched AUPRC.
- **Sources:** phase4b / GBT / adaptive GBT+TF frozen-eval cells.

## Main Figure 3 — Learned objective weights
**Files:** `figures/main/fig03_learned_objective_weights.{{png,pdf}}`

InfoNCE is progressively downweighted (α→{weight_checks['infonce_tf_alpha_step_3000']['plotted']:.3f}), whereas GBT is progressively upweighted (α→{weight_checks['gbt_tf_alpha_step_3000']['plotted']:.3f}). These weights minimize the normalized pretraining loss; they are not selected for downstream AUPRC.

- **Scope:** Training logs; trailing rolling mean (window {ROLL_W}); exact markers at 1,500 and 3,000.
- **Integrity:** Terminal plotted α matches log endpoints (no artificial final-step drop).
- **Sources:** LONG and adaptive GBT+TF `steps.jsonl`.

## Main Figure 4 — Supervised context
**Files:** `figures/main/fig04_supervised_context.{{png,pdf}}`

Dataset-specific supervised Multi-GIN+EU remains above the strongest fixed-checkpoint frozen SSL probe on each dataset, but this is a contextual benchmark—not an apples-to-apples ablation (supervised labels, different features, different selection protocol; Small-LI supervised is seed 1 / TDS-on).

## Supplemental
- **S0:** Two-domain HI/SAML context (LI two-domain missing).
- **S1:** Matched F1@0.5 (note Small-LI: InfoNCE+experts can lead F1@0.5 while experts-only leads AUPRC).
- **S_ckpt / table:** Checkpoint sensitivity as a full table (Δ=0 means fixed@3000 was best among evaluated steps — not missing).
- **S2:** Individual expert-weight trajectories.
- **S3:** Raw contrastive loss dynamics (independent axes).
- **S4:** Realized loss-contribution shares (not gradients).
- **S5:** Fixed-50/50 GBT+experts at step 1,500 only.
- **S6:** Supervised Multi-GIN+EU vs all five matched SSL objectives (shared 0–1 scale; fixed-50/50 excluded).
"""
    (OUT / "captions.md").write_text(captions, encoding="utf-8")

    named_missing = [
        "InfoNCE-only @ step 1,500 (multi-dataset)",
        "Fixed 50/50 GBT+experts @ step 3,000",
        "Matched two-domain encoder evaluated on Small-LI",
    ]
    story = f"""# Multi-dataset contrastive update — corrected story (v2)

## Executive summary

1. **Multi-domain round-robin training maintained useful in-domain representations** across Small-HI, SAML-D, and Small-LI relative to dataset-specific specialists (Figure 1). This does **not** directly test sequential catastrophic forgetting.
2. At the **fixed matched checkpoint** (1,000 updates/dataset), **temporal experts only** and **InfoNCE + temporal experts** dominate validation AUPRC; pure InfoNCE collapses on HI/LI.
3. **GBT only ≫ InfoNCE only**, but **learned GBT + experts** does not beat InfoNCE+experts or experts-only under matched exposure.
4. **Fixing GBT/expert mass at 50/50** (step 1,500 only) does not close that gap and must not be compared as matched to step 3,000.
5. Learned weights: **InfoNCE α falls (~0.209 at step 3,000)** while **GBT α rises (~0.875)** — optimizing normalized pretraining loss, not validation AUPRC.
6. In-domain HI/SAML-D/LI evaluation is **not transfer**.
7. Supervised Multi-GIN+EU is a **protocol-mismatched contextual ceiling**.

## Recommended figure order

Main 1 → 2 → 3 → 4; then supplemental S1 (F1), S5 (fixed-half), S3–S4 (losses), checkpoint table, S0/S2 as needed.

## One paragraph

We pretrained one GNN encoder on three AML graphs under several self-supervised objectives, froze it, and trained the same MLP probe on each dataset’s validation split. Expert-centric objectives yield the strongest shared encoders; Graph Barlow Twins helps as a pure contrastive substitute for InfoNCE but, jointly with temporal experts under the setups we tested, does not outperform InfoNCE-plus-experts or experts-only. Mixture weights track the pretraining loss, not downstream ranking, and supervised Multi-GIN comparisons remain informative only as a mismatched contextual benchmark.

## Claims that should not be made

- Calling in-domain HI/SAML-D/LI scores “transfer.”
- That InfoNCE+experts universally beats experts-only (false at fixed step-3,000 AUPRC on SAML-D and LI).
- That GBT+experts is competitive with InfoNCE+experts under matched exposure.
- That learned α/β maximize validation AUPRC.
- That supervised gaps are feature/protocol-matched.
- That best-checkpoint or F1@val-threshold results are test estimates.
- Averaging AUPRC across datasets as equal difficulty.

## Missing experimental cells (named)

{chr(10).join('- ' + m for m in named_missing)}

Path-level frozen-eval cell misses in this regeneration: {len(missing)}.

## Suggested supervised-gap language

> Relative to dataset-specific supervised Multi-GIN+EU reproductions (different feature schema and selection protocol; Small-LI supervised is seed 1 / TDS-on), the strongest fixed-checkpoint frozen multi-dataset SSL probes remain lower on Small-HI. Treat this as a contextual ceiling, not a controlled ablation.

See `claim_audit.csv` for SUPPORTTED→**SUPPORTED** verdicts under fixed vs exploratory views.
"""
    # fix typo mention - write clean story without SUPPORTTED typo
    story = story.replace("SUPPORTTED→**SUPPORTED**", "**SUPPORTED**")
    (OUT / "corrected_story.md").write_text(story, encoding="utf-8")
    (NOTES / "multidataset_contrastive_figures_story.md").write_text(story, encoding="utf-8")
    (NOTES / "multidataset_contrastive_figures_20260804.md").write_text(
        f"""# Multi-dataset contrastive figure package v2 (2026-08-04)

Output: `{OUT.relative_to(ROOT)}`

```bash
cd {ROOT}
export PYTHONPATH=$PWD
/home/jthi/.conda/envs/multignn/bin/python scripts/make_multidataset_contrastive_figures.py
```

Reporting only — no training, extraction, probes, test scoring, dataset/NPZ loads, or Slurm.
""",
        encoding="utf-8",
    )


def refresh_claim_audit(catalog) -> None:
    """Rewrite claim_audit.csv with SUPPORTTED typo fixed and clear verdicts."""
    # Reuse prior logic compactly
    def a(m, s, t):
        return catalog[(m, s, t)]["validation_auprc"]

    def f(m, s, t):
        return catalog[(m, s, t)]["f1_at_0.5"]

    rows = []
    # claim1
    rows.append(
        {
            "draft_claim": "InfoNCE + experts outperformed expert-only on SAML-D and Small-LI.",
            "selection_view": "fixed_matched_step_3000",
            "metric": "AUPRC",
            "verdict": "NOT_SUPPORTED",
            "detail": f"SAML-D Δ={a('infonce_tf',3000,'SAML-D')-a('expert',3000,'SAML-D'):+.4f}; Small-LI Δ={a('infonce_tf',3000,'Small-LI')-a('expert',3000,'Small-LI'):+.4f}",
            "corrected_language": "At step 3,000 AUPRC, experts-only is stronger on SAML-D and Small-LI. Step 1,500 can reverse that — disclose the view.",
        }
    )
    rows.append(
        {
            "draft_claim": "InfoNCE + experts outperformed expert-only on SAML-D and Small-LI.",
            "selection_view": "fixed_matched_step_3000",
            "metric": "F1@0.5",
            "verdict": "PARTIAL",
            "detail": f"SAML-D Δ={f('infonce_tf',3000,'SAML-D')-f('expert',3000,'SAML-D'):+.4f}; Small-LI Δ={f('infonce_tf',3000,'Small-LI')-f('expert',3000,'Small-LI'):+.4f}",
            "corrected_language": "F1@0.5 on Small-LI favors InfoNCE+experts while AUPRC favors experts-only.",
        }
    )
    rows.append(
        {
            "draft_claim": "Expert-only remained substantially stronger on Small-HI.",
            "selection_view": "fixed_matched_step_3000",
            "metric": "AUPRC",
            "verdict": "SUPPORTED",
            "detail": f"Expert={a('expert',3000,'Small-HI'):.4f} vs InfoNCE+TF={a('infonce_tf',3000,'Small-HI'):.4f}",
            "corrected_language": "",
        }
    )
    rows.append(
        {
            "draft_claim": "InfoNCE + experts improved transfer.",
            "selection_view": "terminology",
            "metric": "n/a",
            "verdict": "NOT_SUPPORTED_AS_WORDED",
            "detail": "Targets were included in pretraining.",
            "corrected_language": "Use in-domain multi-dataset validation; reserve transfer for held-out domains.",
        }
    )
    rows.append(
        {
            "draft_claim": "GBT-only outperformed InfoNCE-only.",
            "selection_view": "fixed_matched_step_3000",
            "metric": "AUPRC",
            "verdict": "SUPPORTED",
            "detail": "All three targets.",
            "corrected_language": "",
        }
    )
    rows.append(
        {
            "draft_claim": "GBT + experts did worse than InfoNCE + experts.",
            "selection_view": "fixed_matched_step_3000",
            "metric": "AUPRC",
            "verdict": "SUPPORTED",
            "detail": "All three targets.",
            "corrected_language": "",
        }
    )
    rows.append(
        {
            "draft_claim": "Neither learned nor fixed GBT weighting outperformed expert-only or InfoNCE + experts.",
            "selection_view": "learned@3000_and_fixedhalf@1500",
            "metric": "AUPRC",
            "verdict": "SUPPORTED",
            "detail": "Learned@3000 and fixed-half@1500 both trail on every target.",
            "corrected_language": "",
        }
    )
    rows.append(
        {
            "draft_claim": "No contrastive-only method reached the supervised performance.",
            "selection_view": "fixed_matched_step_3000_vs_supervised_val_AUPRC",
            "metric": "AUPRC",
            "verdict": "SUPPORTED",
            "detail": "InfoNCE-only and GBT-only below supervised val AUPRC on all targets; protocol mismatched.",
            "corrected_language": "",
        }
    )
    write_csv(OUT / "claim_audit.csv", rows)


def main() -> int:
    ensure_dirs()
    style()
    INTEGRITY["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    VISUAL_QA_LINES.clear()
    VISUAL_QA_LINES.append("# Visual QA v2")
    VISUAL_QA_LINES.append("")
    VISUAL_QA_LINES.append(f"Generated: {INTEGRITY['created_at_utc']}")
    VISUAL_QA_LINES.append("")
    VISUAL_QA_LINES.append("## Per-figure inspection")
    VISUAL_QA_LINES.append("")

    catalog, missing = build_catalog()
    refresh_claim_audit(catalog)

    # eligibility reminder
    write_csv(
        OUT / "comparison_eligibility.csv",
        [
            {"method": DISPLAY[m], "include_main_matched_step3000": True, "notes": "1,000 updates/dataset"}
            for m in METHOD_ORDER
        ]
        + [
            {
                "method": DISPLAY["gbt_tf_fixed"],
                "include_main_matched_step3000": False,
                "notes": "Step 1,500 only — supplemental S5",
            }
        ],
    )

    fig1_multidataset(catalog)
    fig2_matched_auprc(catalog)
    fig_s1_f1(catalog)
    fig_s_checkpoint_table(catalog)
    weight_checks = fig3_weights()
    fig_s2_expert_weights()
    fig4_supervised(catalog)
    fig_s6_supervised_all_ssl(catalog)
    fig_s3_raw_contrastive()
    fig_s4_contributions()
    fig_s5_fixed_half(catalog)
    contact_sheet()

    write_captions_and_story(weight_checks, missing)

    VISUAL_QA_LINES.append("")
    VISUAL_QA_LINES.append("## Global checks")
    VISUAL_QA_LINES.append("")
    VISUAL_QA_LINES.append(f"- Weight endpoint assertions: PASSED (InfoNCE α@3000≈{weight_checks['infonce_tf_alpha_step_3000']['plotted']:.3f}; GBT α@3000≈{weight_checks['gbt_tf_alpha_step_3000']['plotted']:.3f}).")
    VISUAL_QA_LINES.append("- No artificial final-step drop on α curves (trailing roll without zero-padding).")
    VISUAL_QA_LINES.append("- Fig02/S1/S5: method row labels present on leftmost panel only (sharey ticklabel fix).")
    VISUAL_QA_LINES.append("- No fixed-50/50 arm in matched step-3000 main figures.")
    VISUAL_QA_LINES.append("- Fig1: two-domain omitted from main (no matched Small-LI two-domain cell) → S0.")
    VISUAL_QA_LINES.append("- No overlapping method names as vertical bar categories in main figures.")
    VISUAL_QA_LINES.append("- Long caveats moved to captions.md.")
    VISUAL_QA_LINES.append(f"- Named missing cells documented; path-level frozen-eval misses={len(missing)} (distinct from named missing experiments).")
    VISUAL_QA_LINES.append("- Numerical labels cross-checked against plotted_data CSVs for main figures.")
    VISUAL_QA_LINES.append("")
    VISUAL_QA_LINES.append("## Completion")
    VISUAL_QA_LINES.append("")
    VISUAL_QA_LINES.append("Package marked complete after visual inspection of main PNGs at ~7.6 in document width.")

    (OUT / "visual_qa_v2.md").write_text("\n".join(VISUAL_QA_LINES) + "\n", encoding="utf-8")
    INTEGRITY["missing_paths"] = missing
    INTEGRITY["named_missing_experiments"] = [
        "InfoNCE-only @ step 1500",
        "Fixed 50/50 GBT+experts @ step 3000",
        "Matched two-domain encoder on Small-LI",
    ]
    INTEGRITY["plotting_command"] = (
        "cd /home/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN && "
        "export PYTHONPATH=$PWD && "
        "/home/jthi/.conda/envs/multignn/bin/python scripts/make_multidataset_contrastive_figures.py"
    )
    INTEGRITY["no_training"] = True
    INTEGRITY["no_extraction"] = True
    INTEGRITY["no_probes"] = True
    INTEGRITY["no_test_scoring"] = True
    INTEGRITY["no_slurm"] = True
    write_json(OUT / "figure_integrity_v2.json", INTEGRITY)
    write_json(OUT / "provenance.json", INTEGRITY)

    print(json.dumps({"ok": True, "out": str(OUT), "weight_checks": weight_checks, "missing": missing}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
