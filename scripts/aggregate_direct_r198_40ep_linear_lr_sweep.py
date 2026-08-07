#!/usr/bin/env python3
"""Aggregate DIRECT_R198 40ep linear-LR sweep: tables, figures, CE audit, report.

Runs after the four eval jobs (afterany). Missing cells reported, never fabricated.
Writes under results/diagnostics/direct_r198_tfmoe_40ep_linear_lr_sweep/ and notes/.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/diagnostics/direct_r198_tfmoe_40ep_linear_lr_sweep"
FIG = OUT / "figures"
NOTES = ROOT / "notes/direct_r198_tfmoe_40ep_linear_lr_sweep.md"

ARMS = [
    {
        "run": "direct_r198_infonce_40ep_seed2_linear_lr6p2e-3",
        "arm": "DIRECT_R198",
        "peak_lr": 0.006213266113989207,
        "label": "InfoNCE",
        "ls": "-",
    },
    {
        "run": "direct_r198_tfmoe_40ep_seed2_linear_lr6p2e-3",
        "arm": "DIRECT_R198_TFMOE",
        "peak_lr": 0.006213266113989207,
        "label": "InfoNCE+TF",
        "ls": "-",
    },
    {
        "run": "direct_r198_infonce_40ep_seed2_linear_lr2e-3",
        "arm": "DIRECT_R198",
        "peak_lr": 0.002,
        "label": "InfoNCE",
        "ls": "--",
    },
    {
        "run": "direct_r198_tfmoe_40ep_seed2_linear_lr2e-3",
        "arm": "DIRECT_R198_TFMOE",
        "peak_lr": 0.002,
        "label": "InfoNCE+TF",
        "ls": "--",
    },
]

C_H = "#0072B2"
C_T = "#D55E00"
TF_NAMES = (
    "sender interarrival",
    "sender recent 7-day count",
    "amount relative to sender history",
)
C_TF = ["#009E73", "#CC79A7", "#E69F00"]
CALIB = 100  # expected; actual from logs
EVAL_EPS = [3, 10, 20, 30, 40]

mpl.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 12,
        "figure.dpi": 140,
        "savefig.dpi": 200,
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def _load_steps(run: str) -> Optional[pd.DataFrame]:
    p = ROOT / "results/diagnostics" / run / "logs" / "steps.jsonl"
    if not p.is_file():
        return None
    rows = [json.loads(l) for l in p.open()]
    return pd.DataFrame(rows) if rows else None


def _load_cell(run: str, ep: int) -> Optional[Dict[str, Any]]:
    p = OUT / "cells" / run / f"epoch_{ep:02d}.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text())


def _save(fig, stem: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{stem}.png", bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _legend_out(ax):
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8)


def _color(arm: str) -> str:
    return C_T if "TFMOE" in arm else C_H


def plot_training(series_rows: List[Dict]) -> None:
    # 1 raw InfoNCE all arms
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    for a in ARMS:
        df = _load_steps(a["run"])
        if df is None or "L_contrast_raw" not in df:
            continue
        ax.plot(
            df["optimizer_step_index"],
            df["L_contrast_raw"],
            color=_color(a["arm"]),
            ls=a["ls"],
            lw=1.4,
            label=f"{a['label']} lr={a['peak_lr']:.4g}",
        )
        for x, y in zip(df["optimizer_step_index"], df["L_contrast_raw"]):
            series_rows.append(
                {
                    "figure": "01_raw_infonce",
                    "series": a["run"],
                    "x": float(x),
                    "y": float(y),
                }
            )
    ax.axvline(CALIB, color="#56B4E9", ls=":", lw=1.0)
    ax.set_title("Raw InfoNCE")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Raw InfoNCE")
    _legend_out(ax)
    fig.tight_layout()
    _save(fig, "01_raw_infonce")

    # 2 TF raw MAE by LR panel
    tf_arms = [a for a in ARMS if a["arm"] == "DIRECT_R198_TFMOE"]
    if tf_arms:
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=True)
        for ax, a in zip(axes, tf_arms):
            df = _load_steps(a["run"])
            if df is None:
                ax.set_title(f"lr={a['peak_lr']:.4g} (missing)")
                continue
            for m, (name, col) in enumerate(zip(TF_NAMES, C_TF)):
                if f"L_tf_raw_{m}" not in df:
                    continue
                ax.plot(df["optimizer_step_index"], df[f"L_tf_raw_{m}"], color=col, lw=1.3, label=name)
            ax.axvline(CALIB, color="#56B4E9", ls=":", lw=1.0)
            ax.set_title(f"TF raw MAE · lr={a['peak_lr']:.4g}")
            ax.set_xlabel("Optimizer step")
            if ax is axes[0]:
                ax.set_ylabel("Raw MAE")
            _legend_out(ax)
        fig.tight_layout()
        _save(fig, "02_tf_raw_mae_by_lr")

    # 3 weighted objective post-calib by LR
    if tf_arms:
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=True)
        for ax, a in zip(axes, tf_arms):
            df = _load_steps(a["run"])
            if df is None or "weighted_contrast" not in df:
                ax.set_title(f"lr={a['peak_lr']:.4g} (missing)")
                continue
            post = df[df["optimizer_step_index"] >= CALIB].copy()
            if post.empty:
                continue
            stack = [
                post["weighted_contrast"].to_numpy(),
                (post["w_tf_0"] * post["L_tf_norm_0"]).to_numpy(),
                (post["w_tf_1"] * post["L_tf_norm_1"]).to_numpy(),
                (post["w_tf_2"] * post["L_tf_norm_2"]).to_numpy(),
            ]
            ax.stackplot(
                post["optimizer_step_index"],
                stack,
                colors=[C_T, *C_TF],
                labels=["InfoNCE", *TF_NAMES],
                alpha=0.8,
            )
            ax.plot(post["optimizer_step_index"], post["L_total"], color="k", lw=1.3, label=r"$L_{total}$")
            ax.set_ylim(0, 1.2)
            ax.set_title(f"Weighted objective · lr={a['peak_lr']:.4g}")
            ax.set_xlabel("Optimizer step")
            _legend_out(ax)
        fig.tight_layout()
        _save(fig, "03_weighted_objective_by_lr")

    # 4 effective weights by LR
    if tf_arms:
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=True)
        for ax, a in zip(axes, tf_arms):
            df = _load_steps(a["run"])
            if df is None or "w_contrast" not in df:
                continue
            ax.plot(df["optimizer_step_index"], df["w_contrast"], color=C_T, lw=1.5, label="InfoNCE")
            for m, (name, col) in enumerate(zip(TF_NAMES, C_TF)):
                ax.plot(df["optimizer_step_index"], df[f"w_tf_{m}"], color=col, lw=1.3, label=name)
            ax.axvline(CALIB, color="#56B4E9", ls=":", lw=1.0)
            ax.set_title(f"Effective weights · lr={a['peak_lr']:.4g}")
            ax.set_xlabel("Optimizer step")
            ax.set_ylim(0, 0.75)
            _legend_out(ax)
        fig.tight_layout()
        _save(fig, "04_effective_weights_by_lr")

    # 5 encoder LR
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    seen = set()
    for a in ARMS:
        key = round(a["peak_lr"], 8)
        if key in seen:
            continue
        seen.add(key)
        df = _load_steps(a["run"])
        if df is None:
            continue
        # dedupe by optimizer step
        g = df.groupby("optimizer_step_index", as_index=False).last()
        ax.plot(
            g["optimizer_step_index"],
            g["encoder_lr"],
            lw=1.6,
            label=f"peak={a['peak_lr']:.4g}",
        )
    ax.axvline(CALIB, color="#56B4E9", ls=":", lw=1.0)
    ax.set_title("Encoder learning rate")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("LR")
    _legend_out(ax)
    fig.tight_layout()
    _save(fig, "05_encoder_lr")

    # 6 geometry vs checkpoint
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for a in ARMS:
        eps, norms, ranks = [], [], []
        for ep in EVAL_EPS:
            c = _load_cell(a["run"], ep)
            if not c or c.get("status") != "ok":
                continue
            eps.append(ep)
            norms.append(c["repr_val"]["mean_l2_norm"])
            ranks.append(c["repr_val"]["effective_rank"])
        if not eps:
            continue
        axes[0].plot(eps, norms, marker="o", color=_color(a["arm"]), ls=a["ls"], label=f"{a['label']} {a['peak_lr']:.4g}")
        axes[1].plot(eps, ranks, marker="o", color=_color(a["arm"]), ls=a["ls"], label=f"{a['label']} {a['peak_lr']:.4g}")
    axes[0].set_title("Mean R198 L2 norm")
    axes[1].set_title("Effective rank")
    for ax in axes:
        ax.set_xlabel("SSL epoch")
        _legend_out(ax)
    fig.tight_layout()
    _save(fig, "06_representation_geometry")


def plot_downstream(series_rows: List[Dict]) -> None:
    def _series(metric_fn, title, stem, ylabel):
        fig, ax = plt.subplots(figsize=(7.5, 3.8))
        for a in ARMS:
            xs, ys = [], []
            for ep in EVAL_EPS:
                c = _load_cell(a["run"], ep)
                if not c or c.get("status") != "ok":
                    continue
                xs.append(ep)
                ys.append(metric_fn(c))
                series_rows.append({"figure": stem, "series": a["run"], "x": ep, "y": ys[-1]})
            if xs:
                ax.plot(
                    xs,
                    ys,
                    marker="o",
                    color=_color(a["arm"]),
                    ls=a["ls"],
                    lw=1.5,
                    label=f"{a['label']} lr={a['peak_lr']:.4g}",
                )
        ax.set_title(title)
        ax.set_xlabel("SSL checkpoint epoch")
        ax.set_ylabel(ylabel)
        ax.set_xticks(EVAL_EPS)
        _legend_out(ax)
        fig.tight_layout()
        _save(fig, stem)

    _series(lambda c: c["primary"]["validation_auprc"], "Validation AUPRC", "07_val_auprc", "AUPRC")
    _series(
        lambda c: c["primary"]["validation_metrics_at_0.5"]["f1"],
        "Validation F1 @ 0.5",
        "08_val_f1_fixed",
        "F1",
    )
    _series(
        lambda c: c["primary"]["validation_metrics_at_val_optimal_f1"]["f1"],
        "Validation F1 @ val-thr (diagnostic)",
        "09_val_f1_val_selected",
        "F1",
    )


def supervised_ce_audit() -> Dict[str, Any]:
    hist_path = ROOT / "results/diagnostics/supervised_Small-HI_small_hi_legacy_supervised_gin_emlps_ports_50ep_seed2_epoch_history.json"
    summary_path = ROOT / "results/diagnostics/supervised_Small-HI_small_hi_legacy_supervised_gin_emlps_ports_50ep_seed2_summary.json"
    audit: Dict[str, Any] = {
        "probe_loss": {
            "class": "torch.nn.functional.binary_cross_entropy_with_logits",
            "logits": "one_logit",
            "class_weights": None,
            "pos_weight": None,
            "reduction": "mean",
            "note": "PaperStyleMLP AML probe",
        },
        "supervised_loss": {
            "class": "torch.nn.CrossEntropyLoss",
            "logits": "two_logit",
            "class_weights": None,
            "reduction": "mean",
            "note": "legacy supervised Multi-GIN head",
        },
        "comparable_directly": False,
        "common_metric_for_comparison": "unweighted_binary_validation_nll_from_positive_class_proba",
        "supervised_native": {},
        "supervised_common_val_nll": {
            "status": "unavailable_in_logs",
            "smallest_offline_recompute": (
                "Val-only inference from checkpoint_last.tar and checkpoint_best_val_f1.tar "
                "with --skip_test_eval; convert 2-logit softmax P(y=1) to binary NLL "
                "mean(-(y log p + (1-y) log(1-p)))."
            ),
        },
    }
    if hist_path.is_file():
        hist = json.loads(hist_path.read_text())
        audit["supervised_loss"]["class_weights"] = hist.get("class_weights")
        eps = hist["epochs"]
        final = eps[-1]
        # best by validation minority f1 argmax
        best = max(eps, key=lambda e: float(e.get("validation_minority_f1_argmax") or -1))
        audit["supervised_native"] = {
            "final_epoch": {
                "epoch": final["epoch"],
                "train_loss_weighted_ce": final["train_loss"],
                "validation_f1_argmax": final.get("validation_minority_f1_argmax"),
                "validation_auprc": final.get("validation_auprc"),
                "split": "train_loss is training objective; validation CE not logged",
            },
            "best_validation_f1_epoch": {
                "epoch": best["epoch"],
                "train_loss_weighted_ce": best["train_loss"],
                "validation_f1_argmax": best.get("validation_minority_f1_argmax"),
                "validation_auprc": best.get("validation_auprc"),
                "diagnostic_label": "best_validation_checkpoint",
            },
        }
    if summary_path.is_file():
        s = json.loads(summary_path.read_text())
        audit["supervised_checkpoints"] = {
            "last": s.get("last_checkpoint_path"),
            "best_val": s.get("best_val_checkpoint_path"),
            "best_validation_epoch": s.get("best_validation_epoch"),
        }
    return audit


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    series_rows: List[Dict] = []
    plot_training(series_rows)
    plot_downstream(series_rows)

    # Primary performance table
    rows = []
    for a in ARMS:
        summ_path = OUT / "cells" / a["run"] / "summary.json"
        summ = json.loads(summ_path.read_text()) if summ_path.is_file() else {}
        sel = summ.get("selected_ssl_epoch_by_primary_auprc")
        ep40 = None
        c40 = _load_cell(a["run"], 40)
        if c40 and c40.get("status") == "ok":
            ep40 = c40
        for kind, src in [("selected_ssl", sel), ("epoch_40", ep40)]:
            if src is None:
                rows.append(
                    {
                        "arm": a["arm"],
                        "run": a["run"],
                        "peak_lr": a["peak_lr"],
                        "schedule": "warmup_linear",
                        "checkpoint_meaning": kind,
                        "ssl_epoch": None,
                        "val_auprc": None,
                        "f1_at_0.5": None,
                        "f1_at_val_thr": None,
                        "final_probe_train_bce": None,
                        "final_probe_val_bce": None,
                        "status": "missing",
                    }
                )
                continue
            if kind == "selected_ssl":
                rows.append(
                    {
                        "arm": a["arm"],
                        "run": a["run"],
                        "peak_lr": a["peak_lr"],
                        "schedule": "warmup_linear",
                        "checkpoint_meaning": kind,
                        "ssl_epoch": src["epoch"],
                        "val_auprc": src["validation_auprc"],
                        "f1_at_0.5": src["f1_at_0.5"],
                        "f1_at_val_thr": src["f1_at_val_thr"],
                        "final_probe_train_bce": src["final_probe_train_bce"],
                        "final_probe_val_bce": src["final_probe_val_bce"],
                        "status": "ok",
                    }
                )
            else:
                p = src["primary"]
                rows.append(
                    {
                        "arm": a["arm"],
                        "run": a["run"],
                        "peak_lr": a["peak_lr"],
                        "schedule": "warmup_linear",
                        "checkpoint_meaning": kind,
                        "ssl_epoch": 40,
                        "val_auprc": p["validation_auprc"],
                        "f1_at_0.5": p["validation_metrics_at_0.5"]["f1"],
                        "f1_at_val_thr": p["validation_metrics_at_val_optimal_f1"]["f1"],
                        "final_probe_train_bce": p["final_probe_train_bce"],
                        "final_probe_val_bce": p["final_probe_val_bce"],
                        "status": "ok",
                    }
                )
    perf = pd.DataFrame(rows)
    perf.to_csv(OUT / "primary_performance_table.csv", index=False)

    ce_audit = supervised_ce_audit()
    ce_rows = []
    for r in rows:
        if r["status"] != "ok":
            continue
        ce_rows.append(
            {
                "method": "frozen_encoder_PaperStyleMLP_probe",
                "representation_training": r["run"],
                "checkpoint_meaning": r["checkpoint_meaning"],
                "ssl_epoch": r["ssl_epoch"],
                "final_train_binary_ce": r["final_probe_train_bce"],
                "final_validation_binary_ce": r["final_probe_val_bce"],
                "loss_definition": "unweighted BCE-with-logits mean (one logit)",
            }
        )
    # supervised native
    sn = ce_audit.get("supervised_native") or {}
    for label, block in sn.items():
        ce_rows.append(
            {
                "method": "supervised_MultiGIN",
                "representation_training": "small_hi_legacy_supervised_gin_emlps_ports_50ep_seed2",
                "checkpoint_meaning": label,
                "ssl_epoch": block.get("epoch"),
                "final_train_binary_ce": block.get("train_loss_weighted_ce"),
                "final_validation_binary_ce": None,
                "loss_definition": "native weighted 2-logit CrossEntropyLoss (NOT directly comparable to probe BCE)",
            }
        )
    ce_rows.append(
        {
            "method": "supervised_MultiGIN_common_unweighted_binary_val_nll",
            "representation_training": "small_hi_legacy_supervised_gin_emlps_ports_50ep_seed2",
            "checkpoint_meaning": "final_and_best",
            "ssl_epoch": None,
            "final_train_binary_ce": None,
            "final_validation_binary_ce": None,
            "loss_definition": ce_audit["supervised_common_val_nll"]["smallest_offline_recompute"],
        }
    )
    ce_df = pd.DataFrame(ce_rows)
    ce_df.to_csv(OUT / "ce_comparison_table.csv", index=False)
    pd.DataFrame(series_rows).to_csv(OUT / "plotting_data.csv", index=False)
    (OUT / "ce_audit.json").write_text(json.dumps(ce_audit, indent=2) + "\n")

    # Aggregate JSON + manifest stubs
    dirty = ""
    try:
        import subprocess

        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=str(ROOT), text=True)
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True).strip()
    except Exception:
        commit = None
    aggregate = {
        "artifact": str(OUT),
        "arms": ARMS,
        "primary_performance": rows,
        "ce_audit": ce_audit,
        "test_evaluated": False,
        "amp": False,
        "git_commit": commit,
        "git_dirty_manifest": dirty,
    }
    (OUT / "aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n")

    # Markdown report
    lines = [
        "# DIRECT_R198 40ep linear-LR sweep",
        "",
        "Exploratory seed-2 AMLWorld (Small-HI). Test locked. AMP off.",
        "",
        "## Primary performance",
        "",
        perf.to_markdown(index=False) if hasattr(perf, "to_markdown") else perf.to_string(index=False),
        "",
        "## CE comparison",
        "",
        "Probe uses unweighted one-logit BCE-with-logits. Supervised Multi-GIN uses weighted two-logit CrossEntropyLoss;",
        "native train losses are **not** directly comparable. Common unweighted validation binary NLL requires a follow-up",
        "val-only inference (see `ce_audit.json`).",
        "",
        ce_df.to_markdown(index=False) if hasattr(ce_df, "to_markdown") else ce_df.to_string(index=False),
        "",
        "## Figures",
        "",
        "See `results/diagnostics/direct_r198_tfmoe_40ep_linear_lr_sweep/figures/`.",
        "",
    ]
    NOTES.write_text("\n".join(lines) + "\n")
    print(json.dumps({"out": str(OUT), "n_perf_rows": len(rows), "notes": str(NOTES)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
