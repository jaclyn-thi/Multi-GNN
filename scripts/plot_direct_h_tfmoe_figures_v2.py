#!/usr/bin/env python3
"""Offline presentation figure package for scheduled DIRECT_H / DIRECT_H_TFMOE.

Writes ONLY under:
  results/diagnostics/direct_h_tfmoe_scheduled_val_analysis/figures_v2/

Does not retrain, submit jobs, touch test data, or overwrite canonical figures/.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/diagnostics/direct_h_tfmoe_scheduled_val_analysis/figures_v2"
ANALYSIS = ROOT / "results/diagnostics/direct_h_tfmoe_scheduled_val_analysis.json"
H_LOG = ROOT / "results/diagnostics/direct_h_infonce_10ep_seed2_sched/logs"
T_LOG = ROOT / "results/diagnostics/direct_h_tfmoe_learned_alpha_10ep_seed2_sched/logs"

# Okabe–Ito / colorblind-friendly
C_H = "#0072B2"  # Direct-R198 InfoNCE
C_T = "#D55E00"  # Direct-R198 InfoNCE + TF experts
C_TF = ["#009E73", "#CC79A7", "#E69F00"]  # three TF targets
C_TOTAL = "#000000"
C_REF = "#666666"

NAME_H = "Direct-R198 InfoNCE"
NAME_T = "Direct-R198 InfoNCE + TF experts"
TF_KEYS = (
    "log1p_sender_interarrival",
    "log1p_sender_past_7d_count",
    "log1p_amount_vs_sender_past_mean",
)
TF_LABELS = (
    "sender interarrival (log1p)",
    "sender recent 7-day count (log1p)",
    "amount relative to sender history (log1p)",
)
TF_SHORT = (
    "sender interarrival",
    "sender recent 7-day count",
    "amount relative to sender history",
)

CALIB_OPT = 100  # end of epoch 1 / start of epoch 2
WARMUP_END = 100
N_OPT_PLANNED = 1000
STEPS_PER_EPOCH = 100
RECON_TOL = 1e-5
ROLL = 7

mpl.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "figure.dpi": 140,
        "savefig.dpi": 200,
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def _load_steps(path: Path) -> pd.DataFrame:
    rows = [json.loads(l) for l in path.open()]
    return pd.DataFrame(rows)


def _save(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _mark_boundaries(ax, *, calib=True, warmup=True, epochs=True, ymax=None):
    if warmup:
        ax.axvline(WARMUP_END, color="#56B4E9", ls=":", lw=1.2, alpha=0.9, label="end warmup (=calib)")
    if calib and not warmup:
        ax.axvline(CALIB_OPT, color="#56B4E9", ls=":", lw=1.2, alpha=0.9, label="end calibration")
    if epochs:
        for e in range(1, 11):
            x = e * STEPS_PER_EPOCH
            if x <= N_OPT_PLANNED:
                ax.axvline(x, color="#DDDDDD", lw=0.8, zorder=0)


def _rolling_median(y: np.ndarray, window: int = ROLL) -> np.ndarray:
    s = pd.Series(y)
    return s.rolling(window, center=True, min_periods=1).median().to_numpy()


def build_tf_contrib(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["contrib_contrast"] = out["weighted_contrast"]
    for m in range(3):
        out[f"contrib_tf_{m}"] = out[f"w_tf_{m}"] * out[f"L_tf_norm_{m}"]
    out["contrib_sum"] = out["contrib_contrast"] + sum(out[f"contrib_tf_{m}"] for m in range(3))
    out["recon_abs_err"] = (out["contrib_sum"] - out["L_total"]).abs()
    out["weight_sum"] = out["w_contrast"] + sum(out[f"w_tf_{m}"] for m in range(3))
    out["calibrated"] = out["epoch"] >= 2
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    analysis = json.loads(ANALYSIS.read_text())
    h = _load_steps(H_LOG / "steps.jsonl")
    t = build_tf_contrib(_load_steps(T_LOG / "steps.jsonl"))

    # Integrity
    recon_max = float(t["recon_abs_err"].max())
    recon_mean = float(t["recon_abs_err"].mean())
    wsum_max = float((t["weight_sum"] - 1.0).abs().max())
    if recon_max > RECON_TOL:
        raise SystemExit(
            f"FAIL fig04: L_total reconstruction max abs err={recon_max} > tol={RECON_TOL}"
        )
    if wsum_max > RECON_TOL:
        raise SystemExit(f"FAIL fig06: weight sum max abs err={wsum_max} > tol={RECON_TOL}")

    # LR identity across arms on common optimizer_step_index
    hm = h.groupby("optimizer_step_index", as_index=False).last()
    tm = t.groupby("optimizer_step_index", as_index=False).last()
    common = sorted(set(hm["optimizer_step_index"]) & set(tm["optimizer_step_index"]))
    lr_diffs = [
        abs(
            float(hm.loc[hm["optimizer_step_index"] == i, "encoder_lr"].iloc[0])
            - float(tm.loc[tm["optimizer_step_index"] == i, "encoder_lr"].iloc[0])
        )
        for i in common
    ]
    lr_max_diff = float(max(lr_diffs)) if lr_diffs else float("nan")

    series_rows: List[Dict[str, Any]] = []

    def add_series(figure: str, series: str, x, y, x_name: str, extra: Optional[Dict] = None):
        for xi, yi in zip(x, y):
            if yi is None or (isinstance(yi, float) and np.isnan(yi)):
                continue
            row = {
                "figure": figure,
                "series": series,
                "x_name": x_name,
                "x": float(xi),
                "y": float(yi),
            }
            if extra:
                row.update(extra)
            series_rows.append(row)

    # ------------------------------------------------------------------ 01
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.plot(h["optimizer_step_index"], h["L_contrast_raw"], "o-", color=C_H, ms=3.5, lw=1.4, label=NAME_H)
    ax.plot(t["optimizer_step_index"], t["L_contrast_raw"], "s-", color=C_T, ms=3.5, lw=1.4, label=NAME_T)
    _mark_boundaries(ax, warmup=True, calib=False, epochs=True)
    ax.set_xlabel("Optimizer step index")
    ax.set_ylabel("Raw InfoNCE")
    ax.set_title("Raw InfoNCE component — auxiliary losses excluded")
    ax.text(0.98, 0.98, "lower is better", transform=ax.transAxes, ha="right", va="top", fontsize=9, style="italic")
    ax.legend(loc="upper right", frameon=False)
    ax.set_xlim(0, N_OPT_PLANNED)
    _save(fig, "01_raw_infonce_comparison")
    add_series("01_raw_infonce_comparison", "DIRECT_H_raw_InfoNCE", h["optimizer_step_index"], h["L_contrast_raw"], "optimizer_step_index")
    add_series("01_raw_infonce_comparison", "TFMOE_raw_InfoNCE", t["optimizer_step_index"], t["L_contrast_raw"], "optimizer_step_index")

    # ------------------------------------------------------------------ 02
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    for m, (lab, col) in enumerate(zip(TF_SHORT, C_TF)):
        ax.plot(t["optimizer_step_index"], t[f"L_tf_raw_{m}"], "-", color=col, lw=1.5, label=lab)
    ax.axvline(CALIB_OPT, color="#56B4E9", ls=":", lw=1.2, label="end calibration")
    ax.set_xlabel("Optimizer step index")
    ax.set_ylabel("Raw TF MAE (standardized targets)")
    ax.set_title("TF-expert raw MAE (not on InfoNCE scale)")
    ax.legend(loc="upper right", frameon=False)
    ax.set_xlim(0, N_OPT_PLANNED)
    _save(fig, "02_tfmoe_raw_expert_losses")
    for m, lab in enumerate(TF_SHORT):
        add_series("02_tfmoe_raw_expert_losses", f"raw_MAE_{lab}", t["optimizer_step_index"], t[f"L_tf_raw_{m}"], "optimizer_step_index")

    # ------------------------------------------------------------------ 03
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    pre = t[~t["calibrated"]]
    post = t[t["calibrated"]]
    # Pre-calib (identity scaling) — dashed / lighter
    ax.plot(pre["optimizer_step_index"], pre["L_contrast_norm"], "--", color=C_T, alpha=0.55, lw=1.3, label="norm. InfoNCE (pre-calib ≡ raw)")
    for m, (lab, col) in enumerate(zip(["TF0", "TF1", "TF2"], C_TF)):
        ax.plot(pre["optimizer_step_index"], pre[f"L_tf_norm_{m}"], "--", color=col, alpha=0.55, lw=1.2, label=f"norm. {TF_SHORT[m]} (pre≡raw)")
    # Post-calib
    ax.plot(post["optimizer_step_index"], post["L_contrast_norm"], "-", color=C_T, lw=1.6, label="norm. InfoNCE (post: raw/μ_c)")
    for m, col in enumerate(C_TF):
        ax.plot(post["optimizer_step_index"], post[f"L_tf_norm_{m}"], "-", color=col, lw=1.4, label=f"norm. {TF_SHORT[m]} (post: raw/μ)")
    ax.axvline(CALIB_OPT, color="#56B4E9", ls=":", lw=1.4, label="end calibration")
    ax.axvspan(0, CALIB_OPT, color="#56B4E9", alpha=0.06)
    ax.set_xlabel("Optimizer step index")
    ax.set_ylabel("Normalized loss")
    ax.set_title(
        "Normalized losses — pre/post calibration use different scaling regimes\n"
        "(epoch 1: identity; epoch ≥2: divide by frozen epoch-1 means μ)"
    )
    ax.legend(loc="upper right", frameon=False, fontsize=7.5, ncol=1)
    ax.set_xlim(0, N_OPT_PLANNED)
    _save(fig, "03_tfmoe_normalized_losses")
    add_series("03_tfmoe_normalized_losses", "L_contrast_norm", t["optimizer_step_index"], t["L_contrast_norm"], "optimizer_step_index", {"calibrated": None})
    for m in range(3):
        add_series("03_tfmoe_normalized_losses", f"L_tf_norm_{m}", t["optimizer_step_index"], t[f"L_tf_norm_{m}"], "optimizer_step_index")

    # ------------------------------------------------------------------ 04 stacked
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = t["optimizer_step_index"].to_numpy()
    stack = np.vstack(
        [
            t["contrib_contrast"].to_numpy(),
            t["contrib_tf_0"].to_numpy(),
            t["contrib_tf_1"].to_numpy(),
            t["contrib_tf_2"].to_numpy(),
        ]
    )
    labels_stack = [
        r"$\alpha\cdot$norm. InfoNCE",
        r"$w_{\mathrm{tf0}}\cdot$norm. " + TF_SHORT[0],
        r"$w_{\mathrm{tf1}}\cdot$norm. " + TF_SHORT[1],
        r"$w_{\mathrm{tf2}}\cdot$norm. " + TF_SHORT[2],
    ]
    colors_stack = [C_T, *C_TF]
    ax.stackplot(x, stack, labels=labels_stack, colors=colors_stack, alpha=0.75)
    ax.plot(x, t["L_total"], color=C_TOTAL, lw=1.8, label=r"logged $L_{\mathrm{total}}$")
    ax.axvline(CALIB_OPT, color="#56B4E9", ls=":", lw=1.3, label="end calibration")
    ax.set_xlabel("Optimizer step index")
    ax.set_ylabel("Objective contribution")
    ax.set_title(
        f"Weighted objective decomposition\n"
        f"recon |contrib_sum − L_total|: max={recon_max:.2e}, mean={recon_mean:.2e} (tol={RECON_TOL:g})"
    )
    ax.legend(loc="upper right", frameon=False, fontsize=7.5)
    ax.set_xlim(0, N_OPT_PLANNED)
    _save(fig, "04_tfmoe_weighted_objective_decomposition")
    for name, col in [
        ("contrib_contrast", t["contrib_contrast"]),
        ("contrib_tf_0", t["contrib_tf_0"]),
        ("contrib_tf_1", t["contrib_tf_1"]),
        ("contrib_tf_2", t["contrib_tf_2"]),
        ("L_total", t["L_total"]),
        ("contrib_sum", t["contrib_sum"]),
    ]:
        add_series("04_tfmoe_weighted_objective_decomposition", name, x, col, "optimizer_step_index")

    # ------------------------------------------------------------------ 05 post-calib only + inset raw
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(9.2, 4.2), gridspec_kw={"width_ratios": [2.2, 1.0]}
    )
    post = t[t["calibrated"]]
    ax.plot(post["optimizer_step_index"], post["L_total"], "-", color=C_T, lw=1.8, label=r"$L_{\mathrm{total}}$ (post-calib)")
    ax.axvline(CALIB_OPT, color="#56B4E9", ls=":", lw=1.2)
    ax.set_xlabel("Optimizer step index")
    ax.set_ylabel("Combined total objective")
    ax.set_title("Post-calibration total objective only")
    ax.legend(frameon=False)
    ax.set_xlim(CALIB_OPT, N_OPT_PLANNED)
    ax2.plot(t["optimizer_step_index"], t["L_contrast_raw"], color=C_T, lw=1.3)
    ax2.axvline(CALIB_OPT, color="#56B4E9", ls=":", lw=1.0)
    ax2.set_title("Raw InfoNCE (separate panel)")
    ax2.set_xlabel("Optimizer step")
    ax2.set_ylabel("Raw InfoNCE")
    ax2.text(0.5, -0.22, "not added into L_total on this scale", transform=ax2.transAxes, ha="center", fontsize=8, style="italic")
    fig.suptitle("Pre- and post-calibration L_total are not plotted as one continuous comparable curve", y=1.02, fontsize=10)
    _save(fig, "05_tfmoe_total_loss_post_calibration")
    add_series("05_tfmoe_total_loss_post_calibration", "L_total_post_calib", post["optimizer_step_index"], post["L_total"], "optimizer_step_index")
    add_series("05_tfmoe_total_loss_post_calibration", "raw_InfoNCE_inset", t["optimizer_step_index"], t["L_contrast_raw"], "optimizer_step_index")

    # ------------------------------------------------------------------ 06 weights
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.plot(t["optimizer_step_index"], t["w_contrast"], "-", color=C_T, lw=1.8, label="InfoNCE (w_contrast=α)")
    for m, (lab, col) in enumerate(zip(TF_SHORT, C_TF)):
        ax.plot(t["optimizer_step_index"], t[f"w_tf_{m}"], "-", color=col, lw=1.5, label=lab)
    ax.axvline(CALIB_OPT, color="#56B4E9", ls=":", lw=1.2, label="α/β unfrozen")
    ax.set_ylim(0, 0.7)
    ax.set_xlabel("Optimizer step index")
    ax.set_ylabel("Effective weight")
    ax.set_title(f"Learned effective weights (sum−1 max |err|={wsum_max:.2e})")
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.set_xlim(0, N_OPT_PLANNED)
    _save(fig, "06_learned_effective_weights")
    add_series("06_learned_effective_weights", "w_contrast", t["optimizer_step_index"], t["w_contrast"], "optimizer_step_index")
    for m in range(3):
        add_series("06_learned_effective_weights", f"w_tf_{m}", t["optimizer_step_index"], t[f"w_tf_{m}"], "optimizer_step_index")

    # ------------------------------------------------------------------ 07 train vs val MAE
    epochs = [1, 3, 5, 10]
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6), sharey=False)
    for m, (ax, key, lab, col) in enumerate(zip(axes, TF_KEYS, TF_LABELS, C_TF)):
        tr, va = [], []
        for ep in epochs:
            cell = analysis["cells"]["DIRECT_H_TFMOE"][str(ep)]["tfmoe"]
            tr.append(cell["train"]["mae"][key])
            va.append(cell["val"]["mae"][key])
        ax.plot(epochs, tr, "o-", color=col, lw=1.8, label="train")
        ax.plot(epochs, va, "s--", color=col, lw=1.8, label="validation")
        ax.set_title(lab, fontsize=10)
        ax.set_xlabel("SSL checkpoint epoch")
        ax.set_xticks(epochs)
        if m == 0:
            ax.set_ylabel("MAE (train-fit standardized scale)")
        ax.legend(frameon=False, fontsize=8)
        add_series("07_expert_train_vs_validation_mae", f"train_{key}", epochs, tr, "ssl_epoch")
        add_series("07_expert_train_vs_validation_mae", f"val_{key}", epochs, va, "ssl_epoch")
    fig.suptitle("TF expert MAE: train vs validation at frozen checkpoints", y=1.03)
    _save(fig, "07_expert_train_vs_validation_mae")

    # ------------------------------------------------------------------ 08 AUPRC
    refs = analysis["references"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=False)
    for ax, probe, title in [
        (axes[0], "primary", "A. Frozen R198+X+TF probe"),
        (axes[1], "diagnostic", "B. Frozen R198-only probe"),
    ]:
        for arm, color, name, marker in [
            ("DIRECT_H", C_H, NAME_H, "o"),
            ("DIRECT_H_TFMOE", C_T, NAME_T, "s"),
        ]:
            ys = [analysis["cells"][arm][str(ep)][probe]["validation_auprc"] for ep in epochs]
            ax.plot(epochs, ys, marker=marker, color=color, lw=1.8, label=name)
            add_series("08_downstream_validation_auprc", f"{arm}_{probe}_auprc", epochs, ys, "ssl_epoch")
        if probe == "primary":
            ax.axhline(
                refs["supervised_multigin"]["validation_auprc"],
                color="#000000",
                ls=":",
                lw=1.2,
                label="Supervised Multi-GIN+EU, validation AUPRC, seed 2",
            )
        ax.set_title(title)
        ax.set_xlabel("SSL checkpoint epoch")
        ax.set_ylabel("Validation AUPRC")
        ax.set_xticks(epochs)
        ax.legend(frameon=False, fontsize=7.5)
    fig.suptitle(
        "Downstream validation AUPRC (locked val-only)",
        y=1.02,
        fontsize=11,
    )
    _save(fig, "08_downstream_validation_auprc")

    # ------------------------------------------------------------------ 09 F1
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    # A fixed 0.5
    ax = axes[0]
    for arm, color, name, marker in [
        ("DIRECT_H", C_H, NAME_H, "o"),
        ("DIRECT_H_TFMOE", C_T, NAME_T, "s"),
    ]:
        ys = [
            analysis["cells"][arm][str(ep)]["primary"]["validation_metrics_at_0.5"]["f1"]
            for ep in epochs
        ]
        ax.plot(epochs, ys, marker=marker, color=color, lw=1.8, label=name)
        add_series("09_downstream_validation_f1", f"{arm}_f1_at_0.5", epochs, ys, "ssl_epoch")
    ax.axhline(
        refs["supervised_multigin"]["validation_f1"],
        color="#000000",
        ls=":",
        lw=1.2,
        label="Supervised Multi-GIN+EU, validation F1 (argmax), seed 2",
    )
    ax.set_title("A. Fixed-threshold F1@0.5")
    ax.set_xlabel("SSL checkpoint epoch")
    ax.set_ylabel("Validation F1")
    ax.set_xticks(epochs)
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1]
    for arm, color, name, marker in [
        ("DIRECT_H", C_H, NAME_H, "o"),
        ("DIRECT_H_TFMOE", C_T, NAME_T, "s"),
    ]:
        ys = [
            analysis["cells"][arm][str(ep)]["primary"]["validation_metrics_at_val_optimal_f1"]["f1"]
            for ep in epochs
        ]
        ax.plot(epochs, ys, marker=marker, color=color, lw=1.8, label=name)
        add_series("09_downstream_validation_f1", f"{arm}_f1_at_val_opt", epochs, ys, "ssl_epoch")
    ax.set_title("B. F1@validation-optimized threshold\n(optimistic diagnostic)")
    ax.set_xlabel("SSL checkpoint epoch")
    ax.set_ylabel("Validation F1")
    ax.set_xticks(epochs)
    ax.legend(frameon=False, fontsize=7)
    fig.suptitle(
        "Downstream validation F1 on frozen R198+X+TF probe",
        y=1.05,
        fontsize=11,
    )
    _save(fig, "09_downstream_validation_f1")

    # ------------------------------------------------------------------ 10 geometry
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    for arm, color, name, marker in [
        ("DIRECT_H", C_H, NAME_H, "o"),
        ("DIRECT_H_TFMOE", C_T, NAME_T, "s"),
    ]:
        norms = [analysis["cells"][arm][str(ep)]["repr_val"]["mean_l2_norm"] for ep in epochs]
        ranks = [analysis["cells"][arm][str(ep)]["repr_val"]["effective_rank"] for ep in epochs]
        axes[0].plot(epochs, norms, marker=marker, color=color, lw=1.8, label=name)
        axes[1].plot(epochs, ranks, marker=marker, color=color, lw=1.8, label=name)
        add_series("10_representation_geometry", f"{arm}_mean_l2_norm", epochs, norms, "ssl_epoch")
        add_series("10_representation_geometry", f"{arm}_effective_rank", epochs, ranks, "ssl_epoch")
    axes[0].set_title("Mean R198 L2 norm (validation seeds)")
    axes[0].set_ylabel("Mean ‖r‖₂")
    axes[1].set_title("R198 effective rank (participation-ratio style)")
    axes[1].set_ylabel("Effective rank (not literal matrix rank)")
    for ax in axes:
        ax.set_xlabel("SSL checkpoint epoch")
        ax.set_xticks(epochs)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Stable norm is desirable; higher effective rank means variation uses more independent directions",
        y=1.02,
        fontsize=10,
    )
    _save(fig, "10_representation_geometry")

    # ------------------------------------------------------------------ 11 LR
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.plot(hm["optimizer_step_index"], hm["encoder_lr"], "-", color=C_H, lw=2.0, label=f"{NAME_H} encoder LR")
    ax.plot(tm["optimizer_step_index"], tm["encoder_lr"], "--", color=C_T, lw=1.4, alpha=0.9, label=f"{NAME_T} encoder LR")
    ax.axvline(WARMUP_END, color="#56B4E9", ls=":", lw=1.3, label="end warmup / end calib")
    ax.axvspan(0, WARMUP_END, color="#56B4E9", alpha=0.08, label="warmup phase")
    ax.axvspan(WARMUP_END, N_OPT_PLANNED, color="#F0E442", alpha=0.08, label="cosine phase")
    for e in range(1, 11):
        ax.axvline(e * STEPS_PER_EPOCH, color="#EEEEEE", lw=0.8, zorder=0)
    ax.set_xlabel("Optimizer step index")
    ax.set_ylabel("Encoder learning rate")
    ax.set_title(f"Encoder LR schedule (arms identical on shared steps; max |ΔLR|={lr_max_diff:.2e})")
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.set_xlim(0, N_OPT_PLANNED)
    _save(fig, "11_learning_rate_schedule")
    add_series("11_learning_rate_schedule", "DIRECT_H_encoder_lr", hm["optimizer_step_index"], hm["encoder_lr"], "optimizer_step_index")
    add_series("11_learning_rate_schedule", "TFMOE_encoder_lr", tm["optimizer_step_index"], tm["encoder_lr"], "optimizer_step_index")

    # ------------------------------------------------------------------ 12 grads
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 6.2), sharex=True)
    # encoder both arms
    ax = axes[0]
    for df, color, name in [(h, C_H, NAME_H), (t, C_T, NAME_T)]:
        x = df["optimizer_step_index"].to_numpy()
        y = df["encoder_grad_norm"].to_numpy()
        ax.plot(x, y, color=color, alpha=0.25, lw=0.9)
        ax.plot(x, _rolling_median(y), color=color, lw=2.0, label=f"{name} (rolling median)")
        add_series("12_gradient_norms", f"{name}_encoder_grad_raw", x, y, "optimizer_step_index")
        add_series("12_gradient_norms", f"{name}_encoder_grad_rollmed", x, _rolling_median(y), "optimizer_step_index")
    ax.axvline(CALIB_OPT, color="#56B4E9", ls=":", lw=1.3, label="calib / αβ unfreeze")
    ax.set_ylabel("Encoder grad norm")
    ax.set_title("Encoder ‖grad‖₂ (pre-clip; no clipping in this path; may be mid-accumulation)")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    x = t["optimizer_step_index"].to_numpy()
    y = t["alpha_grad_norm"].to_numpy()
    ax.plot(x, y, color=C_T, alpha=0.25, lw=0.9, label="α-logit grad (raw)")
    ax.plot(x, _rolling_median(y), color=C_T, lw=2.0, label="α-logit grad (rolling median)")
    ax.axvline(CALIB_OPT, color="#56B4E9", ls=":", lw=1.3, label="α/β unfrozen")
    ax.set_xlabel("Optimizer step index")
    ax.set_ylabel("α-logit grad norm")
    ax.set_title(
        "α-logit grad only — β-logit grads not logged; do not compare magnitudes to encoder "
        "(different parameter counts)"
    )
    ax.legend(frameon=False, fontsize=8)
    ax.set_xlim(0, N_OPT_PLANNED)
    add_series("12_gradient_norms", "alpha_logit_grad_raw", x, y, "optimizer_step_index")
    add_series("12_gradient_norms", "alpha_logit_grad_rollmed", x, _rolling_median(y), "optimizer_step_index")
    add_series(
        "12_gradient_norms",
        "beta_logits_grad_norm",
        [np.nan],
        [np.nan],
        "unavailable",
        {"note": "not logged"},
    )
    _save(fig, "12_gradient_norms")

    # ------------------ figure_data.csv
    pd.DataFrame(series_rows).to_csv(OUT / "figure_data.csv", index=False)

    # ------------------ loss_checkpoint_table.csv
    ckpt_rows = []
    for ep in epochs:
        # last TFMOE step log in that epoch
        sub = t[t["epoch"] == ep]
        if sub.empty:
            raise SystemExit(f"No TFMOE step logs for epoch {ep}")
        r = sub.iloc[-1]
        epj_h = json.loads((H_LOG / f"epoch_{ep:02d}.json").read_text())
        epj_t = json.loads((T_LOG / f"epoch_{ep:02d}.json").read_text())
        # DIRECT_H row (contrast only)
        ckpt_rows.append(
            {
                "arm": "DIRECT_H",
                "epoch": ep,
                "raw_InfoNCE_epoch_mean": epj_h["loss/contrastive"],
                "raw_InfoNCE_last_logged_step": float(
                    h[h["epoch"] == ep].iloc[-1]["L_contrast_raw"]
                ),
                "normalized_InfoNCE": None,
                "combined_total_epoch_mean": epj_h["loss/train"],
                "combined_total_last_logged_step": float(h[h["epoch"] == ep].iloc[-1]["L_total"]),
                "L_tf_raw_0": None,
                "L_tf_raw_1": None,
                "L_tf_raw_2": None,
                "L_tf_norm_0": None,
                "L_tf_norm_1": None,
                "L_tf_norm_2": None,
                "w_contrast": 1.0,
                "w_tf_0": 0.0,
                "w_tf_1": 0.0,
                "w_tf_2": 0.0,
                "contrib_contrast": float(h[h["epoch"] == ep].iloc[-1]["L_total"]),
                "contrib_tf_0": None,
                "contrib_tf_1": None,
                "contrib_tf_2": None,
                "optimizer_step_index_last_log": int(h[h["epoch"] == ep].iloc[-1]["optimizer_step_index"]),
            }
        )
        ckpt_rows.append(
            {
                "arm": "DIRECT_H_TFMOE",
                "epoch": ep,
                "raw_InfoNCE_epoch_mean": epj_t["loss/contrastive"],
                "raw_InfoNCE_last_logged_step": float(r["L_contrast_raw"]),
                "normalized_InfoNCE": float(r["L_contrast_norm"]),
                "combined_total_epoch_mean": epj_t["loss/train"],
                "combined_total_last_logged_step": float(r["L_total"]),
                "L_tf_raw_0": float(r["L_tf_raw_0"]),
                "L_tf_raw_1": float(r["L_tf_raw_1"]),
                "L_tf_raw_2": float(r["L_tf_raw_2"]),
                "L_tf_norm_0": float(r["L_tf_norm_0"]),
                "L_tf_norm_1": float(r["L_tf_norm_1"]),
                "L_tf_norm_2": float(r["L_tf_norm_2"]),
                "w_contrast": float(r["w_contrast"]),
                "w_tf_0": float(r["w_tf_0"]),
                "w_tf_1": float(r["w_tf_1"]),
                "w_tf_2": float(r["w_tf_2"]),
                "contrib_contrast": float(r["contrib_contrast"]),
                "contrib_tf_0": float(r["contrib_tf_0"]),
                "contrib_tf_1": float(r["contrib_tf_1"]),
                "contrib_tf_2": float(r["contrib_tf_2"]),
                "optimizer_step_index_last_log": int(r["optimizer_step_index"]),
            }
        )
    pd.DataFrame(ckpt_rows).to_csv(OUT / "loss_checkpoint_table.csv", index=False)

    # ------------------ plot_integrity.json
    integrity = {
        "L_total_reconstruction": {
            "formula": "weighted_contrast + sum_m (w_tf_m * L_tf_norm_m)",
            "n_points": int(len(t)),
            "max_abs_err": recon_max,
            "mean_abs_err": recon_mean,
            "tolerance": RECON_TOL,
            "passed": recon_max <= RECON_TOL,
            "note": "Logged weighted_tf_m is beta*norm without (1-alpha); contributions use w_tf*norm.",
        },
        "effective_weights_sum_to_one": {
            "max_abs_err": wsum_max,
            "tolerance": RECON_TOL,
            "passed": wsum_max <= RECON_TOL,
        },
        "encoder_lr_arms_identical": {
            "n_common_optimizer_steps": len(common),
            "max_abs_diff": lr_max_diff,
            "passed": lr_max_diff == 0.0,
        },
        "calibration_boundary_optimizer_step_index": CALIB_OPT,
        "warmup_end_optimizer_step_index": WARMUP_END,
        "gradient_norms": {
            "clipping_applied": False,
            "measured": "after backward, before optimizer.step; pre-clip (no clip in path)",
            "alpha_grad_scope": "alpha_logit only; beta_logits not logged",
            "may_be_mid_accumulation": True,
            "accum_steps_recipe": 4,
        },
        "missing_fields": [
            "beta_logits gradient norms",
            "per-step validation TF MAE (only checkpoint epochs 1/3/5/10 in analysis JSON)",
            "dense per-optimizer-step logs (sparse: first 8 microbatches + every 50)",
        ],
        "no_jobs_submitted": True,
        "no_test_data_accessed": True,
        "outputs_dir": str(OUT),
        "did_not_overwrite_canonical_figures": True,
    }
    (OUT / "plot_integrity.json").write_text(json.dumps(integrity, indent=2) + "\n")

    # ------------------ figure_manifest.md
    manifest = f"""# Figure package v2 — scheduled DIRECT_H / DIRECT_H_TFMOE

Offline-only reformatting and reconstruction from existing step logs, epoch JSON, and
`direct_h_tfmoe_scheduled_val_analysis.json`. Canonical `figures/` untouched.
Arm colors: blue = {NAME_H}; vermillion = {NAME_T}.

## 01_raw_infonce_comparison
Raw InfoNCE (`L_contrast_raw`) for both arms versus optimizer-step index. Auxiliary TF MAE
is excluded by construction (logged before MoE combination). Annotates lower-is-better.
TFMOE’s higher raw InfoNCE is expected under a multi-task, down-weighted contrast term.

## 02_tfmoe_raw_expert_losses
Raw MAE for the three TF targets on their train-standardized scales. Plotted alone so
InfoNCE (~7) is not visually compared to MAE (~0.1–1).

## 03_tfmoe_normalized_losses
Normalized InfoNCE and TF losses. Epoch 1 uses identity scaling (norm ≡ raw); after the
calibration boundary (optimizer step 100) each term is divided by its frozen epoch-1 mean.
Pre/post values therefore live in different scaling regimes (dashed vs solid).

## 04_tfmoe_weighted_objective_decomposition
Stacked contributions that enter the optimizer:
`α·L_contrast_norm` and `w_tf_m·L_tf_norm_m`, with logged `L_total` overlaid.
Reconstruction max abs error = {recon_max:.3e} (tolerance {RECON_TOL:g}); plot fails if exceeded.

## 05_tfmoe_total_loss_post_calibration
Post-calibration `L_total` only (optimizer step ≥ 100), avoiding a false continuous curve
across the calibration discontinuity. Raw InfoNCE is shown only in a separate panel.

## 06_learned_effective_weights
The four effective weights `w_contrast=α` and `w_tf_0..2=(1−α)β_m`. Duplicate α/w_contrast
lines removed. Weights sum to 1 at every logged point (max |err|={wsum_max:.3e}).

## 07_expert_train_vs_validation_mae
Per-target train (solid) vs validation (dashed) MAE at frozen checkpoints 1/3/5/10 from
the locked val analysis (full non-truncated target names).

## 08_downstream_validation_auprc
Two panels: (A) R198+X+TF probe AUPRC; (B) R198-only diagnostic AUPRC.
Supervised Multi-GIN+EU seed-2 validation AUPRC retained on panel A. Projected-encoder
baselines omitted.

## 09_downstream_validation_f1
Panel A: fixed-threshold F1@0.5 with Supervised Multi-GIN+EU seed-2 validation F1 (argmax).
Panel B: F1@validation-optimized threshold (optimistic diagnostic). No projected-encoder baseline.

## 10_representation_geometry
Validation-seed mean R198 L2 norm and participation-ratio-style effective rank (not literal
matrix rank). Stable norms are desirable; higher effective rank means more independent directions.

## 11_learning_rate_schedule
Actual encoder LR only for both arms (identical on shared optimizer steps; max |Δ|={lr_max_diff:.3e}).
Marks warmup end, calibration boundary, cosine phase, and epoch boundaries. No LR-factor overlay.

## 12_gradient_norms
Encoder and α-logit gradient norms: faint raw traces, prominent rolling medians, separate
panels. Measured post-backward / pre-step with **no grad clipping** in this training path;
may reflect mid-accumulation grads. β-logit grads were not logged.

## Supporting artifacts
- `data_definition.json` — exact field and objective definitions from code
- `figure_data.csv` — every plotted numeric series
- `loss_checkpoint_table.csv` — epochs 1/3/5/10 loss/weight snapshot
- `plot_integrity.json` — reconstruction and sum-to-one checks
"""
    (OUT / "figure_manifest.md").write_text(manifest)

    print(json.dumps({
        "out": str(OUT),
        "recon_max": recon_max,
        "recon_mean": recon_mean,
        "wsum_max": wsum_max,
        "lr_max_diff": lr_max_diff,
        "n_png": len(list(OUT.glob("*.png"))),
        "n_pdf": len(list(OUT.glob("*.pdf"))),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
