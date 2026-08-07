#!/usr/bin/env python3
"""Presentation_v3 figures from figures_v2 data only (no retrain / no log re-read).

Writes ONLY under:
  results/diagnostics/direct_h_tfmoe_scheduled_val_analysis/presentation_v3/

Does not overwrite figures_v2/ or canonical figures/.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "results/diagnostics/direct_h_tfmoe_scheduled_val_analysis/figures_v2"
OUT = ROOT / "results/diagnostics/direct_h_tfmoe_scheduled_val_analysis/presentation_v3"

C_H = "#0072B2"
C_T = "#D55E00"
C_TF = ["#009E73", "#CC79A7", "#E69F00"]
C_TOTAL = "#000000"
C_REF = "#666666"
C_CAL = "#56B4E9"

NAME_H = "InfoNCE"
NAME_T = "InfoNCE + TF"
TF_SHORT = (
    "sender interarrival",
    "sender recent 7-day count",
    "amount vs sender history",
)
TF_KEYS = (
    "log1p_sender_interarrival",
    "log1p_sender_past_7d_count",
    "log1p_amount_vs_sender_past_mean",
)
TF_FULL = (
    "sender interarrival (log1p)",
    "sender recent 7-day count (log1p)",
    "amount relative to sender history (log1p)",
)

CALIB = 100.0
N_OPT = 1000.0
EPOCHS = [1, 3, 5, 10]

mpl.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "figure.dpi": 140,
        "savefig.dpi": 200,
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def series(df: pd.DataFrame, figure: str, name: str) -> Tuple[np.ndarray, np.ndarray]:
    sub = df[(df["figure"] == figure) & (df["series"] == name)].sort_values("x")
    if sub.empty:
        raise KeyError(f"Missing series {figure}/{name} in v2 figure_data.csv")
    return sub["x"].to_numpy(dtype=float), sub["y"].to_numpy(dtype=float)


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def legend_outside(ax, **kwargs):
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, **kwargs)


def post_mask(x: np.ndarray) -> np.ndarray:
    return x >= CALIB


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(V2 / "figure_data.csv")
    integrity = json.loads((V2 / "plot_integrity.json").read_text())

    # -------- 01 raw InfoNCE
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    x, y = series(df, "01_raw_infonce_comparison", "DIRECT_H_raw_InfoNCE")
    ax.plot(x, y, "-", color=C_H, lw=1.6, label=NAME_H)
    x, y = series(df, "01_raw_infonce_comparison", "TFMOE_raw_InfoNCE")
    ax.plot(x, y, "-", color=C_T, lw=1.6, label=NAME_T)
    ax.axvline(CALIB, color=C_CAL, ls=":", lw=1.1)
    ax.set_xlim(0, N_OPT)
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Raw InfoNCE")
    ax.set_title("Raw InfoNCE")
    ax.text(0.98, 0.05, "lower is better", transform=ax.transAxes, ha="right", va="bottom", fontsize=8, style="italic", color="#555555")
    legend_outside(ax)
    fig.tight_layout()
    save(fig, "01_raw_infonce")

    # -------- 02 raw TF MAE (unchanged content, shorter title)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for m, (lab, col) in enumerate(zip(TF_SHORT, C_TF)):
        # v2 series names use full short labels with spaces
        v2_lab = (
            "sender interarrival",
            "sender recent 7-day count",
            "amount relative to sender history",
        )[m]
        x, y = series(df, "02_tfmoe_raw_expert_losses", f"raw_MAE_{v2_lab}")
        ax.plot(x, y, "-", color=col, lw=1.5, label=lab)
    ax.axvline(CALIB, color=C_CAL, ls=":", lw=1.1)
    ax.set_xlim(0, N_OPT)
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Raw MAE")
    ax.set_title("TF expert raw MAE")
    legend_outside(ax)
    fig.tight_layout()
    save(fig, "02_tf_raw_mae")

    # -------- 03 post-calib normalized only
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    x, y = series(df, "03_tfmoe_normalized_losses", "L_contrast_norm")
    m = post_mask(x)
    ax.plot(x[m], y[m], "-", color=C_T, lw=1.7, label="InfoNCE")
    for i, (lab, col) in enumerate(zip(TF_SHORT, C_TF)):
        x, y = series(df, "03_tfmoe_normalized_losses", f"L_tf_norm_{i}")
        ax.plot(x[m], y[m], "-", color=col, lw=1.5, label=lab)
    ax.set_xlim(CALIB, N_OPT)
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Normalized loss")
    ax.set_title("Normalized losses (post-calibration)")
    legend_outside(ax)
    fig.tight_layout()
    save(fig, "03_normalized_losses_post_calib")

    # -------- 04 post-calib stacked, y 0–1.2
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    x = series(df, "04_tfmoe_weighted_objective_decomposition", "contrib_contrast")[0]
    m = post_mask(x)
    stack_names = ["contrib_contrast", "contrib_tf_0", "contrib_tf_1", "contrib_tf_2"]
    stack_labs = ["InfoNCE", *TF_SHORT]
    stack_cols = [C_T, *C_TF]
    ys = []
    for name in stack_names:
        _, y = series(df, "04_tfmoe_weighted_objective_decomposition", name)
        ys.append(y[m])
    xp = x[m]
    ax.stackplot(xp, ys, labels=stack_labs, colors=stack_cols, alpha=0.8)
    xt, yt = series(df, "04_tfmoe_weighted_objective_decomposition", "L_total")
    ax.plot(xt[m], yt[m], color=C_TOTAL, lw=1.6, label=r"$L_{\mathrm{total}}$")
    ax.set_ylim(0, 1.2)
    ax.set_xlim(CALIB, N_OPT)
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Contribution")
    ax.set_title("Weighted objective (post-calibration)")
    legend_outside(ax)
    fig.tight_layout()
    save(fig, "04_weighted_objective_post_calib")

    # -------- S: full-range reconstruction integrity (supplemental)
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    x = series(df, "04_tfmoe_weighted_objective_decomposition", "contrib_contrast")[0]
    ys = []
    for name in stack_names:
        _, y = series(df, "04_tfmoe_weighted_objective_decomposition", name)
        ys.append(y)
    ax.stackplot(x, ys, labels=stack_labs, colors=stack_cols, alpha=0.75)
    xt, yt = series(df, "04_tfmoe_weighted_objective_decomposition", "L_total")
    ax.plot(xt, yt, color=C_TOTAL, lw=1.5, label=r"$L_{\mathrm{total}}$")
    ax.axvline(CALIB, color=C_CAL, ls=":", lw=1.1)
    ax.set_xlim(0, N_OPT)
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Contribution")
    ax.set_title("Supplemental: full-range weighted reconstruction")
    legend_outside(ax)
    fig.tight_layout()
    save(fig, "S01_weighted_reconstruction_full_range")

    # -------- 06 weights
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    x, y = series(df, "06_learned_effective_weights", "w_contrast")
    ax.plot(x, y, "-", color=C_T, lw=1.7, label="InfoNCE")
    for i, (lab, col) in enumerate(zip(TF_SHORT, C_TF)):
        x, y = series(df, "06_learned_effective_weights", f"w_tf_{i}")
        ax.plot(x, y, "-", color=col, lw=1.5, label=lab)
    ax.axvline(CALIB, color=C_CAL, ls=":", lw=1.1)
    ax.set_ylim(0, 0.7)
    ax.set_xlim(0, N_OPT)
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Effective weight")
    ax.set_title("Learned effective weights")
    legend_outside(ax)
    fig.tight_layout()
    save(fig, "06_effective_weights")

    # -------- 07 train vs val MAE
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    for i, (ax, key, lab, col) in enumerate(zip(axes, TF_KEYS, TF_FULL, C_TF)):
        x, y = series(df, "07_expert_train_vs_validation_mae", f"train_{key}")
        ax.plot(x, y, "o-", color=col, lw=1.6, label="train")
        x, y = series(df, "07_expert_train_vs_validation_mae", f"val_{key}")
        ax.plot(x, y, "s--", color=col, lw=1.6, label="validation")
        ax.set_title(lab, fontsize=9)
        ax.set_xlabel("SSL epoch")
        ax.set_xticks(EPOCHS)
        if i == 0:
            ax.set_ylabel("MAE")
        ax.legend(loc="best", frameon=False, fontsize=8)
    fig.suptitle("TF expert train vs validation MAE", y=1.02, fontsize=12)
    fig.tight_layout()
    save(fig, "07_tf_train_val_mae")

    # -------- 08 AUPRC
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    for ax, probe, title in [
        (axes[0], "primary", "R198+X+TF"),
        (axes[1], "diagnostic", "R198 only"),
    ]:
        for arm, color, name, marker in [
            ("DIRECT_H", C_H, NAME_H, "o"),
            ("DIRECT_H_TFMOE", C_T, NAME_T, "s"),
        ]:
            x, y = series(df, "08_downstream_validation_auprc", f"{arm}_{probe}_auprc")
            ax.plot(x, y, marker=marker, color=color, lw=1.6, label=name)
        if probe == "primary":
            # Supervised AUPRC only; projected-encoder refs omitted (ambiguous provenance).
            ax.axhline(
                0.5509,
                color="#000000",
                ls=":",
                lw=1.1,
                label="Supervised Multi-GIN+EU, val AUPRC, seed 2",
            )
        ax.set_title(title)
        ax.set_xlabel("SSL epoch")
        ax.set_ylabel("Val AUPRC")
        ax.set_xticks(EPOCHS)
        legend_outside(ax, fontsize=8)
    fig.suptitle("Downstream validation AUPRC", y=1.02, fontsize=12)
    fig.tight_layout()
    save(fig, "08_val_auprc")

    # -------- 09a F1@0.5
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    for arm, color, name, marker in [
        ("DIRECT_H", C_H, NAME_H, "o"),
        ("DIRECT_H_TFMOE", C_T, NAME_T, "s"),
    ]:
        x, y = series(df, "09_downstream_validation_f1", f"{arm}_f1_at_0.5")
        ax.plot(x, y, marker=marker, color=color, lw=1.6, label=name)
    ax.axhline(
        0.6101,
        color="#000000",
        ls=":",
        lw=1.1,
        label="Supervised Multi-GIN+EU, val F1 (argmax), seed 2",
    )
    ax.set_xlabel("SSL epoch")
    ax.set_ylabel("Val F1")
    ax.set_title("Validation F1@0.5")
    ax.set_xticks(EPOCHS)
    legend_outside(ax)
    fig.tight_layout()
    save(fig, "09a_val_f1_fixed_threshold")

    # -------- 09b F1@validation-optimized threshold (no projected; no supervised argmax)
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    for arm, color, name, marker in [
        ("DIRECT_H", C_H, NAME_H, "o"),
        ("DIRECT_H_TFMOE", C_T, NAME_T, "s"),
    ]:
        x, y = series(df, "09_downstream_validation_f1", f"{arm}_f1_at_val_opt")
        ax.plot(x, y, marker=marker, color=color, lw=1.6, label=name)
    ax.set_xlabel("SSL epoch")
    ax.set_ylabel("Val F1")
    ax.set_title("Validation F1@validation-optimized threshold")
    ax.set_xticks(EPOCHS)
    legend_outside(ax)
    fig.tight_layout()
    save(fig, "09b_val_f1_val_selected_threshold")

    # -------- 10 geometry (supplementary)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    for arm, color, name, marker in [
        ("DIRECT_H", C_H, NAME_H, "o"),
        ("DIRECT_H_TFMOE", C_T, NAME_T, "s"),
    ]:
        x, y = series(df, "10_representation_geometry", f"{arm}_mean_l2_norm")
        axes[0].plot(x, y, marker=marker, color=color, lw=1.6, label=name)
        x, y = series(df, "10_representation_geometry", f"{arm}_effective_rank")
        axes[1].plot(x, y, marker=marker, color=color, lw=1.6, label=name)
    axes[0].set_title("Mean R198 L2 norm")
    axes[0].set_ylabel(r"Mean $\|r\|_2$")
    axes[1].set_title("R198 effective rank")
    axes[1].set_ylabel("Effective rank")
    for ax in axes:
        ax.set_xlabel("SSL epoch")
        ax.set_xticks(EPOCHS)
        legend_outside(ax, fontsize=8)
    fig.suptitle("Representation geometry (supplementary)", y=1.02, fontsize=12)
    fig.tight_layout()
    save(fig, "10_representation_geometry")

    # -------- 11 one LR curve
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x, y = series(df, "11_learning_rate_schedule", "DIRECT_H_encoder_lr")
    ax.plot(x, y, "-", color=C_H, lw=1.8, label="Encoder LR")
    ax.axvline(CALIB, color=C_CAL, ls=":", lw=1.1)
    ax.axvspan(0, CALIB, color=C_CAL, alpha=0.07)
    ax.axvspan(CALIB, N_OPT, color="#F0E442", alpha=0.07)
    ax.set_xlim(0, N_OPT)
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Learning rate")
    ax.set_title("Encoder learning-rate schedule")
    legend_outside(ax)
    fig.tight_layout()
    save(fig, "11_encoder_lr")

    # -------- 12a encoder grads (supplementary)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for arm_key, color, name in [
        ("Direct-R198 InfoNCE", C_H, NAME_H),
        ("Direct-R198 InfoNCE + TF experts", C_T, NAME_T),
    ]:
        x, y = series(df, "12_gradient_norms", f"{arm_key}_encoder_grad_raw")
        ax.plot(x, y, color=color, alpha=0.22, lw=0.8)
        x, y = series(df, "12_gradient_norms", f"{arm_key}_encoder_grad_rollmed")
        ax.plot(x, y, color=color, lw=1.8, label=name)
    ax.axvline(CALIB, color=C_CAL, ls=":", lw=1.1)
    ax.set_xlim(0, N_OPT)
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel(r"Encoder $\|\mathrm{grad}\|_2$")
    ax.set_title("Encoder gradient norms (supplementary)")
    legend_outside(ax)
    fig.tight_layout()
    save(fig, "12a_encoder_grad_norms")

    # -------- 12b alpha grads (supplementary)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x, y = series(df, "12_gradient_norms", "alpha_logit_grad_raw")
    ax.plot(x, y, color=C_T, alpha=0.22, lw=0.8)
    x, y = series(df, "12_gradient_norms", "alpha_logit_grad_rollmed")
    ax.plot(x, y, color=C_T, lw=1.8, label=r"$\alpha$-logit")
    ax.axvline(CALIB, color=C_CAL, ls=":", lw=1.1)
    ax.set_xlim(0, N_OPT)
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel(r"$\alpha$-logit $\|\mathrm{grad}\|_2$")
    ax.set_title(r"$\alpha$-logit gradient norms (supplementary)")
    legend_outside(ax)
    fig.tight_layout()
    save(fig, "12b_alpha_grad_norms")

    # -------- captions.md
    recon = integrity["L_total_reconstruction"]
    wsum = integrity["effective_weights_sum_to_one"]
    lr = integrity["encoder_lr_arms_identical"]
    g = integrity["gradient_norms"]
    captions = f"""# Presentation v3 captions

Compact figure set redrawn from `figures_v2/figure_data.csv` and
`figures_v2/plot_integrity.json` only. No retraining, no Slurm, no test access.
Canonical `figures/` and `figures_v2/` were not overwritten.

Colors: blue = InfoNCE; vermillion = InfoNCE + TF; green/magenta/gold = TF targets.

---

## 01_raw_infonce
Raw InfoNCE component for both arms vs optimizer step (aux TF MAE excluded).
Vertical dotted line: end of epoch-1 calibration / warmup (step 100). Lower is better.
TF arm stays higher because contrast is down-weighted in a multi-task objective.

## 02_tf_raw_mae
Raw MAE for the three TF targets on train-standardized scales (not InfoNCE-scale).

## 03_normalized_losses_post_calib
Normalized InfoNCE and TF losses **after calibration only** (step ≥ 100).
During epoch 1, “normalized” losses are identity(raw). At step 100, frozen epoch-1 means
μ begin dividing each raw term (`L/μ`), so pre/post values are different scaling regimes
and are not shown together here.

## 04_weighted_objective_post_calib
Post-calibration stacked contributions entering the optimizer:
`α · L_contrast_norm` and `w_tf_m · L_tf_norm_m`, with logged `L_total` overlaid.
Y-axis focused on 0–1.2. Full-run reconstruction (including the calibration cliff) is in
`S01_weighted_reconstruction_full_range`.

## S01_weighted_reconstruction_full_range *(supplemental integrity)*
Full optimizer-step range stacked contributions vs logged `L_total`.
Reconstruction check from v2: max |contrib_sum − L_total| = {recon['max_abs_err']:.3e},
mean = {recon['mean_abs_err']:.3e} (tol {recon['tolerance']:g}); passed={recon['passed']}.
Note: logged `weighted_tf_m` stores `β·norm` without `(1−α)`; stacks use `w_tf·norm`.

## 06_effective_weights
Effective weights `w_contrast=α` and `w_tf_m=(1−α)β_m`. Sum-to-one check (v2):
max |Σw − 1| = {wsum['max_abs_err']:.3e} (tol {wsum['tolerance']:g}); passed={wsum['passed']}.
Dotted line: α/β unfrozen after calibration.

## 07_tf_train_val_mae
Per-target train (solid) vs validation (dashed) MAE at frozen SSL checkpoints 1/3/5/10.

## 08_val_auprc
Locked validation AUPRC for frozen probes. Left: R198+X+TF; right: R198-only.
Horizontal reference (primary panel only): Supervised Multi-GIN+EU, validation AUPRC,
seed 2 (`small_hi_legacy_supervised_gin_emlps_ports_50ep_seed2`, best epoch 43).
Projected-encoder baselines are omitted.

## 09a_val_f1_fixed_threshold
Validation F1@0.5 on the R198+X+TF probe.
Supervised reference: Multi-GIN+EU seed-2 validation minority F1 via **argmax** over
two-class logits at the best-validation checkpoint (epoch 43; value 0.6101). This is a
fixed decision rule, not a validation-optimized threshold for the probe.

## 09b_val_f1_val_selected_threshold
Validation F1@validation-optimized threshold on the R198+X+TF probe (optimistic diagnostic;
threshold fit on validation). No projected-encoder baseline. Supervised argmax F1 is not
plotted here (different decision rule).

## 10_representation_geometry *(supplementary)*
Validation-seed mean R198 L2 norm and participation-ratio-style effective rank
(not literal matrix rank). Stable norms are desirable; higher effective rank means
variation uses more independent directions.

## 11_encoder_lr
Encoder learning-rate schedule (warmup then cosine). A single curve is shown because
both arms are identical on shared optimizer steps (v2 max |ΔLR| = {lr['max_abs_diff']}).
Shaded regions: warmup/calibration (step < 100) and cosine (step ≥ 100).

## 12a_encoder_grad_norms *(supplementary)*
Encoder gradient L2 norms: faint raw traces, bold rolling medians.
Measured post-backward / pre-`optimizer.step`; **no grad clipping** in this path
(may be mid-accumulation; accum_steps={g['accum_steps_recipe']}).
Dotted line: calibration / αβ unfreeze.

## 12b_alpha_grad_norms *(supplementary)*
α-logit gradient norms only (β-logit grads were not logged). Do not compare absolute
magnitudes to the encoder panel (different parameter counts). Zero before unfreeze.

---

## Protocol notes
- Microbatch logs are sparse (first 8 + every 50); x-axis uses optimizer-step index.
- No jobs submitted; no test data accessed for this redraw.
"""
    (OUT / "captions.md").write_text(captions)

    # light manifest
    files = sorted(p.name for p in OUT.glob("*"))
    (OUT / "README.md").write_text(
        "# presentation_v3\n\nCompact redraw from figures_v2 data. See `captions.md`.\n\n"
        + "\n".join(f"- `{f}`" for f in files)
        + "\n"
    )

    print(
        json.dumps(
            {
                "out": str(OUT),
                "n_png": len(list(OUT.glob("*.png"))),
                "n_pdf": len(list(OUT.glob("*.pdf"))),
                "files": files,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
