#!/usr/bin/env python3
"""Collaborator-facing deliverables for corrected 40ep linear-LR matched trajectories.

Uses only full-subgraph re-eval cells under
results/diagnostics/direct_r198_40ep_linear_lr_full_extract_reeval/.
Excludes seed-only validation metrics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/diagnostics/direct_r198_40ep_linear_lr_full_extract_reeval"
PKG = OUT / "collaborator_package"
FIG = PKG / "figures"
CELLS = OUT / "cells"
SM = ROOT / "saved-models"

sys_path_scripts = ROOT / "scripts"
if str(sys_path_scripts) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(sys_path_scripts))
from direct_r198_eval_protocol import (  # noqa: E402
    PROTOCOL_FULL_SUBGRAPH,
    assert_collaborator_merge_allowed,
    infer_protocol,
    official_protocol_block,
)

# Distinct color + marker + linestyle per arm so the legend is unambiguous.
# 6.21e-3 = solid; 2e-3 = dashed; 1e-3 = dotted.
_DASH = (4.0, 2.0)
_DOT = (1.5, 1.5)
ARMS = [
    {
        "run": "direct_r198_infonce_40ep_seed2_linear_lr6p2e-3",
        "method": "DIRECT_H",
        "peak_lr": 0.006213266113989207,
        "label": "DIRECT_H · lr≈6.21e-3 (solid)",
        "color": "#0072B2",
        "ls": "-",
        "dashes": None,
        "marker": "o",
    },
    {
        "run": "direct_r198_tfmoe_40ep_seed2_linear_lr6p2e-3",
        "method": "DIRECT_H_TFMOE",
        "peak_lr": 0.006213266113989207,
        "label": "TFMOE · lr≈6.21e-3 (solid)",
        "color": "#D55E00",
        "ls": "-",
        "dashes": None,
        "marker": "s",
    },
    {
        "run": "direct_r198_infonce_40ep_seed2_linear_lr2e-3",
        "method": "DIRECT_H",
        "peak_lr": 0.002,
        "label": "DIRECT_H · lr=2e-3 (dashed)",
        "color": "#56B4E9",
        "ls": "--",
        "dashes": _DASH,
        "marker": "^",
    },
    {
        "run": "direct_r198_tfmoe_40ep_seed2_linear_lr2e-3",
        "method": "DIRECT_H_TFMOE",
        "peak_lr": 0.002,
        "label": "TFMOE · lr=2e-3 (dashed)",
        "color": "#009E73",
        "ls": "--",
        "dashes": _DASH,
        "marker": "D",
    },
    {
        "run": "direct_r198_infonce_40ep_seed2_linear_lr1e-3",
        "method": "DIRECT_H",
        "peak_lr": 0.001,
        "label": "DIRECT_H · lr=1e-3 (dotted)",
        "color": "#CC79A7",
        "ls": ":",
        "dashes": _DOT,
        "marker": "v",
    },
    {
        "run": "direct_r198_tfmoe_40ep_seed2_linear_lr1e-3",
        "method": "DIRECT_H_TFMOE",
        "peak_lr": 0.001,
        "label": "TFMOE · lr=1e-3 (dotted)",
        "color": "#000000",
        "ls": ":",
        "dashes": _DOT,
        "marker": "P",
    },
]
EPS = [3, 10, 20, 30, 40]
TF_NAMES = (
    "sender interarrival",
    "sender past 7d count",
    "amount vs sender past mean",
)
# Distinct color per TF expert (Okabe–Ito); LR still via solid/dashed + marker.
TF_COLORS = ("#0072B2", "#E69F00", "#CC79A7")
TF_MARKERS = ("o", "s", "^")

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


def _load_steps(run: str) -> pd.DataFrame:
    p = ROOT / "results/diagnostics" / run / "logs" / "steps.jsonl"
    rows = [json.loads(l) for l in p.open() if l.strip()]
    return pd.DataFrame(rows)


def _epoch_end_row(steps: pd.DataFrame, ep: int) -> Optional[Dict[str, Any]]:
    sub = steps[steps["epoch"].astype(int) == int(ep)]
    if sub.empty:
        return None
    return sub.iloc[-1].to_dict()


def _load_cell(run: str, ep: int) -> Dict[str, Any]:
    return json.loads((CELLS / run / f"epoch_{ep:02d}.json").read_text())


def collect() -> pd.DataFrame:
    rows = []
    refused: List[str] = []
    for a in ARMS:
        steps = _load_steps(a["run"])
        for ep in EPS:
            cell_path = CELLS / a["run"] / f"epoch_{ep:02d}.json"
            cell = json.loads(cell_path.read_text())
            try:
                assert_collaborator_merge_allowed(cell, path=cell_path)
            except ValueError as exc:
                refused.append(str(exc))
                continue
            prim = cell["primary"]
            ver = cell["verify"]
            tr = _epoch_end_row(steps, ep) or {}
            m05 = prim["validation_metrics_at_0.5"]
            mopt = prim["validation_metrics_at_val_optimal_f1"]
            hist = prim["epoch_history"]
            assert abs(hist[-1]["train_bce"] - prim["final_probe_train_bce"]) < 1e-12
            assert abs(hist[-1]["val_bce"] - prim["final_probe_val_bce"]) < 1e-12
            rows.append(
                {
                    "method": a["method"],
                    "peak_lr": a["peak_lr"],
                    "ssl_epoch": ep,
                    "run": a["run"],
                    "label": a["label"],
                    "color": a["color"],
                    "ls": a["ls"],
                    "protocol": infer_protocol(cell),
                    "validation_auprc": float(prim["validation_auprc"]),
                    "f1_at_0.5": float(m05["f1"]),
                    "f1_at_val_thr": float(mopt["f1"]),
                    "val_thr": float(mopt["threshold"]),
                    "final_probe_train_bce": float(prim["final_probe_train_bce"]),
                    "final_probe_val_bce": float(prim["final_probe_val_bce"]),
                    "best_probe_epoch": int(prim["best_probe_epoch"]),
                    "L_contrast_raw": float(tr.get("L_contrast_raw", np.nan)),
                    "L_total": float(tr["L_total"]) if "L_total" in tr else np.nan,
                    "L_tf_raw_0": float(tr["L_tf_raw_0"]) if "L_tf_raw_0" in tr else np.nan,
                    "L_tf_raw_1": float(tr["L_tf_raw_1"]) if "L_tf_raw_1" in tr else np.nan,
                    "L_tf_raw_2": float(tr["L_tf_raw_2"]) if "L_tf_raw_2" in tr else np.nan,
                    "w_contrast": float(tr["w_contrast"]) if "w_contrast" in tr else np.nan,
                    "w_tf_0": float(tr["w_tf_0"]) if "w_tf_0" in tr else np.nan,
                    "w_tf_1": float(tr["w_tf_1"]) if "w_tf_1" in tr else np.nan,
                    "w_tf_2": float(tr["w_tf_2"]) if "w_tf_2" in tr else np.nan,
                    "alpha": float(tr["alpha"]) if "alpha" in tr else np.nan,
                    "verify_ok": bool(ver["ok"]),
                    "train_val_intersect": int(ver["train_val_intersect"]),
                    "seed_only_bug_evidence": bool(ver.get("seed_only_bug_evidence")),
                    "n_val": int(ver["n_val"]),
                    "extractor": cell.get("extractor"),
                    "seed_only_r198": cell.get("seed_only_r198"),
                    "checkpoint": cell.get("checkpoint"),
                    "learner": prim["learner"],
                    "mlp_epochs": prim["mlp_epochs"],
                    "mlp_lr": prim["mlp_lr"],
                    "mlp_batch_size": prim["mlp_batch_size"],
                    "input_dim": prim["input_dim"],
                }
            )
    if refused:
        raise SystemExit(
            "Refusing collaborator package build; non-full_subgraph or failed cells:\n- "
            + "\n- ".join(refused)
        )
    return pd.DataFrame(rows)


def _save(fig, stem: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{stem}.png", bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _legend(ax):
    # Long handles + markerscale so solid/dashed and markers are obvious in the key.
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=8,
        handlelength=4.5,
        handletextpad=0.6,
        borderaxespad=0.0,
        markerscale=1.15,
    )


def _style_line(ax, xs, ys, a, *, label=None, lw=2.0, markersize=7, marker=None, zorder=2):
    (line,) = ax.plot(
        xs,
        ys,
        color=a["color"],
        ls=a["ls"],
        marker=a["marker"] if marker is None else marker,
        lw=lw,
        markersize=markersize,
        label=a["label"] if label is None else label,
        zorder=zorder,
    )
    if a.get("dashes") is not None:
        line.set_dashes(a["dashes"])
    return line


def plot_all(df: pd.DataFrame) -> None:
    xs = np.asarray(EPS, dtype=float)

    def line(ax, col, ylabel, title):
        for a in ARMS:
            sub = df[df["run"] == a["run"]].set_index("ssl_epoch").reindex(EPS)
            _style_line(ax, xs, sub[col].to_numpy(dtype=float), a)
        ax.set_xticks(EPS)
        ax.set_xlabel("SSL epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        _legend(ax)

    # 1 raw contrastive
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    line(ax, "L_contrast_raw", "Raw InfoNCE (epoch-end)", "Raw contrastive loss vs SSL epoch")
    fig.tight_layout()
    _save(fig, "01_raw_contrastive_loss_vs_epoch")

    # 2 TFMOE total + expert raw MAE
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 7.0), sharex=True)
    for a in ARMS:
        if a["method"] != "DIRECT_H_TFMOE":
            continue
        sub = df[df["run"] == a["run"]].set_index("ssl_epoch").reindex(EPS)
        _style_line(axes[0], xs, sub["L_total"].to_numpy(dtype=float), a)
        lr_tag = "solid" if a["dashes"] is None else "dashed"
        for i, name in enumerate(TF_NAMES):
            (eline,) = axes[1].plot(
                xs,
                sub[f"L_tf_raw_{i}"].to_numpy(dtype=float),
                color=TF_COLORS[i],
                ls=a["ls"],
                marker=TF_MARKERS[i],
                lw=1.8,
                markersize=6.5,
                label=f"{name} · lr={a['peak_lr']:.2e} ({lr_tag})",
            )
            if a.get("dashes") is not None:
                eline.set_dashes(a["dashes"])
    axes[0].set_ylabel("L_total")
    axes[0].set_title("TFMOE total training loss vs SSL epoch")
    _legend(axes[0])
    axes[1].set_xticks(EPS)
    axes[1].set_xlabel("SSL epoch")
    axes[1].set_ylabel("Expert raw MAE")
    axes[1].set_title("TFMOE expert-head raw losses vs SSL epoch")
    _legend(axes[1])
    fig.tight_layout()
    _save(fig, "02_tfmoe_total_and_expert_losses_vs_epoch")

    # 3–6 downstream
    for stem, col, ylab, title in [
        ("03_val_auprc_vs_epoch", "validation_auprc", "Validation AUPRC", "Validation AUPRC vs SSL epoch"),
        ("04_val_f1_at_0.5_vs_epoch", "f1_at_0.5", "Validation F1@0.5", "F1@0.5 vs SSL epoch"),
        (
            "05_val_f1_at_val_thr_vs_epoch",
            "f1_at_val_thr",
            "Validation F1@val-threshold",
            "F1@validation-selected threshold vs SSL epoch",
        ),
        (
            "06_final_probe_val_bce_vs_epoch",
            "final_probe_val_bce",
            "Final probe-epoch val BCE",
            "Final probe-epoch validation BCE vs SSL epoch",
        ),
    ]:
        fig, ax = plt.subplots(figsize=(8.2, 4.2))
        line(ax, col, ylab, title)
        fig.tight_layout()
        _save(fig, stem)

    # 7 TFMOE learned weights — separate panel per peak LR; direct labels
    tfmoe_arms = [a for a in ARMS if a["method"] == "DIRECT_H_TFMOE" and a["peak_lr"] in (0.006213266113989207, 0.002)]
    # Prefer stable order: 6.21e-3 then 2e-3
    tfmoe_arms = sorted(tfmoe_arms, key=lambda a: -a["peak_lr"])
    weight_specs = [
        ("w_contrast", "w_contrast", "#0072B2", "-", "o"),
        ("w_tf_0", "w_tf_0 (interarrival)", "#E69F00", "--", "s"),
        ("w_tf_1", "w_tf_1 (past 7d)", "#009E73", "-.", "^"),
        ("w_tf_2", "w_tf_2 (amt vs mean)", "#CC79A7", ":", "D"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    if len(tfmoe_arms) == 1:
        axes = [axes]
    for ax, a in zip(axes, tfmoe_arms):
        sub = df[df["run"] == a["run"]].set_index("ssl_epoch").reindex(EPS)
        series = {
            "w_contrast": sub["w_contrast"].to_numpy(dtype=float),
            "w_tf_0": sub["w_tf_0"].to_numpy(dtype=float),
            "w_tf_1": sub["w_tf_1"].to_numpy(dtype=float),
            "w_tf_2": sub["w_tf_2"].to_numpy(dtype=float),
        }
        series["sum_w_tf"] = series["w_tf_0"] + series["w_tf_1"] + series["w_tf_2"]
        for key, lab, color, ls, marker in weight_specs + [
            ("sum_w_tf", "Σw_tf", "#000000", "-", "P")
        ]:
            ys = series[key]
            ax.plot(xs, ys, color=color, ls=ls, marker=marker, lw=2.0, markersize=7, label=lab)
            ax.annotate(
                lab,
                xy=(xs[-1], ys[-1]),
                xytext=(6, 0),
                textcoords="offset points",
                fontsize=7.5,
                color=color,
                va="center",
                ha="left",
            )
        lr_tag = "lr≈6.21e-3" if a["peak_lr"] > 0.005 else "lr=2e-3"
        ax.set_xticks(EPS)
        ax.set_xlim(2, 48)
        ax.set_ylim(0, 1.05)
        ax.axhline(1.0, color="#888888", lw=0.8, ls=":", alpha=0.7)
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_xlabel("SSL epoch")
        ax.set_title(f"TFMOE learned weights · {lr_tag}")
    axes[0].set_ylabel("Weight")
    fig.suptitle(
        "Source: steps.jsonl cols w_contrast, w_tf_0/1/2; Σw_tf = w_tf_0+w_tf_1+w_tf_2 (epoch-end).",
        fontsize=9,
        y=1.02,
    )
    fig.tight_layout()
    _save(fig, "07_tfmoe_learned_weights_vs_epoch")


def tables(df: pd.DataFrame) -> Dict[str, Any]:
    # A trajectory
    A = df[
        [
            "method",
            "peak_lr",
            "ssl_epoch",
            "validation_auprc",
            "f1_at_0.5",
            "f1_at_val_thr",
            "final_probe_train_bce",
            "final_probe_val_bce",
            "L_contrast_raw",
            "L_total",
        ]
    ].copy()
    A.to_csv(PKG / "table_A_complete_trajectory.csv", index=False)
    A.to_json(PKG / "table_A_complete_trajectory.json", orient="records", indent=2)

    # B best by AUPRC
    B_rows = []
    for a in ARMS:
        sub = df[df["run"] == a["run"]]
        best = sub.loc[sub["validation_auprc"].idxmax()]
        last = sub.loc[sub["ssl_epoch"].idxmax()]
        B_rows.append(
            {
                "method": a["method"],
                "peak_lr": a["peak_lr"],
                "best_ssl_epoch_by_auprc": int(best["ssl_epoch"]),
                "best_validation_auprc": float(best["validation_auprc"]),
                "best_f1_at_0.5": float(best["f1_at_0.5"]),
                "best_f1_at_val_thr": float(best["f1_at_val_thr"]),
                "best_final_train_bce": float(best["final_probe_train_bce"]),
                "best_final_val_bce": float(best["final_probe_val_bce"]),
                "best_epoch_L_contrast_raw": float(best["L_contrast_raw"]),
                "best_epoch_L_total": None
                if np.isnan(best["L_total"])
                else float(best["L_total"]),
                "epoch40_L_contrast_raw": float(last["L_contrast_raw"]),
                "epoch40_L_total": None if np.isnan(last["L_total"]) else float(last["L_total"]),
                "epoch40_validation_auprc": float(last["validation_auprc"]),
            }
        )
    B = pd.DataFrame(B_rows)
    B.to_csv(PKG / "table_B_best_checkpoint_summary.csv", index=False)
    B.to_json(PKG / "table_B_best_checkpoint_summary.json", orient="records", indent=2)

    # C matched comparisons across the three peak LRs
    def _row(method: str, lr: float, ep: int) -> pd.Series:
        return df[
            (df["method"] == method) & (np.isclose(df["peak_lr"], lr)) & (df["ssl_epoch"] == ep)
        ].iloc[0]

    C = []
    for ep in EPS:
        h621 = _row("DIRECT_H", 0.006213266113989207, ep)
        h2 = _row("DIRECT_H", 0.002, ep)
        h1 = _row("DIRECT_H", 0.001, ep)
        t621 = _row("DIRECT_H_TFMOE", 0.006213266113989207, ep)
        t2 = _row("DIRECT_H_TFMOE", 0.002, ep)
        t1 = _row("DIRECT_H_TFMOE", 0.001, ep)
        C.append(
            {
                "ssl_epoch": ep,
                "DIRECT_H_auprc_6p21e-3": float(h621["validation_auprc"]),
                "DIRECT_H_auprc_2e-3": float(h2["validation_auprc"]),
                "DIRECT_H_auprc_1e-3": float(h1["validation_auprc"]),
                "DIRECT_H_delta_1e-3_minus_2e-3": float(h1["validation_auprc"] - h2["validation_auprc"]),
                "DIRECT_H_delta_2e-3_minus_6p21e-3": float(h2["validation_auprc"] - h621["validation_auprc"]),
                "TFMOE_auprc_6p21e-3": float(t621["validation_auprc"]),
                "TFMOE_auprc_2e-3": float(t2["validation_auprc"]),
                "TFMOE_auprc_1e-3": float(t1["validation_auprc"]),
                "TFMOE_delta_1e-3_minus_2e-3": float(t1["validation_auprc"] - t2["validation_auprc"]),
                "TFMOE_delta_2e-3_minus_6p21e-3": float(t2["validation_auprc"] - t621["validation_auprc"]),
                "delta_TF_minus_H_at_6p21e-3": float(t621["validation_auprc"] - h621["validation_auprc"]),
                "delta_TF_minus_H_at_2e-3": float(t2["validation_auprc"] - h2["validation_auprc"]),
                "delta_TF_minus_H_at_1e-3": float(t1["validation_auprc"] - h1["validation_auprc"]),
            }
        )
    Cdf = pd.DataFrame(C)
    Cdf.to_csv(PKG / "table_C_matched_comparisons.csv", index=False)
    Cdf.to_json(PKG / "table_C_matched_comparisons.json", orient="records", indent=2)

    # D CE/BCE note
    D = {
        "probe_loss": {
            "name": "unweighted BCEWithLogitsLoss (one logit)",
            "learner": "PaperStyleMLP",
            "mlp_epochs": 20,
            "mlp_lr": 1e-3,
            "mlp_batch_size": 8192,
            "mlp_seed": 2,
            "features": "R198 + edge X + temporal-flow cache",
            "final_bce_definition": "mean train/val BCE at MLP epoch 20 (last probe epoch), not best-AUPRC epoch",
        },
        "final_probe_bce_by_cell": df[
            [
                "method",
                "peak_lr",
                "ssl_epoch",
                "final_probe_train_bce",
                "final_probe_val_bce",
                "best_probe_epoch",
                "validation_auprc",
            ]
        ].to_dict(orient="records"),
        "supervised_multi_gin_eu_seed2": {
            "source": "results/diagnostics/direct_r198_tfmoe_40ep_linear_lr_sweep/ce_audit.json",
            "loss": "weighted CrossEntropyLoss (two logits)",
            "class_weights": {"0": 1.0000182882773443, "1": 6.275014431494497},
            "best_val_f1_epoch": 43,
            "train_loss_weighted_ce_at_best_val_f1_epoch": 0.011318868767954312,
            "validation_f1_argmax": 0.610073571024335,
            "validation_auprc": 0.5508610996276531,
            "directly_comparable_to_probe_bce": False,
            "caveat": (
                "Probe BCE is unweighted one-logit BCEWithLogits; supervised uses weighted "
                "two-logit CrossEntropy. Do not compare these loss numbers directly."
            ),
        },
    }
    (PKG / "table_D_ce_bce_note.json").write_text(json.dumps(D, indent=2) + "\n")
    return {"A": A, "B": B, "C": Cdf, "D": D}


def analysis(df: pd.DataFrame, C: pd.DataFrame, B: pd.DataFrame) -> Dict[str, Any]:
    def mean_delta(col):
        return float(C[col].mean())

    # longer training: compare ep of peak vs ep40
    longer = []
    for a in ARMS:
        sub = df[df["run"] == a["run"]].sort_values("ssl_epoch")
        peak_ep = int(sub.loc[sub["validation_auprc"].idxmax(), "ssl_epoch"])
        a3 = float(sub.loc[sub["ssl_epoch"] == 3, "validation_auprc"].iloc[0])
        a10 = float(sub.loc[sub["ssl_epoch"] == 10, "validation_auprc"].iloc[0])
        a40 = float(sub.loc[sub["ssl_epoch"] == 40, "validation_auprc"].iloc[0])
        peak_a = float(sub["validation_auprc"].max())
        longer.append(
            {
                "method": a["method"],
                "peak_lr": a["peak_lr"],
                "peak_epoch": peak_ep,
                "auprc_ep3": a3,
                "auprc_ep10": a10,
                "auprc_ep40": a40,
                "auprc_peak": peak_a,
                "delta_ep40_minus_peak": a40 - peak_a,
                "raw_contrast_ep3": float(sub.loc[sub["ssl_epoch"] == 3, "L_contrast_raw"].iloc[0]),
                "raw_contrast_ep40": float(sub.loc[sub["ssl_epoch"] == 40, "L_contrast_raw"].iloc[0]),
                "raw_contrast_delta_40_minus_3": float(
                    sub.loc[sub["ssl_epoch"] == 40, "L_contrast_raw"].iloc[0]
                    - sub.loc[sub["ssl_epoch"] == 3, "L_contrast_raw"].iloc[0]
                ),
            }
        )

    peak_lrs = sorted({float(a["peak_lr"]) for a in ARMS}, reverse=True)
    tf_wins = []
    for ep in EPS:
        for lr in peak_lrs:
            h = df[
                (df["method"] == "DIRECT_H")
                & (df["ssl_epoch"] == ep)
                & np.isclose(df["peak_lr"], lr)
            ].iloc[0]
            t = df[
                (df["method"] == "DIRECT_H_TFMOE")
                & (df["ssl_epoch"] == ep)
                & np.isclose(df["peak_lr"], lr)
            ].iloc[0]
            tf_wins.append(float(t["validation_auprc"]) > float(h["validation_auprc"]))

    # Best-by-AUPRC per method across LRs (for 1e-3 vs 2e-3 vs 6.21e-3)
    lr_effect = {}
    for method in ("DIRECT_H", "DIRECT_H_TFMOE"):
        subB = B[B["method"] == method].sort_values("best_validation_auprc", ascending=False)
        best = subB.iloc[0]
        by_lr = {
            f"{float(r.peak_lr):.4g}": {
                "best_ssl_epoch": int(r.best_ssl_epoch_by_auprc),
                "best_auprc": float(r.best_validation_auprc),
            }
            for r in subB.itertuples()
        }
        a1 = float(subB[np.isclose(subB["peak_lr"], 0.001)].iloc[0]["best_validation_auprc"])
        a2 = float(subB[np.isclose(subB["peak_lr"], 0.002)].iloc[0]["best_validation_auprc"])
        a621 = float(
            subB[np.isclose(subB["peak_lr"], 0.006213266113989207)].iloc[0]["best_validation_auprc"]
        )
        lr_effect[method] = {
            "best_peak_lr": float(best["peak_lr"]),
            "best_ssl_epoch": int(best["best_ssl_epoch_by_auprc"]),
            "best_auprc": float(best["best_validation_auprc"]),
            "by_peak_lr": by_lr,
            "delta_best_1e-3_minus_2e-3": a1 - a2,
            "delta_best_1e-3_minus_6p21e-3": a1 - a621,
            "1e-3_improves_vs_2e-3": bool(a1 > a2),
            "1e-3_improves_vs_6p21e-3": bool(a1 > a621),
        }

    bce_agree = []
    for a in ARMS:
        sub = df[df["run"] == a["run"]]
        corr = float(np.corrcoef(sub["validation_auprc"], sub["final_probe_val_bce"])[0, 1])
        bce_agree.append({"method": a["method"], "peak_lr": a["peak_lr"], "corr_auprc_vs_val_bce": corr})

    n_matched = len(EPS) * len(peak_lrs)
    out = {
        "n_cells": int(len(df)),
        "grid_complete": int(len(df)) == n_matched,
        "n_matched_lr_epoch_cells": n_matched,
        "lr_effect_on_best_checkpoint": lr_effect,
        "mean_delta_1e-3_minus_2e-3": {
            "DIRECT_H": mean_delta("DIRECT_H_delta_1e-3_minus_2e-3"),
            "TFMOE": mean_delta("TFMOE_delta_1e-3_minus_2e-3"),
        },
        "tfmoe_vs_direct_h": {
            "tf_wins_out_of": n_matched,
            "tf_wins": int(sum(tf_wins)),
            "mean_delta_tf_minus_h_6p21e-3": mean_delta("delta_TF_minus_H_at_6p21e-3"),
            "mean_delta_tf_minus_h_2e-3": mean_delta("delta_TF_minus_H_at_2e-3"),
            "mean_delta_tf_minus_h_1e-3": mean_delta("delta_TF_minus_H_at_1e-3"),
        },
        "longer_training": longer,
        "bce_auprc_correlation_per_run": bce_agree,
        "best_overall": B.sort_values("best_validation_auprc", ascending=False).iloc[0].to_dict(),
    }
    (PKG / "analysis_answers.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


def write_summaries(df: pd.DataFrame, B: pd.DataFrame, C: pd.DataFrame, ans: Dict[str, Any]) -> None:
    best = ans["best_overall"]
    tf = ans["tfmoe_vs_direct_h"]
    lr_eff = ans["lr_effect_on_best_checkpoint"]
    mean12 = ans["mean_delta_1e-3_minus_2e-3"]
    n_cells = ans["n_cells"]
    n_matched = ans["n_matched_lr_epoch_cells"]

    lines: List[str] = []
    lines.append("# DIRECT_H / TFMOE 40ep linear-LR sweep — collaborator summary")
    lines.append("")
    lines.append("**Scope:** corrected full-subgraph R198 extract + PaperStyleMLP probe (R198+X+TF), seed 2, Small-HI, test locked.")
    lines.append("**Excluded:** seed-only validation metrics (including ID-fixed 1e-3 seed-only, now superseded).")
    lines.append("")
    lines.append("## What was swept")
    lines.append("")
    lines.append("- Methods: DIRECT_H (InfoNCE on R198 only) and DIRECT_H_TFMOE (same + temporal-flow MoE aux heads).")
    lines.append("- Peak LRs: ≈6.21e-3, 2e-3, and 1e-3 under `direct_h_warmup_linear` (40 SSL epochs).")
    lines.append(
        f"- Downstream eval at matched SSL epochs **3, 10, 20, 30, 40** ({n_cells}/{n_matched} cells complete)."
    )
    lines.append("")
    lines.append("## Verification")
    lines.append("")
    lines.append(
        f"- All {n_cells} cells: full-subgraph extract, `seed_only_r198=false`, verify_ok, "
        "train∩val=0, val IDs above train max, PaperStyleMLP 20/1e-3/8192, "
        "ranking via best-val-AUPRC; final BCE = last MLP epoch."
    )
    lines.append("")
    lines.append("## Does lr=1e-3 help vs 2e-3 / 6.21e-3?")
    lines.append("")
    for method, key in (("DIRECT_H", "DIRECT_H"), ("TFMOE", "DIRECT_H_TFMOE")):
        e = lr_eff[key]
        verb_2 = "improves" if e["1e-3_improves_vs_2e-3"] else "worsens"
        verb_6 = "improves" if e["1e-3_improves_vs_6p21e-3"] else "worsens"
        lines.append(
            f"- **{method}:** best-AUPRC checkpoint at lr=1e-3 "
            f"**{verb_2}** vs 2e-3 (Δ={e['delta_best_1e-3_minus_2e-3']:+.4f}) and "
            f"**{verb_6}** vs 6.21e-3 (Δ={e['delta_best_1e-3_minus_6p21e-3']:+.4f}). "
            f"Best overall for this method: lr={e['best_peak_lr']:.4g} @ SSL ep{e['best_ssl_epoch']} "
            f"(AUPRC={e['best_auprc']:.4f}). "
            f"Mean epoch-matched ΔAUPRC (1e-3−2e-3) = "
            f"**{mean12['DIRECT_H' if method=='DIRECT_H' else 'TFMOE']:+.4f}**."
        )
    lines.append("")
    lines.append("## Does longer SSL training help?")
    lines.append("")
    for row in ans["longer_training"]:
        verb = "hurts after peak" if row["delta_ep40_minus_peak"] < -0.01 else (
            "mostly plateaus" if abs(row["delta_ep40_minus_peak"]) <= 0.01 else "still near peak"
        )
        lines.append(
            f"- **{row['method']} lr={row['peak_lr']:.4g}:** peaks at SSL ep **{row['peak_epoch']}** "
            f"(AUPRC={row['auprc_peak']:.4f}); ep40 AUPRC={row['auprc_ep40']:.4f} "
            f"(Δ vs peak={row['delta_ep40_minus_peak']:+.4f}) → {verb}."
        )
    lines.append("")
    lines.append("## Does TFMOE help?")
    lines.append("")
    lines.append(
        f"- Yes at matched checkpoints: TFMOE higher AUPRC on **{tf['tf_wins']}/{tf['tf_wins_out_of']}** "
        f"(3 LRs × 5 epochs). Mean Δ(TF−H): "
        f"**{tf['mean_delta_tf_minus_h_6p21e-3']:+.4f}** (6.21e-3), "
        f"**{tf['mean_delta_tf_minus_h_2e-3']:+.4f}** (2e-3), "
        f"**{tf['mean_delta_tf_minus_h_1e-3']:+.4f}** (1e-3)."
    )
    lines.append("")
    lines.append("## Best configuration (this seed)")
    lines.append("")
    lines.append(
        f"- **{best['method']}**, peak LR **{best['peak_lr']:.4g}**, SSL epoch **{int(best['best_ssl_epoch_by_auprc'])}**: "
        f"AUPRC=**{best['best_validation_auprc']:.4f}**, F1@0.5={best['best_f1_at_0.5']:.4f}, "
        f"F1@val-thr={best['best_f1_at_val_thr']:.4f}, "
        f"final train/val BCE={best['best_final_train_bce']:.4f}/{best['best_final_val_bce']:.4f}."
    )
    lines.append("- Single seed (seed 2); treat gaps of a few AUPRC points cautiously.")
    lines.append("")
    lines.append("## Training loss vs downstream")
    lines.append("")
    lines.append("- DIRECT_H: raw InfoNCE slowly **decreases** through ep40 while downstream peaks mid-run then declines/plateaus → SSL loss keeps improving after downstream peak.")
    lines.append("- TFMOE: **L_total** falls (TF expert MAEs improve; weight mass shifts to TF), but **raw InfoNCE can rise** while downstream is already strong → do not use raw contrast alone as a stopping rule for TFMOE.")
    lines.append("- Probe val BCE generally moves opposite AUPRC within a run (negative correlation), so BCE trends broadly agree with ranking metrics, but selection should stay on AUPRC.")
    lines.append("")
    lines.append("## BCE / supervised CE caveat")
    lines.append("")
    lines.append("- Reported final BCE = last PaperStyleMLP epoch (20), unweighted one-logit BCEWithLogits.")
    lines.append("- Supervised Multi-GIN+EU seed2 best-val checkpoint: weighted two-logit CE train loss ≈ **0.0113** at ep43 (val F1 argmax **0.6101**). **Not directly comparable** to probe BCE.")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- Package dir: `{PKG.relative_to(ROOT)}/`")
    lines.append(f"- Figures: `{FIG.relative_to(ROOT)}/`")
    lines.append(f"- Tables: `table_A_*.csv`, `table_B_*.csv`, `table_C_*.csv`, `table_D_ce_bce_note.json`")
    lines.append(f"- Source cells: `{CELLS.relative_to(ROOT)}/`")
    lines.append(f"- Embeddings: `embeddings/direct_r198_40ep_linear_lr_full_extract/`")
    lines.append("")
    (PKG / "collaborator_summary.md").write_text("\n".join(lines) + "\n")

    email = []
    email.append("Subject: DIRECT_H / TFMOE 40ep LR sweep — corrected matched trajectories incl. 1e-3 (seed 2)")
    email.append("")
    email.append(
        "Hi — update on the corrected full-subgraph re-eval "
        "(seed-only numbers, including ID-fixed 1e-3 seed-only, are provisional/superseded and not used)."
    )
    email.append("")
    email.append(
        f"We finished the matched 6×5 grid (DIRECT_H & TFMOE × {{6.21e-3, 2e-3, 1e-3}} × SSL epochs "
        f"{{3,10,20,30,40}}; {n_cells}/{n_matched} cells) with the same R198+X+TF → PaperStyleMLP protocol."
    )
    email.append("")
    h = lr_eff["DIRECT_H"]
    t = lr_eff["DIRECT_H_TFMOE"]
    email.append(
        f"Main takeaways (one seed): TFMOE beats DIRECT_H on {tf['tf_wins']}/{tf['tf_wins_out_of']} matched cells. "
        f"Best overall: {best['method']} lr={best['peak_lr']:.4g} @ SSL ep{int(best['best_ssl_epoch_by_auprc'])} "
        f"(AUPRC {best['best_validation_auprc']:.3f}). "
        f"Best DIRECT_H LR={h['best_peak_lr']:.4g} (AUPRC {h['best_auprc']:.3f}); "
        f"best TFMOE LR={t['best_peak_lr']:.4g} (AUPRC {t['best_auprc']:.3f}). "
        f"lr=1e-3 vs 2e-3 best-checkpoint ΔAUPRC: H {h['delta_best_1e-3_minus_2e-3']:+.3f}, "
        f"TFMOE {t['delta_best_1e-3_minus_2e-3']:+.3f}."
    )
    email.append("")
    email.append(f"Package: {PKG.relative_to(ROOT)}/ (plots 01–07 + tables A–D + summary).")
    email.append("")
    email.append("Happy to walk through the figures.")
    (PKG / "email_ready.txt").write_text("\n".join(email) + "\n")


def verify_block(df: pd.DataFrame) -> Dict[str, Any]:
    issues = []
    expected = len(ARMS) * len(EPS)
    if len(df) != expected:
        issues.append(f"expected {expected} rows, got {len(df)}")
    if "protocol" in df.columns and not bool((df["protocol"] == PROTOCOL_FULL_SUBGRAPH).all()):
        issues.append("protocol != full_subgraph present")
    if not bool(df["verify_ok"].all()):
        issues.append("verify_ok false present")
    if not bool((df["train_val_intersect"] == 0).all()):
        issues.append("train/val overlap")
    if bool(df["seed_only_bug_evidence"].any()):
        issues.append("seed-only bug evidence")
    if not bool((df["seed_only_r198"] == False).all()):  # noqa: E712
        issues.append("seed_only_r198 not false")
    if not bool((df["learner"] == "PaperStyleMLP").all()):
        issues.append("learner mismatch")
    if not bool((df["mlp_epochs"] == 20).all()):
        issues.append("mlp_epochs mismatch")
    if not bool((df["input_dim"] == 227).all()):
        issues.append("input_dim != 227 (expected R198+X+TF)")
    rep = {
        "n_cells": int(len(df)),
        "issues": issues,
        "sound": len(issues) == 0,
        "n_val_unique": sorted(df["n_val"].unique().tolist()),
        "required_protocol": PROTOCOL_FULL_SUBGRAPH,
        "protocol_block": official_protocol_block(),
    }
    (PKG / "verification.json").write_text(json.dumps(rep, indent=2) + "\n")
    return rep


def main() -> int:
    PKG.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    df = collect()
    ver = verify_block(df)
    if not ver["sound"]:
        print(json.dumps({"status": "verification_failed", "verification": ver}, indent=2))
        return 2
    plot_all(df)
    tabs = tables(df)
    ans = analysis(df, tabs["C"], tabs["B"])
    write_summaries(df, tabs["B"], tabs["C"], ans)
    # also refresh top-level note pointer
    index = {
        "status": "ok",
        "package": str(PKG.relative_to(ROOT)),
        "verification": ver,
        "best": ans["best_overall"],
        "figures": sorted(p.name for p in FIG.glob("*.png")),
        "tables": [
            "table_A_complete_trajectory.csv",
            "table_B_best_checkpoint_summary.csv",
            "table_C_matched_comparisons.csv",
            "table_D_ce_bce_note.json",
        ],
        "summary": "collaborator_summary.md",
        "email": "email_ready.txt",
    }
    (PKG / "README.md").write_text(
        "# Collaborator package — corrected 40ep matched trajectories\n\n"
        "See `collaborator_summary.md` and `email_ready.txt`.\n\n"
        "**Required protocol:** `full_subgraph` only.\n\n"
        "Seed-only / diagnostic-provisional results are refused at merge time.\n"
        "Official command: `python scripts/official_direct_r198_collaborator_eval.py` "
        "(see `notes/direct_r198_official_collaborator_eval.md`).\n"
    )
    index["protocol"] = PROTOCOL_FULL_SUBGRAPH
    index["protocol_block"] = official_protocol_block()
    index["seed_only_excluded"] = True
    (PKG / "package_index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(json.dumps(index, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
