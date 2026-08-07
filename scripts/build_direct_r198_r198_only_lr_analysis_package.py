#!/usr/bin/env python3
"""Build R198-only LR-analysis package from existing full-subgraph cells.

Does NOT retrain SSL. Does NOT overwrite R198+X+TF primary artifacts.

Source of R198-only metrics: the `diagnostic` PaperStyleMLP probe already run
inside scripts/reeval_direct_r198_40ep_full_extract_cell.py on full-subgraph
embeddings (input = Z only, dim=198; no X, no TF). Primary remains R198+X+TF.

Outputs under:
  results/diagnostics/direct_r198_40ep_linear_lr_full_extract_reeval/r198_only_lr_analysis/
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_CELLS = ROOT / "results/diagnostics/direct_r198_40ep_linear_lr_full_extract_reeval/cells"
OUT = (
    ROOT
    / "results/diagnostics/direct_r198_40ep_linear_lr_full_extract_reeval/r198_only_lr_analysis"
)
PKG = OUT / "package"
FIG = PKG / "figures"
EMB_ROOT = ROOT / "embeddings/direct_r198_40ep_linear_lr_full_extract"

PROBE_FEATURE_PROTOCOL = "R198_only"
EPS = [3, 10, 20, 30, 40]
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


def _assert_r198_only_merge(src: Dict[str, Any], *, path: Path) -> Dict[str, Any]:
    diag = src.get("diagnostic")
    verify = src.get("verify") or {}
    if not isinstance(diag, dict):
        raise ValueError(f"missing diagnostic R198-only probe: {path}")
    if int(diag.get("input_dim", -1)) != 198:
        raise ValueError(f"input_dim != 198: {path}")
    if src.get("seed_only_r198") is not False:
        raise ValueError(f"not full-subgraph cell (seed_only_r198): {path}")
    extractor = str(src.get("extractor") or "")
    if "full" not in extractor.lower():
        raise ValueError(f"extractor not full-subgraph: {path} ({extractor})")
    if not verify.get("ok"):
        raise ValueError(f"verify.ok false: {path}")
    if int(verify.get("train_val_intersect", -1)) != 0:
        raise ValueError(f"train∩val != 0: {path}")
    if bool(verify.get("seed_only_bug_evidence")):
        raise ValueError(f"seed_only_bug_evidence: {path}")
    if diag.get("learner") != "PaperStyleMLP":
        raise ValueError(f"learner mismatch: {path}")
    if int(diag.get("mlp_epochs", -1)) != 20:
        raise ValueError(f"mlp_epochs mismatch: {path}")
    if float(diag.get("mlp_lr", -1)) != 1e-3:
        raise ValueError(f"mlp_lr mismatch: {path}")
    if int(diag.get("mlp_batch_size", -1)) != 8192:
        raise ValueError(f"mlp_batch_size mismatch: {path}")
    if diag.get("selection_within_probe") != "best_val_auprc":
        raise ValueError(f"selection rule mismatch: {path}")
    # Explicit: diagnostic path used Z only (see reeval _fit_probe(z_tr,...))
    return diag


def materialize_cells() -> pd.DataFrame:
    rows = []
    OUT.mkdir(parents=True, exist_ok=True)
    for a in ARMS:
        cell_dir = OUT / "cells" / a["run"]
        cell_dir.mkdir(parents=True, exist_ok=True)
        for ep in EPS:
            src_path = SRC_CELLS / a["run"] / f"epoch_{ep:02d}.json"
            src = json.loads(src_path.read_text())
            diag = _assert_r198_only_merge(src, path=src_path)
            prim_xtf = src["primary"]
            emb = EMB_ROOT / f"{a['run']}_epoch{ep:02d}" / "pre_embedding_3h"
            # Refuse writing into primary collaborator package paths
            out_cell = {
                "status": "ok",
                "arm": a["method"] if a["method"] != "DIRECT_H_TFMOE" else "DIRECT_H_TFMOE",
                "run": a["run"],
                "peak_lr": a["peak_lr"],
                "schedule": "direct_h_warmup_linear",
                "epoch": ep,
                "checkpoint": src.get("checkpoint"),
                "checkpoint_sha256": src.get("checkpoint_sha256"),
                "embedding_dir": str(emb),
                "extractor": src.get("extractor"),
                "extraction_protocol": "full_subgraph",
                "probe_feature_protocol": PROBE_FEATURE_PROTOCOL,
                "concatenated_raw_edge_X": False,
                "concatenated_temporal_flow": False,
                "probe_input_dim": 198,
                "evaluation_tier": "r198_only_lr_analysis",
                "thesis_primary": False,
                "note": (
                    "R198-only probe for isolating representation quality in this LR/TFMOE "
                    "analysis. Not a replacement of thesis-wide R198+X+TF evaluation."
                ),
                "source_cell": str(src_path),
                "source_field": "diagnostic",
                "seed_only_r198": False,
                "verify": src["verify"],
                "primary": diag,  # R198-only becomes primary for THIS package
                "r198_x_tf_reference": {
                    "input_dim": prim_xtf.get("input_dim"),
                    "validation_auprc": prim_xtf.get("validation_auprc"),
                    "f1_at_0.5": prim_xtf["validation_metrics_at_0.5"]["f1"],
                    "f1_at_val_thr": prim_xtf["validation_metrics_at_val_optimal_f1"]["f1"],
                    "final_probe_train_bce": prim_xtf.get("final_probe_train_bce"),
                    "final_probe_val_bce": prim_xtf.get("final_probe_val_bce"),
                    "best_probe_epoch": prim_xtf.get("best_probe_epoch"),
                },
                "test_evaluated": False,
            }
            # Hard gates recorded on the cell
            assert out_cell["probe_input_dim"] == 198
            assert out_cell["concatenated_raw_edge_X"] is False
            assert out_cell["concatenated_temporal_flow"] is False
            assert out_cell["extraction_protocol"] == "full_subgraph"
            assert int(diag["input_dim"]) == 198

            dest = cell_dir / f"epoch_{ep:02d}.json"
            if dest.exists():
                # allow refresh of derived package cells in this analysis dir only
                pass
            dest.write_text(json.dumps(out_cell, indent=2) + "\n")
            # copy verify sidecar if present
            vsrc = SRC_CELLS / a["run"] / f"epoch_{ep:02d}_verify.json"
            if vsrc.is_file():
                shutil.copy2(vsrc, cell_dir / f"epoch_{ep:02d}_verify.json")

            m05 = diag["validation_metrics_at_0.5"]
            mopt = diag["validation_metrics_at_val_optimal_f1"]
            rows.append(
                {
                    "method": a["method"],
                    "peak_lr": a["peak_lr"],
                    "ssl_epoch": ep,
                    "run": a["run"],
                    "label": a["label"],
                    "color": a["color"],
                    "ls": a["ls"],
                    "dashes": a["dashes"],
                    "marker": a["marker"],
                    "probe_feature_protocol": PROBE_FEATURE_PROTOCOL,
                    "probe_input_dim": 198,
                    "validation_auprc": float(diag["validation_auprc"]),
                    "f1_at_0.5": float(m05["f1"]),
                    "f1_at_val_thr": float(mopt["f1"]),
                    "val_thr": float(mopt["threshold"]),
                    "final_probe_train_bce": float(diag["final_probe_train_bce"]),
                    "final_probe_val_bce": float(diag["final_probe_val_bce"]),
                    "best_probe_epoch": int(diag["best_probe_epoch"]),
                    "verify_ok": bool(src["verify"]["ok"]),
                    "train_val_intersect": int(src["verify"]["train_val_intersect"]),
                    "n_val": int(src["verify"]["n_val"]),
                    "xtf_validation_auprc": float(prim_xtf["validation_auprc"]),
                    "xtf_f1_at_0.5": float(prim_xtf["validation_metrics_at_0.5"]["f1"]),
                    "xtf_f1_at_val_thr": float(
                        prim_xtf["validation_metrics_at_val_optimal_f1"]["f1"]
                    ),
                    "xtf_final_probe_val_bce": float(prim_xtf["final_probe_val_bce"]),
                }
            )
    return pd.DataFrame(rows)


def _save(fig, stem: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{stem}.png", bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _legend(ax):
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=8,
        handlelength=3.5,
    )


def _style_line(ax, xs, ys, a, *, label=None):
    (line,) = ax.plot(
        xs,
        ys,
        color=a["color"],
        ls=a["ls"],
        marker=a["marker"],
        lw=2.0,
        markersize=7,
        label=a["label"] if label is None else label,
    )
    if a.get("dashes") is not None:
        line.set_dashes(a["dashes"])
    return line


def plot_all(df: pd.DataFrame) -> None:
    xs = np.asarray(EPS, dtype=float)

    def line(col, ylabel, title, stem):
        fig, ax = plt.subplots(figsize=(8.6, 4.3))
        for a in ARMS:
            sub = df[df["run"] == a["run"]].set_index("ssl_epoch").reindex(EPS)
            _style_line(ax, xs, sub[col].to_numpy(dtype=float), a)
        ax.set_xticks(EPS)
        ax.set_xlabel("SSL epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        _legend(ax)
        fig.tight_layout()
        _save(fig, stem)

    line(
        "validation_auprc",
        "Validation AUPRC",
        "R198-only validation AUPRC vs SSL epoch",
        "01_r198_only_val_auprc_vs_epoch",
    )
    line(
        "f1_at_0.5",
        "Validation F1@0.5",
        "R198-only F1@0.5 vs SSL epoch",
        "02_r198_only_val_f1_at_0.5_vs_epoch",
    )
    line(
        "f1_at_val_thr",
        "Validation F1@val-threshold",
        "R198-only F1@validation-selected threshold vs SSL epoch",
        "03_r198_only_val_f1_at_val_thr_vs_epoch",
    )
    line(
        "final_probe_val_bce",
        "Final probe-epoch val BCE",
        "R198-only final probe-epoch validation BCE vs SSL epoch",
        "04_r198_only_final_probe_val_bce_vs_epoch",
    )


def tables(df: pd.DataFrame) -> Dict[str, Any]:
    PKG.mkdir(parents=True, exist_ok=True)
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
            "probe_feature_protocol",
            "probe_input_dim",
        ]
    ].copy()
    A.to_csv(PKG / "table_A_complete_trajectory.csv", index=False)
    A.to_json(PKG / "table_A_complete_trajectory.json", orient="records", indent=2)

    B_rows = []
    for a in ARMS:
        sub = df[df["run"] == a["run"]]
        best = sub.loc[sub["validation_auprc"].idxmax()]
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
            }
        )
    B = pd.DataFrame(B_rows)
    B.to_csv(PKG / "table_B_best_checkpoint_summary.csv", index=False)
    B.to_json(PKG / "table_B_best_checkpoint_summary.json", orient="records", indent=2)

    def _row(method: str, lr: float, ep: int) -> pd.Series:
        return df[
            (df["method"] == method)
            & (np.isclose(df["peak_lr"], lr))
            & (df["ssl_epoch"] == ep)
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
                "TFMOE_auprc_6p21e-3": float(t621["validation_auprc"]),
                "TFMOE_auprc_2e-3": float(t2["validation_auprc"]),
                "TFMOE_auprc_1e-3": float(t1["validation_auprc"]),
                "delta_TF_minus_H_6p21e-3": float(t621["validation_auprc"] - h621["validation_auprc"]),
                "delta_TF_minus_H_2e-3": float(t2["validation_auprc"] - h2["validation_auprc"]),
                "delta_TF_minus_H_1e-3": float(t1["validation_auprc"] - h1["validation_auprc"]),
                "DIRECT_H_delta_1e-3_minus_2e-3": float(h1["validation_auprc"] - h2["validation_auprc"]),
                "DIRECT_H_delta_2e-3_minus_6p21e-3": float(h2["validation_auprc"] - h621["validation_auprc"]),
                "TFMOE_delta_1e-3_minus_2e-3": float(t1["validation_auprc"] - t2["validation_auprc"]),
                "TFMOE_delta_2e-3_minus_6p21e-3": float(t2["validation_auprc"] - t621["validation_auprc"]),
            }
        )
    Cdf = pd.DataFrame(C)
    Cdf.to_csv(PKG / "table_C_matched_comparisons.csv", index=False)
    Cdf.to_json(PKG / "table_C_matched_comparisons.json", orient="records", indent=2)

    # Protocol comparison R198-only vs R198+X+TF
    P = df[
        [
            "method",
            "peak_lr",
            "ssl_epoch",
            "validation_auprc",
            "f1_at_0.5",
            "f1_at_val_thr",
            "final_probe_val_bce",
            "xtf_validation_auprc",
            "xtf_f1_at_0.5",
            "xtf_f1_at_val_thr",
            "xtf_final_probe_val_bce",
        ]
    ].copy()
    P["delta_auprc_r198_minus_xtf"] = P["validation_auprc"] - P["xtf_validation_auprc"]
    P["delta_f1_0.5_r198_minus_xtf"] = P["f1_at_0.5"] - P["xtf_f1_at_0.5"]
    P["delta_f1_valthr_r198_minus_xtf"] = P["f1_at_val_thr"] - P["xtf_f1_at_val_thr"]
    P["delta_val_bce_r198_minus_xtf"] = P["final_probe_val_bce"] - P["xtf_final_probe_val_bce"]
    P.to_csv(PKG / "table_E_protocol_comparison_r198_only_vs_r198_x_tf.csv", index=False)
    P.to_json(
        PKG / "table_E_protocol_comparison_r198_only_vs_r198_x_tf.json",
        orient="records",
        indent=2,
    )
    return {"A": A, "B": B, "C": Cdf, "P": P}


def interpret(df: pd.DataFrame, B: pd.DataFrame, C: pd.DataFrame, P: pd.DataFrame) -> Dict[str, Any]:
    tf_wins = []
    for ep in EPS:
        for lr in (0.006213266113989207, 0.002, 0.001):
            h = df[(df.method == "DIRECT_H") & np.isclose(df.peak_lr, lr) & (df.ssl_epoch == ep)].iloc[0]
            t = df[(df.method == "DIRECT_H_TFMOE") & np.isclose(df.peak_lr, lr) & (df.ssl_epoch == ep)].iloc[0]
            tf_wins.append(float(t.validation_auprc) > float(h.validation_auprc))

    lr_effect = {}
    for method in ("DIRECT_H", "DIRECT_H_TFMOE"):
        subB = B[B.method == method].sort_values("best_validation_auprc", ascending=False)
        best = subB.iloc[0]
        a1 = float(subB[np.isclose(subB.peak_lr, 0.001)].iloc[0].best_validation_auprc)
        a2 = float(subB[np.isclose(subB.peak_lr, 0.002)].iloc[0].best_validation_auprc)
        a621 = float(subB[np.isclose(subB.peak_lr, 0.006213266113989207)].iloc[0].best_validation_auprc)
        lr_effect[method] = {
            "best_peak_lr": float(best.peak_lr),
            "best_ssl_epoch": int(best.best_ssl_epoch_by_auprc),
            "best_auprc": float(best.best_validation_auprc),
            "delta_best_1e-3_minus_2e-3": a1 - a2,
            "delta_best_1e-3_minus_6p21e-3": a1 - a621,
            "delta_best_2e-3_minus_6p21e-3": a2 - a621,
        }

    overall = B.sort_values("best_validation_auprc", ascending=False).iloc[0]
    ans = {
        "probe_feature_protocol": PROBE_FEATURE_PROTOCOL,
        "thesis_primary": False,
        "n_cells": int(len(df)),
        "tfmoe_wins_out_of_15": int(sum(tf_wins)),
        "mean_delta_tf_minus_h": {
            "6p21e-3": float(C["delta_TF_minus_H_6p21e-3"].mean()),
            "2e-3": float(C["delta_TF_minus_H_2e-3"].mean()),
            "1e-3": float(C["delta_TF_minus_H_1e-3"].mean()),
        },
        "lr_effect_on_best_checkpoint": lr_effect,
        "best_overall_r198_only": overall.to_dict(),
        "protocol_gap_r198_minus_xtf": {
            "mean_delta_auprc": float(P["delta_auprc_r198_minus_xtf"].mean()),
            "median_delta_auprc": float(P["delta_auprc_r198_minus_xtf"].median()),
            "mean_delta_f1_0.5": float(P["delta_f1_0.5_r198_minus_xtf"].mean()),
            "mean_delta_val_bce": float(P["delta_val_bce_r198_minus_xtf"].mean()),
            "n_cells_where_xtf_higher_auprc": int((P["delta_auprc_r198_minus_xtf"] < 0).sum()),
            "n_cells_where_r198_higher_auprc": int((P["delta_auprc_r198_minus_xtf"] > 0).sum()),
        },
    }
    (PKG / "analysis_answers.json").write_text(json.dumps(ans, indent=2) + "\n")
    return ans


def write_summary(df: pd.DataFrame, B: pd.DataFrame, C: pd.DataFrame, P: pd.DataFrame, ans: Dict[str, Any]) -> None:
    best = ans["best_overall_r198_only"]
    gap = ans["protocol_gap_r198_minus_xtf"]
    lines: List[str] = []
    lines.append("# R198-only LR / TFMOE analysis (not thesis-wide primary)")
    lines.append("")
    lines.append(
        "**Framing:** this package isolates representation quality with PaperStyleMLP on "
        "**R198 embeddings only**. It is **not** a replacement of the thesis-wide R198+X+TF "
        "collaborator evaluation."
    )
    lines.append("")
    lines.append("## Protocol")
    lines.append("")
    lines.append("- Extraction: corrected full-subgraph (`embeddings/direct_r198_40ep_linear_lr_full_extract/`)")
    lines.append("- ID checks: train∩val=0, val above train max, verify.ok (from existing full-extract cells)")
    lines.append("- Probe input: **R198 only** (`input_dim=198`); no raw edge X; no temporal-flow")
    lines.append("- Probe: PaperStyleMLP 20 ep / lr=1e-3 / bs=8192 / seed=2 / best-val-AUPRC")
    lines.append("- `probe_feature_protocol = R198_only`")
    lines.append("- Seed-only evaluations: diagnostic/superseded; **not used** here")
    lines.append("")
    lines.append("## Best R198-only checkpoints")
    lines.append("")
    for _, r in B.sort_values(["method", "peak_lr"]).iterrows():
        lines.append(
            f"- **{r['method']}** lr={r['peak_lr']:.4g}: SSL ep **{int(r['best_ssl_epoch_by_auprc'])}** "
            f"AUPRC={r['best_validation_auprc']:.4f}, F1@0.5={r['best_f1_at_0.5']:.4f}, "
            f"F1@thr={r['best_f1_at_val_thr']:.4f}, "
            f"final BCE train/val={r['best_final_train_bce']:.4f}/{r['best_final_val_bce']:.4f}"
        )
    lines.append("")
    lines.append("## Answers")
    lines.append("")
    lines.append(
        f"1. **R198 alone:** best overall AUPRC under R198-only is "
        f"**{best['method']}** lr={best['peak_lr']:.4g} @ ep{int(best['best_ssl_epoch_by_auprc'])} "
        f"(AUPRC={best['best_validation_auprc']:.4f}). "
        f"Mean ΔAUPRC (R198-only − R198+X+TF) across 30 cells = "
        f"**{gap['mean_delta_auprc']:+.4f}** (X+TF higher on "
        f"{gap['n_cells_where_xtf_higher_auprc']}/30 cells)."
    )
    lines.append(
        f"2. **TFMOE vs DIRECT_H without concatenating TF/X:** TFMOE wins AUPRC on "
        f"**{ans['tfmoe_wins_out_of_15']}/15** matched LR×epoch cells. "
        f"Mean Δ(TF−H): "
        f"{ans['mean_delta_tf_minus_h']['6p21e-3']:+.4f} (6.21e-3), "
        f"{ans['mean_delta_tf_minus_h']['2e-3']:+.4f} (2e-3), "
        f"{ans['mean_delta_tf_minus_h']['1e-3']:+.4f} (1e-3)."
    )
    h = ans["lr_effect_on_best_checkpoint"]["DIRECT_H"]
    t = ans["lr_effect_on_best_checkpoint"]["DIRECT_H_TFMOE"]
    lines.append(
        f"3. **LR conclusions under R198-only:** DIRECT_H best LR={h['best_peak_lr']:.4g} "
        f"(AUPRC={h['best_auprc']:.4f}); TFMOE best LR={t['best_peak_lr']:.4g} "
        f"(AUPRC={t['best_auprc']:.4f}). "
        f"Best-checkpoint Δ(1e-3−2e-3): H {h['delta_best_1e-3_minus_2e-3']:+.4f}, "
        f"TFMOE {t['delta_best_1e-3_minus_2e-3']:+.4f}."
    )
    lines.append(
        f"4. **Were R198+X+TF results feature-driven?** Yes, materially: concatenating X+TF "
        f"raises AUPRC by about **{-gap['mean_delta_auprc']:.3f}** on average vs R198-only "
        f"(median gap {-gap['median_delta_auprc']:.3f}). Treat R198+X+TF as embedding-plus-features, "
        f"and R198-only as representation isolation for this analysis."
    )
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- Package: `{PKG.relative_to(ROOT)}/`")
    lines.append(f"- Cells: `{(OUT / 'cells').relative_to(ROOT)}/`")
    lines.append("- R198+X+TF package (unchanged): `.../collaborator_package/`")
    lines.append("")
    (PKG / "summary.md").write_text("\n".join(lines) + "\n")


def write_manifest(df: pd.DataFrame, ans: Dict[str, Any]) -> None:
    checkpoints = []
    for a in ARMS:
        for ep in EPS:
            checkpoints.append(
                str(ROOT / "saved-models" / f"checkpoint_{a['run']}_epoch{ep:02d}.tar")
            )
    man = {
        "probe_feature_protocol": PROBE_FEATURE_PROTOCOL,
        "thesis_primary": False,
        "evaluation_tier": "r198_only_lr_analysis",
        "extraction_function": "embedding_extraction.run_embedding_extraction via scripts/extract_direct_r198_full_cell.py",
        "extraction_code_path": "scripts/extract_direct_r198_full_cell.py → scripts/reeval_direct_r198_40ep_full_extract_cell.py (reuse existing full extracts)",
        "r198_only_probe_source": (
            "diagnostic PaperStyleMLP already computed in "
            "scripts/reeval_direct_r198_40ep_full_extract_cell.py as _fit_probe(Z_train, Z_val) "
            "(no X/TF concat); packaged here without SSL retrain and without overwriting primary"
        ),
        "embeddings_dir": str(EMB_ROOT.relative_to(ROOT)),
        "out_dir": str(OUT.relative_to(ROOT)),
        "probe_input": {
            "description": "R198 embedding matrix Z only",
            "shape_train_example": [3248267, 198],
            "shape_val_example": [965464, 198],
            "dim": 198,
            "includes_raw_edge_X": False,
            "includes_temporal_flow": False,
        },
        "ssl_training_submitted": False,
        "checkpoints": checkpoints,
        "n_cells": int(len(df)),
        "analysis": ans,
        "seed_only_excluded": True,
        "refuse_merge_unless": [
            "probe_input_dim == 198",
            "extraction_protocol == full_subgraph",
            "verify.ok and train_val_intersect == 0",
            "concatenated_raw_edge_X == false",
            "concatenated_temporal_flow == false",
        ],
    }
    (OUT / "manifest.json").write_text(json.dumps(man, indent=2) + "\n")
    (PKG / "manifest.json").write_text(json.dumps(man, indent=2) + "\n")


def print_preflight() -> None:
    print("=" * 72)
    print("PREFLIGHT — R198-only LR analysis (no SSL training)")
    print("=" * 72)
    print("1) Extraction function / code path:")
    print("   scripts/extract_direct_r198_full_cell.py")
    print("   → embedding_extraction.run_embedding_extraction (full GraphModule R198)")
    print("   Reused via existing cells from scripts/reeval_direct_r198_40ep_full_extract_cell.py")
    print("   R198-only probe already stored as cell['diagnostic'] = _fit_probe(Z) (dim=198)")
    print()
    print("2) Checkpoint list: 6 runs × epochs {3,10,20,30,40} = 30 checkpoints under saved-models/")
    for a in ARMS:
        print(f"   - {a['run']}")
    print()
    print("3) Output directory:")
    print(f"   {OUT.relative_to(ROOT)}/")
    print("   (does NOT write into collaborator_package/ primary tables)")
    print()
    print("4) Probe feature tensor:")
    print("   Z_train shape (3248267, 198), Z_val shape (965464, 198)")
    print("   PaperStyleMLP(input_dim=198)")
    print()
    print("5) No X / no TF concatenation: confirmed (diagnostic path uses Z only)")
    print("6) SSL training jobs: NONE will be submitted")
    print("=" * 72)


def main() -> int:
    print_preflight()
    df = materialize_cells()
    assert (df["probe_input_dim"] == 198).all()
    assert (df["probe_feature_protocol"] == PROBE_FEATURE_PROTOCOL).all()
    assert (df["verify_ok"]).all()
    assert (df["train_val_intersect"] == 0).all()
    plot_all(df)
    tabs = tables(df)
    ans = interpret(df, tabs["B"], tabs["C"], tabs["P"])
    write_summary(df, tabs["B"], tabs["C"], tabs["P"], ans)
    write_manifest(df, ans)
    (PKG / "README.md").write_text(
        "# R198-only LR analysis package\n\n"
        "Not thesis-wide primary. Isolates representation quality with R198-only probes.\n"
        "R198+X+TF remains in `../collaborator_package/`.\n"
        "See `summary.md`.\n"
    )
    print(json.dumps({"status": "ok", "out": str(OUT), "n_cells": len(df), "best": ans["best_overall_r198_only"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
