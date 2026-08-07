#!/usr/bin/env python3
"""Aggregate corrected 40ep full-subgraph re-eval; mark seed-only metrics INVALID.

Reads cells under results/diagnostics/direct_r198_40ep_linear_lr_full_extract_reeval/.
Does not overwrite results/diagnostics/direct_r198_tfmoe_40ep_linear_lr_sweep/.
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
FIG = OUT / "figures"
NOTES = ROOT / "notes/direct_r198_40ep_linear_lr_full_extract_reeval.md"
SEED_ONLY_NOTE = ROOT / "notes/direct_r198_tfmoe_40ep_linear_lr_sweep.md"
SEED_ONLY_DIR = ROOT / "results/diagnostics/direct_r198_tfmoe_40ep_linear_lr_sweep"

ARMS = [
    {
        "run": "direct_r198_infonce_40ep_seed2_linear_lr6p2e-3",
        "arm": "DIRECT_H",
        "peak_lr": 0.006213266113989207,
        "label": "DIRECT_H",
        "ls": "-",
    },
    {
        "run": "direct_r198_tfmoe_40ep_seed2_linear_lr6p2e-3",
        "arm": "DIRECT_H_TFMOE",
        "peak_lr": 0.006213266113989207,
        "label": "DIRECT_H_TFMOE",
        "ls": "-",
    },
    {
        "run": "direct_r198_infonce_40ep_seed2_linear_lr2e-3",
        "arm": "DIRECT_H",
        "peak_lr": 0.002,
        "label": "DIRECT_H",
        "ls": "--",
    },
    {
        "run": "direct_r198_tfmoe_40ep_seed2_linear_lr2e-3",
        "arm": "DIRECT_H_TFMOE",
        "peak_lr": 0.002,
        "label": "DIRECT_H_TFMOE",
        "ls": "--",
    },
    {
        "run": "direct_r198_infonce_40ep_seed2_linear_lr1e-3",
        "arm": "DIRECT_H",
        "peak_lr": 0.001,
        "label": "DIRECT_H",
        "ls": ":",
    },
    {
        "run": "direct_r198_tfmoe_40ep_seed2_linear_lr1e-3",
        "arm": "DIRECT_H_TFMOE",
        "peak_lr": 0.001,
        "label": "DIRECT_H_TFMOE",
        "ls": ":",
    },
]
EVAL_EPS = [3, 10, 20, 30, 40]
C_H = "#0072B2"
C_T = "#D55E00"

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


def _color(arm: str) -> str:
    return C_T if "TFMOE" in arm else C_H


def _primary(cell: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if cell.get("status") != "ok":
        return None
    return cell.get("primary")


def collect_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for a in ARMS:
        for ep in EVAL_EPS:
            cell = _load_cell(a["run"], ep)
            if cell is None:
                rows.append(
                    {
                        "run": a["run"],
                        "arm": a["arm"],
                        "peak_lr": a["peak_lr"],
                        "epoch": ep,
                        "status": "missing",
                    }
                )
                continue
            prim = _primary(cell)
            verify = cell.get("verify") or {}
            if prim is None:
                rows.append(
                    {
                        "run": a["run"],
                        "arm": a["arm"],
                        "peak_lr": a["peak_lr"],
                        "epoch": ep,
                        "status": cell.get("status", "failed"),
                        "verify_ok": verify.get("ok"),
                        "cell_path": str(OUT / "cells" / a["run"] / f"epoch_{ep:02d}.json"),
                    }
                )
                continue
            m05 = prim["validation_metrics_at_0.5"]
            mopt = prim["validation_metrics_at_val_optimal_f1"]
            rows.append(
                {
                    "run": a["run"],
                    "arm": a["arm"],
                    "peak_lr": a["peak_lr"],
                    "epoch": ep,
                    "status": "ok",
                    "verify_ok": verify.get("ok"),
                    "n_val": verify.get("n_val"),
                    "train_val_intersect": verify.get("train_val_intersect"),
                    "val_auprc": prim["validation_auprc"],
                    "f1_at_0.5": m05["f1"],
                    "f1_at_val_thr": mopt["f1"],
                    "val_thr": mopt["threshold"],
                    "final_probe_train_bce": prim["final_probe_train_bce"],
                    "final_probe_val_bce": prim["final_probe_val_bce"],
                    "best_probe_epoch": prim["best_probe_epoch"],
                    "embedding_dir": cell.get("embedding_dir"),
                    "cell_path": str(OUT / "cells" / a["run"] / f"epoch_{ep:02d}.json"),
                    "verify_path": str(
                        OUT / "cells" / a["run"] / f"epoch_{ep:02d}_verify.json"
                    ),
                }
            )
    return rows


def _series_on_grid(df: pd.DataFrame, run: str, col: str) -> np.ndarray:
    """Values on the shared EVAL_EPS grid; NaN where missing (breaks line connections)."""
    y = []
    for ep in EVAL_EPS:
        rows = df[(df["run"] == run) & (df["epoch"] == ep) & (df["status"] == "ok")]
        if rows.empty or col not in rows.columns or pd.isna(rows.iloc[0][col]):
            y.append(np.nan)
        else:
            y.append(float(rows.iloc[0][col]))
    return np.asarray(y, dtype=float)


def _load_steps(run: str) -> Optional[pd.DataFrame]:
    p = ROOT / "results/diagnostics" / run / "logs" / "steps.jsonl"
    if not p.is_file():
        return None
    rows = [json.loads(l) for l in p.open() if l.strip()]
    return pd.DataFrame(rows) if rows else None


def plot_curves(df: pd.DataFrame) -> None:
    """Matched-epoch trajectories; never connect across missing checkpoints."""
    xs = np.asarray(EVAL_EPS, dtype=float)

    def _one(col: str, ylabel: str, stem: str, title: str) -> None:
        fig, ax = plt.subplots(figsize=(8.0, 4.2))
        any_line = False
        for a in ARMS:
            ys = _series_on_grid(df, a["run"], col)
            if np.all(np.isnan(ys)):
                continue
            any_line = True
            ax.plot(
                xs,
                ys,
                color=_color(a["arm"]),
                ls=a["ls"],
                marker="o",
                lw=1.6,
                label=f"{a['label']} lr={a['peak_lr']:.4g}",
            )
        if not any_line:
            plt.close(fig)
            return
        ax.set_xticks(EVAL_EPS)
        ax.set_xlabel("SSL epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8)
        fig.tight_layout()
        _save(fig, stem)

    _one(
        "val_auprc",
        "Validation AUPRC",
        "01_val_auprc_vs_epoch",
        "Matched checkpoints — full-subgraph + PaperStyleMLP",
    )
    _one(
        "f1_at_0.5",
        "Validation F1@0.5",
        "02_val_f1_at_0.5_vs_epoch",
        "Matched checkpoints — F1@0.5",
    )
    _one(
        "f1_at_val_thr",
        "Validation F1@val-threshold",
        "03_val_f1_at_val_thr_vs_epoch",
        "Matched checkpoints — F1@validation-selected threshold",
    )
    _one(
        "final_probe_val_bce",
        "Final probe-epoch val BCE",
        "04_final_probe_val_bce_vs_epoch",
        "Matched checkpoints — last MLP-epoch validation BCE",
    )

    # Combined: training InfoNCE + LR schedule + downstream AUPRC
    fig, axes = plt.subplots(3, 1, figsize=(9.0, 9.0), sharex=False)
    for a in ARMS:
        steps = _load_steps(a["run"])
        if steps is None or steps.empty:
            continue
        c, ls = _color(a["arm"]), a["ls"]
        lab = f"{a['label']} lr={a['peak_lr']:.4g}"
        if "L_contrast_raw" in steps.columns:
            axes[0].plot(
                steps["optimizer_step_index"],
                steps["L_contrast_raw"],
                color=c,
                ls=ls,
                lw=1.2,
                label=lab,
            )
        lr_col = "encoder_lr" if "encoder_lr" in steps.columns else (
            "lr" if "lr" in steps.columns else None
        )
        if lr_col is not None:
            axes[1].plot(
                steps["optimizer_step_index"],
                steps[lr_col],
                color=c,
                ls=ls,
                lw=1.2,
                label=lab,
            )
        ys = _series_on_grid(df, a["run"], "val_auprc")
        if not np.all(np.isnan(ys)):
            axes[2].plot(xs, ys, color=c, ls=ls, marker="o", lw=1.6, label=lab)
    axes[0].set_ylabel("Raw InfoNCE")
    axes[0].set_title("Training loss (full 40ep)")
    axes[0].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=7)
    axes[1].set_ylabel("LR")
    axes[1].set_title("Learning-rate schedule")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=7)
    axes[2].set_xticks(EVAL_EPS)
    axes[2].set_xlabel("SSL epoch")
    axes[2].set_ylabel("Val AUPRC")
    axes[2].set_title("Downstream (matched checkpoints)")
    axes[2].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=7)
    fig.tight_layout()
    _save(fig, "05_train_loss_lr_and_downstream_auprc")


def trajectory_answers(df: pd.DataFrame) -> Dict[str, Any]:
    """Peak / LR / length / TFMOE consistency from completed matched cells only."""
    ok = df[df["status"] == "ok"].copy()
    peaks = []
    for a in ARMS:
        sub = ok[ok["run"] == a["run"]]
        if sub.empty:
            peaks.append({"run": a["run"], "arm": a["arm"], "peak_lr": a["peak_lr"], "status": "no_cells"})
            continue
        best = sub.loc[sub["val_auprc"].idxmax()]
        peaks.append(
            {
                "run": a["run"],
                "arm": a["arm"],
                "peak_lr": a["peak_lr"],
                "peak_ssl_epoch": int(best["epoch"]),
                "peak_val_auprc": float(best["val_auprc"]),
                "peak_f1_0.5": float(best["f1_at_0.5"]),
                "peak_f1_val_thr": float(best["f1_at_val_thr"]),
                "n_epochs_observed": int(len(sub)),
                "full_grid": bool(set(sub["epoch"].astype(int)) >= set(EVAL_EPS)),
            }
        )

    # Lower vs higher LR at shared epochs (exact peak LRs)
    lr_cmp = []
    for arm in ("DIRECT_H", "DIRECT_H_TFMOE"):
        for ep in EVAL_EPS:
            hi = ok[(ok["arm"] == arm) & (ok["epoch"] == ep) & np.isclose(ok["peak_lr"], 0.006213266113989207)]
            mid = ok[(ok["arm"] == arm) & (ok["epoch"] == ep) & np.isclose(ok["peak_lr"], 0.002)]
            lo = ok[(ok["arm"] == arm) & (ok["epoch"] == ep) & np.isclose(ok["peak_lr"], 0.001)]
            if hi.empty or mid.empty:
                continue
            row = {
                "arm": arm,
                "epoch": ep,
                "auprc_6p21e-3": float(hi.iloc[0]["val_auprc"]),
                "auprc_2e-3": float(mid.iloc[0]["val_auprc"]),
                "delta_6p21e-3_minus_2e-3": float(hi.iloc[0]["val_auprc"]) - float(mid.iloc[0]["val_auprc"]),
            }
            if not lo.empty:
                row["auprc_1e-3"] = float(lo.iloc[0]["val_auprc"])
                row["delta_1e-3_minus_2e-3"] = float(lo.iloc[0]["val_auprc"]) - float(mid.iloc[0]["val_auprc"])
            lr_cmp.append(row)

    tf_vs_h = []
    for peak in (0.006213266113989207, 0.002, 0.001):
        for ep in EVAL_EPS:
            h = ok[
                (ok["arm"] == "DIRECT_H")
                & (ok["epoch"] == ep)
                & (np.isclose(ok["peak_lr"], peak))
            ]
            t = ok[
                (ok["arm"] == "DIRECT_H_TFMOE")
                & (ok["epoch"] == ep)
                & (np.isclose(ok["peak_lr"], peak))
            ]
            if h.empty or t.empty:
                continue
            tf_vs_h.append(
                {
                    "peak_lr": peak,
                    "epoch": ep,
                    "auprc_h": float(h.iloc[0]["val_auprc"]),
                    "auprc_tf": float(t.iloc[0]["val_auprc"]),
                    "delta_tf_minus_h": float(t.iloc[0]["val_auprc"]) - float(h.iloc[0]["val_auprc"]),
                    "tf_wins": bool(t.iloc[0]["val_auprc"] > h.iloc[0]["val_auprc"]),
                }
            )

    longer = []
    for a in ARMS:
        sub = ok[ok["run"] == a["run"]].sort_values("epoch")
        if len(sub) < 2:
            continue
        first, last = sub.iloc[0], sub.iloc[-1]
        longer.append(
            {
                "run": a["run"],
                "arm": a["arm"],
                "peak_lr": a["peak_lr"],
                "first_epoch": int(first["epoch"]),
                "last_epoch": int(last["epoch"]),
                "auprc_first": float(first["val_auprc"]),
                "auprc_last": float(last["val_auprc"]),
                "delta_last_minus_first": float(last["val_auprc"]) - float(first["val_auprc"]),
            }
        )

    n_tf_win = sum(1 for r in tf_vs_h if r["tf_wins"])
    return {
        "peaks": peaks,
        "lr_comparison_matched_epochs": lr_cmp,
        "tfmoe_vs_h_matched_epochs": tf_vs_h,
        "tfmoe_wins_count": n_tf_win,
        "tfmoe_comparisons_count": len(tf_vs_h),
        "longer_training": longer,
        "grid_complete": bool((df["status"] == "ok").sum() == len(ARMS) * len(EVAL_EPS)),
    }


def _fmt(x: Any, nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "—"


def build_tables(df: pd.DataFrame) -> Dict[str, Any]:
    ok = df[df["status"] == "ok"].copy()
    lr_rows = []
    for arm in ("DIRECT_H", "DIRECT_H_TFMOE"):
        for ep in EVAL_EPS:
            hi = ok[
                (ok["arm"] == arm)
                & (ok["epoch"] == ep)
                & np.isclose(ok["peak_lr"], 0.006213266113989207)
            ]
            mid = ok[
                (ok["arm"] == arm)
                & (ok["epoch"] == ep)
                & np.isclose(ok["peak_lr"], 0.002)
            ]
            lo = ok[
                (ok["arm"] == arm)
                & (ok["epoch"] == ep)
                & np.isclose(ok["peak_lr"], 0.001)
            ]
            if hi.empty and mid.empty and lo.empty:
                continue
            hr = hi.iloc[0].to_dict() if not hi.empty else {}
            mr = mid.iloc[0].to_dict() if not mid.empty else {}
            lr = lo.iloc[0].to_dict() if not lo.empty else {}
            row = {
                "arm": arm,
                "epoch": ep,
                "auprc_lr6p2e-3": hr.get("val_auprc"),
                "auprc_lr2e-3": mr.get("val_auprc"),
                "auprc_lr1e-3": lr.get("val_auprc"),
                "f1_0.5_lr6p2e-3": hr.get("f1_at_0.5"),
                "f1_0.5_lr2e-3": mr.get("f1_at_0.5"),
                "f1_0.5_lr1e-3": lr.get("f1_at_0.5"),
                "f1_valthr_lr6p2e-3": hr.get("f1_at_val_thr"),
                "f1_valthr_lr2e-3": mr.get("f1_at_val_thr"),
                "f1_valthr_lr1e-3": lr.get("f1_at_val_thr"),
            }
            if hr.get("val_auprc") is not None and mr.get("val_auprc") is not None:
                row["delta_auprc_6p2e-3_minus_2e-3"] = float(hr["val_auprc"]) - float(mr["val_auprc"])
            if lr.get("val_auprc") is not None and mr.get("val_auprc") is not None:
                row["delta_auprc_1e-3_minus_2e-3"] = float(lr["val_auprc"]) - float(mr["val_auprc"])
            lr_rows.append(row)

    hvst = []
    for peak in (0.006213266113989207, 0.002, 0.001):
        for ep in EVAL_EPS:
            h = ok[
                (ok["arm"] == "DIRECT_H")
                & (ok["epoch"] == ep)
                & (np.isclose(ok["peak_lr"], peak))
            ]
            t = ok[
                (ok["arm"] == "DIRECT_H_TFMOE")
                & (ok["epoch"] == ep)
                & (np.isclose(ok["peak_lr"], peak))
            ]
            if h.empty and t.empty:
                continue
            hr = h.iloc[0].to_dict() if not h.empty else {}
            tr = t.iloc[0].to_dict() if not t.empty else {}
            da = None
            if hr and tr and hr.get("val_auprc") is not None and tr.get("val_auprc") is not None:
                da = float(tr["val_auprc"]) - float(hr["val_auprc"])
            hvst.append(
                {
                    "peak_lr": peak,
                    "epoch": ep,
                    "auprc_DIRECT_H": hr.get("val_auprc"),
                    "auprc_DIRECT_H_TFMOE": tr.get("val_auprc"),
                    "delta_auprc_tfmoe_minus_h": da,
                    "f1_0.5_DIRECT_H": hr.get("f1_at_0.5"),
                    "f1_0.5_DIRECT_H_TFMOE": tr.get("f1_at_0.5"),
                    "f1_valthr_DIRECT_H": hr.get("f1_at_val_thr"),
                    "f1_valthr_DIRECT_H_TFMOE": tr.get("f1_at_val_thr"),
                    "final_val_bce_DIRECT_H": hr.get("final_probe_val_bce"),
                    "final_val_bce_DIRECT_H_TFMOE": tr.get("final_probe_val_bce"),
                }
            )
    return {"lr_comparison": lr_rows, "direct_h_vs_tfmoe": hvst}


def write_note(df: pd.DataFrame, tables: Dict[str, Any], artifacts: Dict[str, str]) -> None:
    ok = df[df["status"] == "ok"]
    n_ok = int((df["status"] == "ok").sum())
    n_tot = len(df)
    lines: List[str] = []
    lines.append("# DIRECT_R198 40ep linear-LR sweep — corrected full-subgraph re-eval")
    lines.append("")
    lines.append(
        "> **INVALID (do not use):** prior seed-only validation metrics under "
        f"`{SEED_ONLY_DIR.relative_to(ROOT)}/` and `{SEED_ONLY_NOTE.relative_to(ROOT)}`. "
        "Those extracts treated loader `input_id` as a global edge index instead of "
        "`split_inds[input_id]`, so validation IDs landed in the train range "
        "(~100% train∩val overlap). Retraining was not redone; only extraction/probe."
    )
    lines.append("")
    lines.append("## Protocol (matched to 10ep full-extract analysis)")
    lines.append("")
    lines.append("- Extractor: `scripts/extract_direct_r198_full_cell.py` (full GraphModule R198)")
    lines.append(
        "- Embeddings root: `embeddings/direct_r198_40ep_linear_lr_full_extract/` "
        "(does **not** overwrite `embeddings/<run>_epochXX/` seed-only artifacts)"
    )
    lines.append(
        "- Probe: PaperStyleMLP, 20 epochs, lr=1e-3, bs=8192, seed=2; "
        "features R198+X+TF; ranking metrics from best-val-AUPRC probe epoch; "
        "also report last probe-epoch train/val BCE"
    )
    lines.append(
        "- ID gate before probe: train∩val=0, all val IDs above ref train max, "
        "no seed-only train-range signature; Jaccard≥0.999 and relative |n−n_ref|/n_ref≤1% "
        "vs prior full extract (exact set equality not required)"
    )
    lines.append(f"- Cells complete: **{n_ok}/{n_tot}**")
    lines.append("- Matched SSL epoch grid: **3, 10, 20, 30, 40** (plots break on missing points)")
    lines.append("")
    lines.append("## Primary results (R198+X+TF → PaperStyleMLP)")
    lines.append("")
    lines.append(
        "| Arm | Peak LR | SSL ep | Val AUPRC | F1@0.5 | F1@val-thr | "
        "Final train BCE | Final val BCE | Verify |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for _, r in df.sort_values(["arm", "peak_lr", "epoch"]).iterrows():
        if r["status"] != "ok":
            lines.append(
                f"| {r['arm']} | {r['peak_lr']:.4g} | {int(r['epoch'])} | — | — | — | — | — | "
                f"{r['status']} |"
            )
            continue
        lines.append(
            f"| {r['arm']} | {r['peak_lr']:.4g} | {int(r['epoch'])} | "
            f"{_fmt(r['val_auprc'])} | {_fmt(r['f1_at_0.5'])} | {_fmt(r['f1_at_val_thr'])} | "
            f"{_fmt(r['final_probe_train_bce'])} | {_fmt(r['final_probe_val_bce'])} | "
            f"{'ok' if r.get('verify_ok') else 'FAIL'} |"
        )
    lines.append("")
    traj = tables.get("trajectory_answers") or {}
    if traj:
        lines.append("## Trajectory answers (corrected full-extract only)")
        lines.append("")
        lines.append(f"- Grid complete: **{traj.get('grid_complete')}**")
        lines.append("- Per-run peaks (by val AUPRC):")
        for p in traj.get("peaks") or []:
            if p.get("status") == "no_cells":
                lines.append(f"  - {p['arm']} lr={p['peak_lr']:.4g}: no cells yet")
            else:
                lines.append(
                    f"  - {p['arm']} lr={p['peak_lr']:.4g}: SSL epoch **{p['peak_ssl_epoch']}** "
                    f"(AUPRC={_fmt(p['peak_val_auprc'])}, F1@0.5={_fmt(p['peak_f1_0.5'])}, "
                    f"F1@val-thr={_fmt(p['peak_f1_val_thr'])}; "
                    f"{p['n_epochs_observed']}/5 epochs observed)"
                )
        tf_n = traj.get("tfmoe_comparisons_count") or 0
        tf_w = traj.get("tfmoe_wins_count") or 0
        if tf_n:
            lines.append(
                f"- TFMOE beats DIRECT_H on AUPRC at **{tf_w}/{tf_n}** matched "
                "(arm×LR×epoch) checkpoints with both present"
            )
        lines.append("")
    lines.append("## Learning-rate comparison (corrected only)")
    lines.append("")
    lines.append(
        "| Arm | SSL ep | AUPRC 6.21e-3 | AUPRC 2e-3 | AUPRC 1e-3 | "
        "Δ(1e-3−2e-3) | F1@0.5 2e-3 | F1@0.5 1e-3 |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in tables["lr_comparison"]:
        lines.append(
            f"| {row['arm']} | {row['epoch']} | {_fmt(row['auprc_lr6p2e-3'])} | "
            f"{_fmt(row['auprc_lr2e-3'])} | {_fmt(row.get('auprc_lr1e-3'))} | "
            f"{_fmt(row.get('delta_auprc_1e-3_minus_2e-3'))} | "
            f"{_fmt(row['f1_0.5_lr2e-3'])} | {_fmt(row.get('f1_0.5_lr1e-3'))} |"
        )
    lines.append("")
    lines.append("## DIRECT_H vs DIRECT_H_TFMOE (corrected only)")
    lines.append("")
    lines.append(
        "| Peak LR | SSL ep | AUPRC H | AUPRC TFMOE | Δ (TF−H) | "
        "F1@0.5 H | F1@0.5 TF | Final val BCE H | Final val BCE TF |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in tables["direct_h_vs_tfmoe"]:
        lines.append(
            f"| {row['peak_lr']:.4g} | {row['epoch']} | {_fmt(row['auprc_DIRECT_H'])} | "
            f"{_fmt(row['auprc_DIRECT_H_TFMOE'])} | {_fmt(row['delta_auprc_tfmoe_minus_h'])} | "
            f"{_fmt(row['f1_0.5_DIRECT_H'])} | {_fmt(row['f1_0.5_DIRECT_H_TFMOE'])} | "
            f"{_fmt(row['final_val_bce_DIRECT_H'])} | {_fmt(row['final_val_bce_DIRECT_H_TFMOE'])} |"
        )
    lines.append("")
    if not ok.empty:
        best = ok.loc[ok["val_auprc"].idxmax()]
        lines.append("## Best corrected cell")
        lines.append("")
        lines.append(
            f"- **{best['arm']}** peak_lr={best['peak_lr']:.4g} SSL epoch {int(best['epoch'])}: "
            f"AUPRC={_fmt(best['val_auprc'])}, F1@0.5={_fmt(best['f1_at_0.5'])}, "
            f"F1@val-thr={_fmt(best['f1_at_val_thr'])}, "
            f"final val BCE={_fmt(best['final_probe_val_bce'])}"
        )
        lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    for k, v in artifacts.items():
        lines.append(f"- **{k}:** `{v}`")
    lines.append("")
    lines.append("## Proposed seed-only ID fix (not used for these numbers)")
    lines.append("")
    lines.append(
        "Smallest fix: in `scripts/extract_direct_r198_seed_only_cell.py` "
        "`extract_split_seed_only`, resolve "
        "`batch_edge_inds = split_inds[input_id]` then "
        "`edge_attr[batch_edge_inds, 0]` / `y[batch_edge_inds]` "
        "(same as full extract). Do **not** change training's "
        "`get_hetero_seed_edge_ids` without auditing train loaders "
        "(all-edge seeds). Regression: `tests/test_seed_only_val_edge_id_resolution.py`."
    )
    lines.append("")
    NOTES.write_text("\n".join(lines) + "\n")


def stamp_seed_only_invalid() -> None:
    """Banner on the old note without rewriting its body tables as corrected."""
    banner = (
        "> **INVALID — seed-only validation metrics.** "
        "Val edge IDs were resolved from raw `input_id` (train-range leakage). "
        "Do not use AUPRC/F1/BCE from this note for LR or DIRECT_H-vs-TFMOE claims. "
        "Corrected full-subgraph re-eval: "
        "`notes/direct_r198_40ep_linear_lr_full_extract_reeval.md` and "
        "`results/diagnostics/direct_r198_40ep_linear_lr_full_extract_reeval/`.\n\n"
    )
    if SEED_ONLY_NOTE.is_file():
        text = SEED_ONLY_NOTE.read_text()
        if "INVALID — seed-only validation metrics" not in text:
            lines = text.splitlines(keepends=True)
            if lines and lines[0].startswith("#"):
                SEED_ONLY_NOTE.write_text(lines[0] + "\n" + banner + "".join(lines[1:]))
            else:
                SEED_ONLY_NOTE.write_text(banner + text)

    inv = SEED_ONLY_DIR / "SEED_ONLY_VALIDATION_METRICS_INVALID.md"
    SEED_ONLY_DIR.mkdir(parents=True, exist_ok=True)
    inv.write_text(
        "# Seed-only validation metrics are INVALID / DIAGNOSTIC-PROVISIONAL\n\n"
        "Cause (historical bug): seed-only extract indexed "
        "`loader.data.edge_attr[input_id]` instead of "
        "`loader.data.edge_attr[split_inds[input_id]]`.\n\n"
        "Even with the ID fix, seed-only R198 extract is **not** the collaborator "
        "protocol (neighborhoods differ from full-subgraph).\n\n"
        "- Outputs from `scripts/eval_direct_r198_40ep_linear_arm.py` are stamped "
        "`protocol=seed_only`, `evaluation_tier=diagnostic_provisional`, "
        "`collaborator_merge_allowed=false`.\n"
        "- Collaborator package build **refuses** any non-`full_subgraph` cell.\n\n"
        "Official path:\n"
        "- `python scripts/official_direct_r198_collaborator_eval.py ...`\n"
        "- docs: `notes/direct_r198_official_collaborator_eval.md`\n"
        "- package: `../direct_r198_40ep_linear_lr_full_extract_reeval/collaborator_package/`\n"
    )


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    rows = collect_rows()
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "corrected_cells.csv", index=False)
    tables = build_tables(df)
    traj = trajectory_answers(df)
    tables["trajectory_answers"] = traj
    (OUT / "lr_comparison_table.json").write_text(
        json.dumps(tables["lr_comparison"], indent=2) + "\n"
    )
    (OUT / "direct_h_vs_tfmoe_table.json").write_text(
        json.dumps(tables["direct_h_vs_tfmoe"], indent=2) + "\n"
    )
    (OUT / "trajectory_answers.json").write_text(json.dumps(traj, indent=2) + "\n")
    (OUT / "aggregate.json").write_text(
        json.dumps(
            {
                "n_cells": len(rows),
                "n_ok": int((df["status"] == "ok").sum()),
                "seed_only_prior_invalid": True,
                "extractor": "full_subgraph",
                "matched_epochs": EVAL_EPS,
                "rows": rows,
                "tables": tables,
                "trajectory_answers": traj,
            },
            indent=2,
        )
        + "\n"
    )
    plot_curves(df)
    stamp_seed_only_invalid()
    artifacts = {
        "out_dir": str(OUT.relative_to(ROOT)),
        "embeddings": "embeddings/direct_r198_40ep_linear_lr_full_extract/",
        "cells": str((OUT / "cells").relative_to(ROOT)),
        "csv": str((OUT / "corrected_cells.csv").relative_to(ROOT)),
        "aggregate_json": str((OUT / "aggregate.json").relative_to(ROOT)),
        "lr_table": str((OUT / "lr_comparison_table.json").relative_to(ROOT)),
        "h_vs_tf_table": str((OUT / "direct_h_vs_tfmoe_table.json").relative_to(ROOT)),
        "trajectory_answers": str((OUT / "trajectory_answers.json").relative_to(ROOT)),
        "fig_auprc": str((FIG / "01_val_auprc_vs_epoch.png").relative_to(ROOT)),
        "fig_f1_0.5": str((FIG / "02_val_f1_at_0.5_vs_epoch.png").relative_to(ROOT)),
        "fig_f1_val_thr": str((FIG / "03_val_f1_at_val_thr_vs_epoch.png").relative_to(ROOT)),
        "fig_bce": str((FIG / "04_final_probe_val_bce_vs_epoch.png").relative_to(ROOT)),
        "fig_combined": str(
            (FIG / "05_train_loss_lr_and_downstream_auprc.png").relative_to(ROOT)
        ),
        "note": str(NOTES.relative_to(ROOT)),
        "seed_only_invalid_marker": str(
            (SEED_ONLY_DIR / "SEED_ONLY_VALIDATION_METRICS_INVALID.md").relative_to(ROOT)
        ),
        "seed_only_note_bannered": str(SEED_ONLY_NOTE.relative_to(ROOT)),
        "proposed_fix": "scripts/extract_direct_r198_seed_only_cell.py (split_inds[input_id])",
        "proposed_test": "tests/test_seed_only_val_edge_id_resolution.py",
    }
    (OUT / "artifact_index.json").write_text(json.dumps(artifacts, indent=2) + "\n")
    write_note(df, tables, artifacts)
    print(json.dumps({"status": "ok", "n_ok": int((df["status"] == "ok").sum()), "artifacts": artifacts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
