#!/usr/bin/env python3
"""Build TFMOE weight-ablation comparison package (R198-only primary).

Baselines (not retrained):
  - DIRECT_H lr=2e-3 epochs 3/10/20 from r198_only_lr_analysis source cells
  - adaptive TFMOE lr=2e-3 epochs 3/10/20 from same

Ablation cells: results/diagnostics/tfmoe_weight_ablation_lr2e-3/cells/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASELINE_CELLS = (
    ROOT
    / "results/diagnostics/direct_r198_40ep_linear_lr_full_extract_reeval/cells"
)
DEFAULT_OUT = ROOT / "results/diagnostics/tfmoe_weight_ablation_lr2e-3"

EPS = [3, 10, 20]
ABLATION_ARMS = [
    {
        "label": "FIXED_BALANCED",
        "run": "direct_r198_tfmoe_wtabl_fixed_balanced_20ep_seed2_linear_lr2e-3",
        "weight_mode": "fixed_balanced",
        "color": "#E69F00",
        "marker": "o",
    },
    {
        "label": "ADAPTIVE_CONTRAST_FLOOR",
        "run": "direct_r198_tfmoe_wtabl_contrast_floor_20ep_seed2_linear_lr2e-3",
        "weight_mode": "adaptive_contrast_floor",
        "color": "#56B4E9",
        "marker": "s",
    },
    {
        "label": "EXPERT_ONLY",
        "run": "direct_r198_tfmoe_wtabl_expert_only_20ep_seed2_linear_lr2e-3",
        "weight_mode": "expert_only",
        "color": "#009E73",
        "marker": "D",
    },
    {
        "label": "FIXED_CURRENT_EARLY",
        "run": "direct_r198_tfmoe_wtabl_fixed_ep10_20ep_seed2_linear_lr2e-3",
        "weight_mode": "fixed_current_early",
        "color": "#CC79A7",
        "marker": "^",
    },
]
BASELINES = [
    {
        "label": "DIRECT_H",
        "run": "direct_r198_infonce_40ep_seed2_linear_lr2e-3",
        "weight_mode": "contrast_only",
        "source": "baseline_diagnostic",
        "color": "#0072B2",
        "marker": "v",
        "ls": "--",
    },
    {
        "label": "TFMOE_adaptive",
        "run": "direct_r198_tfmoe_40ep_seed2_linear_lr2e-3",
        "weight_mode": "adaptive",
        "source": "baseline_diagnostic",
        "color": "#D55E00",
        "marker": "P",
        "ls": "-",
    },
]

mpl.rcParams.update(
    {
        "font.size": 11,
        "figure.dpi": 140,
        "savefig.dpi": 200,
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def _load_probe(cell: Dict[str, Any]) -> Dict[str, Any]:
    # Ablation cells store R198-only under primary; baselines under diagnostic.
    if cell.get("probe_feature_protocol") in ("R198_only",) or cell.get("probe_input_dim") == 198:
        p = cell.get("primary") or cell.get("diagnostic")
    else:
        p = cell.get("diagnostic") or cell.get("primary")
    if not isinstance(p, dict):
        raise ValueError("missing probe block")
    if int(p.get("input_dim", -1)) != 198:
        raise ValueError(f"expected input_dim=198, got {p.get('input_dim')}")
    return p


def _row_from_cell(label: str, weight_mode: str, run: str, ep: int, cell: Dict[str, Any]) -> Dict[str, Any]:
    p = _load_probe(cell)
    verify = cell.get("verify") or {}
    if not verify.get("ok"):
        raise ValueError(f"verify failed for {run} ep{ep}")
    if int(verify.get("train_val_intersect", -1)) != 0:
        raise ValueError(f"train∩val != 0 for {run} ep{ep}")
    return {
        "method": label,
        "weight_mode": weight_mode,
        "run": run,
        "ssl_epoch": ep,
        "validation_auprc": p["validation_auprc"],
        "f1_at_0.5": p["validation_metrics_at_0.5"]["f1"],
        "f1_at_val_thr": p["validation_metrics_at_val_optimal_f1"]["f1"],
        "final_probe_train_bce": p["final_probe_train_bce"],
        "final_probe_val_bce": p["final_probe_val_bce"],
        "best_probe_epoch": p.get("best_probe_epoch"),
        "probe_input_dim": 198,
        "probe_feature_protocol": "R198_only",
        "extraction_protocol": "full_subgraph",
        "concatenated_raw_edge_X": False,
        "concatenated_temporal_flow": False,
    }


def _epoch_mean_from_steps(log_dir: Path, ep: int) -> Optional[Dict[str, float]]:
    steps = log_dir / "steps.jsonl"
    if not steps.is_file():
        return None
    rows = []
    with steps.open() as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if int(d.get("epoch", -1)) == int(ep):
                rows.append(d)
    if not rows:
        return None
    keys = [
        "L_contrast_raw",
        "L_tf_raw_0",
        "L_tf_raw_1",
        "L_tf_raw_2",
        "w_contrast",
        "w_tf_0",
        "w_tf_1",
        "w_tf_2",
        "sum_w_tf",
        "sum_weights",
        "L_total",
        "encoder_lr",
        "alpha_beta_lr",
    ]
    out: Dict[str, float] = {"ssl_epoch": float(ep), "n_logged_steps": float(len(rows))}
    for k in keys:
        vals = [float(r[k]) for r in rows if k in r and r[k] is not None]
        if vals:
            out[k] = sum(vals) / len(vals)
    return out


def collect(out: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for b in BASELINES:
        for ep in EPS:
            path = BASELINE_CELLS / b["run"] / f"epoch_{ep:02d}.json"
            cell = json.loads(path.read_text())
            rows.append(_row_from_cell(b["label"], b["weight_mode"], b["run"], ep, cell))
    for a in ABLATION_ARMS:
        for ep in EPS:
            path = out / "cells" / a["run"] / f"epoch_{ep:02d}.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            cell = json.loads(path.read_text())
            # Merge gates
            if int((cell.get("primary") or {}).get("input_dim", -1)) != 198:
                raise ValueError(f"{path}: primary input_dim != 198")
            if cell.get("concatenated_raw_edge_X") is True:
                raise ValueError(f"{path}: raw X concatenated")
            if cell.get("concatenated_temporal_flow") is True:
                raise ValueError(f"{path}: TF concatenated")
            rows.append(_row_from_cell(a["label"], a["weight_mode"], a["run"], ep, cell))
    return pd.DataFrame(rows)


def _series_for_methods(methods: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    series = BASELINES + ABLATION_ARMS
    if methods is None:
        return series
    keep = set(methods)
    return [s for s in series if s["label"] in keep]


def _plot_metric(
    df: pd.DataFrame,
    col: str,
    ylabel: str,
    out_path: Path,
    series: Optional[List[Dict[str, Any]]] = None,
) -> None:
    series = series if series is not None else _series_for_methods()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for s in series:
        sub = df[df["method"] == s["label"]].sort_values("ssl_epoch")
        if sub.empty:
            continue
        ax.plot(
            sub["ssl_epoch"],
            sub[col],
            label=s["label"],
            color=s["color"],
            marker=s["marker"],
            linestyle=s.get("ls", "-"),
            linewidth=2.0,
            markersize=7,
        )
    ax.set_xlabel("SSL epoch")
    ax.set_ylabel(ylabel)
    ax.set_xticks(EPS)
    ax.legend(fontsize=8, frameon=False, loc="best")
    ax.set_title(ylabel + " (R198-only)")
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"))
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def _tfmoe_log_series(series: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Runs that log TF expert losses / objective weights."""
    out: List[Dict[str, Any]] = []
    for s in series:
        if s["label"] == "DIRECT_H":
            continue
        out.append(s)
    return out


def _plot_objective_weights(fig_dir: Path, series: List[Dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for a in _tfmoe_log_series(series):
        log_dir = ROOT / "results/diagnostics" / a["run"] / "logs"
        xs, ys = [], []
        for ep in EPS:
            m = _epoch_mean_from_steps(log_dir, ep)
            if m and "w_contrast" in m:
                xs.append(ep)
                ys.append(m["w_contrast"])
        if xs:
            ax.plot(
                xs,
                ys,
                label=a["label"],
                color=a["color"],
                marker=a["marker"],
                linewidth=2.0,
                markersize=7,
            )
    ax.axhline(0.25, color="gray", linestyle=":", linewidth=1, label="floor 0.25")
    ax.set_xlabel("SSL epoch")
    ax.set_ylabel("w_contrast (epoch-mean of logged steps)")
    ax.set_xticks(EPS)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("Objective weights vs SSL epoch")
    fig.tight_layout()
    fig.savefig(fig_dir / "05_objective_weights_vs_epoch.png")
    fig.savefig(fig_dir / "05_objective_weights_vs_epoch.pdf")
    plt.close(fig)


def _plot_expert_raw_losses(fig_dir: Path, series: List[Dict[str, Any]]) -> None:
    """One panel per expert head — much easier to read than overlaid tf0/tf1/tf2."""
    expert_keys = [
        ("L_tf_raw_0", "Expert 0 (tf0)"),
        ("L_tf_raw_1", "Expert 1 (tf1)"),
        ("L_tf_raw_2", "Expert 2 (tf2)"),
    ]
    methods = _tfmoe_log_series(series)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.2), sharey=False)
    for ax, (key, title) in zip(axes, expert_keys):
        for a in methods:
            log_dir = ROOT / "results/diagnostics" / a["run"] / "logs"
            xs, ys = [], []
            for ep in EPS:
                m = _epoch_mean_from_steps(log_dir, ep)
                if m and key in m:
                    xs.append(ep)
                    ys.append(m[key])
            if xs:
                ax.plot(
                    xs,
                    ys,
                    label=a["label"],
                    color=a["color"],
                    marker=a["marker"],
                    linewidth=2.0,
                    markersize=7,
                )
        ax.set_title(title)
        ax.set_xlabel("SSL epoch")
        ax.set_xticks(EPS)
        ax.grid(True, axis="y", alpha=0.25)
    axes[0].set_ylabel("raw expert MAE (epoch-mean)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=min(4, len(labels)),
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, 1.08),
    )
    fig.suptitle("Expert-head raw losses vs SSL epoch", y=1.12, fontsize=12)
    fig.tight_layout()
    fig.savefig(fig_dir / "06_expert_raw_losses_vs_epoch.png", bbox_inches="tight")
    fig.savefig(fig_dir / "06_expert_raw_losses_vs_epoch.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_raw_contrastive(fig_dir: Path, series: List[Dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for a in series:
        log_dir = ROOT / "results/diagnostics" / a["run"] / "logs"
        xs, ys = [], []
        for ep in EPS:
            m = _epoch_mean_from_steps(log_dir, ep)
            if m and "L_contrast_raw" in m:
                xs.append(ep)
                ys.append(m["L_contrast_raw"])
        if xs:
            ax.plot(
                xs,
                ys,
                label=a["label"],
                color=a["color"],
                marker=a["marker"],
                linestyle=a.get("ls", "-"),
                linewidth=2.0,
                markersize=7,
            )
    ax.set_xlabel("SSL epoch")
    ax.set_ylabel("raw InfoNCE (epoch-mean of logged steps)")
    ax.set_xticks(EPS)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("Raw contrastive loss vs SSL epoch")
    fig.tight_layout()
    fig.savefig(fig_dir / "04_raw_contrastive_loss_vs_epoch.png")
    fig.savefig(fig_dir / "04_raw_contrastive_loss_vs_epoch.pdf")
    plt.close(fig)


def _plot_weights(fig_dir: Path, series: Optional[List[Dict[str, Any]]] = None) -> None:
    series = series if series is not None else _series_for_methods()
    _plot_raw_contrastive(fig_dir, series)
    _plot_objective_weights(fig_dir, series)
    _plot_expert_raw_losses(fig_dir, series)


def interpret(df: pd.DataFrame) -> Dict[str, Any]:
    def best(method: str) -> Dict[str, Any]:
        sub = df[df["method"] == method]
        i = sub["validation_auprc"].idxmax()
        r = sub.loc[i]
        return {
            "method": method,
            "best_ssl_epoch": int(r["ssl_epoch"]),
            "validation_auprc": float(r["validation_auprc"]),
            "f1_at_0.5": float(r["f1_at_0.5"]),
            "f1_at_val_thr": float(r["f1_at_val_thr"]),
        }

    adaptive = df[df["method"] == "TFMOE_adaptive"].set_index("ssl_epoch")
    expert = df[df["method"] == "EXPERT_ONLY"].set_index("ssl_epoch")
    fixed = df[df["method"] == "FIXED_BALANCED"].set_index("ssl_epoch")
    floor = df[df["method"] == "ADAPTIVE_CONTRAST_FLOOR"].set_index("ssl_epoch")

    answers = {
        "best_by_method": {m: best(m) for m in df["method"].unique()},
        "expert_only_vs_adaptive_delta_auprc": {
            int(ep): float(expert.loc[ep, "validation_auprc"] - adaptive.loc[ep, "validation_auprc"])
            for ep in EPS
            if ep in expert.index and ep in adaptive.index
        },
        "fixed_balanced_vs_adaptive_delta_auprc": {
            int(ep): float(fixed.loc[ep, "validation_auprc"] - adaptive.loc[ep, "validation_auprc"])
            for ep in EPS
            if ep in fixed.index and ep in adaptive.index
        },
        "floor_vs_adaptive_delta_auprc": {
            int(ep): float(floor.loc[ep, "validation_auprc"] - adaptive.loc[ep, "validation_auprc"])
            for ep in EPS
            if ep in floor.index and ep in adaptive.index
        },
    }
    return answers


def _write_package(
    df: pd.DataFrame,
    pkg: Path,
    *,
    exclude_methods: Optional[List[str]] = None,
    summary_title: str = "TFMOE objective-weighting ablation (R198-only)",
    summary_extra: Optional[List[str]] = None,
) -> Dict[str, Any]:
    fig = pkg / "figures"
    pkg.mkdir(parents=True, exist_ok=True)
    fig.mkdir(parents=True, exist_ok=True)

    excl = set(exclude_methods or [])
    dff = df[~df["method"].isin(excl)].copy()
    series = [s for s in _series_for_methods() if s["label"] not in excl]

    dff.to_csv(pkg / "table_trajectory_r198_only.csv", index=False)
    dff.to_json(pkg / "table_trajectory_r198_only.json", orient="records", indent=2)

    best_rows = []
    for method, g in dff.groupby("method"):
        r = g.loc[g["validation_auprc"].idxmax()]
        best_rows.append(r.to_dict())
    pd.DataFrame(best_rows).to_csv(pkg / "table_best_checkpoint.csv", index=False)

    _plot_metric(dff, "validation_auprc", "Validation AUPRC", fig / "01_r198_only_val_auprc_vs_epoch", series)
    _plot_metric(dff, "f1_at_0.5", "F1@0.5", fig / "02_r198_only_val_f1_at_0.5_vs_epoch", series)
    _plot_metric(
        dff,
        "f1_at_val_thr",
        "F1@val-selected threshold",
        fig / "03_r198_only_val_f1_at_val_thr_vs_epoch",
        series,
    )
    _plot_weights(fig, series)

    answers = interpret(dff)
    (pkg / "analysis_answers.json").write_text(json.dumps(answers, indent=2) + "\n")

    lines = [
        f"# {summary_title}",
        "",
        "- Protocol: full-subgraph extract + PaperStyleMLP on **R198 only** (dim=198)",
        "- Not thesis-wide primary; baselines reused (no SSL retrain)",
        "- EXPERT_ONLY = learned β among experts; w_contrast=0 (option a)",
    ]
    if summary_extra:
        lines.extend(summary_extra)
    lines += ["", "## Best checkpoint by method"]
    for m, info in answers["best_by_method"].items():
        lines.append(
            f"- **{m}**: SSL ep {info['best_ssl_epoch']} AUPRC={info['validation_auprc']:.4f} "
            f"F1@0.5={info['f1_at_0.5']:.4f} F1@thr={info['f1_at_val_thr']:.4f}"
        )
    lines += [
        "",
        "## Matched ΔAUPRC vs adaptive TFMOE",
        f"- EXPERT_ONLY − adaptive: {answers['expert_only_vs_adaptive_delta_auprc']}",
        f"- FIXED_BALANCED − adaptive: {answers['fixed_balanced_vs_adaptive_delta_auprc']}",
        f"- CONTRAST_FLOOR − adaptive: {answers['floor_vs_adaptive_delta_auprc']}",
    ]
    (pkg / "summary.md").write_text("\n".join(lines) + "\n")
    return {"n_rows": int(len(dff)), "methods": sorted(dff["method"].unique().tolist()), "pkg": str(pkg)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    args = ap.parse_args()
    out = Path(args.out)

    df = collect(out)
    assert (df["probe_input_dim"] == 198).all()
    assert (df["probe_feature_protocol"] == "R198_only").all()

    full = _write_package(df, out / "package")
    # Focused copy without FIXED_CURRENT_EARLY (original package kept intact above).
    focused = _write_package(
        df,
        out / "package" / "no_fixed_early",
        exclude_methods=["FIXED_CURRENT_EARLY"],
        summary_title="TFMOE weight ablation (R198-only; FIXED_CURRENT_EARLY omitted)",
        summary_extra=[
            "- This copy hides FIXED_CURRENT_EARLY; full package remains one level up.",
            "- Expert-loss figure uses one panel per expert head.",
        ],
    )

    manifest = {
        "probe_feature_protocol": "R198_only",
        "peak_lr": 0.002,
        "ssl_epochs_evaluated": EPS,
        "n_rows": full["n_rows"],
        "out_dir": str(out),
        "no_fixed_early": focused,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "ok", "full": full, "no_fixed_early": focused}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
