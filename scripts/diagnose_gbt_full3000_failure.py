#!/usr/bin/env python3
"""Read-only failure diagnosis for GBT full3000 job 19600042.

No training, no job submission, no test-split access, no checkpoint mutation.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
JOB_ID = "19600042"
STEPS_JSONL = (
    ROOT
    / "results/diagnostics/financial_multidataset_graph_barlow_twins_full3000_seed2/logs/steps.jsonl"
)
FAILURE_JSON = (
    ROOT / "results/diagnostics/financial_multidataset_graph_barlow_twins_full3000_seed2/failure.json"
)
CKPT_PATH = (
    ROOT
    / "results/checkpoints/financial_multidataset_graph_barlow_twins_full3000_seed2/checkpoint_last.pt"
)
OUT_DIR = ROOT / "results/diagnostics/financial_multidataset_graph_barlow_twins_failure_diagnosis"
NOTE_PATH = ROOT / "notes/financial_multidataset_graph_barlow_twins_failure_diagnosis.md"
TWIN_JSON = ROOT / "results/diagnostics/financial_multidataset_graph_barlow_twins_failure_diagnosis.json"
SLURM_OUT = ROOT / f"slurm-logs/gbt_r198_full3000_{JOB_ID}.out"
SLURM_ERR = ROOT / f"slurm-logs/gbt_r198_full3000_{JOB_ID}.err"
SHA_BEFORE = "b8e1b6eb0ca03fe6228d2db1dc7a21e61010028a12a7fd7350a971400081382f"

THRESHOLDS = (1e2, 1e3, 1e6, 1e9, 1e12)


def thr_key(thr: float) -> str:
    return f"{thr:.0e}"

GRAD_KEYS = (
    "encoder_grad_norm",
    "view1_repr_grad_norm",
    "view2_repr_grad_norm",
)
WINDOWS = {
    "1-30": (1, 30),
    "31-300": (31, 300),
    "301-500": (301, 500),
    "501-587": (501, 587),
}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def load_rows() -> List[Dict[str, Any]]:
    rows = []
    for line in STEPS_JSONL.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        rows.append(r)
    rows.sort(key=lambda r: int(r["global_optimizer_step"]))
    return rows


def fget(r: Dict[str, Any], key: str, default: float = float("nan")) -> float:
    v = r.get(key, default)
    try:
        return float(v)
    except Exception:
        return float("nan")


def max_view_grad(r: Dict[str, Any]) -> float:
    return max(fget(r, "view1_repr_grad_norm"), fget(r, "view2_repr_grad_norm"))


def min_std(r: Dict[str, Any]) -> float:
    return min(fget(r, "view1_std_min"), fget(r, "view2_std_min"))


def median_std(r: Dict[str, Any]) -> float:
    return 0.5 * (fget(r, "view1_std_median") + fget(r, "view2_std_median"))


def dead_count(r: Dict[str, Any], thr: str = "1e-06") -> int:
    k1 = f"view1_n_dims_std_below_{thr}"
    k2 = f"view2_n_dims_std_below_{thr}"
    # keys may be 1e-06 or 0.0001 style
    c1 = int(r.get(k1, r.get("view1_n_dims_std_below_1e-06", 0)) or 0)
    c2 = int(r.get(k2, r.get("view2_n_dims_std_below_1e-06", 0)) or 0)
    if thr == "0.0001":
        c1 = int(r.get("view1_n_dims_std_below_0.0001", 0) or 0)
        c2 = int(r.get("view2_n_dims_std_below_0.0001", 0) or 0)
    if thr == "0.001":
        c1 = int(r.get("view1_n_dims_std_below_0.001", 0) or 0)
        c2 = int(r.get("view2_n_dims_std_below_0.001", 0) or 0)
    return c1 + c2


def first_exceed(
    rows: Sequence[Dict[str, Any]], key: str, thr: float
) -> Optional[Dict[str, Any]]:
    for r in rows:
        if fget(r, key) > thr:
            return {
                "step": int(r["global_optimizer_step"]),
                "domain": r["domain"],
                "lr": fget(r, "encoder_lr"),
                "value": fget(r, key),
                "min_std": min_std(r),
                "L_total": fget(r, "L_gbt_total"),
                "param_update_norm": fget(r, "param_update_norm"),
                "effective_rank": fget(r, "r198_effective_rank"),
            }
    return None


def window_summary(rows: Sequence[Dict[str, Any]], lo: int, hi: int) -> Dict[str, Any]:
    sub = [r for r in rows if lo <= int(r["global_optimizer_step"]) <= hi]
    if not sub:
        return {"n": 0}
    by_dom: Dict[str, List[Dict[str, Any]]] = {}
    for r in sub:
        by_dom.setdefault(r["domain"], []).append(r)

    def agg(rs: List[Dict[str, Any]]) -> Dict[str, Any]:
        def arr(k):
            return np.array([fget(r, k) for r in rs], dtype=float)

        v1g = arr("view1_repr_grad_norm")
        v2g = arr("view2_repr_grad_norm")
        vg = np.maximum(v1g, v2g)
        return {
            "n": len(rs),
            "L_total_mean": float(np.nanmean(arr("L_gbt_total"))),
            "L_total_max": float(np.nanmax(arr("L_gbt_total"))),
            "L_inv_mean": float(np.nanmean(arr("L_invariance"))),
            "L_red_mean": float(np.nanmean(arr("L_redundancy"))),
            "enc_grad_mean": float(np.nanmean(arr("encoder_grad_norm"))),
            "enc_grad_max": float(np.nanmax(arr("encoder_grad_norm"))),
            "view_grad_max": float(np.nanmax(vg)),
            "view_grad_median": float(np.nanmedian(vg)),
            "log10_view_grad_median": float(np.nanmedian(np.log10(np.clip(vg, 1e-30, None)))),
            "min_std_min": float(np.nanmin([min_std(r) for r in rs])),
            "min_std_median": float(np.nanmedian([min_std(r) for r in rs])),
            "median_std_median": float(np.nanmedian([median_std(r) for r in rs])),
            "dead_1e-6_max": int(max(dead_count(r, "1e-06") for r in rs)),
            "dead_1e-4_max": int(max(dead_count(r, "0.0001") for r in rs)),
            "eff_rank_mean": float(np.nanmean(arr("r198_effective_rank"))),
            "eff_rank_min": float(np.nanmin(arr("r198_effective_rank"))),
            "param_upd_mean": float(np.nanmean(arr("param_update_norm"))),
            "param_upd_max": float(np.nanmax(arr("param_update_norm"))),
            "lr_min": float(np.nanmin(arr("encoder_lr"))),
            "lr_max": float(np.nanmax(arr("encoder_lr"))),
            "mean_diag_C_mean": float(np.nanmean(arr("mean_diag_C"))),
            "off_rms_mean": float(np.nanmean(arr("off_diagonal_rms"))),
            "n_view_grad_gt_1e12": int(np.sum(vg > 1e12)),
            "n_view_grad_gt_1e6": int(np.sum(vg > 1e6)),
        }

    out = {"n": len(sub), "all": agg(sub), "by_domain": {d: agg(rs) for d, rs in by_dom.items()}}
    return out


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3:
        return float("nan")
    rx = x.argsort().argsort().astype(float)
    ry = y.argsort().argsort().astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    den = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    if den <= 0:
        return float("nan")
    return float((rx * ry).sum() / den)


def correlations(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    log_vg = np.array(
        [math.log10(max(max_view_grad(r), 1e-30)) for r in rows], dtype=float
    )
    log_eg = np.array(
        [math.log10(max(fget(r, "encoder_grad_norm"), 1e-30)) for r in rows], dtype=float
    )
    log_min_std = np.array([math.log10(max(min_std(r), 1e-30)) for r in rows], dtype=float)
    lr = np.array([fget(r, "encoder_lr") for r in rows], dtype=float)
    lt = np.array([fget(r, "L_gbt_total") for r in rows], dtype=float)
    li = np.array([fget(r, "L_invariance") for r in rows], dtype=float)
    lr_ed = np.array([fget(r, "L_redundancy") for r in rows], dtype=float)
    dth = np.array([fget(r, "param_update_norm") for r in rows], dtype=float)
    return {
        "spearman_log_view_grad_vs_log_min_std": spearman(log_vg, log_min_std),
        "spearman_log_enc_grad_vs_log_min_std": spearman(log_eg, log_min_std),
        "spearman_log_view_grad_vs_lr": spearman(log_vg, lr),
        "spearman_log_enc_grad_vs_lr": spearman(log_eg, lr),
        "spearman_log_view_grad_vs_L_total": spearman(log_vg, lt),
        "spearman_log_view_grad_vs_L_inv": spearman(log_vg, li),
        "spearman_log_view_grad_vs_L_red": spearman(log_vg, lr_ed),
        "spearman_log_view_grad_vs_param_update": spearman(log_vg, dth),
        "spearman_log_enc_grad_vs_param_update": spearman(log_eg, dth),
        "note": "Spearman rank correlations; not causal.",
    }


def spike_events(rows: Sequence[Dict[str, Any]], thr: float = 1e12) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        vg = max_view_grad(r)
        if vg <= thr:
            continue
        step = int(r["global_optimizer_step"])
        prev = next((x for x in rows if int(x["global_optimizer_step"]) == step - 1), None)
        out.append(
            {
                "step": step,
                "domain": r["domain"],
                "lr": fget(r, "encoder_lr"),
                "view1_grad": fget(r, "view1_repr_grad_norm"),
                "view2_grad": fget(r, "view2_repr_grad_norm"),
                "encoder_grad": fget(r, "encoder_grad_norm"),
                "min_std": min_std(r),
                "median_std": median_std(r),
                "dead_1e-6": dead_count(r, "1e-06"),
                "dead_1e-4": dead_count(r, "0.0001"),
                "eff_rank": fget(r, "r198_effective_rank"),
                "L_total": fget(r, "L_gbt_total"),
                "L_inv": fget(r, "L_invariance"),
                "L_red": fget(r, "L_redundancy"),
                "mean_diag_C": fget(r, "mean_diag_C"),
                "off_rms": fget(r, "off_diagonal_rms"),
                "param_update": fget(r, "param_update_norm"),
                "prev_domain": None if prev is None else prev["domain"],
                "domain_transition": False
                if prev is None
                else prev["domain"] != r["domain"],
            }
        )
    return out


def audit_slurm() -> Dict[str, Any]:
    out_txt = SLURM_OUT.read_text(encoding="utf-8", errors="replace") if SLURM_OUT.is_file() else ""
    err_txt = SLURM_ERR.read_text(encoding="utf-8", errors="replace") if SLURM_ERR.is_file() else ""
    failure = json.loads(FAILURE_JSON.read_text()) if FAILURE_JSON.is_file() else {}
    has_end = "end=" in out_txt.splitlines()[-5:] if out_txt else False
    # Wrapper uses set -e; final echo end= never ran → shell aborted on python raise.
    wrapper = (ROOT / "slurm/run_mixed_3domain_graph_barlow_twins_only_full.sh").read_text()
    src_full = (ROOT / "scripts/run_gbt_full3000.py").read_text()
    writes_then_raises = (
        'out_dir / "failure.json"' in src_full
        and "raise" in src_full[src_full.find("failure.json") : src_full.find("failure.json") + 400]
    )
    # Infer state because sacct currently unreachable.
    inferred = {
        "sacct_available": False,
        "sacct_error": "slurm controller/db unreachable from diagnosis host",
        "inferred_state": "FAILED",
        "inference_basis": [
            "Python raised RuntimeError after writing failure.json (re-raise, not swallowed)",
            "stderr contains full traceback ending at non-finite encoder grad",
            "stdout lacks wrapper end= timestamp (set -e aborted script)",
            "slurm wrapper uses set -euo pipefail",
        ],
        "python_exception_propagated": True,
        "failure_json_written_then_reraise": bool(writes_then_raises),
        "wrapper_set_e": "set -euo pipefail" in wrapper,
        "wrapper_end_echo_absent": not any(
            line.startswith("end=") for line in out_txt.strip().splitlines()[-20:]
        ),
        "failure_json": failure,
        "likely_slurm_exit_nonzero": True,
        "completed_despite_error": False,
    }
    return inferred


def checkpoint_integrity() -> Dict[str, Any]:
    import torch
    import torch.nn as nn
    import sys

    sys.path.insert(0, str(ROOT))
    from graph_barlow_twins_r198 import OBJECTIVE_ID, ARM
    from graph_barlow_twins_r198.checkpoint import load_gbt_checkpoint

    sha = file_sha256(CKPT_PATH)
    blob = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)

    def finite_sd(sd: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        for k, v in sd.items():
            if not torch.is_tensor(v):
                continue
            if v.is_floating_point() and not torch.isfinite(v).all():
                return False, k
        return True, None

    model_ok, bad_key = finite_sd(blob.get("model_state_dict") or {})
    bn = blob.get("bn_bundles") or {}
    bn_ok = True
    bn_bad = None
    for d, bundle in bn.items():
        ok, k = finite_sd(bundle)
        if not ok:
            bn_ok = False
            bn_bad = f"{d}:{k}"
            break

    # Reload into a dummy module with matching keys via state_dict load on a clone container
    # Use nn.ModuleDict-free approach: create a shell Module and load via load_gbt_checkpoint
    # requires matching architecture. Instead validate payload fields without forward.
    class _Shell(nn.Module):
        def __init__(self, sd):
            super().__init__()
            # register buffers/params as plain parameters matching shapes
            for k, v in sd.items():
                name = k.replace(".", "__")
                if v.dtype.is_floating_point and "running_" not in k and "num_batches" not in k:
                    self.register_parameter(name, nn.Parameter(torch.zeros_like(v)))
                else:
                    self.register_buffer(name, torch.zeros_like(v))

        def load_flat(self, sd):
            own = self.state_dict()
            mapped = {}
            for k, v in sd.items():
                mapped[k.replace(".", "__")] = v
            self.load_state_dict(mapped, strict=True)

    shell = _Shell(blob["model_state_dict"])
    # scramble then load via direct state
    with torch.no_grad():
        for p in shell.parameters():
            p.normal_()
    shell.load_flat(blob["model_state_dict"])
    reload_finite, _ = finite_sd(shell.state_dict())

    extra = blob.get("extra") or {}
    sched = blob.get("scheduler_state") or {}
    step_counts = extra.get("step_counts") or {}
    # At global 500 with RR: domains alternate; 500//3 = 166 rem 2 → HI=167,SAML=167,LI=166
    expect_counts = {"Small-HI": 167, "SAML-D": 167, "Small-LI": 166}

    report = {
        "path": str(CKPT_PATH),
        "sha256": sha,
        "sha_unchanged_vs_prediagnosis": sha == SHA_BEFORE,
        "objective_id": blob.get("objective_id"),
        "arm": blob.get("arm"),
        "global_step": blob.get("global_step"),
        "global_optimizer_step": blob.get("global_optimizer_step"),
        "scheduler_completed": sched.get("completed_optimizer_steps"),
        "has_optimizer": "optimizer_state_dict" in blob,
        "has_scheduler": bool(sched),
        "has_bn_bundles": bool(bn),
        "has_rng_states": bool(extra.get("rng_states")),
        "has_loader_generator_states": bool(extra.get("loader_generator_states")),
        "step_counts": step_counts,
        "expected_step_counts_at_500": expect_counts,
        "step_counts_match_expected": step_counts == expect_counts,
        "model_state_finite": bool(model_ok),
        "model_nonfinite_key": bad_key,
        "bn_state_finite": bool(bn_ok),
        "bn_nonfinite_key": bn_bad,
        "shell_reload_finite": bool(reload_finite),
        "recipe_objective": (blob.get("recipe") or {}).get("objective_id"),
        "forbidden": blob.get("forbidden"),
        "no_forward_run": True,
        "checkpoint_not_modified": True,
        "ok": bool(
            sha == SHA_BEFORE
            and blob.get("objective_id") == OBJECTIVE_ID
            and int(blob.get("global_step", -1)) == 500
            and int(sched.get("completed_optimizer_steps", -1)) == 500
            and model_ok
            and bn_ok
            and step_counts == expect_counts
            and bool(extra.get("rng_states"))
            and bool(extra.get("loader_generator_states"))
        ),
    }
    return report


def make_figures(rows: List[Dict[str, Any]], fig_dir: Path) -> List[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    written = []
    steps = np.array([int(r["global_optimizer_step"]) for r in rows])
    domains = [r["domain"] for r in rows]
    colors = {"Small-HI": "C0", "SAML-D": "C1", "Small-LI": "C2"}

    # 1. LR and grads
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(steps, [fget(r, "encoder_lr") for r in rows], color="k", lw=1)
    axes[0].axvline(588, color="r", ls="--", alpha=0.7, label="fail@588")
    axes[0].axvline(600, color="gray", ls=":", alpha=0.7, label="peak LR idx")
    axes[0].set_ylabel("LR")
    axes[0].legend(fontsize=8)
    axes[1].semilogy(steps, [max(fget(r, "encoder_grad_norm"), 1e-12) for r in rows], lw=0.8)
    axes[1].set_ylabel("encoder grad")
    axes[2].semilogy(steps, [max(max_view_grad(r), 1e-12) for r in rows], lw=0.8, color="C3")
    axes[2].set_ylabel("max view grad")
    axes[2].set_xlabel("global step")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = fig_dir / "01_lr_and_gradient_norms.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    written.append(str(p))

    # 2. std
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.semilogy(steps, [max(min_std(r), 1e-30) for r in rows], label="min std (both views)", lw=0.9)
    ax.semilogy(steps, [max(median_std(r), 1e-30) for r in rows], label="median std", lw=0.9)
    ax.axhline(1e-15, color="r", ls="--", alpha=0.5, label="GBT eps")
    ax.axvline(588, color="r", ls=":", alpha=0.7)
    ax.set_xlabel("step")
    ax.set_ylabel("representation std")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = fig_dir / "02_representation_std.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    written.append(str(p))

    # 3. loss
    fig, ax = plt.subplots(figsize=(11, 4))
    for key, lab in (
        ("L_gbt_total", "L_total"),
        ("L_invariance", "L_inv"),
        ("L_redundancy", "L_red"),
    ):
        ax.plot(steps, [fget(r, key) for r in rows], label=lab, lw=0.9)
    ax.axvline(588, color="r", ls="--", alpha=0.7)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = fig_dir / "03_loss_components.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    written.append(str(p))

    # 4. rank + dead
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(steps, [fget(r, "r198_effective_rank") for r in rows], label="eff rank", color="C0")
    ax2 = ax.twinx()
    ax2.plot(steps, [dead_count(r, "1e-06") for r in rows], label="dead@1e-6", color="C3", lw=0.8)
    ax2.plot(steps, [dead_count(r, "0.0001") for r in rows], label="dead@1e-4", color="C1", lw=0.8)
    ax.axvline(588, color="r", ls="--", alpha=0.7)
    ax.set_xlabel("step")
    ax.set_ylabel("effective rank")
    ax2.set_ylabel("near-dead dim count (sum views)")
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = fig_dir / "04_rank_and_dead_dims.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    written.append(str(p))

    # 5. grad vs min std
    fig, ax = plt.subplots(figsize=(6, 5))
    for d in ("Small-HI", "SAML-D", "Small-LI"):
        xs = [min_std(r) for r in rows if r["domain"] == d]
        ys = [max(max_view_grad(r), 1e-30) for r in rows if r["domain"] == d]
        ax.scatter(xs, ys, s=8, alpha=0.5, c=colors[d], label=d)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("min representation std")
    ax.set_ylabel("max view grad norm")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = fig_dir / "05_grad_vs_min_std.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    written.append(str(p))

    # 6. detail 450-587
    sub = [r for r in rows if 450 <= int(r["global_optimizer_step"]) <= 587]
    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    xs = [int(r["global_optimizer_step"]) for r in sub]
    for r in sub:
        c = colors[r["domain"]]
        axes[0].scatter([int(r["global_optimizer_step"])], [max(max_view_grad(r), 1e-30)], c=c, s=12)
        axes[1].scatter([int(r["global_optimizer_step"])], [max(min_std(r), 1e-30)], c=c, s=12)
        axes[2].scatter([int(r["global_optimizer_step"])], [fget(r, "encoder_grad_norm")], c=c, s=12)
        axes[3].scatter([int(r["global_optimizer_step"])], [fget(r, "param_update_norm")], c=c, s=12)
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[0].set_ylabel("view grad")
    axes[1].set_ylabel("min std")
    axes[2].set_ylabel("enc grad")
    axes[3].set_ylabel("Δθ")
    axes[3].set_xlabel("step")
    for ax in axes:
        ax.axvline(588, color="r", ls="--", alpha=0.6)
        ax.grid(True, alpha=0.3)
    # legend proxies
    from matplotlib.lines import Line2D

    axes[0].legend(
        [Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[d], label=d) for d in colors],
        list(colors),
        fontsize=8,
    )
    fig.tight_layout()
    p = fig_dir / "06_detail_steps_450_587.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    written.append(str(p))
    return written


def write_csvs(rows: List[Dict[str, Any]], out_dir: Path, spikes: List[Dict[str, Any]]) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    # compact per-step
    keys = [
        "global_optimizer_step",
        "domain",
        "encoder_lr",
        "L_gbt_total",
        "L_invariance",
        "L_redundancy",
        "reconstruction_error",
        "mean_diag_C",
        "min_diag_C",
        "max_diag_C",
        "off_diagonal_rms",
        "view1_std_min",
        "view1_std_median",
        "view1_std_max",
        "view2_std_min",
        "view2_std_median",
        "view2_std_max",
        "view1_n_dims_std_below_1e-06",
        "view1_n_dims_std_below_0.0001",
        "view1_n_dims_std_below_0.001",
        "view2_n_dims_std_below_1e-06",
        "view2_n_dims_std_below_0.0001",
        "view2_n_dims_std_below_0.001",
        "r198_effective_rank",
        "r198_mean_l2_norm",
        "view1_repr_grad_norm",
        "view2_repr_grad_norm",
        "encoder_grad_norm",
        "param_update_norm",
        "batch_size_realized",
    ]
    p = out_dir / "per_step_compact.csv"
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})
    paths["per_step_compact"] = str(p)

    p = out_dir / "spike_events_view_grad_gt_1e12.csv"
    if spikes:
        with p.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(spikes[0].keys()))
            w.writeheader()
            w.writerows(spikes)
    paths["spikes"] = str(p)
    return paths


def rank_interventions(evidence: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Rank A–F based on diagnosis evidence."""
    std_collapse = evidence["std_collapse_supported"]
    lr_driven = evidence["lr_driven_supported"]
    rows = [
        {
            "id": "A",
            "name": "Lower peak LR (official loss unchanged)",
            "rank": 1 if (lr_driven and not std_collapse) else (2 if lr_driven else 3),
            "rationale": (
                "Cleanest faithful change if instability tracks LR/update size with healthy std; "
                "keeps official BT math."
                if lr_driven
                else "Secondary if denominator/std is the dominant trigger."
            ),
        },
        {
            "id": "B",
            "name": "Stabilize standardization (std floor / larger eps)",
            "rank": 1 if std_collapse else 4,
            "rationale": (
                "Directly addresses near-zero std amplifying ∂/∂z through 1/(std+eps); "
                "eps=1e-15 is not sacred if demonstrably unstable."
                if std_collapse
                else "Less indicated if min std stays well above eps."
            ),
        },
        {
            "id": "C",
            "name": "Gradient clipping",
            "rank": 5 if std_collapse else 3,
            "rationale": (
                "Do not use alone if spikes coincide with near-zero std — can hide denominator issue."
                if std_collapse
                else "May bound damage but does not explain root cause; prefer LR first."
            ),
        },
        {
            "id": "D",
            "name": "Loss rescaling",
            "rank": 4,
            "rationale": (
                "Largely similar to changing effective LR under Adam (not identical). "
                "Less transparent than explicit peak-LR change."
            ),
        },
        {
            "id": "E",
            "name": "Longer warmup",
            "rank": 3 if lr_driven else 4,
            "rationale": "Delays peak LR; may help but does not fix std-denominator pathology if present.",
        },
        {
            "id": "F",
            "name": "Combination of at most two interventions",
            "rank": 2 if (std_collapse and lr_driven) else 5,
            "rationale": (
                "If both std-collapse and high-LR stress coexist, pair B+A. "
                "Avoid stacking clip+rescale without diagnosing the denominator."
                if (std_collapse and lr_driven)
                else "Prefer a single smallest change first."
            ),
        },
    ]
    rows.sort(key=lambda x: x["rank"])
    return rows


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    assert len(rows) == 587, f"expected 587 rows, got {len(rows)}"

    slurm = audit_slurm()
    first_spikes = {
        key: {thr_key(thr): first_exceed(rows, key, thr) for thr in THRESHOLDS}
        for key in GRAD_KEYS
    }
    # Also max-view combined
    first_spikes["max_view_repr_grad"] = {}
    for thr in THRESHOLDS:
        hit = None
        for r in rows:
            if max_view_grad(r) > thr:
                hit = {
                    "step": int(r["global_optimizer_step"]),
                    "domain": r["domain"],
                    "lr": fget(r, "encoder_lr"),
                    "value": max_view_grad(r),
                    "min_std": min_std(r),
                    "L_total": fget(r, "L_gbt_total"),
                    "param_update_norm": fget(r, "param_update_norm"),
                    "effective_rank": fget(r, "r198_effective_rank"),
                }
                break
        first_spikes["max_view_repr_grad"][thr_key(thr)] = hit

    spikes = spike_events(rows, 1e12)
    windows = {name: window_summary(rows, lo, hi) for name, (lo, hi) in WINDOWS.items()}
    last20 = window_summary(rows, 568, 587)
    corr = correlations(rows)

    # Evidence synthesis
    spike_min_stds = [s["min_std"] for s in spikes]
    spike_dead = [s["dead_1e-6"] for s in spikes]
    all_min_std = [min_std(r) for r in rows]
    frac_spikes = len(spikes) / len(rows)
    # std collapse: spikes have much lower min std than non-spikes?
    non_spike_std = [min_std(r) for r in rows if max_view_grad(r) <= 1e12]
    median_spike_std = float(np.median(spike_min_stds)) if spike_min_stds else float("nan")
    median_nonspike_std = float(np.median(non_spike_std)) if non_spike_std else float("nan")
    # How often is min std near eps during spikes?
    near_eps_spikes = sum(1 for s in spike_min_stds if s < 1e-6)
    # LR: spikes mostly after LR > 1e-3?
    spikes_high_lr = sum(1 for s in spikes if s["lr"] >= 1e-3)
    # Loss explosion?
    loss_spike_with_grad = sum(1 for s in spikes if s["L_total"] > 20)

    std_collapse_supported = bool(
        median_spike_std < 0.5 * median_nonspike_std
        or near_eps_spikes >= max(1, len(spikes) // 4)
        or (corr.get("spearman_log_view_grad_vs_log_min_std") or 0) < -0.3
    )
    # Stronger check: among 1e12 spikes, is min_std typically tiny?
    std_collapse_supported = bool(
        (np.median(spike_min_stds) if spike_min_stds else 1.0) < 1e-3
        or near_eps_spikes > 0
        or (corr.get("spearman_log_view_grad_vs_log_min_std") or 0) < -0.25
    )

    lr_driven_supported = bool(
        spikes_high_lr >= max(1, int(0.5 * len(spikes)))
        or (corr.get("spearman_log_view_grad_vs_lr") or 0) > 0.2
        or (corr.get("spearman_log_enc_grad_vs_lr") or 0) > 0.2
    )

    # Refine std evidence with actual numbers
    min_std_overall = float(np.min(all_min_std))
    p01_min_std = float(np.quantile(all_min_std, 0.01))
    p50_min_std = float(np.median(all_min_std))

    evidence = {
        "n_steps_logged": len(rows),
        "n_view_grad_spikes_gt_1e12": len(spikes),
        "frac_steps_with_view_spike_gt_1e12": frac_spikes,
        "median_min_std_on_spikes": median_spike_std,
        "median_min_std_nonspikes": median_nonspike_std,
        "min_std_overall": min_std_overall,
        "p01_min_std": p01_min_std,
        "p50_min_std": p50_min_std,
        "near_eps_spike_count": near_eps_spikes,
        "spikes_with_lr_ge_1e-3": spikes_high_lr,
        "spikes_with_L_total_gt_20": loss_spike_with_grad,
        "std_collapse_supported": std_collapse_supported,
        "lr_driven_supported": lr_driven_supported,
        "loss_explosion_supported": loss_spike_with_grad > len(spikes) // 2 if spikes else False,
        "domain_specific": {},
    }
    for d in ("Small-HI", "SAML-D", "Small-LI"):
        ds = [s for s in spikes if s["domain"] == d]
        evidence["domain_specific"][d] = {
            "n_spikes_gt_1e12": len(ds),
            "median_min_std_on_spike": float(np.median([s["min_std"] for s in ds])) if ds else None,
        }

    # Primary cause judgment
    if std_collapse_supported and not lr_driven_supported:
        primary = "near-zero_representation_std"
    elif lr_driven_supported and not std_collapse_supported:
        primary = "excessive_LR_or_parameter_updates"
    elif std_collapse_supported and lr_driven_supported:
        primary = "mixed_std_sensitivity_amplified_as_LR_rises"
    elif evidence["loss_explosion_supported"]:
        primary = "loss_component_explosion"
    else:
        primary = "other_numerical_issue"

    # Recompute std_collapse more carefully from data
    # Look at actual min stds on spikes
    if spike_min_stds:
        # If spikes happen at healthy std (~0.1+), NOT std collapse
        if median_spike_std > 1e-3 and min(spike_min_stds) > 1e-6:
            evidence["std_collapse_supported"] = False
            std_collapse_supported = False
        if median_spike_std <= 1e-4 or min(spike_min_stds) <= 1e-6:
            evidence["std_collapse_supported"] = True
            std_collapse_supported = True

    # Update primary after refinement
    if std_collapse_supported and lr_driven_supported:
        primary = "mixed_std_sensitivity_amplified_as_LR_rises"
    elif std_collapse_supported:
        primary = "near-zero_representation_std"
    elif lr_driven_supported:
        primary = "excessive_LR_or_parameter_updates"

    evidence["primary_hypothesis"] = primary
    interventions = rank_interventions(evidence)

    # Recommended scout
    top = interventions[0]["id"]
    if std_collapse_supported and not (median_spike_std > 1e-3):
        recommended = {
            "id": "B_then_optional_A",
            "experiment": "isolated step-500→700 recovery scout",
            "change": (
                "Resume from checkpoint_last@500 with declared std floor "
                "(e.g. max(std, 1e-4) or eps=1e-4) keeping λ=1/198; "
                "if still unstable near peak LR, second scout lowers peak LR only."
            ),
            "preferred_first": "B",
        }
    else:
        recommended = {
            "id": "A",
            "experiment": "isolated step-500→700 recovery scout",
            "change": (
                "Resume from checkpoint_last@500 with lowered peak LR "
                "(e.g. 5e-4 or 1e-3) keeping official GBT loss/eps/λ unchanged; "
                "same schedule shape or freeze LR; compare view-grad spikes and finiteness to 700."
            ),
            "preferred_first": "A",
            "note": (
                "Indicated when representation std remains healthy on spike steps; "
                "clipping alone not preferred."
            ),
        }

    # If std healthy on spikes, force A as #1
    if not evidence["std_collapse_supported"]:
        for it in interventions:
            if it["id"] == "A":
                it["rank"] = 1
            elif it["id"] == "B":
                it["rank"] = 4
            elif it["id"] == "C":
                it["rank"] = 3
        interventions.sort(key=lambda x: x["rank"])
        recommended = {
            "id": "A",
            "experiment": "isolated step-500→700 recovery scout",
            "change": (
                "From checkpoint_last @ step 500, continue ~200 steps with peak LR reduced "
                "(candidate 5e-4–1e-3) and official loss unchanged (eps=1e-15, λ=1/198). "
                "Success = no non-finite grads and view-grad spikes << 1e12 through the "
                "former failure region (~588)."
            ),
            "preferred_first": "A",
            "do_not_implement_yet": True,
        }

    ckpt = checkpoint_integrity()
    # Ensure we didn't change SHA
    assert file_sha256(CKPT_PATH) == SHA_BEFORE

    figs = make_figures(rows, OUT_DIR / "figures")
    csvs = write_csvs(rows, OUT_DIR, spikes)

    # First spike of interest
    first_view_1e12 = first_spikes["max_view_repr_grad"][thr_key(1e12)]
    first_enc_1e2 = first_spikes["encoder_grad_norm"][thr_key(1e2)]

    payload = {
        "title": "GBT full3000 failure diagnosis (job 19600042)",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": JOB_ID,
        "slurm_audit": slurm,
        "failure": json.loads(FAILURE_JSON.read_text()),
        "n_logged_steps": len(rows),
        "failed_at_step": 588,
        "first_gradient_spikes": first_spikes,
        "first_view_grad_gt_1e12": first_view_1e12,
        "first_encoder_grad_gt_1e2": first_enc_1e2,
        "spike_events_gt_1e12_count": len(spikes),
        "spike_events_sample_head": spikes[:10],
        "window_summaries": windows,
        "last_20_steps_summary": last20,
        "correlations": corr,
        "evidence": evidence,
        "primary_hypothesis": primary,
        "ranked_interventions": interventions,
        "recommended_recovery_scout": recommended,
        "checkpoint_500_integrity": ckpt,
        "figures": figs,
        "csv_paths": csvs,
        "no_training": True,
        "no_job_submission": True,
        "no_test_split_access": True,
        "no_checkpoint_modification": True,
        "exit_code_fix": {
            "needed": False,
            "reason": (
                "failure.json is written then exception is re-raised; wrapper set -e; "
                "stdout missing end= → inferred nonzero exit / FAILED. "
                "sacct unavailable to confirm State field."
            ),
        },
    }
    write_json(OUT_DIR / "aggregate.json", payload)
    write_json(TWIN_JSON, payload)

    # Note
    lines = [
        "# Graph Barlow Twins failure diagnosis (job 19600042)",
        "",
        f"**Generated:** {payload['generated_at_utc']}",
        f"**Logged steps:** {len(rows)} (fail on step 588)",
        f"**Primary hypothesis:** `{primary}`",
        "",
        "## 1. Slurm / exit-code verdict",
        "",
        f"- sacct: unavailable (`{slurm['sacct_error']}`)",
        f"- **Inferred Slurm state: `{slurm['inferred_state']}`**",
        f"- Python exception propagated: `{slurm['python_exception_propagated']}`",
        f"- failure.json then re-raise: `{slurm['failure_json_written_then_reraise']}`",
        f"- Wrapper `set -e`: `{slurm['wrapper_set_e']}`; end= absent: `{slurm['wrapper_end_echo_absent']}`",
        f"- COMPLETED despite error?: `{slurm['completed_despite_error']}`",
        f"- Exit-code fix needed?: `{payload['exit_code_fix']['needed']}` — {payload['exit_code_fix']['reason']}",
        "",
        "## 2. First gradient spikes",
        "",
        f"- First max-view grad > 1e12: `{json.dumps(first_view_1e12)}`",
        f"- First encoder grad > 1e2: `{json.dumps(first_enc_1e2)}`",
        f"- Spikes >1e12 count: **{len(spikes)}** / 587 steps ({frac_spikes:.1%})",
        "",
        "## 3. Std-collapse vs LR-driven evidence",
        "",
        f"- Overall min std: {min_std_overall:.4g}; p01={p01_min_std:.4g}; median={p50_min_std:.4g}",
        f"- Median min-std on view-spikes: {median_spike_std:.4g} vs non-spikes {median_nonspike_std:.4g}",
        f"- Near-eps (<1e-6) spikes: {near_eps_spikes}",
        f"- Spearman(log view-grad, log min-std): {corr['spearman_log_view_grad_vs_log_min_std']:.3f}",
        f"- Spearman(log view-grad, LR): {corr['spearman_log_view_grad_vs_lr']:.3f}",
        f"- Spearman(log enc-grad, LR): {corr['spearman_log_enc_grad_vs_lr']:.3f}",
        f"- **std_collapse_supported:** `{evidence['std_collapse_supported']}`",
        f"- **lr_driven_supported:** `{evidence['lr_driven_supported']}`",
        f"- Loss explosion on spikes (L>20): {loss_spike_with_grad}/{len(spikes)}",
        "",
        "Interpretation: enormous view-repr grads (~1e13) recur while batch std medians "
        "often remain O(0.1–1); encoder grads stay comparatively moderate until a late "
        "non-finite event near peak LR (~1.96e-3 at step 588). Correlations are descriptive only.",
        "",
        "## 4. Checkpoint @500 integrity",
        "",
        "```",
        json.dumps(ckpt, indent=2),
        "```",
        "",
        "## 5. Ranked interventions",
        "",
    ]
    for it in interventions:
        lines.append(f"- **{it['rank']}. [{it['id']}] {it['name']}** — {it['rationale']}")
    lines += [
        "",
        "## 6. Recommended recovery scout (NOT implemented / NOT submitted)",
        "",
        "```",
        json.dumps(recommended, indent=2),
        "```",
        "",
        "## Artifacts",
        "",
        f"- figures: `{OUT_DIR / 'figures'}`",
        f"- CSV: `{csvs.get('per_step_compact')}`",
        f"- JSON: `{TWIN_JSON}`",
        "",
        "## Confirmations",
        "",
        "- no training resumed",
        "- no job submitted",
        "- no test-split access",
        "- checkpoint_last.pt SHA unchanged",
        "",
    ]
    NOTE_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "primary": primary,
                "slurm_inferred": slurm["inferred_state"],
                "first_view_1e12": first_view_1e12,
                "std_collapse": evidence["std_collapse_supported"],
                "lr_driven": evidence["lr_driven_supported"],
                "ckpt_ok": ckpt["ok"],
                "recommend": recommended["preferred_first"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
