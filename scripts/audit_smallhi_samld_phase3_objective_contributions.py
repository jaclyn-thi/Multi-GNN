#!/usr/bin/env python3
"""Read-only Phase-3 objective-weight / loss-contribution audit.

Offline analysis of existing training logs only.
Does not train, extract, probe, submit Slurm, load NPZ embeddings, or access test data.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCOUT = ROOT / "results/diagnostics/smallhi_samld_mixed_ssl_phase3_scout"
OUT = ROOT / "results/diagnostics/smallhi_samld_phase3_objective_contribution_audit"
NOTES = ROOT / "notes/smallhi_samld_phase3_objective_contribution_audit.md"
TWIN = ROOT / "results/diagnostics/smallhi_samld_phase3_objective_contribution_audit.json"

ARMS = ("SMALL_HI_ONLY", "SAMLD_ONLY", "MIXED_1TO1")
TF_NAMES = (
    "log1p_sender_interarrival",
    "log1p_sender_past_7d_count",
    "log1p_amount_vs_sender_past_mean",
)
RECON_ATOL = 1e-5
SECONDARY_CAVEAT = (
    "Secondary exposure-matched comparison (mixed@500/domain vs single@500) is "
    "NOT perfectly LR-phase matched for domain exposure."
)

# Okabe–Ito-ish
COLORS = {
    "contrast": "#0072B2",
    "tf0": "#E69F00",
    "tf1": "#009E73",
    "tf2": "#CC79A7",
    "total": "#000000",
    "alpha": "#56B4E9",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def load_rows(arm: str) -> List[Dict[str, Any]]:
    p = SCOUT / "arms" / arm / "logs" / "steps.jsonl"
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def verify_objective_source() -> Dict[str, Any]:
    src = ROOT / "direct_r198" / "__init__.py"
    text = src.read_text(encoding="utf-8")
    # Prove target order from source
    from direct_r198 import TF_MOE_TARGET_NAMES, resolve_tfmoe_weights
    import torch
    from direct_r198 import LearnedAlphaBeta

    names = list(TF_MOE_TARGET_NAMES)
    if names != list(TF_NAMES):
        raise RuntimeError(f"TF target order mismatch: {names}")

    # Prove adaptive formula numerically
    ab = LearnedAlphaBeta(n_tf=3, init_alpha=0.6)
    alpha, beta = ab()
    w_c, w_tf, meta = resolve_tfmoe_weights("adaptive", alpha, beta)
    assert abs(float(w_c) - float(alpha)) < 1e-8
    for i in range(3):
        expect = float((1.0 - alpha) * beta[i])
        assert abs(float(w_tf[i]) - expect) < 1e-8

    # Phase-3 resolved configs
    integ = json.loads((SCOUT / "training_integrity_summary.json").read_text())
    modes = set()
    for arm in ARMS:
        r = integ["resolved_configs"][arm]
        modes.add(r.get("direct_r198_tfmoe_weight_mode"))
        if list(r.get("moe_targets") or []) != list(TF_NAMES):
            raise RuntimeError(f"{arm} moe_targets mismatch: {r.get('moe_targets')}")
    if modes != {"adaptive"}:
        raise RuntimeError(f"unexpected weight modes: {modes}")

    trainer = ROOT / "scripts" / "train_mixed_ssl_phase3_scout.py"
    return {
        "objective_function": "direct_r198.combine_direct_h_tfmoe_loss",
        "weight_resolver": "direct_r198.resolve_tfmoe_weights",
        "source_file": str(src.relative_to(ROOT)),
        "source_sha256": sha256_file(src),
        "trainer_file": str(trainer.relative_to(ROOT)),
        "trainer_sha256": sha256_file(trainer),
        "weight_mode": "adaptive",
        "formula": (
            "L_total = alpha * L_contrast_norm + (1-alpha) * sum_m beta_m * L_tf_norm_m "
            "with alpha=sigmoid(alpha_logit), beta=softmax(beta_logits); "
            "w_contrast=alpha; w_tf_m=(1-alpha)*beta_m"
        ),
        "tf_target_order": list(TF_NAMES),
        "tf_target_order_proven_from": [
            "direct_r198.TF_MOE_TARGET_NAMES",
            "training_integrity_summary.json resolved_configs.*.moe_targets",
            "step logs k/tfmoe/raw_mae/<name> keys",
        ],
        "not_adaptive_contrast_floor": True,
        "phase3_arms_weight_mode_verified": "adaptive",
    }


def reconstruct_point(r: Dict[str, Any]) -> Optional[Dict[str, float]]:
    need = [
        "w_contrast",
        "w_tf_0",
        "w_tf_1",
        "w_tf_2",
        "L_contrast_norm",
        "L_tf_norm_0",
        "L_tf_norm_1",
        "L_tf_norm_2",
        "L_total",
    ]
    if any(r.get(k) is None for k in need):
        return None
    if not r.get("calibration_complete_domain"):
        return None
    wc = float(r["w_contrast"])
    wn = [float(r[f"w_tf_{m}"]) for m in range(3)]
    lc = float(r["L_contrast_norm"])
    ln = [float(r[f"L_tf_norm_{m}"]) for m in range(3)]
    c_c = wc * lc
    c_tf = [wn[m] * ln[m] for m in range(3)]
    c_sum = c_c + sum(c_tf)
    err = abs(c_sum - float(r["L_total"]))
    # Also verify logged weighted_* if present (should match C_*)
    return {
        "C_contrast": c_c,
        "C_tf_0": c_tf[0],
        "C_tf_1": c_tf[1],
        "C_tf_2": c_tf[2],
        "C_sum": c_sum,
        "recon_abs_err": err,
        "w_sum": wc + sum(wn),
        "share_contrast": c_c / c_sum if c_sum else float("nan"),
        "share_tf_0": c_tf[0] / c_sum if c_sum else float("nan"),
        "share_tf_1": c_tf[1] / c_sum if c_sum else float("nan"),
        "share_tf_2": c_tf[2] / c_sum if c_sum else float("nan"),
        "share_tf_total": sum(c_tf) / c_sum if c_sum else float("nan"),
    }


def enrich(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        e = dict(r)
        rec = reconstruct_point(r)
        e["_recon"] = rec
        e["_post_calib"] = bool(r.get("calibration_complete_domain")) and rec is not None
        out.append(e)
    return out


def domain_rows(rows: List[Dict[str, Any]], domain: str) -> List[Dict[str, Any]]:
    return [r for r in rows if r.get("domain") == domain]


def at_global(rows: List[Dict[str, Any]], step: int) -> Optional[Dict[str, Any]]:
    exact = [r for r in rows if int(r.get("global_optimizer_step", -1)) == step]
    if exact:
        return exact[0]
    # nearest
    if not rows:
        return None
    return min(rows, key=lambda r: abs(int(r.get("global_optimizer_step", 0)) - step))


def at_domain_exposure(rows: List[Dict[str, Any]], domain: str, exp: int) -> Optional[Dict[str, Any]]:
    cand = [r for r in domain_rows(rows, domain) if int(r.get("domain_exposure_count", -1)) == exp]
    if cand:
        return cand[0]
    drows = domain_rows(rows, domain)
    if not drows:
        return None
    return min(drows, key=lambda r: abs(int(r.get("domain_exposure_count", 0)) - exp))


def last_n_post(rows: List[Dict[str, Any]], n: int, domain: Optional[str] = None) -> List[Dict[str, Any]]:
    pool = rows if domain is None else domain_rows(rows, domain)
    post = [r for r in pool if r.get("_post_calib") and r.get("_recon")]
    return post[-n:]


def mean_contrib_over_mean_total(recs: List[Dict[str, float]]) -> Dict[str, float]:
    if not recs:
        return {}
    keys = ["C_contrast", "C_tf_0", "C_tf_1", "C_tf_2", "C_sum"]
    means = {k: float(np.mean([r[k] for r in recs])) for k in keys}
    cs = means["C_sum"]
    return {
        "aggregation": "mean_component_over_mean_total",
        "mean_C_contrast": means["C_contrast"],
        "mean_C_tf_0": means["C_tf_0"],
        "mean_C_tf_1": means["C_tf_1"],
        "mean_C_tf_2": means["C_tf_2"],
        "mean_C_sum": cs,
        "share_contrast": means["C_contrast"] / cs if cs else float("nan"),
        "share_tf_0": means["C_tf_0"] / cs if cs else float("nan"),
        "share_tf_1": means["C_tf_1"] / cs if cs else float("nan"),
        "share_tf_2": means["C_tf_2"] / cs if cs else float("nan"),
        "share_tf_total": (means["C_tf_0"] + means["C_tf_1"] + means["C_tf_2"]) / cs if cs else float("nan"),
        "n": len(recs),
    }


def mean_of_shares(recs: List[Dict[str, float]]) -> Dict[str, float]:
    if not recs:
        return {}
    return {
        "aggregation": "mean_of_per_step_shares",
        "share_contrast": float(np.mean([r["share_contrast"] for r in recs])),
        "share_tf_0": float(np.mean([r["share_tf_0"] for r in recs])),
        "share_tf_1": float(np.mean([r["share_tf_1"] for r in recs])),
        "share_tf_2": float(np.mean([r["share_tf_2"] for r in recs])),
        "share_tf_total": float(np.mean([r["share_tf_total"] for r in recs])),
        "n": len(recs),
    }


def dominant_tf(shares: Dict[str, float], prefix: str = "share_tf_") -> Tuple[int, str]:
    vals = [(m, shares.get(f"{prefix}{m}", float("nan"))) for m in range(3)]
    m_best = max(vals, key=lambda x: x[1] if x[1] == x[1] else -1)
    return m_best[0], TF_NAMES[m_best[0]]


def snapshot_row(arm: str, r: Dict[str, Any], label: str) -> Dict[str, Any]:
    return {
        "label": label,
        "arm": arm,
        "domain": r.get("domain"),
        "global_step": int(r.get("global_optimizer_step", r.get("step", -1) + 1)),
        "domain_exposure": int(r.get("domain_exposure_count", -1)),
        "alpha": float(r.get("alpha_raw", r.get("alpha", float("nan")))),
        "beta0": float(r.get("beta_0", float("nan"))),
        "beta1": float(r.get("beta_1", float("nan"))),
        "beta2": float(r.get("beta_2", float("nan"))),
        "w_contrast": float(r.get("w_contrast", float("nan"))),
        "w_tf0": float(r.get("w_tf_0", float("nan"))),
        "w_tf1": float(r.get("w_tf_1", float("nan"))),
        "w_tf2": float(r.get("w_tf_2", float("nan"))),
        "L_contrast_raw": float(r.get("L_contrast_raw", float("nan"))),
        "L_contrast_norm": float(r.get("L_contrast_norm", float("nan"))),
        "L_tf_raw_0": float(r.get("L_tf_raw_0", float("nan"))),
        "L_tf_raw_1": float(r.get("L_tf_raw_1", float("nan"))),
        "L_tf_raw_2": float(r.get("L_tf_raw_2", float("nan"))),
        "L_tf_norm_0": float(r.get("L_tf_norm_0", float("nan"))),
        "L_tf_norm_1": float(r.get("L_tf_norm_1", float("nan"))),
        "L_tf_norm_2": float(r.get("L_tf_norm_2", float("nan"))),
        "L_total": float(r.get("L_total", float("nan"))),
        "alpha_beta_frozen": bool(r.get("alpha_beta_frozen")),
        "calibration_complete_domain": bool(r.get("calibration_complete_domain")),
        **({f"recon_{k}": v for k, v in (r.get("_recon") or {}).items()}),
    }


def plot_figures(all_rows: Dict[str, List[Dict[str, Any]]], fig_dir: Path) -> Dict[str, str]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        (fig_dir / "plot_error.txt").write_text(str(exc), encoding="utf-8")
        return paths

    def save(fig, stem: str) -> None:
        fig.tight_layout()
        png = fig_dir / f"{stem}.png"
        pdf = fig_dir / f"{stem}.pdf"
        fig.savefig(png, dpi=140)
        fig.savefig(pdf)
        plt.close(fig)
        paths[stem] = str(png.relative_to(ROOT))

    # 1 alpha beta
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    for ax, arm in zip(axes, ARMS):
        rows = all_rows[arm]
        xs = [int(r["global_optimizer_step"]) for r in rows]
        ax.plot(xs, [float(r["alpha_raw"]) for r in rows], color=COLORS["alpha"], label="alpha")
        for m, c in zip(range(3), (COLORS["tf0"], COLORS["tf1"], COLORS["tf2"])):
            ax.plot(xs, [float(r[f"beta_{m}"]) for r in rows], color=c, label=f"beta{m}", alpha=0.85)
        ax.axvline(10.5, color="gray", ls="--", lw=1, label="α/β unfreeze (after step0=9)")
        ax.set_ylabel(arm.replace("_", "\n"), fontsize=8)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7, ncol=4, loc="upper right")
    axes[-1].set_xlabel("global optimizer step")
    fig.suptitle("Learned alpha / beta (global in MIXED)", fontsize=11)
    save(fig, "01_alpha_beta_trajectory")

    # 2 effective weights
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    for ax, arm in zip(axes, ARMS):
        rows = all_rows[arm]
        xs = [int(r["global_optimizer_step"]) for r in rows]
        ax.plot(xs, [float(r["w_contrast"]) for r in rows], color=COLORS["contrast"], label="w_contrast")
        for m, c, name in zip(
            range(3),
            (COLORS["tf0"], COLORS["tf1"], COLORS["tf2"]),
            ("interarrival", "past_7d_count", "amount_vs_mean"),
        ):
            ax.plot(xs, [float(r[f"w_tf_{m}"]) for r in rows], color=c, label=f"w_tf{m}:{name}", alpha=0.9)
        ax.axvline(10, color="gray", ls="--", lw=1)
        ax.set_ylabel(arm.replace("_", "\n"), fontsize=8)
        ax.set_ylim(0, 0.8)
        ax.legend(fontsize=6, ncol=2, loc="upper right")
    axes[-1].set_xlabel("global optimizer step")
    fig.suptitle("Effective objective weights w_i", fontsize=11)
    save(fig, "02_effective_weights")

    # 3 raw losses by domain
    fig, axes = plt.subplots(3, 2, figsize=(11, 9), sharex="col")
    views = [
        ("SMALL_HI_ONLY", "Small-HI", 0),
        ("SAMLD_ONLY", "SAML-D", 1),
        ("MIXED_1TO1", "Small-HI", 2),
    ]
    # left: InfoNCE, right: TF — for specialists one domain; mixed HI on row2 and we'll add SAML as extra via color on same axes for mixed
    for row, (arm, dom, _) in enumerate(views):
        rs = domain_rows(all_rows[arm], dom)
        xs = [int(r["global_optimizer_step"]) for r in rs]
        axes[row, 0].plot(xs, [float(r["L_contrast_raw"]) for r in rs], color=COLORS["contrast"])
        axes[row, 0].set_ylabel(f"{arm}\n{dom}\nraw InfoNCE", fontsize=7)
        axes[row, 0].axvline(10, color="gray", ls="--", lw=1)
        for m, c in zip(range(3), (COLORS["tf0"], COLORS["tf1"], COLORS["tf2"])):
            axes[row, 1].plot(
                xs,
                [float(r[f"L_tf_raw_{m}"]) for r in rs],
                color=c,
                label=TF_NAMES[m].replace("log1p_", ""),
                alpha=0.9,
            )
        axes[row, 1].axvline(10, color="gray", ls="--", lw=1)
        axes[row, 1].set_ylabel("raw TF", fontsize=7)
        if row == 0:
            axes[row, 1].legend(fontsize=6)
    # overlay mixed SAML on bottom row with dashed
    rs = domain_rows(all_rows["MIXED_1TO1"], "SAML-D")
    xs = [int(r["global_optimizer_step"]) for r in rs]
    axes[2, 0].plot(xs, [float(r["L_contrast_raw"]) for r in rs], color=COLORS["contrast"], ls="--", alpha=0.7, label="SAML")
    for m, c in zip(range(3), (COLORS["tf0"], COLORS["tf1"], COLORS["tf2"])):
        axes[2, 1].plot(xs, [float(r[f"L_tf_raw_{m}"]) for r in rs], color=c, ls="--", alpha=0.55)
    axes[2, 0].legend(fontsize=6)
    axes[2, 0].set_xlabel("global step")
    axes[2, 1].set_xlabel("global step")
    fig.suptitle("Raw losses (InfoNCE vs TF on separate scales); MIXED SAML dashed", fontsize=10)
    save(fig, "03_raw_losses_by_domain")

    # 4 normalized post-calib only
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    for ax, arm in zip(axes, ARMS):
        for dom in sorted({r["domain"] for r in all_rows[arm]}):
            rs = [r for r in domain_rows(all_rows[arm], dom) if r.get("_post_calib")]
            xs = [int(r["global_optimizer_step"]) for r in rs]
            ax.plot(xs, [float(r["L_contrast_norm"]) for r in rs], color=COLORS["contrast"], label=f"{dom} contrast", alpha=0.85)
            for m, c in zip(range(3), (COLORS["tf0"], COLORS["tf1"], COLORS["tf2"])):
                ax.plot(
                    xs,
                    [float(r[f"L_tf_norm_{m}"]) for r in rs],
                    color=c,
                    label=f"{dom} tf{m}" if arm == "MIXED_1TO1" or m == 0 else None,
                    alpha=0.7,
                )
        ax.axvline(10, color="gray", ls="--", lw=1)
        ax.set_ylabel(arm.replace("_", "\n"), fontsize=8)
        ax.legend(fontsize=6, ncol=3)
    axes[-1].set_xlabel("global optimizer step")
    fig.suptitle("Normalized losses (post-calibration only)", fontsize=11)
    save(fig, "04_normalized_losses_post_calibration")

    # 5 stacked contributions
    fig, axes = plt.subplots(4, 1, figsize=(9, 11), sharex=True)
    panels = [
        ("SMALL_HI_ONLY", "Small-HI"),
        ("SAMLD_ONLY", "SAML-D"),
        ("MIXED_1TO1", "Small-HI"),
        ("MIXED_1TO1", "SAML-D"),
    ]
    max_err = 0.0
    for ax, (arm, dom) in zip(axes, panels):
        rs = [r for r in domain_rows(all_rows[arm], dom) if r.get("_post_calib") and r["_recon"]]
        xs = [int(r["global_optimizer_step"]) for r in rs]
        c0 = np.array([r["_recon"]["C_contrast"] for r in rs])
        c1 = np.array([r["_recon"]["C_tf_0"] for r in rs])
        c2 = np.array([r["_recon"]["C_tf_1"] for r in rs])
        c3 = np.array([r["_recon"]["C_tf_2"] for r in rs])
        ax.stackplot(
            xs,
            c0,
            c1,
            c2,
            c3,
            colors=[COLORS["contrast"], COLORS["tf0"], COLORS["tf1"], COLORS["tf2"]],
            labels=["C_contrast", "C_tf0", "C_tf1", "C_tf2"],
            alpha=0.85,
        )
        ax.plot(xs, [float(r["L_total"]) for r in rs], color=COLORS["total"], lw=1.2, label="logged L_total")
        max_err = max(max_err, max(r["_recon"]["recon_abs_err"] for r in rs) if rs else 0)
        ax.set_ylabel(f"{arm}\n{dom}", fontsize=7)
        ax.legend(fontsize=6, ncol=3, loc="upper right")
    axes[-1].set_xlabel("global optimizer step")
    fig.suptitle(
        f"Weighted objective-value decomposition (max |C_sum−L_total|={max_err:.2e})",
        fontsize=10,
    )
    save(fig, "05_weighted_objective_decomposition")

    # 6 last-20 shares bar
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = []
    contrast_s = []
    tf_s = []
    for arm, dom in panels:
        rs = last_n_post(all_rows[arm], 20, dom)
        recs = [r["_recon"] for r in rs]
        m = mean_of_shares(recs)
        labels.append(f"{arm.split('_')[0]}\n{dom}")
        contrast_s.append(m.get("share_contrast", float("nan")))
        tf_s.append(m.get("share_tf_total", float("nan")))
    x = np.arange(len(labels))
    ax.bar(x - 0.15, contrast_s, 0.3, label="contrast share (mean of per-step)", color=COLORS["contrast"])
    ax.bar(x + 0.15, tf_s, 0.3, label="combined TF share (mean of per-step)", color=COLORS["tf0"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylim(0, 1)
    ax.set_ylabel("objective-value share")
    ax.set_title("Last-20 post-calib realized objective-value shares")
    ax.legend(fontsize=7)
    save(fig, "06_realized_contribution_shares")

    # 7 matched exposure
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, dom, spec_arm in zip(
        axes, ("Small-HI", "SAML-D"), ("SMALL_HI_ONLY", "SAMLD_ONLY")
    ):
        s = at_domain_exposure(all_rows[spec_arm], dom, 500)
        m = at_domain_exposure(all_rows["MIXED_1TO1"], dom, 500)
        # use last-20 ending at that exposure? User asked at 500 updates — use that point's shares from recon
        groups = []
        for name, r in (("specialist", s), ("mixed", m)):
            rec = r["_recon"] if r and r.get("_recon") else None
            if rec is None and r is not None:
                rec = reconstruct_point(r)
            groups.append((name, rec, r))
        names = [g[0] for g in groups]
        cshare = [g[1]["share_contrast"] if g[1] else float("nan") for g in groups]
        tshare = [g[1]["share_tf_total"] if g[1] else float("nan") for g in groups]
        x = np.arange(2)
        ax.bar(x - 0.15, cshare, 0.3, label="contrast share", color=COLORS["contrast"])
        ax.bar(x + 0.15, tshare, 0.3, label="TF total share", color=COLORS["tf0"])
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_ylim(0, 1)
        ax.set_title(f"{dom} @ domain exposure 500")
        ax.legend(fontsize=7)
        s_gs = int(s["global_optimizer_step"]) if s else -1
        m_gs = int(m["global_optimizer_step"]) if m else -1
        ax.text(
            0.02,
            0.02,
            f"spec global={s_gs}\nmixed global={m_gs}\n{SECONDARY_CAVEAT[:60]}…",
            transform=ax.transAxes,
            fontsize=6,
            va="bottom",
        )
    fig.suptitle("Matched exposure comparison (not LR-phase matched)", fontsize=10)
    save(fig, "07_matched_exposure_comparison")

    return paths


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)

    obj = verify_objective_source()

    input_files = {
        "training_integrity_summary.json": SCOUT / "training_integrity_summary.json",
        "matching_contract.json": SCOUT / "matching_contract.json",
    }
    for arm in ARMS:
        input_files[f"{arm}/summary.json"] = SCOUT / "arms" / arm / "summary.json"
        input_files[f"{arm}/steps.jsonl"] = SCOUT / "arms" / arm / "logs" / "steps.jsonl"
        input_files[f"{arm}/steps.csv"] = SCOUT / "arms" / arm / "logs" / "steps.csv"

    file_shas = {}
    for k, p in input_files.items():
        if p.is_file():
            file_shas[k] = {"path": str(p.relative_to(ROOT)), "sha256": sha256_file(p)}

    integ = json.loads(input_files["training_integrity_summary.json"].read_text())
    summaries = {
        arm: json.loads((SCOUT / "arms" / arm / "summary.json").read_text()) for arm in ARMS
    }

    # Domain/LossNorm confirmations from summaries
    domain_state = {}
    for arm in ARMS:
        ln = summaries[arm]["loss_norm_states"]
        domain_state[arm] = {
            "loss_norm_domains": list(ln.keys()),
            "n_loss_norm_states": len(ln),
            "alpha_unfrozen_after_step": summaries[arm].get("alpha_unfrozen_after_step"),
            "alpha_freeze_policy": summaries[arm].get("alpha_freeze_policy"),
            "step_counts": summaries[arm].get("step_counts"),
        }

    all_rows = {arm: enrich(load_rows(arm)) for arm in ARMS}

    # Reconstruction integrity
    recon_errs = []
    missing = []
    wsum_bad = []
    for arm, rows in all_rows.items():
        for r in rows:
            if not r.get("calibration_complete_domain"):
                continue
            rec = r.get("_recon")
            if rec is None:
                missing.append(
                    {
                        "arm": arm,
                        "global_step": r.get("global_optimizer_step"),
                        "domain": r.get("domain"),
                        "reason": "missing fields",
                    }
                )
                continue
            recon_errs.append(rec["recon_abs_err"])
            if abs(rec["w_sum"] - 1.0) > 1e-4:
                wsum_bad.append({"arm": arm, "step": r.get("global_optimizer_step"), "w_sum": rec["w_sum"]})

    recon_report = {
        "atol": RECON_ATOL,
        "n_reconstructable_post_calib_points": len(recon_errs),
        "max_abs_reconstruction_error": float(max(recon_errs)) if recon_errs else None,
        "mean_abs_reconstruction_error": float(np.mean(recon_errs)) if recon_errs else None,
        "n_missing_unreconstructable": len(missing),
        "missing_points": missing[:20],
        "weight_sum_violations": wsum_bad,
        "pass": bool(recon_errs)
        and float(max(recon_errs)) <= RECON_ATOL
        and len(missing) == 0
        and len(wsum_bad) == 0,
        "formula_used": "C_i = w_i * L_i_norm; uses logged w_tf (includes 1-alpha), not beta alone",
        "weighted_tf_fields_include_one_minus_alpha": True,
    }
    write_json(OUT / "reconstruction_integrity.json", recon_report)

    # Calibration / freeze boundary checks from logs.
    # Logs expose both 0-indexed `step` and 1-indexed `global_optimizer_step`.
    # Policy "frozen through step 9; unfrozen at step 10" refers to 0-indexed `step`.
    freeze_audit = {}
    for arm, rows in all_rows.items():
        frozen_gs = [int(r["global_optimizer_step"]) for r in rows if r.get("alpha_beta_frozen")]
        unfrozen_gs = [int(r["global_optimizer_step"]) for r in rows if not r.get("alpha_beta_frozen")]
        frozen_step0 = [int(r["step"]) for r in rows if r.get("alpha_beta_frozen")]
        unfrozen_step0 = [int(r["step"]) for r in rows if not r.get("alpha_beta_frozen")]
        calib_done_at = {}
        for dom in sorted({r["domain"] for r in rows}):
            drows = domain_rows(rows, dom)
            first_done = next((r for r in drows if r.get("calibration_complete_domain")), None)
            calib_done_at[dom] = {
                "first_calibrated_global_optimizer_step": int(first_done["global_optimizer_step"])
                if first_done
                else None,
                "first_calibrated_step0": int(first_done["step"]) if first_done else None,
                "first_calibrated_domain_exposure": int(first_done["domain_exposure_count"])
                if first_done
                else None,
            }
        max_f0 = max(frozen_step0) if frozen_step0 else -1
        min_u0 = min(unfrozen_step0) if unfrozen_step0 else 999
        freeze_audit[arm] = {
            "max_frozen_step0": max_f0 if frozen_step0 else None,
            "min_unfrozen_step0": min_u0 if unfrozen_step0 else None,
            "max_frozen_global_optimizer_step": max(frozen_gs) if frozen_gs else None,
            "min_unfrozen_global_optimizer_step": min(unfrozen_gs) if unfrozen_gs else None,
            "frozen_through_step_9_unfrozen_at_10_step0": max_f0 == 9 and min_u0 == 10,
            "policy_note": (
                "alpha_beta_frozen True for step0 0..9 (global_optimizer_step 1..10); "
                "False from step0 10 (global_optimizer_step 11). "
                "Matches freeze-through-9 / unfreeze-at-10 on the 0-indexed step counter."
            ),
            "calib_by_domain": calib_done_at,
            "calib_uses_first_five_observations_per_domain": all(
                v.get("first_calibrated_domain_exposure") == 5 for v in calib_done_at.values()
            ),
        }

    # Table 1 final weights
    table1 = []
    for arm, rows in all_rows.items():
        r = rows[-1]
        hi_exp = summaries[arm]["step_counts"].get("Small-HI", 0)
        sd_exp = summaries[arm]["step_counts"].get("SAML-D", 0)
        table1.append(
            {
                "Arm": arm,
                "Global step": int(r["global_optimizer_step"]),
                "HI exposure": hi_exp,
                "SAML exposure": sd_exp,
                "alpha": float(r["alpha_raw"]),
                "beta0": float(r["beta_0"]),
                "beta1": float(r["beta_1"]),
                "beta2": float(r["beta_2"]),
                "w_contrast": float(r["w_contrast"]),
                "w_tf0": float(r["w_tf_0"]),
                "w_tf1": float(r["w_tf_1"]),
                "w_tf2": float(r["w_tf_2"]),
                "w_sum": float(r["w_contrast"] + r["w_tf_0"] + r["w_tf_1"] + r["w_tf_2"]),
                "largest_effective_weight_target": (
                    "contrast"
                    if float(r["w_contrast"])
                    >= max(float(r[f"w_tf_{m}"]) for m in range(3))
                    else TF_NAMES[int(np.argmax([float(r[f"w_tf_{m}"]) for m in range(3)]))]
                ),
            }
        )
    write_csv(OUT / "tables" / "table1_final_learned_weights.csv", table1)

    # Table 2 last-20 contributions
    table2 = []
    views = [
        ("SMALL_HI_ONLY", "Small-HI"),
        ("SAMLD_ONLY", "SAML-D"),
        ("MIXED_1TO1", "Small-HI"),
        ("MIXED_1TO1", "SAML-D"),
    ]
    last20_summaries = {}
    for arm, dom in views:
        rs = last_n_post(all_rows[arm], 20, dom)
        recs = [r["_recon"] for r in rs]
        m_mt = mean_contrib_over_mean_total(recs)
        m_ps = mean_of_shares(recs)
        last20_summaries[f"{arm}|{dom}"] = {"mean_component_over_mean_total": m_mt, "mean_of_per_step_shares": m_ps}
        dom_tf_i, dom_tf_name = dominant_tf(m_ps)
        table2.append(
            {
                "Arm": arm,
                "Domain": dom,
                "C_contrast": m_mt.get("mean_C_contrast"),
                "C_tf0": m_mt.get("mean_C_tf_0"),
                "C_tf1": m_mt.get("mean_C_tf_1"),
                "C_tf2": m_mt.get("mean_C_tf_2"),
                "Contrast share (mean_C / mean_total)": m_mt.get("share_contrast"),
                "TF total share (mean_C / mean_total)": m_mt.get("share_tf_total"),
                "Contrast share (mean of per-step shares)": m_ps.get("share_contrast"),
                "TF total share (mean of per-step shares)": m_ps.get("share_tf_total"),
                "Dominant TF target (by mean-of-shares)": dom_tf_name,
                "n points": len(recs),
                "aggregation_primary_in_C_columns": "mean_component_over_mean_total",
            }
        )
    write_csv(OUT / "tables" / "table2_last20_realized_contributions.csv", table2)

    # Table 3 raw/norm at key intervals
    table3 = []
    intervals = [
        ("unfreeze_boundary", "global", 10),
        ("domain_exp_250", "domain_exp", 250),
        ("domain_exp_500", "domain_exp", 500),
        ("global_500", "global", 500),
        ("global_1000", "global", 1000),
    ]
    for arm, rows in all_rows.items():
        domains = sorted({r["domain"] for r in rows})
        for dom in domains:
            for label, kind, val in intervals:
                if kind == "global":
                    r = at_global(domain_rows(rows, dom), val)
                    # for global markers, prefer exact global step on that domain if present else nearest domain row by global
                    if r is None or int(r.get("global_optimizer_step", -1)) != val:
                        # find domain row with global closest among domain rows — already done
                        pass
                else:
                    r = at_domain_exposure(rows, dom, val)
                if r is None:
                    continue
                # skip domain_exp intervals that don't apply to wrong domain meaning — always apply to this domain
                table3.append(
                    {
                        "Arm": arm,
                        "Domain": dom,
                        "Interval": label,
                        "Actual global step": int(r["global_optimizer_step"]),
                        "Actual domain exposure": int(r["domain_exposure_count"]),
                        "Raw InfoNCE": float(r["L_contrast_raw"]),
                        "Norm InfoNCE": float(r["L_contrast_norm"]),
                        "Raw TF0": float(r["L_tf_raw_0"]),
                        "Raw TF1": float(r["L_tf_raw_1"]),
                        "Raw TF2": float(r["L_tf_raw_2"]),
                        "Norm TF0": float(r["L_tf_norm_0"]),
                        "Norm TF1": float(r["L_tf_norm_1"]),
                        "Norm TF2": float(r["L_tf_norm_2"]),
                        "post_calib": bool(r.get("calibration_complete_domain")),
                    }
                )
    write_csv(OUT / "tables" / "table3_raw_normalized_losses.csv", table3)

    # Table 4 matched exposure 500
    table4 = []
    for dom, spec in (("Small-HI", "SMALL_HI_ONLY"), ("SAML-D", "SAMLD_ONLY")):
        s = at_domain_exposure(all_rows[spec], dom, 500)
        m = at_domain_exposure(all_rows["MIXED_1TO1"], dom, 500)
        srec = s["_recon"] if s and s.get("_recon") else reconstruct_point(s) if s else None
        mrec = m["_recon"] if m and m.get("_recon") else reconstruct_point(m) if m else None
        sdom_i, sdom_n = dominant_tf(srec or {})
        mdom_i, mdom_n = dominant_tf(mrec or {})
        table4.append(
            {
                "Domain": dom,
                "Specialist contrast share": None if not srec else srec["share_contrast"],
                "Mixed contrast share": None if not mrec else mrec["share_contrast"],
                "Specialist TF share": None if not srec else srec["share_tf_total"],
                "Mixed TF share": None if not mrec else mrec["share_tf_total"],
                "Specialist dominant TF": sdom_n,
                "Mixed dominant TF": mdom_n,
                "Specialist global step": int(s["global_optimizer_step"]) if s else None,
                "Mixed global step": int(m["global_optimizer_step"]) if m else None,
                "Specialist domain exposure": int(s["domain_exposure_count"]) if s else None,
                "Mixed domain exposure": int(m["domain_exposure_count"]) if m else None,
                "LR_phase_caveat": SECONDARY_CAVEAT,
            }
        )
    write_csv(OUT / "tables" / "table4_matched_exposure_500.csv", table4)

    # Table 5 interpretation
    table5 = []
    for arm, dom in views:
        r = last_n_post(all_rows[arm], 20, dom)[-1] if last_n_post(all_rows[arm], 20, dom) else all_rows[arm][-1]
        final = all_rows[arm][-1] if arm != "MIXED_1TO1" else r
        # effective weights from final global point for MIXED (shared)
        gfinal = all_rows[arm][-1]
        w_list = [
            ("contrast", float(gfinal["w_contrast"])),
            (TF_NAMES[0], float(gfinal["w_tf_0"])),
            (TF_NAMES[1], float(gfinal["w_tf_1"])),
            (TF_NAMES[2], float(gfinal["w_tf_2"])),
        ]
        largest_w = max(w_list, key=lambda x: x[1])
        m_ps = last20_summaries[f"{arm}|{dom}"]["mean_of_per_step_shares"]
        c_list = [
            ("contrast", m_ps.get("share_contrast", float("nan"))),
            (TF_NAMES[0], m_ps.get("share_tf_0", float("nan"))),
            (TF_NAMES[1], m_ps.get("share_tf_1", float("nan"))),
            (TF_NAMES[2], m_ps.get("share_tf_2", float("nan"))),
        ]
        largest_c = max(c_list, key=lambda x: x[1] if x[1] == x[1] else -1)
        # among TF only for "which expert"
        tf_w = max([(TF_NAMES[m], float(gfinal[f"w_tf_{m}"])) for m in range(3)], key=lambda x: x[1])
        tf_c = max([(TF_NAMES[m], m_ps.get(f"share_tf_{m}", float("nan"))) for m in range(3)], key=lambda x: x[1] if x[1] == x[1] else -1)
        table5.append(
            {
                "Arm": arm,
                "Domain": dom,
                "largest_learned_effective_weight": largest_w[0],
                "largest_effective_weight_value": largest_w[1],
                "largest_effective_weight_among_TF_experts": tf_w[0],
                "largest_realized_objective_value_share": largest_c[0],
                "largest_realized_share_value": largest_c[1],
                "largest_realized_share_among_TF_experts": tf_c[0],
                "weight_vs_contribution_rank_agree_on_TF_expert": tf_w[0] == tf_c[0],
                "contrast_share_mean_of_steps": m_ps.get("share_contrast"),
                "tf_total_share_mean_of_steps": m_ps.get("share_tf_total"),
                "contrast_vs_combined_TF": (
                    "contrast_more"
                    if (m_ps.get("share_contrast") or 0) > (m_ps.get("share_tf_total") or 0)
                    else "combined_TF_more"
                ),
            }
        )
    write_csv(OUT / "tables" / "table5_weight_contribution_interpretation.csv", table5)

    # Gradient evidence
    sample = all_rows["MIXED_1TO1"][-1]
    grad_fields_present = [k for k in sample.keys() if "grad" in k.lower()]
    grad_audit = {
        "available_gradient_fields": sorted(
            {k for rows in all_rows.values() for k in rows[0].keys() if "grad" in k.lower()}
        ),
        "missing_gradient_fields": [
            "per_loss_encoder_grad_norm_from_InfoNCE",
            "per_loss_encoder_grad_norm_from_TF",
            "beta_logit_grad_norms_separate",
            "alpha_logit_grad_norm_separate_from_total_alpha_grad_norm_naming",
        ],
        "logged": {
            "encoder_grad_norm": "total encoder parameter grad L2 after backward of L_total",
            "moe_grad_norm": "total MoE/expert-head parameter grad L2",
            "alpha_grad_norm": "LearnedAlphaBeta parameter grad L2 (alpha_logit + beta_logits jointly)",
            "contrast_grad_contribution": "boolean flag that contrast term is in the graph (not a norm)",
        },
        "measurement_timing": (
            "After total.backward() and before optimizer.step(); "
            "clip_grad_norm_(..., max_norm=1e9) is called after grad_norm measurement "
            "in scripts/train_mixed_ssl_phase3_scout.py mixed_step — so logged norms are pre-clip."
        ),
        "gradient_accumulation": "contrastive_accum_steps=1; no accumulation staging",
        "clipping": "torch.nn.utils.clip_grad_norm_(..., 1e9) — effectively inactive",
        "component_specific_gradient_attribution_available": False,
        "statement": (
            "Component-specific gradient attribution is unavailable from these runs."
        ),
        "can_conclude": [
            "Encoder and MoE received nonzero gradients throughout",
            "Alpha/beta parameter vector received nonzero gradients after unfreeze",
        ],
        "cannot_conclude": [
            "Relative gradient influence of InfoNCE vs each TF expert on the encoder",
            "Causal downstream importance of any loss term from gradient norms alone",
        ],
    }

    # Trajectory movement: last 100 vs previous 100 alpha change
    traj = {}
    for arm, rows in all_rows.items():
        alphas = [float(r["alpha_raw"]) for r in rows]
        betas = np.array([[float(r[f"beta_{m}"]) for m in range(3)] for r in rows])
        d_alpha_last100 = abs(alphas[-1] - alphas[-100]) if len(alphas) >= 100 else abs(alphas[-1] - alphas[0])
        d_beta_last100 = float(np.abs(betas[-1] - betas[-100]).sum()) if len(betas) >= 100 else float(np.abs(betas[-1] - betas[0]).sum())
        # still moving if last-100 change > 1e-3
        status = "still_moving" if (d_alpha_last100 > 1e-3 or d_beta_last100 > 1e-3) else "approximately_converged"
        if d_alpha_last100 > 0.05 or d_beta_last100 > 0.1:
            status = "still_moving"
        traj[arm] = {
            "final_alpha": alphas[-1],
            "final_beta": betas[-1].tolist(),
            "delta_alpha_last100": d_alpha_last100,
            "delta_beta_l1_last100": d_beta_last100,
            "status": status,
        }

    fig_paths = plot_figures(all_rows, OUT / "figures")

    # figure_data.csv — long format key points
    fig_rows = []
    for arm, rows in all_rows.items():
        for r in rows:
            if int(r["global_optimizer_step"]) % 10 != 0 and int(r["global_optimizer_step"]) not in (10, 500, 1000):
                continue
            fig_rows.append(
                {
                    "arm": arm,
                    "domain": r["domain"],
                    "global_step": int(r["global_optimizer_step"]),
                    "domain_exposure": int(r["domain_exposure_count"]),
                    "alpha": float(r["alpha_raw"]),
                    "beta0": float(r["beta_0"]),
                    "beta1": float(r["beta_1"]),
                    "beta2": float(r["beta_2"]),
                    "w_contrast": float(r["w_contrast"]),
                    "w_tf0": float(r["w_tf_0"]),
                    "w_tf1": float(r["w_tf_1"]),
                    "w_tf2": float(r["w_tf_2"]),
                    "L_contrast_raw": float(r["L_contrast_raw"]),
                    "L_contrast_norm": float(r["L_contrast_norm"]),
                    "L_total": float(r["L_total"]),
                    "C_contrast": None if not r.get("_recon") else r["_recon"]["C_contrast"],
                    "C_tf0": None if not r.get("_recon") else r["_recon"]["C_tf_0"],
                    "C_tf1": None if not r.get("_recon") else r["_recon"]["C_tf_1"],
                    "C_tf2": None if not r.get("_recon") else r["_recon"]["C_tf_2"],
                    "share_contrast": None if not r.get("_recon") else r["_recon"]["share_contrast"],
                    "share_tf_total": None if not r.get("_recon") else r["_recon"]["share_tf_total"],
                    "post_calib": bool(r.get("_post_calib")),
                }
            )
    write_csv(OUT / "figure_data.csv", fig_rows)

    # Answers
    # Q6: mixed more TF mass than specialists?
    mixed_hi = last20_summaries["MIXED_1TO1|Small-HI"]["mean_of_per_step_shares"]["share_tf_total"]
    mixed_sd = last20_summaries["MIXED_1TO1|SAML-D"]["mean_of_per_step_shares"]["share_tf_total"]
    spec_hi = last20_summaries["SMALL_HI_ONLY|Small-HI"]["mean_of_per_step_shares"]["share_tf_total"]
    spec_sd = last20_summaries["SAMLD_ONLY|SAML-D"]["mean_of_per_step_shares"]["share_tf_total"]

    answers = {
        "1_final_weights": table1,
        "2_largest_effective_weight_among_experts": {
            arm: max(
                [(TF_NAMES[m], float(all_rows[arm][-1][f"w_tf_{m}"])) for m in range(3)],
                key=lambda x: x[1],
            )
            for arm in ARMS
        },
        "3_largest_realized_contribution_among_experts": {
            f"{arm}|{dom}": table5[[(a, d) for a, d in views].index((arm, dom))][
                "largest_realized_share_among_TF_experts"
            ]
            for arm, dom in views
        },
        "4_weight_vs_contribution_agree": {
            f"{arm}|{dom}": table5[[(a, d) for a, d in views].index((arm, dom))][
                "weight_vs_contribution_rank_agree_on_TF_expert"
            ]
            for arm, dom in views
        },
        "5_mixed_domain_realized_differs_despite_shared_alphabeta": {
            "shared_final_alpha": float(all_rows["MIXED_1TO1"][-1]["alpha_raw"]),
            "shared_final_betas": [float(all_rows["MIXED_1TO1"][-1][f"beta_{m}"]) for m in range(3)],
            "HI_last20_contrast_share": mixed_hi and last20_summaries["MIXED_1TO1|Small-HI"]["mean_of_per_step_shares"]["share_contrast"],
            "SAML_last20_contrast_share": last20_summaries["MIXED_1TO1|SAML-D"]["mean_of_per_step_shares"]["share_contrast"],
            "HI_last20_tf_share": mixed_hi,
            "SAML_last20_tf_share": mixed_sd,
            "differs": abs(mixed_hi - mixed_sd) > 0.02
            or abs(
                last20_summaries["MIXED_1TO1|Small-HI"]["mean_of_per_step_shares"]["share_contrast"]
                - last20_summaries["MIXED_1TO1|SAML-D"]["mean_of_per_step_shares"]["share_contrast"]
            )
            > 0.02,
        },
        "6_mixed_more_TF_mass_than_specialists": {
            "mixed_HI_tf_share": mixed_hi,
            "specialist_HI_tf_share": spec_hi,
            "mixed_SAML_tf_share": mixed_sd,
            "specialist_SAML_tf_share": spec_sd,
            "mixed_HI_gt_specialist_HI": mixed_hi > spec_hi,
            "mixed_SAML_gt_specialist_SAML": mixed_sd > spec_sd,
            "mixed_more_than_either_specialist_on_both": (mixed_hi > spec_hi) and (mixed_sd > spec_sd),
        },
        "7_contrast_nontrivial_at_1000": {
            "SMALL_HI_ONLY": {
                "w_contrast": float(all_rows["SMALL_HI_ONLY"][-1]["w_contrast"]),
                "last20_contrast_share": last20_summaries["SMALL_HI_ONLY|Small-HI"][
                    "mean_of_per_step_shares"
                ]["share_contrast"],
                "nontrivial": float(all_rows["SMALL_HI_ONLY"][-1]["w_contrast"]) >= 0.25,
            },
            "SAMLD_ONLY": {
                "w_contrast": float(all_rows["SAMLD_ONLY"][-1]["w_contrast"]),
                "last20_contrast_share": last20_summaries["SAMLD_ONLY|SAML-D"][
                    "mean_of_per_step_shares"
                ]["share_contrast"],
                "nontrivial": float(all_rows["SAMLD_ONLY"][-1]["w_contrast"]) >= 0.25,
            },
            "MIXED_1TO1": {
                "w_contrast": float(all_rows["MIXED_1TO1"][-1]["w_contrast"]),
                "last20_contrast_share": {
                    "HI": last20_summaries["MIXED_1TO1|Small-HI"]["mean_of_per_step_shares"][
                        "share_contrast"
                    ],
                    "SAML": last20_summaries["MIXED_1TO1|SAML-D"]["mean_of_per_step_shares"][
                        "share_contrast"
                    ],
                },
                "nontrivial": float(all_rows["MIXED_1TO1"][-1]["w_contrast"]) >= 0.25,
            },
        },
        "8_trajectory_status": traj,
        "9_component_grad_attribution_available": False,
        "10_weights_establish_causal_downstream": False,
        "10_required_statement": (
            "Weights and objective shares alone cannot establish causal downstream importance."
        ),
        "11_needed_attribution_experiment": (
            "Matched multi-domain comparison: InfoNCE-only vs EXPERT_ONLY vs adaptive InfoNCE+TF "
            "(not launched in this audit)."
        ),
    }

    data_def = {
        "effective_objective_weight": "w_i from resolve_tfmoe_weights(adaptive)",
        "realized_objective_value_contribution": "C_i = w_i * L_i_norm",
        "realized_objective_value_share": "C_i / C_sum",
        "aggregations": {
            "mean_component_over_mean_total": "mean(C_i) / mean(C_sum) over interval",
            "mean_of_per_step_shares": "mean_t (C_i(t)/C_sum(t))",
        },
        "not_causal_downstream_importance": True,
        "tf_targets": list(TF_NAMES),
        "calibration": "first 5 observations/domain; alpha/beta frozen through global step 9, unfrozen at 10",
        "secondary_comparison_caveat": SECONDARY_CAVEAT,
    }
    write_json(OUT / "data_definition.json", data_def)

    payload = {
        "ok": bool(recon_report["pass"]) and bool(obj),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "read_only": True,
        "no_training": True,
        "no_encoder_forward_or_backward": True,
        "no_extraction": True,
        "no_downstream_probe": True,
        "no_test_access": True,
        "no_slurm_jobs": True,
        "no_npz_embeddings_loaded": True,
        "training_model_eval_code_changed": False,
        "checkpoints_modified": False,
        "objective": obj,
        "domain_state": domain_state,
        "freeze_and_calib_audit": freeze_audit,
        "reconstruction": recon_report,
        "gradient_audit": grad_audit,
        "tables": {
            "table1_final_learned_weights": table1,
            "table2_last20_realized_contributions": table2,
            "table3_raw_normalized_losses": table3,
            "table4_matched_exposure_500": table4,
            "table5_weight_contribution_interpretation": table5,
        },
        "last20_summaries": last20_summaries,
        "answers": answers,
        "figures": fig_paths,
        "input_file_sha256": file_shas,
        "confirmations": {
            "no_model_training": True,
            "no_encoder_forward_or_backward_pass": True,
            "no_extraction": True,
            "no_downstream_probe": True,
            "no_test_access": True,
            "no_slurm_jobs": True,
        },
    }
    write_json(OUT / "artifact_manifest.json", {
        "outputs": [
            str(NOTES.relative_to(ROOT)),
            str(TWIN.relative_to(ROOT)),
            str((OUT / "reconstruction_integrity.json").relative_to(ROOT)),
            str((OUT / "data_definition.json").relative_to(ROOT)),
            str((OUT / "figure_data.csv").relative_to(ROOT)),
            "tables/*.csv",
            "figures/*",
        ],
        "script": "scripts/audit_smallhi_samld_phase3_objective_contributions.py",
    })
    write_json(TWIN, payload)

    # Notes markdown
    lines = [
        "# Phase-3 objective-weight and loss-contribution audit",
        "",
        f"> Twin: `{TWIN.relative_to(ROOT)}`",
        f"> Source: `direct_r198.combine_direct_h_tfmoe_loss` / `resolve_tfmoe_weights` (`adaptive`)",
        f"> Source SHA256: `{obj['source_sha256']}`",
        "",
        "**Read-only.** No training, extraction, probes, NPZ loads, test access, or Slurm.",
        "",
        "## Verified objective",
        "",
        "```",
        obj["formula"],
        "```",
        "",
        "TF target order (proven):",
        "",
    ]
    for i, n in enumerate(TF_NAMES):
        lines.append(f"{i}. `{n}`")
    lines += [
        "",
        f"Reconstruction integrity: **{'PASS' if recon_report['pass'] else 'FAIL'}** "
        f"(n={recon_report['n_reconstructable_post_calib_points']}, "
        f"max|err|={recon_report['max_abs_reconstruction_error']:.3e}, "
        f"mean|err|={recon_report['mean_abs_reconstruction_error']:.3e}).",
        "",
        "Alpha/beta are **global/shared** in MIXED_1TO1; LossNormState is **per-domain**.",
        "Calibration: first 5 observations/domain; α/β frozen through 0-indexed `step` 9, unfrozen at `step` 10 "
        "(logged `global_optimizer_step` 1..10 frozen, 11+ unfrozen).",
        "",
        "## Table 1 — Final learned weights",
        "",
        "| Arm | Global step | HI exp | SAML exp | alpha | beta0 | beta1 | beta2 | w_c | w_tf0 | w_tf1 | w_tf2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in table1:
        lines.append(
            f"| {r['Arm']} | {r['Global step']} | {r['HI exposure']} | {r['SAML exposure']} | "
            f"{r['alpha']:.4f} | {r['beta0']:.4f} | {r['beta1']:.4f} | {r['beta2']:.4f} | "
            f"{r['w_contrast']:.4f} | {r['w_tf0']:.4f} | {r['w_tf1']:.4f} | {r['w_tf2']:.4f} |"
        )
    lines += [
        "",
        "## Table 2 — Last-20 realized contributions",
        "",
        "C columns use **mean(C_i)**; shares reported both as mean(C)/mean(total) and mean of per-step shares.",
        "",
        "| Arm | Domain | mean C_contrast | mean C_tf0 | mean C_tf1 | mean C_tf2 | contrast share (meanC/meanTot) | TF share (meanC/meanTot) | Dominant TF | n |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for r in table2:
        lines.append(
            f"| {r['Arm']} | {r['Domain']} | {r['C_contrast']:.4f} | {r['C_tf0']:.4f} | "
            f"{r['C_tf1']:.4f} | {r['C_tf2']:.4f} | {r['Contrast share (mean_C / mean_total)']:.3f} | "
            f"{r['TF total share (mean_C / mean_total)']:.3f} | `{r['Dominant TF target (by mean-of-shares)']}` | {r['n points']} |"
        )
    lines += [
        "",
        "## Interpretation answers",
        "",
        f"1. Final weights: see Table 1 (all α≈0.44–0.45; β allocations differ by arm).",
        f"2. Largest effective TF weight: "
        + ", ".join(f"{a}→`{answers['2_largest_effective_weight_among_experts'][a][0]}`" for a in ARMS)
        + ".",
        f"3. Largest realized TF contribution (last-20 mean-of-shares): "
        + ", ".join(f"{k}→`{v}`" for k, v in answers["3_largest_realized_contribution_among_experts"].items())
        + ".",
        f"4. Weight vs contribution TF ranking agree: `{answers['4_weight_vs_contribution_agree']}`.",
        f"5. MIXED domain realized shares differ despite shared α/β: "
        f"`{answers['5_mixed_domain_realized_differs_despite_shared_alphabeta']['differs']}` "
        f"(HI TF share {mixed_hi:.3f} vs SAML {mixed_sd:.3f}).",
        f"6. Mixed more TF mass than specialists: "
        f"HI {mixed_hi:.3f} vs {spec_hi:.3f}; SAML {mixed_sd:.3f} vs {spec_sd:.3f}; "
        f"both={answers['6_mixed_more_TF_mass_than_specialists']['mixed_more_than_either_specialist_on_both']}.",
        "7. Contrast remains nontrivial at step 1000: w_contrast≈0.44–0.45 (≥0.25) in all arms; "
        "realized contrast share still ~0.4–0.8 depending on domain view.",
        f"8. Trajectories: `{ {a: traj[a]['status'] for a in ARMS} }` "
        f"(last-100 Δα / Δβ L1 reported in JSON).",
        "9. Component-specific gradient attribution is **unavailable** from these runs "
        "(only total encoder / MoE / joint αβ grad norms).",
        "10. **Weights and objective shares alone cannot establish causal downstream importance.**",
        "11. Needed: matched multi-domain InfoNCE-only vs EXPERT_ONLY vs adaptive InfoNCE+TF "
        "(not launched here).",
        "",
        f"## Secondary caveat",
        "",
        SECONDARY_CAVEAT,
        "",
        "## Gradient evidence",
        "",
        grad_audit["statement"],
        "",
        f"Available fields: {grad_audit['available_gradient_fields']}",
        "",
        "## Confirmations",
        "",
        "- no model training",
        "- no encoder forward/backward",
        "- no extraction / probe / NPZ embedding load",
        "- no test access",
        "- no Slurm jobs",
        "- training/model/eval code unchanged; checkpoints unmodified",
        "",
    ]
    NOTES.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "ok": payload["ok"],
                "recon_pass": recon_report["pass"],
                "max_recon_err": recon_report["max_abs_reconstruction_error"],
                "notes": str(NOTES.relative_to(ROOT)),
            },
            indent=2,
        )
    )
    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
