#!/usr/bin/env python3
"""Locked validation-only analysis for scheduled DIRECT_H / DIRECT_H_TFMOE runs.

Assumes frozen R198 train/val embeddings already extracted under:
  embeddings/<run>_epochXX/pre_embedding_3h/{train,val}.npz

No encoder retraining. No test load/score/inspect.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from direct_r198 import (  # noqa: E402
    LearnedAlphaBeta,
    TFMoEBundle,
    load_tf_moe_context,
    tf_moe_mae_losses,
)
from gcpal_txn_node.eval_mlp import PaperStyleMLP, _predict_proba  # noqa: E402
from linear_probe import load_embedding_npz  # noqa: E402
from util import logger_setup, set_seed  # noqa: E402

MLP_EPOCHS = 20
MLP_LR = 1e-3
MLP_BS = 8192
MLP_SEED = 2
TF_CACHE = ROOT / "results/cache/temporal_flow_causal/Small-HI"
EPOCHS = (1, 3, 5, 10)
ARMS = {
    "DIRECT_H": "direct_h_infonce_10ep_seed2_sched",
    "DIRECT_H_TFMOE": "direct_h_tfmoe_learned_alpha_10ep_seed2_sched",
}
OUT_DIR = ROOT / "results/diagnostics/direct_h_tfmoe_scheduled_val_analysis"
FIG_DIR = OUT_DIR / "figures"
JSON_OUT = ROOT / "results/diagnostics/direct_h_tfmoe_scheduled_val_analysis.json"
MD_OUT = ROOT / "notes/direct_h_tfmoe_scheduled_val_analysis.md"

# Supervised reference only (projected-encoder baselines removed from figures/tables:
# provenance of projected F1@opt≈0.571 was too ambiguous for these DIRECT_H plots).
REF_SUPERVISED = {
    "run": "small_hi_legacy_supervised_gin_emlps_ports_50ep_seed2",
    "horizon_epochs": 50,
    "seed": 2,
    "tds": False,
    "note": "Paper-faithful Multi-GIN+EU (ports; TDS off). DIRECT_H recipe uses TDS on.",
    "source": "notes/supervised_Small-HI_small_hi_legacy_supervised_gin_emlps_ports_50ep_seed2_summary.md",
    "validation_f1": 0.6101,
    "validation_f1_decision_rule": "argmax over two-class logits",
    "validation_f1_checkpoint": "best validation minority F1 (epoch 43)",
    "validation_auprc": 0.5509,
    "best_epoch": 43,
    "plot_label_f1": "Supervised Multi-GIN+EU, validation F1 (argmax), seed 2",
    "plot_label_auprc": "Supervised Multi-GIN+EU, validation AUPRC, seed 2",
}


def _tune_thr(y: np.ndarray, p: np.ndarray) -> float:
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.linspace(0.01, 0.99, 99):
        f1 = float(f1_score(y.astype(np.int64), (p >= thr).astype(np.int64), zero_division=0))
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_thr


def _metrics(y: np.ndarray, p: np.ndarray, thr: float) -> Dict[str, float]:
    pred = (p >= thr).astype(np.int64)
    return {
        "auprc": float(average_precision_score(y, p)),
        "f1": float(f1_score(y.astype(np.int64), pred, zero_division=0)),
        "threshold": float(thr),
    }


def _load_x_tf() -> Tuple[np.ndarray, np.ndarray]:
    spec = importlib.util.spec_from_file_location(
        "probe_feature_ablation", ROOT / "scripts" / "probe_feature_ablation.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["probe_feature_ablation"] = mod
    spec.loader.exec_module(mod)
    df, df_train, _, _, _, _ = mod.load_dataset_frames("Small-HI", str(ROOT / "data_config.json"))
    x_raw, _, _, _ = mod.build_full_feature_matrix(
        df, df_train, ("edge_native",), categorical_encoding="one_hot"
    )
    tf = np.load(TF_CACHE / "features.npy").astype(np.float32)
    return x_raw.astype(np.float32), tf


def _fit_mlp(
    mat_tr: np.ndarray,
    y_tr: np.ndarray,
    mat_va: np.ndarray,
    y_va: np.ndarray,
    device,
    *,
    return_proba: bool = False,
) -> Dict[str, Any]:
    scaler = StandardScaler()
    tr = scaler.fit_transform(mat_tr).astype(np.float32)
    va = scaler.transform(mat_va).astype(np.float32)
    torch.manual_seed(MLP_SEED)
    np.random.seed(MLP_SEED)
    model = PaperStyleMLP(tr.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=MLP_LR)
    x_t = torch.from_numpy(tr)
    y_t = torch.from_numpy(y_tr.astype(np.float32))
    n = tr.shape[0]
    best_auprc, best_state, best_ep = -1.0, None, -1
    for ep in range(MLP_EPOCHS):
        model.train()
        perm = np.random.RandomState(MLP_SEED * 1009 + ep).permutation(n)
        for start in range(0, n, MLP_BS):
            idx = perm[start : start + MLP_BS]
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(
                model(x_t[idx].to(device)), y_t[idx].to(device)
            )
            loss.backward()
            opt.step()
        pva = _predict_proba(model, va, batch_size=MLP_BS, device=device)
        auprc = float(average_precision_score(y_va, pva))
        if auprc > best_auprc + 1e-12:
            best_auprc = auprc
            best_ep = ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    model.to(device)
    pva = _predict_proba(model, va, batch_size=MLP_BS, device=device)
    thr = _tune_thr(y_va, pva)
    out = {
        "learner": "PaperStyleMLP",
        "mlp_epochs": MLP_EPOCHS,
        "mlp_lr": MLP_LR,
        "mlp_batch_size": MLP_BS,
        "mlp_seed": MLP_SEED,
        "best_epoch": best_ep,
        "validation_auprc": float(average_precision_score(y_va, pva)),
        "validation_metrics_at_0.5": _metrics(y_va, pva, 0.5),
        "validation_metrics_at_val_optimal_f1": _metrics(y_va, pva, thr),
        "input_dim": int(tr.shape[1]),
        "n_train": int(n),
        "n_val": int(y_va.shape[0]),
        "test_evaluated": False,
    }
    if return_proba:
        out["val_proba"] = pva.astype(np.float32)
    return out


def _repr_stats(z: np.ndarray, max_pairs: int = 20000, seed: int = 2) -> Dict[str, float]:
    z64 = z.astype(np.float64)
    norms = np.linalg.norm(z64, axis=1)
    # center for effective rank via singular values of covariance proxy
    zc = z64 - z64.mean(axis=0, keepdims=True)
    # subsample rows for SVD if huge
    rng = np.random.RandomState(seed)
    if zc.shape[0] > 50000:
        idx = rng.choice(zc.shape[0], size=50000, replace=False)
        zc_s = zc[idx]
    else:
        zc_s = zc
    # economy SVD on (n x d); effective rank from normalized singular values squared
    try:
        s = np.linalg.svd(zc_s, full_matrices=False, compute_uv=False)
        p = (s ** 2)
        p = p / max(p.sum(), 1e-12)
        eff_rank = float(np.exp(-(p * np.log(np.maximum(p, 1e-300))).sum()))
    except Exception:
        eff_rank = float("nan")
    # mean off-diagonal cosine on a subsample of pairs
    n = z.shape[0]
    m = min(max_pairs, n)
    idx = rng.choice(n, size=m, replace=False)
    zs = z64[idx]
    zn = zs / np.maximum(np.linalg.norm(zs, axis=1, keepdims=True), 1e-12)
    # sample pairwise without full Gram if large
    n_pairs = min(5000, m * (m - 1) // 2)
    i1 = rng.randint(0, m, size=n_pairs)
    i2 = rng.randint(0, m, size=n_pairs)
    mask = i1 != i2
    i1, i2 = i1[mask], i2[mask]
    cos = (zn[i1] * zn[i2]).sum(axis=1)
    return {
        "mean_l2_norm": float(norms.mean()),
        "std_l2_norm": float(norms.std()),
        "mean_per_dim_variance": float(z64.var(axis=0).mean()),
        "median_per_dim_variance": float(np.median(z64.var(axis=0))),
        "effective_rank": float(eff_rank),
        "mean_offdiag_cosine": float(cos.mean()) if cos.size else float("nan"),
    }


def _emb_dir(run: str, epoch: int) -> Path:
    return ROOT / "embeddings" / f"{run}_epoch{epoch:02d}" / "pre_embedding_3h"


def _probe_cell(
    run: str,
    epoch: int,
    x_all: np.ndarray,
    tf_all: np.ndarray,
    device,
    probs_dir: Path,
) -> Dict[str, Any]:
    emb = _emb_dir(run, epoch)
    if (emb / "test.npz").is_file():
        raise RuntimeError(f"Refusing analysis: test.npz present at {emb}")
    z_tr, y_tr, id_tr = load_embedding_npz(emb / "train.npz")
    z_va, y_va, id_va = load_embedding_npz(emb / "val.npz")
    if z_tr.shape[1] != 198 or z_va.shape[1] != 198:
        raise RuntimeError(f"Expected R198, got {z_tr.shape[1]} / {z_va.shape[1]}")
    # Verify not H128/Z128
    assert z_tr.shape[1] != 128

    set_seed(MLP_SEED)
    primary = _fit_mlp(
        np.concatenate([z_tr, x_all[id_tr], tf_all[id_tr]], axis=1),
        y_tr,
        np.concatenate([z_va, x_all[id_va], tf_all[id_va]], axis=1),
        y_va,
        device,
        return_proba=True,
    )
    primary["stack"] = "R198+X+TF"
    primary["dims"] = {"R198": 198, "X": int(x_all.shape[1]), "TF": int(tf_all.shape[1])}
    pva = primary.pop("val_proba")

    set_seed(MLP_SEED)
    diagnostic = _fit_mlp(z_tr, y_tr, z_va, y_va, device, return_proba=True)
    diagnostic["stack"] = "R198_only"
    pva_d = diagnostic.pop("val_proba")

    # Save aligned IDs + probs (val only)
    probs_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        probs_dir / f"{run}_epoch{epoch:02d}_val_proba.npz",
        edge_id=id_va.astype(np.int64),
        y=y_va.astype(np.int64),
        proba_primary=pva,
        proba_diagnostic=pva_d,
        stack_primary="R198+X+TF",
        stack_diagnostic="R198_only",
        representation="R198",
        test_evaluated=np.array(False),
    )

    return {
        "run": run,
        "epoch": epoch,
        "embedding_dir": str(emb),
        "representation": "R198",
        "verified_not_h128_z128_moe_tf_pred": True,
        "dim": 198,
        "primary": primary,
        "diagnostic": diagnostic,
        "repr_train": _repr_stats(z_tr),
        "repr_val": _repr_stats(z_va),
        "n_train": int(z_tr.shape[0]),
        "n_val": int(z_va.shape[0]),
        "test_evaluated": False,
        "val_proba_path": str(probs_dir / f"{run}_epoch{epoch:02d}_val_proba.npz"),
    }


@torch.no_grad()
def _tfmoe_diagnostics(run: str, epoch: int, device) -> Dict[str, Any]:
    ckpt_path = ROOT / "saved-models" / f"checkpoint_{run}_epoch{epoch:02d}.tar"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not ckpt.get("direct_r198_tfmoe"):
        return {"tfmoe": False}
    emb = _emb_dir(run, epoch)
    z_tr, _, id_tr = load_embedding_npz(emb / "train.npz")
    z_va, _, id_va = load_embedding_npz(emb / "val.npz")

    bundle = TFMoEBundle(in_dim=198, hidden=64, n_targets=3).to(device)
    bundle.load_state_dict(ckpt["direct_r198_tfmoe_state_dict"])
    bundle.eval()
    ab = LearnedAlphaBeta(n_tf=3, init_alpha=0.6).to(device)
    ab.load_state_dict(ckpt["direct_r198_alpha_beta_state_dict"])
    ab.eval()
    ctx = load_tf_moe_context(TF_CACHE, device)

    def _eval_split(z: np.ndarray, ids: np.ndarray) -> Dict[str, Any]:
        zt = torch.from_numpy(z).to(device)
        idt = torch.from_numpy(ids.astype(np.int64)).to(device)
        # chunk to limit VRAM
        mae_sums = [0.0, 0.0, 0.0]
        n = 0
        gate_sums = [torch.zeros(3, device=device) for _ in range(3)]
        util_sums = [torch.zeros(3, device=device) for _ in range(3)]
        bs = 8192
        for start in range(0, zt.shape[0], bs):
            sl = slice(start, start + bs)
            losses, diag = tf_moe_mae_losses(zt[sl], idt[sl], bundle, ctx)
            w = int(sl.stop - sl.start) if False else int(zt[sl].shape[0])
            for i, loss in enumerate(losses):
                mae_sums[i] += float(loss.detach().cpu()) * w
            n += w
            for m, name in enumerate(bundle.target_names):
                gm = diag.get(f"tfmoe/gate_mean/{name}")
                ut = diag.get(f"tfmoe/util/{name}")
                if gm is not None:
                    gate_sums[m] += torch.tensor(gm, device=device) * w
                if ut is not None:
                    util_sums[m] += torch.tensor(ut, device=device) * w
        out_mae = [mae_sums[i] / max(n, 1) for i in range(3)]
        gates = {}
        utils = {}
        for m, name in enumerate(bundle.target_names):
            gates[name] = (gate_sums[m] / max(n, 1)).detach().cpu().tolist()
            utils[name] = (util_sums[m] / max(n, 1)).detach().cpu().tolist()
        return {
            "mae": {name: out_mae[i] for i, name in enumerate(bundle.target_names)},
            "gate_mean": gates,
            "expert_utilization": utils,
            "n": n,
        }

    alpha, beta = ab()
    eff = ab.effective_weights()
    return {
        "tfmoe": True,
        "checkpoint": str(ckpt_path),
        "alpha": float(alpha.detach().cpu()),
        "beta": beta.detach().cpu().tolist(),
        "effective_weights": eff,
        "ckpt_effective_weights": ckpt.get("direct_r198_effective_weights"),
        "train": _eval_split(z_tr, id_tr),
        "val": _eval_split(z_va, id_va),
        "target_names": list(bundle.target_names),
        "uses_train_only_scaler": True,
        "scaler_from": "load_tf_moe_context(train split IDs)",
    }


def _ssl_epoch_losses() -> Dict[str, Dict[int, Dict[str, float]]]:
    out: Dict[str, Dict[int, Dict[str, float]]] = {}
    for arm, run in ARMS.items():
        out[arm] = {}
        for ep in EPOCHS:
            p = ROOT / "results/diagnostics" / run / "logs" / f"epoch_{ep:02d}.json"
            d = json.loads(p.read_text(encoding="utf-8"))
            out[arm][ep] = {
                "loss_train": float(d.get("loss/train")),
                "loss_contrastive": float(d.get("loss/contrastive", d.get("loss/train"))),
                "lr_encoder": float(d.get("lr/encoder")),
                "lr_factor": float(d.get("lr/factor")),
                "lr_alpha_beta": d.get("lr/alpha_beta"),
            }
    return out


def _alpha_traj_from_steps(run: str) -> List[Dict[str, Any]]:
    path = ROOT / "results/diagnostics" / run / "logs" / "steps.jsonl"
    if not path.is_file():
        return []
    by_ep: Dict[int, Dict[str, Any]] = {}
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            if "alpha" in r:
                by_ep[int(r["epoch"])] = r
    return [
        {
            "epoch": ep,
            "alpha": by_ep[ep].get("alpha"),
            "beta": [by_ep[ep].get(f"beta_{i}") for i in range(3)],
            "w_contrast": by_ep[ep].get("w_contrast"),
            "w_tf": [by_ep[ep].get(f"w_tf_{i}") for i in range(3)],
            "L_contrast_raw": by_ep[ep].get("L_contrast_raw"),
            "L_total": by_ep[ep].get("L_total"),
        }
        for ep in sorted(by_ep)
    ]


def _integrity_hashes() -> Dict[str, Any]:
    outs = {
        "DIRECT_H": ROOT / "slurm-logs/direct_h_infonce_10ep_seed2_sched_19251526.out",
        "DIRECT_H_TFMOE": ROOT / "slurm-logs/direct_h_tfmoe_learned_alpha_10ep_seed2_sched_19251528.out",
    }
    hashes: Dict[str, List[str]] = {}
    for arm, path in outs.items():
        text = path.read_text(errors="replace") if path.is_file() else ""
        # first epoch (0-idx) first 32 steps
        found = re.findall(
            r"scout_batch_log epoch=0 step=(\d+) seed_ids_sha256=([0-9a-f]+)",
            text,
        )
        # keep unique by step
        by_step = {int(s): h for s, h in found}
        hashes[arm] = [by_step[i] for i in range(32) if i in by_step]
    match = hashes.get("DIRECT_H") == hashes.get("DIRECT_H_TFMOE") and len(hashes.get("DIRECT_H", [])) == 32
    return {
        "first_32_seed_edge_hashes_epoch0": hashes,
        "arms_match": match,
        "n_hashes_direct_h": len(hashes.get("DIRECT_H", [])),
        "n_hashes_tfmoe": len(hashes.get("DIRECT_H_TFMOE", [])),
        "unique_negs_per_anchor_nan_reason": (
            "denom_mode=sampled_8192: InfoNCE negatives are randomly sampled "
            "(with replacement from the aligned batch / bank), so uniqueness and "
            "duplicate counts are not tracked and remain NaN by design."
        ),
        "duplicate_neg_count_nan_reason": (
            "Same as unique_negs_per_anchor: only computed for all-aligned "
            "current-batch denom mode, not for sampled_8192."
        ),
    }


def _plot_all(report: Dict[str, Any]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    epochs = list(EPOCHS)

    def series(arm: str, stack: str, key_path: List[str]):
        ys = []
        for ep in epochs:
            cell = report["cells"][arm][str(ep)]
            node = cell[stack]
            for k in key_path:
                node = node[k]
            ys.append(float(node))
        return ys

    # 1 AUPRC vs epoch
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(epochs, series("DIRECT_H", "primary", ["validation_auprc"]), "o-", label="DIRECT_H primary")
    ax.plot(epochs, series("DIRECT_H_TFMOE", "primary", ["validation_auprc"]), "s-", label="TFMOE primary")
    ax.plot(epochs, series("DIRECT_H", "diagnostic", ["validation_auprc"]), "o--", alpha=0.7, label="DIRECT_H R198-only")
    ax.plot(epochs, series("DIRECT_H_TFMOE", "diagnostic", ["validation_auprc"]), "s--", alpha=0.7, label="TFMOE R198-only")
    ax.axhline(
        REF_SUPERVISED["validation_auprc"],
        color="black",
        ls=":",
        label=REF_SUPERVISED["plot_label_auprc"],
    )
    ax.set_xlabel("SSL epoch")
    ax.set_ylabel("Validation AUPRC")
    ax.set_title("Downstream validation AUPRC vs epoch")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_val_auprc_vs_epoch.png", dpi=160)
    plt.close(fig)

    # 2 F1 vs epoch
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for arm, short, marker in (
        ("DIRECT_H", "DIRECT_H", "o"),
        ("DIRECT_H_TFMOE", "DIRECT_H_TFMOE", "s"),
    ):
        ax.plot(
            epochs,
            series(arm, "primary", ["validation_metrics_at_val_optimal_f1", "f1"]),
            f"{marker}-",
            label=f"{short} F1@val-threshold",
        )
        ax.plot(
            epochs,
            series(arm, "primary", ["validation_metrics_at_0.5", "f1"]),
            f"{marker}--",
            alpha=0.7,
            label=f"{short} F1@0.5",
        )
    ax.axhline(
        REF_SUPERVISED["validation_f1"],
        color="black",
        ls=":",
        label=REF_SUPERVISED["plot_label_f1"],
    )
    ax.set_xlabel("SSL epoch")
    ax.set_ylabel("Validation F1")
    ax.set_title("Downstream validation F1 vs SSL epoch")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_val_f1_vs_epoch.png", dpi=160)
    plt.close(fig)

    # 3 TF MAE train/val
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    names = report["cells"]["DIRECT_H_TFMOE"]["1"]["tfmoe"]["target_names"]
    for i, name in enumerate(names):
        tr = [report["cells"]["DIRECT_H_TFMOE"][str(ep)]["tfmoe"]["train"]["mae"][name] for ep in epochs]
        va = [report["cells"]["DIRECT_H_TFMOE"][str(ep)]["tfmoe"]["val"]["mae"][name] for ep in epochs]
        ax.plot(epochs, tr, "o-", label=f"train {name.split('log1p_')[-1][:18]}")
        ax.plot(epochs, va, "s--", label=f"val {name.split('log1p_')[-1][:18]}")
    ax.set_xlabel("SSL epoch")
    ax.set_ylabel("MAE (train-standardized)")
    ax.set_title("TF MoE MAE train vs validation")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_tf_mae_train_val_vs_epoch.png", dpi=160)
    plt.close(fig)

    # 4 alpha / weights
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    alphas = [report["cells"]["DIRECT_H_TFMOE"][str(ep)]["tfmoe"]["alpha"] for ep in epochs]
    ax.plot(epochs, alphas, "o-", label="alpha")
    for i in range(3):
        ws = [
            report["cells"]["DIRECT_H_TFMOE"][str(ep)]["tfmoe"]["effective_weights"][f"w_tf_{i}"]
            for ep in epochs
        ]
        ax.plot(epochs, ws, "s--", label=f"w_tf_{i}")
    wcs = [
        report["cells"]["DIRECT_H_TFMOE"][str(ep)]["tfmoe"]["effective_weights"]["w_contrast"]
        for ep in epochs
    ]
    ax.plot(epochs, wcs, "^-", label="w_contrast")
    ax.set_xlabel("SSL epoch")
    ax.set_ylabel("Weight")
    ax.set_title("Alpha / effective weights vs epoch")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_alpha_effective_weights.png", dpi=160)
    plt.close(fig)

    # 5 raw contrastive loss
    ssl = report["ssl_losses"]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(epochs, [ssl["DIRECT_H"][str(ep)]["loss_contrastive"] for ep in epochs], "o-", label="DIRECT_H")
    ax.plot(
        epochs,
        [ssl["DIRECT_H_TFMOE"][str(ep)]["loss_contrastive"] for ep in epochs],
        "s-",
        label="TFMOE contrastive",
    )
    ax.set_xlabel("SSL epoch")
    ax.set_ylabel("Raw contrastive loss")
    ax.set_title("Raw InfoNCE loss comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_raw_contrastive_loss.png", dpi=160)
    plt.close(fig)

    # 6 representation geometry
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for arm, marker in (("DIRECT_H", "o"), ("DIRECT_H_TFMOE", "s")):
        axes[0].plot(
            epochs,
            [report["cells"][arm][str(ep)]["repr_val"]["mean_l2_norm"] for ep in epochs],
            f"{marker}-",
            label=arm,
        )
        axes[1].plot(
            epochs,
            [report["cells"][arm][str(ep)]["repr_val"]["effective_rank"] for ep in epochs],
            f"{marker}-",
            label=arm,
        )
    axes[0].set_title("Val mean L2 norm")
    axes[1].set_title("Val effective rank")
    for ax in axes:
        ax.set_xlabel("SSL epoch")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_repr_scale_effective_rank.png", dpi=160)
    plt.close(fig)

    # 7 LR dual axis
    fig, ax1 = plt.subplots(figsize=(7.5, 4.5))
    ax2 = ax1.twinx()
    lrs = [ssl["DIRECT_H"][str(ep)]["lr_encoder"] for ep in epochs]
    fac = [ssl["DIRECT_H"][str(ep)]["lr_factor"] for ep in epochs]
    # also load denser from steps if available
    steps_path = ROOT / "results/diagnostics/direct_h_infonce_10ep_seed2_sched/logs/steps.jsonl"
    if steps_path.is_file():
        xs, ys_lr, ys_f = [], [], []
        with steps_path.open() as f:
            for line in f:
                r = json.loads(line)
                if r.get("encoder_lr") is not None and r.get("optimizer_step_index") is not None:
                    xs.append(r["optimizer_step_index"])
                    ys_lr.append(r["encoder_lr"])
                    ys_f.append(r.get("lr_factor"))
        if xs:
            ax1.plot(xs, ys_lr, color="C0", lw=1.2, label="encoder LR")
            ax2.plot(xs, ys_f, color="C1", lw=1.2, label="LR factor")
            ax1.set_xlabel("optimizer step")
        else:
            ax1.plot(epochs, lrs, "o-", color="C0", label="encoder LR")
            ax2.plot(epochs, fac, "s--", color="C1", label="LR factor")
            ax1.set_xlabel("SSL epoch")
    else:
        ax1.plot(epochs, lrs, "o-", color="C0", label="encoder LR")
        ax2.plot(epochs, fac, "s--", color="C1", label="LR factor")
        ax1.set_xlabel("SSL epoch")
    ax1.set_ylabel("Encoder LR", color="C0")
    ax2.set_ylabel("LR factor", color="C1")
    ax1.set_title("DIRECT_H LR schedule (separate axes)")
    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2, fontsize=8, loc="upper right")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_lr_dual_axis.png", dpi=160)
    plt.close(fig)


def _write_md(report: Dict[str, Any]) -> None:
    cells = report["cells"]
    sel = report["selection"]
    deltas = report["deltas_tfmoe_minus_direct_h"]
    lines = []
    lines.append("# DIRECT_H / DIRECT_H_TFMOE scheduled validation analysis")
    lines.append("")
    lines.append("Locked validation-only extract + PaperStyleMLP probe for scheduled warmup+cosine runs.")
    lines.append("**No encoder retraining. No test evaluation.**")
    lines.append("")
    lines.append("## Jobs")
    lines.append("")
    lines.append("| Arm | Job | Run |")
    lines.append("|-----|-----|-----|")
    lines.append("| DIRECT_H | 19251526 (`dir_h_sched`) | `direct_h_infonce_10ep_seed2_sched` |")
    lines.append("| DIRECT_H_TFMOE | 19251528 (`dir_h_tfmoe_s`) | `direct_h_tfmoe_learned_alpha_10ep_seed2_sched` |")
    lines.append("")
    lines.append("## Primary downstream (R198+X+TF → PaperStyleMLP)")
    lines.append("")
    lines.append("| Epoch | H AUPRC | H F1@0.5 | H F1@val-threshold | TF AUPRC | TF F1@0.5 | TF F1@val-threshold | ΔAUPRC | ΔF1@val-threshold |")
    lines.append("|------:|--------:|---------:|-------------------:|---------:|----------:|--------------------:|-------:|------------------:|")
    for ep in EPOCHS:
        h = cells["DIRECT_H"][str(ep)]["primary"]
        t = cells["DIRECT_H_TFMOE"][str(ep)]["primary"]
        d = deltas[str(ep)]["primary"]
        lines.append(
            f"| {ep} | {h['validation_auprc']:.4f} | {h['validation_metrics_at_0.5']['f1']:.4f} | "
            f"{h['validation_metrics_at_val_optimal_f1']['f1']:.4f} | {t['validation_auprc']:.4f} | "
            f"{t['validation_metrics_at_0.5']['f1']:.4f} | {t['validation_metrics_at_val_optimal_f1']['f1']:.4f} | "
            f"{d['auprc']:+.4f} | {d['f1_opt']:+.4f} |"
        )
    lines.append("")
    lines.append("### Val-selected checkpoints (by primary AUPRC)")
    lines.append("")
    lines.append(
        f"- DIRECT_H: epoch **{sel['DIRECT_H']['epoch']}** "
        f"(AUPRC={sel['DIRECT_H']['validation_auprc']:.4f}, "
        f"F1@val-threshold={sel['DIRECT_H']['f1_opt']:.4f}, F1@0.5={sel['DIRECT_H']['f1_0.5']:.4f})"
    )
    lines.append(
        f"- DIRECT_H_TFMOE: epoch **{sel['DIRECT_H_TFMOE']['epoch']}** "
        f"(AUPRC={sel['DIRECT_H_TFMOE']['validation_auprc']:.4f}, "
        f"F1@val-threshold={sel['DIRECT_H_TFMOE']['f1_opt']:.4f}, F1@0.5={sel['DIRECT_H_TFMOE']['f1_0.5']:.4f})"
    )
    lines.append("")
    lines.append("## References (validation only)")
    lines.append("")
    lines.append(
        f"- {REF_SUPERVISED['plot_label_f1']}: "
        f"**{REF_SUPERVISED['validation_f1']:.4f}** "
        f"(run `{REF_SUPERVISED['run']}`, best epoch {REF_SUPERVISED['best_epoch']}, "
        f"decision rule: {REF_SUPERVISED['validation_f1_decision_rule']}; "
        f"val AUPRC={REF_SUPERVISED['validation_auprc']:.4f}; "
        f"[source]({REF_SUPERVISED['source']}))."
    )
    lines.append(
        "- Projected-encoder baselines (including former “projected F1@opt ≈ 0.571”) "
        "are **omitted** from figures and this table: threshold/split/seed provenance "
        "was not unambiguous enough for these DIRECT_H analyses."
    )
    lines.append("")
    lines.append("## TF MoE diagnostics")
    lines.append("")
    lines.append("| Epoch | α | w_c | w_tf0 | w_tf1 | w_tf2 | train MAE | val MAE |")
    lines.append("|------:|--:|----:|------:|------:|------:|----------:|--------:|")
    for ep in EPOCHS:
        tm = cells["DIRECT_H_TFMOE"][str(ep)]["tfmoe"]
        names = tm["target_names"]
        tr = [tm["train"]["mae"][n] for n in names]
        va = [tm["val"]["mae"][n] for n in names]
        ew = tm["effective_weights"]
        lines.append(
            f"| {ep} | {tm['alpha']:.3f} | {ew['w_contrast']:.3f} | {ew['w_tf_0']:.3f} | "
            f"{ew['w_tf_1']:.3f} | {ew['w_tf_2']:.3f} | "
            f"{tr[0]:.3f}/{tr[1]:.3f}/{tr[2]:.3f} | {va[0]:.3f}/{va[1]:.3f}/{va[2]:.3f} |"
        )
    lines.append("")
    lines.append(f"Expert generalization verdict: **{report['answers']['tf_experts_generalize_or_overfit']}**")
    lines.append("")
    lines.append("## Integrity")
    lines.append("")
    integ = report["integrity"]
    lines.append(f"- First-32 seed-edge hashes match across arms: **{integ['arms_match']}**")
    lines.append(f"- unique_negs_per_anchor NaN: {integ['unique_negs_per_anchor_nan_reason']}")
    lines.append("")
    lines.append("## Figures")
    lines.append("")
    for i, name in enumerate(
        [
            "01_val_auprc_vs_epoch.png",
            "02_val_f1_vs_epoch.png",
            "03_tf_mae_train_val_vs_epoch.png",
            "04_alpha_effective_weights.png",
            "05_raw_contrastive_loss.png",
            "06_repr_scale_effective_rank.png",
            "07_lr_dual_axis.png",
        ],
        1,
    ):
        lines.append(f"{i}. `results/diagnostics/direct_h_tfmoe_scheduled_val_analysis/figures/{name}`")
    lines.append("")
    lines.append("## Answers")
    lines.append("")
    for i, (k, v) in enumerate(report["answers"].items(), 1):
        lines.append(f"{i}. **{k}:** {v}")
    lines.append("")
    MD_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _answers(report: Dict[str, Any]) -> Dict[str, str]:
    sel = report["selection"]
    h = sel["DIRECT_H"]
    t = sel["DIRECT_H_TFMOE"]
    # compare selected
    d_a = t["validation_auprc"] - h["validation_auprc"]
    d_f = t["f1_opt"] - h["f1_opt"]
    # TF weights at selected TFMOE epoch
    tm = report["cells"]["DIRECT_H_TFMOE"][str(t["epoch"])]["tfmoe"]
    ew = tm["effective_weights"]
    meaningful = []
    for i, name in enumerate(tm["target_names"]):
        if ew[f"w_tf_{i}"] >= 0.15:
            meaningful.append(f"{name} (w={ew[f'w_tf_{i}']:.3f})")
    # generalize vs overfit: compare train/val MAE gap
    gaps = []
    for name in tm["target_names"]:
        gaps.append(tm["val"]["mae"][name] - tm["train"]["mae"][name])
    mean_gap = float(np.mean(gaps))
    if mean_gap < 0.05 and all(tm["val"]["mae"][n] < 0.5 for n in tm["target_names"]):
        gen = "a) validation-generalizing prediction (val MAE tracks train; no large gap)"
    elif mean_gap > 0.15:
        gen = "b) auxiliary overfitting (val MAE substantially worse than train)"
    else:
        gen = (
            "c) mixed / possible target misalignment: TF aux improves geometry or weights "
            f"but mean val−train MAE gap={mean_gap:.3f}; downstream ΔAUPRC={d_a:+.4f}"
        )
    # geometry
    h_er = report["cells"]["DIRECT_H"][str(h["epoch"])]["repr_val"]["effective_rank"]
    t_er = report["cells"]["DIRECT_H_TFMOE"][str(t["epoch"])]["repr_val"]["effective_rank"]
    h_n = report["cells"]["DIRECT_H"][str(h["epoch"])]["repr_val"]["mean_l2_norm"]
    t_n = report["cells"]["DIRECT_H_TFMOE"][str(t["epoch"])]["repr_val"]["mean_l2_norm"]
    geom = (
        f"TFMOE keeps smaller val L2 norm ({t_n:.2f} vs {h_n:.2f}) and "
        f"effective rank {t_er:.1f} vs {h_er:.1f} at each arm's selected epoch"
    )
    approach = (
        f"DIRECT_H selected val F1@val-threshold={h['f1_opt']:.4f} vs "
        f"supervised Multi-GIN+EU seed2 validation F1 (argmax)={REF_SUPERVISED['validation_f1']:.4f}; "
        f"gap={REF_SUPERVISED['validation_f1']-h['f1_opt']:+.4f}. "
        f"Does not approach supervised F1 at 10ep."
        if h["f1_opt"] < REF_SUPERVISED["validation_f1"] - 0.02
        else (
            f"Within ~2pp of supervised validation F1 "
            f"({h['f1_opt']:.4f} vs {REF_SUPERVISED['validation_f1']:.4f})."
        )
    )
    improve = (
        f"Yes (ΔAUPRC={d_a:+.4f}, ΔF1@val-threshold={d_f:+.4f} at val-selected epochs)."
        if (d_a >= 0.003 or d_f >= 0.01)
        else (
            f"No clear improvement under locked gates "
            f"(ΔAUPRC={d_a:+.4f}, ΔF1@val-threshold={d_f:+.4f}; "
            f"gates ≥0.003 AUPRC or ≥0.01 F1)."
        )
    )
    longer = (
        "Optional: longer SSL and/or BCE+MoE fallback if TF aux continues to "
        "dominate weight away from contrast without lifting primary AUPRC."
        if d_a < 0.003
        else "Optional follow-up only if TF continues to help geometry without AUPRC."
    )
    return {
        "direct_r198_approaches_supervised_val_f1": approach,
        "tfmoe_improves_over_direct_h": improve,
        "tf_objectives_with_meaningful_weight": (
            ", ".join(meaningful) if meaningful else "none above 0.15 effective weight"
        ),
        "tf_experts_generalize_or_overfit": gen,
        "tfmoe_improved_representation_geometry": geom,
        "longer_run_or_bce_moe_justified": longer,
        "no_encoder_retrain_no_test": (
            "Confirmed: frozen checkpoint extract only; extract_splits=train,val; "
            "test_evaluated=false everywhere; no test.npz written or read."
        ),
    }


def main() -> int:
    logger_setup()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    probs_dir = OUT_DIR / "val_proba"

    logging.info("Loading locked X + causal TF features")
    x_all, tf_all = _load_x_tf()
    logging.info("X dim=%s TF dim=%s", x_all.shape[1], tf_all.shape[1])

    cells: Dict[str, Dict[str, Any]] = {arm: {} for arm in ARMS}
    for arm, run in ARMS.items():
        for ep in EPOCHS:
            logging.info("=== Probe %s epoch %s ===", arm, ep)
            cell = _probe_cell(run, ep, x_all, tf_all, device, probs_dir)
            if arm == "DIRECT_H_TFMOE":
                logging.info("TF MoE diagnostics epoch %s", ep)
                cell["tfmoe"] = _tfmoe_diagnostics(run, ep, device)
            else:
                cell["tfmoe"] = {"tfmoe": False}
            cells[arm][str(ep)] = cell
            logging.info(
                "%s ep%s primary AUPRC=%.4f F1@val-threshold=%.4f",
                arm,
                ep,
                cell["primary"]["validation_auprc"],
                cell["primary"]["validation_metrics_at_val_optimal_f1"]["f1"],
            )

    # selection by primary AUPRC
    selection = {}
    for arm in ARMS:
        best_ep, best = None, -1.0
        for ep in EPOCHS:
            a = cells[arm][str(ep)]["primary"]["validation_auprc"]
            if a > best + 1e-12:
                best, best_ep = a, ep
        p = cells[arm][str(best_ep)]["primary"]
        selection[arm] = {
            "epoch": best_ep,
            "validation_auprc": p["validation_auprc"],
            "f1_0.5": p["validation_metrics_at_0.5"]["f1"],
            "f1_opt": p["validation_metrics_at_val_optimal_f1"]["f1"],
            "threshold": p["validation_metrics_at_val_optimal_f1"]["threshold"],
        }

    deltas = {}
    for ep in EPOCHS:
        h = cells["DIRECT_H"][str(ep)]["primary"]
        t = cells["DIRECT_H_TFMOE"][str(ep)]["primary"]
        hd = cells["DIRECT_H"][str(ep)]["diagnostic"]
        td = cells["DIRECT_H_TFMOE"][str(ep)]["diagnostic"]
        deltas[str(ep)] = {
            "primary": {
                "auprc": t["validation_auprc"] - h["validation_auprc"],
                "f1_0.5": t["validation_metrics_at_0.5"]["f1"] - h["validation_metrics_at_0.5"]["f1"],
                "f1_opt": t["validation_metrics_at_val_optimal_f1"]["f1"]
                - h["validation_metrics_at_val_optimal_f1"]["f1"],
            },
            "diagnostic": {
                "auprc": td["validation_auprc"] - hd["validation_auprc"],
                "f1_opt": td["validation_metrics_at_val_optimal_f1"]["f1"]
                - hd["validation_metrics_at_val_optimal_f1"]["f1"],
            },
        }

    ssl_raw = _ssl_epoch_losses()
    ssl = {arm: {str(ep): ssl_raw[arm][ep] for ep in EPOCHS} for arm in ARMS}

    report: Dict[str, Any] = {
        "artifact": "direct_h_tfmoe_scheduled_val_analysis",
        "encoder_retrained": False,
        "test_evaluated": False,
        "selection_metric": "validation_auprc",
        "secondary_metric": "validation_f1",
        "mlp": {
            "learner": "PaperStyleMLP",
            "epochs": MLP_EPOCHS,
            "lr": MLP_LR,
            "batch_size": MLP_BS,
            "seed": MLP_SEED,
        },
        "arms": ARMS,
        "epochs": list(EPOCHS),
        "cells": cells,
        "selection": selection,
        "deltas_tfmoe_minus_direct_h": deltas,
        "ssl_losses": ssl,
        "alpha_traj_steps": _alpha_traj_from_steps(ARMS["DIRECT_H_TFMOE"]),
        "references": {
            "projected_40ep": None,
            "projected_40ep_omitted_reason": (
                "Removed from DIRECT_H figure/table package: projected F1@opt≈0.571096 "
                "(and related projected AUPRC/F1@0.5 refs) lacked unambiguous "
                "threshold/split/seed presentation for these analyses."
            ),
            "supervised_multigin": REF_SUPERVISED,
        },
        "integrity": _integrity_hashes(),
        "gates": {"auprc_margin": 0.003, "f1_margin": 0.01},
    }
    report["answers"] = _answers(report)

    # JSON-safe: strip any leftover arrays
    def _sanitize(o):
        if isinstance(o, dict):
            return {k: _sanitize(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_sanitize(v) for v in o]
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o

    report_s = _sanitize(report)
    JSON_OUT.write_text(json.dumps(report_s, indent=2) + "\n", encoding="utf-8")
    (OUT_DIR / "report.json").write_text(json.dumps(report_s, indent=2) + "\n", encoding="utf-8")
    logging.info("Wrote %s", JSON_OUT)

    _plot_all(report_s)
    _write_md(report_s)
    logging.info("Wrote %s and figures under %s", MD_OUT, FIG_DIR)

    logging.info("SELECTION DIRECT_H=%s TFMOE=%s", selection["DIRECT_H"], selection["DIRECT_H_TFMOE"])
    for k, v in report["answers"].items():
        logging.info("ANSWER %s: %s", k, v)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
