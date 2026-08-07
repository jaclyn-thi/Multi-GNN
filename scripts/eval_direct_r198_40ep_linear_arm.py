#!/usr/bin/env python3
"""DIAGNOSTIC / PROVISIONAL seed-only extract + PaperStyleMLP for one DIRECT_R198 arm.

NOT the collaborator-facing protocol. Official metrics must use:
  python scripts/official_direct_r198_collaborator_eval.py ...

This script:
  - uses seed-only R198 extract
  - stamps protocol=seed_only / evaluation_tier=diagnostic_provisional
  - refuses to write into the official full-subgraph out/embeddings dirs
  - must not be merged into collaborator tables

Processes eval checkpoints one-at-a-time via subprocess extract so host RAM is released.
Never loads test.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from gcpal_txn_node.eval_mlp import PaperStyleMLP, _predict_proba  # noqa: E402
from direct_r198_eval_protocol import (  # noqa: E402
    PROTOCOL_SEED_ONLY,
    TIER_DIAGNOSTIC,
    diagnostic_seed_only_protocol_block,
    refuse_seed_only_write_into_official,
)

MLP_EPOCHS = 20
MLP_LR = 1e-3
MLP_BS = 8192
MLP_SEED = 2
EVAL_EPOCHS = (3, 10, 20, 30, 40)
TF_CACHE = ROOT / "results/cache/temporal_flow_causal/Small-HI"


def _sha256_file(path: Path, nbytes: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(nbytes)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _cell_complete(emb: Path) -> bool:
    return (
        (emb / "train.npz").is_file()
        and (emb / "val.npz").is_file()
        and (emb / "meta.json").is_file()
        and not (emb / "test.npz").is_file()
    )


def _extract_one(run: str, epoch: int) -> None:
    emb = ROOT / "embeddings" / f"{run}_epoch{epoch:02d}" / "pre_embedding_3h"
    if _cell_complete(emb):
        logging.info("Reuse existing embeddings for %s ep%02d", run, epoch)
        return
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    cmd = [
        sys.executable,
        str(ROOT / "scripts/extract_direct_r198_seed_only_cell.py"),
        "--run",
        run,
        "--epoch",
        str(epoch),
        "--splits",
        "train,val",
    ]
    logging.info("Subprocess extract: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(ROOT), env=env)


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


def _bce_np(logits: np.ndarray, y: np.ndarray) -> float:
    """Mean unweighted binary NLL from logits (matches BCE-with-logits, reduction=mean)."""
    t = torch.from_numpy(logits.astype(np.float32))
    yb = torch.from_numpy(y.astype(np.float32))
    return float(nn.functional.binary_cross_entropy_with_logits(t, yb).item())


def _metrics(y: np.ndarray, p: np.ndarray, thr: float) -> Dict[str, float]:
    pred = (p >= thr).astype(np.int64)
    return {
        "auprc": float(average_precision_score(y, p)),
        "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "threshold": float(thr),
    }


def _tune_thr(y: np.ndarray, p: np.ndarray) -> float:
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.linspace(0.01, 0.99, 99):
        f1 = float(f1_score(y.astype(np.int64), (p >= thr).astype(np.int64), zero_division=0))
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    return best_thr


def _fit_probe(
    mat_tr: np.ndarray,
    y_tr: np.ndarray,
    mat_va: np.ndarray,
    y_va: np.ndarray,
    device,
) -> Dict[str, Any]:
    """Train PaperStyleMLP for MLP_EPOCHS; metrics from **final** epoch weights."""
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
    history: List[Dict[str, float]] = []
    for ep in range(MLP_EPOCHS):
        model.train()
        perm = np.random.RandomState(MLP_SEED * 1009 + ep).permutation(n)
        last_batch_loss = None
        for start in range(0, n, MLP_BS):
            idx = perm[start : start + MLP_BS]
            opt.zero_grad(set_to_none=True)
            logits = model(x_t[idx].to(device))
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, y_t[idx].to(device)
            )
            loss.backward()
            opt.step()
            last_batch_loss = float(loss.detach().cpu())
        # Epoch-end full-split CE (unweighted BCE-with-logits mean)
        model.eval()
        with torch.no_grad():
            # accumulate train logits in batches
            tr_logits = []
            for start in range(0, n, MLP_BS):
                tr_logits.append(model(x_t[start : start + MLP_BS].to(device)).detach().cpu().numpy())
            tr_log = np.concatenate(tr_logits, axis=0)
            va_log = []
            xva = torch.from_numpy(va)
            for start in range(0, va.shape[0], MLP_BS):
                va_log.append(model(xva[start : start + MLP_BS].to(device)).detach().cpu().numpy())
            va_log_a = np.concatenate(va_log, axis=0)
        tr_ce = _bce_np(tr_log, y_tr)
        va_ce = _bce_np(va_log_a, y_va)
        pva = 1.0 / (1.0 + np.exp(-va_log_a))
        history.append(
            {
                "epoch": ep + 1,
                "train_bce": tr_ce,
                "val_bce": va_ce,
                "val_auprc": float(average_precision_score(y_va, pva)),
                "last_batch_bce": last_batch_loss if last_batch_loss is not None else float("nan"),
            }
        )
    # Final-epoch model (already at last state)
    pva = 1.0 / (1.0 + np.exp(-va_log_a))
    thr = _tune_thr(y_va, pva)
    return {
        "learner": "PaperStyleMLP",
        "loss": "binary_cross_entropy_with_logits",
        "logits": "one_logit",
        "class_weights": None,
        "pos_weight": None,
        "reduction": "mean",
        "mlp_epochs": MLP_EPOCHS,
        "mlp_lr": MLP_LR,
        "mlp_batch_size": MLP_BS,
        "mlp_seed": MLP_SEED,
        "selection_within_probe": "final_scheduled_epoch",
        "final_probe_train_bce": float(history[-1]["train_bce"]),
        "final_probe_val_bce": float(history[-1]["val_bce"]),
        "validation_auprc": float(average_precision_score(y_va, pva)),
        "validation_auroc": float(roc_auc_score(y_va, pva)),
        "validation_metrics_at_0.5": _metrics(y_va, pva, 0.5),
        "validation_metrics_at_val_optimal_f1": {
            **_metrics(y_va, pva, thr),
            "optimistic_diagnostic": True,
        },
        "epoch_history": history,
        "input_dim": int(tr.shape[1]),
        "n_train": int(n),
        "n_val": int(y_va.shape[0]),
        "test_evaluated": False,
    }


def _stack(z: np.ndarray, edge_id: np.ndarray, x_raw: np.ndarray, tf: np.ndarray, mode: str):
    eid = edge_id.astype(np.int64)
    if mode == "primary":
        return np.concatenate([z, x_raw[eid], tf[eid]], axis=1)
    if mode == "diagnostic":
        return z
    raise ValueError(mode)


def _repr_stats(z: np.ndarray) -> Dict[str, float]:
    # Match analyze_direct_h_tfmoe_scheduled_val effective-rank definition
    zn = z - z.mean(axis=0, keepdims=True)
    # covariance eigenvalues via SVD on subsample if huge
    n, d = zn.shape
    rs = np.random.RandomState(2)
    if n > 50000:
        idx = rs.choice(n, 50000, replace=False)
        zn = zn[idx]
    try:
        s = np.linalg.svd(zn, compute_uv=False)
        p = (s ** 2)
        p = p / max(p.sum(), 1e-12)
        eff = float(np.exp(-np.sum(p * np.log(np.maximum(p, 1e-12)))))
    except Exception:
        eff = float("nan")
    norms = np.linalg.norm(z, axis=1)
    return {
        "mean_l2_norm": float(norms.mean()),
        "std_l2_norm": float(norms.std()),
        "effective_rank": eff,
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--arm", required=True, choices=["DIRECT_R198", "DIRECT_R198_TFMOE"])
    ap.add_argument("--peak_lr", type=float, required=True)
    ap.add_argument(
        "--out_dir",
        type=str,
        default=str(ROOT / "results/diagnostics/direct_r198_tfmoe_40ep_linear_lr_sweep"),
    )
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    refuse_seed_only_write_into_official(
        out_dir=out_dir,
        embeddings_hint=None,
        root=ROOT,
    )
    # Also block if out_dir itself is under official path via absolute resolution
    cell_dir = out_dir / "cells" / args.run
    cell_dir.mkdir(parents=True, exist_ok=True)
    logging.warning(
        "SEED-ONLY diagnostic/provisional eval for %s — NOT collaborator protocol "
        "(use scripts/official_direct_r198_collaborator_eval.py for official metrics)",
        args.run,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info("Loading X+TF once (no test)")
    x_raw, tf = _load_x_tf()

    cells = []
    for ep in EVAL_EPOCHS:
        ckpt = ROOT / "saved-models" / f"checkpoint_{args.run}_epoch{ep:02d}.tar"
        if not ckpt.is_file():
            cells.append({"epoch": ep, "status": "missing_checkpoint", "checkpoint": str(ckpt)})
            logging.error("Missing checkpoint %s", ckpt)
            continue
        _extract_one(args.run, ep)
        emb = ROOT / "embeddings" / f"{args.run}_epoch{ep:02d}" / "pre_embedding_3h"
        if not _cell_complete(emb):
            cells.append({"epoch": ep, "status": "extract_incomplete", "embedding_dir": str(emb)})
            continue
        if (emb / "test.npz").is_file():
            raise RuntimeError(f"test.npz present: {emb}")
        tr = np.load(emb / "train.npz")
        va = np.load(emb / "val.npz")
        z_tr, y_tr, id_tr = tr["Z"], tr["y"].reshape(-1), tr["edge_id"].reshape(-1)
        z_va, y_va, id_va = va["Z"], va["y"].reshape(-1), va["edge_id"].reshape(-1)
        assert z_tr.shape[1] == 198 and z_va.shape[1] == 198
        coverage = {
            "n_train": int(z_tr.shape[0]),
            "n_val": int(z_va.shape[0]),
            "train_edge_id_sha256": hashlib.sha256(id_tr.astype(np.int64).tobytes()).hexdigest(),
            "val_edge_id_sha256": hashlib.sha256(id_va.astype(np.int64).tobytes()).hexdigest(),
            "checkpoint_sha256": _sha256_file(ckpt),
        }
        prim = _fit_probe(
            _stack(z_tr, id_tr, x_raw, tf, "primary"),
            y_tr,
            _stack(z_va, id_va, x_raw, tf, "primary"),
            y_va,
            device,
        )
        diag = _fit_probe(
            _stack(z_tr, id_tr, x_raw, tf, "diagnostic"),
            y_tr,
            _stack(z_va, id_va, x_raw, tf, "diagnostic"),
            y_va,
            device,
        )
        cell = {
            "status": "ok",
            "arm": args.arm,
            "run": args.run,
            "peak_lr": float(args.peak_lr),
            "schedule": "direct_h_warmup_linear",
            "epoch": ep,
            "checkpoint": str(ckpt),
            "embedding_dir": str(emb),
            "extractor": "extract_direct_r198_seed_only_cell",
            "protocol": PROTOCOL_SEED_ONLY,
            "evaluation_tier": TIER_DIAGNOSTIC,
            "collaborator_merge_allowed": False,
            "protocol_block": diagnostic_seed_only_protocol_block(),
            "seed_only_r198": True,
            "diagnostic_provisional": True,
            "coverage": coverage,
            "repr_val": _repr_stats(z_va.astype(np.float32)),
            "primary": prim,
            "diagnostic": diag,
            "test_evaluated": False,
        }
        (cell_dir / f"epoch_{ep:02d}.json").write_text(json.dumps(cell, indent=2) + "\n")
        cells.append(cell)
        logging.info(
            "OK ep=%s primary AUPRC=%.4f final_val_BCE=%.4f",
            ep,
            prim["validation_auprc"],
            prim["final_probe_val_bce"],
        )
        # Release large arrays for this checkpoint
        del tr, va, z_tr, z_va, y_tr, y_va, id_tr, id_va
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ok = [c for c in cells if c.get("status") == "ok"]
    selected = None
    if ok:
        selected = max(ok, key=lambda c: c["primary"]["validation_auprc"])
    summary = {
        "run": args.run,
        "arm": args.arm,
        "peak_lr": float(args.peak_lr),
        "protocol": PROTOCOL_SEED_ONLY,
        "evaluation_tier": TIER_DIAGNOSTIC,
        "collaborator_merge_allowed": False,
        "diagnostic_provisional": True,
        "warning": (
            "DIAGNOSTIC / PROVISIONAL seed-only results. Do not merge into collaborator "
            "tables. Official path: scripts/official_direct_r198_collaborator_eval.py"
        ),
        "cells": cells,
        "selected_ssl_epoch_by_primary_auprc": (
            None
            if selected is None
            else {
                "epoch": selected["epoch"],
                "validation_auprc": selected["primary"]["validation_auprc"],
                "f1_at_0.5": selected["primary"]["validation_metrics_at_0.5"]["f1"],
                "f1_at_val_thr": selected["primary"]["validation_metrics_at_val_optimal_f1"]["f1"],
                "final_probe_train_bce": selected["primary"]["final_probe_train_bce"],
                "final_probe_val_bce": selected["primary"]["final_probe_val_bce"],
            }
        ),
        "epoch40": next((c for c in ok if c["epoch"] == 40), None),
        "test_evaluated": False,
        "amp": False,
    }
    (cell_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"run": args.run, "n_ok": len(ok), "selected": summary["selected_ssl_epoch_by_primary_auprc"]}, indent=2))
    return 0 if len(ok) == len(EVAL_EPOCHS) else 2


if __name__ == "__main__":
    raise SystemExit(main())
