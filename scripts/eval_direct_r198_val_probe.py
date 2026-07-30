#!/usr/bin/env python3
"""Validation-only downstream probes for DIRECT_H / DIRECT_H_TFMOE scouts.

Primary: frozen R198 + locked X + causal TF -> PaperStyleMLP
Diagnostic: frozen R198 only -> PaperStyleMLP

No test access. Does not select by SSL loss.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gcpal_txn_node.eval_mlp import PaperStyleMLP, _predict_proba  # noqa: E402
from linear_probe import load_embedding_npz  # noqa: E402
from util import logger_setup, set_seed  # noqa: E402

MLP_EPOCHS = 20
MLP_LR = 1e-3
MLP_BS = 8192
MLP_SEED = 2
TF_CACHE = ROOT / "results/cache/temporal_flow_causal/Small-HI"


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
    df, df_train, _, _, _, dspec = mod.load_dataset_frames("Small-HI", str(ROOT / "data_config.json"))
    x_raw, _, _, _ = mod.build_full_feature_matrix(
        df, df_train, ("edge_native",), categorical_encoding="one_hot"
    )
    tf = np.load(TF_CACHE / "features.npy").astype(np.float32)
    return x_raw.astype(np.float32), tf


def _fit_mlp(mat_tr: np.ndarray, y_tr: np.ndarray, mat_va: np.ndarray, y_va: np.ndarray, device) -> Dict[str, Any]:
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
    return {
        "learner": "PaperStyleMLP",
        "best_epoch": best_ep,
        "validation_auprc": float(average_precision_score(y_va, pva)),
        "validation_metrics_at_0.5": _metrics(y_va, pva, 0.5),
        "validation_metrics_at_val_optimal_f1": _metrics(y_va, pva, thr),
        "input_dim": int(tr.shape[1]),
        "n_train": int(n),
        "n_val": int(y_va.shape[0]),
        "test_evaluated": False,
    }


def _resolve_emb_dir(run: str, embeddings_root: Path) -> Path:
    base = embeddings_root / run
    pre = base / "pre_embedding_3h"
    if (pre / "train.npz").is_file() and (pre / "val.npz").is_file():
        return pre
    if (base / "train.npz").is_file() and (base / "val.npz").is_file():
        return base
    raise FileNotFoundError(f"Missing train/val npz under {base} or {pre}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, help="unique_name / embeddings subdir")
    p.add_argument("--embeddings_dir", type=str, default="embeddings")
    p.add_argument("--out", type=str, default="")
    p.add_argument("--checkpoint_epoch", type=int, default=0, help="optional label only")
    args = p.parse_args()
    logger_setup(logging.INFO)

    emb_dir = _resolve_emb_dir(args.run, Path(args.embeddings_dir))
    if (emb_dir / "test.npz").is_file():
        logging.warning("test.npz present under %s but will not be read", emb_dir)

    z_tr, y_tr, id_tr = load_embedding_npz(emb_dir / "train.npz")
    z_va, y_va, id_va = load_embedding_npz(emb_dir / "val.npz")
    if int(z_tr.shape[1]) != 198:
        logging.warning("expected R198 (dim 198), got dim=%s", z_tr.shape[1])

    x_all, tf_all = _load_x_tf()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    set_seed(MLP_SEED)
    primary = _fit_mlp(
        np.concatenate([z_tr, x_all[id_tr], tf_all[id_tr]], axis=1),
        y_tr,
        np.concatenate([z_va, x_all[id_va], tf_all[id_va]], axis=1),
        y_va,
        device,
    )
    primary["stack"] = "R198+X+TF"

    diagnostic = _fit_mlp(z_tr, y_tr, z_va, y_va, device)
    diagnostic["stack"] = "R198_only"

    out = {
        "ok": True,
        "run": args.run,
        "embedding_dir": str(emb_dir),
        "representation": "R198",
        "checkpoint_epoch_label": int(args.checkpoint_epoch) or None,
        "selection": "downstream_validation_not_ssl_loss",
        "test_evaluated": False,
        "primary": primary,
        "diagnostic": diagnostic,
        "gates": {
            "q1_compare_direct_h_f1_to_supervised_eu": "external_reference",
            "q2_tfmoe_improve_over_direct_h": {
                "auprc_margin": 0.003,
                "f1_margin": 0.01,
            },
        },
        "not_exact_papagei": True,
        "not_four_arm_projection_ablation": True,
    }
    out_path = Path(args.out) if args.out else (
        ROOT / "results/diagnostics" / args.run / "val_probe.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    logging.info(
        "Wrote %s primary_auprc=%.4f primary_f1@opt=%.4f diagnostic_auprc=%.4f",
        out_path,
        primary["validation_auprc"],
        primary["validation_metrics_at_val_optimal_f1"]["f1"],
        diagnostic["validation_auprc"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
