"""Frozen MLP on H||X for the txn-node baseline evaluations."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class PaperStyleMLP(nn.Module):
    def __init__(self, d_in: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _metrics_at_threshold(y: np.ndarray, proba: np.ndarray, thr: float) -> Dict[str, float]:
    pred = (proba >= float(thr)).astype(np.int64)
    y = y.astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    n = int(y.shape[0])
    return {
        "auroc": float(roc_auc_score(y, proba)) if len(np.unique(y)) > 1 else float("nan"),
        "auprc": float(average_precision_score(y, proba)) if len(np.unique(y)) > 1 else float("nan"),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "threshold": float(thr),
        "positive_prediction_rate": float(pred.mean()) if n else 0.0,
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
        "n": float(n),
    }


def _ranking_only(y: np.ndarray, proba: np.ndarray) -> Dict[str, float]:
    y = y.astype(np.int64)
    return {
        "auroc": float(roc_auc_score(y, proba)) if len(np.unique(y)) > 1 else float("nan"),
        "auprc": float(average_precision_score(y, proba)) if len(np.unique(y)) > 1 else float("nan"),
        "n": float(y.shape[0]),
        "positive_rate": float(y.mean()) if y.size else 0.0,
    }


def _select_threshold_f1(y_val: np.ndarray, proba_val: np.ndarray) -> float:
    """Validation-selected threshold maximizing F1 over a dense grid."""
    if len(np.unique(y_val)) < 2:
        return 0.5
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.linspace(0.01, 0.99, 99):
        pred = (proba_val >= thr).astype(np.int64)
        f1 = float(f1_score(y_val.astype(np.int64), pred, zero_division=0))
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    return best_thr


def _predict_proba(
    model: nn.Module,
    x: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        chunks = []
        xt = torch.from_numpy(x.astype(np.float32))
        for start in range(0, xt.shape[0], batch_size):
            chunks.append(model(xt[start : start + batch_size].to(device)).cpu())
        return torch.sigmoid(torch.cat(chunks)).numpy()


def train_eval_mlp(
    h_train: np.ndarray,
    x_train: np.ndarray,
    y_train: np.ndarray,
    h_test: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    h_val: Optional[np.ndarray] = None,
    x_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    epochs: int = 15,
    batch_size: int = 8192,
    lr: float = 1e-3,
    seed: int = 0,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """BCE MLP on concat(H, X); fixed 0.5 threshold (paper-style primary). Legacy API."""
    suite = train_eval_mlp_suite(
        h_train,
        x_train,
        y_train,
        h_test,
        x_test,
        y_test,
        h_val=h_val,
        x_val=x_val,
        y_val=y_val,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        device=device,
        representations=("HxX",),
    )
    return suite["HxX"]["threshold_0.5"]


def train_eval_mlp_suite(
    h_train: np.ndarray,
    x_train: np.ndarray,
    y_train: np.ndarray,
    h_test: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    h_val: Optional[np.ndarray] = None,
    x_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    epochs: int = 15,
    batch_size: int = 8192,
    lr: float = 1e-3,
    seed: int = 0,
    device: Optional[torch.device] = None,
    representations: Sequence[str] = ("X", "H", "HxX"),
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Train separate MLPs for X / H / H||X; report fixed-0.5 and val-selected thresholds."""
    device = device or torch.device("cpu")
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    feats = {
        "X": (x_train, x_test, x_val),
        "H": (h_train, h_test, h_val),
        "HxX": (
            np.concatenate([h_train, x_train], axis=1),
            np.concatenate([h_test, x_test], axis=1),
            None
            if h_val is None or x_val is None
            else np.concatenate([h_val, x_val], axis=1),
        ),
    }
    for name in representations:
        tr, te, va = feats[name]
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = PaperStyleMLP(int(tr.shape[1])).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        x_t = torch.from_numpy(tr.astype(np.float32))
        y_t = torch.from_numpy(y_train.astype(np.float32))
        n = tr.shape[0]
        model.train()
        for ep in range(epochs):
            perm = np.random.RandomState(seed * 1009 + ep).permutation(n)
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                xb = x_t[idx].to(device)
                yb = y_t[idx].to(device)
                opt.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = nn.functional.binary_cross_entropy_with_logits(logits, yb)
                loss.backward()
                opt.step()
        proba_te = _predict_proba(model, te, batch_size=batch_size, device=device)
        fixed = _metrics_at_threshold(y_test, proba_te, 0.5)
        val_rank: Dict[str, float] = {}
        if va is not None and y_val is not None:
            proba_va = _predict_proba(model, va, batch_size=batch_size, device=device)
            thr = _select_threshold_f1(y_val, proba_va)
            selected = _metrics_at_threshold(y_test, proba_te, thr)
            selected["validation_selected_threshold"] = float(thr)
            val_rank = _ranking_only(y_val, proba_va)
            val_at_selected = _metrics_at_threshold(y_val, proba_va, thr)
        else:
            selected = _metrics_at_threshold(y_test, proba_te, 0.5)
            selected["validation_selected_threshold"] = 0.5
            val_at_selected = {}
        out[name] = {
            "threshold_0.5": fixed,
            "threshold_val_selected": selected,
            "val_ranking": val_rank,
            "val_at_selected_threshold": val_at_selected,
        }
    return out
