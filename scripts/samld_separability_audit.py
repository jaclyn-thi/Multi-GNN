#!/usr/bin/env python3
"""SAML-D Candidate A X-only separability audit (train/val only; test locked).

Uses the same base edge features + ports as protocol A, with legacy per-graph
edge z-norm on the train graph / train∪val graph (matching get_data without
train_fit). Does not load or inspect the test split.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from data_util import GraphData, z_norm  # noqa: E402
from dataset_specs import (  # noqa: E402
    DEFAULT_EDGE_FEATURE_COLS,
    FORMATTED_TRANSACTION_COLUMNS,
    get_dataset_spec,
)
from dataset_splits import temporal_edge_split  # noqa: E402
from gcpal_txn_node.eval_mlp import PaperStyleMLP, _predict_proba  # noqa: E402
from util import set_seed  # noqa: E402

FORMATTED = REPO / "aml-data" / "SAML-D" / "formatted_transactions.csv"
OUT_JSON = REPO / "results" / "diagnostics" / "samld_separability_audit.json"
OUT_MD = REPO / "notes" / "samld_separability_audit.md"
INTEGRITY = REPO / "results" / "diagnostics" / "samld_protocol_and_integrity.json"

FEATURE_NAMES = list(DEFAULT_EDGE_FEATURE_COLS) + ["in_port", "out_port"]
# Explicit Candidate A inventory requested by protocol card:
CANDIDATE_A_FEATURE_LIST = [
    "Timestamp",
    "Amount Received",  # amount
    "Received Currency",  # currency / payment-format fields
    "Payment Format",
    "in_port",
    "out_port",
]

EXPECTED = {
    "train": {
        "n": 5_707_315,
        "n_positives": 5_751,
        "index_sha256": "290713933cc655e9c70984bc3cb7f575ab26a03b8078a1337cda58892054935f",
    },
    "val": {
        "n": 1_899_523,
        "n_positives": 1_986,
        "index_sha256": "b08cdb815f82e6d37019e5e6ec9c5a6fd12c3f9d523f63b2768f6e4d0a99a38c",
    },
}

PERM_SUBSET_N = 200_000
PERM_SEED = 2
LOGISTIC_SEED = 2
MLP_SEED = 3
NEAR_ID_AUPRC = 0.5  # single-feature "nearly identifies" threshold (absolute)


def sha256_int64(arr: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(arr.astype(np.int64)).tobytes())
    return h.hexdigest()


def _safe_auroc(y: np.ndarray, s: np.ndarray) -> float:
    if np.unique(y).size < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y, s))
    except ValueError:
        return float("nan")


def _safe_auprc(y: np.ndarray, s: np.ndarray) -> float:
    if np.unique(y).size < 2:
        return float("nan")
    try:
        return float(average_precision_score(y, s))
    except ValueError:
        return float("nan")


def univariate_rank(y: np.ndarray, x: np.ndarray, name: str) -> Dict[str, Any]:
    """Score both raw and negated feature directions; keep better AUPRC."""
    out = {"feature": name}
    best = None
    for sign, arr in (("pos", x), ("neg", -x)):
        auprc = _safe_auprc(y, arr)
        auroc = _safe_auroc(y, arr)
        row = {
            "sign": sign,
            "auprc": auprc,
            "auroc": auroc,
            "score_min": float(np.min(arr)),
            "score_max": float(np.max(arr)),
        }
        out[f"direction_{sign}"] = row
        if best is None or (
            math.isfinite(auprc) and (not math.isfinite(best["auprc"]) or auprc > best["auprc"])
        ):
            best = {"sign": sign, "auprc": auprc, "auroc": auroc}
    out["best"] = best
    return out


def fit_eval_logistic(
    x_tr: np.ndarray, y_tr: np.ndarray, x_va: np.ndarray, y_va: np.ndarray
) -> Dict[str, Any]:
    scaler = StandardScaler()
    xt = scaler.fit_transform(x_tr)
    xv = scaler.transform(x_va)
    clf = LogisticRegression(
        max_iter=200,
        solver="lbfgs",
        class_weight="balanced",
        random_state=LOGISTIC_SEED,
        n_jobs=1,
    )
    clf.fit(xt, y_tr)
    proba = clf.predict_proba(xv)[:, 1]
    pred = (proba >= 0.5).astype(np.int64)
    return {
        "learner": "logistic_regression",
        "val_auprc": _safe_auprc(y_va, proba),
        "val_auroc": _safe_auroc(y_va, proba),
        "val_f1_at_0.5": float(f1_score(y_va, pred, zero_division=0)),
        "val_precision_at_0.5": float(precision_score(y_va, pred, zero_division=0)),
        "val_recall_at_0.5": float(recall_score(y_va, pred, zero_division=0)),
        "test_evaluated": False,
    }


def fit_eval_mlp(
    x_tr: np.ndarray, y_tr: np.ndarray, x_va: np.ndarray, y_va: np.ndarray
) -> Dict[str, Any]:
    set_seed(MLP_SEED)
    scaler = StandardScaler()
    xt = scaler.fit_transform(x_tr).astype(np.float32)
    xv = scaler.transform(x_va).astype(np.float32)
    device = torch.device("cpu")
    model = PaperStyleMLP(d_in=xt.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    # class-balanced BCE
    n_pos = max(int(y_tr.sum()), 1)
    n_neg = max(int((1 - y_tr).sum()), 1)
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    xt_t = torch.from_numpy(xt)
    y_t = torch.from_numpy(y_tr.astype(np.float32))
    bs = 8192
    model.train()
    for _epoch in range(8):
        perm = torch.randperm(xt_t.shape[0])
        for i in range(0, xt_t.shape[0], bs):
            idx = perm[i : i + bs]
            opt.zero_grad()
            logits = model(xt_t[idx].to(device))
            loss = loss_fn(logits, y_t[idx].to(device))
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        proba = _predict_proba(model, xv, device=device, batch_size=8192)
    pred = (proba >= 0.5).astype(np.int64)
    return {
        "learner": "paper_style_mlp",
        "epochs": 8,
        "val_auprc": _safe_auprc(y_va, proba),
        "val_auroc": _safe_auroc(y_va, proba),
        "val_f1_at_0.5": float(f1_score(y_va, pred, zero_division=0)),
        "test_evaluated": False,
    }


def fit_eval_hgb(
    x_tr: np.ndarray, y_tr: np.ndarray, x_va: np.ndarray, y_va: np.ndarray
) -> Dict[str, Any]:
    clf = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.1,
        max_iter=100,
        l2_regularization=0.0,
        random_state=LOGISTIC_SEED,
    )
    clf.fit(x_tr, y_tr)
    proba = clf.predict_proba(x_va)[:, 1]
    pred = (proba >= 0.5).astype(np.int64)
    return {
        "learner": "hist_gradient_boosting",
        "note": "LightGBM/XGBoost not installed; sklearn HGB used as tree control",
        "val_auprc": _safe_auprc(y_va, proba),
        "val_auroc": _safe_auroc(y_va, proba),
        "val_f1_at_0.5": float(f1_score(y_va, pred, zero_division=0)),
        "test_evaluated": False,
    }


def category_label_audit(
    df: pd.DataFrame, tr: np.ndarray, va: np.ndarray, col: str
) -> Dict[str, Any]:
    y = df["Is Laundering"].to_numpy()
    out: Dict[str, Any] = {"column": col}
    for split, inds in (("train", tr), ("val", va)):
        sub = df.iloc[inds]
        ys = y[inds]
        counts = {}
        for cat, g in sub.groupby(col):
            idx = g.index.to_numpy()
            # map to positions in inds — use label via original index
            # simpler: use values aligned
            pass
        # rebuild with numpy
        vals = sub[col].to_numpy()
        cats = {}
        for c in np.unique(vals):
            m = vals == c
            n = int(m.sum())
            pos = int(ys[m].sum())
            cats[str(c)] = {
                "n": n,
                "n_pos": pos,
                "pos_rate": float(pos / n) if n else 0.0,
            }
        out[split] = cats
    # categories overwhelmingly positive on train
    train_cats = out["train"]
    hot = []
    for c, st in train_cats.items():
        if st["n"] >= 20 and st["pos_rate"] >= 0.5:
            hot.append({"category": c, **st})
    out["train_high_pos_rate_categories"] = sorted(
        hot, key=lambda r: -r["pos_rate"]
    )[:20]
    # only in positives
    only_pos = [
        c
        for c, st in train_cats.items()
        if st["n_pos"] == st["n"] and st["n"] >= 5
    ]
    out["train_categories_all_positive"] = only_pos
    return out


def main() -> int:
    t0 = time.perf_counter()
    import importlib.util as iu

    tree_deps = {
        "lightgbm": iu.find_spec("lightgbm") is not None,
        "xgboost": iu.find_spec("xgboost") is not None,
    }

    print("loading formatted CSV…", flush=True)
    df = pd.read_csv(FORMATTED)
    assert list(df.columns) == list(FORMATTED_TRANSACTION_COLUMNS)
    spec = get_dataset_spec("SAML-D")
    timestamps = torch.tensor(df["Timestamp"].to_numpy(), dtype=torch.long)
    y_all = torch.tensor(df["Is Laundering"].to_numpy(), dtype=torch.long)
    tr_t, va_t, te_t, _ = temporal_edge_split(timestamps, y_all, spec)
    tr = tr_t.numpy().astype(np.int64)
    va = va_t.numpy().astype(np.int64)
    # Explicitly do not use te_t beyond confirming we ignore it
    del te_t

    split_check = {}
    for name, inds, exp in (
        ("train", tr, EXPECTED["train"]),
        ("val", va, EXPECTED["val"]),
    ):
        yi = y_all.numpy()[inds]
        st = {
            "n": int(inds.shape[0]),
            "n_positives": int(yi.sum()),
            "prevalence": float(yi.mean()),
            "index_sha256": sha256_int64(inds),
        }
        st["matches_protocol"] = (
            st["n"] == exp["n"]
            and st["n_positives"] == exp["n_positives"]
            and st["index_sha256"] == exp["index_sha256"]
        )
        split_check[name] = st
    assert split_check["train"]["matches_protocol"], split_check["train"]
    assert split_check["val"]["matches_protocol"], split_check["val"]
    print("split hashes OK; test not loaded as cohort", flush=True)

    # Account overlap train/val
    from_id = df["from_id"].to_numpy()
    to_id = df["to_id"].to_numpy()
    tr_acc = set(from_id[tr].tolist()) | set(to_id[tr].tolist())
    va_acc = set(from_id[va].tolist()) | set(to_id[va].tolist())
    overlap = tr_acc & va_acc
    account_overlap = {
        "n_train_accounts": len(tr_acc),
        "n_val_accounts": len(va_acc),
        "n_overlap": len(overlap),
        "frac_val_accounts_also_in_train": float(len(overlap) / len(va_acc)) if va_acc else 0.0,
    }

    # Build Candidate A edge features with ports on train and train∪val graphs
    print("building ports + legacy z-norm (train / train∪val)…", flush=True)
    n_nodes = int(max(from_id.max(), to_id.max()) + 1)
    x_nodes = torch.ones((n_nodes, 1), dtype=torch.float32)

    def build_graph(edge_inds: np.ndarray) -> GraphData:
        ei = torch.stack(
            [
                torch.from_numpy(from_id[edge_inds].astype(np.int64)),
                torch.from_numpy(to_id[edge_inds].astype(np.int64)),
            ],
            dim=0,
        )
        ea = torch.tensor(
            df.loc[edge_inds, list(DEFAULT_EDGE_FEATURE_COLS)].to_numpy(),
            dtype=torch.float32,
        )
        yy = torch.from_numpy(y_all.numpy()[edge_inds].astype(np.int64))
        ts = torch.from_numpy(timestamps.numpy()[edge_inds])
        g = GraphData(x=x_nodes.clone(), y=yy, edge_index=ei, edge_attr=ea, timestamps=ts)
        g.add_ports()
        g.edge_attr = z_norm(g.edge_attr)
        return g

    e_tr = tr
    e_val = np.concatenate([tr, va])
    g_tr = build_graph(e_tr)
    g_va = build_graph(e_val)
    assert g_tr.edge_attr.shape[1] == 6
    assert g_va.edge_attr.shape[1] == 6

    # Seed features: train = all train graph edges; val seeds = trailing |va| edges
    x_tr = g_tr.edge_attr.numpy()
    y_tr = g_tr.y.numpy().astype(np.int64)
    x_va = g_va.edge_attr.numpy()[-len(va) :]
    y_va = g_va.y.numpy().astype(np.int64)[-len(va) :]
    assert x_va.shape[0] == len(va)
    assert int(y_va.sum()) == EXPECTED["val"]["n_positives"]

    # Ports independence: recompute ports on val-only graph vs train∪val slice for audit
    g_va_only = build_graph(va)
    ports_indep = {
        "val_only_graph_edge_dim": int(g_va_only.edge_attr.shape[1]),
        "ports_built_independently_per_split_graph": True,
        "note": (
            "Candidate A uses ports on train graph and on train∪val graph separately "
            "(get_data). Val seed ports therefore depend on train∪val adjacency, not "
            "val-only. This audit matches that construction."
        ),
        "max_abs_diff_val_seed_ports_vs_val_only_ports": float(
            np.max(
                np.abs(
                    x_va[:, 4:6]
                    - g_va_only.edge_attr.numpy()[:, 4:6]
                )
            )
        ),
    }

    prevalence = {
        "train": float(y_tr.mean()),
        "val": float(y_va.mean()),
        "val_prevalence_baseline_auprc": float(y_va.mean()),
    }

    # Univariate
    print("univariate rankings…", flush=True)
    uni = []
    for j, name in enumerate(FEATURE_NAMES):
        uni.append(univariate_rank(y_va, x_va[:, j], name))
    uni_sorted = sorted(
        uni,
        key=lambda r: -(r["best"]["auprc"] if r.get("best") and math.isfinite(r["best"]["auprc"]) else -1),
    )
    nearly = [
        r
        for r in uni_sorted
        if r.get("best") and math.isfinite(r["best"]["auprc"]) and r["best"]["auprc"] >= NEAR_ID_AUPRC
    ]

    # X-only learners
    print("logistic…", flush=True)
    logistic = fit_eval_logistic(x_tr, y_tr, x_va, y_va)
    print("mlp…", flush=True)
    mlp = fit_eval_mlp(x_tr, y_tr, x_va, y_va)
    print("hgb…", flush=True)
    hgb = fit_eval_hgb(x_tr, y_tr, x_va, y_va)

    # Label permutation on fixed subset
    print("permutation sanity…", flush=True)
    rng = np.random.default_rng(PERM_SEED)
    n_sub = min(PERM_SUBSET_N, len(tr))
    sub_pos = np.arange(n_sub)  # fixed prefix of train (deterministic)
    x_sub = x_tr[sub_pos]
    y_sub = y_tr[sub_pos].copy()
    y_perm = y_sub.copy()
    rng.shuffle(y_perm)
    # evaluate on full val with model trained on permuted subset labels
    scaler = StandardScaler()
    xt = scaler.fit_transform(x_sub)
    xv = scaler.transform(x_va)
    clf = LogisticRegression(
        max_iter=200, solver="lbfgs", class_weight="balanced", random_state=LOGISTIC_SEED
    )
    clf.fit(xt, y_perm)
    proba_p = clf.predict_proba(xv)[:, 1]
    perm = {
        "subset_n": int(n_sub),
        "subset_policy": "first_N_train_edges",
        "perm_seed": PERM_SEED,
        "train_subset_prevalence": float(y_sub.mean()),
        "permuted_label_prevalence": float(y_perm.mean()),
        "val_auprc": _safe_auprc(y_va, proba_p),
        "val_prevalence": float(y_va.mean()),
        "abs_auprc_minus_prevalence": float(
            abs(_safe_auprc(y_va, proba_p) - float(y_va.mean()))
        ),
        "falls_near_prevalence": bool(
            abs(_safe_auprc(y_va, proba_p) - float(y_va.mean())) < 0.01
        ),
        "test_evaluated": False,
    }

    # Leakage / category audits (train/val only)
    print("audits…", flush=True)
    # Account / EdgeID in X?
    x_cols_model = list(FEATURE_NAMES)
    leakage = {
        "account_ids_in_X": False,
        "edge_id_in_X": False,
        "model_input_feature_names": x_cols_model,
        "formatter_columns_excluded_from_X": [
            c
            for c in FORMATTED_TRANSACTION_COLUMNS
            if c not in DEFAULT_EDGE_FEATURE_COLS and c != "Is Laundering"
        ],
        "label_col_in_X": False,
        "formatter_uses_labels_for_features": False,
        "note_formatter": (
            "SAML-D formatter maps raw fields to AML schema; label is Is Laundering only. "
            "Amount Sent discarded from edge_attr (Amount Received kept)."
        ),
    }

    cat_currency = category_label_audit(df, tr, va, "Received Currency")
    cat_payment = category_label_audit(df, tr, va, "Payment Format")

    # Amount / time ranges nearly identifying positives
    amt = df["Amount Received"].to_numpy()
    ts_np = df["Timestamp"].to_numpy()
    amt_audit = {}
    for split, inds in (("train", tr), ("val", va)):
        yi = y_all.numpy()[inds]
        a = amt[inds]
        t = ts_np[inds]
        pos = yi == 1
        amt_audit[split] = {
            "amount_pos_min": float(a[pos].min()) if pos.any() else None,
            "amount_pos_max": float(a[pos].max()) if pos.any() else None,
            "amount_neg_min": float(a[~pos].min()) if (~pos).any() else None,
            "amount_neg_max": float(a[~pos].max()) if (~pos).any() else None,
            "timestamp_pos_min": int(t[pos].min()) if pos.any() else None,
            "timestamp_pos_max": int(t[pos].max()) if pos.any() else None,
            "disjoint_amount_ranges": bool(
                pos.any()
                and (~pos).any()
                and (a[pos].max() < a[~pos].min() or a[~pos].max() < a[pos].min())
            ),
        }

    # Duplicate feature rows with conflicting labels (train∪val only)
    feat_cols = list(DEFAULT_EDGE_FEATURE_COLS) + ["from_id", "to_id"]
    tv = np.concatenate([tr, va])
    sub = df.iloc[tv][feat_cols + ["Is Laundering"]].copy()
    # hash rows
    key = pd.util.hash_pandas_object(sub[feat_cols], index=False).to_numpy()
    # conflict: same key different labels
    conflicts = 0
    by_key: Dict[int, set] = defaultdict(set)
    labs = sub["Is Laundering"].to_numpy()
    for k, lab in zip(key, labs):
        by_key[int(k)].add(int(lab))
    for labs_set in by_key.values():
        if len(labs_set) > 1:
            conflicts += 1
    dup_audit = {
        "n_train_val_rows": int(len(tv)),
        "n_unique_feature_hashes": len(by_key),
        "n_hashes_with_conflicting_labels": conflicts,
    }

    # Prevalence by calendar day and payment format (train/val)
    day = (ts_np // 86400).astype(np.int64)
    day_prev = {}
    for split, inds in (("train", tr), ("val", va)):
        d = day[inds]
        yi = y_all.numpy()[inds]
        rows = []
        for dd in np.unique(d):
            m = d == dd
            n = int(m.sum())
            pos = int(yi[m].sum())
            rows.append({"day": int(dd), "n": n, "n_pos": pos, "prevalence": float(pos / n)})
        day_prev[split] = rows

    pay_prev = {}
    for split, inds in (("train", tr), ("val", va)):
        pf = df["Payment Format"].to_numpy()[inds]
        yi = y_all.numpy()[inds]
        rows = []
        for c in np.unique(pf):
            m = pf == c
            n = int(m.sum())
            pos = int(yi[m].sum())
            rows.append(
                {
                    "payment_format": int(c) if np.issubdtype(type(c), np.integer) else str(c),
                    "n": n,
                    "n_pos": pos,
                    "prevalence": float(pos / n),
                }
            )
        pay_prev[split] = sorted(rows, key=lambda r: -r["prevalence"])

    best_x = max(
        [
            ("logistic", logistic["val_auprc"]),
            ("mlp", mlp["val_auprc"]),
            ("hgb", hgb["val_auprc"]),
        ],
        key=lambda t: t[1] if math.isfinite(t[1]) else -1,
    )

    payload = {
        "audit_id": "samld_separability_audit",
        "protocol_id": "samld_supervised_multigin_eu_v1",
        "test_inspected": False,
        "test_evaluated": False,
        "candidate_a_features_explicit": CANDIDATE_A_FEATURE_LIST,
        "model_input_feature_names_after_ports": FEATURE_NAMES,
        "edge_dim": 6,
        "normalization": "legacy_per_graph_edge_znorm",
        "tree_dependencies": tree_deps,
        "split_check": split_check,
        "prevalence": prevalence,
        "account_overlap_train_val": account_overlap,
        "ports_construction": ports_indep,
        "univariate_val": uni_sorted,
        "nearly_identifying_single_features": nearly,
        "x_only_learners": {"logistic": logistic, "mlp": mlp, "hgb": hgb},
        "best_x_only_val_auprc": {"learner": best_x[0], "val_auprc": best_x[1]},
        "label_permutation_sanity": perm,
        "leakage_audit": leakage,
        "category_audits": {
            "Received Currency": cat_currency,
            "Payment Format": cat_payment,
        },
        "amount_time_range_audit": amt_audit,
        "duplicate_conflicting_label_audit": dup_audit,
        "prevalence_by_calendar_day": {
            "train_n_days": len(day_prev["train"]),
            "val_n_days": len(day_prev["val"]),
            "train_max_day_prevalence": max(r["prevalence"] for r in day_prev["train"]),
            "val_max_day_prevalence": max(r["prevalence"] for r in day_prev["val"]),
            # omit full day tables from JSON top-level size; attach compact
            "train_top5_days_by_prevalence": sorted(
                day_prev["train"], key=lambda r: -r["prevalence"]
            )[:5],
            "val_top5_days_by_prevalence": sorted(
                day_prev["val"], key=lambda r: -r["prevalence"]
            )[:5],
        },
        "prevalence_by_payment_format": pay_prev,
        "elapsed_sec": time.perf_counter() - t0,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# SAML-D separability audit (Candidate A features; train/val only)",
        "",
        f"- **test_inspected / test_evaluated:** `false`",
        f"- **edge_dim:** 6 — `{FEATURE_NAMES}`",
        f"- **Norm:** legacy per-graph z-norm (train graph; train∪val graph)",
        f"- **Val prevalence (AUPRC baseline):** `{prevalence['val']:.6f}`",
        f"- **Best X-only val AUPRC:** `{best_x[0]}` = `{best_x[1]:.4f}`",
        f"- **Permutation control falls near prevalence:** `{perm['falls_near_prevalence']}` "
        f"(AUPRC={perm['val_auprc']:.6f})",
        "",
        "## Candidate A features (explicit)",
        "",
    ]
    for f in CANDIDATE_A_FEATURE_LIST:
        lines.append(f"- `{f}`")
    lines.extend(
        [
            "",
            "## Prevalence",
            "",
            f"- Train π={prevalence['train']:.6f} (n={split_check['train']['n']}, pos={split_check['train']['n_positives']})",
            f"- Val π={prevalence['val']:.6f} (n={split_check['val']['n']}, pos={split_check['val']['n_positives']})",
            "",
            "## Univariate val AUPRC (best direction)",
            "",
        ]
    )
    for r in uni_sorted:
        b = r["best"]
        lines.append(
            f"- `{r['feature']}` ({b['sign']}): AUPRC={b['auprc']:.4f} AUROC={b['auroc']:.4f}"
        )
    lines.extend(
        [
            "",
            f"Nearly-identifying (AUPRC≥{NEAR_ID_AUPRC}): "
            + (
                ", ".join(f"`{r['feature']}`" for r in nearly)
                if nearly
                else "**none**"
            ),
            "",
            "## X-only learners (val)",
            "",
            f"- Logistic: AUPRC={logistic['val_auprc']:.4f} AUROC={logistic['val_auroc']:.4f}",
            f"- MLP: AUPRC={mlp['val_auprc']:.4f} AUROC={mlp['val_auroc']:.4f}",
            f"- HGB (sklearn; LGBM/XGB absent={not tree_deps['lightgbm']}/{not tree_deps['xgboost']}): "
            f"AUPRC={hgb['val_auprc']:.4f}",
            "",
            "## Audits (summary)",
            "",
            f"- Account/EdgeID in X: `{leakage['account_ids_in_X']}` / `{leakage['edge_id_in_X']}`",
            f"- Conflicting-label feature hashes (train∪val): `{dup_audit['n_hashes_with_conflicting_labels']}`",
            f"- Val accounts also in train: `{account_overlap['frac_val_accounts_also_in_train']:.3f}`",
            f"- Amount ranges disjoint by label (train): `{amt_audit['train']['disjoint_amount_ranges']}`",
            f"- Ports max |diff| val-seed (train∪val) vs val-only graph: "
            f"`{ports_indep['max_abs_diff_val_seed_ports_vs_val_only_ports']:.4f}`",
            "",
            f"Twin JSON: `{OUT_JSON.relative_to(REPO)}`",
            "",
        ]
    )
    OUT_MD.write_text("\n".join(lines))
    print(json.dumps({"out_json": str(OUT_JSON), "best_x_auprc": best_x}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
