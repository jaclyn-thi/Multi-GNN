#!/usr/bin/env python3
"""CPU-only SAML-D integrity validation (no GNN, no train, no test eval)."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dataset_specs import (  # noqa: E402
    DEFAULT_EDGE_FEATURE_COLS,
    FORMATTED_TRANSACTION_COLUMNS,
    get_dataset_spec,
)
from dataset_splits import temporal_edge_split  # noqa: E402

FORMATTED = REPO / "aml-data" / "SAML-D" / "formatted_transactions.csv"
RAW_CANDIDATES = [
    REPO / "raw-aml-data" / "SAML-D.csv",
    Path("/orcd/pool/007/jthi/Multi-GNN/raw-aml-data/SAML-D.csv"),
]
OUT_JSON = REPO / "results" / "diagnostics" / "samld_integrity_compute.json"
MODEL_INPUT_EDGE_COLS = list(DEFAULT_EDGE_FEATURE_COLS)
LABEL_COL = "Is Laundering"
FEATURE_HASH_COLS = [
    "from_id",
    "to_id",
    "Timestamp",
    "Amount Sent",
    "Sent Currency",
    "Amount Received",
    "Received Currency",
    "Payment Format",
]


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_int64_array(arr: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def dtype_map(df: pd.DataFrame) -> Dict[str, str]:
    return {c: str(df[c].dtype) for c in df.columns}


def split_stats(y: np.ndarray, inds: np.ndarray) -> Dict[str, Any]:
    yi = y[inds]
    n = int(yi.shape[0])
    pos = int(yi.sum())
    return {
        "n": n,
        "n_positives": pos,
        "positive_rate": float(pos / n) if n else 0.0,
        "index_sha256": sha256_int64_array(inds.astype(np.int64)),
    }


def category_unseen(train_vals: np.ndarray, other_vals: np.ndarray) -> Dict[str, Any]:
    tr = set(map(int, np.unique(train_vals)))
    ot = set(map(int, np.unique(other_vals)))
    unseen = sorted(ot - tr)
    return {
        "n_train_categories": len(tr),
        "n_other_categories": len(ot),
        "n_unseen_in_other": len(unseen),
        "unseen_values_sample": unseen[:20],
    }


def label_determinism_report(df: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
    out: Dict[str, Any] = {"perfect_separating_categories": {}, "high_pos_rate_categories": {}}
    for col in ("Sent Currency", "Received Currency", "Payment Format"):
        vals = df[col].to_numpy()
        perfect = []
        high = []
        for v in np.unique(vals):
            m = vals == v
            n = int(m.sum())
            pos = int(y[m].sum())
            rate = pos / n if n else 0.0
            if n >= 10 and (rate == 0.0 or rate == 1.0):
                perfect.append({"value": int(v), "n": n, "positive_rate": rate})
            elif n >= 50 and rate >= 0.5:
                high.append({"value": int(v), "n": n, "positive_rate": rate})
        out["perfect_separating_categories"][col] = perfect[:50]
        out["high_pos_rate_categories"][col] = sorted(high, key=lambda d: -d["positive_rate"])[:20]
    return out


def cross_split_feature_identical(feat_hash: np.ndarray, split_id: np.ndarray) -> Dict[str, Any]:
    order = np.argsort(feat_hash, kind="mergesort")
    h = feat_hash[order]
    s = split_id[order]
    n = h.shape[0]
    multi = 0
    pairs = Counter()
    i = 0
    while i < n:
        j = i + 1
        while j < n and h[j] == h[i]:
            j += 1
        splits_present = set(int(x) for x in s[i:j])
        if len(splits_present) > 1:
            multi += 1
            pairs[tuple(sorted(splits_present))] += 1
        i = j
    return {
        "n_feature_hashes_spanning_multiple_splits": int(multi),
        "split_pair_counts": {str(k): int(v) for k, v in pairs.items()},
    }


def account_overlap(from_id, to_id, tr, va, te) -> Dict[str, Any]:
    def nodes(inds):
        return set(from_id[inds].tolist()) | set(to_id[inds].tolist())

    tr_n, va_n, te_n = nodes(tr), nodes(va), nodes(te)
    return {
        "n_accounts_train": len(tr_n),
        "n_accounts_val": len(va_n),
        "n_accounts_test": len(te_n),
        "n_accounts_all": len(tr_n | va_n | te_n),
        "val_accounts_also_in_train": len(va_n & tr_n),
        "val_accounts_also_in_train_frac": float(len(va_n & tr_n) / len(va_n)) if va_n else 0.0,
        "test_accounts_also_in_train": len(te_n & tr_n),
        "test_accounts_also_in_train_frac": float(len(te_n & tr_n) / len(te_n)) if te_n else 0.0,
        "test_accounts_also_in_val": len(te_n & va_n),
        "interpretation": "allowed_entity_overlap_not_entity_inductive",
    }


def main() -> int:
    t0 = time.time()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    raw_path = next((p for p in RAW_CANDIDATES if p.is_file()), None)
    if not FORMATTED.is_file():
        raise SystemExit(f"missing formatted: {FORMATTED}")
    if raw_path is None:
        raise SystemExit(f"missing raw SAML-D.csv in {RAW_CANDIDATES}")

    print("[1/8] hashing files...", flush=True)
    formatted_sha = sha256_file(FORMATTED)
    raw_sha = sha256_file(raw_path)
    print(f"  formatted sha256={formatted_sha}", flush=True)
    print(f"  raw sha256={raw_sha}", flush=True)

    print("[2/8] loading formatted CSV...", flush=True)
    df = pd.read_csv(FORMATTED)
    cols = list(df.columns)
    col_match = tuple(cols) == FORMATTED_TRANSACTION_COLUMNS

    print("[3/8] basic counts...", flush=True)
    n_rows = int(len(df))
    y = df[LABEL_COL].to_numpy().astype(np.int64)
    n_pos = int(y.sum())
    edge_ids = df["EdgeID"].to_numpy().astype(np.int64)
    n_unique_edge = int(np.unique(edge_ids).shape[0])
    ts = df["Timestamp"].to_numpy().astype(np.int64)
    ts_diff = np.diff(ts)
    n_ts_decreases = int(np.sum(ts_diff < 0))
    n_ts_equal_consec = int(np.sum(ts_diff == 0))

    amt_s = df["Amount Sent"].to_numpy()
    amt_r = df["Amount Received"].to_numpy()
    amt_equal = bool(np.all(amt_s == amt_r))
    n_amt_ne = int(np.sum(amt_s != amt_r))

    print("[4/8] duplicate rows...", flush=True)
    n_dup_rows = int(df.duplicated(keep=False).sum())
    n_dup_groups = int(df.duplicated().sum())

    print("[5/8] temporal split (same as loader)...", flush=True)
    spec = get_dataset_spec("SAML-D")
    timestamps_t = torch.tensor(ts, dtype=torch.long)
    y_t = torch.tensor(y, dtype=torch.long)
    tr_inds_t, val_inds_t, te_inds_t, split_buckets = temporal_edge_split(
        timestamps_t, y_t, spec
    )
    tr = tr_inds_t.numpy().astype(np.int64)
    va = val_inds_t.numpy().astype(np.int64)
    te = te_inds_t.numpy().astype(np.int64)

    e_tr, e_va, e_te = set(edge_ids[tr].tolist()), set(edge_ids[va].tolist()), set(edge_ids[te].tolist())
    cross_overlap = {
        "train_val": int(len(e_tr & e_va)),
        "train_test": int(len(e_tr & e_te)),
        "val_test": int(len(e_va & e_te)),
    }

    bucket_sec = 24 * 3600
    day = (ts // bucket_sec).astype(np.int64)
    train_days = sorted(set(int(x) for x in day[tr]))
    val_days = sorted(set(int(x) for x in day[va]))
    test_days = sorted(set(int(x) for x in day[te]))

    print("[6/8] account overlap + missing/unseen...", flush=True)
    from_id = df["from_id"].to_numpy().astype(np.int64)
    to_id = df["to_id"].to_numpy().astype(np.int64)
    acct = account_overlap(from_id, to_id, tr, va, te)
    missing = {c: int(df[c].isna().sum()) for c in df.columns}
    unseen = {
        col: {
            "val": category_unseen(df[col].to_numpy()[tr], df[col].to_numpy()[va]),
            "test": category_unseen(df[col].to_numpy()[tr], df[col].to_numpy()[te]),
        }
        for col in ("Sent Currency", "Received Currency", "Payment Format")
    }

    print("[7/8] feature-identical cross-split + label determinism...", flush=True)
    feat_hash = pd.util.hash_pandas_object(df[FEATURE_HASH_COLS], index=False).to_numpy(dtype=np.uint64)
    split_id = np.empty(n_rows, dtype=np.int8)
    split_id[tr] = 0
    split_id[va] = 1
    split_id[te] = 2
    cross_feat = cross_split_feature_identical(feat_hash, split_id)
    det = label_determinism_report(df, y)

    print("[8/8] raw peek + assemble...", flush=True)
    raw_header = pd.read_csv(raw_path, nrows=0).columns.tolist()
    raw_nrows = sum(1 for _ in open(raw_path, "rb")) - 1

    label_exclusion = {
        "label_col": LABEL_COL,
        "in_FORMATTED_TRANSACTION_COLUMNS": LABEL_COL in FORMATTED_TRANSACTION_COLUMNS,
        "in_DEFAULT_EDGE_FEATURE_COLS": LABEL_COL in DEFAULT_EDGE_FEATURE_COLS,
        "node_x": "placeholder Feature=1 only (data_loading.get_data)",
        "edge_attr_semantic_cols": MODEL_INPUT_EDGE_COLS,
        "formatted_cols_not_in_edge_attr": [
            c for c in FORMATTED_TRANSACTION_COLUMNS
            if c not in MODEL_INPUT_EDGE_COLS and c != LABEL_COL
        ],
        "ports_tds_built_from": "topology+timestamps only (data_util.ports / time_deltas)",
        "labels_enter_X": False,
        "labels_enter_edge_attr": False,
        "labels_enter_graph_construction": False,
        "labels_enter_normalization_stats": False,
        "labels_enter_sampling": False,
        "labels_enter_morphology": False,
    }

    graph_edge_counts = {
        "train_graph_edges": int(tr.shape[0]),
        "val_graph_edges": int(tr.shape[0] + va.shape[0]),
        "test_graph_edges": int(n_rows),
        "train_seed_edges": int(tr.shape[0]),
        "val_seed_edges": int(va.shape[0]),
        "test_seed_edges": int(te.shape[0]),
        "policy": {
            "train_graph": "train_edges_only",
            "val_graph": "train_union_val_edges",
            "test_graph": "all_edges_transductive",
        },
    }

    mp_context = {
        "future_split_leakage": {
            "train_seeds_see_val_or_test_edges": False,
            "val_seeds_see_test_edges": False,
            "test_is_last_split": True,
            "verdict": "no_future_split_edge_leakage_under_current_loader",
        },
        "within_split_future_context": {
            "val_seeds_can_aggregate_later_val_edges": True,
            "test_seeds_can_aggregate_later_test_edges": True,
            "train_seeds_can_aggregate_later_train_edges": True,
            "mechanism": "Neighbor sampling uses split-graph edges without temporal neighbor filter",
        },
        "legitimate_earlier_history_context": {
            "val_seeds_see_train_edges": True,
            "test_seeds_see_train_and_val_edges": True,
            "note": "Expanding temporal context intentional under current Multi-GNN loader",
        },
    }

    norm_scope = {
        "legacy_per_graph_edge_znorm": {
            "train": "z_norm(train_edge_attr)",
            "val": "z_norm(train_union_val_edge_attr)",
            "test": "z_norm(all_edge_attr)",
            "transductive_attrs": True,
        },
        "train_fit_edge_znorm": {
            "fit": "mean/std on train edge_attr only",
            "apply": "same transform to val and test",
            "transductive_attrs": False,
        },
        "node_x": "always train-fit clone to val/test",
    }

    integrity = {
        "computed_at_unix": time.time(),
        "elapsed_sec": None,
        "paths": {"formatted": str(FORMATTED), "raw": str(raw_path)},
        "sha256": {
            "formatted_transactions_csv": formatted_sha,
            "raw_SAML_D_csv": raw_sha,
        },
        "formatted": {
            "ordered_columns": cols,
            "columns_match_FORMATTED_TRANSACTION_COLUMNS": col_match,
            "expected_columns": list(FORMATTED_TRANSACTION_COLUMNS),
            "dtypes": dtype_map(df),
            "n_rows": n_rows,
            "n_positives": n_pos,
            "positive_rate": float(n_pos / n_rows),
            "EdgeID_n_unique": n_unique_edge,
            "EdgeID_unique": bool(n_unique_edge == n_rows),
            "missing_counts": missing,
        },
        "raw": {
            "ordered_columns": raw_header,
            "n_rows_linecount_minus_header": int(raw_nrows),
            "has_Laundering_type": "Laundering_type" in raw_header,
            "note": "Laundering_type not written to formatted_transactions.csv",
        },
        "amounts": {
            "Amount_Sent_equals_Amount_Received_all_rows": amt_equal,
            "n_rows_Amount_Sent_ne_Amount_Received": n_amt_ne,
            "formatter_policy": "both set from raw Amount (format_saml_d_files.py)",
        },
        "timestamps": {
            "min": int(ts.min()),
            "max": int(ts.max()),
            "n_calendar_day_buckets": int(ts.max() // bucket_sec + 1),
            "n_consecutive_decreases": n_ts_decreases,
            "n_consecutive_ties": n_ts_equal_consec,
            "sorted_nondecreasing": bool(n_ts_decreases == 0),
        },
        "duplicates": {
            "n_rows_in_duplicate_groups": n_dup_rows,
            "n_extra_duplicate_rows": n_dup_groups,
        },
        "split": {
            "mode": spec.split_mode,
            "fractions_target": list(spec.split_fractions),
            "bucket_boundaries": {
                "train_buckets": split_buckets[0],
                "val_buckets": split_buckets[1],
                "test_buckets": split_buckets[2],
                "train_day_min_max": [train_days[0], train_days[-1]] if train_days else None,
                "val_day_min_max": [val_days[0], val_days[-1]] if val_days else None,
                "test_day_min_max": [test_days[0], test_days[-1]] if test_days else None,
                "n_train_days": len(train_days),
                "n_val_days": len(val_days),
                "n_test_days": len(test_days),
            },
            "train": split_stats(y, tr),
            "val": split_stats(y, va),
            "test": split_stats(y, te),
            "EdgeID_cross_split_overlap": cross_overlap,
            "split_index_hashes": {
                "train": sha256_int64_array(tr),
                "val": sha256_int64_array(va),
                "test": sha256_int64_array(te),
            },
        },
        "entity_overlap_allowed": acct,
        "unseen_categories_vs_train": unseen,
        "feature_identical_rows_crossing_splits": cross_feat,
        "label_determinism_scan": det,
        "label_exclusion_from_model_inputs": label_exclusion,
        "normalization_scope_declared": norm_scope,
        "graph_edge_counts_by_split": graph_edge_counts,
        "message_passing_temporal_context": mp_context,
    }
    integrity["elapsed_sec"] = float(time.time() - t0)

    payload = {
        "artifact": "samld_integrity_compute",
        "no_gnn_constructed": True,
        "no_training": True,
        "no_test_evaluation": True,
        "integrity": integrity,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {OUT_JSON}", flush=True)
    print(json.dumps({
        "n_rows": n_rows,
        "n_pos": n_pos,
        "split_n": [int(tr.shape[0]), int(va.shape[0]), int(te.shape[0])],
        "cross_overlap": cross_overlap,
        "elapsed_sec": integrity["elapsed_sec"],
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
