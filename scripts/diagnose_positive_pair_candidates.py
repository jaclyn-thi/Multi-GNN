#!/usr/bin/env python3
"""Non-training diagnostic for candidate contrastive soft-positive rules (label-free)."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataset_specs import get_dataset_spec
from dataset_splits import temporal_edge_split
from transaction_knn.richer_features import compute_causal_edge_stats, resolve_amount_column

DEFAULT_DATA = "Small-HI"
DEFAULT_SAMPLE_ANCHORS = 3000
DEFAULT_MAX_TRAIN_ROWS = 400_000
DEFAULT_TEMPORAL_WINDOW_SEC = 86_400.0  # 1 calendar day
DEFAULT_AMOUNT_LOG_DELTA = 0.5


def _load_train_edges(data: str, max_rows: int) -> Tuple[pd.DataFrame, str]:
    spec = get_dataset_spec(data)
    csv_path = _ROOT / "aml-data" / data / "formatted_transactions.csv"
    if not csv_path.is_file():
        raise FileNotFoundError("Missing {0}".format(csv_path))
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    amount_col = resolve_amount_column(pd.DataFrame(columns=header))
    usecols = ["EdgeID", "from_id", "to_id", "Timestamp", amount_col]
    df = pd.read_csv(csv_path, usecols=usecols)
    ts = torch.tensor(df["Timestamp"].astype(np.float64).to_numpy())
    y = torch.zeros(len(df))
    tr_inds, _, _, _ = temporal_edge_split(ts, y, spec)
    train = df.iloc[tr_inds.numpy()].reset_index(drop=True)
    if max_rows and len(train) > max_rows:
        train = train.iloc[:max_rows].copy()
    train = train.sort_values("Timestamp", kind="mergesort").reset_index(drop=True)
    return train, amount_col


def _build_indices(train: pd.DataFrame) -> Dict[str, Any]:
    n = len(train)
    by_sender = defaultdict(list)
    by_receiver = defaultdict(list)
    by_pair = defaultdict(list)
    for i in range(n):
        s = int(train.at[i, "from_id"])
        r = int(train.at[i, "to_id"])
        t = float(train.at[i, "Timestamp"])
        amt = float(max(train.at[i, train.columns[-1]], 0.0))
        eid = int(train.at[i, "EdgeID"])
        row = (i, t, amt, s, r, eid)
        by_sender[s].append(row)
        by_receiver[r].append(row)
        by_pair[(s, r)].append(row)
    return {"by_sender": dict(by_sender), "by_receiver": dict(by_receiver), "by_pair": dict(by_pair), "n": n}


def rule_endpoint_role_temporal(
    anchor_idx: int, train: pd.DataFrame, indices: Dict[str, Any], window_sec: float
) -> List[int]:
    s = int(train.at[anchor_idx, "from_id"])
    r = int(train.at[anchor_idx, "to_id"])
    t0 = float(train.at[anchor_idx, "Timestamp"])
    hits = set()
    for u, role in ((s, "sender"), (r, "receiver")):
        bucket = indices["by_sender"] if role == "sender" else indices["by_receiver"]
        for j, t, _, sj, rj, _ in bucket.get(u, []):
            if j == anchor_idx:
                continue
            if abs(t - t0) > window_sec:
                continue
            if role == "sender" and sj != u:
                continue
            if role == "receiver" and rj != u:
                continue
            hits.add(j)
    return sorted(hits)


def rule_shared_sender_temporal_amount(
    anchor_idx: int, train: pd.DataFrame, indices: Dict[str, Any],
    window_sec: float, amount_log_delta: float, amount_col: str,
) -> List[int]:
    s = int(train.at[anchor_idx, "from_id"])
    t0 = float(train.at[anchor_idx, "Timestamp"])
    a0 = np.log1p(max(float(train.at[anchor_idx, amount_col]), 0.0))
    hits = []
    for j, t, amt, _, _, _ in indices["by_sender"].get(s, []):
        if j == anchor_idx or abs(t - t0) > window_sec:
            continue
        if abs(np.log1p(max(amt, 0.0)) - a0) <= amount_log_delta:
            hits.append(j)
    return hits


def rule_repeat_pair_temporal(
    anchor_idx: int, train: pd.DataFrame, indices: Dict[str, Any], window_sec: float
) -> List[int]:
    s = int(train.at[anchor_idx, "from_id"])
    r = int(train.at[anchor_idx, "to_id"])
    t0 = float(train.at[anchor_idx, "Timestamp"])
    hits = []
    for j, t, _, _, _, _ in indices["by_pair"].get((s, r), []):
        if j == anchor_idx:
            continue
        dt = t - t0
        if 0 < dt <= window_sec:
            hits.append(j)
    return hits


RULES = {
    "endpoint_role_temporal_v1": {
        "description": (
            "Shared endpoint u; u plays the same role (sender/receiver) on both edges; "
            "|Δt| <= W. Label-free; exclusion of identity only."
        ),
        "fn": lambda i, tr, idx, w, ac: rule_endpoint_role_temporal(i, tr, idx, w),
        "self_supervised": True,
        "uses_labels": False,
    },
    "shared_sender_temporal_amount_v1": {
        "description": (
            "Same sender; |Δt| <= W; |log1p(amount)-log1p(anchor_amount)| <= δ."
        ),
        "fn": lambda i, tr, idx, w, ac: rule_shared_sender_temporal_amount(
            i, tr, idx, w, DEFAULT_AMOUNT_LOG_DELTA, ac),
        "self_supervised": True,
        "uses_labels": False,
    },
    "repeat_pair_forward_temporal_v1": {
        "description": (
            "Same ordered (sender, receiver); forward repeat 0 < Δt <= W (prior edge earlier)."
        ),
        "fn": lambda i, tr, idx, w, ac: rule_repeat_pair_temporal(i, tr, idx, w),
        "self_supervised": True,
        "uses_labels": False,
    },
}


def _hub_concentration(
    train: pd.DataFrame, anchor_list: List[int], pos_lists: List[List[int]], top_k: int = 10
) -> Dict[str, Any]:
    node_counts = defaultdict(int)
    for anchor_idx, hits in zip(anchor_list, pos_lists):
        if not hits:
            continue
        s = int(train.at[anchor_idx, "from_id"])
        r = int(train.at[anchor_idx, "to_id"])
        node_counts[s] += 1
        node_counts[r] += 1
    ranked = sorted(node_counts.items(), key=lambda x: -x[1])[:top_k]
    total = sum(node_counts.values()) or 1
    top_share = sum(c for _, c in ranked) / total
    return {"top_nodes": ranked, "top10_share_of_hubbed_anchors": top_share}


def _summarize_counts(counts: List[int]) -> Dict[str, Any]:
    arr = np.asarray(counts, dtype=np.int64)
    return {
        "anchors_examined": int(arr.size),
        "fraction_zero": float((arr == 0).mean()),
        "fraction_one": float((arr == 1).mean()),
        "fraction_many_ge5": float((arr >= 5).mean()),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "max": int(arr.max(initial=0)),
        "p90": float(np.percentile(arr, 90)) if arr.size else 0.0,
    }


def _sample_pairs(
    train: pd.DataFrame, amount_col: str, anchor_indices: List[int],
    pos_fn, indices, window_sec: float, n_samples: int = 20, seed: int = 1,
) -> List[Dict[str, Any]]:
    rng = np.random.RandomState(seed)
    causal = compute_causal_edge_stats(train, amount_col)
    samples = []
    for anchor_idx in anchor_indices:
        hits = pos_fn(anchor_idx, train, indices, window_sec, amount_col)
        if not hits:
            continue
        j = hits[rng.randint(0, len(hits))]
        ai, aj = int(anchor_idx), int(j)
        samples.append({
            "anchor_edge_id": int(train.at[ai, "EdgeID"]),
            "positive_edge_id": int(train.at[aj, "EdgeID"]),
            "anchor_sender": int(train.at[ai, "from_id"]),
            "anchor_receiver": int(train.at[ai, "to_id"]),
            "positive_sender": int(train.at[aj, "from_id"]),
            "positive_receiver": int(train.at[aj, "to_id"]),
            "anchor_timestamp": float(train.at[ai, "Timestamp"]),
            "positive_timestamp": float(train.at[aj, "Timestamp"]),
            "delta_t_sec": float(train.at[aj, "Timestamp"] - train.at[ai, "Timestamp"]),
            "anchor_amount": float(train.at[ai, amount_col]),
            "positive_amount": float(train.at[aj, amount_col]),
            "amount_log_delta": float(
                abs(np.log1p(max(train.at[aj, amount_col], 0)) - np.log1p(max(train.at[ai, amount_col], 0)))
            ),
            "sender_dt_anchor": float(causal["sender_dt"][ai]),
            "pair_past_count_anchor": float(causal["pair_past_count"][ai]),
            "why_qualified": "rule-specific match (see rule description)",
        })
        if len(samples) >= n_samples:
            break
    return samples


def run_diagnostic(
    data: str,
    max_train_rows: int,
    n_anchors: int,
    window_sec: float,
    seed: int,
) -> Dict[str, Any]:
    train, amount_col = _load_train_edges(data, max_train_rows)
    indices = _build_indices(train)
    rng = np.random.RandomState(seed)
    n = len(train)
    anchor_pool = np.arange(n, dtype=np.int64)
    if n_anchors < n:
        anchor_pool = rng.choice(anchor_pool, size=n_anchors, replace=False)
    anchor_list = sorted(anchor_pool.tolist())

    out = {
        "dataset": data,
        "sampling": {
            "train_rows_used": n,
            "max_train_rows_cap": max_train_rows,
            "anchors_examined": len(anchor_list),
            "temporal_window_sec": window_sec,
            "amount_log_delta": DEFAULT_AMOUNT_LOG_DELTA,
            "seed": seed,
            "note": "Train split only; no laundering labels used.",
        },
        "rules": {},
    }

    all_sets = {}
    for rule_name, spec in RULES.items():
        pos_lists = [
            spec["fn"](i, train, indices, window_sec, amount_col) for i in anchor_list
        ]
        counts = [len(x) for x in pos_lists]
        all_sets[rule_name] = [set(x) for x in pos_lists]
        dts = []
        amount_d = []
        for i, hits in zip(anchor_list, pos_lists):
            t0 = float(train.at[i, "Timestamp"])
            a0 = np.log1p(max(float(train.at[i, amount_col]), 0.0))
            for j in hits[:50]:
                dts.append(abs(float(train.at[j, "Timestamp"]) - t0))
                amount_d.append(
                    abs(np.log1p(max(float(train.at[j, amount_col]), 0.0)) - a0)
                )
        out["rules"][rule_name] = {
            "definition": spec["description"],
            "self_supervised": spec["self_supervised"],
            "uses_labels": spec["uses_labels"],
            "count_summary": _summarize_counts(counts),
            "temporal_distance_sec": {
                "mean": float(np.mean(dts)) if dts else None,
                "median": float(np.median(dts)) if dts else None,
                "p90": float(np.percentile(dts, 90)) if dts else None,
            },
            "amount_log_delta": {
                "mean": float(np.mean(amount_d)) if amount_d else None,
                "median": float(np.median(amount_d)) if amount_d else None,
            },
            "hub_concentration": _hub_concentration(train, anchor_list, pos_lists),
            "estimated_storage": {
                "note": "O(E) index + O(anchors * avg_pos) lists; full train ~5M edges feasible offline",
                "avg_positives_per_anchor": float(np.mean(counts)) if counts else 0.0,
            },
        }

    rule_names = list(RULES.keys())
    overlap = {}
    for i in range(len(rule_names)):
        for j in range(i + 1, len(rule_names)):
            a, b = rule_names[i], rule_names[j]
            inter = [len(sa & sb) for sa, sb in zip(all_sets[a], all_sets[b])]
            union = [len(sa | sb) for sa, sb in zip(all_sets[a], all_sets[b])]
            overlap["{0}_vs_{1}".format(a, b)] = {
                "mean_intersection": float(np.mean(inter)),
                "mean_union": float(np.mean(union)),
                "jaccard_mean": float(np.mean([
                    (inter[k] / union[k]) if union[k] else 0.0 for k in range(len(inter))
                ])),
            }
    out["rule_overlap"] = overlap

    for rule_name, spec in RULES.items():
        out["rules"][rule_name]["samples"] = _sample_pairs(
            train, amount_col, anchor_list, spec["fn"], indices, window_sec,
        )
    return out


def write_samples_md(payload: Dict[str, Any], path: Path) -> None:
    lines = [
        "# Positive-pair candidate samples (label-free diagnostic)",
        "",
        "Dataset: **{0}** | anchors: **{1}** | window: **{2}s**".format(
            payload["dataset"],
            payload["sampling"]["anchors_examined"],
            payload["sampling"]["temporal_window_sec"],
        ),
        "",
    ]
    for rule_name, block in payload["rules"].items():
        lines.append("## {0}".format(rule_name))
        lines.append("")
        lines.append(block["definition"])
        lines.append("")
        cs = block["count_summary"]
        lines.append(
            "Anchors: {0} | zero: {1:.1%} | one: {2:.1%} | mean: {3:.2f} | max: {4}".format(
                cs["anchors_examined"], cs["fraction_zero"], cs["fraction_one"],
                cs["mean"], cs["max"],
            )
        )
        lines.append("")
        for i, s in enumerate(block.get("samples", [])[:20], 1):
            lines.append("### Sample {0}".format(i))
            lines.append("- anchor EdgeID `{0}` → positive `{1}`".format(
                s["anchor_edge_id"], s["positive_edge_id"]))
            lines.append("- roles: ({0}→{1}) vs ({2}→{3})".format(
                s["anchor_sender"], s["anchor_receiver"],
                s["positive_sender"], s["positive_receiver"]))
            lines.append("- Δt={0:.0f}s | amounts {1:.2g} vs {2:.2g} | log Δ={3:.3f}".format(
                s["delta_t_sec"], s["anchor_amount"], s["positive_amount"], s["amount_log_delta"]))
            lines.append("- pair_past_count(anchor)={0}".format(s["pair_past_count_anchor"]))
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--max-train-rows", type=int, default=DEFAULT_MAX_TRAIN_ROWS)
    parser.add_argument("--anchors", type=int, default=DEFAULT_SAMPLE_ANCHORS)
    parser.add_argument("--window-sec", type=float, default=DEFAULT_TEMPORAL_WINDOW_SEC)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--out-json",
        default="results/diagnostics/positive_pair_candidate_diagnostics.json",
    )
    parser.add_argument(
        "--out-md",
        default="notes/positive_pair_candidate_samples.md",
    )
    args = parser.parse_args()
    payload = run_diagnostic(
        args.data, args.max_train_rows, args.anchors, args.window_sec, args.seed,
    )
    out_json = _ROOT / args.out_json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    # samples are large — keep in JSON but strip for compactness optional
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_samples_md(payload, _ROOT / args.out_md)
    print("Wrote {0} and {1}".format(out_json, _ROOT / args.out_md))


if __name__ == "__main__":
    main()
