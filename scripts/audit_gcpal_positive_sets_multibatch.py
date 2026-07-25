#!/usr/bin/env python3
"""
Multi-batch GCPAL positive-set / KNN feature diagnostic (decision-grade).

Read-only. tds=False. Seed batch size ≤2048. Dense matrices only over seed
transactions. No third KNN message-passing view. No contrastive training.

Extends the single-batch audit (scripts/audit_gcpal_positive_sets.py) which was
dominated by majority–majority pairs (only two minority seeds in the first batch).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_loading import get_data  # noqa: E402
from train_util import (  # noqa: E402
    AddEgoIds,
    FORWARD_EDGE_TYPE as FORWARD,
    add_arange_ids,
    get_hetero_seed_edge_ids,
)
from transaction_knn.features import (  # noqa: E402
    build_features_detailed,
    load_train_frame,
    standardize_features,
)

BYTES_F32 = 4
K_DEFAULT = 15
MAX_BATCHES = 64
TARGET_MINORITY_ANCHORS = 100


def matrix_memory_bytes(n: int, *, dtype_bytes: int = BYTES_F32) -> int:
    return int(n) * int(n) * int(dtype_bytes)


def build_directed_chain_mask(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    chain = dst.unsqueeze(1) == src.unsqueeze(0)
    eye = torch.eye(src.numel(), dtype=torch.bool, device=src.device)
    return chain & ~eye


def build_shared_endpoint_mask(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    s_i, d_i = src.unsqueeze(1), dst.unsqueeze(1)
    s_j, d_j = src.unsqueeze(0), dst.unsqueeze(0)
    share = (s_i == s_j) | (s_i == d_j) | (d_i == s_j) | (d_i == d_j)
    eye = torch.eye(src.numel(), dtype=torch.bool, device=src.device)
    return share & ~eye


def build_batch_knn_mask(
    features_scaled_l2: torch.Tensor,
    *,
    k: int = 15,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    b = features_scaled_l2.shape[0]
    t0 = time.perf_counter()
    sim = features_scaled_l2 @ features_scaled_l2.T
    t_sim = time.perf_counter() - t0
    sim_no_self = sim.clone()
    sim_no_self.fill_diagonal_(float("-inf"))
    kk = min(int(k), max(b - 1, 0))
    t1 = time.perf_counter()
    if kk <= 0:
        mask = torch.zeros((b, b), dtype=torch.bool, device=sim.device)
        top_sims = sim.new_empty((b, 0))
    else:
        top_sims, top_idx = torch.topk(sim_no_self, k=kk, dim=1)
        mask = torch.zeros((b, b), dtype=torch.bool, device=sim.device)
        rows = torch.arange(b, device=sim.device).unsqueeze(1).expand_as(top_idx)
        valid = torch.isfinite(top_sims)
        mask[rows[valid], top_idx[valid]] = True
        mask.fill_diagonal_(False)
    t_topk = time.perf_counter() - t1
    finite = top_sims[torch.isfinite(top_sims)].detach().cpu().numpy() if top_sims.numel() else np.array([])
    meta = {
        "k_requested": int(k),
        "k_effective": int(kk),
        "self_neighbors_removed_before_topk": True,
        "runtime_seconds": {"similarity_matrix": t_sim, "topk": t_topk},
        "cosine_similarity_of_selected_neighbors": {
            "min": float(finite.min()) if finite.size else float("nan"),
            "mean": float(finite.mean()) if finite.size else float("nan"),
            "median": float(np.median(finite)) if finite.size else float("nan"),
            "max": float(finite.max()) if finite.size else float("nan"),
            "n": int(finite.size),
        },
    }
    return mask, sim, meta


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _agg(xs: List[float]) -> Dict[str, Any]:
    arr = np.asarray([x for x in xs if x is not None and np.isfinite(x)], dtype=np.float64)
    if arr.size == 0:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p95": None,
            "max": None,
        }
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def _data_args() -> SimpleNamespace:
    return SimpleNamespace(
        data="Small-HI",
        model="gin",
        ports=True,
        tds=False,
        reverse_mp=True,
        ego=True,
        emlps=True,
        load_pattern_metadata=False,
        pattern_metadata=None,
        temporal_flow_edge_features=False,
        temporal_flow_cache=None,
        correct_reverse_edge_features=False,
    )


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (x / norms).astype(np.float32)


def prepare_feature_variants(
    df_train: pd.DataFrame,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]:
    """Build full-train L2-normalized feature matrices for each KNN variant."""
    protocols: Dict[str, Dict[str, Any]] = {}
    matrices: Dict[str, np.ndarray] = {}

    # 1) Existing ordinal edge_native (first audit protocol)
    t0 = time.perf_counter()
    r1 = build_features_detailed(
        df_train, "edge_native", categorical_encoding="ordinal", scaling="none"
    )
    scaler1 = StandardScaler().fit(r1.features)
    x1 = _l2_normalize(scaler1.transform(r1.features).astype(np.float32))
    matrices["edge_native_ordinal"] = x1
    protocols["edge_native_ordinal"] = {
        "name": "edge_native_ordinal",
        "columns": r1.names,
        "transformations": [
            "Timestamp raw",
            "log1p(Amount Received)",
            "Received Currency / Payment Format: ordinal factorize (sort=True)",
        ],
        "scaler_fitting_scope": "StandardScaler fit on full train split; transform all train rows",
        "categorical_encoding": "ordinal",
        "similarity": "cosine (row L2-normalize then inner product)",
        "k": K_DEFAULT,
        "self_neighbor_handling": "diagonal set to -inf before top-k",
        "candidate_pool": "batch-local (seed transactions only)",
        "feature_dim": int(x1.shape[1]),
        "prep_seconds": time.perf_counter() - t0,
    }

    # 2) Standardized continuous + one-hot categoricals
    t0 = time.perf_counter()
    r2 = build_features_detailed(
        df_train, "edge_native", categorical_encoding="one_hot", scaling="none"
    )
    # Identify continuous vs one-hot columns by name
    cont_idx = [
        i
        for i, n in enumerate(r2.names)
        if n in ("Timestamp",) or n.startswith("log1p_")
    ]
    cat_idx = [i for i in range(len(r2.names)) if i not in cont_idx]
    x_raw = r2.features.astype(np.float32)
    x_cont = x_raw[:, cont_idx]
    x_cat = x_raw[:, cat_idx]
    scaler2 = StandardScaler().fit(x_cont)
    x_cont_s = scaler2.transform(x_cont).astype(np.float32)
    x2 = _l2_normalize(np.concatenate([x_cont_s, x_cat], axis=1))
    matrices["edge_native_onehot"] = x2
    protocols["edge_native_onehot"] = {
        "name": "edge_native_onehot",
        "columns": [r2.names[i] for i in cont_idx] + [r2.names[i] for i in cat_idx],
        "transformations": [
            "Timestamp + log1p(Amount): StandardScaler on train",
            "Currency/Payment Format: one-hot (train vocabulary); not scaled",
            "Concat then row L2-normalize",
        ],
        "scaler_fitting_scope": "StandardScaler on continuous columns only, full train split",
        "categorical_encoding": "one_hot",
        "similarity": "cosine (row L2-normalize then inner product)",
        "k": K_DEFAULT,
        "self_neighbor_handling": "diagonal set to -inf before top-k",
        "candidate_pool": "batch-local (seed transactions only)",
        "feature_dim": int(x2.shape[1]),
        "n_continuous": len(cont_idx),
        "n_onehot": len(cat_idx),
        "prep_seconds": time.perf_counter() - t0,
    }

    # 3) Exact global-cache feature definition: edge_native+degree_fan, ordinal, legacy_standard
    t0 = time.perf_counter()
    r3 = build_features_detailed(
        df_train,
        "edge_native+degree_fan",
        categorical_encoding="ordinal",
        scaling="none",
    )
    x3_std = standardize_features(r3.features)  # global StandardScaler (legacy_standard)
    x3 = _l2_normalize(x3_std)
    matrices["global_cache_matched"] = x3
    protocols["global_cache_matched"] = {
        "name": "global_cache_matched",
        "columns": r3.names,
        "transformations": [
            "edge_native (ordinal) + degree_fan train-graph degrees",
            "legacy_standard = global StandardScaler over all columns",
            "row L2-normalize for cosine (matches precompute --metric cosine)",
        ],
        "scaler_fitting_scope": "StandardScaler fit on full train feature matrix (same as cache precompute default)",
        "categorical_encoding": "ordinal",
        "similarity": "cosine",
        "k": K_DEFAULT,
        "self_neighbor_handling": "diagonal set to -inf before top-k (batch-local); cache excludes self",
        "candidate_pool": "batch-local for primary; also compared to global/cache neighbors",
        "feature_set": "edge_native+degree_fan",
        "matches_cache_path": "morphology_cache/Small-HI/transaction_knn_edge_native_degree_fan_k15.npz",
        "feature_dim": int(x3.shape[1]),
        "prep_seconds": time.perf_counter() - t0,
        "note": (
            "Cache metadata feature_names omit '_ordinal' suffixes from an older dump; "
            "current builder uses '*_ordinal' names with equivalent factorize codes."
        ),
    }
    return matrices, protocols


def build_train_structural_index(
    src: np.ndarray, dst: np.ndarray
) -> Dict[str, Any]:
    """Maps for directed-chain / shared-endpoint lookup on the full train graph."""
    n = int(src.shape[0])
    out_edges: Dict[int, List[int]] = defaultdict(list)
    in_edges: Dict[int, List[int]] = defaultdict(list)
    for i in range(n):
        s, d = int(src[i]), int(dst[i])
        out_edges[s].append(i)
        in_edges[d].append(i)
    return {"out_edges": out_edges, "in_edges": in_edges, "n_edges": n}


def directed_chain_neighbors_full(
    edge_id: int, src: np.ndarray, dst: np.ndarray, index: Dict[str, Any]
) -> np.ndarray:
    """Edges j with receiver(i)==sender(j), j!=i."""
    recv = int(dst[edge_id])
    cand = index["out_edges"].get(recv, [])
    return np.asarray([j for j in cand if j != edge_id], dtype=np.int64)


def evaluate_knn_class_split(
    *,
    seed_ids: np.ndarray,
    labels: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    ts: np.ndarray,
    amount: np.ndarray,
    currency: np.ndarray,
    payfmt: np.ndarray,
    knn_idx: np.ndarray,
    knn_sim: np.ndarray,
    train_pos_rate: float,
    k: int,
) -> Dict[str, Any]:
    """Per-class diagnostics for a single batch KNN."""
    b = int(seed_ids.shape[0])
    labs = labels.astype(np.int64)
    out: Dict[str, Any] = {}
    for cls_name, cls in (("minority", 1), ("majority", 0)):
        anchors = np.where(labs == cls)[0]
        base = train_pos_rate if cls == 1 else (1.0 - train_pos_rate)
        if anchors.size == 0:
            out[cls_name] = {"n_anchors": 0, "class_base_rate_train": float(base)}
            continue
        precisions = []
        recalls = []
        lifts = []
        has_same = []
        sims = []
        dup_rates = []
        same_acct = []
        same_cur = []
        same_pay = []
        dt_all = []
        damt_all = []
        n_neighbors = []
        other_same = int((labs == cls).sum()) - 1
        for a in anchors:
            nbrs = knn_idx[a]
            sims_a = knn_sim[a]
            valid = nbrs >= 0
            nbrs = nbrs[valid]
            sims_a = sims_a[valid]
            n_neighbors.append(int(nbrs.size))
            if nbrs.size == 0:
                continue
            nbr_labs = labs[nbrs]
            same = nbr_labs == cls
            prec = float(same.mean())
            precisions.append(prec)
            lifts.append(prec / base if base > 0 else float("nan"))
            has_same.append(float(same.any()))
            sims.extend(sims_a.tolist())
            if other_same > 0:
                recalls.append(float(same.sum()) / float(other_same))
            dup_rates.append(float((sims_a >= 0.9999).mean()))
            sa, da = int(src[a]), int(dst[a])
            same_acct.append(
                float(
                    np.mean(
                        [
                            (int(src[j]) in (sa, da)) or (int(dst[j]) in (sa, da))
                            for j in nbrs
                        ]
                    )
                )
            )
            same_cur.append(float(np.mean(currency[nbrs] == currency[a])))
            same_pay.append(float(np.mean(payfmt[nbrs] == payfmt[a])))
            dt_all.extend(np.abs(ts[nbrs] - ts[a]).tolist())
            damt_all.extend(np.abs(amount[nbrs] - amount[a]).tolist())

        out[cls_name] = {
            "n_anchors": int(anchors.size),
            "neighbors_per_anchor": _agg([float(x) for x in n_neighbors]),
            "same_label_precision_at_k": _agg(precisions),
            "same_class_neighbor_recall_in_batch": _agg(recalls),
            "lift_over_train_class_base_rate": _agg(lifts),
            "fraction_with_at_least_one_same_label_neighbor": float(np.mean(has_same))
            if has_same
            else None,
            "average_similarity": _agg(sims),
            "exact_feature_duplicate_rate_sim_ge_0.9999": _agg(dup_rates),
            "same_account_pair_rate": _agg(same_acct),
            "same_currency_rate": _agg(same_cur),
            "same_payment_format_rate": _agg(same_pay),
            "temporal_distance_abs": _agg(dt_all),
            "amount_distance_abs_log1p": _agg(damt_all),
            "class_base_rate_train": float(base),
        }
    return out


def knn_from_features(
    x_batch: np.ndarray, *, k: int, device: torch.device
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    xt = torch.from_numpy(x_batch).to(device)
    mask, sim, meta = build_batch_knn_mask(xt, k=k)
    b = x_batch.shape[0]
    kk = min(k, max(b - 1, 0))
    if kk <= 0:
        return (
            np.full((b, 0), -1, dtype=np.int64),
            np.full((b, 0), np.nan, dtype=np.float32),
            meta,
        )
    sim_no = sim.clone()
    sim_no.fill_diagonal_(float("-inf"))
    top_sims, top_idx = torch.topk(sim_no, k=kk, dim=1)
    return (
        top_idx.detach().cpu().numpy().astype(np.int64),
        top_sims.detach().cpu().numpy().astype(np.float32),
        meta,
    )


def hub_stats(src: np.ndarray, dst: np.ndarray, knn_idx: np.ndarray) -> Dict[str, Any]:
    accounts = np.concatenate([src, dst])
    uniq, counts = np.unique(accounts, return_counts=True)
    order = np.argsort(-counts)
    top_n = max(1, int(math.ceil(0.01 * len(uniq))))
    top = set(int(x) for x in uniq[order[:top_n]].tolist())
    touch = 0
    total = 0
    for a in range(knn_idx.shape[0]):
        for j in knn_idx[a]:
            if j < 0:
                continue
            total += 1
            if (
                int(src[a]) in top
                or int(dst[a]) in top
                or int(src[j]) in top
                or int(dst[j]) in top
            ):
                touch += 1
    return {
        "n_unique_accounts": int(len(uniq)),
        "top_1pct_account_count": int(top_n),
        "fraction_knn_pairs_touching_top_1pct_accounts": float(touch / total)
        if total
        else None,
        "max_account_endpoint_degree_in_batch": int(counts.max()) if counts.size else 0,
    }


def structural_batch_and_full(
    seed_ids: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    labels: np.ndarray,
    full_src: np.ndarray,
    full_dst: np.ndarray,
    struct_index: Dict[str, Any],
) -> Dict[str, Any]:
    b = seed_ids.shape[0]
    src_t = torch.from_numpy(src.astype(np.int64))
    dst_t = torch.from_numpy(dst.astype(np.int64))
    chain = build_directed_chain_mask(src_t, dst_t).numpy()
    shared = build_shared_endpoint_mask(src_t, dst_t).numpy()
    seed_set = set(int(x) for x in seed_ids.tolist())
    id_to_local = {int(e): i for i, e in enumerate(seed_ids.tolist())}

    chain_in_batch = []
    chain_in_full = []
    chain_absent = []
    for local_i, eid in enumerate(seed_ids.tolist()):
        full_nbrs = directed_chain_neighbors_full(int(eid), full_src, full_dst, struct_index)
        chain_in_full.append(int(full_nbrs.size))
        in_batch = [n for n in full_nbrs.tolist() if int(n) in seed_set]
        chain_in_batch.append(len(in_batch))
        chain_absent.append(int(full_nbrs.size) - len(in_batch))

    # minority structural coverage
    min_idx = np.where(labels == 1)[0]
    return {
        "directed_chain_nonid_pairs_in_batch": int(chain.sum()),
        "shared_endpoint_nonid_pairs_in_batch": int(shared.sum()),
        "directed_chain_density": float(chain.mean()) if b else None,
        "shared_endpoint_density": float(shared.mean()) if b else None,
        "directed_chain_full_graph_neighbors_per_seed": _agg([float(x) for x in chain_in_full]),
        "directed_chain_neighbors_present_in_batch": _agg([float(x) for x in chain_in_batch]),
        "directed_chain_neighbors_absent_from_batch": _agg([float(x) for x in chain_absent]),
        "fraction_full_chain_neighbors_captured_in_batch": float(
            np.sum(chain_in_batch) / max(np.sum(chain_in_full), 1)
        ),
        "minority_anchors_with_any_directed_chain_in_batch": int(
            sum(1 for i in min_idx if chain[i].any())
        )
        if min_idx.size
        else 0,
        "minority_anchors_with_any_shared_endpoint_in_batch": int(
            sum(1 for i in min_idx if shared[i].any())
        )
        if min_idx.size
        else 0,
    }


def global_cache_overlap_matched(
    seed_ids: np.ndarray,
    knn_idx: np.ndarray,
    cache_neigh: np.ndarray,
) -> Dict[str, Any]:
    """Overlap between batch-local KNN and global cache (same feature family)."""
    b = seed_ids.shape[0]
    seed_set = set(int(x) for x in seed_ids.tolist())
    hit = 0
    total = 0
    cache_in_batch = 0
    cache_slots = 0
    for local_i, eid in enumerate(seed_ids.tolist()):
        eid = int(eid)
        if eid < 0 or eid >= cache_neigh.shape[0]:
            continue
        cache_set = {int(x) for x in cache_neigh[eid] if int(x) >= 0}
        for j in knn_idx[local_i]:
            if j < 0:
                continue
            total += 1
            tid = int(seed_ids[j])
            if tid in cache_set:
                hit += 1
        for nb in cache_set:
            cache_slots += 1
            if nb in seed_set and nb != eid:
                cache_in_batch += 1
    return {
        "batch_local_knn_pairs": int(total),
        "batch_local_pairs_also_in_global_cache": int(hit),
        "fraction_batch_knn_in_global_cache": float(hit / total) if total else None,
        "global_cache_neighbors_landing_in_seed_batch": int(cache_in_batch),
        "global_cache_neighbor_slots_examined": int(cache_slots),
        "features_matched": True,
    }


def neighbor_stability(
    history: Dict[int, List[np.ndarray]],
) -> Dict[str, Any]:
    """Jaccard of neighbor sets for seeds appearing in ≥2 batches."""
    jacs = []
    n_multi = 0
    for eid, nbr_lists in history.items():
        if len(nbr_lists) < 2:
            continue
        n_multi += 1
        # pairwise among appearances
        for i in range(len(nbr_lists)):
            for j in range(i + 1, len(nbr_lists)):
                a = set(int(x) for x in nbr_lists[i].tolist() if int(x) >= 0)
                b = set(int(x) for x in nbr_lists[j].tolist() if int(x) >= 0)
                union = a | b
                inter = a & b
                jacs.append(float(len(inter) / len(union)) if union else 1.0)
    return {
        "n_seeds_seen_in_multiple_batches": n_multi,
        "pairwise_neighbor_jaccard_across_batch_compositions": _agg(jacs),
    }


def decide(
    pooled_min: Dict[str, Any],
    stability: Dict[str, Any],
    cache_overlap: Dict[str, Any],
    n_min_anchors: int,
    best_variant: str,
) -> Dict[str, Any]:
    """Map evidence to A/B/C/D."""
    if n_min_anchors < 30:
        return {
            "decision": "D",
            "rationale": (
                f"Only {n_min_anchors} minority anchors observed (target ≥100, hard cap 64 batches). "
                "Evidence remains insufficient for a training scout."
            ),
            "recommended_scout": None,
        }

    prec = (pooled_min or {}).get("same_label_precision_at_k", {}).get("mean")
    lift = (pooled_min or {}).get("lift_over_train_class_base_rate", {}).get("mean")
    frac = (pooled_min or {}).get("fraction_with_at_least_one_same_label_neighbor")
    jac = (
        (stability or {})
        .get("pairwise_neighbor_jaccard_across_batch_compositions", {})
        .get("mean")
    )
    cache_frac = (cache_overlap or {}).get("fraction_batch_knn_in_global_cache")

    # Useful minority signal: precision well above chance and some lift
    useful = (
        prec is not None
        and lift is not None
        and prec >= 0.02
        and lift >= 2.0
        and (frac or 0) >= 0.05
    )
    unstable = jac is not None and jac < 0.25
    cache_disagreement = cache_frac is not None and cache_frac < 0.05

    if useful and not unstable:
        return {
            "decision": "A",
            "rationale": (
                f"Minority-conditioned KNN shows mean same-label P@15={prec:.4f} "
                f"(lift={lift:.2f}) under {best_variant}; neighbor sets stable enough "
                f"for a small positive-set scout."
            ),
            "recommended_scout": {
                "feature_definition": best_variant,
                "retrieval": "batch-local",
                "note": "Do not auto-submit; await explicit approval.",
            },
        }
    if useful and (unstable or cache_disagreement):
        return {
            "decision": "B",
            "rationale": (
                f"Minority signal exists (P@15={prec}, lift={lift}) under {best_variant}, "
                f"but neighbor Jaccard across batches={jac} and/or batch↔global overlap="
                f"{cache_frac}. Prefer global/approximate retrieval for any scout."
            ),
            "recommended_scout": {
                "feature_definition": best_variant,
                "retrieval": "global",
                "note": "Do not auto-submit; await explicit approval.",
            },
        }
    if n_min_anchors >= 50 and (prec is None or lift is None or lift < 1.5):
        return {
            "decision": "C",
            "rationale": (
                f"With {n_min_anchors} minority anchors, tested KNN features show no useful "
                f"minority-conditioned lift (best P@15={prec}, lift={lift})."
            ),
            "recommended_scout": None,
        }
    return {
        "decision": "D",
        "rationale": (
            f"Borderline/insufficient: n_minority={n_min_anchors}, best P@15={prec}, "
            f"lift={lift}, stability_jaccard={jac}."
        ),
        "recommended_scout": None,
    }


def write_markdown(path: Path, payload: Dict[str, Any]) -> None:
    lines = [
        "# GCPAL positive-set multi-batch KNN diagnostic",
        "",
        "Read-only. `tds=False`. Seed B≤2048. Pairwise over seed transactions only. "
        "**No** third KNN MP view. **No** contrastive training.",
        "",
        f"- Batches processed: **{payload['n_batches']}** (cap {MAX_BATCHES})",
        f"- Minority anchors achieved: **{payload['n_minority_anchors_total']}** "
        f"(target {TARGET_MINORITY_ANCHORS})",
        f"- Train positive rate: **{payload['train_positive_rate']:.6f}**",
        f"- Primary sampling: natural LinkNeighborLoader batches (no edge-drop filter on seeds)",
        "",
        "## Feature protocols",
        "",
    ]
    for name, proto in payload["feature_protocols"].items():
        lines += [
            f"### `{name}`",
            f"- columns: `{proto['columns']}`",
            f"- transformations: {proto['transformations']}",
            f"- scaler: {proto['scaler_fitting_scope']}",
            f"- categorical: {proto['categorical_encoding']}",
            f"- similarity / k / self: {proto['similarity']} / {proto['k']} / {proto['self_neighbor_handling']}",
            f"- candidate pool: {proto['candidate_pool']}",
            f"- dim: {proto['feature_dim']}",
            "",
        ]

    lines += ["## Pooled minority / majority KNN metrics", ""]
    for vname, block in payload["pooled_by_variant"].items():
        lines.append(f"### {vname}")
        for cls in ("minority", "majority"):
            m = block.get(cls, {})
            if not m or m.get("n_anchors", 0) == 0:
                lines.append(f"- **{cls}**: no anchors")
                continue
            p = m.get("same_label_precision_at_k", {})
            lift = m.get("lift_over_train_class_base_rate", {})
            lines.append(
                f"- **{cls}** (n={m['n_anchors']}): P@15 mean={p.get('mean')} "
                f"lift={lift.get('mean')} "
                f"frac≥1 same={m.get('fraction_with_at_least_one_same_label_neighbor')} "
                f"avg_sim={m.get('average_similarity', {}).get('mean')}"
            )
        lines.append("")

    lines += [
        "## Cross-batch summary (minority P@15 by variant)",
        "",
        "| Variant | mean | SD | median | p25 | p75 |",
        "|---------|-----:|---:|-------:|----:|----:|",
    ]
    for vname, s in payload["across_batch_minority_precision"].items():
        lines.append(
            f"| {vname} | {s.get('mean')} | {s.get('std')} | {s.get('median')} | "
            f"{s.get('p25')} | {s.get('p75')} |"
        )

    dec = payload["decision"]
    lines += [
        "",
        "## Batch↔global overlap (matched cache features)",
        "",
        f"```json\n{json.dumps(payload.get('global_cache_overlap_pooled', {}), indent=2)}\n```",
        "",
        "## Neighbor stability",
        "",
        f"```json\n{json.dumps(payload.get('neighbor_stability', {}), indent=2)}\n```",
        "",
        "## Structural coverage",
        "",
        f"```json\n{json.dumps(payload.get('structural_pooled', {}), indent=2)}\n```",
        "",
        "## Runtime / memory",
        "",
        f"```json\n{json.dumps(payload.get('resources', {}), indent=2)}\n```",
        "",
        "## Decision",
        "",
        f"**{dec['decision']}** — {dec['rationale']}",
        "",
    ]
    if dec.get("recommended_scout"):
        lines += [
            "### Recommended scout (not launched)",
            "",
            f"```json\n{json.dumps(dec['recommended_scout'], indent=2)}\n```",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)-5s] %(message)s")
    device = torch.device(
        args.device
        if torch.cuda.is_available() or not str(args.device).startswith("cuda")
        else "cpu"
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t_all = time.perf_counter()
    with open(args.data_config, "r", encoding="utf-8") as f:
        data_config = json.load(f)

    logging.info("Loading Small-HI graph (tds=False)...")
    t0 = time.perf_counter()
    tr_data, _, _, tr_inds, _, _ = get_data(_data_args(), data_config)
    assert isinstance(tr_data, HeteroData)
    data_load_s = time.perf_counter() - t0

    add_arange_ids([tr_data])
    fwd = tr_data[FORWARD]
    full_src = fwd.edge_index[0].detach().cpu().numpy().astype(np.int64)
    full_dst = fwd.edge_index[1].detach().cpu().numpy().astype(np.int64)
    full_y = fwd.y.detach().cpu().numpy().astype(np.int64)
    n_train = int(full_y.shape[0])
    train_pos_rate = float(full_y.mean())

    logging.info("Building structural index...")
    struct_index = build_train_structural_index(full_src, full_dst)

    logging.info("Loading train frame + feature variants...")
    t0 = time.perf_counter()
    _, df_train, _, _ = load_train_frame("Small-HI", args.data_config)
    assert len(df_train) == n_train
    matrices, protocols = prepare_feature_variants(df_train)
    # raw columns for distance diagnostics (train-local row = edge id after add_arange)
    ts_all = df_train["Timestamp"].astype(float).to_numpy()
    amt_col = "Amount Received" if "Amount Received" in df_train.columns else "Amount Sent"
    amount_all = np.log1p(np.maximum(df_train[amt_col].astype(float).to_numpy(), 0.0))
    currency_all = pd.factorize(df_train["Received Currency"].fillna("?").astype(str), sort=True)[0]
    payfmt_all = pd.factorize(df_train["Payment Format"].fillna("?").astype(str), sort=True)[0]
    feat_prep_s = time.perf_counter() - t0

    cache_path = Path(args.knn_cache)
    cache_neigh = None
    if cache_path.is_file():
        z = np.load(cache_path, allow_pickle=True)
        cache_neigh = z["neighbor_ids"].astype(np.int64)
        logging.info("Loaded global KNN cache %s shape=%s", cache_path, cache_neigh.shape)

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    loader = LinkNeighborLoader(
        tr_data,
        num_neighbors=list(args.num_neighbors),
        edge_label_index=(FORWARD, tr_data[FORWARD].edge_index),
        edge_label=tr_data[FORWARD].y,
        batch_size=int(args.batch_size),
        shuffle=True,
        transform=AddEgoIds(),
        num_workers=0,
    )

    variant_names = list(matrices.keys())
    batch_rows: List[Dict[str, Any]] = []
    pooled_prec: Dict[str, List[float]] = {v: [] for v in variant_names}
    pooled_class_metrics: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        v: {"minority": [], "majority": []} for v in variant_names
    }
    neigh_history: Dict[str, Dict[int, List[np.ndarray]]] = {
        v: defaultdict(list) for v in variant_names
    }
    cache_overlaps = []
    structural_rows = []
    n_min_total = 0
    n_batches = 0
    sim_runtime = []

    logging.info(
        "Processing up to %d batches (target %d minority anchors)...",
        MAX_BATCHES,
        TARGET_MINORITY_ANCHORS,
    )
    for batch in loader:
        if n_batches >= MAX_BATCHES or n_min_total >= TARGET_MINORITY_ANCHORS:
            break
        seed_ids_t = get_hetero_seed_edge_ids(batch, tr_data)
        seed_ids = seed_ids_t.detach().cpu().numpy().astype(np.int64)
        # Natural seeds: no view-drop filter (documented departure from single-batch audit)
        assert seed_ids.size <= int(args.batch_size)
        b = int(seed_ids.size)
        labels = full_y[seed_ids]
        src = full_src[seed_ids]
        dst = full_dst[seed_ids]
        n_min = int((labels == 1).sum())
        n_min_total += n_min
        n_batches += 1
        n_mp = int(batch[FORWARD].edge_index.shape[1])
        logging.info(
            "batch %d: seeds=%d minority=%d mp_fwd=%d cum_min=%d",
            n_batches,
            b,
            n_min,
            n_mp,
            n_min_total,
        )

        struct = structural_batch_and_full(
            seed_ids, src, dst, labels, full_src, full_dst, struct_index
        )
        structural_rows.append(struct)

        batch_variant = {}
        for vname in variant_names:
            x_batch = matrices[vname][seed_ids]
            t1 = time.perf_counter()
            knn_idx, knn_sim, meta = knn_from_features(x_batch, k=K_DEFAULT, device=device)
            sim_runtime.append(time.perf_counter() - t1)
            assert knn_idx.shape[0] == b
            class_metrics = evaluate_knn_class_split(
                seed_ids=seed_ids,
                labels=labels,
                src=src,
                dst=dst,
                ts=ts_all[seed_ids],
                amount=amount_all[seed_ids],
                currency=currency_all[seed_ids],
                payfmt=payfmt_all[seed_ids],
                knn_idx=knn_idx,
                knn_sim=knn_sim,
                train_pos_rate=train_pos_rate,
                k=K_DEFAULT,
            )
            hubs = hub_stats(src, dst, knn_idx)
            # stability history: store global neighbor edge ids
            for local_i, eid in enumerate(seed_ids.tolist()):
                nbr_eids = np.array(
                    [int(seed_ids[j]) for j in knn_idx[local_i] if j >= 0],
                    dtype=np.int64,
                )
                neigh_history[vname][int(eid)].append(nbr_eids)

            if vname == "global_cache_matched" and cache_neigh is not None:
                ov = global_cache_overlap_matched(seed_ids, knn_idx, cache_neigh)
                cache_overlaps.append(ov)
            else:
                ov = None

            # track minority precision for across-batch agg
            mp = class_metrics.get("minority", {}).get("same_label_precision_at_k", {})
            if mp and mp.get("mean") is not None:
                pooled_prec[vname].append(float(mp["mean"]))
            for cls in ("minority", "majority"):
                pooled_class_metrics[vname][cls].append(class_metrics.get(cls, {}))

            batch_variant[vname] = {
                "class_metrics": class_metrics,
                "hub": hubs,
                "knn_meta": {
                    "k_effective": meta.get("k_effective"),
                    "sim_summary": meta.get("cosine_similarity_of_selected_neighbors"),
                },
                "cache_overlap": ov,
            }

        batch_rows.append(
            {
                "batch_index": n_batches,
                "n_seeds": b,
                "n_minority": n_min,
                "n_majority": int((labels == 0).sum()),
                "n_message_passing_forward_edges": n_mp,
                "dense_seed_sim_gib": matrix_memory_bytes(b) / (1024**3),
                "mistaken_mp_sim_gib": matrix_memory_bytes(n_mp) / (1024**3),
                "structural": struct,
                "variants": batch_variant,
            }
        )

    # Pool class metrics (recompute means of batch-level means; also weight by anchors)
    pooled_by_variant: Dict[str, Any] = {}
    for vname in variant_names:
        pooled_by_variant[vname] = {}
        for cls in ("minority", "majority"):
            rows = [r for r in pooled_class_metrics[vname][cls] if r.get("n_anchors", 0) > 0]
            if not rows:
                pooled_by_variant[vname][cls] = {"n_anchors": 0}
                continue
            # Weighted by n_anchors for key scalars
            def wmean(key_path: Tuple[str, str]) -> Optional[float]:
                num = 0.0
                den = 0.0
                for r in rows:
                    n = float(r.get("n_anchors", 0))
                    block = r.get(key_path[0], {})
                    val = block.get(key_path[1]) if isinstance(block, dict) else None
                    if val is None:
                        continue
                    num += n * float(val)
                    den += n
                return float(num / den) if den else None

            frac_vals = []
            for r in rows:
                n = r.get("n_anchors", 0)
                f = r.get("fraction_with_at_least_one_same_label_neighbor")
                if f is not None and n:
                    frac_vals.extend([f] * int(n))
            pooled_by_variant[vname][cls] = {
                "n_anchors": int(sum(r.get("n_anchors", 0) for r in rows)),
                "same_label_precision_at_k": {
                    "mean": wmean(("same_label_precision_at_k", "mean"))
                },
                "lift_over_train_class_base_rate": {
                    "mean": wmean(("lift_over_train_class_base_rate", "mean"))
                },
                "fraction_with_at_least_one_same_label_neighbor": float(np.mean(frac_vals))
                if frac_vals
                else None,
                "average_similarity": {"mean": wmean(("average_similarity", "mean"))},
                "exact_feature_duplicate_rate_sim_ge_0.9999": {
                    "mean": wmean(("exact_feature_duplicate_rate_sim_ge_0.9999", "mean"))
                },
                "same_account_pair_rate": {"mean": wmean(("same_account_pair_rate", "mean"))},
                "same_currency_rate": {"mean": wmean(("same_currency_rate", "mean"))},
                "same_payment_format_rate": {
                    "mean": wmean(("same_payment_format_rate", "mean"))
                },
            }

    across_batch_prec = {v: _agg(pooled_prec[v]) for v in variant_names}
    # pick best variant by minority lift then precision
    best_variant = max(
        variant_names,
        key=lambda v: (
            pooled_by_variant[v].get("minority", {}).get("lift_over_train_class_base_rate", {}).get("mean")
            or -1.0,
            pooled_by_variant[v].get("minority", {}).get("same_label_precision_at_k", {}).get("mean")
            or -1.0,
        ),
    )
    stability = {
        v: neighbor_stability(neigh_history[v]) for v in variant_names
    }
    cache_pooled = {
        "n_batches_with_overlap": len(cache_overlaps),
        "fraction_batch_knn_in_global_cache": _agg(
            [
                float(x["fraction_batch_knn_in_global_cache"])
                for x in cache_overlaps
                if x.get("fraction_batch_knn_in_global_cache") is not None
            ]
        ),
        "global_cache_neighbors_landing_in_seed_batch": _agg(
            [float(x["global_cache_neighbors_landing_in_seed_batch"]) for x in cache_overlaps]
        ),
        "note": "Computed only for global_cache_matched batch-local KNN vs existing cache.",
    }
    structural_pooled = {
        "directed_chain_nonid_pairs_in_batch": _agg(
            [float(s["directed_chain_nonid_pairs_in_batch"]) for s in structural_rows]
        ),
        "shared_endpoint_nonid_pairs_in_batch": _agg(
            [float(s["shared_endpoint_nonid_pairs_in_batch"]) for s in structural_rows]
        ),
        "fraction_full_chain_neighbors_captured_in_batch": _agg(
            [float(s["fraction_full_chain_neighbors_captured_in_batch"]) for s in structural_rows]
        ),
        "directed_chain_neighbors_absent_from_batch": _agg(
            [
                float(s["directed_chain_neighbors_absent_from_batch"]["mean"])
                for s in structural_rows
                if s.get("directed_chain_neighbors_absent_from_batch", {}).get("mean") is not None
            ]
        ),
    }

    decision = decide(
        pooled_by_variant[best_variant].get("minority", {}),
        stability[best_variant],
        {
            "fraction_batch_knn_in_global_cache": cache_pooled["fraction_batch_knn_in_global_cache"].get(
                "mean"
            )
        },
        n_min_total,
        best_variant,
    )

    peak = {}
    if device.type == "cuda":
        peak = {
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / (1024**2),
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / (1024**2),
        }

    payload = {
        "title": "GCPAL positive-set multi-batch KNN diagnostic",
        "constraints": {
            "tds": False,
            "batch_size_cap": int(args.batch_size),
            "max_batches": MAX_BATCHES,
            "target_minority_anchors": TARGET_MINORITY_ANCHORS,
            "pairwise_scope": "seed transactions only",
            "no_third_knn_view": True,
            "no_contrastive_training": True,
            "primary_batches": "natural LinkNeighborLoader seeds (no view edge-drop filter)",
            "secondary_minority_enriched": "not required for primary decision; omitted unless insufficient",
        },
        "n_batches": n_batches,
        "n_minority_anchors_total": n_min_total,
        "train_positive_rate": train_pos_rate,
        "n_train_edges": n_train,
        "feature_protocols": protocols,
        "batches": batch_rows,
        "pooled_by_variant": pooled_by_variant,
        "across_batch_minority_precision": across_batch_prec,
        "neighbor_stability": stability,
        "global_cache_overlap_pooled": cache_pooled,
        "structural_pooled": structural_pooled,
        "best_variant_by_minority_lift": best_variant,
        "decision": decision,
        "resources": {
            "data_load_seconds": data_load_s,
            "feature_prep_seconds": feat_prep_s,
            "knn_similarity_seconds_per_batch": _agg(sim_runtime),
            "wall_seconds": time.perf_counter() - t_all,
            "device": str(device),
            **peak,
        },
        "first_audit_limitation": {
            "note": "Single batch had only 2 minority seeds; overall purity dominated by majority–majority pairs; directed-chain had 9 non-id pairs.",
            "reference": "notes/gcpal_positive_set_audit.md",
        },
    }
    return payload


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_config", default="data_config.json")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--num_neighbors", type=int, nargs=2, default=[100, 100])
    p.add_argument(
        "--knn_cache",
        default="morphology_cache/Small-HI/transaction_knn_edge_native_degree_fan_k15.npz",
    )
    p.add_argument(
        "--output_json",
        default="results/diagnostics/gcpal_positive_set_multibatch_audit.json",
    )
    p.add_argument(
        "--output_md",
        default="notes/gcpal_positive_set_multibatch_audit.md",
    )
    args = p.parse_args()
    if int(args.batch_size) > 2048:
        raise SystemExit("batch_size must be ≤ 2048")

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    if out_json.exists():
        raise SystemExit(f"ABORT: refusing overwrite {out_json}")
    if out_md.exists():
        raise SystemExit(f"ABORT: refusing overwrite {out_md}")

    payload = run(args)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(_jsonable(payload), indent=2) + "\n", encoding="utf-8")
    write_markdown(out_md, payload)
    logging.info("Wrote %s", out_json)
    logging.info("Wrote %s", out_md)
    logging.info("DECISION %s", payload["decision"]["decision"])
    print(out_json)
    print(out_md)


if __name__ == "__main__":
    main()
