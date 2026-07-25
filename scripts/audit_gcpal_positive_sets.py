#!/usr/bin/env python3
"""
Read-only GCPAL vs edge-centric contrastive positive-set audit.

One deterministic Small-HI LinkNeighborLoader seed batch (B<=2048), tds=False.
Dense similarity / positive masks are constructed ONLY over surviving seed
transactions (shape exactly [B_seed, B_seed]). No training, no optimizer step,
no permanent KNN cache generation.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.data import HeteroData

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_loading import get_data  # noqa: E402
from gcpal_positive_diagnostics import (  # noqa: E402
    merge_positive_masks,
    multipositive_infonce_reference,
)
from graph_augmentations import generate_views  # noqa: E402
from train_util import (  # noqa: E402
    AddEgoIds,
    FORWARD_EDGE_TYPE as FORWARD,
    add_arange_ids,
    attach_edge_id_from_batch,
    get_hetero_seed_edge_ids,
)
from transaction_knn.features import (  # noqa: E402
    build_features_detailed,
    load_train_frame,
)


BYTES_F32 = 4
BYTES_BOOL = 1


# ---------------------------------------------------------------------------
# Paper / memory static evidence (no data load)
# ---------------------------------------------------------------------------


def paper_ambiguity_audit() -> Dict[str, Any]:
    return {
        "explicit": {
            "transactions_as_nodes": True,
            "two_random_views_edge_and_feature_dropping": True,
            "A_knn_definition": "top-k(X X^T) over transaction feature matrix X",
            "loss_combines_random_random_and_random_knn_contrast": True,
            "batch_size_search_up_to_approx": 2048,
            "k_optimum_reported_around": 15,
            "eq8_uses_P_i": True,
            "eq9_M_P": "A + A_knn",
            "ablation_without_neighbor_positives": (
                "positives are the same nodes across the two random views (identity)"
            ),
        },
        "implied_but_omitted": {
            "identity_in_M_P": (
                "Eq. 9 writes M_P = A + A_knn without I, but the ablation that removes "
                "neighbor positives falls back to same-node-across-views identity. "
                "The implied practical positive set is therefore I ∪ A ∪ A_knn "
                "(equivalently I + A + A_knn with duplicates collapsed), even though "
                "I is absent from Eq. 9."
            ),
            "connected_neighbors_definition": (
                "A is described as connected neighbors on the transaction graph; "
                "exact adjacency (shared account vs directed payment chain) is not "
                "formalized beyond the transaction-as-node framing."
            ),
        },
        "unreproducible_without_code": {
            "knn_scope": (
                "Paper does not state whether KNN is global, batch-local, chunked, "
                "approximate, or precomputed."
            ),
            "literal_global_XXT": (
                "Literal dense X X^T over millions of AMLWorld transactions is "
                "infeasible; the paper does not release code clarifying the workaround."
            ),
            "feature_matrix_X_columns": (
                "Exact raw feature columns / categorical encoding for X are not "
                "specified at implementation level."
            ),
            "code_release": False,
        },
    }


def matrix_memory_bytes(n: int, *, dtype_bytes: int = BYTES_F32) -> int:
    return int(n) * int(n) * int(dtype_bytes)


def memory_feasibility_report(
    *,
    n_full: int = 5_078_345,
    n_train: int = 3_248_921,
) -> Dict[str, Any]:
    """Separate dense matrix storage from GNN activations / autograd."""

    def row(n: int, label: str) -> Dict[str, Any]:
        sim = matrix_memory_bytes(n, dtype_bytes=BYTES_F32)
        mask = matrix_memory_bytes(n, dtype_bytes=BYTES_BOOL)
        return {
            "label": label,
            "n": n,
            "similarity_matrix_float32_bytes": sim,
            "similarity_matrix_float32_gib": sim / (1024**3),
            "boolean_mask_bytes": mask,
            "boolean_mask_gib": mask / (1024**3),
            "feasible_on_single_gpu_80gib": (sim + mask) < 40 * (1024**3),
        }

    return {
        "note": (
            "Figures below are dense pairwise matrix storage only. They exclude "
            "GNN activations, optimizer state, and autograd graphs."
        ),
        "full_small_hi": row(n_full, "full Small-HI transactions"),
        "train_split_small_hi": row(n_train, "train-split Small-HI transactions"),
        "batch_8192": row(8192, "seed batch B=8192"),
        "batch_2048": row(2048, "seed batch B=2048"),
        "neighbor_expanded_warning": (
            "LinkNeighborLoader with num_neighbors=[100,100] can return far more "
            "than B seed edges. Applying BxB construction to all sampled MP edges "
            "would use n_sampled^2 storage, not B^2."
        ),
    }


# ---------------------------------------------------------------------------
# Multi-positive InfoNCE reference imported from gcpal_positive_diagnostics
# ---------------------------------------------------------------------------


def _percentiles(x: np.ndarray, ps: Sequence[float]) -> Dict[str, float]:
    if x.size == 0:
        return {f"p{int(p)}": float("nan") for p in ps}
    vals = np.percentile(x.astype(np.float64), list(ps))
    return {f"p{int(p)}": float(v) for p, v in zip(ps, vals)}


def summarize_positive_mask(
    mask: torch.Tensor,
    labels: torch.Tensor,
    *,
    name: str,
    src: Optional[torch.Tensor] = None,
    dst: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """Summarize a BxB boolean positive mask. Labels are diagnostic-only."""
    b = int(mask.shape[0])
    assert mask.shape == (b, b), f"{name}: expected [{b},{b}], got {tuple(mask.shape)}"
    identity = torch.eye(b, dtype=torch.bool, device=mask.device)
    non_id = mask & ~identity
    pos_per_anchor = non_id.sum(dim=1).detach().cpu().numpy().astype(np.int64)
    total_pairs = int(non_id.sum().item())
    # Include identity in total positive entries when reporting mask density
    total_with_id = int(mask.sum().item())

    labs = labels.detach().cpu().long().view(-1)
    assert labs.numel() == b
    li = labs.unsqueeze(1)
    lj = labs.unsqueeze(0)
    pair_lab = non_id.detach().cpu()
    mm = int(((li == 1) & (lj == 1) & pair_lab).sum().item())
    jj = int(((li == 0) & (lj == 0) & pair_lab).sum().item())
    cross = int(((li != lj) & pair_lab).sum().item())
    same = mm + jj
    purity = float(same / total_pairs) if total_pairs > 0 else float("nan")

    # Laundering / non-laundering same-label rates among non-id positives
    pos_from_min = int(non_id[labs == 1].sum().item()) if (labs == 1).any() else 0
    pos_from_maj = int(non_id[labs == 0].sum().item()) if (labs == 0).any() else 0
    min_same = float(mm / pos_from_min) if pos_from_min > 0 else float("nan")
    maj_same = float(jj / pos_from_maj) if pos_from_maj > 0 else float("nan")

    hub = {}
    if src is not None and dst is not None and total_pairs > 0:
        s = src.detach().cpu().long()
        d = dst.detach().cpu().long()
        accounts = torch.cat([s, d]).numpy()
        uniq, counts = np.unique(accounts, return_counts=True)
        order = np.argsort(-counts)
        top1pct_n = max(1, int(math.ceil(0.01 * len(uniq))))
        top_accounts = uniq[order[:top1pct_n]]
        top_set = set(int(x) for x in top_accounts.tolist())
        is_top_endpoint = torch.tensor(
            [(int(s[i]) in top_set) or (int(d[i]) in top_set) for i in range(b)],
            dtype=torch.bool,
        )
        # A non-id positive contributes if either endpoint transaction touches a hub account
        touch = is_top_endpoint.unsqueeze(1) | is_top_endpoint.unsqueeze(0)
        contrib = int((non_id.detach().cpu() & touch).sum().item())
        hub = {
            "n_unique_accounts_in_seed_batch": int(len(uniq)),
            "top_1pct_account_count": int(top1pct_n),
            "fraction_nonid_positives_touching_top_1pct_accounts": float(contrib / total_pairs),
            "max_account_endpoint_degree_in_seed_batch": int(counts.max()) if counts.size else 0,
        }

    sim_bytes = matrix_memory_bytes(b, dtype_bytes=BYTES_F32)
    mask_bytes = matrix_memory_bytes(b, dtype_bytes=BYTES_BOOL)
    sparse_nnz = total_with_id
    sparse_est = sparse_nnz * (8 + 8)  # rough int64 index pair estimate

    return {
        "name": name,
        "anchors": b,
        "anchors_with_at_least_one_non_identity_positive": int((pos_per_anchor > 0).sum()),
        "fraction_anchors_with_no_non_identity_positives": float((pos_per_anchor == 0).mean()) if b else float("nan"),
        "positives_per_anchor_non_identity": {
            "min": int(pos_per_anchor.min()) if b else 0,
            "mean": float(pos_per_anchor.mean()) if b else float("nan"),
            "median": float(np.median(pos_per_anchor)) if b else float("nan"),
            "p25": float(np.percentile(pos_per_anchor, 25)) if b else float("nan"),
            "p75": float(np.percentile(pos_per_anchor, 75)) if b else float("nan"),
            "p95": float(np.percentile(pos_per_anchor, 95)) if b else float("nan"),
            "max": int(pos_per_anchor.max()) if b else 0,
            **(_percentiles(pos_per_anchor, [25, 50, 75, 95]) if b else {}),
        },
        "total_non_identity_positive_pairs": total_pairs,
        "total_positive_mask_entries_including_identity": total_with_id,
        "positive_mask_density_including_identity": float(total_with_id / (b * b)) if b else float("nan"),
        "positive_mask_density_non_identity": float(total_pairs / (b * b)) if b else float("nan"),
        "label_diagnostics_do_not_affect_construction": True,
        "pair_counts_non_identity": {
            "minority_minority": mm,
            "majority_majority": jj,
            "cross_class": cross,
        },
        "label_agreement_purity_non_identity": purity,
        "same_label_rate_among_positives_from_minority_anchors": min_same,
        "same_label_rate_among_positives_from_majority_anchors": maj_same,
        "memory": {
            "similarity_matrix_float32_bytes": sim_bytes,
            "mask_bool_bytes": mask_bytes,
            "estimated_sparse_index_pair_bytes": int(sparse_est),
        },
        "hub_concentration": hub,
    }


def build_identity_mask(b: int, device: torch.device) -> torch.Tensor:
    return torch.eye(b, dtype=torch.bool, device=device)


def build_directed_chain_mask(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """receiver(e_i) == sender(e_j); excludes identity."""
    chain = dst.unsqueeze(1) == src.unsqueeze(0)
    eye = torch.eye(src.numel(), dtype=torch.bool, device=src.device)
    return chain & ~eye


def build_shared_endpoint_mask(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Share any account; excludes identity. Matches multi_positive_mode=same_endpoint."""
    s_i, d_i = src.unsqueeze(1), dst.unsqueeze(1)
    s_j, d_j = src.unsqueeze(0), dst.unsqueeze(0)
    share = (s_i == s_j) | (s_i == d_j) | (d_i == s_j) | (d_i == d_j)
    eye = torch.eye(src.numel(), dtype=torch.bool, device=src.device)
    return share & ~eye


def build_same_pair_mask(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Same ordered (sender, receiver); excludes identity. Matches same_pair."""
    same = (src.unsqueeze(1) == src.unsqueeze(0)) & (dst.unsqueeze(1) == dst.unsqueeze(0))
    eye = torch.eye(src.numel(), dtype=torch.bool, device=src.device)
    return same & ~eye


def build_batch_knn_mask(
    features_scaled_l2: torch.Tensor,
    *,
    k: int = 15,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Batch-local cosine KNN. ``features_scaled_l2`` must already be L2-normalized.
    Self-neighbors are removed BEFORE selecting top-k.
    """
    b = features_scaled_l2.shape[0]
    assert features_scaled_l2.ndim == 2
    t0 = time.perf_counter()
    sim = features_scaled_l2 @ features_scaled_l2.T
    t_sim = time.perf_counter() - t0
    assert sim.shape == (b, b), f"similarity must be [{b},{b}], got {tuple(sim.shape)}"

    # Remove self before top-k
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
        # Ensure diagonal never set
        mask.fill_diagonal_(False)
    t_topk = time.perf_counter() - t1

    finite = top_sims[torch.isfinite(top_sims)].detach().cpu().numpy() if top_sims.numel() else np.array([])
    meta = {
        "k_requested": int(k),
        "k_effective": int(kk),
        "self_neighbors_removed_before_topk": True,
        "similarity": "cosine (L2-normalized rows → inner product)",
        "runtime_seconds": {
            "similarity_matrix": t_sim,
            "topk": t_topk,
        },
        "cosine_similarity_of_selected_neighbors": {
            "min": float(finite.min()) if finite.size else float("nan"),
            "mean": float(finite.mean()) if finite.size else float("nan"),
            "median": float(np.median(finite)) if finite.size else float("nan"),
            "p95": float(np.percentile(finite, 95)) if finite.size else float("nan"),
            "max": float(finite.max()) if finite.size else float("nan"),
            "n": int(finite.size),
        },
    }
    return mask, sim, meta


def mask_overlap(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    a_n = a & ~torch.eye(a.shape[0], dtype=torch.bool, device=a.device)
    b_n = b & ~torch.eye(b.shape[0], dtype=torch.bool, device=b.device)
    inter = int((a_n & b_n).sum().item())
    ua = int(a_n.sum().item())
    ub = int(b_n.sum().item())
    union = int((a_n | b_n).sum().item())
    return {
        "intersection_non_identity": inter,
        "only_a": ua - inter,
        "only_b": ub - inter,
        "union": union,
        "jaccard": float(inter / union) if union else float("nan"),
        "recall_a_in_b": float(inter / ua) if ua else float("nan"),
        "recall_b_in_a": float(inter / ub) if ub else float("nan"),
    }


# ---------------------------------------------------------------------------
# Data / batch helpers
# ---------------------------------------------------------------------------


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
    )


def _peak_cuda() -> Dict[str, Any]:
    if not torch.cuda.is_available():
        return {"device": "cpu", "peak_allocated_bytes": None, "peak_reserved_bytes": None}
    return {
        "device": torch.cuda.get_device_name(0),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / (1024**2),
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / (1024**2),
    }


def load_first_seed_batch(
    tr_data: HeteroData,
    *,
    seed: int,
    batch_size: int,
    num_neighbors: Sequence[int],
    device: torch.device,
) -> Dict[str, Any]:
    add_arange_ids([tr_data])
    torch.manual_seed(seed)
    np.random.seed(seed)
    loader = LinkNeighborLoader(
        tr_data,
        num_neighbors=list(num_neighbors),
        edge_label_index=(FORWARD, tr_data[FORWARD].edge_index),
        edge_label=tr_data[FORWARD].y,
        batch_size=batch_size,
        shuffle=True,
        transform=AddEgoIds(),
        num_workers=0,
    )
    batch = next(iter(loader))
    seed_ids = get_hetero_seed_edge_ids(batch, tr_data).to(device)
    assert int(seed_ids.numel()) <= int(batch_size), (
        f"seed count {seed_ids.numel()} exceeds batch_size {batch_size}"
    )

    # Apply standard contrastive views; keep seeds surviving both views
    attach_edge_id_from_batch(batch, tr_data)
    torch.manual_seed(seed + 17)
    view1, view2 = generate_views(
        batch,
        edge_attr_mask_rate=0.1,
        edge_drop_rate=0.1,
    )
    eid1 = view1[FORWARD].edge_id.detach().long().to(device)
    eid2 = view2[FORWARD].edge_id.detach().long().to(device)
    shared = seed_ids[torch.isin(seed_ids, eid1) & torch.isin(seed_ids, eid2)]
    shared = torch.unique(shared, sorted=True)

    # Endpoints / labels from the *loader graph* (train split), indexed by seed id
    fwd = tr_data[FORWARD]
    src_all = fwd.edge_index[0]
    dst_all = fwd.edge_index[1]
    y_all = fwd.y
    shared_cpu = shared.detach().cpu().long()
    src = src_all[shared_cpu].to(device)
    dst = dst_all[shared_cpu].to(device)
    labels = y_all[shared_cpu].to(device)

    n_mp = int(batch[FORWARD].edge_index.shape[1])
    return {
        "batch": batch,
        "view1": view1,
        "view2": view2,
        "seed_ids_requested": seed_ids,
        "shared_seed_ids": shared,
        "src": src,
        "dst": dst,
        "labels": labels,
        "n_message_passing_forward_edges_in_batch": n_mp,
        "n_seed_requested": int(seed_ids.numel()),
        "n_shared_surviving_both_views": int(shared.numel()),
    }


def current_impl_gap_doc() -> Dict[str, Any]:
    return {
        "representation": {
            "gcpal": "transactions as nodes",
            "ours": "transactions as edges; z = concat(h_sender, h_receiver, edge_attr) then embedding head",
        },
        "default_positives": {
            "gcpal_implied": "I ∪ A ∪ A_knn across random and KNN views",
            "ours": "identity across two random views only (same edge_id)",
        },
        "knn_soft_positives_are_not_gcpal": {
            "flag": "--enable_knn_soft_positives",
            "why_not_equivalent": [
                "Uses an offline sparse train-split feature-KNN cache, not a KNN message-passing view",
                "Adds low-weight soft positives into the identity InfoNCE numerator (default weight 0.025, m=1)",
                "Does not build random↔KNN contrast as a separate view pair",
                "Requires --contrastive_asymmetric; incompatible with endpoint multipos / morph_contrast",
                "Prior Small-HI ablations underperformed vs identity baseline (see notes/results-archive.md)",
            ],
        },
        "knn_filter_is_not_gcpal": {
            "flag": "--enable_knn_negative_filter",
            "why_not_equivalent": [
                "Only excludes cached neighbors from the negative pool",
                "Never adds structural or KNN positives",
            ],
        },
        "endpoint_multipos_is_not_gcpal": {
            "flag": "--multi_positive_mode same_endpoint|same_pair|...",
            "why_not_equivalent": [
                "Batch-local weak endpoint positives only; no A_knn tier",
                "No third KNN graph view",
                "Weak weight default 0.1; still identity-primary",
            ],
        },
        "mapping_to_existing_flags": {
            "identity": "default contrastive path",
            "directed_chain": "not implemented as a training flag",
            "shared_endpoint": "--multi_positive_mode same_endpoint",
            "same_ordered_pair": "--multi_positive_mode same_pair",
            "feature_knn": "offline cache via --enable_knn_soft_positives / --enable_knn_negative_filter (not batch-local)",
        },
        "closest_to_transaction_node_line_graph": (
            "shared-endpoint adjacency approximates undirected line-graph adjacency "
            "(transactions share an account). Directed-chain (receiver→sender) is the "
            "directed payment-flow line-graph edge and is closer to money-flow succession."
        ),
    }


def run_audit(args: argparse.Namespace) -> Dict[str, Any]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)-5s] %(message)s")
    device = torch.device(args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    out: Dict[str, Any] = {
        "audit": "gcpal_vs_edge_centric_positive_set_diagnostic",
        "read_only": True,
        "tds": False,
        "seed": int(args.seed),
        "batch_size_seed": int(args.batch_size),
        "num_neighbors": list(args.num_neighbors),
        "loader_num_workers": 0,
        "encoder_flags": {
            "reverse_mp": True,
            "ego": True,
            "ports": True,
            "emlps": True,
            "tds": False,
            "edge_dim_expected": 6,
        },
        "paper_ambiguity": paper_ambiguity_audit(),
        "memory_feasibility_static": memory_feasibility_report(),
        "current_implementation_gap": current_impl_gap_doc(),
    }

    # ---- load data ----
    t_load0 = time.perf_counter()
    with open(args.data_config, "r", encoding="utf-8") as f:
        data_config = json.load(f)
    data_args = _data_args()
    tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(data_args, data_config)
    out["data_load_seconds"] = time.perf_counter() - t_load0
    n_train = int(tr_data[FORWARD].edge_index.shape[1])
    # Full graph size from test split message-passing edges (includes all time)
    n_full = int(te_data[FORWARD].edge_index.shape[1])
    out["dataset_sizes"] = {
        "n_train_forward_edges": n_train,
        "n_full_forward_edges_te_mp": n_full,
    }
    out["memory_feasibility_static"] = memory_feasibility_report(n_full=n_full, n_train=n_train)

    # ---- one seed batch + views ----
    t_batch0 = time.perf_counter()
    batch_info = load_first_seed_batch(
        tr_data,
        seed=int(args.seed),
        batch_size=int(args.batch_size),
        num_neighbors=args.num_neighbors,
        device=device,
    )
    out["batch_prep_seconds"] = time.perf_counter() - t_batch0
    shared = batch_info["shared_seed_ids"]
    b = int(shared.numel())
    assert b <= int(args.batch_size)
    src, dst, labels = batch_info["src"], batch_info["dst"], batch_info["labels"]
    out["batch"] = {
        "n_seed_requested": batch_info["n_seed_requested"],
        "n_shared_surviving_both_views": b,
        "n_message_passing_forward_edges_in_batch": batch_info["n_message_passing_forward_edges_in_batch"],
        "seed_ids_first_32": shared[:32].detach().cpu().tolist(),
        "positive_label_count": int((labels == 1).sum().item()),
        "constraint": "All pairwise matrices use only shared surviving seeds, not MP edges",
    }

    # Optional note: encoder edge_dim without training
    if args.touch_encoder:
        out["encoder_touch"] = {"edge_dim": 6, "note": "tds=False → 4 raw + 2 ports"}

    # ---- feature prep for batch-local KNN (train-fit scaling) ----
    t_feat0 = time.perf_counter()
    _, df_train, _, _ = load_train_frame("Small-HI", args.data_config)
    feat_result = build_features_detailed(
        df_train,
        "edge_native",
        categorical_encoding="ordinal",
        scaling="none",
    )
    X_train = feat_result.features.astype(np.float32)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_std = scaler.transform(X_train).astype(np.float32)
    shared_cpu = shared.detach().cpu().numpy().astype(np.int64)
    X_batch = X_train_std[shared_cpu]
    # L2 normalize for cosine
    norms = np.linalg.norm(X_batch, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    X_batch_l2 = (X_batch / norms).astype(np.float32)
    x_t = torch.from_numpy(X_batch_l2).to(device)
    t_feat = time.perf_counter() - t_feat0
    out["knn_feature_protocol"] = {
        "feature_set": "edge_native",
        "columns_named": feat_result.names,
        "categorical_columns": ["Received Currency", "Payment Format"],
        "categorical_encoding": (
            "ordinal factorize (documented choice: category IDs are treated as "
            "numeric magnitudes after StandardScaler; not one-hot). Labels unused."
        ),
        "continuous": "Timestamp raw; Amount via log1p(Amount Received)",
        "scaling": "StandardScaler fit on full train split only; transform seed batch",
        "similarity": "cosine = L2-normalize rows then inner product",
        "k": 15,
        "self_excluded_before_topk": True,
        "labels_used_in_construction": False,
        "learned_representations_used_in_construction": False,
        "feature_prep_seconds": t_feat,
        "n_train_fit_rows": int(X_train.shape[0]),
        "feature_dim": int(X_train.shape[1]),
    }

    # ---- build positive definitions ----
    identity = build_identity_mask(b, device)
    directed = build_directed_chain_mask(src, dst)
    shared_ep = build_shared_endpoint_mask(src, dst)
    same_pair = build_same_pair_mask(src, dst)
    knn_mask, sim, knn_meta = build_batch_knn_mask(x_t, k=15)
    assert sim.shape == (b, b), f"FATAL: sim shape {tuple(sim.shape)} != [{b},{b}]"
    assert knn_mask.shape == (b, b)

    directed_u_knn = merge_positive_masks(identity, directed, knn_mask)
    shared_u_knn = merge_positive_masks(identity, shared_ep, knn_mask)

    # Random baseline purity: expected same-label rate under random pairs
    p_pos = float((labels == 1).float().mean().item()) if b else float("nan")
    random_same = p_pos * p_pos + (1 - p_pos) * (1 - p_pos)

    definitions = [
        ("identity_only", identity),
        ("directed_chain", merge_positive_masks(identity, directed)),
        ("shared_endpoint", merge_positive_masks(identity, shared_ep)),
        ("same_ordered_pair", merge_positive_masks(identity, same_pair)),
        ("feature_knn_k15", merge_positive_masks(identity, knn_mask)),
        ("directed_chain_union_knn", directed_u_knn),
        ("shared_endpoint_union_knn", shared_u_knn),
    ]

    summaries = {}
    masks_cpu = {}
    for name, mask in definitions:
        assert mask.shape == (b, b)
        summaries[name] = summarize_positive_mask(
            mask, labels, name=name, src=src, dst=dst
        )
        summaries[name]["random_pair_same_label_baseline"] = random_same
        masks_cpu[name] = mask.detach().cpu()

    # Overlaps among non-identity tiers
    tiers = {
        "identity": identity,
        "directed_chain": directed,
        "shared_endpoint": shared_ep,
        "same_ordered_pair": same_pair,
        "feature_knn_k15": knn_mask,
    }
    overlaps = {}
    keys = list(tiers.keys())
    for i, ka in enumerate(keys):
        for kb in keys[i + 1 :]:
            overlaps[f"{ka}__vs__{kb}"] = mask_overlap(tiers[ka], tiers[kb])

    # Global sparse cache overlap (batch-local KNN vs cache neighbors restricted to batch)
    cache_path = Path(args.knn_cache)
    cache_overlap = {"available": False}
    if cache_path.is_file():
        z = np.load(cache_path, allow_pickle=True)
        cache_eids = z["edge_ids"].astype(np.int64)
        neigh = z["neighbor_ids"].astype(np.int64)
        # Map seed id → row in cache (assume edge_ids == 0..N-1 identity)
        id_to_row = {int(e): i for i, e in enumerate(cache_eids)}
        seed_list = shared_cpu.tolist()
        seed_set = set(seed_list)
        pos_in_batch = 0
        cache_edges = 0
        for sid in seed_list:
            row = id_to_row.get(int(sid))
            if row is None:
                continue
            for nb in neigh[row]:
                if int(nb) < 0:
                    continue
                cache_edges += 1
                if int(nb) in seed_set and int(nb) != int(sid):
                    pos_in_batch += 1
        # Also: of batch-local KNN pairs, how many appear in cache neighbor lists
        cache_sets = {}
        for sid in seed_list:
            row = id_to_row.get(int(sid))
            if row is None:
                continue
            cache_sets[int(sid)] = {int(x) for x in neigh[row] if int(x) >= 0}
        knn_ii, knn_jj = knn_mask.detach().cpu().nonzero(as_tuple=True)
        hit = 0
        for a, c in zip(knn_ii.tolist(), knn_jj.tolist()):
            sid = int(seed_list[a])
            tid = int(seed_list[c])
            if tid in cache_sets.get(sid, ()):
                hit += 1
        n_knn = int(knn_mask.sum().item())
        cache_overlap = {
            "available": True,
            "cache_path": str(cache_path),
            "cache_k": int(z["k"]) if "k" in z else int(neigh.shape[1]),
            "batch_local_knn_pairs": n_knn,
            "batch_local_knn_pairs_also_in_global_cache_neighbor_list": hit,
            "fraction_batch_knn_supported_by_global_cache": float(hit / n_knn) if n_knn else float("nan"),
            "global_cache_neighbors_of_seeds_that_also_fall_in_seed_batch": pos_in_batch,
            "global_cache_neighbor_slots_examined": cache_edges,
            "note": (
                "Global cache is train-split feature-KNN (edge_native+degree_fan); "
                "batch-local KNN here uses edge_native only — overlap is diagnostic, "
                "not exact feature parity."
            ),
        }

    # Mistaken MP-edge dense cost
    n_mp = batch_info["n_message_passing_forward_edges_in_batch"]
    mistaken = {
        "n_message_passing_forward_edges": n_mp,
        "similarity_float32_bytes": matrix_memory_bytes(n_mp),
        "similarity_float32_gib": matrix_memory_bytes(n_mp) / (1024**3),
        "vs_seed_B_float32_gib": matrix_memory_bytes(b) / (1024**3),
        "ratio_mp_over_seed": float(matrix_memory_bytes(n_mp) / max(matrix_memory_bytes(b), 1)),
    }

    out["positive_definitions"] = summaries
    out["overlaps_non_identity"] = overlaps
    out["knn_batch_local"] = knn_meta
    out["knn_vs_global_cache"] = cache_overlap
    out["mistaken_dense_over_message_passing_edges"] = mistaken
    out["cuda_memory"] = _peak_cuda()
    out["runtimes_seconds"] = {
        "data_load": out["data_load_seconds"],
        "batch_and_views": out["batch_prep_seconds"],
        "feature_prep": t_feat,
        "similarity": knn_meta["runtime_seconds"]["similarity_matrix"],
        "topk": knn_meta["runtime_seconds"]["topk"],
    }

    # Answers stub filled after metrics
    knn_sum = summaries["feature_knn_k15"]
    dir_sum = summaries["directed_chain"]
    sh_sum = summaries["shared_endpoint"]
    out["diagnostic_answers"] = {
        "seed_only_B2048_knn_comfortably_feasible": bool(
            b <= 2048 and matrix_memory_bytes(b) < 64 * (1024**2)
        ),
        "knn_provides_positives_beyond_identity": bool(
            knn_sum["total_non_identity_positive_pairs"] > 0
        ),
        "knn_more_label_consistent_than_random": bool(
            knn_sum["label_agreement_purity_non_identity"] > random_same
            if knn_sum["total_non_identity_positive_pairs"] > 0
            else False
        ),
        "defensible_adjacency_mapping": "directed_chain",
        "notes_for_adjacency": {
            "directed_chain": (
                "Default recommendation for GCPAL transaction-node adjacency mapped into "
                "edge-centric AMLWorld: directed money-flow succession (receiver→sender). "
                "Less hub-saturated than any-shared-account in typical payment graphs."
            ),
            "shared_endpoint": (
                "Undirected line-graph / share-any-account; matches existing "
                "--multi_positive_mode same_endpoint. Often hub-dominated."
            ),
            "measured_hub_fractions": {
                "directed_chain": dir_sum.get("hub_concentration", {}),
                "shared_endpoint": sh_sum.get("hub_concentration", {}),
            },
        },
    }

    return out


def write_markdown(report: Dict[str, Any], path: Path) -> None:
    mem = report["memory_feasibility_static"]
    ans = report["diagnostic_answers"]
    defs = report["positive_definitions"]
    lines = [
        "# GCPAL vs edge-centric positive-set audit",
        "",
        "Machine-readable twin: `results/diagnostics/gcpal_positive_set_audit.json`",
        "",
        "Read-only diagnostic. `tds=False`. Dense matrices only over surviving seed transactions "
        f"(B={report['batch']['n_shared_surviving_both_views']}). No training run.",
        "",
        "## 1. What GCPAL explicitly specifies",
        "",
        json.dumps(report["paper_ambiguity"]["explicit"], indent=2),
        "",
        "## 2. What is implied but omitted",
        "",
        json.dumps(report["paper_ambiguity"]["implied_but_omitted"], indent=2),
        "",
        "## 3. What cannot be reproduced without code",
        "",
        json.dumps(report["paper_ambiguity"]["unreproducible_without_code"], indent=2),
        "",
        "## 4. How our current implementation differs",
        "",
        json.dumps(report["current_implementation_gap"], indent=2),
        "",
        "## 5. Memory: dense pairwise storage only",
        "",
        "| Scope | N | float32 sim GiB | feasible? |",
        "|-------|--:|----------------:|:---------:|",
    ]
    for key in ("full_small_hi", "train_split_small_hi", "batch_8192", "batch_2048"):
        r = mem[key]
        lines.append(
            f"| {r['label']} | {r['n']} | {r['similarity_matrix_float32_gib']:.4g} | {r['feasible_on_single_gpu_80gib']} |"
        )
    lines += [
        "",
        f"Mistaken MP-edge dense cost this batch: "
        f"{report['mistaken_dense_over_message_passing_edges']['similarity_float32_gib']:.4g} GiB "
        f"(n={report['mistaken_dense_over_message_passing_edges']['n_message_passing_forward_edges']}), "
        f"ratio vs seed: {report['mistaken_dense_over_message_passing_edges']['ratio_mp_over_seed']:.1f}×",
        "",
        "## 6. Batch-local KNN feasibility (B≈2048)",
        "",
        f"- Comfortably feasible: **{ans['seed_only_B2048_knn_comfortably_feasible']}**",
        f"- Peak CUDA allocated MiB: {report['cuda_memory'].get('peak_allocated_mib')}",
        f"- Feature protocol: {json.dumps(report['knn_feature_protocol'], indent=2)}",
        "",
        "## 7. Positive-set measurements (non-identity stats)",
        "",
        "| Definition | anchors | frac no non-id pos | median | mean | p95 | max | pairs | purity | dens |",
        "|------------|--------:|-------------------:|-------:|-----:|----:|----:|------:|-------:|-----:|",
    ]
    for name, s in defs.items():
        pp = s["positives_per_anchor_non_identity"]
        lines.append(
            f"| {name} | {s['anchors']} | {s['fraction_anchors_with_no_non_identity_positives']:.3f} | "
            f"{pp['median']:.2f} | {pp['mean']:.2f} | {pp['p95']:.2f} | {pp['max']} | "
            f"{s['total_non_identity_positive_pairs']} | "
            f"{s['label_agreement_purity_non_identity']:.4f} | "
            f"{s['positive_mask_density_non_identity']:.4g} |"
        )
    lines += [
        "",
        "## 8. Hub / near-duplicate risk",
        "",
    ]
    for name in ("shared_endpoint", "directed_chain", "feature_knn_k15"):
        h = defs[name].get("hub_concentration", {})
        lines.append(f"- **{name}**: {json.dumps(h)}")
    lines += [
        "",
        f"KNN cosine sims: {json.dumps(report['knn_batch_local']['cosine_similarity_of_selected_neighbors'])}",
        "",
        f"Global cache overlap: {json.dumps(report['knn_vs_global_cache'], indent=2)}",
        "",
        "## 9. Diagnostic answers",
        "",
        f"1. Seed-only B=2048 KNN feasible? **{ans['seed_only_B2048_knn_comfortably_feasible']}**",
        f"2. KNN beyond identity? **{ans['knn_provides_positives_beyond_identity']}**",
        f"3. KNN more label-consistent than random? **{ans['knn_more_label_consistent_than_random']}**",
        f"4. More defensible adjacency mapping? **{ans['defensible_adjacency_mapping']}** "
        f"({ans['notes_for_adjacency']})",
        "",
        "## 10. Recommendation",
        "",
        report.get("recommendation", {}).get("choice", "PENDING_JOB"),
        "",
        report.get("recommendation", {}).get("rationale", ""),
        "",
        "## 11. Smallest next training experiment (not launched)",
        "",
        report.get("recommendation", {}).get("next_experiment", ""),
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def choose_recommendation(report: Dict[str, Any]) -> Dict[str, Any]:
    defs = report["positive_definitions"]
    ans = report["diagnostic_answers"]
    knn = defs["feature_knn_k15"]
    shared = defs["shared_endpoint"]
    directed = defs["directed_chain"]
    knn_purity = knn["label_agreement_purity_non_identity"]
    random_p = knn.get("random_pair_same_label_baseline", float("nan"))
    knn_hub = knn.get("hub_concentration", {}).get(
        "fraction_nonid_positives_touching_top_1pct_accounts", float("nan")
    )
    knn_sims = report["knn_batch_local"]["cosine_similarity_of_selected_neighbors"]
    mean_sim = knn_sims.get("mean", float("nan"))

    # Decision logic from measurements
    if not ans["seed_only_B2048_knn_comfortably_feasible"]:
        choice = "A"
        rationale = "Seed-batch KNN itself is not comfortably feasible; do not pursue GCPAL mapping."
        next_exp = "None."
    elif mean_sim == mean_sim and mean_sim > 0.99 and knn_purity <= random_p + 0.02:
        choice = "A"
        rationale = (
            "Batch-local KNN neighbors are near-duplicates (mean cosine > 0.99) and not meaningfully "
            "more label-consistent than random pairs — matching prior offline softpos failure modes."
        )
        next_exp = (
            "Do not launch a GCPAL-style contrastive scout. Prefer non-GCPAL objective work already in-tree."
        )
    elif (
        shared["total_non_identity_positive_pairs"] > 0
        and shared["fraction_anchors_with_no_non_identity_positives"] < 0.5
        and shared.get("hub_concentration", {}).get(
            "fraction_nonid_positives_touching_top_1pct_accounts", 1.0
        )
        > 0.5
        and directed["fraction_anchors_with_no_non_identity_positives"]
        < shared["fraction_anchors_with_no_non_identity_positives"] + 0.2
    ):
        # Shared-endpoint heavily hub-dominated → prefer directed-chain ablation
        choice = "D"
        rationale = (
            "Shared-endpoint positives are plentiful but hub-dominated; directed-chain is the more "
            "defensible money-flow line-graph mapping and should be ablated before any KNN view work."
        )
        next_exp = (
            "1–3 epoch contrastive scout with a directed-chain multi-positive definition only "
            "(identity + directed-chain), tds=False, B_seed≤2048 if memory-sensitive — not launched here."
        )
    elif ans["knn_more_label_consistent_than_random"] and knn_hub == knn_hub and knn_hub < 0.35:
        choice = "B"
        rationale = (
            "Batch-local KNN is feasible, adds non-identity positives, and shows better-than-random "
            "label agreement without extreme hub concentration — warranting a tiny memory/stability scout."
        )
        next_exp = (
            "1–3 epoch scout: identity ∪ batch-local k=15 feature-KNN soft/hard positives, "
            "tds=False, seed batch 2048 — not launched here."
        )
    else:
        choice = "D"
        rationale = (
            "Measurements support a controlled positive-set ablation (identity vs directed-chain vs "
            "shared-endpoint vs batch-local KNN) before implementing a third KNN message-passing view."
        )
        next_exp = (
            "Single smallest scout: identity + one structural definition "
            f"({ans['defensible_adjacency_mapping']}), tds=False — not launched here."
        )

    # Map letter names
    names = {
        "A": "do not pursue GCPAL mapping",
        "B": "run a 1–3 epoch memory/stability scout",
        "C": "implement the third KNN message-passing view first",
        "D": "run a controlled positive-set ablation first",
    }
    # Which ONE positive definition for a tiny scout
    if choice == "A":
        one_def = "none"
    elif choice == "B":
        one_def = "feature_knn_k15_union_identity"
    else:
        one_def = f"{ans['defensible_adjacency_mapping']}_union_identity"

    return {
        "choice": choice,
        "choice_name": names[choice],
        "rationale": rationale,
        "next_experiment": next_exp,
        "one_positive_definition_for_tiny_scout": one_def,
        "answers": {
            "1_seed_B2048_knn_feasible": ans["seed_only_B2048_knn_comfortably_feasible"],
            "2_knn_beyond_identity": ans["knn_provides_positives_beyond_identity"],
            "3_knn_more_label_consistent_than_random": ans["knn_more_label_consistent_than_random"],
            "4_defensible_adjacency": ans["defensible_adjacency_mapping"],
            "5_one_positive_definition": one_def,
            "6_mistaken_mp_dense_gib": report["mistaken_dense_over_message_passing_edges"][
                "similarity_float32_gib"
            ],
        },
    }


def parse_args() -> argparse.Namespace:
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
        default="results/diagnostics/gcpal_positive_set_audit.json",
    )
    p.add_argument(
        "--output_md",
        default="notes/gcpal_positive_set_audit.md",
    )
    p.add_argument("--touch_encoder", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.batch_size) > 2048:
        raise SystemExit("batch_size must be <= 2048 for this diagnostic")
    report = run_audit(args)
    report["recommendation"] = choose_recommendation(report)
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    write_markdown(report, out_md)
    print(json.dumps({"wrote": str(out_json), "recommendation": report["recommendation"]}, indent=2))


if __name__ == "__main__":
    main()
