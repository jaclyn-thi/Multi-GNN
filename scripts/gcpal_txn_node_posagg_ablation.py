#!/usr/bin/env python3
"""B_gcpal positive-aggregation ablation (C/D): 5ep train + expanding-window eval.

Does not overwrite historical A/B scout or replay artifacts.
Default sum_logsumexp behavior elsewhere is unchanged when this flag is omitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch

from gcpal_txn_node.adjacency import (
    adjacency_list_from_edge_index,
    build_directed_flow_adjacency,
)
from gcpal_txn_node.checkpointing import save_training_checkpoint
from gcpal_txn_node.data import load_small_hi_frame
from gcpal_txn_node.eval_mlp import train_eval_mlp_suite
from gcpal_txn_node.extraction import (
    TEMPORAL_EXPANDING_WINDOW_V1,
    extract_temporal_expanding_window,
    sha256_file,
    sha256_json,
    temporal_expanding_window_config,
)
from gcpal_txn_node.features import fit_feature_preprocessor
from gcpal_txn_node.knn_adapter import load_train_knn_cache
from gcpal_txn_node.loss import (
    DEFAULT_POSITIVE_AGGREGATION,
    POSITIVE_AGGREGATION_MODES,
    validate_positive_aggregation,
)
from gcpal_txn_node.model import SharedTxnNodeEncoder
from gcpal_txn_node.spec import DEFAULT_KNN_CACHE, LAMBDA_MIX, NOT_EXACT_REPRODUCTION, TEMPERATURE
from gcpal_txn_node.train_step import StepConfig, run_positive_complete_step

PROTECTED = (
    "poscomplete_scout_A_identity_5ep",
    "poscomplete_scout_B_gcpal_5ep",
    "poscomplete_scout_A_identity_20ep",
    "poscomplete_scout_B_gcpal_20ep",
)


def _refuse_protected(path: Path) -> None:
    s = str(path)
    for p in PROTECTED:
        if p in s:
            raise SystemExit(f"Refusing protected historical path: {path}")


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return None
    if isinstance(obj, torch.Tensor):
        return None
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if not math.isfinite(f) else f
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _grad_global_norm(params) -> float:
    total = 0.0
    for p in params:
        if p.grad is None:
            continue
        total += float(p.grad.detach().pow(2).sum().cpu())
    return float(math.sqrt(total))


def _bin_key(n_pos: float) -> str:
    p = int(round(n_pos))
    if p <= 1:
        return "1"
    if p <= 4:
        return "2-4"
    if p <= 8:
        return "5-8"
    if p <= 16:
        return "9-16"
    return "17+"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--positive_aggregation",
        choices=list(POSITIVE_AGGREGATION_MODES),
        required=True,
        help="Aggregation mode for B_gcpal ablation (C/D).",
    )
    p.add_argument("--data_config", default="data_config.json")
    p.add_argument("--knn_cache", default=DEFAULT_KNN_CACHE)
    p.add_argument("--max_total_nodes", type=int, default=2048)
    p.add_argument("--n_epochs", type=int, default=5)
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--output_json", required=True)
    p.add_argument("--output_md", required=True)
    p.add_argument("--tag", default=None, help="Short run tag, e.g. C_logmeanexp")
    args = p.parse_args()
    assert NOT_EXACT_REPRODUCTION
    if int(args.n_epochs) != 5:
        raise SystemExit("Ablation trains exactly 5 epochs")
    agg = validate_positive_aggregation(args.positive_aggregation)
    if agg == DEFAULT_POSITIVE_AGGREGATION:
        raise SystemExit(
            "Refusing to retrain default sum_logsumexp B here; use historical B reference."
        )

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    ckpt_dir = Path(args.ckpt_dir)
    _refuse_protected(out_json)
    _refuse_protected(out_md)
    _refuse_protected(ckpt_dir)
    if out_json.exists():
        raise SystemExit(f"Refusing overwrite {out_json}")
    if out_md.exists():
        raise SystemExit(f"Refusing overwrite {out_md}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))

    df, tr, va, te, meta = load_small_hi_frame(args.data_config)
    df_train = df.iloc[tr].reset_index(drop=True).copy()
    n = len(df_train)
    allowed = set(range(n))
    labels = df_train[meta["label_col"]].to_numpy().astype(np.int64)
    prep = fit_feature_preprocessor(df_train, amount_col=meta["amount_col"])
    x_train = prep.transform(df_train)

    flow_ei, flow_stats = build_directed_flow_adjacency(
        df_train["from_id"].to_numpy(),
        df_train["to_id"].to_numpy(),
        df_train["Timestamp"].astype(float).to_numpy(),
        policy="immediate_next",
    )
    flow_adj = adjacency_list_from_edge_index(flow_ei, n)
    knn = load_train_knn_cache(args.knn_cache, expected_k=15)
    if knn.node_ids.shape[0] != n:
        raise SystemExit("KNN/train size mismatch")
    knn_adj = knn.adjacency_lists()

    cfg = StepConfig(
        edge_drop=0.1,
        feature_drop=0.1,
        lambda_mix=LAMBDA_MIX,
        temperature=TEMPERATURE,
        include_identity=True,
        use_knn_contrast=True,
        max_total_nodes=int(args.max_total_nodes),
        batching_mode="positive_complete",
        knn_k=15,
        positive_aggregation=agg,
    )
    pos_flow, pos_knn = flow_adj, knn_adj

    resolved_config = {
        "mode": "B_gcpal",
        "positive_aggregation": agg,
        "seed": int(args.seed),
        "max_total_nodes": int(args.max_total_nodes),
        "n_epochs": 5,
        "lambda_mix": float(cfg.lambda_mix),
        "temperature": float(cfg.temperature),
        "knn_k": 15,
        "edge_drop": 0.1,
        "feature_drop": 0.1,
        "optimizer": "Adam",
        "lr": 1e-3,
        "batching_mode": "positive_complete",
        "use_knn_contrast": True,
        "include_identity": True,
        "steps_per_epoch_formula": "ceil(n_train/2048)",
        "not_exact_reproduction": True,
        "tag": args.tag,
    }
    config_hash = sha256_json(resolved_config)

    encoder = SharedTxnNodeEncoder(in_dim=x_train.shape[1], emb_dim=128).to(device)
    opt = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    rng = np.random.RandomState(args.seed)
    steps_per_epoch = int(np.ceil(n / 2048.0))
    assert steps_per_epoch == 1587, f"expected 1587 steps/epoch, got {steps_per_epoch}"

    history: List[dict] = []
    opt_diags: List[dict] = []
    batch_diags: List[dict] = []
    total_opt_steps = 0
    total_anchor_exposures = 0
    t0 = time.perf_counter()
    ckpt_hashes: Dict[str, str] = {}

    # Accumulators across all steps for |P| distribution / loss-by-bin
    all_n_pos: List[float] = []
    loss_by_bin_sum: Dict[str, float] = defaultdict(float)
    loss_by_bin_n: Dict[str, int] = defaultdict(int)
    sim_sums: Dict[str, float] = defaultdict(float)
    sim_ns: Dict[str, int] = defaultdict(int)

    for epoch in range(1, 6):
        perm = rng.permutation(n).astype(np.int64)
        cursor = 0
        losses: List[float] = []
        grad_norms: List[float] = []
        emb_snap: List[dict] = []
        log_num_vals: List[float] = []
        log_den_vals: List[float] = []
        for step_i in range(steps_per_epoch):
            if cursor >= n:
                perm = rng.permutation(n).astype(np.int64)
                cursor = 0
            out = run_positive_complete_step(
                encoder=encoder,
                x_all=x_train,
                flow_edge_index=flow_ei,
                flow_adj=flow_adj,
                knn_adj=knn_adj,
                knn_graph_edge_fn=knn.edge_index_for_nodes,
                candidate_stream=perm,
                stream_start=cursor,
                device=device,
                cfg=cfg,
                seed=args.seed * 10007 + (epoch - 1) * 97 + step_i,
                pos_flow_adj=pos_flow,
                pos_knn_adj=pos_knn,
                allowed_ids=allowed,
                labels=labels,
            )
            loss = out["loss"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norms.append(_grad_global_norm(encoder.parameters()))
            opt.step()
            total_opt_steps += 1
            total_anchor_exposures += int(out["n_anchors"])
            losses.append(float(loss.detach().cpu()))

            n_pos = out["n_pos_per_anchor"].numpy()
            loss_vec = out["loss_vec_per_anchor"].numpy()
            all_n_pos.extend(n_pos.tolist())
            for np_, lv in zip(n_pos.tolist(), loss_vec.tolist()):
                bk = _bin_key(np_)
                loss_by_bin_sum[bk] += float(lv)
                loss_by_bin_n[bk] += 1
            log_num_vals.extend(out["log_num_per_anchor"].numpy().tolist())
            log_den_vals.extend(out["log_denom_per_anchor"].numpy().tolist())
            for k, v in out["similarity_diagnostics"].items():
                if v is None or (isinstance(v, float) and not math.isfinite(v)):
                    continue
                sim_sums[k] += float(v)
                sim_ns[k] += 1
            if step_i % 200 == 0:
                emb_snap.append(out["embedding_diagnostics"])
            if len(batch_diags) < 8:
                batch_diags.append(
                    {
                        "epoch": epoch,
                        "realized_n_anchors": int(out["n_anchors"]),
                        "n_nodes": int(out["n_sampled_nodes"]),
                        "growth_stats": out["growth_stats"],
                        "positive_stats": out["positive_stats"],
                        "similarity_diagnostics": out["similarity_diagnostics"],
                        "embedding_diagnostics": out["embedding_diagnostics"],
                    }
                )
            cursor = int(out["stream_next"])

        ep_hist = {
            "epoch": epoch,
            "loss_mean": float(np.mean(losses)),
            "loss_std": float(np.std(losses)),
            "n_steps": len(losses),
            "grad_norm_mean": float(np.mean(grad_norms)),
            "grad_norm_p50": float(np.median(grad_norms)),
            "grad_norm_p95": float(np.percentile(grad_norms, 95)),
            "log_num_mean": float(np.mean(log_num_vals)),
            "log_denom_mean": float(np.mean(log_den_vals)),
            "embedding_snapshot_tail": emb_snap[-1] if emb_snap else None,
        }
        history.append(ep_hist)
        opt_diags.append(ep_hist)
        logging.info(
            "epoch %s loss=%.4f grad_mean=%.4f steps=%d",
            epoch,
            ep_hist["loss_mean"],
            ep_hist["grad_norm_mean"],
            ep_hist["n_steps"],
        )

        ckpt_path = ckpt_dir / f"epoch_{epoch:02d}.pt"
        save_training_checkpoint(
            ckpt_path,
            epoch=epoch,
            encoder=encoder,
            optimizer=opt,
            rng=rng,
            history=history,
            total_opt_steps=total_opt_steps,
            total_anchor_exposures=total_anchor_exposures,
            meta={
                **resolved_config,
                "config_hash": config_hash,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            },
        )
        ckpt_hashes[str(ckpt_path)] = sha256_file(ckpt_path)

    train_s = time.perf_counter() - t0
    n_pos_arr = np.asarray(all_n_pos, dtype=np.float64)
    pos_dist = {
        "mean": float(n_pos_arr.mean()),
        "std": float(n_pos_arr.std()),
        "p50": float(np.median(n_pos_arr)),
        "p10": float(np.percentile(n_pos_arr, 10)),
        "p90": float(np.percentile(n_pos_arr, 90)),
        "min": float(n_pos_arr.min()),
        "max": float(n_pos_arr.max()),
        "histogram_bins": {
            k: int(loss_by_bin_n[k]) for k in ("1", "2-4", "5-8", "9-16", "17+")
        },
    }
    loss_by_bin = {
        k: (loss_by_bin_sum[k] / max(loss_by_bin_n[k], 1)) for k in loss_by_bin_n
    }
    sim_means = {k: sim_sums[k] / max(sim_ns[k], 1) for k in sim_sums}

    # Collapse verdict from last-epoch embedding snapshot + similarity gap
    last_emb = history[-1].get("embedding_snapshot_tail") or {}
    collapse_verdict = "ok"
    if float(last_emb.get("anchor_mean_pairwise_cosine") or 0) > 0.9:
        collapse_verdict = "suspect_collapse_high_anchor_cosine"
    if float(last_emb.get("h_var_mean") or 1.0) < 1e-4:
        collapse_verdict = "suspect_collapse_low_h_variance"

    # --- Canonical expanding-window eval at fixed epoch 5 ---
    ep5 = ckpt_dir / "epoch_05.pt"
    try:
        blob = torch.load(str(ep5), map_location=str(device), weights_only=False)
    except TypeError:
        blob = torch.load(str(ep5), map_location=str(device))
    encoder.load_state_dict(blob["model_state_dict"])
    encoder.eval()
    for p_ in encoder.parameters():
        p_.requires_grad_(False)

    x_full = np.zeros((len(df), x_train.shape[1]), dtype=np.float32)
    x_full[tr] = x_train
    x_full[va] = prep.transform(df.iloc[va])
    x_full[te] = prep.transform(df.iloc[te])
    y_all = df[meta["label_col"]].to_numpy().astype(np.int64)
    flow_full, _ = build_directed_flow_adjacency(
        df["from_id"].to_numpy(),
        df["to_id"].to_numpy(),
        df["Timestamp"].astype(float).to_numpy(),
        policy="immediate_next",
    )
    ext = extract_temporal_expanding_window(
        encoder=encoder,
        x_all=x_full,
        flow_ei=flow_full,
        tr=tr,
        va=va,
        te=te,
        device=device,
        config=temporal_expanding_window_config(seed=int(args.seed)),
        checkpoint_path=ep5,
        verify_expected_edges=True,
    )
    # Cache H
    emb_root = Path("embeddings/gcpal_txn_node_posagg") / f"B_{agg}" / "ep05"
    emb_root.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        np.save(emb_root / f"h_{split}.npy", ext["embeddings"][split])
        np.save(emb_root / f"ids_{split}.npy", ext["split_node_ids"][split])

    temporal = train_eval_mlp_suite(
        ext["embeddings"]["train"],
        x_full[tr],
        y_all[tr],
        ext["embeddings"]["test"],
        x_full[te],
        y_all[te],
        h_val=ext["embeddings"]["val"],
        x_val=x_full[va],
        y_val=y_all[va],
        seed=int(args.seed),
        device=device,
    )

    payload = {
        "not_exact_reproduction": True,
        "tag": args.tag,
        "mode": "B_gcpal",
        "positive_aggregation": agg,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "resolved_config": resolved_config,
        "config_hash": config_hash,
        "ckpt_dir": str(ckpt_dir),
        "checkpoint_sha256": ckpt_hashes,
        "seed": int(args.seed),
        "n_epochs": 5,
        "steps_per_epoch": steps_per_epoch,
        "total_opt_steps": total_opt_steps,
        "total_anchor_exposures": total_anchor_exposures,
        "train_seconds": train_s,
        "history": history,
        "optimization_diagnostics": {
            "n_pos_distribution": pos_dist,
            "loss_by_positive_count_bin": loss_by_bin,
            "mean_similarity_by_type": sim_means,
            "per_epoch": opt_diags,
            "collapse_verdict": collapse_verdict,
            "note": "Do not compare raw loss magnitude across aggregation modes as training quality.",
        },
        "batch_diagnostics_head": batch_diags,
        "extraction_mode": TEMPORAL_EXPANDING_WINDOW_V1,
        "temporal_primary_ep5": temporal,
        "flow_stats": {
            "n_nodes": flow_stats.n_nodes,
            "n_edges": flow_stats.n_edges,
            "policy": flow_stats.policy,
        },
        "gnn_training_occurred": True,
        "historical_artifacts_unchanged": True,
        "defaults_unchanged": True,
        "eval_notes": [
            "Primary eval: frozen_checkpoint_temporal_expanding_window_v1 at fixed epoch 5.",
            "No legacy chunking / online h_anchors / per-split sensitivity for primary comparison.",
        ],
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n")

    hx = temporal["HxX"]
    md = [
        f"# Positive-aggregation ablation (`{args.tag or agg}`)",
        "",
        "**B_gcpal only** · 5 epochs · seed 2 · expanding-window eval at ep5",
        "",
        f"- positive_aggregation: `{agg}`",
        f"- job: `{os.environ.get('SLURM_JOB_ID')}`",
        f"- config_hash: `{config_hash}`",
        f"- mean |P|: {pos_dist['mean']:.4f}",
        f"- collapse_verdict: {collapse_verdict}",
        f"- train_seconds: {train_s:.1f}",
        "",
        "## Temporal HxX (ep5 expanding-window)",
        "",
        f"- val AUPRC: {hx['val_ranking'].get('auprc')}",
        f"- test @0.5 AUPRC/AUROC/F1: {hx['threshold_0.5'].get('auprc')} / "
        f"{hx['threshold_0.5'].get('auroc')} / {hx['threshold_0.5'].get('f1')}",
        f"- test @val-thr F1: {hx['threshold_val_selected'].get('f1')} "
        f"(thr={hx['threshold_val_selected'].get('validation_selected_threshold')})",
        "",
        "Raw loss magnitudes are **not** comparable across aggregation modes.",
        "",
    ]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md) + "\n")
    logging.info("Wrote %s and %s", out_json, out_md)


if __name__ == "__main__":
    main()
