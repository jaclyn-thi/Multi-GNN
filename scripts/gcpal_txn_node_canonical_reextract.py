#!/usr/bin/env python3
"""Canonical frozen checkpoint re-extraction + MLP eval (no GNN retraining).

Phases
------
smoke :
    B_gcpal epoch-5 feasibility gate on Advanced GPU. Writes smoke JSON and a
    pass/fail gate file. Does not overwrite historical scout artifacts.
arm :
    Re-extract + evaluate one arm (A_identity | B_gcpal) for epochs 5/10/15/20
    under expanding-window, per-split isolation, and joint-full-graph random-40.
aggregate :
    Build notes/gcpal_txn_node_canonical_reextraction.md + companion JSON from
    smoke + arm outputs; recompute checkpoint selection from expanding-window
    temporal validation HxX AUPRC.

Not an exact GCPAL reproduction. Extraction only — never trains the GNN.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import resource
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit

from gcpal_txn_node.adjacency import build_directed_flow_adjacency
from gcpal_txn_node.data import load_small_hi_frame
from gcpal_txn_node.eval_mlp import train_eval_mlp_suite
from gcpal_txn_node.extraction import (
    EXPECTED_EDGE_COUNTS,
    EXPECTED_N_NODES_FULL,
    JOINT_FULL_GRAPH_RANDOM40_V1,
    LEGACY_CHUNKED_EXTRACTION_MODE,
    PER_TEMPORAL_SPLIT_V1,
    TEMPORAL_EXPANDING_WINDOW_V1,
    extract_joint_full_graph,
    extract_split_embeddings,
    extract_temporal_expanding_window,
    joint_full_graph_random40_config,
    load_encoder_from_checkpoint,
    per_temporal_split_config,
    sha256_file,
    sha256_json,
    temporal_expanding_window_config,
)
from gcpal_txn_node.features import fit_feature_preprocessor
from gcpal_txn_node.spec import NOT_EXACT_REPRODUCTION

EVAL_EPOCHS = (5, 10, 15, 20)
VAL_SELECT_REP = "HxX"
VAL_SELECT_KEY = "auprc"
SIX_HOUR_SECONDS = 6 * 3600
# Leave headroom for queue/startup and MLP eval variance.
SUITE_TIME_BUDGET = 0.85 * SIX_HOUR_SECONDS

PROTECTED_PATH_SUBSTRINGS = (
    "poscomplete_scout_A_identity_5ep",
    "poscomplete_scout_B_gcpal_5ep",
    "poscomplete_scout_A_identity_20ep",
    "poscomplete_scout_B_gcpal_20ep",
)

DEFAULT_CKPT = {
    "A_identity": "checkpoints/gcpal_txn_node_poscomplete_A_identity_20ep_seed2",
    "B_gcpal": "checkpoints/gcpal_txn_node_poscomplete_B_gcpal_20ep_seed2",
}

SMOKE_GATE = Path("results/diagnostics/gcpal_txn_node_canonical_reextract_smoke_gate.json")
SMOKE_JSON = Path("results/diagnostics/gcpal_txn_node_canonical_reextract_smoke_B_ep05.json")
EMB_ROOT = Path("embeddings/gcpal_txn_node_canonical")


def _refuse_protected(path: Path) -> None:
    name = str(path)
    for s in PROTECTED_PATH_SUBSTRINGS:
        if s in name:
            raise SystemExit(f"Refusing to write protected historical path: {path}")


def _cpu_rss_mib() -> Optional[float]:
    try:
        # Linux: ru_maxrss is kilobytes
        return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
    except Exception:
        return None


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _load_data(data_config: str, seed: int):
    df, tr, va, te, meta = load_small_hi_frame(data_config)
    y_all = df[meta["label_col"]].to_numpy().astype(np.int64)
    df_train = df.iloc[tr].reset_index(drop=True).copy()
    prep = fit_feature_preprocessor(df_train, amount_col=meta["amount_col"])
    x_full = prep.transform(df)
    flow_full, flow_stats = build_directed_flow_adjacency(
        df["from_id"].to_numpy(),
        df["to_id"].to_numpy(),
        df["Timestamp"].astype(float).to_numpy(),
        policy="immediate_next",
    )
    return {
        "df": df,
        "tr": tr,
        "va": va,
        "te": te,
        "y_all": y_all,
        "x_full": x_full,
        "flow_full": flow_full,
        "flow_stats": flow_stats,
        "meta": meta,
        "in_dim": int(x_full.shape[1]),
        "n_nodes": int(x_full.shape[0]),
        "seed": int(seed),
    }


def _cache_h(
    *,
    mode: str,
    epoch: int,
    extraction_mode: str,
    split: str,
    h: np.ndarray,
    node_ids: np.ndarray,
) -> Path:
    out_dir = EMB_ROOT / mode / f"ep{epoch:02d}" / extraction_mode
    out_dir.mkdir(parents=True, exist_ok=True)
    h_path = out_dir / f"h_{split}.npy"
    id_path = out_dir / f"ids_{split}.npy"
    np.save(h_path, h.astype(np.float32, copy=False))
    np.save(id_path, node_ids.astype(np.int64, copy=False))
    return h_path


def _random40_indices(y_all: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sss = StratifiedShuffleSplit(n_splits=1, train_size=0.4, random_state=seed)
    tr_r, te_r = next(sss.split(np.arange(len(y_all)), y_all))
    sss_inner = StratifiedShuffleSplit(n_splits=1, train_size=0.75, random_state=seed + 1)
    tr_r2, va_r = next(sss_inner.split(tr_r, y_all[tr_r]))
    return tr_r[tr_r2], tr_r[va_r], te_r


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return None  # never dump raw arrays into summary JSON
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        if not np.isfinite(f):
            return None
        return f
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _write_json(path: Path, payload: dict) -> None:
    _refuse_protected(path)
    if path.exists():
        raise SystemExit(f"Refusing overwrite existing {path}")
    _ensure_parent(path)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n")


def _metric_snapshot(suite: Dict[str, Any]) -> Dict[str, Any]:
    """Compact X/H/HxX metrics for tables."""
    out = {}
    for rep in ("X", "H", "HxX"):
        block = suite[rep]
        out[rep] = {
            "val_ranking": block.get("val_ranking") or {},
            "threshold_0.5": block["threshold_0.5"],
            "threshold_val_selected": block["threshold_val_selected"],
            "val_at_selected_threshold": block.get("val_at_selected_threshold") or {},
        }
    return out


def evaluate_temporal(
    *,
    h_tr: np.ndarray,
    h_va: np.ndarray,
    h_te: np.ndarray,
    x_full: np.ndarray,
    y_all: np.ndarray,
    tr: np.ndarray,
    va: np.ndarray,
    te: np.ndarray,
    seed: int,
    device: torch.device,
) -> Dict[str, Any]:
    return train_eval_mlp_suite(
        h_tr,
        x_full[tr],
        y_all[tr],
        h_te,
        x_full[te],
        y_all[te],
        h_val=h_va,
        x_val=x_full[va],
        y_val=y_all[va],
        seed=seed,
        device=device,
    )


def evaluate_random40(
    *,
    h_all: np.ndarray,
    x_full: np.ndarray,
    y_all: np.ndarray,
    seed: int,
    device: torch.device,
) -> Dict[str, Any]:
    tr_idx, va_idx, te_idx = _random40_indices(y_all, seed)
    return {
        "split_sizes": {
            "train": int(tr_idx.shape[0]),
            "val": int(va_idx.shape[0]),
            "test": int(te_idx.shape[0]),
        },
        "labels": [
            "random-40",
            "transductive",
            "diagnostic-only",
            "not thesis-primary",
        ],
        "metrics": train_eval_mlp_suite(
            h_all[tr_idx],
            x_full[tr_idx],
            y_all[tr_idx],
            h_all[te_idx],
            x_full[te_idx],
            y_all[te_idx],
            h_val=h_all[va_idx],
            x_val=x_full[va_idx],
            y_val=y_all[va_idx],
            seed=seed,
            device=device,
        ),
    }


def run_smoke(*, data_config: str, seed: int, device_str: str, ckpt_dir: Path) -> Dict[str, Any]:
    t_wall0 = time.perf_counter()
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        logging.warning("Smoke running on CPU — GPU feasibility cannot be proven")

    data = _load_data(data_config, seed)
    assert data["n_nodes"] == EXPECTED_N_NODES_FULL, (
        f"n_nodes {data['n_nodes']} != {EXPECTED_N_NODES_FULL}"
    )
    assert int(data["flow_stats"].n_edges) == EXPECTED_EDGE_COUNTS["full"]

    ckpt = ckpt_dir / "epoch_05.pt"
    if not ckpt.is_file():
        raise SystemExit(f"Missing smoke checkpoint {ckpt}")
    ckpt_hash = sha256_file(ckpt)
    encoder, meta = load_encoder_from_checkpoint(
        ckpt, in_dim=data["in_dim"], emb_dim=128, map_location=str(device)
    )
    encoder.to(device)

    smoke: Dict[str, Any] = {
        "not_exact_reproduction": bool(NOT_EXACT_REPRODUCTION),
        "phase": "smoke",
        "mode": "B_gcpal",
        "epoch": 5,
        "checkpoint_path": str(ckpt),
        "checkpoint_hash_sha256": ckpt_hash,
        "checkpoint_meta": meta,
        "device": str(device),
        "n_nodes": data["n_nodes"],
        "n_edges_full": int(data["flow_stats"].n_edges),
        "gnn_training_occurred": False,
        "forwards": {},
        "failures": [],
        "pass": False,
    }

    try:
        t0 = time.perf_counter()
        out = extract_temporal_expanding_window(
            encoder=encoder,
            x_all=data["x_full"],
            flow_ei=data["flow_full"],
            tr=data["tr"],
            va=data["va"],
            te=data["te"],
            device=device,
            config=temporal_expanding_window_config(seed=seed),
            checkpoint_path=ckpt,
            verify_expected_edges=True,
        )
        smoke["forwards"]["temporal_expanding_window"] = {
            "wall_seconds_total": time.perf_counter() - t0,
            "scope_edge_checks": out["scope_edge_checks"],
            "forward_diagnostics": out["forward_diagnostics"],
            "coverage": out["coverage"],
            "config_hash": out["config_hash"],
            "row_id_hashes_sha256": out["row_id_hashes_sha256"],
            "extraction_mode": out["extraction_mode"],
        }
        for split, h in out["embeddings"].items():
            _cache_h(
                mode="B_gcpal",
                epoch=5,
                extraction_mode=TEMPORAL_EXPANDING_WINDOW_V1,
                split=split,
                h=h,
                node_ids=out["split_node_ids"][split],
            )
        # Quick MLP on smoke embeddings (temporal)
        t_mlp = time.perf_counter()
        temporal = evaluate_temporal(
            h_tr=out["embeddings"]["train"],
            h_va=out["embeddings"]["val"],
            h_te=out["embeddings"]["test"],
            x_full=data["x_full"],
            y_all=data["y_all"],
            tr=data["tr"],
            va=data["va"],
            te=data["te"],
            seed=seed,
            device=device,
        )
        smoke["forwards"]["temporal_expanding_window"]["mlp_wall_seconds"] = (
            time.perf_counter() - t_mlp
        )
        smoke["forwards"]["temporal_expanding_window"]["metrics_snapshot"] = _metric_snapshot(
            temporal
        )
    except Exception as e:
        smoke["failures"].append({"stage": "temporal_expanding_window", "error": str(e), "tb": traceback.format_exc()})
        logging.exception("Expanding-window smoke failed")

    # Joint full-graph (same full scope as expanding test encode, separate call)
    try:
        t0 = time.perf_counter()
        all_ids = np.arange(data["n_nodes"], dtype=np.int64)
        joint = extract_joint_full_graph(
            encoder=encoder,
            x_all=data["x_full"],
            flow_ei=data["flow_full"],
            all_node_ids=all_ids,
            device=device,
            config=joint_full_graph_random40_config(seed=seed),
            checkpoint_path=ckpt,
            verify_expected_edges=True,
        )
        smoke["forwards"]["joint_full_graph"] = {
            "wall_seconds_total": time.perf_counter() - t0,
            "scope_edge_checks": joint["scope_edge_checks"],
            "forward_diagnostics": joint["forward_diagnostics"],
            "coverage": joint["coverage"],
            "config_hash": joint["config_hash"],
            "extraction_mode": joint["extraction_mode"],
            "label": joint["label"],
        }
        _cache_h(
            mode="B_gcpal",
            epoch=5,
            extraction_mode=JOINT_FULL_GRAPH_RANDOM40_V1,
            split="all",
            h=joint["embeddings"]["all"],
            node_ids=all_ids,
        )
    except Exception as e:
        smoke["failures"].append({"stage": "joint_full_graph", "error": str(e), "tb": traceback.format_exc()})
        logging.exception("Joint full-graph smoke failed")

    # Per-split isolation smoke (train only is cheapest extra; do all three for completeness)
    try:
        t0 = time.perf_counter()
        per = extract_split_embeddings(
            encoder=encoder,
            x_all=data["x_full"],
            flow_ei=data["flow_full"],
            split_node_ids={"train": data["tr"], "val": data["va"], "test": data["te"]},
            device=device,
            config=per_temporal_split_config(seed=seed),
            checkpoint_path=ckpt,
        )
        smoke["forwards"]["per_temporal_split"] = {
            "wall_seconds_total": time.perf_counter() - t0,
            "forward_diagnostics": per["forward_diagnostics"],
            "coverage": per["coverage"],
            "extraction_mode": per["extraction_mode"],
        }
    except Exception as e:
        smoke["failures"].append({"stage": "per_temporal_split", "error": str(e), "tb": traceback.format_exc()})
        logging.exception("Per-split smoke failed")

    wall = time.perf_counter() - t_wall0
    smoke["wall_seconds"] = wall
    smoke["peak_cpu_rss_mib"] = _cpu_rss_mib()
    if device.type == "cuda":
        smoke["peak_gpu_allocated_mib"] = float(torch.cuda.max_memory_allocated(device) / (1024**2))
        smoke["peak_gpu_reserved_mib"] = float(torch.cuda.max_memory_reserved(device) / (1024**2))
    else:
        smoke["peak_gpu_allocated_mib"] = None
        smoke["peak_gpu_reserved_mib"] = None

    # Feasibility: one checkpoint smoke cost * 8 (A/B × 4 epochs), plus margin for MLP on all protocols
    ew = smoke["forwards"].get("temporal_expanding_window", {})
    jw = smoke["forwards"].get("joint_full_graph", {})
    pw = smoke["forwards"].get("per_temporal_split", {})
    per_ckpt_est = (
        float(ew.get("wall_seconds_total") or 0.0)
        + float(ew.get("mlp_wall_seconds") or 0.0)
        + float(jw.get("wall_seconds_total") or 0.0)
        + float(pw.get("wall_seconds_total") or 0.0)
        + 120.0  # MLP for joint + per-split protocols
    )
    suite_est = per_ckpt_est * 8
    smoke["timing_estimate"] = {
        "per_checkpoint_seconds": per_ckpt_est,
        "full_suite_A_B_8ckpt_seconds": suite_est,
        "six_hour_budget_seconds": SIX_HOUR_SECONDS,
        "fits_six_hour_advanced_gpu": suite_est <= SUITE_TIME_BUDGET,
    }

    # Pass criteria
    reasons = []
    if smoke["failures"]:
        reasons.append("runtime_failures")
    ew_checks = (ew.get("scope_edge_checks") or []) if ew else []
    if not ew_checks or not all(c.get("matches_expected") for c in ew_checks):
        reasons.append("edge_count_mismatch")
    cov = ew.get("coverage") or {}
    for split in ("train", "val", "test"):
        c = cov.get(split) or {}
        if c.get("coverage") != 1.0 or not c.get("all_finite"):
            reasons.append(f"coverage_or_finite_{split}")
    # No silently dropped edges (retained fraction must be 1.0)
    for name, block in (ew.get("forward_diagnostics") or {}).items():
        if float(block.get("retained_edge_fraction_of_induced_scope", 0)) != 1.0:
            reasons.append(f"dropped_edges_{name}")
        if int(block.get("duplicate_output_ids", 0)) != 0:
            reasons.append(f"duplicate_ids_{name}")
    if not smoke["timing_estimate"]["fits_six_hour_advanced_gpu"]:
        reasons.append("suite_exceeds_six_hour_estimate")
        smoke["scalable_inference_required"] = True
        smoke["scalable_inference_note"] = (
            "Full-scope one-shot encode timing/memory does not fit the 6h Advanced GPU "
            "envelope for the A/B×4 suite. Need layer-wise or halo-partition inference "
            "with numerical agreement vs full-graph; do not silently change graph semantics."
        )

    smoke["pass_reasons_failed"] = reasons
    smoke["pass"] = len(reasons) == 0
    smoke["slurm_job_id"] = os.environ.get("SLURM_JOB_ID")
    return smoke


def run_arm(
    *,
    mode: str,
    data_config: str,
    seed: int,
    device_str: str,
    ckpt_dir: Path,
    epochs: Sequence[int],
    require_smoke_gate: bool,
) -> Dict[str, Any]:
    if require_smoke_gate:
        if not SMOKE_GATE.is_file():
            raise SystemExit(f"Missing smoke gate {SMOKE_GATE}; run smoke first")
        gate = json.loads(SMOKE_GATE.read_text())
        if not gate.get("pass"):
            raise SystemExit(f"Smoke gate failed; refusing full arm. Gate={gate}")

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    data = _load_data(data_config, seed)
    arm: Dict[str, Any] = {
        "not_exact_reproduction": bool(NOT_EXACT_REPRODUCTION),
        "phase": "arm",
        "mode": mode,
        "seed": seed,
        "device": str(device),
        "gnn_training_occurred": False,
        "ckpt_dir": str(ckpt_dir),
        "epochs": list(epochs),
        "by_epoch": {},
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "wall_seconds": None,
    }
    t0 = time.perf_counter()
    for ep in epochs:
        ckpt = ckpt_dir / f"epoch_{int(ep):02d}.pt"
        if not ckpt.is_file():
            raise SystemExit(f"Missing checkpoint {ckpt}")
        ckpt_hash = sha256_file(ckpt)
        encoder, meta = load_encoder_from_checkpoint(
            ckpt, in_dim=data["in_dim"], emb_dim=128, map_location=str(device)
        )
        encoder.to(device)
        ep_rec: Dict[str, Any] = {
            "epoch": int(ep),
            "checkpoint_path": str(ckpt),
            "checkpoint_hash_sha256": ckpt_hash,
            "checkpoint_meta": meta,
            "protocols": {},
        }

        # 1) Temporal expanding-window (thesis-primary candidate)
        ew = extract_temporal_expanding_window(
            encoder=encoder,
            x_all=data["x_full"],
            flow_ei=data["flow_full"],
            tr=data["tr"],
            va=data["va"],
            te=data["te"],
            device=device,
            config=temporal_expanding_window_config(seed=seed),
            checkpoint_path=ckpt,
            verify_expected_edges=(ep == epochs[0]),
        )
        for split, h in ew["embeddings"].items():
            _cache_h(
                mode=mode,
                epoch=int(ep),
                extraction_mode=TEMPORAL_EXPANDING_WINDOW_V1,
                split=split,
                h=h,
                node_ids=ew["split_node_ids"][split],
            )
        temporal_metrics = evaluate_temporal(
            h_tr=ew["embeddings"]["train"],
            h_va=ew["embeddings"]["val"],
            h_te=ew["embeddings"]["test"],
            x_full=data["x_full"],
            y_all=data["y_all"],
            tr=data["tr"],
            va=data["va"],
            te=data["te"],
            seed=seed,
            device=device,
        )
        ep_rec["protocols"][TEMPORAL_EXPANDING_WINDOW_V1] = {
            "extraction_mode": TEMPORAL_EXPANDING_WINDOW_V1,
            "protocol_role": "thesis_primary_candidate",
            "config_hash": ew["config_hash"],
            "row_id_hashes_sha256": ew["row_id_hashes_sha256"],
            "scope_edge_checks": ew.get("scope_edge_checks"),
            "forward_diagnostics": ew["forward_diagnostics"],
            "coverage": ew["coverage"],
            "temporal_primary": _metric_snapshot(temporal_metrics),
            "graph_scope_note": ew["graph_scope_note"],
        }

        # 2) Per-split isolation (sensitivity)
        per = extract_split_embeddings(
            encoder=encoder,
            x_all=data["x_full"],
            flow_ei=data["flow_full"],
            split_node_ids={"train": data["tr"], "val": data["va"], "test": data["te"]},
            device=device,
            config=per_temporal_split_config(seed=seed),
            checkpoint_path=ckpt,
        )
        for split, h in per["embeddings"].items():
            _cache_h(
                mode=mode,
                epoch=int(ep),
                extraction_mode=PER_TEMPORAL_SPLIT_V1,
                split=split,
                h=h,
                node_ids=per["split_node_ids"][split],
            )
        per_metrics = evaluate_temporal(
            h_tr=per["embeddings"]["train"],
            h_va=per["embeddings"]["val"],
            h_te=per["embeddings"]["test"],
            x_full=data["x_full"],
            y_all=data["y_all"],
            tr=data["tr"],
            va=data["va"],
            te=data["te"],
            seed=seed,
            device=device,
        )
        ep_rec["protocols"][PER_TEMPORAL_SPLIT_V1] = {
            "extraction_mode": PER_TEMPORAL_SPLIT_V1,
            "protocol_role": "sensitivity",
            "config_hash": per["config_hash"],
            "row_id_hashes_sha256": per["row_id_hashes_sha256"],
            "forward_diagnostics": per["forward_diagnostics"],
            "coverage": per["coverage"],
            "temporal_primary": _metric_snapshot(per_metrics),
            "graph_scope_note": per["graph_scope_note"],
        }

        # 3) Joint full-graph random-40 (diagnostic)
        all_ids = np.arange(data["n_nodes"], dtype=np.int64)
        joint = extract_joint_full_graph(
            encoder=encoder,
            x_all=data["x_full"],
            flow_ei=data["flow_full"],
            all_node_ids=all_ids,
            device=device,
            config=joint_full_graph_random40_config(seed=seed),
            checkpoint_path=ckpt,
            verify_expected_edges=(ep == epochs[0]),
        )
        _cache_h(
            mode=mode,
            epoch=int(ep),
            extraction_mode=JOINT_FULL_GRAPH_RANDOM40_V1,
            split="all",
            h=joint["embeddings"]["all"],
            node_ids=all_ids,
        )
        r40 = evaluate_random40(
            h_all=joint["embeddings"]["all"],
            x_full=data["x_full"],
            y_all=data["y_all"],
            seed=seed,
            device=device,
        )
        ep_rec["protocols"][JOINT_FULL_GRAPH_RANDOM40_V1] = {
            "extraction_mode": JOINT_FULL_GRAPH_RANDOM40_V1,
            "protocol_role": "diagnostic_random40",
            "config_hash": joint["config_hash"],
            "coverage": joint["coverage"],
            "forward_diagnostics": joint["forward_diagnostics"],
            "scope_edge_checks": joint.get("scope_edge_checks"),
            "label": joint["label"],
            "random40_diagnostic": {
                "split_sizes": r40["split_sizes"],
                "labels": r40["labels"],
                "metrics": _metric_snapshot(r40["metrics"]),
            },
            "graph_scope_note": joint["graph_scope_note"],
        }

        arm["by_epoch"][str(ep)] = ep_rec
        logging.info("Completed %s epoch %s", mode, ep)

    arm["wall_seconds"] = time.perf_counter() - t0
    arm["peak_cpu_rss_mib"] = _cpu_rss_mib()
    if device.type == "cuda":
        arm["peak_gpu_allocated_mib"] = float(torch.cuda.max_memory_allocated(device) / (1024**2))
        arm["peak_gpu_reserved_mib"] = float(torch.cuda.max_memory_reserved(device) / (1024**2))
    return arm


def _val_auprc(ep_rec: dict, protocol: str, rep: str = VAL_SELECT_REP) -> float:
    block = ep_rec["protocols"][protocol]["temporal_primary"][rep]["val_ranking"]
    return float(block["auprc"])


def _select_checkpoint(by_epoch: dict, protocol: str) -> Dict[str, Any]:
    best_ep, best_val = None, -1.0
    curve = {}
    for ep_s, rec in by_epoch.items():
        v = _val_auprc(rec, protocol)
        curve[ep_s] = v
        if v > best_val:
            best_val = v
            best_ep = int(ep_s)
    return {
        "protocol": protocol,
        "selection_metric": f"temporal_val_{VAL_SELECT_REP}_{VAL_SELECT_KEY}",
        "selected_epoch": best_ep,
        "selected_val_auprc": best_val,
        "curve_val_auprc": curve,
        "never_used_test_for_selection": True,
    }


def _load_legacy_chunked_metrics() -> Dict[str, Any]:
    """Read historical 20ep scout JSONs for comparison (do not modify)."""
    out = {}
    for mode in ("A_identity", "B_gcpal"):
        path = Path(f"results/diagnostics/gcpal_txn_node_poscomplete_scout_{mode}_20ep_seed2.json")
        if not path.is_file():
            out[mode] = {"missing": True, "path": str(path)}
            continue
        blob = json.loads(path.read_text())
        curve = blob.get("learning_curve") or {}
        slim = {}
        for ep, rec in curve.items():
            tp = rec.get("temporal_primary") or {}
            slim[ep] = {
                rep: {
                    "val_auprc": (tp.get(rep) or {}).get("val_ranking", {}).get("auprc"),
                    "test_auprc_0.5": (tp.get(rep) or {}).get("threshold_0.5", {}).get("auprc"),
                    "test_auroc_0.5": (tp.get(rep) or {}).get("threshold_0.5", {}).get("auroc"),
                    "test_f1_0.5": (tp.get(rep) or {}).get("threshold_0.5", {}).get("f1"),
                }
                for rep in ("X", "H", "HxX")
            }
        out[mode] = {
            "path": str(path),
            "extraction_mode": LEGACY_CHUNKED_EXTRACTION_MODE,
            "selected_epoch_legacy": (blob.get("checkpoint_selection") or {}).get("selected_epoch"),
            "by_epoch": slim,
        }
    return out


def run_aggregate(*, arm_paths: Sequence[Path], smoke_path: Path, out_json: Path, out_md: Path) -> Dict[str, Any]:
    smoke = json.loads(smoke_path.read_text()) if smoke_path.is_file() else {}
    arms = {}
    for p in arm_paths:
        blob = json.loads(p.read_text())
        arms[blob["mode"]] = blob

    selections = {}
    for mode, arm in arms.items():
        selections[mode] = _select_checkpoint(arm["by_epoch"], TEMPORAL_EXPANDING_WINDOW_V1)

    legacy = _load_legacy_chunked_metrics()

    def _hx_x_test(arm, ep, protocol, key="auprc"):
        return arm["by_epoch"][str(ep)]["protocols"][protocol]["temporal_primary"]["HxX"][
            "threshold_0.5"
        ].get(key)

    comparisons_temporal = []
    for mode in sorted(arms.keys()):
        arm = arms[mode]
        sel = selections[mode]["selected_epoch"]
        row = {
            "mode": mode,
            "validation_selected_epoch_expanding": sel,
            "fixed_epoch_20": 20,
            "legacy_chunked_val_selected_epoch": legacy.get(mode, {}).get("selected_epoch_legacy"),
            "curves": {},
        }
        for ep in EVAL_EPOCHS:
            ep_s = str(ep)
            row["curves"][ep_s] = {
                "legacy_chunked_induce_4096_v0": (legacy.get(mode) or {}).get("by_epoch", {}).get(ep_s),
                PER_TEMPORAL_SPLIT_V1: {
                    "val_HxX_auprc": _val_auprc(arm["by_epoch"][ep_s], PER_TEMPORAL_SPLIT_V1),
                    "test_HxX_auprc_0.5": _hx_x_test(arm, ep, PER_TEMPORAL_SPLIT_V1),
                    "test_HxX_auroc_0.5": _hx_x_test(arm, ep, PER_TEMPORAL_SPLIT_V1, "auroc"),
                    "test_HxX_f1_0.5": _hx_x_test(arm, ep, PER_TEMPORAL_SPLIT_V1, "f1"),
                },
                TEMPORAL_EXPANDING_WINDOW_V1: {
                    "val_HxX_auprc": _val_auprc(arm["by_epoch"][ep_s], TEMPORAL_EXPANDING_WINDOW_V1),
                    "test_HxX_auprc_0.5": _hx_x_test(arm, ep, TEMPORAL_EXPANDING_WINDOW_V1),
                    "test_HxX_auroc_0.5": _hx_x_test(arm, ep, TEMPORAL_EXPANDING_WINDOW_V1, "auroc"),
                    "test_HxX_f1_0.5": _hx_x_test(arm, ep, TEMPORAL_EXPANDING_WINDOW_V1, "f1"),
                },
            }
        comparisons_temporal.append(row)

    comparisons_random40 = []
    for mode in sorted(arms.keys()):
        arm = arms[mode]
        curve = {}
        for ep in EVAL_EPOCHS:
            r = arm["by_epoch"][str(ep)]["protocols"][JOINT_FULL_GRAPH_RANDOM40_V1][
                "random40_diagnostic"
            ]["metrics"]
            curve[str(ep)] = {
                "val_HxX_auprc": r["HxX"]["val_ranking"].get("auprc"),
                "test_HxX_auprc_0.5": r["HxX"]["threshold_0.5"].get("auprc"),
                "test_HxX_auroc_0.5": r["HxX"]["threshold_0.5"].get("auroc"),
                "test_HxX_f1_0.5": r["HxX"]["threshold_0.5"].get("f1"),
                "label": [
                    "random-40",
                    "transductive",
                    "diagnostic-only",
                    "not thesis-primary",
                ],
            }
        comparisons_random40.append({"mode": mode, "curves": curve})

    # B vs A under expanding-window at newly selected epochs and at ep20
    def _beat(metric_a, metric_b) -> Optional[bool]:
        if metric_a is None or metric_b is None:
            return None
        return float(metric_b) > float(metric_a)

    b_vs_a = {}
    if "A_identity" in arms and "B_gcpal" in arms:
        for label, ep_getter in (
            ("val_selected", lambda m: selections[m]["selected_epoch"]),
            ("fixed_ep20", lambda m: 20),
        ):
            ea, eb = ep_getter("A_identity"), ep_getter("B_gcpal")
            a_val = _val_auprc(arms["A_identity"]["by_epoch"][str(ea)], TEMPORAL_EXPANDING_WINDOW_V1)
            b_val = _val_auprc(arms["B_gcpal"]["by_epoch"][str(eb)], TEMPORAL_EXPANDING_WINDOW_V1)
            a_te = _hx_x_test(arms["A_identity"], ea, TEMPORAL_EXPANDING_WINDOW_V1)
            b_te = _hx_x_test(arms["B_gcpal"], eb, TEMPORAL_EXPANDING_WINDOW_V1)
            b_vs_a[label] = {
                "A_epoch": ea,
                "B_epoch": eb,
                "A_val_HxX_auprc": a_val,
                "B_val_HxX_auprc": b_val,
                "A_test_HxX_auprc_0.5": a_te,
                "B_test_HxX_auprc_0.5": b_te,
                "B_beats_A_on_val_HxX_auprc": _beat(a_val, b_val),
                "B_beats_A_on_test_HxX_auprc_0.5": _beat(a_te, b_te),
            }

    payload = {
        "not_exact_reproduction": bool(NOT_EXACT_REPRODUCTION),
        "title": "gcpal_txn_node_canonical_reextraction",
        "gnn_training_occurred": False,
        "smoke": {
            "path": str(smoke_path),
            "pass": smoke.get("pass"),
            "slurm_job_id": smoke.get("slurm_job_id"),
            "timing_estimate": smoke.get("timing_estimate"),
            "peak_gpu_allocated_mib": smoke.get("peak_gpu_allocated_mib"),
            "peak_gpu_reserved_mib": smoke.get("peak_gpu_reserved_mib"),
        },
        "arm_job_ids": {m: arms[m].get("slurm_job_id") for m in arms},
        "artifact_classification": {
            "original_online_augmented_5ep": {
                "status": "noncanonical",
                "table_eligible": False,
            },
            "replay_legacy_chunked": {
                "status": "internally_comparable_A_B_diagnostic",
                "table_eligible": False,
                "wording": (
                    "B beats A under a shared frozen-checkpoint legacy-chunked extraction; "
                    "canonical graph-preserving re-extraction is pending."
                    if not arms
                    else (
                        "B beats A under a shared frozen-checkpoint legacy-chunked extraction; "
                        "canonical graph-preserving re-extraction results are reported separately."
                    )
                ),
            },
            "per_split_full_induce_v1": {
                "status": "sensitivity_analysis",
                "table_eligible": False,
                "mode_id": PER_TEMPORAL_SPLIT_V1,
            },
            "temporal_expanding_window_v1": {
                "status": "candidate_canonical_thesis_primary_extraction",
                "table_eligible": "candidate_after_review",
                "mode_id": TEMPORAL_EXPANDING_WINDOW_V1,
            },
            "joint_full_graph_random40_v1": {
                "status": "candidate_GCPAL_aligned_diagnostic",
                "table_eligible": False,
                "never_primary": True,
                "mode_id": JOINT_FULL_GRAPH_RANDOM40_V1,
            },
        },
        "checkpoint_selection_expanding_window": selections,
        "comparisons_temporal": comparisons_temporal,
        "comparisons_random40": comparisons_random40,
        "b_vs_a_expanding_window": b_vs_a,
        "embedding_cache_root": str(EMB_ROOT),
        "config_hash_note": "Per-protocol config_hash recorded in arm JSON by epoch",
    }

    # Write MD
    lines = [
        "# Canonical transaction-node checkpoint re-extraction",
        "",
        "Status: **extraction + frozen MLP eval only** · **No GNN retraining**",
        "",
        f"Companion: [`{out_json}`](../{out_json})",
        "",
        "## Smoke / feasibility",
        "",
        f"- Smoke job ID: `{smoke.get('slurm_job_id')}`",
        f"- Pass: **{smoke.get('pass')}**",
        f"- Peak GPU allocated MiB: `{smoke.get('peak_gpu_allocated_mib')}`",
        f"- Peak GPU reserved MiB: `{smoke.get('peak_gpu_reserved_mib')}`",
        f"- Suite estimate (s): `{((smoke.get('timing_estimate') or {}).get('full_suite_A_B_8ckpt_seconds'))}`",
        f"- Fits 6h Advanced GPU: `{(smoke.get('timing_estimate') or {}).get('fits_six_hour_advanced_gpu')}`",
        "",
        "## Full-suite job IDs",
        "",
    ]
    for m, jid in payload["arm_job_ids"].items():
        lines.append(f"- `{m}`: `{jid}`")
    lines += [
        "",
        "## Checkpoint selection (expanding-window temporal val HxX AUPRC)",
        "",
        "Selection never uses test metrics. Legacy epoch-5 selection is **not** carried forward.",
        "",
    ]
    for mode, sel in selections.items():
        lines.append(
            f"- **{mode}**: selected epoch **{sel['selected_epoch']}** "
            f"(val HxX AUPRC={sel['selected_val_auprc']:.6f}); "
            f"curve={sel['curve_val_auprc']}"
        )
    lines += ["", "## B vs A (expanding-window)", ""]
    for k, v in b_vs_a.items():
        lines.append(
            f"- **{k}**: A@ep{v['A_epoch']} val={v['A_val_HxX_auprc']:.6f} test={v['A_test_HxX_auprc_0.5']:.6f}; "
            f"B@ep{v['B_epoch']} val={v['B_val_HxX_auprc']:.6f} test={v['B_test_HxX_auprc_0.5']:.6f}; "
            f"B>A val={v['B_beats_A_on_val_HxX_auprc']} test={v['B_beats_A_on_test_HxX_auprc_0.5']}"
        )
    lines += [
        "",
        "## Artifact classification",
        "",
        "| Artifact | Status | Table eligible |",
        "|----------|--------|----------------|",
        "| Original online augmented 5ep | noncanonical | No |",
        "| Replay legacy-chunked | internally comparable A/B diagnostic | No |",
        "| Per-split full induce v1 | sensitivity | No |",
        "| Temporal expanding-window v1 | candidate thesis-primary extraction | candidate after review |",
        "| Joint full-graph random-40 v1 | GCPAL-aligned diagnostic | No (never primary) |",
        "",
        "Corrected legacy wording: **B beats A under a shared frozen-checkpoint "
        "legacy-chunked extraction; canonical graph-preserving re-extraction results "
        "are reported in this note.**",
        "",
        "## Confirmation",
        "",
        "- No GNN training occurred in this suite.",
        "- Historical original-scout and legacy-chunked artifacts were not rewritten.",
        "",
    ]
    _refuse_protected(out_md)
    if out_md.exists():
        raise SystemExit(f"Refusing overwrite {out_md}")
    _ensure_parent(out_md)
    out_md.write_text("\n".join(lines) + "\n")

    if out_json.exists():
        raise SystemExit(f"Refusing overwrite {out_json}")
    _write_json(out_json, payload)
    return payload


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=["smoke", "arm", "aggregate"], required=True)
    p.add_argument("--mode", choices=["A_identity", "B_gcpal"], default="B_gcpal")
    p.add_argument("--data_config", default="data_config.json")
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--ckpt_dir", default=None)
    p.add_argument("--epochs", default="5,10,15,20")
    p.add_argument("--output_json", default=None)
    p.add_argument(
        "--no_require_smoke_gate",
        action="store_true",
        help="Skip smoke gate check (debug only).",
    )
    p.add_argument("--arm_json", action="append", default=[])
    p.add_argument("--smoke_json", default=str(SMOKE_JSON))
    p.add_argument("--output_md", default="notes/gcpal_txn_node_canonical_reextraction.md")
    args = p.parse_args()
    assert NOT_EXACT_REPRODUCTION

    if args.phase == "smoke":
        ckpt_dir = Path(args.ckpt_dir or DEFAULT_CKPT["B_gcpal"])
        smoke = run_smoke(
            data_config=args.data_config,
            seed=args.seed,
            device_str=args.device,
            ckpt_dir=ckpt_dir,
        )
        out = Path(args.output_json or SMOKE_JSON)
        if out.exists():
            # Allow re-smoke only if prior failed? Prefer unique path via job id.
            stem = out.stem
            out = out.with_name(f"{stem}_job{os.environ.get('SLURM_JOB_ID', 'local')}.json")
        _write_json(out, smoke)
        gate = {
            "pass": bool(smoke["pass"]),
            "smoke_json": str(out),
            "slurm_job_id": smoke.get("slurm_job_id"),
            "timing_estimate": smoke.get("timing_estimate"),
            "pass_reasons_failed": smoke.get("pass_reasons_failed"),
            "scalable_inference_required": smoke.get("scalable_inference_required", False),
        }
        # Gate file: overwrite allowed only for the gate pointer itself
        _ensure_parent(SMOKE_GATE)
        SMOKE_GATE.write_text(json.dumps(gate, indent=2) + "\n")
        logging.info("Smoke pass=%s wrote %s gate=%s", smoke["pass"], out, SMOKE_GATE)
        if not smoke["pass"]:
            raise SystemExit(2)
        return

    if args.phase == "arm":
        require = not args.no_require_smoke_gate
        ckpt_dir = Path(args.ckpt_dir or DEFAULT_CKPT[args.mode])
        epochs = [int(x) for x in args.epochs.split(",") if x.strip()]
        arm = run_arm(
            mode=args.mode,
            data_config=args.data_config,
            seed=args.seed,
            device_str=args.device,
            ckpt_dir=ckpt_dir,
            epochs=epochs,
            require_smoke_gate=require,
        )
        out = Path(
            args.output_json
            or f"results/diagnostics/gcpal_txn_node_canonical_reextract_{args.mode}_seed{args.seed}.json"
        )
        if out.exists():
            out = out.with_name(f"{out.stem}_job{os.environ.get('SLURM_JOB_ID', 'local')}.json")
        _write_json(out, arm)
        logging.info("Wrote arm JSON %s", out)
        return

    if args.phase == "aggregate":
        arms = [Path(x) for x in args.arm_json]
        if len(arms) < 2:
            raise SystemExit("aggregate requires --arm_json for A and B")
        out_json = Path(args.output_json or "results/diagnostics/gcpal_txn_node_canonical_reextraction.json")
        out_md = Path(args.output_md)
        run_aggregate(
            arm_paths=arms,
            smoke_path=Path(args.smoke_json),
            out_json=out_json,
            out_md=out_md,
        )
        logging.info("Wrote %s and %s", out_json, out_md)
        return


if __name__ == "__main__":
    main()
