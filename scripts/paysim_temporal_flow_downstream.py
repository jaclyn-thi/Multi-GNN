#!/usr/bin/env python3
"""PaySim temporal-flow downstream Phase-1 (validation-only, frozen H).

Subcommands:
  smoke       — unit tests + timestamp multiplicity + tiny causal cache integrity
  build_cache — full PaySim TF cache under temporal_flow_cache/PaySim/
  integrity   — cache + frozen-H join gates (train/val only)
  ablation    — seed-2 validation-only 7-stack logistic (+ optional random H+X+TF)
  aggregate   — write notes/JSON gate report (no test metrics)

Encoder training/finetuning is forbidden. Test evaluation is forbidden in Phase 1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from linear_probe import load_embedding_npz  # noqa: E402
from morphology.temporal_flow_causal import (  # noqa: E402
    TEMPORAL_FLOW_CAUSAL_FEATURE_NAMES,
    compute_temporal_flow_causal_features,
)
from ranking_metrics import alert_budget_metrics  # noqa: E402
from train_util import extract_param  # noqa: E402
from util import create_parser, logger_setup, set_seed  # noqa: E402

TAG = "paysim_temporal_flow_downstream"
CACHE_ROOT = ROOT / "temporal_flow_cache"
CACHE_DIR = CACHE_ROOT / "PaySim"
SMOKE_CACHE_DIR = ROOT / "results" / "diagnostics" / TAG / "smoke_cache" / "PaySim"
RESULT_ROOT = ROOT / "results" / "diagnostics" / TAG
CELLS = RESULT_ROOT / "cells"
NOTES_MD = ROOT / "notes" / "paysim_temporal_flow_downstream_validation.md"
OUT_JSON = ROOT / "results" / "diagnostics" / "paysim_temporal_flow_downstream_validation.json"
SUBMISSION_JSON = RESULT_ROOT / "submission.json"
INTEGRITY_JSON = RESULT_ROOT / "integrity.json"
SMOKE_JSON = RESULT_ROOT / "smoke.json"
MULTIPLICITY_JSON = RESULT_ROOT / "timestamp_multiplicity.json"
ABLATION_JSON = RESULT_ROOT / "ablation_seed2_validation.json"

LOCKED_SEED = 2
DOWNSTREAM_SEED = 1
GATE_MARGIN = 0.003
FEATURE_CONTRACT = "paysim_legacy_duplicate_v1"
BN_PROTOCOL = "frozen_aml_bn"
NORM_PROTOCOL = "paysim_train_fit_edge_znorm"
SOURCE_UNIQUE = "gin_emlps_ports_tds_corrected_asym_proj_8192neg_queue0_40ep_seed2"
SOURCE_CKPT = ROOT / f"saved-models/checkpoint_{SOURCE_UNIQUE}.tar"
SOURCE_SHA256 = "18e06f555aa4880dfc1e95caa3f54a207e5aa186d266887772640feb93a06ae6"
EMB_DIR = (
    ROOT
    / "embeddings"
    / "final_corrected_no_preserve_multiseed"
    / "seed2_P1_strict_inductive_legacy"
)
RANDOM_EMB_DIR = (
    ROOT
    / "embeddings"
    / "final_corrected_no_preserve_multiseed"
    / "controls_random_paysim_legacy_duplicate_v1"
)
EXPECTED_ID_HASH = {
    "train": "2511d0de4504e52960b414e6b84d47486089a573b6c57aa040feb561e2d2809a",
    "val": "a8de85f31dfe91bd767da6daedf9f2bab474d08c8412c796111e8767ebd0b1e3",
}
STACKS = ("X", "TF", "X+TF", "H", "H+X", "H+TF", "H+X+TF")


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _sha256_arr(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(arr)
    return hashlib.sha256(a.tobytes()).hexdigest()


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        tmp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def gin_model_class_weight() -> Dict[int, float]:
    args = create_parser().parse_args(["--data", "PaySim", "--model", "gin", "--testing"])
    return {0: float(extract_param("w_ce1", args)), 1: float(extract_param("w_ce2", args))}


def tune_thr_max_f1(y: np.ndarray, proba: np.ndarray) -> float:
    y = y.astype(np.int64)
    if len(np.unique(y)) < 2:
        return 0.5
    prec, rec, thrs = precision_recall_curve(y, proba)
    if thrs.size == 0:
        return 0.5
    f1 = (2 * prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-12)
    return float(thrs[int(np.argmax(f1))])


def metrics_block(y: np.ndarray, proba: np.ndarray, thr: float) -> Dict[str, float]:
    y = y.astype(np.int64)
    pred = (proba >= float(thr)).astype(np.int64)
    out: Dict[str, float] = {
        "auroc": float(roc_auc_score(y, proba)) if len(np.unique(y)) > 1 else float("nan"),
        "auprc": float(average_precision_score(y, proba)) if len(np.unique(y)) > 1 else float("nan"),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "threshold": float(thr),
        "positive_prediction_rate": float(pred.mean()) if y.size else 0.0,
        "tp": float(((pred == 1) & (y == 1)).sum()),
        "fp": float(((pred == 1) & (y == 0)).sum()),
        "tn": float(((pred == 0) & (y == 0)).sum()),
        "fn": float(((pred == 0) & (y == 1)).sum()),
        "n": float(y.shape[0]),
        "n_positives": float(int(y.sum())),
        "positive_rate": float(y.mean()) if y.size else 0.0,
    }
    out.update(alert_budget_metrics(y, proba))
    return out


def ids_hash(edge_ids: np.ndarray) -> Dict[str, Any]:
    ids = np.asarray(edge_ids, dtype=np.int64).reshape(-1)
    return {
        "n": int(ids.shape[0]),
        "n_unique": int(np.unique(ids).shape[0]),
        "edge_id_sum": int(ids.sum()),
        "sha256_of_ids_bytes": _sha256_arr(ids),
    }


def verify_frozen_embeddings(emb_dir: Path) -> Dict[str, Any]:
    if not SOURCE_CKPT.is_file():
        raise SystemExit(f"missing checkpoint {SOURCE_CKPT}")
    sha = _sha256_file(SOURCE_CKPT)
    if sha != SOURCE_SHA256:
        raise SystemExit(f"checkpoint sha mismatch: {sha} != {SOURCE_SHA256}")
    meta_path = emb_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("feature_contract_id") != FEATURE_CONTRACT:
        raise SystemExit(f"feature_contract mismatch: {meta.get('feature_contract_id')}")
    if not bool(meta.get("train_fit_edge_znorm")):
        raise SystemExit("expected train_fit_edge_znorm=True")
    if meta.get("bn_protocol") != BN_PROTOCOL:
        raise SystemExit(f"bn_protocol mismatch: {meta.get('bn_protocol')}")
    if not bool(meta.get("encoder_frozen", True)):
        raise SystemExit("encoder_frozen must be true")
    if bool(meta.get("include_temporal_flow_edge_features")):
        raise SystemExit("encoder-input TF must be off for this ablation")

    splits = {}
    for sp in ("train", "val"):
        z, y, ids = load_embedding_npz(emb_dir / f"{sp}.npz")
        h = ids_hash(ids)
        if h["sha256_of_ids_bytes"] != EXPECTED_ID_HASH[sp]:
            raise SystemExit(
                f"{sp} edge_id hash mismatch: {h['sha256_of_ids_bytes']} != {EXPECTED_ID_HASH[sp]}"
            )
        if z.shape[1] != 128:
            raise SystemExit(f"{sp} expected post-128 H, got dim={z.shape[1]}")
        splits[sp] = {"Z": z, "y": y, "ids": ids, "ids_meta": h}
    return {
        "checkpoint_path": str(SOURCE_CKPT),
        "checkpoint_sha256": sha,
        "embeddings_dir": str(emb_dir),
        "feature_contract_id": FEATURE_CONTRACT,
        "normalization_protocol": NORM_PROTOCOL,
        "bn_protocol": BN_PROTOCOL,
        "encoder_frozen": True,
        "representation": "post_embedding_128",
        "include_temporal_flow_edge_features": False,
        "splits": {sp: splits[sp]["ids_meta"] for sp in splits},
        "coverage": {
            sp: {
                "n": int(splits[sp]["y"].shape[0]),
                "n_positives": int(splits[sp]["y"].sum()),
                "positive_rate": float(splits[sp]["y"].mean()),
            }
            for sp in splits
        },
        "_arrays": splits,
    }


def load_x_matrix() -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "probe_feature_ablation", ROOT / "scripts" / "probe_feature_ablation.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["probe_feature_ablation"] = mod
    spec.loader.exec_module(mod)
    df, df_train, tr_ids, va_ids, te_ids, dspec = mod.load_dataset_frames(
        "PaySim", str(ROOT / "data_config.json")
    )
    x_raw, names, _, meta = mod.build_full_feature_matrix(
        df, df_train, ("edge_native",), categorical_encoding="one_hot"
    )
    return x_raw.astype(np.float32), list(names), {
        "x_source": "edge_native_one_hot_train_fit",
        "feature_names": names,
        "n_rows": int(x_raw.shape[0]),
        "n_features": int(x_raw.shape[1]),
        "label_col": dspec.label_col,
        "split_counts": {
            "train": int(len(tr_ids)),
            "val": int(len(va_ids)),
            "test": int(len(te_ids)),
        },
        "meta": meta,
    }


def load_tf_cache(cache_dir: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    meta = json.loads((cache_dir / "meta.json").read_text(encoding="utf-8"))
    features = np.load(cache_dir / "features.npy").astype(np.float32)
    edge_id = np.load(cache_dir / "edge_id.npy").astype(np.int64)
    if features.shape[0] != edge_id.shape[0]:
        raise ValueError("TF features/edge_id length mismatch")
    if features.shape[1] != 5:
        raise ValueError(f"Expected 5 TF cols, got {features.shape[1]}")
    if meta.get("feature_names") != list(TEMPORAL_FLOW_CAUSAL_FEATURE_NAMES):
        raise ValueError(f"TF feature name order mismatch: {meta.get('feature_names')}")
    return features, edge_id, meta


def tf_lookup(features: np.ndarray, cache_edge_id: np.ndarray, query_ids: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Map query edge_ids to TF rows. Dense caches (edge_id[i]==i) use direct indexing."""
    q = np.asarray(query_ids, dtype=np.int64).reshape(-1)
    dense = cache_edge_id.shape[0] > 0 and np.array_equal(
        cache_edge_id, np.arange(cache_edge_id.shape[0], dtype=np.int64)
    )
    if dense:
        if q.min() < 0 or q.max() >= features.shape[0]:
            raise ValueError("query edge_id out of range for dense TF cache")
        unmatched = 0
        out = features[q]
    else:
        order = np.argsort(cache_edge_id)
        sorted_ids = cache_edge_id[order]
        pos = np.searchsorted(sorted_ids, q)
        in_range = pos < sorted_ids.shape[0]
        matched = in_range.copy()
        matched[in_range] = sorted_ids[pos[in_range]] == q[in_range]
        unmatched = int((~matched).sum())
        if unmatched:
            raise ValueError(f"{unmatched} query edge_ids missing from TF cache")
        out = features[order[pos]]
    return out.astype(np.float32), {
        "dense_indexable": bool(dense),
        "n_query": int(q.shape[0]),
        "n_unmatched": 0,
        "n_dropped": 0,
    }


def cmd_smoke(args: argparse.Namespace) -> int:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    CELLS.mkdir(parents=True, exist_ok=True)

    # 1) Focused unit tests
    test_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_temporal_flow_causal_features.py",
            "tests/test_paysim_temporal_flow_downstream.py",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    if test_proc.returncode != 0:
        write_json(
            SMOKE_JSON,
            {
                "ok": False,
                "stage": "pytest",
                "stdout": test_proc.stdout[-4000:],
                "stderr": test_proc.stderr[-4000:],
            },
        )
        print(test_proc.stdout)
        print(test_proc.stderr)
        return test_proc.returncode

    # 2) Timestamp multiplicity on full PaySim (Timestamp column only)
    import importlib.util

    build_mod_path = ROOT / "scripts" / "build_temporal_flow_causal_cache.py"
    spec = importlib.util.spec_from_file_location("build_tf_cache", build_mod_path)
    build_mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(build_mod)

    cfg = json.loads((ROOT / "data_config.json").read_text(encoding="utf-8"))
    csv_path = Path(cfg["paths"]["aml_data"]) / "PaySim" / "formatted_transactions.csv"
    ts = pd.read_csv(csv_path, usecols=["Timestamp"])["Timestamp"].to_numpy()
    # Match builder: re-zero timestamps for multiplicity of the featurization timeline
    ts = ts - ts.min()
    multiplicity = build_mod.timestamp_multiplicity_stats(ts)
    multiplicity["tie_policy_selected"] = build_mod.TIE_POLICY_ID
    multiplicity["tie_policy_description"] = build_mod.TIE_POLICY_DESCRIPTION
    multiplicity["blocker"] = False
    multiplicity["blocker_reason"] = None
    multiplicity["limitation"] = (
        "PaySim step timestamps are coarse: every edge shares its timestamp with others. "
        "Canonical AML policy B (simultaneous batch) avoids arbitrary within-step ordering; "
        "within-step transactions still cannot influence each other."
    )
    # Policy A would be a severe artifact; we do not use it.
    if multiplicity["fraction_edges_sharing_timestamp"] > 0.5:
        multiplicity["policy_A_would_be_severe_artifact"] = True
        multiplicity["proceed_with_policy_B"] = True
    write_json(MULTIPLICITY_JSON, multiplicity)

    # 3) Tiny causal cache on earliest timestamps
    smoke_root = SMOKE_CACHE_DIR.parent
    if smoke_root.exists():
        import shutil

        shutil.rmtree(smoke_root)
    build_mod.build_cache(
        "PaySim",
        str(ROOT / "data_config.json"),
        smoke_root,
        overwrite=True,
        max_timestamps=int(args.smoke_timestamps),
        write_test_split_files=False,
    )
    features, edge_id, meta = load_tf_cache(SMOKE_CACHE_DIR)
    # Deterministic rebuild on same subset via recompute
    df = pd.read_csv(csv_path)
    df["Timestamp"] = df["Timestamp"] - df["Timestamp"].min()
    uniq = np.sort(df["Timestamp"].unique())
    keep = set(uniq[: int(args.smoke_timestamps)].tolist())
    sub = df.loc[df["Timestamp"].isin(keep)].copy()
    # Preserve original edge ids as index into features via edge_id array order
    sub_edge_ids = sub.index.to_numpy(dtype=np.int64)
    sub = sub.reset_index(drop=True)
    amount_col = "Amount Received"
    feat2, _ = compute_temporal_flow_causal_features(sub, amount_col=amount_col)
    if not np.allclose(features, feat2, atol=0, rtol=0):
        raise SystemExit("smoke deterministic recompute mismatch")

    # Labels must not affect TF
    sub2 = sub.copy()
    sub2["Is Laundering"] = 1 - sub2["Is Laundering"].astype(int)
    feat3, _ = compute_temporal_flow_causal_features(sub2, amount_col=amount_col)
    if not np.array_equal(feat2, feat3):
        raise SystemExit("labels changed TF output — abort")

    # Unsorted input: shuffle rows then compute; compare after mapping back by edge identity
    # (canonical API assumes CSV row order identity; unsorted check uses same rows reordered)
    perm = np.random.RandomState(0).permutation(len(sub))
    shuffled = sub.iloc[perm].reset_index(drop=True)
    feat_shuf, _ = compute_temporal_flow_causal_features(shuffled, amount_col=amount_col)
    # After shuffle, features are in shuffled row order; unpermute
    feat_unperm = np.empty_like(feat_shuf)
    feat_unperm[perm] = feat_shuf
    if not np.allclose(feat_unperm, feat2):
        raise SystemExit("unsorted-input causality check failed")

    payload = {
        "ok": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pytest_ok": True,
        "pytest_summary": test_proc.stdout.strip().splitlines()[-1] if test_proc.stdout else "",
        "timestamp_multiplicity": multiplicity,
        "smoke_cache_dir": str(SMOKE_CACHE_DIR),
        "smoke_n_rows": int(features.shape[0]),
        "smoke_edge_id_sha256": _sha256_arr(edge_id),
        "smoke_features_sha256": _sha256_arr(features),
        "deterministic_rebuild_ok": True,
        "labels_do_not_affect_tf": True,
        "unsorted_input_ok": True,
        "tie_policy_id": meta["timestamp_handling"]["tie_policy_id"],
        "cache_version": meta.get("cache_version"),
        "smoke_split": meta.get("smoke_split"),
        "encoder_training": False,
        "validation_only": True,
        "test_evaluated": False,
    }
    write_json(SMOKE_JSON, payload)
    logging.info("Smoke OK: %s", SMOKE_JSON)
    return 0


def cmd_build_cache(args: argparse.Namespace) -> int:
    smoke = json.loads(SMOKE_JSON.read_text(encoding="utf-8")) if SMOKE_JSON.is_file() else None
    if not smoke or not smoke.get("ok"):
        raise SystemExit("smoke.json missing or not ok — refuse full cache build")
    if smoke.get("timestamp_multiplicity", {}).get("blocker"):
        raise SystemExit("timestamp multiplicity blocker set — refuse full cache")

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "build_tf_cache", ROOT / "scripts" / "build_temporal_flow_causal_cache.py"
    )
    build_mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(build_mod)

    out = build_mod.build_cache(
        "PaySim",
        str(ROOT / "data_config.json"),
        CACHE_ROOT,
        overwrite=bool(args.overwrite),
        max_timestamps=None,
        write_test_split_files=True,  # store for parity; Phase-1 ablation ignores test metrics
    )
    features, edge_id, meta = load_tf_cache(out)
    manifest = {
        "ok": True,
        "cache_dir": str(out),
        "cache_version": meta["cache_version"],
        "n_rows": int(features.shape[0]),
        "n_features": int(features.shape[1]),
        "feature_names": meta["feature_names"],
        "features_sha256": _sha256_arr(features),
        "edge_id_sha256": _sha256_arr(edge_id),
        "meta_path": str(out / "meta.json"),
        "tie_policy_id": meta["timestamp_handling"]["tie_policy_id"],
        "scaler_policy": meta["causal_history_policy"]["scaler_policy"],
        "labels_used_in_feature_construction": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(RESULT_ROOT / "cache_manifest.json", manifest)
    logging.info("Full PaySim TF cache ready: %s", out)
    return 0


def cmd_integrity(args: argparse.Namespace) -> int:
    features, edge_id, meta = load_tf_cache(CACHE_DIR)
    emb = verify_frozen_embeddings(EMB_DIR)
    arrays = emb.pop("_arrays")

    checks: Dict[str, Any] = {}
    checks["n_features_is_5"] = features.shape[1] == 5
    checks["feature_names_locked_order"] = meta.get("feature_names") == list(
        TEMPORAL_FLOW_CAUSAL_FEATURE_NAMES
    )
    checks["finite_values"] = bool(np.isfinite(features).all())
    checks["unique_edge_ids"] = int(np.unique(edge_id).shape[0]) == int(edge_id.shape[0])
    checks["dense_csv_indexable"] = bool(
        np.array_equal(edge_id, np.arange(edge_id.shape[0], dtype=np.int64))
    )
    checks["cache_version"] = meta.get("cache_version") == "temporal_flow_causal_paysim_v1"
    checks["uses_labels"] = meta.get("labels_used_in_feature_construction") is False
    checks["tie_policy_B"] = meta.get("timestamp_handling", {}).get("tie_policy_id") == (
        "B_simultaneous_batch_strictly_earlier_timestamps"
    )
    checks["scaler_deferred_train_only"] = "train_only" in str(
        meta.get("causal_history_policy", {}).get("scaler_policy", "")
    )

    join_info = {}
    for sp in ("train", "val"):
        ids = arrays[sp]["ids"]
        tf_sp, info = tf_lookup(features, edge_id, ids)
        join_info[sp] = {
            **info,
            "tf_sha256": _sha256_arr(tf_sp),
            "coverage_ok": info["n_unmatched"] == 0 and tf_sp.shape[0] == ids.shape[0],
        }
        checks[f"join_{sp}_ok"] = join_info[sp]["coverage_ok"]

    # Expected train/val sizes from locked P1
    checks["train_n_match"] = arrays["train"]["ids"].shape[0] == 3792812
    checks["val_n_match"] = arrays["val"]["ids"].shape[0] == 1276274
    checks["no_test_metrics_in_phase1"] = True

    # Past-only spot check: first chronological row has all-zero TF defaults
    # (may not be in train embed sample; check global min timestamp row)
    # Recompute vs cache on a small random sample of rows
    cfg = json.loads((ROOT / "data_config.json").read_text(encoding="utf-8"))
    csv_path = Path(cfg["paths"]["aml_data"]) / "PaySim" / "formatted_transactions.csv"
    # Sample-based recompute is expensive on full CSV; instead verify cache self-hash + meta
    checks["features_sha256_matches_meta"] = (
        meta.get("code_metadata", {}).get("features_sha256") == _sha256_arr(features)
    )

    bool_keys = [k for k, v in checks.items() if isinstance(v, bool)]
    ok = all(bool(checks[k]) for k in bool_keys)

    payload = {
        "ok": ok,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "join_info": join_info,
        "frozen_embeddings": emb,
        "cache_meta_summary": {
            "cache_version": meta.get("cache_version"),
            "n_rows": meta.get("n_rows"),
            "tie_policy_id": meta.get("timestamp_handling", {}).get("tie_policy_id"),
            "timestamp_multiplicity": meta.get("timestamp_multiplicity"),
        },
        "encoder_training": False,
        "encoder_frozen": True,
        "validation_only": True,
        "test_evaluated": False,
    }
    write_json(INTEGRITY_JSON, payload)
    if not ok:
        raise SystemExit(f"integrity failed: {json.dumps(checks, indent=2)}")
    logging.info("Integrity OK")
    return 0


def _stack_matrix(
    name: str,
    z: Optional[np.ndarray],
    x: Optional[np.ndarray],
    tf: Optional[np.ndarray],
) -> np.ndarray:
    # Support optional "random_" prefix for the secondary control stack.
    core = name[len("random_") :] if name.startswith("random_") else name
    include_x = core in ("X", "X+TF", "H+X", "H+X+TF")
    include_tf = core in ("TF", "X+TF", "H+TF", "H+X+TF")
    include_h = core in ("H", "H+X", "H+TF", "H+X+TF")
    blocks: List[np.ndarray] = []
    if include_h:
        assert z is not None
        blocks.append(z.astype(np.float32))
    if include_x:
        assert x is not None
        blocks.append(x.astype(np.float32))
    if include_tf:
        assert tf is not None
        blocks.append(tf.astype(np.float32))
    if not blocks:
        raise ValueError(f"empty stack {name}")
    return np.concatenate(blocks, axis=1).astype(np.float32)


def run_one_stack(
    name: str,
    x_tr: Optional[np.ndarray],
    x_va: Optional[np.ndarray],
    tf_tr: Optional[np.ndarray],
    tf_va: Optional[np.ndarray],
    z_tr: Optional[np.ndarray],
    z_va: Optional[np.ndarray],
    y_tr: np.ndarray,
    y_va: np.ndarray,
    ids_tr: np.ndarray,
    ids_va: np.ndarray,
) -> Dict[str, Any]:
    mat_tr = _stack_matrix(name, z_tr, x_tr, tf_tr)
    mat_va = _stack_matrix(name, z_va, x_va, tf_va)
    scaler = StandardScaler()
    tr_s = scaler.fit_transform(mat_tr).astype(np.float32)
    va_s = scaler.transform(mat_va).astype(np.float32)
    cw = gin_model_class_weight()
    set_seed(DOWNSTREAM_SEED)
    clf = LogisticRegression(
        class_weight=cw,
        max_iter=1000,
        random_state=DOWNSTREAM_SEED,
        solver="lbfgs",
        n_jobs=1,
        C=1.0,
    )
    clf.fit(tr_s, y_tr)
    proba_tr = clf.predict_proba(tr_s)[:, 1].astype(np.float64)
    proba_va = clf.predict_proba(va_s)[:, 1].astype(np.float64)
    thr = tune_thr_max_f1(y_va, proba_va)
    cell = {
        "stack": name,
        "feature_dim": int(mat_tr.shape[1]),
        "feature_matrix_sha256_train": _sha256_arr(mat_tr),
        "feature_matrix_sha256_val": _sha256_arr(mat_va),
        "scaled_train_sha256": _sha256_arr(tr_s),
        "learner": "LogisticRegression",
        "class_weight_mode": "model",
        "class_weight": {str(k): float(v) for k, v in cw.items()},
        "C": 1.0,
        "downstream_seed": DOWNSTREAM_SEED,
        "scaler": "StandardScaler_fit_train_only",
        "ids": {"train": ids_hash(ids_tr), "val": ids_hash(ids_va)},
        "coverage": {
            "train": {
                "n": int(y_tr.shape[0]),
                "n_positives": int(y_tr.sum()),
                "positive_rate": float(y_tr.mean()),
            },
            "val": {
                "n": int(y_va.shape[0]),
                "n_positives": int(y_va.sum()),
                "positive_rate": float(y_va.mean()),
            },
        },
        "validation": {
            "threshold_0.5": metrics_block(y_va, proba_va, 0.5),
            "threshold_val_selected_max_f1": metrics_block(y_va, proba_va, thr),
            "validation_selected_threshold": thr,
            "threshold_provenance": "max_f1_on_paysim_validation_only_diagnostic",
        },
        "train_fit_diagnostics": {
            "threshold_0.5": metrics_block(y_tr, proba_tr, 0.5),
        },
        "test_evaluated": False,
        "encoder_training": False,
        "encoder_frozen": True,
        "validation_only": True,
        "exploratory_posthoc": True,
        "table_eligible": False,
    }
    write_json(CELLS / f"seed2_{name.replace('+', 'plus')}_validation.json", cell)
    return cell


def cmd_ablation(args: argparse.Namespace) -> int:
    integ = json.loads(INTEGRITY_JSON.read_text(encoding="utf-8"))
    if not integ.get("ok"):
        raise SystemExit("integrity.json not ok — refuse ablation")

    emb = verify_frozen_embeddings(EMB_DIR)
    arrays = emb.pop("_arrays")
    features, edge_id, meta = load_tf_cache(CACHE_DIR)
    x_raw, x_names, x_meta = load_x_matrix()

    ids_tr = arrays["train"]["ids"]
    ids_va = arrays["val"]["ids"]
    y_tr = arrays["train"]["y"]
    y_va = arrays["val"]["y"]
    z_tr = arrays["train"]["Z"]
    z_va = arrays["val"]["Z"]

    # Intersected cohort: embeddings define the cohort; require full TF/X coverage
    tf_tr, join_tr = tf_lookup(features, edge_id, ids_tr)
    tf_va, join_va = tf_lookup(features, edge_id, ids_va)
    if ids_tr.max() >= x_raw.shape[0] or ids_va.max() >= x_raw.shape[0]:
        raise SystemExit("X matrix too short for embedding edge_ids")
    x_tr = x_raw[ids_tr]
    x_va = x_raw[ids_va]

    # Label consistency vs CSV
    cfg = json.loads((ROOT / "data_config.json").read_text(encoding="utf-8"))
    csv_path = Path(cfg["paths"]["aml_data"]) / "PaySim" / "formatted_transactions.csv"
    y_all = pd.read_csv(csv_path, usecols=["Is Laundering"])["Is Laundering"].to_numpy(dtype=np.int64)
    if not np.array_equal(y_tr, y_all[ids_tr]) or not np.array_equal(y_va, y_all[ids_va]):
        raise SystemExit("label mismatch vs CSV for embedding edge_ids")

    cells = {}
    for name in STACKS:
        logging.info("Fitting stack %s", name)
        cells[name] = run_one_stack(
            name,
            x_tr=x_tr,
            x_va=x_va,
            tf_tr=tf_tr,
            tf_va=tf_va,
            z_tr=z_tr,
            z_va=z_va,
            y_tr=y_tr,
            y_va=y_va,
            ids_tr=ids_tr,
            ids_va=ids_va,
        )

    random_cell = None
    if RANDOM_EMB_DIR.is_dir() and (RANDOM_EMB_DIR / "train.npz").is_file():
        rz_tr, ry_tr, rid_tr = load_embedding_npz(RANDOM_EMB_DIR / "train.npz")
        rz_va, ry_va, rid_va = load_embedding_npz(RANDOM_EMB_DIR / "val.npz")
        if np.array_equal(rid_tr, ids_tr) and np.array_equal(rid_va, ids_va):
            if not np.array_equal(ry_tr, y_tr) or not np.array_equal(ry_va, y_va):
                raise SystemExit("random control label mismatch")
            logging.info("Fitting secondary control random_H+X+TF")
            random_cell = run_one_stack(
                "random_H+X+TF",
                x_tr=x_tr,
                x_va=x_va,
                tf_tr=tf_tr,
                tf_va=tf_va,
                z_tr=rz_tr,
                z_va=rz_va,
                y_tr=y_tr,
                y_va=y_va,
                ids_tr=ids_tr,
                ids_va=ids_va,
            )
        else:
            logging.warning("random control edge_ids differ — skip random_H+X+TF")

    def auprc(stack: str) -> float:
        return float(cells[stack]["validation"]["threshold_0.5"]["auprc"])

    gate = {
        "margin_abs": GATE_MARGIN,
        "hxxtf_vs_hx": auprc("H+X+TF") - auprc("H+X"),
        "hxxtf_vs_xtf": auprc("H+X+TF") - auprc("X+TF"),
        "hxxtf_vs_h": auprc("H+X+TF") - auprc("H"),
        "requires": {
            "hxxtf_beats_hx_by_margin": (auprc("H+X+TF") - auprc("H+X")) >= GATE_MARGIN,
            "hxxtf_beats_xtf_by_margin": (auprc("H+X+TF") - auprc("X+TF")) >= GATE_MARGIN,
            "hxxtf_beats_h": auprc("H+X+TF") > auprc("H"),
            "xtf_reported": True,
        },
    }
    if random_cell is not None:
        rand_a = float(random_cell["validation"]["threshold_0.5"]["auprc"])
        gate["hxxtf_vs_random_hxxtf"] = auprc("H+X+TF") - rand_a
        gate["requires"]["pretrained_beats_random_hxxtf"] = auprc("H+X+TF") > rand_a
    gate["pass"] = all(gate["requires"].values())
    gate["central_transfer_criterion"] = "H+X+TF > X+TF (with margin)"
    gate["central_pass"] = gate["requires"]["hxxtf_beats_xtf_by_margin"]

    winner = max(STACKS, key=lambda s: auprc(s))
    summary = {
        "ok": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "encoder_training": False,
        "encoder_frozen": True,
        "validation_only": True,
        "test_evaluated": False,
        "exploratory_posthoc": True,
        "table_eligible_until_confirmation": False,
        "seed": LOCKED_SEED,
        "downstream_seed": DOWNSTREAM_SEED,
        "feature_contract_id": FEATURE_CONTRACT,
        "bn_protocol": BN_PROTOCOL,
        "normalization_protocol": NORM_PROTOCOL,
        "x_meta": {k: v for k, v in x_meta.items() if k != "meta"},
        "x_feature_names": x_names,
        "tf_cache_version": meta.get("cache_version"),
        "join": {"train": join_tr, "val": join_va},
        "stacks": {k: {
            "feature_dim": cells[k]["feature_dim"],
            "val_auprc_at_0.5": cells[k]["validation"]["threshold_0.5"]["auprc"],
            "val_auroc_at_0.5": cells[k]["validation"]["threshold_0.5"]["auroc"],
            "val_f1_at_0.5": cells[k]["validation"]["threshold_0.5"]["f1"],
            "val_f1_at_val_selected": cells[k]["validation"]["threshold_val_selected_max_f1"]["f1"],
            "cell": str(CELLS / f"seed2_{k.replace('+', 'plus')}_validation.json"),
        } for k in STACKS},
        "random_H+X+TF": None if random_cell is None else {
            "val_auprc_at_0.5": random_cell["validation"]["threshold_0.5"]["auprc"],
            "feature_dim": random_cell["feature_dim"],
        },
        "winner_val_auprc_at_0.5": winner,
        "gate": gate,
        "frozen_embeddings": emb,
    }
    write_json(ABLATION_JSON, summary)
    logging.info("Ablation done. gate.pass=%s winner=%s", gate["pass"], winner)
    return 0


def cmd_aggregate(args: argparse.Namespace) -> int:
    abl = json.loads(ABLATION_JSON.read_text(encoding="utf-8"))
    integ = json.loads(INTEGRITY_JSON.read_text(encoding="utf-8"))
    smoke = json.loads(SMOKE_JSON.read_text(encoding="utf-8")) if SMOKE_JSON.is_file() else {}
    manifest = json.loads((RESULT_ROOT / "cache_manifest.json").read_text(encoding="utf-8"))

    stacks = abl["stacks"]
    gate = abl["gate"]

    def a(s: str) -> float:
        return float(stacks[s]["val_auprc_at_0.5"])

    answers = {
        "1_did_tf_improve_hx": a("H+X+TF") > a("H+X"),
        "1_delta_auprc": a("H+X+TF") - a("H+X"),
        "2_did_h_improve_xtf": a("H+X+TF") > a("X+TF"),
        "2_delta_auprc": a("H+X+TF") - a("X+TF"),
        "3_winner_val_auprc": abl["winner_val_auprc_at_0.5"],
        "4_predeclared_gate_pass": bool(gate["pass"]),
        "5_attribution": (
            "both"
            if (a("H+X+TF") > a("H+X") and a("H+X+TF") > a("X+TF"))
            else ("TF" if a("H+X+TF") > a("H+X") else ("transferred_H" if a("H+X+TF") > a("X+TF") else "neither"))
        ),
        "6_multiseed_test_confirmation_justified": bool(gate["pass"]),
        "7_followup_jobs_auto_submitted": False,
    }

    out = {
        "title": "paysim_temporal_flow_downstream_validation",
        "phase": 1,
        "encoder_training": False,
        "encoder_frozen": True,
        "validation_only": True,
        "test_evaluated": False,
        "exploratory_posthoc": True,
        "table_eligible_until_confirmation": True,
        "table_eligible": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "smoke": {"ok": smoke.get("ok"), "path": str(SMOKE_JSON)},
        "cache_manifest": manifest,
        "integrity": {"ok": integ.get("ok"), "path": str(INTEGRITY_JSON)},
        "ablation": abl,
        "answers": answers,
        "paths": {
            "cache_dir": str(CACHE_DIR),
            "cells": str(CELLS),
            "notes": str(NOTES_MD),
            "json": str(OUT_JSON),
        },
    }
    write_json(OUT_JSON, out)

    lines = [
        "# PaySim temporal-flow downstream validation (Phase 1)",
        "",
        "> Validation-only. Encoder frozen. No test metrics. Exploratory/post-hoc.",
        f"> Twin: `{OUT_JSON.relative_to(ROOT)}`",
        "",
        "## Status flags",
        "",
        "- `encoder_training=false`",
        "- `encoder_frozen=true`",
        "- `validation_only=true`",
        "- `test_evaluated=false`",
        "- `exploratory_posthoc=true`",
        "- `table_eligible=false` until confirmation",
        "",
        "## Frozen representation",
        "",
        f"- Checkpoint: `{SOURCE_UNIQUE}.tar`",
        f"- SHA256: `{SOURCE_SHA256}`",
        f"- Embeddings: `{EMB_DIR.relative_to(ROOT)}`",
        f"- Contract: `{FEATURE_CONTRACT}`",
        f"- BN: `{BN_PROTOCOL}`",
        f"- Norm: `{NORM_PROTOCOL}`",
        f"- H: post-128",
        "",
        "## TF cache",
        "",
        f"- Dir: `{CACHE_DIR.relative_to(ROOT)}`",
        f"- Version: `{manifest.get('cache_version')}`",
        f"- Features SHA256: `{manifest.get('features_sha256')}`",
        f"- Tie policy: B simultaneous batch (strictly earlier timestamps)",
        "",
        "## Validation AUPRC @ 0.5 (seed-2 logistic)",
        "",
        "| Stack | Dim | Val AUPRC | Val AUROC | Val F1@0.5 |",
        "|-------|-----|-----------|-----------|------------|",
    ]
    for s in STACKS:
        st = stacks[s]
        lines.append(
            f"| {s} | {st['feature_dim']} | {st['val_auprc_at_0.5']:.6f} | "
            f"{st['val_auroc_at_0.5']:.6f} | {st['val_f1_at_0.5']:.6f} |"
        )
    if abl.get("random_H+X+TF"):
        r = abl["random_H+X+TF"]
        lines.append(
            f"| random_H+X+TF (control) | {r['feature_dim']} | {r['val_auprc_at_0.5']:.6f} | — | — |"
        )

    lines += [
        "",
        "## Predeclared gate",
        "",
        f"- Margin: {GATE_MARGIN}",
        f"- H+X+TF − H+X = {gate['hxxtf_vs_hx']:.6f} (need ≥ {GATE_MARGIN}): "
        f"{'PASS' if gate['requires']['hxxtf_beats_hx_by_margin'] else 'FAIL'}",
        f"- H+X+TF − X+TF = {gate['hxxtf_vs_xtf']:.6f} (need ≥ {GATE_MARGIN}): "
        f"{'PASS' if gate['requires']['hxxtf_beats_xtf_by_margin'] else 'FAIL'}",
        f"- H+X+TF > H: {'PASS' if gate['requires']['hxxtf_beats_h'] else 'FAIL'}",
        f"- Gate overall: **{'PASS' if gate['pass'] else 'FAIL'}**",
        f"- Central transfer criterion (H+X+TF > X+TF w/ margin): "
        f"{'PASS' if gate['central_pass'] else 'FAIL'}",
        "",
        "## Exact answers",
        "",
        f"1. Did TF improve H+X? **{answers['1_did_tf_improve_hx']}** (Δ={answers['1_delta_auprc']:.6f})",
        f"2. Did H improve X+TF? **{answers['2_did_h_improve_xtf']}** (Δ={answers['2_delta_auprc']:.6f})",
        f"3. Winner (val AUPRC): **{answers['3_winner_val_auprc']}**",
        f"4. Predeclared gate pass? **{answers['4_predeclared_gate_pass']}**",
        f"5. Attribution: **{answers['5_attribution']}**",
        f"6. Multiseed/test confirmation justified? **{answers['6_multiseed_test_confirmation_justified']}**",
        f"7. Follow-up jobs auto-submitted? **{answers['7_followup_jobs_auto_submitted']}**",
        "",
    ]
    NOTES_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logging.info("Wrote %s and %s", NOTES_MD, OUT_JSON)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sm = sub.add_parser("smoke")
    sm.add_argument("--smoke_timestamps", type=int, default=8)

    bc = sub.add_parser("build_cache")
    bc.add_argument("--overwrite", action="store_true")

    sub.add_parser("integrity")
    sub.add_parser("ablation")
    sub.add_parser("aggregate")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    logger_setup()
    args = build_parser().parse_args(argv)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    CELLS.mkdir(parents=True, exist_ok=True)
    if args.cmd == "smoke":
        return cmd_smoke(args)
    if args.cmd == "build_cache":
        return cmd_build_cache(args)
    if args.cmd == "integrity":
        return cmd_integrity(args)
    if args.cmd == "ablation":
        return cmd_ablation(args)
    if args.cmd == "aggregate":
        return cmd_aggregate(args)
    raise SystemExit(f"unknown cmd {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
