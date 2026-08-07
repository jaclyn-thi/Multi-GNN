#!/usr/bin/env python3
"""Phase-1 SAML-D TF cache + integrity for smallhi_samld_shared_core_v1 (CPU).

Wraps ``scripts/build_temporal_flow_causal_cache.py`` with SAML-D defaults:
unique cache root, train/val only, no test split files, MoE train scaler,
and a completion JSON with integrity gates.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from util import logger_setup  # noqa: E402


def _load_builder():
    path = _ROOT / "scripts" / "build_temporal_flow_causal_cache.py"
    spec = importlib.util.spec_from_file_location("build_tf_cache_phase1", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def integrity_report(cache_dir: Path, builder) -> Dict[str, Any]:
    cache_dir = Path(cache_dir)
    meta = json.loads((cache_dir / "meta.json").read_text(encoding="utf-8"))
    feat = np.load(cache_dir / "features.npy")
    edge_id = np.load(cache_dir / "edge_id.npy")
    tr = np.load(cache_dir / "split_train_edge_id.npy")
    va = np.load(cache_dir / "split_val_edge_id.npy")
    test_exists = (cache_dir / "split_test_edge_id.npy").is_file()
    scaler_path = cache_dir / "moe_target_train_scaler.json"
    moe = json.loads(scaler_path.read_text(encoding="utf-8")) if scaler_path.is_file() else {}
    name_to_i = {n: i for i, n in enumerate(meta["feature_names"])}
    moe_idx = [name_to_i[n] for n in builder.MOE_TARGET_NAMES]
    moe_cols = feat[:, moe_idx]
    report: Dict[str, Any] = {
        "cache_dir": str(cache_dir),
        "cache_version": meta.get("cache_version"),
        "feature_contract_id": meta.get("feature_contract_id"),
        "test_split_present": test_exists,
        "n_rows": int(feat.shape[0]),
        "n_train": int(tr.shape[0]),
        "n_val": int(va.shape[0]),
        "n_pos_train": meta.get("prevalence", {}).get("n_pos_train"),
        "n_pos_val": meta.get("prevalence", {}).get("n_pos_val"),
        "unique_edge_ids": bool(len(set(edge_id.tolist())) == edge_id.size),
        "train_val_disjoint": bool(len(set(tr.tolist()) & set(va.tolist())) == 0),
        "finite_features": bool(np.isfinite(feat).all()),
        "moe_shape": [int(moe_cols.shape[0]), int(moe_cols.shape[1])],
        "moe_finite": bool(np.isfinite(moe_cols).all()),
        "target_coverage": float(np.isfinite(moe_cols).mean()),
        "causal_past_only": bool(meta.get("causal_history_policy", {}).get("past_only")),
        "uses_labels": bool(meta.get("labels_used_in_feature_construction")),
        "moe_scaler_sha256": moe.get("scaler_sha256")
        or meta.get("moe_target_train_scaler", {}).get("scaler_sha256"),
        "reuses_small_hi_statistics": bool(meta.get("reuses_small_hi_target_statistics")),
        "tie_policy_id": meta.get("timestamp_handling", {}).get("tie_policy_id"),
        "coverage": meta.get("coverage"),
    }
    gates = {
        "version_ok": report["cache_version"] == builder.CACHE_VERSION_SAMLD_SHARED_CORE,
        "no_test_split": not report["test_split_present"],
        "unique_ids": report["unique_edge_ids"],
        "disjoint": report["train_val_disjoint"],
        "finite": report["finite_features"],
        "moe_width_3": report["moe_shape"][1] == 3,
        "moe_finite": report["moe_finite"],
        "causal": report["causal_past_only"],
        "label_free": not report["uses_labels"],
        "scaler_dataset_specific": not report["reuses_small_hi_statistics"],
        "scaler_present": bool(report["moe_scaler_sha256"]),
        "contract_id": report["feature_contract_id"] == "smallhi_samld_shared_core_v1",
    }
    report["gates"] = gates
    report["ok"] = all(gates.values())
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    builder = _load_builder()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_config", default="data_config.json")
    p.add_argument(
        "--cache_root",
        default=builder.DEFAULT_CACHE_ROOT_SAMLD_SHARED_CORE,
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--max_timestamps", type=int, default=None)
    p.add_argument(
        "--completion_json",
        default="results/diagnostics/small_hi_samld_shared_core_phase1_cache_completion.json",
    )
    args = p.parse_args(argv)
    logger_setup()
    out = builder.build_cache(
        "SAML-D",
        args.data_config,
        Path(args.cache_root),
        overwrite=bool(args.overwrite),
        max_timestamps=args.max_timestamps,
        write_test_split_files=False,
        train_val_only=True,
    )
    report = integrity_report(out, builder)
    completion = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "account": os.environ.get("SLURM_JOB_ACCOUNT"),
        "qos": os.environ.get("SLURM_JOB_QOS"),
        "cache_dir": str(out),
        "integrity": report,
        "test_data_accessed": False,
        "encoder_trained": False,
        "embeddings_extracted": False,
        "probes_fit": False,
        "mixed_trainer": False,
        "category_adapter": False,
    }
    Path(args.completion_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.completion_json).write_text(
        json.dumps(completion, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(completion, indent=2))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
