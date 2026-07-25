#!/usr/bin/env python3
"""Bounded end-to-end preflight for the forensic GCPAL eval-protocol pipeline.

Exercises the same branches/schemas as the full Slurm forensic job on a
deterministic subset (max_batches extraction + forensic --smoke). Does not
change scientific settings of the full run.
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch

from linear_probe import load_embedding_npz, tune_threshold_max_f1
from train_util import infer_pre_embedding_dim, log_seed_coverage, resolve_embedding_head_linear


RUN = "gin_emlps_ports_tds_corrected_preserve_seed_asym_proj_8192neg_queue0_40ep_seed2"
GRAPH = (
    "--reverse_mp --ego --ports --emlps --tds --correct_reverse_edge_features"
)
COMMON = (
    f"--data Small-HI --model gin --tqdm --batch_size 8192 "
    f"--num_neighs 100 100 --loader_num_workers 0 --seed 2"
)


def _check_api_contracts() -> dict:
    """Regression guards for both known forensic failures."""
    sig_infer = inspect.signature(infer_pre_embedding_dim)
    params = list(sig_infer.parameters)
    assert params == ["model", "emb_dim"], params

    # Historical failure #1: positional int only.
    try:
        infer_pre_embedding_dim(64)  # type: ignore[arg-type]
        raise AssertionError("infer_pre_embedding_dim(64) should TypeError")
    except TypeError:
        pass

    sig_cov = inspect.signature(log_seed_coverage)
    assert "split_name" in sig_cov.parameters
    assert "split" not in sig_cov.parameters

    a = torch.tensor([1, 2, 3])
    b = torch.tensor([1, 2, 3, 4])
    # Historical failure #2: kwarg split=
    try:
        log_seed_coverage(a, b, split="fullgraph_all")  # type: ignore[call-arg]
        raise AssertionError("log_seed_coverage(..., split=) should TypeError")
    except TypeError:
        pass
    log_seed_coverage(a, b, split_name="fullgraph_all")

    # Source audit: extract script must not use split=
    src = (_ROOT / "scripts/extract_fullgraph_embeddings_diagnostic.py").read_text()
    assert 'log_seed_coverage(edge_ids, expected, split="' not in src
    assert "split_name=" in src
    assert "infer_pre_embedding_dim(model, emb_dim)" in src
    assert "infer_pre_embedding_dim(actual_n_hidden)" not in src

    return {
        "infer_pre_embedding_dim_params": params,
        "log_seed_coverage_params": list(sig_cov.parameters),
        "source_audit_ok": True,
    }


def _run(cmd: list, env_extra: Optional[Dict[str, str]] = None) -> None:
    logging.info("RUN: %s", " ".join(cmd))
    env = None
    if env_extra:
        import os

        env = os.environ.copy()
        env.update(env_extra)
    subprocess.run(cmd, check=True, cwd=str(_ROOT), env=env)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_batches", type=int, default=2)
    p.add_argument(
        "--preflight_root",
        default="results/diagnostics/forensic_preflight",
    )
    p.add_argument("--python", default=sys.executable)
    args = p.parse_args()

    t0 = time.perf_counter()
    root = Path(args.preflight_root)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    fg_name = "forensic_preflight_fullgraph_post"
    fg_pre_name = "forensic_preflight_fullgraph_pre3h"
    # Clean prior preflight extract dirs if present
    for name in (fg_name, fg_pre_name):
        d = _ROOT / "embeddings" / name
        if d.exists():
            shutil.rmtree(d)

    report: dict = {"stages": {}}

    # --- Stage: API contracts ---
    report["stages"]["api_contracts"] = _check_api_contracts()
    logging.info("API contracts OK")

    # --- Stage: checkpoint + post-embedding extract (bounded) ---
    py = args.python
    _run(
        [
            py,
            "scripts/extract_fullgraph_embeddings_diagnostic.py",
            *COMMON.split(),
            *GRAPH.split(),
            "--unique_name",
            RUN,
            "--output_unique_name",
            fg_name,
            "--representation_source",
            "post_embedding",
            "--device",
            args.device,
            "--max_batches",
            str(int(args.max_batches)),
        ]
    )
    post_npz = _ROOT / "embeddings" / fg_name / "all.npz"
    assert post_npz.is_file(), post_npz
    z, y, e = load_embedding_npz(post_npz)
    assert z.ndim == 2 and z.shape[1] == 128, z.shape
    assert y.shape[0] == z.shape[0] == e.shape[0]
    report["stages"]["post_embedding_extract"] = {
        "path": str(post_npz),
        "n": int(z.shape[0]),
        "dim": int(z.shape[1]),
        "n_pos": int(y.sum()),
        "schema": ["Z", "y", "edge_id"],
    }
    logging.info("Post-embedding extract OK: n=%d dim=%d", z.shape[0], z.shape[1])

    # --- Stage: pre-3h extract (bounded) ---
    _run(
        [
            py,
            "scripts/extract_fullgraph_embeddings_diagnostic.py",
            *COMMON.split(),
            *GRAPH.split(),
            "--unique_name",
            RUN,
            "--output_unique_name",
            fg_pre_name,
            "--representation_source",
            "pre_embedding_3h",
            "--device",
            args.device,
            "--max_batches",
            str(int(args.max_batches)),
        ]
    )
    # pre_embedding writes under output/pre_embedding_3h/all.npz
    pre_npz = _ROOT / "embeddings" / fg_pre_name / "pre_embedding_3h" / "all.npz"
    assert pre_npz.is_file(), pre_npz
    zp, yp, ep = load_embedding_npz(pre_npz)
    assert zp.shape[1] == 198, zp.shape  # 3 * 66 for gin D
    report["stages"]["pre_embedding_3h_extract"] = {
        "path": str(pre_npz),
        "n": int(zp.shape[0]),
        "dim": int(zp.shape[1]),
    }
    logging.info("Pre-3h extract OK: n=%d dim=%d", zp.shape[0], zp.shape[1])

    # --- Stage: raw/temporal assets exist ---
    temporal_dir = _ROOT / "embeddings" / RUN
    for split in ("train", "val", "test"):
        assert (temporal_dir / f"{split}.npz").is_file()
    pre3h_dir = temporal_dir / "pre_embedding_3h"
    assert (pre3h_dir / "train.npz").is_file()
    tf_dir = _ROOT / "results/cache/temporal_flow_causal/Small-HI"
    assert (tf_dir / "features.npy").is_file() and (tf_dir / "edge_id.npy").is_file()
    report["stages"]["assets"] = {
        "temporal_post": str(temporal_dir),
        "temporal_pre3h": str(pre3h_dir),
        "tf_cache": str(tf_dir),
        "checkpoint_meta": str(temporal_dir / "meta.json"),
    }

    # --- Stage: threshold helper schema ---
    rng = np.random.RandomState(0)
    proba = rng.rand(200).astype(np.float64)
    yb = (rng.rand(200) > 0.9).astype(np.int64)
    thr, vf1 = tune_threshold_max_f1(yb, proba)
    report["stages"]["threshold_selection"] = {"threshold": float(thr), "val_f1": float(vf1)}

    # --- Stage: forensic classifier matrix (smoke) ---
    out_json = root / "forensic_preflight_smoke.json"
    out_md = root / "forensic_preflight_smoke.md"
    _run(
        [
            py,
            "scripts/forensic_gcpal_eval_protocol_audit.py",
            "--embedding_dir",
            str(temporal_dir),
            "--fullgraph_embedding_dir",
            str(_ROOT / "embeddings" / fg_name),
            "--device",
            args.device,
            "--smoke",
            "--smoke_n",
            "2000",
            "--output_json",
            str(out_json),
            "--output_md",
            str(out_md),
        ]
    )
    assert out_json.is_file() and out_md.is_file()
    payload = json.loads(out_json.read_text())
    assert "temporal_protocol" in payload
    assert "random_protocol_40" in payload
    assert "random_protocol_60" in payload
    assert "decision" in payload
    assert "provenance" in payload
    # Markdown serialization non-empty
    assert len(out_md.read_text()) > 100
    report["stages"]["forensic_smoke"] = {
        "output_json": str(out_json),
        "output_md": str(out_md),
        "decision_verdict": payload["decision"].get("verdict"),
        "has_temporal": True,
        "has_random40": "pooled_over_split_seeds" in payload["random_protocol_40"]
        or "skipped" not in payload["random_protocol_40"],
        "smoke": payload["provenance"].get("smoke"),
    }

    report["ok"] = True
    report["elapsed_seconds"] = time.perf_counter() - t0
    report_path = root / "preflight_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    logging.info("PREFLIGHT PASS in %.1fs → %s", report["elapsed_seconds"], report_path)
    print(json.dumps(report, indent=2))
    print(report_path)


if __name__ == "__main__":
    main()
