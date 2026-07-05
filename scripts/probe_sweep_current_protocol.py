#!/usr/bin/env python3
"""CPU probe sweep over class weights and regularization C (current-protocol comparison)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.probe_feature_ablation import (  # noqa: E402
    FEATURE_MODES,
    load_dataset_frames,
    run_probe_for_mode,
)
from util import logger_setup, set_seed  # noqa: E402

DEFAULT_C_GRID = (0.01, 0.1, 1.0, 10.0)
DEFAULT_CLASS_WEIGHTS = ("model", "none", "balanced")
SSL_FEATURE_MODES = ("embedding", "embedding+raw", "embedding+raw+morph")
BASELINE_FEATURE_MODE = "raw+morph"

RUNS = (
    {
        "run_label": "GINe emlps+tds seed1 (20ep)",
        "run_name": "gin_20ep_seed1",
        "embedding_dir": "embeddings/hi_contrastive_gin_emlps_tds_proj_asym_8192neg_queue0_accum4_20ep",
    },
    {
        "run_label": "GINe emlps+tds seed1 (40ep)",
        "run_name": "gin_40ep_seed1",
        "embedding_dir": "embeddings/gin_emlps_tds_asym_proj_8192neg_queue0_40ep_seed1",
    },
    {
        "run_label": "GINe emlps+tds seed2 (40ep)",
        "run_name": "gin_40ep_seed2",
        "embedding_dir": "embeddings/gin_emlps_tds_asym_proj_8192neg_queue0_40ep_seed2",
    },
    {
        "run_label": "FNF + emlps+tds seed1",
        "run_name": "fnf_seed1",
        "embedding_dir": "embeddings/same_pair_fnf_emlps_tds_asym_proj_8192neg_queue0_20ep",
    },
    {
        "run_label": "FNF + emlps+tds seed2",
        "run_name": "fnf_seed2",
        "embedding_dir": "embeddings/fnf_emlps_tds_asym_proj_8192neg_queue0_20ep_seed2",
    },
)

PROTOCOL = {
    "description": "Current-protocol CPU probe sweep (frozen embeddings only; no SSL retrain).",
    "threshold_tuning": "max_f1_on_val",
    "class_weight_modes": list(DEFAULT_CLASS_WEIGHTS),
    "class_weight_notes": {
        "model": "Shared GIN weights via --class_weight model --model gin",
        "none": "No class weighting",
        "balanced": "Exploratory sklearn balanced; may be extreme under rare positives",
    },
    "probe_C_grid": list(DEFAULT_C_GRID),
    "probe_C_note": "sklearn LogisticRegression C (inverse regularization; default reference 1.0)",
    "ssl_feature_modes": list(SSL_FEATURE_MODES),
    "baseline_feature_mode": BASELINE_FEATURE_MODE,
}


def _make_args(
    *,
    class_weight: str,
    probe_c: float,
    data: str,
    data_config: str,
    categorical_encoding: str,
    probe_max_iter: int,
    probe_n_jobs: int,
    seed: int,
) -> Namespace:
    return Namespace(
        data=data,
        data_config=data_config,
        class_weight=class_weight,
        model="gin",
        probe_C=probe_c,
        probe_max_iter=probe_max_iter,
        probe_n_jobs=probe_n_jobs,
        seed=seed,
        categorical_encoding=categorical_encoding,
        report_all_splits=True,
    )


def _flatten_row(
    *,
    run_spec: Dict[str, str],
    features: str,
    class_weight: str,
    probe_c: float,
    row: Dict[str, Any],
) -> Dict[str, Any]:
    test = row["test"]
    t05 = row["test_at_threshold_0.5"]
    out: Dict[str, Any] = {
        "run_label": run_spec["run_label"],
        "run_name": run_spec["run_name"],
        "embedding_dir": run_spec["embedding_dir"],
        "features": features,
        "class_weight_mode": class_weight,
        "class_weight": row.get("class_weight"),
        "probe_C": probe_c,
        "threshold": test.get("threshold"),
        "val_f1_at_selected_threshold": row.get("val_f1_at_selected_threshold"),
        "test_auroc": test.get("auroc"),
        "test_auprc": test.get("auprc"),
        "test_f1": test.get("f1"),
        "test_precision": test.get("precision"),
        "test_recall": test.get("recall"),
        "test_f1_at_0_5": t05.get("f1"),
    }
    for split in ("train", "val"):
        if split in row and row[split]:
            s = row[split]
            out[f"{split}_auroc"] = s.get("auroc")
            out[f"{split}_auprc"] = s.get("auprc")
            out[f"{split}_f1"] = s.get("f1")
            out[f"{split}_precision"] = s.get("precision")
            out[f"{split}_recall"] = s.get("recall")
            out[f"{split}_f1_at_0_5"] = s.get("f1_at_0_5")
    return out


def run_sweep(
    *,
    runs: Sequence[Dict[str, str]],
    c_grid: Sequence[float],
    class_weights: Sequence[str],
    data: str,
    data_config: str,
    categorical_encoding: str,
    probe_max_iter: int,
    probe_n_jobs: int,
    seed: int,
    per_run_json_dir: Path | None,
) -> List[Dict[str, Any]]:
    df, df_train, tr_np, _, _, _ = load_dataset_frames(data, data_config)
    rows: List[Dict[str, Any]] = []
    per_run_buckets: Dict[str, List[Dict[str, Any]]] = {}

    def _run_cell(run_spec: Dict[str, str], features: str, class_weight: str, probe_c: float) -> None:
        embedding_dir = Path(run_spec["embedding_dir"])
        args = _make_args(
            class_weight=class_weight,
            probe_c=probe_c,
            data=data,
            data_config=data_config,
            categorical_encoding=categorical_encoding,
            probe_max_iter=probe_max_iter,
            probe_n_jobs=probe_n_jobs,
            seed=seed,
        )
        logging.info(
            "probe_sweep %s | %s | cw=%s | C=%s",
            run_spec["run_name"],
            features,
            class_weight,
            probe_c,
        )
        result = run_probe_for_mode(
            features=features,
            embedding_dir=embedding_dir,
            df=df,
            df_train=df_train,
            tr_np=tr_np,
            args=args,
        )
        flat = _flatten_row(
            run_spec=run_spec,
            features=features,
            class_weight=class_weight,
            probe_c=probe_c,
            row=result,
        )
        rows.append(flat)
        per_run_buckets.setdefault(run_spec["run_name"], []).append(flat)

    # Shared no-embedding baseline once (raw+morph is identical across embedding dirs).
    baseline_run = runs[0]
    for class_weight in class_weights:
        for probe_c in c_grid:
            _run_cell(baseline_run, BASELINE_FEATURE_MODE, class_weight, probe_c)

    for run_spec in runs:
        for features in SSL_FEATURE_MODES:
            for class_weight in class_weights:
                for probe_c in c_grid:
                    _run_cell(run_spec, features, class_weight, probe_c)

    if per_run_json_dir is not None:
        per_run_json_dir.mkdir(parents=True, exist_ok=True)
        for run_name, bucket in per_run_buckets.items():
            path = per_run_json_dir / f"probe_sweep_current_protocol_{run_name}.json"
            path.write_text(
                json.dumps(
                    {
                        "protocol": PROTOCOL,
                        "run_name": run_name,
                        "rows": bucket,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            logging.info("Wrote %s", path)

    return rows


def write_markdown(path: Path, payload: Dict[str, Any]) -> None:
    lines = [
        "# Current-protocol probe sweep",
        "",
        "Frozen embeddings only; val max-F1 threshold; sklearn `LogisticRegression` (lbfgs).",
        "",
        f"- **C grid:** {payload['protocol']['probe_C_grid']}",
        f"- **Class weights:** {payload['protocol']['class_weight_modes']}",
        f"- **Total rows:** {len(payload['rows'])}",
        "",
        "See `scripts/summarize_probe_sweep_current_protocol.py` output for focus tables.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="Small-HI")
    parser.add_argument("--data_config", default="data_config.json")
    parser.add_argument("--categorical_encoding", default="ordinal", choices=["ordinal", "one_hot"])
    parser.add_argument("--probe_max_iter", type=int, default=1000)
    parser.add_argument("--probe_n_jobs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--output_json",
        default=str(_ROOT / "results/diagnostics/probe_sweep_current_protocol.json"),
    )
    parser.add_argument(
        "--output_md",
        default=str(_ROOT / "results/diagnostics/probe_sweep_current_protocol.md"),
    )
    parser.add_argument(
        "--per_run_json_dir",
        default=str(_ROOT / "results/diagnostics"),
        help="Directory for per-embedding JSON files (probe_sweep_current_protocol_<run_name>.json).",
    )
    parser.add_argument("--skip_summarize", action="store_true")
    parser.add_argument("--testing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger_setup()
    set_seed(args.seed)

    rows = run_sweep(
        runs=RUNS,
        c_grid=DEFAULT_C_GRID,
        class_weights=DEFAULT_CLASS_WEIGHTS,
        data=args.data,
        data_config=args.data_config,
        categorical_encoding=args.categorical_encoding,
        probe_max_iter=args.probe_max_iter,
        probe_n_jobs=args.probe_n_jobs,
        seed=args.seed,
        per_run_json_dir=Path(args.per_run_json_dir),
    )

    payload = {"protocol": PROTOCOL, "rows": rows}
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logging.info("Wrote %s (%d rows)", out_json, len(rows))

    write_markdown(Path(args.output_md), payload)
    logging.info("Wrote %s", args.output_md)

    if not args.skip_summarize:
        from scripts.summarize_probe_sweep_current_protocol import main as summarize_main

        summarize_main()


if __name__ == "__main__":
    main()
