#!/usr/bin/env python3
"""Targeted 40ep current-protocol probe sweep (checkpointed, split per seed)."""

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

from scripts.probe_sweep_engine import (  # noqa: E402
    SweepCellSpec,
    make_cell_specs,
    run_checkpointed_sweep,
)
from util import logger_setup, set_seed  # noqa: E402

TARGETED_C_GRID = (0.1, 1.0, 10.0)
TARGETED_CLASS_WEIGHTS = ("model", "none")
TARGETED_SSL_MODES = ("embedding", "embedding+raw", "embedding+raw+morph")
BASELINE_MODE = "raw+morph"

GIN_40EP_RUNS: Dict[int, Dict[str, str]] = {
    1: {
        "run_label": "GINe emlps+tds seed1 (40ep)",
        "run_name": "gin_40ep_seed1",
        "embedding_dir": "embeddings/gin_emlps_tds_asym_proj_8192neg_queue0_40ep_seed1",
    },
    2: {
        "run_label": "GINe emlps+tds seed2 (40ep)",
        "run_name": "gin_40ep_seed2",
        "embedding_dir": "embeddings/gin_emlps_tds_asym_proj_8192neg_queue0_40ep_seed2",
    },
    3: {
        "run_label": "GINe emlps+tds seed3 (40ep)",
        "run_name": "gin_40ep_seed3",
        "embedding_dir": "embeddings/gin_emlps_tds_asym_proj_8192neg_queue0_40ep_seed3",
    },
    4: {
        "run_label": "GINe emlps+tds seed4 (40ep)",
        "run_name": "gin_40ep_seed4",
        "embedding_dir": "embeddings/gin_emlps_tds_asym_proj_8192neg_queue0_40ep_seed4",
    },
}

BASELINE_RUN = GIN_40EP_RUNS[1]

PROTOCOL_40EP = {
    "description": "Current-protocol 40ep targeted probe sweep (checkpointed; frozen embeddings only).",
    "threshold_tuning": "max_f1_on_val",
    "class_weight_modes": list(TARGETED_CLASS_WEIGHTS),
    "class_weight_notes": {
        "model": "Shared GIN weights via --class_weight model --model gin (~{0: 1.0, 1: 6.275})",
        "none": "No class weighting",
    },
    "probe_C_grid": list(TARGETED_C_GRID),
    "ssl_feature_modes": list(TARGETED_SSL_MODES),
    "baseline_feature_mode": BASELINE_MODE,
    "cells_per_seed": len(TARGETED_SSL_MODES) * len(TARGETED_CLASS_WEIGHTS) * len(TARGETED_C_GRID),
    "baseline_cells_once": len(TARGETED_CLASS_WEIGHTS) * len(TARGETED_C_GRID),
}


def build_seed_cell_specs(seed_num: int, include_baseline: bool) -> List[SweepCellSpec]:
    run_spec = GIN_40EP_RUNS[seed_num]
    specs = make_cell_specs(
        run_spec=run_spec,
        feature_modes=TARGETED_SSL_MODES,
        class_weights=TARGETED_CLASS_WEIGHTS,
        c_grid=TARGETED_C_GRID,
    )
    if include_baseline and seed_num == 1:
        specs.extend(
            make_cell_specs(
                run_spec=BASELINE_RUN,
                feature_modes=(BASELINE_MODE,),
                class_weights=TARGETED_CLASS_WEIGHTS,
                c_grid=TARGETED_C_GRID,
            )
        )
    return specs


def write_seed_markdown(path: Path, payload: Dict[str, Any], seed_num: int) -> None:
    lines = [
        f"# 40ep targeted probe sweep — seed {seed_num}",
        "",
        f"- **Expected cells:** {payload.get('expected_cells')}",
        f"- **Recorded:** {len(payload.get('cells', []))}",
        f"- **Summary:** `{payload.get('summary')}`",
        "",
        "| feature | cw | C | test F1 | test AUPRC | F1@0.5 | status |",
        "|---------|----|---|--------:|-----------:|-------:|--------|",
    ]
    for row in payload.get("cells", []):
        if row.get("status") != "completed":
            lines.append(
                f"| `{row.get('feature_mode')}` | {row.get('class_weight_policy')} | "
                f"{row.get('probe_C')} | — | — | — | {row.get('status')} |"
            )
            continue
        test = row["test"]
        lines.append(
            f"| `{row.get('feature_mode')}` | {row.get('class_weight_policy')} | "
            f"{row.get('probe_C')} | {test['f1']:.4f} | {test['auprc']:.4f} | "
            f"{test['f1_at_0_5']:.4f} | {row.get('status')} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-num", type=int, required=True, choices=[1, 2, 3, 4])
    parser.add_argument("--data", default="Small-HI")
    parser.add_argument("--data_config", default="data_config.json")
    parser.add_argument("--categorical_encoding", default="ordinal", choices=["ordinal", "one_hot"])
    parser.add_argument("--probe_max_iter", type=int, default=1000)
    parser.add_argument("--probe_n_jobs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1, help="LogisticRegression random_state.")
    parser.add_argument(
        "--cache_root",
        default=str(_ROOT / "results/cache/probe_features_current_protocol"),
    )
    parser.add_argument(
        "--partial_json",
        default=None,
        help="Default: results/diagnostics/probe_sweep_40ep_seed{N}_partial.json",
    )
    parser.add_argument(
        "--final_json",
        default=None,
        help="Default: results/diagnostics/probe_sweep_40ep_seed{N}.json",
    )
    parser.add_argument(
        "--output_md",
        default=None,
        help="Default: results/diagnostics/probe_sweep_40ep_seed{N}.md",
    )
    parser.add_argument("--include_baseline", action="store_true", default=True)
    parser.add_argument("--no_baseline", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-run completed cells.")
    parser.add_argument("--consolidate", action="store_true", help="Merge all seed finals.")
    parser.add_argument("--testing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger_setup()
    set_seed(args.seed)

    diag = _ROOT / "results/diagnostics"
    seed_num = args.seed_num
    partial_path = Path(
        args.partial_json or diag / f"probe_sweep_40ep_seed{seed_num}_partial.json"
    )
    final_path = Path(args.final_json or diag / f"probe_sweep_40ep_seed{seed_num}.json")
    md_path = Path(args.output_md or diag / f"probe_sweep_40ep_seed{seed_num}.md")

    include_baseline = args.include_baseline and not args.no_baseline
    cell_specs = build_seed_cell_specs(seed_num, include_baseline=include_baseline)
    run_spec = GIN_40EP_RUNS[seed_num]
    run_specs_by_name = {run_spec["run_name"]: run_spec}
    if include_baseline and seed_num == 1:
        run_specs_by_name[BASELINE_RUN["run_name"]] = BASELINE_RUN

    for spec in cell_specs:
        emb = Path(spec.embedding_dir)
        if not (emb / "train.npz").is_file():
            raise FileNotFoundError(f"Missing embeddings: {emb / 'train.npz'}")

    sweep_args = Namespace(
        data=args.data,
        data_config=args.data_config,
        categorical_encoding=args.categorical_encoding,
        probe_max_iter=args.probe_max_iter,
        probe_n_jobs=args.probe_n_jobs,
        seed=args.seed,
        model="gin",
    )

    protocol = dict(PROTOCOL_40EP)
    protocol["seed_num"] = seed_num
    protocol["run_spec"] = run_spec
    protocol["include_baseline"] = include_baseline

    logging.info(
        "40ep targeted sweep seed=%d cells=%d partial=%s",
        seed_num,
        len(cell_specs),
        partial_path,
    )

    payload = run_checkpointed_sweep(
        cell_specs=cell_specs,
        run_specs_by_name=run_specs_by_name,
        partial_path=partial_path,
        final_path=final_path,
        protocol=protocol,
        args=sweep_args,
        cache_root=Path(args.cache_root),
        force=args.force,
    )

    write_seed_markdown(md_path, payload, seed_num)
    logging.info("Wrote %s", md_path)

    if args.consolidate:
        from scripts.summarize_probe_sweep_40ep_current_protocol import consolidate

        consolidate()


if __name__ == "__main__":
    main()
