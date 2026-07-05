#!/usr/bin/env python3
"""Small-LI current-protocol probe sweep (checkpointed; frozen embeddings only)."""

from __future__ import annotations

import argparse
import logging
import sys
from argparse import Namespace
from pathlib import Path
from typing import Dict, List

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.probe_sweep_engine import SweepCellSpec, make_cell_specs, run_checkpointed_sweep
from util import logger_setup, set_seed

RUN_SPEC: Dict[str, str] = {
    "run_label": "Small-LI GINe emlps+tds seed1 (20ep)",
    "run_name": "small_li_gin_20ep_seed1",
    "embedding_dir": "embeddings/small_li_gin_emlps_tds_asym_proj_8192neg_queue0_20ep_seed1",
}

FEATURE_MODES = ("raw+morph", "embedding", "embedding+raw", "embedding+raw+morph")
CLASS_WEIGHTS = ("model", "none", "balanced")
C_GRID = (0.01, 0.1, 1.0, 10.0)

PROTOCOL = {
    "description": "Small-LI current-protocol probe sweep (checkpointed; frozen SSL embeddings only).",
    "dataset": "Small-LI",
    "threshold_tuning": "max_f1_on_val",
    "feature_modes": list(FEATURE_MODES),
    "class_weight_modes": list(CLASS_WEIGHTS),
    "class_weight_notes": {
        "model": "Shared GIN model weights via --class_weight model --model gin (~{0: 1.0, 1: 6.275}).",
        "none": "No class weighting.",
        "balanced": "Exploratory sklearn balanced weights; may be unstable/extreme on low-prevalence Small-LI.",
    },
    "probe_C_grid": list(C_GRID),
    "expected_cells": len(FEATURE_MODES) * len(CLASS_WEIGHTS) * len(C_GRID),
    "alert_budget_metrics": "precision/recall/lift at fixed k=100,500,1000 per split.",
}


def build_cell_specs() -> List[SweepCellSpec]:
    return make_cell_specs(
        run_spec=RUN_SPEC,
        feature_modes=FEATURE_MODES,
        class_weights=CLASS_WEIGHTS,
        c_grid=C_GRID,
    )


def write_markdown(path: Path, payload: dict) -> None:
    lines = [
        "# Small-LI current-protocol probe sweep",
        "",
        f"- **Expected cells:** {payload.get('expected_cells')}",
        f"- **Recorded:** {len(payload.get('cells', []))}",
        f"- **Summary:** `{payload.get('summary')}`",
        "",
        "| feature | cw | C | AUROC | AUPRC | F1 | F1@0.5 | P@500 | R@500 | lift@500 | status |",
        "|---------|----|---|------:|------:|---:|-------:|------:|------:|---------:|--------|",
    ]
    for row in payload.get("cells", []):
        if row.get("status") != "completed":
            lines.append(
                f"| `{row.get('feature_mode')}` | {row.get('class_weight_policy')} | "
                f"{row.get('probe_C')} | — | — | — | — | — | — | — | {row.get('status')} |"
            )
            continue
        test = row["test"]
        lines.append(
            f"| `{row['feature_mode']}` | {row['class_weight_policy']} | {row['probe_C']} | "
            f"{test['auroc']:.4f} | {test['auprc']:.4f} | {test['f1']:.4f} | "
            f"{test['f1_at_0_5']:.4f} | {test.get('precision_at_500', float('nan')):.4f} | "
            f"{test.get('recall_at_500', float('nan')):.4f} | {test.get('lift_at_500', float('nan')):.1f} | "
            f"{row.get('status')} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="Small-LI")
    parser.add_argument("--data_config", default="data_config.json")
    parser.add_argument("--categorical_encoding", default="ordinal", choices=["ordinal", "one_hot"])
    parser.add_argument("--probe_max_iter", type=int, default=1000)
    parser.add_argument("--probe_n_jobs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--cache_root",
        default=str(_ROOT / "results/cache/probe_features_small_li_current_protocol"),
    )
    parser.add_argument(
        "--partial_json",
        default=str(_ROOT / "results/diagnostics/probe_sweep_small_li_current_protocol_partial.json"),
    )
    parser.add_argument(
        "--final_json",
        default=str(_ROOT / "results/diagnostics/probe_sweep_small_li_current_protocol.json"),
    )
    parser.add_argument(
        "--output_md",
        default=str(_ROOT / "results/diagnostics/probe_sweep_small_li_current_protocol.md"),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--testing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger_setup()
    set_seed(args.seed)

    emb = Path(RUN_SPEC["embedding_dir"])
    for split in ("train", "val", "test"):
        if not (emb / f"{split}.npz").is_file():
            raise FileNotFoundError(f"Missing embeddings: {emb / f'{split}.npz'}")

    cell_specs = build_cell_specs()
    sweep_args = Namespace(
        data=args.data,
        data_config=args.data_config,
        categorical_encoding=args.categorical_encoding,
        probe_max_iter=args.probe_max_iter,
        probe_n_jobs=args.probe_n_jobs,
        seed=args.seed,
        model="gin",
    )
    protocol = dict(PROTOCOL)
    protocol["run_spec"] = RUN_SPEC

    logging.info(
        "Small-LI probe sweep cells=%d partial=%s",
        len(cell_specs),
        args.partial_json,
    )
    payload = run_checkpointed_sweep(
        cell_specs=cell_specs,
        run_specs_by_name={RUN_SPEC["run_name"]: RUN_SPEC},
        partial_path=Path(args.partial_json),
        final_path=Path(args.final_json),
        protocol=protocol,
        args=sweep_args,
        cache_root=Path(args.cache_root),
        force=args.force,
    )
    write_markdown(Path(args.output_md), payload)
    logging.info("Wrote %s", args.output_md)


if __name__ == "__main__":
    main()
