#!/usr/bin/env python3
"""Compute alert-budget metrics for current-protocol key runs."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.probe_sweep_engine import SweepCellSpec, make_cell_specs, run_checkpointed_sweep
from util import logger_setup, set_seed

SMALL_HI_RUNS = [
    {
        "run_label": "Small-HI GINe emlps+tds seed1 (20ep)",
        "run_name": "small_hi_gin_20ep_seed1",
        "embedding_dir": "embeddings/hi_contrastive_gin_emlps_tds_proj_asym_8192neg_queue0_accum4_20ep",
    },
    {
        "run_label": "Small-HI FNF + emlps+tds seed1",
        "run_name": "small_hi_fnf_20ep_seed1",
        "embedding_dir": "embeddings/same_pair_fnf_emlps_tds_asym_proj_8192neg_queue0_20ep",
    },
    {
        "run_label": "Small-HI GINe emlps+tds seed2 (40ep)",
        "run_name": "small_hi_gin_40ep_seed2",
        "embedding_dir": "embeddings/gin_emlps_tds_asym_proj_8192neg_queue0_40ep_seed2",
    },
]

SMALL_LI_RUNS = [
    {
        "run_label": "Small-LI GINe emlps+tds seed1 (20ep)",
        "run_name": "small_li_gin_20ep_seed1",
        "embedding_dir": "embeddings/small_li_gin_emlps_tds_asym_proj_8192neg_queue0_20ep_seed1",
    },
    {
        "run_label": "Small-LI FNF + emlps+tds seed1 (20ep)",
        "run_name": "small_li_fnf_20ep_seed1",
        "embedding_dir": "embeddings/small_li_fnf_emlps_tds_asym_proj_8192neg_queue0_20ep_seed1",
    },
]

FEATURE_MODES = ("embedding", "embedding+raw", "embedding+raw+morph")
BASELINE_MODE = "raw+morph"


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2))


def existing_runs(runs: List[Dict[str, str]]) -> List[Dict[str, str]]:
    out = []
    for run in runs:
        emb = Path(run["embedding_dir"])
        if all((emb / f"{split}.npz").is_file() for split in ("train", "val", "test")):
            out.append(run)
        else:
            logging.info("Skipping alert-budget run without embeddings: %s", run["run_name"])
    return out


def build_cells(runs: List[Dict[str, str]], include_raw_baseline: bool) -> List[SweepCellSpec]:
    cells: List[SweepCellSpec] = []
    for run in runs:
        cells.extend(
            make_cell_specs(
                run_spec=run,
                feature_modes=FEATURE_MODES,
                class_weights=("model",),
                c_grid=(1.0,),
            )
        )
    if include_raw_baseline and runs:
        cells.extend(
            make_cell_specs(
                run_spec=runs[0],
                feature_modes=(BASELINE_MODE,),
                class_weights=("model",),
                c_grid=(1.0,),
            )
        )
    return cells


def row_line(row: Dict[str, Any]) -> str:
    test = row["test"]
    return (
        f"| {row.get('run_label', row.get('run_name'))} | `{row['feature_mode']}` | "
        f"{test['auroc']:.4f} | {test['auprc']:.4f} | {test['f1']:.4f} | {test['f1_at_0_5']:.4f} | "
        f"{test.get('precision_at_100', float('nan')):.4f} | {test.get('recall_at_100', float('nan')):.4f} | "
        f"{test.get('lift_at_100', float('nan')):.1f} | {test.get('precision_at_500', float('nan')):.4f} | "
        f"{test.get('recall_at_500', float('nan')):.4f} | {test.get('lift_at_500', float('nan')):.1f} | "
        f"{test.get('precision_at_1000', float('nan')):.4f} | {test.get('recall_at_1000', float('nan')):.4f} | "
        f"{test.get('lift_at_1000', float('nan')):.1f} |"
    )


def write_markdown(path: Path, payload: Dict[str, Any]) -> None:
    lines = [
        "# Alert-Budget Metrics (Current Protocol)",
        "",
        "CPU fallback evaluation for current-protocol key runs. Metrics use `cw=model`, C=1.0, val-tuned F1 threshold, and fixed alert budgets on the test split.",
        "",
        "| Run | Features | AUROC | AUPRC | F1 | F1@0.5 | P@100 | R@100 | lift@100 | P@500 | R@500 | lift@500 | P@1000 | R@1000 | lift@1000 |",
        "|-----|----------|------:|------:|---:|-------:|------:|------:|---------:|------:|------:|---------:|-------:|-------:|----------:|",
    ]
    for row in payload["rows"]:
        lines.append(row_line(row))
    lines.append("")
    atomic_write_text(path, "\n".join(lines))


def run_group(
    *,
    data: str,
    runs: List[Dict[str, str]],
    partial_path: Path,
    final_path: Path,
    cache_root: Path,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    cells = build_cells(runs, include_raw_baseline=True)
    if not cells:
        return []
    sweep_args = Namespace(
        data=data,
        data_config=args.data_config,
        categorical_encoding="ordinal",
        probe_max_iter=args.probe_max_iter,
        probe_n_jobs=args.probe_n_jobs,
        seed=args.seed,
        model="gin",
    )
    payload = run_checkpointed_sweep(
        cell_specs=cells,
        run_specs_by_name={r["run_name"]: r for r in runs},
        partial_path=partial_path,
        final_path=final_path,
        protocol={
            "description": f"Alert-budget metrics for {data} current-protocol key runs.",
            "class_weight_policy": "model",
            "probe_C": 1.0,
            "feature_modes": list(FEATURE_MODES) + [BASELINE_MODE],
        },
        args=sweep_args,
        cache_root=cache_root,
        force=args.force,
    )
    return [c for c in payload.get("cells", []) if c.get("status") == "completed"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_config", default="data_config.json")
    parser.add_argument("--probe_max_iter", type=int, default=1000)
    parser.add_argument("--probe_n_jobs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--testing", action="store_true")
    parser.add_argument(
        "--output_json",
        default="results/diagnostics/alert_budget_metrics_current_protocol.json",
    )
    parser.add_argument(
        "--output_md",
        default="results/diagnostics/alert_budget_metrics_current_protocol.md",
    )
    parser.add_argument(
        "--notes_md",
        default="notes/alert_budget_metrics_current_protocol.md",
    )
    args = parser.parse_args()

    logger_setup()
    set_seed(args.seed)
    diag = _ROOT / "results/diagnostics"
    cache = _ROOT / "results/cache/alert_budget_metrics_current_protocol"
    rows = []
    rows.extend(
        run_group(
            data="Small-HI",
            runs=existing_runs(SMALL_HI_RUNS),
            partial_path=diag / "alert_budget_metrics_small_hi_partial.json",
            final_path=diag / "alert_budget_metrics_small_hi.json",
            cache_root=cache,
            args=args,
        )
    )
    rows.extend(
        run_group(
            data="Small-LI",
            runs=existing_runs(SMALL_LI_RUNS),
            partial_path=diag / "alert_budget_metrics_small_li_partial.json",
            final_path=diag / "alert_budget_metrics_small_li.json",
            cache_root=cache,
            args=args,
        )
    )
    payload = {
        "description": "Alert-budget metrics for current-protocol key runs.",
        "rows": rows,
        "cells_completed": len(rows),
    }
    out_json = Path(args.output_json)
    atomic_write_json(out_json, payload)
    write_markdown(Path(args.output_md), payload)
    # notes/alert_budget_metrics_current_protocol.md is curated by hand (adds
    # thesis takeaways / interpretation); only the raw diagnostics markdown is
    # regenerated here to avoid clobbering the interpreted note.
    _ = args.notes_md
    print(args.output_md)


if __name__ == "__main__":
    main()
