#!/usr/bin/env python3
"""Explicit positive-weight probe sweeps for current-protocol embeddings."""

from __future__ import annotations

import argparse
import logging
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.probe_sweep_engine import SweepCellSpec, make_cell_specs, run_checkpointed_sweep
from util import logger_setup, set_seed

POS_WEIGHTS = (1.0, 3.0, 6.275, 10.0, 20.0, 50.0)
SMALL_LI_C_GRID = (0.1, 1.0, 10.0)
SMALL_HI_C_GRID = (1.0,)

SMALL_LI_RUNS = [
    {
        "run_label": "Small-LI GINe emlps+tds seed1 (20ep)",
        "run_name": "small_li_gin_20ep_seed1",
        "embedding_dir": "embeddings/small_li_gin_emlps_tds_asym_proj_8192neg_queue0_20ep_seed1",
    }
]

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

SSL_MODES = ("embedding", "embedding+raw", "embedding+raw+morph")
SMALL_LI_MODES = ("raw+morph",) + SSL_MODES


def policy_label(weight: float) -> str:
    return f"pos_{weight:g}"


def build_cells(sweep: str) -> tuple[list[SweepCellSpec], dict[str, dict[str, str]], dict[str, Any], str, Sequence[float]]:
    class_weights = [policy_label(w) for w in POS_WEIGHTS]
    if sweep == "small_li":
        run_specs = SMALL_LI_RUNS
        c_grid = SMALL_LI_C_GRID
        cells: List[SweepCellSpec] = []
        for run in run_specs:
            cells.extend(
                make_cell_specs(
                    run_spec=run,
                    feature_modes=SMALL_LI_MODES,
                    class_weights=class_weights,
                    c_grid=c_grid,
                )
            )
        data = "Small-LI"
        description = "Small-LI current-protocol explicit positive-weight probe sweep."
    elif sweep == "small_hi":
        run_specs = SMALL_HI_RUNS
        c_grid = SMALL_HI_C_GRID
        cells = []
        for run in run_specs:
            cells.extend(
                make_cell_specs(
                    run_spec=run,
                    feature_modes=SSL_MODES,
                    class_weights=class_weights,
                    c_grid=c_grid,
                )
            )
        # Raw/morph baseline once, using the seed1 embedding directory only for label alignment.
        cells.extend(
            make_cell_specs(
                run_spec=SMALL_HI_RUNS[0],
                feature_modes=("raw+morph",),
                class_weights=class_weights,
                c_grid=c_grid,
            )
        )
        data = "Small-HI"
        description = "Small-HI key-run explicit positive-weight probe sweep."
    else:
        raise ValueError(f"unknown sweep {sweep}")

    protocol = {
        "description": description,
        "class_weight_mode": "explicit positive weights {0: 1.0, 1: w}",
        "positive_weight_grid": list(POS_WEIGHTS),
        "class_weight_policies": class_weights,
        "probe_C_grid": list(c_grid),
        "threshold_tuning": "max_f1_on_val",
        "alert_budget_metrics": "precision/recall/lift at k=100,500,1000 per split",
        "run_specs": run_specs,
    }
    return cells, {r["run_name"]: r for r in run_specs}, protocol, data, c_grid


def write_markdown(path: Path, payload: Dict[str, Any]) -> None:
    lines = [
        f"# {payload['protocol']['description']}",
        "",
        f"- **Expected cells:** {payload.get('expected_cells')}",
        f"- **Recorded:** {len(payload.get('cells', []))}",
        f"- **Summary:** `{payload.get('summary')}`",
        "",
        "| run | feature | weight | C | AUROC | AUPRC | F1 | F1@0.5 | P@500 | R@500 | lift@500 | status |",
        "|-----|---------|--------|---|------:|------:|---:|-------:|------:|------:|---------:|--------|",
    ]
    for row in payload.get("cells", []):
        if row.get("status") != "completed":
            lines.append(
                f"| {row.get('run_label', row.get('run_name'))} | `{row.get('feature_mode')}` | "
                f"{row.get('class_weight_policy')} | {row.get('probe_C')} | — | — | — | — | — | — | — | {row.get('status')} |"
            )
            continue
        test = row["test"]
        lines.append(
            f"| {row.get('run_label', row.get('run_name'))} | `{row['feature_mode']}` | "
            f"{row['class_weight_policy']} | {row['probe_C']} | {test['auroc']:.4f} | "
            f"{test['auprc']:.4f} | {test['f1']:.4f} | {test['f1_at_0_5']:.4f} | "
            f"{test.get('precision_at_500', float('nan')):.4f} | {test.get('recall_at_500', float('nan')):.4f} | "
            f"{test.get('lift_at_500', float('nan')):.1f} | {row.get('status')} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", required=True, choices=["small_li", "small_hi"])
    parser.add_argument("--data_config", default="data_config.json")
    parser.add_argument("--categorical_encoding", default="ordinal", choices=["ordinal", "one_hot"])
    parser.add_argument("--probe_max_iter", type=int, default=1000)
    parser.add_argument("--probe_n_jobs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cache_root", required=True)
    parser.add_argument("--partial_json", required=True)
    parser.add_argument("--final_json", required=True)
    parser.add_argument("--output_md", required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--testing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger_setup()
    set_seed(args.seed)
    cells, runs, protocol, data, _ = build_cells(args.sweep)
    for spec in cells:
        emb = Path(spec.embedding_dir)
        for split in ("train", "val", "test"):
            if not (emb / f"{split}.npz").is_file():
                raise FileNotFoundError(f"Missing embeddings: {emb / f'{split}.npz'}")

    sweep_args = Namespace(
        data=data,
        data_config=args.data_config,
        categorical_encoding=args.categorical_encoding,
        probe_max_iter=args.probe_max_iter,
        probe_n_jobs=args.probe_n_jobs,
        seed=args.seed,
        model="gin",
    )
    logging.info("%s weight sweep cells=%d", args.sweep, len(cells))
    payload = run_checkpointed_sweep(
        cell_specs=cells,
        run_specs_by_name=runs,
        partial_path=Path(args.partial_json),
        final_path=Path(args.final_json),
        protocol=protocol,
        args=sweep_args,
        cache_root=Path(args.cache_root),
        force=args.force,
    )
    write_markdown(Path(args.output_md), payload)


if __name__ == "__main__":
    main()
