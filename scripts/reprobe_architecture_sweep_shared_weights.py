#!/usr/bin/env python3
"""Reprobe architecture-sweep embeddings with shared GIN class weights."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_util import extract_param

ARCHITECTURE_SWEEP_RUNS: List[Dict[str, str]] = [
    {
        "encoder": "gin",
        "unique_name": "hi_contrastive_gin_emlps_tds_proj_asym_8192neg_queue0_accum4_20ep",
    },
    {
        "encoder": "gat",
        "unique_name": "gate_emlps_tds_asym_proj_8192neg_queue0_20ep_seed1",
    },
    {
        "encoder": "pna",
        "unique_name": "pna_emlps_tds_asym_proj_8192neg_queue0_20ep_seed1",
    },
    {
        "encoder": "rgcn",
        "unique_name": "rgcn_emlps_tds_asym_proj_8192neg_queue0_20ep_seed1",
    },
]


def gin_model_class_weights() -> Dict[int, float]:
    args = SimpleNamespace(model="gin")
    return {
        0: float(extract_param("w_ce1", args)),
        1: float(extract_param("w_ce2", args)),
    }


def load_original_probe(unique_name: str) -> Optional[Dict[str, Any]]:
    path = ROOT / "embeddings" / unique_name / "probe_results.json"
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def summarize_probe_row(
    encoder: str,
    unique_name: str,
    probe: Dict[str, Any],
    probe_source: str,
) -> Dict[str, Any]:
    test_sel = probe["splits_at_selected_threshold"]["test"]
    test_05 = probe.get("splits_at_threshold_0.5", {}).get("test", {})
    return {
        "encoder": encoder,
        "unique_name": unique_name,
        "probe_source": probe_source,
        "class_weight": probe.get("class_weight"),
        "threshold": probe["classification_threshold"]["value"],
        "test_auroc": test_sel["auroc"],
        "test_auprc": test_sel["auprc"],
        "test_f1": test_sel["f1"],
        "test_precision": test_sel["precision"],
        "test_recall": test_sel["recall"],
        "test_f1_at_0.5": test_05.get("f1"),
        "test_precision_at_0.5": test_05.get("precision"),
        "test_recall_at_0.5": test_05.get("recall"),
    }


def format_markdown_table(rows: List[Dict[str, Any]], title: str) -> str:
    lines = [
        f"## {title}",
        "",
        "| Encoder | AUROC | AUPRC | F1 | P | R | F1@0.5 | threshold | class weights |",
        "|---------|-------|-------|-----|---|---|--------|-------------|---------------|",
    ]
    for row in rows:
        cw = row["class_weight"]
        cw_str = f"{{0: {cw['0']:.4g}, 1: {cw['1']:.4g}}}" if isinstance(cw, dict) else str(cw)
        lines.append(
            f"| {row['encoder']} | {row['test_auroc']:.3f} | {row['test_auprc']:.3f} | "
            f"{row['test_f1']:.3f} | {row['test_precision']:.3f} | {row['test_recall']:.3f} | "
            f"{row['test_f1_at_0.5']:.3f} | {row['threshold']:.3f} | `{cw_str}` |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_json",
        default=str(ROOT / "results/diagnostics/architecture_sweep_shared_probe_weights.json"),
    )
    parser.add_argument(
        "--output_md",
        default=str(ROOT / "results/diagnostics/architecture_sweep_shared_probe_weights.md"),
    )
    parser.add_argument(
        "--probe_dir",
        default=str(ROOT / "results/diagnostics/architecture_sweep_shared_probe_weights"),
        help="Per-run probe_results.json written here (does not overwrite embeddings/).",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    probe_dir = Path(args.probe_dir)
    probe_dir.mkdir(parents=True, exist_ok=True)
    shared_weights = gin_model_class_weights()

    shared_rows: List[Dict[str, Any]] = []
    original_rows: List[Dict[str, Any]] = []

    for spec in ARCHITECTURE_SWEEP_RUNS:
        unique_name = spec["unique_name"]
        encoder = spec["encoder"]
        emb = ROOT / "embeddings" / unique_name
        if not (emb / "train.npz").is_file():
            print(f"SKIP (missing npz): {unique_name}")
            continue

        original = load_original_probe(unique_name)
        if original is not None:
            original_rows.append(
                summarize_probe_row(encoder, unique_name, original, "embeddings/probe_results.json")
            )

        out_probe = probe_dir / f"{unique_name}.probe_results.json"
        cmd = [
            args.python,
            str(ROOT / "linear_probe.py"),
            "--unique_name",
            unique_name,
            "--class_weight",
            "model",
            "--model",
            "gin",
            "--probe_output",
            str(out_probe),
            "--testing",
        ]
        print("Running:", " ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, cwd=ROOT, check=True)
            with out_probe.open(encoding="utf-8") as f:
                probe = json.load(f)
        else:
            probe = {"class_weight": shared_weights}

        if not args.dry_run:
            shared_rows.append(
                summarize_probe_row(encoder, unique_name, probe, str(out_probe.relative_to(ROOT)))
            )

    payload = {
        "policy": {
            "class_weight_mode": "model",
            "model_for_weights": "gin",
            "shared_class_weight": {str(k): v for k, v in shared_weights.items()},
        },
        "runs": shared_rows,
        "original_per_architecture_model_weights": original_rows,
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {out_json}")

    if shared_rows:
        md = [
            "# Architecture sweep — shared GIN probe class weights",
            "",
            "Linear probe on frozen embeddings. All runs use `--class_weight model --model gin` "
            f"(weights `{payload['policy']['shared_class_weight']}`). "
            "Does not overwrite `embeddings/*/probe_results.json`.",
            "",
            format_markdown_table(shared_rows, "Shared-weight reprobe (test split)"),
        ]
        if original_rows:
            md.append(format_markdown_table(original_rows, "Original per-architecture model weights (reference)"))
        out_md = Path(args.output_md)
        out_md.write_text("\n".join(md), encoding="utf-8")
        print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
