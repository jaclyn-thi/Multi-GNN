#!/usr/bin/env python3
"""Compare best-checkpoint vs last-checkpoint morphology probe results."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RUNS = [
    "morph_degree_fan_only_asym_proj_8192neg_queue0_20ep_weight005",
    "morph_degree_fan_only_asym_proj_8192neg_queue0_20ep_defaultweight",
    "morph_motif_participation_only_asym_proj_8192neg_queue0_10ep_weight005",
    "morph_flow_balance_only_asym_proj_8192neg_queue0_10ep_weight005",
    "morph_motif_participation_only_asym_proj_8192neg_queue0_10ep",
    "morph_flow_balance_only_asym_proj_8192neg_queue0_10ep",
]


def load_probe(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def row_from_probe(run: str, ckpt_type: str, path: Path, data: dict) -> dict:
    test_sel = data["splits_at_selected_threshold"]["test"]
    test_fixed = (data.get("splits_at_threshold_0.5") or {}).get("test", {})
    meta = data.get("extraction_meta") or {}
    row = {
        "run": run,
        "checkpoint_type": ckpt_type,
        "epoch": meta.get("checkpoint_epoch"),
        "auroc": test_sel.get("auroc"),
        "f1": test_sel.get("f1"),
        "precision": test_sel.get("precision"),
        "recall": test_sel.get("recall"),
        "f1_at_0.5": test_fixed.get("f1"),
        "path": str(path.relative_to(ROOT)),
    }
    # Optional recall-oriented fields when present (future enriched probes).
    for k in (
        "precision_at_100",
        "recall_at_100",
        "lift_at_100",
        "precision_at_500",
        "recall_at_500",
        "lift_at_500",
        "precision_at_1000",
        "recall_at_1000",
        "lift_at_1000",
        "recall_at_precision_ge_0.95",
        "recall_at_precision_ge_0.90",
        "recall_at_precision_ge_0.80",
        "recall_at_precision_ge_0.70",
    ):
        if k in test_sel:
            row[k] = test_sel[k]
    return row


def main() -> None:
    rows: list[dict] = []
    for run in RUNS:
        best_path = ROOT / "embeddings" / run / "probe_results.json"
        last_path = ROOT / "embeddings" / f"{run}_lastckpt_probe" / "probe_results_lastckpt.json"
        last_fallback = ROOT / "embeddings" / f"{run}_lastckpt_probe" / "probe_results.json"

        best = load_probe(best_path)
        if best is not None:
            rows.append(row_from_probe(run, "best", best_path, best))
        else:
            rows.append({"run": run, "checkpoint_type": "best", "path": str(best_path.relative_to(ROOT)), "missing": True})

        last = load_probe(last_path) or load_probe(last_fallback)
        if last is not None:
            p = last_path if last_path.is_file() else last_fallback
            rows.append(row_from_probe(run, "last", p, last))
        else:
            rows.append(
                {
                    "run": run,
                    "checkpoint_type": "last",
                    "path": str(last_path.relative_to(ROOT)),
                    "missing": True,
                }
            )

    print(
        "run\tcheckpoint_type\tepoch\tAUROC\tF1\tprec\trec\tF1@0.5\tpath"
    )
    for r in rows:
        if r.get("missing"):
            print(f"{r['run']}\t{r['checkpoint_type']}\t-\t-\t-\t-\t-\t-\t{r['path']} (missing)")
            continue
        print(
            f"{r['run']}\t{r['checkpoint_type']}\t{r['epoch']}\t"
            f"{r['auroc']:.4f}\t{r['f1']:.4f}\t{r['precision']:.4f}\t{r['recall']:.4f}\t"
            f"{(r['f1_at_0.5'] if r['f1_at_0.5'] is not None else float('nan')):.4f}\t{r['path']}"
        )


if __name__ == "__main__":
    main()
