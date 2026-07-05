#!/usr/bin/env python3
"""Consolidate current-protocol probe feature ablation JSONs into comparison tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]

COMPARE_MODES = (
    "raw",
    "morph",
    "raw+morph",
    "embedding",
    "embedding+raw",
    "embedding+raw+morph",
)

FOCUS_MODES = ("raw+morph", "embedding", "embedding+raw+morph")

RUNS = (
    {
        "label": "GINe emlps+tds seed1 (20ep)",
        "json": "results/diagnostics/probe_feature_ablation_current_protocol_gin_emlps_tds_seed1.json",
    },
    {
        "label": "GINe emlps+tds seed1 (40ep)",
        "json": "results/diagnostics/probe_feature_ablation_current_protocol_gin_40ep_seed1.json",
        "optional": True,
    },
    {
        "label": "GINe emlps+tds seed2 (40ep)",
        "json": "results/diagnostics/probe_feature_ablation_current_protocol_gin_40ep_seed2.json",
        "optional": True,
    },
    {
        "label": "FNF + emlps+tds seed1",
        "json": "results/diagnostics/probe_feature_ablation_current_protocol_fnf_emlps_tds_seed1.json",
        "legacy_json": "results/diagnostics/probe_feature_ablation_same_pair_fnf_emlps_tds.json",
        "optional": True,
    },
    {
        "label": "FNF + emlps+tds seed2",
        "json": "results/diagnostics/probe_feature_ablation_current_protocol_fnf_emlps_tds_seed2.json",
        "optional": True,
    },
)

POLICY = {
    "class_weight_mode": "model",
    "model_for_weights": "gin",
    "shared_class_weight": {"0": 1.0000182882773443, "1": 6.275014431494497},
    "note": "All probes use --class_weight model --model gin regardless of SSL encoder.",
}


def _row(run_label: str, mode: str, entry: Dict[str, Any]) -> Dict[str, Any]:
    test = entry["test"]
    t05 = entry.get("test_at_threshold_0.5", {})
    return {
        "run_label": run_label,
        "features": mode,
        "auroc": test.get("auroc"),
        "auprc": test.get("auprc"),
        "f1": test.get("f1"),
        "precision": test.get("precision"),
        "recall": test.get("recall"),
        "f1_at_0_5": t05.get("f1"),
        "threshold": test.get("threshold"),
        "val_f1": test.get("val_f1"),
        "class_weight": entry.get("class_weight"),
        "feature_dim": test.get("feature_dim"),
        "embedding_dir": entry.get("embedding_dir"),
    }


def _resolve_path(spec: Dict[str, Any]) -> Optional[Path]:
    path = ROOT / spec["json"]
    if path.is_file():
        return path
    legacy = spec.get("legacy_json")
    if legacy and (ROOT / legacy).is_file():
        return ROOT / legacy
    if spec.get("optional"):
        return None
    return path


def _load_rows(path: Path, run_label: str) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    by_mode = {r["features"]: r for r in payload["runs"]}
    rows = []
    for mode in COMPARE_MODES:
        if mode not in by_mode:
            raise KeyError(f"{path}: missing features={mode!r}")
        rows.append(_row(run_label, mode, by_mode[mode]))
    return rows


def _write_md(path: Path, rows: List[Dict[str, Any]], title: str, modes: tuple) -> None:
    cw = POLICY.get("shared_class_weight", {})
    lines = [
        f"# {title}",
        "",
        "Frozen embeddings; logistic regression; val max-F1 threshold; shared GIN class weights "
        f"(`{cw}`).",
        "",
        "| Run | Features | AUROC | AUPRC | F1 | Prec | Recall | F1@0.5 | Thr |",
        "|-----|----------|------:|------:|---:|-----:|-------:|-------:|----:|",
    ]
    mode_set = set(modes)
    for r in rows:
        if r["features"] not in mode_set:
            continue
        f1_05 = r["f1_at_0_5"]
        f1_05_str = f"{f1_05:.4f}" if f1_05 is not None else "—"
        lines.append(
            f"| {r['run_label']} | `{r['features']}` | "
            f"{r['auroc']:.4f} | {r['auprc']:.4f} | {r['f1']:.4f} | "
            f"{r['precision']:.4f} | {r['recall']:.4f} | {f1_05_str} | "
            f"{r['threshold']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_json",
        default=str(ROOT / "results/diagnostics/probe_feature_ablation_current_protocol_comparison.json"),
    )
    parser.add_argument(
        "--output_md",
        default=str(ROOT / "notes/probe_feature_ablation_current_protocol_comparison.md"),
    )
    parser.add_argument(
        "--focus_md",
        default=str(ROOT / "notes/probe_feature_ablation_current_protocol_stack_comparison.md"),
    )
    args = parser.parse_args()

    all_rows: List[Dict[str, Any]] = []
    included_runs: List[str] = []
    for spec in RUNS:
        path = _resolve_path(spec)
        if path is None:
            continue
        rows = _load_rows(path, spec["label"])
        all_rows.extend(rows)
        included_runs.append(spec["label"])

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"policy": POLICY, "included_runs": included_runs, "rows": all_rows}
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out} ({len(all_rows)} rows from {len(included_runs)} runs)")

    if all_rows:
        _write_md(
            Path(args.output_md),
            all_rows,
            "Current-protocol probe feature ablation comparison",
            COMPARE_MODES,
        )
        print(f"Wrote {args.output_md}")
        _write_md(
            Path(args.focus_md),
            all_rows,
            "Current-protocol downstream stack comparison (focus modes)",
            FOCUS_MODES,
        )
        print(f"Wrote {args.focus_md}")


if __name__ == "__main__":
    main()
