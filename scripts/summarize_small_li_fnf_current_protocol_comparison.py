#!/usr/bin/env python3
"""Summarize the Small-LI same-pair FNF scout against the plain Small-LI scout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

FEATURES = ("raw+morph", "embedding", "embedding+raw", "embedding+raw+morph")
SMALL_HI_FNF_NOTE = (
    "On Small-HI, same-pair FNF did not improve embedding-only GINe emlps+tds, "
    "but FNF seed1 remained strongest with the downstream full stack."
)


def load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def by_feature(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {row["features"]: row for row in payload.get("runs", [])}


def load_probe(path: Path) -> Optional[Dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def metric_row(row: Dict[str, Any]) -> str:
    test = row["test"]
    return (
        f"{test['auroc']:.3f} / {test['auprc']:.3f} / {test['f1']:.3f} / "
        f"{test.get('f1_at_0_5', row.get('test_at_threshold_0.5', {}).get('f1', float('nan'))):.3f}"
    )


def probe_metric(probe: Optional[Dict[str, Any]]) -> str:
    if not probe:
        return "not available"
    test = probe["splits_at_selected_threshold"]["test"]
    t05 = probe.get("splits_at_threshold_0.5", {}).get("test", {})
    return (
        f"{test['auroc']:.3f} / {test['auprc']:.3f} / {test['f1']:.3f} / "
        f"{t05.get('f1', float('nan')):.3f}"
    )


def write_note(
    *,
    plain_json: Path,
    fnf_json: Path,
    fnf_probe: Path,
    plain_probe: Path,
    fnf_last_probe: Path,
    output_md: Path,
) -> None:
    plain = by_feature(load_json(plain_json))
    fnf = by_feature(load_json(fnf_json))
    plain_embedding_probe = load_probe(plain_probe)
    fnf_embedding_probe = load_probe(fnf_probe)
    fnf_last = load_probe(fnf_last_probe)

    lines = [
        "# Small-LI FNF Current-Protocol Comparison",
        "",
        "Small-LI FNF scout: same-pair false-negative filtering added to the current GINe emlps+tds SSL recipe. This is a dataset comparison, not a frozen benchmark.",
        "",
        "Metrics are test AUROC / AUPRC / F1 / F1@0.5 with shared probe weights (`cw=model --model gin`).",
        "",
        "## Embedding-Only Probe",
        "",
        f"- Plain Small-LI: {probe_metric(plain_embedding_probe)}",
        f"- Small-LI FNF: {probe_metric(fnf_embedding_probe)}",
    ]
    if fnf_last:
        lines.append(f"- Small-LI FNF last checkpoint: {probe_metric(fnf_last)}")
    lines.extend(
        [
            "",
            "## Feature Stacks",
            "",
            "| Features | Plain Small-LI | FNF Small-LI | Delta F1 | Delta AUPRC |",
            "|----------|----------------|--------------|---------:|------------:|",
        ]
    )
    for feature in FEATURES:
        p = plain.get(feature)
        f = fnf.get(feature)
        if not p or not f:
            continue
        delta_f1 = f["test"]["f1"] - p["test"]["f1"]
        delta_auprc = f["test"]["auprc"] - p["test"]["auprc"]
        lines.append(
            f"| `{feature}` | {metric_row(p)} | {metric_row(f)} | {delta_f1:+.3f} | {delta_auprc:+.3f} |"
        )

    best_plain = max(plain.values(), key=lambda r: r["test"]["f1"])
    best_fnf = max(fnf.values(), key=lambda r: r["test"]["f1"])
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Plain best F1 stack: `{best_plain['features']}` ({best_plain['test']['f1']:.3f}).",
            f"- FNF best F1 stack: `{best_fnf['features']}` ({best_fnf['test']['f1']:.3f}).",
            f"- {SMALL_HI_FNF_NOTE}",
            "- Check the last-checkpoint row above to see whether FNF changes the best-vs-last behavior seen in the plain Small-LI scout.",
            "",
            "Artifacts:",
            "",
            "- FNF feature ablation: `results/diagnostics/probe_feature_ablation_small_li_fnf_current_protocol_seed1.json`",
            "- FNF embeddings: `embeddings/small_li_fnf_emlps_tds_asym_proj_8192neg_queue0_20ep_seed1/`",
            "",
        ]
    )
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plain_json",
        default="results/diagnostics/probe_feature_ablation_small_li_current_protocol_seed1.json",
    )
    parser.add_argument(
        "--fnf_json",
        default="results/diagnostics/probe_feature_ablation_small_li_fnf_current_protocol_seed1.json",
    )
    parser.add_argument(
        "--plain_probe",
        default="embeddings/small_li_gin_emlps_tds_asym_proj_8192neg_queue0_20ep_seed1/probe_results.json",
    )
    parser.add_argument(
        "--fnf_probe",
        default="embeddings/small_li_fnf_emlps_tds_asym_proj_8192neg_queue0_20ep_seed1/probe_results.json",
    )
    parser.add_argument(
        "--fnf_last_probe",
        default="embeddings/small_li_fnf_emlps_tds_asym_proj_8192neg_queue0_20ep_seed1_last_ckpt/probe_results.json",
    )
    parser.add_argument(
        "--output_md",
        default="notes/small_li_fnf_current_protocol_comparison.md",
    )
    args = parser.parse_args()
    write_note(
        plain_json=Path(args.plain_json),
        fnf_json=Path(args.fnf_json),
        fnf_probe=Path(args.fnf_probe),
        plain_probe=Path(args.plain_probe),
        fnf_last_probe=Path(args.fnf_last_probe),
        output_md=Path(args.output_md),
    )
    print(args.output_md)


if __name__ == "__main__":
    main()
