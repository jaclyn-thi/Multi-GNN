#!/usr/bin/env python3
"""Write a concise Small-LI current-protocol dataset comparison note."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


FEATURE_ORDER = (
    "raw",
    "morph",
    "raw+morph",
    "embedding",
    "embedding+raw",
    "embedding+raw+morph",
)


def load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def run_by_feature(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {row["features"]: row for row in payload.get("runs", [])}


def hi_row(rows: Iterable[Dict[str, Any]], label: str, feature: str) -> Optional[Dict[str, Any]]:
    for row in rows:
        if row.get("run_label") == label and row.get("features") == feature:
            return row
    return None


def metric_cell(row: Optional[Dict[str, Any]]) -> str:
    if not row:
        return "—"
    test = row.get("test", row)
    f1_05 = test.get("f1_at_0_5", row.get("f1_at_0_5"))
    return (
        f"{test.get('auroc', float('nan')):.3f} / "
        f"{test.get('auprc', float('nan')):.3f} / "
        f"{test.get('f1', float('nan')):.3f}"
        + (f" / {f1_05:.3f}" if f1_05 is not None else "")
    )


def positive_rate(audit: Dict[str, Any], split: str) -> float:
    return float(audit["splits"][split]["positive_rate"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit_json",
        default="results/diagnostics/small_li_dataset_audit.json",
    )
    parser.add_argument(
        "--small_li_json",
        default="results/diagnostics/probe_feature_ablation_small_li_current_protocol_seed1.json",
    )
    parser.add_argument(
        "--small_li_none_json",
        default="results/diagnostics/probe_feature_ablation_small_li_current_protocol_seed1_cw_none.json",
    )
    parser.add_argument(
        "--small_hi_json",
        default="results/diagnostics/probe_feature_ablation_current_protocol_comparison.json",
    )
    parser.add_argument(
        "--output_md",
        default="notes/small_li_current_protocol_comparison.md",
    )
    args = parser.parse_args()

    audit = load_json(Path(args.audit_json))
    small_li = load_json(Path(args.small_li_json))
    small_hi = load_json(Path(args.small_hi_json))
    small_li_none = (
        load_json(Path(args.small_li_none_json))
        if Path(args.small_li_none_json).is_file()
        else None
    )

    li = run_by_feature(small_li)
    hi_rows = small_hi.get("rows", [])
    hi_gin_embed = hi_row(hi_rows, "GINe emlps+tds seed1 (20ep)", "embedding")
    hi_gin_full = hi_row(hi_rows, "GINe emlps+tds seed1 (20ep)", "embedding+raw+morph")
    hi_fnf_full = hi_row(hi_rows, "FNF + emlps+tds seed1", "embedding+raw+morph")
    hi_raw_morph = hi_row(hi_rows, "GINe emlps+tds seed1 (20ep)", "raw+morph")

    embed = li.get("embedding")
    raw_morph = li.get("raw+morph")
    emb_raw = li.get("embedding+raw")
    emb_full = li.get("embedding+raw+morph")

    def f1(feature: str) -> Optional[float]:
        row = li.get(feature)
        return None if not row else float(row["test"]["f1"])

    interpretation = []
    li_test_prev = positive_rate(audit, "test")
    hi_test_prev = 0.0018666033265165429
    if li_test_prev < hi_test_prev:
        interpretation.append(
            "Small-LI has lower test positive prevalence than the Small-HI reference "
            f"({li_test_prev:.4%} vs {hi_test_prev:.4%}), so AUPRC/F1 should be read with that class-balance shift in mind."
        )
    else:
        interpretation.append(
            "Small-LI has equal or higher test positive prevalence than the Small-HI reference, "
            "so metric differences are not just from rarer positives."
        )
    if f1("embedding") is not None and f1("raw+morph") is not None:
        if f1("embedding") > f1("raw+morph"):
            interpretation.append("Frozen SSL embeddings beat the raw+morph-only baseline on F1.")
        else:
            interpretation.append("Raw+morph-only matches or beats embedding-only on F1; inspect whether SSL transfer is weaker on Small-LI.")
    if f1("embedding+raw") is not None and f1("embedding") is not None:
        raw_delta = f1("embedding+raw") - f1("embedding")
        interpretation.append(f"`embedding+raw` changes F1 vs embedding-only by {raw_delta:+.3f}.")
    if f1("embedding+raw+morph") is not None and f1("embedding+raw") is not None:
        morph_delta = f1("embedding+raw+morph") - f1("embedding+raw")
        interpretation.append(f"Adding morphology on top of embedding+raw changes F1 by {morph_delta:+.3f}.")

    lines = [
        "# Small-LI Current-Protocol Dataset Comparison",
        "",
        "Small-LI scout for the current GINe emlps+tds SSL recipe. This is a dataset comparison, not a frozen benchmark.",
        "",
        "## Dataset Audit",
        "",
        f"- Dataset key: `{audit['dataset_key']}`",
        f"- CSV: `{audit['csv_path']}`",
        f"- Edges / nodes: {audit['total_edges']:,} / {audit['n_nodes']:,}",
        f"- Label: `{audit['dataset_spec']['label_col']}`; overall positive rate {audit['overall_positive_rate']:.4%}",
        f"- Split mode: `{audit['dataset_spec']['split_mode']}`; train/val/test positive rates "
        f"{positive_rate(audit, 'train'):.4%} / {positive_rate(audit, 'val'):.4%} / {positive_rate(audit, 'test'):.4%}",
        f"- Pattern metadata: {audit['pattern_metadata']['status']}",
        f"- Raw/morph generation: {audit['feature_generation_check']['status']}",
        "",
        "## Small-LI Probes",
        "",
        "Primary probe policy: `--class_weight model --model gin` (shared GIN weights). Metrics are test AUROC / AUPRC / F1 / F1@0.5.",
        "",
        "| Features | Metrics | Threshold | Precision | Recall |",
        "|----------|---------|----------:|----------:|-------:|",
    ]
    for feature in FEATURE_ORDER:
        row = li.get(feature)
        if not row:
            continue
        test = row["test"]
        lines.append(
            f"| `{feature}` | {metric_cell(row)} | {test['threshold']:.3f} | "
            f"{test['precision']:.3f} | {test['recall']:.3f} |"
        )

    if small_li_none:
        lines.extend(
            [
                "",
                "Exploratory secondary probe (`cw=none`) is saved separately at "
                "`results/diagnostics/probe_feature_ablation_small_li_current_protocol_seed1_cw_none.json`.",
            ]
        )

    lines.extend(
        [
            "",
            "## Small-HI References",
            "",
            "| Reference | Metrics |",
            "|-----------|---------|",
            f"| Small-HI GINe emlps+tds 20ep seed1 `embedding` | {metric_cell(hi_gin_embed)} |",
            f"| Small-HI GINe emlps+tds 20ep seed1 `embedding+raw+morph` | {metric_cell(hi_gin_full)} |",
            f"| Small-HI FNF seed1 `embedding+raw+morph` | {metric_cell(hi_fnf_full)} |",
            f"| Small-HI `raw+morph` only | {metric_cell(hi_raw_morph)} |",
            "",
            "## Interpretation",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in interpretation)
    lines.extend(
        [
            "",
            "Artifacts:",
            "",
            "- Audit: `results/diagnostics/small_li_dataset_audit.json`",
            "- Embeddings: `embeddings/small_li_gin_emlps_tds_asym_proj_8192neg_queue0_20ep_seed1/`",
            "- Feature ablation: `results/diagnostics/probe_feature_ablation_small_li_current_protocol_seed1.json`",
            "",
        ]
    )

    out = Path(args.output_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
