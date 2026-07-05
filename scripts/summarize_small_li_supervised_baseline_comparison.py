#!/usr/bin/env python3
"""Summarize Small-LI supervised baseline against SSL scout results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

SSL_DEFAULT = {
    "raw+morph": {"auroc": 0.858, "auprc": 0.016, "f1": 0.057, "f1_at_0_5": 0.050},
    "embedding": {"auroc": 0.899, "auprc": 0.017, "f1": 0.052, "f1_at_0_5": 0.052},
    "embedding+raw": {"auroc": 0.909, "auprc": 0.027, "f1": 0.076, "f1_at_0_5": 0.081},
    "embedding+raw+morph": {"auroc": 0.925, "auprc": 0.039, "f1": 0.056, "f1_at_0_5": 0.073},
}


def load_json(path: Path) -> Dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_note(supervised: Dict, output_md: Path) -> None:
    test = supervised["splits"]["test"]
    val = supervised["splits"]["val"]
    best_log = (supervised.get("training_log") or {}).get("best_by_val_f1_argmax")
    lines = [
        "# Small-LI Supervised Baseline Comparison",
        "",
        "Small-LI comparison run using labels through the repo's canonical supervised GINe CE path. Report separately from SSL probes.",
        "",
        f"- **run:** `{supervised['run_name']}`",
        f"- **checkpoint epoch:** {supervised['checkpoint_epoch']} (canonical final checkpoint)",
        f"- **CE class weights:** `{supervised['ce_class_weight']}`",
        f"- **val-tuned threshold:** {supervised['classification_threshold']['value']:.4f}",
        "",
        "## Supervised Final Checkpoint",
        "",
        "| Split | AUROC | AUPRC | F1 | F1@0.5 | Precision | Recall | P@500 | R@500 | lift@500 |",
        "|-------|------:|------:|---:|-------:|----------:|-------:|------:|------:|---------:|",
    ]
    for split in ("train", "val", "test"):
        row = supervised["splits"][split]
        lines.append(
            f"| {split} | {row['auroc']:.4f} | {row['auprc']:.4f} | {row['f1']:.4f} | "
            f"{row['f1_at_0_5']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row.get('precision_at_500', float('nan')):.4f} | {row.get('recall_at_500', float('nan')):.4f} | "
            f"{row.get('lift_at_500', float('nan')):.1f} |"
        )
    if best_log:
        lines.extend(
            [
                "",
                f"Best epoch in the canonical training log by argmax Validation F1: **{best_log.get('epoch_index')}** "
                f"(val {best_log.get('val_f1_argmax'):.4f}, test {best_log.get('test_f1_argmax', float('nan')):.4f}). "
                "The saved checkpoint evaluated above is the final canonical checkpoint.",
            ]
        )
    lines.extend(
        [
            "",
            "## SSL Scout Reference",
            "",
            "| Small-LI SSL / feature stack | AUROC | AUPRC | F1 | F1@0.5 |",
            "|------------------------------|------:|------:|---:|-------:|",
        ]
    )
    for name, row in SSL_DEFAULT.items():
        lines.append(
            f"| `{name}` | {row['auroc']:.3f} | {row['auprc']:.3f} | {row['f1']:.3f} | {row['f1_at_0_5']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Placeholder",
            "",
            f"- Supervised test metrics: AUROC {test['auroc']:.4f}, AUPRC {test['auprc']:.4f}, F1 {test['f1']:.4f}.",
            f"- Validation F1 at selected threshold: {val['f1']:.4f}. Compare to the SSL scout after the probe sweep finishes.",
            "",
            "Artifacts:",
            "",
            "- JSON: `results/diagnostics/supervised_small_li_gin_emlps_tds_seed1.json`",
            "- Markdown: `results/diagnostics/supervised_small_li_gin_emlps_tds_seed1.md`",
            "",
        ]
    )
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--supervised_json",
        default="results/diagnostics/supervised_small_li_gin_emlps_tds_seed1.json",
    )
    parser.add_argument(
        "--output_md",
        default="notes/small_li_supervised_baseline_comparison.md",
    )
    args = parser.parse_args()
    write_note(load_json(Path(args.supervised_json)), Path(args.output_md))
    print(args.output_md)


if __name__ == "__main__":
    main()
