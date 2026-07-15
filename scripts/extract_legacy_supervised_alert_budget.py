#!/usr/bin/env python3
"""Extract legacy supervised alert-budget from an existing eval JSON (no re-eval)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_json",
        default=str(root / "results/diagnostics/eval_small_li_legacy_supervised_gin_emlps_tds_100ep_seed1.json"),
    )
    parser.add_argument(
        "--output_json",
        default=str(root / "results/diagnostics/small_li_legacy_supervised_alert_budget_seed1.json"),
    )
    parser.add_argument(
        "--output_md",
        default=str(root / "notes/small_li_legacy_supervised_alert_budget_seed1.md"),
    )
    args = parser.parse_args()

    def _resolve(path_str: str) -> Path:
        p = Path(path_str)
        if not p.is_absolute():
            p = root / p
        return p.resolve()

    eval_path = _resolve(args.eval_json)
    out_path = _resolve(args.output_json)
    note_path = _resolve(args.output_md)

    with eval_path.open(encoding="utf-8") as f:
        ev = json.load(f)
    test = ev["splits"]["test"]
    alert = test["alert_budget"]
    dataset = ev.get("data", "unknown")
    out = {
        "run_name": ev["run_name"],
        "data": dataset,
        "checkpoint_path": ev["checkpoint_path"],
        "checkpoint_epoch": ev["checkpoint_epoch"],
        "checkpoint_policy": "best_val_f1 (NOT last checkpoint)",
        "split": "test",
        "n": test["n"],
        "positive_rate": test["positive_rate"],
        "ranking_source": "model logits / softmax scores (same eval run as paper_argmax)",
        "alert_budget": alert,
        "provenance": {
            "source_json": str(eval_path.relative_to(root)),
            "note": "Extracted from eval JSON; no cluster re-evaluation required.",
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    note_path.write_text(
        "\n".join([
            f"# {dataset} legacy supervised alert-budget (seed 1, formal 100ep)",
            "",
            f"**Checkpoint:** `{ev['checkpoint_path']}` @ epoch **{ev['checkpoint_epoch']}**",
            "",
            "Metrics extracted from existing eval JSON — no model re-run.",
            "",
            f"**Source:** `{eval_path.relative_to(root)}`",
            "",
            "## Test split alert-budget",
            "",
            "| K | Precision | Recall | Lift |",
            "|---|-----------|--------|------|",
            f"| 100 | {alert['precision_at_100']} | {alert['recall_at_100']} | {alert['lift_at_100']:.2f} |",
            f"| 500 | {alert['precision_at_500']} | {alert['recall_at_500']} | {alert['lift_at_500']:.2f} |",
            f"| 1000 | {alert['precision_at_1000']} | {alert['recall_at_1000']} | {alert['lift_at_1000']:.2f} |",
            "",
            "## Caveats",
            "",
            f"- Uses **best-validation** checkpoint (epoch {ev['checkpoint_epoch']}), not last epoch.",
            "- Decision/ranking from **in-GNN supervised** model, not frozen SSL + linear probe.",
            "- Comparable to SSL alert-budget only at ranking semantics level, not protocol level.",
        ]),
        encoding="utf-8",
    )
    print(f"Wrote {out_path} and {note_path}")


if __name__ == "__main__":
    main()
