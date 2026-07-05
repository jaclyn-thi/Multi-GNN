#!/usr/bin/env python3
"""Re-run linear_probe.py on a list of runs and emit an AUPRC summary JSON."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--skip_existing_auprc",
        action="store_true",
        help="Skip linear_probe when probe_results.json already has test auprc.",
    )
    args = parser.parse_args()

    rows = []
    for name in args.runs:
        emb = ROOT / "embeddings" / name
        if not (emb / "test.npz").is_file():
            print(f"SKIP (no npz): {name}")
            continue
        probe_path = emb / "probe_results.json"
        needs_probe = True
        if args.skip_existing_auprc and probe_path.is_file():
            with probe_path.open(encoding="utf-8") as f:
                existing = json.load(f)
            test_existing = existing.get("splits_at_selected_threshold", {}).get("test", {})
            if test_existing.get("auprc") is not None:
                needs_probe = False
                res = existing
                print(f"SKIP (has auprc): {name}")
        if needs_probe:
            subprocess.run(
                [
                    args.python,
                    str(ROOT / "linear_probe.py"),
                    "--unique_name",
                    name,
                    "--class_weight",
                    "model",
                    "--model",
                    "gin",
                    "--testing",
                ],
                cwd=ROOT,
                check=True,
            )
            with probe_path.open(encoding="utf-8") as f:
                res = json.load(f)
        test = res["splits_at_selected_threshold"]["test"]
        row = {
            "unique_name": name,
            "test_auroc": test["auroc"],
            "test_auprc": test["auprc"],
            "test_f1": test["f1"],
            "test_precision": test["precision"],
            "test_recall": test["recall"],
            "threshold": res["classification_threshold"]["value"],
        }
        rows.append(row)
        print(
            f"{name}: AUROC={row['test_auroc']:.4f} AUPRC={row['test_auprc']:.4f} "
            f"F1={row['test_f1']:.4f} R={row['test_recall']:.4f}"
        )

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump({"runs": rows}, f, indent=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
