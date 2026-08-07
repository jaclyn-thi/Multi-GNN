#!/usr/bin/env python3
"""Small-HI morphology downstream probe runner (scaffold; fit gated).

Infrastructure task: default path refuses embedding load and probe fit.
Pass --execute-probe only in a later authorized run (NOT used now).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from morphology_downstream_probe.config import (
    BASELINES,
    ENCODER_SPECS,
    PROBE_CONFIG,
    RESULT_ROOT,
)
from morphology_downstream_probe.cohort import refuse_test_access
from morphology_downstream_probe.probe_regression import fit_paperstyle_regressor
from morphology_downstream_probe.residualize import DEFAULT_RESIDUALIZATION


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path(RESULT_ROOT) / "probe_plan.json")
    p.add_argument("--execute-probe", action="store_true")
    p.add_argument("--load-embeddings", action="store_true")
    p.add_argument("--include-test", action="store_true")
    p.add_argument("--encoder", choices=[e["name"] for e in ENCODER_SPECS] + ["all"], default="all")
    args = p.parse_args()

    if args.include_test:
        refuse_test_access("test")
    refuse_test_access()

    plan = {
        "status": "scaffold_only",
        "execute_probe": False,
        "load_embeddings": False,
        "encoders": [
            e for e in ENCODER_SPECS if args.encoder == "all" or e["name"] == args.encoder
        ],
        "baselines": list(BASELINES),
        "probe": PROBE_CONFIG,
        "residualization": DEFAULT_RESIDUALIZATION.to_dict(),
        "primary_claim": "frozen_r198_alone_vs_baselines",
        "supplemental_only": ["supplemental_r198_plus_degree"],
        "metrics": {
            "primary": "spearman_rho",
            "secondary": ["mae", "r2"],
            "vs_train_mean": True,
        },
        "max_cells": 8,
        "test_access": False,
        "overwrite_embeddings": False,
    }

    if args.load_embeddings or args.execute_probe:
        raise RuntimeError(
            "Refuse embedding load / probe execute in infrastructure-only task. "
            "Re-run later with authorization after triangle cache + ladder concurrency clear."
        )

    # Demonstrate gated fit API without executing.
    gated = fit_paperstyle_regressor(
        __import__("numpy").zeros((4, 198), dtype="float32"),
        __import__("numpy").zeros(4, dtype="float64"),
        __import__("numpy").zeros((2, 198), dtype="float32"),
        __import__("numpy").zeros(2, dtype="float64"),
        execute=False,
    )
    plan["fit_gate_demo"] = gated

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "out": str(args.out), "executed": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
