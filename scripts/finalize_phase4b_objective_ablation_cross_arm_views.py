#!/usr/bin/env python3
"""Cross-arm view-hash comparison for Phase-4B objective ablation training."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase4b_objective_ablation import RESULT_ROOT as TRAIN_RESULT_ROOT  # noqa: E402
from phase4b_objective_ablation_frozen_eval import RESULT_ROOT  # noqa: E402
from phase4b_objective_ablation_frozen_eval.views import compare_cross_arm_views  # noqa: E402


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    out_dir = ROOT / RESULT_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    train_root = ROOT / TRAIN_RESULT_ROOT
    report = compare_cross_arm_views()
    report["train_result_root"] = str(train_root)
    write_json(out_dir / "cross_arm_view_match.json", report)
    also = train_root / "cross_arm_view_match.json"
    write_json(also, report)
    print(json.dumps({"ok": report["ok"], "n_mismatches": len(report["mismatches"])}, indent=2))
    return 0 if report["ok"] or not report["missing_arms"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
