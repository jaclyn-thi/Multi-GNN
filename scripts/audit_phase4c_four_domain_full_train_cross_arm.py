#!/usr/bin/env python3
"""Post-hoc cross-arm audit for Phase-4C full-training artifacts (login-safe).

Reads only written train summaries / step logs. Never loads graphs or submits jobs.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phase4c_four_domain import arm_result_root  # noqa: E402

ARMS_ORDER = [
    "FOUR_DOMAIN_INFONCE_TF_ADAPTIVE_SHORT",
    "FOUR_DOMAIN_PROJECTION_INFONCE_TF_ADAPTIVE_SHORT",
    "FOUR_DOMAIN_EXPERT_ONLY_SHORT",
    "FOUR_DOMAIN_PROJECTION_INFONCE_TF_ADAPTIVE_LONG",
]
CONTRASTIVE = [
    "FOUR_DOMAIN_INFONCE_TF_ADAPTIVE_SHORT",
    "FOUR_DOMAIN_PROJECTION_INFONCE_TF_ADAPTIVE_SHORT",
    "FOUR_DOMAIN_PROJECTION_INFONCE_TF_ADAPTIVE_LONG",
]


def _load_rows(arm: str) -> Optional[List[Dict[str, Any]]]:
    result = ROOT / arm_result_root(arm)
    # Prefer full step dump if present; else reconstruct from head/tail only (partial).
    for name in ("steps_full.json", "train_steps.json", "steps.json"):
        path = result / name
        if path.is_file():
            blob = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(blob, list):
                return blob
            for key in ("rows", "steps", "all"):
                if isinstance(blob.get(key), list):
                    return blob[key]
    head = result / "steps_head.json"
    summary = result / "train_summary.json"
    if not summary.is_file():
        return None
    if head.is_file():
        blob = json.loads(head.read_text(encoding="utf-8"))
        rows = list(blob.get("head") or []) + list(blob.get("tail") or [])
        # Deduplicate by step.
        by = {int(r["global_optimizer_step"]): r for r in rows}
        return [by[k] for k in sorted(by)]
    return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/diagnostics/phase4c_four_domain_full_train_cross_arm_audit.json",
    )
    args = ap.parse_args()
    per_arm: Dict[str, Any] = {}
    rows_by: Dict[str, List[Dict[str, Any]]] = {}
    for arm in ARMS_ORDER:
        summary_path = ROOT / arm_result_root(arm) / "train_summary.json"
        if not summary_path.is_file():
            per_arm[arm] = {"present": False, "ok": False, "reason": "missing_train_summary"}
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows = _load_rows(arm) or []
        rows_by[arm] = rows
        per_arm[arm] = {
            "present": True,
            "ok": bool(summary.get("ok")) and summary.get("end") == "SUCCESS",
            "steps": summary.get("steps"),
            "optimizer_steps": summary.get("optimizer_steps"),
            "exposures": summary.get("exposures"),
            "gates": summary.get("gates"),
            "alpha_unfrozen_at_completed": summary.get("alpha_unfrozen_at_completed"),
            "projection_present": summary.get("projection_present"),
            "contrastive_optimization_active": summary.get("contrastive_optimization_active"),
            "expert_alpha_absent_from_optimizer": summary.get("expert_alpha_absent_from_optimizer"),
            "test_evaluated": summary.get("test_evaluated"),
            "n_hash_rows_available": len(rows),
        }

    def overlap_map(arms: List[str], key: str, max_step: int) -> Dict[str, Any]:
        available = [a for a in arms if a in rows_by and rows_by[a]]
        if len(available) < 2:
            return {"compared": False, "reason": "insufficient_arms_with_rows"}
        by_arm = {
            a: {
                int(r["global_optimizer_step"]): r.get(key)
                for r in rows_by[a]
                if int(r["global_optimizer_step"]) <= max_step
            }
            for a in available
        }
        steps = sorted(set.intersection(*(set(m) for m in by_arm.values())))
        mismatches = []
        for step in steps:
            vals = {a: by_arm[a][step] for a in available}
            if len(set(vals.values())) != 1:
                mismatches.append({"step": step, "values": vals})
        return {
            "compared": True,
            "arms": available,
            "max_step": max_step,
            "n_steps_compared": len(steps),
            "all_match": len(mismatches) == 0,
            "mismatches_head": mismatches[:10],
        }

    audit = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "per_arm": per_arm,
        "cross_arm": {
            "seed_ids_all_arms_through_4000": overlap_map(ARMS_ORDER, "seed_ids_sha256", 4000),
            "view1_all_arms_through_4000": overlap_map(ARMS_ORDER, "view1_aug_sha256", 4000),
            "view2_contrastive_arms_through_4000": overlap_map(
                CONTRASTIVE, "view2_aug_sha256", 4000
            ),
            "proj_short_vs_long_through_4000_view1": overlap_map(
                [
                    "FOUR_DOMAIN_PROJECTION_INFONCE_TF_ADAPTIVE_SHORT",
                    "FOUR_DOMAIN_PROJECTION_INFONCE_TF_ADAPTIVE_LONG",
                ],
                "view1_aug_sha256",
                4000,
            ),
            "infonce_vs_proj_short_through_4000_view1": overlap_map(
                [
                    "FOUR_DOMAIN_INFONCE_TF_ADAPTIVE_SHORT",
                    "FOUR_DOMAIN_PROJECTION_INFONCE_TF_ADAPTIVE_SHORT",
                ],
                "view1_aug_sha256",
                4000,
            ),
            "infonce_vs_expert_view1_through_4000": overlap_map(
                [
                    "FOUR_DOMAIN_INFONCE_TF_ADAPTIVE_SHORT",
                    "FOUR_DOMAIN_EXPERT_ONLY_SHORT",
                ],
                "view1_aug_sha256",
                4000,
            ),
        },
        "note": (
            "EXPERT SHORT is not failed for lacking contrastive view2 semantics; "
            "view1 comparisons remain required. Full step dumps may be limited to "
            "steps_head.json head/tail unless train writes a full step log."
        ),
        "extraction_submitted": False,
        "probes_submitted": False,
        "test_evaluated": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "per_arm_ok": {a: per_arm[a].get("ok") for a in ARMS_ORDER}}, indent=2))
    missing = [a for a, v in per_arm.items() if not v.get("present")]
    failed = [a for a, v in per_arm.items() if v.get("present") and not v.get("ok")]
    if missing:
        print("MISSING:", missing)
        return 2
    if failed:
        print("FAILED:", failed)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
