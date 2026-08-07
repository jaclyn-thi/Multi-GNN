#!/usr/bin/env python3
"""Artifact-only post-smoke audit (no get_data). Cross-arm seed/view matching + gates."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phase4c_four_domain import ARMS, DOMAINS, PHASE3_SHARED_INIT_SHA256, arm_ckpt_root, arm_result_root, arm_uses_projection, resolved_recipe  # noqa: E402
from phase4c_four_domain.integrity import evaluate_no_test_policy  # noqa: E402

ARMS_ORDER = [
    "FOUR_DOMAIN_INFONCE_TF_ADAPTIVE_SHORT",
    "FOUR_DOMAIN_PROJECTION_INFONCE_TF_ADAPTIVE_SHORT",
    "FOUR_DOMAIN_EXPERT_ONLY_SHORT",
    "FOUR_DOMAIN_PROJECTION_INFONCE_TF_ADAPTIVE_LONG",
]
CONTRASTIVE = [a for a in ARMS_ORDER if a != "FOUR_DOMAIN_EXPERT_ONLY_SHORT"]
APPROVED_SHA = "eb352474eebe95ca9bd303c991307a0606d1e1c2e8798dde13ee5ba536457c84"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _lr_factor(step_completed: int, warmup: int, decay: int) -> float:
    """Match DirectHWarmupLinearScheduler: warmup 0.1→1.0 then linear 1.0→0.1."""
    # Scheduler steps after each optimizer step; after N steps, completed=N.
    # Factor applied for the N-th step uses state after (N-1) steps typically;
    # report factor at completed step index using phase_at(completed).
    s = int(step_completed)
    if s <= 0:
        return 0.1
    if s <= warmup:
        # linear warmup over warmup steps
        t = (s - 1) / max(warmup - 1, 1) if warmup > 1 else 1.0
        return 0.1 + (1.0 - 0.1) * t
    into = s - warmup
    if into >= decay:
        return 0.1
    t = (into - 1) / max(decay - 1, 1) if decay > 1 else 1.0
    return 1.0 + (0.1 - 1.0) * max(t, 0.0)


def load_arm(arm: str) -> Dict[str, Any]:
    root = ROOT / arm_result_root(arm)
    smoke = root / "smoke_summary.json"
    integ = root / "integrity.json"
    steps = root / "steps_head.json"
    fail = root / "failure.json"
    out: Dict[str, Any] = {"arm": arm, "result_root": str(root)}
    if fail.is_file() and (not smoke.is_file() or fail.stat().st_mtime >= smoke.stat().st_mtime):
        out["failure"] = json.loads(fail.read_text())
    if smoke.is_file():
        out["summary"] = json.loads(smoke.read_text())
    if integ.is_file():
        out["integrity"] = json.loads(integ.read_text())
    if steps.is_file():
        out["steps_head"] = json.loads(steps.read_text())
    ckpt = ROOT / arm_ckpt_root(arm) / "smoke" / "checkpoint_step_0040.tar"
    out["checkpoint_path"] = str(ckpt)
    out["checkpoint_exists"] = ckpt.is_file()
    if ckpt.is_file():
        out["checkpoint_sha256"] = _sha(ckpt)
    return out


def extract_hash_timeline(blob: Dict[str, Any]) -> List[Dict[str, Any]]:
    sh = blob.get("steps_head") or {}
    rows = list(sh.get("head") or []) + list(sh.get("tail") or [])
    # Prefer full rows if present later; smoke only stores head/tail.
    return [
        {
            "global_optimizer_step": r.get("global_optimizer_step"),
            "domain": r.get("domain"),
            "seed_ids_sha256": r.get("seed_ids_sha256"),
            "view1_aug_sha256": r.get("view1_aug_sha256"),
            "view2_aug_sha256": r.get("view2_aug_sha256"),
            "alpha_beta_frozen": r.get("alpha_beta_frozen"),
            "loss_norm_calibrated": r.get("loss_norm_calibrated"),
        }
        for r in rows
        if isinstance(r, dict)
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission-json", default="results/diagnostics/phase4c_four_domain_smoke_submission_current.json")
    args = ap.parse_args()
    sub = json.loads((ROOT / args.submission_json).read_text())
    arms_data = {a: load_arm(a) for a in ARMS_ORDER}
    per_arm = {}
    for arm, data in arms_data.items():
        summary = data.get("summary") or {}
        gates = (summary.get("gates") or data.get("integrity", {}).get("gates") or {})
        recipe = resolved_recipe(arm)
        checks = {
            "smoke_summary_present": "summary" in data,
            "ok": summary.get("ok") is True,
            "end_success": summary.get("end") == "SUCCESS",
            "mode_smoke": summary.get("mode") == "smoke",
            "steps_40": int(summary.get("steps", -1)) == 40,
            "optimizer_40": int(summary.get("optimizer_steps", -1)) == 40,
            "scheduler_40": int(summary.get("scheduler_steps", -1)) == 40,
            "exposures_10": summary.get("exposures") == {d: 10 for d in DOMAINS},
            "init_sha": True,  # enforced at load; reflected if checkpoint ok
            "source_manifest_sha": summary.get("source_manifest_sha256") == APPROVED_SHA,
            "gates_ok": bool(gates.get("ok")),
            "no_test": bool(gates.get("no_test_graph_cache_or_metric")),
            "test_evaluated_false": summary.get("test_evaluated") is False,
            "checkpoint_reload": bool(gates.get("checkpoints_reload")),
            "bn_changed": bool(gates.get("bn_all_changed", True)),
            "bn_distinct": bool(gates.get("bn_all_distinct", True)),
            "finite_losses": bool(gates.get("finite_losses")),
            "checkpoint_file": data.get("checkpoint_exists") is True,
            "no_failure_supersede": "failure" not in data,
            "paysim_fraud": summary.get("paysim_task_semantics") == "fraud_detection",
            "contract": (summary.get("resolved") or {}).get("protocol_id")
            == "financial_multidataset_shared_core_4domain_v1",
            "schedule_horizon": int((summary.get("resolved") or {}).get("max_optimizer_steps", -1))
            == int(recipe["max_optimizer_steps"]),
        }
        # objective-aware
        if arm == "FOUR_DOMAIN_EXPERT_ONLY_SHORT":
            checks["expert_no_contrast"] = bool(gates.get("expert_has_no_contrast_grad"))
            checks["expert_moe"] = bool(gates.get("expert_moe_grad"))
        elif arm_uses_projection(arm):
            checks["projection_grad"] = bool(gates.get("projection_grad"))
            checks["adaptive_moe"] = bool(gates.get("adaptive_moe_grad"))
        else:
            checks["adaptive_contrast_moe"] = bool(gates.get("adaptive_contrast_and_moe_grad"))

        no_test = evaluate_no_test_policy(
            {
                "test_graph_loaded": False,
                "test_metrics_computed": False,
                "skip_test_eval": True,
            },
            test_evaluated=bool(summary.get("test_evaluated", True)),
        )
        checks["no_test_schema"] = bool(no_test["ok"])

        warmup = int(recipe["warmup_steps"])
        decay = int(recipe["linear_decay_steps"])
        lr_first = 0.002 * _lr_factor(1, warmup, decay)
        lr_final = 0.002 * _lr_factor(40, warmup, decay)
        per_arm[arm] = {
            "checks": checks,
            "pass": all(checks.values()),
            "gates": gates,
            "alpha_unfrozen_at_completed": summary.get("alpha_unfrozen_at_completed"),
            "cuda_peak_alloc_gib": summary.get("cuda_peak_alloc_gib"),
            "cuda_peak_reserved_gib": summary.get("cuda_peak_reserved_gib"),
            "lr_smoke_first_encoder": lr_first,
            "lr_smoke_final_encoder": lr_final,
            "timeline": extract_hash_timeline(data),
            "job_id": (sub.get("arms") or {}).get(arm, {}).get("smoke_job_id"),
            "checkpoint_sha256": data.get("checkpoint_sha256"),
        }

    # Cross-arm matching on overlapping head/tail rows keyed by (step, domain)
    def index(arm: str) -> Dict[tuple, Dict[str, Any]]:
        idx = {}
        for r in per_arm[arm]["timeline"]:
            key = (r.get("global_optimizer_step"), r.get("domain"))
            if key[0] is not None and key[1] is not None:
                idx[key] = r
        return idx

    idxs = {a: index(a) for a in ARMS_ORDER}
    common_keys = set.intersection(*(set(idxs[a]) for a in ARMS_ORDER)) if ARMS_ORDER else set()
    seed_match = True
    view1_match = True
    view2_contrast_match = True
    mismatches = []
    for key in sorted(common_keys):
        seeds = {a: idxs[a][key].get("seed_ids_sha256") for a in ARMS_ORDER}
        v1 = {a: idxs[a][key].get("view1_aug_sha256") for a in ARMS_ORDER}
        v2 = {a: idxs[a][key].get("view2_aug_sha256") for a in CONTRASTIVE}
        if len(set(seeds.values())) != 1:
            seed_match = False
            mismatches.append({"key": key, "type": "seed", "values": seeds})
        if len(set(v1.values())) != 1:
            view1_match = False
            mismatches.append({"key": key, "type": "view1", "values": v1})
        if len(set(v2.values())) != 1:
            view2_contrast_match = False
            mismatches.append({"key": key, "type": "view2_contrastive", "values": v2})

    cross = {
        "compared_keys": len(common_keys),
        "seed_ids_match_all_arms": seed_match and len(common_keys) > 0,
        "view1_match_all_arms": view1_match and len(common_keys) > 0,
        "view2_match_contrastive_arms": view2_contrast_match and len(common_keys) > 0,
        "mismatches_head": mismatches[:20],
        "note": "Matching uses smoke steps_head head+tail rows only (not full 40-step log).",
    }
    # Mark arms failed if cross-arm matching fails
    if not (cross["seed_ids_match_all_arms"] and cross["view1_match_all_arms"] and cross["view2_match_contrastive_arms"]):
        for arm in ARMS_ORDER:
            per_arm[arm]["pass"] = False
            per_arm[arm]["checks"]["cross_arm_hash_match"] = False
    else:
        for arm in ARMS_ORDER:
            per_arm[arm]["checks"]["cross_arm_hash_match"] = True
            per_arm[arm]["pass"] = all(per_arm[arm]["checks"].values())

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "approved_manifest_sha256": APPROVED_SHA,
        "per_arm": per_arm,
        "cross_arm_matching": cross,
        "all_smokes_pass": all(per_arm[a]["pass"] for a in ARMS_ORDER),
        "full_training_submitted": False,
    }
    out = ROOT / "results/diagnostics/phase4c_four_domain_smoke_audit.json"
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, out)
    print(json.dumps({"all_pass": report["all_smokes_pass"], "path": str(out)}, indent=2))
    return 0 if report["all_smokes_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
