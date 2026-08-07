#!/usr/bin/env python3
"""Artifact-only Phase-4C smoke acceptance gate (no get_data / graphs / loaders)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phase4c_four_domain import ARMS, DOMAINS, resolved_recipe  # noqa: E402
from phase4c_four_domain import arm_result_root  # noqa: E402

APPROVED = ROOT / "results/diagnostics/phase4c_four_domain_source_manifest.approved.json"
SHARED_MEM = ROOT / "results/diagnostics/phase4c_four_domain_shared_memory_preflight.json"


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def _fail(msg: str) -> int:
    print(f"GATE_FAIL: {msg}", file=sys.stderr)
    return 1


def _preflight_ok(result_dir: Path, arm: str) -> bool:
    pref = result_dir / "preflight.json"
    if pref.is_file():
        blob = _load(pref)
        return bool(blob.get("ok") and blob.get("end") == "SUCCESS" and blob.get("arm") == arm
                    and blob.get("mode") == "preflight")
    # Accept legacy CPU preflight that only wrote summary.json (mode=preflight).
    summary = result_dir / "summary.json"
    if summary.is_file():
        blob = _load(summary)
        return bool(blob.get("ok") and blob.get("mode") == "preflight" and blob.get("arm") == arm
                    and blob.get("end") in {"SUCCESS", None})
    return False


def accept_arm(arm: str, *, smoke_job_id: str | None, smoke_log: Path | None) -> int:
    if arm not in ARMS:
        return _fail(f"unknown arm {arm}")
    result_dir = ROOT / arm_result_root(arm)
    smoke_path = result_dir / "smoke_summary.json"
    failure_path = result_dir / "failure.json"
    out_path = result_dir / "smoke_acceptance.json"

    try:
        smoke = _load(smoke_path)
    except FileNotFoundError:
        return _fail(f"missing smoke artifact {smoke_path}")

    if failure_path.is_file():
        fail = _load(failure_path)
        if fail.get("mode") == "smoke" and failure_path.stat().st_mtime >= smoke_path.stat().st_mtime:
            return _fail("failure.json supersedes smoke_summary.json")

    recipe = resolved_recipe(arm)
    checks = {}

    checks["smoke_ok"] = bool(smoke.get("ok") is True)
    checks["smoke_end_success"] = smoke.get("end") == "SUCCESS"
    checks["smoke_mode"] = smoke.get("mode") == "smoke"
    checks["smoke_arm"] = smoke.get("arm") == arm
    checks["exact_40_steps"] = int(smoke.get("steps", -1)) == 40
    checks["exact_40_optimizer"] = int(smoke.get("optimizer_steps", -1)) == 40
    checks["exact_40_scheduler"] = int(smoke.get("scheduler_steps", -1)) == 40
    exposures = smoke.get("exposures") or {}
    checks["exact_10_per_domain"] = all(int(exposures.get(d, -1)) == 10 for d in DOMAINS)
    checks["four_domains"] = set(exposures) == set(DOMAINS)
    checks["resolved_recipe_match"] = smoke.get("resolved") == recipe

    if not APPROVED.is_file():
        return _fail(f"missing frozen approved manifest {APPROVED}")
    approved_sha = _sha_file(APPROVED)
    checks["source_sha_match"] = smoke.get("source_manifest_sha256") == approved_sha

    checks["preflight_passed"] = _preflight_ok(result_dir, arm)
    try:
        mem = _load(SHARED_MEM)
        checks["memory_preflight_passed"] = bool(
            mem.get("ok") and mem.get("end") == "SUCCESS" and mem.get("mode") == "memory_preflight"
        )
    except FileNotFoundError:
        checks["memory_preflight_passed"] = False

    gates = smoke.get("gates") or {}
    checks["objective_gates_ok"] = bool(gates.get("ok"))
    checks["checkpoint_reload"] = bool(gates.get("checkpoints_reload"))
    checks["finite_losses"] = bool(gates.get("finite_losses"))
    checks["required_gradients_finite"] = bool(gates.get("required_gradients_finite", True))
    checks["seed_view_hashes"] = bool(gates.get("seed_and_view_hashes_logged", True))
    checks["bn_changed"] = bool(gates.get("bn_all_changed"))
    checks["bn_distinct"] = bool(gates.get("bn_all_distinct"))
    checks["lossnorm_ok"] = bool(gates.get("lossnorm_calibrated", True))
    checks["alpha_beta_ok"] = bool(gates.get("alpha_frozen_through_20")) and bool(
        gates.get("first_alpha_update_is_21")
    )
    checks["no_test"] = bool(gates.get("no_test_graph_cache_or_metric")) and smoke.get("test_evaluated") is False

    log_path = smoke_log
    if log_path is None and smoke_job_id:
        cand = ROOT / "slurm-logs" / f"p4c_smoke_{smoke_job_id}.out"
        log_path = cand if cand.is_file() else None
        if log_path is None:
            # also try err-less alternate patterns
            matches = list((ROOT / "slurm-logs").glob(f"*_{smoke_job_id}.out"))
            log_path = matches[0] if matches else None
    if log_path is None or not Path(log_path).is_file():
        checks["stdout_end_marker"] = False
    else:
        text = Path(log_path).read_text(encoding="utf-8", errors="replace")
        checks["stdout_end_marker"] = f"end=SUCCESS arm={arm}" in text

    ok = all(bool(v) for v in checks.values())
    payload = {
        "ok": ok,
        "arm": arm,
        "checks": checks,
        "smoke_summary": str(smoke_path),
        "approved_manifest_sha256": approved_sha,
        "smoke_job_id": smoke_job_id,
        "smoke_log": str(log_path) if log_path else None,
        "end": "SUCCESS" if ok else "FAILED",
    }
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, out_path)
    print(json.dumps(payload, indent=2))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=tuple(ARMS))
    ap.add_argument("--smoke-job-id", default=os.environ.get("SMOKE_JOB_ID"))
    ap.add_argument("--smoke-log")
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("Refuse: smoke acceptance gate requires SLURM_JOB_ID")
    return accept_arm(
        args.arm,
        smoke_job_id=args.smoke_job_id,
        smoke_log=Path(args.smoke_log) if args.smoke_log else None,
    )


if __name__ == "__main__":
    raise SystemExit(main())
