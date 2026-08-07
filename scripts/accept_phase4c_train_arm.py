#!/usr/bin/env python3
"""Artifact-only post-train integrity acceptance for one Phase-4C arm."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phase4c_four_domain import ARMS, DOMAINS, resolved_recipe, arm_ckpt_root, arm_result_root  # noqa: E402

APPROVED = ROOT / "results/diagnostics/phase4c_four_domain_source_manifest.approved.json"


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=tuple(ARMS))
    ap.add_argument("--train-job-id", default=os.environ.get("TRAIN_JOB_ID"))
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("Refuse: train integrity gate requires SLURM_JOB_ID")
    arm = args.arm
    recipe = resolved_recipe(arm)
    result_dir = ROOT / arm_result_root(arm)
    ckpt_dir = ROOT / arm_ckpt_root(arm)
    summary_path = result_dir / "train_summary.json"
    failure_path = result_dir / "failure.json"
    out_path = result_dir / "train_acceptance.json"
    checks = {}
    if not summary_path.is_file():
        print("GATE_FAIL: missing train_summary.json", file=sys.stderr)
        return 1
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if failure_path.is_file() and failure_path.stat().st_mtime >= summary_path.stat().st_mtime:
        if json.loads(failure_path.read_text()).get("mode") == "full":
            print("GATE_FAIL: failure.json supersedes train_summary.json", file=sys.stderr)
            return 1
    expected_steps = int(recipe["max_optimizer_steps"])
    checks["ok"] = summary.get("ok") is True
    checks["end"] = summary.get("end") == "SUCCESS"
    checks["mode_full"] = summary.get("mode") == "full"
    checks["arm"] = summary.get("arm") == arm
    checks["steps"] = int(summary.get("steps", -1)) == expected_steps
    checks["optimizer"] = int(summary.get("optimizer_steps", -1)) == expected_steps
    checks["scheduler"] = int(summary.get("scheduler_steps", -1)) == expected_steps
    exposures = summary.get("exposures") or {}
    per = expected_steps // len(DOMAINS)
    checks["balanced_exposures"] = all(int(exposures.get(d, -1)) == per for d in DOMAINS)
    checks["resolved"] = summary.get("resolved") == recipe
    checks["source_sha"] = summary.get("source_manifest_sha256") == _sha_file(APPROVED)
    checks["gates_ok"] = bool((summary.get("gates") or {}).get("ok"))
    checks["no_test"] = summary.get("test_evaluated") is False
    for step in recipe["checkpoint_steps"]:
        p = ckpt_dir / f"checkpoint_step_{int(step):04d}.tar"
        checks[f"ckpt_{step}"] = p.is_file()
    log_ok = False
    if args.train_job_id:
        matches = list((ROOT / "slurm-logs").glob(f"*_{args.train_job_id}.out"))
        if matches:
            text = matches[0].read_text(encoding="utf-8", errors="replace")
            log_ok = f"end=SUCCESS arm={arm}" in text or "end=SUCCESS" in text
    checks["stdout_end_marker"] = log_ok
    ok = all(bool(v) for v in checks.values())
    payload = {"ok": ok, "arm": arm, "checks": checks, "end": "SUCCESS" if ok else "FAILED"}
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, out_path)
    print(json.dumps(payload, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
