#!/usr/bin/env python3
"""Submit four independent Phase-4C full-training jobs (no array, no deps)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phase4c_four_domain import ARMS, arm_ckpt_root, arm_result_root, resolved_recipe  # noqa: E402
from phase4c_four_domain.paths import assert_disjoint  # noqa: E402
from phase4c_four_domain.source_manifest import verify_manifest  # noqa: E402

ARMS_ORDER = [
    "FOUR_DOMAIN_INFONCE_TF_ADAPTIVE_SHORT",
    "FOUR_DOMAIN_PROJECTION_INFONCE_TF_ADAPTIVE_SHORT",
    "FOUR_DOMAIN_EXPERT_ONLY_SHORT",
    "FOUR_DOMAIN_PROJECTION_INFONCE_TF_ADAPTIVE_LONG",
]
SHORT = {
    "FOUR_DOMAIN_INFONCE_TF_ADAPTIVE_SHORT": "infonce_short",
    "FOUR_DOMAIN_PROJECTION_INFONCE_TF_ADAPTIVE_SHORT": "proj_short",
    "FOUR_DOMAIN_EXPERT_ONLY_SHORT": "expert_short",
    "FOUR_DOMAIN_PROJECTION_INFONCE_TF_ADAPTIVE_LONG": "proj_long",
}
EXPECTED_SHA = "7dcfaa52d38abd6e929633c028b2e2a21743385d26c6b2cdbd34e87b2f42d3aa"
APPROVED = ROOT / "results/diagnostics/phase4c_four_domain_source_manifest.approved.json"
AUTH = ROOT / "results/diagnostics/phase4c_four_domain_cross_arm_replay_authorization_v1.json"
OUT = ROOT / "results/diagnostics/phase4c_four_domain_full_train_submission_independent.json"
WRAPPER = ROOT / "slurm/run_phase4c_four_domain_train_arm.sh"


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sbatch(args: list[str], *, dry: bool) -> str:
    cmd = ["sbatch", "--parsable", *args]
    print("CMD:", " ".join(cmd), flush=True)
    if dry:
        return "0"
    out = subprocess.check_output(cmd, cwd=str(ROOT), text=True, stderr=subprocess.STDOUT).strip()
    job = out.split(";")[0].strip()
    if not re.fullmatch(r"\d+", job):
        raise RuntimeError(f"non-numeric sbatch id: {out!r}")
    return job


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    os.chdir(ROOT)

    man_sha = sha_file(APPROVED)
    if man_sha != EXPECTED_SHA:
        raise SystemExit(f"approved manifest SHA mismatch: {man_sha}")
    verify_manifest(APPROVED)
    auth = json.loads(AUTH.read_text(encoding="utf-8"))
    if auth.get("authorization_verdict") != "FULL_TRAINING_AUTHORIZED":
        raise SystemExit("authorization sidecar not FULL_TRAINING_AUTHORIZED")
    if auth.get("new_source_manifest_sha256") != EXPECTED_SHA:
        raise SystemExit("authorization sidecar manifest SHA mismatch")
    if auth.get("full_training_authorized") is not True:
        raise SystemExit("full_training_authorized is not true")
    assert_disjoint()

    jobs = {}
    manual = []
    for arm in ARMS_ORDER:
        if arm not in ARMS:
            raise SystemExit(f"unknown arm {arm}")
        recipe = resolved_recipe(arm)
        result = ROOT / arm_result_root(arm)
        ckpt = ROOT / arm_ckpt_root(arm)
        smoke = result / "smoke_summary.json"
        if not smoke.is_file():
            raise SystemExit(f"missing smoke_summary for {arm}")
        smoke_blob = json.loads(smoke.read_text(encoding="utf-8"))
        if smoke_blob.get("resolved") != recipe:
            raise SystemExit(f"smoke resolved recipe mismatch for {arm}")
        if list(ckpt.glob("checkpoint_step_*.tar")):
            raise SystemExit(f"refusing non-empty full ckpt dir: {ckpt}")
        if (result / "train_summary.json").is_file():
            raise SystemExit(f"refusing existing train_summary.json under {result}")
        job_name = f"p4c_tr_{SHORT[arm]}"
        sbatch_args = [
            f"--job-name={job_name}",
            f"--export=ALL,ARM={arm}",
            str(WRAPPER.relative_to(ROOT)),
        ]
        manual.append(
            f"cd {ROOT} && sbatch --parsable --job-name={job_name} "
            f"--export=ALL,ARM={arm} {WRAPPER.relative_to(ROOT)}"
        )
        try:
            job_id = sbatch(sbatch_args, dry=args.dry_run)
        except (subprocess.CalledProcessError, FileNotFoundError, OSError) as exc:
            print("SBATCH_FAILED:", exc, flush=True)
            if isinstance(exc, subprocess.CalledProcessError) and exc.output:
                print(exc.output, flush=True)
            print("MANUAL_SBATCH_COMMANDS:", flush=True)
            for line in manual:
                print(line, flush=True)
            # Print remaining arms too.
            for rest in ARMS_ORDER[ARMS_ORDER.index(arm) + 1 :]:
                print(
                    f"cd {ROOT} && sbatch --parsable --job-name=p4c_tr_{SHORT[rest]} "
                    f"--export=ALL,ARM={rest} {WRAPPER.relative_to(ROOT)}",
                    flush=True,
                )
            return 3
        jobs[arm] = {
            "job_id": job_id,
            "job_name": job_name,
            "result_root": arm_result_root(arm),
            "ckpt_root": arm_ckpt_root(arm),
            "max_optimizer_steps": recipe["max_optimizer_steps"],
            "warmup_steps": recipe["warmup_steps"],
            "linear_decay_steps": recipe["linear_decay_steps"],
            "checkpoint_steps": recipe["checkpoint_steps"],
            "projection": recipe["projection"],
            "weight_mode": recipe["weight_mode"],
            "dependency": None,
        }

    payload = {
        "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "four_independent_full_trains_only",
        "approved_manifest_sha256": man_sha,
        "authorization_sidecar": str(AUTH.relative_to(ROOT)),
        "authorization_verdict": "FULL_TRAINING_AUTHORIZED",
        "no_job_array": True,
        "no_cross_arm_dependencies": True,
        "no_extract_probe": True,
        "no_test_eval": True,
        "rolling_checkpoint_every_100": False,
        "rolling_checkpoint_note": "not already supported in authorized train loop; milestone ckpts only",
        "slurm_flags": {
            "partition": "mit_normal_gpu",
            "account": "mit_amf_advanced_gpu",
            "qos": "mit_amf_advanced_gpu",
            "gres": "gpu:1",
            "cpus": 16,
            "mem": "128G",
            "time": "06:00:00",
            "workers": 0,
        },
        "jobs": jobs,
        "dry_run": bool(args.dry_run),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"submission": str(OUT), "jobs": {a: j["job_id"] for a, j in jobs.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
