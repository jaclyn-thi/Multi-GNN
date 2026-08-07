#!/usr/bin/env python3
"""Submit four independent Phase-4C smoke jobs only (no DAG / no full training)."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
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
APPROVED = ROOT / "results/diagnostics/phase4c_four_domain_source_manifest.approved.json"
EXPECTED_SHA = "eb352474eebe95ca9bd303c991307a0606d1e1c2e8798dde13ee5ba536457c84"
OUT = ROOT / "results/diagnostics/phase4c_four_domain_smoke_submission_current.json"


def _atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass


def sbatch(args: list[str], *, dry: bool) -> str:
    cmd = ["sbatch", "--parsable", *args]
    print("CMD:", " ".join(cmd), flush=True)
    if dry:
        return "0"
    out = subprocess.check_output(cmd, cwd=str(ROOT), text=True).strip()
    job = out.split(";")[0].strip()
    if not re.fullmatch(r"\d+", job):
        raise RuntimeError(f"non-numeric sbatch id: {out!r}")
    return job


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    os.chdir(ROOT)
    import hashlib

    sha = hashlib.sha256(APPROVED.read_bytes()).hexdigest()
    if sha != EXPECTED_SHA:
        raise SystemExit(f"approved manifest SHA mismatch: {sha} != {EXPECTED_SHA}")
    verify_manifest(APPROVED)
    assert_disjoint()
    for arm in ARMS_ORDER:
        if arm not in ARMS:
            raise SystemExit(f"unknown arm {arm}")
        r = resolved_recipe(arm)
        print(arm, "horizon", r["max_optimizer_steps"], "warmup", r["warmup_steps"], "decay", r["linear_decay_steps"])

    payload = {
        "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
        "approved_manifest_sha256": sha,
        "scope": "four_independent_smokes_only",
        "no_full_training": True,
        "no_extract_probe_dag": True,
        "slurm_flags": {
            "partition": "mit_normal_gpu",
            "account": "mit_amf_advanced_gpu",
            "qos": "mit_amf_advanced_gpu",
            "gres": "gpu:1",
            "cpus": 16,
            "mem": "128G",
            "time": "02:00:00",
            "workers": 0,
        },
        "arms": {},
        "dry_run": bool(args.dry_run),
    }
    for arm in ARMS_ORDER:
        tag = SHORT[arm]
        job = sbatch(
            [
                f"--job-name=p4c_smoke_{tag}",
                f"--export=ALL,ARM={arm}",
                "slurm/run_phase4c_four_domain_smoke_arm.sh",
            ],
            dry=bool(args.dry_run),
        )
        payload["arms"][arm] = {
            "smoke_job_id": job if not args.dry_run else None,
            "dry_placeholder": job if args.dry_run else None,
            "result_root": arm_result_root(arm),
            "ckpt_root": arm_ckpt_root(arm),
            "smoke_summary": f"{arm_result_root(arm)}/smoke_summary.json",
            "integrity": f"{arm_result_root(arm)}/integrity.json",
            "recipe": {
                "max_optimizer_steps": resolved_recipe(arm)["max_optimizer_steps"],
                "warmup_steps": resolved_recipe(arm)["warmup_steps"],
                "linear_decay_steps": resolved_recipe(arm)["linear_decay_steps"],
                "smoke_steps": 40,
            },
        }
    _atomic(OUT, payload)
    print(f"wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
