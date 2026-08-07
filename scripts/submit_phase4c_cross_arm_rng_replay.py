#!/usr/bin/env python3
"""Freeze a new Phase-4C source manifest and submit the paired RNG replay job."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phase4c_four_domain.source_manifest import build_manifest, verify_manifest  # noqa: E402

APPROVED = ROOT / "results/diagnostics/phase4c_four_domain_source_manifest.approved.json"
REASON = "ISOLATE_PROJECTION_INITIALIZATION_FROM_AUGMENTATION_RNG"
OLD_EXPECTED = "eb352474eebe95ca9bd303c991307a0606d1e1c2e8798dde13ee5ba536457c84"


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    os.chdir(ROOT)
    if not APPROVED.is_file():
        raise SystemExit(f"missing approved manifest: {APPROVED}")
    old_sha = sha_file(APPROVED)
    if old_sha != OLD_EXPECTED:
        raise SystemExit(
            f"refusing to replace unexpected approved manifest SHA {old_sha} "
            f"(expected frozen {OLD_EXPECTED})"
        )
    # Preserve historical approved artifact unchanged under a versioned name.
    preserved = ROOT / (
        "results/diagnostics/"
        f"phase4c_four_domain_source_manifest.approved.pre_proj_rng_isolation_{old_sha[:12]}.json"
    )
    if not preserved.is_file():
        shutil.copy2(APPROVED, preserved)
        print(f"preserved_old_approved={preserved}")

    manifest = build_manifest()
    manifest["reason"] = REASON
    manifest["supersedes_source_manifest_sha256"] = old_sha
    manifest["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    new_path = ROOT / "results/diagnostics/phase4c_four_domain_source_manifest.json"
    new_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    # Replace approved pointer with the new freeze (historical smoke artifacts untouched).
    APPROVED.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    verify_manifest(APPROVED)
    new_sha = sha_file(APPROVED)
    freeze_sidecar = ROOT / (
        "results/diagnostics/phase4c_four_domain_source_manifest_isolate_projection_rng.json"
    )
    freeze_sidecar.write_text(
        json.dumps(
            {
                "reason": REASON,
                "old_source_manifest_sha256": old_sha,
                "new_source_manifest_sha256": new_sha,
                "preserved_old_approved_path": str(preserved),
                "approved_path": str(APPROVED),
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"old_sha={old_sha}")
    print(f"new_sha={new_sha}")
    print(f"freeze_sidecar={freeze_sidecar}")

    env = os.environ.copy()
    env["PHASE4C_OLD_MANIFEST_SHA"] = old_sha
    env["SOURCE_MANIFEST_PATH"] = str(APPROVED.relative_to(ROOT))
    cmd = [
        "sbatch",
        "--parsable",
        "--export=ALL,PHASE4C_OLD_MANIFEST_SHA={0},SOURCE_MANIFEST_PATH={1}".format(
            old_sha, str(APPROVED.relative_to(ROOT))
        ),
        "slurm/run_phase4c_cross_arm_rng_replay.sh",
    ]
    print("CMD:", " ".join(cmd), flush=True)
    try:
        out = subprocess.check_output(cmd, cwd=str(ROOT), text=True, env=env, stderr=subprocess.STDOUT)
        job = out.strip().split(";")[0].strip()
        if not re.fullmatch(r"\d+", job):
            raise RuntimeError(f"non-numeric sbatch id: {out!r}")
        print(f"job_id={job}")
        sub = {
            "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
            "job_id": job,
            "old_source_manifest_sha256": old_sha,
            "new_source_manifest_sha256": new_sha,
            "reason": REASON,
            "script": "slurm/run_phase4c_cross_arm_rng_replay.sh",
            "full_training_submitted": False,
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
        }
        (ROOT / "results/diagnostics/phase4c_four_domain_cross_arm_rng_replay_submission.json").write_text(
            json.dumps(sub, indent=2) + "\n", encoding="utf-8"
        )
        return 0
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as exc:
        print("SBATCH_FAILED:", exc, flush=True)
        if isinstance(exc, subprocess.CalledProcessError) and exc.output:
            print(exc.output, flush=True)
        print(
            "MANUAL_SBATCH_COMMAND:\n"
            f"cd {ROOT} && \\\n"
            f"  PHASE4C_OLD_MANIFEST_SHA={old_sha} \\\n"
            f"  SOURCE_MANIFEST_PATH=results/diagnostics/phase4c_four_domain_source_manifest.approved.json \\\n"
            "  sbatch --parsable \\\n"
            f"    --export=ALL,PHASE4C_OLD_MANIFEST_SHA={old_sha},"
            "SOURCE_MANIFEST_PATH=results/diagnostics/phase4c_four_domain_source_manifest.approved.json \\\n"
            "    slurm/run_phase4c_cross_arm_rng_replay.sh",
            flush=True,
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
