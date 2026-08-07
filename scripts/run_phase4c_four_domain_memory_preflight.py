#!/usr/bin/env python3
"""One actual batch/domain GPU-memory preflight for the projection arm (Slurm only)."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phase4c_four_domain.source_manifest import verify_manifest
from phase4c_four_domain.train import train_arm

APPROVED = ROOT / "results/diagnostics/phase4c_four_domain_source_manifest.approved.json"
AUTH = (
    ROOT
    / "results/diagnostics/phase4c_four_domain_cpu_preflight_retry_fix_numpy_tf_scaler"
    / "cpu_preflight_authorization.json"
)


def main() -> int:
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit(
            "Refuse: real-data Phase-4C memory preflight requires SLURM_JOB_ID."
        )
    if not APPROVED.is_file():
        raise SystemExit(f"Refuse: missing approved manifest {APPROVED}")
    verify_manifest(APPROVED)
    man_sha = hashlib.sha256(APPROVED.read_bytes()).hexdigest()
    if not AUTH.is_file():
        raise SystemExit(f"Refuse: missing authorization sidecar {AUTH}")
    auth = json.loads(AUTH.read_text(encoding="utf-8"))
    if not auth.get("authorize_gpu_memory_preflight_only"):
        raise SystemExit("Refuse: GPU memory preflight not authorized by sidecar")
    if auth.get("verdict") != "PASS_REVALIDATED_NO_TEST_GATE_LOGIC":
        raise SystemExit(f"Refuse: unexpected revalidation verdict {auth.get('verdict')!r}")
    print(json.dumps({"manifest_sha256": man_sha, "authorization": auth}, indent=2, default=str))
    result = train_arm(
        "FOUR_DOMAIN_PROJECTION_INFONCE_TF_ADAPTIVE_SHORT",
        mode="memory_preflight",
        source_manifest_path=str(APPROVED.relative_to(ROOT)),
        result_dir_override=(
            "results/diagnostics/phase4c_four_domain_memory_preflight_after_no_test_gate_fix"
        ),
        ckpt_dir_override=(
            "results/checkpoints/phase4c_four_domain_seed2/memory_preflight_after_no_test_gate_fix"
        ),
    )
    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
