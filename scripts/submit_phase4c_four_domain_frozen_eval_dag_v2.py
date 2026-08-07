#!/usr/bin/env python3
"""Submit Phase-4C four-domain frozen-eval DAG v2 (9 extract + 40 probe + 1 finalize).

Reuses PROJECTION SHORT@4000 cells from preflight 19778019.
Login-safe gates only before sbatch. No training. No test. No historical re-extract.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PYTHON = "/home/jthi/.conda/envs/multignn/bin/python"
OUT = ROOT / "results/diagnostics/phase4c_four_domain_frozen_eval_v2"
SUBMISSION = OUT / "submission_manifest.json"


def run(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=True, text=True, capture_output=True, **kw)


def sbatch_parsable(args: list[str]) -> str:
    cmd = ["sbatch", "--parsable", *args]
    try:
        cp = subprocess.run(cmd, check=True, text=True, capture_output=True)
    except FileNotFoundError as e:
        raise RuntimeError("sbatch not found") from e
    except subprocess.CalledProcessError as e:
        err = (e.stderr or "") + (e.stdout or "")
        raise RuntimeError(
            "SBATCH_FAILED\n"
            + " ".join(cmd)
            + "\n"
            + err
            + "\n=== copy-paste on normal SSH ===\n"
            + f"cd {ROOT} && module load miniforge && conda activate multignn && "
            + f"PYTHONPATH=. {PYTHON} scripts/submit_phase4c_four_domain_frozen_eval_dag_v2.py"
        ) from e
    jid = cp.stdout.strip().split(";")[0].strip()
    if not jid.isdigit():
        raise RuntimeError(f"unexpected sbatch output: {cp.stdout!r}")
    return jid


def main() -> int:
    os.chdir(ROOT)
    OUT.mkdir(parents=True, exist_ok=True)
    (ROOT / "slurm-logs").mkdir(parents=True, exist_ok=True)
    (OUT / "cells").mkdir(parents=True, exist_ok=True)
    (OUT / "extract_jobs").mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)

    run([PYTHON, "scripts/run_phase4c_four_domain_frozen_eval_dag_v2.py", "authorize"])
    run([PYTHON, "scripts/run_phase4c_four_domain_frozen_eval_dag_v2.py", "cheap-gates"])
    run([PYTHON, "scripts/run_phase4c_four_domain_frozen_eval_dag_v2.py", "dry-run-deps"])

    run(
        [
            PYTHON,
            "-m",
            "compileall",
            "-q",
            "phase4c_four_domain_frozen_eval",
            "scripts/run_phase4c_four_domain_frozen_eval_dag_v2.py",
            "scripts/submit_phase4c_four_domain_frozen_eval_dag_v2.py",
        ]
    )
    for sh in (
        "slurm/run_phase4c_four_domain_frozen_eval_extract_ckpt_v2.sh",
        "slurm/run_phase4c_four_domain_frozen_eval_probe_cell_v2.sh",
        "slurm/run_phase4c_four_domain_frozen_eval_finalize_v2.sh",
    ):
        run(["bash", "-n", sh])
        os.chmod(ROOT / sh, 0o755)
    run([PYTHON, "scripts/run_phase4c_four_domain_frozen_eval_dag_v2.py", "--help"])

    from phase4c_four_domain_frozen_eval import TARGETS, cell_name
    from phase4c_four_domain_frozen_eval.inventory import write_inventory_artifacts
    from phase4c_four_domain_frozen_eval.paths_dag_v2 import (
        EXTRACT_CHECKPOINTS,
        EXPECTED_EVAL_SOURCE_MANIFEST_V2_SHA,
        EXPECTED_TRAINING_MANIFEST_SHA,
        LOGICAL_EMB_ROOT,
        PHYSICAL_EMB_ROOT,
        REUSED_ARM,
        REUSED_STEP,
        assert_dag_v2_paths_unique,
        ensure_embedding_root,
        physical_emb_path,
    )
    from phase4c_four_domain_frozen_eval.plan import estimate_storage

    assert_dag_v2_paths_unique()
    inv = write_inventory_artifacts(OUT)
    if not inv.get("ok"):
        raise SystemExit(f"checkpoint inventory failed: {inv}")

    emb_info = ensure_embedding_root()
    phys = physical_emb_path()
    usage = shutil.disk_usage(str(phys))
    est = estimate_storage()
    free_gib = usage.free / (1024**3)
    # Nine new checkpoints × 4 domains remain; four cells already present.
    need_gib = float(est["total_est_gib"]) * (36 / 40.0)
    if free_gib < need_gib * 1.1:
        raise SystemExit(
            f"insufficient POOL free space: free={free_gib:.1f}GiB need~{need_gib * 1.1:.1f}GiB"
        )

    storage_manifest = {
        "logical": str(ROOT / LOGICAL_EMB_ROOT),
        "physical": str(phys.resolve()),
        "free_gib": free_gib,
        "projected_remaining_est_gib": need_gib,
        "destination_sufficient": True,
        "embedding_root": emb_info,
        "reused_preflight_cells": 4,
        "new_extract_cells": 36,
    }
    (OUT / "storage_manifest.json").write_text(json.dumps(storage_manifest, indent=2) + "\n")

    extract_script = "slurm/run_phase4c_four_domain_frozen_eval_extract_ckpt_v2.sh"
    probe_script = "slurm/run_phase4c_four_domain_frozen_eval_probe_cell_v2.sh"
    final_script = "slurm/run_phase4c_four_domain_frozen_eval_finalize_v2.sh"

    extracts = []
    probes = []

    print("=== submitting 4 reusable-cell probes (no extract dependency) ===", flush=True)
    try:
        for target in TARGETS:
            mem = "64G" if target == "Small-HI" else "96G"
            cell = cell_name(REUSED_ARM, REUSED_STEP, target)
            pname = f"p4c4dv2_pr_{target.replace('-', '')[:6]}"
            pjid = sbatch_parsable(
                [
                    f"--job-name={pname}",
                    f"--mem={mem}",
                    f"--export=ALL,ARM={REUSED_ARM},STEP={REUSED_STEP},DATA={target},MEM={mem}",
                    probe_script,
                ]
            )
            probes.append(
                {
                    "cell": cell,
                    "arm": REUSED_ARM,
                    "step": REUSED_STEP,
                    "target": target,
                    "job_id": pjid,
                    "dependency": None,
                    "extract_job_id": None,
                    "reused_preflight_cell": True,
                    "mem": mem,
                }
            )
            print(f"  reused probe {cell} -> {pjid} ({mem}) no-dep", flush=True)

        print("=== submitting 9 independent GPU extract jobs (account concurrency ≤4) ===", flush=True)
        for i, (arm, step) in enumerate(EXTRACT_CHECKPOINTS):
            jname = f"p4c4dv2_e{i}"
            jid = sbatch_parsable(
                [
                    f"--job-name={jname}",
                    f"--export=ALL,ARM={arm},STEP={step}",
                    extract_script,
                ]
            )
            extracts.append(
                {
                    "idx": i,
                    "arm": arm,
                    "step": int(step),
                    "job_id": jid,
                    "job_name": jname,
                    "resources": {
                        "partition": "mit_normal_gpu",
                        "account": "mit_amf_advanced_gpu",
                        "qos": "mit_amf_advanced_gpu",
                        "gres": "gpu:1",
                        "cpus": 16,
                        "mem": "128G",
                        "time": "06:00:00",
                        "loader_workers": 0,
                    },
                }
            )
            print(f"extract[{i}] {arm}@{step} -> {jid}", flush=True)
            for target in TARGETS:
                mem = "64G" if target == "Small-HI" else "96G"
                cell = cell_name(arm, step, target)
                pname = f"p4c4dv2_p{i}_{target.replace('-', '')[:6]}"
                pjid = sbatch_parsable(
                    [
                        f"--job-name={pname}",
                        f"--dependency=afterany:{jid}",
                        f"--mem={mem}",
                        f"--export=ALL,ARM={arm},STEP={step},DATA={target},MEM={mem}",
                        probe_script,
                    ]
                )
                probes.append(
                    {
                        "cell": cell,
                        "arm": arm,
                        "step": int(step),
                        "target": target,
                        "job_id": pjid,
                        "dependency": f"afterany:{jid}",
                        "extract_job_id": jid,
                        "reused_preflight_cell": False,
                        "mem": mem,
                    }
                )
                print(f"  probe {cell} -> {pjid} ({mem}) afterany:{jid}", flush=True)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 2

    probe_ids = [p["job_id"] for p in probes]
    assert len(probe_ids) == 40
    dep = "afterany:" + ":".join(probe_ids)
    fin = sbatch_parsable(
        [
            "--job-name=p4c4dv2_fin",
            f"--dependency={dep}",
            final_script,
        ]
    )
    print(f"finalize -> {fin}", flush=True)

    payload = {
        "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
        "preflight_job_id": "19778019",
        "preflight_verdict": "FULL_FROZEN_EVAL_V2_AUTHORIZED",
        "eval_source_manifest_v2_sha256": EXPECTED_EVAL_SOURCE_MANIFEST_V2_SHA,
        "authorized_manifest_sha256": EXPECTED_TRAINING_MANIFEST_SHA,
        "encoder_retrain": False,
        "test_eval": False,
        "historical_reextract": False,
        "max_concurrent_gpu_intended": 4,
        "gpu_concurrency_note": (
            "Nine independent extract jobs; account advanced GPU slots (≤4) provide natural "
            "concurrency; remaining stay pending; not moved to preemptable."
        ),
        "probe_dependency_policy": "afterany_per_checkpoint_extract; reused probes independent",
        "skipped_extract": {"arm": REUSED_ARM, "step": REUSED_STEP, "reason": "reused_from_preflight"},
        "storage": storage_manifest,
        "result_root": str(OUT),
        "embedding_logical": str(ROOT / LOGICAL_EMB_ROOT),
        "embedding_physical": PHYSICAL_EMB_ROOT,
        "extracts": extracts,
        "probes": probes,
        "finalize": {"job_id": fin, "dependency": dep, "n_probe_deps": len(probe_ids)},
        "n_extract": len(extracts),
        "n_probe": len(probes),
        "n_cells_expected": 40,
    }
    SUBMISSION.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (ROOT / "results/diagnostics/phase4c_four_domain_frozen_eval_v2_submission.json").write_text(
        json.dumps(
            {
                "submission_manifest": str(SUBMISSION),
                "finalize_job_id": fin,
                "n_extract": len(extracts),
                "n_probe": len(probes),
                "extract_job_ids": [e["job_id"] for e in extracts],
                "preflight_job_id": "19778019",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"ok": True, "submission": str(SUBMISSION), "finalize": fin}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
