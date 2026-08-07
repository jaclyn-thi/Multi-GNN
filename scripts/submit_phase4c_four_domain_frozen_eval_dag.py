#!/usr/bin/env python3
"""Submit Phase-4C four-domain frozen-eval DAG (10 extract + 40 probe + 1 finalize).

Login-safe gates only. No real-data load. No training. No test.
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
OUT = ROOT / "results/diagnostics/phase4c_four_domain_frozen_eval_v1"
SUBMISSION = OUT / "submission_manifest.json"


def run(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=True, text=True, capture_output=True, **kw)


def sbatch_parsable(args: list[str]) -> str:
    cmd = ["sbatch", "--parsable", *args]
    try:
        cp = subprocess.run(cmd, check=True, text=True, capture_output=True)
    except FileNotFoundError:
        raise RuntimeError("sbatch not found")
    except subprocess.CalledProcessError as e:
        err = (e.stderr or "") + (e.stdout or "")
        if "Unable to contact slurm controller" in err or "slurm_submit" in err.lower():
            raise RuntimeError(
                "SLURM_CONTROLLER_UNREACHABLE_FROM_SANDBOX\n"
                + " ".join(cmd)
                + "\n"
                + err
            )
        raise RuntimeError(f"sbatch failed: {err}") from e
    jid = cp.stdout.strip().split(";")[0].strip()
    if not jid.isdigit():
        raise RuntimeError(f"unexpected sbatch output: {cp.stdout!r}")
    return jid


def main() -> int:
    os.chdir(ROOT)
    OUT.mkdir(parents=True, exist_ok=True)
    (ROOT / "slurm-logs").mkdir(parents=True, exist_ok=True)

    # A. Authorization + source freeze
    run([PYTHON, "scripts/run_phase4c_four_domain_frozen_eval_dag.py", "authorize"])
    run([PYTHON, "scripts/run_phase4c_four_domain_frozen_eval_dag.py", "cheap-gates"])
    run([PYTHON, "scripts/run_phase4c_four_domain_frozen_eval_dag.py", "dry-run-deps"])

    # compileall / bash -n / help
    run([PYTHON, "-m", "compileall", "-q", "phase4c_four_domain_frozen_eval", "scripts/run_phase4c_four_domain_frozen_eval_dag.py"])
    for sh in (
        "slurm/run_phase4c_four_domain_frozen_eval_extract_ckpt.sh",
        "slurm/run_phase4c_four_domain_frozen_eval_probe_cell.sh",
        "slurm/run_phase4c_four_domain_frozen_eval_finalize.sh",
    ):
        run(["bash", "-n", sh])
        os.chmod(ROOT / sh, 0o755)
    run([PYTHON, "scripts/run_phase4c_four_domain_frozen_eval_dag.py", "--help"])

    from phase4c_four_domain_frozen_eval import INVENTORY_CHECKPOINTS, TARGETS, cell_name
    from phase4c_four_domain_frozen_eval.paths_dag import (
        LOGICAL_EMB_ROOT,
        PHYSICAL_EMB_ROOT,
        ensure_embedding_root,
        physical_emb_path,
    )
    from phase4c_four_domain_frozen_eval.plan import estimate_storage
    from phase4c_four_domain_frozen_eval.inventory import write_inventory_artifacts

    inv = write_inventory_artifacts(OUT)
    if not inv.get("ok"):
        raise SystemExit(f"checkpoint inventory failed: {inv}")

    emb_info = ensure_embedding_root()
    phys = physical_emb_path()
    if not str(phys).startswith("/orcd/pool/007/jthi/Multi-GNN"):
        raise SystemExit("physical root outside POOL project")
    usage = shutil.disk_usage(str(phys))
    est = estimate_storage()
    # Re-estimate against POOL free
    free_gib = usage.free / (1024**3)
    need_gib = float(est["total_est_gib"])
    if free_gib < need_gib * 1.2:
        raise SystemExit(f"insufficient POOL free space: free={free_gib:.1f}GiB need~{need_gib*1.2:.1f}GiB")
    # collision: physical must be empty or only our structure
    existing = [p for p in phys.iterdir()] if phys.is_dir() else []
    if existing:
        raise SystemExit(f"physical embedding root not empty (collision risk): {existing[:5]}")

    storage_manifest = {
        "logical": str(ROOT / LOGICAL_EMB_ROOT),
        "physical": str(phys.resolve()),
        "free_gib": free_gib,
        "projected_total_est_gib": need_gib,
        "destination_sufficient": True,
        "embedding_root": emb_info,
    }
    (OUT / "storage_manifest.json").write_text(json.dumps(storage_manifest, indent=2) + "\n")

    extract_script = "slurm/run_phase4c_four_domain_frozen_eval_extract_ckpt.sh"
    probe_script = "slurm/run_phase4c_four_domain_frozen_eval_probe_cell.sh"
    final_script = "slurm/run_phase4c_four_domain_frozen_eval_finalize.sh"

    extracts = []
    probes = []
    copy_cmds = []

    print("=== submitting 10 independent GPU extract jobs (account concurrency ≤4) ===", flush=True)
    try:
        for i, (arm, step) in enumerate(INVENTORY_CHECKPOINTS):
            jname = f"p4c4d_e{i}"
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
                pname = f"p4c4d_p{i}_{target.replace('-', '')[:6]}"
                pjid = sbatch_parsable(
                    [
                        f"--job-name={pname}",
                        f"--dependency=afterok:{jid}",
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
                        "dependency": f"afterok:{jid}",
                        "extract_job_id": jid,
                        "mem": mem,
                    }
                )
                print(f"  probe {cell} -> {pjid} ({mem}) afterok:{jid}", flush=True)
    except RuntimeError as e:
        msg = str(e)
        print(msg, file=sys.stderr)
        if "SLURM_CONTROLLER_UNREACHABLE_FROM_SANDBOX" in msg:
            # Build copy-paste block
            lines = [
                "cd /home/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN",
                "module load miniforge && conda activate multignn",
                f"PYTHON={PYTHON} PYTHONPATH=. {PYTHON} scripts/submit_phase4c_four_domain_frozen_eval_dag.py",
            ]
            print("\n=== copy-paste on normal SSH (outside Cursor sandbox) ===\n" + "\n".join(lines))
            return 2
        raise

    probe_ids = [p["job_id"] for p in probes]
    dep = "afterany:" + ":".join(probe_ids)
    fin = sbatch_parsable(
        [
            "--job-name=p4c4d_final",
            f"--dependency={dep}",
            final_script,
        ]
    )
    print(f"finalize -> {fin} ({dep[:60]}...)", flush=True)

    payload = {
        "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
        "preflight_job_id": "19742414",
        "preflight_verdict": "FULL_FROZEN_EVAL_AUTHORIZED",
        "authorized_manifest_sha256": "7dcfaa52d38abd6e929633c028b2e2a21743385d26c6b2cdbd34e87b2f42d3aa",
        "encoder_retrain": False,
        "test_eval": False,
        "max_concurrent_gpu_intended": 4,
        "gpu_concurrency_note": (
            "Ten independent extract jobs; account advanced GPU slots (≤4) provide natural concurrency; "
            "no shared afterok across extracts; remaining jobs stay pending (not moved to preemptable)."
        ),
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
    # also twin top-level pointer
    (ROOT / "results/diagnostics/phase4c_four_domain_frozen_eval_v1_submission.json").write_text(
        json.dumps(
            {
                "submission_manifest": str(SUBMISSION),
                "finalize_job_id": fin,
                "n_extract": len(extracts),
                "n_probe": len(probes),
                "extract_job_ids": [e["job_id"] for e in extracts],
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
