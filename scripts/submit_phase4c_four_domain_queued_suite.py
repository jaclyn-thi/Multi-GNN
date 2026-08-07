#!/usr/bin/env python3
"""Queue Phase-4C independent smoke→gate→train→downstream DAGs with artifact gates.

Login-safe: --dry-run prints commands only (no sbatch, no get_data).
"""
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
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phase4c_four_domain import ARMS, arm_result_root, resolved_recipe  # noqa: E402
from phase4c_four_domain_frozen_eval import all_extract_cells  # noqa: E402

DEP_SIMPLE = re.compile(r"^(?:afterok|afterany|after):\d+(?::\d+)*$")

ARMS_ORDER = [
    "FOUR_DOMAIN_INFONCE_TF_ADAPTIVE_SHORT",
    "FOUR_DOMAIN_PROJECTION_INFONCE_TF_ADAPTIVE_SHORT",
    "FOUR_DOMAIN_EXPERT_ONLY_SHORT",
    "FOUR_DOMAIN_PROJECTION_INFONCE_TF_ADAPTIVE_LONG",
]

SHORT_NAME = {
    "FOUR_DOMAIN_INFONCE_TF_ADAPTIVE_SHORT": "infonce_short",
    "FOUR_DOMAIN_PROJECTION_INFONCE_TF_ADAPTIVE_SHORT": "proj_short",
    "FOUR_DOMAIN_EXPERT_ONLY_SHORT": "expert_short",
    "FOUR_DOMAIN_PROJECTION_INFONCE_TF_ADAPTIVE_LONG": "proj_long",
}


def validate_dep(dep: str) -> str:
    if not dep:
        return ""
    if not DEP_SIMPLE.match(dep):
        raise ValueError(f"invalid dependency string: {dep!r}")
    return dep


_dry_seq = 9000000


def sbatch(args: List[str], *, dry: bool) -> str:
    global _dry_seq
    cmd = ["sbatch", "--parsable", *args]
    print("CMD:", " ".join(cmd), flush=True)
    if dry:
        _dry_seq += 1
        return str(_dry_seq)
    out = subprocess.check_output(cmd, cwd=str(ROOT), text=True).strip()
    job = out.split(";")[0].strip()
    if not job.isdigit():
        raise RuntimeError(f"sbatch did not return numeric job id: {out!r}")
    return job


def atomic_write(path: Path, payload: Dict[str, Any]) -> None:
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


def validate_paths() -> None:
    roots = []
    for arm in ARMS_ORDER:
        r = resolved_recipe(arm)
        roots.extend([r["result_root"], r["ckpt_root"], r["embedding_root"]])
    if len(roots) != len(set(roots)):
        raise RuntimeError("path collision among arm roots")
    for arm in ARMS_ORDER:
        cells = all_extract_cells(arm)
        names = [f"{arm}:{s}:{t}" for s, t in cells]
        if len(names) != len(set(names)):
            raise RuntimeError(f"extract cell collision for {arm}")


def submit_suite(*, dry: bool, cpu0: Optional[str]) -> Dict[str, Any]:
    validate_paths()
    plan: Dict[str, Any] = {
        "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
        "current_smoke_status": {
            "smoke_job_id": None,
            "scope": "none_were_queued; submitting_four_independent_smokes",
            "per_arm_smoke_artifact": {
                arm: f"{arm_result_root(arm)}/smoke_summary.json" for arm in ARMS_ORDER
            },
        },
        "cpu_preflight_existing": cpu0,
        "full_jobs_already_queued_before_submit": False,
        "slurm_flags": {
            "gpu_partition": "mit_normal_gpu",
            "gpu_account": "mit_amf_advanced_gpu",
            "gpu_qos": "mit_amf_advanced_gpu",
            "gres": "gpu:1",
            "cpus": 16,
            "mem": "128G",
            "time_train_or_extract": "06:00:00",
            "workers": 0,
            "cpu_partition": "mit_normal",
            "cpu_account": "mit_amf_advanced_cpu",
            "cpu_qos": "mit_amf_advanced_cpu",
        },
        "failure_isolation": (
            "Each arm has its own smoke→gate→train→train_gate→extract→probe→finalize DAG. "
            "No cross-arm afterok. Training is four independent jobs (not an array)."
        ),
        "arms": {},
        "dry_run": dry,
    }

    dep_a = validate_dep(f"afterok:{cpu0}") if cpu0 else ""
    pre_args = ([f"--dependency={dep_a}"] if dep_a else []) + [
        "--job-name=p4c_cpu_preflight_all",
        "slurm/run_phase4c_four_domain_cpu_preflight_all.sh",
    ]
    preflight_all = sbatch(pre_args, dry=dry)
    plan["preflight_all_job_id"] = preflight_all
    plan["preflight_all_dependency"] = dep_a or None

    mem_args = ([f"--dependency={dep_a}"] if dep_a else []) + [
        "--job-name=p4c_mem_preflight",
        "slurm/run_phase4c_four_domain_memory_preflight.sh",
    ]
    mem_job = sbatch(mem_args, dry=dry)
    plan["memory_preflight_job_id"] = mem_job
    plan["memory_preflight_dependency"] = dep_a or None

    smoke_dep = validate_dep(f"afterok:{preflight_all}:{mem_job}")

    for arm in ARMS_ORDER:
        tag = SHORT_NAME[arm]
        arm_plan: Dict[str, Any] = {
            "arm": arm,
            "result_root": arm_result_root(arm),
            "smoke_artifact": f"{arm_result_root(arm)}/smoke_summary.json",
            "recipe_steps": resolved_recipe(arm)["max_optimizer_steps"],
        }
        smoke = sbatch(
            [
                f"--dependency={smoke_dep}",
                f"--job-name=p4c_smoke_{tag}",
                f"--export=ALL,ARM={arm}",
                "slurm/run_phase4c_four_domain_smoke_arm.sh",
            ],
            dry=dry,
        )
        arm_plan["smoke_job_id"] = smoke
        arm_plan["smoke_dependency"] = smoke_dep

        gate_dep = validate_dep(f"afterany:{smoke}")
        gate = sbatch(
            [
                f"--dependency={gate_dep}",
                f"--job-name=p4c_sgate_{tag}",
                f"--export=ALL,ARM={arm},SMOKE_JOB_ID={smoke}",
                "slurm/run_phase4c_four_domain_smoke_gate.sh",
            ],
            dry=dry,
        )
        arm_plan["smoke_gate_job_id"] = gate
        arm_plan["smoke_gate_dependency"] = gate_dep

        train_dep = validate_dep(f"afterok:{gate}")
        train = sbatch(
            [
                f"--dependency={train_dep}",
                f"--job-name=p4c_train_{tag}",
                f"--export=ALL,ARM={arm}",
                "slurm/run_phase4c_four_domain_train_arm.sh",
            ],
            dry=dry,
        )
        arm_plan["train_job_id"] = train
        arm_plan["train_dependency"] = train_dep

        tgate_dep = validate_dep(f"afterok:{train}")
        tgate = sbatch(
            [
                f"--dependency={tgate_dep}",
                f"--job-name=p4c_tgate_{tag}",
                f"--export=ALL,ARM={arm},TRAIN_JOB_ID={train}",
                "slurm/run_phase4c_four_domain_train_gate.sh",
            ],
            dry=dry,
        )
        arm_plan["train_gate_job_id"] = tgate
        arm_plan["train_gate_dependency"] = tgate_dep

        extracts = []
        probes = []
        probe_ids: List[str] = []
        for step, target in all_extract_cells(arm):
            ttag = target.lower().replace("-", "")
            ext_dep = validate_dep(f"afterok:{tgate}")
            ext = sbatch(
                [
                    f"--dependency={ext_dep}",
                    f"--job-name=p4c_ext_{tag}_{step}_{ttag}",
                    f"--export=ALL,ARM={arm},STEP={step},TARGET={target}",
                    "slurm/run_phase4c_four_domain_extract_cell.sh",
                ],
                dry=dry,
            )
            extracts.append({"job_id": ext, "dependency": ext_dep, "step": step, "target": target})
            pr_dep = validate_dep(f"afterok:{ext}")
            pr = sbatch(
                [
                    f"--dependency={pr_dep}",
                    f"--job-name=p4c_prb_{tag}_{step}_{ttag}",
                    f"--export=ALL,ARM={arm},STEP={step},TARGET={target}",
                    "slurm/run_phase4c_four_domain_probe_cell.sh",
                ],
                dry=dry,
            )
            probes.append({"job_id": pr, "dependency": pr_dep, "step": step, "target": target})
            probe_ids.append(pr)

        fin_dep = validate_dep("afterok:" + ":".join(probe_ids))
        fin = sbatch(
            [
                f"--dependency={fin_dep}",
                f"--job-name=p4c_fin_{tag}",
                f"--export=ALL,ARM={arm}",
                "slurm/run_phase4c_four_domain_finalize_arm.sh",
            ],
            dry=dry,
        )
        arm_plan["extracts"] = extracts
        arm_plan["probes"] = probes
        arm_plan["finalize_job_id"] = fin
        arm_plan["finalize_dependency"] = fin_dep
        plan["arms"][arm] = arm_plan

    return plan


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--cpu0-job-id", default=os.environ.get("P4C_CPU0_JOB_ID", "19673182"))
    ap.add_argument(
        "--out",
        default="results/diagnostics/phase4c_four_domain_submission_current.json",
    )
    args = ap.parse_args()
    cpu0 = str(args.cpu0_job_id).strip() or None
    if cpu0 is not None and not str(cpu0).isdigit():
        raise SystemExit(f"cpu0 job id must be numeric, got {cpu0!r}")
    os.chdir(ROOT)
    plan = submit_suite(dry=bool(args.dry_run), cpu0=cpu0)
    out = ROOT / args.out
    atomic_write(out, plan)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
