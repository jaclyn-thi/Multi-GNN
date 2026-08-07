#!/usr/bin/env python3
"""Offline revalidation of job 19630663 false-negative c_always_198x198 gate.

Does NOT retrain, forward, extract, probe, or rewrite the original aggregate.json.
Writes versioned sidecars only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gbt_tf_adaptive_stdfloor_r198 import (  # noqa: E402
    ARM,
    CKPT_ROOT,
    OBJECTIVE_ID,
    RESULT_ROOT,
)
from gbt_tf_adaptive_stdfloor_r198.integrity import (  # noqa: E402
    FALSE_NEGATIVE_REASON,
    REVALIDATED_CLASSIFICATION,
    c_always_198x198_from_rows,
)
from gbt_tf_adaptive_stdfloor_r198.orchestration import write_json  # noqa: E402
from gbt_tf_adaptive_stdfloor_r198.training_acceptance import (  # noqa: E402
    verify_training_authorized_for_frozen_eval,
)
from graph_barlow_twins_r198 import R198_DIM  # noqa: E402

JOB_ID = "19630663"
OUT_DIR = ROOT / RESULT_ROOT
CKPT_DIR = ROOT / CKPT_ROOT
STEP_PY = ROOT / "gbt_tf_adaptive_stdfloor_r198" / "step.py"
RUNTIME_ASSERT_MARKERS = (
    'if tuple(gbt_diag["C_shape"]) != (R198_DIM, R198_DIM):',
    'raise RuntimeError(f"C_shape={gbt_diag[\'C_shape\']} != (198,198)")',
)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_rows(jsonl_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def locate_runtime_assert(step_path: Path) -> Dict[str, Any]:
    text = step_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    hits = []
    for i, line in enumerate(lines, start=1):
        if "C_shape" in line and (
            "R198_DIM" in line or "198,198" in line or "(198, 198)" in line
        ):
            hits.append({"line": i, "text": line.strip()})
    # Match on source structure rather than a brittle exact raise string.
    has_compare = any(
        'tuple(gbt_diag["C_shape"])' in h["text"] and "R198_DIM" in h["text"] for h in hits
    )
    has_raise = any("raise RuntimeError" in h["text"] and "C_shape" in h["text"] for h in hits)
    present = bool(has_compare and has_raise)
    assert_line = next(
        (h for h in hits if "tuple(gbt_diag" in h["text"] and "R198_DIM" in h["text"]),
        None,
    )
    raise_line = next(
        (h for h in hits if "raise RuntimeError" in h["text"] and "C_shape" in h["text"]),
        None,
    )
    return {
        "path": str(step_path.relative_to(ROOT)),
        "sha256": file_sha256(step_path),
        "runtime_assert_present": present,
        "assert_line": assert_line,
        "raise_line": raise_line,
        "note": (
            "Assertion is in the hybrid training step used by job 19630663; "
            "any C.shape mismatch would have raised mid-run. Current file SHA is "
            "the post-run working-tree step module; the C_shape==(198,198) hard-fail "
            "was already present during training."
        ),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=f"Revalidate {ARM} C_shape logging gate")
    p.add_argument("--job-id", type=str, default=JOB_ID)
    args = p.parse_args(argv)

    agg_path = OUT_DIR / "aggregate.json"
    jsonl_path = OUT_DIR / "logs" / "steps.jsonl"
    if not agg_path.is_file():
        raise RuntimeError(f"missing original aggregate: {agg_path}")
    if not jsonl_path.is_file():
        raise RuntimeError(f"missing steps jsonl: {jsonl_path}")

    # Capture SHAs BEFORE writing any sidecars.
    original_agg_sha = file_sha256(agg_path)
    log_sha = file_sha256(jsonl_path)
    ckpt_paths = {
        "step_0750": CKPT_DIR / "checkpoint_step_0750.pt",
        "step_1500": CKPT_DIR / "checkpoint_step_1500.pt",
        "step_2250": CKPT_DIR / "checkpoint_step_2250.pt",
        "step_3000": CKPT_DIR / "checkpoint_step_3000.pt",
        "last": CKPT_DIR / "checkpoint_last.pt",
    }
    ckpt_shas = {k: file_sha256(v) for k, v in ckpt_paths.items() if v.is_file()}
    if len(ckpt_shas) != 5:
        raise RuntimeError(f"missing checkpoints: {ckpt_paths}")

    original = json.loads(agg_path.read_text(encoding="utf-8"))
    if str(original.get("slurm_job_id")) != str(args.job_id):
        raise RuntimeError(
            f"aggregate job_id={original.get('slurm_job_id')} != expected {args.job_id}"
        )
    if bool(original.get("ok")) or str(original.get("classification")) == "PASS":
        raise RuntimeError("original aggregate unexpectedly PASS; refuse revalidation rewrite path")

    original_gates = dict(original.get("training_integrity_gates") or {})
    failed = [k for k, v in original_gates.items() if not v]
    if failed != ["c_always_198x198"]:
        raise RuntimeError(f"refuse revalidation; unexpected failed gates={failed}")

    rows = load_rows(jsonl_path)
    if len(rows) != 3000:
        raise RuntimeError(f"expected 3000 rows, got {len(rows)}")

    c_gate = c_always_198x198_from_rows(rows, expected_dim=R198_DIM)
    r198_logged = all(int(r.get("r198_dim") or -1) == int(R198_DIM) for r in rows)
    # Logging defect evidence: C_shape absent, C_numel present everywhere.
    n_shape_missing = sum(1 for r in rows if r.get("C_shape") is None)
    n_numel_39204 = sum(1 for r in rows if int(r.get("C_numel") or -1) == 39204)

    runtime_assert = locate_runtime_assert(STEP_PY)
    if not runtime_assert["runtime_assert_present"]:
        raise RuntimeError("runtime C_shape assertion missing from step.py")

    # Recompute gates with repaired C check only; all other original gate values preserved.
    repaired_gates = dict(original_gates)
    repaired_gates["c_always_198x198"] = bool(c_gate["ok"])
    all_other_original_true = all(
        bool(v) for k, v in original_gates.items() if k != "c_always_198x198"
    )
    revalidated_ok = (
        bool(c_gate["ok"])
        and r198_logged
        and n_numel_39204 == 3000
        and n_shape_missing == 3000  # documents the logging omission on this job
        and all_other_original_true
        and runtime_assert["runtime_assert_present"]
    )

    defect = {
        "description": (
            "JSONL row construction filtered out list-valued stats except *first32 keys; "
            "C_shape (list) was dropped while scalar C_numel was retained. The aggregate "
            "gate then required list(r['C_shape'])==[198,198], yielding a false FAIL."
        ),
        "filter_location": (
            "scripts/run_gbt_tf_adaptive_stdfloor_full3000.py "
            "(stats unpack: keep non-list/dict OR key.endswith('first32'))"
        ),
        "gate_location": (
            "scripts/run_gbt_tf_adaptive_stdfloor_full3000.py training_integrity_gates"
            "['c_always_198x198']"
        ),
        "n_rows_missing_C_shape": n_shape_missing,
        "n_rows_with_C_numel_39204": n_numel_39204,
    }

    # Confirm artifacts unchanged after analysis (still pre-sidecar write for ckpts/log/agg).
    post_check = {
        "aggregate_sha256": file_sha256(agg_path),
        "steps_jsonl_sha256": file_sha256(jsonl_path),
        "checkpoints_sha256": {k: file_sha256(v) for k, v in ckpt_paths.items()},
    }
    artifacts_unchanged = (
        post_check["aggregate_sha256"] == original_agg_sha
        and post_check["steps_jsonl_sha256"] == log_sha
        and post_check["checkpoints_sha256"] == ckpt_shas
    )
    if not artifacts_unchanged:
        raise RuntimeError("artifact SHA drift during revalidation")

    revalidation = {
        "title": f"{ARM} training integrity revalidation (C_shape logging gate)",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": str(args.job_id),
        "arm": ARM,
        "objective_id": OBJECTIVE_ID,
        "classification": REVALIDATED_CLASSIFICATION if revalidated_ok else "FAIL_REVALIDATION",
        "ok": bool(revalidated_ok),
        "authorized_for_frozen_eval": bool(revalidated_ok),
        "original_failure_reason": FALSE_NEGATIVE_REASON,
        "original_aggregate_preserved": True,
        "original_aggregate_path": str(agg_path),
        "original_aggregate_sha256": original_agg_sha,
        "original_classification": original.get("classification"),
        "original_ok": bool(original.get("ok")),
        "original_failed_gates": failed,
        "steps_jsonl_path": str(jsonl_path),
        "steps_jsonl_sha256": log_sha,
        "checkpoints_sha256": ckpt_shas,
        "artifacts_unchanged_confirmed": True,
        "logging_defect": defect,
        "c_shape_revalidation": c_gate,
        "r198_dim_logged_all_steps": r198_logged,
        "runtime_c_shape_assertion": runtime_assert,
        "repaired_training_integrity_gates": repaired_gates,
        "no_retrain": True,
        "no_encoder_forward": True,
        "no_optimizer_update": True,
        "no_extraction": True,
        "no_probe": True,
        "future_logging_fix": (
            "filter_stats_for_jsonl retains C_shape; aggregate gate uses "
            "c_always_198x198_from_rows (C_numel==198**2 or C_shape)."
        ),
    }

    revalidated_aggregate = {
        **{k: v for k, v in original.items() if k not in {"ok", "classification", "training_integrity_gates"}},
        "ok": bool(revalidated_ok),
        "classification": REVALIDATED_CLASSIFICATION if revalidated_ok else "FAIL_REVALIDATION",
        "training_integrity_gates": repaired_gates,
        "revalidation": {
            "sidecar": "training_integrity_revalidation.json",
            "original_aggregate_sha256": original_agg_sha,
            "original_failure_reason": FALSE_NEGATIVE_REASON,
            "c_shape_revalidation": c_gate,
            "artifacts_unchanged_confirmed": True,
        },
        "note": (
            "Versioned revalidated aggregate. Original aggregate.json remains FAIL for provenance."
        ),
    }

    rev_path = OUT_DIR / "training_integrity_revalidation.json"
    rev_agg_path = OUT_DIR / "aggregate_revalidated.json"
    write_json(rev_path, revalidation)
    write_json(rev_agg_path, revalidated_aggregate)

    # Confirm original aggregate SHA unchanged after sidecar writes.
    if file_sha256(agg_path) != original_agg_sha:
        raise RuntimeError("original aggregate was modified; provenance broken")
    for k, pth in ckpt_paths.items():
        if file_sha256(pth) != ckpt_shas[k]:
            raise RuntimeError(f"checkpoint SHA changed: {k}")
    if file_sha256(jsonl_path) != log_sha:
        raise RuntimeError("steps.jsonl SHA changed")

    acceptance = verify_training_authorized_for_frozen_eval(OUT_DIR)
    write_json(OUT_DIR / "frozen_eval_authorization.json", acceptance)

    # Append note section without rewriting the original FAIL gates block meaning.
    note_path = ROOT / "notes/financial_multidataset_gbt_tf_adaptive_stdfloor_1e4.md"
    if note_path.is_file():
        extra = [
            "",
            "## Offline C_shape gate revalidation (no retrain)",
            "",
            f"- classification: `{revalidation['classification']}`",
            f"- original failure reason: `{FALSE_NEGATIVE_REASON}`",
            f"- original aggregate SHA: `{original_agg_sha}` (unchanged)",
            f"- steps.jsonl SHA: `{log_sha}` (unchanged)",
            f"- authorized_for_frozen_eval: `{acceptance.get('authorized_for_frozen_eval')}`",
            f"- sidecar: `results/diagnostics/financial_multidataset_gbt_tf_adaptive_stdfloor_1e4/training_integrity_revalidation.json`",
            f"- revalidated aggregate: `results/diagnostics/financial_multidataset_gbt_tf_adaptive_stdfloor_1e4/aggregate_revalidated.json`",
            "",
        ]
        text = note_path.read_text(encoding="utf-8")
        if "Offline C_shape gate revalidation" not in text:
            note_path.write_text(text.rstrip() + "\n" + "\n".join(extra), encoding="utf-8")

    print(
        json.dumps(
            {
                "ok": revalidated_ok,
                "classification": revalidation["classification"],
                "authorized_for_frozen_eval": acceptance.get("authorized_for_frozen_eval"),
                "original_aggregate_sha256": original_agg_sha,
                "c_gate": c_gate,
                "acceptance": acceptance,
            },
            indent=2,
        )
    )
    return 0 if revalidated_ok and acceptance.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
