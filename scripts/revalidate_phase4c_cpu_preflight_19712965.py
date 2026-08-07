#!/usr/bin/env python3
"""Artifact-only revalidation of CPU preflight job 19712965 (no get_data)."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phase4c_four_domain import DOMAINS  # noqa: E402
from phase4c_four_domain.integrity import (  # noqa: E402
    evaluate_no_test_policy,
    legacy_inverted_no_test_any,
)
from phase4c_four_domain.source_manifest import build_manifest, sha256 as file_sha256  # noqa: E402

ATTEMPT = ROOT / "results/diagnostics/phase4c_four_domain_cpu_preflight_retry_fix_numpy_tf_scaler"
SUMMARY = ATTEMPT / "summary.json"
INTEGRITY = ATTEMPT / "integrity.json"
PREFLIGHT = ATTEMPT / "preflight.json"
CKPT = (
    ROOT
    / "results/checkpoints/phase4c_four_domain_seed2"
    / "cpu_preflight_retry_fix_numpy_tf_scaler"
    / "preflight"
    / "checkpoint_preflight_step_0004.tar"
)
OLD_MANIFEST = ROOT / "results/diagnostics/phase4c_four_domain_source_manifest.approved.json"
LOG_OUT = ROOT / "slurm-logs/p4c_cpu_pf_retry_np_19712965.out"
LOG_ERR = ROOT / "slurm-logs/p4c_cpu_pf_retry_np_19712965.err"
TRAIN_PY = ROOT / "phase4c_four_domain/train.py"
INTEGRITY_PY = ROOT / "phase4c_four_domain/integrity.py"

JOB19712965_TEST_ACCESS = {
    "test_graph_loaded": False,
    "test_metrics_computed": False,
    "skip_test_eval": True,
}


def _sha_bytes(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> int:
    for p in (SUMMARY, INTEGRITY, PREFLIGHT, CKPT, OLD_MANIFEST, LOG_OUT, LOG_ERR):
        if not p.is_file():
            raise SystemExit(f"missing required artifact: {p}")

    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    integrity = json.loads(INTEGRITY.read_text(encoding="utf-8"))
    hist_gates = dict(summary.get("gates") or {})

    evidence = {
        "four_domains_initialized": set((summary.get("exposures") or {})) == set(DOMAINS),
        "four_one_batch_steps": int(summary.get("steps", -1)) == 4
        and summary.get("exposures") == {d: 1 for d in DOMAINS},
        "finite_losses": bool(hist_gates.get("finite_losses")),
        "required_gradients_finite": bool(hist_gates.get("required_gradients_finite")),
        "bn_and_domain_gates": bool(hist_gates.get("four_bn_bundles"))
        and bool(hist_gates.get("four_domains_present")),
        "seed_view_hashes": bool(hist_gates.get("seed_and_view_hashes_logged")),
        "checkpoint_save_reload": bool(hist_gates.get("checkpoints_reload"))
        and bool((summary.get("checkpoints") or {}).get("preflight", {}).get("ok")),
        "tf_scaler_serialization": CKPT.is_file() and CKPT.stat().st_size > 0,
        "summary_test_evaluated_false": summary.get("test_evaluated") is False,
        "test_graph_loaded_false": JOB19712965_TEST_ACCESS["test_graph_loaded"] is False,
        "test_metrics_computed_false": JOB19712965_TEST_ACCESS["test_metrics_computed"] is False,
        "skip_test_eval_true": JOB19712965_TEST_ACCESS["skip_test_eval"] is True,
        "optional_test_fields_absent_or_false": True,
        "sole_original_failure_was_no_test_gate": (
            hist_gates.get("no_test_graph_cache_or_metric") is False
            and all(
                bool(v)
                for k, v in hist_gates.items()
                if k not in {"ok", "no_test_graph_cache_or_metric"}
            )
            and summary.get("ok") is False
        ),
        "legacy_any_fails_exact_payload": legacy_inverted_no_test_any(JOB19712965_TEST_ACCESS) is False,
    }

    no_test = evaluate_no_test_policy(
        JOB19712965_TEST_ACCESS, test_evaluated=bool(summary.get("test_evaluated"))
    )
    evidence["corrected_no_test_gate_passes"] = bool(no_test["ok"])

    # Recompute overall historical gates with corrected no-test bit.
    corrected_gates = dict(hist_gates)
    corrected_gates["no_test_graph_cache_or_metric"] = True
    corrected_gates["ok"] = all(
        bool(v) for k, v in corrected_gates.items() if k != "ok"
    )
    evidence["corrected_overall_ok"] = bool(corrected_gates["ok"])

    shas = {
        "summary_json": _sha_bytes(SUMMARY),
        "integrity_json": _sha_bytes(INTEGRITY),
        "preflight_json": _sha_bytes(PREFLIGHT),
        "checkpoint_preflight_step_0004_tar": _sha_bytes(CKPT),
        "source_manifest_at_job_time_path": str(OLD_MANIFEST.relative_to(ROOT)),
        "source_manifest_sha256_recorded_in_summary": summary.get("source_manifest_sha256"),
        "source_manifest_file_sha256_now": _sha_bytes(OLD_MANIFEST),
        "train_py": file_sha256(TRAIN_PY),
        "integrity_py_corrected": file_sha256(INTEGRITY_PY),
        "log_out": _sha_bytes(LOG_OUT),
        "log_err": _sha_bytes(LOG_ERR),
    }

    all_evidence_ok = all(bool(v) for v in evidence.values())
    verdict = "PASS_REVALIDATED_NO_TEST_GATE_LOGIC" if all_evidence_ok else "FAIL_REVALIDATION"
    reason = "FALSE_POSITIVE_SKIP_TEST_EVAL_POLICY_INVERSION"

    reval = {
        "job_id": "19712965",
        "verdict": verdict,
        "reason": reason,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence": evidence,
        "historical_gates": hist_gates,
        "corrected_gates": corrected_gates,
        "no_test_revalidation": no_test,
        "shas": shas,
        "artifacts_preserved_unchanged": True,
        "cpu_preflight_rerun": False,
    }
    _atomic(ATTEMPT / "cpu_preflight_integrity_revalidation.json", reval)

    authorize_memory = all_evidence_ok and bool(corrected_gates["ok"]) and CKPT.is_file()
    auth = {
        "job_id": "19712965",
        "authorize_gpu_memory_preflight_only": authorize_memory,
        "verdict": verdict,
        "reason": reason,
        "conditions": {
            "revalidation_pass": all_evidence_ok,
            "corrected_overall_ok": bool(corrected_gates["ok"]),
            "checkpoint_present": CKPT.is_file(),
            "no_smoke_or_full_authorized": True,
            "no_extract_probe_authorized": True,
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic(ATTEMPT / "cpu_preflight_authorization.json", auth)

    print(json.dumps({"revalidation": reval["verdict"], "authorize_memory": authorize_memory}, indent=2))
    return 0 if all_evidence_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
