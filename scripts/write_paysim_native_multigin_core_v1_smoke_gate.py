#!/usr/bin/env python3
"""Gate writer for PaySim-native Multi-GIN core_v1 2-epoch smoke (validation-only)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from paysim_native_multigin import (  # noqa: E402
    CONTRACT_NATIVE_MULTIGIN_CORE,
    EXPECTED_EDGE_DIM_PORTS,
    TYPE_ORDER,
    sha256_file,
)

RUN_NAME = "paysim_native_multigin_core_v1_smoke_seed2"
GATE_ID = "paysim_native_multigin_core_v1_smoke"
OUT_JSON = REPO / "results/diagnostics/paysim_native_multigin_core_v1_smoke.json"
OUT_MD = REPO / "notes/paysim_native_multigin_core_v1_smoke.md"
SUBMISSION_JSON = (
    REPO / "results/diagnostics/paysim_native_multigin_core_v1_smoke_submission.json"
)
SUBMISSION_MD = REPO / "notes/paysim_native_multigin_core_v1_smoke_submission.md"
REG_JSON = REPO / "results/diagnostics/thesis_experiment_registry.json"
REG_MD = REPO / "notes/thesis_experiment_registry.md"

# Locked temporal cohorts (prior audits).
EXPECTED_SPLITS = {
    "train": {
        "n": 3792821,
        "n_positives": 3175,
        "index_sha256": "0d2f7e516aeae723cda174f4ab086380d006a526d4ede47cd2b8f5100af92279",
        "step_min": 1,
        "step_max": 280,
    },
    "val": {
        "n": 1276276,
        "n_positives": 780,
        "index_sha256": "696756046b7e6dd4df5f6f600bbb373c7a24b888d60df7ea10ce3bf468f76469",
        "step_min": 281,
        "step_max": 354,
    },
    "test": {
        "n": 1293523,
        "n_positives": 4258,
        "index_sha256": "dcc1018601844cfb174ca14a24d2208512c63cca2948cdbc804ed1c44aebac87",
        "step_min": 355,
        "step_max": 743,
    },
}
FORMATTED_SHA256 = "03c2fa07b95d145e754b74a5e646c2d71cd4fed051210d6292a0bbab90112c93"

# Smoke gate: material improvement over balance-free Multi-GIN Candidate A.
REF_BALANCE_FREE_VAL_AUPRC = 0.168
MARGIN_ABS = 0.05
PASS_THRESHOLD_VAL_AUPRC = REF_BALANCE_FREE_VAL_AUPRC + MARGIN_ABS  # 0.218

REF_TABLE = {
    "legacy_compatibility_x_only": 0.0046,
    "balance_free_multigin_candidate_a": REF_BALANCE_FREE_VAL_AUPRC,
    "native_logistic": 0.5736,
    "native_mlp": 0.6476,
    "native_hgb": 0.6616,
}

PROTECTED = [
    REPO / "notes/paysim_supervised_failure_audit.md",
    REPO / "results/diagnostics/paysim_supervised_failure_audit.json",
    REPO / "notes/paysim_native_tabular_baseline.md",
    REPO / "results/diagnostics/paysim_native_tabular_baseline.json",
    REPO / "results/diagnostics/paysim_supervised_multigin_eu.json",
    REPO / "notes/paysim_supervised_multigin_eu.md",
    REPO
    / "results/diagnostics/supervised_PaySim_paysim_supervised_multigin_eu_seed2_summary.json",
    REPO
    / "results/diagnostics/supervised_PaySim_paysim_supervised_multigin_eu_seed2_epoch_history.json",
]


def _finite(x: Any) -> bool:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return False
    return math.isfinite(v)


def _git_provenance() -> Dict[str, Any]:
    def _run(cmd: List[str]) -> Optional[str]:
        try:
            return subprocess.check_output(cmd, cwd=REPO, text=True).strip()
        except Exception:  # noqa: BLE001
            return None

    return {
        "git_head": _run(["git", "rev-parse", "HEAD"]),
        "git_describe": _run(["git", "describe", "--always", "--dirty"]),
        "git_branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
    }


def recompute_cohort_hashes(formatted: Path) -> Dict[str, Any]:
    df = pd.read_csv(formatted, usecols=["EdgeID", "Timestamp", "Is Laundering"])
    ts = df["Timestamp"].to_numpy()
    # Match get_data re-zero then hourly buckets (step = rezeroed_ts/3600 + 1 if min was 1*3600)
    # Prior audits hash EdgeID order after sort; formatted is already sorted.
    # Locked steps use raw step = Timestamp/3600 when Timestamp is step*3600 before rezero.
    # After format, Timestamp = step*3600; after get_data rezero: Timestamp -= min.
    # Audits used: step from raw; index hashes are of EdgeID arrays for each split.
    # Reconstruct using EdgeID ranges from timestamp buckets matching dataset_specs PaySim.
    from dataset_specs import get_dataset_spec
    from dataset_splits import temporal_edge_split

    y = torch.LongTensor(df["Is Laundering"].to_numpy())
    # Re-zero like get_data
    ts_t = torch.Tensor(ts.astype(np.float64) - float(ts.min()))
    spec = get_dataset_spec("PaySim")
    tr, va, te, _ = temporal_edge_split(ts_t, y, spec)

    def _hash_ids(inds: torch.Tensor) -> str:
        ids = df["EdgeID"].to_numpy()[inds.numpy()].astype(np.int64)
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(ids).tobytes())
        return h.hexdigest()

    out = {}
    for name, inds in (("train", tr), ("val", va), ("test", te)):
        y_s = y[inds].numpy()
        out[name] = {
            "n": int(inds.numel()),
            "n_positives": int(y_s.sum()),
            "positive_rate": float(y_s.mean()) if inds.numel() else 0.0,
            "index_sha256": _hash_ids(inds),
        }
    out["formatted_sha256"] = sha256_file(formatted)
    return out


def append_registry(payload: Dict[str, Any]) -> None:
    if not REG_JSON.is_file():
        return
    reg = json.loads(REG_JSON.read_text())
    rows = reg.get("rows", [])
    run_id = f"{GATE_ID}|{CONTRACT_NATIVE_MULTIGIN_CORE}|gin|seed2"
    best = payload.get("best_metrics") or {}
    row = {
        "run_id": run_id,
        "dataset": "PaySim",
        "objective": "supervised_native_multigin_smoke",
        "encoder": "gin",
        "seed": 2,
        "n_epochs": 2,
        "thesis_role": "thesis_supporting",
        "feature_contract_id": CONTRACT_NATIVE_MULTIGIN_CORE,
        "val_auprc": best.get("max_val_auprc"),
        "val_f1_argmax": best.get("best_val_f1"),
        "gate_pass": payload.get("gate_pass"),
        "test_evaluated": False,
        "source": str(OUT_JSON),
        "table_eligible": False,
        "preserve_seed_edges": False,
        "note": "2-epoch validation-only smoke; not a formal 50-ep claim",
    }
    # Upsert same run_id (gate-only rewrites allowed; do not duplicate).
    updated = False
    for i, existing in enumerate(rows):
        if isinstance(existing, dict) and existing.get("run_id") == run_id:
            rows[i] = row
            updated = True
            break
    if not updated:
        rows.append(row)
    reg["rows"] = rows
    reg["row_count"] = len(rows)
    REG_JSON.write_text(json.dumps(reg, indent=2) + "\n")
    if REG_MD.is_file() and not updated:
        with REG_MD.open("a") as f:
            f.write(
                f"\n\n## {GATE_ID}\n\n"
                f"- Appended `{run_id}` (table_eligible=false); historical rows unchanged.\n"
                f"- See `{OUT_MD.relative_to(REPO)}`.\n"
            )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-id", default="")
    ap.add_argument("--run-name", default=RUN_NAME)
    ap.add_argument("--command", default="", help="Exact train command for submission note")
    args = ap.parse_args()
    run_name = args.run_name

    hist_path = (
        REPO / "results/diagnostics" / f"supervised_PaySim_{run_name}_epoch_history.json"
    )
    summary_path = (
        REPO / "results/diagnostics" / f"supervised_PaySim_{run_name}_summary.json"
    )
    run_dir = REPO / "saved-models" / run_name
    best_path = run_dir / "checkpoint_best_val_f1.tar"
    last_path = run_dir / "checkpoint_last.tar"
    flat_path = REPO / "saved-models" / f"checkpoint_{run_name}.tar"
    summary_md = REPO / "notes" / f"supervised_PaySim_{run_name}_summary.md"

    checks: Dict[str, Any] = {}
    failures: List[str] = []

    # Protected historical artifacts
    protected_status = {}
    protected_ok = True
    for p in PROTECTED:
        exists = p.is_file()
        protected_status[str(p.relative_to(REPO))] = {"exists": exists}
        if not exists:
            protected_ok = False
            failures.append(f"protected historical artifact missing: {p}")
    checks["no_historical_overwrite"] = {
        "pass": protected_ok,
        "status": protected_status,
    }

    if not hist_path.is_file():
        failures.append(f"missing epoch history: {hist_path}")
    if not summary_path.is_file():
        failures.append(f"missing summary: {summary_path}")

    hist = json.loads(hist_path.read_text()) if hist_path.is_file() else {}
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    epochs = hist.get("epochs") or []
    meta_args: Dict[str, Any] = {}
    meta_args.update(hist.get("graph_flags") or {})
    meta_args.update(summary.get("graph_flags") or {})
    meta_args.update(hist.get("args") or {})
    meta_args.update(summary.get("args") or {})
    for k in (
        "supervised_head",
        "seed",
        "n_epochs",
        "dataset",
        "model_architecture",
        "model",
        "data",
        "objective",
        "feature_contract",
    ):
        if k in hist and meta_args.get(k) is None:
            meta_args[k] = hist[k]
        if k in summary and meta_args.get(k) is None:
            meta_args[k] = summary[k]
    if meta_args.get("data") is None and meta_args.get("dataset"):
        meta_args["data"] = meta_args["dataset"]
    if meta_args.get("model") is None and meta_args.get("model_architecture"):
        meta_args["model"] = meta_args["model_architecture"]
    if meta_args.get("objective") is None:
        meta_args["objective"] = "supervised"

    # Flags
    flag_expect = {
        "ports": True,
        "tds": False,
        "ego": True,
        "emlps": True,
        "reverse_mp": True,
        "correct_reverse_edge_features": False,
        "preserve_seed_edges": False,
        "train_fit_edge_znorm": True,
        "skip_test_eval": True,
        "save_model": True,
        "seed": 2,
        "n_epochs": 2,
        "data": "PaySim",
        "model": "gin",
        "objective": "supervised",
        "supervised_head": "legacy",
        "feature_contract": CONTRACT_NATIVE_MULTIGIN_CORE,
    }
    flag_results = {}
    flags_ok = True
    for k, exp in flag_expect.items():
        got = meta_args.get(k, summary.get(k, hist.get(k)))
        # train_fit may be auto-enabled without flag in args; accept True
        match = got == exp
        if k == "train_fit_edge_znorm" and got is None:
            # native path auto-enables; tolerate missing if feature_contract is native
            match = meta_args.get("feature_contract") == CONTRACT_NATIVE_MULTIGIN_CORE
        flag_results[k] = {"expected": exp, "got": got, "pass": match}
        if not match:
            flags_ok = False
            failures.append(f"flag {k}: got={got!r} expected={exp!r}")
    checks["candidate_a_native_flags"] = {"pass": flags_ok, "flags": flag_results}

    # Class weights note (AML-derived)
    cw = hist.get("class_weights") or summary.get("class_weights") or {}
    checks["class_weights_aml_derived"] = {
        "pass": True,
        "weights": cw,
        "note": (
            "Candidate-A Adam/lr/class weights retained from gin model_settings "
            "(~1.0, 6.275); AML-derived; later validation-only weighting ablation may be needed."
        ),
    }

    # Cohort integrity
    formatted = REPO / "aml-data/PaySim/formatted_transactions.csv"
    try:
        cohorts = recompute_cohort_hashes(formatted)
        cohort_ok = True
        for split, exp in EXPECTED_SPLITS.items():
            got = cohorts.get(split) or {}
            for key in ("n", "n_positives", "index_sha256"):
                if got.get(key) != exp[key]:
                    cohort_ok = False
                    failures.append(
                        f"cohort {split}.{key}: got={got.get(key)!r} expected={exp[key]!r}"
                    )
        if cohorts.get("formatted_sha256") != FORMATTED_SHA256:
            cohort_ok = False
            failures.append(
                f"formatted_sha256 mismatch: {cohorts.get('formatted_sha256')}"
            )
        checks["cohort_hashes"] = {
            "pass": cohort_ok,
            "computed": cohorts,
            "expected": EXPECTED_SPLITS,
        }
    except Exception as exc:  # noqa: BLE001
        checks["cohort_hashes"] = {"pass": False, "error": str(exc)}
        failures.append(f"cohort hash recompute failed: {exc}")

    # Checkpoints
    ckpt_hashes = {
        "best": sha256_file(best_path) if best_path.is_file() else None,
        "last": sha256_file(last_path) if last_path.is_file() else None,
        "flat": sha256_file(flat_path) if flat_path.is_file() else None,
    }
    reload_ok = True
    reload_detail: Dict[str, Any] = {}
    for label, path in (("best", best_path), ("last", last_path)):
        if not path.is_file():
            reload_ok = False
            reload_detail[label] = {"exists": False}
            failures.append(f"missing checkpoint: {path}")
            continue
        try:
            payload_ckpt = torch.load(path, map_location="cpu", weights_only=False)
            keys = set(payload_ckpt.keys()) if isinstance(payload_ckpt, dict) else set()
            need = {"model_state_dict", "optimizer_state_dict", "epoch", "supervised_head"}
            missing = sorted(need - keys)
            head_ckpt = payload_ckpt.get("supervised_head") if isinstance(payload_ckpt, dict) else None
            ok = not missing and head_ckpt == "legacy"
            sd = payload_ckpt.get("model_state_dict") or {}
            finite_params = True
            n_tensors = 0
            for t in sd.values():
                if torch.is_tensor(t):
                    n_tensors += 1
                    if not torch.isfinite(t).all():
                        finite_params = False
                        break
            if n_tensors == 0 or not finite_params:
                ok = False
            if not ok:
                reload_ok = False
                failures.append(
                    f"checkpoint {label} reload failed missing={missing} "
                    f"head={head_ckpt!r} n_tensors={n_tensors} finite={finite_params}"
                )
            reload_detail[label] = {
                "exists": True,
                "keys_ok": not missing,
                "supervised_head": head_ckpt,
                "epoch": payload_ckpt.get("epoch"),
                "n_tensors": n_tensors,
                "finite_params": finite_params,
                "sha256": ckpt_hashes[label],
            }
        except Exception as exc:  # noqa: BLE001
            reload_ok = False
            reload_detail[label] = {"exists": True, "error": str(exc)}
            failures.append(f"checkpoint {label} load error: {exc}")
    checks["checkpoints_reload"] = {"pass": reload_ok, "detail": reload_detail}

    # Finite losses / scores
    finite_ok = True
    loss_rows = []
    for row in epochs:
        loss = row.get("train_loss")
        auprc = row.get("validation_auprc")
        auroc = row.get("validation_auroc")
        ok = _finite(loss) and _finite(auprc) and _finite(auroc)
        loss_rows.append(
            {
                "epoch": row.get("epoch"),
                "train_loss": loss,
                "validation_auprc": auprc,
                "validation_auroc": auroc,
                "finite": ok,
            }
        )
        if not ok:
            finite_ok = False
            failures.append(f"nonfinite metrics epoch={row.get('epoch')}")
    checks["finite_losses_and_scores"] = {"pass": finite_ok, "rows": loss_rows}

    # Coverage / collapse
    # NeighborLoader can miss a tiny handful of seed edges; require ≈1.0 coverage.
    VAL_N = EXPECTED_SPLITS["val"]["n"]
    VAL_POS = EXPECTED_SPLITS["val"]["n_positives"]
    MAX_MISSING_SEEDS = 16  # absolute; ≈1.0 coverage
    coverage_ok = True
    coverage_rows = []
    for row in epochs:
        n = row.get("validation_n")
        n_pos = row.get("validation_n_positives")
        cov = row.get("validation_positive_coverage")
        if n is None:
            n_ok = True
            seed_cov = None
            missing = None
        else:
            missing = abs(float(n) - VAL_N)
            seed_cov = float(n) / float(VAL_N)
            n_ok = missing <= MAX_MISSING_SEEDS and seed_cov >= 0.999
        npos_ok = n_pos is None or abs(float(n_pos) - VAL_POS) < 1.0
        row_ok = n_ok and npos_ok
        coverage_rows.append(
            {
                "epoch": row.get("epoch"),
                "validation_n": n,
                "validation_n_positives": n_pos,
                "validation_positive_coverage_recall": cov,
                "seed_edge_coverage": seed_cov,
                "missing_seeds_abs": missing,
                "pass": row_ok,
            }
        )
        if not row_ok:
            coverage_ok = False
            failures.append(
                f"val coverage/counts epoch={row.get('epoch')}: n={n} n_pos={n_pos}"
            )
    if not epochs:
        coverage_ok = False
        failures.append("no epochs recorded")
    checks["validation_coverage_counts"] = {
        "pass": coverage_ok,
        "rows": coverage_rows,
        "note": (
            "validation_n must be ≈ locked val cohort (seed-edge coverage ≥ 0.999; "
            f"≤{MAX_MISSING_SEEDS} missing); n_positives exact; "
            "validation_positive_coverage is argmax recall (TP/P), not seed coverage."
        ),
    }

    # Collapse: AUPRC near prevalence or all-zero F1
    val_prev = EXPECTED_SPLITS["val"]["n_positives"] / EXPECTED_SPLITS["val"]["n"]
    auprcs = [float(r.get("validation_auprc", float("nan"))) for r in epochs]
    f1s = [float(r.get("validation_minority_f1_argmax", float("nan"))) for r in epochs]
    max_auprc = max(auprcs) if auprcs and all(_finite(x) for x in auprcs) else float("nan")
    max_f1 = max(f1s) if f1s and all(_finite(x) for x in f1s) else float("nan")
    collapse = bool(
        (not _finite(max_auprc))
        or max_auprc < max(val_prev * 2.0, 0.002)
        or (all(_finite(f) and f < 1e-6 for f in f1s) and len(f1s) >= 1)
    )
    if collapse:
        failures.append(
            f"model collapse suspected: max_auprc={max_auprc} max_f1={max_f1} val_prev={val_prev}"
        )
    checks["no_collapse"] = {
        "pass": not collapse,
        "max_val_auprc": max_auprc,
        "max_val_f1": max_f1,
        "val_prevalence": val_prev,
    }

    # Material improvement vs balance-free Multi-GIN
    material_ok = _finite(max_auprc) and max_auprc >= PASS_THRESHOLD_VAL_AUPRC
    if not material_ok:
        failures.append(
            f"val AUPRC {max_auprc} < pass threshold {PASS_THRESHOLD_VAL_AUPRC} "
            f"(ref {REF_BALANCE_FREE_VAL_AUPRC} + {MARGIN_ABS})"
        )
    checks["material_vs_balance_free_multigin"] = {
        "pass": material_ok,
        "max_val_auprc": max_auprc,
        "reference_val_auprc": REF_BALANCE_FREE_VAL_AUPRC,
        "margin_abs": MARGIN_ABS,
        "pass_threshold": PASS_THRESHOLD_VAL_AUPRC,
    }

    # Test untouched
    TEST_METRIC_KEY_PREFIXES = (
        "test_minority_",
        "test_precision",
        "test_recall",
        "test_auroc",
        "test_auprc",
        "test_f1",
    )
    test_metric_keys = sorted(
        {
            k
            for r in epochs
            for k in r.keys()
            if any(k.startswith(p) for p in TEST_METRIC_KEY_PREFIXES)
        }
    )
    skip_flag = bool(
        meta_args.get("skip_test_eval", False)
        or (summary.get("args") or {}).get("skip_test_eval")
        or (hist.get("graph_flags") or {}).get("skip_test_eval")
    )
    # Summary may still contain NaN placeholders for test_* under skip_test_eval.
    def _absent_or_nan(v: Any) -> bool:
        if v is None:
            return True
        try:
            return isinstance(v, float) and math.isnan(v)
        except TypeError:
            return False

    richer = summary.get("richer_ranking_metrics_at_best") or {}
    test_summary_clean = (
        _absent_or_nan(summary.get("test_minority_f1_argmax_at_best"))
        and _absent_or_nan(summary.get("final_epoch_test_minority_f1_argmax"))
        and richer.get("test_auroc") in (None,)
        and richer.get("test_auprc") in (None,)
    )
    test_ok = (
        bool(epochs)
        and all(r.get("test_evaluated") is False for r in epochs)
        and (not test_metric_keys)
        and skip_flag
        and test_summary_clean
    )
    if not test_ok:
        failures.append(
            f"test_evaluated check failed keys={test_metric_keys} skip={skip_flag} "
            f"summary_test_f1={summary.get('test_minority_f1_argmax_at_best')!r}"
        )
    checks["test_evaluated_false"] = {
        "pass": test_ok,
        "epoch_test_evaluated": [r.get("test_evaluated") for r in epochs],
        "test_metric_keys_in_history": test_metric_keys,
        "skip_test_eval": skip_flag,
        "summary_test_placeholders_nan_or_null": test_summary_clean,
    }

    # Best epochs
    best_f1_epoch = summary.get("best_validation_epoch")
    if best_f1_epoch is None and f1s:
        best_f1_epoch = int(np.argmax(f1s)) + 1
    max_auprc_epoch = int(np.argmax(auprcs)) + 1 if auprcs and all(_finite(x) for x in auprcs) else None

    # Scaler hash from history if present
    scaler_meta = None
    for src in (hist, summary):
        if isinstance(src.get("native_edge_scaler"), dict):
            scaler_meta = src["native_edge_scaler"]
            break
        fc = src.get("feature_contract") or src.get("feature_contract_summary")
        if isinstance(fc, dict) and isinstance(fc.get("scaler"), dict):
            scaler_meta = fc["scaler"]
            break
    if scaler_meta is None:
        for src in (hist, summary):
            a = src.get("args") or {}
            if isinstance(a.get("feature_contract_summary"), dict):
                scaler_meta = a["feature_contract_summary"].get("scaler")
                if scaler_meta:
                    break

    gate_pass = len(failures) == 0 and all(
        c.get("pass") for c in checks.values() if isinstance(c, dict) and "pass" in c
    )

    # Approach HGB?
    vs_hgb = None
    if _finite(max_auprc):
        vs_hgb = {
            "native_hgb_val_auprc": REF_TABLE["native_hgb"],
            "smoke_max_val_auprc": max_auprc,
            "delta": float(max_auprc) - REF_TABLE["native_hgb"],
            "approaches_or_exceeds": bool(max_auprc >= REF_TABLE["native_hgb"] - 0.05),
            "exceeds": bool(max_auprc >= REF_TABLE["native_hgb"]),
            "note": "Learner/protocol differ (GNN vs HGB tabular); compare cautiously.",
        }

    formal_justified = bool(
        gate_pass
        and material_ok
        and (not collapse)
        and test_ok
    )

    epoch_report = []
    for row in epochs:
        epoch_report.append(
            {
                "epoch": row.get("epoch"),
                "train_loss": row.get("train_loss"),
                "validation_auroc": row.get("validation_auroc"),
                "validation_auprc": row.get("validation_auprc"),
                "validation_minority_f1_argmax": row.get("validation_minority_f1_argmax"),
                "validation_precision_argmax": row.get("validation_precision_argmax"),
                "validation_recall_argmax": row.get("validation_recall_argmax"),
                "validation_positive_prediction_rate": row.get(
                    "validation_positive_prediction_rate"
                ),
                "validation_tp": row.get("validation_tp"),
                "validation_fp": row.get("validation_fp"),
                "validation_tn": row.get("validation_tn"),
                "validation_fn": row.get("validation_fn"),
                "validation_positive_coverage": row.get("validation_positive_coverage"),
                "scores_finite": _finite(row.get("validation_auprc"))
                and _finite(row.get("validation_auroc"))
                and _finite(row.get("train_loss")),
            }
        )

    provenance = _git_provenance()
    payload = {
        "artifact": GATE_ID,
        "feature_contract_id": CONTRACT_NATIVE_MULTIGIN_CORE,
        "run_name": run_name,
        "job_id": args.job_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gate_pass": gate_pass,
        "classification": (
            "NATIVE_MULTIGIN_SMOKE_PASS" if gate_pass else "NATIVE_MULTIGIN_SMOKE_FAIL"
        ),
        "scientific_question": (
            "Does supervised Multi-GIN using PaySim native balance features materially "
            "improve over balance-free compatibility-contract Multi-GIN, and can it "
            "compete with native HGB val AUPRC 0.6616?"
        ),
        "deployment_caveat": (
            "newbalanceOrig/newbalanceDest make this a post-transaction supervised "
            "ceiling; may be unavailable pre-authorization."
        ),
        "edge_dim_with_ports": EXPECTED_EDGE_DIM_PORTS,
        "type_onehot_order": list(TYPE_ORDER),
        "feature_names_with_ports": [
            "time",
            "log1p_amount",
            "type_PAYMENT",
            "type_TRANSFER",
            "type_CASH_OUT",
            "type_DEBIT",
            "type_CASH_IN",
            "oldbalanceOrg",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest",
            "in_port",
            "out_port",
        ],
        "references_val_auprc": REF_TABLE,
        "pass_threshold_val_auprc": PASS_THRESHOLD_VAL_AUPRC,
        "checks": checks,
        "failures": failures,
        "epoch_report": epoch_report,
        "best_metrics": {
            "best_val_f1_epoch": best_f1_epoch,
            "best_val_f1": summary.get("validation_minority_f1_argmax_at_best", max_f1),
            "max_val_auprc_epoch": max_auprc_epoch,
            "max_val_auprc": max_auprc,
            "max_val_f1": max_f1,
        },
        "vs_native_hgb": vs_hgb,
        "formal_seed2_justified": formal_justified,
        "formal_50ep_authorized": False,
        "jobs_submitted_this_gate": 0,
        "scaler": scaler_meta,
        "code_provenance": provenance,
        "paths": {
            "epoch_history": str(hist_path.relative_to(REPO)),
            "summary": str(summary_path.relative_to(REPO)),
            "summary_md": str(summary_md.relative_to(REPO)) if summary_md.is_file() else None,
            "checkpoint_dir": str(run_dir.relative_to(REPO)),
            "best_ckpt": str(best_path.relative_to(REPO)) if best_path.is_file() else None,
            "last_ckpt": str(last_path.relative_to(REPO)) if last_path.is_file() else None,
        },
        "checkpoint_sha256": ckpt_hashes,
        "note": "Stop after smoke; do not auto-submit formal 50-epoch.",
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")

    # Submission / status note
    sub = {
        "job_id": args.job_id,
        "run_name": run_name,
        "exact_command": args.command,
        "code_provenance": provenance,
        "dataset_hash_formatted": FORMATTED_SHA256,
        "scaler_sha256": (scaler_meta or {}).get("scaler_sha256") if isinstance(scaler_meta, dict) else None,
        "gate_pass": gate_pass,
        "checkpoint_dir": str(run_dir.relative_to(REPO)),
        "created_utc": payload["created_utc"],
    }
    SUBMISSION_JSON.write_text(json.dumps(sub, indent=2) + "\n")
    SUBMISSION_MD.write_text(
        "\n".join(
            [
                f"# {GATE_ID} submission / status",
                "",
                f"- **Job ID:** `{args.job_id or 'unknown'}`",
                f"- **Run name:** `{run_name}`",
                f"- **Gate pass:** `{gate_pass}`",
                f"- **Command:** `{args.command or 'see slurm script'}`",
                f"- **Git HEAD:** `{provenance.get('git_head')}`",
                f"- **Git describe:** `{provenance.get('git_describe')}`",
                f"- **Formatted SHA256:** `{FORMATTED_SHA256}`",
                f"- **Scaler SHA256:** `{(scaler_meta or {}).get('scaler_sha256') if isinstance(scaler_meta, dict) else 'see training logs / contract_summary'}`",
                f"- **Checkpoints:** `saved-models/{run_name}/`",
                f"- **Twin JSON:** `{OUT_JSON.relative_to(REPO)}`",
                "",
                "Runtime/memory: see Slurm out/err for `Elapsed`, `MaxRSS`, GPU util.",
                "",
                "**Formal 50-epoch: NOT submitted.**",
                "",
            ]
        )
    )

    # Markdown report
    lines = [
        f"# PaySim-native Multi-GIN smoke (`{CONTRACT_NATIVE_MULTIGIN_CORE}`)",
        "",
        f"- **Gate pass:** `{gate_pass}`",
        f"- **Job:** `{args.job_id or 'unknown'}`",
        f"- **Run:** `{run_name}`",
        f"- **Edge dim (ports on, TDS off):** `{EXPECTED_EDGE_DIM_PORTS}`",
        "",
        "## Scientific question",
        "",
        payload["scientific_question"],
        "",
        "## Deployment caveat",
        "",
        payload["deployment_caveat"],
        "",
        "## Protocol",
        "",
        "- Candidate-A Multi-GIN (legacy head, emlps/reverse_mp/ego/ports on, tds off)",
        "- Native feature contract (not AML transfer / not contrastive / not paper table claim)",
        "- Adam + AML-derived class weights ~(1.0, 6.275); may need later weighting ablation",
        "- 2 epochs, seed 2, `--skip_test_eval`, `--save_model`",
        "- Locked steps: train 1–280 / val 281–354 / test 355–743 (test locked)",
        "",
        "## Features (exact order)",
        "",
    ]
    for i, name in enumerate(payload["feature_names_with_ports"], 1):
        lines.append(f"{i}. `{name}`")
    lines.extend(
        [
            "",
            "## Epoch report",
            "",
            "```json",
            json.dumps(epoch_report, indent=2),
            "```",
            "",
            f"- **Best val-F1 epoch:** `{best_f1_epoch}` (F1=`{payload['best_metrics']['best_val_f1']}`)",
            f"- **Max val-AUPRC epoch:** `{max_auprc_epoch}` (AUPRC=`{max_auprc}`)",
            "",
            "## Validation-only comparisons (different learners/protocols)",
            "",
            "| Reference | Val AUPRC |",
            "|-----------|----------:|",
            f"| Legacy compatibility X-only | ~{REF_TABLE['legacy_compatibility_x_only']} |",
            f"| Balance-free Multi-GIN Candidate A | ~{REF_TABLE['balance_free_multigin_candidate_a']} |",
            f"| Native logistic | {REF_TABLE['native_logistic']} |",
            f"| Native MLP | {REF_TABLE['native_mlp']} |",
            f"| Native HGB | {REF_TABLE['native_hgb']} |",
            f"| **This smoke (max)** | **{max_auprc}** |",
            "",
            f"- Pass threshold vs Multi-GIN: `{PASS_THRESHOLD_VAL_AUPRC}` "
            f"(~{REF_BALANCE_FREE_VAL_AUPRC} + {MARGIN_ABS})",
            f"- Material exceed Multi-GIN: `{material_ok}`",
            f"- vs HGB: `{json.dumps(vs_hgb)}`",
            "",
            "## Checks",
            "",
        ]
    )
    for name, c in checks.items():
        lines.append(f"- `{name}`: **{'PASS' if c.get('pass') else 'FAIL'}**")
    if failures:
        lines.extend(["", "## Failures", ""])
        for f in failures:
            lines.append(f"- {f}")
    lines.extend(
        [
            "",
            "## Formal seed-2",
            "",
            f"- **Justified by this smoke:** `{formal_justified}`",
            "- **Auto-submitted:** `false` (hard stop after smoke)",
            "",
            f"Twin JSON: `{OUT_JSON.relative_to(REPO)}`",
            "",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n")

    append_registry(payload)

    print(
        json.dumps(
            {
                "gate_pass": gate_pass,
                "failures": failures,
                "max_val_auprc": max_auprc,
                "formal_seed2_justified": formal_justified,
                "out": str(OUT_JSON),
            }
        )
    )
    return 0 if gate_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
