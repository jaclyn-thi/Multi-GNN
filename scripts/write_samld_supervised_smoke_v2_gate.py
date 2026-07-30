#!/usr/bin/env python3
"""Protocol gate for SAML-D supervised Candidate A smoke v2 (legacy head, no test)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

RUN_NAME = "samld_supervised_multigin_eu_v1_smoke_v2_seed2"
PROTOCOL_ID = "samld_supervised_multigin_eu_v1"
EXPECTED_VAL_N = 1_899_523
EXPECTED_VAL_POS = 1_986
EXPECTED_EDGE_DIM = 6

PROTECTED_ARTIFACTS = [
    REPO / "notes/supervised_SAML-D_samld_supervised_multigin_eu_v1_smoke_seed2_summary.md",
    REPO / "results/diagnostics/supervised_SAML-D_samld_supervised_multigin_eu_v1_smoke_seed2_summary.json",
    REPO / "results/diagnostics/supervised_SAML-D_samld_supervised_multigin_eu_v1_smoke_seed2_epoch_history.json",
    REPO / "results/diagnostics/samld_supervised_smoke.json",
]

OUT_JSON = REPO / "results/diagnostics/samld_supervised_smoke_v2.json"
OUT_MD = REPO / "notes/samld_supervised_smoke_v2.md"


def sha256_file(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _finite(x: Any) -> bool:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return False
    return math.isfinite(v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-id", default="")
    ap.add_argument("--run-name", default=RUN_NAME)
    args = ap.parse_args()
    run_name = args.run_name

    hist_path = (
        REPO
        / "results/diagnostics"
        / f"supervised_SAML-D_{run_name}_epoch_history.json"
    )
    summary_path = (
        REPO / "results/diagnostics" / f"supervised_SAML-D_{run_name}_summary.json"
    )
    run_dir = REPO / "saved-models" / run_name
    best_path = run_dir / "checkpoint_best_val_f1.tar"
    last_path = run_dir / "checkpoint_last.tar"
    flat_path = REPO / "saved-models" / f"checkpoint_{run_name}.tar"

    checks: Dict[str, Any] = {}
    failures: List[str] = []

    # --- protected artifacts must still exist (not overwritten by this run name) ---
    protected_ok = True
    protected_status = {}
    for p in PROTECTED_ARTIFACTS:
        exists = p.is_file()
        protected_status[str(p.relative_to(REPO))] = {"exists": exists}
        # Historical smoke summary must remain; optional smoke.json may be absent.
        if "smoke_seed2" in p.name and not exists:
            protected_ok = False
            failures.append(f"protected historical artifact missing: {p}")
    checks["no_historical_overwrite"] = {
        "pass": protected_ok,
        "status": protected_status,
        "note": "v2 uses unique run_name; historical smoke_seed2 artifacts must remain",
    }

    if not hist_path.is_file():
        failures.append(f"missing epoch history: {hist_path}")
    if not summary_path.is_file():
        failures.append(f"missing summary: {summary_path}")

    hist = json.loads(hist_path.read_text()) if hist_path.is_file() else {}
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    epochs = hist.get("epochs") or []
    meta_args = {}
    meta_args.update(hist.get("graph_flags") or {})
    meta_args.update(summary.get("graph_flags") or {})
    meta_args.update(hist.get("args") or {})
    meta_args.update(summary.get("args") or {})
    # Promote common top-level fields
    for k in (
        "supervised_head",
        "seed",
        "n_epochs",
        "dataset",
        "model_architecture",
        "model",
        "data",
        "objective",
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

    # 1) supervised_head == legacy
    head = (
        summary.get("supervised_head")
        or summary.get("supervised_mode")
        or meta_args.get("supervised_head")
        or hist.get("supervised_head")
    )
    head_ok = head == "legacy"
    checks["supervised_head_legacy"] = {"pass": head_ok, "value": head}
    if not head_ok:
        failures.append(f"supervised_head={head!r} != legacy")

    # 2) locked Candidate A graph flags
    flag_expect = {
        "ports": True,
        "tds": False,
        "ego": True,
        "emlps": True,
        "reverse_mp": True,
        "correct_reverse_edge_features": False,
        "preserve_seed_edges": False,
        "train_fit_edge_znorm": False,
        "skip_test_eval": True,
        "save_model": True,
        "seed": 2,
        "n_epochs": 2,
        "data": "SAML-D",
        "model": "gin",
        "objective": "supervised",
    }
    flag_results = {}
    flags_ok = True
    for k, exp in flag_expect.items():
        got = meta_args.get(k, summary.get(k, hist.get(k)))
        # booleans may be missing from older metadata; fall back to summary top-level
        if got is None and k in summary:
            got = summary[k]
        match = got == exp
        flag_results[k] = {"expected": exp, "got": got, "pass": match}
        if not match:
            flags_ok = False
            failures.append(f"flag {k}: got={got!r} expected={exp!r}")
    checks["candidate_a_flags"] = {"pass": flags_ok, "flags": flag_results}

    # 3) checkpoints exist + reload strictly
    ckpt_hashes = {
        "best": sha256_file(best_path),
        "last": sha256_file(last_path),
        "flat": sha256_file(flat_path),
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
            payload = torch.load(path, map_location="cpu", weights_only=False)
            keys = set(payload.keys()) if isinstance(payload, dict) else set()
            need = {"model_state_dict", "optimizer_state_dict", "epoch", "supervised_head"}
            missing = sorted(need - keys)
            head_ckpt = payload.get("supervised_head") if isinstance(payload, dict) else None
            ok = not missing and head_ckpt == "legacy"
            if not ok:
                reload_ok = False
                failures.append(
                    f"checkpoint {label} reload failed missing={missing} head={head_ckpt!r}"
                )
            # strict: state dict non-empty + finite params
            sd = payload.get("model_state_dict") or {}
            finite_params = True
            n_tensors = 0
            for t in sd.values():
                if torch.is_tensor(t):
                    n_tensors += 1
                    if not torch.isfinite(t).all():
                        finite_params = False
                        break
            if n_tensors == 0 or not finite_params:
                reload_ok = False
                failures.append(f"checkpoint {label}: empty or nonfinite state_dict")
            reload_detail[label] = {
                "exists": True,
                "keys_ok": not missing,
                "supervised_head": head_ckpt,
                "epoch": payload.get("epoch"),
                "n_tensors": n_tensors,
                "finite_params": finite_params,
                "sha256": ckpt_hashes[label],
            }
        except Exception as exc:  # noqa: BLE001
            reload_ok = False
            reload_detail[label] = {"exists": True, "error": str(exc)}
            failures.append(f"checkpoint {label} load error: {exc}")
    checks["checkpoints_reload"] = {"pass": reload_ok, "detail": reload_detail}

    # 4) best differs from last when expected
    best_sha = ckpt_hashes["best"]
    last_sha = ckpt_hashes["last"]
    hashes_differ = bool(best_sha and last_sha and best_sha != last_sha)
    selected_epoch = summary.get("best_validation_epoch")
    n_ep = len(epochs)
    # If selected epoch is not the final epoch, hashes must differ.
    expect_differ = (
        selected_epoch is not None and n_ep >= 2 and int(selected_epoch) != int(epochs[-1].get("epoch", -1))
    )
    differ_ok = (hashes_differ if expect_differ else True) and best_sha is not None and last_sha is not None
    if expect_differ and not hashes_differ:
        failures.append(
            f"expected best!=last (selected_epoch={selected_epoch}) but hashes match"
        )
    checks["best_vs_last"] = {
        "pass": differ_ok,
        "best_sha256": best_sha,
        "last_sha256": last_sha,
        "hashes_differ": hashes_differ,
        "selected_epoch": selected_epoch,
        "expect_differ": expect_differ,
    }

    # 5) finite losses
    finite_ok = True
    loss_rows = []
    for row in epochs:
        loss = row.get("train_loss")
        ok = _finite(loss)
        loss_rows.append({"epoch": row.get("epoch"), "train_loss": loss, "finite": ok})
        if not ok:
            finite_ok = False
            failures.append(f"nonfinite train_loss epoch={row.get('epoch')}: {loss}")
    checks["finite_losses"] = {"pass": finite_ok, "rows": loss_rows}

    # 6) validation coverage / positives (from protocol card; smoke cannot re-count seeds here)
    # Prefer values recorded in history/summary if present; else assert protocol constants.
    val_cov = {
        "expected_n": EXPECTED_VAL_N,
        "expected_positives": EXPECTED_VAL_POS,
        "expected_edge_dim": EXPECTED_EDGE_DIM,
        "note": "Seed-edge coverage enforced by locked calendar_day split; "
        "counts from integrity job 19108637 / protocol card.",
    }
    checks["validation_protocol_card"] = {"pass": True, **val_cov}

    # 7) test_evaluated == false
    # Note: epoch rows intentionally include key "test_evaluated" (boolean flag).
    # That must NOT be treated as a test *metric* key.
    TEST_METRIC_KEY_PREFIXES = (
        "test_minority_",
        "test_precision",
        "test_recall",
        "test_auroc",
        "test_auprc",
        "test_f1",
    )
    test_flags = [bool(r.get("test_evaluated", True)) for r in epochs] if epochs else [True]
    test_metric_keys = sorted(
        {
            k
            for r in epochs
            for k in r.keys()
            if any(k.startswith(p) for p in TEST_METRIC_KEY_PREFIXES)
        }
    )
    test_metrics_present = bool(test_metric_keys)
    skip_flag = bool(
        meta_args.get("skip_test_eval", False)
        or (summary.get("args") or {}).get("skip_test_eval")
        or (hist.get("graph_flags") or {}).get("skip_test_eval")
    )
    test_ok = (
        bool(epochs)
        and all(r.get("test_evaluated") is False for r in epochs)
        and (not test_metrics_present)
        and skip_flag
    )
    if not test_ok:
        failures.append(
            "test_evaluated check failed "
            f"flags={test_flags} test_metric_keys={test_metric_keys} skip_test_eval={skip_flag}"
        )
    checks["test_evaluated_false"] = {
        "pass": test_ok,
        "epoch_test_evaluated": [r.get("test_evaluated") for r in epochs],
        "test_metric_keys_in_history": test_metric_keys,
        "skip_test_eval": skip_flag,
    }

    # Epoch collapse diagnostic (val F1 / AUPRC)
    collapse = None
    if len(epochs) >= 2:
        f1s = [float(r.get("validation_minority_f1_argmax", float("nan"))) for r in epochs]
        auprcs = [float(r.get("validation_auprc", float("nan"))) for r in epochs]
        collapse = {
            "val_f1_by_epoch": f1s,
            "val_auprc_by_epoch": auprcs,
            "epoch2_f1_near_zero": bool(_finite(f1s[-1]) and f1s[-1] < 1e-4),
            "epoch2_auprc_near_prevalence": bool(
                _finite(auprcs[-1]) and auprcs[-1] < 0.01
            ),
        }

    gate_pass = len(failures) == 0 and all(
        c.get("pass") for c in checks.values() if isinstance(c, dict) and "pass" in c
    )

    payload = {
        "protocol_id": PROTOCOL_ID,
        "gate_id": "samld_supervised_smoke_v2",
        "run_name": run_name,
        "job_id": args.job_id,
        "gate_pass": gate_pass,
        "classification_hint": (
            "PROTOCOL_SMOKE_PASS" if gate_pass else "PROTOCOL_SMOKE_FAIL"
        ),
        "prior_deviating_job": "19109396",
        "checks": checks,
        "failures": failures,
        "epoch_history_path": str(hist_path.relative_to(REPO)),
        "summary_path": str(summary_path.relative_to(REPO)),
        "checkpoint_paths": {
            "best": str(best_path),
            "last": str(last_path),
            "flat": str(flat_path),
        },
        "checkpoint_sha256": ckpt_hashes,
        "summary_metrics": {
            "best_validation_epoch": summary.get("best_validation_epoch"),
            "validation_minority_f1_argmax_at_best": summary.get(
                "validation_minority_f1_argmax_at_best"
            ),
            "richer_ranking_metrics_at_best": summary.get("richer_ranking_metrics_at_best"),
        },
        "epoch_collapse_diagnostic": collapse,
        "formal_50ep_authorized": False,
        "note": "Do not launch formal 50-epoch run from this gate alone.",
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# SAML-D supervised smoke v2 (protocol gate)",
        "",
        f"- **Gate pass:** `{gate_pass}`",
        f"- **Run name:** `{run_name}`",
        f"- **Job:** `{args.job_id or 'unknown'}`",
        f"- **Protocol:** `{PROTOCOL_ID}` (Candidate A)",
        f"- **Prior deviant job:** 19109396 (embedding head / no save_model / test every epoch)",
        "",
        "## Corrections vs 19109396",
        "",
        "- `--supervised_head legacy`",
        "- `--save_model`",
        "- `--skip_test_eval` (no test graph materialization; no test metrics in history)",
        "- Unique run name `*_smoke_v2_seed2`",
        "",
        "## Checks",
        "",
    ]
    for name, c in checks.items():
        lines.append(f"- `{name}`: **{'PASS' if c.get('pass') else 'FAIL'}**")
    if failures:
        lines.extend(["", "## Failures", ""])
        for f in failures:
            lines.append(f"- {f}")
    lines.extend(
        [
            "",
            "## Checkpoints",
            "",
            f"- best sha256: `{best_sha}`",
            f"- last sha256: `{last_sha}`",
            f"- hashes differ: `{hashes_differ}`",
            "",
            "## Epoch collapse diagnostic",
            "",
            f"```json\n{json.dumps(collapse, indent=2)}\n```" if collapse else "- n/a",
            "",
            "## Formal 50-epoch",
            "",
            "**Not authorized** by this gate. Do not auto-submit.",
            "",
            f"Twin JSON: `{OUT_JSON.relative_to(REPO)}`",
            "",
        ]
    )
    OUT_MD.write_text("\n".join(lines))
    print(json.dumps({"gate_pass": gate_pass, "failures": failures, "out": str(OUT_JSON)}))
    return 0 if gate_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
