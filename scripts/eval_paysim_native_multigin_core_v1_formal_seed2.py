#!/usr/bin/env python3
"""Post-selection eval for PaySim native Multi-GIN formal seed-2.

Loads only checkpoint_best_val_f1.tar; evaluates val+test once.
Threshold from validation only (no test-driven selection).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch_geometric.nn import to_hetero

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from data_loading import get_data  # noqa: E402
from train_util import AddEgoIds, add_arange_ids, get_loaders  # noqa: E402
from training import get_model  # noqa: E402
from util import create_parser, logger_setup, set_seed  # noqa: E402

import importlib.util as _ilu  # noqa: E402

_spec = _ilu.spec_from_file_location(
    "evaluate_supervised_gnn",
    REPO / "scripts" / "evaluate_supervised_gnn.py",
)
_mod = _ilu.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
build_model_config = _mod.build_model_config
collect_split_predictions = _mod.collect_split_predictions
split_metrics = _mod.split_metrics
tune_threshold = _mod.tune_threshold

RUN_NAME = "paysim_native_multigin_core_v1_formal_seed2"
CONTRACT = "paysim_native_multigin_core_v1"
OUT_JSON = REPO / "results/diagnostics/paysim_native_multigin_core_v1_formal_seed2.json"
OUT_MD = REPO / "notes/paysim_native_multigin_core_v1_formal_seed2.md"
OUT_DIR = REPO / "results/diagnostics/paysim_native_multigin_core_v1_formal_seed2"
REG_CSV = REPO / "results/diagnostics/thesis_experiment_registry.csv"
REG_JSON = REPO / "results/diagnostics/thesis_experiment_registry.json"
REG_MD = REPO / "notes/thesis_experiment_registry.md"

EXPECTED_VAL = {"n": 1_276_276, "n_positives": 780}
EXPECTED_TEST = {"n": 1_293_523, "n_positives": 4258}
SMOKE_SCALER_SHA = "45ce032c08ae0f3ef73f11f3a778bbc351da7bd43b3316ab583c941d4bcbae27"

# Comparators (protocol differences labeled in MD/JSON)
NATIVE_HGB_TEST = {"auprc": 0.6992, "f1": 0.8134, "source": "paysim_native_core_v1 HGB"}
BALANCE_FREE_GIN_SEED2 = {
    "val_auprc": 0.1662,
    "test_auprc": 0.2531,
    "test_f1_argmax": 0.1943,
    "source": "paysim_supervised_multigin_eu_seed2 (legacy duplicate X, edge_dim=6)",
}
SMOKE = {
    "max_val_auprc": 0.6665,
    "best_val_f1": 0.6754,
    "job_id": "19123387",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_state_dict(sd: Dict[str, Any]) -> str:
    h = hashlib.sha256()
    for k in sorted(sd.keys()):
        h.update(k.encode())
        t = sd[k]
        if torch.is_tensor(t):
            h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def append_registry_rows(rows: List[Dict[str, Any]]) -> None:
    if REG_CSV.is_file():
        with REG_CSV.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            existing = list(reader)
        for row in rows:
            existing.append({k: row.get(k, "") for k in fieldnames})
        with REG_CSV.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(existing)

    if REG_JSON.is_file():
        payload = json.loads(REG_JSON.read_text())
        payload_rows = payload.get("rows") or []
        existing_ids = {r.get("run_id") for r in payload_rows}
        for row in rows:
            if row.get("run_id") in existing_ids:
                continue
            payload_rows.append(row)
            existing_ids.add(row.get("run_id"))
        payload["rows"] = payload_rows
        payload["row_count"] = len(payload_rows)
        REG_JSON.write_text(json.dumps(payload, indent=2) + "\n")

    with REG_MD.open("a", encoding="utf-8") as f:
        f.write(
            "\n### PaySim native Multi-GIN formal seed-2 (appended)\n"
            f"- Added {len(rows)} row(s) for `{RUN_NAME}` at "
            f"{datetime.now(timezone.utc).isoformat()}\n"
            f"- Source: `{OUT_JSON.relative_to(REPO)}`\n"
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-job-id", required=True)
    ap.add_argument("--eval-job-id", default="")
    args_ns = ap.parse_args()

    logger_setup()
    parser = create_parser()
    argv = [
        "--data", "PaySim",
        "--model", "gin",
        "--objective", "supervised",
        "--supervised_head", "legacy",
        "--feature_contract", CONTRACT,
        "--train_fit_edge_znorm",
        "--unique_name", RUN_NAME,
        "--seed", "2",
        "--batch_size", "8192",
        "--num_neighs", "100", "100",
        "--loader_num_workers", "8",
        "--reverse_mp", "--ego", "--ports", "--emlps",
        "--tqdm",
        # NOT skip_test_eval — this job evaluates test once
    ]
    args = parser.parse_args(argv)
    set_seed(args.seed)

    with open("data_config.json", encoding="utf-8") as f:
        data_config = json.load(f)

    ckpt_path = (
        Path(data_config["paths"]["model_to_load"])
        / RUN_NAME
        / "checkpoint_best_val_f1.tar"
    )
    if not ckpt_path.is_file():
        raise SystemExit(f"missing best-val checkpoint: {ckpt_path}")

    train_final = OUT_DIR / "train_finalize.json"
    train_meta = json.loads(train_final.read_text()) if train_final.is_file() else {}
    hist_path = (
        REPO / "results/diagnostics" / f"supervised_PaySim_{RUN_NAME}_epoch_history.json"
    )
    hist = json.loads(hist_path.read_text()) if hist_path.is_file() else {}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    if checkpoint.get("supervised_head") != "legacy":
        raise SystemExit(f"bad supervised_head in ckpt: {checkpoint.get('supervised_head')}")
    ckpt_args = checkpoint.get("args") or {}
    for k, exp in (
        ("ports", True),
        ("tds", False),
        ("ego", True),
        ("emlps", True),
        ("reverse_mp", True),
        ("correct_reverse_edge_features", False),
        ("preserve_seed_edges", False),
        ("train_fit_edge_znorm", True),
        ("feature_contract", CONTRACT),
        ("seed", 2),
        ("data", "PaySim"),
        ("model", "gin"),
        ("supervised_head", "legacy"),
    ):
        got = ckpt_args.get(k, checkpoint.get(k))
        if got != exp:
            raise SystemExit(f"checkpoint protocol mismatch {k}: got={got!r} expected={exp!r}")

    model_state_sha = sha256_state_dict(checkpoint.get("model_state_dict") or {})
    expected_sha = ((train_meta.get("checkpoints") or {}).get("best") or {}).get(
        "model_state_sha256"
    )
    if expected_sha and expected_sha != model_state_sha:
        raise SystemExit(f"model state hash mismatch: {model_state_sha} != {expected_sha}")

    tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(args, data_config)
    # Confirm scaler matches smoke (train-fit continuous)
    scaler = getattr(args, "native_edge_scaler", None) or {}
    scaler_sha = scaler.get("scaler_sha256") if isinstance(scaler, dict) else None
    if scaler_sha != SMOKE_SCALER_SHA:
        raise SystemExit(
            f"scaler sha mismatch vs smoke: got={scaler_sha} expected={SMOKE_SCALER_SHA}"
        )

    transform = AddEgoIds() if args.ego else None
    add_arange_ids([tr_data, val_data, te_data])
    _, val_loader, te_loader = get_loaders(
        tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform, args, train_shuffle=False
    )

    sample_batch = next(iter(val_loader))
    config = build_model_config(args)
    model = get_model(sample_batch, config, args)
    if args.reverse_mp:
        model = to_hetero(model, te_data.metadata(), aggr="mean")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    y_val, p_val, cov_val = collect_split_predictions(
        val_loader, val_inds, model, val_data, device, args
    )
    y_te, p_te, cov_te = collect_split_predictions(
        te_loader, te_inds, model, te_data, device, args
    )

    # Allow NeighborLoader −2 style missing seeds on val (≤16)
    if abs(int(y_val.shape[0]) - EXPECTED_VAL["n"]) > 16:
        raise SystemExit(
            f"val cohort n mismatch: {y_val.shape[0]} vs {EXPECTED_VAL['n']}"
        )
    if int(y_val.sum()) != EXPECTED_VAL["n_positives"]:
        raise SystemExit(
            f"val positives mismatch: {int(y_val.sum())} vs {EXPECTED_VAL['n_positives']}"
        )
    if abs(int(y_te.shape[0]) - EXPECTED_TEST["n"]) > 16:
        raise SystemExit(
            f"test cohort n mismatch: {y_te.shape[0]} vs {EXPECTED_TEST['n']}"
        )
    if int(y_te.sum()) != EXPECTED_TEST["n_positives"]:
        raise SystemExit(
            f"test positives mismatch: {int(y_te.sum())} vs {EXPECTED_TEST['n_positives']}"
        )

    thr, _ = tune_threshold(y_val, p_val)
    val_metrics = split_metrics(y_val, p_val, thr)
    te_metrics = split_metrics(y_te, p_te, thr)
    fixed05_val = split_metrics(y_val, p_val, 0.5)
    fixed05_te = split_metrics(y_te, p_te, 0.5)

    selected_epoch = checkpoint.get("selected_epoch")
    max_auprc_ep = train_meta.get("max_validation_auprc_epoch_diagnostic")

    payload = {
        "artifact": "paysim_native_multigin_core_v1_formal_seed2",
        "protocol_id": CONTRACT,
        "feature_contract_id": CONTRACT,
        "run_name": RUN_NAME,
        "seed": 2,
        "train_job_id": args_ns.train_job_id,
        "eval_job_id": args_ns.eval_job_id or None,
        "checkpoint_path": str(ckpt_path),
        "checkpoint_archive_sha256": sha256_file(ckpt_path),
        "model_state_sha256": model_state_sha,
        "supervised_head": "legacy",
        "selection_rule": "validation_minority_f1_argmax",
        "selected_epoch": selected_epoch,
        "max_validation_auprc_epoch_diagnostic": max_auprc_ep,
        "max_validation_auprc_diagnostic": train_meta.get(
            "max_validation_auprc_diagnostic"
        ),
        "scaler_sha256": scaler_sha,
        "test_used_for_selection": False,
        "test_evaluated_exactly_once": True,
        "graph_flags": {
            "emlps": True,
            "reverse_mp": True,
            "ego": True,
            "ports": True,
            "tds": False,
            "correct_reverse_edge_features": False,
            "preserve_seed_edges": False,
            "train_fit_edge_znorm": True,
            "reverse_edge_feature_semantics": "inherited_legacy",
            "edge_dim": 13,
            "includes_balance_deltas": False,
            "includes_isFlaggedFraud": False,
        },
        "deployment_caveat": (
            "newbalanceOrig/newbalanceDest are post-transaction fields; "
            "may be unavailable pre-authorization."
        ),
        "coverage": {"val": cov_val, "test": cov_te},
        "cohorts": {
            "val": {
                "n": int(y_val.shape[0]),
                "n_positives": int(y_val.sum()),
                "prevalence": float(y_val.mean()),
            },
            "test": {
                "n": int(y_te.shape[0]),
                "n_positives": int(y_te.sum()),
                "prevalence": float(y_te.mean()),
            },
        },
        "primary_metric": "paper_argmax",
        "splits": {
            "val": {
                "paper_argmax": val_metrics["paper_argmax"],
                "fixed_0.5": fixed05_val["paper_argmax"],
                "validation_tuned_threshold": val_metrics["validation_tuned_threshold"],
                "auroc": val_metrics["auroc"],
                "auprc": val_metrics["auprc"],
                "alert_budget": val_metrics["alert_budget"],
            },
            "test": {
                "paper_argmax": te_metrics["paper_argmax"],
                "fixed_0.5": fixed05_te["paper_argmax"],
                "validation_tuned_threshold": te_metrics["validation_tuned_threshold"],
                "auroc": te_metrics["auroc"],
                "auprc": te_metrics["auprc"],
                "alert_budget": te_metrics["alert_budget"],
            },
        },
        "validation_tuned_threshold_value": thr,
        "comparisons": {
            "native_hgb_test": NATIVE_HGB_TEST,
            "balance_free_multigin_seed2": BALANCE_FREE_GIN_SEED2,
            "smoke_two_epoch": SMOKE,
            "protocol_difference_note": (
                "Native HGB is tabular (no graph MP). Balance-free Multi-GIN uses "
                "paysim_legacy_duplicate_v1 edge_dim=6 without balances. "
                "This run uses paysim_native_multigin_core_v1 edge_dim=13 with "
                "train-fit continuous z-norm. Do not treat gaps as pure architecture effects."
            ),
        },
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "epoch_history_path": str(hist_path.relative_to(REPO)) if hist_path.is_file() else None,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "eval_cell.json").write_text(json.dumps(payload, indent=2) + "\n")
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")

    va = payload["splits"]["val"]
    te = payload["splits"]["test"]
    ab_te = te["alert_budget"]
    md = f"""# PaySim native Multi-GIN formal seed-2 (`{CONTRACT}`)

> Twin: `{OUT_JSON.relative_to(REPO)}`  
> Train job: `{args_ns.train_job_id}` · Eval job: `{args_ns.eval_job_id or 'n/a'}`  
> Checkpoint: `{ckpt_path}` (best-val F1; selected epoch **{selected_epoch}**)

## Protocol (locked to smoke)

- gin + legacy head + emlps + reverse_mp + ego + ports; TDS/preserve/correct_reverse off
- native 13-d edge contract; train-fit continuous z-norm; one-hots unchanged
- Adam + AML-derived class weights ~(1.0, 6.275); seed=2; 50 epochs
- Training used `--skip_test_eval`. This eval scores val+test **once** after selection.
- Scaler SHA256: `{scaler_sha}` (must match smoke `{SMOKE_SCALER_SHA}`)

## Deployment caveat

{payload['deployment_caveat']}

## Primary (paper argmax / fixed-0.5)

| Split | AUROC | AUPRC | F1 | P | R | PPR |
|-------|------:|------:|---:|--:|--:|----:|
| Val | {va['auroc']:.4f} | {va['auprc']:.4f} | {va['paper_argmax']['f1']:.4f} | {va['paper_argmax']['precision']:.4f} | {va['paper_argmax']['recall']:.4f} | {va['paper_argmax']['positive_prediction_rate']:.6f} |
| Test | {te['auroc']:.4f} | {te['auprc']:.4f} | {te['paper_argmax']['f1']:.4f} | {te['paper_argmax']['precision']:.4f} | {te['paper_argmax']['recall']:.4f} | {te['paper_argmax']['positive_prediction_rate']:.6f} |

Confusion (argmax): val TP/FP/TN/FN = {va['paper_argmax']['tp']:.0f}/{va['paper_argmax']['fp']:.0f}/{va['paper_argmax']['tn']:.0f}/{va['paper_argmax']['fn']:.0f};  
test TP/FP/TN/FN = {te['paper_argmax']['tp']:.0f}/{te['paper_argmax']['fp']:.0f}/{te['paper_argmax']['tn']:.0f}/{te['paper_argmax']['fn']:.0f}.

## Validation-selected threshold (not paper-compatible; reported only)

- thr={thr:.6f}
- Test F1/P/R at val-tuned thr: {te['validation_tuned_threshold']['f1']:.4f} / {te['validation_tuned_threshold']['precision']:.4f} / {te['validation_tuned_threshold']['recall']:.4f}

## Alert budgets (P@K)

Test P@100={ab_te.get('precision_at_100')}, P@500={ab_te.get('precision_at_500')}, P@1000={ab_te.get('precision_at_1000')}  
(full keys in JSON `splits.*.alert_budget`)

## Diagnostics (not selection)

- Max-validation-AUPRC epoch (diagnostic): **{max_auprc_ep}**
- Smoke max val AUPRC: {SMOKE['max_val_auprc']} (job {SMOKE['job_id']})

## Comparisons (protocol differences labeled)

| Reference | Metric | Value |
|-----------|--------|------:|
| Native HGB (`paysim_native_core_v1`, tabular) | test AUPRC / F1@0.5 | {NATIVE_HGB_TEST['auprc']} / {NATIVE_HGB_TEST['f1']} |
| Balance-free Multi-GIN seed2 (legacy X, dim=6) | test AUPRC / F1 argmax | {BALANCE_FREE_GIN_SEED2['test_auprc']} / {BALANCE_FREE_GIN_SEED2['test_f1_argmax']} |
| This formal native Multi-GIN | test AUPRC / F1 argmax | {te['auprc']:.4f} / {te['paper_argmax']['f1']:.4f} |

{payload['comparisons']['protocol_difference_note']}

## Selection integrity

- `test_used_for_selection`: false
- `test_evaluated_exactly_once`: true
"""
    OUT_MD.write_text(md)

    def _row(split: str, block: Dict[str, Any], ab: Dict[str, Any]) -> Dict[str, Any]:
        pa = block["paper_argmax"]
        return {
            "run_id": f"{RUN_NAME}|paper_argmax|{split}",
            "dataset": "PaySim",
            "dataset_positive_rate": payload["cohorts"][split]["prevalence"],
            "objective": "supervised",
            "encoder": "gin",
            "seed": 2,
            "training_epochs": 50,
            "selected_epoch": selected_epoch,
            "checkpoint_policy": "best_val_f1",
            "supervised_head": "legacy",
            "feature_contract_id": CONTRACT,
            "graph_flags": "reverse_mp,ego,ports,emlps,train_fit_edge_znorm",
            "emlps": True,
            "tds": False,
            "reverse_mp": True,
            "ego": True,
            "ports": True,
            "threshold_rule": "paper_argmax",
            "AUROC": block["auroc"],
            "AUPRC": block["auprc"],
            "F1": pa["f1"],
            "F1_fixed": pa["f1"],
            "precision": pa["precision"],
            "recall": pa["recall"],
            "precision_at_100": ab.get("precision_at_100"),
            "recall_at_100": ab.get("recall_at_100"),
            "precision_at_500": ab.get("precision_at_500"),
            "recall_at_500": ab.get("recall_at_500"),
            "precision_at_1000": ab.get("precision_at_1000"),
            "recall_at_1000": ab.get("recall_at_1000"),
            "source_json": str(OUT_JSON.relative_to(REPO)),
            "source_note": str(OUT_MD.relative_to(REPO)),
            "checkpoint_path": str(ckpt_path),
            "status": "evaluated",
            "scout_or_formal": "formal",
            "superseded": False,
            "thesis_role": "thesis_supporting",
            "validation_status": "validated",
            "table_eligible": True,
            "table_group": "paysim_native_multigin_core_v1",
            "duplicate_resolution": "not_duplicate",
            "caveats": (
                "PaySim native Multi-GIN formal seed-2; post-transaction balance ceiling; "
                "not balance-free Multi-GIN; not tabular HGB"
            ),
            "protocol_family": CONTRACT,
            "split_protocol": "hourly_steps_1_280_281_354_355_743",
            "reverse_feature_semantics": "inherited_legacy",
            "preserve_seed_edges": False,
            "job_id": args_ns.eval_job_id or args_ns.train_job_id,
            "paper_comparable": False,
        }

    append_registry_rows(
        [
            _row("val", va, va["alert_budget"]),
            _row("test", te, te["alert_budget"]),
        ]
    )
    print(
        json.dumps(
            {
                "ok": True,
                "out_json": str(OUT_JSON),
                "selected_epoch": selected_epoch,
                "test_auprc": te["auprc"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
