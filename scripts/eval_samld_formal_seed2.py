#!/usr/bin/env python3
"""Post-selection eval for SAML-D formal seed-2 (best-val ckpt only; val+test once)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch_geometric.nn import to_hetero

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from data_loading import get_data  # noqa: E402
from train_util import AddEgoIds, add_arange_ids, extract_param, get_loaders  # noqa: E402
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

RUN_NAME = "samld_supervised_multigin_eu_v1_formal_seed2"
OUT_JSON = REPO / "results/diagnostics/samld_supervised_multigin_eu_formal_seed2.json"
OUT_MD = REPO / "notes/samld_supervised_multigin_eu_formal_seed2.md"
OUT_DIR = REPO / "results/diagnostics/samld_supervised_multigin_eu_formal_seed2"
REG_CSV = REPO / "results/diagnostics/thesis_experiment_registry.csv"
REG_JSON = REPO / "results/diagnostics/thesis_experiment_registry.json"
REG_MD = REPO / "notes/thesis_experiment_registry.md"

# Versioned current get_data calendar_day 60/20/20 seed cohort (formal train /
# smoke graph path). NOT the integrity-audit card (19108637), which differs.
VERSIONED_CURRENT_SEEDS = {
    "val": {
        "n": 1_900_105,
        "n_positives": 1_984,
        "index_sha256": "81269d803f1480b75dde3ab66562324fa10d5d11616fa8cca21be21755f8a97e",
    },
    "test": {
        "n": 1_889_454,
        "n_positives": 2_125,
        "index_sha256": "a9f19af47d06417035b29235f2cb84277a055f8765c6240ca0ae6cda188caf0c",
    },
}
# Integrity audit 19108637 — informational only; never gate criteria.
INTEGRITY_VAL_SEEDS_HISTORICAL = {
    "n": 1_899_523,
    "n_positives": 1_986,
    "index_sha256": "b08cdb815f82e6d37019e5e6ec9c5a6fd12c3f9d523f63b2768f6e4d0a99a38c",
    "note": "integrity_19108637; differs from current get_data; do not gate on this",
}
# NeighborLoader under preserve_seed_edges=false drops ~13% of SAML-D seeds
# (formal train scored ~87%). Fail only if coverage collapses below this floor.
MIN_SEED_EDGE_COVERAGE = 0.85
MIN_SEED_POSITIVE_COVERAGE = 0.90

SMOKE_COMP = {
    "epoch1_val_auprc": 0.983986,
    "epoch2_val_auprc": 0.958508,
    "selected_val_f1": 0.904432,
}
XONLY_HGB_VAL_AUPRC = 0.7235
PREVALENCE_VAL = 0.0010455


def sha256_int64(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(np.asarray(arr, dtype=np.int64))
    return hashlib.sha256(a.tobytes()).hexdigest()


def seed_cohort_stats(inds: torch.Tensor, y_full: torch.Tensor) -> Dict[str, Any]:
    inds_np = inds.detach().cpu().numpy().astype(np.int64)
    y_np = y_full.detach().cpu().numpy().astype(np.int64)[inds_np]
    return {
        "n": int(inds_np.shape[0]),
        "n_positives": int(y_np.sum()),
        "prevalence": float(y_np.mean()) if inds_np.size else 0.0,
        "index_sha256": sha256_int64(inds_np),
    }


def coverage_report(
    *,
    seed: Dict[str, Any],
    scored_n: int,
    scored_positives: int,
    loader_cov: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    seed_n = int(seed["n"])
    seed_pos = int(seed["n_positives"])
    missing_n = seed_n - int(scored_n)
    missing_pos = seed_pos - int(scored_positives)
    edge_cov = float(scored_n / seed_n) if seed_n else 0.0
    pos_cov = float(scored_positives / seed_pos) if seed_pos else 0.0
    return {
        "expected_seed_edges": seed_n,
        "expected_seed_positives": seed_pos,
        "scored_seed_edges": int(scored_n),
        "scored_seed_positives": int(scored_positives),
        "missing_seed_edges": int(missing_n),
        "missing_seed_positives": int(missing_pos),
        "edge_coverage": edge_cov,
        "positive_coverage": pos_cov,
        "loader_batch_coverage": loader_cov,
    }


def assert_versioned_seed_gate(
    *,
    split: str,
    observed_seed: Dict[str, Any],
    expected_seed: Dict[str, Any],
    cov: Dict[str, Any],
    min_edge_coverage: float = MIN_SEED_EDGE_COVERAGE,
    min_positive_coverage: float = MIN_SEED_POSITIVE_COVERAGE,
) -> None:
    """Fail on seed-hash/count mismatch or coverage floor breach — never on scored==integrity."""
    failures: List[str] = []
    for key in ("n", "n_positives", "index_sha256"):
        if observed_seed.get(key) != expected_seed.get(key):
            failures.append(
                f"{split} seed {key}: got={observed_seed.get(key)!r} "
                f"expected={expected_seed.get(key)!r}"
            )
    if float(cov["edge_coverage"]) < float(min_edge_coverage):
        failures.append(
            f"{split} edge_coverage={cov['edge_coverage']:.6f} "
            f"< min={min_edge_coverage} "
            f"(scored={cov['scored_seed_edges']}/{cov['expected_seed_edges']})"
        )
    if float(cov["positive_coverage"]) < float(min_positive_coverage):
        failures.append(
            f"{split} positive_coverage={cov['positive_coverage']:.6f} "
            f"< min={min_positive_coverage} "
            f"(scored_pos={cov['scored_seed_positives']}/{cov['expected_seed_positives']})"
        )
    if failures:
        raise SystemExit(
            "versioned_seed_coverage_gate_failed:\n  - " + "\n  - ".join(failures)
        )


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
    if not REG_CSV.is_file():
        raise SystemExit(f"missing registry csv: {REG_CSV}")
    with REG_CSV.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        existing = list(reader)
    for row in rows:
        # only known columns
        existing.append({k: row.get(k, "") for k in fieldnames})
    with REG_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(existing)

    if REG_JSON.is_file():
        payload = json.loads(REG_JSON.read_text())
        payload_rows = payload.get("rows") or []
        # avoid duplicate run_ids
        existing_ids = {r.get("run_id") for r in payload_rows}
        for row in rows:
            if row.get("run_id") in existing_ids:
                continue
            payload_rows.append(row)
            existing_ids.add(row.get("run_id"))
        payload["rows"] = payload_rows
        payload["row_count"] = len(payload_rows)
        REG_JSON.write_text(json.dumps(payload, indent=2) + "\n")

    # append-only markdown note
    with REG_MD.open("a", encoding="utf-8") as f:
        f.write(
            "\n### SAML-D formal Candidate-A seed-2 (appended)\n"
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
    # Build locked Candidate-A args for get_data / model (eval DOES evaluate test once)
    argv = [
        "--data", "SAML-D",
        "--model", "gin",
        "--objective", "supervised",
        "--supervised_head", "legacy",
        "--unique_name", RUN_NAME,
        "--seed", "2",
        "--batch_size", "4096",
        "--num_neighs", "100", "100",
        "--loader_num_workers", "8",
        "--reverse_mp", "--ego", "--ports", "--emlps",
        "--tqdm",
        # explicitly NOT skip_test_eval — this job evaluates val+test once
    ]
    args = parser.parse_args(argv)
    set_seed(args.seed)

    with open("data_config.json", encoding="utf-8") as f:
        data_config = json.load(f)

    ckpt_path = Path(data_config["paths"]["model_to_load"]) / RUN_NAME / "checkpoint_best_val_f1.tar"
    if not ckpt_path.is_file():
        raise SystemExit(f"missing best-val checkpoint: {ckpt_path}")

    train_final = OUT_DIR / "train_finalize.json"
    train_meta = json.loads(train_final.read_text()) if train_final.is_file() else {}
    hist_path = REPO / "results/diagnostics" / f"supervised_SAML-D_{RUN_NAME}_epoch_history.json"
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
        ("seed", 2),
        ("data", "SAML-D"),
        ("model", "gin"),
        ("supervised_head", "legacy"),
    ):
        got = ckpt_args.get(k, checkpoint.get(k))
        if got != exp:
            raise SystemExit(f"checkpoint protocol mismatch {k}: got={got!r} expected={exp!r}")

    model_state_sha = sha256_state_dict(checkpoint.get("model_state_dict") or {})
    expected_sha = ((train_meta.get("checkpoints") or {}).get("best") or {}).get("model_state_sha256")
    if expected_sha and expected_sha != model_state_sha:
        raise SystemExit(f"model state hash mismatch: {model_state_sha} != {expected_sha}")

    # Full graphs for val+test eval (test unlocked for this job only)
    tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(args, data_config)
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

    # Seed cohort = temporal split inds (versioned current get_data). Scored cohort
    # may be smaller under preserve_seed_edges=false; never equate scored to integrity.
    y_full = te_data["node", "to", "node"].y
    seed_val = seed_cohort_stats(val_inds, y_full)
    seed_te = seed_cohort_stats(te_inds, y_full)

    y_val, p_val, cov_val = collect_split_predictions(val_loader, val_inds, model, val_data, device, args)
    y_te, p_te, cov_te = collect_split_predictions(te_loader, te_inds, model, te_data, device, args)

    cov_val_report = coverage_report(
        seed=seed_val,
        scored_n=int(y_val.shape[0]),
        scored_positives=int(y_val.sum()),
        loader_cov=cov_val,
    )
    cov_te_report = coverage_report(
        seed=seed_te,
        scored_n=int(y_te.shape[0]),
        scored_positives=int(y_te.sum()),
        loader_cov=cov_te,
    )
    assert_versioned_seed_gate(
        split="val",
        observed_seed=seed_val,
        expected_seed=VERSIONED_CURRENT_SEEDS["val"],
        cov=cov_val_report,
    )
    assert_versioned_seed_gate(
        split="test",
        observed_seed=seed_te,
        expected_seed=VERSIONED_CURRENT_SEEDS["test"],
        cov=cov_te_report,
    )

    thr, val_f1_tuned = tune_threshold(y_val, p_val)
    # fixed-0.5 == paper argmax for two-class softmax
    val_metrics = split_metrics(y_val, p_val, thr)
    te_metrics = split_metrics(y_te, p_te, thr)
    # also explicit fixed-0.5 block (same as paper_argmax here)
    fixed05_val = split_metrics(y_val, p_val, 0.5)
    fixed05_te = split_metrics(y_te, p_te, 0.5)

    selected_epoch = checkpoint.get("selected_epoch")
    max_auprc_ep = train_meta.get("max_validation_auprc_epoch_diagnostic")

    payload = {
        "artifact": "samld_supervised_multigin_eu_formal_seed2",
        "protocol_id": "samld_supervised_multigin_eu_v1",
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
        "max_validation_auprc_diagnostic": train_meta.get("max_validation_auprc_diagnostic"),
        "paper_comparable_claim": (
            "Candidate-A locked protocol only (SAML-D Multi-GIN+EU legacy head). "
            "Not claimed as IBM AMLWorld table-comparable."
        ),
        "test_used_for_selection": False,
        "graph_flags": {
            "emlps": True,
            "reverse_mp": True,
            "ego": True,
            "ports": True,
            "tds": False,
            "correct_reverse_edge_features": False,
            "preserve_seed_edges": False,
            "reverse_edge_feature_semantics": "inherited_legacy",
            "normalization": "legacy_per_graph_edge_znorm",
            "edge_dim": 6,
        },
        "coverage": {
            "val": cov_val_report,
            "test": cov_te_report,
            "min_seed_edge_coverage": MIN_SEED_EDGE_COVERAGE,
            "min_seed_positive_coverage": MIN_SEED_POSITIVE_COVERAGE,
            "note": (
                "Metrics are computed on NeighborLoader-scored seeds under "
                "preserve_seed_edges=false. Seed counts/hashes are the locked "
                "temporal cohort; scored counts are not protocol truth."
            ),
        },
        "cohorts": {
            "val": {
                "seed": seed_val,
                "scored": {
                    "n": int(y_val.shape[0]),
                    "n_positives": int(y_val.sum()),
                    "prevalence": float(y_val.mean()),
                },
                # Back-compat keys = scored (selection/eval regime as executed)
                "n": int(y_val.shape[0]),
                "n_positives": int(y_val.sum()),
                "prevalence": float(y_val.mean()),
            },
            "test": {
                "seed": seed_te,
                "scored": {
                    "n": int(y_te.shape[0]),
                    "n_positives": int(y_te.sum()),
                    "prevalence": float(y_te.mean()),
                },
                "n": int(y_te.shape[0]),
                "n_positives": int(y_te.sum()),
                "prevalence": float(y_te.mean()),
            },
            "versioned_current_seeds": VERSIONED_CURRENT_SEEDS,
            "integrity_val_seeds_historical": INTEGRITY_VAL_SEEDS_HISTORICAL,
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
            "xonly_hgb_val_auprc": XONLY_HGB_VAL_AUPRC,
            "smoke_two_epoch": SMOKE_COMP,
            "val_prevalence_baseline": PREVALENCE_VAL,
            "gnn_val_auprc_minus_xonly": float(val_metrics["auprc"] - XONLY_HGB_VAL_AUPRC),
            "xonly_comparison_caveat": (
                "Graph model materially outperforms strongest audited X-only control on val AUPRC; "
                "this is not proof the entire gap is exclusively caused by message passing."
            ),
        },
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "eval_cell.json").write_text(json.dumps(payload, indent=2) + "\n")
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")

    va = payload["splits"]["val"]
    te = payload["splits"]["test"]
    md = f"""# SAML-D supervised Multi-GIN+EU formal seed-2

> Twin: `results/diagnostics/samld_supervised_multigin_eu_formal_seed2.json`  
> Protocol: `samld_supervised_multigin_eu_v1` (Candidate A)  
> Train job: `{args_ns.train_job_id}` · Eval job: `{args_ns.eval_job_id or 'n/a'}`  
> Checkpoint: `{ckpt_path}` (best-val F1; selected epoch **{selected_epoch}**)

## Protocol (locked)

- gin + legacy head + emlps + reverse_mp + ego + ports; TDS off; correct_reverse off; preserve off
- legacy per-graph z-norm; edge_dim=6; seed=2; 50 epochs
- Training used `--skip_test_eval` (test locked). This eval scores val+test **once** after selection.
- Gate: versioned **current get_data** seed counts/hashes + NeighborLoader coverage floors
  (min edge {MIN_SEED_EDGE_COVERAGE}, min positive {MIN_SEED_POSITIVE_COVERAGE}); scored ≠ integrity card.

## Primary (paper argmax)

| Split | AUROC | AUPRC | F1 | P | R | PPR |
|-------|------:|------:|---:|--:|--:|----:|
| Val | {va['auroc']:.4f} | {va['auprc']:.4f} | {va['paper_argmax']['f1']:.4f} | {va['paper_argmax']['precision']:.4f} | {va['paper_argmax']['recall']:.4f} | {va['paper_argmax']['positive_prediction_rate']:.6f} |
| Test | {te['auroc']:.4f} | {te['auprc']:.4f} | {te['paper_argmax']['f1']:.4f} | {te['paper_argmax']['precision']:.4f} | {te['paper_argmax']['recall']:.4f} | {te['paper_argmax']['positive_prediction_rate']:.6f} |

Confusion (argmax): val TP/FP/TN/FN = {va['paper_argmax']['tp']:.0f}/{va['paper_argmax']['fp']:.0f}/{va['paper_argmax']['tn']:.0f}/{va['paper_argmax']['fn']:.0f};  
test TP/FP/TN/FN = {te['paper_argmax']['tp']:.0f}/{te['paper_argmax']['fp']:.0f}/{te['paper_argmax']['tn']:.0f}/{te['paper_argmax']['fn']:.0f}.

## Alert budgets (P@K)

Val P@100/500/1000: see JSON `splits.val.alert_budget`.  
Test P@100/500/1000: see JSON `splits.test.alert_budget`.

## Diagnostics (not selection)

- Max-validation-AUPRC epoch (diagnostic): **{max_auprc_ep}**
- Val-tuned threshold F1 (NOT paper-compatible): thr={thr:.4f}
- Fixed-0.5 equals paper argmax for two-class softmax

## Comparisons (validation)

| Reference | Val AUPRC / F1 |
|-----------|----------------|
| Prevalence | {PREVALENCE_VAL:.6f} |
| X-only HGB | {XONLY_HGB_VAL_AUPRC:.4f} |
| Smoke ep1/ep2 AUPRC | {SMOKE_COMP['epoch1_val_auprc']:.4f} / {SMOKE_COMP['epoch2_val_auprc']:.4f} |
| Smoke selected F1 | {SMOKE_COMP['selected_val_f1']:.4f} |
| This formal (argmax) | {va['auprc']:.4f} / F1 {va['paper_argmax']['f1']:.4f} |

X-only gap is evidence the graph model outperforms the strongest audited feature-only control, not proof that the entire difference is exclusively message passing.

## Paper-comparability

Candidate-A / SAML-D locked protocol only. Do **not** claim IBM AMLWorld table parity beyond that definition.
"""
    OUT_MD.write_text(md)

    # Registry rows: val + test primary argmax
    def _row(split: str, block: Dict[str, Any], ab: Dict[str, Any]) -> Dict[str, Any]:
        pa = block["paper_argmax"]
        return {
            "run_id": f"{RUN_NAME}|paper_argmax|{split}",
            "dataset": "SAML-D",
            "dataset_positive_rate": payload["cohorts"][split]["prevalence"],
            "objective": "supervised",
            "encoder": "gin",
            "seed": 2,
            "training_epochs": 50,
            "selected_epoch": selected_epoch,
            "checkpoint_policy": "best_val_f1",
            "supervised_head": "legacy",
            "graph_flags": "reverse_mp,ego,ports,emlps",
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
            "precision_at_100": ab.get("precision_at_100") or ab.get("p_at_100"),
            "recall_at_100": ab.get("recall_at_100") or ab.get("r_at_100"),
            "precision_at_500": ab.get("precision_at_500") or ab.get("p_at_500"),
            "recall_at_500": ab.get("recall_at_500") or ab.get("r_at_500"),
            "precision_at_1000": ab.get("precision_at_1000") or ab.get("p_at_1000"),
            "recall_at_1000": ab.get("recall_at_1000") or ab.get("r_at_1000"),
            "source_json": str(OUT_JSON.relative_to(REPO)),
            "source_note": str(OUT_MD.relative_to(REPO)),
            "checkpoint_path": str(ckpt_path),
            "status": "evaluated",
            "scout_or_formal": "formal",
            "superseded": False,
            "thesis_role": "thesis_supporting",
            "validation_status": "validated",
            "table_eligible": True,
            "table_group": "samld_supervised_candidate_a",
            "duplicate_resolution": "not_duplicate",
            "caveats": "SAML-D Candidate-A formal seed-2; paper_argmax; not IBM AMLWorld table-comparable",
            "protocol_family": "samld_supervised_multigin_eu_v1",
            "split_protocol": "calendar_day_60_20_20",
            "reverse_feature_semantics": "inherited_legacy",
            "preserve_seed_edges": False,
            "job_id": args_ns.eval_job_id or args_ns.train_job_id,
            "paper_comparable": False,
        }

    append_registry_rows([
        _row("val", va, va["alert_budget"]),
        _row("test", te, te["alert_budget"]),
    ])
    print(json.dumps({"ok": True, "out_json": str(OUT_JSON), "selected_epoch": selected_epoch}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
