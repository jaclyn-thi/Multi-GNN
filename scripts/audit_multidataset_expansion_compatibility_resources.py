#!/usr/bin/env python3
"""Read-only multi-dataset expansion compatibility/resource audit.

Assembles inventory, shared-core mappings, TF/split/leakage matrices,
N-domain trainer plan, and resident-combination estimates from existing
metadata and headers only. No training, extraction, probes, GPU jobs,
dataset rewrites, or unbounded CSV scans.
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "diagnostics" / "multidataset_expansion_compatibility_resource_audit"
TWIN = ROOT / "results" / "diagnostics" / "multidataset_expansion_compatibility_resource_audit.json"
NOTES = ROOT / "notes" / "multidataset_expansion_compatibility_resource_audit.md"

TF_TARGETS = (
    "log1p_sender_interarrival",
    "log1p_sender_past_7d_count",
    "log1p_amount_vs_sender_past_mean",
)
CORE6 = (
    "Timestamp",
    "Amount Received",
    "in_port",
    "out_port",
    "in_td",
    "out_td",
)

# Measured Phase-3 step costs (optimizer wall / 1000), integrity summary.
HI_S_PER_STEP = 1886.30 / 1000.0
SD_S_PER_STEP = 2835.61 / 1000.0
# Small-LI / PaySim extrapolated by train-edge ratio vs nearest measured domain.
HI_TRAIN = 3_248_921
SD_TRAIN = 5_715_293
LI_TRAIN = 4_432_934
PS_TRAIN = 3_792_821
LI_EDGES = 6_924_049
PS_EDGES = 6_362_620
HI_EDGES = 5_078_345
SD_EDGES_FULL = 9_504_852
SD_EDGES_TV = 7_615_398
# Prior-audit Medium edge counts (not re-verified by full scan this audit).
MED_HI_EDGES = 31_898_238
MED_LI_EDGES = 31_251_483

# TF cache clean bytes from meta+du notes.
TF_HI_CLEAN_B = 182_825_484
TF_SD_B = 274_166_266
TF_LI_DIR_B = 476 * 1024**2  # dir du; includes possible temps
TF_PS_B = 229_069_202

# Embedding float32 bytes ≈ n * 198 * 4; Phase-3 measured total ~24.6 GiB for 3×2 cells.
R198 = 198
EMB_BYTES_PER_EDGE = R198 * 4


def sha256_file(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def est_s_per_step(train_edges: int, ref_train: int, ref_s: float) -> float:
    return ref_s * (train_edges / ref_train)


def emb_giB(n_edges: int) -> float:
    return (n_edges * EMB_BYTES_PER_EDGE) / (1024**3)


def tf_cache_bytes_est(n_rows: int) -> float:
    """Linear extrapolation from Small-HI clean TF cache bytes/row."""
    return TF_HI_CLEAN_B * (n_rows / HI_EDGES)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    aml = ROOT / "aml-data"
    raw = ROOT / "raw-aml-data"
    sources = {
        "phase3_scout_note": "notes/smallhi_samld_mixed_ssl_phase3_scout.md",
        "phase3_integrity": "results/diagnostics/smallhi_samld_mixed_ssl_phase3_scout/training_integrity_summary.json",
        "phase3_frozen_note": "notes/smallhi_samld_mixed_ssl_phase3_frozen_eval.md",
        "phase3_frozen_json": "results/diagnostics/smallhi_samld_mixed_ssl_phase3_frozen_eval.json",
        "phase1_note": "notes/small_hi_samld_shared_core_phase1.md",
        "samld_split_note": "notes/samld_shared_core_split_reconciliation.md",
        "prior_compat_audit": "notes/multidataset_ssl_compatibility_audit.md",
        "shared_core_contract": "shared_core_contract.py",
        "trainer": "scripts/train_mixed_ssl_phase3_scout.py",
        "mixed_ssl_phase3": "mixed_ssl_phase3/__init__.py",
    }
    source_sha = {k: sha256_file(ROOT / v) for k, v in sources.items()}

    # ------------------------------------------------------------------
    # Dataset inventory
    # ------------------------------------------------------------------
    inventory = [
        {
            "dataset": "Small-HI",
            "family": "AMLWorld_IBM_simulator_variant",
            "independent_generator": False,
            "role": "reference",
            "raw_path": "raw-aml-data/HI-Small_Trans.csv",
            "formatted_path": "aml-data/Small-HI/formatted_transactions.csv",
            "symlink_root": "aml-data -> /orcd/pool/007/jthi/Multi-GNN/aml-data",
            "raw_size": "193M (du) / 454M (ls)",
            "formatted_size": "143M (dir du) / 250M (csv ls)",
            "compression": "none (csv)",
            "formatter": "format_kaggle_files.py",
            "dataset_spec": "AMLWorld default (dataset_specs)",
            "loader_entrypoint": "data_loading.get_data",
            "graph_construction": "data_loading.get_data -> data_util.GraphData + ports/tds",
            "cache_paths": "results/cache/temporal_flow_causal/Small-HI (349M dir; ~183M clean)",
            "notes_diagnostics": "phase1/2/3 mixed SSL; morphology_cache/Small-HI",
            "approx_edges": HI_EDGES,
            "edge_count_source": "measured_tf_meta",
            "EdgeID_exists": True,
            "EdgeID_assignment": "sequential at format; contiguous row index after sort (HI path)",
            "EdgeID_equals_csv_row_index": True,
            "sort_after_EdgeID": "sort by Timestamp; HI/LI treat EdgeID as row-aligned in TF preflight",
            "sender_col": "from_id (raw: From Bank+Account)",
            "receiver_col": "to_id",
            "timestamp_col_units": "Timestamp; seconds; loader rezeros by min",
            "amount_col_units": "Amount Received (primary); Amount Sent also present",
            "label_col_semantics": "Is Laundering — IBM AMLWorld edge laundering label",
            "categorical_cols": "Sent/Received Currency, Payment Format",
            "balance_cols": "none",
            "leakage_prone_fields": "label; pattern metadata files (not in encoder path)",
            "missing_value_behavior": "formatter/loader assume complete required cols",
            "currency_unit_concerns": "multi-currency codes local to AMLWorld vocab",
            "graph_directed": True,
            "parallel_tx_preserved": True,
            "shared_core_class": "DIRECT_COMPATIBLE",
            "leakage_risk": "LOW",
        },
        {
            "dataset": "SAML-D",
            "family": "independent_SAML_D_generator",
            "independent_generator": True,
            "role": "reference",
            "raw_path": "raw-aml-data/SAML-D.csv",
            "formatted_path": "aml-data/SAML-D/formatted_transactions.csv",
            "symlink_root": "aml-data -> /orcd/pool/007/jthi/Multi-GNN/aml-data",
            "raw_size": "433M (du) / 951M (ls)",
            "formatted_size": "265M (dir) / 483M (csv)",
            "compression": "none (csv)",
            "formatter": "format_saml_d_files.py",
            "dataset_spec": "SAML-D / samld_calendar_day_rezero_v1",
            "loader_entrypoint": "data_loading.get_data",
            "graph_construction": "same as AMLWorld path after format",
            "cache_paths": "results/cache/temporal_flow_causal_samld_shared_core_v1/SAML-D (262M; train∪val only)",
            "notes_diagnostics": "phase1 split reconciliation; phase2/3 scout+frozen eval",
            "approx_edges": SD_EDGES_FULL,
            "edge_count_source": "measured_tf_meta_full_csv",
            "EdgeID_exists": True,
            "EdgeID_assignment": "assigned before sort; unique IDs retained",
            "EdgeID_equals_csv_row_index": False,
            "sort_after_EdgeID": "yes — EdgeID ≠ post-sort row index; TF joins by EdgeID",
            "sender_col": "from_id (raw Sender_account)",
            "receiver_col": "to_id (raw Receiver_account)",
            "timestamp_col_units": "Date+Time → Timestamp seconds; rezero protocol locked",
            "amount_col_units": "Amount → Amount Sent/Received",
            "label_col_semantics": "Is_laundering → Is Laundering; Laundering_type excluded from encoder",
            "categorical_cols": "Payment_currency/Received_currency/Payment_type → local int codes",
            "balance_cols": "none",
            "leakage_prone_fields": "Laundering_type (post-hoc typology); label",
            "missing_value_behavior": "formatter maps required fields",
            "currency_unit_concerns": "SAML-local codes ≠ AMLWorld vocab (excluded from shared-core)",
            "graph_directed": True,
            "parallel_tx_preserved": True,
            "shared_core_class": "DIRECT_COMPATIBLE",
            "leakage_risk": "LOW",
        },
        {
            "dataset": "Small-LI",
            "family": "AMLWorld_IBM_simulator_variant",
            "independent_generator": False,
            "role": "candidate",
            "raw_path": "raw-aml-data/LI-Small_Trans.csv",
            "formatted_path": "aml-data/Small-LI/formatted_transactions.csv",
            "symlink_root": "aml-data -> /orcd/pool/007/jthi/Multi-GNN/aml-data",
            "raw_size": "265M (du) / 621M (ls)",
            "formatted_size": "194M (dir) / 342M (csv)",
            "compression": "none",
            "formatter": "format_kaggle_files.py",
            "dataset_spec": "AMLWorld default calendar_day",
            "loader_entrypoint": "data_loading.get_data",
            "graph_construction": "same ports/TDS path",
            "cache_paths": "results/cache/temporal_flow_causal/Small-LI (476M dir)",
            "notes_diagnostics": "notes/*small_li*; results/diagnostics/small_li_*",
            "approx_edges": LI_EDGES,
            "edge_count_source": "measured_tf_meta",
            "EdgeID_exists": True,
            "EdgeID_assignment": "same as Small-HI formatter family",
            "EdgeID_equals_csv_row_index": True,
            "sort_after_EdgeID": "Timestamp sort (AMLWorld formatter family)",
            "sender_col": "from_id",
            "receiver_col": "to_id",
            "timestamp_col_units": "Timestamp seconds; loader rezero",
            "amount_col_units": "Amount Received",
            "label_col_semantics": "Is Laundering — same AMLWorld semantics (intensity LI)",
            "categorical_cols": "Currency, Payment Format (AMLWorld vocab family)",
            "balance_cols": "none",
            "leakage_prone_fields": "label",
            "missing_value_behavior": "same as Small-HI",
            "currency_unit_concerns": "same AMLWorld family as Small-HI",
            "graph_directed": True,
            "parallel_tx_preserved": True,
            "shared_core_class": "DIRECT_COMPATIBLE",
            "leakage_risk": "LOW",
            "implementation_note": "Need shared_core allowlist + N-domain registry only; loader/TF already exist",
        },
        {
            "dataset": "Medium-HI",
            "family": "AMLWorld_IBM_simulator_variant",
            "independent_generator": False,
            "role": "candidate",
            "raw_path": "NOT_ON_DISK under raw-aml-data/",
            "formatted_path": "aml-data/Medium-HI/formatted_transactions.csv",
            "symlink_root": "aml-data -> /orcd/pool/007/jthi/Multi-GNN/aml-data",
            "raw_size": "absent",
            "formatted_size": "899M (dir) / 1.7G (csv)",
            "compression": "none",
            "formatter": "format_kaggle_files.py (historical; raw absent now)",
            "dataset_spec": "AMLWorld default (assumed calendar_day; no dedicated audit)",
            "loader_entrypoint": "data_loading.get_data",
            "graph_construction": "same; scale risk for ports/TDS RAM",
            "cache_paths": "NO TF cache; not in SUPPORTED_DATASETS",
            "notes_diagnostics": "prior multidataset_ssl_compatibility_audit",
            "approx_edges": MED_HI_EDGES,
            "edge_count_source": "prior_audit_metadata_not_reverified",
            "EdgeID_exists": True,
            "EdgeID_assignment": "standard formatted schema (header confirmed)",
            "EdgeID_equals_csv_row_index": "assumed_AMLWorld_family_unverified",
            "sort_after_EdgeID": "assumed formatter sort; verify in optional CPU job",
            "sender_col": "from_id",
            "receiver_col": "to_id",
            "timestamp_col_units": "Timestamp (units assumed seconds)",
            "amount_col_units": "Amount Received",
            "label_col_semantics": "Is Laundering — AMLWorld HI intensity",
            "categorical_cols": "Currency, Payment Format",
            "balance_cols": "none",
            "leakage_prone_fields": "label",
            "missing_value_behavior": "unknown at scale without scan",
            "currency_unit_concerns": "AMLWorld family",
            "graph_directed": True,
            "parallel_tx_preserved": True,
            "shared_core_class": "DIRECT_COMPATIBLE",
            "leakage_risk": "MANAGEABLE",
            "implementation_note": "TF cache builder allowlist + large RAM graph build; not an independent source vs Small-HI",
        },
        {
            "dataset": "Medium-LI",
            "family": "AMLWorld_IBM_simulator_variant",
            "independent_generator": False,
            "role": "candidate",
            "raw_path": "NOT_ON_DISK under raw-aml-data/",
            "formatted_path": "aml-data/Medium-LI/formatted_transactions.csv",
            "symlink_root": "aml-data -> /orcd/pool/007/jthi/Multi-GNN/aml-data",
            "raw_size": "absent",
            "formatted_size": "880M (dir) / 1.6G (csv)",
            "compression": "none",
            "formatter": "format_kaggle_files.py (historical)",
            "dataset_spec": "AMLWorld default (assumed)",
            "loader_entrypoint": "data_loading.get_data",
            "graph_construction": "same; scale risk",
            "cache_paths": "NO TF cache",
            "notes_diagnostics": "prior multidataset audit",
            "approx_edges": MED_LI_EDGES,
            "edge_count_source": "prior_audit_metadata_not_reverified",
            "EdgeID_exists": True,
            "EdgeID_assignment": "standard formatted schema",
            "EdgeID_equals_csv_row_index": "assumed_unverified",
            "sort_after_EdgeID": "assumed",
            "sender_col": "from_id",
            "receiver_col": "to_id",
            "timestamp_col_units": "Timestamp",
            "amount_col_units": "Amount Received",
            "label_col_semantics": "Is Laundering — AMLWorld LI intensity",
            "categorical_cols": "Currency, Payment Format",
            "balance_cols": "none",
            "leakage_prone_fields": "label",
            "missing_value_behavior": "unknown without scan",
            "currency_unit_concerns": "AMLWorld family",
            "graph_directed": True,
            "parallel_tx_preserved": True,
            "shared_core_class": "DIRECT_COMPATIBLE",
            "leakage_risk": "MANAGEABLE",
        },
        {
            "dataset": "PaySim",
            "family": "independent_PaySim_generator",
            "independent_generator": True,
            "role": "candidate",
            "raw_path": "aml-data/PaySim/PS_20174392719_1491204439457_log.csv",
            "formatted_path": "aml-data/PaySim/formatted_transactions.csv",
            "symlink_root": "aml-data -> /orcd/pool/007/jthi/Multi-GNN/aml-data",
            "raw_size": "471M",
            "formatted_size": "526M (dir) / 348M (csv)",
            "compression": "none",
            "formatter": "format_paysim.py",
            "dataset_spec": "PAYSIM_SPEC hourly_step (dataset_specs.py)",
            "loader_entrypoint": "data_loading.get_data via get_dataset_spec(PaySim)",
            "graph_construction": "same ports/TDS; Timestamp=step*3600",
            "cache_paths": "temporal_flow_cache/PaySim (220M; temporal_flow_causal_paysim_v1)",
            "notes_diagnostics": "notes/paysim_*; expert_only frozen transfer preflight",
            "approx_edges": PS_EDGES,
            "edge_count_source": "measured_tf_meta",
            "EdgeID_exists": True,
            "EdgeID_assignment": "reassigned after Timestamp sort → contiguous row index",
            "EdgeID_equals_csv_row_index": True,
            "sort_after_EdgeID": "sort then reassign EdgeID",
            "sender_col": "from_id (nameOrig)",
            "receiver_col": "to_id (nameDest)",
            "timestamp_col_units": "step*3600 synthetic seconds; coarse ties (all edges share steps)",
            "amount_col_units": "amount → Amount Sent/Received",
            "label_col_semantics": "isFraud → Is Laundering (fraud ≠ AML laundering)",
            "categorical_cols": "type → duplicated into currency+payment slots in formatted CSV",
            "balance_cols": "oldbalance*/newbalance* EXCLUDED from formatted (leakage)",
            "leakage_prone_fields": "balances; isFlaggedFraud; isFraud label; cancelled-fraud balance caveat",
            "missing_value_behavior": "formatter requires step/type/amount/nameOrig/nameDest/isFraud",
            "currency_unit_concerns": "no real currency; type-dup is legacy shim — excluded from shared-core",
            "graph_directed": True,
            "parallel_tx_preserved": True,
            "shared_core_class": "DIRECT_COMPATIBLE",
            "leakage_risk": "MANAGEABLE",
            "implementation_note": "Leakage-safe shared-core uses Timestamp+Amount+ports+TDS only; never balances/flags/labels in encoder",
        },
        {
            "dataset": "AMLSim",
            "family": "independent_AMLSim_generator",
            "independent_generator": True,
            "role": "candidate_archives_only",
            "raw_path": "aml-data/aml-sim/{100vertices-10Kedges.7z, 10Kvertices-1Medges.7z}",
            "formatted_path": "ABSENT",
            "symlink_root": "aml-data -> /orcd/pool/007/jthi/Multi-GNN/aml-data",
            "raw_size": "5.0M total (150K + 4.8M 7z)",
            "formatted_size": "n/a",
            "compression": "7z",
            "formatter": "ABSENT — implementation work",
            "dataset_spec": "ABSENT",
            "loader_entrypoint": "ABSENT",
            "graph_construction": "ABSENT",
            "cache_paths": "ABSENT",
            "notes_diagnostics": "prior audit CURRENTLY_BLOCKED; this audit: DATA_MISSING extracted + FORMATTER/LOADER plan",
            "approx_edges": "unknown (filenames imply ~10K and ~1M)",
            "edge_count_source": "filename_hint_only",
            "EdgeID_exists": False,
            "EdgeID_assignment": "to be defined in formatter",
            "EdgeID_equals_csv_row_index": "n/a",
            "sort_after_EdgeID": "n/a",
            "sender_col": "unknown until extract (typical AMLSim: tx sender account)",
            "receiver_col": "unknown until extract",
            "timestamp_col_units": "unknown until extract",
            "amount_col_units": "unknown until extract",
            "label_col_semantics": "likely alert/SAR membership and/or account-level — UNRESOLVED until extract",
            "categorical_cols": "unknown",
            "balance_cols": "possible in AMLSim account files",
            "leakage_prone_fields": "alert/SAR membership can leak labels if joined into features",
            "missing_value_behavior": "unknown",
            "currency_unit_concerns": "unknown",
            "graph_directed": "expected yes",
            "parallel_tx_preserved": "unknown",
            "shared_core_class": "DATA_MISSING",
            "leakage_risk": "UNRESOLVED",
            "implementation_note": "Example-scale archives; not a full production AMLSim dump. Formatter+loader required — not an automatic scientific blocker.",
        },
        {
            "dataset": "AMLNet",
            "family": "independent_AMLNet",
            "independent_generator": True,
            "role": "other_on_disk_not_in_scope",
            "raw_path": "aml-data/aml-net/AMLNet_August 2025.csv",
            "formatted_path": "ABSENT",
            "symlink_root": "aml-data",
            "raw_size": "181M (dir) / 660M (csv listed previously)",
            "formatted_size": "n/a",
            "compression": "none",
            "formatter": "ABSENT",
            "dataset_spec": "ABSENT",
            "loader_entrypoint": "ABSENT",
            "graph_construction": "ABSENT",
            "cache_paths": "ABSENT",
            "notes_diagnostics": "README v2.0 present",
            "approx_edges": "~1.09M claimed by README (unverified)",
            "edge_count_source": "readme_claim",
            "EdgeID_exists": False,
            "header_sample": "step,type,amount,category,nameOrig,nameDest,...,isFraud,isMoneyLaundering,...",
            "shared_core_class": "FORMATTER_ONLY",
            "leakage_risk": "UNRESOLVED",
            "note": "Immediately plausible candidate; out of implementation scope this audit",
        },
        {
            "dataset": "BankSim",
            "family": "independent_BankSim",
            "independent_generator": True,
            "role": "other_on_disk_not_in_scope",
            "raw_path": "aml-data/banksim/archive (1).zip",
            "formatted_path": "ABSENT",
            "raw_size": "14M zip",
            "formatted_size": "n/a",
            "compression": "zip",
            "formatter": "ABSENT",
            "header_sample": "step,customer,age,gender,zipcodeOri,merchant,zipMerchant,category,amount,fraud",
            "shared_core_class": "FORMATTER_ONLY",
            "leakage_risk": "UNRESOLVED",
            "note": "Merchant-spend graph; different task semantics; out of scope",
        },
    ]
    write_csv(OUT / "dataset_inventory.csv", inventory)

    # ------------------------------------------------------------------
    # Feature mapping to shared core
    # ------------------------------------------------------------------
    feature_rows = []
    for ds, mapping in [
        (
            "Small-HI",
            {
                "Timestamp": ("native", "Timestamp"),
                "Amount Received": ("native", "Amount Received"),
                "in_port": ("deterministic_derived", "ports() on edge_index"),
                "out_port": ("deterministic_derived", "ports()"),
                "in_td": ("deterministic_derived", "time_deltas()"),
                "out_td": ("deterministic_derived", "time_deltas()"),
                "class": "DIRECT_COMPATIBLE",
                "ports_tds_same_semantics": True,
            },
        ),
        (
            "SAML-D",
            {
                "Timestamp": ("native", "formatted Timestamp from Date+Time"),
                "Amount Received": ("native", "Amount"),
                "in_port": ("deterministic_derived", "ports()"),
                "out_port": ("deterministic_derived", "ports()"),
                "in_td": ("deterministic_derived", "time_deltas()"),
                "out_td": ("deterministic_derived", "time_deltas()"),
                "class": "DIRECT_COMPATIBLE",
                "ports_tds_same_semantics": True,
            },
        ),
        (
            "Small-LI",
            {
                "Timestamp": ("native", "Timestamp"),
                "Amount Received": ("native", "Amount Received"),
                "in_port": ("deterministic_derived", "ports()"),
                "out_port": ("deterministic_derived", "ports()"),
                "in_td": ("deterministic_derived", "time_deltas()"),
                "out_td": ("deterministic_derived", "time_deltas()"),
                "class": "DIRECT_COMPATIBLE",
                "ports_tds_same_semantics": True,
                "code_gate": "SHARED_CORE_DATASETS allowlist currently excludes Small-LI",
            },
        ),
        (
            "Medium-HI",
            {
                "Timestamp": ("native", "Timestamp"),
                "Amount Received": ("native", "Amount Received"),
                "in_port": ("deterministic_derived", "ports()"),
                "out_port": ("deterministic_derived", "ports()"),
                "in_td": ("deterministic_derived", "time_deltas()"),
                "out_td": ("deterministic_derived", "time_deltas()"),
                "class": "DIRECT_COMPATIBLE",
                "ports_tds_same_semantics": True,
                "ops": "TF cache + scale",
            },
        ),
        (
            "Medium-LI",
            {
                "Timestamp": ("native", "Timestamp"),
                "Amount Received": ("native", "Amount Received"),
                "in_port": ("deterministic_derived", "ports()"),
                "out_port": ("deterministic_derived", "ports()"),
                "in_td": ("deterministic_derived", "time_deltas()"),
                "out_td": ("deterministic_derived", "time_deltas()"),
                "class": "DIRECT_COMPATIBLE",
                "ports_tds_same_semantics": True,
            },
        ),
        (
            "PaySim",
            {
                "Timestamp": ("deterministic_derived", "step * 3600"),
                "Amount Received": ("native", "amount"),
                "in_port": ("deterministic_derived", "ports()"),
                "out_port": ("deterministic_derived", "ports()"),
                "in_td": ("deterministic_derived", "time_deltas()"),
                "out_td": ("deterministic_derived", "time_deltas()"),
                "class": "DIRECT_COMPATIBLE",
                "ports_tds_same_semantics": True,
                "caveat": "Coarse hourly ties; categoricals excluded from core; balances never used",
            },
        ),
        (
            "AMLSim",
            {
                "Timestamp": ("unavailable", "needs extract+formatter"),
                "Amount Received": ("unavailable", "needs extract+formatter"),
                "in_port": ("implementable_new_loader", "after graph build"),
                "out_port": ("implementable_new_loader", "after graph build"),
                "in_td": ("implementable_new_loader", "after graph build"),
                "out_td": ("implementable_new_loader", "after graph build"),
                "class": "DATA_MISSING",
                "ports_tds_same_semantics": "after_formatter",
            },
        ),
    ]:
        for feat in CORE6:
            kind, detail = mapping[feat]
            feature_rows.append(
                {
                    "dataset": ds,
                    "shared_core_feature": feat,
                    "classification": kind,
                    "source_detail": detail,
                    "dataset_shared_core_class": mapping["class"],
                    "ports_tds_same_semantics": mapping.get("ports_tds_same_semantics"),
                    "notes": mapping.get("caveat")
                    or mapping.get("code_gate")
                    or mapping.get("ops")
                    or "",
                }
            )
    write_csv(OUT / "feature_mapping.csv", feature_rows)

    # ------------------------------------------------------------------
    # Native feature inventory + adapter sketches
    # ------------------------------------------------------------------
    native = [
        {
            "dataset": "Small-HI",
            "safe_numerical": "Timestamp, Amount Sent, Amount Received",
            "safe_categorical": "Received Currency, Payment Format, Sent Currency (AMLWorld vocab)",
            "questionable_post_event": "none beyond label",
            "leakage_fields": "Is Laundering; pattern attempt metadata if joined",
            "unavailable": "balances",
            "adapter_inputs": "optional: currency+payment embeddings",
            "categorical_vocab": "fit train-only AMLWorld codes",
            "numerical_norm": "train-fit z-norm per domain",
            "missing_masks": "not required for core columns",
            "proposed_adapter_out_dim": 16,
            "dataset_id_via_adapter": True,
            "shortcut_risk": "currency/payment can encode dataset identity vs SAML",
        },
        {
            "dataset": "SAML-D",
            "safe_numerical": "Timestamp, Amount",
            "safe_categorical": "Payment_currency, Received_currency, Payment_type, bank locations (local)",
            "questionable_post_event": "Laundering_type",
            "leakage_fields": "Is_laundering, Laundering_type",
            "unavailable": "balances",
            "adapter_inputs": "optional local categoricals",
            "categorical_vocab": "train-only SAML codes — not aligned to AMLWorld",
            "numerical_norm": "train-fit z-norm",
            "missing_masks": "optional for sparse categoricals",
            "proposed_adapter_out_dim": 16,
            "dataset_id_via_adapter": True,
            "shortcut_risk": "HIGH if categoricals shared without domain BN / adapter isolation",
        },
        {
            "dataset": "Small-LI",
            "safe_numerical": "same as Small-HI",
            "safe_categorical": "same AMLWorld family",
            "questionable_post_event": "none",
            "leakage_fields": "Is Laundering",
            "unavailable": "balances",
            "adapter_inputs": "optional shared AMLWorld categorical adapter with Small-HI",
            "categorical_vocab": "can share HI vocab family if codes match",
            "numerical_norm": "train-fit z-norm",
            "missing_masks": "n/a",
            "proposed_adapter_out_dim": 16,
            "dataset_id_via_adapter": "weak (same family)",
            "shortcut_risk": "LOW vs Small-HI; intensity shift may still shortcut",
        },
        {
            "dataset": "Medium-HI",
            "safe_numerical": "same AMLWorld",
            "safe_categorical": "same AMLWorld",
            "questionable_post_event": "none",
            "leakage_fields": "Is Laundering",
            "unavailable": "raw CSV on disk",
            "adapter_inputs": "same as Small-HI",
            "categorical_vocab": "AMLWorld",
            "numerical_norm": "train-fit",
            "missing_masks": "n/a",
            "proposed_adapter_out_dim": 16,
            "dataset_id_via_adapter": "weak",
            "shortcut_risk": "scale/intensity vs diversity confusion if treated as independent source",
        },
        {
            "dataset": "Medium-LI",
            "safe_numerical": "same AMLWorld",
            "safe_categorical": "same AMLWorld",
            "questionable_post_event": "none",
            "leakage_fields": "Is Laundering",
            "unavailable": "raw CSV on disk",
            "adapter_inputs": "same",
            "categorical_vocab": "AMLWorld",
            "numerical_norm": "train-fit",
            "missing_masks": "n/a",
            "proposed_adapter_out_dim": 16,
            "dataset_id_via_adapter": "weak",
            "shortcut_risk": "same as Medium-HI",
        },
        {
            "dataset": "PaySim",
            "safe_numerical": "step/Timestamp, amount",
            "safe_categorical": "type (as categorical only — do not duplicate into currency+payment)",
            "questionable_post_event": "none if balances excluded",
            "leakage_fields": "oldbalanceOrg,newbalanceOrig,oldbalanceDest,newbalanceDest,isFlaggedFraud,isFraud",
            "unavailable": "real currency/payment",
            "adapter_inputs": "optional type embedding → residual; NEVER balances/flags",
            "categorical_vocab": "train-only type codes (CASH_IN/OUT/…)",
            "numerical_norm": "train-fit on amount+time",
            "missing_masks": "n/a for core",
            "proposed_adapter_out_dim": 8,
            "dataset_id_via_adapter": True,
            "shortcut_risk": "MEDIUM — type alone can identify PaySim; balances would be severe leakage",
            "source_warning": "Cancelled fraud txs make balance columns inappropriate for fraud detection",
        },
        {
            "dataset": "AMLSim",
            "safe_numerical": "TBD after extract (tx amount, timestamp)",
            "safe_categorical": "TBD (tx type, etc.)",
            "questionable_post_event": "alert/SAR join fields",
            "leakage_fields": "SAR/alert membership if used as features",
            "unavailable": "extracted CSVs",
            "adapter_inputs": "TBD",
            "categorical_vocab": "TBD",
            "numerical_norm": "train-fit after format",
            "missing_masks": "likely needed",
            "proposed_adapter_out_dim": 16,
            "dataset_id_via_adapter": True,
            "shortcut_risk": "HIGH until label join policy fixed",
        },
    ]
    write_csv(OUT / "native_feature_inventory.csv", native)

    # ------------------------------------------------------------------
    # Split / leakage
    # ------------------------------------------------------------------
    splits = [
        {
            "dataset": "Small-HI",
            "existing_split": "calendar_day 60/20/20",
            "split_type": "temporal",
            "timestamp_convention": "seconds; Timestamp -= min in loader",
            "rezeroed": True,
            "preprocess_fit": "train_fit_edge_znorm after split",
            "train_val_test": "3248921 / 965524 / 863900 (TF meta)",
            "val_test_influence_vocab": "no if train-only factorize/scaler",
            "mp_cross_split": "train graph train-only; val uses train+val edges typical Multi-GNN",
            "test_files_exist": True,
            "ssl_exclude": "test eval; labels from SSL loss",
            "test_labels_inspected_this_audit": False,
            "leakage_risk": "LOW",
            "leakage_reason": "temporal split + train-fit norms + labels excluded from shared-core",
        },
        {
            "dataset": "SAML-D",
            "existing_split": "samld_calendar_day_rezero_v1",
            "split_type": "temporal",
            "timestamp_convention": "rezeroed seconds; protocol locked",
            "rezeroed": True,
            "preprocess_fit": "train-fit scalers locked in phase1",
            "train_val_test": "5715293 / 1900105 / 1889454",
            "val_test_influence_vocab": "no under locked protocol",
            "mp_cross_split": "same Multi-GNN convention",
            "test_files_exist": True,
            "ssl_exclude": "test caches/metrics; Laundering_type",
            "test_labels_inspected_this_audit": False,
            "leakage_risk": "LOW",
            "leakage_reason": "locked rezero split; TF train∪val only; labels out of encoder",
        },
        {
            "dataset": "Small-LI",
            "existing_split": "calendar_day (TF meta)",
            "split_type": "temporal",
            "timestamp_convention": "seconds; loader rezero",
            "rezeroed": True,
            "preprocess_fit": "train_fit_edge_znorm",
            "train_val_test": "4432934 / 1316442 / 1174673",
            "val_test_influence_vocab": "no if train-only",
            "mp_cross_split": "same",
            "test_files_exist": True,
            "ssl_exclude": "test eval during SSL",
            "test_labels_inspected_this_audit": False,
            "leakage_risk": "LOW",
            "leakage_reason": "same protocol family as Small-HI",
        },
        {
            "dataset": "Medium-HI",
            "existing_split": "assumed calendar_day; no dedicated audit",
            "split_type": "temporal_assumed",
            "timestamp_convention": "assumed seconds",
            "rezeroed": "loader will rezero",
            "preprocess_fit": "must be train-fit after split",
            "train_val_test": "unknown exact — optional CPU count job",
            "val_test_influence_vocab": "must enforce train-only",
            "mp_cross_split": "same pattern",
            "test_files_exist": "formatted CSV includes all edges; split at load",
            "ssl_exclude": "test eval",
            "test_labels_inspected_this_audit": False,
            "leakage_risk": "MANAGEABLE",
            "leakage_reason": "schema OK; split boundaries not re-audited at Medium scale",
        },
        {
            "dataset": "Medium-LI",
            "existing_split": "assumed calendar_day",
            "split_type": "temporal_assumed",
            "timestamp_convention": "assumed seconds",
            "rezeroed": "loader will rezero",
            "preprocess_fit": "train-fit after split",
            "train_val_test": "unknown exact",
            "val_test_influence_vocab": "train-only required",
            "mp_cross_split": "same",
            "test_files_exist": "split at load",
            "ssl_exclude": "test eval",
            "test_labels_inspected_this_audit": False,
            "leakage_risk": "MANAGEABLE",
            "leakage_reason": "same as Medium-HI",
        },
        {
            "dataset": "PaySim",
            "existing_split": "hourly_step",
            "split_type": "temporal_hourly",
            "timestamp_convention": "step*3600; extreme timestamp multiplicity",
            "rezeroed": True,
            "preprocess_fit": "train-fit; exclude balances forever",
            "train_val_test": "3792821 / 1276276 / 1293523",
            "val_test_influence_vocab": "type codes train-only",
            "mp_cross_split": "same",
            "test_files_exist": True,
            "ssl_exclude": "balances, isFlaggedFraud, labels, test metrics",
            "test_labels_inspected_this_audit": False,
            "leakage_risk": "MANAGEABLE",
            "leakage_reason": "balances/flags are severe if included; leakage-safe core avoids them; fraud≠AML label semantics differ",
        },
        {
            "dataset": "AMLSim",
            "existing_split": "none",
            "split_type": "to_be_defined_temporal",
            "timestamp_convention": "unknown",
            "rezeroed": "n/a",
            "preprocess_fit": "n/a",
            "train_val_test": "n/a",
            "val_test_influence_vocab": "n/a",
            "mp_cross_split": "n/a",
            "test_files_exist": False,
            "ssl_exclude": "SAR/alert features until policy defined",
            "test_labels_inspected_this_audit": False,
            "leakage_risk": "UNRESOLVED",
            "leakage_reason": "archives unextracted; alert/SAR join policy unknown",
        },
    ]
    write_csv(OUT / "split_leakage_matrix.csv", splits)

    # ------------------------------------------------------------------
    # Temporal targets
    # ------------------------------------------------------------------
    tf_rows = []
    for ds, status, note, n_cache_rows, bytes_est, ram_est, runtime_est, est_method in [
        (
            "Small-HI",
            "immediately_cacheable",
            "cache exists",
            HI_EDGES,
            TF_HI_CLEAN_B,
            "~measured prior jobs; builder historically <<128G",
            "measured prior (exists)",
            "measured",
        ),
        (
            "SAML-D",
            "immediately_cacheable",
            "shared-core train∪val cache exists; job 19512052",
            SD_EDGES_TV,
            TF_SD_B,
            "measured path under 128G",
            "measured job 19512052",
            "measured",
        ),
        (
            "Small-LI",
            "immediately_cacheable",
            "cache exists under temporal_flow_causal/Small-LI",
            LI_EDGES,
            TF_LI_DIR_B,
            "similar to HI/LI scale",
            "cache already built",
            "measured_dir",
        ),
        (
            "Medium-HI",
            "cache_builder_generalization_required",
            "add to SUPPORTED_DATASETS; ~6.3× HI rows",
            MED_HI_EDGES,
            tf_cache_bytes_est(MED_HI_EDGES),
            "extrapolate ~6× HI builder peak; may approach 128G",
            "extrapolate ~6× HI builder wall",
            "extrapolated_by_edges",
        ),
        (
            "Medium-LI",
            "cache_builder_generalization_required",
            "same as Medium-HI",
            MED_LI_EDGES,
            tf_cache_bytes_est(MED_LI_EDGES),
            "similar Medium-HI",
            "similar Medium-HI",
            "extrapolated_by_edges",
        ),
        (
            "PaySim",
            "immediately_cacheable",
            "cache exists; coarse ties use policy B",
            PS_EDGES,
            TF_PS_B,
            "measured exists",
            "cache already built",
            "measured",
        ),
        (
            "AMLSim",
            "formatter_loader_required_first",
            "no CSV yet",
            None,
            None,
            "unknown",
            "unknown",
            "unknown",
        ),
    ]:
        for t in TF_TARGETS:
            tf_rows.append(
                {
                    "dataset": ds,
                    "target": t,
                    "status": status,
                    "timestamp_resolution_ok": ds != "AMLSim",
                    "causal_past_only": True if ds != "AMLSim" else "unknown",
                    "labels_absent_from_targets": True if ds != "AMLSim" else "unknown",
                    "test_cache_construction": "avoid for SSL-dev (SAML-D already train∪val only)",
                    "note": note,
                    "est_cache_rows": n_cache_rows,
                    "est_cache_bytes": bytes_est,
                    "est_peak_RAM": ram_est,
                    "est_runtime": runtime_est,
                    "estimation_method": est_method,
                }
            )
    write_csv(OUT / "temporal_target_matrix.csv", tf_rows)

    # ------------------------------------------------------------------
    # Per-dataset resource estimates
    # ------------------------------------------------------------------
    li_s = est_s_per_step(LI_TRAIN, HI_TRAIN, HI_S_PER_STEP)  # LI closer to HI schema
    # blend HI/SD by train size for LI
    li_s = 0.5 * est_s_per_step(LI_TRAIN, HI_TRAIN, HI_S_PER_STEP) + 0.5 * est_s_per_step(
        LI_TRAIN, SD_TRAIN, SD_S_PER_STEP
    )
    ps_s = 0.5 * est_s_per_step(PS_TRAIN, HI_TRAIN, HI_S_PER_STEP) + 0.5 * est_s_per_step(
        PS_TRAIN, SD_TRAIN, SD_S_PER_STEP
    )
    med_train_approx = int(MED_HI_EDGES * 0.64)  # ~60-65% like HI
    med_s = est_s_per_step(med_train_approx, HI_TRAIN, HI_S_PER_STEP)

    resources = [
        {
            "dataset": "Small-HI",
            "graph_edges": HI_EDGES,
            "formatted_csv_size": "250M",
            "in_memory_graph_tensors": "measured workable under 128G with phase3 recipe",
            "loader_worker_overhead": "num_workers historically 8–16; phase3 used 16 cpus",
            "ports_tds_peak_RAM": "measured under 128G",
            "tf_cache_size": "~183M clean / 349M dir",
            "tf_cache_build_peak_RAM": "prior <128G",
            "expected_GPU_VRAM": "unknown (not logged); batch 8192 R198 historically fits 1 GPU",
            "sec_per_optimizer_step": HI_S_PER_STEP,
            "time_500_updates_sec": HI_S_PER_STEP * 500,
            "R198_train_val_emb_GiB": emb_giB(HI_TRAIN) + emb_giB(965_524),
            "probe_peak_RAM": "64G request measured sufficient (phase3)",
            "checkpoint_size": "~2.8MB",
            "estimation_method": "measured",
        },
        {
            "dataset": "SAML-D",
            "graph_edges": SD_EDGES_FULL,
            "formatted_csv_size": "483M",
            "in_memory_graph_tensors": "measured under 128G",
            "loader_worker_overhead": "16 cpus phase3",
            "ports_tds_peak_RAM": "measured under 128G",
            "tf_cache_size": "262M",
            "tf_cache_build_peak_RAM": "job 19512052 under 128G",
            "expected_GPU_VRAM": "unknown; fits 1 GPU in phase3",
            "sec_per_optimizer_step": SD_S_PER_STEP,
            "time_500_updates_sec": SD_S_PER_STEP * 500,
            "R198_train_val_emb_GiB": emb_giB(SD_TRAIN) + emb_giB(1_900_105),
            "probe_peak_RAM": "96G request (phase3)",
            "checkpoint_size": "~2.9MB",
            "estimation_method": "measured",
        },
        {
            "dataset": "Small-LI",
            "graph_edges": LI_EDGES,
            "formatted_csv_size": "342M",
            "in_memory_graph_tensors": "extrapolated between HI and SAML",
            "loader_worker_overhead": "same recipe",
            "ports_tds_peak_RAM": "extrapolated <128G",
            "tf_cache_size": "476M dir",
            "tf_cache_build_peak_RAM": "already built",
            "expected_GPU_VRAM": "similar HI/SAML",
            "sec_per_optimizer_step": li_s,
            "time_500_updates_sec": li_s * 500,
            "R198_train_val_emb_GiB": emb_giB(LI_TRAIN) + emb_giB(1_316_442),
            "probe_peak_RAM": "~64–96G estimate",
            "checkpoint_size": "~3MB",
            "estimation_method": "extrapolated_by_edges",
        },
        {
            "dataset": "PaySim",
            "graph_edges": PS_EDGES,
            "formatted_csv_size": "348M",
            "in_memory_graph_tensors": "similar Small scale",
            "loader_worker_overhead": "same",
            "ports_tds_peak_RAM": "historical transfer jobs ~50–65G host estimate",
            "tf_cache_size": "220M",
            "tf_cache_build_peak_RAM": "already built",
            "expected_GPU_VRAM": "similar",
            "sec_per_optimizer_step": ps_s,
            "time_500_updates_sec": ps_s * 500,
            "R198_train_val_emb_GiB": emb_giB(PS_TRAIN) + emb_giB(1_276_276),
            "probe_peak_RAM": "~64G estimate",
            "checkpoint_size": "~3MB",
            "estimation_method": "extrapolated_by_edges",
        },
        {
            "dataset": "Medium-HI",
            "graph_edges": MED_HI_EDGES,
            "formatted_csv_size": "1.7G",
            "in_memory_graph_tensors": "extrapolated ~6× HI — may exceed comfortable simultaneous residency",
            "loader_worker_overhead": "reduce workers",
            "ports_tds_peak_RAM": "HIGH — possibly >128G at build",
            "tf_cache_size": f"~{tf_cache_bytes_est(MED_HI_EDGES)/1e9:.2f}e9 bytes est",
            "tf_cache_build_peak_RAM": "unknown; may need >128G or streaming builder",
            "expected_GPU_VRAM": "unknown; neighbor sampling may still fit if batch fixed",
            "sec_per_optimizer_step": med_s,
            "time_500_updates_sec": med_s * 500,
            "R198_train_val_emb_GiB": emb_giB(med_train_approx) + emb_giB(int(MED_HI_EDGES * 0.19)),
            "probe_peak_RAM": "likely >=96–128G",
            "checkpoint_size": "~3MB",
            "estimation_method": "extrapolated_by_edges",
        },
        {
            "dataset": "Medium-LI",
            "graph_edges": MED_LI_EDGES,
            "formatted_csv_size": "1.6G",
            "in_memory_graph_tensors": "similar Medium-HI",
            "loader_worker_overhead": "reduce workers",
            "ports_tds_peak_RAM": "HIGH",
            "tf_cache_size": f"~{tf_cache_bytes_est(MED_LI_EDGES)/1e9:.2f}e9 bytes est",
            "tf_cache_build_peak_RAM": "unknown",
            "expected_GPU_VRAM": "unknown",
            "sec_per_optimizer_step": est_s_per_step(int(MED_LI_EDGES * 0.64), HI_TRAIN, HI_S_PER_STEP),
            "time_500_updates_sec": est_s_per_step(int(MED_LI_EDGES * 0.64), HI_TRAIN, HI_S_PER_STEP)
            * 500,
            "R198_train_val_emb_GiB": "similar Medium-HI",
            "probe_peak_RAM": "likely >=96–128G",
            "checkpoint_size": "~3MB",
            "estimation_method": "extrapolated_by_edges",
        },
        {
            "dataset": "AMLSim",
            "graph_edges": "unknown (~1e4 or ~1e6 from filenames)",
            "formatted_csv_size": "n/a",
            "in_memory_graph_tensors": "small if example archives",
            "loader_worker_overhead": "n/a",
            "ports_tds_peak_RAM": "likely low for 1M-edge example",
            "tf_cache_size": "n/a",
            "tf_cache_build_peak_RAM": "n/a",
            "expected_GPU_VRAM": "low",
            "sec_per_optimizer_step": "unknown",
            "time_500_updates_sec": "unknown",
            "R198_train_val_emb_GiB": "small",
            "probe_peak_RAM": "low",
            "checkpoint_size": "~3MB",
            "estimation_method": "unknown",
        },
    ]
    write_csv(OUT / "resource_estimates.csv", resources)

    # ------------------------------------------------------------------
    # Resident combinations
    # ------------------------------------------------------------------
    # Host RAM heuristic anchored on measured Phase-3 HI+SAML residency OK at 128G.
    # Model: 100 GiB for the HI+SAML pair (measured-feasible), then +30 GiB per
    # additional Small-scale domain, +150 GiB per Medium. Pure non-HI/SAML sets
    # fall back to additive Small-scale estimates.
    EXTRA_SMALL_GIB = 30
    MEDIUM_GIB = 150

    def combo(name: str, domains: List[str], notes: str = "") -> Dict[str, Any]:
        ds = set(domains)
        n_medium = sum(1 for d in domains if d.startswith("Medium"))
        n_small_extra = len(domains) - n_medium
        if {"Small-HI", "SAML-D"}.issubset(ds):
            # counted HI+SAML as the measured base pair
            n_small_extra = len(domains) - 2 - n_medium
            host = 100 + EXTRA_SMALL_GIB * max(n_small_extra, 0) + MEDIUM_GIB * n_medium
        else:
            host = 40 * (len(domains) - n_medium) + MEDIUM_GIB * n_medium
        steps = 500 * len(domains)
        # wall ≈ sum domain_step_times for round-robin mixed of length steps
        step_map = {
            "Small-HI": HI_S_PER_STEP,
            "SAML-D": SD_S_PER_STEP,
            "Small-LI": li_s,
            "PaySim": ps_s,
            "Medium-HI": med_s,
            "Medium-LI": est_s_per_step(int(MED_LI_EDGES * 0.64), HI_TRAIN, HI_S_PER_STEP),
        }
        mean_s = sum(step_map[d] for d in domains) / len(domains)
        wall = mean_s * steps
        tf = sum(
            {
                "Small-HI": TF_HI_CLEAN_B,
                "SAML-D": TF_SD_B,
                "Small-LI": TF_LI_DIR_B,
                "PaySim": TF_PS_B,
                "Medium-HI": tf_cache_bytes_est(MED_HI_EDGES),
                "Medium-LI": tf_cache_bytes_est(MED_LI_EDGES),
            }[d]
            for d in domains
        )
        # embeddings if full (N_encoders ≈ N+1 specialists+mixed) × N targets — storage heavy
        emb = (len(domains) + 1) * sum(
            {
                "Small-HI": emb_giB(HI_TRAIN + 965_524),
                "SAML-D": emb_giB(SD_TRAIN + 1_900_105),
                "Small-LI": emb_giB(LI_TRAIN + 1_316_442),
                "PaySim": emb_giB(PS_TRAIN + 1_276_276),
                "Medium-HI": emb_giB(int(MED_HI_EDGES * 0.83)),
                "Medium-LI": emb_giB(int(MED_LI_EDGES * 0.83)),
            }[d]
            for d in domains
        )
        return {
            "combination": name,
            "domains": "+".join(domains),
            "n_domains": len(domains),
            "est_simultaneous_host_RAM_GiB": host,
            "RAM_basis": "shared_overhead_plus_incremental_per_domain_anchored_on_phase3_HI_SAML_128G",
            "sufficient_128G": host <= 128,
            "sufficient_128G_with_worker_reduction": host <= 145,
            "larger_standard_mem_may_suffice": host <= 192,
            "est_graph_build_time": "dominated by largest domain; Medium >> Small",
            "total_steps_at_500_per_domain": steps,
            "est_mixed_training_wall_sec": round(wall, 1),
            "est_mixed_training_wall_hours": round(wall / 3600, 2),
            "est_tf_cache_bytes": int(tf),
            "est_checkpoint_MB": 3 * (len(domains) + 1),  # specialists+mixed rough
            "est_embedding_matrix_GiB_full_cells": round(emb, 1),
            "estimation_method": "measured_HI_SAML_base_plus_edge_extrapolation",
            "notes": notes,
        }

    combos = [
        combo(
            "1_HI_SAML_LI",
            ["Small-HI", "SAML-D", "Small-LI"],
            "recommended 3-domain smoke; ~155GiB heuristic → try 128G with fewer loader workers first",
        ),
        combo(
            "2_HI_SAML_PaySim",
            ["Small-HI", "SAML-D", "PaySim"],
            "diversity-focused 3-domain; similar RAM to LI combo",
        ),
        combo(
            "3_HI_SAML_LI_PaySim",
            ["Small-HI", "SAML-D", "Small-LI", "PaySim"],
            "4-domain; prefer block residency or higher mem request",
        ),
        combo(
            "4_HI_SAML_MediumHI",
            ["Small-HI", "SAML-D", "Medium-HI"],
            "Medium breaks simultaneous 128G residency",
        ),
        combo(
            "5_HI_SAML_LI_MediumHI_PaySim",
            ["Small-HI", "SAML-D", "Small-LI", "Medium-HI", "PaySim"],
            "infeasible simultaneous under current design",
        ),
        combo(
            "6_all_available_formatted",
            ["Small-HI", "SAML-D", "Small-LI", "Medium-HI", "Medium-LI", "PaySim"],
            "infeasible simultaneous; need block load/unload",
        ),
    ]
    write_csv(OUT / "resident_combination_estimates.csv", combos)

    # ------------------------------------------------------------------
    # Plans (markdown)
    # ------------------------------------------------------------------
    trainer_plan = """# N-domain trainer generalization plan (Phase-4)

## Current hardcodes (Phase-3)

- `mixed_ssl_phase3/__init__.py`: `DOMAINS=("Small-HI","SAML-D")`, fixed TF caches, 3 arms, `i%2` MIXED
- `mixed_ssl_phase2/schedule.py`: loader seed offsets 1 vs 2 only
- `scripts/train_mixed_ssl_phase3_scout.py`: `need_hi`/`need_sd`, dual namespaces, `bn_l1_hi_vs_sd`
- `shared_core_contract.py`: `SHARED_CORE_DATASETS` allowlist length 2
- Slurm/frozen-eval: array `0-2` / six-cell HI×SAML matrix

BN bundles and `LossNormState` dicts are already keyed by domain name once `active_domains` is generalized.

## Required changes for arbitrary N

1. **Domain registry / config** — `DomainSpec{name, data_flag, tf_cache, split_locks, probe_mem_gb}`
2. **Loader dict** — `Dict[str, loader_iter]` replace hi_/sd_ branches
3. **BN / edge-scaler / TF-scaler / LossNorm dicts** — iterate registry
4. **Scheduler** — round-robin `i % N` + optional weights; exposure counters per domain
5. **Checkpoint / resume** — serialize all bundles; schema version field
6. **Logging** — per-domain raw/norm/C contributions; drop HI/SD-only metric names
7. **Validation** — optional per-domain val probe hook (not required for smoke)
8. **Arms** — N specialists + one MIXED_RR; Slurm array `0..(N)`
9. **Frozen eval** — cells `(N+1)×N` with BN policy specialist→source, mixed→target
10. **Contract** — extend `SHARED_CORE_DATASETS` / bump contract id if geometry unchanged but allowlist grows

## Focused tests

1. 3 synthetic domains: balanced RR schedule + distinct loader seeds
2. Matching: MIXED exposure-k stream equals specialist prefix hashes
3. BN/LossNorm independence for N=3
4. Arm factory bijection
5. Preflight missing-cache refusal
6. 2-domain regression parity with Phase-3 seeds/schedule

## Bounded Phase-4 implementation order

1. Extract `DomainSpec` + generic schedule/matching (keep Phase-3 wrapper)
2. Dict-orchestrate trainer; add `step_sec` / CUDA peak logging
3. Allowlist Small-LI (and optionally PaySim) under shared-core
4. Slurm template for variable arm count
5. 2-domain parity smoke → 3-domain smoke (HI+SAML+LI)

**Do not implement in this audit.**
"""
    (OUT / "trainer_generalization_plan.md").write_text(trainer_plan, encoding="utf-8")

    amlsim_plan = """# AMLSim loader / formatter plan

## Status on disk

- Path: `aml-data/aml-sim/`
- Files: `100vertices-10Kedges.7z` (~150K), `10Kvertices-1Medges.7z` (~4.8M)
- No extracted CSVs; no `7z`/`py7zr` on the audit host at inspection time
- Filenames imply **example** IBM AMLSim-scale graphs, not a full bank-scale dump
- Version/source: not recorded beside archives (treat as unlabeled sample packages)

## Classification

- Extracted data: **DATA_MISSING**
- Implementation work: **FORMATTER_ONLY** then **LOADER_REQUIRED** (not automatic scientific blockers)
- Leakage: **UNRESOLVED** until alert/SAR schema inspected

## Proposed dataset ID

`AMLSim-10K` / `AMLSim-1M` (or `amlsim_example_1m_v0`) under `aml-data/AMLSim-1M/`.

## Implementation steps

1. **Extract** archives on a CPU node with `7z` or `py7zr` into a staging dir
2. **Identify files**: transactions, accounts, alerts/SAR (AMLSim typically separates these)
3. **Headers + bounded sample** (first N rows only)
4. **Formatter** `format_amlsim_files.py`:
   - Inputs: transaction CSV (+ optional account map)
   - Outputs: `formatted_transactions.csv` with standard columns
   - Stable **EdgeID** = sequential assignment on causally sorted transactions (document if reassigned)
   - Timestamp: convert to seconds; define rezero policy
   - Amount: choose outgoing amount column; document currency
   - Labels: join alert/SAR → edge label **only as `Is Laundering`**, never as encoder features
5. **Split protocol**: temporal calendar/day or step buckets; train-fit norms; no test in SSL caches
6. **Graph construction**: reuse `data_loading.get_data` after format
7. **Shared-core mapping**: Timestamp, Amount Received, ports, TDS
8. **Integrity tests**: unique EdgeIDs, directed edges, parallel edges preserved, label rate, no SAR columns in `edge_attr`
9. **TF cache**: extend `SUPPORTED_DATASETS`; train∪val only; causal policy B
10. **Effort estimate**: **2–4 engineering days** for example 1M-edge package if schema is standard AMLSim; longer if alert join is nonstandard

## Label handling

- Prefer transaction-level laundering if present
- If only account/alert-level: define explicit join; treat alert membership as **label**, not feature
- Document SAR leakage risk in dataset card

## Do not reject merely because code is absent
"""
    (OUT / "amlsim_loader_plan.md").write_text(amlsim_plan, encoding="utf-8")

    # ------------------------------------------------------------------
    # Scores / rankings / recommendations
    # ------------------------------------------------------------------
    scores = {
        "Small-LI": {
            "independent_diversity": 2,
            "graph_task_relevance": 5,
            "shared_core_compat": 5,
            "temporal_target_compat": 5,
            "split_integrity": 5,
            "leakage_safety": 5,
            "implementation_effort": 5,
            "host_memory": 5,
            "expected_runtime": 4,
            "thesis_value": 3,
        },
        "PaySim": {
            "independent_diversity": 5,
            "graph_task_relevance": 4,
            "shared_core_compat": 4,
            "temporal_target_compat": 4,
            "split_integrity": 4,
            "leakage_safety": 3,
            "implementation_effort": 3,
            "host_memory": 5,
            "expected_runtime": 4,
            "thesis_value": 5,
        },
        "Medium-HI": {
            "independent_diversity": 1,
            "graph_task_relevance": 5,
            "shared_core_compat": 5,
            "temporal_target_compat": 3,
            "split_integrity": 3,
            "leakage_safety": 4,
            "implementation_effort": 2,
            "host_memory": 1,
            "expected_runtime": 1,
            "thesis_value": 3,
        },
        "Medium-LI": {
            "independent_diversity": 1,
            "graph_task_relevance": 5,
            "shared_core_compat": 5,
            "temporal_target_compat": 3,
            "split_integrity": 3,
            "leakage_safety": 4,
            "implementation_effort": 2,
            "host_memory": 1,
            "expected_runtime": 1,
            "thesis_value": 2,
        },
        "AMLSim": {
            "independent_diversity": 5,
            "graph_task_relevance": 4,
            "shared_core_compat": 2,
            "temporal_target_compat": 2,
            "split_integrity": 1,
            "leakage_safety": 2,
            "implementation_effort": 2,
            "host_memory": 5,
            "expected_runtime": 5,
            "thesis_value": 4,
        },
    }

    rankings = {
        "easiest_next_dataset": "Small-LI",
        "strongest_diversity_contribution": "PaySim (available now); AMLSim after formatter",
        "strongest_scale_contribution": "Medium-HI",
        "best_overall_next_addition": "Small-LI",
        "best_realistic_final_collection": "Small-HI + SAML-D + Small-LI + PaySim",
    }

    recommendations = {
        "1_third_training_domain": "Small-LI",
        "2_first_independent_domain_after_SAML_D": "PaySim (leakage-safe shared-core)",
        "3_medium_variant_if_one": "Medium-HI",
        "4_include_paysim_despite_difficulty": True,
        "4_reason": "Only ready independent generator besides SAML-D; shared-core can drop type-dup categoricals",
        "5_amlsim_bounded_enough": True,
        "5_caveat": "Example archives (~1M edges max by filename); extract+formatter first; not a scale dataset",
        "6_max_realistic_domain_count_current_mem": 3,
        "6_note": (
            "HI+SAML measured OK at 128G. Adding Small-LI or PaySim is the max realistic "
            "simultaneous set; may need fewer loader workers. Four Small-scale domains or any "
            "Medium needs higher mem or block load/unload."
        ),
        "7_n_domain_trainer_before_next_smoke": True,
        "8_next_cache_loader_implementation": (
            "Extend SHARED_CORE_DATASETS + Phase-4 domain registry; wire existing Small-LI TF cache; "
            "no new TF build required for LI. For PaySim: register existing paysim TF cache under shared-core path."
        ),
        "9_three_domain_smoke": {
            "datasets": ["Small-HI", "SAML-D", "Small-LI"],
            "schedule": "round_robin",
            "updates_per_domain": 500,
            "total_mixed_steps": 1500,
            "arms": ["SMALL_HI_ONLY", "SAMLD_ONLY", "SMALL_LI_ONLY", "MIXED_RR"] ,
            "resources": "partition=mit_preemptable account=mit_general qos=normal mem=128G gres=gpu:1 cpus=16 time<=08:00:00",
            "expected_wall_mixed_hours": round((sum([HI_S_PER_STEP, SD_S_PER_STEP, li_s]) / 3) * 1500 / 3600, 2),
            "expected_wall_note": "plus 3 specialist arms if run; array concurrency %2",
        },
        "10_final_collection": ["Small-HI", "SAML-D", "Small-LI", "PaySim"],
        "11_human_decisions": [
            "Whether Medium is worth scale cost given it is not an independent generator",
            "Whether PaySim fraud labels are acceptable downstream proxies for AML thesis claims",
            "Priority of AMLSim formatter vs PaySim 3-domain diversity smoke",
            "Whether full (N+1)×N frozen-eval embedding matrices are retained (storage O(N²))",
            "Whether standard-account mem >128G is available if Medium is attempted",
        ],
    }

    # Optional CPU job — not required
    optional_cpu = {
        "needed": False,
        "reason_not_submitted": (
            "Shared-core compatibility, rankings, and residency conclusions are reliable from "
            "existing TF metas, headers, prior Medium edge estimates, and Phase-3 measured step times. "
            "Exact Medium row counts / EdgeID≠index checks would refine estimates ±10% only."
        ),
        "if_run_later": {
            "why": "Verify Medium-HI/LI edge counts, EdgeID alignment, split bucket boundaries",
            "files": [
                "aml-data/Medium-HI/formatted_transactions.csv",
                "aml-data/Medium-LI/formatted_transactions.csv",
            ],
            "fields_read": "EdgeID, Timestamp, from_id, to_id, Amount Received, Is Laundering (counts/min/max only)",
            "expected_runtime": "10–30 min sequential",
            "expected_memory": "<=32G streaming",
            "outputs": "medium_*_metadata.json",
            "slurm": "mit_preemptable / mit_general / qos=normal / no GPU / <=128G",
        },
        "job_id": None,
    }

    alternatives_if_mem_tight = [
        {
            "alt": "fewer_domains_per_scout",
            "faithfulness": 5,
            "eng_cost": 1,
            "note": "Best default",
        },
        {
            "alt": "one_Medium_only_vs_specialists",
            "faithfulness": 3,
            "eng_cost": 3,
            "note": "Scale study, weak diversity",
        },
        {
            "alt": "lower_loader_workers",
            "faithfulness": 5,
            "eng_cost": 1,
            "note": "Easy RAM relief",
        },
        {
            "alt": "short_block_cyclic_load_unload",
            "faithfulness": 3,
            "eng_cost": 4,
            "note": "Changes optimization trajectory vs simultaneous residency",
        },
        {
            "alt": "memory_mapped_tensors",
            "faithfulness": 4,
            "eng_cost": 3,
            "note": "If supported for edge_attr",
        },
        {
            "alt": "graph_sharding",
            "faithfulness": 2,
            "eng_cost": 5,
            "note": "Avoid for thesis unless necessary",
        },
        {
            "alt": "sequential_replay_finetune",
            "faithfulness": 2,
            "eng_cost": 2,
            "note": "Not mixed GFM; use only as control",
        },
    ]

    payload = {
        "ok": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "read_only": True,
        "no_encoder_training": True,
        "no_embedding_extraction": True,
        "no_downstream_probes": True,
        "no_test_evaluation": True,
        "no_destructive_ops": True,
        "no_gpu_jobs": True,
        "no_slurm_jobs_submitted": True,
        "full_csv_scans": False,
        "reference_system": {
            "feature_contract": "smallhi_samld_shared_core_v1",
            "final_feature_order": list(CORE6),
            "model": "shared GIN + R198 InfoNCE + adaptive TF-MoE (3 targets) + per-domain scalers/BN/LossNorm",
            "tf_targets": list(TF_TARGETS),
        },
        "family_distinction": {
            "amlworld_variants_same_simulator": [
                "Small-HI",
                "Small-LI",
                "Medium-HI",
                "Medium-LI",
            ],
            "independent_families": ["SAML-D", "PaySim", "AMLSim", "AMLNet", "BankSim"],
            "label_semantics_differ": {
                "AMLWorld/SAML-D": "laundering edge labels",
                "PaySim": "isFraud (fraud), not AML laundering",
                "AMLSim": "unresolved until extract",
            },
            "do_not_claim_four_amlworld_as_four_independent_sources": True,
        },
        "paysim_safety_policy": {
            "exclude_from_encoder": [
                "oldbalanceOrg",
                "newbalanceOrig",
                "oldbalanceDest",
                "newbalanceDest",
                "isFlaggedFraud",
                "isFraud / Is Laundering",
            ],
            "source_warning_preserved": (
                "Cancelled fraud transactions make balance columns inappropriate for fraud detection"
            ),
            "shared_core_from_safe_fields": True,
            "method": "Timestamp=step*3600, amount, endpoints → ports+TDS; drop type-dup categoricals from encoder",
            "all_native_contract": "diagnostic only; outside default multi-domain GFM",
            "classification": "DIRECT_COMPATIBLE under leakage-safe shared-core",
        },
        "scores": scores,
        "rankings": rankings,
        "recommendations": recommendations,
        "resident_combinations": combos,
        "alternatives_if_mem_tight": alternatives_if_mem_tight,
        "optional_cpu_metadata_job": optional_cpu,
        "n_domain_trainer_summary": {
            "phase3_assumes_exactly_2_domains": True,
            "generalization_needed_before_3domain_smoke": True,
            "plan_path": str((OUT / "trainer_generalization_plan.md").relative_to(ROOT)),
        },
        "source_sha256": {k: {"path": sources[k], "sha256": source_sha[k]} for k in sources},
        "confirmations": {
            "no_encoder_training": True,
            "no_embedding_extraction": True,
            "no_downstream_probes": True,
            "no_test_evaluation": True,
            "no_destructive_operations": True,
            "no_gpu_jobs": True,
            "training_loader_model_code_changed": False,
            "datasets_modified": False,
        },
    }
    write_json(TWIN, payload)
    write_json(
        OUT / "artifact_manifest.json",
        {
            "outputs": [
                str(NOTES.relative_to(ROOT)),
                str(TWIN.relative_to(ROOT)),
                "dataset_inventory.csv",
                "feature_mapping.csv",
                "native_feature_inventory.csv",
                "split_leakage_matrix.csv",
                "temporal_target_matrix.csv",
                "resource_estimates.csv",
                "resident_combination_estimates.csv",
                "trainer_generalization_plan.md",
                "amlsim_loader_plan.md",
            ],
            "script": "scripts/audit_multidataset_expansion_compatibility_resources.py",
        },
    )

    # ------------------------------------------------------------------
    # Notes
    # ------------------------------------------------------------------
    smoke = recommendations["9_three_domain_smoke"]
    lines = [
        "# Multi-dataset expansion compatibility and resource audit",
        "",
        f"> Twin: `{TWIN.relative_to(ROOT)}`",
        f"> Created: {payload['created_at_utc']}",
        "",
        "**Read-only planning audit.** No encoder training, extraction, probes, test evaluation,",
        "dataset rewrites, destructive ops, or GPU/Slurm jobs.",
        "",
        "## Executive recommendations",
        "",
        f"1. **Third training domain:** {recommendations['1_third_training_domain']}",
        f"2. **First independent domain after SAML-D:** {recommendations['2_first_independent_domain_after_SAML_D']}",
        f"3. **If one Medium:** {recommendations['3_medium_variant_if_one']}",
        f"4. **Include leakage-safe PaySim?** Yes — for generator diversity (shared-core drops type-dup categoricals).",
        f"5. **AMLSim attempt?** Yes, bounded (example archives); formatter/loader first — not a blocker by absence of code.",
        f"6. **Max realistic domains under current mem design:** {recommendations['6_max_realistic_domain_count_current_mem']} simultaneous; Medium needs special handling.",
        "7. **N-domain trainer before next smoke?** **Yes.**",
        f"8. **Next implementation:** {recommendations['8_next_cache_loader_implementation']}",
        f"9. **3-domain smoke:** `{smoke['datasets']}`, {smoke['updates_per_domain']}/domain, "
        f"{smoke['total_mixed_steps']} mixed steps, 128G/1GPU, ~{smoke['expected_wall_mixed_hours']} h mixed wall.",
        f"10. **Final collection:** {recommendations['10_final_collection']}",
        "11. **Human decisions:** Medium worth?, PaySim fraud≠AML claims, AMLSim vs PaySim priority, embedding storage, >128G for Medium.",
        "",
        "## Family distinction (critical)",
        "",
        "- **AMLWorld variants (same simulator):** Small-HI, Small-LI, Medium-HI, Medium-LI — **not** four independent sources.",
        "- **Independent families:** SAML-D, PaySim, AMLSim (archives), plus on-disk AMLNet/BankSim (out of scope).",
        "- **Label semantics differ:** AML laundering vs PaySim fraud vs AMLSim unresolved.",
        "",
        "## Reference system",
        "",
        f"Contract `{payload['reference_system']['feature_contract']}` → `{CORE6}`.",
        "Shared GIN, R198 InfoNCE, projection off, adaptive TF-MoE (3 causal targets),",
        "per-domain edge/TF scalers, LossNorm, BN; shared encoder/affine/experts; no test access.",
        "",
        "## Shared-core compatibility summary",
        "",
        "| Dataset | Class | Notes |",
        "|---|---|---|",
        "| Small-HI | DIRECT_COMPATIBLE | reference |",
        "| SAML-D | DIRECT_COMPATIBLE | reference; EdgeID≠row index |",
        "| Small-LI | DIRECT_COMPATIBLE | loader+TF exist; allowlist/registry only |",
        "| Medium-HI/LI | DIRECT_COMPATIBLE | scale/TF-builder ops; not independent diversity |",
        "| PaySim | DIRECT_COMPATIBLE | leakage-safe core from step/amount/ports/TDS |",
        "| AMLSim | DATA_MISSING | extract+formatter+loader = implementation work |",
        "",
        "Ports/TDS algorithms in `data_util.py` are graph-generic; reverse-edge swap via `correct_reverse_edge_features`.",
        "",
        "## PaySim leakage-safe protocol",
        "",
        "Exclude balances, `isFlaggedFraud`, and fraud labels from all encoder inputs.",
        "Preserve source warning about cancelled-fraud balances.",
        "Do not treat type-dup currency/payment slots as real AML fields; shared-core omits them.",
        "All-native PaySim contracts remain diagnostic-only.",
        "",
        "## Resource anchors (measured)",
        "",
        f"- Small-HI: {HI_S_PER_STEP:.3f} s/step (1000-step arm); TF ~183MB clean; emb train+val ~{emb_giB(HI_TRAIN)+emb_giB(965524):.2f} GiB",
        f"- SAML-D: {SD_S_PER_STEP:.3f} s/step; TF 262MB; emb train+val ~{emb_giB(SD_TRAIN)+emb_giB(1900105):.2f} GiB",
        "- Phase-3 HI+SAML simultaneous residency: **OK under 128G** (standard account).",
        "- Phase-3 frozen-eval embeddings retained: **~25 GiB** for 3 encoders × 2 targets.",
        "- Checkpoints: ~2.8–2.9 MB/file.",
        "- Slurm: `mit_preemptable` / `mit_general` / `qos=normal`; do not assume expired advanced limits.",
        "",
        "## Resident combinations (heuristic host GiB sum)",
        "",
        "| Combo | Est RAM GiB | 128G OK? | OK w/ fewer workers? | Mixed steps | Est mixed wall h |",
        "|---|---:|---|---|---:|---:|",
    ]
    for c in combos:
        lines.append(
            f"| {c['combination']} | {c['est_simultaneous_host_RAM_GiB']} | "
            f"{c['sufficient_128G']} | {c['sufficient_128G_with_worker_reduction']} | "
            f"{c['total_steps_at_500_per_domain']} | {c['est_mixed_training_wall_hours']} |"
        )
    lines += [
        "",
        "HI+SAML is measured OK at 128G. Three Small-scale domains are the practical max;",
        "try 128G with reduced loader workers before requesting more memory.",
        "Medium-inclusive combos need special residency. Alternatives by faithfulness:",
        "fewer domains > lower workers > mmap > block cyclic > sharding.",
        "",
        "## Rankings",
        "",
        f"1. Easiest next: **{rankings['easiest_next_dataset']}**",
        f"2. Strongest diversity: **{rankings['strongest_diversity_contribution']}**",
        f"3. Strongest scale: **{rankings['strongest_scale_contribution']}**",
        f"4. Best overall next: **{rankings['best_overall_next_addition']}**",
        f"5. Best realistic final set: **{rankings['best_realistic_final_collection']}**",
        "",
        "Per-axis scores (1–5) are in the twin JSON under `scores` (not collapsed to one number).",
        "",
        "## N-domain trainer",
        "",
        "Phase-3 hardcodes exactly two domains / 1:1 / three arms / fixed caches.",
        f"See `{ (OUT / 'trainer_generalization_plan.md').relative_to(ROOT) }`.",
        "",
        "## AMLSim",
        "",
        f"See `{ (OUT / 'amlsim_loader_plan.md').relative_to(ROOT) }`.",
        "Archives only; example-scale; formatter/loader are implementation work, not automatic blockers.",
        "",
        "## Optional CPU metadata job",
        "",
        "**Not submitted.** Existing metadata suffices for compatibility and ranking decisions.",
        "A later streaming Medium count job is optional (see twin JSON `optional_cpu_metadata_job`).",
        "",
        "## Confirmations",
        "",
        "- no encoder training / embedding extraction / probes / test evaluation",
        "- no destructive ops / no GPU jobs / no Slurm jobs",
        "- no loader/formatter/model/training code modified",
        "- no datasets modified",
        "- no unbounded full CSV scans",
        "",
    ]
    NOTES.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "notes": str(NOTES.relative_to(ROOT)), "twin": str(TWIN.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()