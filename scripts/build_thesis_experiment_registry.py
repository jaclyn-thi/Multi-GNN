#!/usr/bin/env python3
"""Build thesis experiment registry from existing diagnostics JSON (no metric inference)."""

import csv
import json
import re
import statistics as stats
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
DIAG = ROOT / "results" / "diagnostics"

FIELDS = [
    "run_id", "dataset", "dataset_positive_rate", "objective", "encoder", "seed",
    "training_epochs", "selected_epoch", "checkpoint_policy", "supervised_head",
    "graph_flags", "emlps", "tds", "reverse_mp", "ego", "ports",
    "contrastive_variant", "asymmetric", "projection_head", "negative_count",
    "queue_size", "fnf_rule", "positive_rule", "morphology_auxiliary_loss",
    "representation_source", "representation_dim", "probe_feature_stack",
    "probe_class_weight", "probe_C", "threshold_rule",
    "AUROC", "AUPRC", "F1", "F1_fixed", "precision", "recall",
    "precision_at_100", "recall_at_100", "lift_at_100",
    "precision_at_500", "recall_at_500", "lift_at_500",
    "precision_at_1000", "recall_at_1000", "lift_at_1000",
    "paired_test_n", "source_json", "source_note", "checkpoint_path", "status",
    "scout_or_formal", "superseded", "thesis_role", "validation_status",
    "table_eligible", "table_group", "duplicate_resolution", "caveats",
    # Provenance / comparability (2026-07-22 sync)
    "protocol_family", "split_protocol", "graph_representation",
    "batch_size", "accum_steps", "negative_pool_semantics",
    "reverse_feature_semantics", "preserve_seed_edges",
    "positive_set_definition", "knn_scope", "job_id", "paper_comparable",
    # D+ partial fine-tune schema (2026-07-24); empty on historical rows until filled
    "finetune_protocol", "classifier_lr", "encoder_lr", "partial_unfreeze_modules",
]

TEMPORAL_ARM_STACK = {
    "A_embedding": ("A", "embedding"),
    "B_embedding_raw": ("B", "embedding+raw"),
    "C_embedding_temporal_flow": ("C", "embedding+temporal_flow_causal"),
    "D_embedding_raw_temporal_flow": ("D", "embedding+raw+temporal_flow_causal"),
}

TEMPORAL_FLOW_SOURCES = [
    "results/diagnostics/temporal_flow_ablation_small_hi_40ep_seed2.json",
    "results/diagnostics/temporal_flow_ablation_small_hi_40ep_seed2_maxiter5000.json",
    "results/diagnostics/temporal_flow_ablation_small_li_seed1.json",
    "results/diagnostics/temporal_flow_ablation_small_li_seed2.json",
    "results/diagnostics/temporal_flow_ablation_small_li_seed3.json",
    "results/diagnostics/temporal_flow_ablation_small_li_seed1_maxiter5000.json",
    "results/diagnostics/temporal_flow_ablation_small_li_seed2_maxiter5000.json",
    "results/diagnostics/temporal_flow_ablation_small_li_seed3_maxiter5000.json",
    "results/diagnostics/temporal_flow_causal_ablation_summary.json",
    "results/diagnostics/temporal_flow_causal_validation_summary.json",
    "results/diagnostics/temporal_flow_ablation_maxiter5000_comparison.json",
    "results/diagnostics/temporal_flow_causal_leakage_audit.json",
    "results/diagnostics/temporal_flow_shuffle_control_summary.json",
]

PNA_FOLLOWUP_SOURCES = [
    "results/diagnostics/pre_embedding_3h_vs_post_embedding_pna_emlps_tds_seed1.json",
    "results/diagnostics/pna_width_param_audit.json",
    "results/diagnostics/pna_width_aligned_probe.json",
    "results/diagnostics/pna_width65_temporal_flow_probe.json",
    "results/diagnostics/pna_width65_best_stack_comparison.json",
]

TF_AUX_OBJECTIVE_SOURCES = [
    "results/diagnostics/temporal_flow_aux_objective_scout.json",
    "results/diagnostics/tf_aux_tf_bins5_w0.10_post128_seed1.json",
    "results/diagnostics/tf_aux_tf_bins5_w0.10_pre3h_seed1.json",
    "results/diagnostics/tf_aux_tf_reg_w0.10_post128_seed1.json",
    "results/diagnostics/tf_aux_tf_reg_w0.10_pre3h_seed1.json",
    "results/diagnostics/tf_aux_tf_bins10_w0.10_post128_seed1.json",
    "results/diagnostics/tf_aux_tf_bins10_w0.10_pre3h_seed1.json",
    "results/diagnostics/tf_aux_tf_reg_w0.05_post128_seed1.json",
    "results/diagnostics/tf_aux_tf_reg_w0.05_pre3h_seed1.json",
]

TF_AUX_PROBE_FILES = [
    "tf_aux_tf_bins5_w0.10_post128_seed1.json",
    "tf_aux_tf_bins5_w0.10_pre3h_seed1.json",
    "tf_aux_tf_reg_w0.10_post128_seed1.json",
    "tf_aux_tf_reg_w0.10_pre3h_seed1.json",
    "tf_aux_tf_bins10_w0.10_post128_seed1.json",
    "tf_aux_tf_bins10_w0.10_pre3h_seed1.json",
    "tf_aux_tf_reg_w0.05_post128_seed1.json",
    "tf_aux_tf_reg_w0.05_pre3h_seed1.json",
]

TF_SOFT_OBJECTIVE_SOURCES = [
    "results/diagnostics/temporal_flow_soft_positive_scout.json",
    "results/diagnostics/tf_soft_tf_soft_bins5_min3_cap16_w0.05_post128_seed1.json",
    "results/diagnostics/tf_soft_tf_soft_bins5_min3_cap16_w0.05_pre3h_seed1.json",
    "results/diagnostics/tf_soft_tf_soft_bins5_min4_cap16_w0.10_post128_seed1.json",
    "results/diagnostics/tf_soft_tf_soft_bins5_min4_cap16_w0.10_pre3h_seed1.json",
    "results/diagnostics/tf_soft_tf_soft_bins10_min4_cap32_w0.05_post128_seed1.json",
    "results/diagnostics/tf_soft_tf_soft_bins10_min4_cap32_w0.05_pre3h_seed1.json",
    "results/diagnostics/tf_soft_tf_soft_strict_bins10_min5_cap4_w0.01_post128_seed1.json",
    "results/diagnostics/tf_soft_tf_soft_strict_bins10_min5_cap4_w0.01_pre3h_seed1.json",
]

TF_SOFT_PROBE_FILES = [
    "tf_soft_tf_soft_bins5_min3_cap16_w0.05_post128_seed1.json",
    "tf_soft_tf_soft_bins5_min3_cap16_w0.05_pre3h_seed1.json",
    "tf_soft_tf_soft_bins5_min4_cap16_w0.10_post128_seed1.json",
    "tf_soft_tf_soft_bins5_min4_cap16_w0.10_pre3h_seed1.json",
    "tf_soft_tf_soft_bins10_min4_cap32_w0.05_post128_seed1.json",
    "tf_soft_tf_soft_bins10_min4_cap32_w0.05_pre3h_seed1.json",
    "tf_soft_tf_soft_strict_bins10_min5_cap4_w0.01_post128_seed1.json",
    "tf_soft_tf_soft_strict_bins10_min5_cap4_w0.01_pre3h_seed1.json",
]

MORPH_OBJ_RECALL_PROBE_FILES = [
    "morph_obj_baseline_pre3h_seed1.json",
    "morph_obj_baseline_post128_seed1.json",
    "morph_obj_degflow_pre3h_seed1.json",
    "morph_obj_degflow_post128_seed1.json",
    "morph_obj_clustering_pre3h_seed1.json",
    "morph_obj_clustering_post128_seed1.json",
    "morph_obj_degflow_tfreg_pre3h_seed1.json",
    "morph_obj_degflow_tfreg_post128_seed1.json",
]

# Focused degflow multiseed replication (seeds 2–3 + optional seed2 baseline).
# Seed-1 degflow/baseline remain in MORPH_OBJ_RECALL_PROBE_FILES above.
DEGFLOW_MULTISEED_PROBE_FILES = [
    "morph_obj_baseline_pre3h_seed2.json",
    "morph_obj_baseline_post128_seed2.json",
    "morph_obj_degflow_pre3h_seed2.json",
    "morph_obj_degflow_post128_seed2.json",
    "morph_obj_degflow_pre3h_seed3.json",
    "morph_obj_degflow_post128_seed3.json",
]

# Temporal-flow regression aux multiseed confirmation (seeds 2–3).
# Seed-1 tf_reg probes remain in TF_AUX_PROBE_FILES above.
TF_REG_AUX_MULTISEED_PROBE_FILES = [
    "tf_aux_tf_reg_w0.10_pre3h_seed2.json",
    "tf_aux_tf_reg_w0.10_post128_seed2.json",
    "tf_aux_tf_reg_w0.10_pre3h_seed3.json",
    "tf_aux_tf_reg_w0.10_post128_seed3.json",
    "tf_aux_tf_reg_w0.05_pre3h_seed2.json",
    "tf_aux_tf_reg_w0.05_post128_seed2.json",
    "tf_aux_tf_reg_w0.05_pre3h_seed3.json",
    "tf_aux_tf_reg_w0.05_post128_seed3.json",
]

# Contrastive objective resource scout (seed2 large_bs + edge_drop).
CTR_RES_PROBE_FILES = [
    "ctr_res_large_bs_pre3h_seed2.json",
    "ctr_res_large_bs_post128_seed2.json",
    "ctr_res_edge_drop_pre3h_seed2.json",
    "ctr_res_edge_drop_post128_seed2.json",
]

STACK_MAP = {
    "embedding_only": "embedding",
    "embedding_plus_raw": "embedding+raw",
    "embedding_plus_raw_morph": "embedding+raw+morph",
    "raw": "raw",
    "morph": "morph",
    "raw+morph": "raw+morph",
    "embedding": "embedding",
    "embedding+raw": "embedding+raw",
    "embedding+raw+morph": "embedding+raw+morph",
}

REP_MAP = {
    "post": "post_embedding_128",
    "pre": "pre_embedding_3h",
    "post_embedding_128": "post_embedding_128",
    "pre_embedding_3h": "pre_embedding_3h",
    "orig_post_128": "post_embedding_128",
    "orig_pre_3h_198": "pre_embedding_3h",
    "new_post_198": "post_embedding_198_scout",
    "new_pre_3h_198": "pre_embedding_3h_emb198_scout",
}

CANONICAL_PREPOST_SOURCE = "results/diagnostics/pre3h_strong_run_comparison.json"
MULTISEED_LI_SOURCE = "results/diagnostics/pre_embedding_3h_vs_post_embedding_small_li_multiseed.json"
EMB198_SOURCE = "results/diagnostics/small_li_embedding_dim_128_vs_198.json"
LEGACY_EVAL_SOURCE = "results/diagnostics/eval_small_li_legacy_supervised_gin_emlps_tds_100ep_seed1.json"

MULTISEED_STD_DDOF = 1
MULTISEED_STD_LABEL = "sample standard deviation (ddof=1)"

THESIS_ROLE_RULES_MD = """
### `thesis_role` classification rules (conservative)

| Value | Rule |
|-------|------|
| `thesis_primary` | Small-LI multiseed pre/post; **paper-faithful** Small-HI ports TDS-off supervised (paper_argmax); Small-HI strong-run paired pre/post for gin 40ep s2 and FNF HI |
| `thesis_supporting` | Architecture sweep; alert-budget; feature ablation; single-file HI pre/post; 20ep baseline; FNF/LI secondary |
| `diagnostic` | emb198; val-tuned supervised; batch-size E/F; GCPAL audits; A/B/C/D contrastive arms; txn-node scouts; random-40 |
| `negative_result` | degree-aware edge-drop; superseded non-legacy supervised |
| `historical` | Rows from superseded protocols not marked superseded=true |
| `superseded` | `superseded=true` (includes old Small-HI TDS-on supervised for paper table) |

### `paper_comparable` (Multi-GIN+EU)

`true` only for Small-HI legacy supervised with ports, **tds=False**, paper_argmax, formal seeds/aggregate.
Old TDS-on supervised → `false`. Contrastive / txn-node / random-40 → `false`.
"""

HI_PORTS_AGG_SOURCE = (
    "results/diagnostics/eval_small_hi_legacy_supervised_gin_emlps_ports_50ep_seeds1-3_formal_aggregate.json"
)

JUL22_ABCD_PROBES = [
    ("probe_feature_ablation_current_protocol_gin_40ep_seed2_tds_off.json",
     "tds_off", False, "inherited_or_n_a", False, "B_tds_off"),
    ("probe_feature_ablation_current_protocol_gin_40ep_seed2_tds_off_preserve_seed.json",
     "tds_off_preserve", False, "inherited_or_n_a", True, "C_tds_off_preserve"),
    ("probe_feature_ablation_current_protocol_gin_40ep_seed2_tds_corrected.json",
     "tds_corrected", True, "corrected_named_swap", False, "D_tds_corrected"),
    ("probe_feature_ablation_current_protocol_gin_40ep_seed2_tds_corrected_preserve_seed.json",
     "tds_corrected_preserve", True, "corrected_named_swap", True, "Dplus_corrected_preserve"),
]

JUL22_BATCH_EF_PROBES = [
    ("probe_feature_ablation_gin_emlps_ports_tds_corrected_preserve_seed_asym_proj_allneg_bs8192_accum4_10ep_seed2.json",
     8192, 4, "E_bs8192_accum4"),
    ("probe_feature_ablation_gin_emlps_ports_tds_corrected_preserve_seed_asym_proj_allneg_bs2048_accum16_10ep_seed2.json",
     2048, 16, "F_bs2048_accum16"),
]


def _blank_row() -> Dict[str, Any]:
    return {f: None for f in FIELDS}


def _row(**kwargs) -> Dict[str, Any]:
    r = _blank_row()
    for k, v in kwargs.items():
        if k in r:
            r[k] = v
    return r


def _load(path: Path) -> Optional[Any]:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _metrics_from_test_block(block: Dict[str, Any]) -> Dict[str, Any]:
    if not block:
        return {}
    out = {
        "AUROC": block.get("auroc"),
        "AUPRC": block.get("auprc"),
        "F1": block.get("f1") or block.get("f1_at_selected_threshold"),
        "F1_fixed": block.get("f1_at_0_5") or block.get("f1_at_threshold_0.5"),
        "precision": block.get("precision") or block.get("precision_at_selected_threshold"),
        "recall": block.get("recall") or block.get("recall_at_selected_threshold"),
        "precision_at_100": block.get("precision_at_100"),
        "recall_at_100": block.get("recall_at_100"),
        "lift_at_100": block.get("lift_at_100"),
        "precision_at_500": block.get("precision_at_500"),
        "recall_at_500": block.get("recall_at_500"),
        "lift_at_500": block.get("lift_at_500"),
        "precision_at_1000": block.get("precision_at_1000"),
        "recall_at_1000": block.get("recall_at_1000"),
        "lift_at_1000": block.get("lift_at_1000"),
        "dataset_positive_rate": block.get("positive_rate"),
        "paired_test_n": block.get("n"),
    }
    return out


def _ssl_proto(**overrides) -> Dict[str, Any]:
    base = {
        "objective": "contrastive",
        "encoder": "gin",
        "graph_flags": "reverse_mp,ego,ports,emlps,tds",
        "emlps": True, "tds": True, "reverse_mp": True, "ego": True, "ports": True,
        "contrastive_variant": "asymmetric_infonce", "asymmetric": True,
        "projection_head": True, "negative_count": 8192, "queue_size": 0,
        "probe_class_weight": "model (gin weights)", "probe_C": 1.0,
        "threshold_rule": "max_f1_on_val", "status": "evaluated",
        "scout_or_formal": "scout", "superseded": False,
        "thesis_role": "thesis_supporting",
        "duplicate_resolution": "not_duplicate",
        "caveats": "frozen linear probe; eval-mode test split; single-seed unless noted",
    }
    base.update(overrides)
    return base


def _parse_run_meta(run_name: str) -> Dict[str, Any]:
    meta = {"run_name": run_name}
    if "fnf" in run_name or "same_pair" in run_name:
        meta["fnf_rule"] = "same_pair"
    if "small_li" in run_name:
        meta["dataset"] = "Small-LI"
    elif "small_hi" in run_name or run_name.startswith(
        ("gin_", "hi_", "same_pair", "fnf", "gate", "pna", "rgcn")
    ):
        meta["dataset"] = "Small-HI"
    m = re.search(r"seed(\d+)", run_name)
    meta["seed"] = int(m.group(1)) if m else 1
    m = re.search(r"(\d+)ep", run_name)
    meta["training_epochs"] = int(m.group(1)) if m else 20
    if "pna" in run_name:
        meta["encoder"] = "pna"
    elif "gate" in run_name or "gat" in run_name:
        meta["encoder"] = "gat"
    elif "rgcn" in run_name:
        meta["encoder"] = "rgcn"
    return meta


def _run_name_from_embedding_dir(ed: str) -> str:
    if ed and "/" in ed:
        return ed.rstrip("/").split("/")[-1]
    return ed or ""


def _add_ssl_row(rows, source, source_note, run_name, stack, rep, test, **extra):
    meta = _parse_run_meta(run_name)
    rep_name = REP_MAP.get(rep, rep)
    dims = extra.pop("representation_dims", {}) or {}
    dim = dims.get(rep_name) or dims.get(rep)
    if rep_name == "post_embedding_128":
        dim = dim or 128
    elif rep_name == "pre_embedding_3h":
        dim = dim or 198
    metrics = _metrics_from_test_block(test)
    proto = _ssl_proto()
    for k, v in meta.items():
        if k in FIELDS:
            proto[k] = v
        elif k == "fnf_rule":
            proto["fnf_rule"] = v
    row_kw = dict(proto)
    rel_source = str(source.relative_to(ROOT))
    row_kw.update({
        "run_id": "{0}|{1}|{2}".format(run_name, STACK_MAP.get(stack, stack), rep_name),
        "dataset": extra.pop("dataset", meta.get("dataset")),
        "seed": extra.pop("seed", meta.get("seed", 1)),
        "training_epochs": extra.pop("training_epochs", meta.get("training_epochs", 20)),
        "selected_epoch": extra.pop("selected_epoch", None),
        "fnf_rule": extra.pop("fnf_rule", row_kw.get("fnf_rule") or meta.get("fnf_rule")),
        "encoder": extra.pop("encoder", meta.get("encoder", "gin")),
        "representation_source": rep_name,
        "representation_dim": dim,
        "probe_feature_stack": STACK_MAP.get(stack, stack),
        "source_json": rel_source,
        "source_note": source_note,
        "checkpoint_path": extra.pop("checkpoint_path", None),
        "scout_or_formal": extra.pop("scout_or_formal", row_kw.get("scout_or_formal")),
        "superseded": extra.pop("superseded", False),
        "status": extra.pop("status", row_kw.get("status")),
        "caveats": extra.pop("caveats", None) or row_kw.get("caveats"),
    })
    row_kw.update(extra)
    row_kw.update(metrics)
    rows.append(_row(**row_kw))


def ingest_alert_budget(rows, path):
    data = _load(path)
    if not data:
        return
    note = "notes/alert_budget_metrics_current_protocol.md"
    for item in data.get("rows", []):
        if item.get("status") != "completed":
            continue
        label = item.get("run_label", "")
        ds = "Small-LI" if "Small-LI" in label else "Small-HI"
        rn = item.get("run_name", "")
        meta = _parse_run_meta(rn)
        caveats = "alert-budget diagnostic; post_embedding_128 only"
        if meta.get("training_epochs") == 20 and item["feature_mode"] == "embedding":
            caveats += "; historical_20ep_baseline"
        _add_ssl_row(
            rows, path, note, rn, item["feature_mode"], "post_embedding_128",
            item.get("test", {}),
            dataset=ds, seed=meta.get("seed", 1),
            training_epochs=meta.get("training_epochs", 20),
            checkpoint_path=item.get("embedding_dir"),
            caveats=caveats,
        )


def ingest_feature_ablation_comparison(rows, path):
    data = _load(path)
    if not data:
        return
    note = "notes/probe_feature_ablation_current_protocol_comparison.md"
    for item in data.get("rows", []):
        label = item.get("run_label", "")
        seed = 2 if "seed2" in label else 1
        ep = 40 if "40ep" in label else 20
        fnf = "same_pair" if "FNF" in label else None
        run_name = _run_name_from_embedding_dir(item.get("embedding_dir", ""))
        if not run_name:
            run_name = label.lower().replace(" ", "_").replace("+", "_")
        test = {k: item.get(k) for k in (
            "auroc", "auprc", "f1", "precision", "recall", "f1_at_0_5"
        )}
        test["f1_at_threshold_0.5"] = test.pop("f1_at_0_5", None)
        caveats = "6-mode feature ablation; post_embedding_128; full test split (not paired pre/post intersection)"
        if ep == 20 and item["features"] == "embedding":
            caveats += "; historical_20ep_baseline"
        _add_ssl_row(
            rows, path, note, run_name, item["features"], "post_embedding_128", test,
            dataset="Small-HI", seed=seed, training_epochs=ep, fnf_rule=fnf,
            checkpoint_path=item.get("embedding_dir"),
            caveats=caveats,
        )


def ingest_architecture_sweep(rows, path):
    data = _load(path)
    if not data:
        return
    for item in data.get("runs", []):
        proto = _ssl_proto(encoder=item["encoder"])
        proto["caveats"] = (
            "embedding-only; PNA/RGCN not capacity-matched; historical comparison"
        )
        if item["encoder"] == "gin":
            proto["caveats"] += "; historical_20ep_baseline"
        rows.append(_row(
            **proto,
            run_id="{0}|embedding|post_embedding_128".format(item["unique_name"]),
            dataset="Small-HI", seed=1, training_epochs=20,
            representation_source="post_embedding_128", representation_dim=128,
            probe_feature_stack="embedding",
            source_json=str(path.relative_to(ROOT)),
            source_note="results/diagnostics/architecture_sweep_shared_probe_weights.md",
            AUROC=item.get("test_auroc"), AUPRC=item.get("test_auprc"),
            F1=item.get("test_f1"), F1_fixed=item.get("test_f1_at_0.5"),
            precision=item.get("test_precision"), recall=item.get("test_recall"),
        ))


def ingest_pre3h_strong(rows, path):
    data = _load(path)
    if not data:
        return
    note = "notes/pre3h_strong_run_comparison.md"
    stack_map = {
        "embedding_only": "embedding",
        "embedding_plus_raw": "embedding+raw",
        "embedding_plus_raw_morph": "embedding+raw+morph",
    }
    for run in data.get("runs", []):
        rn = run["run_name"]
        for mode, comp in run.get("modes", {}).items():
            stack = stack_map.get(mode, mode)
            for rep_key, rep_name in (
                ("post", "post_embedding_128"), ("pre", "pre_embedding_3h")
            ):
                block = comp.get(rep_key)
                if not block:
                    continue
                caveats = (
                    "strong-run paired pre/post batch; canonical for pre-vs-post tables; "
                    "paired edge-ID inner-join per split"
                )
                _add_ssl_row(
                    rows, path, note, rn, stack, rep_name, block,
                    dataset=run.get("data"),
                    selected_epoch=run.get("checkpoint_epoch"),
                    training_epochs=_parse_run_meta(rn).get("training_epochs", 20),
                    representation_dims=run.get("representation_dims"),
                    caveats=caveats,
                )


def ingest_prepost_json(rows, path, note, skip_seeds=None):
    data = _load(path)
    if not data:
        return
    skip_seeds = skip_seeds or set()
    if "per_seed" in data:
        for ps in data["per_seed"]:
            if ps.get("seed") in skip_seeds:
                continue
            for mode, comp in ps.get("modes", {}).items():
                stack = STACK_MAP.get(mode, mode)
                for rep_key, rep_name in (
                    ("post", "post_embedding_128"), ("pre", "pre_embedding_3h")
                ):
                    block = comp.get(rep_key)
                    if block:
                        _add_ssl_row(
                            rows, path, note, ps["run_name"], stack, rep_name, block,
                            dataset=data.get("data", "Small-LI"), seed=ps.get("seed", 1),
                            training_epochs=20,
                            representation_dims=ps.get("representation_dims"),
                            caveats="paired pre/post probe; no SSL retraining",
                        )
        return
    comparisons = data.get("comparisons") or {}
    run_name = data.get("run_name", path.stem)
    ds = data.get("data", "Small-HI")
    dims = data.get("representation_dims") or {}
    seed = data.get("probe", {}).get("seed", _parse_run_meta(run_name).get("seed", 1))
    if seed in skip_seeds:
        return
    for mode, comp in comparisons.items():
        stack = STACK_MAP.get(mode, mode.replace("embedding_plus_", "embedding+"))
        reps = comp.get("representations") or {}
        if reps:
            for rep_name, rep_block in reps.items():
                test = rep_block.get("test") or rep_block
                _add_ssl_row(
                    rows, path, note, run_name, stack, rep_name, test,
                    dataset=ds, seed=seed, representation_dims=dims,
                    caveats="paired pre/post probe",
                )
        else:
            for rep_key, rep_name in (
                ("post", "post_embedding_128"), ("pre", "pre_embedding_3h")
            ):
                block = comp.get(rep_key)
                if block:
                    _add_ssl_row(
                        rows, path, note, run_name, stack, rep_name, block,
                        dataset=ds, seed=seed, representation_dims=dims,
                        caveats="paired pre/post probe",
                    )


def _temporal_flow_validation_state() -> Dict[str, Any]:
    val_path = DIAG / "temporal_flow_causal_validation_summary.json"
    val = _load(val_path)
    if not val:
        return {
            "passed": False,
            "summary_exists": False,
            "safe_to_cite": None,
            "canonical_probe_max_iter": None,
        }
    safe = (val.get("answers") or {}).get("6_safe_to_cite", "")
    passed = str(safe).startswith("Yes")
    return {
        "passed": passed,
        "summary_exists": True,
        "safe_to_cite": safe,
        "canonical_probe_max_iter": 5000 if passed else None,
    }


def _temporal_probe_max_iter(data: Dict[str, Any]) -> int:
    probe = data.get("probe") or {}
    return int(probe.get("probe_max_iter") or probe.get("max_iter") or 1000)


def _temporal_arm_test_block(arm: Dict[str, Any]) -> Dict[str, Any]:
    if arm.get("test"):
        return arm["test"]
    if arm.get("auroc") is not None:
        return arm
    return {}


def ingest_temporal_flow_ablation(rows, path: Path, source_note: Optional[str] = None) -> bool:
    data = _load(path)
    if not data or "arms" not in data:
        return False
    rel = str(path.relative_to(ROOT))
    note = source_note or "notes/temporal_flow_causal_ablation_summary.md"
    max_iter = _temporal_probe_max_iter(data)
    ds = data.get("data", "Small-HI")
    rn = data["run_name"]
    meta = _parse_run_meta(rn)
    test_n = (data.get("split_pairing") or {}).get("test", {}).get("rows")
    rep_dim = data.get("representation_dim") or 198
    for arm_key, (arm_label, stack) in TEMPORAL_ARM_STACK.items():
        arm = data["arms"].get(arm_key)
        if not arm:
            continue
        test = _temporal_arm_test_block(arm)
        if not test:
            continue
        metrics = _metrics_from_test_block(test)
        if test_n and metrics.get("paired_test_n") is None:
            metrics["paired_test_n"] = test_n
        proto = _ssl_proto(
            thesis_role="thesis_supporting",
            table_group="temporal_flow_ablation",
        )
        proto.pop("caveats", None)
        for k, v in meta.items():
            if k in FIELDS and k not in ("dataset", "seed"):
                proto[k] = v
        rows.append(_row(
            **proto,
            **metrics,
            run_id="{0}|{1}|pre_embedding_3h|tf_arm{2}|maxiter{3}".format(
                rn, stack, arm_label, max_iter
            ),
            dataset=ds,
            seed=meta.get("seed", 1),
            representation_source="pre_embedding_3h",
            representation_dim=rep_dim,
            probe_feature_stack=stack,
            source_json=rel,
            source_note=note,
            checkpoint_path=data.get("embedding_dir"),
            caveats=(
                "temporal_flow_causal ablation arm {0}; probe_max_iter={1}; "
                "paired edge_id join; frozen pre-3h SSL"
            ).format(arm_label, max_iter),
        ))
    return True


def _parse_tf_aux_variant(run_name: str) -> str:
    m = re.search(r"hi_tf_aux_(tf_(?:reg|bins)[^_]*(?:_w[\d.]+)?)_gin", run_name)
    if m:
        return m.group(1)
    m = re.search(r"tf_aux_(tf_(?:reg|bins)[\w.]+)_", run_name)
    return m.group(1) if m else "unknown"


def _tf_aux_selected_epochs() -> Dict[str, Optional[int]]:
    scout = _load(DIAG / "temporal_flow_aux_objective_scout.json") or {}
    out: Dict[str, Optional[int]] = {}
    for v in scout.get("variants") or []:
        rn = v.get("run_name")
        if rn:
            out[rn] = v.get("checkpoint_epoch")
    return out


def ingest_temporal_flow_aux_probe(
    rows,
    path: Path,
    *,
    selected_epochs: Optional[Dict[str, Optional[int]]] = None,
    table_group: str = "temporal_flow_aux_objective",
    source_note: str = "notes/temporal_flow_aux_objective_scout.md",
    thesis_role: str = "diagnostic",
    caveats_extra: str = "single-seed",
) -> bool:
    """Ingest SSL temporal-flow auxiliary objective scout probe JSONs (post-128 and pre-3h)."""
    data = _load(path)
    if not data or "arms" not in data:
        return False
    if "tf_aux" not in path.name and "hi_tf_aux" not in str(data.get("run_name", "")):
        return False
    rel = str(path.relative_to(ROOT))
    note = source_note
    max_iter = _temporal_probe_max_iter(data)
    ds = data.get("data", "Small-HI")
    rn = data["run_name"]
    meta = _parse_run_meta(rn)
    variant = _parse_tf_aux_variant(rn)
    selected = None
    if selected_epochs:
        selected = selected_epochs.get(rn)
    if selected is None:
        selected = (data.get("extraction_meta") or {}).get("checkpoint_epoch")
    raw_rep = data.get("representation") or data.get("representation_source") or "pre_embedding_3h"
    if "post" in str(raw_rep):
        rep_name = "post_embedding_128"
        default_dim = 128
    else:
        rep_name = "pre_embedding_3h"
        default_dim = 198
    rep_dim = data.get("representation_dim") or default_dim
    test_n = (data.get("split_pairing") or {}).get("test", {}).get("rows")
    for arm_key, (arm_label, stack) in TEMPORAL_ARM_STACK.items():
        arm = data["arms"].get(arm_key)
        if not arm:
            continue
        test = _temporal_arm_test_block(arm)
        if not test:
            continue
        metrics = _metrics_from_test_block(test)
        if test_n and metrics.get("paired_test_n") is None:
            metrics["paired_test_n"] = test_n
        proto = _ssl_proto(
            thesis_role=thesis_role,
            scout_or_formal="scout",
            table_group=table_group,
            validation_status="diagnostic_only",
            table_eligible=False,
            morphology_auxiliary_loss="temporal_flow_causal:{0}".format(variant),
        )
        proto.pop("caveats", None)
        for k, v in meta.items():
            if k in FIELDS and k not in ("dataset", "seed"):
                proto[k] = v
        rows.append(_row(
            **proto,
            **metrics,
            run_id="{0}|{1}|{2}|tf_aux_{3}|arm{4}|maxiter{5}".format(
                rn, stack, rep_name, variant, arm_label, max_iter
            ),
            dataset=ds,
            seed=meta.get("seed", 1),
            selected_epoch=selected,
            representation_source=rep_name,
            representation_dim=rep_dim,
            probe_feature_stack=stack,
            source_json=rel,
            source_note=note,
            checkpoint_path=data.get("embedding_dir"),
            caveats=(
                "temporal_flow_aux SSL scout variant={0}; attach=post_embedding_head_pre_projection; "
                "no labels in SSL; arm {1}; probe_max_iter={2}; {3}; "
                "primary evidence uses A/B not D-only; table_eligible=false"
            ).format(variant, arm_label, max_iter, caveats_extra),
        ))
    return True


def _parse_tf_soft_variant(run_name: str) -> str:
    m = re.search(r"hi_tf_soft_(tf_soft_[\w.]+?)_optv2_gin", run_name)
    if m:
        return m.group(1)
    m = re.search(r"tf_soft_(tf_soft_[\w.]+?)_(?:pre3h|post128)", run_name)
    return m.group(1) if m else "unknown"


def ingest_temporal_flow_soft_positive_probe(rows, path: Path) -> bool:
    """Ingest SSL temporal-flow soft-positive scout probe JSONs (post-128 + pre-3h).

    Diagnostic/negative-result scout: identity pair remains the primary positive;
    low-weight temporal-flow soft positives are added. No labels are used in SSL.
    """
    data = _load(path)
    if not data or "arms" not in data:
        return False
    if "tf_soft" not in path.name and "hi_tf_soft" not in str(data.get("run_name", "")):
        return False
    rel = str(path.relative_to(ROOT))
    note = "notes/temporal_flow_soft_positive_scout.md"
    max_iter = _temporal_probe_max_iter(data)
    ds = data.get("data", "Small-HI")
    rn = data["run_name"]
    meta = _parse_run_meta(rn)
    variant = _parse_tf_soft_variant(rn)
    raw_rep = data.get("representation") or data.get("representation_source") or "pre_embedding_3h"
    if "post" in str(raw_rep):
        rep_name = "post_embedding_128"
        default_dim = 128
    else:
        rep_name = "pre_embedding_3h"
        default_dim = 198
    rep_dim = data.get("representation_dim") or default_dim
    test_n = (data.get("split_pairing") or {}).get("test", {}).get("rows")
    for arm_key, (arm_label, stack) in TEMPORAL_ARM_STACK.items():
        arm = data["arms"].get(arm_key)
        if not arm:
            continue
        test = _temporal_arm_test_block(arm)
        if not test:
            continue
        metrics = _metrics_from_test_block(test)
        if test_n and metrics.get("paired_test_n") is None:
            metrics["paired_test_n"] = test_n
        proto = _ssl_proto(
            thesis_role="negative_result",
            scout_or_formal="scout",
            table_group="temporal_flow_soft_positive_objective",
            validation_status="diagnostic_only",
            table_eligible=False,
            morphology_auxiliary_loss="temporal_flow_soft_positive:{0}".format(variant),
        )
        proto.pop("caveats", None)
        for k, v in meta.items():
            if k in FIELDS and k not in ("dataset", "seed"):
                proto[k] = v
        rows.append(_row(
            **proto,
            **metrics,
            run_id="{0}|{1}|{2}|tf_soft_{3}|arm{4}|maxiter{5}".format(
                rn, stack, rep_name, variant, arm_label, max_iter
            ),
            dataset=ds,
            seed=meta.get("seed", 1),
            selected_epoch=None,
            representation_source=rep_name,
            representation_dim=rep_dim,
            probe_feature_stack=stack,
            source_json=rel,
            source_note=note,
            checkpoint_path=data.get("embedding_dir"),
            caveats=(
                "temporal_flow soft-positive SSL scout variant={0}; identity pair primary, "
                "low-weight TF soft positives; no labels in SSL; arm {1}; probe_max_iter={2}; "
                "single-seed; NEGATIVE RESULT: pre-3h A/B AUPRC below baseline (~0.224/0.274), "
                "P@100 and precision-constrained recall collapse; caps fully saturated"
            ).format(variant, arm_label, max_iter),
        ))
    return True


def _parse_morph_obj_variant(run_name: str, path_name: str) -> str:
    m = re.search(r"morph_obj_(baseline|degflow_tfreg|degflow|clustering)_", path_name)
    if m:
        return m.group(1)
    if "hi_morph_obj_degflow_tfreg" in run_name:
        return "degflow_tfreg"
    if "hi_morph_obj_degflow" in run_name:
        return "degflow"
    if "hi_morph_obj_clustering" in run_name:
        return "clustering"
    if "hi_contrastive_gin_emlps_tds" in run_name and "proj_asym_8192neg" in run_name:
        return "baseline"
    return "unknown"


def ingest_morphology_objective_recall_probe(
    rows,
    path: Path,
    *,
    table_group: str = "morphology_objective_recall_scout",
    source_note: str = "notes/morphology_objective_recall_scout.md",
) -> bool:
    """Ingest morphology-objective recall scout probe JSONs (diagnostic_only).

    Regression expert heads only (no M2 contrast, no TF soft positives, no tier2).
    No labels in SSL. Not table-eligible for main thesis tables.
    """
    data = _load(path)
    if not data or "arms" not in data:
        return False
    if "morph_obj" not in path.name and "hi_morph_obj" not in str(data.get("run_name", "")):
        # baseline re-probe uses plain contrastive run name; require morph_obj path prefix
        if "morph_obj_baseline" not in path.name:
            return False
    rel = str(path.relative_to(ROOT))
    note = source_note
    max_iter = _temporal_probe_max_iter(data)
    ds = data.get("data", "Small-HI")
    rn = data["run_name"]
    meta = _parse_run_meta(rn)
    variant = _parse_morph_obj_variant(rn, path.name)
    raw_rep = data.get("representation") or data.get("representation_source") or "pre_embedding_3h"
    if "post" in str(raw_rep):
        rep_name = "post_embedding_128"
        default_dim = 128
    else:
        rep_name = "pre_embedding_3h"
        default_dim = 198
    rep_dim = data.get("representation_dim") or default_dim
    test_n = (data.get("split_pairing") or {}).get("test", {}).get("rows")
    aux_label = {
        "baseline": "none",
        "degflow": "morph_expert:degree_fan+flow_balance",
        "clustering": "morph_expert:local+global_clustering",
        "degflow_tfreg": "morph_expert:degree_fan+flow_balance+tf_reg_w0.05",
    }.get(variant, "morph_expert:{0}".format(variant))
    # Prefer seed from run_name / filename over default
    seed_from_name = meta.get("seed")
    m_seed = re.search(r"_seed(\d+)\.json$", path.name)
    if m_seed:
        seed_from_name = int(m_seed.group(1))
    elif variant == "baseline" and seed_from_name is None:
        seed_from_name = 1
    for arm_key, (arm_label, stack) in TEMPORAL_ARM_STACK.items():
        arm = data["arms"].get(arm_key)
        if not arm:
            continue
        test = _temporal_arm_test_block(arm)
        if not test:
            continue
        metrics = _metrics_from_test_block(test)
        if test_n and metrics.get("paired_test_n") is None:
            metrics["paired_test_n"] = test_n
        proto = _ssl_proto(
            thesis_role="diagnostic_or_scout",
            scout_or_formal="scout",
            table_group=table_group,
            validation_status="diagnostic_only",
            table_eligible=False,
            morphology_auxiliary_loss=aux_label,
        )
        proto.pop("caveats", None)
        for k, v in meta.items():
            if k in FIELDS and k not in ("dataset", "seed"):
                proto[k] = v
        rows.append(_row(
            **proto,
            **metrics,
            run_id="{0}|{1}|{2}|morph_obj_{3}|arm{4}|maxiter{5}".format(
                rn, stack, rep_name, variant, arm_label, max_iter
            ),
            dataset=ds,
            seed=seed_from_name if seed_from_name is not None else 1,
            selected_epoch=None,
            representation_source=rep_name,
            representation_dim=rep_dim,
            probe_feature_stack=stack,
            source_json=rel,
            source_note=note,
            checkpoint_path=data.get("embedding_dir"),
            caveats=(
                "morphology-objective scout variant={0}; expert regression only; "
                "no M2/soft-positives/tier2; no labels in SSL; arm {1}; probe_max_iter={2}; "
                "table_group={3}; diagnostic_only / not table-eligible"
            ).format(variant, arm_label, max_iter, table_group),
        ))
    return True


def ingest_pna_width_aligned_probe(rows, path: Path) -> bool:
    data = _load(path)
    if not data:
        return False
    post = (data.get("embedding_only") or {}).get("post_embedding_128") or {}
    pre = (data.get("embedding_only") or {}).get("pre_embedding_3h") or {}
    run_name = data.get("run_name") or "pna_width65_scout"
    caveats = "; ".join(data.get("caveats") or [])
    rows.append(_row(
        run_id="{0}|embedding|post_embedding_128".format(run_name),
        dataset=data.get("data", "Small-HI"),
        seed=int(data.get("seed", 1)),
        training_epochs=20,
        encoder="pna",
        representation_source="post_embedding_128",
        representation_dim=int(data.get("post_dim", 128)),
        probe_feature_stack="embedding",
        source_json=str(path.relative_to(ROOT)),
        source_note="notes/pna_ssl_fairness_followup.md",
        AUROC=post.get("auroc"),
        AUPRC=post.get("auprc"),
        F1=post.get("f1_at_selected_threshold"),
        precision=post.get("precision_at_selected_threshold"),
        recall=post.get("recall_at_selected_threshold"),
        P_at_100=post.get("precision_at_100"),
        R_at_100=post.get("recall_at_100"),
        lift_at_100=post.get("lift_at_100"),
        status="evaluated",
        thesis_role="thesis_supporting",
        table_group="architecture_ablation",
        caveats=caveats or "PNA width-aligned scout; seed 1",
        scout_or_formal="scout",
    ))
    rows.append(_row(
        run_id="{0}|embedding|pre_embedding_3h".format(run_name),
        dataset=data.get("data", "Small-HI"),
        seed=int(data.get("seed", 1)),
        training_epochs=20,
        encoder="pna",
        representation_source="pre_embedding_3h",
        representation_dim=int(data.get("pre_dim", 195)),
        probe_feature_stack="embedding",
        source_json=str(path.relative_to(ROOT)),
        source_note="notes/pna_ssl_fairness_followup.md",
        AUROC=pre.get("auroc"),
        AUPRC=pre.get("auprc"),
        F1=pre.get("f1_at_selected_threshold"),
        precision=pre.get("precision_at_selected_threshold"),
        recall=pre.get("recall_at_selected_threshold"),
        P_at_100=pre.get("precision_at_100"),
        R_at_100=pre.get("recall_at_100"),
        lift_at_100=pre.get("lift_at_100"),
        status="evaluated",
        thesis_role="thesis_supporting",
        table_group="architecture_ablation",
        caveats=caveats or "PNA width-aligned scout; seed 1",
        scout_or_formal="scout",
    ))
    return True


def ingest_pna_width_audit(rows, path: Path) -> bool:
    data = _load(path)
    if not data or "pna_width_sweep" not in data:
        return False
    rel = str(path.relative_to(ROOT))
    gin_ref = data.get("gin_reference") or {}
    rows.append(_row(
        run_id="pna_width_param_audit|diagnostic|none",
        dataset="Small-HI",
        objective="diagnostic",
        encoder="pna",
        status="audit_only",
        thesis_role="diagnostic",
        validation_status="diagnostic_only",
        table_eligible=False,
        table_group="architecture_ablation",
        representation_dim=gin_ref.get("pre_embedding_3h_dim"),
        source_json=rel,
        source_note="results/diagnostics/pna_width_param_audit.json",
        caveats="PNA width sweep audit only — no probe metrics; capacity-matched PNA probe pending",
    ))
    return True


def compute_temporal_flow_multiseed_aggregates() -> Dict[str, Any]:
    summary = _load(DIAG / "temporal_flow_causal_ablation_summary.json")
    if not summary:
        return {}
    li = summary.get("small_li_multiseed") or {}
    out: Dict[str, Any] = {
        "source_json": "results/diagnostics/temporal_flow_causal_ablation_summary.json",
        "n_seeds": li.get("D_minus_B", {}).get("n", 3),
        "std_convention": "sample",
        "std_ddof": MULTISEED_STD_DDOF,
        "arms": {},
    }
    multiseed = _load(DIAG / "temporal_flow_ablation_small_li_multiseed.json")
    if not multiseed:
        return out
    for arm_key, (arm_label, stack) in TEMPORAL_ARM_STACK.items():
        if arm_label not in ("B", "D"):
            continue
        metrics: Dict[str, Dict[str, Any]] = defaultdict(list)
        for ps in multiseed.get("per_seed", []):
            arm = (ps.get("arms") or {}).get(arm_key) or {}
            test = _temporal_arm_test_block(arm)
            for m in (
                "auroc", "auprc", "f1_at_selected_threshold",
                "precision_at_100", "recall_at_100", "lift_at_100",
                "precision_at_500", "recall_at_500", "lift_at_500",
                "precision_at_1000", "recall_at_1000", "lift_at_1000",
            ):
                if test.get(m) is not None:
                    metrics[m].append(float(test[m]))
        arm_out: Dict[str, Any] = {"arm": arm_label, "stack": stack, "metrics": {}}
        for m, vals in metrics.items():
            arm_out["metrics"][m] = {
                "mean": stats.mean(vals),
                "std": _sample_std(vals),
                "per_seed": vals,
            }
        out["arms"][arm_label] = arm_out
    dmb = li.get("D_minus_B") or {}
    if dmb.get("metrics"):
        out["D_minus_B"] = dmb
    return out


def build_dataset_metadata() -> Dict[str, Any]:
    hi_pair = _load(DIAG / "pre_embedding_3h_vs_post_embedding_small_hi.json")
    li_audit = _load(DIAG / "small_li_dataset_audit.json")
    rows = []
    if hi_pair:
        sp = hi_pair.get("split_pairing") or {}
        for split in ("train", "val", "test"):
            b = sp.get(split) or {}
            rows.append({
                "dataset": "Small-HI",
                "split": split,
                "n_transactions": b.get("joined_rows") or b.get("post_rows"),
                "n_positives": b.get("positives"),
                "positive_rate": b.get("positive_rate"),
                "task": "edge-level AML detection",
                "source_json": "results/diagnostics/pre_embedding_3h_vs_post_embedding_small_hi.json",
            })
    if li_audit:
        for split, b in (li_audit.get("splits") or {}).items():
            rows.append({
                "dataset": "Small-LI",
                "split": split,
                "n_transactions": b.get("n_edges"),
                "n_positives": b.get("n_positive"),
                "positive_rate": b.get("positive_rate"),
                "task": "edge-level AML detection",
                "source_json": "results/diagnostics/small_li_dataset_audit.json",
            })
    return {"rows": rows}


def assign_table_metadata(rows: List[Dict[str, Any]], tf_validation: Dict[str, Any]) -> None:
    canonical_maxiter = tf_validation.get("canonical_probe_max_iter")
    tf_passed = tf_validation.get("passed", False)
    for r in rows:
        src = str(r.get("source_json", ""))
        is_temporal = "temporal_flow_ablation" in src
        maxiter_match = re.search(r"maxiter(\d+)", r.get("run_id") or "")
        probe_max_iter = int(maxiter_match.group(1)) if maxiter_match else None

        if r.get("validation_status") is None:
            if r.get("superseded"):
                r["validation_status"] = "superseded"
            elif r.get("thesis_role") == "diagnostic" or r.get("status") == "audit_only":
                r["validation_status"] = "diagnostic_only"
            elif is_temporal:
                if tf_passed and probe_max_iter == 5000:
                    r["validation_status"] = "validated"
                elif probe_max_iter == 5000:
                    r["validation_status"] = "pending_validation"
                else:
                    r["validation_status"] = "provisional"
            else:
                r["validation_status"] = "validated"

        if is_temporal and tf_passed and probe_max_iter == 1000:
            r["validation_status"] = "superseded"
            r["superseded"] = True
            r["caveats"] = (r.get("caveats") or "") + (
                "; superseded by validated maxiter5000 temporal-flow probe"
            )

        if r.get("table_eligible") is None:
            r["table_eligible"] = (
                r.get("validation_status") == "validated"
                and not r.get("superseded")
                and r.get("thesis_role") not in (
                    "diagnostic",
                    "diagnostic_or_scout",
                    "negative_result",
                    "superseded",
                )
                and r.get("status") not in ("audit_only", "failed", "preliminary")
                and r.get("split_protocol") != "random_40"
            )
        # Hard guards
        if r.get("split_protocol") == "random_40" or r.get("status") in (
            "failed", "audit_only"
        ):
            r["table_eligible"] = False
        if r.get("paper_comparable") is False and r.get("protocol_family") == (
            "supervised_legacy_tds_on_not_paper"
        ):
            r["table_eligible"] = False
        if r.get("objective") == "supervised" and r.get("tds") is True and (
            r.get("dataset") == "Small-HI"
        ):
            r["paper_comparable"] = False


        if r.get("table_group") is None:
            if is_temporal:
                r["table_group"] = "temporal_flow_ablation"
            elif r.get("objective") == "supervised":
                r["table_group"] = "supervised_vs_ssl"
            elif "architecture_sweep" in src:
                r["table_group"] = "architecture_ablation"
            elif "degree_aware" in src or "same_pair_fnf" in src or (
                "probe_feature_ablation" in src and "current_protocol" not in src
            ):
                r["table_group"] = "contrastive_ablations"
            elif src == CANONICAL_PREPOST_SOURCE or src == MULTISEED_LI_SOURCE:
                r["table_group"] = "representation_readout_ablation"
            elif r.get("dataset") == "Small-HI" and r.get("thesis_role") == "thesis_primary":
                r["table_group"] = "main_results_small_hi"
            elif r.get("dataset") == "Small-LI" and r.get("thesis_role") == "thesis_primary":
                r["table_group"] = "main_results_small_li"
            elif "alert_budget" in src or "probe_feature_ablation_current_protocol" in src:
                r["table_group"] = "main_results_small_hi" if r.get("dataset") == "Small-HI" else "main_results_small_li"
            elif EMB198_SOURCE in src:
                r["table_group"] = "representation_readout_ablation"
            elif "pna" in src:
                r["table_group"] = "architecture_ablation"


def collect_pending_sources() -> List[str]:
    pending = []
    for rel in TEMPORAL_FLOW_SOURCES + PNA_FOLLOWUP_SOURCES + TF_AUX_OBJECTIVE_SOURCES + TF_SOFT_OBJECTIVE_SOURCES:
        p = ROOT / rel
        if not p.is_file():
            pending.append(rel)
    return pending


def ingest_emb198(rows, path):
    data = _load(path)
    if not data:
        return
    note = "notes/small_li_embedding_dim_128_vs_198.md"
    rep_labels = {
        "orig_post_128": ("post_embedding_128", 128),
        "orig_pre_3h_198": ("pre_embedding_3h", 198),
        "new_post_198": ("post_embedding_198_scout", 198),
        "new_pre_3h_198": ("pre_embedding_3h_emb198_scout", 198),
    }
    rn = "small_li_gin_emlps_tds_asym_proj_8192neg_queue0_20ep_seed1"
    for mode, block in data.get("blocks", {}).items():
        stack = STACK_MAP.get(mode, mode)
        for rep_key, (rep_name, dim) in rep_labels.items():
            rep_block = (block.get("representations") or {}).get(rep_key)
            if not rep_block:
                continue
            test = rep_block.get("test") or rep_block
            scout = "new_" in rep_key
            c = "emb198 scout"
            if scout:
                c += "; retrained SSL export head; optional/not currently planned for multiseed"
            else:
                c += (
                    "; orig 128-d checkpoint; emb198 paired-intersection join "
                    "(AUPRC +raw pre=0.0829 — use multiseed 0.0818 for three-seed aggregate)"
                )
            _add_ssl_row(
                rows, path, note, rn + ("_emb198" if scout else ""), stack, rep_name, test,
                dataset="Small-LI", seed=1, training_epochs=20,
                representation_dim=dim, scout_or_formal="scout",
                thesis_role="diagnostic", caveats=c,
            )


def ingest_ablation_runs(rows, path, tag, fnf):
    data = _load(path)
    if not data:
        return
    note_path = ROOT / "notes" / "probe_feature_ablation_{0}.md".format(tag)
    note = str(note_path.relative_to(ROOT)) if note_path.exists() else str(path.name)
    role = "negative_result" if tag == "degree_aware" else "thesis_supporting"
    for item in data.get("runs", []):
        test = item.get("splits_at_selected_threshold", {}).get("test") or item.get("test", {})
        if not test and "auroc" in item:
            test = item
        ed = item.get("embedding_dir") or data.get("embedding_dir", tag)
        run_name = _run_name_from_embedding_dir(str(ed))
        _add_ssl_row(
            rows, path, note, run_name,
            item.get("features", "embedding"), "post_embedding_128", test,
            dataset=data.get("data", "Small-HI"), fnf_rule=fnf,
            thesis_role=role,
            caveats="feature ablation ({0}); post_embedding_128; full test split".format(tag),
        )


def ingest_legacy(rows, summary_path, eval_path, superseded=False, **overrides):
    summary = _load(summary_path)
    ev = _load(eval_path) or {}
    if not summary:
        return
    dataset = summary.get("data") or ev.get("data") or "Small-LI"
    test = ev.get("splits", {}).get("test", {})
    alert = test.get("alert_budget", {})
    paper = test.get("paper_argmax", {})
    val_tuned = test.get("validation_tuned_threshold", {})
    caveats_extra = ""
    if dataset == "Small-HI":
        caveats_extra = (
            "; Small-HI legacy supervised reference; compare to SSL temporal-flow with protocol caveats"
        )
    tds = overrides.get("tds", True)
    paper_comparable = overrides.get(
        "paper_comparable",
        bool(dataset == "Small-HI" and not tds and not superseded),
    )
    graph_flags = overrides.get(
        "graph_flags",
        "emlps,reverse_mp,ports,ego" + (",tds" if tds else ""),
    )
    protocol_family = overrides.get(
        "protocol_family",
        "supervised_multigin_eu_ports_no_tds" if (dataset == "Small-HI" and not tds)
        else "supervised_legacy",
    )
    # Always use eval JSON for ranking metrics when available
    base = dict(
        run_id="{0}|supervised|paper_argmax".format(summary["run_name"]),
        dataset=dataset,
        dataset_positive_rate=test.get("positive_rate"),
        objective="supervised", encoder="gin", seed=summary.get("seed", 1),
        training_epochs=summary.get("n_epochs", 100),
        selected_epoch=summary.get("best_validation_epoch"),
        checkpoint_policy="best_val_minority_f1", supervised_head="legacy",
        graph_flags=graph_flags,
        emlps=True, tds=tds, reverse_mp=True, ego=True, ports=True,
        representation_source="logits_direct", probe_feature_stack="in_gnn_end_to_end",
        threshold_rule="paper_argmax",
        AUROC=test.get("auroc"),
        AUPRC=test.get("auprc"),
        F1=paper.get("f1") or summary.get("test_minority_f1_argmax_at_best"),
        precision=paper.get("precision"), recall=paper.get("recall"),
        precision_at_100=alert.get("precision_at_100"),
        recall_at_100=alert.get("recall_at_100"), lift_at_100=alert.get("lift_at_100"),
        precision_at_500=alert.get("precision_at_500"),
        recall_at_500=alert.get("recall_at_500"), lift_at_500=alert.get("lift_at_500"),
        precision_at_1000=alert.get("precision_at_1000"),
        recall_at_1000=alert.get("recall_at_1000"), lift_at_1000=alert.get("lift_at_1000"),
        source_json=str(eval_path.relative_to(ROOT)),
        source_note=str(summary_path.relative_to(ROOT)).replace(".json", ".md"),
        checkpoint_path=summary.get("best_val_checkpoint_path"),
        status="evaluated",
        scout_or_formal="formal" if not superseded else "scout",
        superseded=superseded,
        thesis_role="superseded" if superseded else "thesis_primary",
        table_group="main_results" if dataset == "Small-HI" and not superseded else None,
        protocol_family=protocol_family,
        split_protocol="temporal",
        graph_representation="account_nodes_transaction_edges",
        reverse_feature_semantics=overrides.get(
            "reverse_feature_semantics",
            "inherited_trailing_swap" if tds else "ports_only_inherited",
        ),
        paper_comparable=paper_comparable,
        caveats=(
            "paper_argmax in-GNN; NOT frozen probe; best-val checkpoint ep{0} only"
            + caveats_extra
        ).format(summary.get("best_validation_epoch", "?"))
        + ("; SUPERSEDED — not paper-compatible (TDS-on)" if superseded and dataset == "Small-HI" and tds else "")
        + ("; SUPERSEDED by 100ep formal" if superseded and dataset == "Small-LI" else ""),
    )
    for k, v in overrides.items():
        if k in FIELDS:
            base[k] = v
    rows.append(_row(**base))
    if val_tuned.get("f1") is not None:
        rows.append(_row(
            **dict(base,
                   run_id="{0}|supervised|val_tuned_threshold".format(summary["run_name"]),
                   threshold_rule="max_f1_on_val",
                   F1=val_tuned.get("f1"), precision=val_tuned.get("precision"),
                   recall=val_tuned.get("recall"),
                   thesis_role="diagnostic",
                   paper_comparable=False,
                   table_eligible=False,
                   caveats="val-tuned threshold; NOT paper-compatible; diagnostic only")))


def ingest_ports_hi_aggregate(rows, path: Path):
    data = _load(path)
    if not data:
        return
    agg = (data.get("formal_test_aggregate") or {})
    f1 = agg.get("paper_argmax_f1") or {}
    rows.append(_row(
        run_id="small_hi_legacy_supervised_gin_emlps_ports_50ep_seeds1-3|supervised|paper_argmax|aggregate",
        dataset="Small-HI",
        objective="supervised", encoder="gin", seed=None,
        training_epochs=50, supervised_head="legacy",
        graph_flags="emlps,reverse_mp,ports,ego",
        emlps=True, tds=False, reverse_mp=True, ego=True, ports=True,
        representation_source="logits_direct", probe_feature_stack="in_gnn_end_to_end",
        threshold_rule="paper_argmax",
        AUROC=(agg.get("auroc") or {}).get("mean"),
        AUPRC=(agg.get("auprc") or {}).get("mean"),
        F1=f1.get("mean"),
        precision=(agg.get("precision") or {}).get("mean"),
        recall=(agg.get("recall") or {}).get("mean"),
        source_json=str(path.relative_to(ROOT)),
        source_note="notes/eval_small_hi_legacy_supervised_gin_emlps_ports_50ep_seeds1-3_formal_aggregate.md",
        status="evaluated", scout_or_formal="formal", superseded=False,
        thesis_role="thesis_primary", validation_status="validated",
        table_eligible=True, table_group="main_results",
        protocol_family="supervised_multigin_eu_ports_no_tds",
        split_protocol="temporal",
        graph_representation="account_nodes_transaction_edges",
        reverse_feature_semantics="ports_only_inherited",
        paper_comparable=True,
        caveats=(
            "formal aggregate seeds 1-3 paper_argmax; mean F1 0.660±0.060 vs paper "
            "0.6479±0.0122; mean reproduced, low variance not; F1 field is mean only"
        ),
    ))


def ingest_jul22_probe_arm(
    rows, rel, tag, tds, reverse_sem, preserve, arm_id, *,
    batch_size=None, accum=None, thesis_role="diagnostic", table_eligible=False,
    protocol_family="edge_centric_contrastive_ablation",
    negative_pool="sampled_8192",
    training_epochs=40,
):
    path = DIAG / rel
    data = _load(path)
    if not data:
        return
    note = "notes/" + rel.replace(".json", ".md")
    for item in data.get("runs", []):
        test = item.get("splits_at_selected_threshold", {}).get("test") or item.get("test", {})
        if not test and "auroc" in item:
            test = item
        feats = item.get("features", "embedding")
        ed = item.get("embedding_dir") or data.get("embedding_dir", tag)
        run_name = _run_name_from_embedding_dir(str(ed))
        fixed = item.get("splits_at_threshold_0_5", {}).get("test") or {}
        rows.append(_row(
            **_ssl_proto(
                run_id="{0}|{1}|{2}|val_tuned".format(run_name, arm_id, feats),
                dataset=data.get("data", "Small-HI"),
                seed=2,
                training_epochs=training_epochs,
                tds=tds,
                graph_flags=(
                    "emlps,reverse_mp,ports,ego"
                    + (",tds" if tds else "")
                    + (",correct_reverse" if reverse_sem == "corrected_named_swap" else "")
                    + (",preserve_seed" if preserve else "")
                ),
                representation_source="post_embedding_128",
                probe_feature_stack=feats,
                AUROC=test.get("auroc"),
                AUPRC=test.get("auprc"),
                F1=test.get("f1") or test.get("f1_at_selected_threshold"),
                F1_fixed=fixed.get("f1") or fixed.get("f1_at_0_5"),
                precision=test.get("precision"),
                recall=test.get("recall"),
                source_json=str(path.relative_to(ROOT)),
                source_note=note,
                status="evaluated",
                scout_or_formal="scout",
                thesis_role=thesis_role,
                validation_status="diagnostic_only",
                table_eligible=table_eligible,
                table_group="contrastive_ablations",
                protocol_family=protocol_family,
                split_protocol="temporal",
                graph_representation="account_nodes_transaction_edges",
                reverse_feature_semantics=reverse_sem,
                preserve_seed_edges=preserve,
                batch_size=batch_size,
                accum_steps=accum,
                negative_pool_semantics=negative_pool,
                paper_comparable=False,
                caveats=(
                    "Jul21-22 arm {0}; val-tuned F1 separate from F1_fixed; "
                    "not GCPAL reproduction; not paper Multi-GIN row"
                ).format(arm_id),
            )
        ))


def ingest_gcpal_audit_only(rows, path: Path, note: str):
    data = _load(path)
    if not data:
        return
    rows.append(_row(
        run_id="{0}|gcpal_positive_audit".format(path.stem),
        dataset="Small-HI",
        objective="contrastive",
        status="audit_only",
        scout_or_formal="scout",
        thesis_role="diagnostic",
        validation_status="diagnostic_only",
        table_eligible=False,
        paper_comparable=False,
        protocol_family="gcpal_positive_set_audit",
        split_protocol="temporal",
        graph_representation="account_nodes_transaction_edges",
        source_json=str(path.relative_to(ROOT)),
        source_note=note,
        caveats="positive-set / KNN coverage audit only; no performance metrics ingested",
    ))


def ingest_txn_node_poscomplete(rows, path: Path):
    data = _load(path)
    if not data:
        return
    mode = data.get("mode", path.stem)
    note = "notes/" + path.name.replace(".json", ".md")
    for split_key, split_protocol, eligible_note in (
        ("temporal_primary", "temporal", "preliminary five-epoch; temporal primary"),
        ("random40_diagnostic", "random_40", "DIAGNOSTIC random-40; never primary table"),
    ):
        block = data.get(split_key) or {}
        for stack, metrics in block.items():
            if not isinstance(metrics, dict):
                continue
            thr05 = metrics.get("threshold_0.5") or metrics.get("fixed_0.5") or {}
            val_thr = (
                metrics.get("threshold_val_selected")
                or metrics.get("val_selected_threshold")
                or {}
            )
            # Prefer AUPRC/AUROC from thr05 block if present; keep F1 protocols separate rows
            base_m = thr05 if thr05 else metrics
            common = dict(
                dataset="Small-HI",
                objective="contrastive",
                encoder="gin_txn_node",
                seed=data.get("seed", 2),
                training_epochs=data.get("n_epochs", 5),
                batch_size=data.get("max_total_nodes", 2048),
                probe_feature_stack=stack,
                representation_source="txn_node_embedding",
                status="preliminary",
                scout_or_formal="scout",
                thesis_role="diagnostic",
                validation_status="diagnostic_only",
                table_eligible=False,
                paper_comparable=False,
                protocol_family="txn_node_gcpal_inspired_poscomplete",
                split_protocol=split_protocol,
                graph_representation="transaction_nodes",
                positive_set_definition=mode,
                knn_scope="global_cached_k15",
                source_json=str(path.relative_to(ROOT)),
                source_note=note,
                caveats=(
                    "NOT AN EXACT GCPAL REPRODUCTION; {0}; "
                    "positive-complete batching; λ=0.3 τ=0.5 k=15"
                ).format(eligible_note),
            )
            if base_m.get("auroc") is not None or base_m.get("auprc") is not None:
                rows.append(_row(
                    **common,
                    run_id="{0}|{1}|{2}|f1_fixed_0.5".format(path.stem, split_protocol, stack),
                    threshold_rule="fixed_0.5",
                    AUROC=base_m.get("auroc"),
                    AUPRC=base_m.get("auprc"),
                    F1=thr05.get("f1"),
                    F1_fixed=thr05.get("f1"),
                    precision=thr05.get("precision"),
                    recall=thr05.get("recall"),
                ))
            if val_thr.get("f1") is not None or val_thr.get("auroc") is not None:
                rows.append(_row(
                    **common,
                    run_id="{0}|{1}|{2}|f1_val_selected".format(path.stem, split_protocol, stack),
                    threshold_rule="max_f1_on_val",
                    AUROC=val_thr.get("auroc", base_m.get("auroc")),
                    AUPRC=val_thr.get("auprc", base_m.get("auprc")),
                    F1=val_thr.get("f1"),
                    precision=val_thr.get("precision"),
                    recall=val_thr.get("recall"),
                ))


def ingest_failed_forensic_provenance(rows):
    """Failed forensic jobs — provenance only, no metrics."""
    for job_id, note in (
        ("18558352", "forensic extract TypeError infer_pre_embedding_dim"),
        ("18566110", "forensic extract TypeError log_seed_coverage(split=)"),
    ):
        rows.append(_row(
            run_id="forensic_gcpal_eval_protocol_audit|failed|{0}".format(job_id),
            dataset="Small-HI",
            status="failed",
            scout_or_formal="scout",
            thesis_role="diagnostic",
            validation_status="diagnostic_only",
            table_eligible=False,
            paper_comparable=False,
            protocol_family="gcpal_forensic_eval_protocol",
            job_id=job_id,
            caveats=(
                "FAILED diagnostic; no scientific conclusion; no metrics; {0}"
            ).format(note),
            source_note="notes/documentation_audit_2026-07-22.md",
        ))


def ingest_final_dplus_multiseed_and_finetune(rows, path: Path):
    """Primary frozen D+ multiseed + secondary partial-FT locked eval."""
    data = _load(path)
    if not data or not data.get("per_seed"):
        return
    note = "notes/final_dplus_multiseed_and_finetune_analysis.md"
    common = dict(
        dataset="Small-HI",
        objective="downstream_probe",
        status="evaluated",
        scout_or_formal="formal",
        encoder="gin",
        representation_source="pre_embedding_3h",
        representation_dim=198,
        probe_feature_stack="H+X+TF|mlp|none",
        split_protocol="temporal",
        graph_representation="account_nodes_transaction_edges",
        graph_flags="emlps,reverse_mp,ports,ego,tds,correct_reverse,preserve_seed",
        reverse_feature_semantics="corrected_named_swap",
        preserve_seed_edges=True,
        emlps=True,
        tds=True,
        reverse_mp=True,
        ego=True,
        ports=True,
        source_json=str(path.relative_to(ROOT)),
        source_note=note,
        paper_comparable=False,
    )
    for r in data.get("per_seed") or []:
        seed = r.get("encoder_seed")
        rows.append(_row(
            **common,
            run_id="final_dplus_frozen|seed{0}|H+X+TF|mlp|f1_fixed_0.5".format(seed),
            seed=seed,
            selected_epoch=r.get("checkpoint_epoch"),
            checkpoint_path=None,
            thesis_role="primary_result",
            validation_status="formal",
            table_eligible=True,
            table_group="final_dplus_frozen_multiseed_primary",
            protocol_family="edge_dplus_frozen_hxxtf_mlp_multiseed",
            threshold_rule="fixed_0.5",
            AUROC=r.get("test_auroc"),
            AUPRC=r.get("test_auprc"),
            F1_fixed=r.get("test_f1_0.5"),
            F1=r.get("test_f1_0.5"),
            precision=r.get("test_precision_0.5"),
            recall=r.get("test_recall_0.5"),
            precision_at_100=r.get("P100"),
            precision_at_500=r.get("P500"),
            precision_at_1000=r.get("P1000"),
            finetune_protocol="none_encoder_frozen",
            caveats=(
                "PRIMARY: SSL contrastive encoder frozen; supervised downstream MLP uses AML labels; "
                "checkpoint_policy best from contrastive score (predeclared); never test selection."
            ),
        ))
        rows.append(_row(
            **common,
            run_id="final_dplus_frozen|seed{0}|H+X+TF|mlp|f1_val_selected".format(seed),
            seed=seed,
            selected_epoch=r.get("checkpoint_epoch"),
            thesis_role="primary_result",
            validation_status="formal",
            table_eligible=True,
            table_group="final_dplus_frozen_multiseed_primary",
            protocol_family="edge_dplus_frozen_hxxtf_mlp_multiseed",
            threshold_rule="val_selected",
            AUROC=r.get("test_auroc"),
            AUPRC=r.get("test_auprc"),
            F1=r.get("test_f1_val_thr"),
            precision=r.get("test_precision_val_thr"),
            recall=r.get("test_recall_val_thr"),
            finetune_protocol="none_encoder_frozen",
            caveats="Val-selected threshold F1; keep separate from fixed-0.5.",
        ))
    ft = data.get("finetune_secondary") or {}
    t05 = ft.get("test_metrics_threshold_0.5") or {}
    tv = ft.get("test_metrics_val_threshold") or {}
    if t05:
        rows.append(_row(
            **common,
            run_id="final_dplus_partial_finetune|seed2|H+X+TF|mlp|f1_fixed_0.5",
            seed=2,
            selected_epoch=ft.get("best_epoch"),
            thesis_role="secondary_sensitivity",
            validation_status="formal",
            table_eligible=True,
            table_group="final_dplus_partial_finetune_secondary",
            protocol_family="edge_dplus_partial_finetune_hxxtf",
            threshold_rule="fixed_0.5",
            AUROC=t05.get("auroc"),
            AUPRC=t05.get("auprc"),
            F1_fixed=t05.get("f1"),
            F1=t05.get("f1"),
            precision=t05.get("precision"),
            recall=t05.get("recall"),
            precision_at_100=t05.get("precision_at_100"),
            precision_at_500=t05.get("precision_at_500"),
            precision_at_1000=t05.get("precision_at_1000"),
            finetune_protocol="partial_final_block",
            classifier_lr=1e-3,
            encoder_lr=1e-4,
            partial_unfreeze_modules="convs.1,emlps.1,batch_norms.1",
            caveats=(
                "SECONDARY: SSL-pretrained D+ with supervised partial fine-tuning; "
                "AML labels update classifier + final encoder block; not the primary thesis claim."
            ),
        ))
        rows.append(_row(
            **common,
            run_id="final_dplus_partial_finetune|seed2|H+X+TF|mlp|f1_val_selected",
            seed=2,
            selected_epoch=ft.get("best_epoch"),
            thesis_role="secondary_sensitivity",
            validation_status="formal",
            table_eligible=True,
            table_group="final_dplus_partial_finetune_secondary",
            protocol_family="edge_dplus_partial_finetune_hxxtf",
            threshold_rule="val_selected",
            AUROC=tv.get("auroc"),
            AUPRC=tv.get("auprc"),
            F1=tv.get("f1"),
            precision=tv.get("precision"),
            recall=tv.get("recall"),
            finetune_protocol="partial_final_block",
            classifier_lr=1e-3,
            encoder_lr=1e-4,
            partial_unfreeze_modules="convs.1,emlps.1,batch_norms.1",
            caveats="Stored validation threshold from FT selection; test not used for selection.",
        ))


def ingest_edge_dplus_neighbor_positive(rows, path: Path):
    """10ep poscomplete identity vs neighbor SupCon scout (val-gated; no GNN retrain)."""
    data = _load(path)
    if not data or not data.get("val_rows"):
        return
    note = "notes/edge_dplus_neighbor_positive_10ep_seed2.md"
    job_id = str(data.get("valgate_job_id") or "")
    common = dict(
        dataset="Small-HI",
        objective="downstream_probe",
        seed=2,
        status="evaluated",
        scout_or_formal="scout",
        thesis_role="diagnostic",
        validation_status="diagnostic_only",
        table_eligible=False,
        paper_comparable=False,
        protocol_family="edge_dplus_neighbor_positive_10ep",
        graph_flags="emlps,reverse_mp,ports,ego,tds,correct_reverse",
        reverse_feature_semantics="corrected_named_swap",
        preserve_seed_edges=False,
        source_json=str(path.relative_to(ROOT)),
        source_note=note,
        job_id=job_id or None,
        caveats=(
            "NOT exact GCPAL; matched identity poscomplete required; selection=max H+X+TF "
            "val AUPRC (never SSL loss / never test); 10ep horizon; unequal batching vs D+ 40ep; "
            "recommend_40ep={0}"
        ).format(data.get("recommend_40ep")),
    )
    for role, key in (
        ("selected_identity", "selected_identity"),
        ("selected_neighbor", "selected_neighbor"),
        ("winner", "winner"),
    ):
        sel = data.get(key) or {}
        if not sel:
            continue
        rows.append(_row(
            **common,
            run_id="edge_dplus_neighbor_positive|{0}|{1}|val".format(role, sel.get("tag")),
            encoder="gin",
            representation_source="pre_embedding_3h",
            representation_dim=198,
            probe_feature_stack="{0}|mlp|none".format(sel.get("stack")),
            split_protocol="temporal",
            graph_representation="account_nodes_transaction_edges",
            threshold_rule="val_selected",
            AUPRC=sel.get("val_auprc"),
            AUROC=sel.get("val_auroc"),
            F1=sel.get("val_f1_at_selected"),
            epochs=sel.get("epoch"),
            unique_name=sel.get("arm"),
        ))
    for tr in data.get("test_rows") or []:
        rows.append(_row(
            **common,
            run_id="edge_dplus_neighbor_positive|{0}|test_fixed_0.5".format(tr.get("tag")),
            encoder="gin",
            representation_source="pre_embedding_3h",
            representation_dim=198,
            probe_feature_stack="{0}|mlp|none".format(tr.get("stack")),
            split_protocol="temporal",
            graph_representation="account_nodes_transaction_edges",
            threshold_rule="fixed_0.5",
            AUROC=tr.get("test_auroc"),
            AUPRC=tr.get("test_auprc"),
            F1=tr.get("test_f1_0.5"),
            F1_fixed=tr.get("test_f1_0.5"),
            precision=tr.get("test_p_0.5"),
            recall=tr.get("test_r_0.5"),
            precision_at_100=tr.get("p_at_100"),
            precision_at_500=tr.get("p_at_500"),
            precision_at_1000=tr.get("p_at_1000"),
            epochs=tr.get("epoch"),
            unique_name=tr.get("arm"),
        ))
        rows.append(_row(
            **common,
            run_id="edge_dplus_neighbor_positive|{0}|test_val_thr".format(tr.get("tag")),
            encoder="gin",
            representation_source="pre_embedding_3h",
            representation_dim=198,
            probe_feature_stack="{0}|mlp|none".format(tr.get("stack")),
            split_protocol="temporal",
            graph_representation="account_nodes_transaction_edges",
            threshold_rule="val_selected",
            AUROC=tr.get("test_auroc"),
            AUPRC=tr.get("test_auprc"),
            F1=tr.get("test_f1_val_thr"),
            precision=tr.get("test_p_val_thr"),
            recall=tr.get("test_r_val_thr"),
            epochs=tr.get("epoch"),
            unique_name=tr.get("arm"),
        ))


def ingest_gcpal_challenge_fullstack(rows, path: Path):
    """No-GNN full-stack challenge eval (temporal primary + reconstructed random 40/60)."""
    data = _load(path)
    if not data:
        return
    note = "notes/gcpal_challenge_fullstack_eval.md"
    job_id = str(data.get("slurm_job_id") or "")
    gate = (data.get("comparability_gate") or {}).get("verdict", "PARTIAL")
    common = dict(
        dataset="Small-HI",
        objective="downstream_probe",
        seed=2,
        status="evaluated",
        scout_or_formal="scout",
        thesis_role="diagnostic",
        validation_status="diagnostic_only",
        table_eligible=False,
        paper_comparable=False,
        protocol_family="gcpal_challenge_fullstack",
        graph_flags="emlps,reverse_mp,ports,ego,tds,correct_reverse,preserve_seed",
        reverse_feature_semantics="corrected_named_swap",
        preserve_seed_edges=True,
        source_json=str(path.relative_to(ROOT)),
        source_note=note,
        job_id=job_id or None,
        caveats=(
            "NOT exact GCPAL reproduction; no GNN training; selection=max val AUPRC "
            "(never test); comparability_gate={0}; fixed-0.5 and val-threshold separate"
        ).format(gate),
    )

    sel = (data.get("temporal_primary") or {}).get("selected") or {}
    metrics = sel.get("metrics") or {}
    thr05 = metrics.get("threshold_0.5") or {}
    thr_val = metrics.get("threshold_val_selected") or {}
    tag = sel.get("tag") or "unknown"
    cand = sel.get("candidate") or ""
    stack = sel.get("stack") or ""
    if cand.startswith("edge_pre3h"):
        rep = "pre_embedding_3h"
        dim = 198
        enc = "gin"
        graph_rep = "account_nodes_transaction_edges"
    elif cand.startswith("edge_post128"):
        rep = "post_embedding_128"
        dim = 128
        enc = "gin"
        graph_rep = "account_nodes_transaction_edges"
    elif "txn" in cand:
        rep = "txn_node_embedding"
        dim = None
        enc = "gin_txn_node"
        graph_rep = "transaction_nodes"
    else:
        rep = "features_only"
        dim = None
        enc = None
        graph_rep = "transaction_features"

    if thr05.get("auprc") is not None or thr05.get("auroc") is not None:
        rows.append(_row(
            **common,
            run_id="gcpal_challenge_fullstack|{0}|temporal|f1_fixed_0.5".format(tag),
            encoder=enc,
            representation_source=rep,
            representation_dim=dim,
            probe_feature_stack="{0}|{1}|{2}".format(
                stack, sel.get("learner"), sel.get("weight"),
            ),
            split_protocol="temporal",
            graph_representation=graph_rep,
            threshold_rule="fixed_0.5",
            AUROC=thr05.get("auroc"),
            AUPRC=thr05.get("auprc"),
            F1=thr05.get("f1"),
            F1_fixed=thr05.get("f1"),
            precision=thr05.get("precision"),
            recall=thr05.get("recall"),
            precision_at_100=thr05.get("precision_at_100"),
            recall_at_100=thr05.get("recall_at_100"),
            lift_at_100=thr05.get("lift_at_100"),
            precision_at_500=thr05.get("precision_at_500"),
            recall_at_500=thr05.get("recall_at_500"),
            lift_at_500=thr05.get("lift_at_500"),
            precision_at_1000=thr05.get("precision_at_1000"),
            recall_at_1000=thr05.get("recall_at_1000"),
            lift_at_1000=thr05.get("lift_at_1000"),
            paired_test_n=thr05.get("n"),
            table_group="gcpal_challenge_temporal_primary",
        ))
    if thr_val.get("f1") is not None:
        rows.append(_row(
            **common,
            run_id="gcpal_challenge_fullstack|{0}|temporal|f1_val_selected".format(tag),
            encoder=enc,
            representation_source=rep,
            representation_dim=dim,
            probe_feature_stack="{0}|{1}|{2}".format(
                stack, sel.get("learner"), sel.get("weight"),
            ),
            split_protocol="temporal",
            graph_representation=graph_rep,
            threshold_rule="max_f1_on_val",
            AUROC=thr_val.get("auroc", thr05.get("auroc")),
            AUPRC=thr_val.get("auprc", thr05.get("auprc")),
            F1=thr_val.get("f1"),
            precision=thr_val.get("precision"),
            recall=thr_val.get("recall"),
            precision_at_100=thr_val.get("precision_at_100"),
            recall_at_100=thr_val.get("recall_at_100"),
            lift_at_100=thr_val.get("lift_at_100"),
            precision_at_500=thr_val.get("precision_at_500"),
            recall_at_500=thr_val.get("recall_at_500"),
            lift_at_500=thr_val.get("lift_at_500"),
            precision_at_1000=thr_val.get("precision_at_1000"),
            recall_at_1000=thr_val.get("recall_at_1000"),
            lift_at_1000=thr_val.get("lift_at_1000"),
            paired_test_n=thr_val.get("n"),
            table_group="gcpal_challenge_temporal_primary",
        ))

    rp = data.get("random_protocols") or {}
    for ratio_key, split_name, target in (
        ("random_40", "random_40", 0.581),
        ("random_60", "random_60", 0.658),
    ):
        block = rp.get(ratio_key) or {}
        sel_tag = block.get("selected_by_val_auprc")
        after = block.get("selected_test_metrics_after_selection") or {}
        if not sel_tag or not after:
            continue
        f05 = after.get("agg_test_f1_0.5") or {}
        fvt = after.get("agg_test_f1_val_thr") or {}
        auprc = after.get("agg_test_auprc") or {}
        base_caveat = common["caveats"]
        rows.append(_row(
            **{k: v for k, v in common.items() if k != "caveats"},
            run_id="gcpal_challenge_fullstack|{0}|{1}|mean_f1_fixed_0.5".format(
                sel_tag, split_name,
            ),
            encoder="gin",
            representation_source="pre_embedding_3h",
            representation_dim=198,
            probe_feature_stack=sel_tag,
            split_protocol=split_name,
            graph_representation="account_nodes_transaction_edges",
            threshold_rule="fixed_0.5",
            AUPRC=auprc.get("mean"),
            F1=f05.get("mean"),
            F1_fixed=f05.get("mean"),
            table_group="gcpal_challenge_random_diagnostic",
            caveats=(
                base_caveat
                + "; reconstructed {0} label ratio; target F1={1}; "
                "mean over seeds {2}; exceeds_target={3}; "
                "edge H=temporal-extract then re-label (PARTIAL scope)"
            ).format(
                split_name,
                target,
                (block.get("split_seeds") or []),
                bool(after.get("exceeds_target_f1_0.5_mean")),
            ),
        ))
        rows.append(_row(
            **{k: v for k, v in common.items() if k != "caveats"},
            run_id="gcpal_challenge_fullstack|{0}|{1}|mean_f1_val_selected".format(
                sel_tag, split_name,
            ),
            encoder="gin",
            representation_source="pre_embedding_3h",
            representation_dim=198,
            probe_feature_stack=sel_tag,
            split_protocol=split_name,
            graph_representation="account_nodes_transaction_edges",
            threshold_rule="max_f1_on_val",
            AUPRC=auprc.get("mean"),
            F1=fvt.get("mean"),
            table_group="gcpal_challenge_random_diagnostic",
            caveats=(
                base_caveat
                + "; reconstructed {0} label ratio; target F1={1}; "
                "mean over seeds {2}; exceeds_target={3}; "
                "edge H=temporal-extract then re-label (PARTIAL scope)"
            ).format(
                split_name,
                target,
                (block.get("split_seeds") or []),
                bool(after.get("exceeds_target_f1_val_thr_mean")),
            ),
        ))


def ingest_gcpal_posagg_provenance(rows, path: Path):
    """Positive-aggregation ablation provenance (selected D SupCon on val)."""
    data = _load(path)
    if not data:
        return
    sel = data.get("selected_condition") or data.get("selected_aggregation") or ""
    rows.append(_row(
        run_id="gcpal_txn_node_posagg_ablation|selected|{0}".format(sel or "unknown"),
        dataset="Small-HI",
        objective="contrastive",
        encoder="gin_txn_node",
        seed=2,
        training_epochs=5,
        status="evaluated",
        scout_or_formal="scout",
        thesis_role="diagnostic",
        validation_status="diagnostic_only",
        table_eligible=False,
        paper_comparable=False,
        protocol_family="txn_node_gcpal_inspired_posagg",
        split_protocol="temporal",
        graph_representation="transaction_nodes",
        positive_set_definition=str(sel),
        source_json=str(path.relative_to(ROOT)),
        source_note="notes/gcpal_txn_node_posagg_ablation.md",
        job_id=",".join(str(x) for x in (data.get("job_ids") or [])),
        caveats=(
            "NOT exact GCPAL reproduction; selection=temporal val HxX AUPRC among "
            "B/C/D aggregations; no test-driven selection; feeds challenge eval candidate"
        ),
    ))


def ingest_supervised_nonlegacy(rows, path):
    data = _load(path)
    if not data:
        return
    test = data.get("splits", {}).get("test", {})
    rows.append(_row(
        run_id="{0}|supervised|val_tuned".format(data["run_name"]),
        dataset="Small-LI", objective="supervised", encoder="gin", seed=1,
        training_epochs=20, selected_epoch=data.get("checkpoint_epoch"),
        supervised_head="standard", graph_flags="emlps,tds",
        representation_source="logits_direct", probe_feature_stack="in_gnn_end_to_end",
        threshold_rule="max_f1_on_val",
        source_json=str(path.relative_to(ROOT)),
        checkpoint_path=data.get("checkpoint_path"),
        status="evaluated", scout_or_formal="scout", superseded=True,
        thesis_role="superseded",
        caveats="historical non-legacy supervised; superseded",
        **_metrics_from_test_block(test),
    ))


def classify_thesis_roles(rows):
    primary_runs = {
        "gin_emlps_tds_asym_proj_8192neg_queue0_40ep_seed2",
        "same_pair_fnf_emlps_tds_asym_proj_8192neg_queue0_20ep",
    }
    for r in rows:
        if r.get("superseded"):
            r["thesis_role"] = "superseded"
            continue
        if r.get("thesis_role") in (
            "diagnostic",
            "diagnostic_or_scout",
            "negative_result",
            "superseded",
        ):
            continue
        src = str(r.get("source_json", ""))
        rid = (r.get("run_id") or "").split("|")[0]
        if src == MULTISEED_LI_SOURCE:
            r["thesis_role"] = "thesis_primary"
        elif src == CANONICAL_PREPOST_SOURCE and rid in primary_runs:
            r["thesis_role"] = "thesis_primary"
        elif src == CANONICAL_PREPOST_SOURCE:
            r["thesis_role"] = "thesis_supporting"
        elif r.get("objective") == "supervised" and r.get("scout_or_formal") == "formal":
            r["thesis_role"] = "thesis_primary"
        elif "architecture_sweep" in src:
            r["thesis_role"] = "thesis_supporting"
        elif "degree_aware" in src:
            r["thesis_role"] = "negative_result"
        elif EMB198_SOURCE in src:
            r["thesis_role"] = "diagnostic"
        elif "historical_20ep_baseline" in str(r.get("caveats", "")):
            r["thesis_role"] = "thesis_supporting"
        elif r.get("thesis_role") is None:
            r["thesis_role"] = "thesis_supporting"


def resolve_duplicates(rows):
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        key = (
            r.get("run_id"),
            r.get("representation_source"),
            r.get("probe_feature_stack"),
            r.get("threshold_rule"),
        )
        groups[key].append(i)

    for key, idxs in groups.items():
        if len(idxs) <= 1:
            continue
        sources = [rows[i].get("source_json") for i in idxs]
        src_set = set(sources)
        has_multiseed = MULTISEED_LI_SOURCE in src_set
        has_emb198 = EMB198_SOURCE in src_set
        has_strong = CANONICAL_PREPOST_SOURCE in src_set
        has_ablation_fnf = "probe_feature_ablation_same_pair_fnf" in " ".join(src_set)

        for i in idxs:
            r = rows[i]
            src = r.get("source_json")
            if has_multiseed and has_emb198:
                if src == EMB198_SOURCE:
                    r["duplicate_resolution"] = "pairing_intersection_diff"
                    r["caveats"] = (r.get("caveats") or "") + (
                        "; NOT duplicate of multiseed — different inner-join row set"
                    )
                elif src == MULTISEED_LI_SOURCE:
                    r["duplicate_resolution"] = "canonical_multiseed_source"
                else:
                    r["duplicate_resolution"] = "cross_source_replication"
            elif has_strong and has_ablation_fnf:
                if src == CANONICAL_PREPOST_SOURCE:
                    r["duplicate_resolution"] = "canonical_prepost_source"
                else:
                    r["duplicate_resolution"] = "cross_source_replication"
                    r["caveats"] = (r.get("caveats") or "") + (
                        "; cross-source replication of strong-run metrics"
                    )
            elif "architecture_sweep" in src and any(
                "alert_budget" in s for s in src_set if s != src
            ):
                r["duplicate_resolution"] = "cross_source_replication"
            elif "architecture_sweep" in src and CANONICAL_PREPOST_SOURCE not in src_set:
                other = [s for s in src_set if s != src]
                if any("pre_embedding_3h_vs_post_embedding_pna" in s for s in other):
                    r["duplicate_resolution"] = "cross_source_replication"
                else:
                    r["duplicate_resolution"] = "cross_source_replication"
            else:
                r["duplicate_resolution"] = "cross_source_replication"
                r["caveats"] = (r.get("caveats") or "") + (
                    "; cross-source replication (sources: {0})".format(
                        ", ".join(sorted(src_set))
                    )
                )


def _sample_std(vals):
    """Sample SD (ddof=1) for thesis multiseed reporting."""
    if len(vals) <= 1:
        return 0.0
    return stats.stdev(vals)


def compute_multiseed_aggregates():
    data = _load(DIAG / "pre_embedding_3h_vs_post_embedding_small_li_multiseed.json")
    if not data:
        return {}
    out = {
        "source_json": MULTISEED_LI_SOURCE,
        "n_seeds": 3,
        "std_convention": "sample",
        "std_ddof": MULTISEED_STD_DDOF,
        "aggregation_note": (
            "Mean ± {0} over three seeds.".format(MULTISEED_STD_LABEL)
        ),
        "stacks": {},
    }
    stack_keys = {
        "embedding_only": "embedding",
        "embedding_plus_raw": "embedding+raw",
    }
    for mode, stack in stack_keys.items():
        block = {"stack": stack, "representations": {}}
        for rep_key, rep_name in (("post", "post_embedding_128"), ("pre", "pre_embedding_3h")):
            metrics = defaultdict(list)
            for ps in data["per_seed"]:
                b = ps["modes"][mode][rep_key]
                for m in ("auroc", "auprc", "f1_at_selected_threshold",
                          "precision_at_100", "recall_at_100", "lift_at_100",
                          "precision_at_500", "recall_at_500", "lift_at_500",
                          "precision_at_1000", "recall_at_1000", "lift_at_1000"):
                    if b.get(m) is not None:
                        metrics[m].append(float(b[m]))
            rep_out = {}
            for m, vals in metrics.items():
                rep_out[m] = {
                    "mean": stats.mean(vals),
                    "std": _sample_std(vals),
                    "per_seed": vals,
                }
            block["representations"][rep_name] = rep_out
        deltas = {}
        for metric in ("auprc", "recall_at_100", "auroc", "lift_at_100", "precision_at_100"):
            dvals = []
            for ps in data["per_seed"]:
                po = ps["modes"][mode]["post"]
                pr = ps["modes"][mode]["pre"]
                if po.get(metric) is not None and pr.get(metric) is not None:
                    dvals.append(float(pr[metric]) - float(po[metric]))
            if dvals:
                deltas["delta_{0}_pre_minus_post".format(metric)] = {
                    "mean": stats.mean(dvals),
                    "std": _sample_std(dvals),
                }
        block["deltas_pre_minus_post"] = deltas
        out["stacks"][stack] = block
    # explicit seed1 +raw pre for reporting convention
    s1 = data["per_seed"][0]["modes"]["embedding_plus_raw"]["pre"]
    out["reporting_conventions"] = {
        "seed1_pre3h_plus_raw_auprc_multiseed": s1["auprc"],
        "seed1_pre3h_plus_raw_auprc_emb198_paired": None,
    }
    emb = _load(DIAG / "small_li_embedding_dim_128_vs_198.json")
    if emb:
        try:
            out["reporting_conventions"]["seed1_pre3h_plus_raw_auprc_emb198_paired"] = (
                emb["blocks"]["embedding_plus_raw"]["representations"]
                ["orig_pre_3h_198"]["test"]["auprc"]
            )
        except (KeyError, TypeError):
            pass
    return out


def build_registry():
    rows = []
    ingest_alert_budget(rows, DIAG / "alert_budget_metrics_current_protocol.json")
    ingest_alert_budget(rows, DIAG / "alert_budget_metrics_small_hi.json")
    ingest_alert_budget(rows, DIAG / "alert_budget_metrics_small_li.json")
    ingest_feature_ablation_comparison(
        rows, DIAG / "probe_feature_ablation_current_protocol_comparison.json"
    )
    ingest_architecture_sweep(rows, DIAG / "architecture_sweep_shared_probe_weights.json")
    ingest_pre3h_strong(rows, DIAG / "pre3h_strong_run_comparison.json")
    ingest_prepost_json(
        rows, DIAG / "pre_embedding_3h_vs_post_embedding_small_li_multiseed.json",
        "notes/pre_embedding_3h_vs_post_embedding_small_li_multiseed.md",
    )
    ingest_prepost_json(
        rows, DIAG / "pre_embedding_3h_vs_post_embedding_small_hi.json",
        "notes/pre_embedding_3h_vs_post_embedding_small_hi.md",
    )
    ingest_prepost_json(
        rows, DIAG / "pre_embedding_3h_vs_post_embedding_small_li.json",
        "notes/pre_embedding_3h_vs_post_embedding_small_li.md",
        skip_seeds={1, 2, 3},
    )
    for s in (2, 3):
        ingest_prepost_json(
            rows,
            DIAG / "pre_embedding_3h_vs_post_embedding_small_li_seed{0}.json".format(s),
            "notes/pre_embedding_3h_vs_post_embedding_small_li_seed{0}.md".format(s),
            skip_seeds={1, 2, 3},
        )
    ingest_legacy(
        rows,
        DIAG / "supervised_Small-LI_small_li_legacy_supervised_gin_emlps_tds_100ep_seed1_summary.json",
        DIAG / "eval_small_li_legacy_supervised_gin_emlps_tds_100ep_seed1.json",
    )
    hi_summary = DIAG / "supervised_Small-HI_small_hi_legacy_supervised_gin_emlps_tds_100ep_seed1_summary.json"
    hi_eval = DIAG / "eval_small_hi_legacy_supervised_gin_emlps_tds_100ep_seed1.json"
    if hi_summary.is_file() and hi_eval.is_file():
        # Historical TDS-on formal — retain for provenance; not paper-comparable.
        ingest_legacy(
            rows, hi_summary, hi_eval, superseded=True,
            tds=True, paper_comparable=False,
            protocol_family="supervised_legacy_tds_on_not_paper",
            reverse_feature_semantics="inherited_malformed_under_tds",
            table_group=None,
            table_eligible=False,
        )
    for seed in (1, 2, 3):
        ingest_legacy(
            rows,
            DIAG / (
                "supervised_Small-HI_small_hi_legacy_supervised_gin_emlps_ports_50ep_seed{0}_summary.json"
                .format(seed)
            ),
            DIAG / (
                "eval_small_hi_legacy_supervised_gin_emlps_ports_50ep_seed{0}.json"
                .format(seed)
            ),
            superseded=False,
            tds=False,
            paper_comparable=True,
            protocol_family="supervised_multigin_eu_ports_no_tds",
            reverse_feature_semantics="ports_only_inherited",
            table_group="main_results",
            table_eligible=True,
        )
    ingest_ports_hi_aggregate(rows, DIAG / Path(HI_PORTS_AGG_SOURCE).name)
    for rel, tag, tds, rev, preserve, arm in JUL22_ABCD_PROBES:
        ingest_jul22_probe_arm(
            rows, rel, tag, tds, rev, preserve, arm,
            thesis_role="diagnostic", table_eligible=False,
        )
    for rel, bs, accum, arm in JUL22_BATCH_EF_PROBES:
        ingest_jul22_probe_arm(
            rows, rel, arm, True, "corrected_named_swap", True, arm,
            batch_size=bs, accum=accum,
            thesis_role="diagnostic", table_eligible=False,
            protocol_family="edge_centric_batch_size_diagnostic",
            negative_pool="all_inbatch_num_neg_0",
            training_epochs=10,
        )
    ingest_gcpal_audit_only(
        rows, DIAG / "gcpal_positive_set_audit.json",
        "notes/gcpal_positive_set_audit.md",
    )
    multibatch = DIAG / "gcpal_positive_set_multibatch_audit.json"
    if multibatch.is_file():
        ingest_gcpal_audit_only(
            rows, multibatch, "notes/gcpal_positive_set_audit.md",
        )
    for smoke in (
        "gcpal_txn_node_smoke.json",
        "gcpal_txn_node_poscomplete_smoke.json",
    ):
        p = DIAG / smoke
        if p.is_file():
            rows.append(_row(
                run_id="{0}|smoke".format(p.stem),
                dataset="Small-HI",
                status="diagnostic",
                thesis_role="diagnostic",
                validation_status="diagnostic_only",
                table_eligible=False,
                paper_comparable=False,
                protocol_family="txn_node_gcpal_inspired",
                graph_representation="transaction_nodes",
                source_json=str(p.relative_to(ROOT)),
                source_note="notes/{0}.md".format(p.stem),
                caveats="smoke only; NOT exact GCPAL reproduction",
            ))
    for scout in (
        "gcpal_txn_node_scout_control_5ep_seed2.json",
        "gcpal_txn_node_scout_gcpal_5ep_seed2.json",
    ):
        p = DIAG / scout
        if p.is_file():
            rows.append(_row(
                run_id="{0}|ordinary_batch_scout".format(p.stem),
                dataset="Small-HI",
                status="preliminary",
                thesis_role="diagnostic",
                validation_status="diagnostic_only",
                table_eligible=False,
                paper_comparable=False,
                protocol_family="txn_node_gcpal_inspired_ordinary_batch",
                graph_representation="transaction_nodes",
                knn_scope="global_cached_but_ordinary_batch_low_hit_rate",
                source_json=str(p.relative_to(ROOT)),
                source_note="notes/{0}.md".format(p.stem),
                caveats=(
                    "ordinary batching ~1.3% KNN coverage; NOT exact GCPAL reproduction; "
                    "superseded for positive design by poscomplete scouts"
                ),
            ))
    for pc in (
        "gcpal_txn_node_poscomplete_scout_A_identity_5ep_seed2.json",
        "gcpal_txn_node_poscomplete_scout_B_gcpal_5ep_seed2.json",
    ):
        ingest_txn_node_poscomplete(rows, DIAG / pc)
    ingest_gcpal_posagg_provenance(rows, DIAG / "gcpal_txn_node_posagg_ablation.json")
    ingest_gcpal_challenge_fullstack(
        rows, DIAG / "gcpal_challenge_fullstack_eval.json",
    )
    ingest_edge_dplus_neighbor_positive(
        rows, DIAG / "edge_dplus_neighbor_positive_10ep_seed2.json",
    )
    ingest_final_dplus_multiseed_and_finetune(
        rows, DIAG / "final_dplus_multiseed_and_finetune_analysis.json",
    )
    ingest_failed_forensic_provenance(rows)
    ingest_legacy(
        rows,
        DIAG / "supervised_Small-LI_small_li_legacy_supervised_gin_emlps_tds_seed1_scout_summary.json",
        DIAG / "eval_small_li_legacy_supervised_gin_emlps_tds_seed1_scout.json",
        superseded=True,
    )
    ingest_supervised_nonlegacy(rows, DIAG / "supervised_small_li_gin_emlps_tds_seed1.json")
    ingest_ablation_runs(
        rows, DIAG / "probe_feature_ablation_same_pair_fnf_emlps_tds.json",
        "same_pair_fnf", "same_pair",
    )
    ingest_ablation_runs(
        rows, DIAG / "probe_feature_ablation_degree_aware_edgedrop_emlps_tds.json",
        "degree_aware", None,
    )
    ingest_emb198(rows, DIAG / "small_li_embedding_dim_128_vs_198.json")
    pna = DIAG / "pre_embedding_3h_vs_post_embedding_pna_emlps_tds_seed1.json"
    if pna.is_file():
        ingest_prepost_json(
            rows, pna,
            "notes/pre_embedding_3h_vs_post_embedding_pna_emlps_tds_seed1.md",
        )
        for r in rows:
            if "pna_emlps" in str(r.get("run_id", "")):
                r["encoder"] = "pna"
                r["caveats"] = (r.get("caveats") or "") + "; PNA not capacity-matched"
    pna_w65 = DIAG / "pre_embedding_3h_vs_post_embedding_pna_width65_seed1.json"
    if pna_w65.is_file():
        ingest_prepost_json(
            rows, pna_w65,
            "notes/pre_embedding_3h_vs_post_embedding_pna_width65_seed1.md",
        )
        for r in rows:
            if "pna_width65" in str(r.get("run_id", "")):
                r["encoder"] = "pna"
                r["caveats"] = (
                    (r.get("caveats") or "")
                    + "; PNA width-aligned scout; GIN-matched LR/dropout; seed 1"
                )
    ingest_pna_width_aligned_probe(rows, DIAG / "pna_width_aligned_probe.json")
    ingest_pna_width_audit(rows, DIAG / "pna_width_param_audit.json")
    for rel in (
        "temporal_flow_ablation_small_hi_40ep_seed2.json",
        "temporal_flow_ablation_small_hi_40ep_seed2_maxiter5000.json",
        "temporal_flow_ablation_small_li_seed1.json",
        "temporal_flow_ablation_small_li_seed2.json",
        "temporal_flow_ablation_small_li_seed3.json",
        "temporal_flow_ablation_small_li_seed1_maxiter5000.json",
        "temporal_flow_ablation_small_li_seed2_maxiter5000.json",
        "temporal_flow_ablation_small_li_seed3_maxiter5000.json",
        "pna_width65_temporal_flow_probe.json",
    ):
        ingest_temporal_flow_ablation(rows, DIAG / rel)
        for r in rows:
            if "pna_width65" in str(r.get("run_id", "")) and rel.endswith("pna_width65_temporal_flow_probe.json"):
                r["encoder"] = "pna"
                r["scout_or_formal"] = "scout"
                r["training_epochs"] = 20
                r["caveats"] = (
                    (r.get("caveats") or "")
                    + "; width65 PNA; one seed; downstream-only best-stack probe"
                )
    tf_aux_epochs = _tf_aux_selected_epochs()
    for rel in TF_AUX_PROBE_FILES:
        ingest_temporal_flow_aux_probe(
            rows, DIAG / rel, selected_epochs=tf_aux_epochs
        )
    for rel in TF_SOFT_PROBE_FILES:
        ingest_temporal_flow_soft_positive_probe(rows, DIAG / rel)
    for rel in MORPH_OBJ_RECALL_PROBE_FILES:
        ingest_morphology_objective_recall_probe(rows, DIAG / rel)
    for rel in DEGFLOW_MULTISEED_PROBE_FILES:
        p = DIAG / rel
        if not p.is_file():
            continue
        ingest_morphology_objective_recall_probe(
            rows,
            p,
            table_group="degflow_morphology_multiseed_scout",
            source_note="notes/degflow_morphology_multiseed_scout.md",
        )
    for rel in TF_REG_AUX_MULTISEED_PROBE_FILES:
        p = DIAG / rel
        if not p.is_file():
            continue
        ingest_temporal_flow_aux_probe(
            rows,
            p,
            selected_epochs=tf_aux_epochs,
            table_group="temporal_flow_regression_aux_multiseed",
            source_note="notes/temporal_flow_regression_aux_multiseed.md",
            thesis_role="diagnostic_or_scout",
            caveats_extra="multiseed_confirmation",
        )
    for rel in CTR_RES_PROBE_FILES:
        p = DIAG / rel
        if not p.is_file():
            continue
        before = len(rows)
        ingest_temporal_flow_ablation(
            rows,
            p,
            source_note="notes/contrastive_objective_resource_scout.md",
        )
        rep_name = "post_embedding_128" if "post128" in rel else "pre_embedding_3h"
        default_dim = 128 if rep_name == "post_embedding_128" else 198
        for r in rows[before:]:
            r["thesis_role"] = "diagnostic_or_scout"
            r["validation_status"] = "diagnostic_only"
            r["table_eligible"] = False
            r["table_group"] = "contrastive_objective_resource_scout"
            r["scout_or_formal"] = "scout"
            r["representation_source"] = rep_name
            r["representation_dim"] = r.get("representation_dim") or default_dim
            # Fix run_id representation token hardcoded by ablation ingest.
            rid = str(r.get("run_id") or "")
            r["run_id"] = rid.replace("|pre_embedding_3h|", f"|{rep_name}|", 1)
            cave = r.get("caveats") or ""
            r["caveats"] = (
                cave
                + "; contrastive objective resource scout (large_bs / edge_drop); "
                "no labels in SSL; diagnostic_only; not main-table eligible"
            )
    classify_thesis_roles(rows)
    resolve_duplicates(rows)
    tf_validation = _temporal_flow_validation_state()
    assign_table_metadata(rows, tf_validation)
    return rows


def _filter(rows, **kw):
    out = rows
    for k, v in kw.items():
        if v is None:
            out = [r for r in out if r.get(k) is None]
        else:
            out = [r for r in out if r.get(k) == v]
    return out


def _pick_row(rows, **filters):
    c = _filter(rows, **filters)
    c = [r for r in c if r.get("status") == "evaluated" and not r.get("superseded")]
    return c[0] if len(c) == 1 else (c[0] if c else None)


def _pick_best(rows, metric, prefer_source=None, **filters):
    filters.setdefault("superseded", False)
    c = _filter(rows, **filters)
    c = [r for r in c if r.get(metric) is not None and r.get("status") == "evaluated"]
    if prefer_source:
        pref = [r for r in c if r.get("source_json") == prefer_source]
        if pref:
            c = pref
    return max(c, key=lambda x: float(x[metric])) if c else None


def _fmt_row(r):
    if not r:
        return "*No matching row.*"
    lines = [
        "- **run_id:** `{0}`".format(r.get("run_id")),
        "- **seed:** {0} | **training_epochs:** {1} | **selected_epoch:** {2}".format(
            r.get("seed"), r.get("training_epochs"), r.get("selected_epoch")),
        "- **rep:** {0} | **stack:** {1}".format(
            r.get("representation_source"), r.get("probe_feature_stack")),
        "- **AUROC / AUPRC / F1:** {0} / {1} / {2}".format(
            r.get("AUROC"), r.get("AUPRC"), r.get("F1")),
    ]
    if r.get("precision_at_100") is not None:
        lines.append("- **P@100 / R@100 / lift@100:** {0} / {1} / {2}".format(
            r.get("precision_at_100"), r.get("recall_at_100"), r.get("lift_at_100")))
    if r.get("paired_test_n") is not None:
        lines.append("- **paired_test_n:** {0}".format(int(r.get("paired_test_n"))))
    lines.extend([
        "- **threshold_rule:** {0} | **thesis_role:** {1}".format(
            r.get("threshold_rule"), r.get("thesis_role")),
        "- **source:** `{0}`".format(r.get("source_json")),
        "- **caveats:** {0}".format(r.get("caveats")),
    ])
    return "\n".join(lines)


def write_registry_md(rows, aggregates):
    role_counts = defaultdict(int)
    for r in rows:
        role_counts[r.get("thesis_role") or "unset"] += 1
    dup_groups = defaultdict(list)
    for r in rows:
        if r.get("duplicate_resolution") not in (None, "not_duplicate"):
            dup_groups[r.get("duplicate_resolution")].append(r)

    lines = [
        "# Thesis experiment registry",
        "",
        "Traceable source of truth for thesis-relevant evaluated configurations.",
        "Every metric is copied from a cited JSON file — **no inferred values**.",
        "",
        "## Registry files",
        "",
        "| File | Description |",
        "|------|-------------|",
        "| `results/diagnostics/thesis_experiment_registry.csv` | One row per evaluated config |",
        "| `results/diagnostics/thesis_experiment_registry.json` | Rows + multiseed aggregates |",
        "",
        "**Total rows:** {0} | **thesis_primary:** {1} | **thesis_supporting:** {2} | "
        "**diagnostic:** {3} | **negative_result:** {4} | **superseded:** {5}".format(
            len(rows),
            role_counts["thesis_primary"],
            role_counts["thesis_supporting"],
            role_counts["diagnostic"],
            role_counts["negative_result"],
            role_counts["superseded"],
        ),
        "",
        "## Field conventions",
        "",
        "- `representation_source`: `post_embedding_128` vs `pre_embedding_3h` (198-d)",
        "- `threshold_rule`: `max_f1_on_val` (SSL probe) vs `paper_argmax` (legacy supervised)",
        "- `paired_test_n`: test rows after inner-join when present in source JSON",
        "- `thesis_role`: see classification rules below",
        "- Legacy supervised **canonical AUPRC = 0.292** from `eval_..._100ep_seed1.json` (not summary JSON 0.260)",
        "- Small-LI seed1 pre-3h +raw: **0.0818** in multiseed aggregate; **0.0829** only in emb198 paired join",
        "",
        THESIS_ROLE_RULES_MD.strip(),
        "",
        "---",
        "",
        "## Current headline results",
        "",
        "Frozen linear probe (`cw=model`, C=1.0) unless noted. "
        "Pre-vs-post tables use **paired** strong-run JSON (`pre3h_strong_run_comparison.json`).",
        "",
        "### Small-HI embedding-only SSL",
        "",
        "#### Best current (40ep seed2, strong-run paired)",
    ]
    emb_best_pre = _pick_row(
        rows,
        dataset="Small-HI",
        run_id="gin_emlps_tds_asym_proj_8192neg_queue0_40ep_seed2|embedding|pre_embedding_3h",
        source_json=CANONICAL_PREPOST_SOURCE,
    )
    emb_best_post = _pick_row(
        rows,
        dataset="Small-HI",
        run_id="gin_emlps_tds_asym_proj_8192neg_queue0_40ep_seed2|embedding|post_embedding_128",
        source_json=CANONICAL_PREPOST_SOURCE,
    )
    lines.append(_fmt_row(emb_best_pre))
    lines.append("")
    lines.append("Post-128 same run: AUPRC **{0}** F1 **{1}** (paired n={2})".format(
        emb_best_post.get("AUPRC") if emb_best_post else "?",
        emb_best_post.get("F1") if emb_best_post else "?",
        int(emb_best_post.get("paired_test_n") or 0) if emb_best_post else "?",
    ))
    lines.extend(["", "#### Historical 20ep seed1 baseline (post-128)"])
    baseline = _pick_row(
        rows,
        dataset="Small-HI",
        run_id="hi_contrastive_gin_emlps_tds_proj_asym_8192neg_queue0_accum4_20ep|embedding|post_embedding_128",
    ) or _pick_best(rows, "AUPRC", dataset="Small-HI", probe_feature_stack="embedding",
                    representation_source="post_embedding_128", training_epochs=20, seed=1)
    lines.append(_fmt_row(baseline))
    lines.extend([
        "",
        "### Small-HI +raw F1: pre-3h vs post-128 (canonical paired comparison)",
        "",
        "Both from `{0}`, gin 40ep seed2, val-tuned F1, paired n≈862914.".format(
            CANONICAL_PREPOST_SOURCE),
        "",
    ])
    hi_raw_pre = _pick_row(
        rows,
        run_id="gin_emlps_tds_asym_proj_8192neg_queue0_40ep_seed2|embedding+raw|pre_embedding_3h",
        source_json=CANONICAL_PREPOST_SOURCE,
    )
    hi_raw_post = _pick_row(
        rows,
        run_id="gin_emlps_tds_asym_proj_8192neg_queue0_40ep_seed2|embedding+raw|post_embedding_128",
        source_json=CANONICAL_PREPOST_SOURCE,
    )
    hi_raw_ablation = _pick_row(
        rows,
        run_id="gin_emlps_tds_asym_proj_8192neg_queue0_40ep_seed2|embedding+raw|post_embedding_128",
        source_json="results/diagnostics/probe_feature_ablation_current_protocol_comparison.json",
    )
    lines.append("**Canonical paired (use in main pre/post table):**")
    lines.append("- pre-3h +raw: AUPRC **{0}**, F1 **{1}**".format(
        hi_raw_pre.get("AUPRC"), hi_raw_pre.get("F1")))
    lines.append("- post-128 +raw: AUPRC **{0}**, F1 **{1}**".format(
        hi_raw_post.get("AUPRC"), hi_raw_post.get("F1")))
    lines.append("")
    lines.append("**Interpretation:** pre-3h has stronger **AUPRC** (+0.037) and slightly higher **tuned F1** (+0.001) under the *same paired rows*. The higher post-128 F1 **0.347** appears only in the non-paired feature-ablation file (full test n≈863050).")
    lines.append("")
    lines.append("**Non-paired post-128 reference** (`probe_feature_ablation_current_protocol_comparison.json`, full test n≈863050): F1 **{0}**, AUPRC **{1}** — not comparable row-for-row to paired pre-3h.".format(
        hi_raw_ablation.get("F1") if hi_raw_ablation else "?",
        hi_raw_ablation.get("AUPRC") if hi_raw_ablation else "?",
    ))
    lines.extend([
        "",
        "### Small-HI FNF full-stack alert-budget (recovered from strong-run JSON)",
        "",
    ])
    fnf_post = _pick_row(
        rows,
        run_id="same_pair_fnf_emlps_tds_asym_proj_8192neg_queue0_20ep|embedding+raw+morph|post_embedding_128",
        source_json=CANONICAL_PREPOST_SOURCE,
    )
    fnf_pre = _pick_row(
        rows,
        run_id="same_pair_fnf_emlps_tds_asym_proj_8192neg_queue0_20ep|embedding+raw+morph|pre_embedding_3h",
        source_json=CANONICAL_PREPOST_SOURCE,
    )
    if fnf_post:
        lines.append("- **post-128:** P@100={0}, lift@100={1:.1f} | source `{2}`".format(
            fnf_post.get("precision_at_100"), fnf_post.get("lift_at_100"),
            CANONICAL_PREPOST_SOURCE))
    if fnf_pre:
        lines.append("- **pre-3h:** P@100={0}, lift@100={1:.1f}".format(
            fnf_pre.get("precision_at_100"), fnf_pre.get("lift_at_100")))
    lines.extend([
        "",
        "### Small-LI multiseed aggregates (mean ± sample SD, n=3, ddof=1)",
        "",
        "Convention: **mean ± sample standard deviation** over seeds 1–3.",
        "",
    ])
    for stack, block in aggregates.get("stacks", {}).items():
        lines.append("#### `{0}`".format(stack))
        for rep, rep_data in block.get("representations", {}).items():
            a = rep_data.get("auroc", {})
            p = rep_data.get("auprc", {})
            r = rep_data.get("recall_at_100", {})
            lines.append("- **{0}** AUROC {1:.4f}±{2:.4f} | AUPRC {3:.4f}±{4:.4f} | R@100 {5:.4f}±{6:.4f}".format(
                rep, a.get("mean", 0), a.get("std", 0),
                p.get("mean", 0), p.get("std", 0),
                r.get("mean", 0), r.get("std", 0),
            ))
        d = block.get("deltas_pre_minus_post", {})
        if d:
            lines.append("- Δpre-post: " + ", ".join(
                "{0} mean={1:.4f}±{2:.4f}".format(k, v["mean"], v["std"])
                for k, v in d.items()
            ))
        lines.append("")
    rc = aggregates.get("reporting_conventions", {})
    lines.append("Seed1 +raw pre-3h AUPRC: multiseed **{0:.4f}** | emb198 paired **{1}**".format(
        rc.get("seed1_pre3h_plus_raw_auprc_multiseed", 0),
        rc.get("seed1_pre3h_plus_raw_auprc_emb198_paired"),
    ))
    lines.extend([
        "",
        "### Legacy supervised Small-LI (formal)",
        "",
    ])
    legacy = _pick_row(rows, scout_or_formal="formal", threshold_rule="paper_argmax")
    lines.append(_fmt_row(legacy))
    lines.extend([
        "",
        "## Duplicate resolution",
        "",
        "Rows sharing `run_id|rep|stack|threshold` across sources:",
        "",
    ])
    resolutions = {
        "canonical_multiseed_source": "Small-LI multiseed JSON is canonical; emb198 rows are **not** duplicates (different join).",
        "canonical_prepost_source": "Strong-run JSON is canonical for HI pre/post and FNF alert-budget; ablation JSON is cross-source replication.",
        "pairing_intersection_diff": "Same config, different paired row intersection (multiseed vs emb198).",
        "cross_source_replication": "Same metrics re-ingested from a second diagnostic file; keep both for provenance.",
    }
    for res, desc in resolutions.items():
        n = len(dup_groups.get(res, []))
        if n:
            lines.append("- **{0}** ({1} rows): {2}".format(res, n, desc))
    lines.extend([
        "",
        "Distinct evaluations (same checkpoint, different stack/rep/threshold) are **not** duplicates.",
        "",
        "## Optional / not currently planned",
        "",
        "- **emb198 multiseed replication** — one-seed scout did not beat orig pre-3h; pre-3h replicates 3/3 on Small-LI without emb198.",
        "",
        "## Remaining ambiguities",
        "",
        "1. HI 40ep seed2: `training_epochs=40`, `selected_epoch=36` (best ckpt) — both fields preserved.",
        "2. Feature ablation F1 on full test vs paired strong-run — use strong-run for pre/post tables only.",
        "3. PNA capacity-matched comparison still pending (separate workstream).",
        "",
    ])
    (ROOT / "notes" / "thesis_experiment_registry.md").write_text("\n".join(lines), encoding="utf-8")


def write_table_plan_md(aggregates):
    stacks = aggregates.get("stacks", {})
    emb = stacks.get("embedding", {})
    raw = stacks.get("embedding+raw", {})

    def _ms(rep, metric):
        d = stacks.get("embedding+raw" if metric.startswith("+") else "embedding", {})
        # fix - use proper stack from caller
        return 0, 0

    def fmt(rep, stack_name, metric_key):
        b = stacks.get(stack_name, {}).get("representations", {}).get(rep, {})
        m = b.get(metric_key, {})
        return "{0:.3f} ± {1:.3f}".format(m.get("mean", 0), m.get("std", 0))

    text = """# Thesis results table plan

Draft table structure organized by scientific question. Populated cells cite registry rows or named JSON sources; **—** means not available (do not infer).

**Registry:** `results/diagnostics/thesis_experiment_registry.csv`

---

## Table 1 — Dataset and task summary

**Placement:** Main

| dataset | train | val | test | pos (test) | prev (test) | split | source |
|---------|------:|----:|-----:|-----------:|------------:|-------|--------|
| Small-HI | 3,248,254 | 965,466 | 863,050 | 1,611 | 0.187% | calendar_day | `pre_embedding_3h_vs_post_embedding_small_hi.json` |
| Small-LI | 4,432,934 | 1,316,442 | 1,174,673 | 802 | 0.068% | calendar_day | `small_li_dataset_audit.json` |

---

## Table 2 — Main Small-HI comparison

**Placement:** Main | **Canonical pre/post source:** `pre3h_strong_run_comparison.json` (paired n≈862914 for 40ep s2)

| row | AUROC | AUPRC | F1 | P@100 | lift@100 | source |
|-----|------:|------:|---:|------:|---------:|--------|
| raw+morph baseline (no SSL) | 0.905 | 0.066 | 0.136 | — | — | feature ablation 20ep s1 |
| SSL post-128 embedding (20ep s1 baseline) | 0.944 | **0.213** | 0.259 | 0.85 | 455 | alert_budget / architecture |
| SSL post-128 embedding (40ep s2) | 0.949 | 0.245 | 0.304 | 0.80 | 429 | strong-run paired |
| SSL pre-3h embedding (40ep s2) | 0.958 | **0.295** | 0.340 | 0.83 | 445 | strong-run paired |
| SSL post-128 +raw (40ep s2) | 0.955 | 0.284 | 0.343 | 0.79 | 423 | strong-run paired |
| SSL pre-3h +raw (40ep s2) | **0.960** | **0.321** | 0.344 | 0.84 | 450 | strong-run paired |
| FNF full stack post-128 | 0.959 | 0.277 | 0.320 | **0.80** | **429** | strong-run paired |
| FNF full stack pre-3h | 0.968 | 0.291 | 0.314 | 0.73 | **391** | strong-run paired |

**Footnote (F1):** Under paired protocol, pre-3h +raw wins AUPRC (+0.037) and F1 (+0.001): 0.344 vs 0.343. Non-paired ablation post-128 F1=0.347 uses full test n≈863050 — cite separately, not in pre/post table.

**Thesis-critical missing:** Small-HI supervised baseline.

**Optional / not planned:** emb198 multiseed (see registry).

---

## Table 3 — Main Small-LI comparison

**Placement:** Main | Aggregates from multiseed JSON (n=3). **Mean ± sample standard deviation (ddof=1)** over three seeds.

| row | AUROC | AUPRC | F1 | P@100 | R@100 | lift@100 |
|-----|------:|------:|---:|------:|------:|---------:|
| legacy supervised (paper_argmax ep35) | 0.959 | **0.292** | **0.357** | 0.97 | 0.121 | 1419 |
| SSL post-128 +raw mean ± sample SD | {post_raw_auroc} | {post_raw_auprc} | {post_raw_f1} | {post_raw_p100} | {post_raw_r100} | {post_raw_lift} |
| SSL pre-3h +raw mean ± sample SD | {pre_raw_auroc} | {pre_raw_auprc} | {pre_raw_f1} | {pre_raw_p100} | {pre_raw_r100} | {pre_raw_lift} |
| SSL post-128 embedding mean ± sample SD | {post_emb_auroc} | {post_emb_auprc} | — | — | {post_emb_r100} | — |
| SSL pre-3h embedding mean ± sample SD | {pre_emb_auroc} | {pre_emb_auprc} | — | — | {pre_emb_r100} | — |

Δpre-post (+raw): ΔAUPRC **{d_auprc}**, ΔR@100 **{d_r100}**

Seed1 +raw pre-3h AUPRC: **0.0818** (multiseed); 0.0829 emb198 paired join only.

**Footnote:** Legacy F1 is paper_argmax; SSL F1 is val-tuned probe — do not mix without footnote.

---

## Table 4 — Representation-source ablation

**Placement:** Main summary + Appendix detail (unchanged structure; use strong-run HI + multiseed LI)

---

## Table 5 — Architecture ablation

**Placement:** Appendix (PNA not capacity-matched; fairness scout pending)

---

## Table 6 — Contrastive ablations

**Placement:** Appendix | degree_aware → negative_result

---

## Cross-table notes

| Topic | Status |
|-------|--------|
| FNF HI alert-budget | **Recovered** from `pre3h_strong_run_comparison.json` |
| emb198 multiseed | **Optional / not currently planned** |
| Legacy AUPRC | Use eval JSON **0.292** |
""".format(
        post_raw_auroc=fmt("post_embedding_128", "embedding+raw", "auroc"),
        post_raw_auprc=fmt("post_embedding_128", "embedding+raw", "auprc"),
        post_raw_f1=fmt("post_embedding_128", "embedding+raw", "f1_at_selected_threshold"),
        post_raw_p100=fmt("post_embedding_128", "embedding+raw", "precision_at_100"),
        post_raw_r100=fmt("post_embedding_128", "embedding+raw", "recall_at_100"),
        post_raw_lift=fmt("post_embedding_128", "embedding+raw", "lift_at_100"),
        pre_raw_auroc=fmt("pre_embedding_3h", "embedding+raw", "auroc"),
        pre_raw_auprc=fmt("pre_embedding_3h", "embedding+raw", "auprc"),
        pre_raw_f1=fmt("pre_embedding_3h", "embedding+raw", "f1_at_selected_threshold"),
        pre_raw_p100=fmt("pre_embedding_3h", "embedding+raw", "precision_at_100"),
        pre_raw_r100=fmt("pre_embedding_3h", "embedding+raw", "recall_at_100"),
        pre_raw_lift=fmt("pre_embedding_3h", "embedding+raw", "lift_at_100"),
        post_emb_auroc=fmt("post_embedding_128", "embedding", "auroc"),
        post_emb_auprc=fmt("post_embedding_128", "embedding", "auprc"),
        post_emb_r100=fmt("post_embedding_128", "embedding", "recall_at_100"),
        pre_emb_auroc=fmt("pre_embedding_3h", "embedding", "auroc"),
        pre_emb_auprc=fmt("pre_embedding_3h", "embedding", "auprc"),
        pre_emb_r100=fmt("pre_embedding_3h", "embedding", "recall_at_100"),
        d_auprc="{0:.4f}±{1:.4f}".format(
            raw.get("deltas_pre_minus_post", {}).get("delta_auprc_pre_minus_post", {}).get("mean", 0),
            raw.get("deltas_pre_minus_post", {}).get("delta_auprc_pre_minus_post", {}).get("std", 0),
        ),
        d_r100="{0:.4f}±{1:.4f}".format(
            raw.get("deltas_pre_minus_post", {}).get("delta_recall_at_100_pre_minus_post", {}).get("mean", 0),
            raw.get("deltas_pre_minus_post", {}).get("delta_recall_at_100_pre_minus_post", {}).get("std", 0),
        ),
    )
    (ROOT / "notes" / "thesis_results_table_plan.md").write_text(text, encoding="utf-8")


def main():
    rows = build_registry()
    aggregates = compute_multiseed_aggregates()
    tf_aggregates = compute_temporal_flow_multiseed_aggregates()
    dataset_metadata = build_dataset_metadata()
    pending_sources = collect_pending_sources()
    tf_validation = _temporal_flow_validation_state()
    out_j = DIAG / "thesis_experiment_registry.json"
    out_c = DIAG / "thesis_experiment_registry.csv"
    payload = {
        "row_count": len(rows),
        "fields": FIELDS,
        "multiseed_aggregates": aggregates,
        "temporal_flow_multiseed_aggregates": tf_aggregates,
        "dataset_metadata": dataset_metadata,
        "temporal_flow_validation": tf_validation,
        "pending_sources": pending_sources,
        "thesis_role_counts": dict(
            (k, sum(1 for r in rows if r.get("thesis_role") == k))
            for k in sorted({r.get("thesis_role") for r in rows})
        ),
        "validation_status_counts": dict(
            (k, sum(1 for r in rows if r.get("validation_status") == k))
            for k in sorted({r.get("validation_status") for r in rows})
        ),
        "rows": rows,
    }
    out_j.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with out_c.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if v is None else v) for k, v in r.items()})
    write_registry_md(rows, aggregates)
    write_table_plan_md(aggregates)
    print("Wrote {0} rows".format(len(rows)))
    if pending_sources:
        print("Pending optional sources ({0}):".format(len(pending_sources)))
        for p in pending_sources:
            print("  - {0}".format(p))
    print("Temporal-flow validation passed: {0}".format(tf_validation.get("passed")))


if __name__ == "__main__":
    main()
