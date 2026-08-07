#!/usr/bin/env python3
"""Shared protocol tags/guards for DIRECT_R198 collaborator vs diagnostic eval.

Official collaborator-facing metrics MUST use:
  protocol == \"full_subgraph\"

Seed-only extracts are diagnostic/provisional only and must never merge into
collaborator tables.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROTOCOL_FULL_SUBGRAPH = "full_subgraph"
PROTOCOL_SEED_ONLY = "seed_only"

TIER_OFFICIAL = "official_collaborator"
TIER_DIAGNOSTIC = "diagnostic_provisional"

OFFICIAL_EMB_ROOT = "embeddings/direct_r198_40ep_linear_lr_full_extract"
OFFICIAL_OUT_DIR = "results/diagnostics/direct_r198_40ep_linear_lr_full_extract_reeval"
OFFICIAL_PKG_DIR = f"{OFFICIAL_OUT_DIR}/collaborator_package"
SEED_ONLY_SWEEP_DIR = "results/diagnostics/direct_r198_tfmoe_40ep_linear_lr_sweep"

PROBE_PROTOCOL = {
    "learner": "PaperStyleMLP",
    "mlp_epochs": 20,
    "mlp_lr": 1e-3,
    "mlp_batch_size": 8192,
    "mlp_seed": 2,
    "features": "R198 + edge X + temporal-flow cache",
    "selection_within_probe": "best_val_auprc",
    "final_bce_definition": "last_probe_epoch_mean_bce",
    "test_evaluated": False,
}

EXTRACTOR_FULL = "extract_direct_r198_full_cell"
EXTRACTOR_SEED_ONLY = "extract_direct_r198_seed_only_cell"


def official_protocol_block(
    *,
    extractor: str = EXTRACTOR_FULL,
    embeddings_dir: str = OFFICIAL_EMB_ROOT,
) -> Dict[str, Any]:
    return {
        "protocol": PROTOCOL_FULL_SUBGRAPH,
        "evaluation_tier": TIER_OFFICIAL,
        "collaborator_merge_allowed": True,
        "extractor_script": extractor,
        "embeddings_dir": embeddings_dir,
        "id_checks": [
            "train_val_intersect == 0",
            "val IDs above train max / no seed-only train-range signature",
            "Jaccard and relative-n agreement vs reference full extract",
        ],
        "probe": dict(PROBE_PROTOCOL),
        "seed_only_r198": False,
    }


def diagnostic_seed_only_protocol_block() -> Dict[str, Any]:
    return {
        "protocol": PROTOCOL_SEED_ONLY,
        "evaluation_tier": TIER_DIAGNOSTIC,
        "collaborator_merge_allowed": False,
        "extractor_script": EXTRACTOR_SEED_ONLY,
        "warning": (
            "DIAGNOSTIC / PROVISIONAL ONLY. Seed-only R198 extract is not the "
            "collaborator protocol. Do not merge into collaborator tables or "
            "present as official LR-grid metrics."
        ),
        "seed_only_r198": True,
        "probe_note": (
            "May use PaperStyleMLP, but extraction neighborhoods differ from "
            "full-subgraph; ID-fixed seed-only is still not official."
        ),
    }


def infer_protocol(cell: Dict[str, Any]) -> Optional[str]:
    """Return protocol tag, inferring legacy official cells when safe."""
    explicit = cell.get("protocol")
    if explicit in (PROTOCOL_FULL_SUBGRAPH, PROTOCOL_SEED_ONLY):
        return str(explicit)
    # Legacy full-subgraph cells stamped before protocol field existed.
    if cell.get("seed_only_r198") is False and cell.get("verify") and cell.get("status") == "ok":
        extractor = str(cell.get("extractor") or "")
        if "full" in extractor.lower() or extractor == "full_subgraph_run_embedding_extraction":
            return PROTOCOL_FULL_SUBGRAPH
    if cell.get("seed_only_r198") is True:
        return PROTOCOL_SEED_ONLY
    # Seed-only arm eval cells historically lacked seed_only_r198 / protocol.
    emb = str(cell.get("embedding_dir") or "")
    if "/embeddings/direct_r198_40ep_linear_lr_full_extract/" in emb.replace("\\", "/"):
        if cell.get("verify") and cell.get("status") == "ok":
            return PROTOCOL_FULL_SUBGRAPH
    if "pre_embedding_3h" in emb and "/embeddings/" in emb.replace("\\", "/"):
        # Default seed-only layout: embeddings/<run>_epochXX/pre_embedding_3h
        if "/embeddings/direct_r198_40ep_linear_lr_full_extract/" not in emb.replace("\\", "/"):
            if "coverage" in cell and "verify" not in cell:
                return PROTOCOL_SEED_ONLY
    return None


def assert_collaborator_merge_allowed(cell: Dict[str, Any], *, path: Optional[Path] = None) -> str:
    """Raise ValueError unless cell is official full_subgraph."""
    proto = infer_protocol(cell)
    where = f" ({path})" if path else ""
    if proto == PROTOCOL_SEED_ONLY:
        raise ValueError(
            f"Refusing collaborator merge{where}: protocol=seed_only "
            f"(diagnostic/provisional only)."
        )
    if proto != PROTOCOL_FULL_SUBGRAPH:
        raise ValueError(
            f"Refusing collaborator merge{where}: protocol must be "
            f"'{PROTOCOL_FULL_SUBGRAPH}', got {proto!r}."
        )
    if cell.get("collaborator_merge_allowed") is False:
        raise ValueError(f"Refusing collaborator merge{where}: collaborator_merge_allowed=false.")
    if cell.get("seed_only_r198") is True:
        raise ValueError(f"Refusing collaborator merge{where}: seed_only_r198=true.")
    verify = cell.get("verify") or {}
    if not verify.get("ok"):
        raise ValueError(f"Refusing collaborator merge{where}: verify.ok is not true.")
    if int(verify.get("train_val_intersect", -1)) != 0:
        raise ValueError(f"Refusing collaborator merge{where}: train∩val != 0.")
    return PROTOCOL_FULL_SUBGRAPH


def is_official_out_dir(path: Path, root: Path) -> bool:
    try:
        rel = path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return str(rel).startswith(OFFICIAL_OUT_DIR) or str(rel) == OFFICIAL_OUT_DIR


def is_official_emb_dir(path: Path, root: Path) -> bool:
    try:
        rel = path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return str(rel).startswith(OFFICIAL_EMB_ROOT)


def refuse_seed_only_write_into_official(
    *,
    out_dir: Path,
    embeddings_hint: Optional[Path],
    root: Path,
) -> None:
    if is_official_out_dir(out_dir, root):
        raise SystemExit(
            f"Refusing seed-only / diagnostic write into official out dir: {out_dir}\n"
            f"Official collaborator path is {OFFICIAL_OUT_DIR} and requires full_subgraph."
        )
    if embeddings_hint is not None and is_official_emb_dir(embeddings_hint, root):
        raise SystemExit(
            f"Refusing seed-only write into official embeddings root: {embeddings_hint}"
        )


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


def build_run_manifest(
    *,
    run: str,
    arm: str,
    peak_lr: float,
    epochs: List[int],
    protocol_block: Dict[str, Any],
    cells: List[Dict[str, Any]],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    manifest = {
        "run": run,
        "arm": arm,
        "peak_lr": peak_lr,
        "epochs": epochs,
        "protocol": protocol_block.get("protocol"),
        "evaluation_tier": protocol_block.get("evaluation_tier"),
        "collaborator_merge_allowed": protocol_block.get("collaborator_merge_allowed"),
        "protocol_block": protocol_block,
        "cells": cells,
    }
    if extra:
        manifest.update(extra)
    return manifest
