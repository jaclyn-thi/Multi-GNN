"""
Load AMLWorld laundering-pattern metadata keyed by CSV ``EdgeID``.

Auxiliary metadata for diagnostics/evaluation only — does not modify
``formatted_transactions.csv`` or ``Is Laundering`` labels.

Loaded once in ``data_loading.get_data()`` when
``aml-data/{dataset}/laundering_attempt_metadata.csv`` exists (or when
``--load_pattern_metadata`` / ``--pattern_metadata`` is set). Lookup via
``te_data.pattern_metadata_by_edge_id[edge_id]`` using CSV ``EdgeID`` values
from ``te_data.csv_edge_ids``.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Union

import pandas as pd

REQUIRED_METADATA_COLUMNS = (
    "attempt_id",
    "pattern_type",
    "pattern_detail",
    "tx_idx_in_attempt",
    "n_tx_in_attempt",
    "EdgeID",
    "join_status",
)


@dataclass(frozen=True)
class LaunderingPatternMetadata:
    attempt_id: int
    pattern_type: str
    pattern_detail: Optional[str]
    tx_idx_in_attempt: int
    n_tx_in_attempt: int


def default_pattern_metadata_path(data_config: Mapping[str, object], dataset: str) -> Path:
    aml_data = data_config["paths"]["aml_data"]
    return Path(str(aml_data)) / dataset / "laundering_attempt_metadata.csv"


def resolve_pattern_metadata_path(
    data_config: Mapping[str, object],
    dataset: str,
    *,
    explicit_path: Optional[str] = None,
    load_requested: bool = False,
) -> Optional[Path]:
    if explicit_path:
        return Path(explicit_path)
    default = default_pattern_metadata_path(data_config, dataset)
    if load_requested or default.exists():
        return default
    return None


def load_laundering_pattern_metadata(
    metadata_path: Path,
    df_edges: pd.DataFrame,
) -> Dict[int, LaunderingPatternMetadata]:
    """
    Load pattern metadata and return ``EdgeID -> LaunderingPatternMetadata``.

    Validates duplicate ``EdgeID``s, checks metadata IDs against ``df_edges``,
    and logs pattern-type counts plus laundering-edge coverage.
    """
    if not metadata_path.exists():
        logging.info("Laundering pattern metadata not found at %s (skipping)", metadata_path)
        return {}

    df_meta = pd.read_csv(metadata_path)
    missing_cols = [c for c in REQUIRED_METADATA_COLUMNS if c not in df_meta.columns]
    if missing_cols:
        raise ValueError(f"{metadata_path} missing columns: {missing_cols}")

    df_meta = df_meta[df_meta["join_status"] == "matched"].copy()
    if df_meta.empty:
        logging.warning("No matched rows in %s; pattern metadata disabled", metadata_path)
        return {}

    edge_ids = df_meta["EdgeID"].astype(int)
    dup_mask = edge_ids.duplicated(keep=False)
    if dup_mask.any():
        dup_ids = sorted(edge_ids[dup_mask].unique().tolist())
        raise ValueError(
            f"Duplicate EdgeID(s) in {metadata_path}: {dup_ids[:20]}"
            + (f" ... ({len(dup_ids)} total)" if len(dup_ids) > 20 else "")
        )

    formatted_edge_ids = set(df_edges["EdgeID"].astype(int).tolist())
    meta_edge_ids = set(edge_ids.tolist())
    missing_in_formatted = sorted(meta_edge_ids - formatted_edge_ids)
    if missing_in_formatted:
        raise ValueError(
            f"{len(missing_in_formatted)} metadata EdgeID(s) missing from "
            f"formatted_transactions.csv (first few: {missing_in_formatted[:10]})"
        )

    pattern_type_counts = Counter(df_meta["pattern_type"].astype(str).tolist())
    logging.info(
        "Loaded laundering pattern metadata from %s (%d matched rows, %d unique EdgeIDs)",
        metadata_path,
        len(df_meta),
        len(meta_edge_ids),
    )
    logging.info("Pattern metadata counts by pattern_type:")
    for pattern_type, count in sorted(pattern_type_counts.items()):
        logging.info("  %s: %d", pattern_type, count)

    laundering_mask = df_edges["Is Laundering"].astype(int) == 1
    laundering_total = int(laundering_mask.sum())
    laundering_edge_ids = set(df_edges.loc[laundering_mask, "EdgeID"].astype(int).tolist())
    laundering_with_pattern = laundering_edge_ids & meta_edge_ids
    laundering_without_pattern = laundering_total - len(laundering_with_pattern)
    logging.info(
        "Laundering edges in formatted_transactions: %d with pattern metadata, "
        "%d without known pattern metadata",
        len(laundering_with_pattern),
        laundering_without_pattern,
    )

    mapping: Dict[int, LaunderingPatternMetadata] = {}
    for row in df_meta.itertuples(index=False):
        detail = str(row.pattern_detail).strip()
        mapping[int(row.EdgeID)] = LaunderingPatternMetadata(
            attempt_id=int(row.attempt_id),
            pattern_type=str(row.pattern_type),
            pattern_detail=detail if detail else None,
            tx_idx_in_attempt=int(row.tx_idx_in_attempt),
            n_tx_in_attempt=int(row.n_tx_in_attempt),
        )
    return mapping


def lookup_pattern_metadata(
    edge_id: Union[int, pd.Series, pd.Index],
    metadata_by_edge_id: Mapping[int, LaunderingPatternMetadata],
) -> Optional[LaunderingPatternMetadata]:
    """Return metadata for a CSV ``EdgeID``, or ``None`` if unknown."""
    return metadata_by_edge_id.get(int(edge_id))
