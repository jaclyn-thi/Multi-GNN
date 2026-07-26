"""
Edge-level downstream dataset specifications.

Each spec describes how to load ``formatted_transactions.csv`` for a dataset
folder under ``data_config.json`` → ``paths.aml_data``. Formatters (e.g.
``format_paysim.py``) must emit the shared CSV schema so the GNN pipeline and
frozen-embedding probes stay unchanged.

PaySim is the first non-AMLWorld adapter; IBM AML splits (``Small-HI``, etc.)
use the default AMLWorld spec.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Sequence, Tuple


# Columns expected in every formatted_transactions.csv (see format_kaggle_files.py).
FORMATTED_TRANSACTION_COLUMNS: Tuple[str, ...] = (
    "EdgeID",
    "from_id",
    "to_id",
    "Timestamp",
    "Amount Sent",
    "Sent Currency",
    "Amount Received",
    "Received Currency",
    "Payment Format",
    "Is Laundering",
)

# Edge attributes consumed by models.py / get_data (after optional ports/tds).
DEFAULT_EDGE_FEATURE_COLS: Tuple[str, ...] = (
    "Timestamp",
    "Amount Received",
    "Received Currency",
    "Payment Format",
)

# Historical PaySim notes (documentation only). Runtime behavior is controlled by
# ``--feature_contract`` (see feature_contracts.py). Omitting the flag preserves
# the legacy_duplicate mapping described below.


@dataclass(frozen=True)
class EdgeDatasetSpec:
    """
    Configuration for loading one edge-classification dataset.

    Parameters
    ----------
    name :
        Folder name under ``aml_data`` (matches ``--data``).
    label_col :
        Binary edge label column in the formatted CSV. PaySim maps ``isFraud``
        into ``Is Laundering`` at format time so probes need no code change.
    edge_feature_cols :
        Numeric edge channels passed to the encoder (must match pretrain
        ``edge_dim`` when loading a frozen AML checkpoint — 4 cols by default).
    split_mode :
        ``calendar_day`` — bucket ``Timestamp`` into 86400s days (AMLWorld).
        ``hourly_step`` — bucket ``Timestamp // 3600`` (PaySim; formatter sets
        ``Timestamp = step * 3600``).
    split_fractions :
        Target train / val / test proportions (temporal order preserved).
    supports_pattern_metadata :
        If True, auto-load ``laundering_attempt_metadata.csv`` when present.
    feature_mapping :
        Human-readable notes for ``meta.json`` / thesis (harmonization policy).
    """

    name: str
    label_col: str = "Is Laundering"
    edge_feature_cols: Tuple[str, ...] = DEFAULT_EDGE_FEATURE_COLS
    split_mode: str = "calendar_day"
    split_fractions: Tuple[float, float, float] = (0.6, 0.2, 0.2)
    supports_pattern_metadata: bool = True
    feature_mapping: Dict[str, str] = field(default_factory=dict)

    def formatted_csv_name(self) -> str:
        return "formatted_transactions.csv"

    def validate_edge_feature_cols(self) -> None:
        if len(self.edge_feature_cols) != len(DEFAULT_EDGE_FEATURE_COLS):
            raise ValueError(
                f"{self.name}: edge_feature_cols must have "
                f"{len(DEFAULT_EDGE_FEATURE_COLS)} columns for checkpoint "
                f"compatibility (got {len(self.edge_feature_cols)})."
            )


PAYSIM_FEATURE_MAPPING: Dict[str, str] = {
    "Timestamp": "PaySim step * 3600 (synthetic seconds; 1 step = 1 hour)",
    "Amount Received": "PaySim amount",
    "Received Currency": "Integer code for PaySim type (PAYMENT, TRANSFER, ...)",
    "Payment Format": "Same type code as Received Currency (AML schema placeholder)",
    "Is Laundering": "PaySim isFraud (0/1); column name kept for pipeline compatibility",
    "excluded": "oldbalance*, newbalance*, isFlaggedFraud (leakage / not in AML pretrain)",
}


PAYSIM_SPEC = EdgeDatasetSpec(
    name="PaySim",
    label_col="Is Laundering",
    edge_feature_cols=DEFAULT_EDGE_FEATURE_COLS,
    split_mode="hourly_step",
    split_fractions=(0.6, 0.2, 0.2),
    supports_pattern_metadata=False,
    feature_mapping=PAYSIM_FEATURE_MAPPING,
)

AMLWORLD_DEFAULT_SPEC = EdgeDatasetSpec(
    name="AMLWorld",
    label_col="Is Laundering",
    edge_feature_cols=DEFAULT_EDGE_FEATURE_COLS,
    split_mode="calendar_day",
    split_fractions=(0.6, 0.2, 0.2),
    supports_pattern_metadata=True,
    feature_mapping={
        "Timestamp": "Seconds from dataset start (AMLWorld formatter)",
        "Amount Received": "Native AML amount",
        "Received Currency": "AML currency code",
        "Payment Format": "AML payment format code",
        "Is Laundering": "Native AML edge label",
    },
)

# Explicit adapters; all other ``--data`` values use AMLWORLD_DEFAULT_SPEC with
# ``name`` overridden to the folder name (Small-HI, Small-LI, ...).
_DATASET_REGISTRY: Dict[str, EdgeDatasetSpec] = {
    "PaySim": PAYSIM_SPEC,
}


def get_dataset_spec(data_name: str) -> EdgeDatasetSpec:
    """
    Resolve the loading spec for ``--data {data_name}``.

    Unknown names default to AMLWorld conventions (same as Small-HI today).
    """
    if data_name in _DATASET_REGISTRY:
        return _DATASET_REGISTRY[data_name]
    return EdgeDatasetSpec(
        name=data_name,
        label_col=AMLWORLD_DEFAULT_SPEC.label_col,
        edge_feature_cols=AMLWORLD_DEFAULT_SPEC.edge_feature_cols,
        split_mode=AMLWORLD_DEFAULT_SPEC.split_mode,
        split_fractions=AMLWORLD_DEFAULT_SPEC.split_fractions,
        supports_pattern_metadata=AMLWORLD_DEFAULT_SPEC.supports_pattern_metadata,
        feature_mapping=dict(AMLWORLD_DEFAULT_SPEC.feature_mapping),
    )


def list_registered_datasets() -> Sequence[str]:
    return sorted(_DATASET_REGISTRY.keys())


def spec_summary(spec: EdgeDatasetSpec) -> Mapping[str, object]:
    """JSON-serializable summary for extraction ``meta.json``."""
    return {
        "dataset": spec.name,
        "label_col": spec.label_col,
        "edge_feature_cols": list(spec.edge_feature_cols),
        "split_mode": spec.split_mode,
        "split_fractions": list(spec.split_fractions),
        "supports_pattern_metadata": spec.supports_pattern_metadata,
        "feature_mapping": dict(spec.feature_mapping),
    }
