"""Shared-core edge contract for Small-HI + SAML-D mixed SSL (Phase 1).

Contract ID: ``smallhi_samld_shared_core_v1``

Final model-input order (edge_dim=6):
  [Timestamp, Amount Received, in_port, out_port, in_td, out_td]

This is NOT the historical supervised Multi-GIN ports-only edge_dim=6 geometry.
Categorical currency/payment slots and labels are excluded from encoder inputs.
Temporal-flow features remain prediction targets only (never concatenated here).
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

CONTRACT_SHARED_CORE_V1 = "smallhi_samld_shared_core_v1"
SHARED_CORE_DATASETS: Tuple[str, ...] = ("Small-HI", "SAML-D")

SHARED_CORE_BASE_FEATURE_NAMES: Tuple[str, ...] = (
    "Timestamp",
    "Amount Received",
)
SHARED_CORE_EXCLUDED_CATEGORICALS: Tuple[str, ...] = (
    "Received Currency",
    "Payment Format",
    "Sent Currency",
)
SHARED_CORE_FINAL_FEATURE_NAMES: Tuple[str, ...] = (
    "Timestamp",
    "Amount Received",
    "in_port",
    "out_port",
    "in_td",
    "out_td",
)
SHARED_CORE_EXPECTED_BASE_DIM = 2
SHARED_CORE_EXPECTED_EDGE_DIM = 6  # base2 + ports2 + tds2

# Historical supervised Multi-GIN ports-only geometry (DO NOT ALIAS).
HISTORICAL_SUPERVISED_PORTS_ONLY_EDGE_DIM = 6
HISTORICAL_SUPERVISED_PORTS_ONLY_NOTE = (
    "Historical supervised Multi-GIN+EU often used edge_dim=6 = base4 + ports2 "
    "(TDS off). smallhi_samld_shared_core_v1 is a different contract: base2 + "
    "ports2 + tds2 with categoricals dropped."
)


def is_shared_core_contract(contract_id: Optional[str]) -> bool:
    return contract_id is not None and str(contract_id) == CONTRACT_SHARED_CORE_V1


def assert_dataset_allowed(data: str) -> None:
    if data not in SHARED_CORE_DATASETS:
        raise ValueError(
            f"{CONTRACT_SHARED_CORE_V1} supports only {SHARED_CORE_DATASETS}; got --data={data!r}"
        )


def shared_core_summary(*, dataset: str) -> Dict[str, Any]:
    return {
        "feature_contract_id": CONTRACT_SHARED_CORE_V1,
        "description": (
            "Small-HI + SAML-D mixed-SSL shared core: Timestamp + Amount Received "
            "+ ports + TDS; categoricals and labels excluded from encoder inputs."
        ),
        "dataset": dataset,
        "supported_datasets": list(SHARED_CORE_DATASETS),
        "base_feature_names": list(SHARED_CORE_BASE_FEATURE_NAMES),
        "final_feature_names": list(SHARED_CORE_FINAL_FEATURE_NAMES),
        "excluded_categoricals": list(SHARED_CORE_EXCLUDED_CATEGORICALS),
        "labels_in_encoder_inputs": False,
        "temporal_flow_as_encoder_input": False,
        "expected_base_dim": SHARED_CORE_EXPECTED_BASE_DIM,
        "expected_edge_dim_with_ports_tds": SHARED_CORE_EXPECTED_EDGE_DIM,
        "not_historical_supervised_ports_only_dim6": True,
        "historical_supervised_ports_only_note": HISTORICAL_SUPERVISED_PORTS_ONLY_NOTE,
        "normalization": "train_fit_edge_znorm_per_dataset",
        "preserve_seed_edges": False,
        "projection": "off_for_later_trainer",
    }


def select_shared_core_base_edge_attr(
    edge_attr: torch.Tensor,
    *,
    source_feature_names: Sequence[str],
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Select Timestamp + Amount Received from a base edge_attr matrix.

    Accepts either already-base-2 columns or the standard base-4 AML columns
    ``[Timestamp, Amount Received, Received Currency, Payment Format]``.
    """
    names = list(source_feature_names)
    if edge_attr.dim() != 2:
        raise ValueError(f"edge_attr must be 2-D, got {tuple(edge_attr.shape)}")
    if edge_attr.shape[1] != len(names):
        raise ValueError(
            f"edge_attr width {edge_attr.shape[1]} != len(source_feature_names)={len(names)}"
        )
    try:
        idx_ts = names.index("Timestamp")
        idx_amt = names.index("Amount Received")
    except ValueError as exc:
        raise ValueError(
            f"{CONTRACT_SHARED_CORE_V1} requires Timestamp and Amount Received in "
            f"source columns; got {names}"
        ) from exc

    for cat in SHARED_CORE_EXCLUDED_CATEGORICALS:
        if cat in ("Received Currency", "Payment Format") and cat in names:
            # Allowed in source; must not remain after selection.
            pass

    out = torch.stack([edge_attr[:, idx_ts], edge_attr[:, idx_amt]], dim=1).contiguous()
    if out.shape[1] != SHARED_CORE_EXPECTED_BASE_DIM:
        raise RuntimeError(f"shared-core base width {out.shape[1]} != 2")
    meta = {
        "selected_indices": [idx_ts, idx_amt],
        "selected_names": list(SHARED_CORE_BASE_FEATURE_NAMES),
        "dropped_source_names": [n for n in names if n not in SHARED_CORE_BASE_FEATURE_NAMES],
    }
    return out, meta


def assert_shared_core_final_schema(
    edge_attr: torch.Tensor,
    *,
    feature_names: Sequence[str],
    ports: bool,
    tds: bool,
) -> None:
    if not ports or not tds:
        raise ValueError(
            f"{CONTRACT_SHARED_CORE_V1} requires ports=True and tds=True "
            f"(got ports={ports} tds={tds})"
        )
    if list(feature_names) != list(SHARED_CORE_FINAL_FEATURE_NAMES):
        raise ValueError(
            f"{CONTRACT_SHARED_CORE_V1} feature name mismatch: "
            f"got {list(feature_names)} expected {list(SHARED_CORE_FINAL_FEATURE_NAMES)}"
        )
    if int(edge_attr.shape[1]) != SHARED_CORE_EXPECTED_EDGE_DIM:
        raise ValueError(
            f"{CONTRACT_SHARED_CORE_V1} edge_dim={edge_attr.shape[1]} != "
            f"{SHARED_CORE_EXPECTED_EDGE_DIM} "
            f"(do not confuse with historical supervised ports-only dim=6)"
        )
    for cat in SHARED_CORE_EXCLUDED_CATEGORICALS:
        if cat in feature_names:
            raise ValueError(f"Categorical {cat!r} must be absent from shared-core inputs")


def assert_no_label_column_in_features(feature_names: Sequence[str], label_col: str) -> None:
    if label_col in feature_names:
        raise ValueError(f"Label column {label_col!r} must not appear in encoder features")


def train_fit_scaler_provenance(
    mean: torch.Tensor,
    std: torch.Tensor,
    *,
    dataset: str,
    n_train_edges: int,
) -> Dict[str, Any]:
    """Hashable train-fit z-norm provenance (per dataset)."""
    m = mean.detach().cpu().numpy().astype(np.float64).reshape(-1)
    s = std.detach().cpu().numpy().astype(np.float64).reshape(-1)
    payload = np.concatenate([m, s]).tobytes()
    return {
        "policy": "train_fit_edge_znorm",
        "dataset": dataset,
        "feature_contract_id": CONTRACT_SHARED_CORE_V1,
        "feature_names": list(SHARED_CORE_FINAL_FEATURE_NAMES),
        "n_train_edges": int(n_train_edges),
        "mean": m.tolist(),
        "std": s.tolist(),
        "scaler_sha256": hashlib.sha256(payload).hexdigest(),
        "applied_to": ["train", "validation"],
        "test_not_used": True,
    }


def apply_train_fit_znorm_with_provenance(
    tr_edge_attr: torch.Tensor,
    val_edge_attr: torch.Tensor,
    *,
    dataset: str,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    mean = tr_edge_attr.mean(0)
    std = tr_edge_attr.std(0)
    std = torch.where(std == 0, torch.ones_like(std), std)
    prov = train_fit_scaler_provenance(
        mean, std, dataset=dataset, n_train_edges=int(tr_edge_attr.shape[0])
    )
    tr = (tr_edge_attr - mean) / std
    va = (val_edge_attr - mean) / std
    return tr, va, prov
