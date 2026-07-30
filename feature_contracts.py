"""Versioned financial edge-feature contracts (PaySim v1 adapters).

v1 preserves AMLWorld edge_dim=8 geometry:
  [Timestamp, Amount, Currency, PaymentFormat] + [in_port, out_port] + [in_td, out_td]

Neutral / missing semantic slots use raw constant 0.0 applied to base columns
after Timestamp re-zeroing and before ports, TDS, and edge z-normalization.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

# Base semantic slots (indices into base-4 edge_attr before ports/TDS).
SLOT_TIMESTAMP = 0
SLOT_AMOUNT = 1
SLOT_CURRENCY = 2
SLOT_PAYMENT_FORMAT = 3

BASE_SLOT_NAMES: Tuple[str, ...] = (
    "Timestamp",
    "Amount Received",
    "Received Currency",
    "Payment Format",
)

NEUTRAL_RAW_VALUE = 0.0
NEUTRAL_APPLICATION_POINT = "after_timestamp_rezero_before_ports_tds_znorm"

CONTRACT_LEGACY = "paysim_legacy_duplicate_v1"
CONTRACT_TYPE_ONLY = "paysim_type_only_v1"
CONTRACT_STRUCTURE_ONLY = "paysim_structure_only_v1"

PAYSIM_V1_CONTRACT_IDS: Tuple[str, ...] = (
    CONTRACT_LEGACY,
    CONTRACT_TYPE_ONLY,
    CONTRACT_STRUCTURE_ONLY,
)


@dataclass(frozen=True)
class FeatureContract:
    """Declarative PaySim→AMLWorld base-slot mapping for a fixed edge geometry."""

    contract_id: str
    description: str
    # Base-column index → policy: "type_code" | "neutral" | "passthrough"
    slot_policies: Mapping[int, str]
    raw_to_canonical: Mapping[str, str]
    neutral_raw_value: float = NEUTRAL_RAW_VALUE
    neutral_application_point: str = NEUTRAL_APPLICATION_POINT
    expected_base_dim: int = 4
    expected_edge_dim_with_ports_tds: int = 8

    def ordered_slots(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for i, name in enumerate(BASE_SLOT_NAMES):
            out.append(
                {
                    "index": i,
                    "name": name,
                    "policy": self.slot_policies.get(i, "passthrough"),
                }
            )
        return out

    def summary(self) -> Dict[str, Any]:
        return {
            "feature_contract_id": self.contract_id,
            "description": self.description,
            "ordered_slots": self.ordered_slots(),
            "raw_to_canonical_mapping": dict(self.raw_to_canonical),
            "neutral_policy": {
                "raw_value": self.neutral_raw_value,
                "application_point": self.neutral_application_point,
                "uses_nan": False,
                "drops_dimensions": False,
            },
            "expected_base_dim": self.expected_base_dim,
            "expected_edge_dim_with_ports_tds": self.expected_edge_dim_with_ports_tds,
        }


_REGISTRY: Dict[str, FeatureContract] = {
    CONTRACT_LEGACY: FeatureContract(
        contract_id=CONTRACT_LEGACY,
        description=(
            "Historical PaySim adapter: type codes duplicated into currency and "
            "payment-format slots (bit-exact with format_paysim.py defaults)."
        ),
        slot_policies={
            SLOT_TIMESTAMP: "passthrough",
            SLOT_AMOUNT: "passthrough",
            SLOT_CURRENCY: "type_code",
            SLOT_PAYMENT_FORMAT: "type_code",
        },
        raw_to_canonical={
            "Timestamp": "PaySim step*3600 (re-zeroed at load)",
            "Amount Received": "PaySim amount",
            "Received Currency": "PaySim type integer code (duplicate)",
            "Payment Format": "PaySim type integer code (duplicate)",
        },
    ),
    CONTRACT_TYPE_ONLY: FeatureContract(
        contract_id=CONTRACT_TYPE_ONLY,
        description=(
            "Currency slot neutralized; payment-format/type slot keeps PaySim type codes."
        ),
        slot_policies={
            SLOT_TIMESTAMP: "passthrough",
            SLOT_AMOUNT: "passthrough",
            SLOT_CURRENCY: "neutral",
            SLOT_PAYMENT_FORMAT: "type_code",
        },
        raw_to_canonical={
            "Timestamp": "PaySim step*3600 (re-zeroed at load)",
            "Amount Received": "PaySim amount",
            "Received Currency": "neutral/missing (raw 0.0)",
            "Payment Format": "PaySim type integer code",
        },
    ),
    CONTRACT_STRUCTURE_ONLY: FeatureContract(
        contract_id=CONTRACT_STRUCTURE_ONLY,
        description=(
            "Both categorical semantic slots neutralized; Timestamp/Amount + ports/TDS only."
        ),
        slot_policies={
            SLOT_TIMESTAMP: "passthrough",
            SLOT_AMOUNT: "passthrough",
            SLOT_CURRENCY: "neutral",
            SLOT_PAYMENT_FORMAT: "neutral",
        },
        raw_to_canonical={
            "Timestamp": "PaySim step*3600 (re-zeroed at load)",
            "Amount Received": "PaySim amount",
            "Received Currency": "neutral/missing (raw 0.0)",
            "Payment Format": "neutral/missing (raw 0.0)",
        },
    ),
}


def list_feature_contracts() -> Sequence[str]:
    return tuple(_REGISTRY.keys())


def get_feature_contract(contract_id: str) -> FeatureContract:
    if contract_id not in _REGISTRY:
        raise ValueError(
            f"Unknown feature_contract={contract_id!r}. "
            f"Known: {list(list_feature_contracts())}"
        )
    return _REGISTRY[contract_id]


def resolve_feature_contract_id(raw: Optional[str]) -> Optional[str]:
    """Return normalized contract id, or None when flag omitted (legacy passthrough)."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    # Native Multi-GIN is a separate family (not AML slot neutralization).
    if s == "paysim_native_multigin_core_v1":
        return s
    return get_feature_contract(s).contract_id


def apply_feature_contract_to_base_edge_attr(
    edge_attr: torch.Tensor,
    contract_id: Optional[str],
) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
    """Apply a v1 contract to base-4 (or wider) edge_attr columns 0..3.

    Legacy / omitted contract: return ``edge_attr`` unchanged (bit-exact path).
    Neutral slots: set to ``NEUTRAL_RAW_VALUE`` in-place on a clone.
    Native Multi-GIN contracts must not use this path (builder replaces base attr).
    """
    if contract_id is None:
        return edge_attr, None
    if contract_id == "paysim_native_multigin_core_v1":
        raise ValueError(
            "paysim_native_multigin_core_v1 must be built via paysim_native_multigin "
            "loader, not apply_feature_contract_to_base_edge_attr"
        )

    contract = get_feature_contract(contract_id)
    if edge_attr.dim() != 2:
        raise ValueError(f"edge_attr must be 2-D, got shape={tuple(edge_attr.shape)}")
    if edge_attr.shape[1] < contract.expected_base_dim:
        raise ValueError(
            f"edge_attr width {edge_attr.shape[1]} < base_dim {contract.expected_base_dim}"
        )

    # Legacy duplicate: formatted CSV already holds type in both slots — no-op.
    if contract.contract_id == CONTRACT_LEGACY:
        return edge_attr, contract.summary()

    out = edge_attr.clone()
    for idx, policy in contract.slot_policies.items():
        if policy == "neutral":
            out[:, idx] = float(contract.neutral_raw_value)
        # type_code / passthrough: leave values from CSV as-is
    return out, contract.summary()


def ensure_contract_output_dir(
    out_dir: Path,
    contract_id: str,
    *,
    meta_name: str = "meta.json",
) -> None:
    """Create ``out_dir``; refuse if existing meta records a different contract."""
    import json

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / meta_name
    if not meta_path.is_file():
        return
    try:
        prev = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Corrupt meta at {meta_path}; refusing overwrite") from exc
    prev_id = prev.get("feature_contract_id") or (
        (prev.get("feature_contract") or {}).get("feature_contract_id")
    )
    if prev_id is not None and str(prev_id) != str(contract_id):
        raise RuntimeError(
            f"Refusing cross-contract overwrite of {out_dir}: "
            f"existing={prev_id!r} requested={contract_id!r}"
        )
