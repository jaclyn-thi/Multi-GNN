"""PaySim-native Multi-GIN feature contract (supervised ceiling; not AML transfer).

Contract ID: ``paysim_native_multigin_core_v1``

Base edge features (width 11), fixed order:
  0  Timestamp/step (seconds after re-zero, i.e. step*3600 re-zeroed)
  1  log1p(amount)
  2–6 type one-hots: PAYMENT, TRANSFER, CASH_OUT, DEBIT, CASH_IN
  7–10 oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest

With ports (no TDS): edge_dim = 13.

Continuous columns are train-fit z-normalized; one-hots left as 0/1.
New-balance fields make this a post-transaction supervised ceiling.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

CONTRACT_NATIVE_MULTIGIN_CORE = "paysim_native_multigin_core_v1"

TYPE_ORDER: Tuple[str, ...] = (
    "PAYMENT",
    "TRANSFER",
    "CASH_OUT",
    "DEBIT",
    "CASH_IN",
)

BASE_FEATURE_NAMES: Tuple[str, ...] = (
    "time",
    "log1p_amount",
    "type_PAYMENT",
    "type_TRANSFER",
    "type_CASH_OUT",
    "type_DEBIT",
    "type_CASH_IN",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
)

# Continuous indices in the *base* 11-d tensor (before ports).
CONTINUOUS_BASE_INDICES: Tuple[int, ...] = (0, 1, 7, 8, 9, 10)
# After ports append at 11,12 — also continuous.
PORT_INDICES: Tuple[int, ...] = (11, 12)
ONEHOT_BASE_INDICES: Tuple[int, ...] = (2, 3, 4, 5, 6)

EXPECTED_BASE_DIM = 11
EXPECTED_EDGE_DIM_PORTS = 13

FORBIDDEN_IN_X: Tuple[str, ...] = (
    "isFraud",
    "Is Laundering",
    "isFlaggedFraud",
    "EdgeID",
    "from_id",
    "to_id",
    "nameOrig",
    "nameDest",
)


def is_native_multigin_contract(contract_id: Optional[str]) -> bool:
    return str(contract_id or "").strip() == CONTRACT_NATIVE_MULTIGIN_CORE


def contract_summary() -> Dict[str, Any]:
    return {
        "feature_contract_id": CONTRACT_NATIVE_MULTIGIN_CORE,
        "description": (
            "PaySim-native supervised Multi-GIN core: time, log1p(amount), "
            "5 type one-hots, four balance fields; ports appended separately. "
            "Post-transaction ceiling (newbalance* included). Not AMLWorld-transfer compatible."
        ),
        "base_feature_names": list(BASE_FEATURE_NAMES),
        "type_onehot_order": list(TYPE_ORDER),
        "expected_base_dim": EXPECTED_BASE_DIM,
        "expected_edge_dim_with_ports": EXPECTED_EDGE_DIM_PORTS,
        "continuous_base_indices": list(CONTINUOUS_BASE_INDICES),
        "onehot_base_indices": list(ONEHOT_BASE_INDICES),
        "port_indices_after_ports": list(PORT_INDICES),
        "normalization": {
            "mode": "train_fit_continuous_only",
            "onehots_unchanged": True,
            "independent_val_znorm": False,
        },
        "forbidden_in_X": list(FORBIDDEN_IN_X),
        "includes_balance_deltas": False,
        "includes_isFlaggedFraud": False,
        "deployment_caveat": (
            "newbalanceOrig/newbalanceDest are post-transaction fields; "
            "may be unavailable in a pre-authorization setting."
        ),
    }


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_float_array(arr: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(arr.astype(np.float64)).tobytes())
    return h.hexdigest()


def load_aligned_paysim_native(
    *,
    formatted_csv: Path,
    raw_csv: Path,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Rebuild native table aligned to formatted EdgeID/Timestamp/label order.

    Same factorization/sort policy as ``format_paysim.py`` / tabular baseline.
    """
    raw = pd.read_csv(raw_csv)
    required = {
        "step",
        "type",
        "amount",
        "nameOrig",
        "nameDest",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "isFraud",
    }
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"raw PaySim missing columns: {sorted(missing)}")

    n = len(raw)
    type_str = raw["type"].astype(str).str.strip()
    # Fixed category order (do not rely on appearance order).
    type_to_code = {t: i for i, t in enumerate(TYPE_ORDER)}
    unknown = sorted(set(type_str.unique()) - set(TYPE_ORDER))
    if unknown:
        raise ValueError(f"unexpected PaySim type values: {unknown}")
    type_codes = type_str.map(type_to_code).to_numpy(dtype=np.int64)

    from_names = raw["nameOrig"].astype(str).str.strip()
    to_names = raw["nameDest"].astype(str).str.strip()
    account_codes, _ = pd.factorize(pd.concat([from_names, to_names], ignore_index=True))
    from_id = account_codes[:n].astype(np.int64)
    to_id = account_codes[n:].astype(np.int64)
    ts = raw["step"].to_numpy(dtype=np.int64) * 3600

    df = pd.DataFrame(
        {
            "EdgeID": np.arange(n, dtype=np.int64),
            "from_id": from_id,
            "to_id": to_id,
            "Timestamp": ts,
            "step": raw["step"].to_numpy(dtype=np.int64),
            "type_code": type_codes,
            "type_str": type_str.to_numpy(),
            "amount": raw["amount"].to_numpy(dtype=np.float64),
            "oldbalanceOrg": raw["oldbalanceOrg"].to_numpy(dtype=np.float64),
            "newbalanceOrig": raw["newbalanceOrig"].to_numpy(dtype=np.float64),
            "oldbalanceDest": raw["oldbalanceDest"].to_numpy(dtype=np.float64),
            "newbalanceDest": raw["newbalanceDest"].to_numpy(dtype=np.float64),
            "Is Laundering": raw["isFraud"].to_numpy(dtype=np.int64),
        }
    )
    df = df.sort_values("Timestamp", kind="mergesort").reset_index(drop=True)
    df["EdgeID"] = np.arange(len(df), dtype=np.int64)

    fmt = pd.read_csv(
        formatted_csv,
        usecols=["EdgeID", "Timestamp", "Is Laundering", "from_id", "to_id"],
    )
    if len(fmt) != len(df):
        raise ValueError(f"row count mismatch formatted={len(fmt)} native={len(df)}")
    if not np.array_equal(fmt["EdgeID"].to_numpy(), df["EdgeID"].to_numpy()):
        raise ValueError("EdgeID alignment failed vs formatted")
    if not np.array_equal(fmt["Timestamp"].to_numpy(), df["Timestamp"].to_numpy()):
        raise ValueError("Timestamp alignment failed vs formatted")
    if not np.array_equal(fmt["Is Laundering"].to_numpy(), df["Is Laundering"].to_numpy()):
        raise ValueError("label alignment failed vs formatted")
    if not np.array_equal(fmt["from_id"].to_numpy(), df["from_id"].to_numpy()):
        raise ValueError("from_id alignment failed vs formatted")
    if not np.array_equal(fmt["to_id"].to_numpy(), df["to_id"].to_numpy()):
        raise ValueError("to_id alignment failed vs formatted")

    meta = {
        "formatted_sha256": sha256_file(formatted_csv),
        "raw_sha256": sha256_file(raw_csv),
        "n_rows": int(len(df)),
        "type_order": list(TYPE_ORDER),
        "alignment": "EdgeID/Timestamp/label/from_id/to_id match formatted",
    }
    return df, meta


def build_native_base_edge_attr(
    df: pd.DataFrame,
    *,
    timestamp_rezeroed: np.ndarray,
) -> torch.Tensor:
    """Build float32 [N, 11] base edge_attr. Never includes labels/IDs."""
    n = len(df)
    out = np.zeros((n, EXPECTED_BASE_DIM), dtype=np.float32)
    # 0: time (re-zeroed Timestamp seconds — same timeline as split)
    out[:, 0] = timestamp_rezeroed.astype(np.float32)
    # 1: log1p(amount)
    out[:, 1] = np.log1p(df["amount"].to_numpy(dtype=np.float64)).astype(np.float32)
    # 2–6: type one-hots
    tc = df["type_code"].to_numpy(dtype=np.int64)
    for i in range(len(TYPE_ORDER)):
        out[:, 2 + i] = (tc == i).astype(np.float32)
    # 7–10: balances
    for j, name in enumerate(
        ("oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest")
    ):
        out[:, 7 + j] = df[name].to_numpy(dtype=np.float64).astype(np.float32)
    return torch.from_numpy(out)


def continuous_indices_with_ports(*, ports: bool) -> List[int]:
    idxs = list(CONTINUOUS_BASE_INDICES)
    if ports:
        idxs.extend(PORT_INDICES)
    return idxs


def apply_train_fit_continuous_znorm(
    tr_attr: torch.Tensor,
    val_attr: torch.Tensor,
    te_attr: torch.Tensor,
    continuous_indices: Sequence[int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Fit mean/std on train continuous cols only; apply to val/test. One-hots unchanged."""
    idx = torch.tensor(list(continuous_indices), dtype=torch.long)
    mean = tr_attr[:, idx].mean(0)
    std = tr_attr[:, idx].std(0)
    std = torch.where(std == 0, torch.ones_like(std), std)

    def _apply(ea: torch.Tensor) -> torch.Tensor:
        out = ea.clone()
        out[:, idx] = (ea[:, idx] - mean) / std
        return out

    meta = {
        "mode": "train_fit_continuous_only",
        "continuous_indices": list(continuous_indices),
        "mean": mean.detach().cpu().numpy().astype(np.float64).tolist(),
        "std": std.detach().cpu().numpy().astype(np.float64).tolist(),
        "scaler_sha256": sha256_float_array(
            np.concatenate(
                [mean.detach().cpu().numpy(), std.detach().cpu().numpy()]
            )
        ),
        "zero_variance_replaced_with_1": bool(
            (tr_attr[:, idx].std(0) == 0).any().item()
        ),
    }
    return _apply(tr_attr), _apply(val_attr), _apply(te_attr), meta


def assert_no_forbidden_names(names: Sequence[str]) -> None:
    bad = [n for n in names if n in FORBIDDEN_IN_X]
    if bad:
        raise AssertionError(f"forbidden columns in X: {bad}")
