#!/usr/bin/env python3
"""
Format PaySim CSV into Multi-GNN ``formatted_transactions.csv``.

PaySim source columns (Kaggle / original release):
  step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig,
  nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud

Output schema matches IBM AMLWorld (``format_kaggle_files.py``) so downstream
loading, extraction, and linear probes work without encoder changes.

Harmonization (for frozen AML checkpoint transfer):
  - Timestamp  = step * 3600  (synthetic seconds; enables hourly_step split)
  - Amount Received / Amount Sent = amount
  - Received Currency / Sent Currency = integer code for ``type``
  - Payment Format = same type code (AML 4-feature contract)
  - Is Laundering = isFraud

Excluded from output (leakage / not in AML pretrain):
  - balance columns, isFlaggedFraud

Usage::

  python format_paysim.py /path/to/PS_20174392719_149120730945s.csv

Writes ``formatted_transactions.csv`` next to the input file. Copy the output
tree to ``aml-data/PaySim/formatted_transactions.csv`` and pass ``--data PaySim``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from dataset_specs import FORMATTED_TRANSACTION_COLUMNS


def format_paysim_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Map raw PaySim rows to the shared formatted transaction schema."""
    required = {
        "step",
        "type",
        "amount",
        "nameOrig",
        "nameDest",
        "isFraud",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"PaySim input missing columns: {sorted(missing)}")

    n = len(df)
    type_codes, _ = pd.factorize(df["type"].astype(str).str.strip())

    from_names = df["nameOrig"].astype(str).str.strip()
    to_names = df["nameDest"].astype(str).str.strip()
    account_codes, _ = pd.factorize(pd.concat([from_names, to_names], ignore_index=True))
    from_id = account_codes[:n].astype(np.int64)
    to_id = account_codes[n:].astype(np.int64)

    out = pd.DataFrame(
        {
            "EdgeID": np.arange(n, dtype=np.int64),
            "from_id": from_id,
            "to_id": to_id,
            "Timestamp": df["step"].to_numpy(dtype=np.int64) * 3600,
            "Amount Sent": df["amount"].to_numpy(dtype=np.float64),
            "Sent Currency": type_codes.astype(np.int64),
            "Amount Received": df["amount"].to_numpy(dtype=np.float64),
            "Received Currency": type_codes.astype(np.int64),
            "Payment Format": type_codes.astype(np.int64),
            "Is Laundering": df["isFraud"].to_numpy(dtype=np.int64),
        }
    )
    out = out.sort_values("Timestamp", kind="mergesort").reset_index(drop=True)
    out["EdgeID"] = np.arange(len(out), dtype=np.int64)
    return out[list(FORMATTED_TRANSACTION_COLUMNS)]


def format_paysim_file(in_path: Path, out_path: Path | None = None) -> Path:
    in_path = Path(in_path)
    if out_path is None:
        out_path = in_path.parent / "formatted_transactions.csv"
    else:
        out_path = Path(out_path)

    df = pd.read_csv(in_path)
    formatted = format_paysim_dataframe(df)
    formatted.to_csv(out_path, index=False)
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Format PaySim CSV for Multi-GNN.")
    parser.add_argument(
        "input_csv",
        type=str,
        help="Path to PaySim CSV (e.g. PS_20174392719_149120730945s.csv).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output path (default: formatted_transactions.csv beside input).",
    )
    args = parser.parse_args(argv)

    out = format_paysim_file(args.input_csv, args.output)
    print(f"Wrote {out} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
