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


def _encode_type(type_name: str, type_vocab: dict[str, int]) -> int:
    key = str(type_name).strip()
    if key not in type_vocab:
        type_vocab[key] = len(type_vocab)
    return type_vocab[key]


def _encode_account(name: str, account_vocab: dict[str, int]) -> int:
    key = str(name).strip()
    if key not in account_vocab:
        account_vocab[key] = len(account_vocab)
    return account_vocab[key]


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

    type_vocab: dict[str, int] = {}
    account_vocab: dict[str, int] = {}

    rows = []
    for i, row in df.iterrows():
        type_code = _encode_type(row["type"], type_vocab)
        from_id = _encode_account(row["nameOrig"], account_vocab)
        to_id = _encode_account(row["nameDest"], account_vocab)
        step = int(row["step"])
        amount = float(row["amount"])
        is_fraud = int(row["isFraud"])

        rows.append(
            {
                "EdgeID": int(i),
                "from_id": from_id,
                "to_id": to_id,
                "Timestamp": step * 3600,
                "Amount Sent": amount,
                "Sent Currency": type_code,
                "Amount Received": amount,
                "Received Currency": type_code,
                "Payment Format": type_code,
                "Is Laundering": is_fraud,
            }
        )

    out = pd.DataFrame(rows, columns=list(FORMATTED_TRANSACTION_COLUMNS))
    out = out.sort_values("Timestamp", kind="mergesort").reset_index(drop=True)
    out["EdgeID"] = np.arange(len(out), dtype=np.int64)
    return out


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
