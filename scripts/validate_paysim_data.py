#!/usr/bin/env python3
"""
Validate PaySim raw or formatted data under aml-data/PaySim/.

Lightweight checks (default): file presence, schema, fraud rate, temporal span.
Optional ``--load-graph``: full ``get_data()`` smoke test (slow on ~6M edges).

Examples::

  python scripts/validate_paysim_data.py
  python scripts/validate_paysim_data.py --format-raw
  python scripts/validate_paysim_data.py --load-graph --testing
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset_specs import FORMATTED_TRANSACTION_COLUMNS, get_dataset_spec
from format_paysim import format_paysim_file

RAW_GLOB = "PS_*.csv"
FORMATTED_NAME = "formatted_transactions.csv"


def _load_data_config() -> dict:
    with (REPO_ROOT / "data_config.json").open(encoding="utf-8") as f:
        return json.load(f)


def _paysim_dir(data_config: dict) -> Path:
    return Path(str(data_config["paths"]["aml_data"])) / "PaySim"


def _find_raw_csv(paysim_dir: Path) -> Path | None:
    matches = sorted(paysim_dir.glob(RAW_GLOB))
    if not matches:
        # Also accept any single large CSV that is not the formatted output.
        for path in sorted(paysim_dir.glob("*.csv")):
            if path.name != FORMATTED_NAME:
                return path
    return matches[0] if matches else None


def summarize_formatted(path: Path) -> None:
    spec = get_dataset_spec("PaySim")
    df = pd.read_csv(path)
    missing_cols = set(FORMATTED_TRANSACTION_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"{path} missing columns: {sorted(missing_cols)}")

    y = df[spec.label_col]
    ts = df["Timestamp"]
    step_hours = (ts // 3600).astype(int)
    logging.info("Formatted CSV: %s", path)
    logging.info("Rows: %d", len(df))
    logging.info("Unique accounts (from_id ∪ to_id): %d", pd.unique(df[["from_id", "to_id"]].values.ravel()).size)
    logging.info(
        "Positive rate (%s): %.4f%% (%d / %d)",
        spec.label_col,
        100.0 * float(y.mean()),
        int(y.sum()),
        len(y),
    )
    logging.info(
        "Timestamp span: %d .. %d (step hours %d .. %d)",
        int(ts.min()),
        int(ts.max()),
        int(step_hours.min()),
        int(step_hours.max()),
    )
    logging.info("Edge feature columns: %s", list(spec.edge_feature_cols))
    logging.info("Split mode: %s (%s)", spec.split_mode, spec.split_fractions)


def run_load_graph(testing: bool) -> None:
    from types import SimpleNamespace

    from data_loading import get_data

    args = SimpleNamespace(
        data="PaySim",
        model="gin",
        reverse_mp=True,
        ego=True,
        ports=True,
        tds=False,
        emlps=False,
        load_pattern_metadata=False,
        pattern_metadata=None,
    )
    data_config = _load_data_config()
    logging.info("Running get_data(PaySim) smoke test …")
    tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(args, data_config)
    logging.info(
        "Graph load OK: train=%d val=%d test=%d edges",
        int(tr_inds.shape[0]),
        int(val_inds.shape[0]),
        int(te_inds.shape[0]),
    )
    logging.info("Train graph: %s", tr_data)
    if not testing:
        logging.info("(Pass --testing if you only wanted a load smoke test.)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate PaySim dataset under aml-data/PaySim/.")
    parser.add_argument(
        "--format-raw",
        action="store_true",
        help=f"Run format_paysim.py on raw CSV in aml-data/PaySim/ → {FORMATTED_NAME}.",
    )
    parser.add_argument(
        "--load-graph",
        action="store_true",
        help="Full get_data() smoke test (loads entire graph; slow on full PaySim).",
    )
    parser.add_argument(
        "--testing",
        action="store_true",
        help="No-op flag for symmetry with other scripts (validate always runs locally).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    data_config = _load_data_config()
    paysim_dir = _paysim_dir(data_config)
    formatted_path = paysim_dir / FORMATTED_NAME

    if not paysim_dir.is_dir():
        logging.error("Missing PaySim directory: %s", paysim_dir)
        return 1

    if args.format_raw:
        raw_path = _find_raw_csv(paysim_dir)
        if raw_path is None:
            logging.error("No raw PaySim CSV found under %s (expected %s)", paysim_dir, RAW_GLOB)
            return 1
        logging.info("Formatting %s → %s", raw_path, formatted_path)
        format_paysim_file(raw_path, formatted_path)

    if not formatted_path.is_file():
        raw_path = _find_raw_csv(paysim_dir)
        if raw_path is not None:
            logging.error(
                "Found raw CSV %s but no %s. Re-run with --format-raw.",
                raw_path,
                formatted_path.name,
            )
        else:
            logging.error(
                "No formatted PaySim data at %s. Copy raw CSV there and use --format-raw.",
                formatted_path,
            )
        return 1

    summarize_formatted(formatted_path)

    if args.load_graph:
        run_load_graph(args.testing)

    logging.info("PaySim validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
