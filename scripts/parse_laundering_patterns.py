#!/usr/bin/env python3
"""
Parse AMLWorld laundering-pattern ``patterns.txt`` files into edge-level metadata.

Encodes pattern transaction rows using the patterns.txt field order (not Kaggle CSV
column order), joins to ``formatted_transactions.csv`` via a strong composite key,
and writes ``laundering_attempt_metadata.csv`` plus an audit summary.

Example (cluster: ``module load miniforge && conda activate multignn``)::

  python scripts/parse_laundering_patterns.py \\
    --patterns aml-data/Small-HI/patterns.txt \\
    --raw-transactions raw-aml-data/HI-Small_Trans.csv \\
    --formatted-transactions aml-data/Small-HI/formatted_transactions.csv \\
    --output aml-data/Small-HI/laundering_attempt_metadata.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

BEGIN_RE = re.compile(r"^BEGIN LAUNDERING ATTEMPT - ([A-Z-]+)(?::\s*(.*))?\s*$")
END_RE = re.compile(r"^END LAUNDERING ATTEMPT - ([A-Z-]+)\s*$")

PATTERN_TX_FIELDS = (
    "timestamp_raw",
    "from_bank",
    "from_account",
    "to_bank",
    "to_account",
    "amount_sent",
    "sent_currency",
    "amount_received",
    "received_currency",
    "payment_format",
    "is_laundering",
)

OUTPUT_FIELDS = [
    "attempt_id",
    "pattern_type",
    "pattern_detail",
    "tx_idx_in_attempt",
    "n_tx_in_attempt",
    "EdgeID",
    "join_status",
    *PATTERN_TX_FIELDS,
    "from_id",
    "to_id",
    "Timestamp",
    "Amount Sent",
    "Amount Received",
    "Sent Currency",
    "Received Currency",
    "Payment Format",
    "Is Laundering",
]

JoinKey = Tuple[
    int,
    int,
    int,
    float,
    float,
    int,
    int,
    int,
    int,
]


@dataclass
class EncodeState:
    currency: Dict[str, int] = field(default_factory=dict)
    payment_format: Dict[str, int] = field(default_factory=dict)
    account: Dict[str, int] = field(default_factory=dict)
    first_ts: int = -1

    def get_dict_val(self, name: str, collection: Dict[str, int]) -> int:
        if name in collection:
            return collection[name]
        val = len(collection)
        collection[name] = val
        return val


@dataclass
class PatternTxRow:
    attempt_id: int
    pattern_type: str
    pattern_detail: str
    tx_idx_in_attempt: int
    n_tx_in_attempt: int
    fields: Dict[str, str]
    join_key: Optional[JoinKey] = None


@dataclass
class ParseStats:
    attempts: int = 0
    tx_rows_parsed: int = 0
    malformed_blocks: int = 0
    malformed_tx_rows: int = 0
    begin_end_mismatches: int = 0
    pattern_type_counts: Counter = field(default_factory=Counter)


def _norm_str(value: Any) -> str:
    return str(value).strip()


def _parse_timestamp(ts_raw: str, state: EncodeState) -> int:
    dt_obj = datetime.strptime(ts_raw.strip(), "%Y/%m/%d %H:%M")
    ts = int(dt_obj.timestamp())
    if state.first_ts == -1:
        start = datetime(dt_obj.year, dt_obj.month, dt_obj.day)
        state.first_ts = int(start.timestamp()) - 10
    return ts - state.first_ts


def _lookup(collection: Dict[str, int], key: str) -> int:
    try:
        return collection[key]
    except KeyError as exc:
        raise KeyError(f"unknown encoding key: {key!r}") from exc


def encode_pattern_row(parts: Sequence[str], state: EncodeState) -> JoinKey:
    if len(parts) != 11:
        raise ValueError(f"expected 11 fields, got {len(parts)}")

    ts = _parse_timestamp(parts[0], state)
    from_key = _norm_str(parts[1]) + _norm_str(parts[2])
    to_key = _norm_str(parts[3]) + _norm_str(parts[4])
    from_id = _lookup(state.account, from_key)
    to_id = _lookup(state.account, to_key)

    amount_sent = round(float(parts[5]), 6)
    sent_currency = _lookup(state.currency, _norm_str(parts[6]))
    amount_received = round(float(parts[7]), 6)
    received_currency = _lookup(state.currency, _norm_str(parts[8]))
    payment_format = _lookup(state.payment_format, _norm_str(parts[9]))
    is_laundering = int(parts[10])

    return (
        from_id,
        to_id,
        ts,
        amount_sent,
        amount_received,
        sent_currency,
        received_currency,
        payment_format,
        is_laundering,
    )


# Raw Kaggle CSV layout (matches ``format_kaggle_files.py`` / datatable column indices).
RAW_COL_TIMESTAMP = 0
RAW_COL_FROM_BANK = 1
RAW_COL_FROM_ACCOUNT = 2
RAW_COL_TO_BANK = 3
RAW_COL_TO_ACCOUNT = 4
RAW_COL_AMOUNT_RECEIVED = 5
RAW_COL_RECEIVING_CURRENCY = 6
RAW_COL_AMOUNT_PAID = 7
RAW_COL_PAYMENT_CURRENCY = 8
RAW_COL_PAYMENT_FORMAT = 9
RAW_COL_IS_LAUNDERING = 10
RAW_MIN_COLUMNS = 11


def build_encode_state_from_raw(raw_path: Path) -> EncodeState:
    state = EncodeState()
    with raw_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader)  # header; use fixed indices because "Account" appears twice

        for row in reader:
            if not row or all(not cell.strip() for cell in row):
                continue
            if len(row) < RAW_MIN_COLUMNS:
                continue

            _parse_timestamp(row[RAW_COL_TIMESTAMP], state)

            state.get_dict_val(_norm_str(row[RAW_COL_RECEIVING_CURRENCY]), state.currency)
            state.get_dict_val(_norm_str(row[RAW_COL_PAYMENT_CURRENCY]), state.currency)
            state.get_dict_val(_norm_str(row[RAW_COL_PAYMENT_FORMAT]), state.payment_format)

            from_key = _norm_str(row[RAW_COL_FROM_BANK]) + _norm_str(row[RAW_COL_FROM_ACCOUNT])
            to_key = _norm_str(row[RAW_COL_TO_BANK]) + _norm_str(row[RAW_COL_TO_ACCOUNT])
            state.get_dict_val(from_key, state.account)
            state.get_dict_val(to_key, state.account)

    return state


def parse_patterns_file(patterns_path: Path, state: EncodeState) -> Tuple[List[PatternTxRow], ParseStats]:
    text = patterns_path.read_text(encoding="utf-8")
    blocks = [block.strip() for block in text.split("\n\n") if block.strip()]
    rows: List[PatternTxRow] = []
    stats = ParseStats()

    for attempt_id, block in enumerate(blocks):
        lines = [line.strip() for line in block.split("\n") if line.strip()]
        if len(lines) < 2:
            stats.malformed_blocks += 1
            continue

        begin_match = BEGIN_RE.match(lines[0])
        end_match = END_RE.match(lines[-1])
        if begin_match is None or end_match is None:
            stats.malformed_blocks += 1
            continue

        pattern_type = begin_match.group(1)
        pattern_detail = begin_match.group(2) or ""
        if end_match.group(1) != pattern_type:
            stats.begin_end_mismatches += 1

        tx_lines = lines[1:-1]
        n_tx = len(tx_lines)
        stats.attempts += 1
        stats.pattern_type_counts[pattern_type] += 1

        for tx_idx, line in enumerate(tx_lines):
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 11:
                stats.malformed_tx_rows += 1
                continue

            try:
                join_key = encode_pattern_row(parts, state)
            except (ValueError, TypeError, KeyError):
                stats.malformed_tx_rows += 1
                continue

            fields = dict(zip(PATTERN_TX_FIELDS, parts))
            rows.append(
                PatternTxRow(
                    attempt_id=attempt_id,
                    pattern_type=pattern_type,
                    pattern_detail=pattern_detail,
                    tx_idx_in_attempt=tx_idx,
                    n_tx_in_attempt=n_tx,
                    fields=fields,
                    join_key=join_key,
                )
            )
            stats.tx_rows_parsed += 1

    return rows, stats


def _formatted_join_key(row: Dict[str, str]) -> JoinKey:
    return (
        int(row["from_id"]),
        int(row["to_id"]),
        int(float(row["Timestamp"])),
        round(float(row["Amount Sent"]), 6),
        round(float(row["Amount Received"]), 6),
        int(row["Sent Currency"]),
        int(row["Received Currency"]),
        int(row["Payment Format"]),
        int(row["Is Laundering"]),
    )


def build_formatted_lookup(formatted_path: Path) -> Dict[JoinKey, List[str]]:
    lookup: Dict[JoinKey, List[str]] = {}
    with formatted_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "EdgeID",
            "from_id",
            "to_id",
            "Timestamp",
            "Amount Sent",
            "Amount Received",
            "Sent Currency",
            "Received Currency",
            "Payment Format",
            "Is Laundering",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = required - set(reader.fieldnames or [])
            raise ValueError(f"{formatted_path} missing columns: {sorted(missing)}")

        for row in reader:
            key = _formatted_join_key(row)
            lookup.setdefault(key, []).append(_norm_str(row["EdgeID"]))
    return lookup


def join_rows(
    rows: Iterable[PatternTxRow],
    lookup: Dict[JoinKey, List[str]],
) -> Tuple[List[Dict[str, str]], Counter]:
    output_rows: List[Dict[str, str]] = []
    join_counts: Counter = Counter()

    for row in rows:
        out: Dict[str, str] = {
            "attempt_id": str(row.attempt_id),
            "pattern_type": row.pattern_type,
            "pattern_detail": row.pattern_detail,
            "tx_idx_in_attempt": str(row.tx_idx_in_attempt),
            "n_tx_in_attempt": str(row.n_tx_in_attempt),
            "EdgeID": "",
            "join_status": "unmatched",
        }
        out.update(row.fields)

        if row.join_key is None:
            join_counts["unmatched"] += 1
            output_rows.append(out)
            continue

        key = row.join_key
        out.update(
            {
                "from_id": str(key[0]),
                "to_id": str(key[1]),
                "Timestamp": str(key[2]),
                "Amount Sent": str(key[3]),
                "Amount Received": str(key[4]),
                "Sent Currency": str(key[5]),
                "Received Currency": str(key[6]),
                "Payment Format": str(key[7]),
                "Is Laundering": str(key[8]),
            }
        )

        matches = lookup.get(key, [])
        if len(matches) == 1:
            out["EdgeID"] = matches[0]
            out["join_status"] = "matched"
            join_counts["matched"] += 1
        elif len(matches) > 1:
            out["join_status"] = "ambiguous"
            join_counts["ambiguous"] += 1
        else:
            join_counts["unmatched"] += 1

        output_rows.append(out)

    return output_rows, join_counts


def collect_laundering_edge_ids(formatted_path: Path) -> Tuple[int, set[str]]:
    laundering_ids: set[str] = set()
    with formatted_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if int(row["Is Laundering"]) == 1:
                laundering_ids.add(_norm_str(row["EdgeID"]))
    return len(laundering_ids), laundering_ids


def build_audit(
    stats: ParseStats,
    join_counts: Counter,
    output_rows: Sequence[Dict[str, str]],
    laundering_total: int,
    laundering_ids: set[str],
) -> Dict[str, Any]:
    matched_edge_ids = [
        row["EdgeID"] for row in output_rows if row["join_status"] == "matched" and row["EdgeID"]
    ]
    matched_unique = set(matched_edge_ids)
    duplicate_edge_ids = {
        edge_id: count for edge_id, count in Counter(matched_edge_ids).items() if count > 1
    }

    laundering_not_in_patterns = laundering_ids - matched_unique

    return {
        "attempts": stats.attempts,
        "pattern_type_counts": dict(sorted(stats.pattern_type_counts.items())),
        "tx_rows_parsed": stats.tx_rows_parsed,
        "malformed_blocks": stats.malformed_blocks,
        "malformed_tx_rows": stats.malformed_tx_rows,
        "begin_end_mismatches": stats.begin_end_mismatches,
        "join_status_counts": {
            "matched": join_counts["matched"],
            "ambiguous": join_counts["ambiguous"],
            "unmatched": join_counts["unmatched"],
        },
        "duplicate_edge_id_count": len(duplicate_edge_ids),
        "duplicate_edge_ids": dict(sorted(duplicate_edge_ids.items(), key=lambda item: -item[1])),
        "laundering_edge_audit": {
            "formatted_is_laundering_total": laundering_total,
            "pattern_matched_unique_edge_ids": len(matched_unique),
            "pattern_matched_rows": join_counts["matched"],
            "laundering_edges_not_in_patterns": len(laundering_not_in_patterns),
            "note": (
                "Unmatched laundering edges are counted only; no pattern label is assigned."
            ),
        },
    }


def write_csv(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def print_audit(audit: Dict[str, Any]) -> None:
    print("=== Laundering pattern parse audit ===")
    print(f"attempts: {audit['attempts']}")
    print("pattern_type counts:")
    for pattern_type, count in audit["pattern_type_counts"].items():
        print(f"  {pattern_type}: {count}")
    print(f"tx_rows_parsed: {audit['tx_rows_parsed']}")
    print(f"malformed_blocks: {audit['malformed_blocks']}")
    print(f"malformed_tx_rows: {audit['malformed_tx_rows']}")
    print(f"begin_end_mismatches: {audit['begin_end_mismatches']}")
    join_counts = audit["join_status_counts"]
    print(
        "join_status counts: "
        f"matched={join_counts['matched']} "
        f"ambiguous={join_counts['ambiguous']} "
        f"unmatched={join_counts['unmatched']}"
    )
    print(f"duplicate_edge_id_count: {audit['duplicate_edge_id_count']}")

    laundering = audit["laundering_edge_audit"]
    print("--- laundering edge coverage ---")
    print(f"formatted Is Laundering=1 edges: {laundering['formatted_is_laundering_total']}")
    print(f"pattern-matched unique EdgeIDs: {laundering['pattern_matched_unique_edge_ids']}")
    print(f"pattern-matched rows: {laundering['pattern_matched_rows']}")
    print(
        "laundering edges not in any pattern block: "
        f"{laundering['laundering_edges_not_in_patterns']}"
    )


def audit_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_audit.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse AMLWorld laundering pattern files.")
    parser.add_argument("--patterns", type=Path, required=True, help="Path to patterns.txt")
    parser.add_argument(
        "--raw-transactions",
        type=Path,
        required=True,
        help="Raw Kaggle transaction CSV used to build formatted_transactions.csv",
    )
    parser.add_argument(
        "--formatted-transactions",
        type=Path,
        required=True,
        help="Path to formatted_transactions.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for laundering_attempt_metadata.csv",
    )
    args = parser.parse_args()

    for path_arg, label in (
        (args.patterns, "patterns"),
        (args.raw_transactions, "raw transactions"),
        (args.formatted_transactions, "formatted transactions"),
    ):
        if not path_arg.exists():
            raise SystemExit(f"Missing {label} file: {path_arg}")

    encode_state = build_encode_state_from_raw(args.raw_transactions)
    pattern_rows, stats = parse_patterns_file(args.patterns, encode_state)
    lookup = build_formatted_lookup(args.formatted_transactions)
    output_rows, join_counts = join_rows(pattern_rows, lookup)
    laundering_total, laundering_ids = collect_laundering_edge_ids(args.formatted_transactions)

    audit = build_audit(stats, join_counts, output_rows, laundering_total, laundering_ids)

    write_csv(args.output, output_rows)
    audit_path = audit_output_path(args.output)
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print_audit(audit)
    print(f"Wrote metadata CSV: {args.output}")
    print(f"Wrote audit JSON: {audit_path}")


if __name__ == "__main__":
    main()
