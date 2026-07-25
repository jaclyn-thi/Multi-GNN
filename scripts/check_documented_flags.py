#!/usr/bin/env python3
"""Validate that documented CLI flags exist on their owning parsers.

Does not load AMLWorld or require a GPU. Exits non-zero on stale/unknown flags.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from typing import Dict, List, Set

ROOT = Path(__file__).resolve().parents[1]
CLI_REF = ROOT / "notes" / "cli-reference.md"


def _flags_from_py_file(path: Path) -> Set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    flags: Set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = getattr(func, "attr", None) or getattr(func, "id", None)
        if name != "add_argument":
            continue
        for arg in node.args:
            val = None
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                val = arg.value
            elif isinstance(arg, ast.Str):
                val = arg.s
            if val and val.startswith("--"):
                flags.add(val)
    return flags


def _documented_flags_by_owner(md: str) -> Dict[str, Set[str]]:
    owners: Dict[str, Set[str]] = {
        "main.py": set(),
        "linear_probe.py": set(),
        "embedding_extraction.py": set(),
        "scripts/gcpal_txn_node": set(),
        "scripts/label_efficiency_probe.py": set(),
        "unscoped": set(),
    }
    current = "unscoped"
    for line in md.splitlines():
        if not line.startswith("#"):
            for m in re.finditer(r"`(--[a-zA-Z0-9_]+)`|\|\s*(--[a-zA-Z0-9_]+)\s*\|", line):
                flag = m.group(1) or m.group(2)
                if flag:
                    owners[current].add(flag)
            continue
        low = line.lower()
        # Prefer more specific owners before generic main.py mentions.
        if "transaction-node" in low or "gcpal_txn_node" in low:
            current = "scripts/gcpal_txn_node"
        elif "label-efficiency" in low:
            current = "scripts/label_efficiency_probe.py"
        elif "linear probe" in low:
            current = "linear_probe.py"
        elif "embedding extraction" in low:
            current = "embedding_extraction.py"
        elif re.search(r"\bmain\.py\b", low) or "graph form" in low or "training objective" in low:
            current = "main.py"
        elif "contrastive" in low or "morphology" in low or "logging and misc" in low:
            current = "main.py"
    return owners


def main() -> int:
    md = CLI_REF.read_text(encoding="utf-8")
    documented = _documented_flags_by_owner(md)
    errors: List[str] = []

    util_flags = _flags_from_py_file(ROOT / "util.py")
    for fl in sorted(documented["main.py"]):
        if fl not in util_flags:
            errors.append("main.py / util.py missing documented flag: {0}".format(fl))

    critical = [
        "--preserve_seed_edges",
        "--correct_reverse_edge_features",
        "--supervised_head",
        "--contrastive_num_neg_samples",
        "--contrastive_accum_steps",
    ]
    for fl in critical:
        if fl not in util_flags:
            errors.append("critical util.py flag missing: {0}".format(fl))
        if fl not in md:
            errors.append("critical flag absent from cli-reference.md: {0}".format(fl))

    emb_flags = _flags_from_py_file(ROOT / "embedding_extraction.py") | util_flags
    if "--representation_source" in md and "--representation_source" not in emb_flags:
        errors.append("embedding_extraction / util missing --representation_source")

    lp_flags = _flags_from_py_file(ROOT / "linear_probe.py")
    for fl in sorted(documented["linear_probe.py"]):
        if fl not in lp_flags:
            errors.append("linear_probe.py missing documented flag: {0}".format(fl))

    txn_flags: Set[str] = set()
    for p in sorted((ROOT / "scripts").glob("gcpal_txn_node*.py")):
        txn_flags |= _flags_from_py_file(p)
    for fl in sorted(documented["scripts/gcpal_txn_node"]):
        if fl not in txn_flags:
            errors.append("gcpal_txn_node scripts missing documented flag: {0}".format(fl))

    # Soft: ensure txn-node section exists and is not empty of ownership
    if "--max_total_nodes" in md and "--max_total_nodes" not in txn_flags:
        errors.append("--max_total_nodes documented but missing from gcpal_txn_node scripts")

    if errors:
        print("check_documented_flags: FAILED ({0} issues)".format(len(errors)))
        for e in errors:
            print("  -", e)
        return 1
    print(
        "check_documented_flags: OK "
        "(util={0}, txn={1}, documented main={2}, lp={3})".format(
            len(util_flags),
            len(txn_flags),
            len(documented["main.py"]),
            len(documented["linear_probe.py"]),
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
