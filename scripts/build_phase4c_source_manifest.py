#!/usr/bin/env python3
"""Build the approved immutable-source manifest before submission."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from phase4c_four_domain.source_manifest import build_manifest
ap = argparse.ArgumentParser(); ap.add_argument("--output", type=Path,
    default=ROOT / "results/diagnostics/phase4c_four_domain_source_manifest.json"); args = ap.parse_args()
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(build_manifest(), indent=2) + "\n", encoding="utf-8")
print(args.output)
