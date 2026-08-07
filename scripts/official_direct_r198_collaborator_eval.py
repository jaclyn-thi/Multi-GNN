#!/usr/bin/env python3
"""Official collaborator-facing DIRECT_R198 evaluation wrapper.

ALWAYS:
  - full-subgraph extraction (`extract_direct_r198_full_cell`)
  - corrected train/val ID checks
  - PaperStyleMLP protocol (20 ep / 1e-3 / 8192 / seed 2 / R198+X+TF / best-val-AUPRC)
  - stamps protocol=full_subgraph into cell JSON + run manifest

NEVER:
  - seed-only extraction
  - training / checkpoint overwrite
  - writing into seed-only embedding paths
  - overwriting existing official cell artifacts unless --allow_overwrite

Usage examples:

  # One (run, epoch) cell — same as collaborator package protocol
  python scripts/official_direct_r198_collaborator_eval.py \\
    --run direct_r198_tfmoe_40ep_seed2_linear_lr2e-3 \\
    --arm DIRECT_H_TFMOE --peak_lr 0.002 --epoch 10

  # Matched SSL epochs 3,10,20,30,40 for one run
  python scripts/official_direct_r198_collaborator_eval.py \\
    --run direct_r198_tfmoe_40ep_seed2_linear_lr2e-3 \\
    --arm DIRECT_H_TFMOE --peak_lr 0.002 --epochs 3,10,20,30,40

  # Rebuild collaborator tables (refuses seed-only cells)
  python scripts/official_direct_r198_collaborator_eval.py --build-package

See notes/direct_r198_official_collaborator_eval.md
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from direct_r198_eval_protocol import (  # noqa: E402
    OFFICIAL_EMB_ROOT,
    OFFICIAL_OUT_DIR,
    OFFICIAL_PKG_DIR,
    PROTOCOL_FULL_SUBGRAPH,
    TIER_OFFICIAL,
    assert_collaborator_merge_allowed,
    build_run_manifest,
    infer_protocol,
    official_protocol_block,
    write_json,
)

DEFAULT_EPOCHS = (3, 10, 20, 30, 40)


def _parse_epochs(s: Optional[str], single: Optional[int]) -> List[int]:
    if single is not None and s:
        raise SystemExit("Pass only one of --epoch or --epochs")
    if single is not None:
        return [int(single)]
    if s:
        return [int(x.strip()) for x in s.split(",") if x.strip()]
    return list(DEFAULT_EPOCHS)


def _run_cell(
    *,
    run: str,
    arm: str,
    peak_lr: float,
    epoch: int,
    embeddings_dir: str,
    out_dir: str,
    allow_overwrite: bool,
    skip_extract: bool,
    skip_probe: bool,
) -> dict:
    cell_path = ROOT / out_dir / "cells" / run / f"epoch_{epoch:02d}.json"
    emb_path = ROOT / embeddings_dir / f"{run}_epoch{epoch:02d}"
    seed_only_emb = ROOT / "embeddings" / f"{run}_epoch{epoch:02d}"

    if emb_path.resolve() == seed_only_emb.resolve():
        raise SystemExit(
            f"Refusing embeddings_dir that resolves to seed-only path: {emb_path}"
        )
    if cell_path.is_file() and not allow_overwrite:
        raise SystemExit(
            f"Refusing to overwrite official cell artifact: {cell_path}\n"
            "Pass --allow_overwrite only if you intentionally replace it."
        )

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    cmd = [
        sys.executable,
        str(ROOT / "scripts/reeval_direct_r198_40ep_full_extract_cell.py"),
        "--run",
        run,
        "--epoch",
        str(epoch),
        "--arm",
        arm,
        "--peak_lr",
        str(peak_lr),
        "--embeddings_dir",
        embeddings_dir,
        "--out_dir",
        out_dir,
    ]
    if skip_extract:
        cmd.append("--skip_extract")
    if skip_probe:
        cmd.append("--skip_probe")
    logging.info("Official cell: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(ROOT), env=env)

    if not cell_path.is_file():
        raise SystemExit(f"Expected cell JSON missing after eval: {cell_path}")
    cell = json.loads(cell_path.read_text())
    # Ensure protocol stamp (reeval script also stamps; enforce here)
    proto = infer_protocol(cell)
    if proto != PROTOCOL_FULL_SUBGRAPH:
        raise SystemExit(f"Cell missing full_subgraph protocol after official eval: {cell_path}")
    assert_collaborator_merge_allowed(cell, path=cell_path)
    return {
        "epoch": epoch,
        "status": cell.get("status"),
        "cell_path": str(cell_path),
        "embedding_dir": cell.get("embedding_dir"),
        "protocol": cell.get("protocol"),
        "validation_auprc": (cell.get("primary") or {}).get("validation_auprc"),
        "verify_ok": (cell.get("verify") or {}).get("ok"),
    }


def _build_package() -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    cmd = [sys.executable, str(ROOT / "scripts/build_direct_r198_40ep_collaborator_package.py")]
    logging.info("Building collaborator package (full_subgraph gate enforced): %s", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(ROOT), env=env)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-package", action="store_true", help="Rebuild collaborator tables/plots only")
    ap.add_argument("--run", type=str, help="Checkpoint unique_name / run id")
    ap.add_argument("--arm", choices=["DIRECT_H", "DIRECT_H_TFMOE"])
    ap.add_argument("--peak_lr", type=float)
    ap.add_argument("--epoch", type=int, help="Single SSL epoch")
    ap.add_argument("--epochs", type=str, help="Comma-separated SSL epochs (default 3,10,20,30,40)")
    ap.add_argument("--embeddings_dir", default=OFFICIAL_EMB_ROOT)
    ap.add_argument("--out_dir", default=OFFICIAL_OUT_DIR)
    ap.add_argument(
        "--allow_overwrite",
        action="store_true",
        help="Allow replacing existing official cell JSON (default: refuse)",
    )
    ap.add_argument("--skip_extract", action="store_true")
    ap.add_argument("--skip_probe", action="store_true")
    args = ap.parse_args()

    if args.build_package:
        if args.run or args.arm or args.peak_lr is not None:
            logging.warning("--build-package ignores --run/--arm/--peak_lr")
        return _build_package()

    if not args.run or not args.arm or args.peak_lr is None:
        ap.error("--run, --arm, and --peak_lr are required unless --build-package")

    if args.embeddings_dir.rstrip("/") == "embeddings" or args.embeddings_dir == "embeddings":
        raise SystemExit(
            "Refusing default seed-only embeddings root. "
            f"Official path is --embeddings_dir {OFFICIAL_EMB_ROOT}"
        )

    epochs = _parse_epochs(args.epochs, args.epoch)
    protocol_block = official_protocol_block(
        embeddings_dir=args.embeddings_dir,
    )
    cell_summaries = []
    for ep in epochs:
        cell_summaries.append(
            _run_cell(
                run=args.run,
                arm=args.arm,
                peak_lr=float(args.peak_lr),
                epoch=ep,
                embeddings_dir=args.embeddings_dir,
                out_dir=args.out_dir,
                allow_overwrite=bool(args.allow_overwrite),
                skip_extract=bool(args.skip_extract),
                skip_probe=bool(args.skip_probe),
            )
        )

    manifest = build_run_manifest(
        run=args.run,
        arm=args.arm,
        peak_lr=float(args.peak_lr),
        epochs=epochs,
        protocol_block=protocol_block,
        cells=cell_summaries,
        extra={
            "submitted_utc": datetime.now(timezone.utc).isoformat(),
            "wrapper": "scripts/official_direct_r198_collaborator_eval.py",
            "training_submitted": False,
            "out_dir": args.out_dir,
            "embeddings_dir": args.embeddings_dir,
            "collaborator_package": OFFICIAL_PKG_DIR,
            "evaluation_tier": TIER_OFFICIAL,
            "protocol": PROTOCOL_FULL_SUBGRAPH,
        },
    )
    man_path = ROOT / args.out_dir / "cells" / args.run / "official_eval_manifest.json"
    write_json(man_path, manifest)
    print(json.dumps({"status": "ok", "manifest": str(man_path), "cells": cell_summaries}, indent=2))
    return 0 if all(c.get("status") == "ok" for c in cell_summaries) else 2


if __name__ == "__main__":
    raise SystemExit(main())
