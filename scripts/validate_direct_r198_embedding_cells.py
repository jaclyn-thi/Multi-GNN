#!/usr/bin/env python3
"""Validate completed DIRECT_H embedding cells; report resume plan. No overwrite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EPOCHS = (1, 3, 5, 10)
ARMS = {
    "DIRECT_H": "direct_h_infonce_10ep_seed2_sched",
    "DIRECT_H_TFMOE": "direct_h_tfmoe_learned_alpha_10ep_seed2_sched",
}


def validate_npz(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {"ok": False, "reason": "missing"}
    try:
        d = np.load(path)
        Z = np.asarray(d["Z"])
        y = np.asarray(d["y"]).reshape(-1)
        eid = np.asarray(d["edge_id"]).reshape(-1)
    except Exception as e:
        return {"ok": False, "reason": f"unreadable: {e}"}
    if Z.ndim != 2 or Z.shape[1] != 198:
        return {"ok": False, "reason": f"bad dim {getattr(Z, 'shape', None)}"}
    if Z.shape[0] != y.shape[0] or Z.shape[0] != eid.shape[0]:
        return {"ok": False, "reason": "row mismatch"}
    if not np.isfinite(Z).all():
        return {"ok": False, "reason": "non-finite"}
    if eid.size != np.unique(eid).size:
        return {"ok": False, "reason": "duplicate edge_id"}
    return {
        "ok": True,
        "n": int(Z.shape[0]),
        "dim": 198,
        "n_pos": int((y == 1).sum()),
        "bytes": int(path.stat().st_size),
    }


def cell_status(run: str, epoch: int) -> Dict[str, Any]:
    sub = f"{run}_epoch{epoch:02d}"
    emb = ROOT / "embeddings" / sub / "pre_embedding_3h"
    ckpt = ROOT / "saved-models" / f"checkpoint_{run}_epoch{epoch:02d}.tar"
    train = validate_npz(emb / "train.npz")
    val = validate_npz(emb / "val.npz")
    meta_path = emb / "meta.json"
    meta = None
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {"ok": False}
    complete = bool(train.get("ok") and val.get("ok") and meta_path.is_file())
    need: List[str] = []
    if not train.get("ok"):
        need.append("train")
    if not val.get("ok"):
        need.append("val")
    # If train ok but val missing -> only val; meta written after val/train as needed
    return {
        "run": run,
        "epoch": epoch,
        "subdir": sub,
        "embedding_dir": str(emb),
        "checkpoint": str(ckpt),
        "checkpoint_exists": ckpt.is_file(),
        "train": train,
        "val": val,
        "meta_exists": meta_path.is_file(),
        "meta_checkpoint_path": (meta or {}).get("checkpoint_path") if isinstance(meta, dict) else None,
        "complete": complete,
        "need_splits": need,
        "test_npz_present": (emb / "test.npz").is_file(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        type=str,
        default=str(
            ROOT / "results/diagnostics/direct_h_tfmoe_scheduled_val_analysis/cell_validation.json"
        ),
    )
    args = ap.parse_args()
    cells = []
    resume = []
    reuse = []
    for arm, run in ARMS.items():
        for ep in EPOCHS:
            st = cell_status(run, ep)
            st["arm"] = arm
            cells.append(st)
            if st["test_npz_present"]:
                raise SystemExit(f"test.npz present — abort: {st['embedding_dir']}")
            if st["complete"]:
                reuse.append({"arm": arm, "run": run, "epoch": ep, "action": "reuse"})
            else:
                resume.append(
                    {
                        "arm": arm,
                        "run": run,
                        "epoch": ep,
                        "action": "extract",
                        "splits": ",".join(st["need_splits"]) or "train,val",
                        "note": (
                            "train exists; val only"
                            if st["train"].get("ok") and not st["val"].get("ok")
                            else "missing/invalid"
                        ),
                    }
                )
    report = {
        "ok": True,
        "n_complete": len(reuse),
        "n_incomplete": len(resume),
        "reuse": reuse,
        "resume": resume,
        "cells": cells,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"n_complete": len(reuse), "n_incomplete": len(resume), "resume": resume}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
