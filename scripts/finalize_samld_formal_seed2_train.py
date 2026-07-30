#!/usr/bin/env python3
"""Post-train finalize for SAML-D formal seed-2 (hashes, max-AUPRC diagnostic, cells)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "results" / "diagnostics" / "samld_supervised_multigin_eu_formal_seed2"


def sha256_file(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_state_dict(sd: Dict[str, Any]) -> str:
    h = hashlib.sha256()
    for k in sorted(sd.keys()):
        h.update(k.encode())
        t = sd[k]
        if torch.is_tensor(t):
            h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-id", required=True)
    ap.add_argument("--run-name", default="samld_supervised_multigin_eu_v1_formal_seed2")
    args = ap.parse_args()
    run = args.run_name
    hist_path = (
        REPO
        / "results/diagnostics"
        / f"supervised_SAML-D_{run}_epoch_history.json"
    )
    summary_path = (
        REPO / "results/diagnostics" / f"supervised_SAML-D_{run}_summary.json"
    )
    run_dir = REPO / "saved-models" / run
    best = run_dir / "checkpoint_best_val_f1.tar"
    last = run_dir / "checkpoint_last.tar"

    if not hist_path.is_file() or not best.is_file() or not last.is_file():
        raise SystemExit(f"missing train artifacts: hist={hist_path.is_file()} best={best.is_file()} last={last.is_file()}")

    hist = json.loads(hist_path.read_text())
    epochs = hist.get("epochs") or []
    # max val AUPRC diagnostic (does not override F1 selection)
    best_auprc_ep = None
    best_auprc = float("-inf")
    for row in epochs:
        a = row.get("validation_auprc")
        if a is None or not math.isfinite(float(a)):
            continue
        if float(a) > best_auprc:
            best_auprc = float(a)
            best_auprc_ep = int(row["epoch"])

    # reload + hash model states separately from archive
    ckpt_meta = {}
    for label, path in (("best", best), ("last", last)):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        sd = payload.get("model_state_dict") or {}
        finite = all(torch.isfinite(t).all().item() for t in sd.values() if torch.is_tensor(t))
        ckpt_meta[label] = {
            "path": str(path),
            "archive_sha256": sha256_file(path),
            "model_state_sha256": sha256_state_dict(sd),
            "supervised_head": payload.get("supervised_head"),
            "epoch": payload.get("epoch"),
            "selected_epoch": payload.get("selected_epoch"),
            "best_validation_f1": payload.get("best_validation_f1"),
            "seed": payload.get("seed"),
            "n_tensors": sum(1 for t in sd.values() if torch.is_tensor(t)),
            "finite_params": finite,
            "reload_ok": bool(sd) and finite and payload.get("supervised_head") == "legacy",
        }

    # any test keys?
    test_metric_keys = sorted(
        {
            k
            for r in epochs
            for k in r
            if k.startswith(("test_minority_", "test_precision", "test_recall", "test_auroc", "test_auprc", "test_f1"))
        }
    )

    cell = {
        "phase": "train_finalize",
        "run_name": run,
        "train_job_id": args.job_id,
        "protocol_id": "samld_supervised_multigin_eu_v1",
        "n_epochs_completed": len(epochs),
        "skip_test_eval": True,
        "test_evaluated_during_training": False,
        "test_metric_keys_in_history": test_metric_keys,
        "selection_rule": "validation_minority_f1_argmax (paper-parity; earliest on tie)",
        "selected_epoch_by_val_f1": (json.loads(summary_path.read_text()).get("best_validation_epoch")
                                     if summary_path.is_file() else None),
        "max_validation_auprc_epoch_diagnostic": best_auprc_ep,
        "max_validation_auprc_diagnostic": best_auprc if best_auprc_ep is not None else None,
        "checkpoints": ckpt_meta,
        "epoch_history_path": str(hist_path.relative_to(REPO)),
        "summary_path": str(summary_path.relative_to(REPO)) if summary_path.is_file() else None,
        "protected_smoke_artifacts_untouched": True,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "train_finalize.json"
    out.write_text(json.dumps(cell, indent=2) + "\n")
    print(json.dumps({"ok": True, "out": str(out), "max_auprc_epoch": best_auprc_ep}))
    if not all(c["reload_ok"] for c in ckpt_meta.values()):
        return 2
    if test_metric_keys:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
