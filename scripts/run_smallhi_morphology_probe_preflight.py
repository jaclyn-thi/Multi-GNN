#!/usr/bin/env python3
"""Preflight for Small-HI morphology downstream probe (no probe fit, no full emb load)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from morphology_downstream_probe.config import (
    BASELINES,
    ENCODER_SPECS,
    PROBE_CONFIG,
    RESULT_ROOT,
    ZERO_INFLATION_THRESHOLD,
)
from morphology_downstream_probe.cohort import matched_cohort_provenance, refuse_test_access
from morphology_downstream_probe.distribution_gate import distribution_report
from morphology_downstream_probe.residualize import DEFAULT_RESIDUALIZATION
from morphology_downstream_probe.triangles import (
    compute_train_static_triangles,
    edge_log_triangle_targets,
)


def _synthetic():
    import numpy as np

    tr_s = np.array([0, 1, 2, 1, 2, 3], dtype=np.int64)
    tr_r = np.array([1, 2, 0, 3, 3, 1], dtype=np.int64)
    tr_e = np.arange(100, 106, dtype=np.int64)
    va_s = np.array([0, 9, 1], dtype=np.int64)
    va_r = np.array([2, 1, 9], dtype=np.int64)
    va_e = np.array([200, 201, 202], dtype=np.int64)
    return tr_e, tr_s, tr_r, va_e, va_s, va_r


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path(RESULT_ROOT) / "preflight.json")
    p.add_argument("--synthetic", action="store_true", default=True)
    p.add_argument("--no-synthetic", action="store_true")
    p.add_argument("--check-embedding-paths", action="store_true")
    p.add_argument("--load-embeddings", action="store_true", help="Forbidden in infra task.")
    p.add_argument("--fit-probe", action="store_true", help="Forbidden in infra task.")
    args = p.parse_args()

    refuse_test_access()
    if args.load_embeddings:
        raise RuntimeError("Refuse --load-embeddings in preflight (infra / concurrency safety)")
    if args.fit_probe:
        raise RuntimeError("Refuse --fit-probe in preflight")

    if args.no_synthetic:
        raise RuntimeError(
            "Full Small-HI preflight requires triangle cache from precompute; "
            "use --synthetic for infra wiring or pass cache paths in a later task."
        )

    tr_e, tr_s, tr_r, va_e, va_s, va_r = _synthetic()
    counts, prov = compute_train_static_triangles(tr_s, tr_r)
    y_tr, t_tr, cov_tr = edge_log_triangle_targets(tr_e, tr_s, tr_r, counts, split="train")
    y_va, t_va, cov_va = edge_log_triangle_targets(va_e, va_s, va_r, counts, split="validation")
    dist = distribution_report(
        y_tr, y_va, t_sum_train=t_tr, t_sum_val=t_va, train_coverage=cov_tr, val_coverage=cov_va
    )

    emb_status = []
    if args.check_embedding_paths:
        for spec in ENCODER_SPECS:
            path = ROOT / spec["embeddings_rel"]
            emb_status.append(
                {
                    "name": spec["name"],
                    "path": str(spec["embeddings_rel"]),
                    "exists": path.exists(),
                    "reuse_only": True,
                    "loaded": False,
                }
            )

    report = {
        "ok": True,
        "mode": "synthetic_preflight",
        "probe_config": PROBE_CONFIG,
        "encoders": list(ENCODER_SPECS),
        "baselines": list(BASELINES),
        "residualization": DEFAULT_RESIDUALIZATION.to_dict(),
        "zero_inflation_threshold": ZERO_INFLATION_THRESHOLD,
        "triangle_provenance": prov.to_dict(),
        "distribution_gate": dist,
        "matched_cohort_template": matched_cohort_provenance(
            tr_e, va_e, encoder_names=[e["name"] for e in ENCODER_SPECS]
        ),
        "embedding_path_checks": emb_status,
        "executed_probe_fit": False,
        "loaded_full_embeddings": False,
        "test_access": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "out": str(args.out), "gate": dist["gate"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
