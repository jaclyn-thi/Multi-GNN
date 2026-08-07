#!/usr/bin/env python3
"""Phase-4C four-domain frozen-eval DAG CLI (extract / probe / finalize / gates)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def cmd_authorize(_: argparse.Namespace) -> int:
    from phase4c_four_domain_frozen_eval.preflight import verify_authorization
    from phase4c_four_domain_frozen_eval.source_freeze import verify_no_critical_drift

    auth = verify_authorization()
    pf = ROOT / "results/diagnostics/phase4c_four_domain_frozen_eval_preflight_v1/preflight_summary.json"
    summary = json.loads(pf.read_text(encoding="utf-8"))
    if summary.get("verdict") != "FULL_FROZEN_EVAL_AUTHORIZED":
        raise SystemExit(f"preflight not authorized: {summary.get('verdict')}")
    if str(summary.get("slurm_job_id")) != "19742414":
        print(f"WARNING: preflight job_id={summary.get('slurm_job_id')} (expected 19742414)")
    drift = verify_no_critical_drift()
    if not drift["ok"]:
        print(json.dumps(drift, indent=2))
        raise SystemExit(
            "CRITICAL extract/probe source drift vs preflight freeze — "
            "require new bounded compute-node preflight before DAG submit"
        )
    out = {
        "ok": True,
        "training_authorization": auth,
        "preflight_verdict": summary["verdict"],
        "preflight_job_id": summary.get("slurm_job_id"),
        "source_drift": drift,
    }
    print(json.dumps(out, indent=2))
    return 0


def cmd_extract(ns: argparse.Namespace) -> int:
    from phase4c_four_domain_frozen_eval.dag_extract import run_checkpoint_extract

    summary = run_checkpoint_extract(ns.arm, int(ns.step))
    print(json.dumps({k: summary[k] for k in summary if k != "domains"}, indent=2, default=str))
    print(json.dumps({"domains_ok": summary["domains"]}, indent=2, default=str))
    return 0 if summary["ok"] else 1


def cmd_probe(ns: argparse.Namespace) -> int:
    from phase4c_four_domain_frozen_eval.dag_probe import run_cell_probe

    payload = run_cell_probe(ns.arm, int(ns.step), ns.target)
    print(json.dumps({"ok": payload["ok"], "cell": payload["cell"], "metrics": payload["metrics"]}, indent=2))
    return 0


def cmd_finalize(_: argparse.Namespace) -> int:
    from phase4c_four_domain_frozen_eval.dag_finalize import run_finalize

    agg = run_finalize()
    print(json.dumps({"ok": agg["ok"], "n_ok": agg["n_ok_cells"], "missing": agg["missing_or_failed_cells"]}, indent=2))
    return 0 if agg["ok"] else 0  # finalizer always exits 0 after writing partial report


def cmd_cheap_gates(_: argparse.Namespace) -> int:
    """Login-safe synthetic tests (no real data)."""
    import tempfile

    import numpy as np

    from phase4c_four_domain_frozen_eval import INVENTORY_CHECKPOINTS, TARGETS, all_plan_cells
    from phase4c_four_domain_frozen_eval.cell_io import (
        cell_is_complete,
        promote_staging_cell,
        save_npz_arrays,
        sha_ordered_ids,
    )
    from phase4c_four_domain_frozen_eval.dag_finalize import matched_cohorts
    from phase4c_four_domain_frozen_eval.paths_dag import ensure_embedding_root
    from phase4c_four_domain_frozen_eval.source_freeze import verify_no_critical_drift
    from phase4c_four_domain_frozen_eval.preflight import verify_authorization

    assert len(list(all_plan_cells())) == 40
    assert len(INVENTORY_CHECKPOINTS) == 10
    assert len(TARGETS) == 4
    drift = verify_no_critical_drift()
    assert drift["ok"], drift
    auth = verify_authorization()
    assert auth["manifest_sha256"]

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        staging = td / "staging"
        staging.mkdir()
        Z = np.random.randn(32, 198).astype(np.float32)
        y = np.array([0, 1] * 16, dtype=np.int64)
        eid = np.arange(32, dtype=np.int64)
        tr = save_npz_arrays(staging / "train.npz", Z, y, eid)
        va = save_npz_arrays(staging / "val.npz", Z, y, eid)
        assert tr["dim"] == 198
        meta = {
            "ok": True,
            "cell": "synthetic",
            "outputs": {"train": tr, "val": va},
            "test_evaluated": False,
        }
        final = td / "cell"
        promote_staging_cell(staging, final, meta)
        assert cell_is_complete(final)
        assert not cell_is_complete(td / "missing")

        # matched cohort synthetic
        h = sha_ordered_ids(eid)
        extracts = {
            "a": {
                "target": "Small-HI",
                "arm": "A",
                "step": 1,
                "cell": "a",
                "edge_id_sha256": {"train": h, "val": h},
                "realized_counts": {"train": 32, "val": 32},
            },
            "b": {
                "target": "Small-HI",
                "arm": "B",
                "step": 2,
                "cell": "b",
                "edge_id_sha256": {"train": h, "val": h},
                "realized_counts": {"train": 32, "val": 32},
            },
        }
        card = matched_cohorts(extracts)
        assert card["Small-HI"]["ok"]

        # partial failure finalizer path: mismatched hashes
        extracts["c"] = {
            "target": "Small-HI",
            "arm": "C",
            "step": 3,
            "cell": "c",
            "edge_id_sha256": {"train": "deadbeef", "val": h},
            "realized_counts": {"train": 32, "val": 32},
        }
        card2 = matched_cohorts(extracts)
        assert not card2["Small-HI"]["ok"]

    emb = ensure_embedding_root()
    assert emb["ok"]
    print(json.dumps({"ok": True, "authorization": auth, "source_drift": drift, "embedding_root": emb}, indent=2))
    return 0


def cmd_dry_run_deps(_: argparse.Namespace) -> int:
    from phase4c_four_domain_frozen_eval import INVENTORY_CHECKPOINTS, TARGETS, cell_name

    extracts = []
    probes = []
    for i, (arm, step) in enumerate(INVENTORY_CHECKPOINTS):
        extracts.append({"idx": i, "arm": arm, "step": step, "job_placeholder": f"EXTRACT_{i}"})
        for target in TARGETS:
            probes.append(
                {
                    "cell": cell_name(arm, step, target),
                    "arm": arm,
                    "step": step,
                    "target": target,
                    "dependency": f"afterok:EXTRACT_{i}",
                }
            )
    assert len(extracts) == 10
    assert len(probes) == 40
    # independence: no shared afterok across extracts
    deps = {p["dependency"] for p in probes}
    assert len(deps) == 10
    print(json.dumps({"ok": True, "n_extract": 10, "n_probe": 40, "unique_extract_deps": len(deps)}, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase-4C four-domain frozen-eval DAG")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("authorize", help="Verify training + preflight + source freeze")
    p.set_defaults(func=cmd_authorize)

    p = sub.add_parser("extract", help="GPU: extract one checkpoint × four domains")
    p.add_argument("--arm", required=True)
    p.add_argument("--step", type=int, required=True)
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("probe", help="CPU: probe one checkpoint-domain cell")
    p.add_argument("--arm", required=True)
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--target", required=True, choices=["Small-HI", "SAML-D", "Small-LI", "PaySim"])
    p.set_defaults(func=cmd_probe)

    p = sub.add_parser("finalize", help="Aggregate artifacts; partial-safe")
    p.set_defaults(func=cmd_finalize)

    p = sub.add_parser("cheap-gates", help="Login-safe synthetic checks")
    p.set_defaults(func=cmd_cheap_gates)

    p = sub.add_parser("dry-run-deps", help="Dependency graph dry run")
    p.set_defaults(func=cmd_dry_run_deps)

    ns = ap.parse_args()
    return int(ns.func(ns))


if __name__ == "__main__":
    raise SystemExit(main())
