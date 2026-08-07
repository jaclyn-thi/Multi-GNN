#!/usr/bin/env python3
"""Phase-4C four-domain frozen-eval DAG v2 CLI (extract / probe / finalize / gates)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def cmd_authorize(_: argparse.Namespace) -> int:
    from phase4c_four_domain_frozen_eval.authorize_v2 import (
        verify_reusable_cells,
        verify_v2_authorization,
    )

    auth = verify_v2_authorization()
    if not auth.get("ok"):
        print(json.dumps(auth, indent=2))
        raise SystemExit(
            "PRODUCTION_SOURCE_DRIFT_REQUIRES_NEW_COMPUTE_NODE_PREFLIGHT — "
            "do not submit DAG; do not regenerate manifest silently"
        )
    reuse = verify_reusable_cells()
    out = {"ok": auth["ok"] and reuse["ok"], "authorization": auth, "reusable_cells": reuse}
    print(json.dumps(out, indent=2))
    return 0 if out["ok"] else 1


def cmd_extract(ns: argparse.Namespace) -> int:
    from phase4c_four_domain_frozen_eval.bind_v2 import bind_v2_paths
    from phase4c_four_domain_frozen_eval.paths_dag_v2 import REUSED_ARM, REUSED_STEP

    if ns.arm == REUSED_ARM and int(ns.step) == REUSED_STEP:
        raise SystemExit(
            "PROJECTION SHORT@4000 extract is reused from preflight — do not re-extract"
        )
    bind_v2_paths()
    from phase4c_four_domain_frozen_eval.dag_extract import run_checkpoint_extract

    summary = run_checkpoint_extract(ns.arm, int(ns.step))
    print(json.dumps({k: summary[k] for k in summary if k != "domains"}, indent=2, default=str))
    print(json.dumps({"domains_ok": summary["domains"]}, indent=2, default=str))
    return 0 if summary["ok"] else 1


def cmd_probe(ns: argparse.Namespace) -> int:
    from phase4c_four_domain_frozen_eval.bind_v2 import bind_v2_paths
    from phase4c_four_domain_frozen_eval.cell_io import atomic_json, cell_is_complete
    from phase4c_four_domain_frozen_eval.paths_dag_v2 import RESULT_ROOT, cell_emb_dir
    from phase4c_four_domain_frozen_eval import cell_name

    bind_v2_paths()
    cell = cell_name(ns.arm, int(ns.step), ns.target)
    emb = cell_emb_dir(ns.arm, int(ns.step), ns.target)
    cells_dir = ROOT / RESULT_ROOT / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    if not cell_is_complete(emb):
        err = {
            "ok": False,
            "cell": cell,
            "arm": ns.arm,
            "step": int(ns.step),
            "target": ns.target,
            "error": "extraction cell incomplete or invalid; probe skipped",
            "emb_dir": str(emb),
        }
        atomic_json(cells_dir / f"{cell}_probe_FAILED.json", err)
        print(json.dumps(err, indent=2))
        return 1
    from phase4c_four_domain_frozen_eval.dag_probe import run_cell_probe

    payload = run_cell_probe(ns.arm, int(ns.step), ns.target)
    print(json.dumps({"ok": payload["ok"], "cell": payload["cell"], "metrics": payload["metrics"]}, indent=2))
    return 0


def cmd_finalize(_: argparse.Namespace) -> int:
    from phase4c_four_domain_frozen_eval.finalize_v2 import run_finalize_v2

    agg = run_finalize_v2()
    print(
        json.dumps(
            {
                "ok": agg["ok"],
                "n_ok": agg["n_ok_cells"],
                "missing": agg["missing_or_failed_cells"],
                "historical": agg.get("historical_classification"),
            },
            indent=2,
        )
    )
    return 0


def cmd_cheap_gates(_: argparse.Namespace) -> int:
    import tempfile

    import numpy as np

    from phase4c_four_domain_frozen_eval import INVENTORY_CHECKPOINTS, TARGETS, all_plan_cells
    from phase4c_four_domain_frozen_eval.authorize_v2 import (
        verify_reusable_cells,
        verify_v2_authorization,
    )
    from phase4c_four_domain_frozen_eval.bind_v2 import bind_v2_paths
    from phase4c_four_domain_frozen_eval.cell_io import (
        cell_is_complete,
        promote_staging_cell,
        save_npz_arrays,
        sha_ordered_ids,
    )
    from phase4c_four_domain_frozen_eval.dag_finalize import matched_cohorts
    from phase4c_four_domain_frozen_eval.paths_dag_v2 import (
        EXTRACT_CHECKPOINTS,
        assert_dag_v2_paths_unique,
        ensure_embedding_root,
    )

    assert len(list(all_plan_cells())) == 40
    assert len(INVENTORY_CHECKPOINTS) == 10
    assert len(EXTRACT_CHECKPOINTS) == 9
    assert len(TARGETS) == 4
    assert_dag_v2_paths_unique()
    bind = bind_v2_paths()
    auth = verify_v2_authorization()
    assert auth["ok"], auth
    reuse = verify_reusable_cells()
    assert reuse["ok"], reuse

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        staging = td / "staging"
        staging.mkdir()
        Z = np.random.randn(32, 198).astype(np.float32)
        y = np.array([0, 1] * 16, dtype=np.int64)
        eid = np.arange(32, dtype=np.int64)
        tr = save_npz_arrays(staging / "train.npz", Z, y, eid)
        va = save_npz_arrays(staging / "val.npz", Z, y, eid)
        meta = {"ok": True, "cell": "synthetic", "outputs": {"train": tr, "val": va}, "test_evaluated": False}
        final = td / "cell"
        promote_staging_cell(staging, final, meta)
        assert cell_is_complete(final)
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
        assert matched_cohorts(extracts)["Small-HI"]["ok"]
        extracts["c"] = {
            "target": "Small-HI",
            "arm": "C",
            "step": 3,
            "cell": "c",
            "edge_id_sha256": {"train": "deadbeef", "val": h},
            "realized_counts": {"train": 32, "val": 32},
        }
        assert not matched_cohorts(extracts)["Small-HI"]["ok"]

    emb = ensure_embedding_root()
    print(
        json.dumps(
            {
                "ok": True,
                "authorization": auth,
                "reusable_cells": reuse,
                "embedding_root": emb,
                "bind": bind,
                "n_extract_checkpoints": len(EXTRACT_CHECKPOINTS),
            },
            indent=2,
        )
    )
    return 0


def cmd_dry_run_deps(_: argparse.Namespace) -> int:
    from phase4c_four_domain_frozen_eval import INVENTORY_CHECKPOINTS, TARGETS, cell_name
    from phase4c_four_domain_frozen_eval.paths_dag_v2 import (
        EXTRACT_CHECKPOINTS,
        REUSED_ARM,
        REUSED_STEP,
    )

    extracts = []
    probes = []
    for i, (arm, step) in enumerate(EXTRACT_CHECKPOINTS):
        extracts.append({"idx": i, "arm": arm, "step": step, "job_placeholder": f"EXTRACT_{i}"})
        for target in TARGETS:
            probes.append(
                {
                    "cell": cell_name(arm, step, target),
                    "dependency": f"afterany:EXTRACT_{i}",
                    "extract_idx": i,
                }
            )
    # reusable probes: no extract dependency
    for target in TARGETS:
        probes.append(
            {
                "cell": cell_name(REUSED_ARM, REUSED_STEP, target),
                "dependency": None,
                "reused_preflight_cell": True,
            }
        )
    assert len(extracts) == 9
    assert len(probes) == 40
    assert len(INVENTORY_CHECKPOINTS) == 10
    deps = {p["dependency"] for p in probes if p["dependency"]}
    assert len(deps) == 9
    assert all(d.startswith("afterany:") for d in deps)
    print(
        json.dumps(
            {
                "ok": True,
                "n_extract": 9,
                "n_probe": 40,
                "n_reused_probes": 4,
                "unique_extract_deps": len(deps),
                "dependency_policy": "afterany_per_checkpoint_extract; reused probes independent",
            },
            indent=2,
        )
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase-4C four-domain frozen-eval DAG v2")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("authorize", help="Verify v2 auth + source freeze + reusable cells")
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

    p = sub.add_parser("cheap-gates", help="Login-safe synthetic + reusable checks")
    p.set_defaults(func=cmd_cheap_gates)

    p = sub.add_parser("dry-run-deps", help="Dependency graph dry run (9 extracts, afterany)")
    p.set_defaults(func=cmd_dry_run_deps)

    ns = ap.parse_args()
    return int(ns.func(ns))


if __name__ == "__main__":
    raise SystemExit(main())
