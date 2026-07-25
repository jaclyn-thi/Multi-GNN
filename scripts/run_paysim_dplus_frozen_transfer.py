#!/usr/bin/env python3
"""CLI for locked AMLWorld→PaySim frozen D+ transfer.

Modes:
  smoke — seed2 only, few batches + one MLP step + freeze/integrity checks
  run   — full extract+eval for --role seed1|seed2|seed3|controls

Never writes thesis experiment registry. Never overwrites legacy PaySim caches.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paysim_dplus_frozen_transfer import (  # noqa: E402
    DEFAULT_EXTRACT_BS,
    DEFAULT_SMOKE_MAX_BATCHES,
    EMBEDDINGS_ROOT,
    LEGACY_PAYSIM_EMBED_DIR,
    MLP_EPOCHS,
    RESULTS_DIR,
    assert_not_legacy_paysim_path,
    project_full_runtime,
    run_role,
    write_json,
)
from util import logger_setup  # noqa: E402


def _build_smoke_projection(report: Dict[str, Any], *, max_mlp_epochs_smoke: int = 1) -> Dict[str, Any]:
    """Amortize one-time setup; scale extract by sec/batch only."""
    rt = report.get("runtime") or {}
    stages = rt.get("stage_timings") or {}
    extract_timing = ((report.get("extract") or {}).get("timing")) or {}

    one_time = float(stages.get("one_time_setup_sec") or 0.0)
    smoke_n_batches = int(rt.get("smoke_n_batches") or extract_timing.get("n_batches_total") or 0)
    full_n_batches = int(rt.get("full_n_batches") or 0)
    sec_per_batch = float(rt.get("extract_sec_per_batch") or extract_timing.get("sec_per_batch") or float("nan"))
    probe_sec = float(stages.get("downstream_probe_sec") or 0.0)

    # Smoke ran 1 MLP epoch on one stack; full seed2 role runs 15 epochs × 5 stacks.
    # Scale conservatively by epochs × stacks (seed2 includes X-only).
    n_full_stacks = 5
    epoch_scale = float(MLP_EPOCHS) / float(max(max_mlp_epochs_smoke, 1))
    projected_downstream = probe_sec * epoch_scale * float(n_full_stacks)

    if not math.isfinite(sec_per_batch) or smoke_n_batches <= 0 or full_n_batches <= 0:
        raise RuntimeError(
            f"Cannot project runtime: sec_per_batch={sec_per_batch} "
            f"smoke_n_batches={smoke_n_batches} full_n_batches={full_n_batches}"
        )

    projection = project_full_runtime(
        one_time_setup_sec=one_time,
        extract_sec_per_batch=sec_per_batch,
        smoke_n_batches=smoke_n_batches,
        full_n_batches=full_n_batches,
        downstream_probe_sec=probe_sec,
        projected_downstream_sec=projected_downstream,
    )
    projection["measured_stage_timings"] = dict(stages)
    projection["measured_extract"] = {
        "dual_extract_sec": extract_timing.get("dual_extract_sec"),
        "pre3h_extract_sec": extract_timing.get("pre3h_extract_sec"),
        "post128_extract_sec": extract_timing.get("post128_extract_sec"),
        "n_batches_total": smoke_n_batches,
        "sec_per_batch": sec_per_batch,
        "capture_mode": extract_timing.get("capture_mode"),
    }
    projection["downstream_scaling"] = {
        "smoke_epochs": max_mlp_epochs_smoke,
        "full_epochs": MLP_EPOCHS,
        "full_n_stacks_seed2": n_full_stacks,
        "projected_downstream_sec": projected_downstream,
    }
    # Explicit bug-class check: one-time must not be multiplied by batch count.
    bad_legacy = one_time * float(full_n_batches)
    projection["legacy_bug_would_have_estimated_sec"] = bad_legacy + projected_downstream
    projection["legacy_bug_note"] = (
        "Job 18851404 failed because wall time (dominated by one-time ports/TDS) "
        "was divided by smoke batches then multiplied by full batch count."
    )
    return projection


def cmd_smoke(args: argparse.Namespace) -> int:
    assert_not_legacy_paysim_path(EMBEDDINGS_ROOT / "smoke_dplus_seed2")
    if LEGACY_PAYSIM_EMBED_DIR.is_dir():
        logging.info("Legacy PaySim dir present and will not be written: %s", LEGACY_PAYSIM_EMBED_DIR)

    t0 = time.time()
    report = run_role(
        "seed2",
        data_config=args.data_config,
        batch_size=args.batch_size,
        device_str=args.device,
        max_batches=args.max_batches,
        max_mlp_epochs=1,
        smoke=True,
    )
    elapsed = time.time() - t0
    report["runtime"]["smoke_wall_sec"] = elapsed
    report["runtime"]["projection"] = _build_smoke_projection(report, max_mlp_epochs_smoke=1)
    report["prior_smoke_job_18851404"] = {
        "failed_gate": "projected_fits_6h",
        "reason": "one_time_setup_multiplied_by_batch_count",
        "technical_checks_passed": True,
    }

    edge_emb = (report.get("load") or {}).get("edge_emb_keys") or {}
    edge_dim_ok = any(
        isinstance(shape, list) and len(shape) >= 2 and int(shape[-1]) == 8
        for shape in edge_emb.values()
    )
    report["smoke_gate"] = {
        "strict_load_ok": bool(edge_emb) and edge_dim_ok,
        "edge_dim_8_ok": edge_dim_ok,
        "encoder_hash_unchanged": report.get("extract", {}).get("encoder_hash_unchanged", False),
        "pre_post_aligned": all(
            s.get("pre_post_id_aligned") for s in report.get("extract", {}).get("splits", {}).values()
        ),
        "finite": all(s.get("finite") for s in report.get("extract", {}).get("splits", {}).values()),
        "mlp_step_ran": "pre3h_HxX" in report.get("stacks", {}),
        "train_fit_edge_znorm": bool((report.get("integrity") or {}).get("train_fit_edge_znorm")),
        "legacy_untouched": True,
        "projected_fits_6h": bool(report["runtime"]["projection"]["fits_6h_gpu"]),
    }
    gate = report["smoke_gate"]
    gate["passed"] = bool(
        gate["strict_load_ok"]
        and gate["edge_dim_8_ok"]
        and gate["encoder_hash_unchanged"]
        and gate["pre_post_aligned"]
        and gate["finite"]
        and gate["mlp_step_ran"]
        and gate["train_fit_edge_znorm"]
        and gate["projected_fits_6h"]
    )

    out_json = RESULTS_DIR / "smoke_seed2.json"
    out_md = RESULTS_DIR / "smoke_seed2.md"
    # Also write uniquely named copy for this rerun.
    stamp = RESULTS_DIR / f"smoke_seed2_retime_{int(time.time())}.json"
    write_json(out_json, report)
    write_json(stamp, report)

    proj = report["runtime"]["projection"]
    stages = report["runtime"].get("stage_timings") or {}
    lines = [
        "# PaySim D+ frozen transfer smoke (seed2, corrected runtime projection)",
        "",
        f"- wall_sec: {elapsed:.1f}",
        f"- gate_passed: {gate['passed']}",
        f"- encoder_hash_unchanged: {gate['encoder_hash_unchanged']}",
        f"- strict_load_ok / edge_dim_8: {gate['strict_load_ok']} / {gate['edge_dim_8_ok']}",
        f"- pre/post ID aligned: {gate['pre_post_aligned']}",
        f"- projected total hours: {proj['est_total_hours']:.2f}",
        f"- fits 6h GPU: {gate['projected_fits_6h']}",
        f"- GPU mem GB: {report['runtime'].get('gpu_mem_gb')}",
        f"- RSS GB: {report['runtime'].get('rss_gb')}",
        "",
        "## Measured stage timings (seconds)",
        f"- data_loading_csv: {stages.get('data_loading_csv_sec')}",
        f"- ports_construction: {stages.get('ports_construction_sec')}",
        f"- tds_construction: {stages.get('tds_construction_sec')}",
        f"- other_graph_prep: {stages.get('other_graph_prep_sec')}",
        f"- model_construct: {stages.get('model_construct_sec')}",
        f"- checkpoint_load: {stages.get('checkpoint_load_sec')}",
        f"- paysim_x_build: {stages.get('paysim_x_build_sec')}",
        f"- one_time_setup (sum): {stages.get('one_time_setup_sec')}",
        f"- dual_extract (pre+post one forward): {(report.get('extract') or {}).get('timing', {}).get('dual_extract_sec')}",
        f"- downstream_probe: {stages.get('downstream_probe_sec')}",
        "",
        "## Projection formula",
        f"- {proj['formula']}",
        f"- one_time_setup_sec: {proj['one_time_setup_sec']:.1f}",
        f"- projected_pre3h_extract_sec: {proj['projected_pre3h_extract_sec']:.1f}",
        f"- projected_post128_extract_sec: {proj['projected_post128_extract_sec']:.1f} (dual; included above)",
        f"- projected_downstream_eval_sec: {proj['projected_downstream_eval_sec']:.1f}",
        f"- margin_sec: {proj['margin_sec']:.1f}",
        f"- est_total_sec / hours: {proj['est_total_sec']:.1f} / {proj['est_total_hours']:.2f}",
        f"- inputs: {json.dumps(proj['inputs'])}",
        "",
        "Do not submit full jobs if any gate fails.",
        "",
    ]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines))
    logging.info("Wrote %s and %s (stamp=%s)", out_json, out_md, stamp)

    if not gate["passed"]:
        logging.error("SMOKE GATE FAILED: %s", json.dumps(gate, indent=2))
        logging.error("projection=%s", json.dumps(proj, indent=2, default=str))
    else:
        logging.info("SMOKE GATE PASSED")
    return 0 if gate["passed"] else 2


def cmd_run(args: argparse.Namespace) -> int:
    role = args.role
    if role == "controls":
        # D: CONTROL random-init + SECONDARY FT seed2 (AML-supervised encoder block).
        roles = ["random_init", "ft_seed2"]
    else:
        roles = [role]

    for r in roles:
        logging.info("=== PaySim D+ transfer role=%s ===", r)
        report = run_role(
            r,
            data_config=args.data_config,
            batch_size=args.batch_size,
            device_str=args.device,
            skip_extract=args.skip_extract,
            max_batches=None,
            max_mlp_epochs=None,
            smoke=False,
        )
        if r == "ft_seed2":
            report["experiment_class"] = "SECONDARY_supervised_partial_ft_transfer"
        elif r == "random_init":
            report["experiment_class"] = "CONTROL_random_init_dplus"
        else:
            report["experiment_class"] = "PRIMARY_frozen_dplus_transfer"
        out = RESULTS_DIR / f"role_{r}.json"
        write_json(out, report)
        logging.info("Wrote %s", out)
    return 0


def main() -> None:
    logger_setup()
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)

    ps = sub.add_parser("smoke", help="Bounded seed2 smoke gate")
    ps.add_argument("--data_config", default="data_config.json")
    ps.add_argument("--device", default="cuda:0")
    ps.add_argument("--batch_size", type=int, default=DEFAULT_EXTRACT_BS)
    ps.add_argument("--max_batches", type=int, default=DEFAULT_SMOKE_MAX_BATCHES)
    ps.set_defaults(func=cmd_smoke)

    pr = sub.add_parser("run", help="Full extract+eval for one role group")
    pr.add_argument(
        "--role",
        required=True,
        choices=("seed1", "seed2", "seed3", "controls"),
        help="controls = random_init CONTROL + FT seed2 SECONDARY",
    )
    pr.add_argument("--data_config", default="data_config.json")
    pr.add_argument("--device", default="cuda:0")
    pr.add_argument("--batch_size", type=int, default=DEFAULT_EXTRACT_BS)
    pr.add_argument("--skip_extract", action="store_true")
    pr.set_defaults(func=cmd_run)

    args = p.parse_args()
    logging.info("embeddings_root=%s results_dir=%s", EMBEDDINGS_ROOT, RESULTS_DIR)
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
