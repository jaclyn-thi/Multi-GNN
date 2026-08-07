#!/usr/bin/env python3
"""Post-smoke integrity aggregation for fixed-half GBT+TF arm."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gbt_tf_fixed_half_stdfloor_r198 import (  # noqa: E402
    OBJECTIVE_ID,
    SMOKE_CKPT_ROOT,
    SMOKE_RESULT_ROOT,
    WEIGHT_MODE,
)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--result-root", type=str, default=SMOKE_RESULT_ROOT)
    p.add_argument("--ckpt-root", type=str, default=SMOKE_CKPT_ROOT)
    args = p.parse_args(argv)

    root = ROOT / args.result_root
    ckpt_dir = ROOT / args.ckpt_root
    errors = []

    gates = json.loads((root / "smoke_integrity_gates.json").read_text(encoding="utf-8"))
    summary = json.loads((root / "smoke_summary.json").read_text(encoding="utf-8"))
    recipe = json.loads((root / "resolved_recipe.json").read_text(encoding="utf-8"))
    prov = json.loads((root / "memory_preflight_provenance.json").read_text(encoding="utf-8"))
    mem = json.loads(
        (
            ROOT
            / "results/diagnostics/financial_multidataset_gbt_tf_adaptive_stdfloor_1e4_memory_preflight"
            / "aggregate.json"
        ).read_text(encoding="utf-8")
    )

    rows = []
    with (root / "logs" / "steps.jsonl").open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    if len(rows) != 30:
        errors.append(f"expected 30 steps, got {len(rows)}")
    steps_per = summary.get("steps_per_domain") or {}
    for d, n in steps_per.items():
        if int(n) != 10:
            errors.append(f"{d} updates={n} != 10")

    beta_traj = []
    for row in rows:
        step = int(row["global_optimizer_step"])
        w_gbt = float(row.get("w_gbt", row.get("w_contrast", -1)))
        sum_tf = float(row.get("sum_w_tf", -1))
        if abs(w_gbt - 0.5) > 1e-5:
            errors.append(f"w_gbt={w_gbt} at step {step}")
        if abs(sum_tf - 0.5) > 1e-4:
            errors.append(f"sum_w_tf={sum_tf} at step {step}")
        for k in (
            "L_total",
            "L_gbt_raw",
            "L_gbt_norm",
            "weighted_gbt",
            "encoder_grad_norm",
            "moe_grad_norm",
            "view1_repr_grad_norm",
            "view2_repr_grad_norm",
        ):
            v = float(row.get(k, float("nan")))
            if not (v == v) or v < 0 and k.endswith("_grad_norm"):
                errors.append(f"bad {k} at {step}")
            if k.endswith("_grad_norm") and v <= 0:
                errors.append(f"zero {k} at {step}")
        recon = float(row.get("total_loss_reconstruction_error", float("nan")))
        if not (recon == recon) or recon > 1e-4:
            errors.append(f"recon={recon} at {step}")
        c_numel = row.get("C_numel")
        if c_numel is not None and int(c_numel) != 198 * 198:
            errors.append(f"C_numel={c_numel} at {step}")
        frozen = bool(row.get("alpha_beta_frozen"))
        betas = [float(row.get(f"beta_{i}", float("nan"))) for i in range(3)]
        beta_traj.append({"step": step, "frozen": frozen, "beta": betas, "w_gbt": w_gbt})
        if step <= 15 and not frozen:
            errors.append(f"beta not frozen at step {step}")
        if step >= 16 and frozen:
            errors.append(f"beta still frozen at step {step}")

    # First beta numerical change at/after 16
    first_beta_change = None
    prev = None
    for item in beta_traj:
        if prev is not None and item["beta"] != prev:
            first_beta_change = item["step"]
            break
        prev = item["beta"]
    expected_first = int(summary.get("beta_first_update_at") or 0) or first_beta_change
    if expected_first is None or int(expected_first) < 16:
        errors.append(f"first beta update={expected_first} expected >=16")

    ckpt_path = ckpt_dir / "checkpoint_step_0030.pt"
    if not ckpt_path.is_file():
        errors.append(f"missing checkpoint {ckpt_path}")
        ckpt_info = {}
    else:
        blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        ckpt_info = {
            "objective_id": blob.get("objective_id"),
            "weight_mode": (blob.get("recipe") or {}).get("weight_mode"),
            "alpha_policy": (blob.get("recipe") or {}).get("alpha_policy"),
            "global_step": blob.get("global_step"),
            "test_evaluated": blob.get("test_evaluated"),
        }
        if blob.get("objective_id") != OBJECTIVE_ID:
            errors.append("ckpt objective mismatch")
        if (blob.get("recipe") or {}).get("weight_mode") != WEIGHT_MODE:
            errors.append("ckpt weight_mode mismatch")
        groups = (blob.get("optimizer_state_dict") or {}).get("param_groups") or []
        if len(groups) < 2 or len(groups[1].get("params", [])) != 1:
            errors.append(f"optimizer must have beta-only group; groups={[(len(g.get('params', []))) for g in groups]}")
        a = float(torch.sigmoid(blob["alpha_beta_state_dict"]["alpha_logit"]))
        if abs(a - 0.5) > 1e-4:
            errors.append(f"ckpt alpha={a} != 0.5")
        if blob.get("test_evaluated") is True:
            errors.append("test_evaluated true in ckpt")

    if not gates.get("ok"):
        errors.extend(list(gates.get("errors") or ["gates_ok false"]))
    if summary.get("status") != "smoke_complete":
        errors.append(f"status={summary.get('status')}")
    if not bool(mem.get("ok")) or str(mem.get("job_id")) != "19625552":
        errors.append("trusted memory PASS missing/mismatched")
    if not bool(prov.get("matches_trusted_job")):
        errors.append("memory provenance mismatch")
    if recipe.get("weight_mode") != WEIGHT_MODE:
        errors.append("recipe weight_mode mismatch")

    payload = {
        "ok": not errors,
        "errors": errors,
        "n_steps": len(rows),
        "steps_per_domain": steps_per,
        "alpha_unfrozen_at": summary.get("alpha_unfrozen_at"),
        "beta_first_update_at": summary.get("beta_first_update_at") or first_beta_change,
        "w_gbt_always_0_5": all(abs(float(r.get("w_gbt", -1)) - 0.5) < 1e-5 for r in rows),
        "sum_w_tf_always_0_5": all(abs(float(r.get("sum_w_tf", -1)) - 0.5) < 1e-4 for r in rows),
        "beta_trajectory_head": beta_traj[:5],
        "beta_trajectory_tail": beta_traj[-5:],
        "loss_head": [
            {
                "step": int(r["global_optimizer_step"]),
                "L_total": r.get("L_total"),
                "L_gbt_norm": r.get("L_gbt_norm"),
                "weighted_gbt": r.get("weighted_gbt"),
                "weighted_tf_0": r.get("weighted_tf_0"),
                "weighted_tf_1": r.get("weighted_tf_1"),
                "weighted_tf_2": r.get("weighted_tf_2"),
                "recon": r.get("total_loss_reconstruction_error"),
            }
            for r in rows[:3]
        ],
        "grad_tail": [
            {
                "step": int(r["global_optimizer_step"]),
                "encoder": r.get("encoder_grad_norm"),
                "moe": r.get("moe_grad_norm"),
                "view1": r.get("view1_repr_grad_norm"),
                "view2": r.get("view2_repr_grad_norm"),
                "alpha_beta": r.get("alpha_beta_grad_norm"),
            }
            for r in rows[-3:]
        ],
        "checkpoint": ckpt_info,
        "smoke_gates_ok": gates.get("ok"),
        "memory_verdict": mem.get("verdict"),
        "memory_job_id": mem.get("job_id"),
        "schedule_horizon": summary.get("schedule_horizon"),
        "executed_stop_step": summary.get("executed_stop_step"),
        "test_evaluated": False,
    }
    out = root / "smoke_final_integrity.json"
    out.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, default=str))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
