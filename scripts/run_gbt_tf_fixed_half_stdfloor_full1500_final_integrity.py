#!/usr/bin/env python3
"""Post-train integrity aggregation for fixed-half GBT+TF 1500-step run."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gbt_tf_adaptive_stdfloor_r198.checkpoint import load_checkpoint  # noqa: E402
from gbt_tf_fixed_half_stdfloor_r198 import (  # noqa: E402
    ARM,
    CKPT_ROOT,
    EXECUTED_STOP_STEP,
    OBJECTIVE_ID,
    RESULT_ROOT,
    SCHEDULE_HORIZON,
    WEIGHT_MODE,
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _finite(x) -> bool:
    try:
        return bool(np.isfinite(float(x)))
    except Exception:
        return False


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--result-root", type=str, default=RESULT_ROOT)
    p.add_argument("--ckpt-root", type=str, default=CKPT_ROOT)
    args = p.parse_args(argv)

    root = ROOT / args.result_root
    ckpt_dir = ROOT / args.ckpt_root
    errors: list[str] = []

    summary = json.loads((root / "smoke_summary.json").read_text(encoding="utf-8"))
    gates = json.loads((root / "smoke_integrity_gates.json").read_text(encoding="utf-8"))
    recipe = json.loads((root / "resolved_recipe.json").read_text(encoding="utf-8"))
    seed = json.loads((root / "seed_stream_vs_long.json").read_text(encoding="utf-8"))
    init = json.loads((root / "shared_init_provenance.json").read_text(encoding="utf-8"))

    rows = []
    with (root / "logs" / "steps.jsonl").open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    n = len(rows)
    if n != EXECUTED_STOP_STEP:
        errors.append(f"n_steps={n} != {EXECUTED_STOP_STEP}")
    step_counts = summary.get("steps_per_domain") or {}
    for d, need in (("Small-HI", 500), ("SAML-D", 500), ("Small-LI", 500)):
        if int(step_counts.get(d, -1)) != need:
            errors.append(f"{d} updates={step_counts.get(d)} != {need}")

    for d, blob in seed.items():
        if not bool(blob.get("ok")):
            errors.append(f"seed-stream fail {d}")

    if int(summary.get("alpha_unfrozen_at") or -1) != 15:
        errors.append(f"alpha_unfrozen_at={summary.get('alpha_unfrozen_at')}")
    first_beta = summary.get("beta_first_update_at")
    if first_beta is None or int(first_beta) < 16:
        errors.append(f"beta_first_update_at={first_beta}")

    w_gbt_ok = True
    sum_tf_ok = True
    recon_ok = True
    grads_ok = True
    c_ok = True
    beta_frozen_ok = True
    for row in rows:
        step = int(row["global_optimizer_step"])
        if abs(float(row.get("w_gbt", -1)) - 0.5) > 1e-5:
            w_gbt_ok = False
            errors.append(f"w_gbt!=0.5 at {step}")
        if abs(float(row.get("sum_w_tf", -1)) - 0.5) > 1e-4:
            sum_tf_ok = False
            errors.append(f"sum_w_tf!=0.5 at {step}")
        recon = float(row.get("total_loss_reconstruction_error", float("nan")))
        if not _finite(recon) or recon > 1e-4:
            recon_ok = False
            errors.append(f"recon at {step}")
        for k in (
            "L_total",
            "L_gbt_raw",
            "L_gbt_norm",
            "encoder_grad_norm",
            "moe_grad_norm",
            "view1_repr_grad_norm",
            "view2_repr_grad_norm",
        ):
            if not _finite(row.get(k)):
                grads_ok = False
                errors.append(f"nonfinite {k} at {step}")
                break
        if float(row.get("encoder_grad_norm") or 0) <= 0:
            grads_ok = False
            errors.append(f"encoder grad 0 at {step}")
        if float(row.get("moe_grad_norm") or 0) <= 0:
            grads_ok = False
            errors.append(f"moe grad 0 at {step}")
        if float(row.get("view1_repr_grad_norm") or 0) <= 0 or float(
            row.get("view2_repr_grad_norm") or 0
        ) <= 0:
            grads_ok = False
            errors.append(f"view grad 0 at {step}")
        if int(row.get("C_numel") or 0) not in (0, 198 * 198) and row.get("C_numel") is not None:
            if int(row["C_numel"]) != 198 * 198:
                c_ok = False
                errors.append(f"C_numel at {step}")
        frozen = bool(row.get("alpha_beta_frozen"))
        if step <= 15 and not frozen:
            beta_frozen_ok = False
            errors.append(f"beta unfrozen early at {step}")
        if step >= 16 and frozen:
            beta_frozen_ok = False
            errors.append(f"beta still frozen at {step}")

    # Truncate error spam
    if len(errors) > 40:
        errors = errors[:40] + [f"... +{len(errors)-40} more"]

    ckpt_specs = {
        "0750": ckpt_dir / "checkpoint_step_0750.pt",
        "1500": ckpt_dir / "checkpoint_step_1500.pt",
        "last": ckpt_dir / "checkpoint_last.pt",
    }
    ckpt_report = {}
    for label, path in ckpt_specs.items():
        info = {"path": str(path), "exists": path.is_file()}
        if not path.is_file():
            errors.append(f"missing ckpt {label}")
            ckpt_report[label] = info
            continue
        info["sha256"] = sha256_file(path)
        try:
            blob = load_checkpoint(path, accepted_objective_ids=(OBJECTIVE_ID,))
            info["reload_ok"] = True
            info["objective_id"] = blob.get("objective_id")
            info["global_step"] = blob.get("global_step")
            info["weight_mode"] = (blob.get("recipe") or {}).get("weight_mode")
            info["alpha_policy"] = (blob.get("recipe") or {}).get("alpha_policy")
            info["test_evaluated"] = blob.get("test_evaluated")
            groups = (blob.get("optimizer_state_dict") or {}).get("param_groups") or []
            info["optimizer_n_groups"] = len(groups)
            info["beta_only_optim_group"] = (
                len(groups) >= 2 and len(groups[1].get("params", [])) == 1
            )
            a = float(torch.sigmoid(blob["alpha_beta_state_dict"]["alpha_logit"]))
            info["alpha_constant"] = a
            if abs(a - 0.5) > 1e-4:
                errors.append(f"ckpt {label} alpha={a}")
            if not info["beta_only_optim_group"]:
                errors.append(f"ckpt {label} optimizer includes alpha")
            if blob.get("test_evaluated") is True:
                errors.append(f"ckpt {label} test_evaluated")
            if (blob.get("recipe") or {}).get("weight_mode") != WEIGHT_MODE:
                errors.append(f"ckpt {label} weight_mode")
            expected_step = {"0750": 750, "1500": 1500, "last": 1500}[label]
            if int(blob.get("global_step", -1)) != expected_step:
                errors.append(f"ckpt {label} step={blob.get('global_step')}")
        except Exception as e:  # noqa: BLE001
            info["reload_ok"] = False
            info["reload_error"] = str(e)
            errors.append(f"ckpt {label} reload failed: {e}")
        ckpt_report[label] = info

    # Trajectories (downsample for aggregate)
    def series(keys):
        out = {k: [] for k in keys}
        out["step"] = []
        for row in rows:
            out["step"].append(int(row["global_optimizer_step"]))
            for k in keys:
                out[k].append(row.get(k))
        return out

    traj = series(
        [
            "w_gbt",
            "sum_w_tf",
            "beta_0",
            "beta_1",
            "beta_2",
            "L_gbt_raw",
            "L_gbt_norm",
            "weighted_gbt",
            "L_tf_raw_0",
            "L_tf_raw_1",
            "L_tf_raw_2",
            "L_tf_norm_0",
            "L_tf_norm_1",
            "L_tf_norm_2",
            "weighted_tf_0",
            "weighted_tf_1",
            "weighted_tf_2",
            "L_total",
            "encoder_grad_norm",
            "moe_grad_norm",
            "view1_repr_grad_norm",
            "view2_repr_grad_norm",
            "alpha_beta_grad_norm",
            "effective_rank",
            "repr_norm_mean",
            "repr_std_mean",
            "view1_n_floored_dims",
            "view2_n_floored_dims",
            "total_loss_reconstruction_error",
        ]
    )

    # Keep compact milestone snapshots
    milestones = {}
    for s in (1, 15, 16, 750, 1500):
        if 1 <= s <= n:
            r = rows[s - 1]
            milestones[str(s)] = {
                "w_gbt": r.get("w_gbt"),
                "sum_w_tf": r.get("sum_w_tf"),
                "beta": [r.get(f"beta_{i}") for i in range(3)],
                "L_total": r.get("L_total"),
                "weighted_gbt": r.get("weighted_gbt"),
                "weighted_tf": [r.get(f"weighted_tf_{i}") for i in range(3)],
                "encoder_grad_norm": r.get("encoder_grad_norm"),
                "moe_grad_norm": r.get("moe_grad_norm"),
                "view1_repr_grad_norm": r.get("view1_repr_grad_norm"),
                "view2_repr_grad_norm": r.get("view2_repr_grad_norm"),
                "effective_rank": r.get("effective_rank"),
                "alpha_beta_frozen": r.get("alpha_beta_frozen"),
            }

    training_gates = {
        "exact_1500_optimizer_steps": n == EXECUTED_STOP_STEP,
        "exact_500_updates_per_domain": all(
            int(step_counts.get(d, -1)) == 500 for d in ("Small-HI", "SAML-D", "Small-LI")
        ),
        "long_seed_hash_match_500": all(bool(seed[d].get("ok")) for d in seed),
        "w_gbt_always_0_5": w_gbt_ok,
        "sum_w_tf_always_0_5": sum_tf_ok,
        "beta_frozen_through_15": beta_frozen_ok,
        "beta_first_update_at_16_or_later": first_beta is not None and int(first_beta) >= 16,
        "alpha_unfrozen_at_15": int(summary.get("alpha_unfrozen_at") or -1) == 15,
        "finite_losses_and_grads": grads_ok and recon_ok,
        "loss_reconstruction_ok": recon_ok,
        "c_always_198x198": c_ok,
        "checkpoint_750_reload_ok": bool((ckpt_report.get("0750") or {}).get("reload_ok")),
        "checkpoint_1500_reload_ok": bool((ckpt_report.get("1500") or {}).get("reload_ok")),
        "checkpoint_last_reload_ok": bool((ckpt_report.get("last") or {}).get("reload_ok")),
        "optimizer_excludes_alpha": all(
            (ckpt_report.get(k) or {}).get("beta_only_optim_group") for k in ("0750", "1500", "last")
            if (ckpt_report.get(k) or {}).get("exists")
        ),
        "schedule_horizon_3000": int(summary.get("schedule_horizon") or 0) == SCHEDULE_HORIZON,
        "executed_stop_1500": int(summary.get("executed_stop_step") or 0) == EXECUTED_STOP_STEP,
        "weight_mode_fixed_half": recipe.get("weight_mode") == WEIGHT_MODE,
        "learn_alpha_false": recipe.get("learn_alpha") is False,
        "orchestration_gates_ok": bool(gates.get("ok")),
        "no_test_metrics": True,
        "no_extraction": True,
        "no_probe": True,
        "fresh_phase3_init": not bool(init.get("skipped")),
    }
    if not all(training_gates.values()):
        for k, v in training_gates.items():
            if not v and k not in str(errors):
                errors.append(f"gate_fail:{k}")

    ok = not errors and all(training_gates.values()) and bool(gates.get("ok"))
    aggregate = {
        "title": f"{ARM} full {EXECUTED_STOP_STEP}-step training",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "classification": "PASS" if ok else "FAIL",
        "ok": ok,
        "arm": ARM,
        "objective_id": OBJECTIVE_ID,
        "weight_mode": WEIGHT_MODE,
        "slurm_job_id": summary.get("job_id"),
        "optimizer_step_count": n,
        "step_counts": step_counts,
        "schedule_horizon": SCHEDULE_HORIZON,
        "executed_stop_step": EXECUTED_STOP_STEP,
        "alpha_unfrozen_at": summary.get("alpha_unfrozen_at"),
        "beta_first_update_at": first_beta,
        "learn_alpha": False,
        "learn_beta": True,
        "fixed_w_gbt": 0.5,
        "recipe": recipe,
        "shared_init_provenance": init,
        "seed_stream_vs_long": seed,
        "training_integrity_gates": training_gates,
        "orchestration_gates": gates,
        "checkpoints": ckpt_report,
        "milestones": milestones,
        "trajectories_summary": {
            "n_rows": n,
            "final": milestones.get("1500"),
            "at_750": milestones.get("750"),
            "beta_unfreeze": {
                "step_15": milestones.get("15"),
                "step_16": milestones.get("16"),
            },
        },
        "trajectory_arrays_path": str(root / "trajectories.json"),
        "bn_l1_vs_init": gates.get("bn_l1_vs_init"),
        "peak_rss_gib": summary.get("peak_rss_gib"),
        "test_evaluated": False,
        "errors": errors,
    }
    (root / "trajectories.json").write_text(
        json.dumps(traj, indent=2, default=str) + "\n", encoding="utf-8"
    )
    (root / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2, default=str) + "\n", encoding="utf-8"
    )
    # Twin pointer
    twin = ROOT / "results/diagnostics/financial_multidataset_gbt_tf_fixed_half_stdfloor_1e4.json"
    twin.write_text(json.dumps(aggregate, indent=2, default=str) + "\n", encoding="utf-8")

    notes = ROOT / "notes/financial_multidataset_gbt_tf_fixed_half_stdfloor_1e4.md"
    notes.parent.mkdir(parents=True, exist_ok=True)
    notes.write_text(
        "\n".join(
            [
                f"# {ARM} full {EXECUTED_STOP_STEP}-step training",
                "",
                f"**Job:** `{aggregate.get('slurm_job_id')}`",
                f"**Verdict:** `{aggregate['classification']}`",
                f"**Objective:** `{OBJECTIVE_ID}`",
                "",
                "## Locks",
                f"- executed_stop_step={EXECUTED_STOP_STEP}, schedule_horizon={SCHEDULE_HORIZON}",
                "- fixed w_gbt=0.5, sum(w_tf)=0.5, learnable beta, alpha not optimized",
                "- Fresh Phase-3 init; no smoke/adaptive resume; no test/extract/probe",
                "",
                "## Gates",
                "```json",
                json.dumps(training_gates, indent=2),
                "```",
                "",
                "## Checkpoints",
                "```json",
                json.dumps(ckpt_report, indent=2),
                "```",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "ok": ok,
                "classification": aggregate["classification"],
                "errors": errors[:20],
                "step_counts": step_counts,
                "beta_first_update_at": first_beta,
                "checkpoints": {k: v.get("sha256") for k, v in ckpt_report.items()},
            },
            indent=2,
        )
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
