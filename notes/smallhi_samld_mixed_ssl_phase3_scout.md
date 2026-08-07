# Small-HI + SAML-D mixed SSL Phase-3 scout

> Twin: `results/diagnostics/smallhi_samld_mixed_ssl_phase3_scout.json`  
> Follows Phase-2 smoke job `19515484` (`ok=true`).

## Purpose

Does training on both Small-HI and SAML-D improve, preserve, or damage
representations relative to single-domain training under the exact same
`smallhi_samld_shared_core_v1` feature and model protocol?

**Primary comparison:** equal optimizer budget — all arms at step 1000.  
**Secondary (diagnostic only):** mixed@500/domain vs single@500 — **not perfectly LR-phase matched** for domain exposure (single reaches exposure 500 at global step 500 / mid-decay; mixed reaches 500/domain at global step 1000 / end of schedule).

## Arms (seed 2, 1000 steps)

| Array | Arm | Schedule |
|---|---|---|
| 0 | `SMALL_HI_ONLY` | 1000× Small-HI |
| 1 | `SAMLD_ONLY` | 1000× SAML-D (`samld_calendar_day_rezero_v1`) |
| 2 | `MIXED_1TO1` | alternating → 500/500 |

## Locked recipe

- Contract `smallhi_samld_shared_core_v1` = base2 + ports2 + TDS2 (`edge_dim=6`); **not** historical base4+ports2
- Shared GIN from scratch; shared init SHA artifact
- R198 InfoNCE; projection **false**; preserve_seed_edges **false**; AMP **false**
- Adaptive TF-MoE (3 causal targets); shared experts; global α/β
- Domain BN bundles; per-domain edge/TF scalers; per-domain LossNormState
- α/β frozen through global step 10 in **all** arms (matched); LossNorm calib = first 5 obs/domain
- LR: encoder 2e-3, α/β 1e-3; warmup 200 + linear decay 800
- Checkpoints: 250 / 500 / 1000 + rolling `checkpoint_last.tar` every 100

## Runtime estimate

Phase-2: ~850s / 50 mixed steps ≈ **17 s/step** → 1000 steps ≈ **4.72 h** compute + ~1 h load/overhead ≈ **~6 h/arm**. Slurm limit **08:00:00**.

## Status

- Focused tests: **23 passed**; preflight `ok=true`
- Array job **`19517322`** (tasks 0–2, `%2`): all arms `ok=true`, no preemption
  - 0 → `SMALL_HI_ONLY` job `19517334` (~31.4 min)
  - 1 → `SAMLD_ONLY` job `19517335` (~47.3 min)
  - 2 → `MIXED_1TO1` job `19517322` (~46.8 min)
- Integrity: [`training_integrity_summary.json`](../results/diagnostics/smallhi_samld_mixed_ssl_phase3_scout/training_integrity_summary.json) (`ok=true`)
- Init SHA equal; seed-stream match 500/500 both domains
- No extraction / probe / test / DAG
- Stop for human review before 6-cell validation eval

## Manual resume (no auto-resubmit)

```bash
sbatch --export=ALL,PHASE3_ARM=MIXED_1TO1,PHASE3_RESUME=results/checkpoints/smallhi_samld_mixed_ssl_phase3_scout_seed2/MIXED_1TO1/checkpoint_last.tar \
  slurm/run_mixed_ssl_phase3_scout.sh
```

Checkpoints live under `results/checkpoints/...` (not `saved-models/`) so the scout remains writable on the shared filesystem.

## Proposed next eval (NOT submitted)

Six cells: {SMALL_HI_ONLY, SAMLD_ONLY, MIXED_1TO1}@step1000 × {Small-HI, SAML-D} targets; frozen full-subgraph R198; validation-only; no test access.

## Post-training integrity
**ok=True** — validation-free training-integrity only.

## Seed-stream matching
- Small-HI: `{'domain': 'Small-HI', 'ok': True, 'n_compared': 500, 'n_mismatches': 0, 'first_mismatch_index': None, 'first_8_single': ['a8dc63a6255687784c99a02878b452d8866a2b8451e7e5c4b0bbb6eb59a6868c', 'ee690620ae5a508ca0c0624282d3a11542c33102639441435536171400385770', 'cc0f584442b6552712c7725abd4afe70c0e44286dd4561330875ae198e6b7f3c', '8aa107fb972878f46a29340f8b6b5a59153c0a97279eee43cad3b1a2aaddf07a', '346c939ff61c08d9c8d68f97c233a00a7e5c20a5e7cf706ac865060cfd2d9304', '394642830e5601b2db6da623d9f12d93a6e7d1f04a9590845eaa24a189ac1506', '609f815bd73245b57cfea1f18b5c43c3532c9083e1eb6ca440a269474b421815', '016053fa982226d5bc044411dd12758cc84b809455259c36f8c112c716e6e9fd'], 'first_8_mixed': ['a8dc63a6255687784c99a02878b452d8866a2b8451e7e5c4b0bbb6eb59a6868c', 'ee690620ae5a508ca0c0624282d3a11542c33102639441435536171400385770', 'cc0f584442b6552712c7725abd4afe70c0e44286dd4561330875ae198e6b7f3c', '8aa107fb972878f46a29340f8b6b5a59153c0a97279eee43cad3b1a2aaddf07a', '346c939ff61c08d9c8d68f97c233a00a7e5c20a5e7cf706ac865060cfd2d9304', '394642830e5601b2db6da623d9f12d93a6e7d1f04a9590845eaa24a189ac1506', '609f815bd73245b57cfea1f18b5c43c3532c9083e1eb6ca440a269474b421815', '016053fa982226d5bc044411dd12758cc84b809455259c36f8c112c716e6e9fd']}`
- SAML-D: `{'domain': 'SAML-D', 'ok': True, 'n_compared': 500, 'n_mismatches': 0, 'first_mismatch_index': None, 'first_8_single': ['d8f063f8af948f521f75948c992d3122c1e5f020c56b47083462d58d6a63851f', '593eaba573fd53e43f4d9eb3983ab9cdc080b6ffff2dbb0c0e833b7f8d548511', 'ab93d07a96ea7e9702ca2b30e4f83c4a6e0ef6c7ad24f97070cb8d96e92ffcd2', '2db5219acfff78af58c47ddb51764d2368494aee139553a759018df9470c1267', '6a640682eb94e95c4efee75f6b3f3b3bb7112530c90c4dd0e62d57634512da24', 'ece9edc2e8a53773f2331208fc94780178dc92fa1382858d8f6808efd780f017', '03186ddf4424b6f38777641a6469e312fc73a57f51452d3c03749968d55b291d', 'fcd51a84f578501852f95732d0b4d716290b51b571475f9f8290289b72289897'], 'first_8_mixed': ['d8f063f8af948f521f75948c992d3122c1e5f020c56b47083462d58d6a63851f', '593eaba573fd53e43f4d9eb3983ab9cdc080b6ffff2dbb0c0e833b7f8d548511', 'ab93d07a96ea7e9702ca2b30e4f83c4a6e0ef6c7ad24f97070cb8d96e92ffcd2', '2db5219acfff78af58c47ddb51764d2368494aee139553a759018df9470c1267', '6a640682eb94e95c4efee75f6b3f3b3bb7112530c90c4dd0e62d57634512da24', 'ece9edc2e8a53773f2331208fc94780178dc92fa1382858d8f6808efd780f017', '03186ddf4424b6f38777641a6469e312fc73a57f51452d3c03749968d55b291d', 'fcd51a84f578501852f95732d0b4d716290b51b571475f9f8290289b72289897']}`

## Init SHA equality
- equal=True: `{'SMALL_HI_ONLY': '8821c986c7394caf504393830dc33a9c3c97ba4d5fdd3bcbaa19f70421c7aebc', 'SAMLD_ONLY': '8821c986c7394caf504393830dc33a9c3c97ba4d5fdd3bcbaa19f70421c7aebc', 'MIXED_1TO1': '8821c986c7394caf504393830dc33a9c3c97ba4d5fdd3bcbaa19f70421c7aebc'}`

## Secondary comparison caveat
Secondary exposure-matched diagnostic (mixed@500/domain vs single@500) is NOT perfectly LR-phase matched for domain exposure: single-domain arms reach domain exposure 500 at global step 500 (mid-decay), while MIXED_1TO1 reaches 500/domain at global step 1000 (end of schedule).

## Proposed next eval (NOT submitted)
{
  "encoders": [
    "SMALL_HI_ONLY@step1000",
    "SAMLD_ONLY@step1000",
    "MIXED_1TO1@step1000"
  ],
  "targets": [
    "Small-HI",
    "SAML-D"
  ],
  "cells": 6,
  "protocol": "frozen full-subgraph R198; same downstream probe; validation-only",
  "no_test_access": true,
  "submitted": false
}

Stop for human review before extraction.

