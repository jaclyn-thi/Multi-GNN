# Graph Barlow Twins R198 — implementation note

**Mode:** `MIXED_3DOMAIN_GRAPH_BARLOW_TWINS_ONLY`  
**Objective:** `edge_aligned_graph_barlow_twins_r198`  
**Phase:** implementation + focused synthetic tests only  
**Official reference commit:** `ec62580aa89bf3f0d20c92e7549031deedc105ab`  
**Audits:** `notes/financial_multidataset_graph_barlow_twins_audit.md`,
`results/diagnostics/financial_multidataset_graph_barlow_twins_audit.json`

## What was implemented

Isolated package `graph_barlow_twins_r198/` (does **not** modify Phase-3/4 trainers
or checkpoints):

| File | Role |
|------|------|
| `__init__.py` | Arm/recipe constants, LONG-matched schedule knobs, memory-risk note |
| `loss.py` | `edge_aligned_graph_barlow_twins_r198` + DxD-only `_feature_cross_correlation` |
| `integrity.py` | Objective-aware gates; test-split refusal |
| `checkpoint.py` | Save/load preserving objective config |
| `step.py` | Dual-view live GNN step (no InfoNCE/TF/αβ/proj/detach) |
| `scripts/train_mixed_3domain_graph_barlow_twins_only.py` | Config/recipe entry; `--run-train` blocked |
| `slurm/run_mixed_3domain_graph_barlow_twins_only_smoke.sh` | Proposed advanced-GPU smoke (not executed) |
| `tests/test_graph_barlow_twins_r198.py` | Focused synthetic coverage |

## Exact implemented formula

For aligned seeds \(Z_a,Z_b\in\mathbb{R}^{B\times 198}\) with identical EdgeIDs/order,
both requiring grad:

```text
λ = 1/198
ε = 1e-15
Za_norm = (Za - mean_0) / (std_0_unbiased + ε)   # NOT (Za-mean)/std + ε
Zb_norm = (Zb - mean_0) / (std_0_unbiased + ε)
C = mm(Za_norm^T, Zb_norm) / B          # shape (198,198) only
L_invariance = Σ_i (1 - C_ii)^2
L_redundancy = λ * Σ_{i≠j} C_ij^2
L_total = L_invariance + L_redundancy
```

**Formula check:** code in `graph_barlow_twins_r198/loss.py` uses
`(z - mean) / (std_unbiased + 1e-15)`. Any prose that looks like
“divide by std, then add eps” is a rendering typo only — semantics are unchanged.

**Identical-view subtlety:** unbiased `std` + divide-by-`B` ⇒ \(C_{ii}\approx(B-1)/B\),
so \(L_{\mathrm{invariance}}\approx D/B^2\) (not exact zero). Tests use that analytic
target with a justified tolerance.

## Hard memory contract (loss path)

Allowed: \(B\times198\), \(198\times198\), length-198, scalars.  
Forbidden: \(B\times B\), \(N\times N\), dense adj, all-pairs seeds, full-dataset reps.  
Enforced via `_feature_cross_correlation` (`torch.mm` on \(D\times B\) and \(B\times D\) only)
plus runtime asserts on shapes / EdgeIDs / finiteness.

**Not claimed safe:** the full symmetric encoder path (two live autograd graphs)
may use substantially more VRAM than asymmetric InfoNCE — measure on GPU before
smoke/`--run-train`.

## Logging fields prepared

`L_gbt_total`, `L_invariance`, `L_redundancy`, `reconstruction_error`,
mean/min/max diagonal \(C\), off-diagonal RMS, per-view std min/median/max,
near-dead dim counts at thresholds \((10^{-6},10^{-4},10^{-3})\) (batch-local;
not auto-labeled collapse), R198 mean L2, effective rank, encoder / view1 /
view2 grad norms, LR, batch size, domain, seed/view hashes, \(C\) shape + bytes.

## Focused tests

```bash
/home/jthi/.conda/envs/multignn/bin/python -m pytest -q \
  tests/test_graph_barlow_twins_r198.py \
  tests/test_phase4b_objective_ablation.py --tb=line
```

**Result: 26 passed** (GBT suite + existing Phase-4B objective ablation tests).  
Report: `results/diagnostics/financial_multidataset_graph_barlow_twins_focused_tests.json`

## Proposed 30-step smoke configuration

See `results/diagnostics/financial_multidataset_shared_core_mixed_3domain_gbt_only/smoke30_recipe.json`.

- Arm: `MIXED_3DOMAIN_GRAPH_BARLOW_TWINS_ONLY`
- Steps: 30; projection/TF/InfoNCE/αβ off; both views require grad
- Augment: edge_drop=0.1, attr_mask=0.1; val-only / no test
- Init: Phase-3 shared init (when train is enabled)
- **Prerequisite:** dual-view VRAM measurement before enabling `--run-train`

## Exact advanced-account sbatch command (DO NOT execute yet)

```bash
sbatch slurm/run_mixed_3domain_graph_barlow_twins_only_smoke.sh
```

Equivalent explicit form:

```bash
sbatch --partition=mit_normal_gpu \
  --account=mit_amf_advanced_gpu \
  --qos=mit_amf_advanced_gpu \
  --gres=gpu:1 --cpus-per-task=16 --mem=128G --time=01:00:00 \
  slurm/run_mixed_3domain_graph_barlow_twins_only_smoke.sh
```

Current smoke script runs focused pytest + recipe dump; full dataset train remains
gated (`--run-train` raises until approved).

## Confirmations

- No Slurm job submitted
- No full financial graph / dataset load for training
- No encoder training run
- No test-split access
- No `dvc pull` / official artifact download
- Historical Phase-3/Phase-4 trainers and checkpoints untouched

**Stop for human review.**
