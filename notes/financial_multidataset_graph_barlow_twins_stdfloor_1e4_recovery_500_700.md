# GBT std-floor recovery scout (`GBT_STDFLOOR_1E4_RECOVERY_500_TO_700`)

## Historical automated verdict (immutable)

**`FAIL_INTEGRITY`**

Preserved exactly as written by job `19603171` automation: resumed NeighborLoader
seed batches did not match original steps 501–587 (`seed_ids_sha256` match rate
0/87). This historical label must **not** be overwritten or relabeled.

Raw aggregate field remains:
`"verdict": "FAIL_INTEGRITY"`
in `results/diagnostics/financial_multidataset_graph_barlow_twins_stdfloor_1e4_recovery_500_700.json`.

## Human-reviewed conclusion (addendum)

**`NUMERICAL_STABILITY_PASS_STREAM_UNMATCHED`**

Evidence supporting (but not proving via exact paired replay) that
`std_floor=1e-4` fixes the denominator instability:

- 200/200 steps completed through peak LR
- zero >1e12 view-gradient spikes
- max view grad ~263 vs original ~2.6e13
- original failure region around step 588 remained finite
- floor active lightly on only ~9–19% of domain steps
- source checkpoint SHA unchanged
  (`b8e1b6eb0ca03fe6228d2db1dc7a21e61010028a12a7fd7350a971400081382f`)

Authorized follow-up (separate DAG; not a continuation of this recovery
checkpoint): fresh full 3000-step std-floor training from Phase-3 shared init
plus gated validation-only evaluation.

## Job metadata

- **Intervention:** B only — `std_safe = clamp_min(std_raw, 1e-4)`
- **Objective:** `edge_aligned_graph_barlow_twins_r198_stdfloor_1e4`
- **Job:** `19603171`
- **Steps:** 500 → 700 (200 new optimizer steps)

## Integrity detail

| Gate | Result |
|------|--------|
| Source SHA unchanged | yes |
| Final exposures | HI/SAML/LI = 234/233/233 (new 67/66/67) |
| Optimizer/scheduler @700 | ok |
| Recovery ckpts 550/600/650/700 reload | ok |
| Domain RR + LR schedule | matched |
| Batch-stream seed equality 501–587 | **failed** (basis of FAIL_INTEGRITY) |
| Aug hash equality 501–587 | failed once seeds diverged |

## Gradients

| Metric | Original 501–587 | Recovery 501–700 |
|--------|------------------|------------------|
| Max view-repr grad | ~2.63×10¹³ | ~263 |
| >1e12 view spikes | 6 | 0 |
| Max encoder grad | ~137 | ~627 (finite) |
| Non-finite @588 region | NaN encoder @588 | finite |

## Confirmations

- historical automated `FAIL_INTEGRITY` unchanged
- human conclusion recorded separately as `NUMERICAL_STABILITY_PASS_STREAM_UNMATCHED`
- recovery checkpoint is **not** the starting point for the authorized full run
- no test access
