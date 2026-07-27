# Final exploratory SSL scout — DAG submission

> Queued without waiting for smoke. No result analysis. No J/JM/JC.
> Final allowed infrastructure attempt after failed paired-batch smokes.

## Design class

- `matched_configuration_one_seed_exploratory_ablation`
- `exact_batch_pairing=false` (first-N batch hashes logged only; not a hard gate)

## Fixes vs prior smoke `18994271`

| Issue | Fix |
|-------|-----|
| M-only morph init reseeding global RNG | Explicit CPU/CUDA/numpy/python RNG save/restore around morph-head init (no `set_seed` restore) |
| Cross-process batch-hash hard gate | Relaxed; hashes diagnostic only |
| Optimizer restore mismatch | Explicit `load_model(..., load_optimizer=False)` for weight continuation; true-resume remains strict |
| Heavy smoke | Reduced: 3 opt steps/arm, accum=1, single Small-HI load, no PaySim/logistic |

## Job table

| Role | Job ID | Script | Dependency | Partition / GPU |
|------|--------|--------|------------|-----------------|
| Smoke | **19001714** | `slurm/run_final_exploratory_ssl_scout_smoke.sh` | (none) | `mit_normal_gpu` / 1 GPU |
| C0 | **19001715** | `slurm/run_final_exploratory_ssl_scout_arm.sh` (`ARM=C0`) | `afterok:19001714` | `mit_normal_gpu` / **1 GPU** |
| M | **19001716** | `slurm/run_final_exploratory_ssl_scout_arm.sh` (`ARM=M`) | `afterok:19001714` | `mit_normal_gpu` / **1 GPU** |
| Aggregate | **19001717** | `slurm/run_final_exploratory_ssl_scout_aggregate.sh` | `afterok:19001715:19001716` | `mit_normal` / **no GPU** |

## Dependency graph

```text
19001714 (reduced smoke)
    ├─ afterok → 19001715 (C0 500-step + val-only)
    └─ afterok → 19001716 (M 500-step + val-only)
                    └─ both afterok → 19001717 (CPU aggregate)
```

If smoke fails, C0/M/agg become `DependencyNeverSatisfied` and never execute. No automatic resubmission.

## Stale cleanup

- Queue empty at submit; no live `DependencyNeverSatisfied` jobs to cancel.
- Prior failed dependents `18994272–18994274` already gone.
- Removed leftover smoke staged ckpts for `18994271`.

## Locked recipe (full arms)

- Source: `saved-models/checkpoint_gin_emlps_ports_tds_corrected_asym_proj_8192neg_queue0_40ep_seed2.tar`
- SHA256: `18e06f555aa4880dfc1e95caa3f54a207e5aa186d266887772640feb93a06ae6`
- Provenance: `checkpoint_weight_continuation_with_optimizer_reset`
- Exactly **500** optimizer steps; `loader_num_workers=0`
- Matched except M: `degree_fan` + `flow_balance`, `λ_morph=0.05`
- Validation only; test never inspected
- Unique names: `final_exploratory_ssl_scout_{c0,m}_seed2`

## Aggregate

- Predeclared M-vs-C0 gate; validation artifacts only
- `table_eligible=false`, `exploratory_posthoc=true`
- No follow-up work

## Not done

- Wait / poll / analyze results / resubmit failures
- J / JM / JC / multiseed / test evaluation
