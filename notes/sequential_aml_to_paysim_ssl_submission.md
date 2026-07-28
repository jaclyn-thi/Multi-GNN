# Sequential AMLWorld→PaySim SSL scout — submission

> Queued without waiting. Exploratory / post-hoc. Validation only. No automatic follow-ups beyond this DAG.

## Design

- `sequential_domain_adaptive_ssl=true`
- `joint_multidomain_pretraining=false`
- `supervised_encoder_updates=false`
- `exploratory_posthoc=true`, `table_eligible=false`, `test_evaluated=false`

## Arms / controls

| ID | Role |
|----|------|
| A `aml_init` | AML seed-2 weight continuation + PaySim SSL (500 steps) |
| B `random_init` | Matched PaySim-only SSL from scratch |
| C `frozen_aml` | Frozen AML BN (eval) |
| D `bn_only` | Label-free PaySim-train BN recal (eval) |
| E `x_only` | Located multiseed val X-only cell |

## Job table

Filled at submit time in `results/diagnostics/sequential_aml_to_paysim_ssl/submission.json`.

## Expected artifacts

- `results/diagnostics/sequential_aml_to_paysim_ssl/smoke.json`
- `results/diagnostics/sequential_aml_to_paysim_ssl/cells/{aml_init,random_init}_train.json`
- `results/diagnostics/sequential_aml_to_paysim_ssl/eval_summary.json`
- `results/diagnostics/sequential_aml_to_paysim_ssl_scout.json`
- `notes/sequential_aml_to_paysim_ssl_scout.md`
- `saved-models/sequential_aml_to_paysim_ssl/`
- `embeddings/sequential_aml_to_paysim_ssl/`

## Not done

- Poll / analyze results
- Multiseed / automatic resubmit
- Test evaluation

## Submitted job IDs

| Role | Job ID | Dependency |
|------|--------|------------|
| Smoke | **19017925** | — |
| aml_init | **19017926** | afterok:19017925 |
| random_init | **19017927** | afterok:19017925 |
| eval | **19017928** | afterok:19017926:19017927 |
| aggregate | **19017929** | afterok:19017928 |

```text
19017925 (smoke)
  ├─ afterok → 19017926 (aml_init 500-step)
  └─ afterok → 19017927 (random_init 500-step)
                 └─ both afterok → 19017928 (val-only eval)
                                   └─ afterok → 19017929 (CPU aggregate)
```
