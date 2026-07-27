# Schema-level categorical masking scout (seed-2)

> Validation-only gate. **Not** a final transfer result. Test never evaluated.

- Selected arm: **none passed**
- Control AMLWorld post-128 H+X+TF val AUPRC: 0.4998
- Control PaySim type_only val AUPRC: 0.0141

## Gate results

| Arm | Pass | PS AUPRC | ΔAUPRC | ΔF1 | Beat rand | AML AUPRC | AML regress |
|-----|------|---------:|-------:|----:|-----------|----------:|------------:|
| `schema_mask_p025` | False | 0.0044 | -0.0096 | -0.0251 | False | 0.4479 | +0.0519 |
| `schema_mask_p050` | False | 0.0065 | -0.0076 | -0.0236 | False | 0.4836 | +0.0161 |

## Predeclared thresholds

{
  "paysim_type_only_val_auprc_improve_abs": 0.003,
  "paysim_type_only_val_f1_improve_abs": 0.01,
  "must_beat_matched_random_val_auprc": true,
  "amlworld_hxxtf_val_auprc_max_regression_abs": 0.02
}

