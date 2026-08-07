# Phase-4C frozen-eval v2 full-coverage preflight

- Verdict: `FULL_FROZEN_EVAL_V2_AUTHORIZED`
- Job: `19778019`
- Checkpoint: `FOUR_DOMAIN_PROJECTION_INFONCE_TF_ADAPTIVE_SHORT` step `4000`
- Eval source manifest v2 SHA: `b125be76774ba66d198cd9ab6139977ac4ab49438b5f2310f5e0294a99ca1d84`
- Training manifest SHA: `7dcfaa52d38abd6e929633c028b2e2a21743385d26c6b2cdbd34e87b2f42d3aa`
- Reusable cells: four_domain_projection_infonce_tf_adaptive_short__step4000__small_hi, four_domain_projection_infonce_tf_adaptive_short__step4000__saml_d, four_domain_projection_infonce_tf_adaptive_short__step4000__small_li, four_domain_projection_infonce_tf_adaptive_short__step4000__paysim
- Seed-complete exact EdgeID gate exercised on all eight train/val cells.
- No full DAG / probes / historical re-extraction / training / test eval submitted.

## Historical cohort warning

Historical Phase-4B/GBT/ladder metrics used incomplete same-loader cohorts (esp. SAML-D ~87%). Direct four- vs three-domain comparisons are not fully cohort-matched until historical checkpoints are re-extracted with seed-complete.
