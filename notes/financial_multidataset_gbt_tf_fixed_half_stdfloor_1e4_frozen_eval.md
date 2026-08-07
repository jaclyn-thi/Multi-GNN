# GBT+TF fixed-half @1500 + EXPERT_ONLY@1500 frozen R198 validation eval

> Twin: `results/diagnostics/financial_multidataset_gbt_tf_fixed_half_stdfloor_1e4_frozen_eval.json`
> Objective: `edge_aligned_gbt_tf_fixed_half_stdfloor_1e4` (gbt_std_floor=0.0001)

**ok=True** — validation-only; direct encoder R198; no test; no retrain.

Matched-arm comparability: `MATCHED_ARM_COMPARABLE`
Baseline comparability: `COMPARABLE_REUSE_AUTHORIZED`

**Alpha interpretation:** beta_unfrozen_after_step_15: inherited training field alpha_unfrozen_at=15 means beta became eligible for updates after step 15; alpha remained fixed at 0.5 and never unfroze. Do not rewrite historical training artifacts.

## Protocol locks

- contract: `financial_multidataset_shared_core_v1`
- probe: PaperStyleMLP 20ep lr=0.001 bs=8192 seed=2
- extract: full-subgraph R198 train/val; projection bypassed; no raw/TF concat
- new arms: `GBT_TF_FIXED_HALF_1500` (.pt) + `EXPERT_ONLY_1500` (.tar) @1500
- F1@val-thr is optimistic (same-val threshold); not a test estimate

## Primary matched-step-1500 table

| Arm | Step | Target | AUPRC | AUROC | F1@0.5 | P | R | F1@val-thr* | Final val BCE |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| GBT_STDFLOOR@1500 | 1500 | Small-HI | 0.2040 | 0.9557 | 0.2480 | 0.4806 | 0.1671 | 0.2982 | 0.005707 |
| GBT_STDFLOOR@1500 | 1500 | SAML-D | 0.8718 | 0.9798 | 0.8562 | 0.9666 | 0.7685 | 0.8608 | 0.003765 |
| GBT_STDFLOOR@1500 | 1500 | Small-LI | 0.0444 | 0.9457 | 0.0698 | 0.2627 | 0.0403 | 0.1102 | 0.004448 |
| GBT_TF_ADAPTIVE@1500 | 1500 | Small-HI | 0.2394 | 0.9584 | 0.1937 | 0.7117 | 0.1121 | 0.3166 | 0.005813 |
| GBT_TF_ADAPTIVE@1500 | 1500 | SAML-D | 0.8837 | 0.9958 | 0.8651 | 0.9844 | 0.7716 | 0.8733 | 0.001910 |
| GBT_TF_ADAPTIVE@1500 | 1500 | Small-LI | 0.0551 | 0.9327 | 0.0662 | 0.3684 | 0.0364 | 0.1121 | 0.004297 |
| GBT_TF_FIXED_HALF_1500 | 1500 | Small-HI | 0.2601 | 0.9656 | 0.3114 | 0.5252 | 0.2213 | 0.3471 | 0.005251 |
| GBT_TF_FIXED_HALF_1500 | 1500 | SAML-D | 0.8996 | 0.9960 | 0.8671 | 0.9410 | 0.8040 | 0.8695 | 0.001715 |
| GBT_TF_FIXED_HALF_1500 | 1500 | Small-LI | 0.0392 | 0.9459 | 0.0253 | 0.5000 | 0.0130 | 0.0946 | 0.004149 |
| EXPERT_ONLY_1500 | 1500 | Small-HI | 0.3509 | 0.9755 | 0.4194 | 0.5034 | 0.3594 | 0.4281 | 0.004696 |
| EXPERT_ONLY_1500 | 1500 | SAML-D | 0.9275 | 0.9986 | 0.8877 | 0.9604 | 0.8253 | 0.8905 | 0.001163 |
| EXPERT_ONLY_1500 | 1500 | Small-LI | 0.0801 | 0.9596 | 0.0887 | 0.4368 | 0.0494 | 0.1562 | 0.003936 |
| ADAPTIVE_LONG@1500 | 1500 | Small-HI | 0.2917 | 0.9717 | 0.3259 | 0.5774 | 0.2271 | 0.3906 | 0.005075 |
| ADAPTIVE_LONG@1500 | 1500 | SAML-D | 0.9319 | 0.9980 | 0.8994 | 0.9734 | 0.8359 | 0.8999 | 0.001220 |
| ADAPTIVE_LONG@1500 | 1500 | Small-LI | 0.0926 | 0.9634 | 0.0985 | 0.5060 | 0.0545 | 0.1596 | 0.003620 |

*F1@val-thr optimistic diagnostic.


## Secondary unmatched-checkpoint-3000 table (context only)

| Arm | Step | Target | AUPRC | F1@0.5 | match_kind |
|---|---:|---|---:|---:|---|
| GBT_STDFLOOR@3000 | 3000 | Small-HI | 0.2071 | 0.2353 | unmatched_checkpoint_3000 |
| GBT_STDFLOOR@3000 | 3000 | SAML-D | 0.8150 | 0.8228 | unmatched_checkpoint_3000 |
| GBT_STDFLOOR@3000 | 3000 | Small-LI | 0.0491 | 0.0963 | unmatched_checkpoint_3000 |
| INFONCE_ONLY@3000 | 3000 | Small-HI | 0.0101 | 0.0333 | unmatched_checkpoint_3000 |
| INFONCE_ONLY@3000 | 3000 | SAML-D | 0.5807 | 0.4288 | unmatched_checkpoint_3000 |
| INFONCE_ONLY@3000 | 3000 | Small-LI | 0.0088 | 0.0000 | unmatched_checkpoint_3000 |
| EXPERT_ONLY@3000 | 3000 | Small-HI | 0.4175 | 0.4684 | unmatched_checkpoint_3000 |
| EXPERT_ONLY@3000 | 3000 | SAML-D | 0.9501 | 0.9073 | unmatched_checkpoint_3000 |
| EXPERT_ONLY@3000 | 3000 | Small-LI | 0.1118 | 0.0852 | unmatched_checkpoint_3000 |
| PROJECTION_ON_ADAPTIVE@3000 | 3000 | Small-HI | 0.3917 | 0.4316 | unmatched_checkpoint_3000 |
| PROJECTION_ON_ADAPTIVE@3000 | 3000 | SAML-D | 0.9450 | 0.9157 | unmatched_checkpoint_3000 |
| PROJECTION_ON_ADAPTIVE@3000 | 3000 | Small-LI | 0.1206 | 0.1250 | unmatched_checkpoint_3000 |
| ADAPTIVE_LONG@3000 | 3000 | Small-HI | 0.3881 | 0.4310 | unmatched_checkpoint_3000 |
| ADAPTIVE_LONG@3000 | 3000 | SAML-D | 0.9147 | 0.8727 | unmatched_checkpoint_3000 |
| ADAPTIVE_LONG@3000 | 3000 | Small-LI | 0.1056 | 0.1365 | unmatched_checkpoint_3000 |

## Fixed-half β / weighted contributions @1500

- Nominal (effective_weights fixed_half): α=0.5000 β=[0.30296608805656433, 0.5810646414756775, 0.11596924066543579] w_gbt=0.5000 w_tf=[0.15148304402828217, 0.29053232073783875, 0.057984620332717896]
- EXPERT_ONLY effective_weights: β=[0.43916046619415283, 0.45549148321151733, 0.10534801334142685] w_contrast=0.0000 w_tf=[0.43916046619415283, 0.45549148321151733, 0.10534801334142685]
- Realized steps.jsonl@1499: w_gbt=0.5000 w_tf=[0.15158845484256744, 0.2903648912906647, 0.058046650141477585] weighted_gbt=0.06894055008888245 weighted_tf=[0.05493203550577164, 0.12853460013866425, 0.02404276467859745]
- Aggregate milestone@1500: {"w_gbt": 0.5, "sum_w_tf": 0.5000000074505806, "beta": [0.3030810058116913, 0.580886721611023, 0.11603228747844696], "L_total": 0.19693681597709656, "weighted_gbt": 0.05057089775800705, "weighted_tf": [0.05344915762543678, 0.050291065126657486, 0.04262568801641464], "encoder_grad_norm": 0.4288239586099044, "moe_grad_norm": 0.31836977192118904, "view1_repr_grad_norm": 0.012485767714679241, "view2_repr_grad_norm": 0.009226636029779911, "effective_rank": 12.300110816955566, "alpha_beta_frozen": false}

## Scientific answers (Q1–Q12)

### Q1_vs_adaptive_gbt_tf
- Q: Does fixed-half GBT+TF beat adaptive GBT+TF at matched step 1500?
- Verdict: `{"beats_all": false, "beats_any": true, "beats_majority": true}`
- Note: Seed-2 validation-only; do not treat as causal proof of objective design.

### Q2_vs_standalone_gbt
- Q: Does fixed-half beat standalone GBT at step 1500?
- Verdict: `{"beats_all": false, "beats_any": true}`

### Q3_vs_expert_only_1500
- Q: Does fixed-half match or beat EXPERT_ONLY at step 1500?
- Verdict: `{"matches_or_exceeds_all": false, "exceeds_any": false}`

### Q4_vs_adaptive_long_1500
- Q: Does fixed-half match or beat adaptive InfoNCE+TF LONG at step 1500?
- Verdict: `{"matches_or_exceeds_all": false, "exceeds_any": false}`

### Q5_expert_gap_closed
- Q: How much of the expert-only gap is closed by protecting 50% TF mass?
- Note: Fraction (AUPRC_fh − AUPRC_GBT) / (AUPRC_expert − AUPRC_GBT); not a causal attribution to the 50% TF mass alone.

### Q6_all_vs_selected_targets
- Q: Does fixed weighting improve all three targets or only selected domains?
- Verdict: `"selected_domains"`
- Note: Do not compare absolute AUPRC across datasets as equal difficulty.

### Q7_beta_allocation
- Q: What beta allocation is learned within the fixed TF half?

### Q8_nominal_and_realized_weights
- Q: What are the nominal and realized weighted contributions at step 1500?
- Note: Weights and loss shares are not gradient compatibility measurements.

### Q9_easier_objective_hypothesis
- Q: Do the results support the hypothesis that learned alpha favored the easier GBT loss rather than the downstream-optimal mixture?
- Verdict: `"supports_but_does_not_prove"`
- Note: Fixed-half beating adaptive supports—but does not by itself prove—the easier-objective hypothesis.

### Q10_retain_gbt_in_final_model
- Q: Does fixed-half provide sufficient evidence to retain GBT in the final model?
- Verdict: `"inconclusive_for_retention_improves_vs_adaptive_but_expert_gap_remains"`
- Note: Careful retention language only; seed-2 validation on pretraining domains; do not infer unseen-domain transfer.

### Q11_strongest_matched_1500_encoder
- Q: Which single common step-1500 encoder is strongest across the three validation targets?
- Note: Mean AUPRC across heterogeneous targets is a convenience summary only; do not treat datasets as equal difficulty.

### Q12_continue_gbt_experiments
- Q: Is there any scientifically compelling reason to continue GBT experiments?
- Verdict: `false`
- Rationale: This is the final authorized GBT experiment. Residual uncertainty (seed-2, validation-only, no gradient diagnostic) does not authorize further GBT training, weight sweeps, longer runs, or new seeds under the current stop condition.


## Final recommendation on GBT

- Verdict: `inconclusive_for_retention_improves_vs_adaptive_but_expert_gap_remains`
- Note: Careful retention language only; seed-2 validation on pretraining domains; do not infer unseen-domain transfer.
- Q12: no further GBT experiments authorized under this stop condition.

Confirmation: no test data loaded/scored; baselines not re-extracted; recovery checkpoints never evaluated; historical training artifacts unchanged.

