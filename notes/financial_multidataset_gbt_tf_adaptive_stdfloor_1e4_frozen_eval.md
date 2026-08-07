# GBT+TF adaptive stdfloor frozen R198 validation eval

> Twin: `results/diagnostics/financial_multidataset_gbt_tf_adaptive_stdfloor_1e4_frozen_eval.json`
> Objective: `edge_aligned_gbt_tf_adaptive_stdfloor_1e4` (gbt_std_floor=0.0001)

**ok=True** — validation-only; direct encoder R198; no test; no retrain.

Baseline comparability: `COMPARABLE_REUSE_AUTHORIZED`

## Protocol locks

- contract: `financial_multidataset_shared_core_v1`
- probe: PaperStyleMLP 20ep lr=0.001 bs=8192 seed=2
- extract: full-subgraph R198 train/val; projection bypassed; no raw/TF concat
- checkpoints: `results/checkpoints/financial_multidataset_gbt_tf_adaptive_stdfloor_1e4_seed2` steps 1500 + 3000 only

## Main table (GBT+TF cells)

| Arm | Step | Target | AUPRC | AUROC | F1@0.5 | P | R | F1@val-thr | Final val BCE |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| GBT_TF_ADAPTIVE_1500 | 1500 | Small-HI | 0.2394 | 0.9584 | 0.1937 | 0.7117 | 0.1121 | 0.3166 | 0.005813 |
| GBT_TF_ADAPTIVE_1500 | 1500 | SAML-D | 0.8837 | 0.9958 | 0.8651 | 0.9844 | 0.7716 | 0.8733 | 0.001910 |
| GBT_TF_ADAPTIVE_1500 | 1500 | Small-LI | 0.0551 | 0.9327 | 0.0662 | 0.3684 | 0.0364 | 0.1121 | 0.004297 |
| GBT_TF_ADAPTIVE_3000 | 3000 | Small-HI | 0.2339 | 0.9544 | 0.2028 | 0.6596 | 0.1198 | 0.3162 | 0.005529 |
| GBT_TF_ADAPTIVE_3000 | 3000 | SAML-D | 0.8783 | 0.9956 | 0.8595 | 0.9777 | 0.7669 | 0.8638 | 0.002212 |
| GBT_TF_ADAPTIVE_3000 | 3000 | Small-LI | 0.0470 | 0.9384 | 0.0779 | 0.2713 | 0.0455 | 0.1131 | 0.004354 |

## GBT+TF @3000 vs @1500

- **Small-HI**: ΔAUPRC=-0.0055 (0.2394 → 0.2339)
- **SAML-D**: ΔAUPRC=-0.0054 (0.8837 → 0.8783)
- **Small-LI**: ΔAUPRC=-0.0081 (0.0551 → 0.0470)

## Adaptive α/β (checkpoint)

- **GBT_TF_ADAPTIVE_1500**: α=0.8153 β=[0.3469260036945343, 0.5146719813346863, 0.13840197026729584] w_gbt=0.8153 w_tf=[0.06407225131988525, 0.09505252540111542, 0.025560857728123665]
- **GBT_TF_ADAPTIVE_3000**: α=0.8751 β=[0.27781346440315247, 0.6213424801826477, 0.10084406286478043] w_gbt=0.8751 w_tf=[0.03469742462038994, 0.07760237157344818, 0.012594887055456638]

## Scientific answers (Q1–Q10)

### Q1_vs_standalone_GBT
- Q: Does GBT+TF improve over standalone GBT?
- Verdict: `{"at_1500": true, "at_3000": false, "any_target_improves_1500": true, "any_target_improves_3000": true}`

### Q2_vs_EXPERT_ONLY
- Q: Does it match or exceed EXPERT_ONLY?
- Verdict: `{"matches_or_exceeds_all": false, "exceeds_any": false}`

### Q3_vs_ADAPTIVE_LONG
- Q: Does it improve over identity-InfoNCE adaptive LONG?
- Verdict: `{"improves_all_at_3000": false, "improves_any_at_3000": false}`

### Q4_vs_PROJECTION_ON
- Q: Does it beat projection-on adaptive?
- Verdict: `{"beats_all": false, "beats_any": false}`

### Q5_complementarity
- Q: Is there evidence of objective complementarity?
- Verdict: `false`
- Note: Metric-level hint only: positive Δ vs both standalone GBT and EXPERT_ONLY. Do not infer causal gradient compatibility from weights or metrics alone.

### Q6_alpha_beta_shares
- Q: What alpha/beta and realized shares were learned?

### Q7_gbt_deprioritized
- Q: Was GBT deprioritized like InfoNCE?
- Verdict: `{"gbt_deprioritized_like_infonce": false, "gbt_upweighted_from_init": true, "interpretation": "\u03b1\u21920 would resemble InfoNCE deprioritization; observed \u03b1 stays high \u2192 GBT retained."}`
- Note: Weight trajectory ≠ gradient compatibility diagnosis.

### Q8_stronger_checkpoint
- Q: Which checkpoint is stronger per target?

### Q9_samld_peaks_early
- Q: Does SAML-D peak earlier again?
- Verdict: `true`

### Q10_gradient_diagnostic_warranted
- Q: Is a later no-update GBT-vs-TF gradient diagnostic warranted?
- Verdict: `true`


## Deltas vs reused baselines (AUPRC)

- **GBT_TF_ADAPTIVE_1500 / Small-HI vs GBT_STDFLOOR@1500** [matched_step_1500]: Δ=+0.0354 (retention 1.174)
- **GBT_TF_ADAPTIVE_1500 / Small-HI vs GBT_STDFLOOR@3000** [matched_step_3000]: Δ=+0.0323 (retention 1.156)
- **GBT_TF_ADAPTIVE_1500 / Small-HI vs INFONCE_ONLY@3000** [matched_step_3000]: Δ=+0.2293 (retention 23.787)
- **GBT_TF_ADAPTIVE_1500 / Small-HI vs EXPERT_ONLY@3000** [matched_step_3000]: Δ=-0.1781 (retention 0.573)
- **GBT_TF_ADAPTIVE_1500 / Small-HI vs PROJECTION_ON_ADAPTIVE@3000** [matched_step_3000]: Δ=-0.1522 (retention 0.611)
- **GBT_TF_ADAPTIVE_1500 / Small-HI vs ADAPTIVE_LONG@3000** [matched_step_3000]: Δ=-0.1486 (retention 0.617)
- **GBT_TF_ADAPTIVE_1500 / Small-HI vs ADAPTIVE_LONG@1500** [matched_step_1500_historical_adaptive_long]: Δ=-0.0523 (retention 0.821)
- **GBT_TF_ADAPTIVE_1500 / SAML-D vs GBT_STDFLOOR@1500** [matched_step_1500]: Δ=+0.0119 (retention 1.014)
- **GBT_TF_ADAPTIVE_1500 / SAML-D vs GBT_STDFLOOR@3000** [matched_step_3000]: Δ=+0.0687 (retention 1.084)
- **GBT_TF_ADAPTIVE_1500 / SAML-D vs INFONCE_ONLY@3000** [matched_step_3000]: Δ=+0.3030 (retention 1.522)
- **GBT_TF_ADAPTIVE_1500 / SAML-D vs EXPERT_ONLY@3000** [matched_step_3000]: Δ=-0.0664 (retention 0.930)
- **GBT_TF_ADAPTIVE_1500 / SAML-D vs PROJECTION_ON_ADAPTIVE@3000** [matched_step_3000]: Δ=-0.0613 (retention 0.935)
- **GBT_TF_ADAPTIVE_1500 / SAML-D vs ADAPTIVE_LONG@3000** [matched_step_3000]: Δ=-0.0310 (retention 0.966)
- **GBT_TF_ADAPTIVE_1500 / SAML-D vs ADAPTIVE_LONG@1500** [matched_step_1500_historical_adaptive_long]: Δ=-0.0481 (retention 0.948)
- **GBT_TF_ADAPTIVE_1500 / Small-LI vs GBT_STDFLOOR@1500** [matched_step_1500]: Δ=+0.0107 (retention 1.241)
- **GBT_TF_ADAPTIVE_1500 / Small-LI vs GBT_STDFLOOR@3000** [matched_step_3000]: Δ=+0.0060 (retention 1.123)
- **GBT_TF_ADAPTIVE_1500 / Small-LI vs INFONCE_ONLY@3000** [matched_step_3000]: Δ=+0.0463 (retention 6.260)
- **GBT_TF_ADAPTIVE_1500 / Small-LI vs EXPERT_ONLY@3000** [matched_step_3000]: Δ=-0.0567 (retention 0.493)
- **GBT_TF_ADAPTIVE_1500 / Small-LI vs PROJECTION_ON_ADAPTIVE@3000** [matched_step_3000]: Δ=-0.0655 (retention 0.457)
- **GBT_TF_ADAPTIVE_1500 / Small-LI vs ADAPTIVE_LONG@3000** [matched_step_3000]: Δ=-0.0506 (retention 0.521)
- **GBT_TF_ADAPTIVE_1500 / Small-LI vs ADAPTIVE_LONG@1500** [matched_step_1500_historical_adaptive_long]: Δ=-0.0375 (retention 0.595)
- **GBT_TF_ADAPTIVE_3000 / Small-HI vs GBT_STDFLOOR@1500** [matched_step_1500]: Δ=+0.0299 (retention 1.147)
- **GBT_TF_ADAPTIVE_3000 / Small-HI vs GBT_STDFLOOR@3000** [matched_step_3000]: Δ=+0.0268 (retention 1.129)
- **GBT_TF_ADAPTIVE_3000 / Small-HI vs INFONCE_ONLY@3000** [matched_step_3000]: Δ=+0.2239 (retention 23.241)
- **GBT_TF_ADAPTIVE_3000 / Small-HI vs EXPERT_ONLY@3000** [matched_step_3000]: Δ=-0.1836 (retention 0.560)
- **GBT_TF_ADAPTIVE_3000 / Small-HI vs PROJECTION_ON_ADAPTIVE@3000** [matched_step_3000]: Δ=-0.1577 (retention 0.597)
- **GBT_TF_ADAPTIVE_3000 / Small-HI vs ADAPTIVE_LONG@3000** [matched_step_3000]: Δ=-0.1541 (retention 0.603)
- **GBT_TF_ADAPTIVE_3000 / Small-HI vs ADAPTIVE_LONG@1500** [matched_step_1500_historical_adaptive_long]: Δ=-0.0578 (retention 0.802)
- **GBT_TF_ADAPTIVE_3000 / SAML-D vs GBT_STDFLOOR@1500** [matched_step_1500]: Δ=+0.0065 (retention 1.007)
- **GBT_TF_ADAPTIVE_3000 / SAML-D vs GBT_STDFLOOR@3000** [matched_step_3000]: Δ=+0.0632 (retention 1.078)
- **GBT_TF_ADAPTIVE_3000 / SAML-D vs INFONCE_ONLY@3000** [matched_step_3000]: Δ=+0.2976 (retention 1.512)
- **GBT_TF_ADAPTIVE_3000 / SAML-D vs EXPERT_ONLY@3000** [matched_step_3000]: Δ=-0.0718 (retention 0.924)
- **GBT_TF_ADAPTIVE_3000 / SAML-D vs PROJECTION_ON_ADAPTIVE@3000** [matched_step_3000]: Δ=-0.0668 (retention 0.929)
- **GBT_TF_ADAPTIVE_3000 / SAML-D vs ADAPTIVE_LONG@3000** [matched_step_3000]: Δ=-0.0364 (retention 0.960)
- **GBT_TF_ADAPTIVE_3000 / SAML-D vs ADAPTIVE_LONG@1500** [matched_step_1500_historical_adaptive_long]: Δ=-0.0536 (retention 0.942)
- **GBT_TF_ADAPTIVE_3000 / Small-LI vs GBT_STDFLOOR@1500** [matched_step_1500]: Δ=+0.0026 (retention 1.059)
- **GBT_TF_ADAPTIVE_3000 / Small-LI vs GBT_STDFLOOR@3000** [matched_step_3000]: Δ=-0.0020 (retention 0.958)
- **GBT_TF_ADAPTIVE_3000 / Small-LI vs INFONCE_ONLY@3000** [matched_step_3000]: Δ=+0.0382 (retention 5.343)
- **GBT_TF_ADAPTIVE_3000 / Small-LI vs EXPERT_ONLY@3000** [matched_step_3000]: Δ=-0.0648 (retention 0.421)
- **GBT_TF_ADAPTIVE_3000 / Small-LI vs PROJECTION_ON_ADAPTIVE@3000** [matched_step_3000]: Δ=-0.0736 (retention 0.390)
- **GBT_TF_ADAPTIVE_3000 / Small-LI vs ADAPTIVE_LONG@3000** [matched_step_3000]: Δ=-0.0586 (retention 0.445)
- **GBT_TF_ADAPTIVE_3000 / Small-LI vs ADAPTIVE_LONG@1500** [matched_step_1500_historical_adaptive_long]: Δ=-0.0456 (retention 0.508)

Confirmation: no test data loaded/scored; baselines not re-extracted; no gradient diagnostic submitted; recovery checkpoints never evaluated.

