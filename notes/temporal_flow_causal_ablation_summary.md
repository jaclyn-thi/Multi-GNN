# Temporal flow causal ablation — consolidated summary

## Answers

1_temporal_flow_vs_pre3h_alone. HI ΔAUPRC(C−A)=+0.1776 (modest improvement); LI mean ΔAUPRC(C−A) from per-seed payloads — see multiseed JSON.
2_temporal_flow_vs_pre3h_plus_raw_primary. HI ΔAUPRC(D−B)=+0.1800 (modest improvement); LI mean ΔAUPRC(D−B)=+0.0669 (modest improvement); D beats B on 3/3 LI seeds.
3_holds_on_small_hi. modest improvement
4_li_direction_consistent. yes
5_feature_level. See per-run feature_diagnostics (coefficients, correlation, univariate AUPRC on train).
6_gain_type. Compare ΔAUPRC (ranking), ΔF1 (thresholded), and Δlift@100 (alert budget) in per-run JSONs.
7_redundancy_with_raw. Inspect temporal_flow correlation matrix and whether D−B ≪ C−A (suggests raw overlap).
8_thesis_placement. Include in main method only if D beats B on HI and ≥2/3 LI seeds without F1 regression >0.01; otherwise appendix or future work.

## Sources

- `results/diagnostics/temporal_flow_ablation_small_hi_40ep_seed2.json`
- `results/diagnostics/temporal_flow_ablation_small_li_multiseed.json`
