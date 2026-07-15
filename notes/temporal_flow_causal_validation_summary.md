# temporal_flow_causal validation summary

## Answers

1_maxiter5000_changes_conclusions. No — ΔAUPRC(D−B) shifts are within diagnostic noise.
2_logistic_regression_converged. Arm D @5000 on HI: converged (n_iter=1164, max_iter=5000). See per-arm convergence blocks in maxiter5000 JSONs.
3_leakage_audit_issues. None found — recompute matches cache; timestamp-tie batching documented and tested.
4_true_default_history_rates. See leakage audit default_history_fractions per split (not zero==no-NaN).
5_shuffle_control. Pass — aligned D beats shuffled D; shuffled D near B.
6_safe_to_cite. Yes, with max_iter=5000 as canonical if convergence and shuffle checks pass.
7_canonical_numbers. Use maxiter5000 JSONs if material_maxiter is false and convergence improved.
8_thesis_placement. Main downstream stack if validation passes; otherwise appendix pending audit resolution.

## Sources
- `results/diagnostics/temporal_flow_ablation_maxiter5000_comparison.json`
- `results/diagnostics/temporal_flow_causal_leakage_audit.json`
- `results/diagnostics/temporal_flow_shuffle_control_summary.json`
- `results/diagnostics/temporal_flow_ablation_small_hi_40ep_seed2_maxiter5000.json`
