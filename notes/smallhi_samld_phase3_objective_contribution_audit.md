# Phase-3 objective-weight and loss-contribution audit

> Twin: `results/diagnostics/smallhi_samld_phase3_objective_contribution_audit.json`
> Source: `direct_r198.combine_direct_h_tfmoe_loss` / `resolve_tfmoe_weights` (`adaptive`)
> Source SHA256: `cb09b54d7ed22ef45781f4311d345ca41609917d25bf572803bb9c9e1f11720b`

**Read-only.** No training, extraction, probes, NPZ loads, test access, or Slurm.

## Verified objective

```
L_total = alpha * L_contrast_norm + (1-alpha) * sum_m beta_m * L_tf_norm_m with alpha=sigmoid(alpha_logit), beta=softmax(beta_logits); w_contrast=alpha; w_tf_m=(1-alpha)*beta_m
```

TF target order (proven):

0. `log1p_sender_interarrival`
1. `log1p_sender_past_7d_count`
2. `log1p_amount_vs_sender_past_mean`

Reconstruction integrity: **PASS** (n=2984, max|err|=2.806e-07, mean|err|=1.964e-08).

Alpha/beta are **global/shared** in MIXED_1TO1; LossNormState is **per-domain**.
Calibration: first 5 observations/domain; α/β frozen through 0-indexed `step` 9, unfrozen at `step` 10 (logged `global_optimizer_step` 1..10 frozen, 11+ unfrozen).

## Table 1 — Final learned weights

| Arm | Global step | HI exp | SAML exp | alpha | beta0 | beta1 | beta2 | w_c | w_tf0 | w_tf1 | w_tf2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SMALL_HI_ONLY | 1000 | 1000 | 0 | 0.4465 | 0.3638 | 0.4851 | 0.1511 | 0.4465 | 0.2014 | 0.2685 | 0.0837 |
| SAMLD_ONLY | 1000 | 0 | 1000 | 0.4478 | 0.4974 | 0.1530 | 0.3497 | 0.4478 | 0.2746 | 0.0845 | 0.1931 |
| MIXED_1TO1 | 1000 | 500 | 500 | 0.4428 | 0.4440 | 0.3787 | 0.1773 | 0.4428 | 0.2474 | 0.2110 | 0.0988 |

## Table 2 — Last-20 realized contributions

C columns use **mean(C_i)**; shares reported both as mean(C)/mean(total) and mean of per-step shares.

| Arm | Domain | mean C_contrast | mean C_tf0 | mean C_tf1 | mean C_tf2 | contrast share (meanC/meanTot) | TF share (meanC/meanTot) | Dominant TF | n |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| SMALL_HI_ONLY | Small-HI | 0.4077 | 0.0463 | 0.0344 | 0.0568 | 0.748 | 0.252 | `log1p_amount_vs_sender_past_mean` | 20 |
| SAMLD_ONLY | SAML-D | 0.3980 | 0.0691 | 0.0360 | 0.0552 | 0.713 | 0.287 | `log1p_sender_interarrival` | 20 |
| MIXED_1TO1 | Small-HI | 0.4111 | 0.0683 | 0.0307 | 0.0720 | 0.706 | 0.294 | `log1p_amount_vs_sender_past_mean` | 20 |
| MIXED_1TO1 | SAML-D | 0.4016 | 0.0758 | 0.0912 | 0.0337 | 0.667 | 0.333 | `log1p_sender_past_7d_count` | 20 |

## Interpretation answers

1. Final weights: see Table 1 (all α≈0.44–0.45; β allocations differ by arm).
2. Largest effective TF weight: SMALL_HI_ONLY→`log1p_sender_past_7d_count`, SAMLD_ONLY→`log1p_sender_interarrival`, MIXED_1TO1→`log1p_sender_interarrival`.
3. Largest realized TF contribution (last-20 mean-of-shares): SMALL_HI_ONLY|Small-HI→`log1p_amount_vs_sender_past_mean`, SAMLD_ONLY|SAML-D→`log1p_sender_interarrival`, MIXED_1TO1|Small-HI→`log1p_amount_vs_sender_past_mean`, MIXED_1TO1|SAML-D→`log1p_sender_past_7d_count`.
4. Weight vs contribution TF ranking agree: `{'SMALL_HI_ONLY|Small-HI': False, 'SAMLD_ONLY|SAML-D': True, 'MIXED_1TO1|Small-HI': False, 'MIXED_1TO1|SAML-D': False}`.
5. MIXED domain realized shares differ despite shared α/β: `True` (HI TF share 0.294 vs SAML 0.333).
6. Mixed more TF mass than specialists: HI 0.294 vs 0.252; SAML 0.333 vs 0.287; both=True.
7. Contrast remains nontrivial at step 1000: w_contrast≈0.44–0.45 (≥0.25) in all arms; realized contrast share still ~0.4–0.8 depending on domain view.
8. Trajectories: `{'SMALL_HI_ONLY': 'still_moving', 'SAMLD_ONLY': 'still_moving', 'MIXED_1TO1': 'still_moving'}` (last-100 Δα / Δβ L1 reported in JSON).
9. Component-specific gradient attribution is **unavailable** from these runs (only total encoder / MoE / joint αβ grad norms).
10. **Weights and objective shares alone cannot establish causal downstream importance.**
11. Needed: matched multi-domain InfoNCE-only vs EXPERT_ONLY vs adaptive InfoNCE+TF (not launched here).

## Secondary caveat

Secondary exposure-matched comparison (mixed@500/domain vs single@500) is NOT perfectly LR-phase matched for domain exposure.

## Gradient evidence

Component-specific gradient attribution is unavailable from these runs.

Available fields: ['alpha_grad_norm', 'contrast_grad_contribution', 'encoder_grad_norm', 'moe_grad_norm']

## Confirmations

- no model training
- no encoder forward/backward
- no extraction / probe / NPZ embedding load
- no test access
- no Slurm jobs
- training/model/eval code unchanged; checkpoints unmodified

