# MIXED_3DOMAIN_LONG no-update gradient-conflict diagnostic

> Twin root: `/orcd/home/002/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN/results/diagnostics/financial_multidataset_long_gradient_conflict`

**No optimizer.step / no parameter update / no checkpoint modification / no test access.**

Job: `19573905` · device `cuda` · runtime_s=1358.9062535762787

## Scientific question

Do InfoNCE and TF-expert losses produce aligned, orthogonal, or conflicting encoder gradients, and does this change between LONG@1500 and LONG@3000?

## Integrity

- Checkpoint SHA preserved: **True**
- All batch reconstructions OK: **True**
- BN/state restored every batch: **True**
- Matched batches/views across checkpoints: **True**
- Max recon rel error: **6.299e-06**

## Across-domain summary

### Checkpoint @1500
- mean cos(InfoNCE, TF-agg) = -0.0298 (std 0.0694; conflict_frac=0.17; class=**approximately_orthogonal**)
- alpha_mean=0.3123; contrast_share=0.054; tf_share=1.000
- mean L_contrast_raw=7.5276

### Checkpoint @3000
- mean cos(InfoNCE, TF-agg) = -0.3023 (std 0.1594; conflict_frac=0.83; class=**conflicting**)
- alpha_mean=0.2090; contrast_share=0.026; tf_share=1.007
- mean L_contrast_raw=7.5224

## Ten interpretation answers

### 1_infonce_vs_tf_agg_conflict
```json
{
  "at_1500": "approximately_orthogonal",
  "at_3000": "conflicting",
  "mean_cos_1500": -0.029780121462155393,
  "mean_cos_3000": -0.30226967058479454,
  "note": "diagnostic evidence from 8 batches/domain; not causal"
}
```

### 2_tf_target_most_conflict_with_infonce
```json
{
  "lowest_mean_cosine_target": "log1p_sender_interarrival",
  "means": {
    "log1p_sender_interarrival": -0.20231914010591046,
    "log1p_sender_past_7d_count": 0.01874583263144282,
    "log1p_amount_vs_sender_past_mean": -0.017503143097278285
  }
}
```

### 3_conflict_stronger_at_3000
```json
{
  "mean_cos_tf_agg_1500": -0.029780121462155393,
  "mean_cos_tf_agg_3000": -0.30226967058479454,
  "stronger_conflict": true
}
```

### 4_contrast_magnitude_shrink_vs_tf
```json
{
  "share_contrast_1500": 0.053787988561232236,
  "share_contrast_3000": 0.026238152397204357,
  "share_tf_1500": 0.9995254130084584,
  "share_tf_3000": 1.0065978924731993,
  "contrast_share_decreased": true
}
```

### 5_lower_alpha_smaller_weighted_contrast_grads
```json
{
  "alpha_1500": 0.31233200430870056,
  "alpha_3000": 0.20899246633052826,
  "share_contrast_1500": 0.053787988561232236,
  "share_contrast_3000": 0.026238152397204357,
  "consistent_with_alpha_drop": true
}
```

### 6_shared_ab_conceal_per_domain_behavior
```json
{
  "cos_tf_agg_by_domain_at_3000": {
    "Small-HI": -0.458717208264516,
    "SAML-D": -0.10938744860749974,
    "Small-LI": -0.33870435488236783
  },
  "spread": 0.3493297596570163,
  "note": "same global \u03b1/\u03b2; domain LossNorm/BN differ"
}
```

### 7_rising_raw_infonce_from_conflict
```json
{
  "L_contrast_raw_1500": 7.527550458908081,
  "L_contrast_raw_3000": 7.522407114505768,
  "plausible_if_conflict_and_alpha_downweight": "diagnostic only; conflict alone insufficient without training dynamics"
}
```

### 8_evidence_mode
```json
{
  "primary": "domination_with_tf",
  "mean_cos_tf_agg": -0.16602489602347498,
  "uncertainty": "n=8 batches/domain; report variation via std/frac tables"
}
```

### 9_strengthens_projection_case
```json
{
  "maybe": true,
  "rationale": "projection could isolate InfoNCE geometry from R198/TF path; not proven here"
}
```

### 10_suggest_masking_reweight_or_alt_objective
```json
{
  "if_conflict": false,
  "if_domination": true,
  "suggestions": [
    "objective reweighting / alpha floor if contrast share collapses",
    "projection-on matched ablation (InfoNCE on H, eval on R198)",
    "stronger attribute masking only if identity-shortcut audit remains primary concern",
    "alternative objective (VICReg) later \u2014 not next"
  ]
}
```

## Confirmation

- No optimizer constructed or stepped
- No encoder/MoE/αβ update
- No checkpoint modification
- No embeddings written; no probe fit
- No test split access

Stop after this diagnostic — no automatic follow-up training.

