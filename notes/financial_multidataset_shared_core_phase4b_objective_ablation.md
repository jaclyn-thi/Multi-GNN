# Phase-4B objective ablation frozen R198 validation eval

> Twin: `results/diagnostics/financial_multidataset_shared_core_phase4b_objective_ablation_frozen_eval.json`
> Reference LONG@3000 reuse: `results/diagnostics/financial_multidataset_shared_core_phase4b_mixed_long_frozen_eval`

**ok=True** — validation-only; no encoder retrain; no test.

LONG comparability: `COMPARABLE_REUSE_AUTHORIZED`

## Protocol locks

- contract: `financial_multidataset_shared_core_v1`
- probe: PaperStyleMLP 20ep lr=0.001 bs=8192 seed=2
- extract: full-subgraph R198 train/val; projection bypassed
- projection architecture (training arm C only): ContrastiveProjectionHead: Linear(198→128,bias=False)→BatchNorm1d(128)→ReLU(inplace)→Linear(128→128); InfoNCE on H; TF/eval on R198

## Main table

| Arm | Objective | Projection | Target | AUPRC | AUROC | F1@0.5 | P | R | F1@val-thr | Final val BCE |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| EXPERT_ONLY | TF experts only (w_contrast=0) | False | Small-HI | 0.4175 | 0.9760 | 0.4684 | 0.7209 | 0.3469 | 0.4693 | 0.004285 |
| ADAPTIVE_LONG_3000 | adaptive InfoNCE+TF (reference; reused) | False | Small-HI | 0.3881 | 0.9787 | 0.4310 | 0.6485 | 0.3227 | 0.4507 | 0.004383 |
| EXPERT_ONLY | TF experts only (w_contrast=0) | False | SAML-D | 0.9501 | 0.9998 | 0.9073 | 0.9761 | 0.8476 | 0.9122 | 0.000823 |
| ADAPTIVE_LONG_3000 | adaptive InfoNCE+TF (reference; reused) | False | SAML-D | 0.9147 | 0.9983 | 0.8727 | 0.9432 | 0.8120 | 0.8746 | 0.002001 |
| EXPERT_ONLY | TF experts only (w_contrast=0) | False | Small-LI | 0.1118 | 0.9628 | 0.0852 | 0.6731 | 0.0455 | 0.1850 | 0.004020 |
| ADAPTIVE_LONG_3000 | adaptive InfoNCE+TF (reference; reused) | False | Small-LI | 0.1056 | 0.9613 | 0.1365 | 0.4919 | 0.0792 | 0.1798 | 0.003667 |
| INFONCE_ONLY | InfoNCE only (no TF encoder grad) | False | Small-HI | 0.0101 | 0.8428 | 0.0333 | 0.0198 | 0.1053 | 0.0380 | 1.123908 |
| INFONCE_ONLY | InfoNCE only (no TF encoder grad) | False | SAML-D | 0.5807 | 0.9899 | 0.4288 | 0.2959 | 0.7785 | 0.6476 | 0.009457 |
| INFONCE_ONLY | InfoNCE only (no TF encoder grad) | False | Small-LI | 0.0088 | 0.7380 | 0.0000 | 0.0000 | 0.0000 | 0.0375 | 0.017470 |
| PROJECTION_ON_ADAPTIVE | adaptive InfoNCE+TF | True | Small-HI | 0.3917 | 0.9770 | 0.4316 | 0.6900 | 0.3140 | 0.4459 | 0.004683 |
| PROJECTION_ON_ADAPTIVE | adaptive InfoNCE+TF | True | SAML-D | 0.9450 | 0.9993 | 0.9157 | 0.9794 | 0.8598 | 0.9180 | 0.000924 |
| PROJECTION_ON_ADAPTIVE | adaptive InfoNCE+TF | True | Small-LI | 0.1206 | 0.9608 | 0.1250 | 0.6795 | 0.0688 | 0.1687 | 0.003734 |

## Deltas vs ADAPTIVE LONG@3000

- **EXPERT_ONLY / Small-HI**: AUPRC Δ=+0.0295 (retention 1.076)
- **EXPERT_ONLY / SAML-D**: AUPRC Δ=+0.0354 (retention 1.039)
- **EXPERT_ONLY / Small-LI**: AUPRC Δ=+0.0062 (retention 1.058)
- **INFONCE_ONLY / Small-HI**: AUPRC Δ=-0.3780 (retention 0.026)
- **INFONCE_ONLY / SAML-D**: AUPRC Δ=-0.3340 (retention 0.635)
- **INFONCE_ONLY / Small-LI**: AUPRC Δ=-0.0968 (retention 0.083)
- **PROJECTION_ON_ADAPTIVE / Small-HI**: AUPRC Δ=+0.0036 (retention 1.009)
- **PROJECTION_ON_ADAPTIVE / SAML-D**: AUPRC Δ=+0.0304 (retention 1.033)
- **PROJECTION_ON_ADAPTIVE / Small-LI**: AUPRC Δ=+0.0150 (retention 1.142)

## Ten interpretation answers

### 1_expert_vs_adaptive_per_domain

```json
{
  "Small-HI": {
    "expert_auprc": 0.41753007638036643,
    "adaptive_auprc": 0.3880503294768462,
    "delta": 0.029479746903520232,
    "expert_matches_or_exceeds": true
  },
  "SAML-D": {
    "expert_auprc": 0.9501057567112391,
    "adaptive_auprc": 0.9146700934546281,
    "delta": 0.03543566325661096,
    "expert_matches_or_exceeds": true
  },
  "Small-LI": {
    "expert_auprc": 0.11178695284791693,
    "adaptive_auprc": 0.10562375462613144,
    "delta": 0.0061631982217854875,
    "expert_matches_or_exceeds": true
  }
}
```

### 2_infonce_in_domain_utility

```json
{
  "Small-HI": {
    "infonce_auprc": 0.010065033831665494,
    "retains_useful_signal": false
  },
  "SAML-D": {
    "infonce_auprc": 0.5807135549191194,
    "retains_useful_signal": true
  },
  "Small-LI": {
    "infonce_auprc": 0.008798306334598997,
    "retains_useful_signal": false
  }
}
```

### 3_adaptive_beats_single_objective

```json
{
  "targets": [],
  "anywhere": false
}
```

### 4_projection_on_vs_adaptive

```json
{
  "Small-HI": {
    "projection_auprc": 0.39165069925040813,
    "adaptive_auprc": 0.3880503294768462,
    "delta": 0.003600369773561929
  },
  "SAML-D": {
    "projection_auprc": 0.9450436153667968,
    "adaptive_auprc": 0.9146700934546281,
    "delta": 0.03037352191216869
  },
  "Small-LI": {
    "projection_auprc": 0.12059048281418239,
    "adaptive_auprc": 0.10562375462613144,
    "delta": 0.014966728188050948
  }
}
```

### 5_projection_on_hi_li

```json
{
  "Small-HI": {
    "delta_vs_adaptive": 0.003600369773561929,
    "helps": true
  },
  "Small-LI": {
    "delta_vs_adaptive": 0.014966728188050948,
    "helps": true
  }
}
```

### 6_contrast_cross_domain_vs_specialist

```json
{
  "infonce_mean_auprc": 0.19985896502846126,
  "expert_mean_auprc": 0.49314092864650744,
  "contrast_more_cross_domain_than_specialist": false,
  "note": "proxy: mean AUPRC across three targets; not a causal claim"
}
```

### 7_complementary_tf_infonce

```json
{
  "adaptive_mean_auprc": 0.46944805918586857,
  "expert_mean_auprc": 0.49314092864650744,
  "infonce_mean_auprc": 0.19985896502846126,
  "adaptive_beats_both_singles_on_mean": false
}
```

### 8_strongest_common_checkpoint

```json
{
  "mean_auprc_by_arm": {
    "EXPERT_ONLY": 0.49314092864650744,
    "INFONCE_ONLY": 0.19985896502846126,
    "PROJECTION_ON_ADAPTIVE": 0.4857615991437958
  },
  "best_arm": "EXPERT_ONLY",
  "reference_adaptive_mean": 0.46944805918586857
}
```

### 9_limitations

```json
{
  "single_seed": 2,
  "validation_only": true,
  "no_test_estimate": true,
  "matched_cohort_may_differ_from_long_reference": true,
  "one_checkpoint_step": 3000
}
```

### 10_smallest_next_experiment

```json
{
  "proposal": "If adaptive beats singles on \u22652 targets, run matched second-seed replication on the winning arm only; otherwise extend step-1500 frozen diagnostics before adding new objectives.",
  "do_not_add_unrequested_arms": true
}
```


## Cross-arm view matching

- ok=False
- limitation: Historical MIXED_3DOMAIN_LONG training did not log view1_aug_sha256 or view2_aug_sha256 in steps.jsonl. Cross-arm view m…

Confirmation: no test data loaded/scored; adaptive LONG not re-extracted.

