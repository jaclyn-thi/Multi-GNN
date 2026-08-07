# Phase-4B MIXED_3DOMAIN_LONG frozen R198 six-cell validation eval

> Twin: `results/diagnostics/financial_multidataset_shared_core_phase4b_mixed_long_frozen_eval.json`
> LONG training: `results/diagnostics/financial_multidataset_shared_core_phase4b_mixed_long_3000/arms/MIXED_3DOMAIN_LONG/summary.json`
> SHORT reuse root: `results/diagnostics/financial_multidataset_shared_core_phase4b_frozen_eval`

**ok=True** — validation-only; no encoder retrain; no test.

SHORT comparability: `COMPARABLE_REUSE_AUTHORIZED`

## Checkpoints (predeclared; not val-selected)

| Encoder | step | updates/domain | LR phase | enc LR | sha256 |
|---|---:|---:|---|---:|---|
| SHORT MIXED | 1500 | 500 | linear (end) | 0.0002 | `a88ba7ea3bcb5224…` |
| LONG | 1500 | 500 | linear (mid) | 0.00132472 | `85e71a42cbcbf225…` |
| LONG | 3000 | 1000 | linear (end) | 0.0002 | `092a8c1159dc8b16…` |

## Small-HI

| encoder/checkpoint | step | upd/dom | LR phase | AUPRC | AUROC | F1@0.5 | P@0.5 | R@0.5 | F1@val-thr† | final BCE | train_n | val_n | val_pos | edge_cov | val EdgeID |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| SHORT@1500 | 1500 | 500 | linear | 0.3232 | 0.9750 | 0.3572 | 0.6241 | 0.2502 | 0.4009 | 0.004726 | 3248269 | 965466 | 1035 | 0.9997993179889569 | `ceb56ae8afe3…` |
| LONG@1500 | 1500 | 500 | linear | 0.2917 | 0.9717 | 0.3259 | 0.5774 | 0.2271 | 0.3906 | 0.005075 | 3248269 | 965466 | 1035 | 0.9997993179889569 | `ceb56ae8afe3…` |
| LONG@3000 | 3000 | 1000 | linear | 0.3881 | 0.9787 | 0.4310 | 0.6485 | 0.3227 | 0.4507 | 0.004383 | 3248269 | 965466 | 1035 | 0.9997993179889569 | `ceb56ae8afe3…` |

### Comparisons
- **A — LONG@1500 vs SHORT@1500 (schedule horizon)**: AUPRC Δ=-0.0315 (retention 0.903); F1@0.5 Δ=-0.0313
- **B — LONG@3000 vs LONG@1500 (extra exposure)**: AUPRC Δ=+0.0964 (retention 1.330); F1@0.5 Δ=+0.1050
- **C — LONG@3000 vs SHORT@1500 (overall longer)**: AUPRC Δ=+0.0649 (retention 1.201); F1@0.5 Δ=+0.0737

## SAML-D

| encoder/checkpoint | step | upd/dom | LR phase | AUPRC | AUROC | F1@0.5 | P@0.5 | R@0.5 | F1@val-thr† | final BCE | train_n | val_n | val_pos | edge_cov | val EdgeID |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| SHORT@1500 | 1500 | 500 | linear | 0.9291 | 0.9978 | 0.8875 | 0.9870 | 0.8062 | 0.8966 | 0.001386 | 5027798 | 1653322 | 1883 | 0.8701213880285563 | `e6ebe9714b3f…` |
| LONG@1500 | 1500 | 500 | linear | 0.9319 | 0.9980 | 0.8994 | 0.9734 | 0.8359 | 0.8999 | 0.001220 | 5027798 | 1653322 | 1883 | 0.8701213880285563 | `e6ebe9714b3f…` |
| LONG@3000 | 3000 | 1000 | linear | 0.9147 | 0.9983 | 0.8727 | 0.9432 | 0.8120 | 0.8746 | 0.002001 | 5027798 | 1653322 | 1883 | 0.8701213880285563 | `e6ebe9714b3f…` |

### Comparisons
- **A — LONG@1500 vs SHORT@1500 (schedule horizon)**: AUPRC Δ=+0.0027 (retention 1.003); F1@0.5 Δ=+0.0120
- **B — LONG@3000 vs LONG@1500 (extra exposure)**: AUPRC Δ=-0.0172 (retention 0.982); F1@0.5 Δ=-0.0267
- **C — LONG@3000 vs SHORT@1500 (overall longer)**: AUPRC Δ=-0.0145 (retention 0.984); F1@0.5 Δ=-0.0147

## Small-LI

| encoder/checkpoint | step | upd/dom | LR phase | AUPRC | AUROC | F1@0.5 | P@0.5 | R@0.5 | F1@val-thr† | final BCE | train_n | val_n | val_pos | edge_cov | val EdgeID |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| SHORT@1500 | 1500 | 500 | linear | 0.0761 | 0.9594 | 0.1144 | 0.3103 | 0.0701 | 0.1411 | 0.003779 | 4431773 | 1316415 | 770 | 0.9997380967097638 | `fee6f1feb56c…` |
| LONG@1500 | 1500 | 500 | linear | 0.0926 | 0.9634 | 0.0985 | 0.5060 | 0.0545 | 0.1596 | 0.003620 | 4431773 | 1316415 | 770 | 0.9997380967097638 | `fee6f1feb56c…` |
| LONG@3000 | 3000 | 1000 | linear | 0.1056 | 0.9613 | 0.1365 | 0.4919 | 0.0792 | 0.1798 | 0.003667 | 4431773 | 1316415 | 770 | 0.9997380967097638 | `fee6f1feb56c…` |

### Comparisons
- **A — LONG@1500 vs SHORT@1500 (schedule horizon)**: AUPRC Δ=+0.0165 (retention 1.216); F1@0.5 Δ=-0.0159
- **B — LONG@3000 vs LONG@1500 (extra exposure)**: AUPRC Δ=+0.0130 (retention 1.141); F1@0.5 Δ=+0.0380
- **C — LONG@3000 vs SHORT@1500 (overall longer)**: AUPRC Δ=+0.0295 (retention 1.387); F1@0.5 Δ=+0.0221

† F1@val-thr is an optimistic validation-selected-threshold diagnostic.

## Three-domain summary

| Target | SHORT@1500 AUPRC | LONG@1500 AUPRC | LONG@3000 AUPRC | SHORT F1@0.5 | LONG@1500 F1 | LONG@3000 F1 |
|---|---:|---:|---:|---:|---:|---:|
| Small-HI | 0.3232 | 0.2917 | 0.3881 | 0.3572 | 0.3259 | 0.4310 |
| SAML-D | 0.9291 | 0.9319 | 0.9147 | 0.8875 | 0.8994 | 0.8727 |
| Small-LI | 0.0761 | 0.0926 | 0.1056 | 0.1144 | 0.0985 | 0.1365 |

## Step-2250

- Scientifically justified to propose: **False**
- LONG@1500 wins over LONG@3000 on 1/3 targets
- Mean AUPRC Δ(3000−1500) = +0.0307
- Submitted: **False** (stop after six-cell eval)

## Caveats

- Training InfoNCE / α are not representation-quality conclusions.
- Do not compare absolute AUPRC across HI vs SAML vs LI as equal difficulty.
- F1@val-thr is optimistic/in-sample, not a test estimate.
- SAML-D coverage uses floors due to extraction_loader_coverage_defect.
- SHORT metrics reused only after comparability card PASS.
- Primary models are fixed predeclared checkpoints (not per-target best).

Embeddings disk ≈ 25.05 GiB

Confirmation: no test data loaded/scored; no encoder retraining.

## Specialist comparison addendum (LI step-500)

Consolidated LI@500 / LI@1000 / SHORT / LONG specialist comparison (no re-probe):

- [`notes/financial_multidataset_shared_core_phase4b_specialist_comparison_addendum.md`](financial_multidataset_shared_core_phase4b_specialist_comparison_addendum.md)
- [`specialist_comparison_with_li_step500.json`](../results/diagnostics/financial_multidataset_shared_core_phase4b_mixed_long_frozen_eval/specialist_comparison_with_li_step500.json)

Stop after this evaluation — no automatic next training phase.

