# Edge-attr mask 0.05 seed2 quickcheck

**Thesis role:** diagnostic_or_scout · **validation_status:** diagnostic_only · **table_eligible:** false

Run: `hi_contrastive_edge_attr_mask_0.05_seed2` · Baseline: `hi_contrastive_gin_emlps_tds_seed2_proj_asym_8192neg_queue0_accum4_20ep`
Selected checkpoint epoch: **17**
edge_attr_mask_rate: **0.05** (baseline hardcoded/default 0.1)
edge_drop_target_rate: **0.1** (unchanged default)

## Recommendation: `keep_diagnostic_promising`

- Primary A/B success: **True** (A up=False, B up=True)
- P@100 collapse: **False**
- Next: keep diagnostic; B-only or mixed — do not promote; no seed3

## Training diagnostics

- peak GPU MiB: 36953
- shared_seed: `2026-07-20 18:07:30,115 [INFO ] Hetero contrastive seed-edge filtering: requested_seed_edges=8192 shared_seed_edges=6681 queue_size=0`
- view_aug log: `2026-07-20 18:07:28,071 [INFO ] Contrastive view aug: edge_drop_target_rate=0.1000 edge_attr_mask_rate=0.0500 policy=random`

## Pre-3h metrics vs matched seed2 baseline

| Variant | Arm | AUROC | AUPRC | F1 | P@100 | R@100 | Lift@100 | P@500 | R@500 | Lift@500 | P@1000 | R@1000 | Lift@1000 | R@P≥0.90 | R@P≥0.80 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | A_embedding | 0.9484 | 0.2598 | 0.2947 | 0.7900 | 0.0490 | 423.2365 | 0.5860 | 0.1819 | 313.9450 | 0.4340 | 0.2694 | 232.5122 | — | 0.0776 |
| baseline | B_embedding_raw | 0.9496 | 0.2725 | 0.3078 | 0.7900 | 0.0490 | 423.2365 | 0.6080 | 0.1887 | 325.7314 | 0.4510 | 0.2800 | 241.6198 | — | 0.0919 |
| baseline | D_embedding_raw_temporal_flow | 0.9734 | 0.5111 | 0.5012 | 0.9300 | 0.0577 | 498.2404 | 0.8460 | 0.2626 | 453.2381 | 0.7090 | 0.4401 | 379.8414 | 0.1359 | 0.3352 |
| attr_mask_0.05 | A_embedding | 0.9648 | 0.2382 | 0.3278 | 0.5500 | 0.0341 | 294.6583 | 0.5180 | 0.1608 | 277.5146 | 0.4160 | 0.2582 | 222.8688 | — | — |
| attr_mask_0.05 | B_embedding_raw | 0.9724 | 0.3470 | 0.1361 | 0.7600 | 0.0472 | 407.1642 | 0.6720 | 0.2086 | 360.0189 | 0.5300 | 0.3290 | 283.9435 | — | — |
| attr_mask_0.05 | D_embedding_raw_temporal_flow | 0.9827 | 0.4898 | 0.1777 | 0.8600 | 0.0534 | 460.7385 | 0.7780 | 0.2415 | 416.8076 | 0.6670 | 0.4140 | 357.3402 | — | 0.1943 |

## Paired deltas (attr_mask_0.05 − seed2 baseline)

| ΔA AUPRC | ΔA P@100 | ΔA R@P≥0.80 | ΔA R@P≥0.90 | ΔB AUPRC | ΔB P@100 | ΔD AUPRC |
|---:|---:|---:|---:|---:|---:|---:|
| -0.0216 | -0.2400 | — | — | 0.0745 | -0.0300 | -0.0214 |

## Notes

- Primary decision uses pre-3h A/B only.
- D reported for completeness only if present; do not count D-only gains.
- Post-128 not extracted/probed.
- Not table-eligible.

