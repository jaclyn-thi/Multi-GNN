# Contrastive objective resource scout (seed 2)

**Thesis role:** diagnostic_or_scout · **validation_status:** diagnostic_only · **table_eligible:** false · **table_group:** `contrastive_objective_resource_scout`

Matched baseline (not retrained): `hi_contrastive_gin_emlps_tds_seed2_proj_asym_8192neg_queue0_accum4_20ep`

Audit reminder: InfoNCE negatives are **per microbatch only**. `accum` does not enlarge the contrastive denominator.

## Recommendation

- **Overall: `replicate_edge_drop_only`**
- True larger batch helped: **False**
- Lower edge drop helped: **True**
- Next: replicate_edge_drop_0.05_on_seeds_1_and_3, consider_edge_drop_0.00_followup, defer_fanout_200_until_after_replication

### Closing (2026-07-20)

Seed1 quickcheck done: **closed / not promoted**. pre-3h+raw improves on seeds 1–2; embedding-only mixed; D mixed. Diagnostic only (`table_eligible=false`). **Do not** train seed3; **do not** run edge_drop_0.00 or fanout_200.

## Training / resource diagnostics

| Variant | Run | bs | accum | edge_drop | OOM fallback | peak GPU MiB | ckpt ep |
|---|---|---:|---:|---:|---|---:|---:|
| baseline | `hi_contrastive_gin_emlps_tds_seed2_proj_asym_8192neg_queue0_accum4_20ep` | 8192 | 4 | 0.1 | False | — | 14 |
| large_bs | `hi_contrastive_large_bs_16384_seed2` | 16384 | 2 | 0.1 | False | 72049 | 8 |
| edge_drop | `hi_contrastive_edge_drop_0.05_seed2` | 8192 | 4 | 0.05 | False | 40941 | 20 |

## Pre-3h metrics (primary)

| Variant | Arm | AUROC | AUPRC | F1 | P@100 | R@100 | P@500 | R@500 | P@1000 | R@1000 | R@P≥0.95 | R@P≥0.90 | R@P≥0.80 | R@P≥0.70 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | A_embedding | 0.9484 | 0.2598 | 0.2947 | 0.7900 | 0.0490 | 0.5860 | 0.1819 | 0.4340 | 0.2694 | — | — | 0.0776 | 0.1260 |
| baseline | B_embedding_raw | 0.9496 | 0.2725 | 0.3078 | 0.7900 | 0.0490 | 0.6080 | 0.1887 | 0.4510 | 0.2800 | — | — | 0.0919 | 0.1434 |
| baseline | D_embedding_raw_temporal_flow | 0.9734 | 0.5111 | 0.5012 | 0.9300 | 0.0577 | 0.8460 | 0.2626 | 0.7090 | 0.4401 | 0.0261 | 0.1359 | 0.3352 | 0.4463 |
| large_bs | A_embedding | 0.9438 | 0.1386 | 0.2369 | 0.5100 | 0.0317 | 0.3920 | 0.1217 | 0.2930 | 0.1819 | — | — | — | — |
| large_bs | B_embedding_raw | 0.9392 | 0.1116 | 0.1811 | 0.4000 | 0.0248 | 0.3320 | 0.1030 | 0.2580 | 0.1601 | — | — | — | — |
| large_bs | D_embedding_raw_temporal_flow | 0.9716 | 0.3077 | 0.4080 | 0.5800 | 0.0360 | 0.6200 | 0.1924 | 0.5100 | 0.3166 | 0.0006 | 0.0006 | 0.0006 | 0.0006 |
| edge_drop | A_embedding | 0.9530 | 0.2870 | 0.3462 | 0.7900 | 0.0490 | 0.6360 | 0.1974 | 0.4530 | 0.2812 | 0.0137 | 0.0205 | 0.0478 | 0.1415 |
| edge_drop | B_embedding_raw | 0.9605 | 0.3251 | 0.3445 | 0.8500 | 0.0528 | 0.6600 | 0.2048 | 0.4940 | 0.3066 | 0.0143 | 0.0403 | 0.0807 | 0.1738 |
| edge_drop | D_embedding_raw_temporal_flow | 0.9782 | 0.4690 | 0.4488 | 0.8600 | 0.0534 | 0.8020 | 0.2489 | 0.6820 | 0.4233 | 0.0199 | 0.0403 | 0.2557 | 0.4084 |

## Paired deltas vs seed2 baseline (pre-3h)

| Variant | ΔA AUPRC | ΔA P@100 | ΔA R@P≥0.80 | ΔB AUPRC | ΔD AUPRC | ΔD P@100 | ΔD R@P≥0.90 | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| large_bs | -0.1211 | -0.2800 | — | -0.1610 | -0.2035 | -0.3500 | -0.1353 | stop |
| edge_drop | 0.0273 | 0.0000 | -0.0298 | 0.0526 | -0.0422 | -0.0700 | -0.0956 | replicate_seeds_1_3 |

## Notes

- Do not count D-only gains as representation improvement.
- Fanout_200 and edge_drop_0.00 were **not** launched in this batch.
- Do not insert into main thesis tables yet.

