# Edge-drop 0.05 seed1 quickcheck — closed

**Status:** **closed** · diagnostic/promising · **not promoted**  
**Thesis role:** diagnostic_or_scout · **validation_status:** diagnostic_only · **table_eligible:** false

Run: `hi_contrastive_edge_drop_0.05_seed1` · Baseline: `hi_contrastive_gin_emlps_tds_proj_asym_8192neg_queue0_accum4_20ep`  
Selected checkpoint epoch: **20**

## Closing decision

**Keep diagnostic only. Do not promote. Do not train seed3. Do not run `edge_drop_0.00` or `fanout_200`.**

Summary across matched seeds 1–2:

- **pre-3h + raw (B):** improves on both matched seeds (seed1 ΔAUPRC **+0.071**; seed2 **+0.053**).
- **pre-3h embedding-only (A):** **mixed** (seed2 +0.027; seed1 **−0.037**, P@100 0.66→0.49).
- **final D stack:** **mixed** (seed1 ΔD AUPRC +0.023; seed2 **−0.042**, with P@100 / R@P≥0.90 down on seed2).

Not thesis-table eligible.

## Recommendation (closed): `keep_diagnostic_promising`

- Seed1 primary A/B success: **True** (A up=False, B up=True)
- Seed1 P@100 collapse: **False**
- Seed2 prior edge_drop helped: **True**
- Explicitly **not** launching: seed3, edge_drop_0.00, fanout_200

## Training diagnostics

- edge_drop_target_rate: `0.05`
- peak GPU MiB: 44693
- shared_seed line: `requested_seed_edges=8192 shared_seed_edges=7402 queue_size=0`
- edge_drop line: `—`

## Pre-3h metrics vs matched seed1 baseline

| Variant | Arm | AUROC | AUPRC | F1 | P@100 | R@100 | Lift@100 | P@500 | R@500 | Lift@500 | P@1000 | R@1000 | Lift@1000 | R@P≥0.90 | R@P≥0.80 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | A_embedding | 0.9608 | 0.1888 | 0.2504 | 0.6600 | 0.0410 | 353.5773 | 0.4580 | 0.1421 | 245.3612 | 0.3320 | 0.2061 | 177.8601 | 0.0006 | 0.0006 |
| baseline | B_embedding_raw | 0.9617 | 0.2113 | 0.2872 | 0.6700 | 0.0416 | 358.9345 | 0.5020 | 0.1558 | 268.9330 | 0.3690 | 0.2291 | 197.6818 | 0.0006 | 0.0006 |
| baseline | D_embedding_raw_temporal_flow | 0.9786 | 0.4337 | 0.4482 | 0.9400 | 0.0583 | 503.5798 | 0.8040 | 0.2495 | 430.7214 | 0.6090 | 0.3780 | 326.2554 | 0.1291 | 0.2514 |
| edge_drop_0.05 | A_embedding | 0.9443 | 0.1521 | 0.2451 | 0.4900 | 0.0304 | 262.5043 | 0.3800 | 0.1179 | 203.5748 | 0.3100 | 0.1924 | 166.0742 | — | — |
| edge_drop_0.05 | B_embedding_raw | 0.9602 | 0.2826 | 0.0753 | 0.7600 | 0.0472 | 407.1496 | 0.5840 | 0.1813 | 312.8623 | 0.4670 | 0.2899 | 250.1827 | 0.0019 | 0.0137 |
| edge_drop_0.05 | D_embedding_raw_temporal_flow | 0.9770 | 0.4570 | 0.1833 | 0.9000 | 0.0559 | 482.1508 | 0.7820 | 0.2427 | 418.9355 | 0.6310 | 0.3917 | 338.0413 | 0.0559 | 0.2142 |

## Paired deltas (edge_drop − seed1 baseline)

| ΔA AUPRC | ΔA P@100 | ΔA R@P≥0.80 | ΔA R@P≥0.90 | ΔB AUPRC | ΔB P@100 | ΔD AUPRC |
|---:|---:|---:|---:|---:|---:|---:|
| -0.0366 | -0.1700 | — | — | 0.0713 | 0.0900 | 0.0233 |

## Notes

- Primary decision uses pre-3h A/B only.
- D is reported for completeness; do not count D-only gains as representation proof.
- Post-128 was not extracted/probed.
- Seed3 not trained; edge_drop_0.00 and fanout_200 not run.
