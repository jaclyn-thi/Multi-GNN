# Temporal-flow regression aux multiseed confirmation

**Thesis role:** diagnostic_or_scout · **validation_status:** diagnostic_only · **table_eligible:** false · **table_group:** `temporal_flow_regression_aux_multiseed`

Primary representation: **pre_embedding_3h**. Post-128 diagnostic only.
SSL: InfoNCE + temporal_flow_causal **regression** (Huber). **No labels.**
Attach point: `post_embedding_head_pre_projection` (fixed; no new attach-point code).

Excluded: bins, soft positives, morphology/degflow/clustering/betweenness.

## Recommendation

- **Overall: `keep_diagnostic`** (best claim via `tf_reg_w0.05`)
- More stable weight (higher mean A AUPRC, lower SD): **tf_reg_w0.10**
- Best pre-3h embedding-only (A): **tf_reg_w0.10_seed3** (0.4165)
- Best pre-3h + raw (B): **tf_reg_w0.10_seed3** (0.4824)
- Best final D by AUPRC: **tf_reg_w0.10_seed3** (0.5726)
- Best final D by R@P≥0.90: **tf_reg_w0.10_seed3** (0.2322)
- Best final D by R@P≥0.80: **tf_reg_w0.10_seed3** (0.4488)

## Per-variant claims

### `tf_reg_w0.10`

- Flags: `--aux_temporal_flow regression --aux_temporal_flow_weight 0.10 --aux_temporal_flow_loss huber`
- Recommendation: **stop**
- Claim 1 (representation): **False** (A most=False, B most=False, mean ΔA=0.0318, collapse=none)
- Claim 2 (final D): **False** (D AUPRC most=False)
- Pre-3h A AUPRC mean±SD: 0.3096±0.1140 (n=3)
- Pre-3h B AUPRC mean±SD: 0.3444±0.1360 (n=3)
- Pre-3h D AUPRC mean±SD: 0.4322±0.1230 (n=3)

### `tf_reg_w0.05`

- Flags: `--aux_temporal_flow regression --aux_temporal_flow_weight 0.05 --aux_temporal_flow_loss huber`
- Recommendation: **keep_diagnostic**
- Claim 1 (representation): **True** (A most=True, B most=True, mean ΔA=0.1259, collapse=none)
- Claim 2 (final D): **False** (D AUPRC most=False)
- Pre-3h A AUPRC mean±SD: 0.3070±0.0779 (n=3)
- Pre-3h B AUPRC mean±SD: 0.3427±0.0812 (n=3)
- Pre-3h D AUPRC mean±SD: 0.4567±0.0436 (n=3)

## Per-seed pre-3h metrics

| Seed | Variant | ckpt ep | Arm | AUROC | AUPRC | F1 | P@100 | R@100 | P@500 | R@500 | P@1000 | R@1000 | R@P≥0.95 | R@P≥0.90 | R@P≥0.80 | R@P≥0.70 |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | baseline | 19 | A_embedding | 0.9608 | 0.1888 | 0.2504 | 0.6600 | 0.0410 | 0.4580 | 0.1421 | 0.3320 | 0.2061 | 0.0006 | 0.0006 | 0.0006 | 0.0006 |
| 1 | baseline | 19 | B_embedding_raw | 0.9617 | 0.2113 | 0.2872 | 0.6700 | 0.0416 | 0.5020 | 0.1558 | 0.3690 | 0.2291 | 0.0006 | 0.0006 | 0.0006 | 0.0782 |
| 1 | baseline | 19 | D_embedding_raw_temporal_flow | 0.9786 | 0.4337 | 0.4482 | 0.9400 | 0.0583 | 0.8040 | 0.2495 | 0.6090 | 0.3780 | 0.0012 | 0.1291 | 0.2514 | 0.3340 |
| 1 | tf_reg_w0.10 | 17 | A_embedding | 0.9700 | 0.3226 | 0.3239 | 0.6900 | 0.0428 | 0.6540 | 0.2030 | 0.5220 | 0.3240 | — | — | — | 0.1390 |
| 1 | tf_reg_w0.10 | 17 | B_embedding_raw | 0.9716 | 0.3404 | 0.2970 | 0.7200 | 0.0447 | 0.6720 | 0.2086 | 0.5330 | 0.3309 | — | — | — | 0.1676 |
| 1 | tf_reg_w0.10 | 17 | D_embedding_raw_temporal_flow | 0.9774 | 0.3801 | 0.3033 | 0.7600 | 0.0472 | 0.6960 | 0.2160 | 0.5620 | 0.3489 | — | — | — | 0.2148 |
| 1 | tf_reg_w0.05 | 17 | A_embedding | 0.9688 | 0.3719 | 0.4158 | 0.9000 | 0.0559 | 0.7180 | 0.2228 | 0.5310 | 0.3296 | 0.0248 | 0.0726 | 0.1589 | 0.2477 |
| 1 | tf_reg_w0.05 | 17 | B_embedding_raw | 0.9666 | 0.3549 | 0.3788 | 0.8800 | 0.0546 | 0.7080 | 0.2197 | 0.5180 | 0.3215 | 0.0354 | 0.0453 | 0.1496 | 0.2222 |
| 1 | tf_reg_w0.05 | 17 | D_embedding_raw_temporal_flow | 0.9765 | 0.4504 | 0.4381 | 0.9100 | 0.0565 | 0.8000 | 0.2483 | 0.6390 | 0.3966 | 0.0503 | 0.0962 | 0.2557 | 0.3464 |
| 2 | baseline | 14 | A_embedding | 0.9484 | 0.2598 | 0.2947 | 0.7900 | 0.0490 | 0.5860 | 0.1819 | 0.4340 | 0.2694 | — | — | 0.0776 | 0.1260 |
| 2 | baseline | 14 | B_embedding_raw | 0.9496 | 0.2725 | 0.3078 | 0.7900 | 0.0490 | 0.6080 | 0.1887 | 0.4510 | 0.2800 | — | — | 0.0919 | 0.1434 |
| 2 | baseline | 14 | D_embedding_raw_temporal_flow | 0.9734 | 0.5111 | 0.5012 | 0.9300 | 0.0577 | 0.8460 | 0.2626 | 0.7090 | 0.4401 | 0.0261 | 0.1359 | 0.3352 | 0.4463 |
| 2 | tf_reg_w0.10 | 20 | A_embedding | 0.9717 | 0.1896 | 0.2823 | 0.6100 | 0.0379 | 0.3560 | 0.1105 | 0.3040 | 0.1887 | — | — | 0.0106 | 0.0261 |
| 2 | tf_reg_w0.10 | 20 | B_embedding_raw | 0.9733 | 0.2104 | 0.2955 | 0.6600 | 0.0410 | 0.3860 | 0.1198 | 0.3280 | 0.2036 | 0.0006 | 0.0112 | 0.0155 | 0.0279 |
| 2 | tf_reg_w0.10 | 20 | D_embedding_raw_temporal_flow | 0.9814 | 0.3438 | 0.4029 | 0.7400 | 0.0459 | 0.5900 | 0.1831 | 0.4880 | 0.3029 | — | 0.0074 | 0.0422 | 0.0869 |
| 2 | tf_reg_w0.05 | 13 | A_embedding | 0.9679 | 0.3285 | 0.2987 | 0.9100 | 0.0565 | 0.6580 | 0.2042 | 0.4910 | 0.3048 | 0.0236 | 0.0670 | 0.1266 | 0.1825 |
| 2 | tf_reg_w0.05 | 13 | B_embedding_raw | 0.9734 | 0.4171 | 0.1490 | 0.9700 | 0.0602 | 0.7640 | 0.2371 | 0.5830 | 0.3619 | 0.0869 | 0.1409 | 0.2129 | 0.2694 |
| 2 | tf_reg_w0.05 | 13 | D_embedding_raw_temporal_flow | 0.9796 | 0.5030 | 0.4147 | 0.9800 | 0.0608 | 0.8500 | 0.2638 | 0.6650 | 0.4128 | 0.1248 | 0.2092 | 0.3079 | 0.3892 |
| 3 | tf_reg_w0.10 | 20 | A_embedding | 0.9731 | 0.4165 | 0.4688 | 0.8900 | 0.0552 | 0.7900 | 0.2452 | 0.5980 | 0.3712 | 0.0043 | 0.0528 | 0.2334 | 0.3284 |
| 3 | tf_reg_w0.10 | 20 | B_embedding_raw | 0.9763 | 0.4824 | 0.3632 | 0.9300 | 0.0577 | 0.8380 | 0.2601 | 0.6770 | 0.4202 | 0.0081 | 0.0931 | 0.3253 | 0.4146 |
| 3 | tf_reg_w0.10 | 20 | D_embedding_raw_temporal_flow | 0.9819 | 0.5726 | 0.4458 | 0.9600 | 0.0596 | 0.8920 | 0.2768 | 0.7700 | 0.4780 | 0.0956 | 0.2322 | 0.4488 | 0.5345 |
| 3 | tf_reg_w0.05 | 20 | A_embedding | 0.9608 | 0.2207 | 0.2905 | 0.7000 | 0.0435 | 0.4940 | 0.1533 | 0.3850 | 0.2390 | 0.0068 | 0.0068 | 0.0298 | 0.0435 |
| 3 | tf_reg_w0.05 | 20 | B_embedding_raw | 0.9637 | 0.2560 | 0.3218 | 0.7700 | 0.0478 | 0.5620 | 0.1744 | 0.4210 | 0.2613 | 0.0037 | 0.0037 | 0.0416 | 0.0813 |
| 3 | tf_reg_w0.05 | 20 | D_embedding_raw_temporal_flow | 0.9764 | 0.4166 | 0.4430 | 0.8800 | 0.0546 | 0.7640 | 0.2371 | 0.5800 | 0.3600 | 0.0317 | 0.0478 | 0.1825 | 0.2955 |

## Post-128 diagnostic (A/B/D)

| Seed | Variant | Arm | AUPRC | P@100 | R@P≥0.90 | R@P≥0.80 |
|---:|---|---|---:|---:|---:|---:|
| 1 | baseline | A_embedding | 0.2131 | 0.8500 | 0.0379 | 0.0677 |
| 1 | baseline | B_embedding_raw | 0.2436 | 0.8400 | 0.0168 | 0.0770 |
| 1 | tf_reg_w0.10 | A_embedding | 0.3731 | 0.7800 | 0.0043 | 0.0813 |
| 1 | tf_reg_w0.10 | B_embedding_raw | 0.4149 | 0.7900 | 0.0062 | 0.1912 |
| 1 | tf_reg_w0.10 | D_embedding_raw_temporal_flow | 0.5112 | 0.8500 | 0.0341 | 0.3426 |
| 1 | tf_reg_w0.05 | A_embedding | 0.2785 | 0.8200 | 0.0304 | 0.0720 |
| 1 | tf_reg_w0.05 | B_embedding_raw | 0.3867 | 0.8500 | 0.0137 | 0.0944 |
| 1 | tf_reg_w0.05 | D_embedding_raw_temporal_flow | 0.4853 | 0.8900 | 0.0466 | 0.2539 |
| 2 | baseline | A_embedding | 0.1415 | 0.7000 | — | — |
| 2 | baseline | B_embedding_raw | 0.1918 | 0.7400 | 0.0006 | 0.0006 |
| 2 | tf_reg_w0.10 | A_embedding | 0.1510 | 0.3200 | — | — |
| 2 | tf_reg_w0.10 | B_embedding_raw | 0.1430 | 0.2700 | — | — |
| 2 | tf_reg_w0.10 | D_embedding_raw_temporal_flow | 0.3068 | 0.7300 | — | — |
| 2 | tf_reg_w0.05 | A_embedding | 0.2364 | 0.7200 | 0.0081 | 0.0304 |
| 2 | tf_reg_w0.05 | B_embedding_raw | 0.3974 | 0.9300 | 0.0633 | 0.1347 |
| 2 | tf_reg_w0.05 | D_embedding_raw_temporal_flow | 0.5205 | 0.9800 | 0.1030 | 0.2936 |
| 3 | tf_reg_w0.10 | A_embedding | 0.3643 | 0.8100 | — | 0.1719 |
| 3 | tf_reg_w0.10 | B_embedding_raw | 0.4265 | 0.8900 | 0.1124 | 0.2638 |
| 3 | tf_reg_w0.10 | D_embedding_raw_temporal_flow | 0.5337 | 0.9500 | 0.2309 | 0.3929 |
| 3 | tf_reg_w0.05 | A_embedding | 0.1715 | 0.5200 | 0.0006 | 0.0043 |
| 3 | tf_reg_w0.05 | B_embedding_raw | 0.2504 | 0.7200 | 0.0019 | 0.0304 |
| 3 | tf_reg_w0.05 | D_embedding_raw_temporal_flow | 0.4288 | 0.8500 | 0.0230 | 0.1633 |

## Paired deltas (variant − matched baseline, pre-3h)

| Variant | Seed | ΔA AUPRC | ΔA P@100 | ΔB AUPRC | ΔD AUPRC | ΔD P@100 | ΔD R@P≥0.90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| tf_reg_w0.10 | 1 | 0.1338 | 0.0300 | 0.1291 | -0.0536 | -0.1800 | — |
| tf_reg_w0.10 | 2 | -0.0702 | -0.1800 | -0.0621 | -0.1673 | -0.1900 | -0.1285 |
| tf_reg_w0.10 | 3 | — | — | — | — | — | — |
| tf_reg_w0.05 | 1 | 0.1831 | 0.2400 | 0.1436 | 0.0167 | -0.0300 | -0.0329 |
| tf_reg_w0.05 | 2 | 0.0688 | 0.1200 | 0.1445 | -0.0081 | 0.0500 | 0.0732 |
| tf_reg_w0.05 | 3 | — | — | — | — | — | — |

## Baseline availability

- Seed 1: matched morph_obj_baseline probes (reuse).
- Seed 2: matched morph_obj_baseline probes (reuse; no retrain).
- Seed 3: no matched baseline; absolute metrics only; **not retrained**.

## Notes

- Do not count D-only gains as representation improvement.
- Do not promote on post-128-only gains.
- Do not insert into main thesis tables yet.

