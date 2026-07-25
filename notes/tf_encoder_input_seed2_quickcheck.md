# Temporal-flow as encoder input — seed2 quickcheck

**Thesis role:** diagnostic_or_scout · **validation_status:** diagnostic_only · **table_eligible:** false

Run: `hi_contrastive_tf_encoder_input_seed2` · Baseline: `hi_contrastive_gin_emlps_tds_seed2_proj_asym_8192neg_queue0_accum4_20ep`
Selected checkpoint epoch: **20**
Flag: `--include_temporal_flow_edge_features`
edge_dim: **8 → 13**
TF features: `['log1p_sender_interarrival', 'log1p_receiver_interarrival', 'log1p_sender_past_7d_count', 'log1p_amount_vs_sender_past_mean', 'pair_repeat_indicator']`
ssl_labels_used: **false** · aux_tf: none · soft_pos: false · morph: false

## Recommendation: `consider_seed1_replication`

- Primary A/B success: **True** (A up=True, B up=True)
- P@100 collapse: **False**
- Next: optional one seed1 replication of --include_temporal_flow_edge_features

## Training diagnostics

- peak GPU MiB: 39873
- training time sec: 5060
- scaler: `temporal_flow_cache split_train_edge_id (train-only)`
- TF append log: `2026-07-20 18:28:45,917 [INFO ] Appended temporal_flow_causal encoder edge features: edge_dim 8 -> 13 (cache=results/cache/temporal_flow_causal/Small-HI, uses_labels=false, past_only=true, train-only scaler)`

## Pre-3h metrics vs matched seed2 baseline

| Variant | Arm | AUROC | AUPRC | F1 | P@100 | R@100 | Lift@100 | P@500 | R@500 | Lift@500 | P@1000 | R@1000 | Lift@1000 | R@P≥0.90 | R@P≥0.80 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | A_embedding | 0.9484 | 0.2598 | 0.2947 | 0.7900 | 0.0490 | 423.2365 | 0.5860 | 0.1819 | 313.9450 | 0.4340 | 0.2694 | 232.5122 | — | 0.0776 |
| baseline | B_embedding_raw | 0.9496 | 0.2725 | 0.3078 | 0.7900 | 0.0490 | 423.2365 | 0.6080 | 0.1887 | 325.7314 | 0.4510 | 0.2800 | 241.6198 | — | 0.0919 |
| baseline | D_embedding_raw_temporal_flow | 0.9734 | 0.5111 | 0.5012 | 0.9300 | 0.0577 | 498.2404 | 0.8460 | 0.2626 | 453.2381 | 0.7090 | 0.4401 | 379.8414 | 0.1359 | 0.3352 |
| tf_encoder_input | A_embedding | 0.9693 | 0.4750 | 0.5006 | 0.8200 | 0.0509 | 439.3001 | 0.7920 | 0.2458 | 424.2996 | 0.6740 | 0.4184 | 361.0833 | 0.0168 | 0.2067 |
| tf_encoder_input | B_embedding_raw | 0.9696 | 0.4925 | 0.4298 | 0.7800 | 0.0484 | 417.8708 | 0.7900 | 0.2452 | 423.2282 | 0.6930 | 0.4302 | 371.2622 | 0.0099 | 0.2297 |
| tf_encoder_input | D_embedding_raw_temporal_flow | 0.9710 | 0.4979 | 0.5417 | 0.8400 | 0.0521 | 450.0147 | 0.7940 | 0.2464 | 425.3711 | 0.6880 | 0.4271 | 368.5835 | 0.0267 | 0.2328 |

## Paired deltas (tf_encoder_input − seed2 baseline)

| ΔA AUPRC | ΔA P@100 | ΔA R@P≥0.80 | ΔA R@P≥0.90 | ΔB AUPRC | ΔB P@100 | ΔD AUPRC |
|---:|---:|---:|---:|---:|---:|---:|
| 0.2153 | 0.0300 | 0.1291 | — | 0.2200 | -0.0100 | -0.0132 |

## Notes

- Primary decision uses pre-3h A/B only.
- D double-counts TF already in the encoder; D-only gains are not success.
- Post-128 not extracted/probed.
- Not table-eligible.

