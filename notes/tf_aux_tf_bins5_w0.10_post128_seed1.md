# Temporal flow causal ablation — Small-HI (`hi_tf_aux_tf_bins5_w0.10_gin_emlps_tds_asym_proj_8192neg_queue0_accum4_20ep_seed1`)

- **embedding:** `embeddings/hi_tf_aux_tf_bins5_w0.10_gin_emlps_tds_asym_proj_8192neg_queue0_accum4_20ep_seed1` (128-d pre-3h)
- **cache:** `results/cache/temporal_flow_causal/Small-HI`
- **primary comparison:** Arm D vs Arm B (ΔAUPRC = +0.2133)
- **no SSL retraining:** True

## Four-arm test metrics

| arm | groups | dim | AUROC | AUPRC | F1 | P@100 | R@100 | lift@100 |
|-----|--------|----:|------:|------:|---:|------:|------:|---------:|
| A_embedding | pre_embedding_3h | 128 | 0.9457 | 0.2012 | 0.2605 | 0.6700 | 0.0416 | 358.93 |
| B_embedding_raw | pre_embedding_3h, raw | 132 | 0.9518 | 0.2579 | 0.3427 | 0.7500 | 0.0466 | 401.79 |
| D_embedding_raw_temporal_flow | pre_embedding_3h, raw, temporal_flow_causal | 137 | 0.9704 | 0.4712 | 0.4926 | 0.9500 | 0.0590 | 508.94 |

## Primary deltas (D − B)

- ΔAUPRC: **+0.2133**
- ΔF1: +0.1499
- ΔP@100: +0.2000
- ΔR@100: +0.0124
- Δlift@100: +107.14

Conservative read: single checkpoint; downstream probe only.
