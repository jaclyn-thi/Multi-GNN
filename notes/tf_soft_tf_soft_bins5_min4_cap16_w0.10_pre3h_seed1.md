# Temporal flow causal ablation — Small-HI (`hi_tf_soft_tf_soft_bins5_min4_cap16_w0.10_optv2_gin_emlps_tds_asym_proj_8192neg_queue0_accum4_20ep_seed1`)

- **embedding:** `embeddings/hi_tf_soft_tf_soft_bins5_min4_cap16_w0.10_optv2_gin_emlps_tds_asym_proj_8192neg_queue0_accum4_20ep_seed1/pre_embedding_3h` (198-d pre-3h)
- **cache:** `results/cache/temporal_flow_causal/Small-HI`
- **primary comparison:** Arm D vs Arm B (ΔAUPRC = +0.1643)
- **no SSL retraining:** True

## Four-arm test metrics

| arm | groups | dim | AUROC | AUPRC | F1 | P@100 | R@100 | lift@100 |
|-----|--------|----:|------:|------:|---:|------:|------:|---------:|
| A_embedding | pre_embedding_3h | 198 | 0.9424 | 0.1149 | 0.2253 | 0.3100 | 0.0192 | 166.07 |
| B_embedding_raw | pre_embedding_3h, raw | 202 | 0.9611 | 0.2477 | 0.1299 | 0.7100 | 0.0441 | 380.36 |
| D_embedding_raw_temporal_flow | pre_embedding_3h, raw, temporal_flow_causal | 207 | 0.9752 | 0.4119 | 0.1972 | 0.7700 | 0.0478 | 412.51 |

## Primary deltas (D − B)

- ΔAUPRC: **+0.1643**
- ΔF1: +0.0673
- ΔP@100: +0.0600
- ΔR@100: +0.0037
- Δlift@100: +32.14

Conservative read: single checkpoint; downstream probe only.
