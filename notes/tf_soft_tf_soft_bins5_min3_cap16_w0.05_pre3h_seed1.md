# Temporal flow causal ablation — Small-HI (`hi_tf_soft_tf_soft_bins5_min3_cap16_w0.05_optv2_gin_emlps_tds_asym_proj_8192neg_queue0_accum4_20ep_seed1`)

- **embedding:** `embeddings/hi_tf_soft_tf_soft_bins5_min3_cap16_w0.05_optv2_gin_emlps_tds_asym_proj_8192neg_queue0_accum4_20ep_seed1/pre_embedding_3h` (198-d pre-3h)
- **cache:** `results/cache/temporal_flow_causal/Small-HI`
- **primary comparison:** Arm D vs Arm B (ΔAUPRC = +0.1995)
- **no SSL retraining:** True

## Four-arm test metrics

| arm | groups | dim | AUROC | AUPRC | F1 | P@100 | R@100 | lift@100 |
|-----|--------|----:|------:|------:|---:|------:|------:|---------:|
| A_embedding | pre_embedding_3h | 198 | 0.9439 | 0.1049 | 0.1117 | 0.5300 | 0.0329 | 283.93 |
| B_embedding_raw | pre_embedding_3h, raw | 202 | 0.9021 | 0.0594 | 0.0548 | 0.2800 | 0.0174 | 150.00 |
| D_embedding_raw_temporal_flow | pre_embedding_3h, raw, temporal_flow_causal | 207 | 0.9666 | 0.2589 | 0.2654 | 0.7100 | 0.0441 | 380.36 |

## Primary deltas (D − B)

- ΔAUPRC: **+0.1995**
- ΔF1: +0.2106
- ΔP@100: +0.4300
- ΔR@100: +0.0267
- Δlift@100: +230.36

Conservative read: single checkpoint; downstream probe only.
