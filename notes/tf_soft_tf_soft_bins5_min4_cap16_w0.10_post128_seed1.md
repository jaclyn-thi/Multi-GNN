# Temporal flow causal ablation — Small-HI (`hi_tf_soft_tf_soft_bins5_min4_cap16_w0.10_optv2_gin_emlps_tds_asym_proj_8192neg_queue0_accum4_20ep_seed1`)

- **embedding:** `embeddings/hi_tf_soft_tf_soft_bins5_min4_cap16_w0.10_optv2_gin_emlps_tds_asym_proj_8192neg_queue0_accum4_20ep_seed1` (128-d pre-3h)
- **cache:** `results/cache/temporal_flow_causal/Small-HI`
- **primary comparison:** Arm D vs Arm B (ΔAUPRC = +nan)
- **no SSL retraining:** True

## Four-arm test metrics

| arm | groups | dim | AUROC | AUPRC | F1 | P@100 | R@100 | lift@100 |
|-----|--------|----:|------:|------:|---:|------:|------:|---------:|
| A_embedding | pre_embedding_3h | 128 | 0.9032 | 0.0635 | 0.1380 | 0.2900 | 0.0180 | 155.36 |
| B_embedding_raw | pre_embedding_3h, raw | 132 | 0.9350 | 0.1765 | 0.0982 | 0.7500 | 0.0466 | 401.79 |

## Primary deltas (D − B)

- ΔAUPRC: **+nan**
- ΔF1: +nan
- ΔP@100: +nan
- ΔR@100: +nan
- Δlift@100: +nan

Conservative read: single checkpoint; downstream probe only.
