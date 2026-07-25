# Temporal flow causal ablation — Small-HI (`hi_tf_soft_tf_soft_strict_bins10_min5_cap4_w0.01_optv2_gin_emlps_tds_asym_proj_8192neg_queue0_accum4_20ep_seed1`)

- **embedding:** `embeddings/hi_tf_soft_tf_soft_strict_bins10_min5_cap4_w0.01_optv2_gin_emlps_tds_asym_proj_8192neg_queue0_accum4_20ep_seed1` (128-d pre-3h)
- **cache:** `results/cache/temporal_flow_causal/Small-HI`
- **primary comparison:** Arm D vs Arm B (ΔAUPRC = +nan)
- **no SSL retraining:** True

## Four-arm test metrics

| arm | groups | dim | AUROC | AUPRC | F1 | P@100 | R@100 | lift@100 |
|-----|--------|----:|------:|------:|---:|------:|------:|---------:|
| A_embedding | pre_embedding_3h | 128 | 0.7072 | 0.0034 | 0.0022 | 0.0100 | 0.0006 | 5.36 |
| B_embedding_raw | pre_embedding_3h, raw | 132 | 0.7848 | 0.0212 | 0.0087 | 0.3100 | 0.0192 | 166.07 |

## Primary deltas (D − B)

- ΔAUPRC: **+nan**
- ΔF1: +nan
- ΔP@100: +nan
- ΔR@100: +nan
- Δlift@100: +nan

Conservative read: single checkpoint; downstream probe only.
