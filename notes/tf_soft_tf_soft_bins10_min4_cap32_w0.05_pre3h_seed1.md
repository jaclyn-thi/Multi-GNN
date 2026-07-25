# Temporal flow causal ablation — Small-HI (`hi_tf_soft_tf_soft_bins10_min4_cap32_w0.05_optv2_gin_emlps_tds_asym_proj_8192neg_queue0_accum4_20ep_seed1`)

- **embedding:** `embeddings/hi_tf_soft_tf_soft_bins10_min4_cap32_w0.05_optv2_gin_emlps_tds_asym_proj_8192neg_queue0_accum4_20ep_seed1/pre_embedding_3h` (198-d pre-3h)
- **cache:** `results/cache/temporal_flow_causal/Small-HI`
- **primary comparison:** Arm D vs Arm B (ΔAUPRC = +0.0985)
- **no SSL retraining:** True

## Four-arm test metrics

| arm | groups | dim | AUROC | AUPRC | F1 | P@100 | R@100 | lift@100 |
|-----|--------|----:|------:|------:|---:|------:|------:|---------:|
| A_embedding | pre_embedding_3h | 198 | 0.9410 | 0.1367 | 0.2254 | 0.3700 | 0.0230 | 198.22 |
| B_embedding_raw | pre_embedding_3h, raw | 202 | 0.9070 | 0.0521 | 0.0130 | 0.1600 | 0.0099 | 85.72 |
| D_embedding_raw_temporal_flow | pre_embedding_3h, raw, temporal_flow_causal | 207 | 0.9580 | 0.1506 | 0.0603 | 0.1900 | 0.0118 | 101.79 |

## Primary deltas (D − B)

- ΔAUPRC: **+0.0985**
- ΔF1: +0.0473
- ΔP@100: +0.0300
- ΔR@100: +0.0019
- Δlift@100: +16.07

Conservative read: single checkpoint; downstream probe only.
