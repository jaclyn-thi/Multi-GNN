# Temporal flow causal ablation — Small-HI (`hi_tf_soft_tf_soft_strict_bins10_min5_cap4_w0.01_optv2_gin_emlps_tds_asym_proj_8192neg_queue0_accum4_20ep_seed1`)

- **embedding:** `embeddings/hi_tf_soft_tf_soft_strict_bins10_min5_cap4_w0.01_optv2_gin_emlps_tds_asym_proj_8192neg_queue0_accum4_20ep_seed1/pre_embedding_3h` (198-d pre-3h)
- **cache:** `results/cache/temporal_flow_causal/Small-HI`
- **primary comparison:** Arm D vs Arm B (ΔAUPRC = +0.1464)
- **no SSL retraining:** True

## Four-arm test metrics

| arm | groups | dim | AUROC | AUPRC | F1 | P@100 | R@100 | lift@100 |
|-----|--------|----:|------:|------:|---:|------:|------:|---------:|
| A_embedding | pre_embedding_3h | 198 | 0.8430 | 0.0077 | 0.0221 | 0.0000 | 0.0000 | 0.00 |
| B_embedding_raw | pre_embedding_3h, raw | 202 | 0.8701 | 0.0104 | 0.0281 | 0.0000 | 0.0000 | 0.00 |
| D_embedding_raw_temporal_flow | pre_embedding_3h, raw, temporal_flow_causal | 207 | 0.9577 | 0.1568 | 0.1179 | 0.7700 | 0.0478 | 412.51 |

## Primary deltas (D − B)

- ΔAUPRC: **+0.1464**
- ΔF1: +0.0898
- ΔP@100: +0.7700
- ΔR@100: +0.0478
- Δlift@100: +412.51

Conservative read: single checkpoint; downstream probe only.
