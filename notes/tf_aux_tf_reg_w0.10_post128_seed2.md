# Temporal flow causal ablation — Small-HI (`hi_tf_aux_tf_reg_w0.10_gin_emlps_tds_asym_proj_8192neg_queue0_accum4_20ep_seed2`)

- **embedding:** `embeddings/hi_tf_aux_tf_reg_w0.10_gin_emlps_tds_asym_proj_8192neg_queue0_accum4_20ep_seed2` (128-d pre-3h)
- **cache:** `results/cache/temporal_flow_causal/Small-HI`
- **primary comparison:** Arm D vs Arm B (ΔAUPRC = +0.1637)
- **no SSL retraining:** True

## Four-arm test metrics

| arm | groups | dim | AUROC | AUPRC | F1 | P@100 | R@100 | lift@100 |
|-----|--------|----:|------:|------:|---:|------:|------:|---------:|
| A_embedding | pre_embedding_3h | 128 | 0.9689 | 0.1510 | 0.2456 | 0.3200 | 0.0199 | 171.44 |
| B_embedding_raw | pre_embedding_3h, raw | 132 | 0.9710 | 0.1430 | 0.2433 | 0.2700 | 0.0168 | 144.65 |
| D_embedding_raw_temporal_flow | pre_embedding_3h, raw, temporal_flow_causal | 137 | 0.9819 | 0.3068 | 0.3833 | 0.7300 | 0.0453 | 391.09 |

## Primary deltas (D − B)

- ΔAUPRC: **+0.1637**
- ΔF1: +0.1400
- ΔP@100: +0.4600
- ΔR@100: +0.0286
- Δlift@100: +246.44

Conservative read: single checkpoint; downstream probe only.
