# Temporal flow causal ablation — Small-HI (`hi_tf_aux_tf_reg_w0.10_gin_emlps_tds_asym_proj_8192neg_queue0_accum4_20ep_seed1`)

- **embedding:** `embeddings/hi_tf_aux_tf_reg_w0.10_gin_emlps_tds_asym_proj_8192neg_queue0_accum4_20ep_seed1/pre_embedding_3h` (198-d pre-3h)
- **cache:** `results/cache/temporal_flow_causal/Small-HI`
- **primary comparison:** Arm D vs Arm B (ΔAUPRC = +0.0397)
- **no SSL retraining:** True

## Four-arm test metrics

| arm | groups | dim | AUROC | AUPRC | F1 | P@100 | R@100 | lift@100 |
|-----|--------|----:|------:|------:|---:|------:|------:|---------:|
| A_embedding | pre_embedding_3h | 198 | 0.9700 | 0.3226 | 0.3239 | 0.6900 | 0.0428 | 369.65 |
| B_embedding_raw | pre_embedding_3h, raw | 202 | 0.9716 | 0.3404 | 0.2970 | 0.7200 | 0.0447 | 385.72 |
| D_embedding_raw_temporal_flow | pre_embedding_3h, raw, temporal_flow_causal | 207 | 0.9774 | 0.3801 | 0.3033 | 0.7600 | 0.0472 | 407.15 |

## Primary deltas (D − B)

- ΔAUPRC: **+0.0397**
- ΔF1: +0.0063
- ΔP@100: +0.0400
- ΔR@100: +0.0025
- Δlift@100: +21.43

Conservative read: single checkpoint; downstream probe only.
