# Temporal flow causal ablation — Small-HI (`hi_tf_aux_tf_reg_w0.10_gin_emlps_tds_asym_proj_8192neg_queue0_accum4_20ep_seed2`)

- **embedding:** `embeddings/hi_tf_aux_tf_reg_w0.10_gin_emlps_tds_asym_proj_8192neg_queue0_accum4_20ep_seed2/pre_embedding_3h` (198-d pre-3h)
- **cache:** `results/cache/temporal_flow_causal/Small-HI`
- **primary comparison:** Arm D vs Arm B (ΔAUPRC = +0.1334)
- **no SSL retraining:** True

## Four-arm test metrics

| arm | groups | dim | AUROC | AUPRC | F1 | P@100 | R@100 | lift@100 |
|-----|--------|----:|------:|------:|---:|------:|------:|---------:|
| A_embedding | pre_embedding_3h | 198 | 0.9717 | 0.1896 | 0.2823 | 0.6100 | 0.0379 | 326.80 |
| B_embedding_raw | pre_embedding_3h, raw | 202 | 0.9733 | 0.2104 | 0.2955 | 0.6600 | 0.0410 | 353.59 |
| D_embedding_raw_temporal_flow | pre_embedding_3h, raw, temporal_flow_causal | 207 | 0.9814 | 0.3438 | 0.4029 | 0.7400 | 0.0459 | 396.45 |

## Primary deltas (D − B)

- ΔAUPRC: **+0.1334**
- ΔF1: +0.1074
- ΔP@100: +0.0800
- ΔR@100: +0.0050
- Δlift@100: +42.86

Conservative read: single checkpoint; downstream probe only.
