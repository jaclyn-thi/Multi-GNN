# Temporal flow causal ablation — Small-HI (`hi_tf_aux_tf_reg_w0.05_gin_emlps_tds_asym_proj_8192neg_queue0_accum4_20ep_seed2`)

- **embedding:** `embeddings/hi_tf_aux_tf_reg_w0.05_gin_emlps_tds_asym_proj_8192neg_queue0_accum4_20ep_seed2` (128-d pre-3h)
- **cache:** `results/cache/temporal_flow_causal/Small-HI`
- **primary comparison:** Arm D vs Arm B (ΔAUPRC = +0.1231)
- **no SSL retraining:** True

## Four-arm test metrics

| arm | groups | dim | AUROC | AUPRC | F1 | P@100 | R@100 | lift@100 |
|-----|--------|----:|------:|------:|---:|------:|------:|---------:|
| A_embedding | pre_embedding_3h | 128 | 0.9547 | 0.2364 | 0.3045 | 0.7200 | 0.0447 | 385.73 |
| B_embedding_raw | pre_embedding_3h, raw | 132 | 0.9713 | 0.3974 | 0.0487 | 0.9300 | 0.0577 | 498.24 |
| D_embedding_raw_temporal_flow | pre_embedding_3h, raw, temporal_flow_causal | 137 | 0.9803 | 0.5205 | 0.1856 | 0.9800 | 0.0608 | 525.03 |

## Primary deltas (D − B)

- ΔAUPRC: **+0.1231**
- ΔF1: +0.1368
- ΔP@100: +0.0500
- ΔR@100: +0.0031
- Δlift@100: +26.79

Conservative read: single checkpoint; downstream probe only.
