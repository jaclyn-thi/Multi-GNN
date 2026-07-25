# Temporal flow causal ablation — Small-HI (`hi_tf_aux_tf_reg_w0.05_gin_emlps_tds_asym_proj_8192neg_queue0_accum4_20ep_seed3`)

- **embedding:** `embeddings/hi_tf_aux_tf_reg_w0.05_gin_emlps_tds_asym_proj_8192neg_queue0_accum4_20ep_seed3/pre_embedding_3h` (198-d pre-3h)
- **cache:** `results/cache/temporal_flow_causal/Small-HI`
- **primary comparison:** Arm D vs Arm B (ΔAUPRC = +0.1606)
- **no SSL retraining:** True

## Four-arm test metrics

| arm | groups | dim | AUROC | AUPRC | F1 | P@100 | R@100 | lift@100 |
|-----|--------|----:|------:|------:|---:|------:|------:|---------:|
| A_embedding | pre_embedding_3h | 198 | 0.9608 | 0.2207 | 0.2905 | 0.7000 | 0.0435 | 375.01 |
| B_embedding_raw | pre_embedding_3h, raw | 202 | 0.9637 | 0.2560 | 0.3218 | 0.7700 | 0.0478 | 412.51 |
| D_embedding_raw_temporal_flow | pre_embedding_3h, raw, temporal_flow_causal | 207 | 0.9764 | 0.4166 | 0.4430 | 0.8800 | 0.0546 | 471.45 |

## Primary deltas (D − B)

- ΔAUPRC: **+0.1606**
- ΔF1: +0.1212
- ΔP@100: +0.1100
- ΔR@100: +0.0068
- Δlift@100: +58.93

Conservative read: single checkpoint; downstream probe only.
