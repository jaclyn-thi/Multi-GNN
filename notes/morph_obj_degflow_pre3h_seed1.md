# Temporal flow causal ablation — Small-HI (`hi_morph_obj_degflow_gin_emlps_tds_asym_proj_8192neg_queue0_accum4_20ep_seed1`)

- **embedding:** `embeddings/hi_morph_obj_degflow_gin_emlps_tds_asym_proj_8192neg_queue0_accum4_20ep_seed1/pre_embedding_3h` (198-d pre-3h)
- **cache:** `results/cache/temporal_flow_causal/Small-HI`
- **primary comparison:** Arm D vs Arm B (ΔAUPRC = +0.1021)
- **no SSL retraining:** True

## Four-arm test metrics

| arm | groups | dim | AUROC | AUPRC | F1 | P@100 | R@100 | lift@100 |
|-----|--------|----:|------:|------:|---:|------:|------:|---------:|
| A_embedding | pre_embedding_3h | 198 | 0.9331 | 0.2828 | 0.3175 | 0.8500 | 0.0528 | 455.36 |
| B_embedding_raw | pre_embedding_3h, raw | 202 | 0.9461 | 0.3719 | 0.0307 | 0.7400 | 0.0459 | 396.44 |
| D_embedding_raw_temporal_flow | pre_embedding_3h, raw, temporal_flow_causal | 207 | 0.9580 | 0.4740 | 0.0468 | 0.8200 | 0.0509 | 439.29 |

## Primary deltas (D − B)

- ΔAUPRC: **+0.1021**
- ΔF1: +0.0160
- ΔP@100: +0.0800
- ΔR@100: +0.0050
- Δlift@100: +42.86

Conservative read: single checkpoint; downstream probe only.
