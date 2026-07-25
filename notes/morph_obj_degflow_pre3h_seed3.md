# Temporal flow causal ablation — Small-HI (`hi_morph_obj_degflow_gin_emlps_tds_asym_proj_8192neg_queue0_accum4_20ep_seed3`)

- **embedding:** `embeddings/hi_morph_obj_degflow_gin_emlps_tds_asym_proj_8192neg_queue0_accum4_20ep_seed3/pre_embedding_3h` (198-d pre-3h)
- **cache:** `results/cache/temporal_flow_causal/Small-HI`
- **primary comparison:** Arm D vs Arm B (ΔAUPRC = +0.1163)
- **no SSL retraining:** True

## Four-arm test metrics

| arm | groups | dim | AUROC | AUPRC | F1 | P@100 | R@100 | lift@100 |
|-----|--------|----:|------:|------:|---:|------:|------:|---------:|
| A_embedding | pre_embedding_3h | 198 | 0.9195 | 0.0828 | 0.1972 | 0.0000 | 0.0000 | 0.00 |
| B_embedding_raw | pre_embedding_3h, raw | 202 | 0.9287 | 0.1185 | 0.1797 | 0.0300 | 0.0019 | 16.07 |
| D_embedding_raw_temporal_flow | pre_embedding_3h, raw, temporal_flow_causal | 207 | 0.9513 | 0.2348 | 0.1552 | 0.0600 | 0.0037 | 32.14 |

## Primary deltas (D − B)

- ΔAUPRC: **+0.1163**
- ΔF1: -0.0246
- ΔP@100: +0.0300
- ΔR@100: +0.0019
- Δlift@100: +16.07

Conservative read: single checkpoint; downstream probe only.
