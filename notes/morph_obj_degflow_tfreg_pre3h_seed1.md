# Temporal flow causal ablation — Small-HI (`hi_morph_obj_degflow_tfreg_w0.05_gin_emlps_tds_asym_proj_8192neg_queue0_accum4_20ep_seed1`)

- **embedding:** `embeddings/hi_morph_obj_degflow_tfreg_w0.05_gin_emlps_tds_asym_proj_8192neg_queue0_accum4_20ep_seed1/pre_embedding_3h` (198-d pre-3h)
- **cache:** `results/cache/temporal_flow_causal/Small-HI`
- **primary comparison:** Arm D vs Arm B (ΔAUPRC = +0.1315)
- **no SSL retraining:** True

## Four-arm test metrics

| arm | groups | dim | AUROC | AUPRC | F1 | P@100 | R@100 | lift@100 |
|-----|--------|----:|------:|------:|---:|------:|------:|---------:|
| A_embedding | pre_embedding_3h | 198 | 0.9295 | 0.1275 | 0.2030 | 0.4600 | 0.0286 | 246.43 |
| B_embedding_raw | pre_embedding_3h, raw | 202 | 0.9380 | 0.1819 | 0.1820 | 0.5900 | 0.0366 | 316.08 |
| D_embedding_raw_temporal_flow | pre_embedding_3h, raw, temporal_flow_causal | 207 | 0.9621 | 0.3134 | 0.3878 | 0.7100 | 0.0441 | 380.36 |

## Primary deltas (D − B)

- ΔAUPRC: **+0.1315**
- ΔF1: +0.2058
- ΔP@100: +0.1200
- ΔR@100: +0.0074
- Δlift@100: +64.29

Conservative read: single checkpoint; downstream probe only.
