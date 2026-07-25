# Temporal flow causal ablation — Small-HI (`hi_contrastive_gin_emlps_tds_proj_asym_8192neg_queue0_accum4_20ep`)

- **embedding:** `embeddings/hi_contrastive_gin_emlps_tds_proj_asym_8192neg_queue0_accum4_20ep/pre_embedding_3h` (198-d pre-3h)
- **cache:** `results/cache/temporal_flow_causal/Small-HI`
- **primary comparison:** Arm D vs Arm B (ΔAUPRC = +0.2224)
- **no SSL retraining:** True

## Four-arm test metrics

| arm | groups | dim | AUROC | AUPRC | F1 | P@100 | R@100 | lift@100 |
|-----|--------|----:|------:|------:|---:|------:|------:|---------:|
| A_embedding | pre_embedding_3h | 198 | 0.9608 | 0.1888 | 0.2504 | 0.6600 | 0.0410 | 353.58 |
| B_embedding_raw | pre_embedding_3h, raw | 202 | 0.9617 | 0.2113 | 0.2872 | 0.6700 | 0.0416 | 358.93 |
| D_embedding_raw_temporal_flow | pre_embedding_3h, raw, temporal_flow_causal | 207 | 0.9786 | 0.4337 | 0.4482 | 0.9400 | 0.0583 | 503.58 |

## Primary deltas (D − B)

- ΔAUPRC: **+0.2224**
- ΔF1: +0.1611
- ΔP@100: +0.2700
- ΔR@100: +0.0168
- Δlift@100: +144.65

Conservative read: single checkpoint; downstream probe only.
