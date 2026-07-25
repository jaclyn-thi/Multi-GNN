# Temporal flow causal ablation — Small-HI (`hi_morph_obj_clustering_gin_emlps_tds_asym_proj_8192neg_queue0_accum4_20ep_seed1`)

- **embedding:** `embeddings/hi_morph_obj_clustering_gin_emlps_tds_asym_proj_8192neg_queue0_accum4_20ep_seed1/pre_embedding_3h` (198-d pre-3h)
- **cache:** `results/cache/temporal_flow_causal/Small-HI`
- **primary comparison:** Arm D vs Arm B (ΔAUPRC = +0.1202)
- **no SSL retraining:** True

## Four-arm test metrics

| arm | groups | dim | AUROC | AUPRC | F1 | P@100 | R@100 | lift@100 |
|-----|--------|----:|------:|------:|---:|------:|------:|---------:|
| A_embedding | pre_embedding_3h | 198 | 0.9487 | 0.0861 | 0.1618 | 0.1400 | 0.0087 | 75.00 |
| B_embedding_raw | pre_embedding_3h, raw | 202 | 0.9605 | 0.1494 | 0.0825 | 0.3400 | 0.0211 | 182.15 |
| D_embedding_raw_temporal_flow | pre_embedding_3h, raw, temporal_flow_causal | 207 | 0.9759 | 0.2696 | 0.2015 | 0.6000 | 0.0372 | 321.43 |

## Primary deltas (D − B)

- ΔAUPRC: **+0.1202**
- ΔF1: +0.1190
- ΔP@100: +0.2600
- ΔR@100: +0.0161
- Δlift@100: +139.29

Conservative read: single checkpoint; downstream probe only.
