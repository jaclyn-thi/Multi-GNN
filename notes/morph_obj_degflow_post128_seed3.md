# Temporal flow causal ablation — Small-HI (`hi_morph_obj_degflow_gin_emlps_tds_asym_proj_8192neg_queue0_accum4_20ep_seed3`)

- **embedding:** `embeddings/hi_morph_obj_degflow_gin_emlps_tds_asym_proj_8192neg_queue0_accum4_20ep_seed3` (128-d pre-3h)
- **cache:** `results/cache/temporal_flow_causal/Small-HI`
- **primary comparison:** Arm D vs Arm B (ΔAUPRC = +nan)
- **no SSL retraining:** True

## Four-arm test metrics

| arm | groups | dim | AUROC | AUPRC | F1 | P@100 | R@100 | lift@100 |
|-----|--------|----:|------:|------:|---:|------:|------:|---------:|
| A_embedding | pre_embedding_3h | 128 | 0.9259 | 0.1072 | 0.1899 | 0.2400 | 0.0149 | 128.58 |
| B_embedding_raw | pre_embedding_3h, raw | 132 | 0.9353 | 0.2197 | 0.1583 | 0.7100 | 0.0441 | 380.37 |

## Primary deltas (D − B)

- ΔAUPRC: **+nan**
- ΔF1: +nan
- ΔP@100: +nan
- ΔR@100: +nan
- Δlift@100: +nan

Conservative read: single checkpoint; downstream probe only.
