# Temporal flow causal ablation — Small-HI (`hi_morph_obj_degflow_gin_emlps_tds_asym_proj_8192neg_queue0_accum4_20ep_seed2`)

- **embedding:** `embeddings/hi_morph_obj_degflow_gin_emlps_tds_asym_proj_8192neg_queue0_accum4_20ep_seed2` (128-d pre-3h)
- **cache:** `results/cache/temporal_flow_causal/Small-HI`
- **primary comparison:** Arm D vs Arm B (ΔAUPRC = +nan)
- **no SSL retraining:** True

## Four-arm test metrics

| arm | groups | dim | AUROC | AUPRC | F1 | P@100 | R@100 | lift@100 |
|-----|--------|----:|------:|------:|---:|------:|------:|---------:|
| A_embedding | pre_embedding_3h | 128 | 0.9311 | 0.0823 | 0.1686 | 0.3000 | 0.0186 | 160.72 |
| B_embedding_raw | pre_embedding_3h, raw | 132 | 0.9372 | 0.1120 | 0.1975 | 0.4800 | 0.0298 | 257.16 |

## Primary deltas (D − B)

- ΔAUPRC: **+nan**
- ΔF1: +nan
- ΔP@100: +nan
- ΔR@100: +nan
- Δlift@100: +nan

Conservative read: single checkpoint; downstream probe only.
