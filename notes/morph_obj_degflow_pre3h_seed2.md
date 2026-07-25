# Temporal flow causal ablation — Small-HI (`hi_morph_obj_degflow_gin_emlps_tds_asym_proj_8192neg_queue0_accum4_20ep_seed2`)

- **embedding:** `embeddings/hi_morph_obj_degflow_gin_emlps_tds_asym_proj_8192neg_queue0_accum4_20ep_seed2/pre_embedding_3h` (198-d pre-3h)
- **cache:** `results/cache/temporal_flow_causal/Small-HI`
- **primary comparison:** Arm D vs Arm B (ΔAUPRC = +0.0764)
- **no SSL retraining:** True

## Four-arm test metrics

| arm | groups | dim | AUROC | AUPRC | F1 | P@100 | R@100 | lift@100 |
|-----|--------|----:|------:|------:|---:|------:|------:|---------:|
| A_embedding | pre_embedding_3h | 198 | 0.9378 | 0.0756 | 0.1803 | 0.0800 | 0.0050 | 42.86 |
| B_embedding_raw | pre_embedding_3h, raw | 202 | 0.9010 | 0.0362 | 0.0200 | 0.0500 | 0.0031 | 26.79 |
| D_embedding_raw_temporal_flow | pre_embedding_3h, raw, temporal_flow_causal | 207 | 0.9550 | 0.1126 | 0.0325 | 0.0300 | 0.0019 | 16.07 |

## Primary deltas (D − B)

- ΔAUPRC: **+0.0764**
- ΔF1: +0.0125
- ΔP@100: -0.0200
- ΔR@100: -0.0012
- Δlift@100: -10.71

Conservative read: single checkpoint; downstream probe only.
