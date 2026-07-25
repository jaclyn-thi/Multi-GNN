# Temporal flow causal ablation — Small-HI (`hi_contrastive_gin_emlps_tds_proj_asym_8192neg_queue0_accum4_20ep`)

- **embedding:** `embeddings/hi_contrastive_gin_emlps_tds_proj_asym_8192neg_queue0_accum4_20ep` (128-d pre-3h)
- **cache:** `results/cache/temporal_flow_causal/Small-HI`
- **primary comparison:** Arm D vs Arm B (ΔAUPRC = +nan)
- **no SSL retraining:** True

## Four-arm test metrics

| arm | groups | dim | AUROC | AUPRC | F1 | P@100 | R@100 | lift@100 |
|-----|--------|----:|------:|------:|---:|------:|------:|---------:|
| A_embedding | pre_embedding_3h | 128 | 0.9441 | 0.2131 | 0.2646 | 0.8500 | 0.0528 | 455.36 |
| B_embedding_raw | pre_embedding_3h, raw | 132 | 0.9488 | 0.2436 | 0.2692 | 0.8400 | 0.0521 | 450.01 |

## Primary deltas (D − B)

- ΔAUPRC: **+nan**
- ΔF1: +nan
- ΔP@100: +nan
- ΔR@100: +nan
- Δlift@100: +nan

Conservative read: single checkpoint; downstream probe only.
