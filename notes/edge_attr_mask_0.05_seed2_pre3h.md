# Temporal flow causal ablation — Small-HI (`hi_contrastive_edge_attr_mask_0.05_seed2`)

- **embedding:** `embeddings/hi_contrastive_edge_attr_mask_0.05_seed2/pre_embedding_3h` (198-d pre-3h)
- **cache:** `results/cache/temporal_flow_causal/Small-HI`
- **primary comparison:** Arm D vs Arm B (ΔAUPRC = +0.1427)
- **no SSL retraining:** True

## Four-arm test metrics

| arm | groups | dim | AUROC | AUPRC | F1 | P@100 | R@100 | lift@100 |
|-----|--------|----:|------:|------:|---:|------:|------:|---------:|
| A_embedding | pre_embedding_3h | 198 | 0.9648 | 0.2382 | 0.3278 | 0.5500 | 0.0341 | 294.66 |
| B_embedding_raw | pre_embedding_3h, raw | 202 | 0.9724 | 0.3470 | 0.1361 | 0.7600 | 0.0472 | 407.16 |
| D_embedding_raw_temporal_flow | pre_embedding_3h, raw, temporal_flow_causal | 207 | 0.9827 | 0.4898 | 0.1777 | 0.8600 | 0.0534 | 460.74 |

## Primary deltas (D − B)

- ΔAUPRC: **+0.1427**
- ΔF1: +0.0416
- ΔP@100: +0.1000
- ΔR@100: +0.0062
- Δlift@100: +53.57

Conservative read: single checkpoint; downstream probe only.
