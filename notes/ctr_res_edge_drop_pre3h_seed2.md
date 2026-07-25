# Temporal flow causal ablation — Small-HI (`hi_contrastive_edge_drop_0.05_seed2`)

- **embedding:** `embeddings/hi_contrastive_edge_drop_0.05_seed2/pre_embedding_3h` (198-d pre-3h)
- **cache:** `results/cache/temporal_flow_causal/Small-HI`
- **primary comparison:** Arm D vs Arm B (ΔAUPRC = +0.1439)
- **no SSL retraining:** True

## Four-arm test metrics

| arm | groups | dim | AUROC | AUPRC | F1 | P@100 | R@100 | lift@100 |
|-----|--------|----:|------:|------:|---:|------:|------:|---------:|
| A_embedding | pre_embedding_3h | 198 | 0.9530 | 0.2870 | 0.3462 | 0.7900 | 0.0490 | 423.24 |
| B_embedding_raw | pre_embedding_3h, raw | 202 | 0.9605 | 0.3251 | 0.3445 | 0.8500 | 0.0528 | 455.38 |
| D_embedding_raw_temporal_flow | pre_embedding_3h, raw, temporal_flow_causal | 207 | 0.9782 | 0.4690 | 0.4488 | 0.8600 | 0.0534 | 460.74 |

## Primary deltas (D − B)

- ΔAUPRC: **+0.1439**
- ΔF1: +0.1043
- ΔP@100: +0.0100
- ΔR@100: +0.0006
- Δlift@100: +5.36

Conservative read: single checkpoint; downstream probe only.
