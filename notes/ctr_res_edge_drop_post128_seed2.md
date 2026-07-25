# Temporal flow causal ablation — Small-HI (`hi_contrastive_edge_drop_0.05_seed2`)

- **embedding:** `embeddings/hi_contrastive_edge_drop_0.05_seed2` (128-d pre-3h)
- **cache:** `results/cache/temporal_flow_causal/Small-HI`
- **primary comparison:** Arm D vs Arm B (ΔAUPRC = +0.2136)
- **no SSL retraining:** True

## Four-arm test metrics

| arm | groups | dim | AUROC | AUPRC | F1 | P@100 | R@100 | lift@100 |
|-----|--------|----:|------:|------:|---:|------:|------:|---------:|
| A_embedding | pre_embedding_3h | 128 | 0.9321 | 0.1459 | 0.2350 | 0.5500 | 0.0341 | 294.66 |
| B_embedding_raw | pre_embedding_3h, raw | 132 | 0.9505 | 0.2408 | 0.2207 | 0.6500 | 0.0403 | 348.23 |
| D_embedding_raw_temporal_flow | pre_embedding_3h, raw, temporal_flow_causal | 137 | 0.9793 | 0.4544 | 0.3549 | 0.7700 | 0.0478 | 412.52 |

## Primary deltas (D − B)

- ΔAUPRC: **+0.2136**
- ΔF1: +0.1342
- ΔP@100: +0.1200
- ΔR@100: +0.0074
- Δlift@100: +64.29

Conservative read: single checkpoint; downstream probe only.
