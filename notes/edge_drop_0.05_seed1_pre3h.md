# Temporal flow causal ablation — Small-HI (`hi_contrastive_edge_drop_0.05_seed1`)

- **embedding:** `embeddings/hi_contrastive_edge_drop_0.05_seed1/pre_embedding_3h` (198-d pre-3h)
- **cache:** `results/cache/temporal_flow_causal/Small-HI`
- **primary comparison:** Arm D vs Arm B (ΔAUPRC = +0.1744)
- **no SSL retraining:** True

## Four-arm test metrics

| arm | groups | dim | AUROC | AUPRC | F1 | P@100 | R@100 | lift@100 |
|-----|--------|----:|------:|------:|---:|------:|------:|---------:|
| A_embedding | pre_embedding_3h | 198 | 0.9443 | 0.1521 | 0.2451 | 0.4900 | 0.0304 | 262.50 |
| B_embedding_raw | pre_embedding_3h, raw | 202 | 0.9602 | 0.2826 | 0.0753 | 0.7600 | 0.0472 | 407.15 |
| D_embedding_raw_temporal_flow | pre_embedding_3h, raw, temporal_flow_causal | 207 | 0.9770 | 0.4570 | 0.1833 | 0.9000 | 0.0559 | 482.15 |

## Primary deltas (D − B)

- ΔAUPRC: **+0.1744**
- ΔF1: +0.1080
- ΔP@100: +0.1400
- ΔR@100: +0.0087
- Δlift@100: +75.00

Conservative read: single checkpoint; downstream probe only.
