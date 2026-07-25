# Temporal flow causal ablation — Small-HI (`hi_contrastive_large_bs_16384_seed2`)

- **embedding:** `embeddings/hi_contrastive_large_bs_16384_seed2` (128-d pre-3h)
- **cache:** `results/cache/temporal_flow_causal/Small-HI`
- **primary comparison:** Arm D vs Arm B (ΔAUPRC = +0.2870)
- **no SSL retraining:** True

## Four-arm test metrics

| arm | groups | dim | AUROC | AUPRC | F1 | P@100 | R@100 | lift@100 |
|-----|--------|----:|------:|------:|---:|------:|------:|---------:|
| A_embedding | pre_embedding_3h | 128 | 0.9301 | 0.1101 | 0.1932 | 0.5300 | 0.0329 | 283.94 |
| B_embedding_raw | pre_embedding_3h, raw | 132 | 0.9379 | 0.1211 | 0.2084 | 0.5100 | 0.0317 | 273.23 |
| D_embedding_raw_temporal_flow | pre_embedding_3h, raw, temporal_flow_causal | 137 | 0.9719 | 0.4081 | 0.4305 | 0.8300 | 0.0515 | 444.67 |

## Primary deltas (D − B)

- ΔAUPRC: **+0.2870**
- ΔF1: +0.2221
- ΔP@100: +0.3200
- ΔR@100: +0.0199
- Δlift@100: +171.44

Conservative read: single checkpoint; downstream probe only.
