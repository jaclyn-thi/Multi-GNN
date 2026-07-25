# Temporal flow causal ablation — Small-HI (`hi_contrastive_large_bs_16384_seed2`)

- **embedding:** `embeddings/hi_contrastive_large_bs_16384_seed2/pre_embedding_3h` (198-d pre-3h)
- **cache:** `results/cache/temporal_flow_causal/Small-HI`
- **primary comparison:** Arm D vs Arm B (ΔAUPRC = +0.1961)
- **no SSL retraining:** True

## Four-arm test metrics

| arm | groups | dim | AUROC | AUPRC | F1 | P@100 | R@100 | lift@100 |
|-----|--------|----:|------:|------:|---:|------:|------:|---------:|
| A_embedding | pre_embedding_3h | 198 | 0.9438 | 0.1386 | 0.2369 | 0.5100 | 0.0317 | 273.23 |
| B_embedding_raw | pre_embedding_3h, raw | 202 | 0.9392 | 0.1116 | 0.1811 | 0.4000 | 0.0248 | 214.30 |
| D_embedding_raw_temporal_flow | pre_embedding_3h, raw, temporal_flow_causal | 207 | 0.9716 | 0.3077 | 0.4080 | 0.5800 | 0.0360 | 310.73 |

## Primary deltas (D − B)

- ΔAUPRC: **+0.1961**
- ΔF1: +0.2269
- ΔP@100: +0.1800
- ΔR@100: +0.0112
- Δlift@100: +96.43

Conservative read: single checkpoint; downstream probe only.
