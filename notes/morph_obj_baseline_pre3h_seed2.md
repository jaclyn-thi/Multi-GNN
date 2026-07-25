# Temporal flow causal ablation — Small-HI (`hi_contrastive_gin_emlps_tds_seed2_proj_asym_8192neg_queue0_accum4_20ep`)

- **embedding:** `embeddings/hi_contrastive_gin_emlps_tds_seed2_proj_asym_8192neg_queue0_accum4_20ep/pre_embedding_3h` (198-d pre-3h)
- **cache:** `results/cache/temporal_flow_causal/Small-HI`
- **primary comparison:** Arm D vs Arm B (ΔAUPRC = +0.2386)
- **no SSL retraining:** True

## Four-arm test metrics

| arm | groups | dim | AUROC | AUPRC | F1 | P@100 | R@100 | lift@100 |
|-----|--------|----:|------:|------:|---:|------:|------:|---------:|
| A_embedding | pre_embedding_3h | 198 | 0.9484 | 0.2598 | 0.2947 | 0.7900 | 0.0490 | 423.24 |
| B_embedding_raw | pre_embedding_3h, raw | 202 | 0.9496 | 0.2725 | 0.3078 | 0.7900 | 0.0490 | 423.24 |
| D_embedding_raw_temporal_flow | pre_embedding_3h, raw, temporal_flow_causal | 207 | 0.9734 | 0.5111 | 0.5012 | 0.9300 | 0.0577 | 498.24 |

## Primary deltas (D − B)

- ΔAUPRC: **+0.2386**
- ΔF1: +0.1934
- ΔP@100: +0.1400
- ΔR@100: +0.0087
- Δlift@100: +75.00

Conservative read: single checkpoint; downstream probe only.
