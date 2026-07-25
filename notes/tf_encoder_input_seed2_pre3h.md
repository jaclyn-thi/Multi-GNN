# Temporal flow causal ablation — Small-HI (`hi_contrastive_tf_encoder_input_seed2`)

- **embedding:** `embeddings/hi_contrastive_tf_encoder_input_seed2/pre_embedding_3h` (198-d pre-3h)
- **cache:** `results/cache/temporal_flow_causal/Small-HI`
- **primary comparison:** Arm D vs Arm B (ΔAUPRC = +0.0054)
- **no SSL retraining:** True

## Four-arm test metrics

| arm | groups | dim | AUROC | AUPRC | F1 | P@100 | R@100 | lift@100 |
|-----|--------|----:|------:|------:|---:|------:|------:|---------:|
| A_embedding | pre_embedding_3h | 198 | 0.9693 | 0.4750 | 0.5006 | 0.8200 | 0.0509 | 439.30 |
| B_embedding_raw | pre_embedding_3h, raw | 202 | 0.9696 | 0.4925 | 0.4298 | 0.7800 | 0.0484 | 417.87 |
| D_embedding_raw_temporal_flow | pre_embedding_3h, raw, temporal_flow_causal | 207 | 0.9710 | 0.4979 | 0.5417 | 0.8400 | 0.0521 | 450.01 |

## Primary deltas (D − B)

- ΔAUPRC: **+0.0054**
- ΔF1: +0.1119
- ΔP@100: +0.0600
- ΔR@100: +0.0037
- Δlift@100: +32.14

Conservative read: single checkpoint; downstream probe only.
