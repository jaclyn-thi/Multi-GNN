# Temporal flow causal ablation — Small-HI (`gin_emlps_tds_asym_proj_8192neg_queue0_40ep_seed2`)

- **embedding:** `embeddings/gin_emlps_tds_asym_proj_8192neg_queue0_40ep_seed2/pre_embedding_3h` (198-d pre-3h)
- **cache:** `results/cache/temporal_flow_causal/Small-HI`
- **primary comparison:** Arm D vs Arm B (ΔAUPRC = +nan)
- **no SSL retraining:** True

## Four-arm test metrics

| arm | groups | dim | AUROC | AUPRC | F1 | P@100 | R@100 | lift@100 |
|-----|--------|----:|------:|------:|---:|------:|------:|---------:|
| D_embedding_raw_temporal_flow | pre_embedding_3h, raw, temporal_flow_causal | 207 | 0.9604 | 0.3203 | 0.3435 | 0.8300 | 0.0515 | 444.65 |

## Primary deltas (D − B)

- ΔAUPRC: **+nan**
- ΔF1: +nan
- ΔP@100: +nan
- ΔR@100: +nan
- Δlift@100: +nan

Conservative read: single checkpoint; downstream probe only.
