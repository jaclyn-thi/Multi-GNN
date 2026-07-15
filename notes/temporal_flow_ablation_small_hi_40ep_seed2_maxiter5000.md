# Temporal flow causal ablation — Small-HI (`gin_emlps_tds_asym_proj_8192neg_queue0_40ep_seed2`)

- **embedding:** `embeddings/gin_emlps_tds_asym_proj_8192neg_queue0_40ep_seed2/pre_embedding_3h` (198-d pre-3h)
- **cache:** `results/cache/temporal_flow_causal/Small-HI`
- **primary comparison:** Arm D vs Arm B (ΔAUPRC = +0.1801)
- **no SSL retraining:** True

## Four-arm test metrics

| arm | groups | dim | AUROC | AUPRC | F1 | P@100 | R@100 | lift@100 |
|-----|--------|----:|------:|------:|---:|------:|------:|---------:|
| A_embedding | pre_embedding_3h | 198 | 0.9581 | 0.2954 | 0.3363 | 0.8300 | 0.0515 | 444.65 |
| B_embedding_raw | pre_embedding_3h, raw | 202 | 0.9604 | 0.3205 | 0.3443 | 0.8300 | 0.0515 | 444.65 |
| C_embedding_temporal_flow | pre_embedding_3h, temporal_flow_causal | 203 | 0.9776 | 0.4735 | 0.4749 | 0.9100 | 0.0565 | 487.51 |
| D_embedding_raw_temporal_flow | pre_embedding_3h, raw, temporal_flow_causal | 207 | 0.9789 | 0.5006 | 0.4649 | 0.9400 | 0.0583 | 503.58 |

## Primary deltas (D − B)

- ΔAUPRC: **+0.1801**
- ΔF1: +0.1206
- ΔP@100: +0.1100
- ΔR@100: +0.0068
- Δlift@100: +58.93

Conservative read: single checkpoint; downstream probe only.
