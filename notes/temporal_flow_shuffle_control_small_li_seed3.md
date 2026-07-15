# Temporal flow causal ablation — Small-LI (`small_li_gin_emlps_tds_asym_proj_8192neg_queue0_20ep_seed3`)

- **embedding:** `embeddings/small_li_gin_emlps_tds_asym_proj_8192neg_queue0_20ep_seed3/pre_embedding_3h` (198-d pre-3h)
- **cache:** `results/cache/temporal_flow_causal/Small-LI`
- **primary comparison:** Arm D vs Arm B (ΔAUPRC = +nan)
- **no SSL retraining:** True

## Four-arm test metrics

| arm | groups | dim | AUROC | AUPRC | F1 | P@100 | R@100 | lift@100 |
|-----|--------|----:|------:|------:|---:|------:|------:|---------:|
| D_embedding_raw_temporal_flow | pre_embedding_3h, raw, temporal_flow_causal | 207 | 0.9343 | 0.0781 | 0.0527 | 0.4300 | 0.0536 | 629.03 |

## Primary deltas (D − B)

- ΔAUPRC: **+nan**
- ΔF1: +nan
- ΔP@100: +nan
- ΔR@100: +nan
- Δlift@100: +nan

Conservative read: single checkpoint; downstream probe only.
