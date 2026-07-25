# Temporal flow causal ablation — Small-HI (`hi_tf_aux_tf_bins10_w0.10_gin_emlps_tds_asym_proj_8192neg_queue0_accum4_20ep_seed1`)

- **embedding:** `embeddings/hi_tf_aux_tf_bins10_w0.10_gin_emlps_tds_asym_proj_8192neg_queue0_accum4_20ep_seed1/pre_embedding_3h` (198-d pre-3h)
- **cache:** `results/cache/temporal_flow_causal/Small-HI`
- **primary comparison:** Arm D vs Arm B (ΔAUPRC = +0.1599)
- **no SSL retraining:** True

## Four-arm test metrics

| arm | groups | dim | AUROC | AUPRC | F1 | P@100 | R@100 | lift@100 |
|-----|--------|----:|------:|------:|---:|------:|------:|---------:|
| A_embedding | pre_embedding_3h | 198 | 0.9535 | 0.1639 | 0.2125 | 0.5000 | 0.0310 | 267.86 |
| B_embedding_raw | pre_embedding_3h, raw | 202 | 0.9573 | 0.2196 | 0.1481 | 0.5900 | 0.0366 | 316.08 |
| D_embedding_raw_temporal_flow | pre_embedding_3h, raw, temporal_flow_causal | 207 | 0.9713 | 0.3795 | 0.3023 | 0.7800 | 0.0484 | 417.86 |

## Primary deltas (D − B)

- ΔAUPRC: **+0.1599**
- ΔF1: +0.1542
- ΔP@100: +0.1900
- ΔR@100: +0.0118
- Δlift@100: +101.79

Conservative read: single checkpoint; downstream probe only.
