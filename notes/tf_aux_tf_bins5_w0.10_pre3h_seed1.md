# Temporal flow causal ablation — Small-HI (`hi_tf_aux_tf_bins5_w0.10_gin_emlps_tds_asym_proj_8192neg_queue0_accum4_20ep_seed1`)

- **embedding:** `embeddings/hi_tf_aux_tf_bins5_w0.10_gin_emlps_tds_asym_proj_8192neg_queue0_accum4_20ep_seed1/pre_embedding_3h` (198-d pre-3h)
- **cache:** `results/cache/temporal_flow_causal/Small-HI`
- **primary comparison:** Arm D vs Arm B (ΔAUPRC = +0.1261)
- **no SSL retraining:** True

## Four-arm test metrics

| arm | groups | dim | AUROC | AUPRC | F1 | P@100 | R@100 | lift@100 |
|-----|--------|----:|------:|------:|---:|------:|------:|---------:|
| A_embedding | pre_embedding_3h | 198 | 0.9604 | 0.2174 | 0.2904 | 0.7700 | 0.0478 | 412.51 |
| B_embedding_raw | pre_embedding_3h, raw | 202 | 0.9703 | 0.3497 | 0.1843 | 0.8500 | 0.0528 | 455.36 |
| D_embedding_raw_temporal_flow | pre_embedding_3h, raw, temporal_flow_causal | 207 | 0.9776 | 0.4758 | 0.2314 | 0.9000 | 0.0559 | 482.15 |

## Primary deltas (D − B)

- ΔAUPRC: **+0.1261**
- ΔF1: +0.0471
- ΔP@100: +0.0500
- ΔR@100: +0.0031
- Δlift@100: +26.79

Conservative read: single checkpoint; downstream probe only.
