# Temporal flow causal ablation — Small-HI (`hi_tf_aux_tf_bins10_w0.10_gin_emlps_tds_asym_proj_8192neg_queue0_accum4_20ep_seed1`)

- **embedding:** `embeddings/hi_tf_aux_tf_bins10_w0.10_gin_emlps_tds_asym_proj_8192neg_queue0_accum4_20ep_seed1` (128-d pre-3h)
- **cache:** `results/cache/temporal_flow_causal/Small-HI`
- **primary comparison:** Arm D vs Arm B (ΔAUPRC = +0.1387)
- **no SSL retraining:** True

## Four-arm test metrics

| arm | groups | dim | AUROC | AUPRC | F1 | P@100 | R@100 | lift@100 |
|-----|--------|----:|------:|------:|---:|------:|------:|---------:|
| A_embedding | pre_embedding_3h | 128 | 0.9533 | 0.1738 | 0.2359 | 0.4900 | 0.0304 | 262.50 |
| B_embedding_raw | pre_embedding_3h, raw | 132 | 0.9592 | 0.2687 | 0.1335 | 0.6700 | 0.0416 | 358.93 |
| D_embedding_raw_temporal_flow | pre_embedding_3h, raw, temporal_flow_causal | 137 | 0.9733 | 0.4074 | 0.2006 | 0.8400 | 0.0521 | 450.01 |

## Primary deltas (D − B)

- ΔAUPRC: **+0.1387**
- ΔF1: +0.0671
- ΔP@100: +0.1700
- ΔR@100: +0.0106
- Δlift@100: +91.07

Conservative read: single checkpoint; downstream probe only.
