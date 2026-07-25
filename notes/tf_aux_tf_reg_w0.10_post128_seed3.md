# Temporal flow causal ablation — Small-HI (`hi_tf_aux_tf_reg_w0.10_gin_emlps_tds_asym_proj_8192neg_queue0_accum4_20ep_seed3`)

- **embedding:** `embeddings/hi_tf_aux_tf_reg_w0.10_gin_emlps_tds_asym_proj_8192neg_queue0_accum4_20ep_seed3` (128-d pre-3h)
- **cache:** `results/cache/temporal_flow_causal/Small-HI`
- **primary comparison:** Arm D vs Arm B (ΔAUPRC = +0.1072)
- **no SSL retraining:** True

## Four-arm test metrics

| arm | groups | dim | AUROC | AUPRC | F1 | P@100 | R@100 | lift@100 |
|-----|--------|----:|------:|------:|---:|------:|------:|---------:|
| A_embedding | pre_embedding_3h | 128 | 0.9678 | 0.3643 | 0.4272 | 0.8100 | 0.0503 | 433.94 |
| B_embedding_raw | pre_embedding_3h, raw | 132 | 0.9713 | 0.4265 | 0.4364 | 0.8900 | 0.0552 | 476.80 |
| D_embedding_raw_temporal_flow | pre_embedding_3h, raw, temporal_flow_causal | 137 | 0.9797 | 0.5337 | 0.5566 | 0.9500 | 0.0590 | 508.95 |

## Primary deltas (D − B)

- ΔAUPRC: **+0.1072**
- ΔF1: +0.1201
- ΔP@100: +0.0600
- ΔR@100: +0.0037
- Δlift@100: +32.14

Conservative read: single checkpoint; downstream probe only.
