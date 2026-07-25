# Temporal flow causal ablation — Small-HI (`hi_tf_aux_tf_reg_w0.10_gin_emlps_tds_asym_proj_8192neg_queue0_accum4_20ep_seed1`)

- **embedding:** `embeddings/hi_tf_aux_tf_reg_w0.10_gin_emlps_tds_asym_proj_8192neg_queue0_accum4_20ep_seed1` (128-d pre-3h)
- **cache:** `results/cache/temporal_flow_causal/Small-HI`
- **primary comparison:** Arm D vs Arm B (ΔAUPRC = +0.0964)
- **no SSL retraining:** True

## Four-arm test metrics

| arm | groups | dim | AUROC | AUPRC | F1 | P@100 | R@100 | lift@100 |
|-----|--------|----:|------:|------:|---:|------:|------:|---------:|
| A_embedding | pre_embedding_3h | 128 | 0.9638 | 0.3731 | 0.3987 | 0.7800 | 0.0484 | 417.86 |
| B_embedding_raw | pre_embedding_3h, raw | 132 | 0.9657 | 0.4149 | 0.3904 | 0.7900 | 0.0490 | 423.22 |
| D_embedding_raw_temporal_flow | pre_embedding_3h, raw, temporal_flow_causal | 137 | 0.9748 | 0.5112 | 0.4063 | 0.8500 | 0.0528 | 455.36 |

## Primary deltas (D − B)

- ΔAUPRC: **+0.0964**
- ΔF1: +0.0159
- ΔP@100: +0.0600
- ΔR@100: +0.0037
- Δlift@100: +32.14

Conservative read: single checkpoint; downstream probe only.
