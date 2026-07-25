# Temporal flow causal ablation — Small-HI (`hi_tf_aux_tf_reg_w0.05_gin_emlps_tds_asym_proj_8192neg_queue0_accum4_20ep_seed1`)

- **embedding:** `embeddings/hi_tf_aux_tf_reg_w0.05_gin_emlps_tds_asym_proj_8192neg_queue0_accum4_20ep_seed1` (128-d pre-3h)
- **cache:** `results/cache/temporal_flow_causal/Small-HI`
- **primary comparison:** Arm D vs Arm B (ΔAUPRC = +0.0987)
- **no SSL retraining:** True

## Four-arm test metrics

| arm | groups | dim | AUROC | AUPRC | F1 | P@100 | R@100 | lift@100 |
|-----|--------|----:|------:|------:|---:|------:|------:|---------:|
| A_embedding | pre_embedding_3h | 128 | 0.9603 | 0.2785 | 0.3333 | 0.8200 | 0.0509 | 439.29 |
| B_embedding_raw | pre_embedding_3h, raw | 132 | 0.9686 | 0.3867 | 0.1069 | 0.8500 | 0.0528 | 455.36 |
| D_embedding_raw_temporal_flow | pre_embedding_3h, raw, temporal_flow_causal | 137 | 0.9780 | 0.4853 | 0.2586 | 0.8900 | 0.0552 | 476.79 |

## Primary deltas (D − B)

- ΔAUPRC: **+0.0987**
- ΔF1: +0.1517
- ΔP@100: +0.0400
- ΔR@100: +0.0025
- Δlift@100: +21.43

Conservative read: single checkpoint; downstream probe only.
