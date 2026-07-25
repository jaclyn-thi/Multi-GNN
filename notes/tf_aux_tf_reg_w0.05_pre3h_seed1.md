# Temporal flow causal ablation — Small-HI (`hi_tf_aux_tf_reg_w0.05_gin_emlps_tds_asym_proj_8192neg_queue0_accum4_20ep_seed1`)

- **embedding:** `embeddings/hi_tf_aux_tf_reg_w0.05_gin_emlps_tds_asym_proj_8192neg_queue0_accum4_20ep_seed1/pre_embedding_3h` (198-d pre-3h)
- **cache:** `results/cache/temporal_flow_causal/Small-HI`
- **primary comparison:** Arm D vs Arm B (ΔAUPRC = +0.0955)
- **no SSL retraining:** True

## Four-arm test metrics

| arm | groups | dim | AUROC | AUPRC | F1 | P@100 | R@100 | lift@100 |
|-----|--------|----:|------:|------:|---:|------:|------:|---------:|
| A_embedding | pre_embedding_3h | 198 | 0.9688 | 0.3719 | 0.4158 | 0.9000 | 0.0559 | 482.15 |
| B_embedding_raw | pre_embedding_3h, raw | 202 | 0.9666 | 0.3549 | 0.3788 | 0.8800 | 0.0546 | 471.44 |
| D_embedding_raw_temporal_flow | pre_embedding_3h, raw, temporal_flow_causal | 207 | 0.9765 | 0.4504 | 0.4381 | 0.9100 | 0.0565 | 487.51 |

## Primary deltas (D − B)

- ΔAUPRC: **+0.0955**
- ΔF1: +0.0593
- ΔP@100: +0.0300
- ΔR@100: +0.0019
- Δlift@100: +16.07

Conservative read: single checkpoint; downstream probe only.
