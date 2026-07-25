# Temporal flow causal ablation — Small-HI (`hi_tf_aux_tf_reg_w0.05_gin_emlps_tds_asym_proj_8192neg_queue0_accum4_20ep_seed2`)

- **embedding:** `embeddings/hi_tf_aux_tf_reg_w0.05_gin_emlps_tds_asym_proj_8192neg_queue0_accum4_20ep_seed2/pre_embedding_3h` (198-d pre-3h)
- **cache:** `results/cache/temporal_flow_causal/Small-HI`
- **primary comparison:** Arm D vs Arm B (ΔAUPRC = +0.0859)
- **no SSL retraining:** True

## Four-arm test metrics

| arm | groups | dim | AUROC | AUPRC | F1 | P@100 | R@100 | lift@100 |
|-----|--------|----:|------:|------:|---:|------:|------:|---------:|
| A_embedding | pre_embedding_3h | 198 | 0.9679 | 0.3285 | 0.2987 | 0.9100 | 0.0565 | 487.53 |
| B_embedding_raw | pre_embedding_3h, raw | 202 | 0.9734 | 0.4171 | 0.1490 | 0.9700 | 0.0602 | 519.67 |
| D_embedding_raw_temporal_flow | pre_embedding_3h, raw, temporal_flow_causal | 207 | 0.9796 | 0.5030 | 0.4147 | 0.9800 | 0.0608 | 525.03 |

## Primary deltas (D − B)

- ΔAUPRC: **+0.0859**
- ΔF1: +0.2657
- ΔP@100: +0.0100
- ΔR@100: +0.0006
- Δlift@100: +5.36

Conservative read: single checkpoint; downstream probe only.
