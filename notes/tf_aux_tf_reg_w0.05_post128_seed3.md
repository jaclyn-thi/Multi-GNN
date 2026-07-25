# Temporal flow causal ablation — Small-HI (`hi_tf_aux_tf_reg_w0.05_gin_emlps_tds_asym_proj_8192neg_queue0_accum4_20ep_seed3`)

- **embedding:** `embeddings/hi_tf_aux_tf_reg_w0.05_gin_emlps_tds_asym_proj_8192neg_queue0_accum4_20ep_seed3` (128-d pre-3h)
- **cache:** `results/cache/temporal_flow_causal/Small-HI`
- **primary comparison:** Arm D vs Arm B (ΔAUPRC = +0.1784)
- **no SSL retraining:** True

## Four-arm test metrics

| arm | groups | dim | AUROC | AUPRC | F1 | P@100 | R@100 | lift@100 |
|-----|--------|----:|------:|------:|---:|------:|------:|---------:|
| A_embedding | pre_embedding_3h | 128 | 0.9505 | 0.1715 | 0.2603 | 0.5200 | 0.0323 | 278.58 |
| B_embedding_raw | pre_embedding_3h, raw | 132 | 0.9618 | 0.2504 | 0.2814 | 0.7200 | 0.0447 | 385.73 |
| D_embedding_raw_temporal_flow | pre_embedding_3h, raw, temporal_flow_causal | 137 | 0.9768 | 0.4288 | 0.4031 | 0.8500 | 0.0528 | 455.37 |

## Primary deltas (D − B)

- ΔAUPRC: **+0.1784**
- ΔF1: +0.1218
- ΔP@100: +0.1300
- ΔR@100: +0.0081
- Δlift@100: +69.65

Conservative read: single checkpoint; downstream probe only.
