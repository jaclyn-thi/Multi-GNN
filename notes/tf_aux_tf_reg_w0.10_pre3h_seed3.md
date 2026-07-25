# Temporal flow causal ablation — Small-HI (`hi_tf_aux_tf_reg_w0.10_gin_emlps_tds_asym_proj_8192neg_queue0_accum4_20ep_seed3`)

- **embedding:** `embeddings/hi_tf_aux_tf_reg_w0.10_gin_emlps_tds_asym_proj_8192neg_queue0_accum4_20ep_seed3/pre_embedding_3h` (198-d pre-3h)
- **cache:** `results/cache/temporal_flow_causal/Small-HI`
- **primary comparison:** Arm D vs Arm B (ΔAUPRC = +0.0902)
- **no SSL retraining:** True

## Four-arm test metrics

| arm | groups | dim | AUROC | AUPRC | F1 | P@100 | R@100 | lift@100 |
|-----|--------|----:|------:|------:|---:|------:|------:|---------:|
| A_embedding | pre_embedding_3h | 198 | 0.9731 | 0.4165 | 0.4688 | 0.8900 | 0.0552 | 476.80 |
| B_embedding_raw | pre_embedding_3h, raw | 202 | 0.9763 | 0.4824 | 0.3632 | 0.9300 | 0.0577 | 498.23 |
| D_embedding_raw_temporal_flow | pre_embedding_3h, raw, temporal_flow_causal | 207 | 0.9819 | 0.5726 | 0.4458 | 0.9600 | 0.0596 | 514.30 |

## Primary deltas (D − B)

- ΔAUPRC: **+0.0902**
- ΔF1: +0.0826
- ΔP@100: +0.0300
- ΔR@100: +0.0019
- Δlift@100: +16.07

Conservative read: single checkpoint; downstream probe only.
