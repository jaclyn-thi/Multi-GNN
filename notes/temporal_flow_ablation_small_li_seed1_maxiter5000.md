# Temporal flow causal ablation — Small-LI (`small_li_gin_emlps_tds_asym_proj_8192neg_queue0_20ep_seed1`)

- **embedding:** `embeddings/small_li_gin_emlps_tds_asym_proj_8192neg_queue0_20ep_seed1/pre_embedding_3h` (198-d pre-3h)
- **cache:** `results/cache/temporal_flow_causal/Small-LI`
- **primary comparison:** Arm D vs Arm B (ΔAUPRC = +0.0634)
- **no SSL retraining:** True

## Four-arm test metrics

| arm | groups | dim | AUROC | AUPRC | F1 | P@100 | R@100 | lift@100 |
|-----|--------|----:|------:|------:|---:|------:|------:|---------:|
| A_embedding | pre_embedding_3h | 198 | 0.9225 | 0.0448 | 0.0975 | 0.2300 | 0.0287 | 336.46 |
| B_embedding_raw | pre_embedding_3h, raw | 202 | 0.9345 | 0.0932 | 0.0260 | 0.4900 | 0.0611 | 716.81 |
| C_embedding_temporal_flow | pre_embedding_3h, temporal_flow_causal | 203 | 0.9459 | 0.0958 | 0.1455 | 0.4500 | 0.0561 | 658.30 |
| D_embedding_raw_temporal_flow | pre_embedding_3h, raw, temporal_flow_causal | 207 | 0.9507 | 0.1565 | 0.0486 | 0.7100 | 0.0885 | 1038.65 |

## Primary deltas (D − B)

- ΔAUPRC: **+0.0634**
- ΔF1: +0.0227
- ΔP@100: +0.2200
- ΔR@100: +0.0274
- Δlift@100: +321.83

Conservative read: single checkpoint; downstream probe only.
