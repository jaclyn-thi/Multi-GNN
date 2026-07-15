# Temporal flow causal ablation — Small-LI (`small_li_gin_emlps_tds_asym_proj_8192neg_queue0_20ep_seed1`)

- **embedding:** `embeddings/small_li_gin_emlps_tds_asym_proj_8192neg_queue0_20ep_seed1/pre_embedding_3h` (198-d pre-3h)
- **cache:** `results/cache/temporal_flow_causal/Small-LI`
- **primary comparison:** Arm D vs Arm B (ΔAUPRC = +0.0697)
- **no SSL retraining:** True

## Four-arm test metrics

| arm | groups | dim | AUROC | AUPRC | F1 | P@100 | R@100 | lift@100 |
|-----|--------|----:|------:|------:|---:|------:|------:|---------:|
| A_embedding | pre_embedding_3h | 198 | 0.9226 | 0.0461 | 0.0941 | 0.2400 | 0.0299 | 351.09 |
| B_embedding_raw | pre_embedding_3h, raw | 202 | 0.9316 | 0.0811 | 0.0533 | 0.4200 | 0.0524 | 614.41 |
| C_embedding_temporal_flow | pre_embedding_3h, temporal_flow_causal | 203 | 0.9458 | 0.0972 | 0.1483 | 0.4600 | 0.0574 | 672.92 |
| D_embedding_raw_temporal_flow | pre_embedding_3h, raw, temporal_flow_causal | 207 | 0.9495 | 0.1508 | 0.0668 | 0.6500 | 0.0810 | 950.87 |

## Primary deltas (D − B)

- ΔAUPRC: **+0.0697**
- ΔF1: +0.0135
- ΔP@100: +0.2300
- ΔR@100: +0.0287
- Δlift@100: +336.46

Conservative read: single checkpoint; downstream probe only.
