# Temporal flow causal ablation — Small-LI (`small_li_gin_emlps_tds_asym_proj_8192neg_queue0_20ep_seed2`)

- **embedding:** `embeddings/small_li_gin_emlps_tds_asym_proj_8192neg_queue0_20ep_seed2/pre_embedding_3h` (198-d pre-3h)
- **cache:** `results/cache/temporal_flow_causal/Small-LI`
- **primary comparison:** Arm D vs Arm B (ΔAUPRC = +0.0752)
- **no SSL retraining:** True

## Four-arm test metrics

| arm | groups | dim | AUROC | AUPRC | F1 | P@100 | R@100 | lift@100 |
|-----|--------|----:|------:|------:|---:|------:|------:|---------:|
| A_embedding | pre_embedding_3h | 198 | 0.9093 | 0.0201 | 0.0617 | 0.1300 | 0.0162 | 190.17 |
| B_embedding_raw | pre_embedding_3h, raw | 202 | 0.9113 | 0.0227 | 0.0626 | 0.1600 | 0.0200 | 234.05 |
| C_embedding_temporal_flow | pre_embedding_3h, temporal_flow_causal | 203 | 0.9373 | 0.0797 | 0.1177 | 0.4700 | 0.0586 | 687.53 |
| D_embedding_raw_temporal_flow | pre_embedding_3h, raw, temporal_flow_causal | 207 | 0.9401 | 0.0979 | 0.1239 | 0.5400 | 0.0673 | 789.93 |

## Primary deltas (D − B)

- ΔAUPRC: **+0.0752**
- ΔF1: +0.0614
- ΔP@100: +0.3800
- ΔR@100: +0.0474
- Δlift@100: +555.88

Conservative read: single checkpoint; downstream probe only.
