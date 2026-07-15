# Temporal flow causal ablation — Small-LI (`small_li_gin_emlps_tds_asym_proj_8192neg_queue0_20ep_seed2`)

- **embedding:** `embeddings/small_li_gin_emlps_tds_asym_proj_8192neg_queue0_20ep_seed2/pre_embedding_3h` (198-d pre-3h)
- **cache:** `results/cache/temporal_flow_causal/Small-LI`
- **primary comparison:** Arm D vs Arm B (ΔAUPRC = +0.0754)
- **no SSL retraining:** True

## Four-arm test metrics

| arm | groups | dim | AUROC | AUPRC | F1 | P@100 | R@100 | lift@100 |
|-----|--------|----:|------:|------:|---:|------:|------:|---------:|
| A_embedding | pre_embedding_3h | 198 | 0.9095 | 0.0203 | 0.0611 | 0.1300 | 0.0162 | 190.17 |
| B_embedding_raw | pre_embedding_3h, raw | 202 | 0.9115 | 0.0232 | 0.0628 | 0.1600 | 0.0200 | 234.05 |
| C_embedding_temporal_flow | pre_embedding_3h, temporal_flow_causal | 203 | 0.9376 | 0.0804 | 0.1218 | 0.4700 | 0.0586 | 687.53 |
| D_embedding_raw_temporal_flow | pre_embedding_3h, raw, temporal_flow_causal | 207 | 0.9401 | 0.0986 | 0.1266 | 0.5300 | 0.0661 | 775.30 |

## Primary deltas (D − B)

- ΔAUPRC: **+0.0754**
- ΔF1: +0.0638
- ΔP@100: +0.3700
- ΔR@100: +0.0461
- Δlift@100: +541.25

Conservative read: single checkpoint; downstream probe only.
