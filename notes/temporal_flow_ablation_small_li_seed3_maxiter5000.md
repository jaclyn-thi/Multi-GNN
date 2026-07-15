# Temporal flow causal ablation — Small-LI (`small_li_gin_emlps_tds_asym_proj_8192neg_queue0_20ep_seed3`)

- **embedding:** `embeddings/small_li_gin_emlps_tds_asym_proj_8192neg_queue0_20ep_seed3/pre_embedding_3h` (198-d pre-3h)
- **cache:** `results/cache/temporal_flow_causal/Small-LI`
- **primary comparison:** Arm D vs Arm B (ΔAUPRC = +0.0558)
- **no SSL retraining:** True

## Four-arm test metrics

| arm | groups | dim | AUROC | AUPRC | F1 | P@100 | R@100 | lift@100 |
|-----|--------|----:|------:|------:|---:|------:|------:|---------:|
| A_embedding | pre_embedding_3h | 198 | 0.9257 | 0.0496 | 0.1128 | 0.2900 | 0.0362 | 424.23 |
| B_embedding_raw | pre_embedding_3h, raw | 202 | 0.9343 | 0.0783 | 0.0528 | 0.4300 | 0.0536 | 629.03 |
| C_embedding_temporal_flow | pre_embedding_3h, temporal_flow_causal | 203 | 0.9457 | 0.0913 | 0.1715 | 0.4700 | 0.0586 | 687.55 |
| D_embedding_raw_temporal_flow | pre_embedding_3h, raw, temporal_flow_causal | 207 | 0.9524 | 0.1341 | 0.0863 | 0.6100 | 0.0761 | 892.35 |

## Primary deltas (D − B)

- ΔAUPRC: **+0.0558**
- ΔF1: +0.0335
- ΔP@100: +0.1800
- ΔR@100: +0.0224
- Δlift@100: +263.32

Conservative read: single checkpoint; downstream probe only.
