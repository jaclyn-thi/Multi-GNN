# Temporal flow causal ablation — Small-HI (`pna_width65_ginparams_emlps_tds_asym_proj_8192neg_queue0_20ep_seed1`)

- **embedding:** `embeddings/pna_width65_ginparams_emlps_tds_asym_proj_8192neg_queue0_20ep_seed1/pre_embedding_3h` (195-d pre-3h)
- **cache:** `results/cache/temporal_flow_causal/Small-HI`
- **primary comparison:** Arm D vs Arm B (ΔAUPRC = +0.1324)
- **no SSL retraining:** True

## Four-arm test metrics

| arm | groups | dim | AUROC | AUPRC | F1 | P@100 | R@100 | lift@100 |
|-----|--------|----:|------:|------:|---:|------:|------:|---------:|
| A_embedding | pre_embedding_3h | 195 | 0.9602 | 0.2303 | 0.2970 | 0.8200 | 0.0509 | 439.29 |
| B_embedding_raw | pre_embedding_3h, raw | 199 | 0.9689 | 0.2742 | 0.2792 | 0.8200 | 0.0509 | 439.29 |
| C_embedding_temporal_flow | pre_embedding_3h, temporal_flow_causal | 200 | 0.9796 | 0.3703 | 0.3941 | 0.9100 | 0.0565 | 487.51 |
| D_embedding_raw_temporal_flow | pre_embedding_3h, raw, temporal_flow_causal | 204 | 0.9824 | 0.4065 | 0.4099 | 0.9300 | 0.0577 | 498.22 |

## Primary deltas (D − B)

- ΔAUPRC: **+0.1324**
- ΔF1: +0.1307
- ΔP@100: +0.1100
- ΔR@100: +0.0068
- Δlift@100: +58.93

Conservative read: single checkpoint; downstream probe only.
