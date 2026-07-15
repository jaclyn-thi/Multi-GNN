# Temporal flow causal ablation — Small-HI (`pna_width65_ginparams_emlps_tds_asym_proj_8192neg_queue0_20ep_seed1`)

- **embedding:** `embeddings/pna_width65_ginparams_emlps_tds_asym_proj_8192neg_queue0_20ep_seed1` (128-d pre-3h)
- **cache:** `results/cache/temporal_flow_causal/Small-HI`
- **primary comparison:** Arm D vs Arm B (ΔAUPRC = +nan)
- **no SSL retraining:** True

## Four-arm test metrics

| arm | groups | dim | AUROC | AUPRC | F1 | P@100 | R@100 | lift@100 |
|-----|--------|----:|------:|------:|---:|------:|------:|---------:|
| D_embedding_raw_temporal_flow | pre_embedding_3h, raw, temporal_flow_causal | 137 | 0.9817 | 0.3995 | 0.4225 | 0.8500 | 0.0528 | 455.36 |

## Primary deltas (D − B)

- ΔAUPRC: **+nan**
- ΔF1: +nan
- ΔP@100: +nan
- ΔR@100: +nan
- Δlift@100: +nan

Conservative read: single checkpoint; downstream probe only.
