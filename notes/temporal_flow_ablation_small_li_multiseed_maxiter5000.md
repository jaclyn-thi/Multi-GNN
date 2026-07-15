# Temporal flow causal ablation — Small-LI multiseed

**Seeds:** [1, 2, 3] (n=3)

## Primary Δ (Arm D − Arm B) — mean ± sample SD (ddof=1)

| metric | mean | sample SD | per-seed |
|--------|-----:|----------:|----------|
| auprc | 0.0648 | 0.0099 | +0.0634, +0.0754, +0.0558 |
| f1_at_selected_threshold | 0.0400 | 0.0213 | +0.0227, +0.0638, +0.0335 |
| precision_at_100 | 0.2567 | 0.1002 | +0.2200, +0.3700, +0.1800 |
| recall_at_100 | 0.0320 | 0.0125 | +0.0274, +0.0461, +0.0224 |
| lift_at_100 | 375.4664 | 146.5239 | +321.83, +541.25, +263.32 |

- D beats B on AUPRC in **3/3** seeds

## Seed 1 — `small_li_gin_emlps_tds_asym_proj_8192neg_queue0_20ep_seed1`

| arm | AUPRC | F1 | P@100 | lift@100 |
|-----|------:|---:|------:|---------:|
| A_embedding | 0.0448 | 0.0975 | 0.2300 | 336.46 |
| B_embedding_raw | 0.0932 | 0.0260 | 0.4900 | 716.81 |
| C_embedding_temporal_flow | 0.0958 | 0.1455 | 0.4500 | 658.30 |
| D_embedding_raw_temporal_flow | 0.1565 | 0.0486 | 0.7100 | 1038.65 |

ΔAUPRC(D−B)=+0.0634; ΔF1=+0.0227; Δlift@100=+321.83

## Seed 2 — `small_li_gin_emlps_tds_asym_proj_8192neg_queue0_20ep_seed2`

| arm | AUPRC | F1 | P@100 | lift@100 |
|-----|------:|---:|------:|---------:|
| A_embedding | 0.0203 | 0.0611 | 0.1300 | 190.17 |
| B_embedding_raw | 0.0232 | 0.0628 | 0.1600 | 234.05 |
| C_embedding_temporal_flow | 0.0804 | 0.1218 | 0.4700 | 687.53 |
| D_embedding_raw_temporal_flow | 0.0986 | 0.1266 | 0.5300 | 775.30 |

ΔAUPRC(D−B)=+0.0754; ΔF1=+0.0638; Δlift@100=+541.25

## Seed 3 — `small_li_gin_emlps_tds_asym_proj_8192neg_queue0_20ep_seed3`

| arm | AUPRC | F1 | P@100 | lift@100 |
|-----|------:|---:|------:|---------:|
| A_embedding | 0.0496 | 0.1128 | 0.2900 | 424.23 |
| B_embedding_raw | 0.0783 | 0.0528 | 0.4300 | 629.03 |
| C_embedding_temporal_flow | 0.0913 | 0.1715 | 0.4700 | 687.55 |
| D_embedding_raw_temporal_flow | 0.1341 | 0.0863 | 0.6100 | 892.35 |

ΔAUPRC(D−B)=+0.0558; ΔF1=+0.0335; Δlift@100=+263.32
