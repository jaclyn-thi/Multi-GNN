# Temporal flow causal ablation — Small-LI multiseed

**Seeds:** [1, 2, 3] (n=3)

## Primary Δ (Arm D − Arm B) — mean ± sample SD (ddof=1)

| metric | mean | sample SD | per-seed |
|--------|-----:|----------:|----------|
| auprc | 0.0669 | 0.0100 | +0.0697, +0.0752, +0.0558 |
| f1_at_selected_threshold | 0.0359 | 0.0241 | +0.0135, +0.0614, +0.0328 |
| precision_at_100 | 0.2633 | 0.1041 | +0.2300, +0.3800, +0.1800 |
| recall_at_100 | 0.0328 | 0.0130 | +0.0287, +0.0474, +0.0224 |
| lift_at_100 | 385.2188 | 152.2535 | +336.46, +555.88, +263.32 |

- D beats B on AUPRC in **3/3** seeds

## Seed 1 — `small_li_gin_emlps_tds_asym_proj_8192neg_queue0_20ep_seed1`

| arm | AUPRC | F1 | P@100 | lift@100 |
|-----|------:|---:|------:|---------:|
| A_embedding | 0.0461 | 0.0941 | 0.2400 | 351.09 |
| B_embedding_raw | 0.0811 | 0.0533 | 0.4200 | 614.41 |
| C_embedding_temporal_flow | 0.0972 | 0.1483 | 0.4600 | 672.92 |
| D_embedding_raw_temporal_flow | 0.1508 | 0.0668 | 0.6500 | 950.87 |

ΔAUPRC(D−B)=+0.0697; ΔF1=+0.0135; Δlift@100=+336.46

## Seed 2 — `small_li_gin_emlps_tds_asym_proj_8192neg_queue0_20ep_seed2`

| arm | AUPRC | F1 | P@100 | lift@100 |
|-----|------:|---:|------:|---------:|
| A_embedding | 0.0201 | 0.0617 | 0.1300 | 190.17 |
| B_embedding_raw | 0.0227 | 0.0626 | 0.1600 | 234.05 |
| C_embedding_temporal_flow | 0.0797 | 0.1177 | 0.4700 | 687.53 |
| D_embedding_raw_temporal_flow | 0.0979 | 0.1239 | 0.5400 | 789.93 |

ΔAUPRC(D−B)=+0.0752; ΔF1=+0.0614; Δlift@100=+555.88

## Seed 3 — `small_li_gin_emlps_tds_asym_proj_8192neg_queue0_20ep_seed3`

| arm | AUPRC | F1 | P@100 | lift@100 |
|-----|------:|---:|------:|---------:|
| A_embedding | 0.0500 | 0.1135 | 0.2900 | 424.23 |
| B_embedding_raw | 0.0790 | 0.0518 | 0.4300 | 629.03 |
| C_embedding_temporal_flow | 0.0913 | 0.1715 | 0.4600 | 672.92 |
| D_embedding_raw_temporal_flow | 0.1349 | 0.0845 | 0.6100 | 892.35 |

ΔAUPRC(D−B)=+0.0558; ΔF1=+0.0328; Δlift@100=+263.32
