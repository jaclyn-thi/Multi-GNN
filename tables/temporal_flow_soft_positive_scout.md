# Temporal-flow soft-positive scout

## Verdict

**Negative result** (`thesis_role=negative_result`, `validation_status=diagnostic_only`).
Do **not** insert into main thesis tables.

Main scout (A/B/C) failed primary criteria:

- pre-3h embedding-only AUPRC below baseline (~0.224) for every variant
- pre-3h + raw did not beat baseline (~0.274)
- P@100 collapsed vs baseline (~0.76)
- recall at precision ≥ 0.90 never achieved; R@P≥0.80 only trivial
- soft-positive caps saturated (avg ≈ max_per_anchor) — positives too broad/low-quality

Optional single scarce-positive sanity: `tf_soft_strict_bins10_min5_cap4_w0.01` (min_shared=5 = all 5 TF features, cap=4, w=0.01). If it still saturates or underperforms, stop soft-positive experiments.

- Primary representation: **pre-3h** (post-128 is diagnostic only)
- SSL soft positives use causal `temporal_flow_causal` bins; **no labels**
- Identity pair remains primary; TF soft positives are low-weight extras

- Best pre-3h embedding-only: **tf_soft_bins10_min4_cap32_w0.05**
- Best pre-3h + raw: **tf_soft_bins5_min4_cap16_w0.10**
- Best final stack (D): **tf_soft_bins5_min4_cap16_w0.10**
- Best recall-oriented (A R@P≥0.90): **tf_soft_bins5_min3_cap16_w0.05**

## Pre-3h primary metrics

| variant | arm | AUPRC | P@100 | R@500 | R@1000 | R@P≥0.90 | R@P≥0.80 |
|---|---|---:|---:|---:|---:|---:|---:|
| tf_soft_bins5_min3_cap16_w0.05 | A_embedding | 0.1049 | 0.5300 | 0.0968 | 0.1390 | — | — |
| tf_soft_bins5_min3_cap16_w0.05 | B_embedding_raw | 0.0594 | 0.2800 | 0.0652 | 0.1006 | — | — |
| tf_soft_bins5_min3_cap16_w0.05 | D_embedding_raw_temporal_flow | 0.2589 | 0.7100 | 0.1639 | 0.2390 | — | — |
| tf_soft_bins5_min4_cap16_w0.10 | A_embedding | 0.1149 | 0.3100 | 0.0925 | 0.1682 | — | — |
| tf_soft_bins5_min4_cap16_w0.10 | B_embedding_raw | 0.2477 | 0.7100 | 0.1533 | 0.2613 | — | 0.0050 |
| tf_soft_bins5_min4_cap16_w0.10 | D_embedding_raw_temporal_flow | 0.4119 | 0.7700 | 0.2185 | 0.3582 | — | 0.0348 |
| tf_soft_bins10_min4_cap32_w0.05 | A_embedding | 0.1367 | 0.3700 | 0.1148 | 0.1868 | — | — |
| tf_soft_bins10_min4_cap32_w0.05 | B_embedding_raw | 0.0521 | 0.1600 | 0.0528 | 0.0937 | — | — |
| tf_soft_bins10_min4_cap32_w0.05 | D_embedding_raw_temporal_flow | 0.1506 | 0.1900 | 0.1099 | 0.1924 | — | — |
| tf_soft_strict_bins10_min5_cap4_w0.01 | A_embedding | 0.0077 | 0.0000 | 0.0025 | 0.0037 | — | — |
| tf_soft_strict_bins10_min5_cap4_w0.01 | B_embedding_raw | 0.0104 | 0.0000 | 0.0031 | 0.0043 | — | — |
| tf_soft_strict_bins10_min5_cap4_w0.01 | D_embedding_raw_temporal_flow | 0.1568 | 0.7700 | 0.1285 | 0.1726 | — | 0.0354 |

Full JSON: `results/diagnostics/temporal_flow_soft_positive_scout.json`

