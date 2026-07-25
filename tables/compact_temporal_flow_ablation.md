# Compact temporal-flow ablation

| Dataset | Baseline stack | Baseline AUPRC | + temporal-flow AUPRC | Δ AUPRC |
| --- | --- | --- | --- | --- |
| Small-HI | pre-3h + raw | 0.320 | 0.501 | +0.180 |
| Small-LI | pre-3h + raw | 0.061 ± 0.033 | 0.128 ± 0.027 | +0.067 ± 0.010 |

**Notes:**
- Baseline = pre-3h + raw; +temporal-flow = pre-3h + raw + temporal_flow_causal.
- Uses validated max_iter=5000 temporal-flow results when available.
- Small-LI values are mean ± sample SD over seeds 1–3.
