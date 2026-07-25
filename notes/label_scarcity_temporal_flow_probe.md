# Label-scarcity temporal-flow probe

Downstream-only evaluation of frozen `pre-3h + raw` vs `pre-3h + raw + temporal_flow_causal` under subsampled train labels.

## Caveats

- Downstream-only probe on frozen pre-3h embeddings.
- Validation/test sets unchanged; only train labels subsampled.
- Small-HI uses one strong run (40ep seed2) × scarcity_seed=1.
- Small-LI uses model seeds 1–3 × scarcity_seed=1.
- Diagnostic appendix only; not inserted into main thesis tables.

## AUPRC by label fraction

| Label fraction | Small-HI pre-3h+raw AUPRC | Small-HI pre-3h+raw+temporal AUPRC | Small-LI pre-3h+raw AUPRC | Small-LI pre-3h+raw+temporal AUPRC |
| --- | --- | --- | --- | --- |
| 1% | 0.058 | 0.139 | 0.004 ± 0.001 | 0.004 ± 0.001 |
| 5% | 0.143 | 0.300 | 0.036 ± 0.037 | 0.068 ± 0.053 |
| 10% | 0.306 | 0.430 | 0.023 ± 0.012 | 0.052 ± 0.018 |
| 25% | 0.298 | 0.475 | 0.047 ± 0.025 | 0.106 ± 0.028 |
| 50% | 0.306 | 0.496 | 0.061 ± 0.037 | 0.121 ± 0.040 |
| 100% | 0.320 | 0.501 | 0.065 ± 0.037 | 0.130 ± 0.029 |

**Notes:**
- Small-HI: temporal-flow improved AUPRC in 6/6 label fractions (mean ΔAUPRC=+0.152).
- Small-HI: temporal-flow gain appears larger at high label fractions.
- Small-LI: temporal-flow improved mean AUPRC in 6/6 label fractions (mean ΔAUPRC=+0.041).
- At 1% labels, performance is degraded vs 100% but not fully collapsed on the available datasets.
- Baseline = pre-3h + raw; +temporal = pre-3h + raw + temporal_flow_causal.
- Small-LI values are mean ± sample SD over model seeds 1–3 when available.
