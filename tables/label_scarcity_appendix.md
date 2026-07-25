# Appendix — Label-scarcity temporal-flow diagnostic

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
- Diagnostic appendix only; not inserted into main thesis tables.
- Train labels subsampled; validation/test unchanged. Frozen pre-3h embeddings.
- Small-LI values are mean ± sample SD over model seeds 1–3 when available.
