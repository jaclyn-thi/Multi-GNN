# Compact AUPRC summary

| Method | Small-HI AUPRC | Small-LI AUPRC |
| --- | --- | --- |
| SSL post-128 | 0.245 | 0.014 ± 0.010 |
| SSL pre-3h | 0.295 | 0.039 ± 0.016 |
| SSL pre-3h + raw | 0.321 | 0.061 ± 0.034 |
| SSL pre-3h + raw + temporal-flow | 0.501 | 0.128 ± 0.027 |
| Legacy supervised GIN | 0.663 | 0.292 |

**Notes:**
- Small-LI SSL rows are mean ± sample SD (ddof=1) over seeds 1–3.
- Small-HI SSL rows use the validated strong-run / temporal-flow protocol.
- Supervised rows use end-to-end paper_argmax evaluation; SSL rows use frozen probe AUPRC.
