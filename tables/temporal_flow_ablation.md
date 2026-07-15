# Table 5 — Temporal-flow ablation

| Dataset / run | Comparison | Feature stack | AUPRC | Δ AUPRC vs pre-3h + raw | F1 | P@100 | R@100 | Lift@100 | Validation status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Small-HI 40ep seed2 | pre-3h only | embedding | 0.295 | — | 0.336 | 0.830 | 0.052 | 445 | validated |
| Small-HI 40ep seed2 | pre-3h + raw | embedding+raw | 0.320 | — | 0.344 | 0.830 | 0.052 | 445 | validated |
| Small-HI 40ep seed2 | pre-3h + temporal-flow | embedding+temporal_flow_causal | 0.473 | — | 0.475 | 0.910 | 0.056 | 488 | validated |
| Small-HI 40ep seed2 | pre-3h + raw + temporal-flow | embedding+raw+temporal_flow_causal | 0.501 | +0.180 | 0.465 | 0.940 | 0.058 | 504 | validated |
| Small-LI multiseed | pre-3h + raw | embedding+raw | 0.061 ± 0.033 | — | 0.056 ± 0.006 | 0.337 ± 0.153 | 0.042 ± 0.019 | 492 ± 224 | validated |
| Small-LI multiseed | pre-3h + raw + temporal-flow | embedding+raw+temporal_flow_causal | 0.128 ± 0.027 | +0.067 ± 0.010 | 0.092 ± 0.029 | 0.600 ± 0.056 | 0.075 ± 0.007 | 878 ± 81 | validated |

**Notes:**
- Primary comparison: pre-3h + raw + temporal-flow versus pre-3h + raw.
- Provisional rows shown only with --include_provisional until validation summary passes.
- Validated max_iter=5000 JSONs preferred when validation summary passes.
- P@100 = precision among the top 100 scored test transactions.
- R@100 = fraction of all positive test transactions recovered in the top 100 scored test transactions.
- Lift@100 = P@100 divided by the test-set positive rate for that dataset.
