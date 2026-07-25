# Appendix — Alert-budget performance

| Dataset | Method | P@100 | R@100 | Lift@100 | P@500 | R@500 | Lift@500 | P@1000 | R@1000 | Lift@1000 | Caveat |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Small-HI | Raw features only | — | — | — | — | — | — | — | — | — | val-tuned F1; no SSL |
| Small-HI | Raw + morphology | 0.230 | 0.014 | 123 | 0.188 | 0.058 | 101 | 0.153 | 0.095 | 82 | val-tuned F1; no SSL |
| Small-HI | SSL post-128 | 0.800 | 0.050 | 429 | 0.590 | 0.183 | 316 | 0.398 | 0.247 | 213 | val-tuned F1 |
| Small-HI | SSL pre-3h | 0.830 | 0.052 | 445 | 0.602 | 0.187 | 322 | 0.447 | 0.277 | 239 | val-tuned F1 |
| Small-HI | SSL post-128 + raw | 0.790 | 0.049 | 423 | 0.630 | 0.196 | 337 | 0.442 | 0.274 | 237 | val-tuned F1 |
| Small-HI | SSL pre-3h + raw | 0.840 | 0.052 | 450 | 0.640 | 0.199 | 343 | 0.490 | 0.304 | 262 | val-tuned F1 |
| Small-HI | SSL pre-3h + raw + temporal-flow | 0.940 | 0.058 | 504 | 0.842 | 0.261 | 451 | 0.682 | 0.423 | 365 | val-tuned F1; validated temporal-flow stack |
| Small-HI | Legacy supervised Multi-GIN+EU (ports TDS-off 50ep mean) | — | — | — | — | — | — | — | — | — | paper_argmax F1 |
| Small-LI | SSL post-128 | 0.120 ± 0.060 | 0.015 ± 0.007 | 176 ± 88 | 0.058 ± 0.034 | 0.036 ± 0.021 | 85 ± 50 | 0.046 ± 0.025 | 0.057 ± 0.031 | 67 ± 36 | frozen probe; mean ± sample SD (n=3) |
| Small-LI | SSL pre-3h | 0.220 ± 0.082 | 0.027 ± 0.010 | 322 ± 120 | 0.123 ± 0.059 | 0.077 ± 0.037 | 180 ± 87 | 0.087 ± 0.032 | 0.108 ± 0.040 | 127 ± 47 | frozen probe; mean ± sample SD (n=3) |
| Small-LI | SSL post-128 + raw | 0.227 ± 0.122 | 0.028 ± 0.015 | 332 ± 179 | 0.095 ± 0.059 | 0.059 ± 0.037 | 138 ± 86 | 0.070 ± 0.033 | 0.087 ± 0.041 | 102 ± 49 | frozen probe; mean ± sample SD (n=3) |
| Small-LI | SSL pre-3h + raw | 0.343 ± 0.159 | 0.043 ± 0.020 | 502 ± 232 | 0.163 ± 0.085 | 0.102 ± 0.053 | 239 ± 125 | 0.111 ± 0.048 | 0.139 ± 0.060 | 163 ± 70 | frozen probe; mean ± sample SD (n=3) |
| Small-LI | SSL pre-3h + raw + temporal-flow | 0.600 ± 0.056 | 0.075 ± 0.007 | 878 ± 81 | 0.271 ± 0.046 | 0.169 ± 0.029 | 396 ± 68 | 0.163 ± 0.023 | 0.203 ± 0.028 | 238 ± 33 | val-tuned F1; mean ± sample SD (n=3) |
| Small-LI | Legacy supervised GIN (100ep seed1) | 0.970 | 0.121 | 1419 | 0.462 | 0.288 | 676 | 0.270 | 0.337 | 395 | paper_argmax F1 |

**Notes:**
- Fixed top-K alert-budget metrics on the test split; threshold-tuned precision/recall are omitted.
- Small-LI SSL rows use mean ± sample SD (ddof=1) over seeds 1–3 where available.
- K=500 and K=1000 may be unavailable (—) for some multiseed aggregates when not present in registry summaries.
- P@100 = precision among the top 100 scored test transactions.
- R@100 = fraction of all positive test transactions recovered in the top 100 scored test transactions.
- Lift@100 = P@100 divided by the test-set positive rate for that dataset.
