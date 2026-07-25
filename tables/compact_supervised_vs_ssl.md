# Compact supervised versus frozen SSL

| Dataset | Method | AUPRC | F1 | P@100 | R@100 | Caveat |
| --- | --- | --- | --- | --- | --- | --- |
| Small-HI | SSL pre-3h + raw + temporal-flow | 0.501 | 0.465 | 0.940 | 0.058 | val-tuned F1; frozen linear probe |
| Small-HI | Legacy supervised GIN | 0.663 | 0.660 | — | — | paper_argmax F1 mean (seeds 1–3); supervised CE; not comparable to SSL val-tuned F1 |
| Small-LI | SSL pre-3h + raw + temporal-flow | 0.128 ± 0.027 | 0.092 ± 0.029 | 0.600 ± 0.056 | 0.075 ± 0.007 | val-tuned F1; frozen linear probe; mean ± sample SD (n=3) |
| Small-LI | Legacy supervised GIN | 0.292 | 0.357 | 0.970 | 0.121 | paper_argmax F1; supervised CE; not comparable to SSL val-tuned F1 |

**Notes:**
- SSL uses frozen linear probe with validation-tuned threshold.
- Supervised uses end-to-end supervised CE and paper_argmax F1.
- F1 values are not directly comparable without the protocol caveat.
