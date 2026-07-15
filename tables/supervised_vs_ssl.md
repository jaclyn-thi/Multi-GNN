# Table 6 — Supervised versus frozen SSL

| Dataset | Method | Training signal | Encoder updated with labels? | AUPRC | F1 | P@100 | R@100 | Lift@100 | Caveat |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Small-HI | SSL pre-3h + raw + temporal-flow | contrastive + frozen probe | no (frozen probe) | 0.501 | 0.465 | 0.940 | 0.058 | 504 | val-tuned F1 |
| Small-HI | Legacy supervised GIN | supervised CE (end-to-end) | yes | 0.639 | 0.539 | 0.990 | 0.061 | 530 | paper_argmax F1 |
| Small-LI | SSL pre-3h + raw (multiseed mean) | contrastive + frozen probe | no (frozen probe) | 0.061 ± 0.034 | 0.054 ± 0.007 | 0.343 ± 0.159 | 0.043 ± 0.020 | 502 ± 232 | val-tuned F1; frozen linear probe |
| Small-LI | SSL pre-3h + raw + temporal-flow (multiseed mean) | contrastive + frozen probe | no (frozen probe) | 0.128 ± 0.027 | 0.092 ± 0.029 | 0.600 ± 0.056 | 0.075 ± 0.007 | 878 ± 81 | val-tuned F1; mean ± sample SD (n=3) |
| Small-LI | Legacy supervised GIN | supervised CE (end-to-end) | yes | 0.292 | 0.357 | 0.970 | 0.121 | 1419 | paper_argmax F1 |

**Notes:**
- SSL rows use frozen linear probe with validation-tuned threshold.
- Supervised rows use end-to-end labeled training and paper_argmax F1.
- SSL and supervised F1 values are not directly comparable without the protocol caveat above.
- Small-LI SSL pre-3h + raw + temporal-flow uses validated temporal-flow multiseed aggregate (same as Table 5).
- Small-HI SSL temporal-flow row uses validated single-seed strong-run protocol when available.
- P@100 = precision among the top 100 scored test transactions.
- R@100 = fraction of all positive test transactions recovered in the top 100 scored test transactions.
- Lift@100 = P@100 divided by the test-set positive rate for that dataset.
