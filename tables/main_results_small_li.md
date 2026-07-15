# Table 3 — Main Small-LI results

| Method | Representation | Features | AUROC | AUPRC | F1 | P@100 | R@100 | Lift@100 | Caveat |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SSL post-128 | post-128 | embedding | 0.888 ± 0.016 | 0.014 ± 0.010 | 0.046 ± 0.037 | 0.120 ± 0.060 | 0.015 ± 0.007 | 176 ± 88 | frozen probe; val-tuned F1; mean ± sample SD (n=3) |
| SSL pre-3h | pre-3h | embedding | 0.919 ± 0.009 | 0.039 ± 0.016 | 0.089 ± 0.026 | 0.220 ± 0.082 | 0.027 ± 0.010 | 322 ± 120 | frozen probe; val-tuned F1; mean ± sample SD (n=3) |
| SSL post-128 + raw | post-128 | embedding+raw | 0.904 ± 0.014 | 0.032 ± 0.021 | 0.039 ± 0.028 | 0.227 ± 0.122 | 0.028 ± 0.015 | 332 ± 179 | frozen probe; val-tuned F1; mean ± sample SD (n=3) |
| SSL pre-3h + raw | pre-3h | embedding+raw | 0.926 ± 0.013 | 0.061 ± 0.034 | 0.054 ± 0.007 | 0.343 ± 0.159 | 0.043 ± 0.020 | 502 ± 232 | frozen probe; val-tuned F1; mean ± sample SD (n=3) |
| SSL pre-3h + raw + temporal-flow | pre-3h | embedding+raw+temporal_flow_causal | 0.947 ± 0.006 | 0.128 ± 0.027 | 0.092 ± 0.029 | 0.600 ± 0.056 | 0.075 ± 0.007 | 878 ± 81 | val-tuned F1; mean ± sample SD (n=3) |
| Legacy supervised GIN (100ep seed1) | logits | in-GNN end-to-end | 0.959 | 0.292 | 0.357 | 0.970 | 0.121 | 1419 | paper_argmax F1 |

**Notes:**
- SSL multiseed rows: mean ± sample SD (ddof=1) over seeds 1–3; frozen linear probe with validation-tuned thresholds.
- SSL pre-3h + raw + temporal-flow uses validated temporal-flow multiseed aggregate (same as Table 5).
- Supervised row uses paper_argmax F1 from results/diagnostics/eval_small_li_legacy_supervised_gin_emlps_tds_100ep_seed1.json; not directly comparable to SSL F1 without footnote.
- P@100 = precision among the top 100 scored test transactions.
- R@100 = fraction of all positive test transactions recovered in the top 100 scored test transactions.
- Lift@100 = P@100 divided by the test-set positive rate for that dataset.
