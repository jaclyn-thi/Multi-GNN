# Table 2 — Main Small-HI results

| Method | Representation | Features | AUROC | AUPRC | F1 | P@100 | R@100 | Lift@100 | Caveat |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Raw features only | — | raw | 0.860 | 0.009 | 0.009 | — | — | — | val-tuned F1; no SSL |
| Raw + morphology | — | raw+morph | 0.905 | 0.066 | 0.136 | 0.230 | 0.014 | 123 | val-tuned F1; no SSL |
| SSL post-128 | post-128 | embedding | 0.949 | 0.245 | 0.304 | 0.800 | 0.050 | 429 | val-tuned F1 |
| SSL pre-3h | pre-3h | embedding | 0.958 | 0.295 | 0.340 | 0.830 | 0.052 | 445 | val-tuned F1 |
| SSL post-128 + raw | post-128 | embedding+raw | 0.955 | 0.284 | 0.343 | 0.790 | 0.049 | 423 | val-tuned F1 |
| SSL pre-3h + raw | pre-3h | embedding+raw | 0.960 | 0.321 | 0.344 | 0.840 | 0.052 | 450 | val-tuned F1 |
| SSL pre-3h + raw + temporal-flow | pre-3h | embedding+raw+temporal_flow_causal | 0.979 | **0.501** | 0.465 | 0.940 | 0.058 | 504 | val-tuned F1; validated temporal-flow stack |
| Legacy supervised GIN (100ep seed1) | logits | in-GNN end-to-end | 0.984 | 0.639 | 0.539 | 0.990 | 0.061 | 530 | paper_argmax F1; paper_argmax F1; supervised CE; not comparable to SSL val-tuned F1 |

**Notes:**
- Small-HI pre/post rows use paired strong-run protocol (results/diagnostics/pre3h_strong_run_comparison.json).
- F1 for SSL rows is validation-tuned; raw-feature rows use val-tuned probe.
- Legacy supervised row uses end-to-end labeled training and paper_argmax F1 (not val-tuned).
- Temporal-flow stack included only when validated or with --include_provisional.
- Contrastive-method variants such as FNF are reported in the appendix.
- P@100 = precision among the top 100 scored test transactions.
- R@100 = fraction of all positive test transactions recovered in the top 100 scored test transactions.
- Lift@100 = P@100 divided by the test-set positive rate for that dataset.
