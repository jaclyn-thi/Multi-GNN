# Alert-Budget Metrics (Current Protocol)

Updated from the completed resume output `results/diagnostics/alert_budget_metrics_current_protocol.json`.

Status: complete. Small-HI has **10/10** cells; Small-LI has **7/7** cells. Resume job `17209094` skipped completed cells and finished the three remaining Small-LI cells.

## Thesis-Relevant Takeaway

Alert-budget metrics give a more realistic and generally more favorable story than thresholded F1 alone. Small-HI is operationally strong: the best current configuration reaches P@500 0.6380 with lift@500 341.8. Small-LI remains much harder in absolute precision, but it still shows very large enrichment above tiny prevalence: the best P@100 is 0.2300, lift@100 336.5.

## Key Alert Rows

| Run | Features | Weight | AUROC | AUPRC | F1 | F1@0.5 | P@100 | R@100 | lift@100 | P@500 | R@500 | lift@500 | P@1000 | R@1000 | lift@1000 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Small-HI GINe emlps+tds seed1 (20ep) | `embedding` | model | 0.9440 | 0.2133 | 0.2593 | 0.2573 | 0.8500 | 0.0528 | 455.4 | 0.5200 | 0.1614 | 278.6 | 0.3450 | 0.2142 | 184.8 |
| Small-HI GINe emlps+tds seed1 (20ep) | `embedding+raw` | model | 0.9489 | 0.2445 | 0.2741 | 0.2678 | 0.8400 | 0.0521 | 450.0 | 0.5520 | 0.1713 | 295.7 | 0.4080 | 0.2533 | 218.6 |
| Small-HI GINe emlps+tds seed1 (20ep) | `embedding+raw+morph` | model | 0.9445 | 0.2755 | 0.2982 | 0.3271 | 0.7700 | 0.0478 | 412.5 | 0.6200 | 0.1924 | 332.1 | 0.4520 | 0.2806 | 242.1 |
| Small-HI FNF + emlps+tds seed1 | `embedding` | model | 0.9421 | 0.1786 | 0.2360 | 0.2385 | 0.6900 | 0.0428 | 369.6 | 0.4380 | 0.1359 | 234.6 | 0.3160 | 0.1962 | 169.3 |
| Small-HI FNF + emlps+tds seed1 | `embedding+raw` | model | 0.9517 | 0.2232 | 0.2561 | 0.2614 | 0.7600 | 0.0472 | 407.1 | 0.4980 | 0.1546 | 266.8 | 0.3770 | 0.2340 | 202.0 |
| Small-HI FNF + emlps+tds seed1 | `embedding+raw+morph` | model | 0.9586 | 0.2763 | 0.3188 | 0.3031 | 0.8000 | 0.0497 | 428.6 | 0.5740 | 0.1782 | 307.5 | 0.4380 | 0.2719 | 234.6 |
| Small-HI GINe emlps+tds seed2 (40ep) | `embedding` | model | 0.9482 | 0.2437 | 0.3001 | 0.3050 | 0.8100 | 0.0503 | 434.0 | 0.5760 | 0.1788 | 308.6 | 0.3920 | 0.2433 | 210.0 |
| Small-HI GINe emlps+tds seed2 (40ep) | `embedding+raw` | model | 0.9553 | 0.2883 | 0.3464 | 0.3392 | 0.8000 | 0.0497 | 428.6 | 0.6380 | 0.1980 | 341.8 | 0.4520 | 0.2806 | 242.2 |
| Small-HI GINe emlps+tds seed2 (40ep) | `embedding+raw+morph` | model | 0.9580 | 0.2154 | 0.2767 | 0.2786 | 0.7000 | 0.0435 | 375.0 | 0.4540 | 0.1409 | 243.2 | 0.3470 | 0.2154 | 185.9 |
| Small-HI GINe emlps+tds seed1 (20ep) | `raw+morph` | model | 0.9052 | 0.0658 | 0.1362 | 0.1323 | 0.2300 | 0.0143 | 123.2 | 0.1880 | 0.0583 | 100.7 | 0.1530 | 0.0950 | 82.0 |
| Small-LI GINe emlps+tds seed1 (20ep) | `embedding` | model | 0.8987 | 0.0139 | 0.0525 | 0.0443 | 0.1200 | 0.0150 | 175.5 | 0.0580 | 0.0362 | 84.8 | 0.0430 | 0.0536 | 62.9 |
| Small-LI GINe emlps+tds seed1 (20ep) | `embedding+raw` | model | 0.9104 | 0.0264 | 0.0450 | 0.0806 | 0.1900 | 0.0237 | 277.9 | 0.1060 | 0.0661 | 155.1 | 0.0750 | 0.0935 | 109.7 |
| Small-LI GINe emlps+tds seed1 (20ep) | `embedding+raw+morph` | model | 0.9238 | 0.0356 | 0.0644 | 0.0756 | 0.2300 | 0.0287 | 336.5 | 0.1160 | 0.0723 | 169.7 | 0.0900 | 0.1122 | 131.7 |
| Small-LI FNF + emlps+tds seed1 (20ep) | `embedding` | model | 0.8995 | 0.0165 | 0.0588 | 0.0431 | 0.1300 | 0.0162 | 190.2 | 0.0840 | 0.0524 | 122.9 | 0.0630 | 0.0786 | 92.2 |
| Small-LI FNF + emlps+tds seed1 (20ep) | `embedding+raw` | model | 0.9110 | 0.0256 | 0.0667 | 0.0747 | 0.1900 | 0.0237 | 277.9 | 0.1040 | 0.0648 | 152.1 | 0.0710 | 0.0885 | 103.9 |
| Small-LI FNF + emlps+tds seed1 (20ep) | `embedding+raw+morph` | model | 0.9193 | 0.0363 | 0.0916 | 0.0914 | 0.1700 | 0.0212 | 248.7 | 0.1000 | 0.0623 | 146.3 | 0.0820 | 0.1022 | 120.0 |
| Small-LI GINe emlps+tds seed1 (20ep) | `raw+morph` | model | 0.8576 | 0.0161 | 0.0566 | 0.0497 | 0.0900 | 0.0112 | 131.7 | 0.0640 | 0.0399 | 93.6 | 0.0500 | 0.0623 | 73.1 |

## Interpretation

- Small-HI: best practical alerting is GINe seed2 `embedding+raw` for P@500/AUPRC/F1, while GINe seed1 embedding-only is strongest at P@100.
- Small-LI: `embedding+raw+morph` is the strongest practical GINe alerting configuration, with P@100 0.2300, P@500 0.1160, and P@1000 0.0900. FNF full-stack has the best thresholded F1 (0.0916) but weaker P@100/P@500 than plain GINe full-stack.
- Embedding-only is not enough for the best Small-LI alerting result. Adding raw and morphology features improves ranking and enrichment.
- Alert budgets avoid overclaiming from threshold-tuned F1. They show Small-LI is hard in absolute precision while still delivering meaningful enrichment over base prevalence.

Artifacts: `results/diagnostics/alert_budget_metrics_small_hi.json`, `results/diagnostics/alert_budget_metrics_small_li.json`, `results/diagnostics/alert_budget_metrics_current_protocol.json`.
