# Small-LI FNF Current-Protocol Comparison

Updated from `results/diagnostics/probe_feature_ablation_small_li_fnf_current_protocol_seed1.json` and the completed alert-budget metrics.

## Thesis-Relevant Takeaway

FNF transfer from Small-HI to Small-LI is mixed, not robust. It slightly helps embedding-only F1 and gives the best Small-LI full-stack thresholded F1 in the alert-budget run, but it does not consistently improve AUPRC or practical alert precision. Treat FNF as promising but unstable rather than a thesis-safe universal improvement.

## Plain vs FNF Feature Stacks (`cw=model`)

| Features | Plain AUPRC | Plain F1 | FNF AUPRC | FNF F1 | Interpretation |
|---|---:|---:|---:|---:|---|
| `raw+morph` | 0.0161 | 0.0566 | 0.0161 | 0.0566 | No embedding; unchanged baseline. |
| `embedding` | 0.0166 | 0.0522 | 0.0164 | 0.0583 | Small F1 gain, ranking nearly unchanged. |
| `embedding+raw` | 0.0272 | 0.0757 | 0.0259 | 0.0668 | Plain is better for F1/AUPRC. |
| `embedding+raw+morph` | 0.0391 | 0.0555 | 0.0358 | 0.0889 | FNF improves F1 but not AUPRC/alerts. |

## Alert-Budget Check

| Run | Features | Weight | AUROC | AUPRC | F1 | F1@0.5 | P@100 | R@100 | lift@100 | P@500 | R@500 | lift@500 | P@1000 | R@1000 | lift@1000 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Small-LI GINe emlps+tds seed1 (20ep) | `embedding+raw+morph` | model | 0.9238 | 0.0356 | 0.0644 | 0.0756 | 0.2300 | 0.0287 | 336.5 | 0.1160 | 0.0723 | 169.7 | 0.0900 | 0.1122 | 131.7 |
| Small-LI FNF + emlps+tds seed1 (20ep) | `embedding` | model | 0.8995 | 0.0165 | 0.0588 | 0.0431 | 0.1300 | 0.0162 | 190.2 | 0.0840 | 0.0524 | 122.9 | 0.0630 | 0.0786 | 92.2 |
| Small-LI FNF + emlps+tds seed1 (20ep) | `embedding+raw` | model | 0.9110 | 0.0256 | 0.0667 | 0.0747 | 0.1900 | 0.0237 | 277.9 | 0.1040 | 0.0648 | 152.1 | 0.0710 | 0.0885 | 103.9 |
| Small-LI FNF + emlps+tds seed1 (20ep) | `embedding+raw+morph` | model | 0.9193 | 0.0363 | 0.0916 | 0.0914 | 0.1700 | 0.0212 | 248.7 | 0.1000 | 0.0623 | 146.3 | 0.0820 | 0.1022 | 120.0 |

## Interpretation

- Embedding-only FNF has slightly better F1 than plain embedding (0.0583 vs 0.0522), but the gain is small.
- Full-stack FNF reaches F1 0.0916, but plain GINe full-stack has better P@100/P@500/P@1000.
- Conservative claim: FNF remains an interesting variant, but current Small-LI evidence is metric-dependent and not robust enough to claim clean transfer.

Artifacts: `results/diagnostics/probe_feature_ablation_small_li_fnf_current_protocol_seed1.json`, `results/diagnostics/alert_budget_metrics_current_protocol.json`.
