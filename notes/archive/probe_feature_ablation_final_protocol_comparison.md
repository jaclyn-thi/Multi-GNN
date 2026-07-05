# Final-protocol probe feature ablation comparison

> **Superseded.** Use [`probe_feature_ablation_current_protocol_comparison.md`](../probe_feature_ablation_current_protocol_comparison.md) (expanded 6-mode table with 40 ep seeds). This file is a Jun 26 snapshot kept for provenance; "final protocol" is legacy naming for what is now called "current protocol".

Frozen embeddings; logistic regression; val max-F1 threshold. Probe policy: `--class_weight model --model gin`.

| Run | Features | AUROC | AUPRC | F1 | Prec | Recall | F1@0.5 | Thr |
|-----|----------|------:|------:|---:|-----:|-------:|-------:|----:|
| emlps+tds baseline | `embedding` | 0.9440 | 0.2133 | 0.2593 | 0.2314 | 0.2948 | 0.2573 | 0.5275 |
| emlps+tds baseline | `embedding+raw` | 0.9489 | 0.2445 | 0.2741 | 0.2117 | 0.3886 | 0.2678 | 0.5212 |
| emlps+tds baseline | `embedding+raw+morph` | 0.9445 | 0.2755 | 0.2982 | 0.2313 | 0.4196 | 0.3271 | 0.3657 |
| FNF + emlps+tds | `embedding` | 0.9421 | 0.1786 | 0.2360 | 0.2049 | 0.2781 | 0.2385 | 0.4788 |
| FNF + emlps+tds | `embedding+raw` | 0.9517 | 0.2232 | 0.2561 | 0.1855 | 0.4134 | 0.2614 | 0.4876 |
| FNF + emlps+tds | `embedding+raw+morph` | 0.9586 | 0.2763 | 0.3188 | 0.2597 | 0.4128 | 0.3031 | 0.5660 |
| degree-aware + emlps+tds | `embedding` | 0.9257 | 0.1529 | 0.2402 | 0.2344 | 0.2464 | 0.2462 | 0.4101 |
| degree-aware + emlps+tds | `embedding+raw` | 0.9450 | 0.2376 | 0.2383 | 0.1676 | 0.4122 | 0.2649 | 0.4464 |
| degree-aware + emlps+tds | `embedding+raw+morph` | 0.9574 | 0.2529 | 0.2906 | 0.2183 | 0.4345 | 0.2073 | 0.7035 |
