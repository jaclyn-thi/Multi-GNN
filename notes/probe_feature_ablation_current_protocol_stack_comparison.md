# Current-protocol downstream stack comparison (focus modes)

Frozen embeddings; logistic regression; val max-F1 threshold; shared GIN class weights (`{'0': 1.0000182882773443, '1': 6.275014431494497}`).

| Run | Features | AUROC | AUPRC | F1 | Prec | Recall | F1@0.5 | Thr |
|-----|----------|------:|------:|---:|-----:|-------:|-------:|----:|
| GINe emlps+tds seed1 (20ep) | `raw+morph` | 0.9052 | 0.0658 | 0.1362 | 0.0867 | 0.3178 | 0.1323 | 0.5367 |
| GINe emlps+tds seed1 (20ep) | `embedding` | 0.9440 | 0.2133 | 0.2593 | 0.2314 | 0.2948 | 0.2573 | 0.5275 |
| GINe emlps+tds seed1 (20ep) | `embedding+raw+morph` | 0.9445 | 0.2755 | 0.2982 | 0.2313 | 0.4196 | 0.3271 | 0.3657 |
| GINe emlps+tds seed1 (40ep) | `raw+morph` | 0.9052 | 0.0658 | 0.1362 | 0.0867 | 0.3178 | 0.1323 | 0.5367 |
| GINe emlps+tds seed1 (40ep) | `embedding` | 0.9484 | 0.1992 | 0.2920 | 0.3097 | 0.2762 | 0.2918 | 0.5021 |
| GINe emlps+tds seed1 (40ep) | `embedding+raw+morph` | 0.9447 | 0.2644 | 0.2622 | 0.1799 | 0.4836 | 0.2484 | 0.5438 |
| GINe emlps+tds seed2 (40ep) | `raw+morph` | 0.9050 | 0.0654 | 0.1351 | 0.0859 | 0.3153 | 0.1318 | 0.5371 |
| GINe emlps+tds seed2 (40ep) | `embedding` | 0.9487 | 0.2419 | 0.3001 | 0.2786 | 0.3253 | 0.3051 | 0.3946 |
| GINe emlps+tds seed2 (40ep) | `embedding+raw+morph` | 0.9581 | 0.2194 | 0.2751 | 0.2105 | 0.3966 | 0.2832 | 0.4078 |
| FNF + emlps+tds seed1 | `raw+morph` | 0.9052 | 0.0658 | 0.1362 | 0.0867 | 0.3178 | 0.1323 | 0.5367 |
| FNF + emlps+tds seed1 | `embedding` | 0.9421 | 0.1786 | 0.2360 | 0.2049 | 0.2781 | 0.2385 | 0.4788 |
| FNF + emlps+tds seed1 | `embedding+raw+morph` | 0.9586 | 0.2763 | 0.3188 | 0.2597 | 0.4128 | 0.3031 | 0.5660 |
| FNF + emlps+tds seed2 | `raw+morph` | 0.9050 | 0.0654 | 0.1351 | 0.0859 | 0.3153 | 0.1318 | 0.5371 |
| FNF + emlps+tds seed2 | `embedding` | 0.9261 | 0.1366 | 0.2063 | 0.1712 | 0.2595 | 0.2180 | 0.4146 |
| FNF + emlps+tds seed2 | `embedding+raw+morph` | 0.9552 | 0.2425 | 0.2624 | 0.1890 | 0.4289 | 0.1700 | 0.7426 |
