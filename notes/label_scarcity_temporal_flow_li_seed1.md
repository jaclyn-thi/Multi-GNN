# Label-scarcity temporal-flow probe — Small-LI (`small_li_gin_emlps_tds_asym_proj_8192neg_queue0_20ep_seed1`)

- scarcity_seed=1
- embedding: `embeddings/small_li_gin_emlps_tds_asym_proj_8192neg_queue0_20ep_seed1/pre_embedding_3h`
- primary: pre-3h+raw+temporal-flow vs pre-3h+raw
- no SSL retraining / no embedding regeneration

| frac | n_lab | n_pos | B AUPRC | D AUPRC | ΔAUPRC | D F1 | D P@100 |
|-----:|------:|------:|-------:|-------:|-------:|-----:|--------:|
| 0.01 | 44318 | 20 | 0.0029 | 0.0029 | -0.0000 | 0.0096 | 0.0100 |
| 0.05 | 221590 | 100 | 0.0179 | 0.0349 | +0.0171 | 0.0664 | 0.2200 |
| 0.10 | 443179 | 199 | 0.0353 | 0.0720 | +0.0367 | 0.0826 | 0.3700 |
| 0.25 | 1107947 | 498 | 0.0655 | 0.1322 | +0.0668 | 0.1134 | 0.6300 |
| 0.50 | 2215894 | 996 | 0.0833 | 0.1528 | +0.0695 | 0.0611 | 0.6700 |
| 1.00 | 4431790 | 1993 | 0.0932 | 0.1565 | +0.0634 | 0.0486 | 0.7100 |
