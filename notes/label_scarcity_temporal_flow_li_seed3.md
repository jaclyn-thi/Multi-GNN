# Label-scarcity temporal-flow probe — Small-LI (`small_li_gin_emlps_tds_asym_proj_8192neg_queue0_20ep_seed3`)

- scarcity_seed=1
- embedding: `embeddings/small_li_gin_emlps_tds_asym_proj_8192neg_queue0_20ep_seed3/pre_embedding_3h`
- primary: pre-3h+raw+temporal-flow vs pre-3h+raw
- no SSL retraining / no embedding regeneration

| frac | n_lab | n_pos | B AUPRC | D AUPRC | ΔAUPRC | D F1 | D P@100 |
|-----:|------:|------:|-------:|-------:|-------:|-----:|--------:|
| 0.01 | 44318 | 20 | 0.0043 | 0.0047 | +0.0004 | 0.0216 | 0.0000 |
| 0.05 | 221588 | 100 | 0.0781 | 0.1292 | +0.0510 | 0.1152 | 0.6100 |
| 0.10 | 443176 | 199 | 0.0223 | 0.0458 | +0.0235 | 0.0547 | 0.2900 |
| 0.25 | 1107940 | 498 | 0.0580 | 0.1095 | +0.0515 | 0.0813 | 0.6000 |
| 0.50 | 2215881 | 996 | 0.0817 | 0.1342 | +0.0525 | 0.0610 | 0.6500 |
| 1.00 | 4431763 | 1993 | 0.0783 | 0.1341 | +0.0558 | 0.0863 | 0.6100 |
