# Label-scarcity temporal-flow probe — Small-HI (`gin_emlps_tds_asym_proj_8192neg_queue0_40ep_seed2`)

- scarcity_seed=1
- embedding: `embeddings/gin_emlps_tds_asym_proj_8192neg_queue0_40ep_seed2/pre_embedding_3h`
- primary: pre-3h+raw+temporal-flow vs pre-3h+raw
- no SSL retraining / no embedding regeneration

| frac | n_lab | n_pos | B AUPRC | D AUPRC | ΔAUPRC | D F1 | D P@100 |
|-----:|------:|------:|-------:|-------:|-------:|-----:|--------:|
| 0.01 | 32482 | 25 | 0.0575 | 0.1394 | +0.0819 | 0.2061 | 0.6600 |
| 0.05 | 162412 | 126 | 0.1431 | 0.2998 | +0.1567 | 0.3567 | 0.8100 |
| 0.10 | 324825 | 253 | 0.3065 | 0.4301 | +0.1236 | 0.4188 | 0.9800 |
| 0.25 | 812063 | 632 | 0.2984 | 0.4753 | +0.1769 | 0.4333 | 0.9000 |
| 0.50 | 1624127 | 1265 | 0.3064 | 0.4962 | +0.1898 | 0.3942 | 0.9600 |
| 1.00 | 3248254 | 2530 | 0.3205 | 0.5006 | +0.1801 | 0.4649 | 0.9400 |
