# Label-scarcity temporal-flow probe — Small-LI (`small_li_gin_emlps_tds_asym_proj_8192neg_queue0_20ep_seed2`)

- scarcity_seed=1
- embedding: `embeddings/small_li_gin_emlps_tds_asym_proj_8192neg_queue0_20ep_seed2/pre_embedding_3h`
- primary: pre-3h+raw+temporal-flow vs pre-3h+raw
- no SSL retraining / no embedding regeneration

| frac | n_lab | n_pos | B AUPRC | D AUPRC | ΔAUPRC | D F1 | D P@100 |
|-----:|------:|------:|-------:|-------:|-------:|-----:|--------:|
| 0.01 | 44318 | 20 | 0.0035 | 0.0042 | +0.0008 | 0.0219 | 0.0100 |
| 0.05 | 221589 | 100 | 0.0116 | 0.0391 | +0.0275 | 0.0199 | 0.2600 |
| 0.10 | 443177 | 199 | 0.0106 | 0.0382 | +0.0275 | 0.0825 | 0.2600 |
| 0.25 | 1107944 | 498 | 0.0182 | 0.0761 | +0.0579 | 0.1238 | 0.4600 |
| 0.50 | 2215888 | 996 | 0.0177 | 0.0760 | +0.0583 | 0.1144 | 0.4500 |
| 1.00 | 4431776 | 1993 | 0.0232 | 0.0986 | +0.0754 | 0.1266 | 0.5300 |
