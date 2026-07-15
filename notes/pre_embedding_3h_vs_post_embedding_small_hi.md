# pre_embedding_3h vs post_embedding_128 — Small-HI (gin_emlps_tds_asym_proj_8192neg_queue0_40ep_seed1)

- **checkpoint dirs:** post=`embeddings/gin_emlps_tds_asym_proj_8192neg_queue0_40ep_seed1`, pre=`embeddings/gin_emlps_tds_asym_proj_8192neg_queue0_40ep_seed1/pre_embedding_3h`
- **representation dims:** post_embedding_128 = 128, pre_embedding_3h = 198
- **no SSL retraining:** True  |  **paired:** True (inner-join on edge_id per split; identical rows/labels/order for both representations)
- **probe:** sklearn LogisticRegression (lbfgs), class_weight=model={0: 1.0000182882773443, 1: 6.275014431494497}, C=1.0, threshold=max_f1_on_val, seed=1

## embedding_only

| metric (test) | post_embedding_128 | pre_embedding_3h | winner |
|---|---|---|---|
| AUROC | 0.9487 | 0.9523 | pre_embedding_3h |
| AUPRC | 0.1978 | 0.2244 | pre_embedding_3h |
| F1 @ val-thr | 0.2922 | 0.2894 | post_embedding_128 |
| precision @ val-thr | 0.3109 | 0.2439 |  |
| recall @ val-thr | 0.2756 | 0.3557 |  |
| F1 @ 0.5 | 0.2926 | 0.3037 |  |
| recall@100 | 0.0366 | 0.0472 | pre_embedding_3h |
| recall@500 | 0.1453 | 0.1564 | pre_embedding_3h |
| recall@1000 | 0.2297 | 0.2464 | pre_embedding_3h |
| precision@100 | 0.5900 | 0.7600 |  |
| precision@500 | 0.4680 | 0.5040 |  |
| precision@1000 | 0.3700 | 0.3970 |  |
| lift@100 | 316.0767 | 407.1496 |  |
| lift@1000 | 198.2176 | 212.6821 |  |
| selected val threshold | 0.5086 | 0.4606 |  |

## embedding_plus_raw

| metric (test) | post_embedding_128 | pre_embedding_3h | winner |
|---|---|---|---|
| AUROC | 0.9605 | 0.9583 | post_embedding_128 |
| AUPRC | 0.3184 | 0.2737 | post_embedding_128 |
| F1 @ val-thr | 0.2762 | 0.3054 | pre_embedding_3h |
| precision @ val-thr | 0.1836 | 0.2413 |  |
| recall @ val-thr | 0.5568 | 0.4159 |  |
| F1 @ 0.5 | 0.1346 | 0.2041 |  |
| recall@100 | 0.0497 | 0.0497 | tie |
| recall@500 | 0.1974 | 0.1769 | post_embedding_128 |
| recall@1000 | 0.3011 | 0.2762 | post_embedding_128 |
| precision@100 | 0.8000 | 0.8000 |  |
| precision@500 | 0.6360 | 0.5700 |  |
| precision@1000 | 0.4850 | 0.4450 |  |
| lift@100 | 428.5785 | 428.5785 |  |
| lift@1000 | 259.8257 | 238.3968 |  |
| selected val threshold | 0.7966 | 0.7284 |  |
