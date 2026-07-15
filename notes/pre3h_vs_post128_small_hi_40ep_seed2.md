# pre_embedding_3h vs post_embedding_128 — Small-HI (gin_emlps_tds_asym_proj_8192neg_queue0_40ep_seed2)

- **checkpoint dirs:** post=`embeddings/gin_emlps_tds_asym_proj_8192neg_queue0_40ep_seed2`, pre=`embeddings/gin_emlps_tds_asym_proj_8192neg_queue0_40ep_seed2/pre_embedding_3h`
- **representation dims:** post_embedding_128 = 128, pre_embedding_3h = 198
- **no SSL retraining:** True  |  **paired:** True (inner-join on edge_id per split; identical rows/labels/order for both representations)
- **probe:** sklearn LogisticRegression (lbfgs), class_weight=model={0: 1.0000182882773443, 1: 6.275014431494497}, C=1.0, threshold=max_f1_on_val, seed=1

## embedding_only

| metric (test) | post_embedding_128 | pre_embedding_3h | winner |
|---|---|---|---|
| AUROC | 0.9487 | 0.9581 | pre_embedding_3h |
| AUPRC | 0.2449 | 0.2953 | pre_embedding_3h |
| F1 @ val-thr | 0.3040 | 0.3398 | pre_embedding_3h |
| precision @ val-thr | 0.2799 | 0.2872 |  |
| recall @ val-thr | 0.3327 | 0.4159 |  |
| F1 @ 0.5 | 0.3095 | 0.3477 |  |
| recall@100 | 0.0497 | 0.0515 | pre_embedding_3h |
| recall@500 | 0.1831 | 0.1868 | pre_embedding_3h |
| recall@1000 | 0.2471 | 0.2775 | pre_embedding_3h |
| precision@100 | 0.8000 | 0.8300 |  |
| precision@500 | 0.5900 | 0.6020 |  |
| precision@1000 | 0.3980 | 0.4470 |  |
| lift@100 | 428.5110 | 444.5801 |  |
| lift@1000 | 213.1842 | 239.4305 |  |
| selected val threshold | 0.3837 | 0.4656 |  |

## embedding_plus_raw

| metric (test) | post_embedding_128 | pre_embedding_3h | winner |
|---|---|---|---|
| AUROC | 0.9551 | 0.9604 | pre_embedding_3h |
| AUPRC | 0.2838 | 0.3212 | pre_embedding_3h |
| F1 @ val-thr | 0.3429 | 0.3443 | pre_embedding_3h |
| precision @ val-thr | 0.3112 | 0.2793 |  |
| recall @ val-thr | 0.3818 | 0.4488 |  |
| F1 @ 0.5 | 0.3415 | 0.3238 |  |
| recall@100 | 0.0490 | 0.0521 | pre_embedding_3h |
| recall@500 | 0.1955 | 0.1986 | pre_embedding_3h |
| recall@1000 | 0.2744 | 0.3042 | pre_embedding_3h |
| precision@100 | 0.7900 | 0.8400 |  |
| precision@500 | 0.6300 | 0.6400 |  |
| precision@1000 | 0.4420 | 0.4900 |  |
| lift@100 | 423.1546 | 449.9365 |  |
| lift@1000 | 236.7523 | 262.4630 |  |
| selected val threshold | 0.5303 | 0.5559 |  |
