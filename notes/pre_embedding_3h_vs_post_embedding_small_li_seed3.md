# pre_embedding_3h vs post_embedding_128 — Small-LI (small_li_gin_emlps_tds_asym_proj_8192neg_queue0_20ep_seed3)

- **checkpoint dirs:** post=`embeddings/small_li_gin_emlps_tds_asym_proj_8192neg_queue0_20ep_seed3`, pre=`embeddings/small_li_gin_emlps_tds_asym_proj_8192neg_queue0_20ep_seed3/pre_embedding_3h`
- **representation dims:** post_embedding_128 = 128, pre_embedding_3h = 198
- **no SSL retraining:** True  |  **paired:** True (inner-join on edge_id per split; identical rows/labels/order for both representations)
- **probe:** sklearn LogisticRegression (lbfgs), class_weight=model={0: 1.0000182882773443, 1: 6.275014431494497}, C=1.0, threshold=max_f1_on_val, seed=1

## embedding_only

| metric (test) | post_embedding_128 | pre_embedding_3h | winner |
|---|---|---|---|
| AUROC | 0.8960 | 0.9258 | pre_embedding_3h |
| AUPRC | 0.0242 | 0.0495 | pre_embedding_3h |
| F1 @ val-thr | 0.0802 | 0.1135 | pre_embedding_3h |
| precision @ val-thr | 0.0921 | 0.0889 |  |
| recall @ val-thr | 0.0711 | 0.1571 |  |
| F1 @ 0.5 | 0.0641 | 0.1368 |  |
| recall@100 | 0.0224 | 0.0362 | pre_embedding_3h |
| recall@500 | 0.0586 | 0.1110 | pre_embedding_3h |
| recall@1000 | 0.0898 | 0.1434 | pre_embedding_3h |
| precision@100 | 0.1800 | 0.2900 |  |
| precision@500 | 0.0940 | 0.1780 |  |
| precision@1000 | 0.0720 | 0.1150 |  |
| lift@100 | 263.3155 | 424.2306 |  |
| lift@1000 | 105.3262 | 168.2294 |  |
| selected val threshold | 0.3836 | 0.3698 |  |

## embedding_plus_raw

| metric (test) | post_embedding_128 | pre_embedding_3h | winner |
|---|---|---|---|
| AUROC | 0.9146 | 0.9344 | pre_embedding_3h |
| AUPRC | 0.0561 | 0.0793 | pre_embedding_3h |
| F1 @ val-thr | 0.0693 | 0.0524 | post_embedding_128 |
| precision @ val-thr | 0.0415 | 0.0281 |  |
| recall @ val-thr | 0.2082 | 0.3778 |  |
| F1 @ 0.5 | 0.0533 | 0.0340 |  |
| recall@100 | 0.0449 | 0.0536 | pre_embedding_3h |
| recall@500 | 0.0948 | 0.1397 | pre_embedding_3h |
| recall@1000 | 0.1309 | 0.1833 | pre_embedding_3h |
| precision@100 | 0.3600 | 0.4300 |  |
| precision@500 | 0.1520 | 0.2240 |  |
| precision@1000 | 0.1050 | 0.1470 |  |
| lift@100 | 526.6311 | 629.0316 |  |
| lift@1000 | 153.6007 | 215.0410 |  |
| selected val threshold | 0.6041 | 0.7392 |  |
