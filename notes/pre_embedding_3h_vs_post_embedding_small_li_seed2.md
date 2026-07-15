# pre_embedding_3h vs post_embedding_128 — Small-LI (small_li_gin_emlps_tds_asym_proj_8192neg_queue0_20ep_seed2)

- **checkpoint dirs:** post=`embeddings/small_li_gin_emlps_tds_asym_proj_8192neg_queue0_20ep_seed2`, pre=`embeddings/small_li_gin_emlps_tds_asym_proj_8192neg_queue0_20ep_seed2/pre_embedding_3h`
- **representation dims:** post_embedding_128 = 128, pre_embedding_3h = 198
- **no SSL retraining:** True  |  **paired:** True (inner-join on edge_id per split; identical rows/labels/order for both representations)
- **probe:** sklearn LogisticRegression (lbfgs), class_weight=model={0: 1.0000182882773443, 1: 6.275014431494497}, C=1.0, threshold=max_f1_on_val, seed=1

## embedding_only

| metric (test) | post_embedding_128 | pre_embedding_3h | winner |
|---|---|---|---|
| AUROC | 0.8702 | 0.9094 | pre_embedding_3h |
| AUPRC | 0.0051 | 0.0202 | pre_embedding_3h |
| F1 @ val-thr | 0.0067 | 0.0611 | pre_embedding_3h |
| precision @ val-thr | 0.0034 | 0.0396 |  |
| recall @ val-thr | 0.1509 | 0.1334 |  |
| F1 @ 0.5 | 0.0246 | 0.0584 |  |
| recall@100 | 0.0075 | 0.0162 | pre_embedding_3h |
| recall@500 | 0.0162 | 0.0374 | pre_embedding_3h |
| recall@1000 | 0.0287 | 0.0648 | pre_embedding_3h |
| precision@100 | 0.0600 | 0.1300 |  |
| precision@500 | 0.0260 | 0.0600 |  |
| precision@1000 | 0.0230 | 0.0520 |  |
| lift@100 | 87.7703 | 190.1689 |  |
| lift@1000 | 33.6453 | 76.0676 |  |
| selected val threshold | 0.1575 | 0.3498 |  |

## embedding_plus_raw

| metric (test) | post_embedding_128 | pre_embedding_3h | winner |
|---|---|---|---|
| AUROC | 0.8887 | 0.9112 | pre_embedding_3h |
| AUPRC | 0.0161 | 0.0224 | pre_embedding_3h |
| F1 @ val-thr | 0.0125 | 0.0624 | pre_embedding_3h |
| precision @ val-thr | 0.0064 | 0.0400 |  |
| recall @ val-thr | 0.2481 | 0.1409 |  |
| F1 @ 0.5 | 0.0278 | 0.0663 |  |
| recall@100 | 0.0150 | 0.0200 | pre_embedding_3h |
| recall@500 | 0.0212 | 0.0411 | pre_embedding_3h |
| recall@1000 | 0.0486 | 0.0711 | pre_embedding_3h |
| precision@100 | 0.1200 | 0.1600 |  |
| precision@500 | 0.0340 | 0.0660 |  |
| precision@1000 | 0.0390 | 0.0570 |  |
| lift@100 | 175.5405 | 234.0541 |  |
| lift@1000 | 57.0507 | 83.3818 |  |
| selected val threshold | 0.3683 | 0.3968 |  |
