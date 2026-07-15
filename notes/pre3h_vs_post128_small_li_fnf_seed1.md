# pre_embedding_3h vs post_embedding_128 — Small-LI (small_li_fnf_emlps_tds_asym_proj_8192neg_queue0_20ep_seed1)

- **checkpoint dirs:** post=`embeddings/small_li_fnf_emlps_tds_asym_proj_8192neg_queue0_20ep_seed1`, pre=`embeddings/small_li_fnf_emlps_tds_asym_proj_8192neg_queue0_20ep_seed1/pre_embedding_3h`
- **representation dims:** post_embedding_128 = 128, pre_embedding_3h = 198
- **no SSL retraining:** True  |  **paired:** True (inner-join on edge_id per split; identical rows/labels/order for both representations)
- **probe:** sklearn LogisticRegression (lbfgs), class_weight=model={0: 1.0000182882773443, 1: 6.275014431494497}, C=1.0, threshold=max_f1_on_val, seed=1

## embedding_only

| metric (test) | post_embedding_128 | pre_embedding_3h | winner |
|---|---|---|---|
| AUROC | 0.8995 | 0.9276 | pre_embedding_3h |
| AUPRC | 0.0166 | 0.0419 | pre_embedding_3h |
| F1 @ val-thr | 0.0589 | 0.0809 | pre_embedding_3h |
| precision @ val-thr | 0.0436 | 0.0581 |  |
| recall @ val-thr | 0.0910 | 0.1334 |  |
| F1 @ 0.5 | 0.0428 | 0.1028 |  |
| recall@100 | 0.0162 | 0.0349 | pre_embedding_3h |
| recall@500 | 0.0511 | 0.0773 | pre_embedding_3h |
| recall@1000 | 0.0786 | 0.1097 | pre_embedding_3h |
| precision@100 | 0.1300 | 0.2800 |  |
| precision@500 | 0.0820 | 0.1240 |  |
| precision@1000 | 0.0630 | 0.0880 |  |
| lift@100 | 190.1744 | 409.6065 |  |
| lift@1000 | 92.1615 | 128.7335 |  |
| selected val threshold | 0.2455 | 0.3446 |  |

## embedding_plus_raw

| metric (test) | post_embedding_128 | pre_embedding_3h | winner |
|---|---|---|---|
| AUROC | 0.9112 | 0.9321 | pre_embedding_3h |
| AUPRC | 0.0255 | 0.0587 | pre_embedding_3h |
| F1 @ val-thr | 0.0662 | 0.0820 | pre_embedding_3h |
| precision @ val-thr | 0.0439 | 0.0506 |  |
| recall @ val-thr | 0.1347 | 0.2170 |  |
| F1 @ 0.5 | 0.0747 | 0.0704 |  |
| recall@100 | 0.0224 | 0.0424 | pre_embedding_3h |
| recall@500 | 0.0661 | 0.0898 | pre_embedding_3h |
| recall@1000 | 0.0873 | 0.1234 | pre_embedding_3h |
| precision@100 | 0.1800 | 0.3400 |  |
| precision@500 | 0.1060 | 0.1440 |  |
| precision@1000 | 0.0700 | 0.0990 |  |
| lift@100 | 263.3185 | 497.3793 |  |
| lift@1000 | 102.4016 | 144.8251 |  |
| selected val threshold | 0.4160 | 0.5708 |  |

## embedding_plus_raw_morph

| metric (test) | post_embedding_128 | pre_embedding_3h | winner |
|---|---|---|---|
| AUROC | 0.9191 | 0.9430 | pre_embedding_3h |
| AUPRC | 0.0360 | 0.0547 | pre_embedding_3h |
| F1 @ val-thr | 0.0898 | 0.0686 | post_embedding_128 |
| precision @ val-thr | 0.0568 | 0.0377 |  |
| recall @ val-thr | 0.2145 | 0.3766 |  |
| F1 @ 0.5 | 0.0915 | 0.0703 |  |
| recall@100 | 0.0212 | 0.0312 | pre_embedding_3h |
| recall@500 | 0.0611 | 0.1022 | pre_embedding_3h |
| recall@1000 | 0.1022 | 0.1309 | pre_embedding_3h |
| precision@100 | 0.1700 | 0.2500 |  |
| precision@500 | 0.0980 | 0.1640 |  |
| precision@1000 | 0.0820 | 0.1050 |  |
| lift@100 | 248.6897 | 365.7201 |  |
| lift@1000 | 119.9562 | 153.6024 |  |
| selected val threshold | 0.2612 | 0.4548 |  |
