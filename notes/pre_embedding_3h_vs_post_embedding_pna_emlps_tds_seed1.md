**PNA interpretation:** `pre_embedding_3h` is **60-d** and `post_embedding` is **128-d**. The embedding head is `Linear(60, 128)` — an **expansion**, not the GIN-style 198→128 compression. Do not assume the GIN pre-3h ranking advantage transfers.

# pre_embedding_3h vs post_embedding_128 — Small-HI (pna_emlps_tds_asym_proj_8192neg_queue0_20ep_seed1)

- **checkpoint dirs:** post=`embeddings/pna_emlps_tds_asym_proj_8192neg_queue0_20ep_seed1`, pre=`embeddings/pna_emlps_tds_asym_proj_8192neg_queue0_20ep_seed1/pre_embedding_3h`
- **representation dims:** post_embedding_128 = 128, pre_embedding_3h = 60
- **no SSL retraining:** True  |  **paired:** True (inner-join on edge_id per split; identical rows/labels/order for both representations)
- **probe:** sklearn LogisticRegression (lbfgs), class_weight=model={0: 1.0000182882773443, 1: 6.275014431494497}, C=1.0, threshold=max_f1_on_val, seed=1

## embedding_only

| metric (test) | post_embedding_128 | pre_embedding_3h | winner |
|---|---|---|---|
| AUROC | 0.9456 | 0.9469 | pre_embedding_3h |
| AUPRC | 0.1119 | 0.1190 | pre_embedding_3h |
| F1 @ val-thr | 0.2083 | 0.2141 | pre_embedding_3h |
| precision @ val-thr | 0.1716 | 0.1878 |  |
| recall @ val-thr | 0.2651 | 0.2489 |  |
| F1 @ 0.5 | 0.1454 | 0.1552 |  |
| recall@100 | 0.0205 | 0.0199 | post_embedding_128 |
| recall@500 | 0.0857 | 0.0900 | pre_embedding_3h |
| recall@1000 | 0.1397 | 0.1533 | pre_embedding_3h |
| precision@100 | 0.3300 | 0.3200 |  |
| precision@500 | 0.2760 | 0.2900 |  |
| precision@1000 | 0.2250 | 0.2470 |  |
| lift@100 | 176.7917 | 171.4344 |  |
| lift@1000 | 120.5398 | 132.3259 |  |
| selected val threshold | 0.2743 | 0.2997 |  |

## embedding_plus_raw

| metric (test) | post_embedding_128 | pre_embedding_3h | winner |
|---|---|---|---|
| AUROC | 0.9596 | 0.9592 | post_embedding_128 |
| AUPRC | 0.2639 | 0.2623 | post_embedding_128 |
| F1 @ val-thr | 0.0320 | 0.0421 | pre_embedding_3h |
| precision @ val-thr | 0.0163 | 0.0216 |  |
| recall @ val-thr | 0.8827 | 0.8616 |  |
| F1 @ 0.5 | 0.0089 | 0.0102 |  |
| recall@100 | 0.0459 | 0.0466 | pre_embedding_3h |
| recall@500 | 0.1664 | 0.1651 | post_embedding_128 |
| recall@1000 | 0.2626 | 0.2595 | post_embedding_128 |
| precision@100 | 0.7400 | 0.7500 |  |
| precision@500 | 0.5360 | 0.5320 |  |
| precision@1000 | 0.4230 | 0.4180 |  |
| lift@100 | 396.4420 | 401.7993 |  |
| lift@1000 | 226.6148 | 223.9362 |  |
| selected val threshold | 0.9357 | 0.9302 |  |
