# pre_embedding_3h vs post_embedding_128 — Small-LI (small_li_gin_emlps_tds_asym_proj_8192neg_queue0_20ep_seed1)

- **checkpoint dirs:** post=`embeddings/small_li_gin_emlps_tds_asym_proj_8192neg_queue0_20ep_seed1`, pre=`embeddings/small_li_gin_emlps_tds_asym_proj_8192neg_queue0_20ep_seed1/pre_embedding_3h`
- **representation dims:** post_embedding_128 = 128, pre_embedding_3h = 198
- **no SSL retraining:** True  |  **paired:** True (inner-join on edge_id per split; identical rows/labels/order for both representations)
- **probe:** sklearn LogisticRegression (lbfgs), class_weight=model={0: 1.0000182882773443, 1: 6.275014431494497}, C=1.0, threshold=max_f1_on_val, seed=1

## embedding_only

| metric (test) | post_embedding_128 | pre_embedding_3h | winner |
|---|---|---|---|
| AUROC | 0.8989 | 0.9225 | pre_embedding_3h |
| AUPRC | 0.0133 | 0.0464 | pre_embedding_3h |
| F1 @ val-thr | 0.0509 | 0.0911 | pre_embedding_3h |
| precision @ val-thr | 0.0315 | 0.0642 |  |
| recall @ val-thr | 0.1334 | 0.1571 |  |
| F1 @ 0.5 | 0.0422 | 0.0978 |  |
| recall@100 | 0.0150 | 0.0299 | pre_embedding_3h |
| recall@500 | 0.0337 | 0.0823 | pre_embedding_3h |
| recall@1000 | 0.0524 | 0.1172 | pre_embedding_3h |
| precision@100 | 0.1200 | 0.2400 |  |
| precision@500 | 0.0540 | 0.1320 |  |
| precision@1000 | 0.0420 | 0.0940 |  |
| lift@100 | 175.5456 | 351.0913 |  |
| lift@1000 | 61.4410 | 137.5107 |  |
| selected val threshold | 0.1866 | 0.2218 |  |

## embedding_plus_raw

| metric (test) | post_embedding_128 | pre_embedding_3h | winner |
|---|---|---|---|
| AUROC | 0.9093 | 0.9324 | pre_embedding_3h |
| AUPRC | 0.0240 | 0.0818 | pre_embedding_3h |
| F1 @ val-thr | 0.0367 | 0.0479 | pre_embedding_3h |
| precision @ val-thr | 0.0196 | 0.0255 |  |
| recall @ val-thr | 0.2781 | 0.4040 |  |
| F1 @ 0.5 | 0.0641 | 0.0201 |  |
| recall@100 | 0.0249 | 0.0549 | pre_embedding_3h |
| recall@500 | 0.0611 | 0.1247 | pre_embedding_3h |
| recall@1000 | 0.0810 | 0.1621 | pre_embedding_3h |
| precision@100 | 0.2000 | 0.4400 |  |
| precision@500 | 0.0980 | 0.2000 |  |
| precision@1000 | 0.0650 | 0.1300 |  |
| lift@100 | 292.5761 | 643.6673 |  |
| lift@1000 | 95.0872 | 190.1744 |  |
| selected val threshold | 0.2232 | 0.7791 |  |
