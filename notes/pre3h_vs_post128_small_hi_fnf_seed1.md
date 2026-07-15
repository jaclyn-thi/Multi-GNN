# pre_embedding_3h vs post_embedding_128 — Small-HI (same_pair_fnf_emlps_tds_asym_proj_8192neg_queue0_20ep)

- **checkpoint dirs:** post=`embeddings/same_pair_fnf_emlps_tds_asym_proj_8192neg_queue0_20ep`, pre=`embeddings/same_pair_fnf_emlps_tds_asym_proj_8192neg_queue0_20ep/pre_embedding_3h`
- **representation dims:** post_embedding_128 = 128, pre_embedding_3h = 198
- **no SSL retraining:** True  |  **paired:** True (inner-join on edge_id per split; identical rows/labels/order for both representations)
- **probe:** sklearn LogisticRegression (lbfgs), class_weight=model={0: 1.0000182882773443, 1: 6.275014431494497}, C=1.0, threshold=max_f1_on_val, seed=1

## embedding_only

| metric (test) | post_embedding_128 | pre_embedding_3h | winner |
|---|---|---|---|
| AUROC | 0.9422 | 0.9626 | pre_embedding_3h |
| AUPRC | 0.1783 | 0.2546 | pre_embedding_3h |
| F1 @ val-thr | 0.2411 | 0.3152 | pre_embedding_3h |
| precision @ val-thr | 0.2234 | 0.2656 |  |
| recall @ val-thr | 0.2619 | 0.3873 |  |
| F1 @ 0.5 | 0.2388 | 0.3008 |  |
| recall@100 | 0.0422 | 0.0422 | tie |
| recall@500 | 0.1366 | 0.1707 | pre_embedding_3h |
| recall@1000 | 0.1968 | 0.2626 | pre_embedding_3h |
| precision@100 | 0.6800 | 0.6800 |  |
| precision@500 | 0.4400 | 0.5500 |  |
| precision@1000 | 0.3170 | 0.4230 |  |
| lift@100 | 364.2917 | 364.2917 |  |
| lift@1000 | 169.8242 | 226.6109 |  |
| selected val threshold | 0.5124 | 0.5674 |  |

## embedding_plus_raw

| metric (test) | post_embedding_128 | pre_embedding_3h | winner |
|---|---|---|---|
| AUROC | 0.9516 | 0.9649 | pre_embedding_3h |
| AUPRC | 0.2232 | 0.2753 | pre_embedding_3h |
| F1 @ val-thr | 0.2527 | 0.3185 | pre_embedding_3h |
| precision @ val-thr | 0.1801 | 0.2508 |  |
| recall @ val-thr | 0.4233 | 0.4364 |  |
| F1 @ 0.5 | 0.2611 | 0.2460 |  |
| recall@100 | 0.0478 | 0.0453 | post_embedding_128 |
| recall@500 | 0.1546 | 0.1782 | pre_embedding_3h |
| recall@1000 | 0.2340 | 0.2706 | pre_embedding_3h |
| precision@100 | 0.7700 | 0.7300 |  |
| precision@500 | 0.4980 | 0.5740 |  |
| precision@1000 | 0.3770 | 0.4360 |  |
| lift@100 | 412.5068 | 391.0779 |  |
| lift@1000 | 201.9676 | 233.5753 |  |
| selected val threshold | 0.4746 | 0.7043 |  |

## embedding_plus_raw_morph

| metric (test) | post_embedding_128 | pre_embedding_3h | winner |
|---|---|---|---|
| AUROC | 0.9586 | 0.9677 | pre_embedding_3h |
| AUPRC | 0.2767 | 0.2906 | pre_embedding_3h |
| F1 @ val-thr | 0.3200 | 0.3142 | post_embedding_128 |
| precision @ val-thr | 0.2620 | 0.2330 |  |
| recall @ val-thr | 0.4109 | 0.4823 |  |
| F1 @ 0.5 | 0.3036 | 0.2355 |  |
| recall@100 | 0.0497 | 0.0453 | post_embedding_128 |
| recall@500 | 0.1794 | 0.1782 | post_embedding_128 |
| recall@1000 | 0.2731 | 0.2762 | pre_embedding_3h |
| precision@100 | 0.8000 | 0.7300 |  |
| precision@500 | 0.5780 | 0.5740 |  |
| precision@1000 | 0.4400 | 0.4450 |  |
| lift@100 | 428.5785 | 391.0779 |  |
| lift@1000 | 235.7182 | 238.3968 |  |
| selected val threshold | 0.5681 | 0.7305 |  |
