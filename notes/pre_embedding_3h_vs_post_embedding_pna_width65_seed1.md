**PNA interpretation:** `pre_embedding_3h` is **195-d** and `post_embedding` is **128-d**. The embedding head is `Linear(195, 128)` — an **expansion**, not the GIN-style 198→128 compression. Do not assume the GIN pre-3h ranking advantage transfers.

# pre_embedding_3h vs post_embedding_128 — Small-HI (pna_width65_ginparams_emlps_tds_asym_proj_8192neg_queue0_20ep_seed1)

- **checkpoint dirs:** post=`embeddings/pna_width65_ginparams_emlps_tds_asym_proj_8192neg_queue0_20ep_seed1`, pre=`embeddings/pna_width65_ginparams_emlps_tds_asym_proj_8192neg_queue0_20ep_seed1/pre_embedding_3h`
- **representation dims:** post_embedding_128 = 128, pre_embedding_3h = 195
- **no SSL retraining:** True  |  **paired:** True (inner-join on edge_id per split; identical rows/labels/order for both representations)
- **probe:** sklearn LogisticRegression (lbfgs), class_weight=model={0: 1.0000182882773443, 1: 6.275014431494497}, C=1.0, threshold=max_f1_on_val, seed=1

## embedding_only

| metric (test) | post_embedding_128 | pre_embedding_3h | winner |
|---|---|---|---|
| AUROC | 0.9536 | 0.9602 | pre_embedding_3h |
| AUPRC | 0.1475 | 0.2304 | pre_embedding_3h |
| F1 @ val-thr | 0.2165 | 0.2984 | pre_embedding_3h |
| precision @ val-thr | 0.2028 | 0.3988 |  |
| recall @ val-thr | 0.2322 | 0.2384 |  |
| F1 @ 0.5 | 0.2150 | 0.2866 |  |
| recall@100 | 0.0348 | 0.0515 | pre_embedding_3h |
| recall@500 | 0.1229 | 0.1670 | pre_embedding_3h |
| recall@1000 | 0.1769 | 0.2439 | pre_embedding_3h |
| precision@100 | 0.5600 | 0.8300 |  |
| precision@500 | 0.3960 | 0.5380 |  |
| precision@1000 | 0.2850 | 0.3930 |  |
| lift@100 | 300.0046 | 444.6497 |  |
| lift@1000 | 152.6809 | 210.5390 |  |
| selected val threshold | 0.5625 | 0.6879 |  |

## embedding_plus_raw

| metric (test) | post_embedding_128 | pre_embedding_3h | winner |
|---|---|---|---|
| AUROC | 0.9613 | 0.9691 | pre_embedding_3h |
| AUPRC | 0.1975 | 0.2757 | pre_embedding_3h |
| F1 @ val-thr | 0.2661 | 0.2788 | pre_embedding_3h |
| precision @ val-thr | 0.2455 | 0.1969 |  |
| recall @ val-thr | 0.2905 | 0.4773 |  |
| F1 @ 0.5 | 0.2169 | 0.1032 |  |
| recall@100 | 0.0397 | 0.0509 | pre_embedding_3h |
| recall@500 | 0.1502 | 0.1763 | pre_embedding_3h |
| recall@1000 | 0.2166 | 0.2669 | pre_embedding_3h |
| precision@100 | 0.6400 | 0.8200 |  |
| precision@500 | 0.4840 | 0.5680 |  |
| precision@1000 | 0.3490 | 0.4300 |  |
| lift@100 | 342.8624 | 439.2925 |  |
| lift@1000 | 186.9672 | 230.3607 |  |
| selected val threshold | 0.7274 | 0.8700 |  |
