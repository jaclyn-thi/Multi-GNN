# pre_embedding_3h vs post_embedding_128 (current-protocol contrastive GINe)

Diagnostic comparing two frozen representations from the **same** contrastively trained checkpoint: the exported 128-d `embedding_head` output (`post_embedding_128`) vs the `3 * n_hidden` tensor fed into `embedding_head` (`pre_embedding_3h` = `cat(src_node, dst_node, edge_attr)`). No contrastive retraining occurred; this is an extraction-location probe, not a new training method.

**Protocol:** identical frozen linear-probe pipeline for both representations, paired by an `edge_id` inner-join per split (same rows/labels/order), same class weights / regularization / val-tuned threshold / seed. Primary comparison is embedding-only.

## Small-HI — `gin_emlps_tds_asym_proj_8192neg_queue0_40ep_seed1`

- dims: post_embedding_128 = 128, pre_embedding_3h = 198
- ΔAUPRC (pre − post) = +0.0266, ΔF1 = -0.0028
- **interpretation:** 3h better: the embedding_head compression appears to discard some information useful for the probe

| metric (test, embedding-only) | post_embedding_128 | pre_embedding_3h | winner |
|---|---|---|---|
| AUROC | 0.9487 | 0.9523 | pre_embedding_3h |
| AUPRC | 0.1978 | 0.2244 | pre_embedding_3h |
| F1 @ val-thr | 0.2922 | 0.2894 | post_embedding_128 |
| precision @ val-thr | 0.3109 | 0.2439 |  |
| recall @ val-thr | 0.2756 | 0.3557 |  |
| recall@100 | 0.0366 | 0.0472 | pre_embedding_3h |
| recall@1000 | 0.2297 | 0.2464 | pre_embedding_3h |
| precision@100 | 0.5900 | 0.7600 |  |
| lift@100 | 316.0767 | 407.1496 |  |

- secondary (embedding+raw): AUPRC post=0.3184 pre=0.2737; F1@val-thr post=0.2762 pre=0.3054

## Small-LI — `small_li_gin_emlps_tds_asym_proj_8192neg_queue0_20ep_seed1`

- dims: post_embedding_128 = 128, pre_embedding_3h = 198
- ΔAUPRC (pre − post) = +0.0331, ΔF1 = +0.0402
- **interpretation:** 3h better: the embedding_head compression appears to discard some information useful for the probe

| metric (test, embedding-only) | post_embedding_128 | pre_embedding_3h | winner |
|---|---|---|---|
| AUROC | 0.8989 | 0.9225 | pre_embedding_3h |
| AUPRC | 0.0133 | 0.0464 | pre_embedding_3h |
| F1 @ val-thr | 0.0509 | 0.0911 | pre_embedding_3h |
| precision @ val-thr | 0.0315 | 0.0642 |  |
| recall @ val-thr | 0.1334 | 0.1571 |  |
| recall@100 | 0.0150 | 0.0299 | pre_embedding_3h |
| recall@1000 | 0.0524 | 0.1172 | pre_embedding_3h |
| precision@100 | 0.1200 | 0.2400 |  |
| lift@100 | 175.5456 | 351.0913 |  |

- secondary (embedding+raw): AUPRC post=0.0240 pre=0.0818; F1@val-thr post=0.0367 pre=0.0479

## Caveats

- Single seed / single checkpoint per dataset.
- This is an extraction-location diagnostic, not a new training method or a claim of universal superiority.
- `pre_embedding_3h` has more dimensions (198 vs 128), which can help a linear probe independent of information content; read alongside AUROC/AUPRC.
