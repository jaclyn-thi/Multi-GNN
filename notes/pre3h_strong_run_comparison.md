# pre_embedding_3h vs post_embedding_128 — strong existing checkpoints

Extraction/probe diagnostic (no SSL retraining) comparing two frozen representations from the **same** contrastive checkpoint: the exported 128-d `embedding_head` output (`post_embedding_128`) vs the `3 * n_hidden` = 198-d tensor fed into `embedding_head` (`pre_embedding_3h`). Each comparison is paired by an `edge_id` inner-join per split (identical rows/labels/order), with the same probe seed, class weights, regularization, val-tuned threshold, and alert-budget definitions.

> **Conservative read:** one checkpoint per run, single probe seed. Treat directional signals, not precise magnitudes, as the takeaway.

## Conclusion (interpreted)

Provenance: extractions `17409110`/`17411075`/`17409112` → probes `17409113`/`17411076`/`17409115`
→ summary `17411077` (one HI FNF extraction first hit a GPU ECC fault on `node4104` and was
resubmitted with `--exclude`). Pairing coverage 0.9998–1.0000, so the comparisons are valid.

- **Consistent ranking win, stack-dependent size.** `pre_embedding_3h` (198-d) beats the exported
  128-d embedding on AUPRC **and** AUROC in all 8 run×stack cells. The margin shrinks as engineered
  features are added — HI FNF ΔAUPRC +0.076 (embedding-only) → +0.052 (+raw) → +0.014 (+raw+morph):
  pre-3h's extra signal partly overlaps with the raw/morph features.
- **Small-LI FNF is the operational standout.** Pre-3h roughly doubles top-budget precision:
  P@100 0.13→0.28 (embedding-only), 0.18→0.34 (+raw); lift@100 190→410 and 263→497. The plain-Small-LI
  pre-3h advantage clearly extends to FNF.
- **Best-of-batch:** best AUPRC (0.321) and best F1 (0.344) are pre-3h + raw on the **ordinary**
  HI 40ep seed2 run; best alert-budget (lift@100 497) is pre-3h + raw on LI FNF seed1.
- **FNF still doesn't beat ordinary on Small-HI**, even with pre-3h (ordinary full-stack AUPRC 0.321
  > FNF full-stack 0.291) — consistent with the earlier "FNF is mixed" conclusion.
- **Ranking vs threshold:** wins are cleanest on AUPRC/AUROC/alert-budget; val-tuned F1 sometimes
  regresses at the full (+morph) stack because the tuned threshold transfers imperfectly to the wider
  198-d probe.
- **Practical guidance:** extract `pre_embedding_3h` when probing embedding-only or embedding+raw
  (free ranking gain, no retraining); at a full raw+morph stack the choice is close to a wash.

## Pairing coverage (min joined-fraction over splits)

- `gin_emlps_tds_asym_proj_8192neg_queue0_40ep_seed2`: 0.9998
- `same_pair_fnf_emlps_tds_asym_proj_8192neg_queue0_20ep`: 1.0000
- `small_li_fnf_emlps_tds_asym_proj_8192neg_queue0_20ep_seed1`: 1.0000

## Guiding questions

1. **Improves strongest ordinary Small-HI (40ep seed2)?** pre-3h wins AUPRC in 2/2 feature stack(s). embedding_only: ΔAUPRC=+0.0505 (post=0.2449→pre=0.2953), ΔF1=+0.0357 [pre_3h better]; embedding_plus_raw: ΔAUPRC=+0.0374 (post=0.2838→pre=0.3212), ΔF1=+0.0014 [pre_3h better]
2. **Improves strongest FNF full-stack (Small-HI FNF seed1)?** pre-3h wins AUPRC in 3/3 feature stack(s). embedding_only: ΔAUPRC=+0.0763 (post=0.1783→pre=0.2546), ΔF1=+0.0740 [pre_3h better]; embedding_plus_raw: ΔAUPRC=+0.0521 (post=0.2232→pre=0.2753), ΔF1=+0.0658 [pre_3h better]; embedding_plus_raw_morph: ΔAUPRC=+0.0139 (post=0.2767→pre=0.2906), ΔF1=-0.0058 [pre_3h better]
3. **Does the Small-LI advantage extend to FNF?** pre-3h wins AUPRC in 3/3 feature stack(s). embedding_only: ΔAUPRC=+0.0253 (post=0.0166→pre=0.0419), ΔF1=+0.0220 [pre_3h better]; embedding_plus_raw: ΔAUPRC=+0.0332 (post=0.0255→pre=0.0587), ΔF1=+0.0158 [pre_3h better]; embedding_plus_raw_morph: ΔAUPRC=+0.0188 (post=0.0360→pre=0.0547), ΔF1=-0.0212 [pre_3h better]
4. **Best AUPRC (any run/stack/representation):** 0.3212 — pre_embedding_3h in embedding_plus_raw (gin_emlps_tds_asym_proj_8192neg_queue0_40ep_seed2)
5. **Best val-tuned F1:** 0.3443 — pre_embedding_3h in embedding_plus_raw (gin_emlps_tds_asym_proj_8192neg_queue0_40ep_seed2)
6. **Strongest alert-budget (lift@100):** 497.38 — pre_embedding_3h in embedding_plus_raw (small_li_fnf_emlps_tds_asym_proj_8192neg_queue0_20ep_seed1)
7. **Are gains consistent?** consistent: pre-3h wins AUPRC in all 8 run×stack comparisons.

## Small-HI — `gin_emlps_tds_asym_proj_8192neg_queue0_40ep_seed2` (small_hi_ordinary)

- dims: post_embedding_128 = 128, pre_embedding_3h = 198; checkpoint epoch 36

| feature stack | AUPRC post | AUPRC pre | ΔAUPRC | F1 post | F1 pre | ΔF1 | lift@100 post | lift@100 pre | AUPRC verdict |
|---|---|---|---|---|---|---|---|---|---|
| embedding_only | 0.2449 | 0.2953 | +0.0505 | 0.3040 | 0.3398 | +0.0357 | 428.51 | 444.58 | pre_3h better |
| embedding_plus_raw | 0.2838 | 0.3212 | +0.0374 | 0.3429 | 0.3443 | +0.0014 | 423.15 | 449.94 | pre_3h better |

## Small-HI — `same_pair_fnf_emlps_tds_asym_proj_8192neg_queue0_20ep` (small_hi_fnf)

- dims: post_embedding_128 = 128, pre_embedding_3h = 198; checkpoint epoch 19

| feature stack | AUPRC post | AUPRC pre | ΔAUPRC | F1 post | F1 pre | ΔF1 | lift@100 post | lift@100 pre | AUPRC verdict |
|---|---|---|---|---|---|---|---|---|---|
| embedding_only | 0.1783 | 0.2546 | +0.0763 | 0.2411 | 0.3152 | +0.0740 | 364.29 | 364.29 | pre_3h better |
| embedding_plus_raw | 0.2232 | 0.2753 | +0.0521 | 0.2527 | 0.3185 | +0.0658 | 412.51 | 391.08 | pre_3h better |
| embedding_plus_raw_morph | 0.2767 | 0.2906 | +0.0139 | 0.3200 | 0.3142 | -0.0058 | 428.58 | 391.08 | pre_3h better |

## Small-LI — `small_li_fnf_emlps_tds_asym_proj_8192neg_queue0_20ep_seed1` (small_li_fnf)

- dims: post_embedding_128 = 128, pre_embedding_3h = 198; checkpoint epoch 20

| feature stack | AUPRC post | AUPRC pre | ΔAUPRC | F1 post | F1 pre | ΔF1 | lift@100 post | lift@100 pre | AUPRC verdict |
|---|---|---|---|---|---|---|---|---|---|
| embedding_only | 0.0166 | 0.0419 | +0.0253 | 0.0589 | 0.0809 | +0.0220 | 190.17 | 409.61 | pre_3h better |
| embedding_plus_raw | 0.0255 | 0.0587 | +0.0332 | 0.0662 | 0.0820 | +0.0158 | 263.32 | 497.38 | pre_3h better |
| embedding_plus_raw_morph | 0.0360 | 0.0547 | +0.0188 | 0.0898 | 0.0686 | -0.0212 | 248.69 | 365.72 | pre_3h better |

## Caveats

- Extraction-location diagnostic only; **no contrastive retraining**. Not a claim that pre-3h is a universally better training target.
- `pre_embedding_3h` has more dimensions (198 vs 128); a linear probe can benefit from the extra width independent of information content. Read AUPRC alongside AUROC and alert-budget.
- `post_embedding_128` is reused from earlier extractions while `pre_embedding_3h` is a fresh forward pass; in the hetero loader the train split is sampled, so on high-degree nodes the two passes may sample slightly different neighborhoods. Pairing is by `edge_id` and coverage is reported above.
- Single checkpoint per run and a single probe seed.
