# pre_embedding_3h vs post_embedding_128 — Small-LI multiseed (plain GINe baseline)

**Interpreted conclusion:** The seed-1 finding that `pre_embedding_3h` beats the exported 128-d embedding on rare-positive ranking **replicates across all three plain Small-LI SSL seeds**. Pre-3h wins AUPRC in **3/3 seeds** for both embedding-only (mean ΔAUPRC **+0.025 ± 0.009**) and embedding+raw (**+0.029 ± 0.026**). Magnitude is seed-dependent — seed 2 is a weak encoder overall (post AUPRC 0.005) but pre-3h still wins directionally; seed 3 is the strongest post-128 baseline. This is a **free extraction lever** (same frozen checkpoint, no retraining), not a new training recipe. Caveats: 198-d vs 128-d confounder; development comparison; absolute Small-LI precision remains low.

Paired probe comparison (same frozen checkpoint, `edge_id` inner-join) across plain Small-LI contrastive seeds. Fair policy: `cw=model`, C=1.0, val-tuned F1; primary stacks: embedding-only and embedding+raw.

**Seeds included:** [1, 2, 3]

## Aggregate (pre − post)

| stack | n seeds | pre-3h AUPRC wins | mean ΔAUPRC | std ΔAUPRC | mean Δlift@100 | std Δlift@100 |
|---|---:|---:|---:|---:|---:|---:|
| embedding_only | 3 | 3 | 0.0245 | 0.0090 | 146.29 | 38.71 |
| embedding_plus_raw | 3 | 3 | 0.0291 | 0.0262 | 170.67 | 157.78 |

## Conclusions

- **multiseed auprc embedding only:** pre-3h wins AUPRC in 3/3 seeds; mean ΔAUPRC=0.0245 ± 0.0090
- **multiseed auprc embedding plus raw:** pre-3h wins AUPRC in 3/3 seeds; mean ΔAUPRC=0.0291 ± 0.0262
- **multiseed alert budget embedding plus raw:** mean Δlift@100=170.67 ± 157.78
- **seed1 result replicated:** yes: every additional seed shows pre-3h winning AUPRC in both embedding-only and +raw
- **conservative read:** 3 seed(s); single checkpoint per seed; development comparison — treat replication as directional unless all three seeds agree with similar magnitude.

## Seed 1 — `small_li_gin_emlps_tds_asym_proj_8192neg_queue0_20ep_seed1`

- min pairing coverage: 1.0000

| stack | AUPRC post | AUPRC pre | ΔAUPRC | F1 post | F1 pre | lift@100 post | lift@100 pre | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| embedding_only | 0.0133 | 0.0464 | +0.0331 | 0.0509 | 0.0911 | 175.55 | 351.09 | pre_3h better |
| embedding_plus_raw | 0.0240 | 0.0818 | +0.0578 | 0.0367 | 0.0479 | 292.58 | 643.67 | pre_3h better |

## Seed 2 — `small_li_gin_emlps_tds_asym_proj_8192neg_queue0_20ep_seed2`

- min pairing coverage: 1.0000

| stack | AUPRC post | AUPRC pre | ΔAUPRC | F1 post | F1 pre | lift@100 post | lift@100 pre | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| embedding_only | 0.0051 | 0.0202 | +0.0151 | 0.0067 | 0.0611 | 87.77 | 190.17 | pre_3h better |
| embedding_plus_raw | 0.0161 | 0.0224 | +0.0063 | 0.0125 | 0.0624 | 175.54 | 234.05 | pre_3h better |

## Seed 3 — `small_li_gin_emlps_tds_asym_proj_8192neg_queue0_20ep_seed3`

- min pairing coverage: 1.0000

| stack | AUPRC post | AUPRC pre | ΔAUPRC | F1 post | F1 pre | lift@100 post | lift@100 pre | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| embedding_only | 0.0242 | 0.0495 | +0.0253 | 0.0802 | 0.1135 | 263.32 | 424.23 | pre_3h better |
| embedding_plus_raw | 0.0561 | 0.0793 | +0.0233 | 0.0693 | 0.0524 | 526.63 | 629.03 | pre_3h better |

## Caveats

- Development numbers; single checkpoint per seed.
- pre_3h is 198-d vs post_128 128-d (linear-probe width confounder).
- Seed 1 probe uses the earlier `pre_embedding_3h_vs_post_embedding_small_li.json` artifact.
