# Strongest runs — master comparison (current protocol)

**Status: current results.** One place to compare our strongest diagnostic runs and decide what to
run next. Only the **strongest** configs are here (best stack per run); full ablation history stays in
[`results.md`](results.md) and [`results-archive.md`](results-archive.md).

**Fair probe policy (all rows):** frozen linear probe, `--class_weight model --model gin`, C=1.0,
val-tuned F1 threshold, test split. Numbers are **development** comparisons, not a frozen benchmark.

**Axes represented:** encoder recipe × feature stack × **representation source** × dataset. The new
axis is *representation source*: `post_128` = exported 128-d `embedding_head` output (what every
`results.md` row uses); `pre_3h` = the 198-d (`3×n_hidden`) tensor fed *into* `embedding_head`,
extracted from the **same frozen checkpoint** (`--representation_source pre_embedding_3h`, no
retraining). Metric priority: **AUPRC + alert-budget (P@100, lift@100)** over thresholded F1.

---

## Small-HI (test prevalence 0.187%, 1611 positives)

| Run | stack | source | AUROC | AUPRC | F1 | P@100 | lift@100 |
|-----|-------|--------|------:|------:|---:|------:|---------:|
| baseline (no emlps/tds) | embedding | post_128 | **0.951** | 0.120 | 0.236 | – | – |
| emlps+tds 20ep s1 | embedding | post_128 | 0.944 | 0.213 | 0.259 | – | – |
| emlps+tds 40ep s1 | embedding | post_128 | 0.949 | 0.198 | 0.292 | 0.59 | 316 |
| emlps+tds 40ep s1 | embedding | pre_3h | 0.952 | 0.224 | 0.289 | 0.76 | 407 |
| emlps+tds 40ep s1 | +raw | post_128 | 0.961 | 0.318 | 0.276 | 0.80 | 429 |
| emlps+tds 40ep s1 | +raw | pre_3h | 0.958 | 0.274 | 0.305 | 0.80 | 429 |
| emlps+tds **40ep s2** | embedding | post_128 | 0.949 | 0.245 | 0.304 | 0.80 | 429 |
| emlps+tds **40ep s2** | embedding | pre_3h | 0.958 | 0.295 | 0.340 | 0.83 | 445 |
| emlps+tds **40ep s2** | +raw | post_128 | 0.955 | 0.288 | 0.346 | 0.79 | 423 |
| **emlps+tds 40ep s2** | **+raw** | **pre_3h** | 0.960 | **0.321** | **0.344** | 0.84 | **450** |
| FNF s1 | +raw+morph | post_128 | 0.959 | 0.277 | 0.320 | 0.80 | 429 |
| FNF s1 | +raw+morph | pre_3h | **0.968** | 0.291 | 0.314 | 0.73 | 391 |
| FNF s1 | embedding | post_128 | 0.942 | 0.178 | 0.236 | 0.68 | 364 |
| FNF s1 | embedding | pre_3h | 0.963 | 0.255 | 0.315 | 0.68 | 364 |

**Small-HI champion:** `emlps+tds 40ep seed2` + `raw` + **`pre_3h`** — **0.321 AUPRC / 0.344 F1 /
lift@100 450**, the best single config on ranking. It edges the previous post-128 leaders (40ep s2
+raw = 0.288 AUPRC; FNF s1 full stack = 0.277).

## Small-LI (test prevalence 0.068%, 802 positives)

| Run | stack | source | AUROC | AUPRC | F1 | P@100 | lift@100 |
|-----|-------|--------|------:|------:|---:|------:|---------:|
| emlps+tds 20ep s1 (plain) | embedding | post_128 | 0.899 | 0.013 | 0.051 | 0.12 | 176 |
| emlps+tds 20ep s1 (plain) | embedding | pre_3h | 0.923 | 0.046 | 0.091 | 0.24 | 351 |
| emlps+tds 20ep s1 (plain) | +raw | post_128 | 0.909 | 0.024 | 0.037 | 0.20 | 293 |
| emlps+tds 20ep s1 (plain) | +raw | pre_3h | 0.932 | 0.082 | 0.048 | 0.44 | 644 |
| emlps+tds 20ep s2 (plain) | embedding | post_128 | 0.888 | 0.005 | 0.007 | 0.06 | 88 |
| emlps+tds 20ep s2 (plain) | embedding | pre_3h | 0.911 | 0.020 | 0.061 | 0.13 | 190 |
| emlps+tds 20ep s2 (plain) | +raw | post_128 | 0.901 | 0.016 | 0.013 | 0.12 | 176 |
| emlps+tds 20ep s2 (plain) | +raw | pre_3h | 0.910 | 0.022 | 0.062 | 0.16 | 234 |
| emlps+tds 20ep s3 (plain) | embedding | post_128 | 0.918 | 0.024 | 0.059 | 0.18 | 263 |
| emlps+tds 20ep s3 (plain) | embedding | pre_3h | 0.931 | 0.050 | 0.081 | 0.29 | 424 |
| emlps+tds 20ep s3 (plain) | +raw | post_128 | 0.926 | 0.056 | 0.066 | 0.34 | 527 |
| emlps+tds 20ep s3 (plain) | +raw | pre_3h | 0.938 | 0.079 | 0.082 | 0.41 | 629 |
| FNF s1 | embedding | post_128 | 0.900 | 0.017 | 0.059 | 0.13 | 190 |
| FNF s1 | embedding | pre_3h | 0.928 | 0.042 | 0.081 | 0.28 | 410 |
| FNF s1 | +raw | post_128 | 0.911 | 0.026 | 0.066 | 0.18 | 263 |
| **FNF s1** | **+raw** | **pre_3h** | 0.932 | **0.059** | 0.082 | 0.34 | **497** |
| FNF s1 | +raw+morph | post_128 | 0.919 | 0.036 | 0.090 | 0.17 | 249 |
| FNF s1 | +raw+morph | pre_3h | **0.943** | 0.055 | 0.069 | 0.25 | 366 |

**Small-LI champion (ranking/alerting):** `FNF s1` + `raw` + **`pre_3h`** — **0.059 AUPRC / lift@100
497**, roughly 2× the post-128 top-budget precision (P@100 0.18 → 0.34). Note the plain 20ep s1 + raw
+ pre_3h is competitive (0.082 AUPRC, lift@100 644 at very low base rate). **Multiseed replication
(Jul 8):** pre-3h wins AUPRC in **3/3 seeds** for both embedding-only and +raw (mean ΔAUPRC +0.025 /
+0.029); see [`pre_embedding_3h_vs_post_embedding_small_li_multiseed.md`](pre_embedding_3h_vs_post_embedding_small_li_multiseed.md).
Absolute precision on Small-LI stays low regardless — this is a ranking/enrichment win, not a solved
problem.

### Small-LI legacy supervised (in-GNN, not frozen probe)

| Run | selection | AUROC | AUPRC | F1 | Notes |
|-----|-----------|------:|------:|---:|-------|
| legacy scout 20ep s1 | best-val ep 11 | 0.944 | 0.191 | ~0.18–0.20 | `--testing` scout only |
| **legacy formal 100ep s1** | **best-val ep 35** | **0.959** | **0.292** | **0.357** | no `--testing`; final epoch F1=0 |

Use `checkpoint_best_val_f1.tar` for the 100-ep run — the last checkpoint collapses to all-negative
argmax. Details: [`supervised_Small-LI_small_li_legacy_supervised_gin_emlps_tds_100ep_seed1_summary.md`](supervised_Small-LI_small_li_legacy_supervised_gin_emlps_tds_100ep_seed1_summary.md).

---

## How to read this / which to pick

- **Default for probing existing checkpoints:** extract `pre_3h` for **embedding-only** and
  **+raw** stacks — it's a free ranking/alert-budget upgrade from the same frozen weights, biggest on
  Small-LI (rare positives).
- **Full `+raw+morph` stack:** pre-3h's edge is marginal and can flip — F1 and top-budget precision
  sometimes favor `post_128` (pre-3h's extra signal overlaps with morphology features). Choose by the
  metric you care about; for AUPRC/AUROC pre-3h still (barely) wins.
- **Not universal:** on **HI 40ep s1 +raw**, `post_128` actually wins AUPRC (0.318 vs 0.274) — the one
  cell where pre-3h loses ranking. So the pre-3h advantage is consistent in direction but
  **checkpoint- and stack-dependent** in size; don't treat it as a guaranteed win everywhere.
- **FNF vs ordinary:** FNF does **not** beat ordinary emlps+tds on Small-HI even with pre-3h (ordinary
  40ep s2 +raw pre_3h 0.321 AUPRC > FNF s1 full-stack pre_3h 0.291). FNF's value is clearest on
  Small-LI alerting.

## Axes tested vs still open

- **Tested:** encoder recipe (emlps+tds / FNF / 40ep), feature stack (embedding / +raw / +raw+morph),
  representation source (post_128 / pre_3h), dataset (HI / LI). Exported-dim 128-vs-198 was scouted
  separately (confounded; see [`small_li_embedding_dim_128_vs_198.md`](small_li_embedding_dim_128_vs_198.md)).
- **Open (to decide next steps):**
  - **pre-3h for the baseline and emlps+tds 20ep s1** (never extracted) to complete the HI ladder.
  - **Alert-budget-tuned thresholds** for pre-3h (val-tuned F1 transfers imperfectly to the 198-d probe).
  - **Dataset transfer** beyond Small-HI/LI (Medium/Large, PaySim, SAML-D).
  - **Legacy supervised multiseed** — formal 100-ep run is single-seed; late-epoch collapse needs monitoring.
- **Recently closed:**
  - ~~**Multi-seed pre-3h replication (plain Small-LI)**~~ — pre-3h wins AUPRC 3/3 seeds (embedding-only and +raw); see multiseed note.

## Caveats

- Development numbers, single seed / single checkpoint per config — directional, not frozen.
- `pre_3h` is 198-d vs `post_128` 128-d; a linear probe can gain from width alone (AUROC + alert-budget
  also improving argues it's not purely dimensional, but it is a confounder).
- `post_128` is reused from earlier extractions while `pre_3h` is a fresh forward pass; paired by
  `edge_id` per split with ~100% coverage.

## Provenance

- Strong-run batch: [`pre3h_strong_run_comparison.md`](pre3h_strong_run_comparison.md) ·
  `results/diagnostics/pre3h_vs_post128_small_hi_40ep_seed2.json`,
  `…_small_hi_fnf_seed1.json`, `…_small_li_fnf_seed1.json`
- Plain Small-LI multiseed pre/post: [`pre_embedding_3h_vs_post_embedding_small_li_multiseed.md`](pre_embedding_3h_vs_post_embedding_small_li_multiseed.md) ·
  `results/diagnostics/pre_embedding_3h_vs_post_embedding_small_li_multiseed.json`
- Legacy supervised 100 ep: [`supervised_Small-LI_small_li_legacy_supervised_gin_emlps_tds_100ep_seed1_summary.md`](supervised_Small-LI_small_li_legacy_supervised_gin_emlps_tds_100ep_seed1_summary.md)
- Earlier pre/post (HI 40ep s1, LI plain 20ep s1):
  [`pre_embedding_3h_vs_post_embedding_current_protocol.md`](pre_embedding_3h_vs_post_embedding_current_protocol.md)
- post_128 leaderboard + history: [`results.md`](results.md);
  thesis-safe synthesis: [`current_protocol_recent_runs_summary.md`](current_protocol_recent_runs_summary.md)
