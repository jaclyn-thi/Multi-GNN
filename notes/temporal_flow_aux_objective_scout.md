# Temporal-flow auxiliary objective scout

## Status

All 4 train→extract→probe chains completed; summarize job wrote this report.

- Dataset: Small-HI | Model: GIN | Seed: 1 | 20 epochs
- Recipe: asym proj, 8192 neg, queue=0, temp=0.5, reverse_mp+ego+ports+emlps+tds, bs=8192/accum=4
- Baseline: `hi_contrastive_gin_emlps_tds_proj_asym_8192neg_queue0_accum4_20ep`
- SSL aux: causal `temporal_flow_causal` only; **no labels**
- Attach point: `post_embedding_head_pre_projection`
- Primary success criterion: improve **A** (embedding-only) or **B** (emb/pre-3h + raw) vs baseline — **not** D-only

## Verdict

**Regression aux works; bins do not (as primary evidence).**

- Best scout: **`tf_reg_w0.10`** (ckpt epoch 17): post-128 AUPRC A **0.373** (+0.160 vs baseline 0.213), B **0.415** (+0.170 vs 0.244).
- Runner-up: **`tf_reg_w0.05`** (ckpt 17): A **0.279** (+0.065), B **0.387** (+0.142).
- `tf_bins5_w0.10` / `tf_bins10_w0.10`: embedding-only **worse** than baseline; B gains are small. Their large D scores are mostly downstream TF features.

## Post-128 vs baseline (primary)

| variant | ckpt | A AUPRC | ΔA | B AUPRC | ΔB | D AUPRC | primary? |
|---|---:|---:|---:|---:|---:|---:|---|
| baseline (no aux) | — | 0.2133 | — | 0.2445 | — | — | — |
| tf_bins5_w0.10 | 20 | 0.2012 | -0.0121 | 0.2579 | +0.0134 | 0.4712 | yes |
| tf_reg_w0.10 | 17 | 0.3731 | +0.1598 | 0.4149 | +0.1704 | 0.5112 | yes |
| tf_bins10_w0.10 | 20 | 0.1738 | -0.0395 | 0.2687 | +0.0242 | 0.4074 | yes |
| tf_reg_w0.05 | 17 | 0.2785 | +0.0652 | 0.3867 | +0.1422 | 0.4853 | yes |

Caveat: registry baseline rows likely used `probe_max_iter≈1000`; scout probes used `5000`. Regression deltas are large enough to remain convincing.

## Pre-3h (among variants; no matched seed1 baseline)

| variant | A | B | D | D−B |
|---|---:|---:|---:|---:|
| tf_bins5_w0.10 | 0.2174 | 0.3497 | 0.4758 | +0.1261 |
| tf_reg_w0.10 | 0.3226 | 0.3404 | 0.3801 | +0.0397 |
| tf_bins10_w0.10 | 0.1639 | 0.2196 | 0.3795 | +0.1599 |
| tf_reg_w0.05 | 0.3719 | 0.3549 | 0.4504 | +0.0955 |

Notes: `tf_reg_w0.05` has the strongest pre-3h embedding-only (0.372). `tf_reg_w0.10` is strong on post-128 but its pre-3h B/D are not the best among scouts — suggests the aux signal is especially visible in the post-128 attach-point representation.

## Convergence

| variant | best ckpt epoch | final aux loss (approx) |
|---|---:|---:|
| tf_bins5_w0.10 | 20 | 0.059 |
| tf_reg_w0.10 | 17 | 0.011 |
| tf_bins10_w0.10 | 20 | 0.090 |
| tf_reg_w0.05 | 17 | 0.008 |

Regression runs selected epoch 17 (train-loss best); bins selected final epoch 20.

## Full metric table

| variant | rep | arm | AUROC | AUPRC | F1 | P@100 | R@100 | Lift@100 | ckpt |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| tf_bins5_w0.10 | post_embedding_128 | A_embedding | 0.9457 | 0.2012 | 0.2605 | 0.6700 | 0.0416 | 358.9 | 20 |
| tf_bins5_w0.10 | post_embedding_128 | B_embedding_raw | 0.9518 | 0.2579 | 0.3427 | 0.7500 | 0.0466 | 401.8 | 20 |
| tf_bins5_w0.10 | post_embedding_128 | D_embedding_raw_temporal_flow | 0.9704 | 0.4712 | 0.4926 | 0.9500 | 0.0590 | 508.9 | 20 |
| tf_bins5_w0.10 | pre_embedding_3h | A_embedding | 0.9604 | 0.2174 | 0.2904 | 0.7700 | 0.0478 | 412.5 | 20 |
| tf_bins5_w0.10 | pre_embedding_3h | B_embedding_raw | 0.9703 | 0.3497 | 0.1843 | 0.8500 | 0.0528 | 455.4 | 20 |
| tf_bins5_w0.10 | pre_embedding_3h | D_embedding_raw_temporal_flow | 0.9776 | 0.4758 | 0.2314 | 0.9000 | 0.0559 | 482.2 | 20 |
| tf_reg_w0.10 | post_embedding_128 | A_embedding | 0.9638 | 0.3731 | 0.3987 | 0.7800 | 0.0484 | 417.9 | 17 |
| tf_reg_w0.10 | post_embedding_128 | B_embedding_raw | 0.9657 | 0.4149 | 0.3904 | 0.7900 | 0.0490 | 423.2 | 17 |
| tf_reg_w0.10 | post_embedding_128 | D_embedding_raw_temporal_flow | 0.9748 | 0.5112 | 0.4063 | 0.8500 | 0.0528 | 455.4 | 17 |
| tf_reg_w0.10 | pre_embedding_3h | A_embedding | 0.9700 | 0.3226 | 0.3239 | 0.6900 | 0.0428 | 369.6 | 17 |
| tf_reg_w0.10 | pre_embedding_3h | B_embedding_raw | 0.9716 | 0.3404 | 0.2970 | 0.7200 | 0.0447 | 385.7 | 17 |
| tf_reg_w0.10 | pre_embedding_3h | D_embedding_raw_temporal_flow | 0.9774 | 0.3801 | 0.3033 | 0.7600 | 0.0472 | 407.1 | 17 |
| tf_bins10_w0.10 | post_embedding_128 | A_embedding | 0.9533 | 0.1738 | 0.2359 | 0.4900 | 0.0304 | 262.5 | 20 |
| tf_bins10_w0.10 | post_embedding_128 | B_embedding_raw | 0.9592 | 0.2687 | 0.1335 | 0.6700 | 0.0416 | 358.9 | 20 |
| tf_bins10_w0.10 | post_embedding_128 | D_embedding_raw_temporal_flow | 0.9733 | 0.4074 | 0.2006 | 0.8400 | 0.0521 | 450.0 | 20 |
| tf_bins10_w0.10 | pre_embedding_3h | A_embedding | 0.9535 | 0.1639 | 0.2125 | 0.5000 | 0.0310 | 267.9 | 20 |
| tf_bins10_w0.10 | pre_embedding_3h | B_embedding_raw | 0.9573 | 0.2196 | 0.1481 | 0.5900 | 0.0366 | 316.1 | 20 |
| tf_bins10_w0.10 | pre_embedding_3h | D_embedding_raw_temporal_flow | 0.9713 | 0.3795 | 0.3023 | 0.7800 | 0.0484 | 417.9 | 20 |
| tf_reg_w0.05 | post_embedding_128 | A_embedding | 0.9603 | 0.2785 | 0.3333 | 0.8200 | 0.0509 | 439.3 | 17 |
| tf_reg_w0.05 | post_embedding_128 | B_embedding_raw | 0.9686 | 0.3867 | 0.1069 | 0.8500 | 0.0528 | 455.4 | 17 |
| tf_reg_w0.05 | post_embedding_128 | D_embedding_raw_temporal_flow | 0.9780 | 0.4853 | 0.2586 | 0.8900 | 0.0552 | 476.8 | 17 |
| tf_reg_w0.05 | pre_embedding_3h | A_embedding | 0.9688 | 0.3719 | 0.4158 | 0.9000 | 0.0559 | 482.2 | 17 |
| tf_reg_w0.05 | pre_embedding_3h | B_embedding_raw | 0.9666 | 0.3549 | 0.3788 | 0.8800 | 0.0546 | 471.4 | 17 |
| tf_reg_w0.05 | pre_embedding_3h | D_embedding_raw_temporal_flow | 0.9765 | 0.4504 | 0.4381 | 0.9100 | 0.0565 | 487.5 | 17 |

Full JSON: `results/diagnostics/temporal_flow_aux_objective_scout.json`

## Multiseed confirmation (Jul 19–20) — keep diagnostic

Focused regression-only confirmation (no bins). Summary: `notes/temporal_flow_regression_aux_multiseed.md`.

- **`tf_reg_w0.05`:** Claim 1 (pre-3h A/B) **passes** on paired seeds 1–2 (mean ΔA +0.126); Claim 2 (D) fails. Overall **`keep_diagnostic`**.
- **`tf_reg_w0.10`:** Claim 1 **fails** (seed2 A/B regress); do not promote. Seed3 absolute A/B look strong but lack a matched baseline.
- Do **not** insert into main thesis tables; do not treat seed1 post-128 gains alone as representation proof.

Registry: `table_group=temporal_flow_regression_aux_multiseed`, `diagnostic_only`, `table_eligible=false`.


