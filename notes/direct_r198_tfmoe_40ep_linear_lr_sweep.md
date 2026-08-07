# DIRECT_H / DIRECT_H_TFMOE 40ep linear-LR sweep — analysis

Exploratory seed-2 Small-HI. **Test locked. AMP off.** Scheduler: `direct_h_warmup_linear` (1-epoch warmup then linear decay to 0.1× peak).

> **INVALID — seed-only validation metrics.** Val edge IDs were resolved from raw `input_id` (train-range leakage; ~100% train∩val). Do **not** use AUPRC/F1/BCE from this note for LR or DIRECT_H-vs-TFMOE claims. Corrected full-subgraph re-eval: [`notes/direct_r198_40ep_linear_lr_full_extract_reeval.md`](direct_r198_40ep_linear_lr_full_extract_reeval.md) and [`results/diagnostics/direct_r198_40ep_linear_lr_full_extract_reeval/`](../results/diagnostics/direct_r198_40ep_linear_lr_full_extract_reeval/). Marker: [`SEED_ONLY_VALIDATION_METRICS_INVALID.md`](../results/diagnostics/direct_r198_tfmoe_40ep_linear_lr_sweep/SEED_ONLY_VALIDATION_METRICS_INVALID.md).

## Completed runs

| Role | Unique name | Method | Peak LR | Scheduler | Epochs | Seed | Train job | Eval job |
|------|-------------|--------|---------|-----------|--------|------|-----------|----------|
| A | `direct_r198_infonce_40ep_seed2_linear_lr6p2e-3` | DIRECT_H InfoNCE (no proj) | ≈6.213e-3 | warmup_linear | 40 | 2 | 19333333 | 19333337 |
| B | `direct_r198_tfmoe_40ep_seed2_linear_lr6p2e-3` | DIRECT_H_TFMOE | ≈6.213e-3 | warmup_linear | 40 | 2 | 19333334 | 19333338 |
| C | `direct_r198_infonce_40ep_seed2_linear_lr2e-3` | DIRECT_H InfoNCE (no proj) | 2e-3 | warmup_linear | 40 | 2 | 19333335 | 19333339 |
| D | `direct_r198_tfmoe_40ep_seed2_linear_lr2e-3` | DIRECT_H_TFMOE | 2e-3 | warmup_linear | 40 | 2 | 19333336 | 19333340 |
| Agg | — | tables/figures | — | — | — | — | 19333341 | — |

Manifest: [`results/diagnostics/direct_r198_tfmoe_40ep_linear_lr_sweep/submission_manifest.json`](../results/diagnostics/direct_r198_tfmoe_40ep_linear_lr_sweep/submission_manifest.json)

## Primary downstream (selected by val AUPRC + final epoch 40)

| arm | run | peak_lr | checkpoint | ssl_epoch | val_auprc | F1@0.5 | F1@val-threshold | final probe train BCE | final probe val BCE |
|-----|-----|--------:|------------|----------:|----------:|-------:|-----------------:|----------------------:|--------------------:|
| DIRECT_R198 | `direct_r198_infonce_40ep_seed2_linear_lr6p2e-3` | 0.00621327 | selected_ssl | 30 | 0.0519297 | 0.101695 | 0.133333 | 0.00280241 | 0.0125981 |
| DIRECT_R198 | `direct_r198_infonce_40ep_seed2_linear_lr6p2e-3` | 0.00621327 | epoch_40 | 40 | 0.00642944 | 0.0408163 | 0.0490798 | 0.00279651 | 0.0413558 |
| DIRECT_R198_TFMOE | `direct_r198_tfmoe_40ep_seed2_linear_lr6p2e-3` | 0.00621327 | selected_ssl | 20 | 0.00073155 | 0 | 0 | 0.00243313 | 0.293068 |
| DIRECT_R198_TFMOE | `direct_r198_tfmoe_40ep_seed2_linear_lr6p2e-3` | 0.00621327 | epoch_40 | 40 | 0.000495422 | 0 | 0 | 0.0024763 | 92.6049 |
| DIRECT_R198 | `direct_r198_infonce_40ep_seed2_linear_lr2e-3` | 0.002 | selected_ssl | 40 | 0.00763341 | 0.0487805 | 0.0689655 | 0.00261981 | 0.0673398 |
| DIRECT_R198 | `direct_r198_infonce_40ep_seed2_linear_lr2e-3` | 0.002 | epoch_40 | 40 | 0.00763341 | 0.0487805 | 0.0689655 | 0.00261981 | 0.0673398 |
| DIRECT_R198_TFMOE | `direct_r198_tfmoe_40ep_seed2_linear_lr2e-3` | 0.002 | selected_ssl | 3 | 0.0150127 | 0 | 0.0727273 | 0.00257383 | 0.0101611 |
| DIRECT_R198_TFMOE | `direct_r198_tfmoe_40ep_seed2_linear_lr2e-3` | 0.002 | epoch_40 | 40 | 0.000514889 | 0.00179493 | 0.00191159 | 0.00237372 | 2.54025 |

### Per-checkpoint curves (all eval epochs)

| arm | peak_lr | ssl_epoch | val_auprc | F1@0.5 | F1@val-threshold | probe train BCE | probe val BCE | mean ‖r‖₂ | eff. rank |
|-----|--------:|----------:|----------:|-------:|-----------------:|----------------:|--------------:|----------:|----------:|
| DIRECT_H | 0.00621327 | 3 | 0.000778991 | 0 | 0 | 0.00270907 | 0.09131 | 26.57 | 5.83 |
| DIRECT_H | 0.00621327 | 10 | 0.00590673 | 0.0337079 | 0.0434783 | 0.00269333 | 0.00617799 | 50.58 | 1.24 |
| DIRECT_H | 0.00621327 | 20 | 0.0158327 | 0.0923077 | 0.0983607 | 0.00275346 | 0.00494982 | 72.18 | 1.12 |
| DIRECT_H | 0.00621327 | 30 | 0.0519297 | 0.101695 | 0.133333 | 0.00280241 | 0.0125981 | 85.35 | 1.13 |
| DIRECT_H | 0.00621327 | 40 | 0.00642944 | 0.0408163 | 0.0490798 | 0.00279651 | 0.0413558 | 41.15 | 2.18 |
| DIRECT_H_TFMOE | 0.00621327 | 3 | 0.000699568 | 0 | 0 | 0.00253981 | 0.0232981 | 14.92 | 10.29 |
| DIRECT_H_TFMOE | 0.00621327 | 10 | 0.000556009 | 0 | 0 | 0.00240167 | 2.82579 | 11.52 | 11.52 |
| DIRECT_H_TFMOE | 0.00621327 | 20 | 0.00073155 | 0 | 0 | 0.00243313 | 0.293068 | 9.09 | 11.33 |
| DIRECT_H_TFMOE | 0.00621327 | 30 | 0.000434556 | 0 | 0 | 0.00246211 | 1.93489 | 7.75 | 11.87 |
| DIRECT_H_TFMOE | 0.00621327 | 40 | 0.000495422 | 0 | 0 | 0.0024763 | 92.6049 | 7.99 | 11.76 |
| DIRECT_H | 0.002 | 3 | 0.000351569 | 0.00124069 | 0.00201005 | 0.00264351 | 0.128313 | 22.89 | 3.04 |
| DIRECT_H | 0.002 | 10 | 0.00086416 | 0 | 0 | 0.00262341 | 0.0468829 | 16.46 | 23.25 |
| DIRECT_H | 0.002 | 20 | 0.00101203 | 0 | 0 | 0.00261831 | 0.0402064 | 14.18 | 31.63 |
| DIRECT_H | 0.002 | 30 | 0.00706985 | 0 | 0.0666667 | 0.00265091 | 0.0554979 | 12.14 | 40.11 |
| DIRECT_H | 0.002 | 40 | 0.00763341 | 0.0487805 | 0.0689655 | 0.00261981 | 0.0673398 | 12.40 | 26.21 |
| DIRECT_H_TFMOE | 0.002 | 3 | 0.0150127 | 0 | 0.0727273 | 0.00257383 | 0.0101611 | 12.53 | 15.61 |
| DIRECT_H_TFMOE | 0.002 | 10 | 0.000630038 | 0 | 0 | 0.00244964 | 0.0675468 | 13.01 | 15.04 |
| DIRECT_H_TFMOE | 0.002 | 20 | 0.00233197 | 0.0152672 | 0.0266667 | 0.0023815 | 0.0789196 | 15.53 | 8.34 |
| DIRECT_H_TFMOE | 0.002 | 30 | 0.000442746 | 0.00112104 | 0.00112141 | 0.00239689 | 20.9685 | 16.18 | 7.45 |
| DIRECT_H_TFMOE | 0.002 | 40 | 0.000514889 | 0.00179493 | 0.00191159 | 0.00237372 | 2.54025 | 17.34 | 6.35 |

## SSL (contrastive) losses — keep separate from downstream BCE

Raw InfoNCE is **not** CE/BCE. For TFMOE, `loss/train` is the weighted multi-task objective (`L_total`), while `loss/contrastive` is raw InfoNCE.

| arm | run | peak_lr | epoch | raw InfoNCE | L_total / train | encoder LR |
|-----|-----|--------:|------:|------------:|----------------:|-----------:|
| DIRECT_H | `direct_r198_infonce_40ep_seed2_linear_lr6p2e-3` | 0.00621327 | 1 | 7.4392 | 7.4392 | 0.00621327 |
| DIRECT_H | `direct_r198_infonce_40ep_seed2_linear_lr6p2e-3` | 0.00621327 | 3 | 7.1470 | 7.1470 | 0.00592643 |
| DIRECT_H | `direct_r198_infonce_40ep_seed2_linear_lr6p2e-3` | 0.00621327 | 10 | 7.0974 | 7.0974 | 0.00492249 |
| DIRECT_H | `direct_r198_infonce_40ep_seed2_linear_lr6p2e-3` | 0.00621327 | 20 | 7.0816 | 7.0816 | 0.00348829 |
| DIRECT_H | `direct_r198_infonce_40ep_seed2_linear_lr6p2e-3` | 0.00621327 | 30 | 7.0771 | 7.0771 | 0.00205409 |
| DIRECT_H | `direct_r198_infonce_40ep_seed2_linear_lr6p2e-3` | 0.00621327 | 40 | 7.0705 | 7.0705 | 0.000621327 |
| DIRECT_H_TFMOE | `direct_r198_tfmoe_40ep_seed2_linear_lr6p2e-3` | 0.00621327 | 1 | 7.5100 | 4.6440 | 0.00621327 |
| DIRECT_H_TFMOE | `direct_r198_tfmoe_40ep_seed2_linear_lr6p2e-3` | 0.00621327 | 3 | 7.4098 | 0.8227 | 0.00592643 |
| DIRECT_H_TFMOE | `direct_r198_tfmoe_40ep_seed2_linear_lr6p2e-3` | 0.00621327 | 10 | 7.3878 | 0.6230 | 0.00492249 |
| DIRECT_H_TFMOE | `direct_r198_tfmoe_40ep_seed2_linear_lr6p2e-3` | 0.00621327 | 20 | 7.4251 | 0.4665 | 0.00348829 |
| DIRECT_H_TFMOE | `direct_r198_tfmoe_40ep_seed2_linear_lr6p2e-3` | 0.00621327 | 30 | 7.4304 | 0.3792 | 0.00205409 |
| DIRECT_H_TFMOE | `direct_r198_tfmoe_40ep_seed2_linear_lr6p2e-3` | 0.00621327 | 40 | 7.4286 | 0.3341 | 0.000621327 |
| DIRECT_H | `direct_r198_infonce_40ep_seed2_linear_lr2e-3` | 0.002 | 1 | 7.5681 | 7.5681 | 0.002 |
| DIRECT_H | `direct_r198_infonce_40ep_seed2_linear_lr2e-3` | 0.002 | 3 | 7.1957 | 7.1957 | 0.00190767 |
| DIRECT_H | `direct_r198_infonce_40ep_seed2_linear_lr2e-3` | 0.002 | 10 | 7.1254 | 7.1254 | 0.00158451 |
| DIRECT_H | `direct_r198_infonce_40ep_seed2_linear_lr2e-3` | 0.002 | 20 | 7.0972 | 7.0972 | 0.00112285 |
| DIRECT_H | `direct_r198_infonce_40ep_seed2_linear_lr2e-3` | 0.002 | 30 | 7.0865 | 7.0865 | 0.000661195 |
| DIRECT_H | `direct_r198_infonce_40ep_seed2_linear_lr2e-3` | 0.002 | 40 | 7.0808 | 7.0808 | 0.0002 |
| DIRECT_H_TFMOE | `direct_r198_tfmoe_40ep_seed2_linear_lr2e-3` | 0.002 | 1 | 7.6294 | 4.7295 | 0.002 |
| DIRECT_H_TFMOE | `direct_r198_tfmoe_40ep_seed2_linear_lr2e-3` | 0.002 | 3 | 7.4064 | 0.7966 | 0.00190767 |
| DIRECT_H_TFMOE | `direct_r198_tfmoe_40ep_seed2_linear_lr2e-3` | 0.002 | 10 | 7.4230 | 0.6052 | 0.00158451 |
| DIRECT_H_TFMOE | `direct_r198_tfmoe_40ep_seed2_linear_lr2e-3` | 0.002 | 20 | 7.4475 | 0.4434 | 0.00112285 |
| DIRECT_H_TFMOE | `direct_r198_tfmoe_40ep_seed2_linear_lr2e-3` | 0.002 | 30 | 7.4643 | 0.3670 | 0.000661195 |
| DIRECT_H_TFMOE | `direct_r198_tfmoe_40ep_seed2_linear_lr2e-3` | 0.002 | 40 | 7.4750 | 0.3307 | 0.0002 |

## Downstream CE/BCE (collaborator request)

Probe definition (all contrastive arms): **unweighted** `binary_cross_entropy_with_logits`, **one logit**, reduction=`mean`, no `pos_weight`. Values below are at the **last of 20 PaperStyleMLP probe epochs** (not best-by-val within the probe).

### Exact final-epoch probe BCE (selected SSL + epoch 40)

| run | checkpoint | ssl_epoch | train BCE | val BCE |
|-----|------------|----------:|----------:|--------:|
| `direct_r198_infonce_40ep_seed2_linear_lr6p2e-3` | selected_ssl | 30 | 0.00280241 | 0.01259815 |
| `direct_r198_infonce_40ep_seed2_linear_lr6p2e-3` | epoch_40 | 40 | 0.00279651 | 0.04135577 |
| `direct_r198_tfmoe_40ep_seed2_linear_lr6p2e-3` | selected_ssl | 20 | 0.00243313 | 0.29306763 |
| `direct_r198_tfmoe_40ep_seed2_linear_lr6p2e-3` | epoch_40 | 40 | 0.00247630 | 92.60485077 |
| `direct_r198_infonce_40ep_seed2_linear_lr2e-3` | selected_ssl | 40 | 0.00261981 | 0.06733979 |
| `direct_r198_infonce_40ep_seed2_linear_lr2e-3` | epoch_40 | 40 | 0.00261981 | 0.06733979 |
| `direct_r198_tfmoe_40ep_seed2_linear_lr2e-3` | selected_ssl | 3 | 0.00257383 | 0.01016114 |
| `direct_r198_tfmoe_40ep_seed2_linear_lr2e-3` | epoch_40 | 40 | 0.00237372 | 2.54024625 |

### Supervised Multi-GIN+EU (seed 2) — **not directly comparable**

Source: [`ce_audit.json`](../results/diagnostics/direct_r198_tfmoe_40ep_linear_lr_sweep/ce_audit.json), epoch history for `small_hi_legacy_supervised_gin_emlps_ports_50ep_seed2`.

| item | value |
|------|------:|
| loss class | `CrossEntropyLoss` (two logits) |
| class weights | ≈[1.000, 6.275] |
| final epoch 50 **train** weighted CE | 0.01165722 |
| best-val-F1 epoch 43 **train** weighted CE | 0.01131887 |
| validation CE/BCE in logs | **not logged** |
| comparable_directly | **false** |

A common unweighted validation binary NLL was **not** available in existing logs (`supervised_common_val_nll.status=unavailable_in_logs`). Do not compare the probe BCE numbers to the supervised train CE numbers.

## Findings

1. **Lower LR did not improve DIRECT_H.** Best InfoNCE checkpoint remains peak LR ≈6.21e-3 at SSL epoch **30** (val AUPRC **0.0519**, F1@0.5 **0.102**, F1@val-thr **0.133**). LR=2e-3 peaks only at epoch 40 with AUPRC **0.0076**.
2. **Lower LR helped TFMOE only relative to a collapsed high-LR TFMOE arm**, not in absolute terms. Best TFMOE is LR=2e-3 at SSL epoch **3** (AUPRC **0.0150**, F1@0.5 **0**, F1@val-thr **0.073**). High-LR TFMOE never exceeds AUPRC ≈0.0007.
3. **Training longer than ~20–30 epochs did not help overall.** InfoNCE lr6.2e-3 improves to ep30 then drops at ep40; both TFMOE arms degrade after their early peaks. Epoch-40 probe val BCE often explodes (e.g. TFMOE lr6.2e-3 val BCE **92.6**), indicating representation/probe failure—not useful convergence.
4. **Best DIRECT_H (this sweep):** `direct_r198_infonce_40ep_seed2_linear_lr6p2e-3`, SSL epoch **30** (selected by val AUPRC).
5. **Best DIRECT_H_TFMOE (this sweep):** `direct_r198_tfmoe_40ep_seed2_linear_lr2e-3`, SSL epoch **3**.
6. **TFMOE does not clearly improve over DIRECT_H here.** Best TFMOE AUPRC (0.015) ≪ best InfoNCE AUPRC (0.052). High-LR TFMOE is near-floor. (Contrast: earlier **10ep cosine** full-extract package had TFMOE ahead at ~0.54 vs ~0.49 — different schedule/extract; do not merge.)
7. **Raw InfoNCE stays ~7.07–7.48** across 40 epochs for all arms (slow decrease). TFMOE `L_total` falls because TF heads + α weighting dominate, **not** because InfoNCE itself converges strongly.
8. **No projected-encoder baselines** are included (ambiguous provenance).

## Figures

- [`01_raw_infonce`](../results/diagnostics/direct_r198_tfmoe_40ep_linear_lr_sweep/figures/01_raw_infonce.png) — InfoNCE vs optimizer step
- [`02_tf_raw_mae_by_lr`](../results/diagnostics/direct_r198_tfmoe_40ep_linear_lr_sweep/figures/02_tf_raw_mae_by_lr.png)
- [`03_weighted_objective_by_lr`](../results/diagnostics/direct_r198_tfmoe_40ep_linear_lr_sweep/figures/03_weighted_objective_by_lr.png)
- [`07_val_auprc`](../results/diagnostics/direct_r198_tfmoe_40ep_linear_lr_sweep/figures/07_val_auprc.png)
- [`08_val_f1_fixed`](../results/diagnostics/direct_r198_tfmoe_40ep_linear_lr_sweep/figures/08_val_f1_fixed.png)
- [`09_val_f1_val_selected`](../results/diagnostics/direct_r198_tfmoe_40ep_linear_lr_sweep/figures/09_val_f1_val_selected.png)
- [`10_contrastive_loss_vs_epoch`](../results/diagnostics/direct_r198_tfmoe_40ep_linear_lr_sweep/figures/10_contrastive_loss_vs_epoch.png) *(added)*
- [`11_tfmoe_total_loss_vs_epoch`](../results/diagnostics/direct_r198_tfmoe_40ep_linear_lr_sweep/figures/11_tfmoe_total_loss_vs_epoch.png) *(added)*
- [`12_final_probe_bce_by_run`](../results/diagnostics/direct_r198_tfmoe_40ep_linear_lr_sweep/figures/12_final_probe_bce_by_run.png) *(added)*

## Supporting artifacts

- Aggregate: [`aggregate.json`](../results/diagnostics/direct_r198_tfmoe_40ep_linear_lr_sweep/aggregate.json)
- CE audit: [`ce_audit.json`](../results/diagnostics/direct_r198_tfmoe_40ep_linear_lr_sweep/ce_audit.json)
- Tables: [`primary_performance_table.csv`](../results/diagnostics/direct_r198_tfmoe_40ep_linear_lr_sweep/primary_performance_table.csv), [`ce_comparison_table.csv`](../results/diagnostics/direct_r198_tfmoe_40ep_linear_lr_sweep/ce_comparison_table.csv)
- Cells: [`cells/`](../results/diagnostics/direct_r198_tfmoe_40ep_linear_lr_sweep/cells/)
- Train logs: `results/diagnostics/<run>/logs/{steps.jsonl,epoch_XX.json}`
