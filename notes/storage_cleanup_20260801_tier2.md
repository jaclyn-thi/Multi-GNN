# Storage cleanup tier-2 (2026-08-01)

Tightly scoped follow-up to `storage_cleanup_audit_20260801`.

## Part A — deleted superseded scratch embeddings

| Path | Kind | Size |
|------|------|-----:|
| `/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/direct_r198_infonce_40ep_seed2_linear_lr1e-3_epoch03` | id_fixed_seed_only_lr1e-3 | 1.386 GiB |
| `/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/direct_r198_infonce_40ep_seed2_linear_lr1e-3_epoch10` | id_fixed_seed_only_lr1e-3 | 1.259 GiB |
| `/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/direct_r198_infonce_40ep_seed2_linear_lr1e-3_epoch20` | id_fixed_seed_only_lr1e-3 | 1.202 GiB |
| `/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/direct_r198_infonce_40ep_seed2_linear_lr1e-3_epoch30` | id_fixed_seed_only_lr1e-3 | 1.181 GiB |
| `/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/direct_r198_infonce_40ep_seed2_linear_lr1e-3_epoch40` | id_fixed_seed_only_lr1e-3 | 1.157 GiB |
| `/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/direct_r198_tfmoe_40ep_seed2_linear_lr1e-3_epoch03` | id_fixed_seed_only_lr1e-3 | 1.702 GiB |
| `/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/direct_r198_tfmoe_40ep_seed2_linear_lr1e-3_epoch10` | id_fixed_seed_only_lr1e-3 | 1.632 GiB |
| `/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/direct_r198_tfmoe_40ep_seed2_linear_lr1e-3_epoch20` | id_fixed_seed_only_lr1e-3 | 1.626 GiB |
| `/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/direct_r198_tfmoe_40ep_seed2_linear_lr1e-3_epoch30` | id_fixed_seed_only_lr1e-3 | 1.659 GiB |
| `/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/direct_r198_tfmoe_40ep_seed2_linear_lr1e-3_epoch40` | id_fixed_seed_only_lr1e-3 | 1.683 GiB |
| `/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/gin_emlps_ports_tds_corrected_asym_proj_8192neg_queue0_40ep_seed2_last_ckpt` | unreferenced_last_ckpt | 2.496 GiB |

**Part A total:** 16.9844 GiB (11 directories)

## Part B — probe cache migration SCRATCH → POOL

Destination root: `/orcd/pool/007/jthi/Multi-GNN/results/cache_archive`

| Source | Destination | Status | Bytes | Checksum |
|--------|-------------|--------|------:|----------|
| `/orcd/scratch/orcd/008/jthi/Multi-GNN/results/cache/probe_weight_sweep_small_hi_key_runs` | `/orcd/pool/007/jthi/Multi-GNN/results/cache_archive/probe_weight_sweep_small_hi_key_runs` | success | 16.076 GiB | match |
| `/orcd/scratch/orcd/008/jthi/Multi-GNN/results/cache/probe_features_small_li_current_protocol` | `/orcd/pool/007/jthi/Multi-GNN/results/cache_archive/probe_features_small_li_current_protocol` | success | 11.242 GiB | match |
| `/orcd/scratch/orcd/008/jthi/Multi-GNN/results/cache/alert_budget_metrics_current_protocol` | `/orcd/pool/007/jthi/Multi-GNN/results/cache_archive/alert_budget_metrics_current_protocol` | success | 11.242 GiB | match |
| `/orcd/scratch/orcd/008/jthi/Multi-GNN/results/cache/probe_weight_sweep_small_li_current_protocol` | `/orcd/pool/007/jthi/Multi-GNN/results/cache_archive/probe_weight_sweep_small_li_current_protocol` | success | 11.242 GiB | match |

### Symlinks created

- `/orcd/scratch/orcd/008/jthi/Multi-GNN/results/cache/probe_weight_sweep_small_hi_key_runs` → `/orcd/pool/007/jthi/Multi-GNN/results/cache_archive/probe_weight_sweep_small_hi_key_runs` (realpath `/orcd/pool/007/jthi/Multi-GNN/results/cache_archive/probe_weight_sweep_small_hi_key_runs`)
- `/orcd/scratch/orcd/008/jthi/Multi-GNN/results/cache/probe_features_small_li_current_protocol` → `/orcd/pool/007/jthi/Multi-GNN/results/cache_archive/probe_features_small_li_current_protocol` (realpath `/orcd/pool/007/jthi/Multi-GNN/results/cache_archive/probe_features_small_li_current_protocol`)
- `/orcd/scratch/orcd/008/jthi/Multi-GNN/results/cache/alert_budget_metrics_current_protocol` → `/orcd/pool/007/jthi/Multi-GNN/results/cache_archive/alert_budget_metrics_current_protocol` (realpath `/orcd/pool/007/jthi/Multi-GNN/results/cache_archive/alert_budget_metrics_current_protocol`)
- `/orcd/scratch/orcd/008/jthi/Multi-GNN/results/cache/probe_weight_sweep_small_li_current_protocol` → `/orcd/pool/007/jthi/Multi-GNN/results/cache_archive/probe_weight_sweep_small_li_current_protocol` (realpath `/orcd/pool/007/jthi/Multi-GNN/results/cache_archive/probe_weight_sweep_small_li_current_protocol`)

## Usage

| Metric | Before | After |
|--------|-------:|------:|
| Scratch Multi-GNN | 814.50 GiB | 747.71 GiB |
| Pool Multi-GNN | 9.05 GiB | 58.86 GiB |
| Quota SCRATCH (file) | 814.5 / 1024.0 GiB | 770.2 / 1024.0 GiB |
| Quota POOL (file) | 5.3 / 1024.0 GiB | 42.9 / 1024.0 GiB |

**Scratch reclaimed (du):** 66.7860 GiB

**Projected free scratch after quota refresh:** 276.29 GiB

## Skipped / restored

None.


## Confirmations

- All protected_explicit paths remain: True
- No Slurm jobs submitted
- No HOME/datasets/checkpoints/full-subgraph/probabilities/temporal/morphology deletions
- Logical cache paths preserved via symlinks to POOL

## git status --short
```
M embedding_extraction.py
 M train_util.py
 M training.py
 M util.py
?? .vscode/
?? aml-data
?? checkpoints/
?? direct_r198/
?? embeddings
?? environment.yml
?? "lit review/"
?? morphology_cache
?? notes/aml_loss_matched_weighted_probe.md
?? notes/common_aml_validation_ce_comparison.md
?? notes/direct_h_tfmoe_scheduled_val_analysis.md
?? notes/direct_r198_40ep_linear_lr_full_extract_reeval.md
?? notes/direct_r198_official_collaborator_eval.md
?? notes/direct_r198_r198_only_lr_analysis.md
?? notes/direct_r198_tfmoe_40ep_linear_lr_sweep.md
?? notes/storage_cleanup_20260801_execution.md
?? notes/storage_cleanup_audit_20260801.md
?? notes/tfmoe_weight_ablation_lr2e-3.md
?? raw-aml-data
?? results/
?? saved-models
?? saved-models.home_backup_20260705/
?? scripts/aggregate_direct_r198_40ep_full_extract_reeval.py
?? scripts/aggregate_direct_r198_40ep_linear_lr_sweep.py
?? scripts/aml_loss_matched_weighted_ce.py
?? scripts/analyze_direct_h_tfmoe_scheduled_val.py
?? scripts/build_direct_r198_40ep_collaborator_package.py
?? scripts/build_direct_r198_r198_only_lr_analysis_package.py
?? scripts/build_tfmoe_weight_ablation_package.py
?? scripts/compare_common_aml_validation_ce.py
?? scripts/compare_seed_only_vs_full_r198_extract.py
?? scripts/direct_r198_eval_protocol.py
?? scripts/eval_aml_loss_matched_weighted_probes.py
?? scripts/eval_direct_r198_40ep_linear_arm.py
?? scripts/extract_direct_r198_epoch_batch.py
?? scripts/extract_direct_r198_full_cell.py
?? scripts/extract_direct_r198_seed_only_cell.py
?? scripts/official_direct_r198_collaborator_eval.py
?? scripts/plot_direct_h_tfmoe_figures_v2.py
?? scripts/plot_direct_h_tfmoe_presentation_v3.py
?? scripts/reeval_direct_r198_40ep_full_extract_cell.py
?? scripts/replot_direct_h_tfmoe_cleaned_baselines.py
?? scripts/validate_direct_r198_embedding_cells.py
?? storage_symlink_smoke
?? temporal_flow_cache/
?? transaction_knn
```
