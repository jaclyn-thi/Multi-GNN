# Storage cleanup execution (2026-08-01)

Executed **only** the approved conservative SAFE_DELETE set from the audit:

1. 20 invalid seed-only R198 EdgeID-bug embedding directories on scratch
2. 36 empty/near-empty failed extraction stubs on scratch

Pre-deletion validation record: `results/diagnostics/storage_cleanup_20260801_execution_predelete.json`.

## Summary

| Metric | Value |
|--------|------:|
| Deleted path count | 56 |
| Skipped | 0 |
| Invalid seed-only deleted | 20 |
| Stubs deleted | 36 |
| Sum of target sizes before delete | 22.5231 GiB |
| Actual reclaimed (du scratch Multi-GNN delta) | 22.5231 GiB |
| Approved estimate (predelete) | 22.5231 GiB |
| Audit conservative estimate (incl. excluded pool dups) | 22.5260 GiB |
| Scratch Multi-GNN bytes before | 898743685651 (837.02 GiB) |
| Scratch Multi-GNN bytes after | 874559703158 (814.50 GiB) |
| Quota file SCRATCH usage (may lag) | 837.0 / 1024.0 GiB |
| Quota file free (may lag) | 187.0 GiB |
| Approx free vs 1024 GiB limit (du-based project only is incomplete for whole scratch) | see note |

## Skipped targets

None.

## Confirmations

- No REVIEW candidates removed
- No HOME files removed
- No POOL / checkpoint / dataset / cache / probability artifacts removed
- No full-subgraph embedding trees removed
- All `protected_explicit` paths still exist
- No Slurm jobs submitted
- Deletion used explicit per-directory `rmtree` on validated realpaths only

## Protected path checks

| Path | Exists |
|------|--------|
| `/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/direct_r198_40ep_linear_lr_full_extract` | yes |
| `/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/tfmoe_weight_ablation_lr2e-3_full_extract` | yes |
| `/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/final_corrected_no_preserve_multiseed` | yes |
| `/home/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN/results/diagnostics/final_corrected_no_preserve_multiseed/probabilities` | yes |
| `/home/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN/results/diagnostics/common_aml_validation_ce_comparison/predictions` | yes |
| `/orcd/pool/007/jthi/Multi-GNN/aml-data` | yes |
| `/orcd/pool/007/jthi/Multi-GNN/raw-aml-data` | yes |
| `/orcd/pool/007/jthi/Multi-GNN/morphology_cache` | yes |
| `/home/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN/temporal_flow_cache` | yes |
| `/orcd/scratch/orcd/008/jthi/Multi-GNN/results/cache/temporal_flow_causal` | yes |
| `/orcd/pool/007/jthi/Multi-GNN/saved-models` | yes |
| `/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/gcpal_txn_node_canonical` | yes |

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

## Deleted paths

### Invalid seed-only R198 (20)

- `/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/direct_r198_infonce_40ep_seed2_linear_lr2e-3_epoch03` (1.094 GiB)
- `/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/direct_r198_infonce_40ep_seed2_linear_lr2e-3_epoch10` (1.033 GiB)
- `/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/direct_r198_infonce_40ep_seed2_linear_lr2e-3_epoch20` (0.989 GiB)
- `/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/direct_r198_infonce_40ep_seed2_linear_lr2e-3_epoch30` (0.953 GiB)
- `/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/direct_r198_infonce_40ep_seed2_linear_lr2e-3_epoch40` (0.933 GiB)
- `/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/direct_r198_infonce_40ep_seed2_linear_lr6p2e-3_epoch03` (1.067 GiB)
- `/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/direct_r198_infonce_40ep_seed2_linear_lr6p2e-3_epoch10` (1.054 GiB)
- `/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/direct_r198_infonce_40ep_seed2_linear_lr6p2e-3_epoch20` (1.015 GiB)
- `/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/direct_r198_infonce_40ep_seed2_linear_lr6p2e-3_epoch30` (1.022 GiB)
- `/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/direct_r198_infonce_40ep_seed2_linear_lr6p2e-3_epoch40` (0.990 GiB)
- `/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/direct_r198_tfmoe_40ep_seed2_linear_lr2e-3_epoch03` (1.239 GiB)
- `/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/direct_r198_tfmoe_40ep_seed2_linear_lr2e-3_epoch10` (1.182 GiB)
- `/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/direct_r198_tfmoe_40ep_seed2_linear_lr2e-3_epoch20` (1.208 GiB)
- `/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/direct_r198_tfmoe_40ep_seed2_linear_lr2e-3_epoch30` (1.236 GiB)
- `/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/direct_r198_tfmoe_40ep_seed2_linear_lr2e-3_epoch40` (1.253 GiB)
- `/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/direct_r198_tfmoe_40ep_seed2_linear_lr6p2e-3_epoch03` (1.241 GiB)
- `/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/direct_r198_tfmoe_40ep_seed2_linear_lr6p2e-3_epoch10` (1.230 GiB)
- `/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/direct_r198_tfmoe_40ep_seed2_linear_lr6p2e-3_epoch20` (1.255 GiB)
- `/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/direct_r198_tfmoe_40ep_seed2_linear_lr6p2e-3_epoch30` (1.261 GiB)
- `/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/direct_r198_tfmoe_40ep_seed2_linear_lr6p2e-3_epoch40` (1.267 GiB)

### Failed extraction stubs (36)

- `/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/degree_aware_edgedrop_asym_proj_8192neg_queue0_20ep` (4688 B)
- `/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/degree_aware_edgedrop_emlps_tds_asym_proj_8192neg_queue0_20ep` (4780 B)
- `/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/degree_aware_edgedrop_samepair_fnf_asym_proj_8192neg_queue0_20ep` (4593 B)
- `/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/degree_aware_edgedrop_samepair_fnf_seed2_asym_proj_8192neg_queue0_20ep` (4638 B)
- `/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/degree_flow_aware_edgedrop_asym_proj_8192neg_queue0_20ep` (4524 B)
- `/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/edge_dplus_identity_poscomplete_10ep_seed2_ep01` (0 B)
- `/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/emlps_tds_asym_proj_8192neg_queue0_20ep_seed3` (4633 B)
- `/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/final_corrected_no_preserve_5seed` (0 B)
- `/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/fnf_emlps_tds_asym_proj_8192neg_queue0_20ep_seed2` (4666 B)
- `/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/gate_emlps_tds_asym_proj_8192neg_queue0_20ep_seed1` (4643 B)
- `/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/gin_emlps_tds_asym_proj_8192neg_queue0_40ep_seed2_last_ckpt` (4734 B)
- `/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/gin_emlps_tds_asym_proj_8192neg_queue0_40ep_seed3` (4675 B)
- `/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/gin_emlps_tds_asym_proj_8192neg_queue0_40ep_seed3_last_ckpt` (4714 B)
- `/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/gin_emlps_tds_asym_proj_8192neg_queue0_40ep_seed4` (4680 B)
- `/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/gin_emlps_tds_sym_proj_8192neg_queue0_20ep_bs4096_accum8_seed1` (4787 B)
- `/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/hi_contrastive_gin_emlps_proj_asym_8192neg_queue0_accum4_20ep` (4781 B)
- `/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/hi_contrastive_gin_lr0p003_h128_proj_asym_8192neg_queue0_accum4_20ep` (4622 B)
- `/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/hi_contrastive_gin_lr0p003_h66_proj_asym_8192neg_queue0_accum4_20ep` (4610 B)
- `/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/hi_contrastive_gin_lrbase_h128_proj_asym_8192neg_queue0_accum4_20ep` (4617 B)
- `/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/hi_contrastive_gin_lrbase_h66_finaldrop0_proj_asym_8192neg_queue0_accum4_20ep` (4690 B)
- `/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/hi_contrastive_gin_tds_proj_asym_8192neg_queue0_accum4_20ep` (4761 B)
- `/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/hi_contrastive_plus_masked_edge_asym_proj_8192neg_queue0_accum4_20ep_bestckpt` (4695 B)
- `/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/hi_contrastive_pna_proj_asym_8192neg_queue0_accum4_20ep` (4525 B)
- `/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/hi_contrastive_proj_asym_8192neg_queue0_bs8192_accum4_seed1_fnfsame_pair_20ep_bestckpt` (4763 B)
- `/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/hi_contrastive_proj_asym_8192neg_queue0_bs8192_accum4_seed2_fnfsame_pair_20ep_bestckpt` (4979 B)
- `/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/hi_contrastive_proj_asym_8192neg_queue0_bs8192_accum4_seed3_fnfsame_pair_20ep_bestckpt` (4760 B)
- `/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/hi_masked_edge_attr_gine_20ep_bestckpt` (4589 B)
- `/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/hi_masked_edge_attr_gine_20ep_seed2` (4357 B)
- `/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/hi_masked_edge_attr_gine_40ep_seed1` (4361 B)
- `/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/morph_degree_fan_only_asym_proj_8192neg_queue0_20ep_weight005` (4571 B)
- `/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/morph_expert_emlps_tds_asym_proj_8192neg_queue0_20ep` (4702 B)
- `/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/morph_expert_emlps_tds_asym_proj_8192neg_queue0_20ep_lastckpt_probe` (8350 B)
- `/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/morph_flow_balance_only_asym_proj_8192neg_queue0_10ep_weight005` (4585 B)
- `/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/morph_motif_participation_only_asym_proj_8192neg_queue0_10ep_weight005` (4624 B)
- `/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/paysim` (24992 B)
- `/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/rgcn_emlps_tds_asym_proj_8192neg_queue0_20ep_seed1` (4665 B)
