# Storage cleanup audit (2026-08-01) — READ-ONLY

**Nothing was deleted, moved, truncated, compressed, or overwritten. No Slurm jobs were submitted.**
Only this note plus `results/diagnostics/storage_cleanup_audit_20260801.json` and `..._candidates.tsv` were written.

## Path resolution

| Role | Path |
|------|------|
| Repository root | `/home/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN` |
| Resolved root | `/home/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN` |
| Home alias | `/orcd/home/002/jthi/ondemand/...` (same home NFS) |

### Symlinks → real paths → filesystem

| Symlink (in repo) | Real path | Mount / quota |
|-------------------|-----------|---------------|
| `embeddings` | `/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings` (~787 GiB) | SCRATCH |
| `results/cache` | `/orcd/scratch/orcd/008/jthi/Multi-GNN/results/cache` (~51 GiB) | SCRATCH |
| `storage_symlink_smoke` | `/orcd/scratch/orcd/008/jthi/Multi-GNN/storage_symlink_smoke` | SCRATCH |
| `saved-models` | `/orcd/pool/007/jthi/Multi-GNN/saved-models` (~673 MiB) | POOL |
| `aml-data` | `/orcd/pool/007/jthi/Multi-GNN/aml-data` (~3.1 GiB) | POOL |
| `raw-aml-data` | `/orcd/pool/007/jthi/Multi-GNN/raw-aml-data` (~891 MiB) | POOL |
| `morphology_cache` | `/orcd/pool/007/jthi/Multi-GNN/morphology_cache` (~711 MiB) | POOL |
| `transaction_knn` | `/orcd/pool/007/jthi/Multi-GNN/transaction_knn` | POOL |

Non-symlink large locals on HOME: `results/` (~27 GiB, mostly diagnostics), `saved-models.home_backup_20260705/` (~218 MiB), `temporal_flow_cache/` (~220 MiB), `slurm-logs/` (~117 MiB).

## Quota / usage

Source: `/home/jthi/orcd/.quota` (Sat Aug 01 20:05:40 EDT 2026) and `quota -s`.

| Space | Usage | Limit | % Used |
|-------|------:|------:|-------:|
| HOME | 129.1 GiB | 200.0 GiB | 64.55% |
| SCRATCH | 837.0 GiB | 1024.0 GiB | **81.74%** |
| POOL | 5.3 GiB | 1024.0 GiB | 0.51% |

Project contribution (approx):
- HOME Multi-GNN tree without following symlinks: **~28 GiB** (of 129 GiB home total)
- SCRATCH Multi-GNN: **~838 GiB** (essentially all of user scratch)
- POOL Multi-GNN: **~5.3 GiB** (all of user pool)

## Duplicate / link analysis

| Pair | Relationship |
|------|----------------|
| `saved-models.home_backup_20260705/` vs pool `saved-models/` | **Separate files** (same sizes; sampled SHA256 match; different fs/inodes; **not** hardlinks) |
| Pool root `checkpoint_seq_aml2ps_smoke_*` vs `sequential_aml_to_paysim_ssl/` copies | **Separate files**, full SHA256 identical |
| Probe cache trees (`probe_features_*` vs `probe_weight_sweep_*`) | **Separate files**, content-identical sample (first 1 MiB SHA256) |
| `results/diagnostics/final_exploratory_ssl_scout/embeddings/` | Regular files on HOME (not symlinked to scratch) |

## Explicitly protected (KEEP)

- Raw + formatted datasets (`raw-aml-data`, `aml-data`)
- `morphology_cache`, `temporal_flow_cache`, scratch `results/cache/temporal_flow_causal`
- `embeddings/final_corrected_no_preserve_multiseed` + home `.../probabilities/`
- Supervised formal AMLWorld / PaySim / SAML-D checkpoints under pool `saved-models/`
- Current 40-epoch DIRECT_H/TFMOE: `embeddings/direct_r198_40ep_linear_lr_full_extract/` + matching checkpoints
- TFMOE weighting ablation reports + `embeddings/tfmoe_weight_ablation_lr2e-3_full_extract/` + EXPERT_ONLY epoch-10/20 checkpoints
- Job **19458946** supervised val preds: `results/diagnostics/common_aml_validation_ce_comparison/predictions/`
- Thesis-facing canonical JSON / registry / source code / `.git`

## Priority findings

1. **Failed/incomplete extraction stubs** — 36 scratch embedding dirs with only meta/probe JSON or empty (~178.08 KiB).
2. **Invalid seed-only R198 EdgeID-bug artifacts** — 20 dirs, **22.52 GiB**, superseded by full-subgraph extract.
3. **Duplicate staged smoke checkpoints** — 2 pool tars with identical SHA256 to experiment-subdir copies (~3 MiB).
4. **Superseded smoke embeddings** — exploratory scout smoke tree on HOME (REVIEW).
5. **ID-fixed but superseded seed-only** — 10 dirs, **14.49 GiB** (REVIEW).
6. **HOME-staged exploratory embeddings** — under `final_exploratory_ssl_scout/embeddings/` (REVIEW; table_eligible=false).
7. **Probe feature caches** — ~50 GiB scratch; regenerable; some cross-tree content dupes (REVIEW).
8. **Home checkpoint backup** — separate copies of pool (REVIEW; do not assume deletable).
9. **Logs** — slurm-logs/wandb on HOME (REVIEW only if needed).

## Candidates

Full per-path table: [`results/diagnostics/storage_cleanup_audit_20260801_candidates.tsv`](../results/diagnostics/storage_cleanup_audit_20260801_candidates.tsv).

### SAFE_DELETE (concrete evidence)

| path | size | status | replacement | conf |
| --- | --- | --- | --- | --- |
| scratch/embeddings/direct_r198_tfmoe_40ep_seed2_linear_lr6p2e-3_epoch40 | 1.27 GiB | invalid | scratch/embeddings/direct_r198_40ep_linear_lr_full_extract/direct_r198 | high |
| scratch/embeddings/direct_r198_tfmoe_40ep_seed2_linear_lr6p2e-3_epoch30 | 1.26 GiB | invalid | scratch/embeddings/direct_r198_40ep_linear_lr_full_extract/direct_r198 | high |
| scratch/embeddings/direct_r198_tfmoe_40ep_seed2_linear_lr6p2e-3_epoch20 | 1.26 GiB | invalid | scratch/embeddings/direct_r198_40ep_linear_lr_full_extract/direct_r198 | high |
| scratch/embeddings/direct_r198_tfmoe_40ep_seed2_linear_lr2e-3_epoch40 | 1.25 GiB | invalid | scratch/embeddings/direct_r198_40ep_linear_lr_full_extract/direct_r198 | high |
| scratch/embeddings/direct_r198_tfmoe_40ep_seed2_linear_lr6p2e-3_epoch03 | 1.24 GiB | invalid | scratch/embeddings/direct_r198_40ep_linear_lr_full_extract/direct_r198 | high |
| scratch/embeddings/direct_r198_tfmoe_40ep_seed2_linear_lr2e-3_epoch03 | 1.24 GiB | invalid | scratch/embeddings/direct_r198_40ep_linear_lr_full_extract/direct_r198 | high |
| scratch/embeddings/direct_r198_tfmoe_40ep_seed2_linear_lr2e-3_epoch30 | 1.24 GiB | invalid | scratch/embeddings/direct_r198_40ep_linear_lr_full_extract/direct_r198 | high |
| scratch/embeddings/direct_r198_tfmoe_40ep_seed2_linear_lr6p2e-3_epoch10 | 1.23 GiB | invalid | scratch/embeddings/direct_r198_40ep_linear_lr_full_extract/direct_r198 | high |
| scratch/embeddings/direct_r198_tfmoe_40ep_seed2_linear_lr2e-3_epoch20 | 1.21 GiB | invalid | scratch/embeddings/direct_r198_40ep_linear_lr_full_extract/direct_r198 | high |
| scratch/embeddings/direct_r198_tfmoe_40ep_seed2_linear_lr2e-3_epoch10 | 1.18 GiB | invalid | scratch/embeddings/direct_r198_40ep_linear_lr_full_extract/direct_r198 | high |
| scratch/embeddings/direct_r198_infonce_40ep_seed2_linear_lr2e-3_epoch03 | 1.09 GiB | invalid | scratch/embeddings/direct_r198_40ep_linear_lr_full_extract/direct_r198 | high |
| scratch/embeddings/direct_r198_infonce_40ep_seed2_linear_lr6p2e-3_epoch03 | 1.07 GiB | invalid | scratch/embeddings/direct_r198_40ep_linear_lr_full_extract/direct_r198 | high |
| scratch/embeddings/direct_r198_infonce_40ep_seed2_linear_lr6p2e-3_epoch10 | 1.05 GiB | invalid | scratch/embeddings/direct_r198_40ep_linear_lr_full_extract/direct_r198 | high |
| scratch/embeddings/direct_r198_infonce_40ep_seed2_linear_lr2e-3_epoch10 | 1.03 GiB | invalid | scratch/embeddings/direct_r198_40ep_linear_lr_full_extract/direct_r198 | high |
| scratch/embeddings/direct_r198_infonce_40ep_seed2_linear_lr6p2e-3_epoch30 | 1.02 GiB | invalid | scratch/embeddings/direct_r198_40ep_linear_lr_full_extract/direct_r198 | high |
| scratch/embeddings/direct_r198_infonce_40ep_seed2_linear_lr6p2e-3_epoch20 | 1.02 GiB | invalid | scratch/embeddings/direct_r198_40ep_linear_lr_full_extract/direct_r198 | high |
| scratch/embeddings/direct_r198_infonce_40ep_seed2_linear_lr6p2e-3_epoch40 | 1013.76 MiB | invalid | scratch/embeddings/direct_r198_40ep_linear_lr_full_extract/direct_r198 | high |
| scratch/embeddings/direct_r198_infonce_40ep_seed2_linear_lr2e-3_epoch20 | 1012.46 MiB | invalid | scratch/embeddings/direct_r198_40ep_linear_lr_full_extract/direct_r198 | high |
| scratch/embeddings/direct_r198_infonce_40ep_seed2_linear_lr2e-3_epoch30 | 975.60 MiB | invalid | scratch/embeddings/direct_r198_40ep_linear_lr_full_extract/direct_r198 | high |
| scratch/embeddings/direct_r198_infonce_40ep_seed2_linear_lr2e-3_epoch40 | 955.63 MiB | invalid | scratch/embeddings/direct_r198_40ep_linear_lr_full_extract/direct_r198 | high |
| pool/saved-models/checkpoint_seq_aml2ps_smoke_aml_init_19017925_finetuned.tar | 2.17 MiB | duplicate | pool/saved-models/sequential_aml_to_paysim_ssl/checkpoint_seq_aml2ps_s | high |
| pool/saved-models/checkpoint_seq_aml2ps_smoke_aml_init_19017925.tar | 873.52 KiB | duplicate | pool/saved-models/sequential_aml_to_paysim_ssl/checkpoint_seq_aml2ps_s | high |

Plus **36** empty/near-empty failed extract dirs under `embeddings/` (negligible bytes; listed in TSV/JSON).

### Largest REVIEW candidates

| path | size | status | quota | conf |
| --- | --- | --- | --- | --- |
| repo/results/diagnostics/final_exploratory_ssl_scout/embeddings | 22.53 GiB | complete | home_quota | medium |
| scratch/results/cache/probe_weight_sweep_small_hi_key_runs | 16.08 GiB | complete | scratch_quota | medium |
| scratch/results/cache/probe_features_small_li_current_protocol | 11.24 GiB | complete | scratch_quota | medium |
| scratch/results/cache/alert_budget_metrics_current_protocol | 11.24 GiB | complete | scratch_quota | medium |
| scratch/results/cache/probe_weight_sweep_small_li_current_protocol | 11.24 GiB | complete | scratch_quota | medium |
| scratch/embeddings/gin_emlps_ports_tds_corrected_asym_proj_8192neg_queue0_40ep_seed2_last_ckpt | 2.50 GiB | complete | scratch_quota | medium |
| repo/results/diagnostics/final_exploratory_ssl_scout/embeddings/smoke_aml_final_exploratory_ssl_scout_smoke_m_seed2_19001714 | 2.07 GiB | superseded | home_quota | medium |
| scratch/embeddings/direct_r198_tfmoe_40ep_seed2_linear_lr1e-3_epoch03 | 1.70 GiB | superseded | scratch_quota | high |
| scratch/embeddings/direct_r198_tfmoe_40ep_seed2_linear_lr1e-3_epoch40 | 1.68 GiB | superseded | scratch_quota | high |
| scratch/embeddings/direct_r198_tfmoe_40ep_seed2_linear_lr1e-3_epoch30 | 1.66 GiB | superseded | scratch_quota | high |
| scratch/embeddings/direct_r198_tfmoe_40ep_seed2_linear_lr1e-3_epoch10 | 1.63 GiB | superseded | scratch_quota | high |
| scratch/embeddings/direct_r198_tfmoe_40ep_seed2_linear_lr1e-3_epoch20 | 1.63 GiB | superseded | scratch_quota | high |
| scratch/embeddings/direct_r198_infonce_40ep_seed2_linear_lr1e-3_epoch03 | 1.39 GiB | superseded | scratch_quota | high |
| scratch/embeddings/direct_r198_infonce_40ep_seed2_linear_lr1e-3_epoch10 | 1.26 GiB | superseded | scratch_quota | high |
| scratch/embeddings/direct_r198_infonce_40ep_seed2_linear_lr1e-3_epoch20 | 1.20 GiB | superseded | scratch_quota | high |

## Recoverable space

| Class | Estimate |
|-------|----------|
| **Conservative immediately recoverable (SAFE_DELETE)** | **22.53 GiB** |
| **Additional after review** | **91.80 GiB** (upper bound if all REVIEW removed; not recommended wholesale) |

Practical review tiers:
- ID-fixed seed-only R198: ~14.49 GiB scratch
- Move/delete exploratory scout HOME embeddings (see measured size in TSV)
- Deduplicate/regenerate probe caches (keep summaries / at least one tree)
- Home backup after full name inventory

## Top 10 largest retained (KEEP) artifacts

| path | size | why |
| --- | --- | --- |
| scratch/embeddings/direct_r198_40ep_linear_lr_full_extract | 95.13 GiB | Current 40-epoch DIRECT_H/TFMOE full-subgraph extracts; collaborator package source. |
| scratch/embeddings/final_corrected_no_preserve_multiseed | 69.07 GiB | Final corrected/no-preserve multiseed embeddings; thesis-facing. |
| scratch/embeddings/gcpal_txn_node_canonical | 59.03 GiB | GCPAL canonical npy embeddings referenced by txn-node reports. |
| scratch/embeddings/tfmoe_weight_ablation_lr2e-3_full_extract | 38.05 GiB | TFMOE weighting ablation full extracts including EXPERT_ONLY epochs. |
| pool/aml-data | 5.68 GiB | Formatted canonical datasets. |
| repo/results/diagnostics/final_corrected_no_preserve_multiseed/probabilities | 3.04 GiB | Final corrected probability arrays; thesis-facing. |
| pool/raw-aml-data | 1.98 GiB | Raw datasets. |
| scratch/results/cache/temporal_flow_causal | 824.15 MiB | Temporal-flow causal cache on scratch. |
| pool/morphology_cache | 757.31 MiB | Morphology cache (expensive rebuild). |
| pool/saved-models | 676.46 MiB | Canonical checkpoints including DIRECT_H/TFMOE/EXPERT_ONLY and supervised formal artifacts |

## Top 10 deletion candidates (by recoverable bytes)

| path | size | action | status | conf |
| --- | --- | --- | --- | --- |
| repo/results/diagnostics/final_exploratory_ssl_scout/embeddings | 22.53 GiB | REVIEW | complete | medium |
| scratch/results/cache/probe_weight_sweep_small_hi_key_runs | 16.08 GiB | REVIEW | complete | medium |
| scratch/results/cache/probe_features_small_li_current_protocol | 11.24 GiB | REVIEW | complete | medium |
| scratch/results/cache/alert_budget_metrics_current_protocol | 11.24 GiB | REVIEW | complete | medium |
| scratch/results/cache/probe_weight_sweep_small_li_current_protocol | 11.24 GiB | REVIEW | complete | medium |
| scratch/embeddings/gin_emlps_ports_tds_corrected_asym_proj_8192neg_queue0_40ep_seed2_last_ckpt | 2.50 GiB | REVIEW | complete | medium |
| repo/results/diagnostics/final_exploratory_ssl_scout/embeddings/smoke_aml_final_exploratory_ssl_scout_smoke_m_seed2_19001714 | 2.07 GiB | REVIEW | superseded | medium |
| scratch/embeddings/direct_r198_tfmoe_40ep_seed2_linear_lr1e-3_epoch03 | 1.70 GiB | REVIEW | superseded | high |
| scratch/embeddings/direct_r198_tfmoe_40ep_seed2_linear_lr1e-3_epoch40 | 1.68 GiB | REVIEW | superseded | high |
| scratch/embeddings/direct_r198_tfmoe_40ep_seed2_linear_lr1e-3_epoch30 | 1.66 GiB | REVIEW | superseded | high |

## Proposed deletion manifest (NOT EXECUTED)

```text
# SAFE_DELETE only — DO NOT RUN without explicit approval
# Invalid EdgeID-bug seed-only R198 embedding dirs (20 paths, ~22.52 GiB):
/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/direct_r198_infonce_40ep_seed2_linear_lr2e-3_epoch03
/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/direct_r198_infonce_40ep_seed2_linear_lr2e-3_epoch10
/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/direct_r198_infonce_40ep_seed2_linear_lr2e-3_epoch20
/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/direct_r198_infonce_40ep_seed2_linear_lr2e-3_epoch30
/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/direct_r198_infonce_40ep_seed2_linear_lr2e-3_epoch40
/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/direct_r198_infonce_40ep_seed2_linear_lr6p2e-3_epoch03
/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/direct_r198_infonce_40ep_seed2_linear_lr6p2e-3_epoch10
/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/direct_r198_infonce_40ep_seed2_linear_lr6p2e-3_epoch20
/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/direct_r198_infonce_40ep_seed2_linear_lr6p2e-3_epoch30
/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/direct_r198_infonce_40ep_seed2_linear_lr6p2e-3_epoch40
/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/direct_r198_tfmoe_40ep_seed2_linear_lr2e-3_epoch03
/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/direct_r198_tfmoe_40ep_seed2_linear_lr2e-3_epoch10
/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/direct_r198_tfmoe_40ep_seed2_linear_lr2e-3_epoch20
/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/direct_r198_tfmoe_40ep_seed2_linear_lr2e-3_epoch30
/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/direct_r198_tfmoe_40ep_seed2_linear_lr2e-3_epoch40
/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/direct_r198_tfmoe_40ep_seed2_linear_lr6p2e-3_epoch03
/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/direct_r198_tfmoe_40ep_seed2_linear_lr6p2e-3_epoch10
/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/direct_r198_tfmoe_40ep_seed2_linear_lr6p2e-3_epoch20
/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/direct_r198_tfmoe_40ep_seed2_linear_lr6p2e-3_epoch30
/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/direct_r198_tfmoe_40ep_seed2_linear_lr6p2e-3_epoch40
# Duplicate staged smoke checkpoints (keep sequential_aml_to_paysim_ssl/ copies):
/orcd/pool/007/jthi/Multi-GNN/saved-models/checkpoint_seq_aml2ps_smoke_aml_init_19017925.tar
/orcd/pool/007/jthi/Multi-GNN/saved-models/checkpoint_seq_aml2ps_smoke_aml_init_19017925_finetuned.tar
# Failed/incomplete embedding stubs (36 dirs) — see JSON/TSV for exact paths
```

Machine-readable: `results/diagnostics/storage_cleanup_audit_20260801.json` → `proposed_deletion_manifest`.

## Method notes

- Tools: `du`, `Path.rglob`, `stat`/`lstat`, `readlink -f`, `df -hT`, `quota -s`, `/home/jthi/orcd/.quota`
- No recursive hashing of large embeddings; reused recorded `checkpoint_sha256` in extract meta/cell JSON; hashed only small suspected checkpoint duplicates
- Cross-checked notes, thesis registry, submission manifests, DIRECT_H / TFMOE / EXPERT_ONLY reports, INVALID seed-only markers

## Confirmation

**No existing project artifacts were deleted or moved.** Audit outputs only were written.
