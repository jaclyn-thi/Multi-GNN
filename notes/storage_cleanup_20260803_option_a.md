# Storage cleanup Option A execution (2026-08-03)

Wave-1 migration of five completed frozen-evaluation embedding trees
SCRATCH → POOL with verified symlink cutover. **No SAFE_DELETE** in this pass
(Option A only). No Slurm submissions. Checkpoints, diagnostics, TF caches, and
datasets were not modified.

## Destination

`/orcd/pool/007/jthi/Multi-GNN/embeddings_archive/`

## Procedure (per tree)

1. Confirm empty `squeue -u jthi`
2. Full SHA256 source manifest
3. `rsync -a` copy with source intact
4. Match path set, file counts, file-size sums, full SHA256 dest manifest
5. `rsync --checksum --dry-run` with zero file diffs
6. Atomic rename source → `*.__migrating_to_pool__`; symlink original path → POOL
7. Validate symlink + representative SHA256s via repo `embeddings/<name>`
8. Remove scratch backup only after validation; restore on failure

Script: `scripts/storage_cleanup_20260803_option_a_migrate.py`  
Records: `results/diagnostics/storage_cleanup_20260803_option_a/`

## Results

| Tree | Status | Size (GiB) | Files | Manifest |
|------|--------|----------:|------:|----------|
| `financial_multidataset_shared_core_phase4b_objective_ablation_frozen_eval` | success | 37.57 | 27 | match |
| `financial_multidataset_shared_core_phase4b_mixed_long_frozen_eval` | success | 25.05 | 18 | match |
| `financial_multidataset_shared_core_phase4b_frozen_eval` | success | 21.18 | 15 | match |
| `expert_only_frozen_transfer_samld_paysim` | success | 35.37 | 24 | match |
| `smallhi_samld_mixed_ssl_phase3_frozen_eval` | success | 24.60 | 18 | match |

**All 5/5 succeeded. Reclaimed ≈ 143.76 GiB SCRATCH (by source `du -sbP`).**

### Symlinks (logical path preserved)

Each of the five paths under
`/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings/<name>`
(and repo `embeddings/<name>`) is now a symlink to
`/orcd/pool/007/jthi/Multi-GNN/embeddings_archive/<name>`.
Post-cutover validation: **all_ok=true**
(`validation.json`).

## Usage

| Metric | Before | After |
|--------|-------:|------:|
| Scratch Multi-GNN (`du -P -sh`) | 892G | **748G** |
| Pool Multi-GNN (`du -P -sh`) | 54G | **156G** |
| Quota SCRATCH (file; may lag) | 891.7 / 1024 | 829.1 / 1024 (snapshot during run; expect ~748 after refresh) |
| Quota POOL (file; may lag) | 53.8 / 1024 | 53.8 / 1024 (refresh pending; `du` shows 156G) |

Notes:

- Per-tree `du -sbP` source and dest sizes matched at **143.76 GiB** total;
  **file-size sums and full SHA256 manifests matched** before every cutover.
- Pool `du -sh` of `embeddings_archive/` reports ~102G (mount accounting);
  scratch reclaim measured by project `du` is **144G** (892→748).
- Quota `.quota` updates asynchronously (POOL still showed 53.8 during validation).

## Confirmations

- No failed/restored trees
- No leftover `*.__migrating_to_pool__` backups
- Checkpoints / diagnostics aggregates / TF caches still present
- No dataset or HOME diagnostic deletions
- Logical embedding paths remain addressable via symlink

## Not done (still awaiting separate approval)

- SAFE_DELETE forensic preflight stubs + `storage_symlink_smoke` (~0.02 GiB)
- Wave-2 migrations (`direct_r198`, TFMOE ablation, GCPAL, …)
- Any REVIEW_DELETE
