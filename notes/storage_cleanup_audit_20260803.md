# Storage cleanup audit (2026-08-03) — READ-ONLY

**Nothing was deleted, moved, truncated, compressed, overwritten, or symlink-replaced. No Slurm jobs were submitted.**

Artifacts:

- [`results/diagnostics/storage_cleanup_audit_20260803.json`](../results/diagnostics/storage_cleanup_audit_20260803.json)
- [`results/diagnostics/storage_cleanup_audit_20260803_candidates.tsv`](../results/diagnostics/storage_cleanup_audit_20260803_candidates.tsv)
- [`results/diagnostics/storage_cleanup_audit_20260803_migration_manifest.tsv`](../results/diagnostics/storage_cleanup_audit_20260803_migration_manifest.tsv)
- [`results/diagnostics/storage_cleanup_audit_20260803_deletion_manifest.tsv`](../results/diagnostics/storage_cleanup_audit_20260803_deletion_manifest.tsv)

Measurement method: `du -sbP` / `du -P -sh`, directory walks for filenames only, existing JSON/notes/job records. **No NPZ / embedding arrays were loaded.**

---

## A. Quota and physical-path snapshot

Source: `/home/jthi/orcd/.quota` (Mon Aug 03 18:11:18 EDT 2026) + live `squeue` / path resolution.

| Space | Usage | Limit | % Used |
|-------|------:|------:|-------:|
| HOME | 129.1 GiB | 200.0 GiB | 64.55% |
| SCRATCH | **891.7 GiB** | 1024.0 GiB | **87.08%** |
| POOL | 53.8 GiB | 1024.0 GiB | 5.26% |

### Topology (logical → physical → quota)

| Role | Logical (repo) | Physical | Space |
|------|----------------|----------|-------|
| Repository | `/home/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN` | same (HOME NFS) | HOME ~28.95 GiB no-follow top-level |
| Embeddings | `embeddings/` | `/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings` (~890.7 GiB, 81 dirs) | SCRATCH |
| Results cache | `results/cache/` | `/orcd/scratch/orcd/008/jthi/Multi-GNN/results/cache` (~1.1 GiB real; probe caches are symlinks) | SCRATCH |
| Saved models | `saved-models/` | `/orcd/pool/007/jthi/Multi-GNN/saved-models` | POOL |
| Datasets | `aml-data/`, `raw-aml-data/` | `/orcd/pool/007/jthi/Multi-GNN/...` | POOL |
| Morphology | `morphology_cache/` | `/orcd/pool/007/jthi/Multi-GNN/morphology_cache` | POOL |
| Prior cache archive | (via cache symlinks) | `/orcd/pool/007/jthi/Multi-GNN/results/cache_archive/` (~49 GiB) | POOL |
| **Proposed** embedding archive | — | `/orcd/pool/007/jthi/Multi-GNN/embeddings_archive/` | **does not exist yet** |

Scratch project breakdown: embeddings **891 GiB**, results **1.1 GiB**, smoke **16 KiB**. Probe feature caches from Aug 1 already charge POOL via symlinks (0 SCRATCH bytes under `du -P`).

**Slurm:** `squeue -u jthi` empty → no path classified `ACTIVE_KEEP` from live jobs. Re-check before any later operation.

---

## B. Top 30 largest SCRATCH embedding directories

| GiB | Files | Status | Directory |
|----:|------:|--------|-----------|
| 95.13 | 90 | complete_train_val | `direct_r198_40ep_linear_lr_full_extract` |
| 69.07 | 88 | complete_with_test | `final_corrected_no_preserve_multiseed` |
| 59.03 | 112 | complete_npy_extracts | `gcpal_txn_node_canonical` |
| 39.58 | 30 | complete_with_test | `paysim_dplus_transfer_final` |
| 38.05 | 36 | complete_train_val | `tfmoe_weight_ablation_lr2e-3_full_extract` |
| 37.57 | 27 | complete_train_val | `financial_multidataset_shared_core_phase4b_objective_ablation_frozen_eval` |
| 36.13 | 45 | complete_train_val | `schema_mask_scout` |
| 35.37 | 24 | complete_train_val | `expert_only_frozen_transfer_samld_paysim` |
| 25.05 | 18 | complete_train_val | `financial_multidataset_shared_core_phase4b_mixed_long_frozen_eval` |
| 24.60 | 18 | complete_train_val | `smallhi_samld_mixed_ssl_phase3_frozen_eval` |
| 22.59 | 28 | complete_with_test | `paysim_preserve_normalization_ablation` |
| 21.18 | 15 | complete_train_val | `financial_multidataset_shared_core_phase4b_frozen_eval` |
| 20.46 | 24 | complete_train_val | `sequential_aml_to_paysim_ssl` |
| 15.64 | 20 | complete_with_test | `paysim_regression_audit` |
| 14.96 | 18 | complete_train_val | `paysim_feature_contract_gate_seed2` |
| 11.33 | 12 | complete_train_val | `joint_replay_scout` |
| 10.42 | 8 | complete_with_test | `small_li_gin_emlps_tds_..._emb198_seed1` |
| 8.61×4 | ~8–10 | complete_with_test | four `small_li_*` seed trees |
| 7.38 | 14 | complete_npy_extracts | `gcpal_txn_node_posagg` |
| ~6.32×N | ~8–10 | complete_with_test | many `hi_*` / `gin_emlps_*` contrastive trees |

Full inventory: JSON `top30_scratch_embedding_dirs` + candidates TSV.

---

## C. Space grouped by category (SCRATCH-centric)

| Category | Approx size | Notes |
|----------|------------:|-------|
| Embeddings | **890.7 GiB** | Dominates SCRATCH |
| Caches (live TF) | **~1.09 GiB** | `temporal_flow_causal` 825 MiB + `..._samld_...` 262 MiB — **PROTECTED** |
| Probe caches | **0 SCRATCH** | Symlinked to POOL `cache_archive` (Aug 1) |
| Checkpoints (multi-dataset) | **~0.1 GiB on HOME** | `results/checkpoints/` — **PROTECTED** |
| Diagnostics/results | **~27.8 GiB on HOME** | JSON/CSV/figures/predictions — **PROTECTED** |
| Partial/failed job residue | **~0.02 GiB** | Two forensic preflight stubs only |

Aug 1 already removed empty extract stubs and invalid seed-only R198 trees; little SAFE_DELETE residue remains.

---

## D. Total immediately SAFE_DELETE

| Path | Size | Evidence |
|------|-----:|----------|
| `embeddings/forensic_preflight_fullgraph_pre3h` | 0.012 GiB | partial (`all.npz` pattern / incomplete product) |
| `embeddings/forensic_preflight_fullgraph_post` | 0.008 GiB | partial |
| `storage_symlink_smoke` | 16 KiB | smoke leftover |

**Total SAFE_DELETE ≈ 0.02 GiB** (negligible vs 100 GiB goal).

---

## E. Total recommended MIGRATE_TO_POOL (Wave-1)

Destination root (proposed): `/orcd/pool/007/jthi/Multi-GNN/embeddings_archive/`

| Logical tree | GiB | Cells | Metrics / ckpts retained? |
|--------------|----:|------:|---------------------------|
| `..._phase4b_objective_ablation_frozen_eval` | 37.57 | 9 | yes (aggregate + twin JSON + notes + ckpts) |
| `..._phase4b_mixed_long_frozen_eval` | 25.05 | 6 | yes |
| `..._phase4b_frozen_eval` | 21.18 | 5 | yes |
| `expert_only_frozen_transfer_samld_paysim` | 35.37 | 8 | yes (diagnostics dir + notes) |
| `smallhi_samld_mixed_ssl_phase3_frozen_eval` | 24.60 | — | yes |

**Wave-1 total ≈ 143.76 GiB** (exceeds 100 GiB alone).  
POOL headroom ≈ 970 GiB — fits easily.  
Migrate **parent trees only** (not nested cells independently). Preserve original logical path via symlink after verified cutover.

### Wave-2 (optional, after Wave-1)

| Tree | GiB | Notes |
|------|----:|-------|
| `direct_r198_40ep_linear_lr_full_extract` | 95.13 | cold; regenerable; still “canonical” historically |
| `tfmoe_weight_ablation_lr2e-3_full_extract` | 38.05 | cold ablation extract |
| `gcpal_txn_node_canonical` | 59.03 | `.npy` extracts (not npz) |
| `gcpal_txn_node_posagg` | 7.38 | `.npy` |

**Wave-2 additional ≈ 199.6 GiB.**

Careful optional: `final_corrected_no_preserve_multiseed` **69.07 GiB** (historically protected multiseed + test).

---

## F. Additional REVIEW_DELETE upper bound

Complete but regenerable / uncertain-usefulness SCRATCH embeddings (old `hi_*`, `small_li_*`, PaySim transfer trees, schema/joint/sequential scouts, epoch sched extracts, etc.):

**Upper bound ≈ 478.3 GiB** if all REVIEW_DELETE candidates were eventually removed.

Do **not** treat this as approved deletion. Prefer migrate-over-delete when uniqueness is uncertain.

---

## G. Projected SCRATCH usage

| Scenario | Projected SCRATCH |
|----------|------------------:|
| Current | **891.7 GiB** (87.1%) |
| SAFE_DELETE only | **~891.7 GiB** (no meaningful change) |
| SAFE_DELETE + Wave-1 migration | **~747.9 GiB** (~73.0%) |
| + Wave-2 migration | **~548.3 GiB** (~53.5%) |
| + all REVIEW_DELETE (upper bound) | **~70.0 GiB** (aggressive; not recommended without review) |

---

## H. Recommended operation order (future — not executed)

1. Re-check `squeue -u jthi` (and any path references in pending scripts).
2. SAFE_DELETE forensic preflight stubs + `storage_symlink_smoke`.
3. `mkdir` POOL `embeddings_archive/`.
4. For each Wave-1 tree, one at a time:
   - copy → pathset/filecount/bytes/SHA256 manifest → `rsync --checksum --dry-run` zero diff → atomic replace with symlink → validate → record quota/`du`.
5. Spot-check logical `embeddings/<name>/...` still resolves; do not load full Z arrays.
6. Optionally Wave-2.
7. Only then human-triage REVIEW_DELETE.

---

## I. Explicit protected-path inventory

Classify **PROTECTED_KEEP** (do not delete/migrate without separate approval beyond embedding cold-archive plan):

- Source code, tests, `.git`
- `notes/`
- `results/diagnostics/**` (JSON, CSV, figures, manifests, probe outputs, predictions)
- `results/checkpoints/**` including Phase-3/4, MIXED_3DOMAIN_LONG, EXPERT_ONLY / INFONCE_ONLY / PROJECTION_ON_ADAPTIVE
- Phase-3 / Phase-4 training-integrity artifacts; shared init / stream-matching records
- Objective-ablation frozen-eval tables and predictions (HOME)
- Gradient-conflict diagnostic (HOME only; no SCRATCH embedding tree)
- TF caches: scratch `temporal_flow_causal`, `temporal_flow_causal_samld_shared_core_v1`; HOME `temporal_flow_cache`
- POOL datasets, `saved-models`, `morphology_cache`, existing `results/cache_archive`

Frozen-eval **embedding matrices** themselves are proposed for **MIGRATE_TO_POOL** (cold, metrics retained), not PROTECTED_KEEP on SCRATCH.

---

## J. Exact blockers requiring human judgment

1. Approve Wave-1 migration of five recent frozen-eval trees (~144 GiB) despite very recent completion?
2. Include Wave-2 (`direct_r198` 95G, TFMOE ablation 38G, GCPAL 66G)?
3. Allow careful migration of historically protected `final_corrected_no_preserve_multiseed` (69G)?
4. Any REVIEW_DELETE of old contrastive/PaySim extracts, or migrate-only policy?
5. Confirm no external absolute SCRATCH paths bypass repo symlinks.
6. HOME diagnostics (~28 GiB) remain a separate HOME-quota issue (out of SCRATCH scope).

---

## Recent work: canonical vs abandoned

| Workstream | SCRATCH embeddings | Verdict |
|------------|--------------------|---------|
| `phase4b_objective_ablation_frozen_eval` | 9 complete cells, 37.57 GiB | **Canonical.** Jobs 19585027–19585032 COMPLETED. Train: 19575819_0 OK; `_1` FAILED gate-only (ckpts revalidated); `_2` CANCELLED → 19579989 COMPLETED. No abandoned extract stubs. |
| `phase4b_mixed_long_frozen_eval` | 6 complete cells, 25.05 GiB | **Canonical** LONG@1500/3000 products. |
| `phase4b_frozen_eval` | 5 complete cells, 21.18 GiB | **Canonical** (incl. step500 LI diagnostic). |
| `expert_only_frozen_transfer_samld_paysim` | 8 complete cells, 35.37 GiB | **Canonical.** |
| `financial_multidataset_long_gradient_conflict` | none on SCRATCH | HOME diagnostics only — **PROTECTED_KEEP**. |
| Smoke / failed / preempted extract dirs for those jobs | **not present** | Prior Aug 1 stub cleanup + successful final DAG left no partial embedding trees. |

---

## Safest ways to reclaim ≥100 GiB (stop for approval)

| Option | Action | SCRATCH reclaim | Risk | Requires approval |
|--------|--------|----------------:|------|-------------------|
| **A (recommended)** | MIGRATE Wave-1 five frozen-eval trees → POOL `embeddings_archive/` + symlink cutover | **~143.8 GiB** | Low if verify gates pass | **Yes — awaiting** |
| B | A + MIGRATE Wave-2 (`direct_r198` + TFMOE ablation + GCPAL) | **~343.3 GiB** | Low–medium (larger; re-grep refs) | Yes |
| C | SAFE_DELETE only | **~0.02 GiB** | Negligible | Yes (tiny) |
| D | REVIEW_DELETE old SSL trees | up to **~478 GiB** | Higher — needs per-tree regenerability proof | Yes, after triage |

**No operations executed in this pass.** Awaiting approval to proceed with Option A (and optionally B/C).
