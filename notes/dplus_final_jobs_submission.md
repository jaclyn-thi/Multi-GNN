# D+ final jobs submission provenance

Submitted **exactly three** Advanced-GPU jobs (train only; no eval). Historical artifacts preserved.

## Source

| Field | Value |
|-------|-------|
| `git_head` | `ed7a15c18ab75f3d8d2e4600113c32a7f25046c2` |
| Dirty-diff + FT manifest sha256 | `f7c1dd21e0750576a92edff4767ab5a986f2bad564940d88caeaa5e641f443bb` |
| `model_settings.json` sha256 | `3a03cda40a4bbef31593a0ccbb9a56833471017952a98b8a91128935a700c11d` |
| `data_config.json` sha256 | `727ce692378fd7c98ab4d72e930b8a992680af59d8e95b6b72c1fa00d5f59ee0` |
| D+ seed-2 init ckpt sha256 | `a320920141f585c5825cbd63ce760a845fb434a9b162d4c87270dc72b0442b87` |

Full JSON: `results/diagnostics/dplus_final_jobs_submission_provenance.json`.

**Working-tree note:** the repo had substantial pre-existing dirty state beyond FT (~222 non-FT porcelain lines). Jobs run against the frozen on-disk tree at submission; **no further source edits** after submit.

## Pre-submit checks

- Focused FT + D+ regression tests: **45 passed**
  (`test_dplus_partial_finetune`, `test_correct_reverse_edge_features`, `test_preserve_seed_edges`, `test_contrastive_projection`, `test_contrastive_edge_identity`, `test_pre_embedding_capture_dimensions`)
- Seed-1 / seed-3 resolved args match seed-2 except `SEED` / `RUN_NAME` (and disjoint log paths)
- FT Stage-2 early-stop patience reset at transition (global best retained)
- Paths disjoint across A/B/C

## Jobs

| Job | Slurm ID | Unique name | Script |
|-----|----------|-------------|--------|
| A D+ seed 1 | **18801429** | `edge_dplus_corrected_preserve_40ep_seed1_final` | `slurm/train_edge_dplus_corrected_preserve_40ep_seed1_final.sh` |
| B D+ seed 3 | **18801432** | `edge_dplus_corrected_preserve_40ep_seed3_final` | `slurm/train_edge_dplus_corrected_preserve_40ep_seed3_final.sh` |
| C partial FT seed 2 | **18801435** | `dplus_partial_finetune_hxxtf_seed2` (label `edge_dplus_pre3h_hxtf_partial_finetune_seed2`) | `slurm/train_edge_dplus_pre3h_hxtf_partial_finetune_seed2.sh` |

Only these three were submitted. No eval / neighbor / txn-node / longer-than-40 jobs.