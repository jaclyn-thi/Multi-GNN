# Final corrected/no-preserve multiseed — Stage 0 preflight

- Timestamp (UTC): `2026-07-27T00:34:03.727827+00:00`
- Git commit: `5d941165a07ae29c6cdd82e16c90a334d0ff80a8`
- Dirty files: 19
- Test used: **false**
- Seed 5 required: **false**

## Checkpoint provenance (seeds 1–4)

| Seed | Role | Epoch | preserve | corrected | emb | sha256 |
|-----:|------|------:|:--------:|:---------:|----:|--------|
| 1 | confirmation | 40 | False | True | 128 | `5e59b5f214700374…` |
| 2 | development | 39 | False | True | 128 | `18e06f555aa4880d…` |
| 3 | confirmation | 40 | False | True | 128 | `4ea55c74a55e6577…` |
| 4 | confirmation | 40 | False | True | 128 | `31aae0f9b3e8040e…` |

Checkpoint epoch field is 1-based (save_model stores epoch+1). Seed2 best checkpoint is epoch 39 of a 40-epoch run; seeds 1/3/4 best==final epoch 40. All trained with n_epochs=40 and checkpoint_policy=best under the same unique_name recipe.

## Recipe consistency

All four checkpoints share: GIN hetero emlps, ports+TDS edge_dim=8, corrected reverse semantics, preserve_seed_edges=False, 128-d embedding, asymmetric 128-d projection head, unique_name template `gin_emlps_ports_tds_corrected_asym_proj_8192neg_queue0_40ep_seed{N}`.

## Existing code paths to reuse

- AMLWorld pre-3h H / H+X+TF PaperStyleMLP: `scripts/final_corrected_no_preserve_5seed.py::run_amlworld` / preserve-ablation `amlworld_eval_a`
- PaySim feature contracts: `feature_contracts.py` + gate script (legacy primary)
- Target-train BN recal: `recalibrate_bn` (labels unused)
- New orchestrator for this pass: `scripts/final_corrected_no_preserve_multiseed.py` (seeds 1–4; P1/P2/P3; post-128 diagnostic; val-selected ensemble)

## Cache reuse

**No prior cache reused.** All embeddings re-extracted under `embeddings/final_corrected_no_preserve_multiseed/`.
- `seed2_amlworld_pre3h_preserve_ablation`: reuse=False — New final-results directory required; working tree dirty vs original extract; re-extract under final_corrected_no_preserve_multiseed/
- `seed2_amlworld_post128_schema_mask_control`: reuse=False — train/val only (no test); not usable for final test eval
- `paysim_fcg_random_legacy`: reuse=False — extract_splits=train,val only; missing test
- `paysim_preserve_random_trainfit`: reuse=False — feature_contract_id missing/None; cannot prove exact legacy contract metadata match

## Proceed

Stop conditions clear. Proceed to script creation and `sbatch` DAG submission.

