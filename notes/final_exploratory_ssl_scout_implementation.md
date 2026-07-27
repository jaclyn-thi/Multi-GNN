# Final exploratory SSL scout — C0 + M implementation (wave 1)

> Locked GO from [`notes/final_exploratory_ssl_scout_preflight.md`](final_exploratory_ssl_scout_preflight.md).
> This turn implements **C0 and M only**. J / JM / JC / CORAL / dual-domain / structural-reliance are **not** implemented.
> Timestamp (UTC): 2026-07-27

## Source checkpoint (verified before any work)

| Field | Value |
|-------|-------|
| Path | `saved-models/checkpoint_gin_emlps_ports_tds_corrected_asym_proj_8192neg_queue0_40ep_seed2.tar` |
| SHA256 | `18e06f555aa4880dfc1e95caa3f54a207e5aa186d266887772640feb93a06ae6` |
| Verified | yes (matches preflight lock) |

## Arms

Both start from the source checkpoint via staged `checkpoint_{unique}.tar` copies + `--finetune`.

| Arm | Objective | Budget |
|-----|-----------|--------|
| **C0** | Unchanged corrected/no-preserve InfoNCE | exactly **500** optimizer steps |
| **M** | InfoNCE + morph expert MSE | same 500 steps |

**Optimizer provenance (explicit):** `checkpoint_weight_continuation_with_optimizer_reset`  
Adam is rebuilt after weight load (not a true optimizer-state resume).

**500-step accounting:** 397 microbatches/epoch × accum=4 → 100 opt steps/epoch → `--n_epochs 5` + `--max_optimizer_steps 500`.

## Morphology objective (M) — aggregation lock

Preflight CLI + existing shared-head path make aggregation **unambiguous**:

\[
\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{contrastive}} + 0.05 \cdot \mathcal{L}_{\mathrm{morph}}
\]

- `--morph_target_groups degree_fan,flow_balance` filters columns.
- `--morph_expert_layout shared` (default): **one** MSE over the filtered vector.
- `--morph_expert_weight 0.05` multiplies that MSE (`cfg.loss_weight` in `morphology_expert_step`).
- Group block weights apply only in `grouped` layout — **not used**.
- Standardization = existing `transform_morph_targets` (log1p on count-like cols).
- **Never** the historical degflow `λ=1.0` recipe.

Targets: AMLWorld **train** topology/features via leakage-audited caches under `morphology_cache/Small-HI/` (`train_node_morphology.csv`, `train_node_flow_balance.csv`). Missing/misaligned IDs raise (no silent fill).

**Morph head init seed (separate from encoder seed 2):** `42` (`--morph_expert_init_seed 42`).

## Matched controls (C0 ↔ M)

Identical: initial base tensors (proven pre-step), seed 2, AMLWorld train split, batch/aug step seeds, step count, batch size 8192, 8192 negatives, queue 0, accum 4, T=0.5, LR, Adam reset, corrected reverse ON, `preserve_seed_edges` OFF, all non-morph flags.

Before step 1 smoke proves: shared trainable tensors equal; match source ckpt; only M has newly initialized morph-head params.

## Label / leakage restrictions

Encoder training never uses AML laundering labels, PaySim fraud labels, or val/test labels/topology.

## Validation-only evaluation (later; not this smoke turn)

AMLWorld: clean frozen pre-3h H+X+TF (PaperStyleMLP) + post-128 H diagnostic.  
PaySim: post-128 H-only, `paysim_legacy_duplicate_v1`, train-fit edge z-norm, frozen AML BN, logistic `class_weight=model`, C=1, downstream seed=1.  
No test. Retain original uncontinued ckpt val metrics as no-continuation reference; **M vs C0** is primary.

Clean extraction discards/ignores the morph head; base embeddings are invariant to its presence.

## Predeclared decision gate (M vs C0)

1. PaySim val AUPRC +≥0.003 **or** F1 +≥0.01 vs C0  
2. PaySim pretrained remains above matched random  
3. AMLWorld pre-3h H+X+TF val AUPRC regression ≤0.02 vs C0  
4. Coverage, leakage, gradient, non-collapse checks pass  

Also report C0 vs original checkpoint separately.

## Code changes

| File | Change |
|------|--------|
| `util.py` | `--max_optimizer_steps`; `--morph_expert_init_seed` |
| `training.py` | Hard stop on max opt steps; scout batch hashes; morph init reseed; optimizer provenance log |
| `scripts/final_exploratory_ssl_scout.py` | Orchestration (smoke + train_arm scaffolding) |
| `slurm/run_final_exploratory_ssl_scout_smoke.sh` | Advanced GPU, 1 GPU, workers=0 |
| `tests/test_final_exploratory_ssl_scout.py` | Focused unit tests |

## Smoke

- Submit: `sbatch slurm/run_final_exploratory_ssl_scout_smoke.sh`
- Artifact: `results/diagnostics/final_exploratory_ssl_scout/smoke.json`
- Submission record: `results/diagnostics/final_exploratory_ssl_scout/submission_smoke.json`
- Full C0/M 500-step jobs and aggregation: **not submitted** this turn.
- Embeddings root for this scout: `results/diagnostics/final_exploratory_ssl_scout/embeddings` (repo `embeddings/` symlink points at missing scratch).

### Smoke pass conditions

1. Source sha256 matches lock  
2. Pre-step C0/M shared tensors identical and match source; only M has morph head  
3. Optimizer provenance logged as `checkpoint_weight_continuation_with_optimizer_reset`  
4. Matched C0/M batch seed-id hashes  
5. Exactly 2 optimizer steps per arm (smoke); finite losses; M has morph component; C0 does not  
6. Finetuned ckpt save/reload; morph state only on M; encoder params moved  
7. Clean extract (no `test.npz`); tiny AMLWorld + PaySim val probe paths run  
8. Projected full-arm runtime (~0.5 h) safely below 6 h  

## Not in this turn

J, JM, JC, CORAL, dual-domain trainer, structural-reliance suite, full 500-step C0/M trains, aggregation, test evaluation.
