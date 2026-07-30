# DIRECT_H / DIRECT_H_TFMOE scout preflight (collaborator sequence)

> Twin: `results/diagnostics/projection_free_tf_expert_scout_preflight.json`  
> **This experiment prioritizes the PhD collaborator’s requested direct-H design.**  
> It is **not** a four-arm projection-location ablation (P/D128/D198/D198TF).  
> It is **not** an exact PaPaGei reproduction (TF MoE is PaPaGei-*inspired* only).  
> Status: design locked; implementation in `direct_r198/`, CLI flags, hetero training path,
> dual-arm smoke (`slurm/direct_r198_dual_arm_smoke.sh`). Full 10ep jobs are **manual only**.

## Scientific sequence (locked)

Two **from-scratch** corrected/no-preserve Multi-GIN trainings on Small-HI, seed **2**,
**10 epochs**, validation-only downstream. No test. No continuation. No historical
overwrite. References (supervised Multi-GIN+EU; projected contrastive) are
**not** retrained.

### Tensor names

| Alias | Code | Dim | Role |
|-------|------|----:|------|
| **R198** | `pre_embedding_3h` | 198 | Edge readout `cat(relu(src‖dst), edge_attr)` = `3*n_hidden` |
| H128 | `post_embedding` | 128 | `embedding_head` output — **bypassed / not used** in these arms |
| Z128 | proj head out | 128 | **Not instantiated** |

Primary frozen downstream: **R198** (+ locked X+TF for primary probe).

---

## Experiment 1: `DIRECT_H`

Run name: `direct_h_infonce_10ep_seed2`

```
graph → Multi-GIN → R198 → InfoNCE
```

- Bypass `GINe.embedding_head` (do not use H128 for SSL).
- Do **not** instantiate contrastive projection.
- Cosine InfoNCE on R198; asymmetric views; established negs/aug/batch/accum/T/graph flags.
- Grads update Multi-GIN through R198.
- No AML labels.
- Frozen primary: R198.

## Experiment 2: `DIRECT_H_TFMOE`

Run name: `direct_h_tfmoe_learned_alpha_10ep_seed2`

```
graph → Multi-GIN → R198 → InfoNCE
                       ├→ TF MoE₁ (sender interarrival)
                       ├→ TF MoE₂ (sender past-7d count)
                       └→ TF MoE₃ (amount vs sender past mean)
```

Targets (cache names, after confirmation):

1. `log1p_sender_interarrival`
2. `log1p_sender_past_7d_count`
3. `log1p_amount_vs_sender_past_mean`

Past-only; no labels; train-only standardization; **not** encoder inputs;
seed EdgeID alignment. Cache: `results/cache/temporal_flow_causal/Small-HI/`.

Each MoE (PaPaGei-inspired scalar):

- 3 experts: `Linear(198,64)→ReLU→Linear(64,1)`
- gate: `Linear(198,3)→Softmax`
- pred = Σ gateᵢ · expertᵢ
- loss = MAE on standardized scalar

Attach to **view-1 gradient-carrying R198 seeds**. Experts discarded at extract.

### Literal learned alphas

```
alpha = sigmoid(alpha_logit)
beta  = softmax(beta_logits)           # length 3

L_aux = Σ_m beta[m] * L_tf_normalized[m]
L_total = alpha * L_contrast_normalized + (1-alpha) * L_aux

w_contrast = alpha
w_tf_m     = (1-alpha) * beta[m]       # ≥0, sum_m w_tf_m + w_contrast = 1
```

Init: `alpha=0.6`, `beta=[1/3,1/3,1/3]`.

**Epoch 1 calibration (fixed weights):** freeze alpha/beta; collect detached mean
raw loss per objective; freeze those means as normalization constants.  
**Epoch ≥2:** learn alpha/beta (param group lr=`1e-3`). Never use labels or val
metrics to learn alpha.

---

## Matched recipe controls (both arms)

Corrected/no-preserve: Small-HI, gin, `--reverse_mp --ego --ports --emlps --tds
--correct_reverse_edge_features`, preserve OFF, batch 8192, 8192 negs, queue 0,
accum 4, asymmetric, T=0.5, seed 2. Identical loader/aug/neg stream barriers;
log first-32 seed-edge hashes. Checkpoints at epochs **1, 3, 5, 10**.

## Evaluation (val only)

- Primary: frozen R198 + locked X + causal TF → PaperStyleMLP  
- Diagnostic: frozen R198 only  
- Report val AUPRC and F1; **do not** select by SSL loss.  
- No test.

## Gates

1. How close is DIRECT_H val F1 to existing supervised Multi-GIN+EU reference?
2. Does DIRECT_H_TFMOE improve over DIRECT_H by **≥0.003 val AUPRC** or
   **≥0.01 val F1**, without collapse/coverage failure?

BCE+expert fallback: **design only if direct contrastive fails** — not submitted now.

## Local deliverables (no online W&B)

Step JSONL/CSV, epoch JSON, PNGs: total/raw/norm InfoNCE; each TF loss;
alpha; betas; effective weights; weighted contributions; MoE gate/utilization;
LR; encoder & alpha grad norms; R198 variance/collapse; val AUPRC/F1 by ckpt.

Offline W&B is **not** required and **not** currently supported (`online|disabled` only).

## Implementability / jobs

| Arm | Code needed | Est. GPU |
|-----|-------------|----------|
| DIRECT_H | R198 SSL path (bypass emb+proj) | ~2 h / 10ep |
| DIRECT_H_TFMOE | + MoE + α/β + calibration logging | ~2.2 h / 10ep |

Smoke: **one** lightweight job covering **both** paths
(`slurm/direct_r198_dual_arm_smoke.sh`). Full trains: **manual only** —
`slurm/train_direct_h_infonce_10ep_seed2.sh` and
`slurm/train_direct_h_tfmoe_learned_alpha_10ep_seed2.sh` (never auto-submitted).

## Explicit non-goals

- Not P / D128 / D198 / D198TF four-arm ablation.
- Not exact PaPaGei reproduction.
- Not projection-head location sweep.
- Not retraining projected contrastive or supervised Multi-GIN references.
