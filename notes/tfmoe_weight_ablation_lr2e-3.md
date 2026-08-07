# TFMOE objective-weighting ablation (AMLWorld Small-HI, seed 2, peak LR 2e-3)

## Audit of production TFMOE weighting (before ablations)

| Item | Behavior |
|------|----------|
| **α** | `α = sigmoid(α_logit)`, init 0.6 (`LearnedAlphaBeta`) |
| **β** | `β = softmax(β_logits)`, init uniform 1/3 |
| **w_contrast** | `α` |
| **w_tf_i** | `(1 − α) · β_i` |
| **Normalization** | Epoch-1 mean calibration; thereafter `L_norm = L_raw / μ` |
| **L_total** | `α · L_c_norm + (1 − α) · Σ_m β_m · L_tf_norm_m` |
| **Gradients** | InfoNCE → encoder; TF MAE → encoder + MoE heads; α/β after epoch 1 |
| **α/β LR** | Separate Adam group base LR **1e-3** (encoder peak LR for this study = **2e-3**); same schedule multiplier |

## Ablation math

1. **FIXED_BALANCED:** `w_contrast=0.5`, `w_tf_i=1/6` (frozen; α/β not learned).
2. **ADAPTIVE_CONTRAST_FLOOR:** keep learned α/β; `w_contrast = max(α, 0.25)`, `w_tf_i = (1 − w_contrast) · β_i`.
3. **EXPERT_ONLY (option a):** `w_contrast=0`, `w_tf_i=β_i` with **learned** β; InfoNCE logged but **excluded** from `L_total` (zero contrastive encoder grad).
4. **FIXED_CURRENT_EARLY:** freeze exact epoch-10 adaptive weights from `checkpoint_direct_r198_tfmoe_40ep_seed2_linear_lr2e-3_epoch10.tar`:
   - w_contrast ≈ 0.371551
   - w_tf ≈ [0.268914, 0.297941, 0.061593]

## Evaluation

- Full-subgraph extract (not seed-only)
- Probe: R198-only, PaperStyleMLP 20ep / lr 1e-3 / bs 8192 / seed 2
- Primary metric selection: best validation AUPRC
- Test locked
- Output: `results/diagnostics/tfmoe_weight_ablation_lr2e-3/`

## Submit

```bash
bash slurm/submit_tfmoe_weight_ablation.sh
```
