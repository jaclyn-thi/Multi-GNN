# DIRECT_H / DIRECT_H_TFMOE scheduled validation analysis

Locked validation-only extract + PaperStyleMLP probe for scheduled warmup+cosine runs.
**No encoder retraining. No test evaluation.**

## Jobs

| Arm | Job | Run |
|-----|-----|-----|
| DIRECT_H | 19251526 (`dir_h_sched`) | `direct_h_infonce_10ep_seed2_sched` |
| DIRECT_H_TFMOE | 19251528 (`dir_h_tfmoe_s`) | `direct_h_tfmoe_learned_alpha_10ep_seed2_sched` |

## Primary downstream (R198+X+TF → PaperStyleMLP)

| Epoch | H AUPRC | H F1@0.5 | H F1@val-threshold | TF AUPRC | TF F1@0.5 | TF F1@val-threshold | ΔAUPRC | ΔF1@val-threshold |
|------:|--------:|---------:|-------------------:|---------:|----------:|--------------------:|-------:|------------------:|
| 1 | 0.4704 | 0.4997 | 0.5122 | 0.4895 | 0.4673 | 0.5286 | +0.0191 | +0.0164 |
| 3 | 0.4874 | 0.5405 | 0.5432 | 0.5423 | 0.5256 | 0.5719 | +0.0549 | +0.0287 |
| 5 | 0.4724 | 0.5142 | 0.5165 | 0.5231 | 0.4803 | 0.5646 | +0.0507 | +0.0480 |
| 10 | 0.4659 | 0.5047 | 0.5105 | 0.5253 | 0.5474 | 0.5630 | +0.0594 | +0.0525 |

### Val-selected checkpoints (by primary AUPRC)

- DIRECT_H: epoch **3** (AUPRC=0.4874, F1@val-threshold=0.5432, F1@0.5=0.5405)
- DIRECT_H_TFMOE: epoch **3** (AUPRC=0.5423, F1@val-threshold=0.5719, F1@0.5=0.5256)

## References (validation only)

- Supervised Multi-GIN+EU, validation F1 (argmax), seed 2: **0.6101** (run `small_hi_legacy_supervised_gin_emlps_ports_50ep_seed2`, best epoch 43, decision rule: argmax over two-class logits; val AUPRC=0.5509; [source](notes/supervised_Small-HI_small_hi_legacy_supervised_gin_emlps_ports_50ep_seed2_summary.md)).
- Projected-encoder baselines (including former “projected F1@opt ≈ 0.571”) are **omitted** from figures and this table: threshold/split/seed provenance was not unambiguous enough for these DIRECT_H analyses.

## TF MoE diagnostics

| Epoch | α | w_c | w_tf0 | w_tf1 | w_tf2 | train MAE | val MAE |
|------:|--:|----:|------:|------:|------:|----------:|--------:|
| 1 | 0.600 | 0.600 | 0.133 | 0.133 | 0.133 | 0.299/0.156/0.355 | 0.242/0.156/0.198 |
| 3 | 0.550 | 0.550 | 0.165 | 0.165 | 0.119 | 0.246/0.133/0.328 | 0.251/0.124/0.191 |
| 5 | 0.504 | 0.504 | 0.194 | 0.198 | 0.104 | 0.199/0.074/0.317 | 0.176/0.063/0.185 |
| 10 | 0.460 | 0.460 | 0.226 | 0.227 | 0.088 | 0.119/0.053/0.313 | 0.163/0.042/0.182 |

Expert generalization verdict: **a) validation-generalizing prediction (val MAE tracks train; no large gap)**

## Integrity

- First-32 seed-edge hashes match across arms: **True**
- unique_negs_per_anchor NaN: denom_mode=sampled_8192: InfoNCE negatives are randomly sampled (with replacement from the aligned batch / bank), so uniqueness and duplicate counts are not tracked and remain NaN by design.

## Figures

1. `results/diagnostics/direct_h_tfmoe_scheduled_val_analysis/figures/01_val_auprc_vs_epoch.png`
2. `results/diagnostics/direct_h_tfmoe_scheduled_val_analysis/figures/02_val_f1_vs_epoch.png`
3. `results/diagnostics/direct_h_tfmoe_scheduled_val_analysis/figures/03_tf_mae_train_val_vs_epoch.png`
4. `results/diagnostics/direct_h_tfmoe_scheduled_val_analysis/figures/04_alpha_effective_weights.png`
5. `results/diagnostics/direct_h_tfmoe_scheduled_val_analysis/figures/05_raw_contrastive_loss.png`
6. `results/diagnostics/direct_h_tfmoe_scheduled_val_analysis/figures/06_repr_scale_effective_rank.png`
7. `results/diagnostics/direct_h_tfmoe_scheduled_val_analysis/figures/07_lr_dual_axis.png`

## Answers

1. **direct_r198_approaches_supervised_val_f1:** DIRECT_H selected val F1@val-threshold=0.5432 vs supervised Multi-GIN+EU seed2 validation F1 (argmax)=0.6101; gap=+0.0669. Does not approach supervised F1 at 10ep.
2. **tfmoe_improves_over_direct_h:** Yes (ΔAUPRC=+0.0549, ΔF1@val-threshold=+0.0287 at val-selected epochs).
3. **tf_objectives_with_meaningful_weight:** log1p_sender_interarrival (w=0.165), log1p_sender_past_7d_count (w=0.165)
4. **tf_experts_generalize_or_overfit:** a) validation-generalizing prediction (val MAE tracks train; no large gap)
5. **tfmoe_improved_representation_geometry:** TFMOE keeps smaller val L2 norm (11.41 vs 43.55) and effective rank 10.2 vs 2.0 at each arm's selected epoch
6. **longer_run_or_bce_moe_justified:** Optional follow-up only if TF continues to help geometry without AUPRC.
7. **no_encoder_retrain_no_test:** Confirmed: frozen checkpoint extract only; extract_splits=train,val; test_evaluated=false everywhere; no test.npz written or read.

