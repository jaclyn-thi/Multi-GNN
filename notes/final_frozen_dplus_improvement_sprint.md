# Final frozen D+ improvement sprint

Secondary frozen evaluations only. **Does not modify** `final_dplus_multiseed_and_finetune_analysis` primary metrics.

## Constraints

- No GNN/encoder training; no supervised gradients into any encoder.
- No new architecture, contrastive objective, or feature/learner/weighting sweep.
- No test-driven selection; no automatic follow-up jobs.
- Used existing best-score embeddings; `_last` extract only for seeds 1 and 3.

## Primary (unchanged)

Locked three-seed frozen D+ best-score aggregate (test F1@0.5 0.6393 ± 0.0186) remains the primary robustness result.

## Experiment A — Equal-weight frozen D+ ensemble

Post-hoc equal-weight ensemble of independently self-supervised frozen D+ encoders.

- Rule: `mean(seed1_probability, seed2_probability, seed3_probability)` (no weight search; no logit averaging).
- Common test IDs: **862849** (sha256 `c4eeeae4f4fbe2cf…`); positives retained: **1611** / 1611.
- Val-selected threshold: **0.35**

| Split / thr | AUROC | AUPRC | F1 | P | R | PPR |
|-------------|------:|------:|---:|--:|--:|----:|
| test @0.5 | 0.9884 | 0.7052 | **0.6994** | 0.7230 | 0.6772 | 0.001749 |
| test @val-thr | 0.9884 | 0.7052 | 0.6335 | 0.5663 | 0.7188 | 0.002370 |

- Confusion @0.5: TP=1091 FP=418 TN=860820 FN=520
- P@100/500/1000 @0.5: 0.980 / 0.956 / 0.893

### Comparisons (fixed-0.5 F1)

- Frozen three-seed mean: 0.6393 ± 0.0186
- Frozen seed-2: 0.6559
- Multi-GIN+EU: 0.6479 ± 0.0122
- Ensemble: **0.6994** (Δ vs Multi-GIN+EU = +0.0515; exceeds=True)

## Experiment B — `_last` / fixed epoch-40 validation gate

| Seed | Checkpoint | Val AUPRC | vs locked best |
|-----:|------------|----------:|---------------:|
| 1 | `_last` ep40 | 0.517043 | -0.002304 |
| 2 | best=ep40 | 0.550000 | +0.000000 |
| 3 | `_last` ep40 | 0.512696 | -0.028299 |
| **mean** | fixed ep40 | **0.526580** | Δ vs best-mean -0.010201 |

- Gate requires mean val AUPRC ≥ 0.541781 (locked best mean 0.536781 + 0.005).
- **Gate passed: False**. `_last` test evaluation permitted: **False**.
- Test was not inspected during the gate.

Gate failed → retained existing best-score checkpoint policy; no `_last` test evaluation.

## Final answers

1. Ensemble fixed-0.5 F1 / AUPRC: **0.699359** / **0.705247**
2. Numerically exceeds Multi-GIN+EU: **True**
3. Fixed-ep40 val gate: mean=0.526580 (Δ=-0.010201; need ≥+0.005); passed=**False**
4. `_last` test permitted: **False**
5. Primary: locked three-seed best-score frozen aggregate
6. Wording: Primary: a self-supervised contrastive Multi-GIN encoder (D+) evaluated with the encoder frozen and a supervised downstream MLP on pre-3h H+X+TF achieves test F1@0.5 of 0.639 ± 0.019 over three encoder seeds (best-score checkpoints). Secondary: a post-hoc equal-weight ensemble of the same three frozen encoders reaches test F1@0.5 of 0.6994 and AUPRC of 0.7052; this is not the robustness statistic. A fixed-horizon epoch-40 (_last) policy was validation-gated and failed the +0.005 val-AUPRC gate, so _last test was not evaluated.
7. No encoder received supervised updates: **true**
8. No new GNN training / automatic follow-up: **true**

