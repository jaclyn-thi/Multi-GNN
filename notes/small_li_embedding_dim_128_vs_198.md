# Small-LI exported embedding dim: 128 vs 198 (emb198 scout)

Four frozen representations, paired on a common `edge_id` join per split. The original 128-d checkpoint was **not** retrained; the emb198 checkpoint is a **new** training run (seed 1, 20 ep) that changed only `embedding_dim: 128 → 198`. Because the altered head also changes SSL optimization, differences are **not** attributable to dimension alone.

- dims: orig_post_128=128, orig_pre_3h=198, new_post_198=198, new_pre_3h=198
- probe: sklearn LogisticRegression (lbfgs), class_weight=model, C=1.0, threshold=max_f1_on_val, seed=1
- run: train `17350448` → extract `17350449` → probe `17350450`; emb198 best epoch 15.

## Conclusion (seed-1 scout; interpreted)

Widening the exported embedding to 198 (removing the `198→128` bottleneck) gave only a small,
**confounded** improvement to the *exported* embedding, and did **not** beat simply probing the
pre-embedding of the existing 128-d checkpoint.

- **Best representation overall stays the existing model's pre-embedding** (`orig_pre_3h_198`):
  AUPRC 0.045 embedding-only, 0.083 with +raw — far above every emb198 variant.
- **Widening the export helped a little** (`orig_post_128` 0.014 → `new_post_198` 0.022 AUPRC) and
  shrank the within-model pre-vs-post gap (+0.032 → +0.007), consistent with the bottleneck being
  what hurt the *export*.
- **But the effect is confounded with retraining.** At fixed dim (198), the emb198 encoder's own
  pre-embedding is *worse* than the original's (`new_pre_3h_198` 0.028 vs `orig_pre_3h_198` 0.045;
  ΔAUPRC −0.017, ΔAUROC −0.036). Since dimension is held constant, this is a worse SSL optimum
  (retraining/seed effect), not a dimension effect — so the export gain cannot be attributed to width.

**Takeaway:** for deployment, extract `pre_embedding_3h` from the existing 128-d checkpoint; a
single-seed emb198 retrain is not justified. Settling the width question cleanly would require
multi-seed emb198 training (not run).

## embedding_only

| representation | dim | AUROC | AUPRC | F1@val-thr | F1@0.5 | P@val-thr | R@val-thr | R@1000 | lift@100 |
|---|---|---|---|---|---|---|---|---|---|
| orig_post_128 | 128 | 0.8989 | 0.0137 | 0.0509 | 0.0454 | 0.0316 | 0.1309 | 0.0536 | 175.51 |
| orig_pre_3h_198 | 198 | 0.9222 | 0.0453 | 0.0960 | 0.0971 | 0.0692 | 0.1571 | 0.1185 | 351.02 |
| new_post_198 | 198 | 0.8938 | 0.0217 | 0.0655 | 0.0525 | 0.0534 | 0.0848 | 0.0623 | 307.14 |
| new_pre_3h_198 | 198 | 0.8858 | 0.0283 | 0.0744 | 0.0637 | 0.0598 | 0.0985 | 0.0848 | 307.14 |

**Contrasts (test):**

- `new_post_198` vs `orig_post_128` (exported-dim change CONFLATED with retraining a different head): ΔAUPRC=+0.0080, ΔAUROC=-0.0051, ΔF1=+0.0146 → AUPRC winner: **new_post_198**
- `new_pre_3h_198` vs `new_post_198` (pre vs post within the new emb198 model): ΔAUPRC=+0.0065, ΔAUROC=-0.0080, ΔF1=+0.0089 → AUPRC winner: **new_pre_3h_198**
- `orig_pre_3h_198` vs `orig_post_128` (pre vs post within the original model): ΔAUPRC=+0.0316, ΔAUROC=+0.0233, ΔF1=+0.0451 → AUPRC winner: **orig_pre_3h_198**
- `new_pre_3h_198` vs `orig_pre_3h_198` (retraining effect on the same-dim (198) 3h representation): ΔAUPRC=-0.0170, ΔAUROC=-0.0364, ΔF1=-0.0216 → AUPRC winner: **orig_pre_3h_198**

## embedding_plus_raw

| representation | dim | AUROC | AUPRC | F1@val-thr | F1@0.5 | P@val-thr | R@val-thr | R@1000 | lift@100 |
|---|---|---|---|---|---|---|---|---|---|
| orig_post_128 | 132 | 0.9081 | 0.0289 | 0.0413 | 0.0824 | 0.0223 | 0.2743 | 0.1010 | 336.39 |
| orig_pre_3h_198 | 202 | 0.9322 | 0.0829 | 0.0447 | 0.0176 | 0.0236 | 0.4152 | 0.1621 | 643.53 |
| new_post_198 | 202 | 0.8998 | 0.0260 | 0.0723 | 0.0705 | 0.0545 | 0.1072 | 0.0786 | 321.77 |
| new_pre_3h_198 | 202 | 0.8907 | 0.0328 | 0.0587 | 0.0824 | 0.0372 | 0.1397 | 0.0985 | 351.02 |

**Contrasts (test):**

- `new_post_198` vs `orig_post_128` (exported-dim change CONFLATED with retraining a different head): ΔAUPRC=-0.0030, ΔAUROC=-0.0083, ΔF1=+0.0310 → AUPRC winner: **orig_post_128**
- `new_pre_3h_198` vs `new_post_198` (pre vs post within the new emb198 model): ΔAUPRC=+0.0068, ΔAUROC=-0.0090, ΔF1=-0.0136 → AUPRC winner: **new_pre_3h_198**
- `orig_pre_3h_198` vs `orig_post_128` (pre vs post within the original model): ΔAUPRC=+0.0540, ΔAUROC=+0.0241, ΔF1=+0.0034 → AUPRC winner: **orig_pre_3h_198**
- `new_pre_3h_198` vs `orig_pre_3h_198` (retraining effect on the same-dim (198) 3h representation): ΔAUPRC=-0.0501, ΔAUROC=-0.0414, ΔF1=+0.0140 → AUPRC winner: **orig_pre_3h_198**

## Interpretation guide

- **Exported-dimension effect** is captured by `new_post_198` vs `orig_post_128`, but this conflates the wider export with retraining a different head (different SSL optimum).
- **Removing the bottleneck within the new model**: compare `new_pre_3h_198` vs `new_post_198` (both 198-d) — a learned 198→198 head vs the raw 198-d pre-embedding.
- **Retraining effect on the pre-embedding**: `new_pre_3h_198` vs `orig_pre_3h_198`.

## Caveats

- Single seed, single checkpoint per model; development/scout run.
- The emb198 model is a distinct training run; do not attribute any delta solely to the exported dimension.
