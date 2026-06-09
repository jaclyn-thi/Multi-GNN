# Contrast projection head ablation (Jun 4, 2026)

Short write-up of the two evening runs probing frozen embeddings with `linear_probe.py` (GIN, Small-HI hetero, model class weights, threshold = max F1 on val).

## Runs

| `unique_name` | Script | Pretrain config | Ckpt epoch (extract) |
|---------------|--------|-----------------|----------------------|
| `hi_contrastive_proj_20ep_bestckpt` | `ablation_contrastive_projection_20ep .sh` | Contrastive + `--contrast_projection_head` (128→128); **no** morph expert | 20 |
| `hi_morphology_global_proj_20ep_bestckpt` | `ablation_m1b_projection_20ep.sh` | M1b morph expert (`local+global`, w=1.0) + contrastive + same projection head | **15** (best ckpt) |

Shared training flags: asymmetric InfoNCE, memory bank 32768, 1024 negatives, `--checkpoint_policy best`, 20 epochs.

Embeddings: encoder `z` (128-d); projection head used in contrastive loss only (GraphCL-style), not at extract time.

## Probe results (val-tuned threshold)

| Run | Val AUROC | Test AUROC | Test F1 | Test prec | Test recall | Threshold |
|-----|-----------|------------|---------|-----------|-------------|-----------|
| contrastive + proj | **0.941** | **0.927** | **0.144** | **0.098** | 0.272 | 0.331 |
| M1b + proj | 0.934 | 0.924 | 0.096 | 0.058 | **0.289** | 0.267 |

Reference (no projection): `hi_morphology_global_20ep` (M1b) — test AUROC **0.920**, test F1 **0.108**. Pure contrastive baseline `hi_contrastive_20ep` — test AUROC 0.839.

At fixed threshold 0.5, test F1: contrastive+proj **0.137**; M1b+proj 0.092 (AUROC unchanged).

## Takeaways

1. **Projection head is a large win for pure contrastive SSL** — test AUROC 0.839 → **0.927** (+0.088 vs `hi_contrastive_20ep`). Val→test F1 is stable (0.146 → 0.144), so the gain looks real, not val overfitting alone.

2. **M1b + projection beats M1b on AUROC but not on F1** — 0.924 vs 0.920 ranking; **0.096 vs 0.108** at the val-optimal operating point. Tuned threshold favors recall (0.289) at the cost of precision (0.058), similar to other “stack more objectives” runs that hurt practical flagging.

3. **Best overall probe in this table: contrastive + projection** — highest test AUROC and F1 among projection runs and prior M1b default.

4. **Morph expert is not required for the projection benefit** — contrastive-only + proj slightly edges morph+proj on both AUROC and F1.

## Open questions / next steps

- Label-efficiency probe on `hi_contrastive_proj_20ep_bestckpt` (M1b still won at low label fractions pre-projection).
- Whether M1b+proj F1 can be recovered with a different threshold policy (e.g. target precision on val).
- Confirm training curves / morph val metrics for epoch-15 vs 20 on the M1b+proj run.

Artifacts: `embeddings/<unique_name>/probe_results.json`.
