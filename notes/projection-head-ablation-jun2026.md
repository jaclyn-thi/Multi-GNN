# Contrast projection head ablation (Jun 4, 2026)

> **Superseded for navigation:** benchmark tables and takeaways live in [`README.md`](../README.md) and [`morphology-metrics-plan.md`](morphology-metrics-plan.md) Project status. This file is a dated run log only.

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

3. ~~**Best overall probe: contrastive + projection**~~ — superseded by clustering+proj (see below).

4. **Morph expert is not required for the baseline projection benefit** — contrastive-only + proj edges M1b+proj on AUROC/F1 in the Jun 4 table; later **M1b + clustering + projection** beats both (0.929 AUROC).

## Label-efficiency (Jun 2026, nine encoders)

**Source:** `embeddings/label_efficiency_summary.json` · stratified train subsamples · val-tuned threshold · test AUROC.

| Encoder | 10% | 25% | 50% | 100% |
|---------|-----|-----|-----|------|
| **M1b + projection** | **0.918** | 0.922 | 0.919 | 0.922 |
| **Contrastive + projection** | 0.906 | **0.918** | **0.925** | **0.928** |
| M1b (no projection) | 0.896 | 0.910 | 0.915 | 0.919 |
| M1b + clustering (no projection) | 0.877 | 0.892 | 0.904 | 0.908 |
| Contrastive (no projection) | 0.818 | 0.849 | 0.857 | 0.863 |

**Takeaways:**

1. **Projection flips the scarcity story vs plain M1b.** Contrastive+proj beats M1b at all fractions (+0.008 to +0.010 test AUROC). The first label-efficiency batch (pre-projection) had M1b winning all fractions vs plain contrastive — that still holds for **non-projection** encoders only.
2. **M1b + projection is best at 10% labels** (0.918 vs 0.906 contrastive+proj). Stacking morph expert with projection helps under extreme scarcity; contrastive+proj leads at 25–100%.
3. **Default SSL recipe (label-efficiency, pending clustering+proj LE):** contrastive+proj @ 25–100%; M1b+proj @ 10%. Full-label leader is now clustering+proj (0.929).
4. **Clustering expert alone regresses** at all label fractions (0.877–0.908) and full-label (0.903). **With projection:** see below.

## M1b + clustering + projection (Jun 2026)

| `unique_name` | Script | Config | Ckpt ep |
|---------------|--------|--------|---------|
| `hi_morphology_global_clustering_proj_20ep_bestckpt` | `ablation_m1b_clustering_projection_20ep.sh` | M1b (11 local incl. clustering) + `--contrast_projection_head` | 20 |

| Run | Val AUROC | Test AUROC | Test F1 | Test prec | Test recall |
|-----|-----------|------------|---------|-----------|-------------|
| **clustering + proj** | **0.930** | **0.929** | **0.156** | 0.117 | 0.235 |
| contrastive + proj | 0.941 | 0.927 | 0.144 | 0.098 | 0.272 |
| M1b + proj | 0.934 | 0.924 | 0.096 | 0.058 | 0.289 |
| clustering expert only | 0.917 | 0.903 | 0.117 | 0.076 | 0.254 |

**Takeaways:**

1. **Best full-label SSL on Small-HI** — test AUROC **0.929**, F1 **0.156** (vs prior best contrastive+proj 0.927 / 0.144).
2. **Interaction effect** — clustering expert **hurts** without projection (0.903) but **helps** with projection (+0.026 vs clustering-only; +0.002 vs contrastive+proj AUROC).
3. Val→test AUROC stable (0.930 → 0.929). Label-efficiency on this encoder **pending** before adopting as default under scarcity.

## MAE vs MSE expert loss (Jun 2026)

| `unique_name` | Loss | Test AUROC | Test F1 |
|---------------|------|------------|---------|
| `hi_morphology_global_clustering_20ep` | MSE | 0.903 | 0.117 |
| `hi_morphology_global_mae_20ep_bestckpt` | MAE | **0.898** | **0.145** |

MAE did not improve AUROC vs MSE on the same 11-dim M1b targets; F1 higher at val-tuned threshold (recall-heavy). Default remains **MSE**.

## Open questions / next steps

- Label-efficiency on **`hi_morphology_global_clustering_proj_20ep_bestckpt`** — does full-label win hold @ 10% labels?
- Whether M1b+proj / clustering+proj F1 can be recovered with a different threshold policy.
- Confirm training curves / morph val metrics for clustering+proj vs contrastive+proj.

Artifacts: `embeddings/<unique_name>/probe_results.json`, `embeddings/<unique_name>/label_efficiency_results.json`.
