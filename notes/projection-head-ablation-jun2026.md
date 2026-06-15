# Contrast projection head ablation (Jun 4, 2026)

> **Superseded for navigation:** benchmark tables and takeaways live in [`README.md`](../README.md) and [`morphology-metrics-plan.md`](morphology-metrics-plan.md) Project status. This file is a dated run log only.

Short write-up of the two evening runs probing frozen embeddings with `linear_probe.py` (GIN, Small-HI hetero, model class weights, threshold = max F1 on val).

## Runs

| `unique_name` | Script | Pretrain config | Ckpt epoch (extract) |
|---------------|--------|-----------------|----------------------|
| `hi_contrastive_proj_20ep_bestckpt` | `slurm/ablation_contrastive_projection_20ep.sh` | Contrastive + `--contrast_projection_head` (128→128); **no** morph expert | 20 |
| `hi_morphology_global_proj_20ep_bestckpt` | `slurm/ablation_m1b_projection_20ep.sh` | M1b morph expert (`local+global`, w=1.0) + contrastive + same projection head | **15** (best ckpt) |

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

## Label-efficiency (Jun 2026, twelve encoders in summary)

**Source:** `embeddings/label_efficiency_summary.json` · stratified train subsamples · val-tuned threshold · test AUROC. Developmental comparisons — not a frozen evaluation protocol.

| Encoder | 10% | 25% | 50% | 100% |
|---------|-----|-----|-----|------|
| **M1b + projection** | **0.918** | 0.922 | 0.919 | 0.922 |
| **M1b + clustering + projection** | 0.916 | **0.926** | **0.930** | **0.929** |
| **Contrastive + projection** | 0.906 | 0.918 | 0.925 | 0.928 |
| M1b (no projection) | 0.896 | 0.910 | 0.915 | 0.919 |
| M1b + clustering (no projection) | 0.877 | 0.892 | 0.904 | 0.908 |
| MAE expert (clustering stack) | 0.872 | 0.885 | 0.894 | 0.898 |
| Contrastive (no projection) | 0.818 | 0.849 | 0.857 | 0.863 |

**Takeaways:**

1. **Projection flips the scarcity story vs plain M1b.** Contrastive+proj beats M1b at all fractions (+0.008 to +0.010 test AUROC). The first label-efficiency batch (pre-projection) had M1b winning all fractions vs plain contrastive — that still holds for **non-projection** encoders only.
2. **M1b + projection is best at 10% labels** (0.918 vs 0.916 clustering+proj vs 0.906 contrastive+proj). Stacking morph expert with projection helps under extreme scarcity.
3. **Default SSL recipe (label-efficiency, updated):** clustering+proj @ **25–100%** (0.926–0.930); M1b+proj @ **10%** (0.918). Full-label leader: clustering+proj (0.929).
4. **Clustering expert alone regresses** at all label fractions (0.877–0.908) and full-label (0.903). **With projection:** see below.

## M1b + clustering + projection (Jun 2026)

| `unique_name` | Script | Config | Ckpt ep |
|---------------|--------|--------|---------|
| `hi_morphology_global_clustering_proj_20ep_bestckpt` | `slurm/ablation_m1b_clustering_projection_20ep.sh` | M1b (11 local incl. clustering) + `--contrast_projection_head` | 20 |

| Run | Val AUROC | Test AUROC | Test F1 | Test prec | Test recall |
|-----|-----------|------------|---------|-----------|-------------|
| **clustering + proj** | **0.930** | **0.929** | **0.156** | 0.117 | 0.235 |
| contrastive + proj | 0.941 | 0.927 | 0.144 | 0.098 | 0.272 |
| M1b + proj | 0.934 | 0.924 | 0.096 | 0.058 | 0.289 |
| clustering expert only | 0.917 | 0.903 | 0.117 | 0.076 | 0.254 |

**Takeaways:**

1. **Best full-label SSL on Small-HI** — test AUROC **0.929**, F1 **0.156** (vs prior best contrastive+proj 0.927 / 0.144).
2. **Interaction effect** — clustering expert **hurts** without projection (0.903) but **helps** with projection (+0.026 vs clustering-only; +0.002 vs contrastive+proj AUROC).
3. Val→test AUROC stable (0.930 → 0.929). Label-efficiency: **0.916 @ 10%**, **0.926–0.930 @ 25–100%** — leads at 25%+ (M1b+proj still best @ 10%).

## MAE vs MSE expert loss (Jun 2026)

| `unique_name` | Loss | Test AUROC | Test F1 |
|---------------|------|------------|---------|
| `hi_morphology_global_clustering_20ep` | MSE | 0.903 | 0.117 |
| `hi_morphology_global_mae_20ep_bestckpt` | MAE | **0.898** | **0.145** |

MAE did not improve AUROC vs MSE on the same 11-dim M1b targets; F1 higher at val-tuned threshold (recall-heavy). Default remains **MSE**.

## Contrastive relax grid — symmetric InfoNCE (Jun 11, 2026)

First completed run from the “relax memory hacks” grid (`slurm/submit_contrastive_relax_grid.sh`). Probes whether dropping **asymmetric** InfoNCE (a VRAM hack) helps downstream metrics when projection is already on.

| `unique_name` | Script | Pretrain config | Ckpt ep |
|---------------|--------|-----------------|---------|
| `hi_contrastive_proj_sym_20ep_bestckpt` | `slurm/ablation_contrastive_proj_sym_20ep.sh` | Contrastive + projection; **symmetric** InfoNCE (no `--contrastive_asymmetric`); 1024 negs + queue 32768 | **20** (best = final) |

**VRAM recipe (required for symmetric on ~44 GiB GPU):** `--batch_size 16384 --contrastive_accum_steps 2`. Baseline asym+proj used `batch_size 32768` — **not a single-variable ablation** (symmetric + smaller subgraph).

Training: contrastive loss fell every epoch (5.34 → 4.998); best checkpoint = epoch 20. Job `slurm-15879714`.

### Probe results vs asymmetric contrastive + proj baseline

| Run | Val AUROC | Test AUROC | Test F1 | Test prec | Test recall | Threshold |
|-----|-----------|------------|---------|-----------|-------------|-----------|
| **sym + proj** (this) | 0.939 | **0.929** | **0.222** | **0.185** | 0.277 | 0.364 |
| asym + proj (`hi_contrastive_proj_20ep_bestckpt`) | **0.941** | 0.927 | 0.144 | 0.098 | **0.272** | 0.331 |
| clustering + proj (reference) | 0.930 | **0.929** | 0.156 | 0.117 | 0.235 | 0.273 |

At fixed threshold 0.5, test F1: sym+proj **0.211**; asym+proj **0.137**.

### Takeaways

1. **Symmetric InfoNCE did not hurt ranking** — test AUROC **0.929** (+0.002 vs asym+proj; ties clustering+proj). Val AUROC −0.002; val→test stable.
2. **Large F1 gain at val-tuned threshold** — test F1 **0.222** vs asym+proj **0.144** (+0.078), driven mainly by **precision** (0.185 vs 0.098) at similar recall (~0.28 vs ~0.27).
3. **Best contrastive+proj F1 so far** — exceeds clustering+proj F1 (0.156) on this probe protocol; AUROC still ~tied with clustering+proj.
4. **Confound resolved (Jun 11 evening):** see **asym @ `bs=16384`** and **8192 negs** sections below — smaller batch drives most F1 gain; symmetric adds +0.009 AUROC / +0.016 F1 on top of asym @ same batch recipe.

## Relax grid — confound control: asym @ `bs=16384` (Jun 11, 2026)

| `unique_name` | Script | Pretrain config | Ckpt ep |
|---------------|--------|-----------------|---------|
| `hi_contrastive_proj_asym_16384_20ep_bestckpt` | `slurm/ablation_contrastive_proj_asym_16384_20ep.sh` | Asym InfoNCE + projection; 1024 negs + queue 32768; `bs=16384 accum=2` (same VRAM recipe as sym, but asymmetric) | **20** |

Job `slurm-15884206`. Training loss 5.34 → 5.002; best ckpt = epoch 20.

### Probe results (1024 negs; isolate sym vs batch size)

| Run | Val AUROC | Test AUROC | Test F1 | Test prec | Test recall | Threshold |
|-----|-----------|------------|---------|-----------|-------------|-----------|
| sym + proj (`bs=16384`) | 0.939 | **0.929** | **0.222** | **0.185** | 0.277 | 0.364 |
| **asym + proj (`bs=16384`)** | 0.937 | 0.920 | 0.206 | 0.161 | **0.286** | 0.329 |
| asym + proj (`bs=32768`, baseline) | **0.941** | 0.927 | 0.144 | 0.098 | 0.272 | 0.331 |

At fixed threshold 0.5, test F1: asym @ 16384 **0.220**; sym **0.211**; baseline **0.137**.

### Takeaways

1. **Smaller subgraph (`bs=16384`) alone boosts F1** — asym at 16384: test F1 **0.206** (+0.062 vs baseline asym) even though test AUROC **drops** 0.007 (0.920 vs 0.927).
2. **Symmetric adds on top of batch recipe** — sym vs asym @ 16384: **+0.009** test AUROC, **+0.016** test F1.
3. **Decomposition vs baseline asym:** ~**80%** of F1 lift (0.206 vs 0.144) from batch size; ~**20%** (0.222 vs 0.206) from symmetric mode.
4. asym @ 16384 is **recall-heavy** (0.286) at the cost of ranking vs sym.

## Relax grid — asymmetric + 8192 negs (Jun 11, 2026)

| `unique_name` | Script | Pretrain config | Ckpt ep |
|---------------|--------|-----------------|---------|
| `hi_contrastive_proj_8192neg_20ep_bestckpt` | `slurm/ablation_contrastive_proj_8192neg_20ep.sh` | Asym InfoNCE + projection; **8192** negs + queue 32768; `bs=8192 accum=4` | **20** |

**VRAM recipe:** halve loader `batch_size` again vs 16384; loss uses k-axis chunking in `contrastive_loss.py`. Do **not** use `--gradient_checkpointing` on hetero (`to_hetero` FX trace fails). Job `slurm-15884204`. Training loss 7.47 → 7.079 (higher than 1024-neg runs — expected).

### Probe results

| Run | Val AUROC | Test AUROC | Test F1 | Test prec | Test recall | Threshold |
|-----|-----------|------------|---------|-----------|-------------|-----------|
| **asym + 8192 negs** | **0.953** | **0.930** | 0.191 | 0.149 | 0.267 | 0.393 |
| sym + proj (`bs=16384`, 1024 negs) | 0.939 | 0.929 | **0.222** | **0.185** | **0.277** | 0.364 |
| asym @ 16384 (1024 negs) | 0.937 | 0.920 | 0.206 | 0.161 | 0.286 | 0.329 |
| asym baseline (`bs=32768`) | 0.941 | 0.927 | 0.144 | 0.098 | 0.272 | 0.331 |
| clustering + proj | 0.930 | 0.929 | 0.156 | 0.117 | 0.235 | 0.273 |

At fixed threshold 0.5, test F1: 8192 **0.196**; sym **0.211**.

### Takeaways

1. **Best contrastive+proj test AUROC: 0.930** (+0.003 vs baseline asym; +0.001 vs sym; ≈ clustering+proj).
2. **8192 negs help ranking vs asym @ 16384** (+0.010 AUROC) but **F1 below sym** (0.191 vs 0.222) at val-tuned threshold.
3. Val AUROC very high (0.953); test holds at 0.930 — monitor val→test gap if stacking more relax-grid knobs.
4. Changes **two** axes vs baseline (negs + batch); vs asym @ 16384 isolates neg-count at similar small-subgraph training.

### Relax grid summary (Jun 11, 2026)

| Script | Status | Test AUROC | Test F1 |
|--------|--------|------------|---------|
| `ablation_contrastive_proj_sym_20ep.sh` | ✅ | 0.929 | **0.222** |
| `ablation_contrastive_proj_asym_16384_20ep.sh` | ✅ | 0.920 | 0.206 |
| `ablation_contrastive_proj_8192neg_20ep.sh` | ✅ | **0.930** | 0.191 |
| `ablation_contrastive_proj_sym_8192neg_20ep.sh` | ⏸ optional | — | — |
| `ablation_contrastive_proj_sym_8192neg_noqueue_20ep.sh` | ⏸ optional | — | — |

**Practical recipes (contrastive+proj only):** best **AUROC** → asym + 8192 negs; best **F1** → sym + 1024 @ `bs=16384`.

## M1b + clustering + triangles + projection (Jun 2026)

| `unique_name` | Script | Config | Ckpt ep |
|---------------|--------|--------|---------|
| `hi_morphology_global_triangles_proj_20ep_bestckpt` | `slurm/ablation_m1b_triangles_projection_20ep.sh` | M1b + **14** local (clustering **and** triangles) + projection; asym @ bs=32768 | 20 |

| Run | Val AUROC | Test AUROC | Test F1 |
|-----|-----------|------------|---------|
| triangles + proj (14 local) | 0.933 | **0.912** | 0.145 |
| clustering + proj (11 local) | 0.930 | **0.929** | 0.156 |
| sym + proj (contrastive) | 0.939 | 0.929 | **0.222** |

**Takeaway:** Stacking triangle counts on top of clustering **regresses** test AUROC (−0.017 vs clustering+proj). Val→test gap −0.021. Triangles overlap triadic signal from clustering; extra expert dims add noise. **Triangles-only** ablation: `slurm/ablation_m1b_triangles_only_projection_20ep.sh` (`--morph_local_subset triangles`).

## M1b + triangles-only + projection (Jun 12, 2026)

| `unique_name` | Script | Config | Ckpt ep |
|---------------|--------|--------|---------|
| `hi_morphology_global_triangles_only_proj_20ep_bestckpt` | `slurm/ablation_m1b_triangles_only_projection_20ep.sh` | M1b + **11** local (degree/ego + **triangles**, no clustering) + projection; asym @ bs=32768 | 20 |

| Run | Val AUROC | Test AUROC | Test F1 | Test prec | Test recall | Threshold |
|-----|-----------|------------|---------|-----------|-------------|-----------|
| **triangles-only + proj** | 0.926 | 0.910 | **0.067** | **0.036** | **0.398** | 0.201 |
| clustering + proj (11 local) | 0.930 | **0.929** | **0.156** | **0.117** | 0.235 | 0.273 |
| triangles + clustering + proj (14 local) | 0.933 | 0.912 | 0.145 | 0.095 | 0.305 | 0.204 |
| M1b (no projection) | 0.914 | 0.920 | 0.108 | 0.069 | 0.248 | 0.156 |

At fixed threshold 0.5, test F1: triangles-only **0.075**; clustering+proj **0.132**; M1b **0.080**.

**Takeaways (probe):**

1. **Clustering beats triangles** on both ranking (AUROC −0.019) and usable flagging (F1 −0.089 at val-tuned threshold).
2. Triangles-only AUROC is similar to stacking both (0.910 vs 0.912) — redundant triadic signal when combined with clustering.
3. Low F1 is **not** from missing ranking signal overall (AUROC ~0.91); it's threshold/precision at the operating point.

### Pattern typology (Jun 12, 2026)

**Source:** `scripts/evaluate_pattern_typology.py` · val-tuned threshold · `--class_weight model` · test split · 1,240 laundering edges with known pattern metadata.

| Run | Thr | Test F1 | Known-meta recall | FAN-OUT recall | GATHER-SCATTER recall | GATHER-SCATTER AUROC |
|-----|-----|---------|-------------------|----------------|----------------------|----------------------|
| M1b | 0.156 | 0.108 | 28% | **5%** | 32% | 0.941 |
| clustering + proj | 0.273 | **0.156** | 29% | 19% | 42% | **0.971** |
| **triangles-only + proj** | 0.201 | 0.067 | **48%** | **68%** | **56%** | 0.948 |

**Key finding — inverted recall vs F1:** At **different** val-optimal thresholds, triangles-only has the **highest per-pattern recall on every type** (known metadata: **596/1240 = 48%** vs clustering **364/1240 = 29%**) yet the **worst overall F1** because it flags ~**17.6K** test edges (~641 TP, 3.6% precision) vs clustering ~**3.2K** (~379 TP, 11.7% precision). Triangles-only is a **recall-heavy, miscalibrated scorer**, not pattern-blind.

**FAN-OUT:** M1b near-zero recall (5%) despite moderate one-vs-rest AUROC on some fan-out variants — scores rank but threshold misses. Projection + triadic experts (clustering or triangles) lift FAN-OUT AUROC to ~0.96; triangles-only catches **68%** of fan-out edges at its low threshold (e.g. Max 16-degree Fan-Out: 100% vs clustering 0% vs M1b 0%) at the cost of global precision.

**Fair comparison (fixed @ 0.5, done — nine-way):** Per-pattern recall at val-tuned thresholds is **not apples-to-apples** across runs. Fixed @ 0.5 results (`results/diagnostics/<run>_thr0.5/`):

| Run | F1 | Prec | Known-meta recall | FAN-OUT recall |
|-----|-----|------|-------------------|----------------|
| **asym @ 16384 + proj** | **0.220** | **0.246** | 26% | 29% |
| sym + proj | 0.211 | 0.225 | 25% | 34% |
| 8192neg + proj | 0.196 | 0.181 | **27%** | **45%** |
| asym + proj (baseline) | 0.137 | 0.125 | 19% | 32% |
| sym + morph + proj | 0.133 | 0.104 | 23% | 35% |
| clustering + proj | 0.132 | 0.182 | 13% | 8% |
| M1b + proj | 0.092 | 0.073 | 16% | 7% |
| M1b | 0.080 | 0.177 | 6% | 1% |
| triangles-only + proj | 0.075 | 0.049 | 21% | 29% |

**asym@16384 best F1 @ 0.5** (0.220) — beats sym (0.211). 8192neg best FAN-OUT (45%). asym@16384 **improves** @ 0.5 vs val-tuned (0.206 → 0.220); baseline asym and M1b+proj known-meta recall halve @ 0.5.

Artifacts: all ten encoders under `results/diagnostics/<unique_name>/` (+ `_thr0.5/`).

## Pattern typology cross-run (Jun 2026)

**Source:** `scripts/evaluate_pattern_typology.py` · **nineteen runs** (ten val-tuned + nine fixed @ 0.5). **Benchmark complete.** See [`downstream-eval-plan.md`](downstream-eval-plan.md) § pattern metadata for full tables.

### SSL leaders — typology explains F1 vs AUROC split

| Run | Test F1 | FAN-OUT recall | FAN-OUT AUROC | Worst type (recall) |
|-----|---------|----------------|---------------|---------------------|
| sym + proj | **0.222** | 49% | 0.922 | BIPARTITE (21%) |
| 8192neg + proj | 0.191 | **53%** | **0.970** | BIPARTITE (18%) |
| clustering + proj | 0.156 | 19% | 0.962 | STACK (15%) |
| M1b (no proj) | 0.108 | **5%** | 0.884 | **FAN-OUT** |

@ fixed 0.5: **asym@16384 F1 0.220 (best)**; sym 0.211; 8192neg FAN-OUT **45%** (best).

### Tier 2 @ val-tuned (Jun 2026)

| Run | Test F1 | Prec | FAN-OUT recall | Notes |
|-----|---------|------|----------------|-------|
| **asym @ 16384 + proj** | **0.206** | **0.161** | 44% | tier-2 F1 surprise; ~2.9K flags |
| asym + proj (baseline) | 0.144 | 0.098 | **47%** | ~4.5K flags |
| sym + morph + proj | 0.134 | 0.093 | 39% | morph hurts sym (−0.088 F1) |
| M1b + proj | 0.096 | 0.058 | 32% | over-flags (~8K); no clustering expert |

**Takeaways:** asym@16384 competitive with sym on typology (F1 0.206 vs 0.222) with better precision than baseline asym. sym+morph typology confirms probe regression. M1b+proj recall-heavy like triangles-only.

### Tier 2 @ fixed 0.5 (Jun 2026)

| Run | F1 @ 0.5 | Prec | FAN-OUT recall | vs val-tuned F1 |
|-----|----------|------|----------------|-----------------|
| **asym @ 16384 + proj** | **0.220** | **0.246** | 29% | **+0.014** (improves) |
| asym + proj (baseline) | 0.137 | 0.125 | 32% | −0.007 |
| sym + morph + proj | 0.133 | 0.104 | 35% | −0.001 |
| M1b + proj | 0.092 | 0.073 | 7% | −0.004 |

**Takeaway:** asym@16384 is the **fixed-threshold leader** across all nine encoders — not just a val-tuned surprise. Aligns with label-efficiency F1 (0.223 @ 100% labels).

### Triangle ablation — typology completes the picture

| Config | Local | FAN-OUT rec | GATHER-SCATTER AUROC | ~flags @ val thr |
|--------|-------|-------------|----------------------|------------------|
| clustering + proj | 11 | 19% | **0.971** | ~3.2K |
| tri + clust + proj | 14 | 26% | 0.918 | ~5.2K |
| tri-only + proj | 11 | **68%** | 0.948 | ~17.6K |

Stacking triangles on clustering inherits over-flagging without clustering's gather/scatter ranking advantage.

**Batch scripts:** `slurm/submit_pattern_typology_remaining.sh` (tier 1/2/all) · `slurm/submit_pattern_typology_tier2.sh` · `slurm/submit_pattern_typology_fixed_thr0.5.sh`.

## M1b + symmetric contrastive + projection (Jun 2026)

| `unique_name` | Script | Config | Ckpt ep |
|---------------|--------|--------|---------|
| `hi_morphology_global_sym_proj_20ep_bestckpt` | `slurm/ablation_m1b_sym_projection_20ep.sh` | M1b + 14 local + **sym** InfoNCE + projection; `bs=16384 accum=2` | 20 |

| Run | Val AUROC | Test AUROC | Test F1 | Test prec | Test recall |
|-----|-----------|------------|---------|-----------|-------------|
| **sym + morph + proj** | 0.938 | **0.930** | 0.134 | 0.093 | 0.239 |
| sym + proj (no morph) | 0.939 | 0.929 | **0.222** | **0.185** | **0.277** |
| clustering + proj | 0.930 | 0.929 | 0.156 | 0.117 | 0.235 |

**Takeaways:**

1. **Morph expert on sym does not combine strengths** — +0.001 test AUROC vs sym-only, **−0.088** test F1 (0.134 vs 0.222).
2. **Sym contrastive+proj remains the val-tuned F1 leader** among SSL runs; **asym@16384 leads @ fixed 0.5** (typology F1 0.220 vs sym 0.211). Morphology stacking hurts threshold-tuned flagging (low precision 0.093).
3. AUROC ~ties 8192neg (**0.930**); choose recipe by metric: **F1 @ val-tuned → sym**; **F1 @ fixed 0.5 → asym@16384**; **AUROC / FAN-OUT → 8192neg**.

## Relax grid — label-efficiency (Jun 12, 2026)

**Source:** `embeddings/label_efficiency_summary.json` · `slurm/run_contrastive_relax_label_efficiency.sh` · fifteen encoders total.

| Encoder | 10% | 25% | 50% | 100% | Test F1 @ 100% |
|---------|-----|-----|-----|------|----------------|
| **sym + proj** | **0.924** | **0.926** | 0.929 | 0.929 | 0.216 |
| **8192 negs + proj** | 0.917 | 0.922 | **0.931** | **0.931** | 0.199 |
| asym @ 16384 + proj | 0.915 | 0.922 | 0.922 | 0.922 | **0.223** |
| Baseline asym + proj | 0.906 | 0.918 | 0.925 | 0.928 | 0.146 |
| clustering + proj | 0.916 | 0.926 | 0.930 | 0.929 | 0.157 |

**Takeaways:** Relax-grid encoders lead **AUROC** under label scarcity vs baseline asym+proj and match or beat morphology SSL. **sym** best @ 10% labels; **8192neg** best @ 50–100%. **F1:** sym dominates (~0.20–0.22); asym@16384 highest F1 @ 100% labels (0.223). Prior LE leaders (clustering+proj @ 25–100%, M1b+proj @ 10%) are now second-tier on AUROC.

## Open questions / next steps

- ~~Label-efficiency on **`hi_morphology_global_clustering_proj_20ep_bestckpt`**~~ — done: 0.916 @ 10% (M1b+proj wins), 0.926–0.930 @ 25–100%.
- Whether M1b+proj / clustering+proj F1 can be recovered with a different threshold policy.
- Confirm training curves / morph val metrics for clustering+proj vs contrastive+proj.
- ~~Relax grid: asym @ `bs=16384` confound control~~ · ~~asym @ 8192 negs~~ — done.
- ~~**M1b + sym + projection**~~ — done: 0.930 AUROC / 0.134 F1; morph hurts sym F1 (−0.088).
- **Optional:** sym @ 8192 negs · no-queue @ 8192 (diminishing returns; see projection ablation note § grid summary).
- ~~Label-efficiency on relax-grid encoders~~ — done (Jun 12): sym @ 10%, 8192neg @ 50–100% AUROC; sym F1 at all fractions.
- ~~Triangles-only probe + pattern typology~~ — done (Jun 2026): **19 runs complete.** asym@16384 best F1 @ fixed 0.5 (0.220); sym best @ val-tuned (0.222); 8192neg best FAN-OUT @ 0.5 (45%). See § Pattern typology cross-run.
- ~~Fixed @ 0.5 typology~~ — done: nine-way table complete.
- ~~Tier-2 typology~~ — done (val-tuned + fixed @ 0.5).

Artifacts: `embeddings/<unique_name>/probe_results.json`, `embeddings/<unique_name>/label_efficiency_results.json`.
