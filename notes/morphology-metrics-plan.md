# Morphology Metrics — Planning & Implementation

## Purpose

Companion to [`contrastive-learning-plan.md`](contrastive-learning-plan.md).

This document owns **morphology metric selection**, **computation strategy** (including what can be precomputed vs must be batch-local), **how metrics attach to transaction (edge) embeddings**, and a **phased implementation plan** for both:

1. **Morphology-aware contrastive learning** (stronger SSL signal than edge-identity InfoNCE alone).
2. **Expert prediction heads** (Papagei-style auxiliary targets on frozen or joint representations).

The contrastive plan continues to own objective routing, homo/hetero training, extraction, and AML linear probe evaluation.

**Metric definitions (concise):** [`morphology-reference.md`](morphology-reference.md) — Tier 0/1/2 columns, edge-native fields, M2 groups, expert target dims, and the `log1p` rule. This document adds compute strategy, phased implementation, and benchmark results.

---

## Project status (Jun 2026)

> **Benchmark numbers in this doc and in [`results.md`](results.md)** are **development sanity checks** (quick ablations while configs and code still change). They guide internal decisions but are not a formal, frozen evaluation suite. Recorded experiments for publication / PI milestones will use fixed recipes once the stack stabilizes.
>
> **This section is a Jun 2026 morphology snapshot.** For the latest cross-protocol results and thesis-safe claims, prefer [`current_protocol_recent_runs_summary.md`](current_protocol_recent_runs_summary.md) and [`results.md` § Recommended configs](results.md#recommended-configs-jun-2026).

**Implemented:** M0 · M1 · M1b · **M1c Tier-1 local clustering** (Jun 2026) · M2 (+ `local_clustering` contrast group) · morphology val throttling · **M4 best checkpoint** · **label-efficiency probe** (incremental summary merge) · **M3 Phase 0–1** (BC precompute + expert wiring) · **contrastive projection head** (Jun 2026) · **`--morph_expert_loss {mse,mae}`** (Jun 2026).

**Small-HI probe results (GIN, hetero, val-tuned F1, `linear_probe.py`):**

| Run | `unique_name` | Ckpt ep | Val AUROC | Test AUROC | Test F1 | Notes |
|-----|---------------|---------|-----------|------------|---------|-------|
| Contrastive baseline | `hi_contrastive_20ep` | 20 | 0.871 | 0.839 | 0.076 | identity InfoNCE |
| M1 | `hi_morphology_20ep` | 20 | 0.921 | 0.910 | 0.079 | local expert |
| M1b | `hi_morphology_global_20ep` | 20 | 0.914 | 0.920 | 0.108 | **best morph-only** config |
| M2 expert + contrast | `hi_morph_global_contrast_10ep_bestckpt` | **9** | 0.912 | 0.906 | 0.058 | M4 best |
| M2 expert + contrast | `hi_morph_global_contrast_20ep_bestckpt` | 20 | 0.912 | 0.891 | 0.107 | M4; AUROC below M1b |
| M2 expert + contrast | `hi_morph_global_contrast_10ep_w05_bestckpt` | 10 | 0.898 | 0.876 | 0.027 | `morph_expert_weight=0.5` ablation |
| M2 contrast only | `hi_morph_contrast_only_10ep` | 10 | — | 0.680 | 0.012 | expert required |
| M3 M1b + BC (4 lift cols) | `hi_morphology_global_bc_20ep_bestckpt` | **14** | 0.913 | 0.896 | 0.033 | **below M1b** |
| M3 M1b + BC (last epoch) | `hi_morphology_global_bc_20ep_last` | 20 | 0.884 | 0.861 | 0.029 | worse than M4 best |
| M3 BC-only (`local+tier2`) | `hi_morphology_bc_only_20ep_bestckpt` | 20 | 0.897 | **0.904** | 0.093 | BC **substitutes** for degrees OK |
| M3 M1b + bc_max | `hi_morphology_global_bc_max_20ep_bestckpt` | 20 | 0.902 | 0.889 | 0.086 | stack still hurts; 1 BC col ≠ fix |
| M5a grouped BC (`w_tier2=0`) | `hi_morph_grouped_bc_w0_20ep_bestckpt` | 19 | 0.914 | 0.885 | 0.043 | 4 heads ≠ M1b sanity; below M1b |
| M5a grouped BC (`w_tier2=1`) | `hi_morph_grouped_bc_w1.0_20ep_bestckpt` | 20 | 0.919 | 0.887 | 0.028 | worse than shared M3 stack (0.896) |
| Contrastive + proj head | `hi_contrastive_proj_20ep_bestckpt` | 20 | **0.941** | **0.927** | **0.144** | asym InfoNCE; `bs=32768` |
| Contrastive + proj, **8192 negs** | `hi_contrastive_proj_8192neg_20ep_bestckpt` | 20 | **0.953** | **0.930** | 0.191 | asym; `bs=8192 accum=4`; **best contrastive+proj AUROC** |
| Contrastive + proj, **symmetric** | `hi_contrastive_proj_sym_20ep_bestckpt` | 20 | 0.939 | **0.929** | **0.222** | sym; `bs=16384 accum=2`; **best contrastive+proj F1** |
| Contrastive + proj, asym @ 16384 | `hi_contrastive_proj_asym_16384_20ep_bestckpt` | 20 | 0.937 | 0.920 | 0.206 | confound control; asym; `bs=16384 accum=2` |
| M1b + proj head | `hi_morphology_global_proj_20ep_bestckpt` | **15** | 0.934 | 0.924 | 0.096 | AUROC ↑ vs M1b, F1 ↓ at val-tuned threshold |
| M1b + clustering expert | `hi_morphology_global_clustering_20ep` | 20 | 0.917 | 0.903 | 0.117 | 11 local dims; AUROC ↓ vs M1b, F1 ↑ |
| M1b + clustering + proj | `hi_morphology_global_clustering_proj_20ep_bestckpt` | 20 | **0.930** | **0.929** | **0.156** | **best full-label SSL**; clustering hurts alone, wins with projection |
| M1b + clustering + triangles + proj | `hi_morphology_global_triangles_proj_20ep_bestckpt` | 20 | 0.933 | **0.912** | 0.145 | 14 local dims; **regresses** vs clustering+proj (−0.017 AUROC); val→test gap −0.021 |
| M1b + sym + proj | `hi_morphology_global_sym_proj_20ep_bestckpt` | 20 | 0.938 | **0.930** | 0.134 | M1b + 14 local + **sym** @ `bs=16384`; +0.001 AUROC vs sym-only; **−0.088 F1** vs sym (0.222) |
| M1b + triangles-only + proj | `hi_morphology_global_triangles_only_proj_20ep_bestckpt` | 20 | 0.926 | 0.910 | 0.067 | 11 local (triangles); F1 collapse is **precision/threshold**, not pattern blind spots — see § Pattern typology |
| M1b + MAE expert | `hi_morphology_global_mae_20ep_bestckpt` | 20 | 0.910 | 0.898 | 0.145 | MAE ↓ vs MSE clustering run (0.903); F1 ↑ |

**Takeaways (Jun 2026):**

- **Best full-label SSL:** **M1b + clustering + projection** — test AUROC **0.929** / F1 **0.156** (`hi_morphology_global_clustering_proj_20ep_bestckpt`). Clustering expert alone regressed (0.903); with projection it beats contrastive+proj (0.927). Interaction effect, not monotonic morphology stacking.
- **Contrast projection head (Jun 4):** contrastive-only + proj **0.927** / **0.144** F1; M1b+proj **0.924** / **0.096** F1. Clustering+proj later reached **0.929** AUROC (table above).
- **Relax grid (Jun 11, contrastive+proj):** sym **0.929** AUROC / **0.222** F1; asym @ 16384 confound **0.920** / **0.206** (batch size drives ~80% of F1 lift vs baseline); asym + 8192 negs **0.930** AUROC / **0.191** F1 (best contrastive AUROC). See [`projection-head-ablation-jun2026.md`](archive/projection-head-ablation-jun2026.md).
- **Default SSL recipe (full-label):** **F1 @ val-tuned** → sym contrastive+proj (**0.222**); **F1 @ fixed 0.5 (typology)** → asym@16384 (**0.220**). **AUROC** → 8192neg or sym+morph (**0.930**). Morphology: clustering+proj (**0.929** / **0.156**). **Sym + morph expert does not combine wins** — `hi_morphology_global_sym_proj_20ep_bestckpt` ties AUROC (+0.001 vs sym) but F1 **0.134** (−0.088). **Label-efficiency:** sym @ 10% (0.924), 8192neg @ 50–100% (0.931). See § Relax-grid label-efficiency.
- **M1b expert-only (8 local dims)** remains best **morph-only** full-label config (0.920 AUROC); plain M1b still beats plain contrastive under label scarcity but trails all projection encoders.
- **M4** helps expert+M2 vs last-epoch saves but does **not** beat M1b on AUROC.
- **Lowering `morph_expert_weight` to 0.5** hurt vs w=1.0 (0.876 vs 0.906 test AUROC).
- **M3 BC stacked on M1b** hurt vs M1b (0.896 best / 0.861 last vs 0.920); **BC-only** (0.904) beats stacked — interference in shared MLP + unified MSE, not “BC is useless.”
- **bc_max** (1 BC col on M1b) still **0.889** — fewer lift cols does not rescue the stack.
- **M5a grouped heads** do **not** fix M3: `w_tier2=1` → 0.887 vs shared stack 0.896; **M5b per-metric deprioritized**.
- **Label-efficiency (fifteen encoders in summary):** relax-grid **sym** best @ **10%** AUROC (0.924); **8192neg** best @ **50–100%** (0.931); **sym** best **F1** at all fractions (~0.20–0.22). Prior leaders: clustering+proj @ 25–100% (0.926–0.930), M1b+proj @ 10% (0.918). Baseline asym+proj (bs=32768) trails relax recipes by +0.003 to +0.018 AUROC.
- **Tier-1 local clustering (M1c):** expert-only regressed (0.903 AUROC; LE 0.877–0.908). **With projection:** **0.929** AUROC — morphology value is conditional on training recipe (asym @ bs=32768; sym+morph does not recover sym F1).
- **MAE vs MSE expert loss:** `--morph_expert_loss mae` → **0.898** test AUROC vs MSE baseline **0.903** (same 11-dim targets); no AUROC win; default MSE retained.
- **Pattern typology (Jun 2026, nineteen runs — complete):** **Projection unlocks FAN-OUT** — M1b worst (5%) vs SSL best (49–53% @ val-tuned; 8192neg **45%** @ 0.5). **sym+proj** best @ val-tuned F1 (**0.222**). **asym@16384** best @ fixed 0.5 F1 (**0.220**, prec **24.6%**) — beats sym (0.211); improves vs val-tuned. **8192neg** best AUROC (0.930). **sym+morph** / **M1b+proj** over-flag or hurt F1. **Clustering+proj** best gather/scatter AUROC (0.971). Full tables: [`downstream-eval-plan.md`](downstream-eval-plan.md) § pattern metadata · [`projection-head-ablation-jun2026.md`](archive/projection-head-ablation-jun2026.md) § Pattern typology cross-run.

**Next:** optional sym+8192 (OOM) · external dataset spike (SAML-D or AMLSim).

---

## Label-efficiency evaluation (implemented)

**Motivation:** Full-label linear probe favors **relax-grid contrastive+proj** (8192 negs **0.930** AUROC) or **M1b + clustering + projection** (**0.929**). In production AML, labeled illicit transactions are scarce — the relevant question (GCPAL, Papagei, RWTH) is: **with only 10–50% of train labels, which frozen encoder separates better?** Current summary (**fifteen** encoders): **sym+proj @ 10%** AUROC; **8192neg @ 50–100%**; morphology clustering+proj still **0.916–0.930**.

**Protocol (no encoder retraining):**

1. Freeze embeddings from a completed pretrain (`embedding_extraction.py`).
2. For each train fraction in `{0.1, 0.25, 0.5, 1.0}`, **stratified subsample** train rows and fit sklearn logistic regression.
3. Tune classification threshold on the **full val** split (same as `linear_probe.py`).
4. Report **val/test AUROC** (primary) and F1 at the val-tuned threshold.

**Implementation:** [`scripts/label_efficiency_probe.py`](../scripts/label_efficiency_probe.py). On cluster: request **~128G RAM**, CPU-only, `--probe_n_jobs 1` (full `train.npz` is ~3.2M×128 floats; multiprocessing sklearn can OOM).

**Outputs:**

| File | Content |
|------|---------|
| `embeddings/{unique_name}/label_efficiency_results.json` | Per-fraction metrics for one encoder |
| `embeddings/label_efficiency_summary.json` | Combined results for all `--unique_names` |

**Compare:** M1b vs expert+M2 (bestckpt and legacy) vs contrastive baseline — if M2 wins at low fractions, it may still be useful under label scarcity even when full-label AUROC loses to M1b.

### Label-efficiency results (Jun 2026)

**Source:** `embeddings/label_efficiency_summary.json` · **fifteen encoders** × four train fractions · `--class_weight model` · `--probe_max_iter 5000` · threshold tuned on **full val** (same protocol as full-label probe, but stratified train subsampling per fraction). Incremental runs merge into the summary (`scripts/label_efficiency_probe.py`). Developmental comparisons only — see project-status note above.

**Test AUROC by train label fraction (primary encoders):**

| Encoder | `unique_name` | 10% | 25% | 50% | 100% |
|---------|---------------|-----|-----|-----|------|
| **M1b + clustering + projection** | `hi_morphology_global_clustering_proj_20ep_bestckpt` | 0.916 | **0.926** | **0.930** | **0.929** |
| **M1b + projection** | `hi_morphology_global_proj_20ep_bestckpt` | **0.918** | 0.922 | 0.919 | 0.922 |
| **Contrastive + projection** | `hi_contrastive_proj_20ep_bestckpt` | 0.906 | 0.918 | 0.925 | 0.928 |
| M1b (8 local dims) | `hi_morphology_global_20ep` | 0.896 | 0.910 | 0.915 | 0.919 |
| M1b + clustering expert (MSE) | `hi_morphology_global_clustering_20ep` | 0.877 | 0.892 | 0.904 | 0.908 |
| M1b + MAE expert | `hi_morphology_global_mae_20ep_bestckpt` | 0.872 | 0.885 | 0.894 | 0.898 |
| M3 BC-only | `hi_morphology_bc_only_20ep_bestckpt` | 0.889 | 0.897 | 0.899 | 0.900 |
| Contrastive baseline | `hi_contrastive_20ep` | 0.818 | 0.849 | 0.857 | 0.863 |
| M2 expert + contrast (10 ep, M4 best) | `hi_morph_global_contrast_10ep_bestckpt` | 0.885 | 0.894 | 0.895 | 0.897 |
| M2 expert + contrast (20 ep, M4 best) | `hi_morph_global_contrast_20ep_bestckpt` | 0.880 | 0.897 | 0.898 | 0.896 |

**Clustering+proj minus contrastive+proj (test AUROC Δ):**

| Train % | Δ AUROC | Interpretation |
|---------|---------|----------------|
| 10% | +0.010 | M1b+proj still best at 10% |
| 25% | **+0.008** | Clustering+proj leads |
| 50% | **+0.005** | |
| 100% | **+0.001** | Near tie |

**Contrastive+proj minus M1b (test AUROC Δ):**

| Train % | Δ AUROC | Interpretation |
|---------|---------|----------------|
| 10% | **+0.010** | Projection closes M1b lead; M1b+proj still best at 10% |
| 25% | +0.008 | Projection leads |
| 50% | +0.010 | |
| 100% | +0.009 | |

**M1b+proj minus contrastive+proj (test AUROC Δ):**

| Train % | Δ AUROC | Interpretation |
|---------|---------|----------------|
| 10% | **+0.012** | Morph + projection helps under extreme scarcity |
| 25% | +0.004 | Near tie |
| 50% | −0.006 | Contrastive+proj leads |
| 100% | −0.006 | |

**Conclusions:**

1. **Projection is the main label-efficiency win.** Contrastive+proj beats plain M1b at **every** fraction (+0.008 to +0.010 AUROC). This flips the first batch (six encoders, pre-projection) where M1b led at all fractions vs plain contrastive.
2. **M1b + projection is best at 10% labels** (0.918 vs 0.916 clustering+proj vs 0.906 contrastive+proj). Morphology + projection helps under extreme scarcity; clustering+proj adds a small gain at 25%+.
3. **Clustering + projection leads at 25–100%** (0.926 / 0.930 / 0.929) — extends the full-label SSL win under most label budgets.
4. **Plain M1b vs plain contrastive:** M1b still wins at all fractions (+0.056 to +0.078 AUROC) — morphology expert helps without projection, but projection encoders dominate both.
5. **M2 does not rescue expert+M2 under label scarcity.** M4-best M2 trails M1b and all projection encoders at every fraction.
6. **F1 at low fractions is noisy** (~253 train positives at 10%). **AUROC is the primary metric** for this analysis.
7. **Clustering expert alone hurts LE** (`hi_morphology_global_clustering_20ep` 0.877–0.908); **with projection** it becomes the LE leader at 25%+.
8. **MAE expert loss** (`hi_morphology_global_mae_20ep_bestckpt`) underperforms MSE clustering expert at every fraction (0.872–0.898 vs 0.877–0.908).

**Historical note:** First batch (six encoders, pre-projection) had M1b winning all fractions vs plain contrastive; largest gap at 10% (+0.078). That story holds for **plain** encoders only — projection changes the ranking.

### Relax-grid label-efficiency (Jun 12, 2026)

**Batch:** `slurm/run_contrastive_relax_label_efficiency.sh` — sym, 8192neg, asym@16384 vs existing summary encoders.

**Test AUROC** (`hi_contrastive_proj_*` relax grid vs baseline asym+proj @ bs=32768):

| Encoder | 10% | 25% | 50% | 100% |
|---------|-----|-----|-----|------|
| **sym + proj** (`bs=16384`) | **0.924** | **0.926** | 0.929 | 0.929 |
| **8192 negs + proj** | 0.917 | 0.922 | **0.931** | **0.931** |
| asym @ 16384 + proj | 0.915 | 0.922 | 0.922 | 0.922 |
| Baseline asym + proj | 0.906 | 0.918 | 0.925 | 0.928 |
| M1b + clustering + proj | 0.916 | 0.926 | 0.930 | 0.929 |

**Takeaways:** (1) Relax recipes beat baseline asym+proj at **every** fraction (+0.003 to +0.018 AUROC). (2) **8192neg** edges morphology/clustering+proj at **50–100%** (+0.001–0.002 AUROC). (3) **sym** wins **10%** AUROC (0.924 vs M1b+proj 0.918) and **F1** at all fractions (~0.20–0.22 vs baseline 0.10–0.15). (4) 8192neg is weak at 10% labels (0.917 AUROC, 0.137 F1) despite best full-label ranking.

**Reproduce:**

```bash
python scripts/label_efficiency_probe.py \
  --unique_names hi_morphology_global_20ep hi_contrastive_20ep \
  --class_weight model --model gin --testing
```

---

## Motivation

**Observed issue (May 2026):** hetero edge-identity InfoNCE at 20 epochs (with memory-constrained asymmetric / 1024 negatives) reached ~**0.83** test probe AUROC vs ~**0.97** for supervised CE on the same Small-HI setup. Train contrastive loss was still decreasing, so budget and objective strength both matter.

Edge-identity contrastive learning only enforces **augmentation invariance** on the same transaction ID. It does not explicitly require embeddings to reflect **structural role** (hub vs leaf, dense vs sparse neighborhood, etc.) that may matter for AML.

**Morphology hypothesis:** add intrinsic, label-free structure targets so pretrain shapes the **128-dim transaction embedding** `z` (see readout below) in ways that transfer better to downstream probes—without using `Is Laundering` during pretrain.

**Papagei analogy (PPG foundation model):**

| Papagei role | Graph / transaction analogue |
|--------------|------------------------------|
| Some morphology dimensions used for **contrast / alignment** | Similar **local** structural profile → softer positives or auxiliary contrast |
| Other morphology targets predicted by **expert heads** | Regression / classification on metrics not used (or not only used) for pairing |
| Pretrain quality without downstream labels | Monitor **contrastive loss + morphology val loss**—not AML F1 during pretrain |

---

## Readout: transaction embeddings are edge-centric (all models)

Morphology integration is **model-agnostic**. We care which **architecture** (GIN, GAT, PNA, RGCN) performs best end-to-end, but morphology losses attach to the **same transaction embedding contract** every model exposes.

All `*e` encoders in `models.py` (`GINe`, `GATe`, PNA edge readout, `RGCN` edge readout) use the same pattern on homogeneous forward edges; hetero training uses forward `('node', 'to', 'node')` only for contrastive loss and extraction:

- Per directed transaction: concatenate **sender node embedding**, **receiver node embedding**, and **edge attribute embedding**, then `embedding_head` → **`z` ∈ R^128**.
- **Nodes** currently use a placeholder feature (`Feature` = 1s); **edge** channels carry timestamp, amount, currency, payment format, etc.
- Downstream AML probe and contrastive seed selection operate on **`z` per seed edge** (`edge_id` / `EdgeID`).

| Component | Model-specific? |
|-----------|-----------------|
| Message passing (GIN / GAT / PNA / RGCN) | Yes |
| Morphology expert head & morph contrast on `z_seed` | **No** |
| Tier 0 offline node tables | **No** (graph statistics) |
| `gradient_checkpointing` under `to_hetero` | GIN-only today (infra; separate from morphology design) |

**Implication for morphology:** metrics must be expressible as **targets or similarity labels on seed transactions** for any model. Benchmark morphology across `--model gin|gat|pna|rgcn` with the same morph code path and probe protocol.

**References:** `models.py` (each architecture’s `embedding_head` and edge tail readout).

---

## Node metrics vs edge metrics — bridging strategy

Many classical morphology metrics are **node-level** (degree, betweenness, clustering). AML labels are **edge-level**. Use a consistent rule:

| Strategy | Description | When to use |
|----------|-------------|-------------|
| **A. Edge-native metrics** | Defined on seed edge or its forward star in the batch subgraph | Tier 0–1 default; matches readout support |
| **B. Endpoint lift** | For node metric `m(v)`, attach `m(sender)`, `m(receiver)` (or max/mean/sum) to the transaction | **Specify global vs local** `m` (see below) |
| **C. Neighborhood aggregate** | Single value per seed from induced subgraph (e.g. mean clustering of 1-hop neighbors) | Local structure role of the *transaction in context* |
| **D. Subgraph vector** | Small vector of Tier-1 stats → expert head predicts vector | v1 expert heads |

**Reverse edges:** hetero training uses synthetic `rev_to` for message passing only. Define morphology on the **forward transaction graph** (train-split directed edges), not on reverse edges as separate transactions.

---

## Global vs local morphology (critical distinction)

The GNN does **not** see the full train graph each step—it sees a **random k-hop subgraph** per seed. Morphology targets must be labeled by what they measure.

| Symbol | Definition | Matches one forward pass? | Compute |
|--------|------------|---------------------------|---------|
| **`morph_local(edge)`** | Stats on the **batch subgraph** (view1): local degree, local clustering, ego size, etc. | **Yes** | Tier 1, **in-batch** |
| **`morph_global(edge)`** | Stats on the **full split graph**: e.g. `deg_train(sender)`, `bc_train(sender)` | **Partially** — global role from partial view | Tier 0 / 2, **offline lookup** |
| **Edge-native** | Amount, timestamp deltas, etc. | N/A (properties of the transaction) | CSV / `edge_attr` or cheap edge table |

Example — “degree sum” is **two different targets**:

```text
deg_sum_global(e) = deg_full(sender) + deg_full(receiver)   # train-split graph
deg_sum_local(e)  = deg_sub(sender) + deg_sub(receiver)     # current batch subgraph only
```

**v0 default (M1):** expert head + (later) morph contrast use **`morph_local` + edge-native** only—aligned with what message passing can observe.

**v1+ optional:** add **`morph_global`** dimensions via endpoint lift (precomputed node table). That trains **inference of global role from a local neighborhood** (useful but a different learning problem than predicting visible structure). Document which mode each run uses.

**Morphology contrast:** prefer **`morph_local`** bins for soft positives first; global bins can group edges that look different locally and add noise.

Recommended target vectors:

```text
# v0 (default)
morph(edge) = [ edge_native(e), morph_local(e) ]

# v1+ (optional ablation)
morph(edge) = [ edge_native(e), morph_local(e), morph_global(e) ]
```

Normalize using **train-split** statistics for binning / standardization. Expert heads predict `morph(edge)` from `z_seed`.

---

## What to precompute (scope)

You do **not** precompute morphology for every possible subgraph sample (stochastic, unbounded). You precompute **cheap global objects** once per split and compute **sample-dependent** metrics when a seed appears in a batch.

| Scope | Approx. size (Small-HI) | Precompute? | Notes |
|-------|-------------------------|-------------|--------|
| **All nodes** in train / val / test split | ~515k nodes | **Yes** (Tier 0 globals) | One graph pass per split; O(nodes + edges) |
| **All train transactions** (forward edges) | ~3.25M train edges | **Optional** | Only for cheap **edge-native** scalars not already in `edge_attr` |
| **Seeds visited this epoch** | Subset per epoch | **No** — use lookup + in-batch | Over many epochs, training covers train seeds |
| **Every sampled subgraph** | Unbounded | **Never** | Tier 1 computed in-batch only |

**Lookup pattern:** offline table keyed by `node_id` → at training time, for each seed edge, `morph_global(e) = f(table[sender], table[receiver])`. No per-edge precompute required for global node metrics.

**Already available without a morphology script:** amount, currency, payment format, timestamp (in `edge_attr` / CSV). Treat as edge-native baselines, not Tier 0 graph passes.

---

## Stochastic subgraphs vs Papagei-style precompute

`LinkNeighborLoader` + `num_neighs` samples a **random k-hop subgraph** per step. Augmentations differ across contrastive views. Therefore:

| Metric class | Depends on sampled subgraph? | Precompute on full graph? |
|--------------|------------------------------|---------------------------|
| **Global node** (train-split in/out degree) | No | **Yes** — once per split, keyed by `node_id` |
| **Global node** (betweenness on train graph) | No | **Yes** — Tier 2, expensive |
| **Local / ego** (clustering in batch) | **Yes** | Only if sampling is **fixed** (not default) |
| **Edge-local** (amount bin, time delta) | Partially | Some from CSV, some from batch |

**Design rule:** metrics that must match what the GNN sees should be computed **online on the forward subgraph in the current batch (view1)**. Papagei-style offline tables are appropriate for **split-safe global endpoint statistics**, not for “clustering coefficient of the full graph at this edge” unless the definition is explicitly global and documented.

> Papagei precomputes morphology on fixed windows; we precompute only **split-safe global** quantities and compute **sample-dependent** morphology in-batch.

---

## GPU memory & batch scale (mandatory for implementers)

Production contrastive runs use **large seed batches** (e.g. `--batch_size 32768`) so steps amortize subgraph sampling. Morphology code must stay compatible with that scale.

**Do not materialize dense seed–seed structures** whose memory grows as **O(B²)** in the number of shared seed edges `B` per step, unless there is an explicit, documented cap and a sparse fallback. Examples to avoid on the training hot path:

| Avoid | Typical size @ B=32k | Prefer |
|-------|----------------------|--------|
| Boolean positive mask `(B, B)` | ~1 GB+ | Bin-id **grouping**: positives = rows in the same morphology bin |
| Full similarity / logits matrix `(B, B)` float32 | ~4 GB+ | Row-chunked matmuls (already used in InfoNCE denominator) |
| Negative mask `(B, B + queue)` | larger still | Rejection sampling by `edge_id` + bin id (no dense neg mask) |
| All-pairs morph contrast without binning | O(B²) pairs | Quantile bins + optional cap on soft positives per anchor |

**Allowed / expected costs (O(batch) or O(subgraph)):**

- Tier 1 local stats on the **forward subgraph** (O(edges in batch)).
- Tier 0 **lookup** per seed (O(B)).
- InfoNCE with **subsampled negatives** (`--contrastive_num_neg_samples`) and optional **memory queue** (queue is not a dense matrix).
- Morphology contrast (M2): **bin-grouped** soft positives in `contrastive_loss.py` — no `(B, B)` mask; `--morph_contrast_max_soft_positives` (default 256) limits numerator work when many seeds share one bin.

**Guideline for new morphology features or losses:** if pairing or similarity is needed, use **(a)** per-seed scalar/vector targets (expert head), **(b)** coarse **bin ids** with group iteration, or **(c)** sampled pairs — not a full dense label matrix over seeds.

**Peak VRAM** is still dominated by GNN message passing on the sampled subgraph; morphology must not add a second quadratic-in-`B` term on top.

---

## Metric tiers

> **Glossary:** column-by-column definitions → [`morphology-reference.md`](morphology-reference.md).

**Legend:** ✅ implemented in code · 🔌 wired into M1 expert loss · ⏳ planned · — not started

| Tier | In M1 expert loss today | Library / plumbing only |
|------|-------------------------|-------------------------|
| **0** | Edge-native + global degree lift (M1b: `local+global`) | — |
| **1** | 11 local features (8 degree/ego + 3 clustering) | ✅ M1c |
| **2** | — | — |
| **3** | — | — |

See [`morphology/tier0_global.py`](../morphology/tier0_global.py), [`morphology/tier1_local.py`](../morphology/tier1_local.py), [`morphology/expert.py`](../morphology/expert.py).

---

### Tier 0 — cheap global & edge-native (mostly precomputable)

**Scope:** train / val / test graphs separately (no leakage across days). Produces **`morph_global`** and edge-native components—not the default v0 expert target set alone.

| Status | Metric | Level | Scope | Compute | Role |
|--------|--------|-------|-------|---------|------|
| ✅ 🔌 | Timestamp, amount, currency, payment format | Edge | **Native** | Already in `edge_attr`; gathered per seed in M1 | Expert targets via `include_edge_native` (default on) |
| ✅ 🔌† | In/out/total degree on split graph | Node | **Global** | `compute_tier0_node_stats` → offline table | Endpoint lift → `morph_global` |
| ✅ 🔌† | Degree **sum** from global endpoint degrees | Edge | **Global** | `lift_node_to_seed_edges` / `lift_global_to_seed_edges_torch` | M1b expert when `local+global` |
| — | Degree **product** from global endpoint degrees | Edge | **Global** | Derived from node table | Optional v1+ expert / features |
| — | Time since previous edge (per node, split-safe) | Node | **Global** | Offline rolling on split (`timestamps` arg reserved) | Endpoint lift |

**Code names (lift):** `sender_deg_in`, `sender_deg_out`, `sender_deg_total`, `receiver_deg_in`, `receiver_deg_out`, `receiver_deg_total`, `deg_sum_out_global`, `deg_sum_in_global`, `deg_sum_total_global` — see `DEFAULT_LIFT_FEATURE_NAMES` in `tier0_global.py`.

**Not Tier 0:** degree sum/product on the **sampled subgraph** — that is **Tier 1 / local** (see below).

**Integration:** optional feature augmentation (concat global scalars to `edge_attr`) or M1b expert targets via `--morph_targets local+global`. Precompute script: [`scripts/precompute_morphology_tier0.py`](../scripts/precompute_morphology_tier0.py).

† Wired when `--morph_targets local+global` (M1b); library-only for `local`.

---

### Tier 1 — local, batch-aligned (primary SSL targets for v0)

Computed on **forward edges and nodes present in the current seed batch subgraph** (before or after augmentation; prefer **view1** for targets).

| Status | Metric | Level | Code name(s) | Notes |
|--------|--------|-------|--------------|-------|
| ✅ 🔌 | Subgraph edge count | Edge / local | `n_edges_sub` | **Batch-level:** total edges in view1 subgraph; **same value for every seed in the batch** (M2 group `local_ego`); log1p in expert |
| ✅ 🔌 | Subgraph node count | Edge / local | `n_nodes_sub` | **Batch-level:** unique nodes in that subgraph; not per-seed k-hop ego; log1p in expert |
| ✅ 🔌 | Sender out/in degree *within subgraph* | Node | `sender_deg_out_local`, `sender_deg_in_local` | **`morph_local`** — differs from global degree |
| ✅ 🔌 | Receiver out/in degree *within subgraph* | Node | `receiver_deg_out_local`, `receiver_deg_in_local` | Same |
| ✅ 🔌 | Out/in degree **sum** on subgraph endpoints | Edge | `deg_sum_out_local`, `deg_sum_in_local` | **`deg_sum_local`** — do not confuse with Tier 0 global sum |
| — | Degree **product** on subgraph endpoints | Edge | — | Not implemented |
| ✅ 🔌 | Local clustering (sender, receiver, mean) | Node | `sender_clustering_local`, `receiver_clustering_local`, `mean_clustering_local` | Undirected clustering on view1 subgraph |
| — | 2-hop reachable count from endpoints | Node | — | Cheap BFS cap within batch |
| — | Triangle / wedge counts involving seed | Edge-local | — | Motif-lite |

**Integration:**

- **Expert head (M1):** MLP on `z_seed` → predict all **11** Tier-1 locals + edge-native (MSE). Degree/ego cols get `log1p`; clustering cols stay in **[0, 1]**.
- **Morphology contrast (M2):** optional bin groups on subsets of the same vector — see **M2 feature groups** below. Hard positive = same `edge_id` across views unchanged.

**M2 `--morph_contrast_features` groups** (comma-separated; map to local column indices):

| Group | Local indices | Columns |
|-------|---------------|---------|
| `local_ego` | 0–1 | `n_edges_sub`, `n_nodes_sub` |
| `local_degree` | 2–7 | sender/receiver degrees, degree sums |
| `local_clustering` | 8–10 | `sender_clustering_local`, `receiver_clustering_local`, `mean_clustering_local` |
| `global_degree` | Tier 0 block | requires `--morph_contrast_scope local+global` |
| `edge_native` | `edge_attr` block | forward edge features (excl. EdgeID) |

Default M2: `local_ego,local_degree`. Clustering contrast is **opt-in** via `local_clustering` (does not change expert targets).

**Cost:** O(batch subgraph) per step for degrees; clustering adds O(E) adjacency build + O(Σ deg²) over active nodes on view1 (CPU). No full-graph precompute.

---

### Tier 2 — split-global, expensive but informative (later)

Computed **offline on train-only directed graph** (val/test graphs separately for eval metrics only—never train on val/test structure for pretrain targets).

| Status | Metric | Level | Compute notes |
|--------|--------|-------|---------------|
| — | Betweenness centrality | Node | Approximate (Brandes sampled) acceptable |
| — | k-core / shell index | Node | Moderate |
| — | PageRank-style stationary | Node | Moderate |
| — | Global clustering coefficient | Node | Cheaper than betweenness |

**Integration:** endpoint lift `{bc(sender), bc(receiver)}` → optional **`morph_global`** slice for expert head (v1+ / M3); optional contrast on coarse BC quantile bins.

**Do not** use Tier 2 in v0; validate Tier 1 local + identity InfoNCE first.

---

### Tier 3 — research / optional (defer)

| Status | Item |
|--------|------|
| — | Full-graph exact betweenness on all 5M+ edges every epoch |
| — | Motif census on full graph |
| — | Morphology contrast with all-pairs O(B²) or dense `(B,B)` masks (see **GPU memory** section) |

---

## Recommended next metrics (earlier tiers)

**M1b done (Jun 2026):** `hi_morphology_global_20ep` test AUROC **0.920** / F1 **0.108** vs M1 **0.910** / **0.079** — global lift helps.

**M2 decision run done (Jun 2026):** expert+M2 @ 20 ep (`hi_morph_global_contrast_20ep`) test AUROC **0.864** / F1 **0.025** — **below** M1b and **below** expert+M2 @ 10 ep (0.900 / 0.037). M2 does not beat M1b at matched epochs; retune (weight, stopping, checkpoint) before Tier 2.

| Priority | Metric | Tier | Rationale |
|----------|--------|------|-----------|
| ~~**1**~~ | ~~Global degree endpoint lift~~ | 0 | ✅ M1b — test AUROC +0.010, F1 +0.029 vs M1 local-only. |
| **1** | **M2 morphology-aware contrast** (+ optional `local_clustering` bins) | — | Degree-only bins underperformed M1b; re-test with clustering group @ 10 ep. |
| ~~**2**~~ | ~~**Local clustering** (sender, receiver, mean)~~ | 1 | ✅ Implemented Jun 2026; benchmark M1b vs `hi_morphology_global_clustering_20ep`. |
| **3** | **Time since previous edge** (per endpoint, split-safe) | 0 | Temporal burstiness; cheap offline rolling per split. |
| **4** | **2-hop reachable count** (capped BFS from endpoints) | 1 | Extends ego-scale beyond 1-hop degree. |
| Lower | Degree **product** (local or global) | 0 / 1 | Redundant with sum for many hubs. |
| Lower | Triangle / wedge counts | 1 | Overlaps local clustering. |
| Defer | Tier 2 (BC, PageRank, k-core) | 2 | After M2 benchmarked. |

**Suggested experiment order:** ~~clustering+projection full-label~~ (0.929) → ~~label-efficiency on clustering+proj~~ (0.916–0.930) → ~~relax grid core~~ (sym 0.929/0.222; 8192neg 0.930/0.191; asym@16384 confound) → optional sym+8192 / no-queue → label-efficiency on winners → new Tier-0/1 metrics → external dataset before more Tier-1 stacks.

---

## Loss design (target architecture)

Total pretrain loss (conceptual):

```text
L = L_edge_infonce          # existing: same edge_id across augmented views
  + λ_exp * L_morph_expert  # predict Tier 0–1 (later Tier 2) targets from z_seed
  + λ_morph * L_morph_con   # optional: morphology-similarity contrast (v1+)
```

**`L_edge_infonce`:** current `edge_identity_infonce_loss` on shared seed edges (hetero forward type).

**`L_morph_expert`:** Papagei expert heads—predict **`morph_local` + edge-native** in v0; add **`morph_global`** in later ablations. Use **stopped-gradient** targets: batch calculator for local; optional join to offline node table on `from_id` / `to_id` for global.

**`L_morph_con`:** pull together `z` for seeds in the same morphology bin; push apart different bins or use as weighted positives in InfoNCE. Start **after** expert-only ablation shows morph targets are stable. **Implementation (M2):** merged into `L_edge_infonce` via bin-grouped soft positives — **no dense `(B,B)` mask** (see **GPU memory & batch scale**).

**Hyperparameters to log:** `λ_exp`, `λ_morph`, temperature, bin edges, which metrics are in contrast vs expert only.

**Checkpoint selection (pretrain-native):** morphology validation loss (+ contrastive train loss)—**not** AML probe val (see contrastive plan).

---

## Integration options (summary)

| Option | Role in our stack | Phase |
|--------|-------------------|-------|
| **A. Expert prediction** | Papagei heads on `z_seed` | v0 |
| **B. Feature augmentation** | Global Tier 0 concat to `edge_attr` | optional ablation |
| **C. Morphology regularizer** | Embedding space geometry | v2 if needed |
| **D. Morphology-aware contrast** | Stronger SSL than identity only | v1 |

Avoid enabling A + B + C + D at once; one ablation axis per run.

---

## Design decisions (agreed direction)

| Decision | Choice |
|----------|--------|
| Models in scope | **All** (`gin`, `gat`, `pna`, `rgcn`) — shared `z_seed` morphology API |
| v0 expert targets | **`morph_local` + edge-native** only |
| v0 morph contrast | **`morph_local`** bins (M2) |
| Global node metrics (degree, later BC) | Precompute **all nodes per split**; lift to edge at train time — **not** v0 expert default |
| Precompute per subgraph | **Never** |
| Precompute all train edges | **Only** if edge-native and not already in `edge_attr` |
| Global vs local degree sum | **Different targets** — document which a run uses |
| Morph contrast @ large `B` | **Bin groups + optional soft-positive cap** — never dense `(B,B)` masks on the hot path |

---

## Temporal leakage (mandatory)

Train / val / test are **day-based** splits. Any offline metric must be computed on **edges in that split’s days only**.

- Pretrain: use **train graph only** for global node tables used during training.
- Morphology val loss: **val graph only** for val-phase global stats (or batch-local only).
- Never compute betweenness / degree on the **union** of train+val+test for training targets.

---

## Prerequisites (sequencing)

Morphology should not block the core pipeline, but implementation starts after:

1. Stable **extract → linear probe** path (Phase 5a/5b).
2. Reproducible **hetero contrastive** baseline (even if AUROC < supervised).
3. **Contrastive recipe at scale** (epochs, symmetric/neg count) benchmarked on stronger GPUs when available.

**Do not** interpret morphology as fixing a broken identity-InfoNCE setup without re-benchmarking plain contrastive at sufficient budget.

---

## Implementation plan

Phases are listed in dependency order below. The phase summary in [`morphology-reference.md`](morphology-reference.md) is a short lookup; this section is the full write-up.

| Phase | Topic | Status |
|-------|-------|--------|
| **M0** | Spec & plumbing (no new loss) | ✅ |
| **M1** | Expert head, Tier-1 local targets | ✅ |
| **M1b** | Tier-0 global endpoint lift in expert | ✅ |
| **M1c** | Tier-1 local clustering in expert (+ optional M2 group) | ✅ |
| **M2** | Morphology-aware contrast (soft positives) | ✅ |
| **M3** | Tier-2 BC precompute + expert wiring | ✅ Phase 0–1 |
| **M4** | Checkpoint policy (`--checkpoint_policy best`) | ✅ |
| **M5** | Expert head layout (grouped / per-metric) | M5a ✅ · M5b proposed |

**Not implemented (Papagei-inspired):** disjoint morphology feature sets for contrast bins vs expert targets (default overlap: `local_ego`, `local_degree`).

---

### Phase M0 — specification & plumbing (no new loss) ✅

- [x] Document `from_id` / `to_id` / `EdgeID` join → [`morphology/IDS.md`](../morphology/IDS.md).
- [x] Utility: `compute_tier0_node_stats(edge_index, num_nodes)` → [`morphology/tier0_global.py`](../morphology/tier0_global.py).
- [x] Utility: `lift_node_to_seed_edges(seed_edge_ids, edge_index, node_table)` (endpoint lift; no separate sender/receiver args).
- [x] Tier 1 local: `compute_local_morphology` + `resolve_seed_positions_in_subgraph` → [`morphology/tier1_local.py`](../morphology/tier1_local.py).
- [x] Unit tests → [`tests/test_morphology_metrics.py`](../tests/test_morphology_metrics.py) (`python -m unittest tests.test_morphology_metrics`).

**Files:**

| Path | Role |
|------|------|
| `morphology/` | Package: graph accessors, Tier 0 global, Tier 1 local |
| `scripts/precompute_morphology_tier0.py` | Offline node tables per train/val/test split |
| `morphology_cache/{data}/` | Suggested output dir for CSV tables (gitignored as needed) |

---

### Phase M1 — expert head only (v0, local targets) ✅

- [x] Online Tier-1 on view1 subgraph: `compute_local_morphology_torch` + `transform_morph_targets` (+ optional edge-native). **14** local dims since M1c (8 degree/ego + 3 clustering + 3 triangle).
- [x] `MorphologyExpertHead` in [`morphology/expert.py`](../morphology/expert.py).
- [x] `L_morph_expert` (MSE, weighted by `--morph_expert_weight`).
- [x] Wired into `train_hetero_contrastive` and `train_homo_contrastive` via `--morph_expert`.
- [x] Logs `morph/expert_train`, `morph/expert_val` (val loader pass, no AML).
- [x] Diagnostic per-group train MSE logging: `morphology/loss_group/{degree_fan,local_motif,centrality,flow_balance,volume_activity,temporal,other}` plus optional per-target keys. This does **not** change the shared expert head or total expert loss.
- [x] Eval: **frozen extract + linear probe** vs contrastive baseline → `embeddings/hi_morphology_20ep/` vs `embeddings/hi_contrastive_20ep/` (Jun 2026: test AUROC 0.910 vs 0.839).

**CLI flags** (`util.py`): `--morph_expert`, `--morph_targets local`, `--morph_expert_weight`, `--morph_expert_hidden`, `--no_morph_edge_native`.

**Example:**

```bash
python main.py --data Small-HI --model gin --objective contrastive \
  --reverse_mp --ego --ports --unique_name hi_contrastive_morph_20ep \
  --morph_expert --morph_targets local --save_model --n_epochs 20 --testing
```

Checkpoint includes `morph_expert_state_dict` when `--morph_expert` is set (encoder extraction unchanged).

**Success:** probe AUROC improves vs identity-only contrastive at **same epochs**; expert loss decreases.

---

### Phase M1b — global endpoint lift (Tier 0 in expert head) ✅

- [x] `--morph_targets local+global`: concat Tier 0 lifted global degrees after local targets (before edge-native).
- [x] `MorphTier0Context` + `lift_global_to_seed_edges_torch` → [`morphology/tier0_global.py`](../morphology/tier0_global.py).
- [x] Train targets use **train-split** graph/table; val morph loss uses **val-split** only (no leakage).
- [x] Optional `--morph_tier0_cache` dir; otherwise compute from split graphs at startup.
- [x] Eval: frozen extract + linear probe vs M1 local-only → `embeddings/hi_morphology_global_20ep/` (Jun 2026).

**Results (Small-HI, 20ep, same probe protocol as M1):**

| Run | Test AUROC | Test F1 (val-tuned) | Test P / R |
|-----|------------|---------------------|------------|
| Contrastive baseline | 0.839 | 0.076 | 0.053 / 0.137 |
| M1 (`local`) | 0.910 | 0.079 | 0.046 / 0.290 |
| **M1b (`local+global`)** | **0.920** | **0.108** | **0.069 / 0.248** |

M1b improves test AUROC **+0.010** and F1 **+0.029** vs M1; val AUROC slightly lower (0.914 vs 0.921). **Default morph expert config for M2:** use `local+global` unless ablating.

**CLI flags:** `--morph_targets local+global`, `--morph_tier0_cache morphology_cache/Small-HI` (optional).

**Example:**

```bash
# Optional: precompute once (reuse across runs)
python scripts/precompute_morphology_tier0.py \
  --data Small-HI --output_dir morphology_cache/Small-HI \
  --reverse_mp --ego --ports

python main.py --data Small-HI --model gin --objective contrastive \
  --reverse_mp --ego --ports --unique_name hi_morphology_global_20ep \
  --morph_expert --morph_targets local+global --save_model --n_epochs 20 --testing
```

Expert target dim with defaults: local **11** + global **9** + edge-native **4** = **24**.

**Deferred to M3:** Tier 2 globals (betweenness, etc.) in `local+global`.

**Note (Jun 2026):** Tier-1 local block is now **11** dims (clustering appended). Prior `hi_morphology_global_20ep` used 8 local dims; re-run as `hi_morphology_global_clustering_20ep` for apples-to-apples with current code.

---

### Phase M1c — Tier-1 local clustering (Jun 2026) ✅

- [x] Undirected local clustering on view1 subgraph: `sender_clustering_local`, `receiver_clustering_local`, `mean_clustering_local` in [`morphology/tier1_local.py`](../morphology/tier1_local.py).
- [x] Wired into M1/M1b expert targets automatically (no new CLI flag).
- [x] M2 opt-in group `local_clustering` in [`morphology/contrast.py`](../morphology/contrast.py) (`--morph_contrast_features`).
- [x] Unit tests: triangle → 1.0, chain → 0.0; contrast column index tests.

**Expert target dims (updated):** `local` → **15** (11 + 4 edge-native); `local+global` (M1b) → **24** (11 + 9 + 4).

**Benchmark:**

| Run | `unique_name` | Result vs baseline |
|-----|---------------|-------------------|
| M1b + clustering expert | `hi_morphology_global_clustering_20ep` | ✅ full-label **0.903** / **0.117** F1; LE **0.877–0.908** AUROC — below M1b at all fractions |
| M1b + clustering + projection | `hi_morphology_global_clustering_proj_20ep_bestckpt` | ✅ **0.929** / **0.156** — best SSL; LE **0.916–0.930** |
| M1b + MAE expert loss | `hi_morphology_global_mae_20ep_bestckpt` | ✅ **0.898** / **0.145** — below MSE 0.903 |
| M1b + M2 + clustering bins | `hi_morph_global_clustering_m2_10ep_bestckpt` | ⏳ pending (low priority) |

### Phase M1d — Tier-1 local triangle counts (Jun 2026) ✅ code · ablation done

- [x] Undirected triangle counts at seed endpoints on view1 subgraph (`sender_triangles_local`, `receiver_triangles_local`, `mean_triangles_local`).
- [x] M2 opt-in group `local_triangles` in [`morphology/contrast.py`](../morphology/contrast.py).
- [x] `--morph_local_subset {all,degree,clustering,triangles}` to ablate expert columns without recomputing metrics.

**Ablation — stacked clustering + triangles + projection** (`hi_morphology_global_triangles_proj_20ep_bestckpt`, default `all` = 14 local dims):

| Metric | Triangles+proj (14 local) | Clustering+proj (11 local) | Δ |
|--------|---------------------------|----------------------------|---|
| Val AUROC | 0.933 | 0.930 | +0.003 |
| **Test AUROC** | **0.912** | **0.929** | **−0.017** |
| Test F1 | 0.145 | 0.156 | −0.011 |

**Conclusion:** Triangle counts **do not help** when stacked with clustering; large val→test AUROC gap suggests redundant/noisy targets. **Do not adopt** as default. **Triangles-only** expert (`--morph_local_subset triangles`, 11 dims) ablation pending.

### M1b + symmetric contrastive + projection (Jun 2026) ✅

**Run:** `hi_morphology_global_sym_proj_20ep_bestckpt` · `slurm/ablation_m1b_sym_projection_20ep.sh` — M1b (14 local) + sym InfoNCE + projection @ `bs=16384 accum=2`.

| Metric | sym + morph + proj | sym + proj (no morph) | clustering + proj |
|--------|-------------------|----------------------|-------------------|
| Val AUROC | 0.938 | 0.939 | 0.930 |
| **Test AUROC** | **0.930** | 0.929 | 0.929 |
| **Test F1** | **0.134** | **0.222** | 0.156 |

**Conclusion:** Morphology expert on the relax-grid sym recipe gives **+0.001** test AUROC vs plain sym but **−0.088** test F1. No “best of both worlds.” **Keep sym contrastive+proj for F1**; do not stack M1b expert on sym by default.

**Example — M1b with clustering (expert only):**

```bash
python main.py --data Small-HI --model gin --objective contrastive \
  --reverse_mp --ego --ports --unique_name hi_morphology_global_clustering_20ep \
  --morph_expert --morph_targets local+global \
  --morph_tier0_cache morphology_cache/Small-HI \
  --save_model --n_epochs 20 --checkpoint_policy best \
  --contrastive_asymmetric --contrastive_num_neg_samples 1024 \
  --contrastive_memory_bank_size 32768 --testing
```

**Example — M2 with clustering bins:**

```bash
# Same M1b expert flags, plus:
  --morph_contrast \
  --morph_contrast_features local_ego,local_degree,local_clustering \
  --morph_contrast_scope local
```

**Success criterion:** test AUROC or label-efficiency ≥ prior M1b / expert+M2 at matched epochs. **Expert-only:** not met (0.903). **With projection:** met — **0.929** full-label; LE **0.926–0.930** @ 25–100%.

---

### Phase M2 — morphology-aware contrast (v1) ✅

- [x] Binning on selected morphology features (view1 subgraph; train-split quantile edges at startup).
- [x] **Merged soft positives** into edge InfoNCE (`contrastive_loss.py`: identity OR same bin).
- [x] CLI: `--morph_contrast`, `--morph_contrast_features`, `--morph_contrast_scope`, `--morph_contrast_bins`.
- [x] **`local_clustering`** feature group (Jun 2026): bins cols 8–10 when listed in `--morph_contrast_features`.
- [x] Hetero + homo contrastive paths; val log `morph/contrast_val`.
- [x] Bin-grouped soft positives (no dense mask); see **GPU memory & batch scale**.
- [x] Ablate on Small-HI @ 10 ep: expert+M2 vs contrast-only vs M1b@20 ep (see **Project status**).
- [x] Ablate @ **20 ep**: expert+M2 vs M1b — **M1b wins** (0.920 vs 0.864 test AUROC); see **Project status** and W&B curves.
- [ ] Tune expert weight vs contrast feature groups; try **10–12 ep** stop for expert+M2.
- [ ] `--morph_contrast_max_soft_positives` caps huge same-bin groups (large `--batch_size` OK).
- [x] Runtime: `--morph_val_every N`, `--morph_val_max_batches K` throttle full val-loader morph passes (always log final epoch).

**Small-HI ablation (Jun 2026, same probe protocol):**

| Run | Ep | Val AUROC | Test AUROC | Test F1 |
|-----|----|-----------|------------|---------|
| M1b (`hi_morphology_global_20ep`) | 20 | 0.914 | **0.920** | **0.108** |
| Expert + M2 (`hi_morph_global_contrast_10ep`) | 10 | 0.913 | 0.900 | 0.037 |
| Expert + M2 (`hi_morph_global_contrast_20ep`) | 20 | 0.897 | 0.864 | 0.025 |
| M2 only (`hi_morph_contrast_only_10ep`) | 10 | — | 0.680 | 0.012 |

Contrast-only confirms the expert head carries representation quality; soft positives alone are insufficient. **More epochs hurt expert+M2** — likely SSL over-training + morphology expert overfitting (see W&B `morph/expert_train` vs `morph/expert_val`).

**Success criterion (revised):** additive gain over **M1b** on probe AUROC or label-efficiency curve — **not met at 20 ep**; partially met at 10 ep AUROC (0.900 vs 0.920) with poor F1.

---

### Phase M3 — Tier 2 global metrics (betweenness centrality)

**Phase 0 (implemented):** BC precompute + endpoint lift plumbing.

- [x] Offline approximate betweenness per split ([`scripts/precompute_morphology_tier2.py`](../scripts/precompute_morphology_tier2.py); run on ~128G RAM — login node can OOM on Small-HI).
- [x] Endpoint lift in `morphology/tier2_global.py`; tests `tests/test_morphology_tier2.py`.

**Phase 1 (implemented + benchmarked):** BC in morphology expert head.

- [x] `--morph_targets local+global+tier2` (M1b + BC) and `local+tier2` (BC-only global ablation).
- [x] `--morph_tier2_cache`, `--morph_tier2_lift {full,max}` (`full` = 4 cols; `max` = `bc_max_global` only).
- [x] Ablations benchmarked: `hi_morphology_bc_only_20ep_bestckpt`, `hi_morphology_global_bc_max_20ep_bestckpt` (see Project status).

**Small-HI M3 findings (Jun 2026):**

| Run | Test AUROC | Test F1 | vs M1b |
|-----|------------|---------|--------|
| M1b | **0.920** | **0.108** | — |
| M1b + BC (4 cols, M4 best ep 14) | 0.896 | 0.033 | −0.024 AUROC |
| M1b + BC (last ep 20) | 0.861 | 0.029 | −0.059 AUROC |

**Interpretation:** split-global BC **did not improve** the AML linear probe when stacked on global degree lift. Global degrees may already capture most useful hub/broker signal on Small-HI; extra BC dims + shared MSE may add noise. **BC-only** and **bc_max-only** ablations still pending.

- [ ] **Phase 2:** Optional coarse morph contrast on BC quantile bins (defer — M2 local bins underperform M1b).
- [x] Precompute on cluster (~128G RAM); login node OOM on Small-HI graph load.

**Success criterion (Phase 1):** probe gain over M1b — **not met** for stacked BC.

---

### Phase M4 — features & checkpoint policy (optional → **recommended**) ✅

- [ ] Option B: concat Tier-0 to `edge_attr` (ablation vs heads-only).
- [x] **Save best checkpoint** by composite morph val (`morph/expert_val` + `morph/contrast_val`) or `loss/train`; `--checkpoint_policy best` (recommended for morph and projection runs).
- [ ] (Future) hetero-safe `gradient_checkpointing` for larger contrastive batches.

**CLI:** `--checkpoint_policy {last,best}` with `--save_model`. **Best** writes the lowest SSL val score to `checkpoint_{unique_name}.tar` (used by extraction) and the final epoch to `checkpoint_{unique_name}_last.tar`. Score = sum of available `morph/expert_val` and `morph/contrast_val` on morph-val epochs; plain contrastive uses `loss/train` every epoch.

**Morph val throttling:** `--morph_val_every N` and `--morph_val_max_batches K` reduce the cost of morph val passes that feed M4 selection (see Phase M2).

**Outcome (Jun 2026):** Last-epoch `hi_morph_global_contrast_20ep` had test AUROC **0.864**; with M4 best (morph val picked epoch 20) → **0.891** AUROC, **0.107** F1. Ten-epoch M4 best (epoch **9**) → **0.906** AUROC. Morph val score still favored later epochs on the 20 ep run — future work: probe-aware selection or early stopping, not only morph val sum.

---

### Phase M5 — expert head layout (M5a grouped **implemented**; M5b per-metric proposed)

**Design ladder (increasing Papagei fidelity, increasing overhead):**

| Stage | Layout | Loss | When |
|-------|--------|------|------|
| **Current (M1–M3)** | Single shared MLP → all targets | One unweighted MSE over full vector | ✅ shipped |
| **M5a grouped** | One small MLP per **block** (local / global / tier2 / edge-native) | Σ `w_block · MSE_block` | Middle ground; lower overhead than per-metric |
| **M5b per-metric** | One small MLP per **scalar morphology target** (full Papagei-style) | Σ `w_i · loss_i` (each metric its own term) | **Preferred long-term direction** if shared/grouped heads keep failing — more overhead (many heads, many loss terms, checkpoint bulk) but likely better fit when targets differ in scale, difficulty, and gradient noise (BC vs degree vs edge-native). **Do not jump here unless grouped ablations or BC-only/bc_max results give a strong reason.** |

**Motivation:** Today a **single shared MLP** predicts all morphology targets with one unweighted MSE over ~12–27 dimensions (`morphology/expert.py`). Papagei uses **separate expert heads per morphology metric** (MoE-style), which can:

- Stop hard targets (e.g. BC, future clustering) from dominating gradients for easier scalars (local degree, edge-native).
- Allow **per-metric loss weights** without retuning a single scalar `morph_expert_weight` for the whole vector.
- Mirror Papagei’s **partition** of morphology signals (contrast vs expert disjoint sets — still a separate item).

**M5a grouped design (minimal Papagei analogue):**

```text
z_seed (128-d, shared encoder output)
    │
    ├─► head_local      → predict morph_local (8)     ─┐
    ├─► head_global     → predict morph_global (9)   ├─► L_expert = Σ_b w_b · MSE_b
    ├─► head_tier2      → predict morph_tier2 (1–4)  │   (detached targets, log1p on counts)
    └─► head_edge_native → predict edge_attr (4)      ─┘

Each head: Linear(128, h) → ReLU → Linear(h, dim_b)   [same h=64 default as today]
Blocks omitted when disabled (e.g. no head_global if morph_targets=local).
```

**Loss:**

```text
L_morph_expert = w_local·MSE(local) + w_global·MSE(global) + w_tier2·MSE(tier2) + w_edge·MSE(edge_native)
```

Default weights: all `1.0` (recovers current behavior if heads share architecture and are summed — **not** identical to one shared trunk, but ablation baseline).

**M5b per-metric design (Papagei-aligned, higher overhead):**

```text
z_seed (128-d)
    ├─► head_i  → predict target_i   (i = 1 … N_metrics)
    └─► L_expert = Σ_i w_i · loss(pred_i, target_i)

Each head: Linear(128, h) → ReLU → Linear(h, 1)   [or small 2-layer MLP per metric]
```

- One loss term and W&B scalar per metric (or per metric group in logs).
- Enables turning BC on/off via `w_i=0` without architectural surgery.
- Checkpoint: dict of N head state dicts; resume logic must match metric list for the run.

**Loss function note:** Papagei uses **MAE**; default **MSE**. Swap via `--morph_expert_loss {mse,mae}`. Ablation done: MAE **0.898** vs MSE **0.903** test AUROC (same 11-dim targets) — keep MSE default.

**CLI sketch (future):**

| Flag | Purpose |
|------|---------|
| `--morph_expert_layout {shared,grouped}` | `shared` = current; `grouped` = block heads (**M5a implemented**) |
| `--morph_expert_group_weight_tier2` | Tier 2 block MSE scale when grouped (default `1.0`; `0` = M1b-like sanity) |
| `--morph_expert_metric_weights` | Comma map or file for per-metric weights (per_metric layout) |
| `--morph_expert_loss {mse,mae}` | Expert regression loss (default `mse`; Papagei uses MAE) — ✅ implemented |

**Implementation notes:**

- Reuse `build_morph_targets()` slice indices (local / global / tier2 / edge-native order unchanged).
- Checkpoint: `morph_expert_state_dict` → dict of block state dicts or prefixed keys.
- **When to build M5a:** ~~after bc_only / bc_max~~ **done** — see `hi_morph_grouped_bc_w0_20ep_bestckpt` / `w1.0` in Project status.
- **When to build M5b:** if M5a still shows BC/tier2 hurting easy metrics, or if per-metric logging shows one scalar dominating val loss.
- **Not in scope for v1:** gating networks (full MoE routing), AML-aware checkpoint selection.

**Success criterion:** new layout ≥ shared on M1b AUROC at matched epochs, **or** enables BC/tier2 with isolated weights without hurting degree/local metrics (diagnostic for M3 negative result).

**Expert loss (MSE vs MAE):** Default `mse`; `mae` uses `F.l1_loss` on same log1p targets. **`hi_morphology_global_mae_20ep_bestckpt`:** test AUROC **0.898** (below MSE 0.903); F1 **0.145**. MAE did not improve separability on Small-HI.

**Contrastive projection head (implemented Jun 2026):** GraphCL-style MLP (`--contrast_projection_head`) maps encoder `z` before InfoNCE only; morphology expert and extraction still use raw **`embedding_head` output** `z ∈ R^128` (see `models.py`, `edge_identity_infonce_loss`). Checkpoint stores `contrast_projection_state_dict` for resume.

**Ablation results (Small-HI, 20 ep, `--checkpoint_policy best`):**

| Run | Test AUROC | Test F1 | Notes |
|-----|------------|---------|-------|
| Sym contrastive + proj (`hi_contrastive_proj_sym_20ep_bestckpt`) | 0.929 | **0.222** | **best SSL F1**; relax grid |
| 8192neg + proj (`hi_contrastive_proj_8192neg_20ep_bestckpt`) | **0.930** | 0.191 | best contrastive AUROC |
| M1b + sym + proj (`hi_morphology_global_sym_proj_20ep_bestckpt`) | **0.930** | 0.134 | morph on sym: +0.001 AUROC, −0.088 F1 vs sym-only |
| M1b + clustering + proj (`hi_morphology_global_clustering_proj_20ep_bestckpt`) | 0.929 | 0.156 | best morph stack |
| Contrastive + proj (`hi_contrastive_proj_20ep_bestckpt`) | 0.927 | 0.144 | +0.088 vs plain contrastive (0.839) |
| M1b + proj (`hi_morphology_global_proj_20ep_bestckpt`) | 0.924 | 0.096 | AUROC ↑ vs M1b; F1 ↓ at val-tuned threshold |
| M1b (no proj) | 0.920 | 0.108 | morph-only baseline |

See **Project status** table for full projection ablation numbers.

---

## Experiments to run (when enabled)

| Run | Pretrain | Full-label probe (test AUROC / F1) |
|-----|----------|--------------------------------------|
| Baseline | identity InfoNCE | ✅ 0.839 / 0.076 |
| M1b | morph expert `local+global` | ✅ **0.920 / 0.108** |
| M2 expert + contrast | + `--morph_contrast` | ✅ 10ep_bestckpt 0.906 / 0.058; 20ep_bestckpt 0.891 / 0.107 |
| M2 contrast only | contrast, no expert | ✅ 0.680 / 0.012 |
| M3 M1b + BC (4 cols) | `local+global+tier2` | ✅ 0.896 / 0.033 (ep 14 best) — **below M1b** |
| M3 BC-only | `local+tier2` | ✅ 0.904 / 0.093 — substitute OK, stack bad |
| M3 bc_max | M1b + `morph_tier2_lift max` | ✅ 0.889 / 0.086 |
| M5a grouped | `--morph_expert_layout grouped` | ✅ w0 → 0.885 / 0.043; w1.0 → 0.887 / 0.028 (below M1b & shared M3 stack) |
| M1b + clustering + projection | M1b + proj + 11 local | ✅ **0.929 / 0.156** — best morph SSL |
| M1b + sym + projection | M1b + sym @ `bs=16384` | ✅ **0.930 / 0.134** — morph hurts sym F1 |
| Sym contrastive + proj | relax grid | ✅ 0.929 / **0.222** — best SSL F1 |
| Contrastive + proj | `--contrast_projection_head` | ✅ 0.927 / 0.144 |
| M1b + proj | M1b + projection head | ✅ 0.924 / 0.096 |
| M1b + clustering expert (MSE) | M1b, 11 local dims | ✅ 0.903 / 0.117; LE 0.877–0.908 |
| M1b + MAE expert | `--morph_expert_loss mae` | ✅ 0.898 / 0.145 — below MSE |
| M1b + M2 + clustering bins | + `local_clustering` in contrast | ⏳ `hi_morph_global_clustering_m2_10ep_bestckpt` |
| Supervised ref | CE | ✅ ~0.972 / ~0.493 |

Match: data split, `num_neighs`, embedding dim, extract settings. Morph pretrain: `--checkpoint_policy best`.

---

## Open questions

1. Which Tier-1 **local** metrics correlate with illicit edges on Small-HI (exploratory EDA, train only)?
2. ~~Does adding **`morph_global`** to the expert head help probe AUROC vs local-only at matched epochs?~~ **Yes (M1b):** test AUROC 0.920 vs 0.910; F1 0.108 vs 0.079.
3. Symmetric vs asymmetric InfoNCE when morphology loss adds memory pressure?
4. ~~Should morph contrast use **same-view** or **cross-view** pairs for soft positives?~~ **v1 uses cross-view** (view1 features → bins; positives in InfoNCE across view1/view2 seed embeddings).
5. Homogeneous pipeline first for faster iteration, or hetero-only (production path)?
6. Fix `gradient_checkpointing` + `to_hetero` FX trace before relying on checkpointing for morph runs?
7. Which architecture benefits most from morphology (GIN vs GAT vs PNA vs RGCN)?
8. ~~Do **grouped expert heads** (M5) explain M3 BC regression vs shared MLP + unified MSE?~~ **No:** grouped BC (`w_tier2=1`) → 0.887 test AUROC, still below M1b (0.920) and shared M3 stack (0.896). Block-separated MSE did not fix interference.
9. ~~Does **local clustering** (Tier 1) help M1b expert?~~ **Expert-only: no** (0.903 AUROC). **With projection: yes** — **0.929** test AUROC (best SSL). Morphology benefit is recipe-dependent.

---

## Relationship to contrastive plan

| Document | Owns |
|----------|------|
| `contrastive-learning-plan.md` | Objectives, homo/hetero contrastive, extraction, AML probe workflow, no AML in pretrain loop |
| **This document** | Metric tiers, node→edge lifting, batch vs offline compute, morph expert + morph contrast, phased implementation |

Cross-reference: contrastive plan § “Related Future Direction: Morphology Metrics” points here for detail.

---

## References in repo

- **Morphology M0–M2:** `morphology/`, `morphology/IDS.md`, `scripts/precompute_morphology_tier0.py`, `tests/test_morphology_metrics.py`, `tests/test_morphology_contrast.py`, `tests/test_morph_val_schedule.py`
- **M3 Phase 0–1:** `morphology/tier2_global.py`, `scripts/precompute_morphology_tier2.py`, `tests/test_morphology_tier2.py`, `tests/test_morphology_expert_tier2.py`
- **Projection head:** `models.py` (`--contrast_projection_head`), Project status table
- **Tier-1 clustering / MAE:** [`morphology/tier1_local.py`](../morphology/tier1_local.py), `--morph_expert_loss {mse,mae}`
- **M2 loss:** [`contrastive_loss.py`](../contrastive_loss.py) (`edge_identity_infonce_loss`, bin-grouped soft positives)
- **M2 training glue:** [`morphology/contrastive_train.py`](../morphology/contrastive_train.py)
- Edge readout (all models): `models.py` (`GINe`, `GATe`, PNA, `RGCN` — shared `embedding_head` on concat sender/receiver/edge)
- Hetero contrastive: `training.py` (`train_hetero_contrastive`)
- Seed edges: `train_util.py` (`get_hetero_seed_edge_ids`)
- Probe eval: `linear_probe.py`, `embedding_extraction.py`
- Label-efficiency: `scripts/label_efficiency_probe.py`, `tests/test_label_efficiency_probe.py`
- M4 checkpoint: `train_util.py` (`CheckpointTracker`, `--checkpoint_policy`), `tests/test_checkpoint_policy.py`
- Results: `embeddings/*/probe_results.json`, `embeddings/*/label_efficiency_results.json`
