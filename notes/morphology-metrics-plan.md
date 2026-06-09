# Morphology Metrics — Planning & Implementation

## Purpose

Companion to [`contrastive-learning-plan.md`](contrastive-learning-plan.md).

This document owns **morphology metric selection**, **computation strategy** (including what can be precomputed vs must be batch-local), **how metrics attach to transaction (edge) embeddings**, and a **phased implementation plan** for both:

1. **Morphology-aware contrastive learning** (stronger SSL signal than edge-identity InfoNCE alone).
2. **Expert prediction heads** (Papagei-style auxiliary targets on frozen or joint representations).

The contrastive plan continues to own objective routing, homo/hetero training, extraction, and AML linear probe evaluation.

---

## Project status (Jun 2026)

**Implemented:** M0 · M1 · M1b · M2 · morphology val throttling · **M4 best checkpoint** · **label-efficiency probe** · **M3 Phase 0–1** (BC precompute + expert wiring: `local+tier2`, `local+global+tier2`, `--morph_tier2_lift`).

**Small-HI probe results (GIN, hetero, val-tuned F1, `linear_probe.py`):**

| Run | `unique_name` | Ckpt ep | Val AUROC | Test AUROC | Test F1 | Notes |
|-----|---------------|---------|-----------|------------|---------|-------|
| Contrastive baseline | `hi_contrastive_20ep` | 20 | 0.871 | 0.839 | 0.076 | identity InfoNCE |
| M1 | `hi_morphology_20ep` | 20 | 0.921 | 0.910 | 0.079 | local expert |
| M1b | `hi_morphology_global_20ep` | 20 | 0.914 | **0.920** | **0.108** | **best AUROC — default config** |
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
| Contrastive + proj head | `hi_contrastive_proj_20ep_bestckpt` | 20 | **0.941** | **0.927** | **0.144** | no morph expert; see [`projection-head-ablation-jun2026.md`](projection-head-ablation-jun2026.md) |
| M1b + proj head | `hi_morphology_global_proj_20ep_bestckpt` | **15** | 0.934 | 0.924 | 0.096 | AUROC ↑ vs M1b, F1 ↓; same note |

**Takeaways (Jun 2026):**

- **Contrast projection head (Jun 4):** contrastive-only + proj reaches **0.927** test AUROC / **0.144** F1; M1b+proj **0.924** AUROC but **0.096** F1 (below M1b 0.108). Details: [`projection-head-ablation-jun2026.md`](projection-head-ablation-jun2026.md).
- **M1b expert-only (no M2 contrast)** was the default for **test AUROC** (0.920) and **label-efficiency** (see below); contrastive+proj now leads full-label AUROC/F1 — re-check label-efficiency.
- **M4** helps expert+M2 vs last-epoch saves but does **not** beat M1b on AUROC.
- **Lowering `morph_expert_weight` to 0.5** hurt vs w=1.0 (0.876 vs 0.906 test AUROC).
- **M3 BC stacked on M1b** hurt vs M1b (0.896 best / 0.861 last vs 0.920); **BC-only** (0.904) beats stacked — interference in shared MLP + unified MSE, not “BC is useless.”
- **bc_max** (1 BC col on M1b) still **0.889** — fewer lift cols does not rescue the stack.
- **M5a grouped heads** do **not** fix M3: `w_tier2=1` → 0.887 vs shared stack 0.896; **M5b per-metric deprioritized**.
- **Label-efficiency:** M1b wins at **all** train fractions; largest gap vs contrastive at **10%** labels (+0.078 test AUROC). M2 does **not** flip the ranking under scarcity.

**Next:** label-efficiency on **`hi_contrastive_proj_20ep_bestckpt`** · optional on `bc_only` · Tier 1 **local clustering** if pursuing more morphology signal.

---

## Label-efficiency evaluation (implemented)

**Motivation:** Full-label linear probe (100% train AML labels) favors **M1b** over expert+M2 on test AUROC. In production AML, labeled illicit transactions are scarce — the relevant question (GCPAL, Papagei, RWTH) is: **with only 10–50% of train labels, which frozen encoder separates better?**

**Protocol (no encoder retraining):**

1. Freeze embeddings from a completed pretrain (`embedding_extraction.py`).
2. For each train fraction in `{0.1, 0.25, 0.5, 1.0}`, **stratified subsample** train rows and fit sklearn logistic regression.
3. Tune classification threshold on the **full val** split (same as `linear_probe.py`).
4. Report **val/test AUROC** (primary) and F1 at the val-tuned threshold.

**Implementation:** [`scripts/label_efficiency_probe.py`](../scripts/label_efficiency_probe.py) · batch Slurm: [`run_label_efficiency.sh`](../run_label_efficiency.sh) (`#SBATCH --mem=128G`, `--probe_n_jobs 1` — full `train.npz` is ~3.2M×128 floats; multiprocessing sklearn can OOM).

**Outputs:**

| File | Content |
|------|---------|
| `embeddings/{unique_name}/label_efficiency_results.json` | Per-fraction metrics for one encoder |
| `embeddings/label_efficiency_summary.json` | Combined results for all `--unique_names` |

**Compare:** M1b vs expert+M2 (bestckpt and legacy) vs contrastive baseline — if M2 wins at low fractions, it may still be useful under label scarcity even when full-label AUROC loses to M1b.

### Label-efficiency results (Jun 2026)

**Source:** `embeddings/label_efficiency_summary.json` · six encoders × four train fractions · `--class_weight model` · `--probe_max_iter 5000` · threshold tuned on **full val** (same protocol as full-label probe, but stratified train subsampling per fraction).

**Test AUROC by train label fraction:**

| Encoder | `unique_name` | 10% | 25% | 50% | 100% |
|---------|---------------|-----|-----|-----|------|
| **M1b** (expert `local+global`) | `hi_morphology_global_20ep` | **0.896** | **0.910** | **0.915** | **0.919** |
| Contrastive baseline | `hi_contrastive_20ep` | 0.818 | 0.849 | 0.857 | 0.863 |
| M2 expert + contrast (10 ep, M4 best) | `hi_morph_global_contrast_10ep_bestckpt` | 0.885 | 0.894 | 0.895 | 0.897 |
| M2 expert + contrast (20 ep, M4 best) | `hi_morph_global_contrast_20ep_bestckpt` | 0.880 | 0.897 | 0.898 | 0.896 |
| M2 expert + contrast (10 ep, last) | `hi_morph_global_contrast_10ep` | 0.886 | 0.888 | 0.888 | 0.891 |
| M2 expert + contrast (20 ep, last) | `hi_morph_global_contrast_20ep` | 0.859 | 0.877 | 0.876 | 0.877 |

**M1b minus contrastive (test AUROC Δ):**

| Train % | Δ AUROC | Interpretation |
|---------|---------|----------------|
| 10% | **+0.078** | Morphology expert helps most when labels are scarcest |
| 25% | +0.061 | |
| 50% | +0.058 | |
| 100% | +0.056 | Gap persists at full train subsample |

**M1b minus best M2 (20 ep M4 best ckpt):**

| Train % | M1b | Best M2 | Δ |
|---------|-----|---------|---|
| 10% | 0.896 | 0.880 | +0.016 |
| 25% | 0.910 | 0.897 | +0.013 |
| 50% | 0.915 | 0.898 | +0.017 |
| 100% | 0.919 | 0.896 | +0.023 |

**Conclusions:**

1. **M1b is the label-efficiency winner**, not a full-label-only artifact. The morphology expert (`local+global` degree lift) improves separability at **every** fraction tested.
2. **Scarcity amplifies the morphology story.** The contrastive-vs-M1b gap is **widest at 10%** train labels (+7.8 pp AUROC). That matches the GCPAL / Papagei motivation: evaluate foundation encoders where downstream labels are limited.
3. **M2 does not rescue expert+M2 under label scarcity.** M4-best M2 trails M1b by ~1.3–2.3 pp at all fractions; adding morph contrast does not beat expert-only when the downstream probe is label-limited.
4. **F1 at low fractions is noisy** (few positives in 10% subsample: ~253 train positives). **AUROC is the primary metric** for this analysis; F1 swings with threshold tuning and class imbalance.
5. **vs supervised Multi-GNN (~0.97 AUROC):** compare on **label-efficiency curves**, not frozen linear probe at 100% labels. Supervised CE uses all labels end-to-end; our SSL path is meant to shine when AML labels are partial.

**Not yet probed at partial labels:** `hi_morphology_bc_only_20ep_bestckpt` (0.904 full-label), grouped M5a runs — extend `run_label_efficiency.sh` if needed (CPU/RAM only).

**Reproduce:**

```bash
sbatch run_label_efficiency.sh
# or locally:
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

Production contrastive runs use **large seed batches** (e.g. `--batch_size 32768` in [`smoketest.sh`](../smoketest.sh)) so steps amortize subgraph sampling. Morphology code must stay compatible with that scale.

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

**Legend:** ✅ implemented in code · 🔌 wired into M1 expert loss · ⏳ planned · — not started

| Tier | In M1 expert loss today | Library / plumbing only |
|------|-------------------------|-------------------------|
| **0** | Edge-native + global degree lift (M1b: `local+global`) | — |
| **1** | 8 local degree / ego-scale features | — |
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
| ✅ 🔌 | Subgraph edge count | Edge / local | `n_edges_sub` | Ego scale proxy; log1p in M1 |
| ✅ 🔌 | Subgraph node count | Edge / local | `n_nodes_sub` | Ego scale proxy; log1p in M1 |
| ✅ 🔌 | Sender out/in degree *within subgraph* | Node | `sender_deg_out_local`, `sender_deg_in_local` | **`morph_local`** — differs from global degree |
| ✅ 🔌 | Receiver out/in degree *within subgraph* | Node | `receiver_deg_out_local`, `receiver_deg_in_local` | Same |
| ✅ 🔌 | Out/in degree **sum** on subgraph endpoints | Edge | `deg_sum_out_local`, `deg_sum_in_local` | **`deg_sum_local`** — do not confuse with Tier 0 global sum |
| — | Degree **product** on subgraph endpoints | Edge | — | Not implemented |
| — | Local clustering (sender, receiver, mean) | Node | — | On induced forward subgraph |
| — | 2-hop reachable count from endpoints | Node | — | Cheap BFS cap within batch |
| — | Triangle / wedge counts involving seed | Edge-local | — | Motif-lite |

**Integration:**

- **Expert head (v0):** MLP on `z_seed` → predict `morph_local` + edge-native (MSE or binned CE). **No `morph_global` in v0.**
- **Morphology contrast (v1):** soft positives from **`morph_local`** bins; keep hard positive = same `edge_id` across views.

**Cost:** O(batch subgraph) per step; no full-graph pass; nothing to precompute for all edges up front.

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
| **1** | **M2 morphology-aware contrast** | — | Next axis: structural similarity contrast on local (+ global) bins; expert targets validated. |
| **2** | **Local clustering** (sender, receiver, mean) | 1 | Defer until after M2; adds triadic signal degree misses. |
| **3** | **Time since previous edge** (per endpoint, split-safe) | 0 | Temporal burstiness; cheap offline rolling per split. |
| **4** | **2-hop reachable count** (capped BFS from endpoints) | 1 | Extends ego-scale beyond 1-hop degree. |
| Lower | Degree **product** (local or global) | 0 / 1 | Redundant with sum for many hubs. |
| Lower | Triangle / wedge counts | 1 | Overlaps local clustering. |
| Defer | Tier 2 (BC, PageRank, k-core) | 2 | After M2 benchmarked. |

**Suggested experiment order:** ~~M1b~~ → **M2** → Tier 1 clustering / time-since-prev if M2 plateaus.

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

- [x] Online Tier-1 on view1 subgraph: `compute_local_morphology_torch` + `transform_morph_targets` (+ optional edge-native).
- [x] `MorphologyExpertHead` in [`morphology/expert.py`](../morphology/expert.py).
- [x] `L_morph_expert` (MSE, weighted by `--morph_expert_weight`).
- [x] Wired into `train_hetero_contrastive` and `train_homo_contrastive` via `--morph_expert`.
- [x] Logs `morph/expert_train`, `morph/expert_val` (val loader pass, no AML).
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

Expert target dim with defaults: local **8** + global **9** + edge-native **4** = **21**.

**Deferred to M3:** Tier 2 globals (betweenness, etc.) in `local+global`.

---

### Phase M2 — morphology-aware contrast (v1) ✅

- [x] Binning on selected morphology features (view1 subgraph; train-split quantile edges at startup).
- [x] **Merged soft positives** into edge InfoNCE (`contrastive_loss.py`: identity OR same bin).
- [x] CLI: `--morph_contrast`, `--morph_contrast_features`, `--morph_contrast_scope`, `--morph_contrast_bins`.
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

- [x] Offline approximate betweenness per split (`scripts/precompute_morphology_tier2.py`, `run_precompute_morphology_tier2.sh`).
- [x] Endpoint lift in `morphology/tier2_global.py`; tests `tests/test_morphology_tier2.py`.

**Phase 1 (implemented + benchmarked):** BC in morphology expert head.

- [x] `--morph_targets local+global+tier2` (M1b + BC) and `local+tier2` (BC-only global ablation).
- [x] `--morph_tier2_cache`, `--morph_tier2_lift {full,max}` (`full` = 4 cols; `max` = `bc_max_global` only).
- [x] Slurm: `run_m3_phase1_bc_expert.sh` (extract/probe only), `ablation_morph_bc_only_20ep.sh`, `ablation_morph_bc_max_global_20ep.sh`.

**Small-HI M3 findings (Jun 2026):**

| Run | Test AUROC | Test F1 | vs M1b |
|-----|------------|---------|--------|
| M1b | **0.920** | **0.108** | — |
| M1b + BC (4 cols, M4 best ep 14) | 0.896 | 0.033 | −0.024 AUROC |
| M1b + BC (last ep 20) | 0.861 | 0.029 | −0.059 AUROC |

**Interpretation:** split-global BC **did not improve** the AML linear probe when stacked on global degree lift. Global degrees may already capture most useful hub/broker signal on Small-HI; extra BC dims + shared MSE may add noise. **BC-only** and **bc_max-only** ablations still pending.

- [ ] **Phase 2:** Optional coarse morph contrast on BC quantile bins (defer — M2 local bins underperform M1b).
- [x] Precompute on Slurm (`run_precompute_morphology_tier2.sh`, 128G); login node OOM on graph load.

**Success criterion (Phase 1):** probe gain over M1b — **not met** for stacked BC.

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

**Loss function note:** Papagei uses **MAE** on morphology targets; we use **MSE** today (see below). Either layout can swap loss via `--morph_expert_loss {mse,mae}` (not implemented).

**CLI sketch (future):**

| Flag | Purpose |
|------|---------|
| `--morph_expert_layout {shared,grouped}` | `shared` = current; `grouped` = block heads (**M5a implemented**) |
| `--morph_expert_group_weight_tier2` | Tier 2 block MSE scale when grouped (default `1.0`; `0` = M1b-like sanity) |
| `--morph_expert_metric_weights` | Comma map or file for per-metric weights (per_metric layout) |
| `--morph_expert_loss {mse,mae}` | Expert regression loss (default `mse`; Papagei uses MAE) |

**Implementation notes:**

- Reuse `build_morph_targets()` slice indices (local / global / tier2 / edge-native order unchanged).
- Checkpoint: `morph_expert_state_dict` → dict of block state dicts or prefixed keys.
- **When to build M5a:** ~~after bc_only / bc_max~~ **done** — run `ablation_morph_grouped_wtier2_20ep.sh` (`WTIER2=0|0.5|1`).
- **When to build M5b:** if M5a still shows BC/tier2 hurting easy metrics, or if per-metric logging shows one scalar dominating val loss.
- **Not in scope for v1:** gating networks (full MoE routing), AML-aware checkpoint selection.

**Success criterion:** new layout ≥ shared on M1b AUROC at matched epochs, **or** enables BC/tier2 with isolated weights without hurting degree/local metrics (diagnostic for M3 negative result).

**Expert loss (MSE vs MAE):** Current code uses `F.mse_loss` on **log1p-transformed** count-like targets (`morph_expert_mse_loss`). Papagei uses MAE. Difference is **moderate, not huge** for our setting: both are proper regression losses on comparable-scale log targets. MSE penalizes large errors more (outlier-sensitive); MAE is more robust and may behave better when one metric has heavy tails (BC, amount). Cheap ablation once head layout is stable — unlikely to explain M1b vs M3 gap on its own.

**Contrastive projection head:** We do **not** have a SimCLR/GraphCL-style projector (separate MLP applied only to contrastive loss, discarded at extraction). InfoNCE and morphology expert both operate on the same **`embedding_head` output** `z ∈ R^128` (see `models.py`, `edge_identity_infonce_loss`). Optional future ablation — not required given M1b @ 0.920; may help pure contrastive baseline or expert+M2 more than M1b.

---

### Phase M4 — features & checkpoint policy (optional → **recommended**)

- [ ] Option B: concat Tier-0 to `edge_attr` (ablation vs heads-only).
- [x] **Save best checkpoint** by composite morph val (`morph/expert_val` + `morph/contrast_val`) or `loss/train`; `--checkpoint_policy best` (default for morph runs in `smoketest.sh`).
- [ ] (Future) hetero-safe `gradient_checkpointing` for larger contrastive batches.

**CLI:** `--checkpoint_policy {last,best}` with `--save_model`. **Best** writes the lowest SSL val score to `checkpoint_{unique_name}.tar` (used by extraction) and the final epoch to `checkpoint_{unique_name}_last.tar`. Score = sum of available `morph/expert_val` and `morph/contrast_val` on morph-val epochs; plain contrastive uses `loss/train` every epoch.

**Outcome (Jun 2026):** Last-epoch `hi_morph_global_contrast_20ep` had test AUROC **0.864**; with M4 best (morph val picked epoch 20) → **0.891** AUROC, **0.107** F1. Ten-epoch M4 best (epoch **9**) → **0.906** AUROC. Morph val score still favored later epochs on the 20 ep run — future work: probe-aware selection or early stopping, not only morph val sum.

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
| Supervised ref | CE | ✅ ~0.972 / ~0.493 |

Match: data split, `num_neighs`, embedding dim, extract settings. Morph pretrain: `--checkpoint_policy best` in [`smoketest.sh`](../smoketest.sh).

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
9. Does **local clustering** (Tier 1) help M1 without redundant global stacks?

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
- **M3 Phase 0–1:** `morphology/tier2_global.py`, `scripts/precompute_morphology_tier2.py`, `run_precompute_morphology_tier2.sh`, `run_m3_phase1_bc_expert.sh`, `tests/test_morphology_tier2.py`, `tests/test_morphology_expert_tier2.py`
- **M3 ablations:** `ablation_morph_bc_only_20ep.sh`, `ablation_morph_bc_max_global_20ep.sh`
- **Other ablations:** `ablation_morph_expert_weight_05_10ep.sh`
- **M2 loss:** [`contrastive_loss.py`](../contrastive_loss.py) (`edge_identity_infonce_loss`, bin-grouped soft positives)
- **M2 training glue:** [`morphology/contrastive_train.py`](../morphology/contrastive_train.py)
- Edge readout (all models): `models.py` (`GINe`, `GATe`, PNA, `RGCN` — shared `embedding_head` on concat sender/receiver/edge)
- Hetero contrastive: `training.py` (`train_hetero_contrastive`)
- Seed edges: `train_util.py` (`get_hetero_seed_edge_ids`)
- Probe eval: `linear_probe.py`, `embedding_extraction.py`
- Label-efficiency: `scripts/label_efficiency_probe.py`, `run_label_efficiency.sh`, `tests/test_label_efficiency_probe.py`
- M4 checkpoint: `train_util.py` (`CheckpointTracker`, `--checkpoint_policy`), `tests/test_checkpoint_policy.py`
- Results: `embeddings/*/probe_results.json`, `embeddings/*/label_efficiency_results.json`
