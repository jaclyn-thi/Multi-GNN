# Morphology Metrics — Planning & Implementation

## Purpose

Companion to [`contrastive-learning-plan.md`](contrastive-learning-plan.md).

This document owns **morphology metric selection**, **computation strategy** (including what can be precomputed vs must be batch-local), **how metrics attach to transaction (edge) embeddings**, and a **phased implementation plan** for both:

1. **Morphology-aware contrastive learning** (stronger SSL signal than edge-identity InfoNCE alone).
2. **Expert prediction heads** (Papagei-style auxiliary targets on frozen or joint representations).

The contrastive plan continues to own objective routing, homo/hetero training, extraction, and AML linear probe evaluation.

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
| — | Morphology contrast with all-pairs O(B²) without binning / sampling |

---

## Recommended next metrics (earlier tiers)

**M1b done (Jun 2026):** `hi_morphology_global_20ep` test AUROC **0.920** / F1 **0.108** vs M1 **0.910** / **0.079** — global lift helps; proceed to **M2** before adding more Tier 0/1 metrics.

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

**`L_morph_con`:** pull together `z` for seeds in the same morphology bin; push apart different bins or use as weighted positives in InfoNCE. Start **after** expert-only ablation shows morph targets are stable.

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

### Phase M2 — morphology-aware contrast (v1)

- [ ] Binning / similarity on **`morph_local`** vector (per batch).
- [ ] Soft-positive InfoNCE or supplementary contrast term `L_morph_con` (same seed batch, no AML labels).
- [ ] Ablate: expert only vs expert + morph contrast vs morph contrast only (diagnostic).
- [ ] Tune `λ_exp`, `λ_morph`; avoid O(B²) all-pairs.

**Success:** additive gain over **M1b** on probe AUROC or label-efficiency curve (future).

---

### Phase M3 — Tier 2 global metrics (v2)

- [ ] Offline approximate betweenness on **train split** graph.
- [ ] Endpoint lift `bc(sender), bc(receiver)` added to morph vector.
- [ ] Expert head output dim increased; optional coarse morph contrast on BC quantiles.
- [ ] Report precompute wall time and memory in README.

**Success:** measurable probe gain over M2 at matched compute; document cost/benefit.

---

### Phase M4 — features & checkpoint policy (optional)

- [ ] Option B: concat Tier-0 to `edge_attr` (ablation vs heads-only).
- [ ] Save best checkpoint by `morph/val + contrastive loss`, not AML.
- [ ] (Future) hetero-safe `gradient_checkpointing` for larger contrastive batches.

---

## Experiments to run (when enabled)

| Run | Pretrain | Probe (test AUROC / F1 @ val-tuned) |
|-----|----------|---------------------------------------|
| Baseline | identity InfoNCE only | ✅ `hi_contrastive_20ep` — 0.839 / 0.076 |
| M1 | InfoNCE + morph expert (**local**) | ✅ `hi_morphology_20ep` — 0.910 / 0.079 |
| M1b | M1 + **global** endpoint lift | ✅ `hi_morphology_global_20ep` — **0.920 / 0.108** |
| M2 | InfoNCE + expert + morph contrast | same — use M1b morph target set |
| M3 | M2 + Tier 2 BC in `morph_global` | same |
| Supervised ref | CE 20ep+ | ✅ `hi_supervised_20ep` — ~0.972 / ~0.493 (ceiling) |

Match: data split, `num_neighs`, embedding dim, epochs, extract settings.

**Model sweep (when morphology wins on one arch):** repeat best morph config across `gin`, `gat`, `pna`, `rgcn` with identical morph code and probe protocol.

**Secondary:** label-efficiency (probe on 10/25/50% train labels) once M1b beats M1.

---

## Open questions

1. Which Tier-1 **local** metrics correlate with illicit edges on Small-HI (exploratory EDA, train only)?
2. ~~Does adding **`morph_global`** to the expert head help probe AUROC vs local-only at matched epochs?~~ **Yes (M1b):** test AUROC 0.920 vs 0.910; F1 0.108 vs 0.079.
3. Symmetric vs asymmetric InfoNCE when morphology loss adds memory pressure?
4. Should morph contrast use **same-view** or **cross-view** pairs for soft positives?
5. Homogeneous pipeline first for faster iteration, or hetero-only (production path)?
6. Fix `gradient_checkpointing` + `to_hetero` FX trace before relying on checkpointing for morph runs?
7. Which architecture benefits most from morphology (GIN vs GAT vs PNA vs RGCN)?

---

## Relationship to contrastive plan

| Document | Owns |
|----------|------|
| `contrastive-learning-plan.md` | Objectives, homo/hetero contrastive, extraction, AML probe workflow, no AML in pretrain loop |
| **This document** | Metric tiers, node→edge lifting, batch vs offline compute, morph expert + morph contrast, phased implementation |

Cross-reference: contrastive plan § “Related Future Direction: Morphology Metrics” points here for detail.

---

## References in repo

- **Morphology M0:** `morphology/`, `morphology/IDS.md`, `scripts/precompute_morphology_tier0.py`, `tests/test_morphology_metrics.py`
- Edge readout (all models): `models.py` (`GINe`, `GATe`, PNA, `RGCN` — shared `embedding_head` on concat sender/receiver/edge)
- Hetero contrastive: `training.py` (`train_hetero_contrastive`)
- Seed edges: `train_util.py` (`get_hetero_seed_edge_ids`)
- Probe eval: `linear_probe.py`, `embedding_extraction.py`
- Baseline results: `embeddings/hi_contrastive_20ep/`, `embeddings/hi_morphology_20ep/`, `embeddings/hi_morphology_global_20ep/`, `embeddings/hi_supervised_20ep/`
