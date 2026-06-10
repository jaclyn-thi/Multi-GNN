# Literature review — development index

Companion to [`contrastive-learning-plan.md`](contrastive-learning-plan.md) and [`morphology-metrics-plan.md`](morphology-metrics-plan.md).

**Last updated:** Jun 2026

---

## Papers in `lit review/`

| Local PDF | Paper | Identifier |
|-----------|--------|------------|
| `2306.11586v3.pdf` | Egressy et al. — *Provably Powerful GNNs for Directed Multigraphs* (Multi-GNN) | [arXiv:2306.11586](https://arxiv.org/abs/2306.11586) · AAAI 2024 |
| `s44196-024-00720-4 (5).pdf` | Hanbin et al. — *Graph Contrastive Pre-training for Anti-Money Laundering* (GCPAL) | [Springer IJCIS 2024](https://link.springer.com/article/10.1007/s44196-024-00720-4) |
| `2010.13902v3.pdf` | You et al. — *Graph Contrastive Learning with Augmentations* (GraphCL) | [arXiv:2010.13902](https://arxiv.org/abs/2010.13902) |
| `2410.20542v2.pdf` | Pillai et al. — *PaPaGei: Open Foundation Models for Optical Physiological Signals* | [arXiv:2410.20542](https://arxiv.org/abs/2410.20542) · ICLR 2025 |
| `Scalable_Semi-Supervised_Graph_Learning_Techniques_for_Anti_Money_Laundering (2).pdf` | RWTH — *Scalable Semi-Supervised Graph Learning Techniques for AML* | [RWTH publication 988578](https://publications.rwth-aachen.de/record/988578) |
| `2509.19359v1.pdf` | Gao & Ye — *Anti-Money Laundering Systems Using Deep Learning* | [arXiv:2509.19359](https://arxiv.org/abs/2509.19359) |

---

## How the stack fits together

```text
[Egressy Multi-GNN]     Directed multigraph GNN + edge AML + reverse_mp / ports / ego
        ↓
[GraphCL]               Augmentation design (edge perturbation, attribute masking)
        ↓
[GCPAL]                 AML-specific contrastive pretrain + richer positives + label scarcity
        ↓
[Papagei]               Morphology experts + morphology contrast + frozen linear probe
        ↓
[This repo M1/M1b/M2]   Transaction-level morphology on stochastic subgraphs
        ↓
[RWTH / Gao & Ye]       Contrasts & future: node-level supervised AML, Tier-2 centrality
```

**Primary research framing (see contrastive plan):** graph foundation model pretrain on transaction graphs → frozen edge embeddings → linear probe on AML (and later other finance tasks). Supervised Multi-GNN remains a **ceiling baseline**, not the GFM target.

---

## Implementation status (repo vs papers)

| Idea | Source paper(s) | In repo? | Where / notes |
|------|-----------------|--------|----------------|
| Hetero directed multigraph + edge readout | Egressy | ✅ | `models.py`, `--reverse_mp --ego --ports` |
| Temporal train/val/test splits | Egressy / IBM AML | ✅ | `data_loading.py` |
| Supervised end-to-end AML (CE + F1) | Egressy | ✅ | `--objective supervised` |
| Two stochastic views per batch | GraphCL, GCPAL | ✅ | `graph_augmentations.py` |
| Edge drop + edge attribute mask | GraphCL | ✅ | `edge_drop_rate=0.1`, attr mask in `generate_views` |
| Identity positives via stable `edge_id` | GCPAL (concept) | ✅ | `contrastive_loss.py`, `train_util.py` |
| Edge-level InfoNCE (seed edges only) | This repo | ✅ | `contrastive_loss.py` |
| Asymmetric InfoNCE, neg subsample, memory bank | This repo | ✅ | CLI in `util.py` |
| Frozen extract → linear probe, AUROC primary | Papagei | ✅ | `embedding_extraction.py`, `linear_probe.py` |
| No AML checkpoint selection during SSL | Papagei / contrastive plan | ✅ | Explicit design choice |
| Morphology **expert** heads (MSE) | Papagei (analogue) | ✅ | M1/M1b — `morphology/expert.py` |
| Morphology-aware **contrast** (soft positives) | Papagei + GCPAL (analogue) | ✅ | M2 — `morphology/contrast.py`, `contrastive_loss.py` |
| Tier-0 global degree precompute | This repo (M1b) | ✅ | `scripts/precompute_morphology_tier0.py` |
| Morph val throttling | This repo | ✅ | `--morph_val_every`, `--morph_val_max_batches` |
| Third **KNN view** | GCPAL | ❌ | Documented divergence in contrastive plan |
| **Projection head** for contrast only | GraphCL, GCPAL | ✅ | `--contrast_projection_head` — InfoNCE only; extract uses encoder `z`; see [`projection-head-ablation-jun2026.md`](projection-head-ablation-jun2026.md) |
| **Neighbor / KNN soft positives** (non-morph) | GCPAL | ❌ | M2 uses morphology bins instead |
| **Label-efficiency** probe (10/25/50% train labels) | GCPAL, Papagei, RWTH | ✅ | `scripts/label_efficiency_probe.py` |
| **Disjoint** morph dims for contrast vs expert | Papagei | ❌ | Default overlap `local_ego,local_degree` |
| **MoE / per-metric expert heads** | Papagei | ⚠️ | Shared MLP default; **M5a grouped** implemented (0.887 AUROC — did not fix M3); **M5b per-metric** deprioritized |
| **Best checkpoint** by SSL + morph val | Papagei spirit / M4 plan | ✅ | `--checkpoint_policy best` |
| Tier-2 **BC** lift (expert) | Gao & Ye, M3 plan | ✅ | `local+global+tier2`, `local+tier2`; **no probe gain vs M1b** (Jun 2026) |
| Tier-1 **local clustering** (expert + M2 group) | This repo (M1c) | ✅ | Expert-only 0.903; **+projection 0.929** (best SSL); LE pending; M2 bins optional |
| Tier-2 PageRank | Gao & Ye, M3 plan | ❌ | BC only so far |
| **Node-level** semi-supervised AML | RWTH | ❌ | Different task; edge-centric here |
| Pipeline: Node2Vec → XGBoost on nodes | RWTH | ⚠️ | Analogue: extract → `linear_probe.py` (edge-level) |

---

## Small-HI probe snapshot (Jun 2026)

GIN, hetero, full-train linear probe, val max-F1 threshold. Full table in [`morphology-metrics-plan.md`](morphology-metrics-plan.md#project-status-jun-2026).

| Config | Ckpt | Test AUROC | Test F1 |
|--------|------|------------|---------|
| **M1b + clustering + projection** | 20 | **0.929** | **0.156** |
| Contrastive + projection | 20 | 0.927 | 0.144 |
| M1b + projection | 20 → ep 15 | 0.924 | 0.096 |
| **M1b expert** (8 local dims) | 20 | 0.920 | 0.108 |
| M1b + clustering (MSE) | 20 | 0.903 | 0.117 |
| M1b + MAE expert | 20 | 0.898 | 0.145 |
| M3 BC-only | 20 | 0.904 | 0.093 |
| Expert + M2 | 10ep best (ep 9) | 0.906 | 0.058 |
| Expert + M2 | 20ep best (ep 20) | 0.891 | 0.107 |
| M3 M1b + BC (4 cols) | 20ep best (ep 14) | 0.896 | 0.033 |
| M5a grouped BC | `w_tier2=1` | 0.887 | 0.028 |
| Expert + M2 | `w=0.5`, 10ep best | 0.876 | 0.027 |
| Expert + M2 last @ 20ep (pre-M4) | 20 | 0.864 | 0.025 |
| Contrastive baseline | 20 | 0.839 | 0.076 |
| M2 only | 10 | 0.680 | 0.012 |
| Supervised CE (in-GNN) | — | ~0.972 | ~0.493 |

**Takeaways:**

- **Projection head** is the largest SSL win on full labels: contrastive 0.839 → **0.927** test AUROC (+0.088); morph expert not required for that gain.
- Expert heads carry representation; M2 contrast alone fails.
- **M4** (`--checkpoint_policy best`) recovers much of the 20 ep expert+M2 regression; **M1b** remains best **morph-only** config (0.920 AUROC).
- **M3 BC** stacked on M1b **hurt** (0.896 best vs 0.920 M1b); **M5a grouped** did not fix interference (0.887).
- **`morph_expert_weight=0.5`** hurt expert+M2 vs w=1.0.
- **Full-label SSL leader:** M1b+clustering+projection **0.929** AUROC (clustering alone 0.903; projection rescues + surpasses contrastive+proj).
- **Label-efficiency** (ten encoders): contrastive+proj @ 25–100%; M1b+proj @ 10%. Clustering+proj LE **pending**.
- **MAE expert loss:** 0.898 AUROC — no win vs MSE (0.903).

---

## Per-paper notes

### 1. Egressy et al. — Multi-GNN (2306.11586)

**Methods:** Reverse message passing, multigraph **port numbering**, **ego IDs** → provably detect directed subgraph patterns (cycles, scatter-gather, fan-in/out) on transaction multigraphs.

**Findings (AML):** Multi-GIN+EU ~**65% minority F1** supervised on Small-HI vs ~29% plain GIN; reverse MP + ports are the main lift; ego IDs help complex synthetic patterns more than aggregate AML F1.

**Repo connection:**
- Foundation for graph form and edge classification.
- Supervised path ≈ original Multi-GNN benchmark (~0.97 probe AUROC in our runs).
- Pattern-level recalls (high on named patterns, ~0% on “lone illicit”) explain low F1 despite decent AUROC — see Egressy appendix pattern breakdown.

**Do not copy for GFM path:** supervised CE + F1 as pretrain objective or checkpoint selection.

**Useful later:** Subgraph pattern vocabulary when choosing Tier-1 metrics or explaining probe F1 vs AUROC gap.

---

### 2. Hanbin et al. — GCPAL

**Methods:** Three views (two perturbed + **KNN** from node feature similarity); InfoNCE with **projection head**; positives include **connected neighbors** and **feature-similar** pairs; strong when **downstream labels are scarce**.

**Repo connection:**
- Same high-level goal: SSL on AML graphs, then downstream with limited labels.
- We implement **two** perturbed views only (`notes/contrastive-learning-plan.md` Phase 2b).
- **M2 soft positives** are the closest analogue to “similar-feature positives,” but on **edge morphology bins** (cross-view), not KNN in node feature space.

**High-value future work:**
1. ~~**Label-efficiency probes**~~ — nine-encoder batch done: contrastive+proj leads vs M1b at all fractions; M1b+proj best @ 10% (0.918 AUROC).
2. **KNN or similarity view** — mitigate link sparsity (GCPAL motivation); engineering cost non-trivial on edge/batch sampling path.
3. ~~**Contrast projection head**~~ — done (Jun 4): contrastive+proj **0.927** test AUROC; M1b+proj 0.924 AUROC / 0.096 F1. See [`projection-head-ablation-jun2026.md`](projection-head-ablation-jun2026.md).

---

### 3. You et al. — GraphCL (2010.13902)

**Methods:** Node drop, **edge perturbation**, **attribute masking**, subgraph sampling; NT-Xent; **projection head** after GNN; systematic study of augmentation pairs by graph domain.

**Findings:** Without augmentation, graph CL can **hurt** vs training from scratch. Edge perturbation helps social graphs but can hurt domains where **edges are semantically binding** (e.g. molecular bonds).

**Repo connection:**
- `graph_augmentations.py`: independent **edge drop** + **edge attr mask** per view; positives aligned by `edge_id` (robust to different edge sets per view).
- **Projection head** implemented (`--contrast_projection_head`) — GraphCL-style; applied before InfoNCE only; encoder `z` at extraction.
- No subgraph/node-drop views; no third KNN view.

**Caution for AML:** Transactions may be “edge-sensitive” like molecules — monitor whether aggressive `edge_drop_rate` (0.1) hurts morph-aware runs; M2 bins use **view1** features only (good).

**Future:** Third view (subgraph or GCPAL KNN); compose **different** augmentation types (GraphCL recommends off-diagonal pairs).

---

### 4. Pillai et al. — PaPaGei (2410.20542)

**Methods:** Large-scale SSL on PPG; split morphology roles:
- **sVRI → contrastive** alignment,
- **IPA, SQI → mixture-of-expert prediction** from encoder embeddings,
- Evaluation: **frozen encoder + linear probe** (logistic / ridge; **AUROC** for classification).

**Repo connection (strongest paper alignment):**

| Papagei | This repo |
|---------|-----------|
| Morphology contrast | **M2** (`--morph_contrast`) |
| Expert heads | **M1 / M1b** (`--morph_expert`) |
| Frozen extract + probe | `embedding_extraction.py` → `linear_probe.py` |
| No downstream metric during SSL | Contrastive + morph val only |

**Empirical validation:** expert+M2 AUROC 0.90 @ 10 ep; contrast-only 0.68 @ 10 ep — matches “experts + contrast” design. **@ 20 ep expert+M2 regressed to 0.864** while M1b expert-only stays 0.920 — M2 needs retuning (weight, early stop, checkpoint), not more epochs as-is.

**Still borrowable:**
- **Partition metrics:** contrast features ≠ expert targets (Papagei uses disjoint sets).
- **MoE experts** if single MLP saturates.
- **Checkpoint policy** by SSL + morph val (M4 in morphology plan).
- **Augmentation discipline** — avoid destroying morphology in views used for targets.

**Explicit non-goals:** PPG architecture, subject-level splits, biosignal morphology definitions — we only borrowed the **protocol and split of contrast vs expert roles**.

---

### 5. RWTH — Scalable semi-supervised graph learning for AML

**Methods:** Compare **pipeline** (Node2Vec / GraphSAGE / Attri2Vec → RF/XGB/LGBM) vs **end-to-end** (SkipGCN, FastGCN, EvolveGCN) on transaction graphs; **node-level** suspiciousness; pattern enumeration (gather-scatter, cycles).

**Repo connection:**
- **Different task:** node classification vs our **edge** embeddings + probe.
- **Analogue:** embed → sklearn classifier ≈ our extract + `linear_probe.py` (but on edges).
- **Cost lesson:** Full pattern enumeration each run is expensive; our Tier-0 cache + in-batch Tier-1 morph avoids that hot path.

**Useful for development:**
- Label-efficiency and scalability arguments.
- Do **not** pivot primary task to node-level without strong reason — labels and Multi-GNN readout are edge-centric.

---

### 6. Gao & Ye — GCN + centrality for AML (2509.19359)

**Methods:** Account-level graph; GCN + classical **centrality** (degree, closeness, betweenness, PageRank) for link / anomaly analysis.

**Repo connection:**
- **M3 implemented (Jun 2026):** offline BC per split (`scripts/precompute_morphology_tier2.py`), endpoint lift in expert via `--morph_targets local+global+tier2` or `local+tier2`, `--morph_tier2_lift {full,max}`.
- **M1b already implements Tier-0 degree lift** (9 global features per seed).
- **Finding:** stacked BC did **not** improve AML linear probe vs M1b on Small-HI (0.896 vs 0.920 test AUROC). Global degrees may subsume most hub signal; shared MLP + unified MSE may dilute BC gradients — see **M5 grouped heads** in morphology plan.

**Future:** PageRank Tier 2; BC contrast bins (Phase 2, deferred). ~~Grouped expert heads~~ — M5a done (0.887 AUROC; did not fix stack). See [`morphology-metrics-plan.md`](morphology-metrics-plan.md) Phase M3 / M5.

---

## Cross-paper design decisions (already in repo)

| Question | Papers | Our choice |
|----------|--------|------------|
| Edge vs node AML task | Egressy vs RWTH | **Edge** (transaction embedding `z`) |
| Pretrain objective | GCPAL, GraphCL | **InfoNCE** + optional morph losses |
| Downstream eval | Papagei, GCPAL | **Frozen encoder + linear probe**; AUROC primary |
| Checkpoint selection | Papagei | **Not** AML val during SSL; **`--checkpoint_policy best`** by morph val composite (M4) |
| Augmentations | GraphCL, GCPAL | Edge drop + attr mask; **no KNN view** |
| Similarity positives | GCPAL, Papagei | **M2 morphology bins** (not KNN) |
| Global structure signal | Gao & Ye, M1b | Tier-0 **degree lift** ✅; Tier-2 **BC lift** ✅ but **no probe gain** vs M1b |
| Expert head layout | Papagei MoE | **Single shared MLP** default; M5a grouped implemented (0.887 AUROC — insufficient); M5b per-metric deprioritized |
| Expert loss | Papagei MAE | **MSE** default; MAE ablation **0.898** vs MSE **0.903** — keep MSE |
| Contrastive projection head | GraphCL, GCPAL | ✅ `--contrast_projection_head` (InfoNCE only; extract uses encoder z) |
| Stochastic subgraphs | GraphCL, Papagei precompute | **In-batch Tier-1** local morph; Tier-0 precompute |

---

## Prioritized development backlog (from literature)

Ordered by impact vs effort for **this** codebase (updated after M2 @ 20 ep):

| Priority | Item | Primary paper | Effort |
|----------|------|---------------|--------|
| 1 | ~~**M4:** save best checkpoint by morph val composite~~ | Papagei | ✅ `--checkpoint_policy best` |
| 2 | ~~Re-run expert+M2 with `--checkpoint_policy best`~~ | This repo | Done — see morphology plan table |
| 3 | ~~**Analyze label-efficiency** (`run_label_efficiency.sh`)~~ | GCPAL, Papagei, RWTH | ✅ nine encoders: contrastive+proj leads; M1b+proj @ 10% |
| 4 | **Tier-1 local clustering** benchmark | This repo | ✅ clustering+proj **0.929**; LE on clustering+proj pending |
| 5 | **Disjoint morph features** (contrast vs expert ablation) | Papagei | Low — CLI/flags |
| 6 | ~~**M3 BC ablations** (`bc_only`, `bc_max`)~~ | Gao & Ye | ✅ done; stack hurts, bc_only 0.904 |
| 7 | ~~**M5 grouped expert heads**~~ | Papagei | ✅ done; 0.887 — **M5b per-metric deprioritized** |
| 8 | ~~**MAE vs MSE** expert loss ablation~~ | Papagei | ✅ MAE 0.898 — no AUROC win vs MSE 0.903 |
| 9 | ~~**Contrast projection head** ablation~~ | GraphCL, GCPAL | ✅ contrastive+proj **0.927** AUROC; M1b+proj 0.924 / 0.096 F1 — see projection ablation note |
| 10 | **KNN view** or neighbor soft positives | GCPAL | High |
| 11 | Tier-2 **PageRank** lift | Gao & Ye | High (precompute) |
| ~~10~~ | ~~**M3 BC** in expert head~~ | Gao & Ye | ✅ implemented; **no AUROC gain** vs M1b |

~~Complete M2 @ 20 ep vs M1b~~ — **done.** **Full-label SSL leader:** M1b+clustering+projection **0.929** AUROC (Jun 2026).

---

## Related work mentioned in papers but not in `lit review/`

Worth knowing for context; not indexed as local PDFs:

- **LaundroGraph** (Cardoso et al. 2022) — self-supervised AML on bipartite account–transaction graph; cited by Egressy.
- **IBM AML dataset** (Altman et al. NeurIPS 2023) — our Small-HI source; see README data section.
- **Weber et al.** — early GCN + graph features for Bitcoin AML; tree-based + GF baselines in Egressy.

Add PDFs to `lit review/` and extend this table if those become relevant.

---

## Quick links in repo

| Topic | Document / code |
|-------|-----------------|
| Contrastive workflow & Phase history | [`contrastive-learning-plan.md`](contrastive-learning-plan.md) |
| Morphology phases M0–M5 | [`morphology-metrics-plan.md`](morphology-metrics-plan.md) |
| Projection head ablation (Jun 2026) | [`projection-head-ablation-jun2026.md`](projection-head-ablation-jun2026.md) |
| CLI & benchmarks | [`README.md`](../README.md) |
| Augmentations | [`graph_augmentations.py`](../graph_augmentations.py) |
| InfoNCE + M2 soft positives | [`contrastive_loss.py`](../contrastive_loss.py) |
| Morphology training glue | [`morphology/contrastive_train.py`](../morphology/contrastive_train.py) |
| Label-efficiency probe | [`scripts/label_efficiency_probe.py`](../scripts/label_efficiency_probe.py) |
