# Morphology feature inventory

Audit of all morphology-related features, caches, and usage paths in Multi-GNN (Jul 2026). Two parallel systems exist:

1. **SSL training-time** — expert MSE targets + optional M2 morphology-bin soft positives (`morphology/`).
2. **Probe-time downstream** — label-free engineered columns stacked with frozen embeddings (`scripts/probe_feature_ablation.py`).

Reference: `notes/morphology-reference.md`, `notes/morphology-metrics-plan.md`.

---

## Summary table

| name / group | level | local/global | time | downstream probe | SSL expert | SSL contrast bin | normalization | cache | leakage risk | prior result | recommendation |
|--------------|-------|--------------|------|------------------|------------|------------------|---------------|-------|--------------|--------------|----------------|
| **deg_in / deg_out / deg_total (Tier 0)** | node | global (split) | static | via `degree_fan` (train-static) | yes (`degree_fan`) | `global_degree` | log1p @ expert | `{split}_node_morphology.csv` | low if split-local | baseline structural signal | **keep** — core probe group |
| **sender/receiver local degrees (Tier 1)** | edge (batch subgraph) | local | static within batch | partial overlap w/ degree_fan | yes | `local_degree` | log1p @ expert | none (on-the-fly) | low | batch-size dependent | expert/contrast only; not in probe ablation |
| **fan-in / fan-out (probe `degree_fan`)** | edge | global **train-static** | static | **yes** (8 cols) | mapped to `degree_fan` | — | log1p | `results/cache/...` | **train-fit only** — OK | helps full stack w/ FNF | **keep** in `embedding+raw+morph` |
| **triangles (Tier 1)** | edge | local (batch) | static | no direct probe cols | yes (`motif_participation`) | `local_triangles` | log1p @ expert | none | low | motif scout 0.190 F1 morph-only | **expert/ablation** only; redundant w/ clustering locally |
| **clustering coeff (Tier 1)** | edge | local (batch) | static | no direct probe cols | yes (`local_density`) | `local_clustering` | raw @ expert | none | low | mixed SSL scouts | same as triangles |
| **betweenness (Tier 2 BC)** | node | global (split) | static | no | yes (`global_role`) | — | raw / max lift | `{split}_node_tier2.csv` | split-local | BC stacked **hurts** vs degree-only | **defer** downstream unless cheap |
| **flow balance (Tier 0 SSL)** | edge lift | global (split) | static cumulative | via `flow_balance` (6 cols, train-static) | yes (10 cols) | — | log1p + ratio clip | `{split}_node_flow_balance.csv` (often missing) | split-local | probe flow helps some stacks | **keep** probe group; optional SSL |
| **flow balance (probe)** | edge | global train-static | static | **yes** (6 cols) | subset of SSL 10 | — | log1p + ratio | probe cache | train-fit aggregates | complements degree | **keep** |
| **volume / amount (edge-native)** | edge | — | per-edge | **yes** (`raw`) | yes (`volume_activity`) | in `edge_native` bins | log1p amount in raw | — | none | raw alone weak | use in `+raw`, not alone |
| **Timestamp (edge-native)** | edge | — | per-edge | **yes** (`raw` + `temporal_behavior`) | yes (`temporal_behavior`) | in `edge_native` | norm / ordinal | — | global sort interarrival leaks future order | weak alone | **replace/enrich** — see temporal plan |
| **temporal_behavior (probe)** | edge | global sort | **global** interarrival | **yes** (2 cols) | partial | — | min-max ts; log1p Δt | probe cache | **uses full-dataset sort** — mild leakage vs strict causal | small lift vs raw | **supersede** with causal past-only features |
| **currency / payment format** | edge | — | static codes | **yes** (`raw`) | yes (`other`) | categoricals binned | ordinal / one-hot | — | none | needed in +raw | **keep** in raw stack |
| **morphology-bin (M2 contrast)** | edge | local+global bins | static | no | no | **yes** (soft positives) | quantile bins | — | calib on train | **negative** SSL (0.012–0.058 F1) | **do not extend** without new design |
| **motif_participation group** | edge | local | static | no | yes (triangles) | optional | log1p | — | low | best morph **scout** 0.190 | expert target only, not typology labels |
| **KNN `temporal_causal` / `flow_rich`** | edge | account/pair causal | **past-only** | **not wired to probe** | no | no | log1p | `transaction_knn_*.npz` | causal if past-only | KNN SSL **failed** | **reuse definitions** for new probe group |
| **ego n_edges / n_nodes (Tier 1)** | edge | local batch | static | no | yes (`local_context_size`) | `local_ego` | log1p | — | batch subsampling | context size signal | contrast/expert only |

---

## Detailed entries

### Degree / fan (Tier 0 + probe `degree_fan`)

| Field | Value |
|-------|-------|
| **Definition** | Tier 0: `deg_in[v]`, `deg_out[v]`, `deg_total[v]` on split graph. Probe: log1p of sender/receiver in/out/total/sum degrees from **train-split** edge list only. |
| **Code** | `morphology/tier0_global.py`, `probe_feature_ablation.py::degree_fan_features_train_static` |
| **Normalization** | log1p on counts; StandardScaler on probe groups (train-fit) |
| **Cache** | `morphology_cache/{dataset}/{split}_node_morphology.csv` |
| **Prior** | Small-HI: essential component of `embedding+raw+morph` (FNF s1 F1 0.319); alone insufficient |
| **Recommendation** | Retain; do not duplicate in new temporal group |

### Triangles / clustering (Tier 1 local)

| Field | Value |
|-------|-------|
| **Definition** | On batch subgraph: local clustering coefficient; triangle counts at sender/receiver/mean |
| **Code** | `morphology/tier1_local.py` cols 8–13 |
| **Local vs global** | **Local** — depends on neighbor sampling / batch |
| **Prior** | SSL scouts: clustering+proj 0.929 AUROC; probe stack adds morph 16-d including degree only |
| **Recommendation** | Already covered indirectly via degree_fan; low priority for new downstream cols |

### Betweenness (Tier 2)

| Field | Value |
|-------|-------|
| **Definition** | Brandes BC on split graph; endpoint lift (4 or 1 dim) |
| **Code** | `morphology/tier2_global.py` |
| **Prior** | Adding BC to degree expert **hurt** in morphology-metrics-plan |
| **Recommendation** | Skip for next downstream scout |

### Flow balance

| Field | Value |
|-------|-------|
| **Definition** | SSL: split-global amount in/out per node + ratios to edge amount. Probe: train-static 6-d subset (no edge-to-node ratio logs). |
| **Code** | `tier0_flow_balance.py`, `flow_balance_features_train_static` |
| **Redundancy** | Overlaps with raw amount + degree-weighted volume |
| **Prior** | Probe `flow_balance` in MORPH_GROUPS; helps some LI/H stacks marginally |
| **Recommendation** | Keep existing; new temporal/flow features should use **causal windows**, not replace this |

### Timestamp / temporal_behavior

| Field | Value |
|-------|-------|
| **Definition** | `timestamp_norm = (t - min) / (max - min)` on **full dataframe**; `log1p_interarrival` from **global** sort order |
| **Leakage** | Global sort uses future transactions when scoring past edges — mild for stationary splits but not strictly causal |
| **Code** | `temporal_behavior_features()` |
| **Recommendation** | **Replace** with past-only interarrival (see `transaction_knn/richer_features.py::compute_causal_edge_stats`) in proposed `temporal_flow_causal` group |

### Edge-native raw (amount, currency, format, timestamp)

| Field | Value |
|-------|-------|
| **Definition** | AML edge columns; amount log1p; categoricals ordinal-fit on train |
| **Prior** | `embedding+raw` strong on HI 40ep s2; essential for LI pre-3h gains |
| **Recommendation** | Always include in ablation baselines |

### Morphology-bin contrast (M2)

| Field | Value |
|-------|-------|
| **Definition** | Quantile bins on morphology columns; same-bin cross-view soft positives |
| **Not** | Typology/motif-label positives |
| **Prior** | Closed negative — weak vs baseline contrastive |
| **Recommendation** | Do not expand; see positive-pair audit for alternative |

### KNN richer causal features (unused in probe)

| Field | Value |
|-------|-------|
| **Definition** | Past-only sender/receiver Δt, counts, amounts, pair history — O(n log n) single pass |
| **Code** | `transaction_knn/richer_features.py` |
| **Recommendation** | **Primary source** for proposed downstream temporal/flow group (subset, not full 20+ cols) |

---

## Forgotten / inconsistent usage

| Issue | Detail |
|-------|--------|
| **Two flow implementations** | SSL 10-d split-global vs probe 6-d train-static — document when comparing |
| **Tier 0 flow cache missing on disk** | Falls back to on-the-fly; probe path never used SSL cache |
| **temporal_behavior global sort** | Inconsistent with causal KNN features — should align |
| **Tier 1 local features not in probe ablation** | triangles/clustering only via SSL; probe `morph` = degree_fan + flow + temporal only |
| **`morph` probe group name** | Misleading — does not include triangles/clustering/betweenness |
| **Pre-3h + morph** | Under-tested vs `+raw`; thesis tables often omit morph on LI multiseed |

---

## Files index

| Path | Role |
|------|------|
| `morphology/tier0_global.py` | Global degrees |
| `morphology/tier0_flow_balance.py` | Global amount flow |
| `morphology/tier1_local.py` | Local degree/cluster/triangle |
| `morphology/tier2_global.py` | Betweenness |
| `morphology/expert.py` | Expert head assembly |
| `morphology/contrast.py` | M2 binning |
| `scripts/probe_feature_ablation.py` | Downstream groups |
| `transaction_knn/richer_features.py` | Causal temporal/flow (KNN only) |
| `morphology_cache/Small-HI/` | Precomputed CSV caches |
| `results/cache/probe_features_*/` | Probe feature matrices |
