# Morphology reference

Quick lookup for morphology metrics, flags, and training concepts. Phased roadmap (M0–M5) and implementation detail: [`morphology-metrics-plan.md`](morphology-metrics-plan.md).

---

## Phase summary

| Phase | Flag | What it adds |
|-------|------|--------------|
| **M1** | `--morph_expert --morph_targets local` | Tier 1 local subgraph stats (11 dims) + edge-native |
| **M1b** | `--morph_targets local+global` | Tier 0 global endpoint degrees (split-safe lift) |
| **M1b + BC** | `--morph_targets local+global+tier2` | M3: M1b + betweenness centrality lift |
| **BC-only** | `--morph_targets local+tier2` | Tier 1 + BC lift only |
| **M2** | `--morph_contrast` | Soft positives in InfoNCE from same morphology bin |
| **Projection** | `--contrast_projection_head` | MLP before InfoNCE only; encoder `z` at extract |

---

## Metrics reference

Label-free structural features on each **seed transaction**. Uses:

- **Expert head (M1+):** MLP predicts detached targets from encoder `z_seed` (MSE/MAE).
- **Morph contrast (M2):** quantile **bins** → soft positives in InfoNCE (plus same-`edge_id` positives).

**Target assembly:** `local` → `global` (M1b) → `tier2` (M3) → `edge_native`. Computed on **view1**. Train targets use the **train-split** graph; val morph loss uses **val-split** only.

| `--morph_targets` | Blocks included | Default expert dims |
|-------------------|-----------------|---------------------|
| `local` (M1) | Tier 1 + edge-native | **18** (14 + 4) |
| `local+global` (M1b) | Tier 1 + Tier 0 lift + edge-native | **27** (14 + 9 + 4) |
| `local+global` + `--morph_flow_balance` | above + flow-balance lift | **37** (14 + 9 + 10 + 4) |
| `local+tier2` | Tier 1 + BC lift + edge-native | **22** (14 + 4 + 4) |
| `local+global+tier2` (M3) | all blocks | **31** (14 + 9 + 4 + 4) |
| `local+global+tier2` + `--morph_flow_balance` | all blocks + flow | **41** (14 + 9 + 10 + 4 + 4) |

**`log1p` rule:** count-like columns (ego, degrees, global lift, BC) get `log1p` before expert loss. Clustering coefficients stay in **[0, 1]**. Edge-native attributes used as-is.

### Diagnostic target groups

The shared expert head and total expert loss are unchanged by default. During
expert training, scalar targets are also grouped for diagnostic MSE logging.
At expert setup, runs also log per-group target counts and member names.

| Group | Scope | Typical targets |
|-------|-------|-----------------|
| `degree_fan` | Tier 1 local + Tier 0 global endpoint lift | sender/receiver in/out/total degree, degree sums |
| `flow_balance` | Tier 0 global precomputed | split-global amount in/out totals, balance ratios, edge-relative flow |
| `volume_activity` | edge-native | `Amount Received`, `Amount Sent` |
| `temporal_behavior` | edge-native | `Timestamp`, burst/interarrival/recency features |
| `motif_participation` | Tier 1 local sampled | triangle counts, wedge/cycle/motif participation |
| `local_density` | Tier 1 local sampled | local clustering coefficient, ego/neighborhood density |
| `local_context_size` | Tier 1 local sampled | `n_edges_sub`, `n_nodes_sub`, ego edge/node counts |
| `global_role` | Tier 2 global precomputed | betweenness centrality endpoint lift (`sender_bc`, `bc_max_global`, …) |
| `other` | edge-native / unrecognized | categorical codes (`Received Currency`, `Payment Format`) |

**Legacy CLI aliases** (expand to the groups above, backward compatible):

| Alias | Expands to |
|-------|------------|
| `local_motif` | `motif_participation`, `local_density`, `local_context_size` |
| `centrality` | `global_role` |
| `temporal` | `temporal_behavior` |

Logged keys include `morphology/loss_total`,
`morphology/loss_group/{group}`, and
`morphology/loss_target/{target}`. These are diagnostics only; future
experiments may add separate expert heads or group-weighted losses.

Targeted expert diagnostics can use `--morph_target_groups` to keep only selected
semantic groups with the same shared expert head. Examples:
`--morph_target_groups degree_fan`,
`--morph_target_groups motif_participation,local_density`, or
`--morph_target_groups degree_fan,local_motif` (legacy alias). The default
`all` preserves the full historical target vector.

**Targeted group scouts (Small-HI):** full tables in
[`results-archive.md` § Morphology target-group scouts](results-archive.md#morphology-target-group-scouts-small-hi).
Best morphology **F1:** `degree_fan` @ 10 ep, w=1.0 (0.208). Best morphology
**AUROC** in targeted scouts: `degree_fan` 20 ep w=0.05 **last epoch** (0.943)
or `motif_participation` @ w=0.05 (0.937). Prefer **`motif_participation` /
`flow_balance` @ w=0.05**. **Motif @ w=1.0:** morph-val best (ep 3) fails;
last epoch (0.925 / 0.128) is better but still below w=0.05. **`--checkpoint_policy best`**
on morph val often mis-ranks downstream quality — probe last epoch
(`slurm/run_morph_lastckpt_extract_probe.sh`) before trusting morph-val best.
All targeted morphology runs trail no-morph baseline (0.951 / 0.233).

### Tier 0 — global (9 cols; M1b+)

Split-global node degrees per train/val/test; **endpoint lift** to each seed edge.

| Col block | Name | M2 group |
|-----------|------|----------|
| per endpoint | `sender_deg_in/out/total`, `receiver_deg_in/out/total` | `global_degree` |
| edge sums | `deg_sum_out/in/total_global` | `global_degree` |

Precompute: `scripts/precompute_morphology_tier0.py` → `morphology_cache/{data}/{split}_node_morphology.csv`.

### Tier 0 flow balance (10 cols; opt-in via `--morph_flow_balance`)

Split-global **Amount Received** aggregates per account on the forward split graph,
lifted to seed edges. Label-free; train targets use the train split only.

| Target | Description |
|--------|-------------|
| `sender_in_amount_log`, `sender_out_amount_log` | log1p total received/sent by sender |
| `receiver_in_amount_log`, `receiver_out_amount_log` | log1p total received/sent by receiver |
| `sender_flow_balance_ratio`, `receiver_flow_balance_ratio` | `(out - in) / (out + in + eps)`, clipped to [-1, 1] |
| `sender_abs_flow_imbalance_log`, `receiver_abs_flow_imbalance_log` | log1p \|out - in\| |
| `edge_to_sender_out_ratio_log` | log1p edge amount / (sender out total + eps) |
| `edge_to_receiver_in_ratio_log` | log1p edge amount / (receiver in total + eps) |

Precompute: `scripts/precompute_morphology_tier0_flow.py` → `{split}_node_flow_balance.csv`.
If cache files are absent, tables are computed from split graphs at startup (same as Tier 0 degrees).

**Default behavior:** `--morph_flow_balance` is **off**. Existing runs unchanged unless explicitly enabled.

### Edge-native (4 cols)

Forward `edge_attr` per seed (excl. `EdgeID`): timestamp, amount received, received currency, payment format. Disable: `--no_morph_edge_native`. M2 group: `edge_native`.

### Tier 1 — local (14 cols)

Stats on the **batch subgraph** from `LinkNeighborLoader`.

| Col | Name | M2 group | log1p? |
|-----|------|----------|--------|
| 0–1 | `n_edges_sub`, `n_nodes_sub` | `local_ego` | yes |
| 2–7 | sender/receiver local degrees, degree sums | `local_degree` | yes |
| 8–10 | `sender/receiver/mean_clustering_local` | `local_clustering` | no |
| 11–13 | `sender/receiver/mean_triangles_local` | `local_triangles` | yes |

### Tier 2 — betweenness centrality (M3)

Sampled Brandes BC per split graph; endpoint lift. `--morph_tier2_lift full` (4 cols) or `max` (1 col). Precompute: `scripts/precompute_morphology_tier2.py`.

### M2 contrast feature groups

| Group | Columns binned |
|-------|----------------|
| `local_ego` | Tier 1 cols 0–1 |
| `local_degree` | Tier 1 cols 2–7 |
| `local_clustering` | Tier 1 cols 8–10 |
| `global_degree` | Tier 0 block (needs `local+global` scope) |
| `edge_native` | Edge-native block |

Code: [`morphology/tier1_local.py`](../morphology/tier1_local.py), [`tier0_global.py`](../morphology/tier0_global.py), [`tier2_global.py`](../morphology/tier2_global.py), [`contrast.py`](../morphology/contrast.py).

---

## Example — M1b pretrain

```bash
python scripts/precompute_morphology_tier0.py \
  --data Small-HI --output_dir morphology_cache/Small-HI \
  --reverse_mp --ego --ports

python main.py \
  --data Small-HI --model gin \
  --objective contrastive \
  --unique_name hi_morphology_global_20ep \
  --save_model --n_epochs 20 \
  --reverse_mp --ego --ports \
  --batch_size 32768 --num_neighs 100 100 \
  --morph_expert --morph_targets local+global \
  --morph_tier0_cache morphology_cache/Small-HI \
  --contrastive_asymmetric --contrastive_num_neg_samples 1024 \
  --contrastive_memory_bank_size 32768 \
  --tqdm --testing
```

Then **extract → probe** (same `--unique_name`). Expert head is **not** used at extraction.

More examples (M2, clustering bins, Tier 2): [`morphology-metrics-plan.md`](morphology-metrics-plan.md).

---

## Flags cheat sheet

| Goal | Flags |
|------|-------|
| Baseline contrastive | *(none)* |
| Contrastive + projection | `--contrast_projection_head --contrast_projection_hidden 128 --contrast_projection_dim 128 --contrastive_memory_bank_size 0` |
| Endpoint false-negative filtering | `--false_neg_filter_mode same_pair` |
| Endpoint weak positives | `--multi_positive_mode same_endpoint --multi_positive_weight 0.1` |
| M1b + projection | M1b expert + projection flags |
| M1 local expert | `--morph_expert --morph_targets local` |
| M1b global expert | `--morph_expert --morph_targets local+global --morph_tier0_cache morphology_cache/Small-HI` |
| Targeted group expert | add `--morph_target_groups degree_fan` or `local_motif` |
| M1b + BC (M3) | `--morph_targets local+global+tier2 --morph_tier2_cache morphology_cache/Small-HI` |
| M2 soft positives | `--morph_contrast --morph_contrast_features local_ego,local_degree` |
| M2 + clustering bins | add `local_clustering` to `--morph_contrast_features` |
| Faster morph val | `--morph_val_every 2 --morph_val_max_batches 10` |

---

## Key concepts

### Contrastive

| Concept | What it does |
|---------|--------------|
| **Asymmetric InfoNCE** | Only `L(z1→z2)`; view2 under `no_grad`. ~half backward VRAM vs symmetric |
| **Contrastive queue** | FIFO of past view2 embeddings as extra negatives; same `edge_id` filtered. Latest asym + projection queue sweep favors `queue=0` on Small-HI |
| **Projection head** | MLP before InfoNCE only; extract uses raw encoder `z` (128-d) |
| **False-negative filtering** | Optional endpoint/pair exclusion from negatives (`same_sender`, `same_receiver`, `same_endpoint`, `same_pair`); `same_pair` is the leading replicated F1/recall ablation, not the default |
| **Endpoint weak positives** | Optional multi-positive InfoNCE with weighted endpoint/pair positives; distinct from M2 morphology-bin positives |

### Morphology

| Concept | What it does |
|---------|--------------|
| **`local_ego`** | Batch-level subgraph edge/node counts (cols 0–1); one value per batch |
| **`log1p` targets** | Transform on count-like expert columns before MSE/MAE |
| **Disjoint contrast vs expert** | Papagei-style split **not implemented** — default M2 overlaps expert on ego/degree |
| **Morph val throttling** | `--morph_val_every`, `--morph_val_max_batches`; feeds `--checkpoint_policy best` |

### Extraction & probe

| Concept | What it does |
|---------|--------------|
| **Dedupe by `edge_id`** | One embedding per transaction before writing `.npz` |
| **AUROC** | Primary downstream metric |
| **F1** | At threshold = argmax F1 on **val** |
| **Best ckpt (SSL)** | Morph: sum of `morph/expert_val` + `morph/contrast_val`; plain contrastive: `loss/train`. For targeted group scouts at low `morph_expert_weight`, morph-val best can lock to early epochs — reprobe last epoch before trusting downstream numbers |

AML labels are **never** used for encoder checkpoint selection during SSL pretrain.

---

## Morphology package (`morphology/`)

| Module | Role |
|--------|------|
| `IDS.md` | Join keys: `EdgeID`, `from_id`, `to_id` |
| `tier0_global.py` | Split-global degrees; endpoint lift (M1b) |
| `tier1_local.py` | Batch-local subgraph stats |
| `tier2_global.py` | Betweenness centrality lift (M3) |
| `expert.py` | `MorphologyExpertHead`, MSE loss |
| `contrast.py` | M2 bin groups and soft positives |
| `contrastive_train.py` | Training-loop glue, morph val gating |

**Training attachment (`training.py`):** batch → two views → InfoNCE; optional expert MSE on `z_seed`; optional M2 bin soft positives; throttled morph val at epoch end.

---

## Tests

```bash
python -m pytest tests/test_morphology_metrics.py tests/test_morphology_contrast.py \
  tests/test_morphology_expert_grouped.py tests/test_morphology_expert_tier2.py tests/test_morphology_tier2.py -q
```
