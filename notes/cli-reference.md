# Command-line reference

Hyperparameters for each architecture come from [`model_settings.json`](../model_settings.json). Most training examples use hetero Multi-GIN with `--reverse_mp --ego --ports`.

---

## Required

| Argument | Description |
|----------|-------------|
| `--data` | Dataset folder name under `aml_data` (e.g. `Small-HI`) |
| `--model` | Architecture: `gin`, `gat`, `pna`, or `rgcn` |

---

## Graph form and Multi-GNN adaptations

| Argument | Description |
|----------|-------------|
| `--reverse_mp` | Heterogeneous graph with reverse message passing |
| `--ego` | Ego IDs on center nodes |
| `--ports` | Port numberings on edges |
| `--emlps` | Edge updates via MLPs |
| `--tds` | Time-delta edge features |

---

## Training objective

| Argument | Description |
|----------|-------------|
| `--objective contrastive` | Self-supervised contrastive pretraining **(default)** |
| `--objective supervised` | Supervised AML edge classification |

---

## Run identity, checkpoints, and modes

| Argument | Description |
|----------|-------------|
| `--unique_name` | Run identifier; names checkpoints and embedding folders |
| `--save_model` | Save checkpoint(s) to `model_to_save` |
| `--checkpoint_policy` | `last` (overwrite each epoch) or `best` (lowest morph val / train loss → main checkpoint; final epoch → `_last.tar`) |
| `--finetune` | Load `checkpoint_{unique_name}.tar` before training; saves `_finetuned` when combined with `--save_model` |
| `--inference` | Load checkpoint and run evaluation only (no training) |

---

## Training hyperparameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--batch_size` | `8192` | Minibatch size for `LinkNeighborLoader` |
| `--n_epochs` | `100` | Training epochs |
| `--num_neighs` | `100 100` | Neighbors sampled per hop (space-separated, descending) |
| `--loader_num_workers` | `10` | CPU workers for subgraph sampling (`0` for single-process / debugging) |
| `--seed` | `1` | Random seed |

---

## Contrastive / memory

| Argument | Default | Description |
|----------|---------|-------------|
| `--amp` | off | CUDA automatic mixed precision |
| `--gradient_checkpointing` | off | Gradient checkpointing in GIN layers (GIN only) |
| `--contrastive_num_neg_samples` | `8192` | InfoNCE negatives per anchor (`0` = all negatives, chunked) |
| `--contrastive_temperature` | `0.5` | InfoNCE logit temperature; lower values sharpen positive/negative separation |
| `--contrastive_asymmetric` | off | **Asymmetric InfoNCE:** loss = `L(z1→z2)` only; view2 under `no_grad`. Saves ~half backward VRAM vs symmetric |
| `--contrastive_accum_steps` | `1` | Gradient accumulation steps before `optimizer.step` |
| `--contrastive_memory_bank_size` | `0` | FIFO queue of past view2 embeddings as extra negatives (MoCo-style). `0` = disabled and is currently recommended for asym + projection on Small-HI |
| `--contrast_projection_head` | off | GraphCL-style MLP before InfoNCE only; extraction uses encoder `z` |
| `--contrast_projection_hidden` | `128` | Hidden width when projection head enabled |
| `--contrast_projection_dim` | `128` | Projection output dim (default matches embedding dim) |
| `--edge_drop_policy` | `random` | `random` (legacy uniform), `degree_aware`, or `degree_flow_aware` |
| `--edge_drop_target_rate` | `0.1` | Target mean drop rate (matches legacy `0.1` random drop) |
| `--edge_drop_min_prob` / `--edge_drop_max_prob` | `0.01` / `0.95` | Per-edge probability clip bounds for non-random policies |
| `--edge_drop_importance_alpha` | `2.0` | Nonuniform drop strength (higher → keep important edges more) |
| `--edge_drop_score_cache_path` | — | Optional precomputed train-split `.npz` (see `scripts/precompute_edge_drop_scores.py`) |

Hetero contrastive training (`--reverse_mp --objective contrastive`) uses the same augmentation and InfoNCE machinery on the heterogeneous graph path.

Practical Small-HI guidance:

- Current strongest asym + projection run: `--contrastive_num_neg_samples 8192 --contrastive_memory_bank_size 0 --batch_size 8192 --contrastive_accum_steps 4`.
- Queue-size ablations found `queue=0` best; larger queues reduced downstream AUROC/F1 in this setup.
- Temperature is configurable with `--contrastive_temperature`; the current recipe should keep the default `0.5`. Lower values `0.05`, `0.10`, and `0.20` underperformed in the first sweep.
- `10240` negatives fits with `queue=0`, `batch_size=8192`, `accum=4`, but did not improve AUROC. Direct `12288` at that batch size OOMed; `12288` fits with `batch_size=4096`, `accum=8` but did not improve metrics.
- `--edge_drop_policy degree_aware` gives a small F1/recall lift on Small-HI (−0.2 AUROC vs baseline). Do **not** combine with `--false_neg_filter_mode same_pair` — 2-seed mean **0.939 AUROC / 0.168 F1** ([`results.md`](results.md)).
- Adding back `queue=32768` with false-negative filtering still hurt, so keep `queue=0` unless explicitly testing queues.

### False-negative filtering and multi-positive InfoNCE

These flags are optional experiments for the edge-level contrastive loss. Defaults preserve the current identity-only InfoNCE behavior.

| Argument | Default | Description |
|----------|---------|-------------|
| `--false_neg_filter_mode` | `none` | Exclude likely false negatives from the negative pool. Modes: `same_sender`, `same_receiver`, `same_endpoint`, `same_pair` |
| `--false_neg_filter_min_negatives` | `1` | Per-anchor fallback threshold. If fewer than this many candidates remain after filtering, that anchor uses the unfiltered candidate set |
| `--knn_cache_path` | — | Sparse offline transaction KNN `.npz` cache from `scripts/precompute_transaction_knn.py` |
| `--enable_knn_negative_filter` | off | Exclude cached KNN neighbors from contrastive negatives |
| `--knn_filter_k` | `0` | Use first K cached neighbors for exclusion (`0` = all cached neighbors) |
| `--enable_knn_soft_positives` | off | Add cached KNN neighbors as low-weight InfoNCE positives (requires `--contrastive_asymmetric`) |
| `--knn_pos_source_k` | `15` | Use top-k cached neighbors as positive candidates |
| `--knn_pos_m` | `1` | KNN positives sampled per anchor per step |
| `--knn_pos_weight` | `0.025` | Total KNN positive mass per anchor (split across `m`, not multiplied) |
| `--knn_pos_weight_mode` | `uniform` | `uniform` or `similarity` (similarity renormalizes within selected positives) |
| `--knn_pos_min_sim` | — | Optional minimum cached similarity for a KNN positive |
| `--knn_pos_seed` | `0` | Base seed for deterministic KNN-positive sampling |
| `--knn_pos_loader_batch_size` | `4096` | Chunk size for auxiliary seed forwards |
| `--multi_positive_mode` | `none` | Add endpoint/pair weak positives in the numerator. Modes: `same_sender`, `same_receiver`, `same_endpoint`, `same_pair` |
| `--multi_positive_weight` | `0.1` | Weight for weak positives. Identity positives remain weight `1.0` |

Mode semantics:

| Mode | Relationship to anchor transaction `(sender, receiver)` |
|------|--------------------------------------------------------|
| `same_sender` | candidate has the same sender |
| `same_receiver` | candidate has the same receiver |
| `same_endpoint` | candidate shares either endpoint with the anchor |
| `same_pair` | candidate has the same ordered sender→receiver pair |

Implementation notes:

- False-negative filtering is **exclusion-only**: filtered candidates are removed from the denominator, not added as positives.
- Multi-positive mode adds weak endpoint/pair positives among the aligned current cross-view seed batch; queue entries matching the weak-positive rule are excluded as negatives but are not used as positives.
- Both features use split-local edge endpoints from the forward `edge_index`; they are compatible with asymmetric/symmetric InfoNCE, projection head, sampled negatives, and the memory queue.
- Logs include before/after candidate counts for filtering, fallback rows, identity positives, weak positives, average positives per anchor, and fraction of anchors without weak positives.
- KNN filter (when enabled): per-epoch `anchors_with_knn_in_pool`, `removed`, and `fallback_rows` — see [`knn-precompute-reference.md`](knn-precompute-reference.md).
- KNN soft positives (when enabled): per-epoch anchor coverage, injected unique ids, similarity min/mean/max, endpoint overlap, identity vs KNN numerator contribution, and drop reasons — see [`knn-precompute-reference.md`](knn-precompute-reference.md).
- Small-HI follow-up: `--false_neg_filter_mode same_pair` is the leading F1/recall candidate across seeds. `same_endpoint` and `same_receiver` were less stable. Default remains `none`. Do **not** stack with `--edge_drop_policy degree_aware` — see [`results.md`](results.md).
- Multi-positive runs so far underperform exclusion-only filtering. Lower `same_pair` weight `0.05` helped relative to `0.1`, but still did not beat no-filter or `same_pair` false-negative filtering.

### Offline feature-KNN negative filtering

Precompute train-only sparse KNN caches, then exclude cached neighbors from
contrastive negatives at train time. **Full reference:**
[`knn-precompute-reference.md`](knn-precompute-reference.md) (feature sets, cache
format, GPU backends, sharding, Slurm).

Quick start:

```bash
sbatch slurm/precompute_transaction_knn_small_hi_gpu_smoke100k.sh   # 100k smoke
sbatch slurm/precompute_transaction_knn_small_hi_gpu_full_k15.sh    # full train
```

Training flags (after cache exists):

```bash
--enable_knn_negative_filter \
--knn_cache_path morphology_cache/Small-HI/transaction_knn_edge_native_degree_fan_k15.npz \
--knn_filter_k 15
```

The cache uses train split-local `edge_id` values, matching contrastive training.
KNN neighbors are removed from negatives only; they are not added as positives.
Queue KNN filtering is intentionally deferred.

**Jun 2026 ablation:** with random 8192 negatives, exclusion did not beat the
no-filter baseline (k=5: 0.947 / 0.209; k=15: 0.928 / 0.176 vs 0.951 / 0.233).
See [`results.md` § Feature-KNN](results.md#feature-knn-small-hi).

**KNN soft positives:** `--enable_knn_soft_positives` adds low-weight cached
neighbors to the InfoNCE numerator via an auxiliary seed forward pass. Requires
`--contrastive_asymmetric`. **Jun 2026 ablation hurt** probe (0.849 / 0.067 vs
0.951 / 0.233 baseline) — do not enable with the current saturated cache. Slurm:
`slurm/ablation_knn_softpos_m1_w0025_asym_proj_8192neg_queue0_20ep.sh`.
Feature-set audit: `python scripts/audit_transaction_knn_cache.py`.

Example:

```bash
python main.py \
  --data Small-HI --model gin \
  --objective contrastive \
  --unique_name hi_contrastive_proj_same_endpoint_pos \
  --reverse_mp --ego --ports \
  --contrast_projection_head \
  --contrastive_asymmetric \
  --contrastive_num_neg_samples 8192 \
  --contrastive_memory_bank_size 32768 \
  --multi_positive_mode same_endpoint \
  --multi_positive_weight 0.1
```

---

## Morphology expert head (contrastive pretrain only)

Requires `--objective contrastive` and `--morph_expert`. Adds `L_morph_expert` (MSE on detached morphology targets) alongside InfoNCE.

| Argument | Default | Description |
|----------|---------|-------------|
| `--morph_expert` | off | Enable morphology expert auxiliary loss |
| `--morph_targets` | `local` | `local` (Tier 1 + edge-native); `local+global` (M1b); `local+tier2`; `local+global+tier2` (M3) |
| `--morph_tier0_cache` | — | Directory with `{train,val,test}_node_morphology.csv` from `scripts/precompute_morphology_tier0.py` |
| `--morph_flow_balance` | off | Append 10 Tier 0 flow-balance expert targets (see morphology reference) |
| `--morph_tier0_flow_cache` | — | Directory with `{train,val,test}_node_flow_balance.csv`; falls back to `--morph_tier0_cache` |
| `--morph_tier2_cache` | — | Directory with `{train,val,test}_node_tier2.csv` from `scripts/precompute_morphology_tier2.py` |
| `--morph_tier2_lift` | `full` | BC lift: `full` (4 cols) or `max` (1 col) |
| `--morph_expert_loss` | `mse` | `mse` or `mae` |
| `--morph_expert_weight` | `1.0` | Scale morphology expert loss vs InfoNCE |
| `--morph_expert_hidden` | `64` | Hidden size of expert MLP(s) |
| `--morph_expert_layout` | `shared` | `shared` or `grouped` (M5a) |
| `--morph_expert_group_weight_tier2` | `1.0` | Tier 2 block weight when `layout=grouped` |
| `--morph_local_subset` | `all` | Tier-1 columns: `all` (14), `degree` (8), `clustering` (11), `triangles` (11) |
| `--morph_target_groups` | `all` | Shared-head target filter by semantic group, e.g. `degree_fan`, `motif_participation`, or legacy `local_motif` |
| `--no_morph_edge_native` | off | Exclude forward `edge_attr` from morphology targets |

Expert diagnostics log the unchanged total loss as `morph/expert_train` and
`morphology/loss_total`, plus per-group MSE keys such as
`morphology/loss_group/degree_fan`, `motif_participation`, `local_density`,
`local_context_size`, `global_role`, `volume_activity`, `temporal_behavior`,
and `other`. Legacy aliases (`local_motif`, `centrality`, `temporal`) expand to
these groups. These logs are diagnostic only; they
do not change the shared expert head or default loss. Group losses are also
printed to stdout once per epoch for Slurm smoke-test inspection.

Use `--morph_target_groups` for targeted diagnostics that keep the same shared
expert head but predict only selected semantic target groups. Default `all`
preserves the historical target vector and training behavior.

---

## Morphology-aware contrast (Phase M2)

Requires `--objective contrastive` and `--morph_contrast`. Soft positives from same morphology bin across views.

| Argument | Default | Description |
|----------|---------|-------------|
| `--morph_contrast` | off | Enable morphology-bin soft positives in edge InfoNCE |
| `--morph_contrast_features` | `local_ego,local_degree` | Bin groups: `local_ego`, `local_degree`, `local_clustering`, `global_degree`, `edge_native` |
| `--morph_contrast_scope` | `local` | `local+global` enables `global_degree` binning |
| `--morph_contrast_bins` | `5` | Quantile buckets per dimension |
| `--morph_contrast_calib_batches` | `32` | Train batches for bin quantile estimation |
| `--morph_contrast_max_soft_positives` | `256` | Cap same-bin positives per anchor (`0` = no cap) |
| `--morph_val_every` | `1` | Run morph val every N epochs (always on final epoch) |
| `--morph_val_max_batches` | `0` | Cap val batches per morph val pass (`0` = full val loader) |

**Target dimensions (defaults):** `local` → **15** (11 local + 4 edge-native); `local+global` → **24**. Count-like columns use `log1p` before MSE; clustering stays in **[0, 1]**.

Checkpoints store `morph_expert_state_dict` and optionally `contrast_projection_state_dict` (resume only — extraction uses the encoder).

Metric column definitions: [`morphology-reference.md`](morphology-reference.md).

---

## Logging and misc

| Argument | Description |
|----------|-------------|
| `--tqdm` | Progress bars during training |
| `--testing` | Disable W&B logging |

---

## Linear probe (`linear_probe.py`)

| Argument | Default | Description |
|----------|---------|-------------|
| `--unique_name` | — | Embedding subfolder under `--embeddings_dir` |
| `--embeddings_dir` | `embeddings` | Root directory from extraction |
| `--class_weight` | `balanced` | `balanced`, `none`, or `model` (reads CE weights from `model_settings.json`) |
| `--threshold_tuning` | `max_f1_val` | `max_f1_val` or `fixed_0.5` |
| `--probe_max_iter` | `1000` | Max iterations for sklearn logistic regression |
| `--testing` | off | Disable W&B |

---

## Label-efficiency probe (`scripts/label_efficiency_probe.py`)

Stratified subsets of **train** labels (default 10/25/50/100%); threshold tuned on **full val**. Does **not** retrain the GNN.

| Argument | Default | Description |
|----------|---------|-------------|
| `--unique_name` / `--unique_names` | — | One or more embedding folders |
| `--train_fractions` | `0.1,0.25,0.5,1.0` | Comma-separated train label fractions |
| `--class_weight` | `model` | Same as `linear_probe.py` |
| `--probe_max_iter` | `5000` | sklearn LR iterations |
| `--probe_n_jobs` | `1` | Avoid OOM on large `train.npz` on Slurm |
| `--testing` | off | Disable W&B |

```bash
python scripts/label_efficiency_probe.py \
  --unique_names hi_morphology_global_20ep hi_morph_global_contrast_10ep_bestckpt \
  --class_weight model --model gin --testing
```

**Outputs:** `embeddings/{unique_name}/label_efficiency_results.json`, `embeddings/label_efficiency_summary.json`. Results: [`results.md`](results.md).
