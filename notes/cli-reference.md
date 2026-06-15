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
| `--contrastive_asymmetric` | off | **Asymmetric InfoNCE:** loss = `L(z1→z2)` only; view2 under `no_grad`. Saves ~half backward VRAM vs symmetric |
| `--contrastive_accum_steps` | `1` | Gradient accumulation steps before `optimizer.step` |
| `--contrastive_memory_bank_size` | `0` | FIFO queue of past view2 embeddings as extra negatives (MoCo-style). `0` = disabled; morph runs often use `32768` |
| `--contrast_projection_head` | off | GraphCL-style MLP before InfoNCE only; extraction uses encoder `z` |
| `--contrast_projection_hidden` | `128` | Hidden width when projection head enabled |
| `--contrast_projection_dim` | `128` | Projection output dim (default matches embedding dim) |

Hetero contrastive training (`--reverse_mp --objective contrastive`) uses the same augmentation and InfoNCE machinery on the heterogeneous graph path.

---

## Morphology expert head (contrastive pretrain only)

Requires `--objective contrastive` and `--morph_expert`. Adds `L_morph_expert` (MSE on detached morphology targets) alongside InfoNCE.

| Argument | Default | Description |
|----------|---------|-------------|
| `--morph_expert` | off | Enable morphology expert auxiliary loss |
| `--morph_targets` | `local` | `local` (Tier 1 + edge-native); `local+global` (M1b); `local+tier2`; `local+global+tier2` (M3) |
| `--morph_tier0_cache` | — | Directory with `{train,val,test}_node_morphology.csv` from `scripts/precompute_morphology_tier0.py` |
| `--morph_tier2_cache` | — | Directory with `{train,val,test}_node_tier2.csv` from `scripts/precompute_morphology_tier2.py` |
| `--morph_tier2_lift` | `full` | BC lift: `full` (4 cols) or `max` (1 col) |
| `--morph_expert_loss` | `mse` | `mse` or `mae` |
| `--morph_expert_weight` | `1.0` | Scale morphology expert loss vs InfoNCE |
| `--morph_expert_hidden` | `64` | Hidden size of expert MLP(s) |
| `--morph_expert_layout` | `shared` | `shared` or `grouped` (M5a) |
| `--morph_expert_group_weight_tier2` | `1.0` | Tier 2 block weight when `layout=grouped` |
| `--morph_local_subset` | `all` | Tier-1 columns: `all` (14), `degree` (8), `clustering` (11), `triangles` (11) |
| `--no_morph_edge_native` | off | Exclude forward `edge_attr` from morphology targets |

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
