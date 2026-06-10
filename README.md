# Multi-GNN — Graph foundation model extensions for AML

This repository extends [IBM Multi-GNN](https://github.com/IBM/Multi-GNN) for anti-money laundering (AML) on financial transaction graphs. It keeps the original Multi-GNN architectures and graph adaptations ([GIN](https://arxiv.org/abs/1810.00826), [GAT](https://arxiv.org/abs/1710.10903), [PNA](https://arxiv.org/abs/2004.05718), [RGCN](https://arxiv.org/abs/1703.06103)) and adds a **graph foundation model (GFM)** workflow:

1. **Self-supervised pretraining** — contrastive learning on transaction graphs (`--objective contrastive`, the default); optional **contrastive projection head** (GraphCL-style; recommended for pure SSL) and/or **morphology expert** + **morphology-aware contrast** (M1/M1b/M2)
2. **Frozen embedding extraction** — seed-edge representations written to disk (`embedding_extraction.py`); use `--checkpoint_policy best` for morph runs so extraction does not use a regressed last epoch
3. **Linear probing** — sklearn logistic regression on frozen features (`linear_probe.py`)
4. **Label-efficiency probing** (optional) — same embeddings, stratified subsets of train labels (`scripts/label_efficiency_probe.py`) to compare encoders under scarce labels (GCPAL / Papagei-style)

The original **supervised** Multi-GNN path (`--objective supervised`) remains available as a baseline and ablation (~0.97 test AUROC in-GNN on Small-HI; still above frozen SSL + linear probe). For design rationale and implementation status, see [`notes/contrastive-learning-plan.md`](notes/contrastive-learning-plan.md), [`notes/morphology-metrics-plan.md`](notes/morphology-metrics-plan.md), [`notes/projection-head-ablation-jun2026.md`](notes/projection-head-ablation-jun2026.md), and [`notes/lit-review-index.md`](notes/lit-review-index.md).

**Quick links:** [Primary workflow](#primary-workflow--contrastive-pretrain--extract--linear-probe) · [Contrastive projection head](#contrastive-projection-head-recommended-for-pure-ssl) · [Morphology (M1/M2)](#morphology-aware-contrastive-pretraining-optional) · [Label-efficiency probe](#label-efficiency-probe-scriptslabel_efficiency_probepy) · [Repository guide](#repository-guide) · [CLI reference](#command-line-arguments) · [Slurm / cluster](#slurm-and-cluster-jobs)

---

## Setup

Create and activate the conda environment:

```bash
conda env create -f env.yml
conda activate multignn
```

Configure paths in [`data_config.json`](data_config.json):

```json
{
  "paths": {
    "aml_data": "aml-data",
    "model_to_load": "saved-models",
    "model_to_save": "saved-models"
  }
}
```

| Path | Purpose |
|------|---------|
| `aml_data` | Root directory for preprocessed datasets (see below) |
| `model_to_load` | Where checkpoints are read from |
| `model_to_save` | Where checkpoints are written when `--save_model` is used |

Training logs to `logs/logs.log`. [Weights & Biases](https://wandb.ai/) logging is enabled by default; pass `--testing` to disable it. You can provide a W&B API key via a `.env` file in the repo root (`load_dotenv()` in `main.py`).

---

## Data preparation

**Currently supported:** the [IBM synthetic AML transaction datasets](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml/data) on Kaggle (from Egressy et al., NeurIPS 2023). All training and evaluation commands in this README assume data prepared in that format. Additional datasets may be supported later; this section will be updated when they are.

### Download and clean (IBM / Kaggle)

1. Download the transaction CSV(s) you need from Kaggle (e.g. `HI-Small_Trans.csv` for the small, high-illicitness split).
2. Run the formatter on each file:

```bash
python format_kaggle_files.py /path/to/HI-Small_Trans.csv
```

This writes `formatted_transactions.csv` in the same directory as the input. Repeat for each Kaggle file you use (e.g. `LI-Small_Trans.csv`, medium splits, etc.).

3. Copy each `formatted_transactions.csv` into its own folder under `aml_data/`. The folder name must match the `--data` argument you pass to `main.py` (e.g. `Small-HI` for the HI small split):

```
aml-data/
  Small-HI/
    formatted_transactions.csv
  Small-LI/
    formatted_transactions.csv
  ...
```

The Kaggle release uses several dataset sizes and illicitness levels; treat each as a separate folder and preprocessing run. Naming is up to you as long as it is consistent with `--data`.

### Formatted schema and splits

The formatted CSV uses columns such as `EdgeID`, `from_id`, `to_id`, `Timestamp`, `Amount Sent`, `Sent Currency`, `Amount Received`, `Received Currency`, `Payment Format`, and `Is Laundering`.

Train / validation / test are assigned by **calendar day** (~60 / 20 / 20) to avoid temporal leakage. Illicit transactions are heavily underrepresented; downstream **AUROC** from the linear probe is often a more informative metric than F1 on short runs.

---

## Usage

All examples below use hetero Multi-GIN with the standard adaptations (`--reverse_mp --ego --ports`). Omit or change flags as needed. Hyperparameters for each architecture are loaded from [`model_settings.json`](model_settings.json).

### Primary workflow — contrastive pretrain → extract → linear probe

**Step 1 — Pretrain and save a checkpoint**

```bash
python main.py \
  --data Small-HI --model gin \
  --objective contrastive \
  --unique_name my_pretrain \
  --save_model \
  --reverse_mp --ego --ports \
  --batch_size 8192 --num_neighs 100 100 \
  --n_epochs 100 --tqdm
```

**Step 2 — Extract frozen seed-edge embeddings**

```bash
python embedding_extraction.py \
  --data Small-HI --model gin \
  --unique_name my_pretrain \
  --reverse_mp --ego --ports \
  --tqdm
```

Requires `saved-models/checkpoint_{unique_name}.tar`.

**Step 3 — Linear probe (downstream evaluation)**

```bash
python linear_probe.py \
  --unique_name my_pretrain \
  --testing
```

Reads `embeddings/{unique_name}/{train,val,test}.npz` and writes metrics to `embeddings/{unique_name}/probe_results.json`.

During contrastive pretraining, monitor **training loss** and **morphology val losses** — AML metrics are not used for encoder checkpoint selection. With `--checkpoint_policy best` (recommended for morph and projection runs), extraction loads the epoch with the lowest SSL val score (morph composite or `loss/train`), not necessarily the final epoch. After extraction, report **AUROC** (primary) and **F1** (secondary) from the probe.

### Contrastive projection head (recommended for pure SSL)

GraphCL/GCPAL-style **projection MLP** applied only during InfoNCE; extraction and morphology expert still use encoder `z` (128-d). On Small-HI this closed most of the gap to supervised for frozen linear probe: test AUROC **0.839 → 0.927** vs plain contrastive (Jun 2026). See [`notes/projection-head-ablation-jun2026.md`](notes/projection-head-ablation-jun2026.md).

```bash
python main.py \
  --data Small-HI --model gin \
  --objective contrastive \
  --unique_name hi_contrastive_proj_20ep_bestckpt \
  --save_model --n_epochs 20 \
  --reverse_mp --ego --ports \
  --batch_size 32768 --num_neighs 100 100 \
  --contrast_projection_head \
  --contrast_projection_hidden 128 --contrast_projection_dim 128 \
  --contrastive_asymmetric --contrastive_num_neg_samples 1024 \
  --contrastive_memory_bank_size 32768 \
  --checkpoint_policy best \
  --tqdm --testing
```

Then **extract → probe** as in the primary workflow.
<!-- Slurm: [`ablation_contrastive_projection_20ep .sh`](ablation_contrastive_projection_20ep%20.sh). To stack projection with M1b: [`ablation_m1b_projection_20ep.sh`](ablation_m1b_projection_20ep.sh) (AUROC ↑ vs M1b, F1 ↓ at val-tuned threshold). -->

### Morphology-aware contrastive pretraining (optional)

Beyond edge-identity InfoNCE, you can add a **morphology expert head** (Papagei-style auxiliary targets) that predicts label-free structural features of each seed transaction from its 128-d embedding. This shapes representations toward graph morphology without using AML labels during pretrain.

| Phase | Flag | What it adds |
|-------|------|--------------|
| **M1** | `--morph_expert --morph_targets local` | Tier 1 **local** subgraph stats (11 dims: ego, degrees, **clustering**) + **edge-native** from `edge_attr` |
| **M1b** | `--morph_targets local+global` | Tier 0 **global** endpoint degrees (split-safe lookup / lift) in addition to Tier 1 local |
| **M1b + BC** | `--morph_targets local+global+tier2` | M3: M1b + betweenness centrality endpoint lift (requires Tier 2 cache) |
| **BC-only ablation** | `--morph_targets local+tier2` | M3 ablation: Tier 1 local + BC lift only (no Tier 0 degrees) |
| **M2** | `--morph_contrast` | **Merged soft positives** in edge InfoNCE: same morphology bin across views (identity positives unchanged) |
| **Projection** | `--contrast_projection_head` | GraphCL-style MLP before InfoNCE only; encoder `z` at extract (stacks with morph expert optionally) |

**Example — M2 pretrain (M1b expert + morphology contrast, Small-HI):**

```bash
python main.py \
  --data Small-HI --model gin \
  --objective contrastive \
  --unique_name hi_morph_global_contrast_20ep \
  --save_model --n_epochs 20 \
  --reverse_mp --ego --ports \
  --batch_size 32768 --num_neighs 100 100 \
  --morph_expert --morph_targets local+global \
  --morph_tier0_cache morphology_cache/Small-HI \
  --morph_contrast \
  --morph_contrast_features local_ego,local_degree \
  --morph_contrast_scope local \
  --contrastive_asymmetric --contrastive_num_neg_samples 1024 \
  --contrastive_memory_bank_size 32768 \
  --tqdm --testing
```

Use `--morph_contrast_scope local+global` and add `global_degree` to `--morph_contrast_features` to include Tier 0 bins. Add **`local_clustering`** to bin on sender/receiver/mean clustering (see Tier-1 table below). Diagnostic: `--morph_contrast` without `--morph_expert` (contrast-only).

**Example — M2 with local clustering bins (M1b expert + triadic soft positives):**

```bash
python main.py \
  --data Small-HI --model gin \
  --objective contrastive \
  --unique_name hi_morph_global_clustering_m2_10ep \
  --save_model --n_epochs 10 \
  --reverse_mp --ego --ports \
  --batch_size 32768 --num_neighs 100 100 \
  --morph_expert --morph_targets local+global \
  --morph_tier0_cache morphology_cache/Small-HI \
  --morph_contrast \
  --morph_contrast_features local_ego,local_degree,local_clustering \
  --morph_contrast_scope local \
  --checkpoint_policy best \
  --contrastive_asymmetric --contrastive_num_neg_samples 1024 \
  --contrastive_memory_bank_size 32768 \
  --morph_val_every 2 --morph_val_max_batches 10 \
  --tqdm --testing
```

<!-- Slurm: [`ablation_m1b_clustering_m2_10ep.sh`](ablation_m1b_clustering_m2_10ep.sh). -->

**Tier-1 local features** (always included when `--morph_expert` is on; computed on **view1** subgraph):

| Index | Name | M2 group | log1p in expert? |
|-------|------|----------|------------------|
| 0–1 | `n_edges_sub`, `n_nodes_sub` | `local_ego` | yes |
| 2–7 | sender/receiver degrees, degree sums | `local_degree` | yes |
| 8–10 | `sender_clustering_local`, `receiver_clustering_local`, `mean_clustering_local` | `local_clustering` | no (in [0, 1]) |

Clustering uses **undirected** local clustering on the batch subgraph: closed triangles → 1.0, chains → 0.0. No extra CLI flag — it is part of `LOCAL_FEATURE_NAMES` in [`morphology/tier1_local.py`](morphology/tier1_local.py).

**Example — M1b pretrain (recommended morph expert config; includes clustering since Jun 2026):**

```bash
# Optional: precompute Tier 0 node tables once (reuse across runs)
python scripts/precompute_morphology_tier0.py \
  --data Small-HI --output_dir morphology_cache/Small-HI \
  --reverse_mp --ego --ports

# Optional (M3 Phase 0): precompute Tier 2 betweenness centrality per split
python scripts/precompute_morphology_tier2.py \
  --data Small-HI --output_dir morphology_cache/Small-HI \
  --reverse_mp --ego --ports
# Default: sampled Brandes (256 sources). Use --bc_exact only on small graphs.
# On Slurm (128G): sbatch run_precompute_morphology_tier2.sh

python main.py \
  --data Small-HI --model gin \
  --objective contrastive \
  --unique_name hi_morphology_global_20ep \
  --save_model --n_epochs 20 \
  --reverse_mp --ego --ports \
  --batch_size 32768 --num_neighs 100 100 \
  --morph_expert \
  --morph_targets local+global \
  --morph_tier0_cache morphology_cache/Small-HI \
  --morph_expert_weight 1.0 \
  --contrastive_asymmetric --contrastive_num_neg_samples 1024 \
  --contrastive_memory_bank_size 32768 \
  --tqdm --testing
```

Then run **extract → probe** as in the primary workflow (same `--unique_name`). The morphology expert head is trained during pretrain but **not** used at extraction — only the GNN encoder embeddings are written to `.npz`.

**Small-HI benchmark (linear probe, val-tuned threshold, GIN hetero):**

| Run | Config | Epochs | Test AUROC | Test F1 |
|-----|--------|--------|------------|---------|
| **M1b + clustering + projection** | M1b + 11 local + `--contrast_projection_head` | 20 | **0.929** | **0.156** |
| Contrastive + projection | `--contrast_projection_head` | 20 | 0.927 | 0.144 |
| M1b + projection | M1b + projection head | 20 → **ep 15** | 0.924 | 0.096 |
| M1b | `--morph_expert`, `local+global` (8 local dims) | 20 | 0.920 | 0.108 |
| M1b + MAE expert | `--morph_expert_loss mae` | 20 | 0.898 | 0.145 |
| M1b + clustering (MSE) | M1b, 11 local dims (incl. clustering) | 20 | 0.903 | 0.117 |
| M3 BC-only | `local+tier2` | 20 | 0.904 | 0.093 |
| M2 expert + contrast | + `--checkpoint_policy best` | 10 → **ep 9** | 0.906 | 0.058 |
| M2 expert + contrast | + `--checkpoint_policy best` | 20 → ep 20 | 0.891 | 0.107 |
| M1 | `--morph_expert`, `local` | 20 | 0.910 | 0.079 |
| M3 M1b + BC (4 lift cols) | `local+global+tier2`, best ckpt | 20 → **ep 14** | 0.896 | 0.033 |
| M3 M1b + bc_max | `lift max` | 20 | 0.889 | 0.086 |
| M5a grouped BC | `layout=grouped`, `w_tier2=1` | 20 | 0.887 | 0.028 |
| M2 expert + contrast | `morph_expert_weight=0.5` (10 ep best) | 10 | 0.876 | 0.027 |
| M3 M1b + BC (last epoch) | same, no M4 | 20 | 0.861 | 0.029 |
| Contrastive baseline | identity InfoNCE | 20 | 0.839 | 0.076 |
| M2 expert + contrast | M1b + `--morph_contrast` | 20 (last) | 0.864 | 0.025 |
| M2 contrast only | `--morph_contrast` (no expert) | 10 | 0.680 | 0.012 |
| Supervised CE (in-GNN) | `--objective supervised` | — | ~0.972 | ~0.493 |

**Default SSL configs (Jun 2026):**

- **Full-label frozen probe:** **M1b + clustering + projection** (`hi_morphology_global_clustering_proj_20ep_bestckpt`) — best test AUROC/F1 among SSL runs (**0.929 / 0.156**). Clustering expert alone regressed (0.903); stacking with projection beats contrastive+proj (0.927).
- **Label-efficiency (10–100% train labels):** **contrastive + projection** leads at 25/50/100% (0.918–0.928 test AUROC); **M1b + projection** is best at **10%** labels (0.918). Label-efficiency on clustering+proj **pending**.
- **Morphology expert path:** **M1b** @ 20 ep (8 local dims) — best morph-only config (0.920 AUROC). **MAE expert loss** (`--morph_expert_loss mae`) did not improve AUROC vs MSE (0.898 vs 0.903 on same 11-dim targets).
- **Morphology negatives:** stacking BC on M1b hurts; M5a grouped heads did not fix interference (0.887 AUROC). See [`notes/morphology-metrics-plan.md`](notes/morphology-metrics-plan.md) and [`notes/projection-head-ablation-jun2026.md`](notes/projection-head-ablation-jun2026.md).

Results: `embeddings/{unique_name}/probe_results.json`. **AUROC** is the primary metric; **F1** is val-threshold-tuned and noisy on imbalanced AML data.

Design detail, metric tiers, and roadmap: [`notes/morphology-metrics-plan.md`](notes/morphology-metrics-plan.md). Literature ↔ implementation index: [`notes/lit-review-index.md`](notes/lit-review-index.md).

### Supervised baseline (original Multi-GNN)

End-to-end AML edge classification with in-graph cross-entropy and F1 logging:

```bash
python main.py \
  --data Small-HI --model gin \
  --objective supervised \
  --unique_name my_supervised \
  --save_model \
  --reverse_mp --ego --ports \
  --n_epochs 100 --tqdm
```

Use `--inference` instead of training to evaluate a saved checkpoint without updating weights.

### Optional — supervised finetune (secondary / ablation)

Loads `checkpoint_{unique_name}.tar`, continues training with supervised CE, and saves `checkpoint_{unique_name}_finetuned.tar`. This is **end-to-end fine-tuning**, not Papagei-style linear probing. Use the **same** `--data` and graph flags as the source checkpoint.

```bash
# Finetune
python main.py \
  --data Small-HI --model gin \
  --objective supervised \
  --unique_name my_pretrain \
  --finetune --save_model \
  --reverse_mp --ego --ports \
  --n_epochs 10 --tqdm

# Extract from finetuned checkpoint
python embedding_extraction.py \
  --data Small-HI --model gin \
  --unique_name my_pretrain \
  --finetune \
  --reverse_mp --ego --ports --tqdm

# Probe finetuned embeddings
python linear_probe.py \
  --unique_name my_pretrain_finetuned \
  --testing
```

---

## Checkpoints and outputs

| Artifact | Path |
|----------|------|
| Base checkpoint | `saved-models/checkpoint_{unique_name}.tar` |
| Finetuned checkpoint | `saved-models/checkpoint_{unique_name}_finetuned.tar` |
| Embeddings (base) | `embeddings/{unique_name}/{train,val,test}.npz` |
| Embeddings (finetuned) | `embeddings/{unique_name}_finetuned/{train,val,test}.npz` |
| Extraction metadata | `embeddings/{unique_name}/meta.json` |
| Probe results | `embeddings/{unique_name}/probe_results.json` |
| Morphology Tier 0 cache (optional) | `morphology_cache/{data}/{train,val,test}_node_morphology.csv` |
| Morphology Tier 2 cache (M3 Phase 0) | `morphology_cache/{data}/{train,val,test}_node_tier2.csv` (column `bc`) + `tier2_meta.json` |

`--unique_name` is required whenever you use `--save_model`, `--finetune`, extraction, or probing.

---

## Command-line arguments

### Required

| Argument | Description |
|----------|-------------|
| `--data` | Dataset folder name under `aml_data` (e.g. `Small-HI`) |
| `--model` | Architecture: `gin`, `gat`, `pna`, or `rgcn` |

### Graph form and Multi-GNN adaptations

| Argument | Description |
|----------|-------------|
| `--reverse_mp` | Heterogeneous graph with reverse message passing |
| `--ego` | Ego IDs on center nodes |
| `--ports` | Port numberings on edges |
| `--emlps` | Edge updates via MLPs |
| `--tds` | Time-delta edge features |

### Training objective

| Argument | Description |
|----------|-------------|
| `--objective contrastive` | Self-supervised contrastive pretraining **(default)** |
| `--objective supervised` | Supervised AML edge classification |

### Run identity, checkpoints, and modes

| Argument | Description |
|----------|-------------|
| `--unique_name` | Run identifier; names checkpoints and embedding folders |
| `--save_model` | Save checkpoint(s) to `model_to_save` |
| `--checkpoint_policy` | `last` (overwrite each epoch) or `best` (lowest morph val / train loss → main checkpoint; final epoch → `_last.tar`) |
| `--finetune` | Load `checkpoint_{unique_name}.tar` before training; saves `_finetuned` when combined with `--save_model` |
| `--inference` | Load checkpoint and run evaluation only (no training) |

### Training hyperparameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--batch_size` | `8192` | Minibatch size for `LinkNeighborLoader` |
| `--n_epochs` | `100` | Training epochs |
| `--num_neighs` | `100 100` | Neighbors sampled per hop (space-separated, descending) |
| `--loader_num_workers` | `10` | CPU workers for subgraph sampling (`0` for single-process / debugging) |
| `--seed` | `1` | Random seed |

### Contrastive / memory (homogeneous path)

| Argument | Default | Description |
|----------|---------|-------------|
| `--amp` | off | CUDA automatic mixed precision |
| `--gradient_checkpointing` | off | Gradient checkpointing in GIN layers (GIN only) |
| `--contrastive_num_neg_samples` | `8192` | InfoNCE negatives per anchor (`0` = all negatives, chunked) |
| `--contrastive_asymmetric` | off | One-directional loss (view 2 under `no_grad`; saves VRAM) |
| `--contrastive_accum_steps` | `1` | Gradient accumulation steps before `optimizer.step` |
| `--contrastive_memory_bank_size` | `0` | Optional queue of past embeddings as extra negatives |
| `--contrast_projection_head` | off | GraphCL-style MLP on seed embeddings before InfoNCE only; morphology expert + extraction use encoder `z` |
| `--contrast_projection_hidden` | `128` | Hidden width when projection head enabled |
| `--contrast_projection_dim` | `128` | Projection output dim (default matches embedding dim) |

Hetero contrastive training (`--reverse_mp --objective contrastive`) uses the same augmentation and InfoNCE machinery on the heterogeneous graph path.

### Morphology expert head (contrastive pretrain only)

Requires `--objective contrastive` and `--morph_expert`. Adds `L_morph_expert` (MSE on detached morphology targets) alongside InfoNCE. Logs `morph/expert_train` and `morph/expert_val` to W&B.

| Argument | Default | Description |
|----------|---------|-------------|
| `--morph_expert` | off | Enable morphology expert auxiliary loss during contrastive pretraining |
| `--morph_targets` | `local` | `local` = Tier 1 (**11** local dims incl. clustering); `local+global` = M1b; `local+tier2` = BC-only global ablation; `local+global+tier2` = M1b + BC (M3) |
| `--morph_tier0_cache` | — | Directory with `{train,val,test}_node_morphology.csv` from `scripts/precompute_morphology_tier0.py`. If omitted with global targets, tables are computed at startup |
| `--morph_tier2_cache` | — | Directory with `{train,val,test}_node_tier2.csv` from `scripts/precompute_morphology_tier2.py` (required for practical Tier 2 runs) |
| `--morph_tier2_lift` | `full` | BC endpoint lift for expert: `full` = 4 cols (`bc_sender`, `bc_receiver`, `bc_sum`, `bc_max`); `max` = `bc_max_global` only |
| `--morph_expert_loss` | `mse` | Expert regression loss: `mse` (default) or `mae` (Papagei-style L1) |
| `--morph_expert_weight` | `1.0` | Scale morphology expert loss vs InfoNCE |
| `--morph_expert_hidden` | `64` | Hidden size of each expert MLP (shared or per-block) |
| `--morph_expert_layout` | `shared` | `shared` = one MLP; `grouped` = block heads + per-block MSE (M5a) |
| `--morph_expert_group_weight_tier2` | `1.0` | Tier 2 block loss weight when `layout=grouped` (`0` disables BC gradients) |
| `--no_morph_edge_native` | off | Exclude forward `edge_attr` columns (timestamp, amount, currency, payment format) from morphology targets |

### Morphology-aware contrast (Phase M2)

Requires `--objective contrastive` and `--morph_contrast`. Merges **soft positives** (same train-split morphology bin, cross-view) with **identity** positives in edge InfoNCE. Bins are computed on **view1** forward subgraph features (detached). Logs `morph/contrast_val` on the val loader (merged InfoNCE loss).

| Argument | Default | Description |
|----------|---------|-------------|
| `--morph_contrast` | off | Enable morphology-bin soft positives in edge InfoNCE |
| `--morph_contrast_features` | `local_ego,local_degree` | Comma-separated **groups** (not individual column names): `local_ego` (cols 0–1), `local_degree` (2–7), `local_clustering` (8–10), `global_degree`, `edge_native` |
| `--morph_contrast_scope` | `local` | `local+global` enables `global_degree` lift for binning (requires Tier 0 cache or startup compute) |
| `--morph_contrast_bins` | `5` | Quantile buckets per feature dimension (edges estimated from train loader at startup) |
| `--morph_contrast_calib_batches` | `32` | Train batches used to estimate bin quantile edges |
| `--morph_contrast_max_soft_positives` | `256` | Cap same-bin positives per anchor in the InfoNCE numerator (`0` = no cap). Safe with large `--batch_size`; no dense `(B,B)` mask is allocated |
| `--morph_val_every` | `1` | Run `morph/expert_val` and `morph/contrast_val` every N epochs (`1` = every epoch). Always runs on the final epoch |
| `--morph_val_max_batches` | `0` | Cap val-loader batches per morphology val pass (`0` = full val loader). Each batch still runs two augmented views + forward(s) |

**Target dimensions (defaults):** `local` → **11** local (8 degree/ego + 3 clustering) + 4 edge-native = **15**; `local+global` → 11 + 9 global + 4 edge-native = **24**. Count-like local/global degree columns use log1p before MSE; clustering coeffs stay in **[0, 1]**. Train morph targets use the **train-split** graph only; val morph loss uses the **val-split** graph (no leakage).

Checkpoints with `--morph_expert` also store `morph_expert_state_dict`; with `--contrast_projection_head`, `contrast_projection_state_dict` is saved too (for resume only — extraction uses the encoder).

### Logging and misc

| Argument | Description |
|----------|-------------|
| `--tqdm` | Progress bars during training |
| `--testing` | Disable W&B logging |

### Linear probe (`linear_probe.py`)

| Argument | Default | Description |
|----------|---------|-------------|
| `--unique_name` | — | Embedding subfolder under `--embeddings_dir` |
| `--embeddings_dir` | `embeddings` | Root directory from extraction |
| `--class_weight` | `balanced` | `balanced`, `none`, or `model` (reads CE weights from `model_settings.json`) |
| `--probe_max_iter` | `1000` | Max iterations for sklearn logistic regression |
| `--testing` | off | Disable W&B |

### Label-efficiency probe (`scripts/label_efficiency_probe.py`)

**Why:** Full-label probes favor **M1b + clustering + projection** (0.929 AUROC; LE pending), but AML teams often have only a **fraction** of train transactions labeled. GCPAL and Papagei evaluate SSL encoders under **label scarcity** — same frozen embeddings, less train supervision. This does **not** retrain the GNN. The Jun 2026 refresh (nine encoders) shows projection beats plain M1b at all fractions; **M1b + projection** still wins at **10%** labels.

**How:** Stratified subsets of **train** labels (default 10/25/50/100%), logistic regression per fraction, threshold tuned on **full val**, metrics on val/test.

| Argument | Default | Description |
|----------|---------|-------------|
| `--unique_name` / `--unique_names` | — | One or more embedding folders (batch compare) |
| `--train_fractions` | `0.1,0.25,0.5,1.0` | Comma-separated train label fractions |
| `--class_weight` | `model` | Same as `linear_probe.py` |
| `--probe_max_iter` | `5000` | sklearn LR iterations |
| `--probe_n_jobs` | `1` | Avoid OOM on large `train.npz` (do not use `-1` on Slurm) |
| `--testing` | off | Disable W&B |

```bash
python scripts/label_efficiency_probe.py \
  --unique_names hi_morphology_global_20ep hi_morph_global_contrast_10ep_bestckpt \
  --class_weight model --model gin --testing
```

**Outputs:** `embeddings/{unique_name}/label_efficiency_results.json`, `embeddings/label_efficiency_summary.json`.

<!-- **Slurm:** [`run_label_efficiency.sh`](run_label_efficiency.sh) — `#SBATCH --mem=64G`, no GPU. Single-encoder runs merge into `label_efficiency_summary.json` (see script docstring). Full batch re-run: pass all `--unique_names` in the script. -->

**Results (Jun 2026, test AUROC, ten encoders in summary):** Source: `embeddings/label_efficiency_summary.json`. Projection encoders lead vs plain M1b at every fraction; **M1b + projection** is strongest at **10%** labels.

| Train labels | M1b+proj | Contrastive+proj | M1b | M1b+clustering | Contrastive |
|--------------|----------|------------------|-----|----------------|-------------|
| 10% | **0.918** | 0.906 | 0.896 | 0.877 | 0.818 |
| 25% | 0.922 | **0.918** | 0.910 | 0.892 | 0.849 |
| 50% | 0.919 | **0.925** | 0.915 | 0.904 | 0.857 |
| 100% | 0.922 | **0.928** | 0.919 | 0.908 | 0.863 |

**Pending label-efficiency:** `hi_morphology_global_clustering_proj_20ep_bestckpt`, `hi_morphology_global_mae_20ep_bestckpt`.

Full tables and interpretation: [`notes/morphology-metrics-plan.md`](notes/morphology-metrics-plan.md) § Label-efficiency results. Source: `embeddings/label_efficiency_summary.json`.

---

## Slurm and cluster jobs

<!-- [`smoketest.sh`](smoketest.sh) is a template Slurm script for train → extract → probe on Small-HI. Edit the `RUN_NAME`, `N_EPOCHS`, and morphology blocks at the top, then:

```bash
sbatch smoketest.sh -->
```

**W&B on batch nodes:** `main.py` loads `.env` via `load_dotenv()`, but bare `wandb login` in a shell script does **not** read `.env` automatically. Either:

```bash
set -a && source .env && set +a && wandb login "$WANDB_API_KEY"
```

or pass `--testing` to disable W&B for smoke runs. **Always** pass `--testing` to `linear_probe.py` on Slurm unless you export the API key first (probe calls `wandb.init()` before writing `probe_results.json`).

**Morphology val cost:** expert + M2 runs two val-loader passes per morph-val epoch (`morph/expert_val`, `morph/contrast_val`). Use `--morph_val_every 2 --morph_val_max_batches 10` to fit long jobs within a 6 h GPU limit. Use `--checkpoint_policy best` so extraction does not use a regressed final epoch.

<!-- [`smoketest_rerun_probe.sh`](smoketest_rerun_probe.sh) re-runs probe only when embeddings already exist.

[`run_label_efficiency.sh`](run_label_efficiency.sh) — label-efficiency on benchmark embeddings (**128G RAM**, CPU-only, nine encoders). If the job dies with `oom_kill`, request more memory or fewer `--unique_names` per job.

[`ablation_contrastive_projection_20ep .sh`](ablation_contrastive_projection_20ep%20.sh) — GPU ablation: contrastive + `--contrast_projection_head` only (result: **0.927** test AUROC, **0.144** F1).

[`ablation_m1b_projection_20ep.sh`](ablation_m1b_projection_20ep.sh) — GPU ablation: M1b + projection head (result: 0.924 test AUROC, 0.096 F1).

[`ablation_morph_expert_weight_05_10ep.sh`](ablation_morph_expert_weight_05_10ep.sh) — GPU ablation: expert+M2 @ 10 ep with `--morph_expert_weight 0.5` and `--checkpoint_policy best` (result: 0.876 test AUROC — worse than w=1.0).

[`run_precompute_morphology_tier2.sh`](run_precompute_morphology_tier2.sh) — CPU precompute for Tier 2 BC cache (**128G**; login node OOMs on Small-HI graph load).

[`run_m3_phase1_bc_expert.sh`](run_m3_phase1_bc_expert.sh) — extract + probe only for an existing M3 checkpoint (passes `$COMMON $GRAPH` like `smoketest.sh`).

[`ablation_morph_bc_only_20ep.sh`](ablation_morph_bc_only_20ep.sh) — train M3 ablation: `--morph_targets local+tier2` (BC without Tier 0 degrees).

[`ablation_morph_bc_max_global_20ep.sh`](ablation_morph_bc_max_global_20ep.sh) — train M3 ablation: M1b + `--morph_tier2_lift max` (result: 0.889 test AUROC).

[`ablation_morph_grouped_wtier2_20ep.sh`](ablation_morph_grouped_wtier2_20ep.sh) — M5a grouped heads on M1b+BC; `WTIER2=0|0.5|1 sbatch ...` for sweep.

[`ablation_m1b_clustering_20ep.sh`](ablation_m1b_clustering_20ep.sh) — M1b with Tier-1 local clustering in expert targets (11 local dims; **0.903** AUROC / **0.117** F1; LE **0.877–0.908** test AUROC).

[`ablation_m1b_clustering_m2_10ep.sh`](ablation_m1b_clustering_m2_10ep.sh) — M1b + M2 with `local_clustering` contrast bins (@ 10 ep; pending).

[`ablation_m1b_clustering_projection_20ep.sh`](ablation_m1b_clustering_projection_20ep.sh) — M1b + clustering + projection → `hi_morphology_global_clustering_proj_20ep_bestckpt` (result: **0.929** test AUROC, **0.156** F1).

[`ablation_m1b_mae_20ep.sh`](ablation_m1b_mae_20ep.sh) — M1b + `--morph_expert_loss mae` → `hi_morphology_global_mae_20ep_bestckpt` (result: **0.898** test AUROC, **0.145** F1 — below MSE baseline 0.903). -->

---

## Repository guide

How the pieces fit together for someone new to the codebase.

### End-to-end data flow

```text
formatted_transactions.csv
    → data_loading.py          (temporal train/val/test graphs)
    → main.py + training.py    (contrastive or supervised training)
    → saved-models/checkpoint_{unique_name}.tar
    → embedding_extraction.py  (frozen encoder → .npz)
    → linear_probe.py                    (full-train LR → probe_results.json)
    → scripts/label_efficiency_probe.py  (optional: partial-train LR curves)
```

Contrastive pretrain never uses AML labels in the loss loop. Downstream evaluation is **always** via frozen embeddings + linear probe (Papagei-style), except when you explicitly run supervised finetune (`--finetune`).

### Directory layout

| Path | Purpose |
|------|---------|
| `main.py` | CLI entry: parse args, build model/loaders, call `training.py` |
| `training.py` | `train_hetero_contrastive`, `train_homo_contrastive`, supervised loops; wires morphology losses |
| `train_util.py` | Loaders, seed-edge selection, checkpoints, extraction helpers |
| `contrastive_loss.py` | Edge InfoNCE, memory queue, **M2 bin-grouped soft positives** |
| `graph_augmentations.py` | Two augmented views per batch (edge drop / attr mask) |
| `models.py` | GINe, GATe, PNA, RGCN — all expose 128-d **edge** embeddings `z` |
| `embedding_extraction.py` | Write `{train,val,test}.npz` + `meta.json` |
| `linear_probe.py` | Logistic regression probe; writes `probe_results.json` |
| `scripts/label_efficiency_probe.py` | Label-efficiency curves on frozen embeddings |
| `util.py` | Shared argparse (contrastive + **all morphology flags**) |
| `morphology/` | Tier 0/1/2 metrics, expert head, contrast binning, training hooks |
| `scripts/precompute_morphology_tier0.py` | Offline `{split}_node_morphology.csv` for M1b/M2 global features |
| `scripts/precompute_morphology_tier2.py` | Offline `{split}_node_tier2.csv` (betweenness centrality; M3 Phase 0) |
| `morphology_cache/{data}/` | Cached Tier 0/2 CSVs (reuse across runs) |
| `embeddings/{unique_name}/` | `.npz` embeddings, `meta.json`, `probe_results.json` |
| `saved-models/` | Checkpoints (`checkpoint_{unique_name}.tar`) |
| `tests/` | Unit tests (`test_morphology_metrics.py`, `test_morphology_contrast.py`, …) |
| `notes/` | Design docs: contrastive plan, morphology plan, projection ablation |
<!-- | `smoketest.sh` | Slurm template for full pipeline | -->

### Morphology package (`morphology/`)

| Module | Role |
|--------|------|
| `IDS.md` | Join keys: `EdgeID`, `from_id`, `to_id` |
| `graph_access.py` | Homo/hetero accessors for forward `edge_index`, `edge_attr`, endpoints |
| `tier0_global.py` | Split-global node degrees; endpoint **lift** to seed edges (M1b) |
| `tier2_global.py` | Split-global betweenness centrality; endpoint lift (M3) |
| `tier1_local.py` | Batch-local subgraph stats on view1 (degrees, ego size, **local clustering**) |
| `expert.py` | `MorphologyExpertHead`, MSE loss, `setup_morphology_expert()` |
| `contrast.py` | M2: feature groups, quantile bins, `build_morph_bin_ids_for_seeds()` |
| `contrastive_train.py` | Training-loop glue: per-step losses, val eval, `--morph_val_*` gating |

**Where morphology attaches in training** (`training.py`):

1. Each train step: sample batch → `generate_views` → GNN forward → InfoNCE on shared seed edges.
2. If `--morph_expert`: predict detached morphology targets from `z_seed` (MSE).
3. If `--morph_contrast`: assign bin ids on view1 features → merged soft positives in InfoNCE.
4. End of epoch (optional/throttled): full val-loader passes log `morph/expert_val` and/or `morph/contrast_val`.

Bin edges for M2 are estimated once at startup from the **train** loader (`--morph_contrast_calib_batches`, default 32 batches).

### Morphology flags cheat sheet

| Goal | Flags |
|------|-------|
| Baseline contrastive | *(none)* |
| Contrastive + projection | `--contrast_projection_head --contrast_projection_hidden 128 --contrast_projection_dim 128` |
| M1b + clustering + projection (best full-label SSL) | M1b expert + projection flags above|
| M1b + MAE expert | add `--morph_expert_loss mae` (ablation: 0.898 AUROC — below MSE default) |
| M1b + projection | M1b expert flags above **and** `--contrast_projection_head …` |
| M1 local expert | `--morph_expert --morph_targets local` |
| M1b global expert | `--morph_expert --morph_targets local+global --morph_tier0_cache morphology_cache/Small-HI` |
| M1b + BC expert (M3) | `--morph_targets local+global+tier2 --morph_tier2_cache morphology_cache/Small-HI` (+ Tier 0 cache) |
| BC-only ablation | `--morph_targets local+tier2 --morph_tier2_cache morphology_cache/Small-HI` |
| BC max-only lift | above + `--morph_tier2_lift max` |
| M2 soft positives (degree bins) | `--morph_contrast --morph_contrast_features local_ego,local_degree` |
| M2 + clustering bins | add `local_clustering` to `--morph_contrast_features` (see clustering M2 example above) |
| M1b + clustering expert only | same M1b flags — clustering is automatic|
| M1b + M2 (prior recipe) | M1b expert + `--morph_contrast --morph_contrast_features local_ego,local_degree` @ **10 ep** |
| M1b + M2 + clustering | M1b expert + `--morph_contrast_features local_ego,local_degree,local_clustering` @ 10 ep |
| Faster morph val | `--morph_val_every 2 --morph_val_max_batches 10` |

Expert head weights are saved in the checkpoint but **not** used at extraction — only the GNN encoder is frozen and probed.

### Tests

```bash
python -m pytest tests/test_morphology_metrics.py tests/test_morphology_contrast.py \
  tests/test_morphology_expert_grouped.py tests/test_morphology_expert_tier2.py tests/test_morphology_tier2.py -q
```

---

## Repository map (quick reference)

| File | Role |
|------|------|
| `main.py` | Entry point for training and inference |
| `training.py` | Contrastive and supervised training loops |
| `train_util.py` | Loaders, evaluation, checkpoints, embedding extraction helpers |
| `models.py` | GINe, GATe, PNA, RGCN |
| `contrastive_loss.py` | Edge-level InfoNCE and memory queue |
| `graph_augmentations.py` | Contrastive view generation |
| `morphology/` | Morphology metrics (Tier 0 global, Tier 1 local, Tier 2 BC lift), expert head, contrastive integration |
| `scripts/precompute_morphology_tier0.py` | Offline split-global node degree tables for M1b |
| `scripts/precompute_morphology_tier2.py` | Offline split-global betweenness centrality (M3 Phase 0) |
| `embedding_extraction.py` | Frozen seed-edge embeddings → `.npz` |
| `linear_probe.py` | Sklearn logistic regression on frozen embeddings |
| `data_loading.py` | CSV → PyG graph objects and temporal splits |
| `format_kaggle_files.py` | Kaggle CSV → formatted transactions |
| `model_settings.json` | Per-architecture hyperparameters |
<!-- | `smoketest.sh` | Slurm train → extract → probe template | -->
| `notes/` | Design documents (contrastive plan, morphology metrics, [projection ablation](notes/projection-head-ablation-jun2026.md), [lit review index](notes/lit-review-index.md)) |

---

## References

### Multi-GNN (architecture and AML benchmark)

This fork builds on the Multi-GNN codebase and IBM synthetic AML datasets:

- Egressy et al., *Provably Powerful Graph Neural Networks for Directed Multigraphs*, AAAI 2024. [arXiv:2306.11586](https://arxiv.org/abs/2306.11586)
- Egressy et al., *Realistic Synthetic Financial Transactions for Anti-Money Laundering Models*, NeurIPS 2023 Datasets and Benchmarks. [arXiv:2306.16424](https://arxiv.org/abs/2306.16424)
- Upstream repository: [IBM/Multi-GNN](https://github.com/IBM/Multi-GNN)

### Downstream evaluation protocol

We adopt Papagei's **pretrain → frozen encoder → linear probe** workflow (not its PPG architecture or biosignal data). Transaction graphs use **temporal** train/val/test splits rather than Papagei's subject-level holdout — see [`notes/contrastive-learning-plan.md`](notes/contrastive-learning-plan.md) when comparing protocols.

- Pillai et al., *PaPaGei: Open Foundation Models for Optical Physiological Signals*, ICLR 2025. [arXiv:2410.20542](https://arxiv.org/abs/2410.20542)

### Additional related work

<!-- Expand as the project evolves. -->

- Hanbin et al., *Graph Contrastive Pre-training for Anti-money Laundering* (GCPAL), *International Journal of Computational Intelligence Systems*, 2024. [Springer](https://link.springer.com/article/10.1007/s44196-024-00720-4) — graph augmentations and contrastive pretraining on AML transaction networks.
- Morphology-aware contrastive pretraining (expert heads, Tier 0/1 metrics): [`notes/morphology-metrics-plan.md`](notes/morphology-metrics-plan.md).
- Contrastive projection head ablation (Jun 2026): [`notes/projection-head-ablation-jun2026.md`](notes/projection-head-ablation-jun2026.md).
- Literature ↔ implementation mapping: [`notes/lit-review-index.md`](notes/lit-review-index.md).

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).
