# Multi-GNN — Graph foundation model extensions for AML

This repository extends [IBM Multi-GNN](https://github.com/IBM/Multi-GNN) for anti-money laundering (AML) on financial transaction graphs. It keeps the original Multi-GNN architectures and graph adaptations ([GIN](https://arxiv.org/abs/1810.00826), [GAT](https://arxiv.org/abs/1710.10903), [PNA](https://arxiv.org/abs/2004.05718), [RGCN](https://arxiv.org/abs/1703.06103)) and adds a **graph foundation model (GFM)** workflow:

1. **Self-supervised pretraining** — contrastive learning on transaction graphs (`--objective contrastive`, the default)
2. **Frozen embedding extraction** — seed-edge representations written to disk (`embedding_extraction.py`)
3. **Linear probing** — sklearn logistic regression on frozen features, Papagei-style (`linear_probe.py`)

The original **supervised** Multi-GNN path (`--objective supervised`) remains available as a baseline and ablation. For design rationale and implementation status, see [`notes/contrastive-learning-plan.md`](notes/contrastive-learning-plan.md).

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

Reads `embeddings/{unique_name}/{train,val,test}.npz` and writes metrics to `embeddings/probe_results.json`.

During contrastive pretraining, monitor **training loss** only — AML metrics are intentionally not used for encoder checkpoint selection. After extraction, report **AUROC** (primary) and **F1** (secondary) from the probe.

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
| Probe results | `embeddings/probe_results.json` |

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

Hetero contrastive training (`--reverse_mp --objective contrastive`) uses the same augmentation and InfoNCE machinery on the heterogeneous graph path.

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

---

## Repository map

| File | Role |
|------|------|
| `main.py` | Entry point for training and inference |
| `training.py` | Contrastive and supervised training loops |
| `train_util.py` | Loaders, evaluation, checkpoints, embedding extraction helpers |
| `models.py` | GINe, GATe, PNA, RGCN |
| `contrastive_loss.py` | Edge-level InfoNCE and memory queue |
| `graph_augmentations.py` | Contrastive view generation |
| `embedding_extraction.py` | Frozen seed-edge embeddings → `.npz` |
| `linear_probe.py` | Sklearn logistic regression on frozen embeddings |
| `data_loading.py` | CSV → PyG graph objects and temporal splits |
| `format_kaggle_files.py` | Kaggle CSV → formatted transactions |
| `model_settings.json` | Per-architecture hyperparameters |
| `notes/` | Design documents (contrastive plan, morphology metrics, etc.) |

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
- *Morphology-aware pretraining signals* — planned; see [`notes/morphology-metrics-plan.md`](notes/morphology-metrics-plan.md).

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).
