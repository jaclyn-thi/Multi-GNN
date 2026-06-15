# Multi-GNN — Graph foundation model extensions for AML

This repository extends [IBM Multi-GNN](https://github.com/IBM/Multi-GNN) for anti-money laundering (AML) on financial transaction graphs. It adds a **graph foundation model (GFM)** workflow on top of the original architectures ([GIN](https://arxiv.org/abs/1810.00826), [GAT](https://arxiv.org/abs/1710.10903), [PNA](https://arxiv.org/abs/2004.05718), [RGCN](https://arxiv.org/abs/1703.06103)):

1. **Self-supervised pretraining** — contrastive learning (`--objective contrastive`, default); optional projection head and/or morphology expert + morphology-aware contrast
2. **Frozen embedding extraction** — seed-edge representations to disk (`embedding_extraction.py`)
3. **Linear probing** — sklearn logistic regression on frozen features (`linear_probe.py`)
4. **Label-efficiency probing** (optional) — same embeddings, stratified train-label subsets (`scripts/label_efficiency_probe.py`)

The original **supervised** path (`--objective supervised`) remains as a baseline (~0.97 test AUROC in-GNN on Small-HI). Downstream evaluation uses **frozen embeddings + linear probe** (Papagei-style), not end-to-end finetune, unless you explicitly use `--finetune`.

---

## Documentation map

| If you want to… | Start here |
| ---------------- | ---------- |
| **Get running** | [Quick start](#quick-start) below |
| **Prepare data** (IBM, PaySim, SAML-D) | [`notes/datasets.md`](notes/datasets.md) |
| **Train / extract / probe** | [Workflows](#workflows) · [`notes/cli-reference.md`](notes/cli-reference.md) |
| **Morphology SSL** (M1/M2, flags, metrics) | [`notes/morphology-reference.md`](notes/morphology-reference.md) · [`notes/morphology-metrics-plan.md`](notes/morphology-metrics-plan.md) |
| **Dev benchmark numbers** | [`notes/results.md`](notes/results.md) |
| **Downstream eval strategy** (PaySim, SAML-D, typology) | [`notes/downstream-eval-plan.md`](notes/downstream-eval-plan.md) |
| **Contrastive design & protocol** | [`notes/contrastive-learning-plan.md`](notes/contrastive-learning-plan.md) |
| **Run on Slurm** | [Slurm](#slurm) |
| **Understand the codebase** | [Repo overview](#repo-overview) |

---

## Quick start

### Setup

```bash
conda env create -f env.yml
conda activate multignn          # on cluster: module load miniforge first
```

Configure paths in [`data_config.json`](data_config.json) (`aml_data`, `model_to_load`, `model_to_save`). Training logs to `logs/logs.log`. [W&B](https://wandb.ai/) is on by default; pass `--testing` to disable. API key via `.env` in repo root.

### Data

Format IBM Kaggle CSVs and place under `aml-data/<name>/formatted_transactions.csv` where `<name>` matches `--data` (e.g. `Small-HI`). See [`notes/datasets.md`](notes/datasets.md) for IBM, PaySim, and SAML-D.

### Primary workflow — pretrain → extract → probe

All examples use hetero GIN with `--reverse_mp --ego --ports`. Hyperparameters from [`model_settings.json`](model_settings.json).

**1. Pretrain**

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

**2. Extract frozen embeddings**

```bash
python embedding_extraction.py \
  --data Small-HI --model gin \
  --unique_name my_pretrain \
  --reverse_mp --ego --ports --tqdm
```

**3. Linear probe**

```bash
python linear_probe.py --unique_name my_pretrain --testing
```

Writes `embeddings/{unique_name}/probe_results.json`. Report **AUROC** (primary) and F1 (secondary). AML labels are not used for SSL checkpoint selection; use `--checkpoint_policy best` for morph/projection runs.

**Recommended pure-SSL add-on:** `--contrast_projection_head` (see [Workflows](#workflows)). Best dev Small-HI probe: test AUROC **0.930** ([`notes/results.md`](notes/results.md)).

---

## Workflows

### Contrastive projection head (recommended for pure SSL)

GraphCL-style projection MLP **only during InfoNCE**; extraction uses encoder `z` (128-d).

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

Then extract → probe as above.

### Morphology-aware pretraining (optional)

Adds a **morphology expert head** (label-free structural targets) and/or **M2 soft positives** in InfoNCE. Phases M0–M5, metric definitions, example commands, and flags cheat sheet: [`notes/morphology-reference.md`](notes/morphology-reference.md).

### Supervised baseline (original Multi-GNN)

```bash
python main.py \
  --data Small-HI --model gin \
  --objective supervised \
  --unique_name my_supervised \
  --save_model \
  --reverse_mp --ego --ports \
  --n_epochs 100 --tqdm
```

Use `--inference` to evaluate a saved checkpoint without training.

### Supervised finetune (secondary / ablation)

End-to-end CE fine-tune — **not** Papagei-style probing. Loads `checkpoint_{unique_name}.tar`, saves `checkpoint_{unique_name}_finetuned.tar`. Same `--data` and graph flags as the source checkpoint.

```bash
python main.py --data Small-HI --model gin --objective supervised \
  --unique_name my_pretrain --finetune --save_model \
  --reverse_mp --ego --ports --n_epochs 10 --tqdm

python embedding_extraction.py --data Small-HI --model gin \
  --unique_name my_pretrain --finetune --reverse_mp --ego --ports --tqdm

python linear_probe.py --unique_name my_pretrain_finetuned --testing
```

### PaySim transfer (external fraud)

AML checkpoint → PaySim frozen extract + linear probe. Use `embeddings/paysim/` and `--embeddings_dir embeddings/paysim`.

```bash
UNIQUE_NAME=hi_contrastive_proj_sym_20ep_bestckpt sbatch slurm/run_paysim_extract_probe.sh
```

Dev result: test AUROC **0.866** (pretrained) vs **0.730** (random init). Details: [`notes/datasets.md`](notes/datasets.md) · [`notes/downstream-eval-plan.md`](notes/downstream-eval-plan.md).

### Label-efficiency probe

```bash
python scripts/label_efficiency_probe.py \
  --unique_names hi_contrastive_proj_sym_20ep_bestckpt \
  --class_weight model --model gin --testing
```

See [`notes/cli-reference.md`](notes/cli-reference.md) and [`notes/results.md`](notes/results.md).

---

## Checkpoints and outputs

| Artifact | Path |
|----------|------|
| Checkpoint | `saved-models/checkpoint_{unique_name}.tar` |
| Finetuned checkpoint | `saved-models/checkpoint_{unique_name}_finetuned.tar` |
| Embeddings | `embeddings/{unique_name}/{train,val,test}.npz` |
| Extraction metadata | `embeddings/{unique_name}/meta.json` |
| Probe results | `embeddings/{unique_name}/probe_results.json` |
| Morphology cache | `morphology_cache/{data}/{split}_node_morphology.csv`, `{split}_node_tier2.csv` |

`--unique_name` is required for `--save_model`, `--finetune`, extraction, and probing.

Full CLI flag list: [`notes/cli-reference.md`](notes/cli-reference.md).

---

## Slurm

Scripts in [`slurm/`](slurm/) (local-only). Submit from repo root:

```bash
sbatch slurm/ablation_contrastive_proj_sym_8192neg_20ep.sh
UNIQUE_NAME=hi_contrastive_proj_sym_20ep_bestckpt sbatch slurm/run_paysim_extract_probe.sh
sbatch slurm/run_saml_d_supervised_smoke.sh
```

`mit_normal_gpu` cap is **6 h** — one ablation at a time. On `TIME_LIMIT`, switch script to `mit_preemptable` and `-t 12:00:00`.

**W&B on batch nodes:** export API key (`set -a && source .env && set +a && wandb login "$WANDB_API_KEY"`) or pass `--testing`. Always `--testing` on `linear_probe.py` unless the key is exported.

**Morphology jobs:** `--morph_val_every 2 --morph_val_max_batches 10` and `--checkpoint_policy best` help fit 6 h GPU limits.

**Label-efficiency:** ~128G RAM, CPU-only, `--probe_n_jobs 1`.

---

## Repo overview

```text
formatted_transactions.csv
    → data_loading.py          (temporal train/val/test graphs)
    → main.py + training.py    (contrastive or supervised)
    → saved-models/checkpoint_{unique_name}.tar
    → embedding_extraction.py  (frozen encoder → .npz)
    → linear_probe.py          (sklearn LR → probe_results.json)
```

| Path | Role |
|------|------|
| `main.py` | CLI entry; training and inference |
| `training.py` | Contrastive and supervised loops; morphology losses |
| `train_util.py` | Loaders, checkpoints, extraction helpers |
| `contrastive_loss.py` | Edge InfoNCE, memory queue, M2 soft positives |
| `models.py` | GINe, GATe, PNA, RGCN (128-d edge embeddings) |
| `embedding_extraction.py` | Frozen seed-edge `.npz` + `meta.json` |
| `linear_probe.py` | Logistic regression probe |
| `data_loading.py` | CSV → PyG graphs and temporal splits |
| `dataset_specs.py` / `dataset_splits.py` | Per-dataset specs and split helpers |
| `morphology/` | Tier 0/1/2 metrics, expert head, M2 contrast |
| `format_*.py` | Kaggle, PaySim, SAML-D formatters |
| `scripts/` | Precompute, validation, label-efficiency, diagnostics |
| `notes/` | Design docs, datasets, results, CLI reference |
| `tests/` | Unit tests |

```bash
python -m pytest tests/ -q
```

---

## References

- Egressy et al., *Provably Powerful GNNs for Directed Multigraphs*, AAAI 2024. [arXiv:2306.11586](https://arxiv.org/abs/2306.11586)
- Egressy et al., *Realistic Synthetic Financial Transactions for AML*, NeurIPS 2023. [arXiv:2306.16424](https://arxiv.org/abs/2306.16424)
- [IBM/Multi-GNN](https://github.com/IBM/Multi-GNN) (upstream)
- Pillai et al., *PaPaGei*, ICLR 2025 — frozen encoder + linear probe protocol. [arXiv:2410.20542](https://arxiv.org/abs/2410.20542)
- You et al., GraphCL, ICML 2020. [arXiv:2010.13902](https://arxiv.org/abs/2010.13902)
- Hanbin et al., GCPAL, 2024. [Springer](https://link.springer.com/article/10.1007/s44196-924-00720-4)

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).
