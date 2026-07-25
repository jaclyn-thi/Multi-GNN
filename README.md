# Multi-GNN — Graph foundation model extensions for AML

This repository extends [IBM Multi-GNN](https://github.com/IBM/Multi-GNN) for anti-money laundering on financial transaction graphs ([GIN](https://arxiv.org/abs/1810.00826), [GAT](https://arxiv.org/abs/1710.10903), [PNA](https://arxiv.org/abs/2004.05718), [RGCN](https://arxiv.org/abs/1703.06103)).

## 1. Project purpose and representation

**Primary thesis system:** accounts as nodes, transactions as edges (edge-centric graphs).

| Path | Role |
|------|------|
| **Supervised Multi-GNN parity** | Paper-faithful Multi-GIN+EU baseline (`--objective supervised --supervised_head legacy`) |
| **Edge-centric contrastive pretraining** | Default SSL path: two augmented views → frozen edge embeddings → linear probe |
| **Transaction-node GCPAL-inspired diagnostics** | Isolated under `gcpal_txn_node/` / `scripts/gcpal_txn_node_*.py` — **not** an exact GCPAL reproduction |

## 2. Current scientific status (brief)

- **Multi-GIN+EU mean reproduced** on Small-HI with paper-faithful **ports, no-TDS** configuration (50ep, best-val minority F1 ckpt, paper_argmax test). Paper target **0.6479 ± 0.0122**; formal seeds 1–3: **0.663 / 0.718 / 0.598** → **0.660 ± 0.060**. Mean yes; paper’s low variance not. Older **TDS-on** supervised is **not** paper-compatible.
- **Strongest semantically valid edge-centric contrastive** tested so far: **corrected reverse-edge TDS + `--preserve_seed_edges`** (does **not** claim GCPAL reproduction).
- **Positive-complete transaction-node** A/B five-epoch scouts are **preliminary diagnostics** only.
- **Temporal** train/val/test is the primary thesis protocol; **random-40** is diagnostic-only.

Full claims and tables: [`notes/current_protocol_recent_runs_summary.md`](notes/current_protocol_recent_runs_summary.md) · protocols: [`notes/thesis_protocol_families.md`](notes/thesis_protocol_families.md) · evaluation: [`notes/evaluation_protocols.md`](notes/evaluation_protocols.md).

## 3. Setup and data

```bash
conda env create -f env.yml   # or environment.yml if present
conda activate multignn
```

Configure [`data_config.json`](data_config.json) (`aml_data`, `model_to_load`, `model_to_save`). Place formatted IBM CSVs under `aml-data/<name>/formatted_transactions.csv`. Details: [`notes/datasets.md`](notes/datasets.md). Cluster: [`notes/engaging-cluster-config.md`](notes/engaging-cluster-config.md).

> `slurm/` and `tests/` may be local-only. Prefer the `python …` commands below.

## 4. Minimal canonical commands

### Paper-faithful Small-HI Multi-GIN+EU (supervised)

```bash
python main.py \
  --data Small-HI --model gin \
  --objective supervised --supervised_head legacy \
  --emlps --reverse_mp --ego --ports \
  --unique_name small_hi_legacy_supervised_gin_emlps_ports_50ep_seed1 \
  --save_model --checkpoint_policy best \
  --n_epochs 50 --batch_size 8192 --num_neighs 100 100 \
  --seed 1 --tqdm
```

Omit `--tds` (edge_dim=6). Select `checkpoint_best_val_f1.tar`. Paper comparison uses **two-class argmax** F1 — see [`notes/evaluation_protocols.md`](notes/evaluation_protocols.md).

### Current edge-centric contrastive (corrected TDS + preserve seeds)

```bash
python main.py \
  --data Small-HI --model gin \
  --objective contrastive \
  --emlps --reverse_mp --ego --ports --tds \
  --correct_reverse_edge_features --preserve_seed_edges \
  --contrast_projection_head --contrastive_asymmetric \
  --contrastive_num_neg_samples 8192 --contrastive_memory_bank_size 0 \
  --contrastive_accum_steps 4 --batch_size 8192 \
  --unique_name gin_emlps_ports_tds_corrected_preserve_seed_asym_proj_8192neg_queue0_40ep_seed2 \
  --save_model --n_epochs 40 --seed 2 --tqdm
```

Ablations and batch-size diagnostics: [`notes/thesis_protocol_families.md`](notes/thesis_protocol_families.md). Full flags: [`notes/cli-reference.md`](notes/cli-reference.md).

### Extract + downstream probe

```bash
python embedding_extraction.py \
  --data Small-HI --model gin \
  --unique_name <run> \
  --reverse_mp --ego --ports --emlps --tds \
  --correct_reverse_edge_features \
  --representation_source post_embedding --tqdm

python linear_probe.py --unique_name <run> --class_weight model --model gin --testing
```

Use matching graph flags for the checkpoint. For pre-`embedding_head` tensors: `--representation_source pre_embedding_3h`.

## 5. Outputs and checkpoints

| Artifact | Location |
|----------|----------|
| Checkpoints | `saved-models/<unique_name>/` (best-val: `checkpoint_best_val_f1.tar` when policy=`best`) |
| Embeddings | `embeddings/<unique_name>/` (optional `pre_embedding_3h/` subdir) |
| Probe JSON | `embeddings/<unique_name>/probe_results.json` |
| Diagnostics / registry | `results/diagnostics/` · registry note [`notes/thesis_experiment_registry.md`](notes/thesis_experiment_registry.md) |

## 6. Documentation map

| Audience | Start here |
|----------|------------|
| New researcher | This README → [`notes/datasets.md`](notes/datasets.md) → [`notes/cli-reference.md`](notes/cli-reference.md) |
| Find an experiment | [`notes/README.md`](notes/README.md) → [`notes/thesis_experiment_registry.md`](notes/thesis_experiment_registry.md) |
| Scientific comparability | [`notes/evaluation_protocols.md`](notes/evaluation_protocols.md) · [`notes/thesis_protocol_families.md`](notes/thesis_protocol_families.md) |
| Current claims | [`notes/current_protocol_recent_runs_summary.md`](notes/current_protocol_recent_runs_summary.md) |
| GCPAL-inspired path | [`notes/thesis_protocol_families.md`](notes/thesis_protocol_families.md#transaction-node-gcpal-inspired-path) (**not exact reproduction**) |
| Doc sync audit (2026-07-22) | [`notes/documentation_audit_2026-07-22.md`](notes/documentation_audit_2026-07-22.md) |
