# Contrastive Learning Implementation Plan for Multi-GNN

## References
- Hanbin et al. (2024): "Graph Contrastive Pre-training for Anti-money Laundering" - Source of graph augmentation techniques and InfoNCE application to AML graphs.
- Papagei (PPG foundation model) - Reference protocol for downstream evaluation: **extract frozen representations** after SSL pretrain, then **linear probing** (e.g. logistic regression for binary tasks) rather than end-to-end fine-tuning on the downstream loss. Subject-level splits in Papagei; we use **temporal** train/val/test on transactions (document when comparing).

## Research framing (GFM north star)

**Primary research question:** Can we pretrain a **graph foundation model (GFM)** on large-scale **financial transaction graph** data (eventually broad / synthetic / largely unlabeled), then **use** the frozen representations for downstream finance tasks (AML, fraud, and others)—primarily via **linear probing** when task labels are available on an eval set?

**Not the main question:** “Does contrastive pretraining on AML-labeled graphs beat supervised CE on the same graphs?” AML is a **development benchmark** and one downstream task, not the definition of pretrain success.

**Papagei-style split (conceptual):**
- **Pretrain:** self-supervised objectives on domain data (here: contrastive / InfoNCE on transactions; future: morphology and other intrinsic signals — see `notes/morphology-metrics-plan.md`). Pretrain quality is **not** measured by downstream task F1 during this phase. Papagei does **not** use downstream-task checkpoint selection during SSL pretrain.
- **Adapt (primary protocol):** load a **frozen** pretrained encoder; **extract** fixed edge embeddings for train/val/test seed transactions; train a **separate** simple classifier on those features (binary AML → **logistic regression**, report **AUROC** and F1). The GNN is not updated on the downstream CE loss in this protocol.
- **Adapt (secondary / optional in repo today):** in-graph `--finetune --objective supervised` (full-model CE fine-tuning). Useful as an ablation or Multi-GNN baseline comparison, but **not** the main GFM evaluation story—it conflates representation quality with extra encoder tuning.

**Explicit decisions (May 2026):**
1. **No** per-epoch AML F1 or AML-val **encoder** selection during contrastive pretrain (declined; ties checkpoints to one task).
2. **No** per-epoch AML linear probe **during** pretrain for the same reason (monitoring only would be optional; not planned).
3. **Yes** to Papagei-style **post-hoc** extraction + linear probe as the **primary** downstream path for task benchmarks (AML now; fraud etc. when labels exist).

**Using AML-labeled data today:** Convenient for engineering and benchmarking; contrastive loss remains **label-agnostic**. Longer term, pretrain should move toward larger finance graphs without leaning on task labels for sampling or checkpointing.

---

## Current System Overview

### Data Loading & Splitting
- **Data Source**: CSV files with transaction records (rows) and features like sender/receiver, bank, currency, amount, etc.
- **Labels**: `Is Laundering` (0=legitimate, 1=illicit)
- **Temporal Split**: Data is split by timestamp into train/val/test (preventing leakage)
- **Imbalanced Classes**: Illicit transactions are the minority class (imbalance varies by dataset)
- **Scale**: Medium datasets have ~32 million transactions

### Current Training
- **Original baseline path**: CrossEntropyLoss with class weights (`w_ce1`, `w_ce2`) and F1-score logging on the AML edge classification task
- **Graph form** (`--reverse_mp`): homo (off) vs hetero with reverse message passing (on)
- **Training objective** (`--objective`, default `contrastive`): contrastive pretraining vs supervised AML classification
- **Checkpoint init** (`--finetune`): load `checkpoint_{unique_name}.tar` before training for either objective (e.g. continued contrastive on more data or supervised adaptation)
- **Contrastive pretrain logging**: `loss/train` only (no per-epoch AML F1 by design — see Research framing)
- **Primary downstream (Papagei-style)**: `embedding_extraction.py` → `linear_probe.py` on `embeddings/{unique_name}/`
- **Secondary downstream (implemented)**: homo/hetero `--objective supervised` with F1 logging; optional `--finetune` loads a pretrain checkpoint then runs **end-to-end** CE (all weights trainable—not linear probe)
- **Subgraph Sampling**: `LinkNeighborLoader` with configurable `num_neighbors` for batch sampling
  - prevents computational overload for large graphs
  - samples k-hop neighbors around seed edges
- **Models**: GINe, GATe, PNA, RGCN
  - all now support an embedding head plus a classifier head
  - all expose dual behavior via `return_embeddings`

### Key Functions
- `get_loaders()` in `train_util.py`: Creates `LinkNeighborLoader` instances
- `resolve_training_setup()` / `validate_training_setup()` in `train_util.py`: map CLI flags to graph form + objective
- `train_homo_contrastive()` / `train_homo_supervised()` in `training.py`: homogeneous contrastive pretraining vs supervised CE + F1
- `train_hetero_contrastive()` / `train_hetero_supervised()` in `training.py`: heterogeneous contrastive pretraining vs supervised F1
- `get_hetero_seed_edge_ids()`, hetero branch of `attach_edge_id_from_batch()` in `train_util.py`
- `train_gnn()` in `training.py`: dispatches on `TrainingSetup` (not `reverse_mp` alone)
- `evaluate_homo()` / `evaluate_hetero()` in `train_util.py`: evaluation helpers (`return_embeddings=False` for classifier mode)

---

## Implementation Plan

The six phases below are preserved from the original implementation plan, but each phase is now updated to reflect what has actually been implemented, where the design diverged, and what still remains.

### Phase 1: Contrastive Loss Functions
**File**: `train_util.py` (or new file `contrastive_loss.py`)

**Original goal**: Implement contrastive loss functions, starting with InfoNCE and leaving room for triplet and pairwise alternatives.

**Current status**: Largely implemented, but the codebase has converged on a more specific loss design than the original plan anticipated.

#### What has been implemented
1. **Edge-level InfoNCE loss**
   - Implemented in `contrastive_loss.py`
   - Operates on transaction / edge embeddings rather than node embeddings
   - Treats the same surviving seed edge across augmented views as the positive pair

2. **Seed-edge-only contrastive loss**
   - Instead of computing loss over all message-passing edges in the sampled subgraph, the current path:
     - recovers the seed edges that formed the batch
     - keeps only seed edges that survive in both views
     - aligns those by stable `edge_id`
     - computes InfoNCE only on those aligned seed edges
   - This was a major design decision added during implementation to stop the loss from scaling with the full sampled subgraph

3. **GPU-batched sampled negatives**
   - Replaced the older CPU-side negative sampling path
   - Negatives are sampled on GPU in row batches
   - This addressed an important CPU bottleneck in the hot path

4. **Optional memory queue**
   - Implemented as `EdgeMemoryQueue`
   - Stores prior detached seed-edge embeddings and IDs
   - Provides extra negatives without making the current batch larger

#### Important divergences from the original plan
- We did **not** keep triplet loss and pairwise contrastive loss as near-term active alternatives
- We standardized on **edge-level InfoNCE**
- The original plan described negatives mostly as "all other samples in the batch"; the implemented path now uses:
  - sampled GPU negatives
  - optional queue negatives
- The temperature is currently effectively fixed in the training call path rather than exposed through a broader contrastive-loss config system

#### Why the design changed
- The original loss scaling caused OOM when trying to use larger `batch_size` or `contrastive_num_neg_samples=0`
- CPU negative sampling became a bottleneck
- The seed-edge-only design was the key change that made larger batches and higher fanout practical

#### Remaining work in this phase
- optionally expose more loss hyperparameters cleanly
- optionally add richer contrastive diagnostics such as positive/negative similarity summaries
- revisit structural positive tiers only if pretrain-native eval signals a need

---

### Phase 2: Model Architecture Changes
**File**: `models.py`

**Original goal**: Add an embedding output and preserve a classifier head so the model can support both pretraining and downstream prediction.

**Current status**: Implemented.

#### What has been implemented
1. **Embedding head added to all main model classes**
   - GINe
   - GATe
   - PNA
   - RGCN

2. **Dual output mode**
   - `return_embeddings=True` returns learned edge embeddings
   - `return_embeddings=False` routes those embeddings through the classifier head

3. **Current architecture shape**
   - `Input -> GNN layers -> embedding head -> classifier head`

#### Important divergences from the original plan
- The original plan described this as a future change; it is now already part of the current codebase
- The embedding head is in place across the main models, not just in a single proof-of-concept model
- The embedding dimension is not yet surfaced as cleanly/configurably as the original plan envisioned

#### Why this matters
- It made contrastive pretraining possible without throwing away the downstream AML classification path
- It is the key architectural bridge needed for later fine-tuning and evaluation

#### Remaining work in this phase
- optionally make embedding-dimension selection cleaner in config
- document encoder vs classifier weights for GFM release (Phase 5 / 6)

#### Phase 2b: Graph Augmentations (Following Hanbin et al. 2024)
**File**: `data_loading.py` or new file `graph_augmentations.py`

**Original goal**: Support multiple graph views for contrastive learning, potentially including KNN-based views.

**Current status**: Partially implemented with a simpler practical design.

#### What has been implemented
1. **Two stochastic views per batch**
   - generated in the training loop
   - currently use independent random edge dropping
   - currently use edge attribute masking

2. **Edge identity preservation**
   - stable `edge_id` handling now drives positive-pair alignment across views

#### Important divergences from the original plan
- We did **not** implement a third KNN graph view
- We did **not** preserve full identical edge ordering across the two views
- Instead:
  - independent edge dropping is allowed
  - seed edges that disappear from one view are skipped
  - surviving seed edges are aligned by stable ID

#### Why the design changed
- The original "preserve full ordering" framing was too rigid for the augmentations we wanted
- Matching by stable edge ID was cleaner and more robust

#### Remaining work in this phase
- optionally evaluate whether additional augmentations are useful
- only revisit KNN or richer structural views if they offer a clear benefit over the current simpler setup

---

### Phase 3: Data Loader Modifications
**File**: `data_loading.py`

**Original goal**: Modify data loading so contrastive learning can operate on graph batches with the right metadata and pair structure.

**Current status**: Implemented in a more targeted form than originally planned.

#### What has been implemented
1. **View generation remains in the training loop**
   - This matches the original plan
   - The data loader still provides sampled subgraphs; augmentation is applied afterward

2. **Stable edge ID handling**
   - `add_arange_ids(...)`
   - `attach_edge_id_from_batch(...)`
   - `get_homo_seed_edge_ids(...)`
   - `select_shared_seed_edge_embeddings(...)`

3. **Seed-edge-aware batch handling**
   - The loss no longer treats every message-passing edge in the sampled subgraph as an anchor
   - The effective contrastive anchors are the surviving shared seed edges only

4. **Loader tuning hooks**
   - `loader_num_workers` is configurable
   - persistent workers are enabled when workers are used
   - homogeneous contrastive batches use non-blocking transfer to GPU

#### Important divergences from the original plan
- The original plan described positive pairs more broadly; the implemented path is now explicitly **seed-edge identity based**
- The loader work ended up being more about **preserving and recovering stable IDs** than about changing the high-level loader API
- Contrastive pair definition now depends on:
  - seed-edge recovery
  - stable edge IDs
  - shared-survival filtering across views

#### What we learned during implementation
- The real bottleneck was not just "data loading for contrastive learning" in the abstract
- The largest practical wins came from:
  - not scaling the loss with all sampled message-passing edges
  - moving negative sampling off CPU
- Neighbor sampling and loader throughput still matter, but the loss redesign was the bigger unlock

#### Remaining work in this phase
- revisit `pin_memory`
- continue tuning `loader_num_workers`
- continue profiling startup preprocessing separately from steady-state training
- eventual **unlabeled or label-agnostic** seed sampling for pretrain on non-AML corpora (hetero helpers for forward/reverse sync are done in Phase 4b)

---

### Phase 4: Training Loop Refactoring
**File**: `training.py`

**Original goal**: Refactor training to support contrastive pretraining followed by downstream evaluation.

**Current status**: **Phase 4a (decoupled controls) and Phase 4b (hetero contrastive) are done.** Contrastive loops are pretrain-only (`loss/train`). Phase 5 focuses on **Papagei-style downstream** (frozen extract + linear probe), not AML metrics inside contrastive training.

#### What has been implemented
1. **Homogeneous contrastive pretraining loop**
   - `train_homo_contrastive()` runs contrastive pretraining on homogeneous graphs
   - uses two augmented views
   - computes embeddings
   - filters to shared seed edges
   - computes edge-level InfoNCE

2. **Practical training controls**
   - `--contrastive_asymmetric`
   - `--gradient_checkpointing`
   - `--contrastive_num_neg_samples`
   - `--contrastive_accum_steps`
   - `--contrastive_memory_bank_size`

3. **Current logging**
   - contrastive train loss
   - first-batch subgraph diagnostics
   - first-batch seed-edge filtering diagnostics
   - W&B logging for contrastive train loss

#### Important divergences from the original plan
- The training loop is no longer just a straightforward "swap CE for contrastive loss" refactor
- It now contains a distinct contrastive training regime with:
  - asymmetric mode
  - queue support
  - GPU negative sampling
  - seed-edge filtering
- The homogeneous AML F1 logging was not just disabled for convenience; the contrastive loop itself is fundamentally a pretraining loop (supervised homo F1 is on `train_homo_supervised()`)

#### Phase 4a: Decouple Graph Form, Objective, and Evaluation
**Status: implemented (May 2026).**

Quick reference — training matrix:

| | `--objective contrastive` (default) | `--objective supervised` |
|--|-------------------------------------|---------------------------|
| **homo** (`--reverse_mp` off) | `train_homo_contrastive` (`loss/train`) | `train_homo_supervised` (+ F1) |
| **hetero** (`--reverse_mp` on) | `train_hetero_contrastive` (`loss/train`; forward anchors) | `train_hetero_supervised` (+ F1) |

CLI / code touchpoints:
- `--objective {contrastive,supervised}` in `util.py` (default `contrastive`); `--reverse_mp` is graph form only
- `--finetune` loads checkpoint for **either** objective (not tied to supervised)
- `train_gnn()` → `resolve_training_setup()` → dispatch; W&B logs `graph_form`, `objective`, `loss` (`infonce` vs `ce`)
- `debug_contrastive_one_batch.py` requires homo + contrastive

Deferred by design (see Research framing; detailed in Phase 5):
- per-epoch AML F1 or AML-driven checkpointing during contrastive pretrain
- making `--finetune` imply supervised-only (`--finetune` can init continued contrastive or downstream supervised)

Control flow (implemented):
1. **Graph form** — `--reverse_mp`
2. **Training objective** — `--objective {contrastive,supervised}`
3. **Task metrics (AML F1, AUROC, etc.)** — **downstream only** (planned: extract + sklearn probe; optional: `--objective supervised` / `--finetune`)

Completed in 4a:
1. Split `train_homo_contrastive` / `train_homo_supervised`; renamed hetero path to `train_hetero_supervised`
2. Classifier-mode calls explicit in supervised train/eval (`return_embeddings=False`)
3. `--finetune` documented and unchanged in behavior: checkpoint init before any objective
4. Homo/hetero **supervised** paths log `f1/train`, `f1/validation`, `f1/test`, `best_test_f1`
5. Objective switch independent of graph form (hetero + contrastive supported in 4b)

#### Phase 4b: Add Heterogeneous Contrastive Training
**Status: implemented (May 2026).**

Run: `--reverse_mp --objective contrastive` (default objective is already `contrastive`).

Design (as implemented):
1. `train_hetero_contrastive(...)` mirrors homo contrastive (AMP, queue, accum, asymmetric mode).
2. Reverse edges participate in **message passing** only; InfoNCE anchors are **forward** `('node', 'to', 'node')` embeddings.
3. `get_hetero_seed_edge_ids(batch, loader_data)` uses forward `input_id` + global id column.
4. `attach_edge_id_from_batch(batch, loader_data)` maps reverse `add_arange_ids` cols to the same transaction `edge_id` as forward (offset by global forward edge count).
5. `generate_views` on `HeteroData` drops forward edges at random per view and removes reverse edges with the same transaction `edge_id` (synchronized aug).
6. `EdgeMemoryQueue` stores forward seed embeddings only (`enqueue` after loss, same as homo).

Code touchpoints: `training.py` (`train_hetero_contrastive`), `train_util.py` (`FORWARD_EDGE_TYPE`, `REVERSE_EDGE_TYPE`, `get_hetero_seed_edge_ids`, hetero branch of `attach_edge_id_from_batch`), `graph_augmentations.py` (`_hetero_random_edge_drop_view`).

#### What comes after Phase 4 (pretrain → extract → linear probe)

**Primary benchmark workflow (Papagei-aligned):**

| Step | Action | Metrics |
|------|--------|---------|
| 1. Pretrain | `--objective contrastive` (+ `--reverse_mp` as needed) | `loss/train`; save `checkpoint_{unique_name}.tar` |
| 2. Extract | Load checkpoint; `model.eval()`; forward with `return_embeddings=True` on seed edges per split | Save `Z`, `y`, `edge_id` per split (`.npz`) |
| 3. Linear probe | `sklearn` logistic regression on `Z_train`, `y_train` | AUROC + F1 on val/test |

**Optional / secondary:** `--finetune --objective supervised` (full in-GNN CE)—not the main GFM claim.

**Science comparisons (same AML graph, different init for step 2–3):**
- SSL pretrain checkpoint → extract → probe (**main**)
- Random-init (or untrained) encoder → extract → probe (control for probe protocol)
- Supervised-from-scratch checkpoint → extract → probe (optional)
- Original Multi-GNN: `--objective supervised` from scratch (in-GNN CE + F1; separate baseline)

---

### Phase 5: Downstream adaptation & evaluation
**Files**: new `embedding_extraction.py` (or `extract_embeddings.py`), `train_util.py` (shared batch/mask logic with `evaluate_*`), `linear_probe.py` (or probe module); existing `training.py` for optional in-GNN paths

**Original goal** (from early plan): Evaluate learned embeddings and compare against baseline AML classification — including per-epoch AML logging during contrastive pretrain.

**Current status**: Contrastive pretrain (Phases 1–4) and **in-GNN supervised** paths exist. **Phase 5a** (`embedding_extraction.py`) and **Phase 5b** (`linear_probe.py`) implement the primary Papagei-style downstream path (frozen extract → sklearn logistic regression).

#### Design decisions (May 2026)

**Declined during contrastive pretrain:**
- Per-epoch AML F1 (untrained head or frozen linear probe) and encoder checkpoint selection on AML val—ties the foundation to one task; Papagei does not do this.

**Primary downstream protocol (Papagei-aligned):**
1. Pretrain with SSL (contrastive) only.
2. **Extract** frozen embeddings `z` per seed transaction (includes `embedding_head` output; GNN in `eval()`, no augmentations, `torch.no_grad()`).
3. **Linear probe** with sklearn `LogisticRegression` on train features; evaluate AUROC (Papagei binary default) and F1 (Multi-GNN tradition) on val/test.
4. Classifier is **not** the in-model MLP head trained with CE unless explicitly running the secondary path below.

**Secondary path (already in repo; optional ablation):**
- `--finetune --objective supervised`: loads checkpoint, trains **all** GNN + embedding + classifier weights with CE, logs `f1/*`. This is **end-to-end fine-tuning**, not Papagei linear probing. Do not treat as the primary GFM result.

#### Phase 5a: Frozen embedding extraction (implemented)

**Goal:** For each split (train / val / test), produce aligned arrays `Z`, `y`, and optional `edge_id` from a fixed contrastive checkpoint.

**Reuse from `evaluate_homo` / `evaluate_hetero`:**
- Seed-edge identification via `input_id` + global id column in `edge_attr`
- Mask seed rows on forward edges (hetero); strip synthetic id columns; optional Small_J/Q missing-edge patch
- **Difference from eval:** `return_embeddings=True`; collect `(edge_id, z, y)` instead of classifier `argmax`

**Extraction rules:**
- `model.eval()`; no `generate_views`
- Same `batch_size` / `num_neighs` as pretrain (embeddings depend on sampled subgraph—document in `meta.json`)
- Train loader `shuffle=False` for reproducibility
- **Dedupe** by global `edge_id` after each split pass; **coverage check** vs expected split seeds

**Hetero:** embeddings from `('node', 'to', 'node')` only (same as contrastive anchors).

**Artifacts (example layout):**
```text
embeddings/{unique_name}/
  train.npz   # Z, y, edge_id
  val.npz
  test.npz
  meta.json   # embedding_dim, model, reverse_mp, num_neighs, checkpoint path, data name
```

**CLI:** `python embedding_extraction.py --data … --model … --unique_name <pretrain_run> [--reverse_mp] [--embeddings_dir embeddings]`

Writes `embeddings/{unique_name}/{train,val,test}.npz` and `meta.json`.

#### Phase 5b: Linear probing (implemented)

**Goal:** Papagei-style downstream classifier on frozen features—separate from the PyTorch `classifier` module.

- **CLI:** `python linear_probe.py --unique_name <same as extraction> [--embeddings_dir embeddings] [--class_weight balanced|model|none]`
- Fit: `LogisticRegression` on `train.npz` (`Z`, `y`); default `class_weight=balanced`; use `--class_weight model --model gin` to mirror `w_ce1`/`w_ce2`
- Metrics logged: **AUROC**, **F1**, precision, recall on train/val/test; W&B keys `aml_probe/linear/{split}/*`
- Writes `embeddings/{unique_name}/probe_results.json`
- No backprop through the GNN; probe weights are sklearn-only

**Other tasks later:** fraud or regression would swap the probe (e.g. ridge for regression) and metrics; each task needs its own labels on the eval set. Pretrain on unlabeled data → extraction still works; probe only where labels exist.

#### Phase 5c: Label-efficiency probing (implemented)

**Goal:** Compare frozen encoders when only a **fraction** of train AML labels are available for the downstream classifier (GCPAL / Papagei / RWTH motivation) — **without** retraining the GNN.

- **Script:** `scripts/label_efficiency_probe.py` (reuses `linear_probe.py` fit/eval helpers)
- **Defaults:** train fractions `0.1, 0.25, 0.5, 1.0` (stratified subsample); val threshold tuning unchanged
- **Batch:** `--unique_names` loops multiple embedding dirs; writes per-run JSON + `embeddings/label_efficiency_summary.json`
- **Slurm:** `run_label_efficiency.sh` (`--mem=128G`, `--probe_n_jobs 1`)
- **Not a replacement** for Phase 5b full-label `probe_results.json`; run both when comparing M1b vs M2 under label scarcity

**Results (Jun 2026):** M1b (`hi_morphology_global_20ep`) beats contrastive and all M2 variants at 10/25/50/100% train fractions (test AUROC). Largest gap at **10%** labels: M1b **0.896** vs contrastive **0.818** (+0.078). M2 does not outperform M1b under scarcity. See [`morphology-metrics-plan.md`](morphology-metrics-plan.md) § Label-efficiency results.

#### What is implemented today (secondary path only)

1. `train_homo_supervised` / `train_hetero_supervised` — in-GNN CE + per-epoch `f1/*`, val-driven `save_model`
2. `evaluate_homo` / `evaluate_hetero` — classifier mode, not feature export
3. `--finetune` — full-state load + supervised training (not frozen extract + sklearn)

#### Remaining work in this phase (priority order)

1. ~~**`extract_seed_embeddings_{homo,hetero}`** + `embedding_extraction.py`**~~ (done)
2. ~~**`linear_probe.py`**~~ (done)
3. **Smoke test (end-to-end):** short pretrain → `embedding_extraction.py` → `linear_probe.py` on Small_HI (homo + hetero)
4. **README / runbook:** Papagei three-step workflow; clarify temporal vs subject-level splits
5. **Optional:** encoder-only export in checkpoint for release
6. **Optional:** `--freeze-encoder` + train only in-model linear layer (still not identical to sklearn probe unless head is a single `Linear`)
7. ~~**Future:** pretrain checkpoint policy via contrastive val / morphology~~ → **done:** `--checkpoint_policy best` in `train_util.py` (see morphology plan M4)
8. ~~**Label-efficiency benchmarks** on existing `.npz` (Phase 5c — `run_label_efficiency.sh`)~~ — done; M1b wins all fractions
9. **Not planned:** `--aml-eval` during contrastive; end-to-end `--finetune` as **primary** GFM metric

---

### Phase 6: Configuration & Documentation
**Files**: `model_settings.json`, `README.md`

**Original goal**: Add configuration options and update documentation so contrastive training is easy to run and explain.

**Current status**: Partially implemented.

#### What has been implemented
1. **New command-line arguments**
   - `--objective {contrastive,supervised}` (default `contrastive`; graph form still `--reverse_mp`)
   - `--amp`
   - `--gradient_checkpointing`
   - `--contrastive_num_neg_samples`
   - `--contrastive_asymmetric`
   - `--contrastive_accum_steps`
   - `--contrastive_memory_bank_size`
   - `--loader_num_workers`

2. **W&B logging for contrastive training**
   - contrastive train loss
   - relevant config values

3. **Documentation update**
   - this plan now reflects the actual design decisions made during implementation

#### Important divergences from the original plan
- We did **not** add a generic `--contrastive-loss-type` switch
- We did **not** build a broad contrastive configuration surface for multiple loss families
- The documentation now needs to emphasize:
  - seed-edge-only loss
  - GPU negative sampling
  - optional memory queue
  - the distinction between graph form, training objective, and downstream evaluation
  - the distinction between pretraining and downstream fine-tuning

#### Remaining work in this phase
1. ~~explicit `--objective` and graph form via `--reverse_mp`~~ (done in Phase 4a)
2. update README with GFM framing (pretrain vs adapt) and example Slurm invocations
3. document the recommended comparison workflow:
   - **GFM (primary):** contrastive pretrain → extract embeddings → sklearn linear probe (AUROC + F1)
   - **Multi-GNN baseline:** `--objective supervised` from scratch (in-GNN CE + F1)
   - **Optional ablation:** contrastive pretrain → `--finetune --objective supervised` (full CE fine-tune—not Papagei)
   - contrastive pretrain only: `loss/train` (no task metrics)
4. document practical run guidance:
   - AMP may be unstable at high fanout
   - peak VRAM is dominated largely by sampled subgraph activations
   - `contrastive_num_neg_samples=1024` increased epoch time relative to `512` in current tests
   - hetero contrastive runs are expected to be slower / heavier than homo runs

---

## Implementation Order

1. **Contrastive Loss Functions** (Phase 1)
   - Implement InfoNCE first as proof of concept
   - Status update: completed in a stronger form than originally planned via seed-edge-only aligned InfoNCE, GPU negatives, and optional queue

2. **Model Architecture** (Phase 2)
   - Add embedding output to one model (e.g., GINe)
   - Status update: effectively completed across the main model classes

3. **Data Loader** (Phase 3)
   - Implement balanced pair sampling
   - Status update: evolved into seed-edge ID recovery, shared-edge alignment, and loader hot-path support rather than balanced pair sampling

4. **Training Loop** (Phase 4)
   - Modify `train_homo()` to use contrastive loss
   - Status update: homo + hetero contrastive; Phase 4a/4b done

5. **Downstream adaptation & evaluation** (Phase 5)
   - Original plan included per-epoch AML during contrastive; **replaced** by Papagei-style extract + linear probe (see Phase 5a/5b)
   - Status update: Phase 5a/5b done; **next:** end-to-end benchmark runs (pretrain → extract → probe)

6. **Configuration & Documentation** (Phase 6)
   - Update configs and README
   - Status update: CLI for objective/graph form in place; README should reflect GFM pretrain vs adapt split

7. **Next arc of development**
   - ~~Decouple graph form / objective (4a)~~ · ~~Hetero contrastive (4b)~~
   - **Implement Phase 5a/5b** (extract + linear probe), then benchmark: SSL pretrain → probe vs supervised-from-scratch vs random-init control
   - Scale pretrain toward larger / less task-labeled finance graphs; keep AML labels for **probe/adapt** eval only
   - Morphology + contrastive val for **pretrain-native** checkpoint selection (`notes/morphology-metrics-plan.md`)
   - Optional: in-GNN `--finetune` as ablation only; encoder-only release artifact

---

## Testing Strategy

1. **Small Dataset Test**
   - Use one of the Small datasets to validate implementation
   - Quick training cycles for iteration
   - Status update: this has already been the main proving ground for the current contrastive path

2. **Sanity Checks**
   - Verify embeddings are learned (not constant)
   - Check positive pairs have higher similarity than negative pairs
   - Ensure no train/val/test leakage in pair generation
   - Status update: we have also learned several systems-level sanity checks matter:
     - no OOM at larger `batch_size`
     - no NaN at higher fanout when AMP is disabled
     - acceptable throughput when queue or negative sampling settings change

3. **Comparison with Baseline (downstream metrics, not pretrain F1)**
   - **Primary:** contrastive pretrain → frozen extract → logistic regression (AUROC + F1 on val/test)
   - **Control:** same extract + probe protocol from random-init encoder weights
   - **Multi-GNN baseline:** supervised from scratch (`--objective supervised`, in-GNN F1)
   - **Optional ablation:** pretrain → `--finetune --objective supervised` (not Papagei; report separately)
   - Homo and hetero for each leg; pretrain monitored via `loss/train` only

4. **Ablations**
   - Test different contrastive loss types
   - Test different embedding dimensions
   - Test different sampling strategies for imbalanced data
   - Status update: the ablations that became most important in practice are:
     - queue on vs off
     - `contrastive_num_neg_samples=512` vs `1024`
     - batch size scaling
     - fanout scaling
     - AMP on vs off
     - homo vs hetero once both contrastive paths exist

5. **Heterogeneous Contrastive Smoke Tests**
   - add one-batch / short-run checks for hetero contrastive before long jobs
   - verify forward/reverse edge augmentation stays synchronized
   - verify seed-edge recovery and alignment operate on forward transaction edges only
   - confirm reverse synthetic edges do not enter the queue as separate contrastive samples

6. **Extraction + linear probe smoke tests** (Phase 5)
   - short contrastive pretrain → extract train/val → shapes, no NaNs, dedupe/coverage logs
   - logistic probe AUROC/F1 on Small_HI; hetero + homo

---

## Related Future Direction: Morphology Metrics

Morphology-aware SSL is the intended way to strengthen contrastive pretrain beyond edge-identity InfoNCE (especially when memory limits asymmetric / subsampled negatives). **Full specification, metric tiers, node→edge lifting, and phased implementation live in the companion doc—not duplicated here.**

**Companion doc (canonical):** [`notes/morphology-metrics-plan.md`](morphology-metrics-plan.md)

Summary of that plan:

| Topic | Decision |
|-------|----------|
| **Models** | Morphology is **model-agnostic** (`gin` / `gat` / `pna` / `rgcn` share edge `z` readout); benchmark best morph config across architectures |
| **Papagei split** | Expert heads predict some metrics; morphology-aware **contrast** uses others (or binned similarity)—AML labels not used in pretrain |
| **Readout** | Transaction embedding `z` is **edge-level** (concat sender/receiver node embs + edge emb → 128-d); morphology targets attach to **seed edges** |
| **Local vs global** | **v0:** `morph_local` + edge-native (batch subgraph); **later:** optional `morph_global` via offline node lookup (e.g. full-graph degree, BC) |
| **Precompute scope** | All **nodes** per split for globals; **never** all subgraphs; train-edge tables only for cheap edge-native scalars |
| **Phases** | M0 plumbing → M1 local expert head → M2 local morph contrast → M3 Tier 2 globals |
| **VRAM @ large B** | No dense `(B,B)` morphology masks; bin-grouped pairing — see morphology plan § **GPU memory & batch scale** |

Contrastive plan still owns: homo/hetero contrastive routing, extraction, linear probe, **no AML F1 in pretrain**. Morphology plan owns metric definitions and loss integration.

**Prerequisite:** stable contrastive + probe baseline (see May 2026 Small-HI runs: contrastive ~0.83 vs supervised ~0.97 test AUROC at 20ep) before morphology ablations.

---

## Notes

- The six-phase structure is preserved; phases 1–4 record what was built.
- **May 2026 GFM decisions:** (1) no AML metrics during contrastive pretrain; (2) **primary** downstream = Papagei-style **frozen extract + sklearn linear probe**; (3) in-GNN `--finetune` supervised is **secondary**, not the main scientific claim.
- The most important architectural win was making the loss scale with shared seed edges rather than full sampled subgraph edges.
- **`return_embeddings=True`** is the extraction hook; `evaluate_*` already shows seed-edge masking—Phase 5a refactors that into export, Phase 5b adds sklearn probe.
- **Biggest near-term gap:** implement Phase 5a/5b, then run SSL pretrain → probe vs baselines on AML (and later other tasks with their own labels).
- **Pretrain checkpoint selection:** eventually contrastive val / morphology—not AML probe val.
- **Declined:** per-epoch AML probe during contrastive; using AML val to pick encoder weights during pretrain; treating end-to-end `--finetune` CE as the definition of GFM downstream success.
