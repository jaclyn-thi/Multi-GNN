# Contrastive Learning Implementation Plan for Multi-GNN

## References
- Hanbin et al. (2024): "Graph Contrastive Pre-training for Anti-money Laundering" - Source of graph augmentation techniques and InfoNCE application to AML graphs.

## Current System Overview

### Data Loading & Splitting
- **Data Source**: CSV files with transaction records (rows) and features like sender/receiver, bank, currency, amount, etc.
- **Labels**: `Is Laundering` (0=legitimate, 1=illicit)
- **Temporal Split**: Data is split by timestamp into train/val/test (preventing leakage)
- **Imbalanced Classes**: Illicit transactions are the minority class (imbalance varies by dataset)
- **Scale**: Medium datasets have ~32 million transactions

### Current Training
- **Original baseline path**: CrossEntropyLoss with class weights (`w_ce1`, `w_ce2`) and F1-score logging on the AML edge classification task
- **Current homogeneous path**: contrastive pretraining on edge / transaction embeddings
- **Current heterogeneous path**: still follows the older supervised classification path with F1 logging
- **Current control-flow limitation**: graph form (`homo` vs `hetero`), training objective (`supervised` vs `contrastive`), and downstream AML evaluation are not yet fully decoupled in the training entry points
- **Subgraph Sampling**: `LinkNeighborLoader` with configurable `num_neighbors` for batch sampling
  - prevents computational overload for large graphs
  - samples k-hop neighbors around seed edges
- **Models**: GINe, GATe, PNA, RGCN
  - all now support an embedding head plus a classifier head
  - all expose dual behavior via `return_embeddings`

### Key Functions
- `get_loaders()` in `train_util.py`: Creates `LinkNeighborLoader` instances
- `train_homo()` in `training.py`: currently contrastive pretraining loop for homogeneous graphs
- `train_hetero()` in `training.py`: supervised training loop with F1 logging
- `evaluate_homo()` / `evaluate_hetero()` in `train_util.py`: evaluation helpers

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
- revisit structural positive tiers only after the evaluation path is restored

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
- use the classifier mode again in a meaningful homogeneous supervised fine-tuning / evaluation path
- optionally make embedding-dimension selection cleaner in config

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
- add hetero-aware transaction-ID plumbing so forward and reverse edge copies can stay synchronized under contrastive augmentation
- add hetero-aware batch helpers for seed-edge recovery and alignment on the forward transaction edges

---

### Phase 4: Training Loop Refactoring
**File**: `training.py`

**Original goal**: Refactor training to support contrastive pretraining followed by downstream evaluation.

**Current status**: Partially implemented. Homogeneous contrastive pretraining exists, but the training control flow still needs a second refactor pass so graph form, objective, and downstream evaluation become separate choices.

#### What has been implemented
1. **Homogeneous contrastive pretraining loop**
   - `train_homo()` now runs contrastive pretraining on homogeneous graphs
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
- The homogeneous AML F1 logging was not just disabled for convenience; the loop itself is now fundamentally a pretraining loop
- The current dispatcher still effectively couples:
  - `reverse_mp=False` with the newer homogeneous contrastive path
  - `reverse_mp=True` with the older heterogeneous supervised path
  rather than letting graph form and objective vary independently

#### Phase 4a: Decouple Graph Form, Objective, and Evaluation
This is now the most important near-term move.

The training entry points should be able to represent all four combinations:
- homogeneous + supervised
- homogeneous + contrastive
- heterogeneous + supervised
- heterogeneous + contrastive

To get there, the training control flow should be refactored so that:
1. **Graph form** is determined by `--reverse_mp`
2. **Training objective** is determined independently (`supervised` vs `contrastive`)
3. **Downstream AML evaluation** is determined independently (`on` vs `off`)

Recommended control surface:
- keep `--reverse_mp` responsible only for graph structure
- add an explicit objective flag such as `--objective {contrastive,supervised}`
- default that objective flag to `contrastive`
- avoid overloading a single flag with both graph-form and objective semantics

Concrete work in this subphase:
1. **Split homogeneous pretraining and homogeneous supervised fine-tuning**
   - likely `train_homo_contrastive(...)`
   - likely `train_homo_supervised(...)`
   - or equivalent branching inside `train_homo()`

2. **Keep heterogeneous supervised training intact, but make it explicit**
   - preserve current F1-based AML supervised behavior
   - make classifier-mode calls explicit in supervised training and evaluation

3. **Make `--finetune` a real downstream classification mode**
   - load pretrained weights
   - run supervised CE on AML labels
   - evaluate on val / test
   - log F1 metrics

4. **Restore baseline-comparable AML metrics**
   - `f1/train`
   - `f1/validation`
   - `f1/test`
   - `best_test_f1`

5. **Add an explicit contrastive on/off switch via objective selection**
   - `contrastive` should remain the default objective
   - `supervised` should provide a clean baseline rerun path
   - this should work independently of `homo` vs `hetero`

#### Phase 4b: Add Heterogeneous Contrastive Training
Once the control flow is clean, the next training-loop extension is hetero contrastive support.

Concrete work in this subphase:
1. add `train_hetero_contrastive(...)`
2. keep reverse edges for message passing, but use forward transaction edges as the contrastive samples
3. recover forward seed-edge IDs for hetero batches
4. run the hetero model on two hetero augmented views
5. compute contrastive loss only on forward-edge embeddings aligned by transaction identity
6. keep queue entries and negatives forward-transaction-only in the first implementation

This preserves the original motivation for Multi-GNN, where reverse message passing improves the graph representation, while keeping the contrastive supervision attached to real transactions rather than synthetic reverse edges.

#### Why this is the next implementation step
- We now have a stable homogeneous contrastive pretraining path
- The missing pieces are:
  - turning the pretrained encoder back into a meaningful AML downstream comparison
  - extending the same ideas to the hetero setting that motivated the original Multi-GNN emphasis

---

### Phase 5: Evaluation Pipeline
**File**: `train_util.py` or new file `evaluation.py`

**Original goal**: Evaluate learned embeddings and compare against baseline AML classification.

**Current status**: This phase is now one of the main remaining gaps, and it needs to support both homogeneous and heterogeneous training modes.

#### What the original plan proposed
1. embedding extraction
2. embedding-based classification
3. F1-score and related metrics for comparison

#### What we know now
- The most direct short-term comparison to baseline Multi-GNN is still the AML edge classification task
- The homogeneous contrastive path does **not** yet provide that comparison cleanly
- `evaluate_homo()` currently calls the model in embedding mode by default, so it is not yet acting as a proper classifier evaluator
- the long-term system should support turning AML evaluation on or off regardless of whether the underlying training run is homogeneous or heterogeneous

#### Updated evaluation priority
1. **First priority: restore direct AML comparison for both graph forms**
   - supervised homogeneous fine-tuning on top of the pretrained encoder
   - preserve heterogeneous supervised AML evaluation
   - classifier-mode evaluation
   - W&B F1 logging

2. **Second priority: make AML evaluation an explicit switch**
   - allow AML F1 evaluation to be enabled for both homo and hetero
   - keep this separate from the choice of training objective
   - leave room for future downstream financial tasks

3. **Third priority: optional embedding-style evaluation**
   - frozen encoder + linear probe
   - logistic regression on extracted embeddings
   - alternate downstream finance tasks

#### Important design clarification
- The AML task is only one downstream use case for the graph foundation model
- However, it is the best immediate benchmark because it allows direct comparison to the original non-contrastive Multi-GNN pipeline

#### Remaining work in this phase
- fix `evaluate_homo()` to use classifier logits when evaluating AML classification
- make classifier-mode behavior explicit in both homo and hetero supervised/eval call sites
- restore homogeneous train / val / test F1 logging
- preserve and cleanly expose heterogeneous AML F1 evaluation
- decide whether AML evaluation should run:
  - every epoch in supervised mode
  - periodically or end-of-run in contrastive mode
- decide whether to support:
  - full fine-tuning first
  - frozen encoder / linear probe later

---

### Phase 6: Configuration & Documentation
**Files**: `model_settings.json`, `README.md`

**Original goal**: Add configuration options and update documentation so contrastive training is easy to run and explain.

**Current status**: Partially implemented.

#### What has been implemented
1. **New command-line arguments**
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
1. add explicit configuration support for:
   - graph form (`homo` vs `hetero`)
   - training objective (`supervised` vs `contrastive`)
   - downstream AML evaluation (`on` vs `off`)
2. prefer an explicit objective flag over a bare boolean:
   - e.g. `--objective contrastive` (default)
   - e.g. `--objective supervised`
   - this keeps baseline reruns and debugging straightforward
3. update README / usage docs when the supervised downstream path is restored
4. document the recommended comparison workflow:
   - supervised from scratch
   - contrastive pretraining only
   - contrastive pretraining -> supervised fine-tuning
5. document practical run guidance:
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
   - Status update: completed for homogeneous pretraining; next arc is to decouple graph form/objective/evaluation, then add hetero contrastive

5. **Evaluation** (Phase 5)
   - Implement embedding-based classification evaluation
   - Status update: partially deferred; first priority is restoring baseline-comparable AML classification metrics for both homo and hetero, ideally as an explicit switch

6. **Configuration & Documentation** (Phase 6)
   - Update configs and README
   - Status update: partially complete; this plan has been updated, but the next documentation pass should happen after the new control-flow and evaluation switches are in place

7. **Next Arc of Development**
   - Step 1: decouple graph form, objective, and downstream AML evaluation
   - Step 2: add an explicit objective flag with `contrastive` as the default and `supervised` as the clean baseline path
   - Step 3: restore classifier-mode AML training/evaluation for homogeneous fine-tuning
   - Step 4: preserve and cleanly expose AML F1 evaluation on the hetero path
   - Step 5: extend contrastive training to hetero while keeping forward transaction edges as the contrastive anchors

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

3. **Comparison with Baseline**
   - Train with CrossEntropyLoss (current)
   - Train with contrastive loss (new)
   - Compare F1-scores and embedding quality
   - Status update: this remains the core next comparison, but it now means comparing:
     - supervised from scratch
     - homogeneous contrastive pretraining -> AML fine-tuning
     - heterogeneous contrastive pretraining -> AML fine-tuning

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

---

## Related Future Direction: Morphology Metrics

Another major project thread is morphology-aware representation learning, including metrics such as:
- betweenness centrality
- triangle counts
- clustering / motif-style structure signals

This should stay visible in the contrastive plan because it is one of the intended ways to make the embeddings more discriminative, but it is large enough to deserve its own companion planning document.

Current recommendation:
- keep the main implementation focus here on contrastive training, homo/hetero support, and AML evaluation
- treat morphology metrics as a parallel future extension
- evaluate whether morphology should be used as:
  - an auxiliary prediction target
  - an additional feature source
  - a regularizer / extra loss term
  - a source of structure-aware positives or task design

Companion doc:
- `notes/morphology-metrics-plan.md`

Open design questions that will likely matter later:
- should morphology targets be defined on nodes, edges, or transaction-local substructures?
- should metrics be computed on homogeneous graphs, heterogeneous graphs, or both?
- how do we avoid temporal leakage when precomputing metrics?
- which metrics are feasible to compute exactly at project scale versus approximately or offline?

---

## Notes

- The six-phase structure is still useful for communicating the implementation plan, but each phase now needs to reflect the design decisions we actually made
- The most important architectural win was making the loss scale with shared seed edges rather than full sampled subgraph edges
- The biggest remaining implementation gaps are now:
  - decoupling graph form, objective, and downstream evaluation
  - adding a clean objective switch so contrastive training can be turned on/off without changing graph form
  - restoring clean AML evaluation for both homo and hetero
  - extending contrastive learning to the hetero path that originally motivated Multi-GNN
- Once the AML fine-tune/evaluation path is back for both graph forms, we can more cleanly assess whether contrastive pretraining improves the baseline task before expanding to broader downstream finance applications
