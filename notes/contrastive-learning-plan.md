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
- **Loss Function**: CrossEntropyLoss with class weights (`w_ce1`, `w_ce2`)
- **Evaluation Metric**: F1-score on node classification (edge labels)
- **Subgraph Sampling**: LinkNeighborLoader with configurable `num_neighbors` for batch sampling
  - Prevents computational overload for large graphs
  - Samples k-hop neighbors around seed edges
- **Models**: GINe, GATe, PNA, RGCN (all output 2-class predictions)

### Key Functions
- `get_loaders()` in `train_util.py`: Creates LinkNeighborLoader instances with edge_label_index and edge_label
- `train_homo()` / `train_hetero()` in `training.py`: Training loops using CrossEntropyLoss
- `evaluate_homo()` / `evaluate_hetero()` in `train_util.py`: Evaluation using F1-score on predicted vs ground truth labels

---

## Implementation Plan

### Phase 1: Contrastive Loss Functions
**File**: `train_util.py` (or new file `contrastive_loss.py`)

Create flexible contrastive loss implementations:

1. **InfoNCE Loss** (Selected based on Hanbin et al. 2024)
   - Treats each sample and its augmented version as positive pair
   - All other samples in batch as negatives
   - Good for large batches
   - Formula: Loss = -log(exp(sim(x_i, x_i+) / τ) / Σ_j exp(sim(x_i, x_j) / τ))
   - Temperature parameter `τ` controls contrast sharpness
   - Will maximize mutual information across different views of the same graph

2. **Triplet Loss** (Alternative)
   - Uses (anchor, positive, negative) triplets
   - Margin-based: Loss = max(0, d(x_a, x_p) - d(x_a, x_n) + margin)
   - Simpler to implement, requires careful negative sampling

3. **Pairwise Contrastive Loss** (Alternative)
   - Similar pairs should have small distance
   - Dissimilar pairs should have large distance
   - Formula: Loss = (1 - y) * d(x_i, x_j)^2 + y * max(0, margin - d(x_i, x_j))^2

**Configuration**: Parameters (temperature, margin) should be extractable from `model_settings.json` like other hyperparameters.

---

### Phase 2: Model Architecture Changes
**File**: `models.py`

Modify all model classes (GINe, GATe, PNA, RGCN):

1. **Add Embedding Output**
   - Current: Final layer outputs 2 logits (for binary classification)
   - New: Add embedding layer before final classification head
   - Embedding dimension should be configurable (e.g., `embedding_dim` parameter)

2. **Dual Output Mode**
   - During training: Return embeddings only
   - During evaluation: Use embeddings to produce classification logits (via simple MLP head)

3. **Example Architecture**
   ```
   Input → GNN Layers → Embedding Layer (e.g., 128 dims) → Classification Head (2 classes)
   ```

---

### Phase 2b: Graph Augmentations (Following Hanbin et al. 2024)
**File**: `data_loading.py` or new file `graph_augmentations.py`

Implement three views of the same graph for contrastive learning:

1. **Random Edge Dropping (View 1)**
   - Randomly drop a specifiable number/percentage of edges
   - Configurable parameter: `drop_rate` (e.g., 0.1 for 10% drop)

2. **Random Edge Dropping (View 2)**
   - Same as View 1, but with different randomization
   - Ensures different edge subsets are dropped for diversity

3. **KNN Graph Construction (View 3)**
   - Connect each node to top k most similar nodes
   - Similarity computed via matrix multiplication of node feature matrix X
   - Note: Current repo uses vector of 1's for node features to avoid memorization
   - Consider using richer features for pre-training (can be changed later)
   - Configurable parameter: `k` (number of nearest neighbors)

**Positive Pairs**:
- Same node across different views
- Neighbors within each view
- Similar feature nodes (via KNN in View 3)

**Negative Pairs**:
- All other nodes not in positive pairs

**Goal**: Maximize mutual information across views using InfoNCE loss.

---

### Phase 3: Data Loader Modifications
**File**: `data_loading.py`

Modify data loading to support contrastive learning with graph augmentations.

### 1. View Generation
- For each batch (from `LinkNeighborLoader`), generate **two augmented views** using random edge dropping
- Use `generate_views(data, drop_rate)` from `graph_augmentations.py`
- Apply augmentations **in the training loop**, not inside the DataLoader itself

### 2. Contrastive Pair Definition
- Positive pairs:
  - Same node across different views (identity)
  - Neighbor nodes within each view (via adjacency matrix)
- Negative pairs:
  - All other nodes in the batch
- Pair relationships are handled **implicitly by the contrastive loss**
- Do not construct explicit pair labels or rely on `Is Laundering` labels

### 3. Batch Structure
- Each batch produces **two correlated graph views**
- Each view is passed independently through the model to produce embeddings
- Contrastive loss is computed between embeddings from the two views
- Batch size should be chosen to ensure a sufficient number of negatives

### 4. Node Features Consideration
- Node features are currently constant (vector of ones)
- Structural information is learned via message passing over edge features
- May experiment with richer node features during pre-training in future work

### 5. Edge Metadata Preservation
- Preserve edge IDs and timestamps for downstream evaluation
- Ensure augmentations are applied **only within batches**
- Maintain temporal split integrity (no leakage across train/val/test)

### 6. (Future Work) Morphology Metrics
- Incorporate graph morphology metrics (e.g., centrality) as an auxiliary loss
- Keep separate from the contrastive augmentation pipeline

---

### Phase 4: Training Loop Refactoring
**File**: `training.py`

Refactor `train_gnn()`, `train_homo()`, and `train_hetero()` to support **contrastive pretraining followed by downstream evaluation**.


### 1. Contrastive Pretraining Phase

- **Objective**: Learn **transaction (edge) embeddings** using multi-view contrastive learning
- **Loss function**: Contrastive loss (InfoNCE implemented; extensible to triplet/pairwise)
- **Input**: Batches from `LinkNeighborLoader` containing subgraph samples
- **Output**: Edge embeddings (no classification head during pretraining)

#### Key Design Decisions
- Operate on **transaction (edge) embeddings**, where each embedding encodes:
  - source node representation
  - destination node representation
  - edge features
- Each edge corresponds to a **single training sample**
- Positive pairs:
  - Same edge across different augmented views (identity)
- Negative pairs:
  - All other edges in the batch
- **Edge ordering must be preserved across views** to ensure correct alignment

#### Important Constraint
- Graph augmentations must **NOT change edge ordering**
- Do NOT drop edges by filtering (this breaks alignment)
- Instead:
  - Use identical edge sets across views
  - Introduce stochasticity via:
    - GNN dropout
    - (future) structure-aware perturbations

#### Training Loop (Pretraining Mode)
```python
for batch in tr_loader:
    optimizer.zero_grad()

    # Generate two aligned views (same edge ordering)
    view1, view2 = generate_views(batch, drop_rate)

    # Compute embeddings (edge-level)
    z1 = model(view1.x, view1.edge_index, view1.edge_attr)
    z2 = model(view2.x, view2.edge_index, view2.edge_attr)

    # Contrastive loss (edge-aligned)
    loss = contrastive_loss(z1, z2)

    loss.backward()
    optimizer.step()
```


### 2. Structural Contrastive Learning (Edge Adjacency)

- **Objective:** Improve contrastive learning by incorporating graph structure into positive pair definitions

- **Key Idea:**
  - Phase 4.1 uses only identity positives (same edge across views)
  - Extend this by defining positive examples on the edge level using tiers:

### Positive Pair Definition for Edge-Level Contrastive Learning

Because the model learns transaction/edge embeddings, positive pairs are defined over transactions rather than nodes.

1. **Identity positives across views**
   - If the same original transaction appears in both augmented views, its embeddings form a positive pair.
   - This requires preserving a stable `edge_id` through augmentation.
   - Since independent edge dropping may remove a transaction from one view, anchors without a matching `edge_id` in the other view are skipped for the identity-only loss.

2. **Structural positives (future extension)**
   - Transactions may also be treated as positives if they share endpoint accounts, e.g. same sender, same receiver, or flow continuation patterns.
   - To avoid dense `E × E` adjacency matrices, structural positives should be sampled or constructed via account-to-transaction lookup tables.

3. **Time-aware structural positives (future extension)**
   - Shared-account positives can be filtered by temporal proximity to better capture AML-relevant transaction patterns.

Labels such as `Is Laundering` are not used for pretraining positive-pair construction.

- **Approach:**
  - Build an edge-level adjacency matrix `A_edge` per batch:
    - `A_edge[i, j] = 1` if edges i and j share a node
  - Pass `A_edge` into `contrastive_loss`:
    ```python
    loss = contrastive_loss(z1, z2, A=A_edge)
    ```

- **Notes:**
  - Edge ordering must remain consistent across views
  - Do not apply edge dropping yet (to preserve alignment)
  - Adjacency is constructed per batch, not globally


### 3. Model Behavior Updates

Model already supports dual modes:
```python
def forward(self, x, edge_index, edge_attr, return_embeddings=True):
    z = self.embedding_head(...)

    if return_embeddings:
        return z
    else:
        return self.classifier(z)
```
- **Pretraining**: return embeddings (`z`)
- **Evaluation**: pass embeddings through classifier head


### 4. Configuration Updates

Update `model_settings.json`:

- Contrastive learning parameters:
  - `embedding_dim`
  - `contrastive_loss_type` (e.g., `"infonce"`)
  - `temperature`
  - `margin` (if applicable)

- Training control:
  - `training_mode`: `"contrastive"` or `"supervised"`


### 5. Logging (wandb)

Track contrastive-specific metrics:

- Training loss (InfoNCE)
- Embedding statistics:
  - Norms
  - Cosine similarity (positive vs negative pairs)

- Optional:
  - Alignment vs uniformity metrics

---

### Phase 5: Evaluation Pipeline
**File**: `train_util.py` or new file `evaluation.py`

Create evaluation functions using learned embeddings:

1. **Embedding Extraction**
   - After training, extract embeddings for all nodes in train/val/test sets
   - Function: `extract_embeddings(model, data_loader, device)`

2. **Classification on Embeddings**
   - Train logistic regression on training set embeddings
   - Evaluate on val/test embeddings
   - Report F1-score, accuracy, precision, recall
   - Can use sklearn's LogisticRegression or simple PyTorch linear layer

3. **Metrics to Log**
   - Training: Contrastive loss per epoch, temperature/margin if applicable
   - Validation/Test: F1-score on embedding classification
   - Optional: Embedding quality metrics (cosine similarity between positive/negative pairs)

---

### Phase 6: Configuration & Documentation
**Files**: `model_settings.json`, `README.md`

1. **Update `model_settings.json`**
   ```json
   {
     "gin": {
       "params": {
         "lr": 0.001,
         "n_hidden": 128,
         "embedding_dim": 64,
         "n_gnn_layers": 3,
         "contrastive_loss_type": "infonce",
         "temperature": 0.5,
         "w_ce1": 1.0,
         "w_ce2": 5.0
       }
     }
   }
   ```

2. **Command-line Arguments**
   - Add `--contrastive-loss` flag to enable/disable contrastive training
   - Add `--contrastive-loss-type` to specify loss (infonce, triplet, pairwise)
   - Add `--embedding-dim` for embedding dimension

3. **Documentation**
   - Update README with contrastive learning setup instructions
   - Document how to evaluate embeddings after training
   - Explain temporal splitting and leakage prevention

---

## Implementation Order

1. **Contrastive Loss Functions** (Phase 1)
   - Implement InfoNCE first as proof of concept
   - Add triplet and pairwise as alternatives

2. **Model Architecture** (Phase 2)
   - Add embedding output to one model (e.g., GINe)
   - Test with contrastive loss before extending to others

3. **Data Loader** (Phase 3)
   - Implement balanced pair sampling
   - Test with small dataset first

4. **Training Loop** (Phase 4)
   - Modify `train_homo()` to use contrastive loss
   - Test end-to-end training

5. **Evaluation** (Phase 5)
   - Implement embedding-based classification evaluation
   - Compare F1-scores with baseline CrossEntropyLoss

6. **Configuration & Documentation** (Phase 6)
   - Update configs and README
   - Add command-line args for easy switching

---

## Testing Strategy

1. **Small Dataset Test**
   - Use one of the Small datasets to validate implementation
   - Quick training cycles for iteration

2. **Sanity Checks**
   - Verify embeddings are learned (not constant)
   - Check positive pairs have higher similarity than negative pairs
   - Ensure no train/val/test leakage in pair generation

3. **Comparison with Baseline**
   - Train with CrossEntropyLoss (current)
   - Train with contrastive loss (new)
   - Compare F1-scores and embedding quality

4. **Ablations**
   - Test different contrastive loss types
   - Test different embedding dimensions
   - Test different sampling strategies for imbalanced data

---

## Notes

- **Subgraph Sampling**: Within-batch pair generation is naturally aligned with LinkNeighborLoader's subgraph sampling
- **Class Imbalance**: Start with balanced sampling per batch. If that fails, fall back to loss weighting
- **Morphological Features**: Phase out for now; can be added later by creating augmentations or pre-computed features
- **Evaluation**: Keeping F1-score on edge labels allows direct comparison with baseline CrossEntropyLoss training
- **TODO Comment**: Line 231 in training.py already has "TODO switch to contrastive loss" comment
