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

Modify data loading to support contrastive learning with multiple graph views:

1. **View Generation**
   - For each original graph/subgraph, generate three augmented views as described in Phase 2b
   - Apply augmentations during batch creation in LinkNeighborLoader

2. **Pair Generation for InfoNCE**
   - Positive pairs:
     - Same node representations across different views
     - Neighbor nodes within the same view
     - Feature-similar nodes (from KNN view)
   - Negative pairs: All other nodes in the batch
   - No longer rely solely on `Is Laundering` labels for pair definition

3. **Batch Structure**
   - Each batch should contain representations from all three views
   - Ensure sufficient negatives for effective contrastive learning
   - Consider batch size implications for computational efficiency

4. **Node Features Consideration**
   - Currently using vector of 1's to prevent memorization for classification
   - For contrastive pre-training, richer features may improve learning
   - Plan to experiment with feature usage during pre-training phase

**LATER:** Add support for pre-computed morphology metrics as additional contrast signals.

3. **Pair Label Format**
   - Create tensor `pair_labels` where:
     - 1 = positive pair (same laundering status)
     - 0 = negative pair (different laundering status)

4. **Edge Metadata Preservation**
   - Keep existing edge ID system for train/val/test tracking
   - Prevent leakage by respecting temporal split in pair generation

---

### Phase 4: Training Loop Refactoring
**File**: `training.py`

Refactor `train_gnn()`, `train_homo()`, and `train_hetero()`:

1. **Training Phase**
   - Loss function: Contrastive loss (InfoNCE, triplet, or pairwise)
   - Input: Batches from LinkNeighborLoader (subgraph samples)
   - Output: Node embeddings for seed edges
   - Compute loss on positive/negative pairs within batch

   ```python
   for batch in tr_loader:
       # Get embeddings for all nodes in subgraph
       embeddings = model(batch.x, batch.edge_index, batch.edge_attr)

       # Get embeddings for seed edges (edges being trained on)
       seed_embeddings = embeddings[seed_edge_nodes]

       # Compute contrastive loss on pairs
       loss = contrastive_loss(seed_embeddings, pair_labels, pair_indices)
       loss.backward()
       optimizer.step()
   ```

2. **Evaluation Phase (Classification)**
   - Use learned embeddings for downstream classification
   - Train simple linear classifier (logistic regression) on val/test embeddings
   - Report F1-score as before
   - Can be done post-training or as separate evaluation script

3. **Configuration**
   - Add `contrastive_loss_type` to `model_settings.json`
   - Update wandb config to log contrastive-specific metrics
   - Add parameters: `embedding_dim`, loss-specific params (temperature, margin, etc.)

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
