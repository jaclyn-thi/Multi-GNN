# GCPAL-style transaction-node baseline — specification & ambiguity report

**Status:** explicit reimplementation under documented assumptions.  
**Not** an exact GCPAL / paper reproduction (no public code; incomplete implementation details).

Machine-readable twin fields live in `gcpal_txn_node/spec.py`.

---

## 1. What the paper explicitly specifies

| Item | Paper statement (as used here) |
|------|--------------------------------|
| Representation | Transactions as **nodes** |
| Structural graph | Financial-flow / payment adjacency between transactions |
| Views | Two random graphs via **edge drop** and **feature drop** |
| Third view | Sparse **KNN graph** from raw transaction features |
| Encoder | Shared vanilla **GIN** |
| Projection | MLP projection head |
| Positives | Matrix from original adjacency + KNN adjacency (Eq. 9: \(M_P = A + A_{knn}\)) |
| Contrast | **Symmetric** multi-positive InfoNCE |
| Mixing | \(\lambda \approx 0.3\) on AMLWorld: \(L=\lambda L(r_1,r_2)+(1-\lambda)L(r_2,\mathrm{KNN})\) |
| Temperature | \(\approx 0.5\) |
| KNN \(k\) | \(\approx 15\) |
| Embedding dim | 128 |
| Downstream | MLP on concatenated \(H \Vert X\) |

---

## 2. Unresolved (must be chosen; not silently claimed as paper truth)

| Ambiguity | Our choice for this baseline | Label |
|-----------|------------------------------|-------|
| Exact AMLWorld transaction-node adjacency | Directed flow: txn \(i\to j\) iff receiver(\(i\))=sender(\(j\)) and \(t_j>t_i\); default **`immediate_next`** (nearest subsequent outgoing per receiver account). Optional `capped_next_k`. | `assumption:adjacency_immediate_next` |
| Do not build all pairwise same-account edges | Enforced | hard constraint |
| Global vs approximate KNN | Reuse existing **sparse global train** KNN cache (`edge_native+degree_fan`, \(k=15\)); never dense \(XX^\top\); never batch-local substitute | `deviation:knn_includes_degree_fan` |
| Feature preprocessing | Timestamp standardized; `log1p(amount)`; one-hot currency & payment format; **fit on train split only** in temporal mode | `assumption:feature_pipeline` |
| Identity in \(M_P\) | Eq. 9 omits \(I\); ablation implies identity. Flag `--include_identity_positives` (default **on** for primary smoke): \(I \cup A \cup A_{knn}\) | `assumption:identity_in_positives` |
| Graph batching / sampling | Positive-aware seed construction: anchors + capped structural/KNN positives present in the sampled node set; \(B\le 2048\) | `assumption:positive_aware_batching` |
| Optimizer / duration | Adam, fixed LR; smoke = 1 batch; scout = 5 epochs (no auto-extend to 40) | `assumption:optimizer_schedule` |
| Split / transductive | Temporal train/val/test primary; stratified random 40/60 diagnostic only | matches thesis diagnostic policy |

---

## 3. Explicit deviations from a literal paper reading

1. **KNN feature space:** cache uses `edge_native+degree_fan` (adds train-graph degree features), not raw AML columns alone.
2. **Adjacency:** `immediate_next` is a sparsity assumption; the paper does not define hop/cap policy.
3. **GIN details:** two-layer `GINConv` (no edge attributes on the txn-node graph), hidden/out 128 — architecture family matches; layer width/MLP internals are not paper-specified.
4. **Feature drop:** entire node-feature **rows** masked (paper equation intent), not per-cell.
5. **No claim** of matching unpublished optimizer, early stopping, or AMLWorld split details.

---

## 4. Namespace isolation

All code lives under `gcpal_txn_node/` and `scripts/gcpal_txn_node_*.py`.  
**No imports** from this package into `main.py`, `training.py`, `contrastive_loss.py`, or default Multi-GNN paths.

---

## 5. Evaluation policy (scouts)

- Primary: **temporal** protocol + frozen MLP on \(H\Vert X\)
- Diagnostic only: stratified **random 40% train / 60% test** (not thesis-primary)
- Do not call results “GCPAL Table 2 reproduction”
