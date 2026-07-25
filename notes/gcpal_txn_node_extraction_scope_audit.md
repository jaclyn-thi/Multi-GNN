# Transaction-node extraction graph-scope audit

Status: **canonical audit** · Date: 2026-07-23 · Diagnostic · **No training jobs submitted**

Companion: [`results/diagnostics/gcpal_txn_node_extraction_scope_audit.json`](../results/diagnostics/gcpal_txn_node_extraction_scope_audit.json)  
Canonical API: `gcpal_txn_node/extraction.py`  
Related: [`gcpal_txn_node_poscomplete_replay_audit.md`](gcpal_txn_node_poscomplete_replay_audit.md)

---

## 1. Code traced

| Piece | Location |
|-------|----------|
| Flow construction | `gcpal_txn_node/adjacency.py` → `build_directed_flow_adjacency(..., policy="immediate_next")` |
| Induce | `induce_edge_index` — both endpoints must lie in `node_ids` |
| Historical encode | Duplicated `encode_nodes_induced` in scout/resume scripts (**chunk=4096**) |
| Canonical encode | `gcpal_txn_node.extraction.encode_nodes_induced` + `extract_split_embeddings` |

Historical helper (resume):

```text
encoder.eval()
for each chunk of node_ids (chunk=4096):
    x = x_all[chunk_ids]
    ei = induce_edge_index(flow_ei, chunk_ids)   # edges inside the chunk only
    h, _ = encoder(x, ei)
```

---

## 2. Exact graph scope by split (Small-HI measured)

| Object | Construction | Scope |
|--------|--------------|-------|
| `flow_full` | immediate-next on **all** CSV rows | 2 605 952 edges; **370 362 cross-split** |
| `x_full` | train fit + transform val/test | All transactions |
| Temporal **train** H | induce on train IDs | Historical: **per 4096 chunk**; canonical v1: **full train set** |
| Temporal **val** H | induce on val IDs | Same policy |
| Temporal **test** H | induce on test IDs | Same policy |
| **Random-40** | No re-encode | Subsets of temporally extracted `h_full` |

### Flow edge split×split (src → dst)

|  | → train | → val | → test |
|--|--------:|------:|-------:|
| train | 1 614 187 | 125 991 | 114 634 |
| val | 0 | 346 599 | 129 737 |
| test | 0 | 0 | 274 804 |

Induced within-split edge counts: train 1 614 187 · val 346 599 · test 274 804.

### Legacy chunk=4096 defect (train)

| Metric | Value |
|--------|------:|
| Full-split induce edges | 1 614 187 |
| Sum of per-chunk induce edges | **66 721** |
| Train nodes losing **all** train–train out-neighbors due to chunking | **1 547 466** |
| Fraction among nodes that had ≥1 such neighbor | **0.959** |

---

## 3. Explicit answers

1. **Can a training transaction aggregate from a validation or test transaction?**  
   Under induce-per-split encode of train IDs: **No** (val/test nodes are not in the subgraph). Cross-split edges exist in `flow_full` but are dropped by induce.

2. **Can it see financial-flow edges constructed using future transactions?**  
   **`flow_full` yes** (full timeline). In train encode MP: only future **train** nodes inside the induced node set (full split for canonical v1; same 4096 chunk historically).

3. **Are train, val, and test encoded on one full graph?**  
   **No joint full-graph forward.** Shared global edge table + features; separate induced encodes per split.

4. **Does induced batching drop neighbors differently by batch composition?**  
   **Yes** for legacy chunk=4096 (severe). Canonical v1 uses **full-split** induce to remove that artifact.

5. **Consistent with edge-centric temporal protocol?**  
   **No.** Different graph object (txn–txn flow vs account–txn Multi-GNN) and induce policy. Treat as a **separate diagnostic protocol**.

---

## 4. Leakage / protocol verdict

| Item | Verdict |
|------|---------|
| Cross-split **message-passing** leakage under clean induce-per-split | **Not supported** by the encode path |
| Full-timeline adjacency constructed | **Yes** |
| Legacy 4096 chunking | **Protocol defect** (not label leakage; representation corruption) |
| Edge-centric equivalence | **Mismatch** |
| Thesis-table eligible today | **No** until canonical extraction is used for reported metrics |

**No silent scope change to historical results.** Smallest correction (implemented as API, not re-eval): keep induce-per-temporal-split, encode each split on the **full** split ID list (`ChunkPolicy.FULL_SPLIT` / mode `frozen_checkpoint_induce_per_temporal_split_v1`).

Optional later **P1**: train-only adjacency for train encode — not applied here.

---

## 5. Extraction modes (updated 2026-07-23)

| Mode ID | Role |
|---------|------|
| `frozen_checkpoint_induce_per_temporal_split_v1` | **Sensitivity** (full per-split isolation) — not sole canonical |
| `frozen_checkpoint_temporal_expanding_window_v1` | **Candidate thesis-primary** (train / train∪val / full) |
| `frozen_checkpoint_joint_full_graph_random40_v1` | **Diagnostic** (joint full-graph encode; random-40 after) |
| `legacy_chunked_induce_4096_v0` | Historical / noncanonical |

Shared requirements: identified checkpoint (+ sha256); `eval` + frozen; no aug / no `h_anchors`; label-free extract; row-ID hashes; config hash; no graph-destructive output chunking for non-legacy modes.

Tests: `tests/test_gcpal_txn_node_extraction.py`.  
Canonical re-extract suite: [`gcpal_txn_node_canonical_reextraction.md`](gcpal_txn_node_canonical_reextraction.md).

---

## 6. Artifact classification

| Artifact | Status | Table eligible |
|----------|--------|----------------|
| Original A/B 5ep | **noncanonical** (online augmented) | **No** |
| Replay ep5/10/15/20 | internally comparable A/B diagnostic under **legacy-chunked** encode | **No** |
| Per-split full induce v1 | sensitivity analysis | **No** |
| Temporal expanding-window v1 | candidate thesis-primary extraction | candidate after review |
| Joint full-graph random-40 v1 | GCPAL-aligned diagnostic; never primary | **No** |

Original extraction wording (locked):

> online augmented-view embeddings accumulated across changing training states, with clean induced fill/evaluation.

Rename for future outputs: prefer **`loss_trajectory_within_tolerance`** over `replay_verified`.

---

## 7. Corrected interpretation

1. **B beats A under a shared frozen-checkpoint legacy-chunked extraction; canonical graph-preserving re-extraction is pending** (or reported separately once the suite completes).  
2. **Val H‖X AUPRC selects epoch 5** for both arms **under the legacy-chunked curve** — do not carry that selection forward to canonical modes.  
3. Longer training **does not improve** that **legacy** validation selection metric.  
4. **B test H and H‖X AUPRC are higher at ep20 than ep5** on the legacy curve — fixed-horizon only; **not** for selection.  
5. Absolute temporal performance remains **weak** on legacy encode.  
6. **One seed**; **not** an exact GCPAL reproduction.

---

## 8. Are existing replay metrics still valid?

| Use | Valid? |
|-----|--------|
| A vs B ranking under a **fixed** (legacy chunked) encode across epochs | Yes, internally consistent |
| Thesis / paper-comparable frozen embeddings | **No** until expanding-window (or reviewed successor) metrics are used |
| Comparison to original 5ep scout numbers | **No** — different extraction |

**Required before table promotion:** graph-preserving re-extract (expanding-window candidate) from saved checkpoints + frozen MLP suite (**no GNN retrain**).

## Confirmation

No training or Slurm jobs were submitted for this audit.
