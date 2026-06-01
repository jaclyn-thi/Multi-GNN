# Morphology ID joins (`EdgeID`, `from_id`, `to_id`)

This document specifies how transaction and node identifiers line up across CSV, PyG graph objects, and morphology tables. See also [`notes/morphology-metrics-plan.md`](../notes/morphology-metrics-plan.md).

## CSV (`formatted_transactions.csv`)

| Column | Role |
|--------|------|
| `EdgeID` | Stable transaction id (integer), row identity in the file |
| `from_id` | Sender node id (account / holding) |
| `to_id` | Receiver node id |
| `Timestamp` | Seconds from start of dataset (after `data_loading` shift) |

## In-memory graph (before `add_arange_ids`)

- `edge_index`: shape `[2, E]`, row 0 = `from_id`, row 1 = `to_id` for each forward transaction in the split.
- Row order matches the order of edges in that split’s tensor (train / val / test subgraph).

## After `add_arange_ids` (`train_util.add_arange_ids`)

A synthetic id column is prepended to `edge_attr`:

- **Homogeneous:** `edge_attr[e, 0] == e` (0 … E−1 within that split graph).
- **Hetero forward:** same for `data['node', 'to', 'node'].edge_attr[e, 0]`.
- **Hetero reverse:** reverse row `r` uses id `E + r` in `edge_attr`; contrastive / morphology use **forward ids only** (`0 … E−1`).

`attach_edge_id_from_batch` copies column 0 into `store.edge_id` and strips it from features before the GNN forward.

## Seed edges in training

| Context | How to get `EdgeID` for seeds |
|---------|------------------------------|
| Hetero loader batch | `get_hetero_seed_edge_ids(batch, loader_data)` → `loader_data[forward].edge_attr[input_id, 0]` |
| Homo loader batch | `get_homo_seed_edge_ids(batch, loader_data)` |
| Full split graph | `edge_attr[:, 0]` or `torch.arange(E)` if `add_arange_ids` was applied |

## Morphology lookups

### Tier 0 global (offline node table)

1. Build **train-only** forward `edge_index` (and val/test separately for eval tables).
2. `compute_tier0_node_stats(edge_index, num_nodes)` → DataFrame indexed by `node_id` with `deg_in`, `deg_out`, `deg_total`.
3. For each seed transaction with `edge_id` and endpoints `(s, t) = edge_index[:, edge_id]`:
   - `lift_node_to_seed_edges([edge_id], edge_index, node_table)` → `[stats(s), stats(t), derived sums]`.

No per-edge precompute is required for global node metrics—only per-node tables and endpoint lift at train time.

### Tier 1 local (batch subgraph)

1. Use **forward** `edge_index` (and optionally `edge_id`) from the **batch subgraph** (`view1`), not the full split graph.
2. `compute_local_morphology(edge_index, seed_edge_ids, num_nodes)` counts degrees and size stats **inside that subgraph only**.
3. Targets align with what message passing can observe under `LinkNeighborLoader` sampling.

## Loader vs global mismatch

- **Global:** `deg_train(sender)` from the full train-split graph.
- **Local:** `deg_sub(sender)` from the current batch’s induced forward edges.

These are different targets; do not mix them without labeling (see morphology plan § Global vs local).
