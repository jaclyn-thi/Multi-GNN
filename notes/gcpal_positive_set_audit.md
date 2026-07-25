# GCPAL vs edge-centric positive-set audit

Machine-readable twin: `results/diagnostics/gcpal_positive_set_audit.json`

Read-only diagnostic. `tds=False`. Dense matrices only over surviving seed transactions (B=1678). No training run.

## 1. What GCPAL explicitly specifies

{
  "transactions_as_nodes": true,
  "two_random_views_edge_and_feature_dropping": true,
  "A_knn_definition": "top-k(X X^T) over transaction feature matrix X",
  "loss_combines_random_random_and_random_knn_contrast": true,
  "batch_size_search_up_to_approx": 2048,
  "k_optimum_reported_around": 15,
  "eq8_uses_P_i": true,
  "eq9_M_P": "A + A_knn",
  "ablation_without_neighbor_positives": "positives are the same nodes across the two random views (identity)"
}

## 2. What is implied but omitted

{
  "identity_in_M_P": "Eq. 9 writes M_P = A + A_knn without I, but the ablation that removes neighbor positives falls back to same-node-across-views identity. The implied practical positive set is therefore I \u222a A \u222a A_knn (equivalently I + A + A_knn with duplicates collapsed), even though I is absent from Eq. 9.",
  "connected_neighbors_definition": "A is described as connected neighbors on the transaction graph; exact adjacency (shared account vs directed payment chain) is not formalized beyond the transaction-as-node framing."
}

## 3. What cannot be reproduced without code

{
  "knn_scope": "Paper does not state whether KNN is global, batch-local, chunked, approximate, or precomputed.",
  "literal_global_XXT": "Literal dense X X^T over millions of AMLWorld transactions is infeasible; the paper does not release code clarifying the workaround.",
  "feature_matrix_X_columns": "Exact raw feature columns / categorical encoding for X are not specified at implementation level.",
  "code_release": false
}

## 4. How our current implementation differs

{
  "representation": {
    "gcpal": "transactions as nodes",
    "ours": "transactions as edges; z = concat(h_sender, h_receiver, edge_attr) then embedding head"
  },
  "default_positives": {
    "gcpal_implied": "I \u222a A \u222a A_knn across random and KNN views",
    "ours": "identity across two random views only (same edge_id)"
  },
  "knn_soft_positives_are_not_gcpal": {
    "flag": "--enable_knn_soft_positives",
    "why_not_equivalent": [
      "Uses an offline sparse train-split feature-KNN cache, not a KNN message-passing view",
      "Adds low-weight soft positives into the identity InfoNCE numerator (default weight 0.025, m=1)",
      "Does not build random\u2194KNN contrast as a separate view pair",
      "Requires --contrastive_asymmetric; incompatible with endpoint multipos / morph_contrast",
      "Prior Small-HI ablations underperformed vs identity baseline (see notes/results-archive.md)"
    ]
  },
  "knn_filter_is_not_gcpal": {
    "flag": "--enable_knn_negative_filter",
    "why_not_equivalent": [
      "Only excludes cached neighbors from the negative pool",
      "Never adds structural or KNN positives"
    ]
  },
  "endpoint_multipos_is_not_gcpal": {
    "flag": "--multi_positive_mode same_endpoint|same_pair|...",
    "why_not_equivalent": [
      "Batch-local weak endpoint positives only; no A_knn tier",
      "No third KNN graph view",
      "Weak weight default 0.1; still identity-primary"
    ]
  },
  "mapping_to_existing_flags": {
    "identity": "default contrastive path",
    "directed_chain": "not implemented as a training flag",
    "shared_endpoint": "--multi_positive_mode same_endpoint",
    "same_ordered_pair": "--multi_positive_mode same_pair",
    "feature_knn": "offline cache via --enable_knn_soft_positives / --enable_knn_negative_filter (not batch-local)"
  },
  "closest_to_transaction_node_line_graph": "shared-endpoint adjacency approximates undirected line-graph adjacency (transactions share an account). Directed-chain (receiver\u2192sender) is the directed payment-flow line-graph edge and is closer to money-flow succession."
}

## 5. Memory: dense pairwise storage only

| Scope | N | float32 sim GiB | feasible? |
|-------|--:|----------------:|:---------:|
| full Small-HI transactions | 5078345 | 9.607e+04 | False |
| train-split Small-HI transactions | 3248921 | 3.932e+04 | False |
| seed batch B=8192 | 8192 | 0.25 | True |
| seed batch B=2048 | 2048 | 0.01562 | True |

Mistaken MP-edge dense cost this batch: 176.4 GiB (n=217599), ratio vs seed: 16816.3×

## 6. Batch-local KNN feasibility (B≈2048)

- Comfortably feasible: **True**
- Peak CUDA allocated MiB: 78.56201171875
- Feature protocol: {
  "feature_set": "edge_native",
  "columns_named": [
    "Timestamp",
    "log1p_Amount Received",
    "Received Currency_ordinal",
    "Payment Format_ordinal"
  ],
  "categorical_columns": [
    "Received Currency",
    "Payment Format"
  ],
  "categorical_encoding": "ordinal factorize (documented choice: category IDs are treated as numeric magnitudes after StandardScaler; not one-hot). Labels unused.",
  "continuous": "Timestamp raw; Amount via log1p(Amount Received)",
  "scaling": "StandardScaler fit on full train split only; transform seed batch",
  "similarity": "cosine = L2-normalize rows then inner product",
  "k": 15,
  "self_excluded_before_topk": true,
  "labels_used_in_construction": false,
  "learned_representations_used_in_construction": false,
  "feature_prep_seconds": 2.9979719490002026,
  "n_train_fit_rows": 3248921,
  "feature_dim": 4
}

## 7. Positive-set measurements (non-identity stats)

| Definition | anchors | frac no non-id pos | median | mean | p95 | max | pairs | purity | dens |
|------------|--------:|-------------------:|-------:|-----:|----:|----:|------:|-------:|-----:|
| identity_only | 1678 | 1.000 | 0.00 | 0.00 | 0.00 | 0 | 0 | nan | 0 |
| directed_chain | 1678 | 0.995 | 0.00 | 0.01 | 0.00 | 1 | 9 | 1.0000 | 3.196e-06 |
| shared_endpoint | 1678 | 0.913 | 0.00 | 1.83 | 6.00 | 49 | 3064 | 0.9687 | 0.001088 |
| same_ordered_pair | 1678 | 0.996 | 0.00 | 0.00 | 0.00 | 1 | 6 | 1.0000 | 2.131e-06 |
| feature_knn_k15 | 1678 | 0.000 | 15.00 | 15.00 | 15.00 | 15 | 25170 | 0.9974 | 0.008939 |
| directed_chain_union_knn | 1678 | 0.000 | 15.00 | 15.00 | 15.00 | 16 | 25178 | 0.9974 | 0.008942 |
| shared_endpoint_union_knn | 1678 | 0.000 | 15.00 | 16.77 | 20.00 | 63 | 28136 | 0.9944 | 0.009993 |

## 8. Hub / near-duplicate risk

- **shared_endpoint**: {"n_unique_accounts_in_seed_batch": 2929, "top_1pct_account_count": 30, "fraction_nonid_positives_touching_top_1pct_accounts": 0.9908616187989556, "max_account_endpoint_degree_in_seed_batch": 49}
- **directed_chain**: {"n_unique_accounts_in_seed_batch": 2929, "top_1pct_account_count": 30, "fraction_nonid_positives_touching_top_1pct_accounts": 0.5555555555555556, "max_account_endpoint_degree_in_seed_batch": 49}
- **feature_knn_k15**: {"n_unique_accounts_in_seed_batch": 2929, "top_1pct_account_count": 30, "fraction_nonid_positives_touching_top_1pct_accounts": 0.15256257449344457, "max_account_endpoint_degree_in_seed_batch": 49}

KNN cosine sims: {"min": 0.7411143183708191, "mean": 0.9745877385139465, "median": 0.9815592765808105, "p95": 0.9991062879562378, "max": 0.9999998807907104, "n": 25170}

Global cache overlap: {
  "available": true,
  "cache_path": "morphology_cache/Small-HI/transaction_knn_edge_native_degree_fan_k15.npz",
  "cache_k": 15,
  "batch_local_knn_pairs": 25170,
  "batch_local_knn_pairs_also_in_global_cache_neighbor_list": 23,
  "fraction_batch_knn_supported_by_global_cache": 0.0009137862534763608,
  "global_cache_neighbors_of_seeds_that_also_fall_in_seed_batch": 24,
  "global_cache_neighbor_slots_examined": 25170,
  "note": "Global cache is train-split feature-KNN (edge_native+degree_fan); batch-local KNN here uses edge_native only \u2014 overlap is diagnostic, not exact feature parity."
}

## 9. Diagnostic answers

1. Seed-only B=2048 KNN feasible? **True**
2. KNN beyond identity? **True**
3. KNN more label-consistent than random? **False**
4. More defensible adjacency mapping? **directed_chain** ({'directed_chain': 'Default recommendation for GCPAL transaction-node adjacency mapped into edge-centric AMLWorld: directed money-flow succession (receiver→sender). Less hub-saturated than any-shared-account in typical payment graphs.', 'shared_endpoint': 'Undirected line-graph / share-any-account; matches existing --multi_positive_mode same_endpoint. Often hub-dominated.', 'measured_hub_fractions': {'directed_chain': {'n_unique_accounts_in_seed_batch': 2929, 'top_1pct_account_count': 30, 'fraction_nonid_positives_touching_top_1pct_accounts': 0.5555555555555556, 'max_account_endpoint_degree_in_seed_batch': 49}, 'shared_endpoint': {'n_unique_accounts_in_seed_batch': 2929, 'top_1pct_account_count': 30, 'fraction_nonid_positives_touching_top_1pct_accounts': 0.9908616187989556, 'max_account_endpoint_degree_in_seed_batch': 49}}})

## 10. Recommendation

D

Measurements support a controlled positive-set ablation (identity vs directed-chain vs shared-endpoint vs batch-local KNN) before implementing a third KNN message-passing view.

## 11. Smallest next training experiment (not launched)

Single smallest scout: identity + one structural definition (directed_chain), tds=False — not launched here.

