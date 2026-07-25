# GCPAL positive-set multi-batch KNN diagnostic

Read-only. `tds=False`. Seed B≤2048. Pairwise over seed transactions only. **No** third KNN MP view. **No** contrastive training.

- Batches processed: **64** (cap 64)
- Minority anchors achieved: **83** (target 100)
- Train positive rate: **0.000779**
- Primary sampling: natural LinkNeighborLoader batches (no edge-drop filter on seeds)

## Feature protocols

### `edge_native_ordinal`
- columns: `['Timestamp', 'log1p_Amount Received', 'Received Currency_ordinal', 'Payment Format_ordinal']`
- transformations: ['Timestamp raw', 'log1p(Amount Received)', 'Received Currency / Payment Format: ordinal factorize (sort=True)']
- scaler: StandardScaler fit on full train split; transform all train rows
- categorical: ordinal
- similarity / k / self: cosine (row L2-normalize then inner product) / 15 / diagonal set to -inf before top-k
- candidate pool: batch-local (seed transactions only)
- dim: 4

### `edge_native_onehot`
- columns: `['Timestamp', 'log1p_Amount Received', 'Received Currency_0', 'Received Currency_1', 'Received Currency_10', 'Received Currency_11', 'Received Currency_12', 'Received Currency_13', 'Received Currency_14', 'Received Currency_2', 'Received Currency_3', 'Received Currency_4', 'Received Currency_5', 'Received Currency_6', 'Received Currency_7', 'Received Currency_8', 'Received Currency_9', 'Payment Format_0', 'Payment Format_1', 'Payment Format_2', 'Payment Format_3', 'Payment Format_4', 'Payment Format_5', 'Payment Format_6']`
- transformations: ['Timestamp + log1p(Amount): StandardScaler on train', 'Currency/Payment Format: one-hot (train vocabulary); not scaled', 'Concat then row L2-normalize']
- scaler: StandardScaler on continuous columns only, full train split
- categorical: one_hot
- similarity / k / self: cosine (row L2-normalize then inner product) / 15 / diagonal set to -inf before top-k
- candidate pool: batch-local (seed transactions only)
- dim: 24

### `global_cache_matched`
- columns: `['Timestamp', 'log1p_Amount Received', 'Received Currency_ordinal', 'Payment Format_ordinal', 'log1p_sender_out_degree_train', 'log1p_sender_in_degree_train', 'log1p_receiver_out_degree_train', 'log1p_receiver_in_degree_train', 'log1p_sender_total_degree_train', 'log1p_receiver_total_degree_train', 'log1p_pair_out_degree_sum_train', 'log1p_pair_in_degree_sum_train']`
- transformations: ['edge_native (ordinal) + degree_fan train-graph degrees', 'legacy_standard = global StandardScaler over all columns', 'row L2-normalize for cosine (matches precompute --metric cosine)']
- scaler: StandardScaler fit on full train feature matrix (same as cache precompute default)
- categorical: ordinal
- similarity / k / self: cosine / 15 / diagonal set to -inf before top-k (batch-local); cache excludes self
- candidate pool: batch-local for primary; also compared to global/cache neighbors
- dim: 12

## Pooled minority / majority KNN metrics

### edge_native_ordinal
- **minority** (n=83): P@15 mean=0.00321285140562249 lift=4.125810435417559 frac≥1 same=0.04819277108433735 avg_sim=0.9664718930022307
- **majority** (n=130989): P@15 mean=0.9994462639356486 lift=1.0002251593452762 frac≥1 same=1.0 avg_sim=0.9783707857923719

### edge_native_onehot
- **minority** (n=83): P@15 mean=0.0024096385542168677 lift=3.0943578265631695 frac≥1 same=0.036144578313253004 avg_sim=0.8843944738667652
- **majority** (n=130989): P@15 mean=0.9993923153852615 lift=1.0001711687513302 frac≥1 same=1.0 avg_sim=0.9193445847620401

### global_cache_matched
- **minority** (n=83): P@15 mean=0.00321285140562249 lift=4.125810435417559 frac≥1 same=0.04819277108433735 avg_sim=0.8577200552067124
- **majority** (n=130989): P@15 mean=0.9995414373217092 lift=1.0003204069025216 frac≥1 same=1.0 avg_sim=0.8915638868930793

## Cross-batch summary (minority P@15 by variant)

| Variant | mean | SD | median | p25 | p75 |
|---------|-----:|---:|-------:|----:|----:|
| edge_native_ordinal | 0.0016908212560386474 | 0.008103375748102765 | 0.0 | 0.0 | 0.0 |
| edge_native_onehot | 0.001497584541062802 | 0.006109280943925893 | 0.0 | 0.0 | 0.0 |
| global_cache_matched | 0.0015458937198067632 | 0.007566724717855543 | 0.0 | 0.0 | 0.0 |

## Batch↔global overlap (matched cache features)

```json
{
  "n_batches_with_overlap": 64,
  "fraction_batch_knn_in_global_cache": {
    "n": 64,
    "mean": 0.0005661010742187499,
    "std": 0.0001956237519635912,
    "min": 0.00026041666666666666,
    "p25": 0.00041503906250000004,
    "median": 0.0005533854166666667,
    "p75": 0.0007161458333333333,
    "p95": 0.0008740234374999998,
    "max": 0.0011393229166666667
  },
  "global_cache_neighbors_landing_in_seed_batch": {
    "n": 64,
    "mean": 17.9375,
    "std": 6.14087269533553,
    "min": 8.0,
    "p25": 14.0,
    "median": 17.0,
    "p75": 22.0,
    "p95": 27.849999999999994,
    "max": 36.0
  },
  "note": "Computed only for global_cache_matched batch-local KNN vs existing cache."
}
```

## Neighbor stability

```json
{
  "edge_native_ordinal": {
    "n_seeds_seen_in_multiple_batches": 0,
    "pairwise_neighbor_jaccard_across_batch_compositions": {
      "n": 0,
      "mean": null,
      "std": null,
      "min": null,
      "p25": null,
      "median": null,
      "p75": null,
      "p95": null,
      "max": null
    }
  },
  "edge_native_onehot": {
    "n_seeds_seen_in_multiple_batches": 0,
    "pairwise_neighbor_jaccard_across_batch_compositions": {
      "n": 0,
      "mean": null,
      "std": null,
      "min": null,
      "p25": null,
      "median": null,
      "p75": null,
      "p95": null,
      "max": null
    }
  },
  "global_cache_matched": {
    "n_seeds_seen_in_multiple_batches": 0,
    "pairwise_neighbor_jaccard_across_batch_compositions": {
      "n": 0,
      "mean": null,
      "std": null,
      "min": null,
      "p25": null,
      "median": null,
      "p75": null,
      "p95": null,
      "max": null
    }
  }
}
```

## Structural coverage

```json
{
  "directed_chain_nonid_pairs_in_batch": {
    "n": 64,
    "mean": 42.671875,
    "std": 37.16375375623362,
    "min": 7.0,
    "p25": 14.75,
    "median": 24.5,
    "p75": 70.25,
    "p95": 102.85,
    "max": 186.0
  },
  "shared_endpoint_nonid_pairs_in_batch": {
    "n": 64,
    "mean": 6047.375,
    "std": 1057.3698351352607,
    "min": 3872.0,
    "p25": 5299.5,
    "median": 5816.0,
    "p75": 6520.5,
    "p95": 8221.499999999998,
    "max": 8718.0
  },
  "fraction_full_chain_neighbors_captured_in_batch": {
    "n": 64,
    "mean": 0.0006181073268761008,
    "std": 0.0001453149862677489,
    "min": 0.0003052636169377698,
    "p25": 0.000521013257761087,
    "median": 0.0006215477527385057,
    "p75": 0.0006772208125064098,
    "p95": 0.0008615901874597303,
    "max": 0.0010903143739778303
  },
  "directed_chain_neighbors_absent_from_batch": {
    "n": 64,
    "mean": 33.87895965576172,
    "std": 29.218188563920254,
    "min": 9.51806640625,
    "p25": 10.7392578125,
    "median": 16.46142578125,
    "p75": 59.451171875,
    "p95": 86.87124023437488,
    "max": 139.734375
  }
}
```

## Runtime / memory

```json
{
  "data_load_seconds": 300.699100783997,
  "feature_prep_seconds": 8.139879081994877,
  "knn_similarity_seconds_per_batch": {
    "n": 192,
    "mean": 0.003367534874693471,
    "std": 0.027970246767122633,
    "min": 0.0010783699981402606,
    "p25": 0.0012704627442872152,
    "median": 0.0013239799955044873,
    "p75": 0.0013982224954816047,
    "p95": 0.0014946317482099401,
    "max": 0.38890310599526856
  },
  "wall_seconds": 349.2982412150013,
  "device": "cuda:0",
  "peak_allocated_mib": 45.7119140625,
  "peak_reserved_mib": 54.0
}
```

## Decision

**D** — Borderline/insufficient: n_minority=83, best P@15=0.00321285140562249, lift=4.125810435417559, stability_jaccard=None.
