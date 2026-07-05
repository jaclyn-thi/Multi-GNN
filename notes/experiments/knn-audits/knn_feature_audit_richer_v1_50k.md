# Transaction KNN feature audit

- **Dataset:** Small-HI
- **Rows audited:** 50000
- **k:** 15

Label-enrichment fields are **analysis only** and must not be used in training.

## Per feature set

### `edge_native+degree_fan` (ordinal, scaling=legacy_standard)

- **Dimensions:** 12 across groups {'edge_native': 4, 'degree_fan': 8}
- **Finite values:** 1.000000 (nan=0, inf=0)
- **Similarity:** min=0.679416, mean=0.996555, p50=0.999977, p90=1.000000, p95=1.000000, p99=1.000000, max=1.000000
- **Diversity:** unique_neighbor_fraction=0.066619, hubness_top1=0.0001, hubness_top10=0.0006
- **Endpoint overlap:** same_sender=0.0161, same_receiver=0.0010, same_pair=0.0007
- **Label enrichment (analysis only):** label_same_fraction=0.9999

Features (12): Timestamp, log1p_Amount Received, Received Currency_ordinal, Payment Format_ordinal, log1p_sender_out_degree_train, log1p_sender_in_degree_train, log1p_receiver_out_degree_train, log1p_receiver_in_degree_train, log1p_sender_total_degree_train, log1p_receiver_total_degree_train, ... (+2 more)

### `edge_native+degree_fan` (one_hot, scaling=legacy_standard)

- **Dimensions:** 32 across groups {'edge_native': 24, 'degree_fan': 8}
- **Finite values:** 1.000000 (nan=0, inf=0)
- **Similarity:** min=0.588918, mean=0.995363, p50=0.999992, p90=1.000000, p95=1.000000, p99=1.000000, max=1.000000
- **Diversity:** unique_neighbor_fraction=0.066588, hubness_top1=0.0001, hubness_top10=0.0008
- **Endpoint overlap:** same_sender=0.0188, same_receiver=0.0007, same_pair=0.0004
- **Jaccard vs baseline:** 0.8149
- **Label enrichment (analysis only):** label_same_fraction=0.9999

Features (32): Timestamp, log1p_Amount Received, Received Currency_0, Received Currency_1, Received Currency_10, Received Currency_11, Received Currency_12, Received Currency_13, Received Currency_14, Received Currency_2, ... (+22 more)

### `richer_v1` (one_hot, scaling=robust)

- **Dimensions:** 67 across groups {'edge_native': 24, 'time_bucket': 3, 'degree_fan': 8, 'degree_causal': 8, 'flow_rich': 10, 'relative_amount': 5, 'temporal_causal': 6, 'pair_history': 3}
- **Finite values:** 1.000000 (nan=0, inf=0)
- **Similarity:** min=0.591943, mean=0.993169, p50=0.999955, p90=1.000000, p95=1.000000, p99=1.000000, max=1.000000
- **Diversity:** unique_neighbor_fraction=0.066473, hubness_top1=0.0001, hubness_top10=0.0006
- **Endpoint overlap:** same_sender=0.0140, same_receiver=0.0001, same_pair=0.0000
- **Jaccard vs baseline:** 0.7552
- **Label enrichment (analysis only):** label_same_fraction=0.9999

Features (67): Timestamp, log1p_Amount Received, Received Currency_0, Received Currency_1, Received Currency_10, Received Currency_11, Received Currency_12, Received Currency_13, Received Currency_14, Received Currency_2, ... (+57 more)

### `richer_v1_no_pair` (one_hot, scaling=robust)

- **Dimensions:** 64 across groups {'edge_native': 24, 'time_bucket': 3, 'degree_fan': 8, 'degree_causal': 8, 'flow_rich': 10, 'relative_amount': 5, 'temporal_causal': 6}
- **Finite values:** 1.000000 (nan=0, inf=0)
- **Similarity:** min=0.601838, mean=0.993177, p50=0.999956, p90=1.000000, p95=1.000000, p99=1.000000, max=1.000000
- **Diversity:** unique_neighbor_fraction=0.066477, hubness_top1=0.0001, hubness_top10=0.0006
- **Endpoint overlap:** same_sender=0.0141, same_receiver=0.0001, same_pair=0.0000
- **Jaccard vs baseline:** 0.7551
- **Label enrichment (analysis only):** label_same_fraction=0.9999

Features (64): Timestamp, log1p_Amount Received, Received Currency_0, Received Currency_1, Received Currency_10, Received Currency_11, Received Currency_12, Received Currency_13, Received Currency_14, Received Currency_2, ... (+54 more)

### `richer_v1` (ordinal, scaling=robust)

- **Dimensions:** 47 across groups {'edge_native': 4, 'time_bucket': 3, 'degree_fan': 8, 'degree_causal': 8, 'flow_rich': 10, 'relative_amount': 5, 'temporal_causal': 6, 'pair_history': 3}
- **Finite values:** 1.000000 (nan=0, inf=0)
- **Similarity:** min=0.654253, mean=0.996500, p50=0.999952, p90=1.000000, p95=1.000000, p99=1.000000, max=1.000000
- **Diversity:** unique_neighbor_fraction=0.066540, hubness_top1=0.0001, hubness_top10=0.0005
- **Endpoint overlap:** same_sender=0.0126, same_receiver=0.0001, same_pair=0.0000
- **Jaccard vs baseline:** 0.7717
- **Label enrichment (analysis only):** label_same_fraction=0.9999

Features (47): Timestamp, log1p_Amount Received, Received Currency_ordinal, Payment Format_ordinal, timestamp_norm_train_span, hour_of_day_fraction, day_phase_fraction, log1p_sender_out_degree_train, log1p_sender_in_degree_train, log1p_receiver_out_degree_train, ... (+37 more)

## Pairwise top-k Jaccard overlap

| A | B | mean Jaccard |
|---|---|--------------|
| `edge_native+degree_fan|ordinal|legacy_standard` | `edge_native+degree_fan|one_hot|legacy_standard` | 0.8149 |
| `edge_native+degree_fan|ordinal|legacy_standard` | `richer_v1|one_hot|robust` | 0.7552 |
| `edge_native+degree_fan|ordinal|legacy_standard` | `richer_v1_no_pair|one_hot|robust` | 0.7551 |
| `edge_native+degree_fan|ordinal|legacy_standard` | `richer_v1|ordinal|robust` | 0.7717 |
| `edge_native+degree_fan|one_hot|legacy_standard` | `richer_v1|one_hot|robust` | 0.7858 |
| `edge_native+degree_fan|one_hot|legacy_standard` | `richer_v1_no_pair|one_hot|robust` | 0.7858 |
| `edge_native+degree_fan|one_hot|legacy_standard` | `richer_v1|ordinal|robust` | 0.7143 |
| `richer_v1|one_hot|robust` | `richer_v1_no_pair|one_hot|robust` | 0.9875 |
| `richer_v1|one_hot|robust` | `richer_v1|ordinal|robust` | 0.8332 |
| `richer_v1_no_pair|one_hot|robust` | `richer_v1|ordinal|robust` | 0.8289 |
