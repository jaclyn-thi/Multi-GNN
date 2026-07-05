# Transaction KNN feature audit

- **Dataset:** Small-HI
- **Rows audited:** 5000
- **k:** 15

Label-enrichment fields are **analysis only** and must not be used in training.

## Per feature set

### `edge_native+degree_fan` (ordinal, scaling=none)

- **Dimensions:** 12 across groups {'edge_native': 4, 'degree_fan': 8}
- **Finite values:** 1.000000 (nan=0, inf=0)
- **Similarity:** min=0.483454, mean=0.996790, p50=0.999995, p90=1.000000, p95=1.000000, p99=1.000000, max=1.000000
- **Diversity:** unique_neighbor_fraction=0.066613, hubness_top1=0.0005, hubness_top10=0.0041
- **Endpoint overlap:** same_sender=0.0191, same_receiver=0.0008, same_pair=0.0006
- **Label enrichment (analysis only):** label_same_fraction=1.0000

Features (12): Timestamp, log1p_Amount Received, Received Currency_ordinal, Payment Format_ordinal, log1p_sender_out_degree_train, log1p_sender_in_degree_train, log1p_receiver_out_degree_train, log1p_receiver_in_degree_train, log1p_sender_total_degree_train, log1p_receiver_total_degree_train, ... (+2 more)

### `edge_native+degree_fan` (one_hot, scaling=none)

- **Dimensions:** 20 across groups {'edge_native': 12, 'degree_fan': 8}
- **Finite values:** 1.000000 (nan=0, inf=0)
- **Similarity:** min=0.022837, mean=0.995561, p50=0.999996, p90=1.000000, p95=1.000000, p99=1.000000, max=1.000000
- **Diversity:** unique_neighbor_fraction=0.066600, hubness_top1=0.0004, hubness_top10=0.0037
- **Endpoint overlap:** same_sender=0.0165, same_receiver=0.0005, same_pair=0.0003
- **Label enrichment (analysis only):** label_same_fraction=1.0000

Features (20): Timestamp, log1p_Amount Received, Received Currency_0, Received Currency_12, Received Currency_2, Received Currency_8, Payment Format_0, Payment Format_1, Payment Format_2, Payment Format_3, ... (+10 more)

### `richer_v1` (one_hot, scaling=robust)

- **Dimensions:** 55 across groups {'edge_native': 12, 'time_bucket': 3, 'degree_fan': 8, 'degree_causal': 8, 'flow_rich': 10, 'relative_amount': 5, 'temporal_causal': 6, 'pair_history': 3}
- **Finite values:** 1.000000 (nan=0, inf=0)
- **Similarity:** min=0.231168, mean=0.995085, p50=0.999991, p90=1.000000, p95=1.000000, p99=1.000000, max=1.000000
- **Diversity:** unique_neighbor_fraction=0.066520, hubness_top1=0.0005, hubness_top10=0.0041
- **Endpoint overlap:** same_sender=0.0161, same_receiver=0.0000, same_pair=0.0000
- **Jaccard vs baseline:** 0.8800
- **Label enrichment (analysis only):** label_same_fraction=1.0000

Features (55): Timestamp, log1p_Amount Received, Received Currency_0, Received Currency_12, Received Currency_2, Received Currency_8, Payment Format_0, Payment Format_1, Payment Format_2, Payment Format_3, ... (+45 more)

### `richer_v1_no_pair` (one_hot, scaling=robust)

- **Dimensions:** 52 across groups {'edge_native': 12, 'time_bucket': 3, 'degree_fan': 8, 'degree_causal': 8, 'flow_rich': 10, 'relative_amount': 5, 'temporal_causal': 6}
- **Finite values:** 1.000000 (nan=0, inf=0)
- **Similarity:** min=0.258600, mean=0.995198, p50=0.999991, p90=1.000000, p95=1.000000, p99=1.000000, max=1.000000
- **Diversity:** unique_neighbor_fraction=0.066520, hubness_top1=0.0005, hubness_top10=0.0041
- **Endpoint overlap:** same_sender=0.0163, same_receiver=0.0000, same_pair=0.0000
- **Jaccard vs baseline:** 0.8797
- **Label enrichment (analysis only):** label_same_fraction=1.0000

Features (52): Timestamp, log1p_Amount Received, Received Currency_0, Received Currency_12, Received Currency_2, Received Currency_8, Payment Format_0, Payment Format_1, Payment Format_2, Payment Format_3, ... (+42 more)

### `richer_v1` (ordinal, scaling=robust)

- **Dimensions:** 47 across groups {'edge_native': 4, 'time_bucket': 3, 'degree_fan': 8, 'degree_causal': 8, 'flow_rich': 10, 'relative_amount': 5, 'temporal_causal': 6, 'pair_history': 3}
- **Finite values:** 1.000000 (nan=0, inf=0)
- **Similarity:** min=0.221611, mean=0.996072, p50=0.999992, p90=1.000000, p95=1.000000, p99=1.000000, max=1.000000
- **Diversity:** unique_neighbor_fraction=0.066533, hubness_top1=0.0005, hubness_top10=0.0041
- **Endpoint overlap:** same_sender=0.0160, same_receiver=0.0000, same_pair=0.0000
- **Jaccard vs baseline:** 0.8830
- **Label enrichment (analysis only):** label_same_fraction=1.0000

Features (47): Timestamp, log1p_Amount Received, Received Currency_ordinal, Payment Format_ordinal, timestamp_norm_train_span, hour_of_day_fraction, day_phase_fraction, log1p_sender_out_degree_train, log1p_sender_in_degree_train, log1p_receiver_out_degree_train, ... (+37 more)

## Pairwise top-k Jaccard overlap

| A | B | mean Jaccard |
|---|---|--------------|
| `edge_native+degree_fan|ordinal|legacy_standard` | `edge_native+degree_fan|one_hot|legacy_standard` | 0.9441 |
| `edge_native+degree_fan|ordinal|legacy_standard` | `richer_v1|one_hot|robust` | 0.8800 |
| `edge_native+degree_fan|ordinal|legacy_standard` | `richer_v1_no_pair|one_hot|robust` | 0.8797 |
| `edge_native+degree_fan|ordinal|legacy_standard` | `richer_v1|ordinal|robust` | 0.8830 |
| `edge_native+degree_fan|one_hot|legacy_standard` | `richer_v1|one_hot|robust` | 0.8792 |
| `edge_native+degree_fan|one_hot|legacy_standard` | `richer_v1_no_pair|one_hot|robust` | 0.8790 |
| `edge_native+degree_fan|one_hot|legacy_standard` | `richer_v1|ordinal|robust` | 0.8731 |
| `richer_v1|one_hot|robust` | `richer_v1_no_pair|one_hot|robust` | 0.9976 |
| `richer_v1|one_hot|robust` | `richer_v1|ordinal|robust` | 0.9643 |
| `richer_v1_no_pair|one_hot|robust` | `richer_v1|ordinal|robust` | 0.9624 |
