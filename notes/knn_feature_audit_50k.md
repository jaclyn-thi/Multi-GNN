# Transaction KNN feature audit (50k train rows)

- **Dataset:** Small-HI
- **Rows audited:** 50000
- **k:** 15

Label-enrichment columns are **analysis only** and must not be used in training.

## Per feature set

### `edge_native+degree_fan` (ordinal)

- **Dimensions:** 12
- **Categorical encoding:** `ordinal` for currency/payment format when `edge_native` is included
- **Similarity:** min=0.679416, mean=0.996555, p25=0.999250, p50=0.999977, p75=0.999999, p95=1.000000, max=1.000000
- **Neighbor diversity:** unique_neighbor_fraction=0.066619, duplicate_neighbor_rows=0
- **Endpoint overlap:** same_sender=0.0161, same_receiver=0.0010, same_pair=0.0007
- **Label enrichment (analysis only):** label_same_fraction=0.9999, neighbor_positive_rate=0.0001, anchor_positive_rate=0.0001

Feature names (12): Timestamp, log1p_Amount Received, Received Currency_ordinal, Payment Format_ordinal, log1p_sender_out_degree_train, log1p_sender_in_degree_train, log1p_receiver_out_degree_train, log1p_receiver_in_degree_train, log1p_sender_total_degree_train, log1p_receiver_total_degree_train, log1p_pair_out_degree_sum_train, log1p_pair_in_degree_sum_train

### `edge_native+degree_fan` (one_hot)

- **Dimensions:** 32
- **Categorical encoding:** `one_hot` for currency/payment format when `edge_native` is included
- **Similarity:** min=0.588918, mean=0.995363, p25=0.999813, p50=0.999992, p75=0.999999, p95=1.000000, max=1.000000
- **Neighbor diversity:** unique_neighbor_fraction=0.066588, duplicate_neighbor_rows=0
- **Endpoint overlap:** same_sender=0.0188, same_receiver=0.0007, same_pair=0.0004
- **Label enrichment (analysis only):** label_same_fraction=0.9999, neighbor_positive_rate=0.0001, anchor_positive_rate=0.0001

Feature names (32): Timestamp, log1p_Amount Received, Received Currency_0, Received Currency_1, Received Currency_10, Received Currency_11, Received Currency_12, Received Currency_13, Received Currency_14, Received Currency_2, Received Currency_3, Received Currency_4, ... (+20 more)

### `edge_native+degree_fan+flow_balance` (ordinal)

- **Dimensions:** 18
- **Categorical encoding:** `ordinal` for currency/payment format when `edge_native` is included
- **Similarity:** min=0.587795, mean=0.995003, p25=0.998713, p50=0.999956, p75=0.999998, p95=1.000000, max=1.000000
- **Neighbor diversity:** unique_neighbor_fraction=0.066608, duplicate_neighbor_rows=0
- **Endpoint overlap:** same_sender=0.0168, same_receiver=0.0015, same_pair=0.0012
- **Label enrichment (analysis only):** label_same_fraction=0.9999, neighbor_positive_rate=0.0001, anchor_positive_rate=0.0001

Feature names (18): Timestamp, log1p_Amount Received, Received Currency_ordinal, Payment Format_ordinal, log1p_sender_out_degree_train, log1p_sender_in_degree_train, log1p_receiver_out_degree_train, log1p_receiver_in_degree_train, log1p_sender_total_degree_train, log1p_receiver_total_degree_train, log1p_pair_out_degree_sum_train, log1p_pair_in_degree_sum_train, ... (+6 more)

### `edge_native+degree_fan+flow_balance` (one_hot)

- **Dimensions:** 38
- **Categorical encoding:** `one_hot` for currency/payment format when `edge_native` is included
- **Similarity:** min=0.577562, mean=0.992622, p25=0.999263, p50=0.999977, p75=0.999999, p95=1.000000, max=1.000000
- **Neighbor diversity:** unique_neighbor_fraction=0.066565, duplicate_neighbor_rows=0
- **Endpoint overlap:** same_sender=0.0191, same_receiver=0.0009, same_pair=0.0006
- **Label enrichment (analysis only):** label_same_fraction=0.9999, neighbor_positive_rate=0.0001, anchor_positive_rate=0.0001

Feature names (38): Timestamp, log1p_Amount Received, Received Currency_0, Received Currency_1, Received Currency_10, Received Currency_11, Received Currency_12, Received Currency_13, Received Currency_14, Received Currency_2, Received Currency_3, Received Currency_4, ... (+26 more)

### `edge_native+degree_fan+flow_balance+temporal_behavior` (ordinal)

- **Dimensions:** 20
- **Categorical encoding:** `ordinal` for currency/payment format when `edge_native` is included
- **Similarity:** min=0.017774, mean=0.994558, p25=0.998738, p50=0.999959, p75=0.999998, p95=1.000000, max=1.000000
- **Neighbor diversity:** unique_neighbor_fraction=0.066601, duplicate_neighbor_rows=0
- **Endpoint overlap:** same_sender=0.0165, same_receiver=0.0013, same_pair=0.0011
- **Label enrichment (analysis only):** label_same_fraction=0.9999, neighbor_positive_rate=0.0001, anchor_positive_rate=0.0001

Feature names (20): Timestamp, log1p_Amount Received, Received Currency_ordinal, Payment Format_ordinal, log1p_sender_out_degree_train, log1p_sender_in_degree_train, log1p_receiver_out_degree_train, log1p_receiver_in_degree_train, log1p_sender_total_degree_train, log1p_receiver_total_degree_train, log1p_pair_out_degree_sum_train, log1p_pair_in_degree_sum_train, ... (+8 more)

### `edge_native+degree_fan+flow_balance+temporal_behavior` (one_hot)

- **Dimensions:** 40
- **Categorical encoding:** `one_hot` for currency/payment format when `edge_native` is included
- **Similarity:** min=0.038576, mean=0.991808, p25=0.999274, p50=0.999978, p75=0.999999, p95=1.000000, max=1.000000
- **Neighbor diversity:** unique_neighbor_fraction=0.066579, duplicate_neighbor_rows=0
- **Endpoint overlap:** same_sender=0.0191, same_receiver=0.0009, same_pair=0.0006
- **Label enrichment (analysis only):** label_same_fraction=0.9999, neighbor_positive_rate=0.0001, anchor_positive_rate=0.0001

Feature names (40): Timestamp, log1p_Amount Received, Received Currency_0, Received Currency_1, Received Currency_10, Received Currency_11, Received Currency_12, Received Currency_13, Received Currency_14, Received Currency_2, Received Currency_3, Received Currency_4, ... (+28 more)

## Top-k neighbor Jaccard overlap

| A | B | mean Jaccard |
|---|---|--------------|
| `edge_native+degree_fan|ordinal` | `edge_native+degree_fan|one_hot` | 0.8149 |
| `edge_native+degree_fan|ordinal` | `edge_native+degree_fan+flow_balance|ordinal` | 0.8420 |
| `edge_native+degree_fan|ordinal` | `edge_native+degree_fan+flow_balance|one_hot` | 0.7873 |
| `edge_native+degree_fan|ordinal` | `edge_native+degree_fan+flow_balance+temporal_behavior|ordinal` | 0.8463 |
| `edge_native+degree_fan|ordinal` | `edge_native+degree_fan+flow_balance+temporal_behavior|one_hot` | 0.7940 |
| `edge_native+degree_fan|one_hot` | `edge_native+degree_fan+flow_balance|ordinal` | 0.7546 |
| `edge_native+degree_fan|one_hot` | `edge_native+degree_fan+flow_balance|one_hot` | 0.8761 |
| `edge_native+degree_fan|one_hot` | `edge_native+degree_fan+flow_balance+temporal_behavior|ordinal` | 0.7547 |
| `edge_native+degree_fan|one_hot` | `edge_native+degree_fan+flow_balance+temporal_behavior|one_hot` | 0.8939 |
| `edge_native+degree_fan+flow_balance|ordinal` | `edge_native+degree_fan+flow_balance|one_hot` | 0.7905 |
| `edge_native+degree_fan+flow_balance|ordinal` | `edge_native+degree_fan+flow_balance+temporal_behavior|ordinal` | 0.9381 |
| `edge_native+degree_fan+flow_balance|ordinal` | `edge_native+degree_fan+flow_balance+temporal_behavior|one_hot` | 0.7847 |
| `edge_native+degree_fan+flow_balance|one_hot` | `edge_native+degree_fan+flow_balance+temporal_behavior|ordinal` | 0.7810 |
| `edge_native+degree_fan+flow_balance|one_hot` | `edge_native+degree_fan+flow_balance+temporal_behavior|one_hot` | 0.9374 |
| `edge_native+degree_fan+flow_balance+temporal_behavior|ordinal` | `edge_native+degree_fan+flow_balance+temporal_behavior|one_hot` | 0.7857 |
