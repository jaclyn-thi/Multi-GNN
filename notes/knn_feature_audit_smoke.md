# Transaction KNN feature audit (50k train rows)

- **Dataset:** Small-HI
- **Rows audited:** 500
- **k:** 15

Label-enrichment columns are **analysis only** and must not be used in training.

## Per feature set

### `edge_native+degree_fan` (ordinal)

- **Dimensions:** 12
- **Categorical encoding:** `ordinal` for currency/payment format when `edge_native` is included
- **Similarity:** min=0.203015, mean=0.979614, p25=0.998623, p50=0.999769, p75=0.999980, p95=0.999999, max=1.000000
- **Neighbor diversity:** unique_neighbor_fraction=0.066400, duplicate_neighbor_rows=0
- **Endpoint overlap:** same_sender=0.0131, same_receiver=0.0012, same_pair=0.0011
- **Label enrichment (analysis only):** label_same_fraction=1.0000, neighbor_positive_rate=0.0000, anchor_positive_rate=0.0000

Feature names (12): Timestamp, log1p_Amount Received, Received Currency_ordinal, Payment Format_ordinal, log1p_sender_out_degree_train, log1p_sender_in_degree_train, log1p_receiver_out_degree_train, log1p_receiver_in_degree_train, log1p_sender_total_degree_train, log1p_receiver_total_degree_train, log1p_pair_out_degree_sum_train, log1p_pair_in_degree_sum_train

### `edge_native+degree_fan` (one_hot)

- **Dimensions:** 17
- **Categorical encoding:** `one_hot` for currency/payment format when `edge_native` is included
- **Similarity:** min=0.121927, mean=0.973396, p25=0.999002, p50=0.999812, p75=0.999982, p95=0.999999, max=1.000000
- **Neighbor diversity:** unique_neighbor_fraction=0.066667, duplicate_neighbor_rows=0
- **Endpoint overlap:** same_sender=0.0131, same_receiver=0.0012, same_pair=0.0011
- **Label enrichment (analysis only):** label_same_fraction=1.0000, neighbor_positive_rate=0.0000, anchor_positive_rate=0.0000

Feature names (17): Timestamp, log1p_Amount Received, Received Currency_0, Payment Format_0, Payment Format_1, Payment Format_2, Payment Format_3, Payment Format_4, Payment Format_5, log1p_sender_out_degree_train, log1p_sender_in_degree_train, log1p_receiver_out_degree_train, ... (+5 more)

### `edge_native+degree_fan+flow_balance` (ordinal)

- **Dimensions:** 18
- **Categorical encoding:** `ordinal` for currency/payment format when `edge_native` is included
- **Similarity:** min=0.309242, mean=0.978111, p25=0.997542, p50=0.999764, p75=0.999978, p95=0.999999, max=1.000000
- **Neighbor diversity:** unique_neighbor_fraction=0.066400, duplicate_neighbor_rows=0
- **Endpoint overlap:** same_sender=0.0131, same_receiver=0.0012, same_pair=0.0011
- **Label enrichment (analysis only):** label_same_fraction=1.0000, neighbor_positive_rate=0.0000, anchor_positive_rate=0.0000

Feature names (18): Timestamp, log1p_Amount Received, Received Currency_ordinal, Payment Format_ordinal, log1p_sender_out_degree_train, log1p_sender_in_degree_train, log1p_receiver_out_degree_train, log1p_receiver_in_degree_train, log1p_sender_total_degree_train, log1p_receiver_total_degree_train, log1p_pair_out_degree_sum_train, log1p_pair_in_degree_sum_train, ... (+6 more)

### `edge_native+degree_fan+flow_balance` (one_hot)

- **Dimensions:** 23
- **Categorical encoding:** `one_hot` for currency/payment format when `edge_native` is included
- **Similarity:** min=0.180926, mean=0.970118, p25=0.998069, p50=0.999772, p75=0.999978, p95=0.999999, max=1.000000
- **Neighbor diversity:** unique_neighbor_fraction=0.066533, duplicate_neighbor_rows=0
- **Endpoint overlap:** same_sender=0.0131, same_receiver=0.0012, same_pair=0.0011
- **Label enrichment (analysis only):** label_same_fraction=1.0000, neighbor_positive_rate=0.0000, anchor_positive_rate=0.0000

Feature names (23): Timestamp, log1p_Amount Received, Received Currency_0, Payment Format_0, Payment Format_1, Payment Format_2, Payment Format_3, Payment Format_4, Payment Format_5, log1p_sender_out_degree_train, log1p_sender_in_degree_train, log1p_receiver_out_degree_train, ... (+11 more)

### `edge_native+degree_fan+flow_balance+temporal_behavior` (ordinal)

- **Dimensions:** 20
- **Categorical encoding:** `ordinal` for currency/payment format when `edge_native` is included
- **Similarity:** min=0.309242, mean=0.978111, p25=0.997542, p50=0.999764, p75=0.999978, p95=0.999999, max=1.000000
- **Neighbor diversity:** unique_neighbor_fraction=0.066400, duplicate_neighbor_rows=0
- **Endpoint overlap:** same_sender=0.0131, same_receiver=0.0012, same_pair=0.0011
- **Label enrichment (analysis only):** label_same_fraction=1.0000, neighbor_positive_rate=0.0000, anchor_positive_rate=0.0000

Feature names (20): Timestamp, log1p_Amount Received, Received Currency_ordinal, Payment Format_ordinal, log1p_sender_out_degree_train, log1p_sender_in_degree_train, log1p_receiver_out_degree_train, log1p_receiver_in_degree_train, log1p_sender_total_degree_train, log1p_receiver_total_degree_train, log1p_pair_out_degree_sum_train, log1p_pair_in_degree_sum_train, ... (+8 more)

### `edge_native+degree_fan+flow_balance+temporal_behavior` (one_hot)

- **Dimensions:** 25
- **Categorical encoding:** `one_hot` for currency/payment format when `edge_native` is included
- **Similarity:** min=0.180926, mean=0.970118, p25=0.998069, p50=0.999772, p75=0.999978, p95=0.999999, max=1.000000
- **Neighbor diversity:** unique_neighbor_fraction=0.066533, duplicate_neighbor_rows=0
- **Endpoint overlap:** same_sender=0.0131, same_receiver=0.0012, same_pair=0.0011
- **Label enrichment (analysis only):** label_same_fraction=1.0000, neighbor_positive_rate=0.0000, anchor_positive_rate=0.0000

Feature names (25): Timestamp, log1p_Amount Received, Received Currency_0, Payment Format_0, Payment Format_1, Payment Format_2, Payment Format_3, Payment Format_4, Payment Format_5, log1p_sender_out_degree_train, log1p_sender_in_degree_train, log1p_receiver_out_degree_train, ... (+13 more)

## Top-k neighbor Jaccard overlap

| A | B | mean Jaccard |
|---|---|--------------|
| `edge_native+degree_fan|ordinal` | `edge_native+degree_fan|one_hot` | 0.9073 |
| `edge_native+degree_fan|ordinal` | `edge_native+degree_fan+flow_balance|ordinal` | 0.9357 |
| `edge_native+degree_fan|ordinal` | `edge_native+degree_fan+flow_balance|one_hot` | 0.8975 |
| `edge_native+degree_fan|ordinal` | `edge_native+degree_fan+flow_balance+temporal_behavior|ordinal` | 0.9357 |
| `edge_native+degree_fan|ordinal` | `edge_native+degree_fan+flow_balance+temporal_behavior|one_hot` | 0.8975 |
| `edge_native+degree_fan|one_hot` | `edge_native+degree_fan+flow_balance|ordinal` | 0.8788 |
| `edge_native+degree_fan|one_hot` | `edge_native+degree_fan+flow_balance|one_hot` | 0.9648 |
| `edge_native+degree_fan|one_hot` | `edge_native+degree_fan+flow_balance+temporal_behavior|ordinal` | 0.8788 |
| `edge_native+degree_fan|one_hot` | `edge_native+degree_fan+flow_balance+temporal_behavior|one_hot` | 0.9648 |
| `edge_native+degree_fan+flow_balance|ordinal` | `edge_native+degree_fan+flow_balance|one_hot` | 0.8885 |
| `edge_native+degree_fan+flow_balance|ordinal` | `edge_native+degree_fan+flow_balance+temporal_behavior|ordinal` | 1.0000 |
| `edge_native+degree_fan+flow_balance|ordinal` | `edge_native+degree_fan+flow_balance+temporal_behavior|one_hot` | 0.8885 |
| `edge_native+degree_fan+flow_balance|one_hot` | `edge_native+degree_fan+flow_balance+temporal_behavior|ordinal` | 0.8885 |
| `edge_native+degree_fan+flow_balance|one_hot` | `edge_native+degree_fan+flow_balance+temporal_behavior|one_hot` | 1.0000 |
| `edge_native+degree_fan+flow_balance+temporal_behavior|ordinal` | `edge_native+degree_fan+flow_balance+temporal_behavior|one_hot` | 0.8885 |
