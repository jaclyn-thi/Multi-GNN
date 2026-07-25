# GCPAL-style txn-node one-batch smoke (Small-HI)

**Not an exact GCPAL reproduction.**

- Anchors: 2048
- Sampled nodes: 4096
- Edges view1/view2/knn: 122/119/7959
- Positives/anchor: {'mean_identity': 1.0, 'mean_structural_pos': 0.0, 'mean_knn_pos': 0.01318359375, 'mean_total_pos': 1.01318359375, 'frac_with_structural': 0.0, 'frac_with_knn': 0.01318359375, 'n_identity_diag': 2048.0}
- Unique negatives (mean): 2046.99
- Loss total / rr / r-knn: 7.6207 / 7.6119 / 7.6244
- Finite grads: True
- Step runtime: 1.48s
- Peak GPU MiB: 251.0517578125
- Projected epoch: 2343.8s; 5ep ≈ 3.26h
- Fits 6h Advanced GPU for 5ep? **True**

## Flow graph assumption

```json
{
  "n_nodes": 3248921,
  "n_edges": 1614187,
  "policy": "immediate_next",
  "mean_out_degree": 0.49683787325084233,
  "max_out_degree": 1,
  "fraction_nodes_with_out_edge": 0.49683787325084233,
  "note": "Implementation assumption: directed receiver\u2192next-sender flow; not all pairwise same-account connections."
}
```

## KNN deviations

```json
[
  "KNN cache feature_set includes degree_fan (train-graph degrees) beyond raw AML columns."
]
```

Smoke OK to proceed to Stage 7 five-epoch matched scouts.