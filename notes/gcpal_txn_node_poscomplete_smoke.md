# Positive-complete txn-node smoke (Small-HI)

**Not an exact GCPAL reproduction.**

- Realized anchors: 124
- Unique transactions: 2045 (cap 2048)
- frac KNN≥1: 1.0000
- frac all available KNN present: 1.0000
- frac structural≥1: 0.5000
- mean/median KNN pos: 15.00/15.00
- pos/neg mask density: 0.00806845910847187/0.9919314980506897
- Loss: 4.8205 (finite=True, grads=True)
- Step time: 0.56s; peak GPU MiB: 227.01220703125
- Matched steps/epoch (legacy opt count): 1587
- Projected 5ep (matched) ≈ 1.24h; full-coverage would be ≈ 20.4h
- Fits 6h with matched steps? **True**
- Gate pass? **True** → {"frac_anchors_knn_ge1_ge_0.95": true, "frac_anchors_knn_all_available_ge_0.90_or_cap": true, "finite_loss": true, "finite_gradients": true, "peak_gpu_safe": true, "no_leakage": true, "views_aligned": true, "n_nodes_le_cap": true, "pass": true}

Smoke OK to submit matched positive-complete 5-epoch scouts A/B.
