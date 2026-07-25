# Edge D+ neighbor-positive smoke

**Passed:** True

- loss finite: True (7.0805)
- grads finite: True
- anchors requested/realized: 126/126
- mean positives/anchor: 16.80158805847168
- structural/knn additions: 114.0/16379.0
- peak alloc/reserved MiB: 2131.6/2680.0
- steady step median: 0.422s (n=3)
- batches/epoch (D+-matched): 397; full-stream would be ~25786
- ~10ep hours @ matched budget: 0.47; ~40ep hours: 1.86
- 6h envelope OK for 10ep: True

NOT an exact GCPAL reproduction. Distinct from `--enable_knn_soft_positives`.

