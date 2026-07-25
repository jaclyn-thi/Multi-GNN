# Positive-complete 5ep diagnostics preflight (resume-to-20)

Source: completed scout JSONs (jobs 18567858 / 18567859). **No model/optimizer checkpoints were saved by the original 5-epoch scouts** — only metrics. Resume-to-20 therefore uses a **deterministic replay of epochs 1–5** (same seed, sampler, step seeds) with loss-history verification against these artifacts, then continues to epoch 20. Existing `*_5ep_seed2` notes/JSON are not overwritten.

## Batch construction coverage (batch 0 of epoch 1)

Shared growth (identical A/B batch membership):

| Metric | A_identity | B_gcpal |
|--------|-----------:|--------:|
| `frac_anchors_knn_ge1` | **1.0** | **1.0** |
| `frac_anchors_knn_all_available` (all ≤15 present) | **1.0** | **1.0** |
| `mean_knn_pos` (growth) | **15.0** | **15.0** |
| `frac_anchors_struct_ge1` | **0.4758** | **0.4758** |
| `mean_structural_pos` (growth) | **0.4758** | **0.4758** |
| realized anchors / nodes | 124 / 2042 | 124 / 2042 |

Loss-mask positives (objective differs):

| Metric | A (identity only) | B (I∪struct∪KNN) |
|--------|------------------:|-----------------:|
| `mean_identity` | 1.0 | 1.0 |
| `mean_knn_pos` | **0.0** | **15.0** |
| `mean_structural_pos` | **0.0** | **0.476** |
| `frac_with_knn` | 0.0 | **1.0** |
| `frac_with_structural` | 0.0 | **0.476** |

## Split leakage

Train-local ids `0..n_train-1` only; sampler raises on out-of-set neighbors. `rejected_not_allowed` not nonzero in recorded growth stats. Val/test CSV rows never enter train minibatches under this id space.

## Matched training budget

- seed=2, cap=2048, steps/epoch=1587, 5 epochs → 7935 opt steps, 981984 anchor exposures
- A loss@ep5 ≈ 6.0809; B loss@ep5 ≈ 3.1463

## Smoke gate (reference)

[`notes/gcpal_txn_node_poscomplete_smoke.md`](gcpal_txn_node_poscomplete_smoke.md): same coverage pattern (124 anchors, 2045 nodes, KNN 100%).

**Preflight verdict: PASS** — coverage diagnostics recovered from JSON; proceed with verified replay → resume to 20.
