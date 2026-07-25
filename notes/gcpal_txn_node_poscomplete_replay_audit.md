# Positive-complete replay audit: original 5ep vs replay epoch 5

Status: **diagnostic** · Date: 2026-07-23 · Protocol family: txn-node GCPAL-inspired positive-complete  
Designation: **diagnostic / provenance** — **NOT an exact GCPAL reproduction** · **not an exact weight resume**

**Classification update (2026-07-23):** Original 5ep metrics are **noncanonical / not table-eligible** (“online augmented-view embeddings accumulated across changing training states, with clean induced fill/evaluation”). Replay ep5–20 runs are **deterministic replay-extension** results under **legacy chunked induce (4096)** — internally comparable A/B diagnostic only. **B beats A under a shared frozen-checkpoint legacy-chunked extraction; canonical graph-preserving re-extraction is pending.** Prefer **`loss_trajectory_within_tolerance`** over `replay_verified`. See [`gcpal_txn_node_extraction_scope_audit.md`](gcpal_txn_node_extraction_scope_audit.md) and [`gcpal_txn_node_canonical_reextraction.md`](gcpal_txn_node_canonical_reextraction.md).

Full-precision companion: [`results/diagnostics/gcpal_txn_node_poscomplete_replay_audit.json`](../results/diagnostics/gcpal_txn_node_poscomplete_replay_audit.json)

## Artifacts compared

| Role | Notes | JSON | Checkpoint |
|------|-------|------|------------|
| Original A | [A 5ep note](gcpal_txn_node_poscomplete_scout_A_identity_5ep_seed2.md) | [A 5ep JSON](../results/diagnostics/gcpal_txn_node_poscomplete_scout_A_identity_5ep_seed2.json) | *(none saved)* |
| Original B | [B 5ep note](gcpal_txn_node_poscomplete_scout_B_gcpal_5ep_seed2.md) | [B 5ep JSON](../results/diagnostics/gcpal_txn_node_poscomplete_scout_B_gcpal_5ep_seed2.json) | *(none saved)* |
| Replay A (job **18576502**) | [A 20ep note](gcpal_txn_node_poscomplete_scout_A_identity_20ep_seed2.md) | [A 20ep JSON](../results/diagnostics/gcpal_txn_node_poscomplete_scout_A_identity_20ep_seed2.json) | `checkpoints/.../A_identity_20ep_seed2/epoch_05.pt` |
| Replay B (job **18576503**) | [B 20ep note](gcpal_txn_node_poscomplete_scout_B_gcpal_20ep_seed2.md) | [B 20ep JSON](../results/diagnostics/gcpal_txn_node_poscomplete_scout_B_gcpal_20ep_seed2.json) | `checkpoints/.../B_gcpal_20ep_seed2/epoch_05.pt` |

## 1. Replay verification (full-precision JSON losses)

Tolerance in resume script: `atol=0.02`, `rtol=0.01`  
(`within` iff `abs_diff ≤ max(atol, rtol·|original|)`).  
`replay_verified=True` for both arms means **tolerance pass only**, not bitwise equality.

### A_identity

- Original 5ep opt steps / anchor exposures: **7935** / **981984**
- Replay steps/epoch: **1587** ⇒ implied through ep5: **7935**

| Epoch | Original loss | Replay loss | Abs diff | Rel diff | Tolerance | Within tol? | Steps (o/r) |
|------:|--------------:|------------:|---------:|---------:|----------:|:-----------:|------------:|
| 1 | 6.156349609389 | 6.157540803896 | 0.001191194507 | 1.934903932100e-04 | 0.061563496094 | yes | 1587/1587 |
| 2 | 6.096243380899 | 6.094811975242 | 0.001431405657 | 2.348012648970e-04 | 0.060962433809 | yes | 1587/1587 |
| 3 | 6.091117451457 | 6.089979031136 | 0.001138420321 | 1.868984353673e-04 | 0.060911174515 | yes | 1587/1587 |
| 4 | 6.084995664586 | 6.083571778654 | 0.001423885932 | 2.339994981842e-04 | 0.060849956646 | yes | 1587/1587 |
| 5 | 6.080860237994 | 6.078867439962 | 0.001992798032 | 3.277164667245e-04 | 0.060808602380 | yes | 1587/1587 |

### B_gcpal

- Original 5ep opt steps / anchor exposures: **7935** / **981984**
- Replay steps/epoch: **1587** ⇒ implied through ep5: **7935**

| Epoch | Original loss | Replay loss | Abs diff | Rel diff | Tolerance | Within tol? | Steps (o/r) |
|------:|--------------:|------------:|---------:|---------:|----------:|:-----------:|------------:|
| 1 | 3.217799723336 | 3.217574673775 | 0.000225049561 | 6.993895855815e-05 | 0.032177997233 | yes | 1587/1587 |
| 2 | 3.157346395704 | 3.157150127036 | 0.000196268668 | 6.216253890882e-05 | 0.031573463957 | yes | 1587/1587 |
| 3 | 3.153302628061 | 3.152609120094 | 0.000693507967 | 2.199306722232e-04 | 0.031533026281 | yes | 1587/1587 |
| 4 | 3.150085994712 | 3.149379494811 | 0.000706499901 | 2.242795601633e-04 | 0.031500859947 | yes | 1587/1587 |
| 5 | 3.146345596157 | 3.145819732057 | 0.000525864101 | 1.671348821959e-04 | 0.031463455962 | yes | 1587/1587 |

### Hashes / seeds recorded?

| Item | Status |
|------|--------|
| Resolved configuration hashes | **Not recorded** in artifacts |
| Data-order / permutation hashes | **Not recorded** |
| Init / model-state hashes | **Not recorded** |
| Sampler stream RNG | `RandomState(seed=2)` (source) |
| Per-step aug seeds | `seed*10007 + epoch_0based*97 + step_i` (scout) ≡ `seed*10007 + (epoch_1based-1)*97 + step_i` (resume) |
| Downstream MLP / random-40 | `seed=2`; SSS `seed` / `seed+1` |

First-step `growth_stats` / `positive_stats` match between original and replay for both arms (see JSON).

## 2. Configuration comparison

### Matched / equivalent

- dataset Small-HI via data_config.json / AMLWorld temporal IDs
- flow adjacency policy immediate_next; identical flow_stats n_nodes/n_edges
- KNN cache path + degree_fan deviation note identical
- positive-complete batching; max_total_nodes=2048; realized_n_anchors=124 on first step
- A: identity-only positives; B: identity+structural+KNN (λ=0.3, τ=0.5, k=15)
- optimizer Adam lr=1e-3; encoder SharedTxnNodeEncoder emb_dim=128; no scheduler
- steps_per_epoch=1587; 5-epoch opt steps=7935; anchor exposures 981984 (original)
- downstream PaperStyleMLP hidden=128 dropout=0.1; 15 epochs; batch 8192; seed=2
- threshold grid linspace(0.01,0.99,99); StratifiedShuffleSplit random40 seeds seed / seed+1
- first-step growth_stats and positive_stats bitwise-equal between original and replay

### Changed or not exact

| Field | Status | Detail |
|-------|--------|--------|
| `training_mean_loss_epochs_1_to_5` | within_loose_tolerance_not_bitwise | abs diffs ~1e-4 to 2e-3; replay_verified uses atol=0.02 rtol=0.01 |
| `embedding_extraction_for_downstream` | CHANGED | Last training epoch: accumulate h_anchors from augmented view1 (h1) during posit… |
| `eval_reporting_schema` | extended_in_replay | Replay adds val_ranking and val_at_selected_threshold; original 5ep JSON lacked these. |
| `checkpoints` | original_had_none | Original 5ep scouts did not save checkpoints; replay reconstructs epoch_05.pt. |

**Primary scientific change:** original 5ep downstream `H` used **last-epoch training-step embeddings from augmented view1** (`h_anchors` from `h1`), with induced fill for unseen train nodes; replay epoch-5 eval uses **clean full-graph `encode_nodes_induced` from `epoch_05.pt` for all splits** (`encode_note` in learning curve).

## 3. Metric discrepancy — original epoch 5 vs replay epoch 5

**X is the evaluator stability control.** Core shared metrics (`auroc`, `auprc`, `f1`, `precision`, `recall`, `threshold`) match exactly for X on temporal and random-40 for both arms (`all_X_core_metrics_equal=true`). Original JSON omitted confusion/PPR fields; those replay-only keys are not part of the control.

### A_identity — temporal (primary)

#### X

| Metric | Orig @0.5 | Replay@0.5 | Δ | Orig @val-thr | Replay @val-thr | Δ |
|--------|----------:|-----------:|--:|-------------:|----------------:|--:|
| auroc | 0.87400594 | 0.87400594 | +0 | 0.87400594 | 0.87400594 | +0 |
| auprc | 0.01395157 | 0.01395157 | +0 | 0.01395157 | 0.01395157 | +0 |
| f1 | 0 | 0 | +0 | 0.00699301 | 0.00699301 | +0 |
| precision | 0 | 0 | +0 | 0.05714286 | 0.05714286 | +0 |
| recall | 0 | 0 | +0 | 0.00372439 | 0.00372439 | +0 |
| threshold | 0.5 | 0.5 | +0 | 0.01 | 0.01 | +0 |
| positive_prediction_rate | — | 0 | — | — | 0.00012154 | — |
| tp | — | 0 | — | — | 6 | — |
| fp | — | 0 | — | — | 99 | — |
| tn | — | 862289 | — | — | 862190 | — |
| fn | — | 1611 | — | — | 1605 | — |

Replay-only val ranking: AUROC=0.90414667 AUPRC=0.02430416 (n=965524).

#### H

| Metric | Orig @0.5 | Replay@0.5 | Δ | Orig @val-thr | Replay @val-thr | Δ |
|--------|----------:|-----------:|--:|-------------:|----------------:|--:|
| auroc | 0.64561334 | 0.44378722 | -0.201826 | 0.64561334 | 0.44378722 | -0.201826 |
| auprc | 0.00506028 | 0.00198942 | -0.00307086 | 0.00506028 | 0.00198942 | -0.00307086 |
| f1 | 0 | 0 | +0 | 0 | 0.00628931 | +0.00628931 |
| precision | 0 | 0 | +0 | 0 | 0.0051526 | +0.0051526 |
| recall | 0 | 0 | +0 | 0 | 0.00806952 | +0.00806952 |
| threshold | 0.5 | 0.5 | +0 | 0.06 | 0.05 | -0.01 |
| positive_prediction_rate | — | 0 | — | — | 0.00292048 | — |
| tp | — | 0 | — | — | 13 | — |
| fp | — | 0 | — | — | 2510 | — |
| tn | — | 862289 | — | — | 859779 | — |
| fn | — | 1611 | — | — | 1598 | — |

Replay-only val ranking: AUROC=0.81047859 AUPRC=0.01433623 (n=965524).

#### HxX

| Metric | Orig @0.5 | Replay@0.5 | Δ | Orig @val-thr | Replay @val-thr | Δ |
|--------|----------:|-----------:|--:|-------------:|----------------:|--:|
| auroc | 0.8884183 | 0.80838245 | -0.0800359 | 0.8884183 | 0.80838245 | -0.0800359 |
| auprc | 0.07930008 | 0.01249186 | -0.0668082 | 0.07930008 | 0.01249186 | -0.0668082 |
| f1 | 0 | 0 | +0 | 0.03653506 | 0.00847971 | -0.0280554 |
| precision | 0 | 0 | +0 | 0.36046512 | 0.175 | -0.185465 |
| recall | 0 | 0 | +0 | 0.01924271 | 0.00434513 | -0.0148976 |
| threshold | 0.5 | 0.5 | +0 | 0.07 | 0.09 | +0.02 |
| positive_prediction_rate | — | 0 | — | — | 0.0000463 | — |
| tp | — | 0 | — | — | 7 | — |
| fp | — | 0 | — | — | 33 | — |
| tn | — | 862289 | — | — | 862256 | — |
| fn | — | 1611 | — | — | 1604 | — |

Replay-only val ranking: AUROC=0.90512294 AUPRC=0.08822348 (n=965524).

### A_identity — random-40 (diagnostic)

#### X

| Metric | Orig @0.5 | Replay@0.5 | Δ | Orig @val-thr | Replay @val-thr | Δ |
|--------|----------:|-----------:|--:|-------------:|----------------:|--:|
| auroc | 0.90920091 | 0.90920091 | +0 | 0.90920091 | 0.90920091 | +0 |
| auprc | 0.03499984 | 0.03499984 | +0 | 0.03499984 | 0.03499984 | +0 |
| f1 | 0 | 0 | +0 | 0.06773463 | 0.06773463 | +0 |
| precision | 0 | 0 | +0 | 0.03916239 | 0.03916239 | +0 |
| recall | 0 | 0 | +0 | 0.25048294 | 0.25048294 | +0 |
| threshold | 0.5 | 0.5 | +0 | 0.02 | 0.02 | +0 |
| positive_prediction_rate | — | 0 | — | — | 0.00651984 | — |
| tp | — | 0 | — | — | 778 | — |
| fp | — | 0 | — | — | 19088 | — |
| tn | — | 3043901 | — | — | 3024813 | — |
| fn | — | 3106 | — | — | 2328 | — |

Replay-only val ranking: AUROC=0.90928285 AUPRC=0.03604078 (n=507835).

#### H

| Metric | Orig @0.5 | Replay@0.5 | Δ | Orig @val-thr | Replay @val-thr | Δ |
|--------|----------:|-----------:|--:|-------------:|----------------:|--:|
| auroc | 0.85067601 | 0.76991188 | -0.0807641 | 0.85067601 | 0.76991188 | -0.0807641 |
| auprc | 0.06435386 | 0.04289554 | -0.0214583 | 0.06435386 | 0.04289554 | -0.0214583 |
| f1 | 0 | 0.00064309 | +0.000643087 | 0.14617536 | 0.11091142 | -0.0352639 |
| precision | 0 | 0.25 | +0.25 | 0.11745163 | 0.27376426 | +0.156313 |
| recall | 0 | 0.00032196 | +0.000321958 | 0.19349646 | 0.06954282 | -0.123954 |
| threshold | 0.5 | 0.5 | +0 | 0.02 | 0.03 | +0.01 |
| positive_prediction_rate | — | 0.00000131 | — | — | 0.00025894 | — |
| tp | — | 1 | — | — | 216 | — |
| fp | — | 3 | — | — | 573 | — |
| tn | — | 3043898 | — | — | 3043328 | — |
| fn | — | 3105 | — | — | 2890 | — |

Replay-only val ranking: AUROC=0.78180891 AUPRC=0.04679951 (n=507835).

#### HxX

| Metric | Orig @0.5 | Replay@0.5 | Δ | Orig @val-thr | Replay @val-thr | Δ |
|--------|----------:|-----------:|--:|-------------:|----------------:|--:|
| auroc | 0.90907241 | 0.9126386 | +0.00356618 | 0.90907241 | 0.9126386 | +0.00356618 |
| auprc | 0.14138593 | 0.17781426 | +0.0364283 | 0.14138593 | 0.17781426 | +0.0364283 |
| f1 | 0.04815605 | 0.00889454 | -0.0392615 | 0.27758998 | 0.29012686 | +0.0125369 |
| precision | 0.45142857 | 0.33333333 | -0.118095 | 0.36297452 | 0.33819117 | -0.0247833 |
| recall | 0.02543464 | 0.00450741 | -0.0209272 | 0.22472634 | 0.25402447 | +0.0292981 |
| threshold | 0.5 | 0.5 | +0 | 0.23 | 0.08 | -0.15 |
| positive_prediction_rate | — | 0.00001378 | — | — | 0.00076567 | — |
| tp | — | 14 | — | — | 789 | — |
| fp | — | 28 | — | — | 1544 | — |
| tn | — | 3043873 | — | — | 3042357 | — |
| fn | — | 3092 | — | — | 2317 | — |

Replay-only val ranking: AUROC=0.913597 AUPRC=0.17314109 (n=507835).

### B_gcpal — temporal (primary)

#### X

| Metric | Orig @0.5 | Replay@0.5 | Δ | Orig @val-thr | Replay @val-thr | Δ |
|--------|----------:|-----------:|--:|-------------:|----------------:|--:|
| auroc | 0.87400594 | 0.87400594 | +0 | 0.87400594 | 0.87400594 | +0 |
| auprc | 0.01395157 | 0.01395157 | +0 | 0.01395157 | 0.01395157 | +0 |
| f1 | 0 | 0 | +0 | 0.00699301 | 0.00699301 | +0 |
| precision | 0 | 0 | +0 | 0.05714286 | 0.05714286 | +0 |
| recall | 0 | 0 | +0 | 0.00372439 | 0.00372439 | +0 |
| threshold | 0.5 | 0.5 | +0 | 0.01 | 0.01 | +0 |
| positive_prediction_rate | — | 0 | — | — | 0.00012154 | — |
| tp | — | 0 | — | — | 6 | — |
| fp | — | 0 | — | — | 99 | — |
| tn | — | 862289 | — | — | 862190 | — |
| fn | — | 1611 | — | — | 1605 | — |

Replay-only val ranking: AUROC=0.90414667 AUPRC=0.02430416 (n=965524).

#### H

| Metric | Orig @0.5 | Replay@0.5 | Δ | Orig @val-thr | Replay @val-thr | Δ |
|--------|----------:|-----------:|--:|-------------:|----------------:|--:|
| auroc | 0.82213208 | 0.7494007 | -0.0727314 | 0.82213208 | 0.7494007 | -0.0727314 |
| auprc | 0.06683665 | 0.01768171 | -0.0491549 | 0.06683665 | 0.01768171 | -0.0491549 |
| f1 | 0.01686747 | 0 | -0.0168675 | 0.12531969 | 0.06121425 | -0.0641054 |
| precision | 0.28571429 | 0 | -0.285714 | 0.2 | 0.15968586 | -0.0403141 |
| recall | 0.00869025 | 0 | -0.00869025 | 0.09124767 | 0.03786468 | -0.053383 |
| threshold | 0.5 | 0.5 | +0 | 0.05 | 0.02 | -0.03 |
| positive_prediction_rate | — | 0 | — | — | 0.00044218 | — |
| tp | — | 0 | — | — | 61 | — |
| fp | — | 0 | — | — | 321 | — |
| tn | — | 862289 | — | — | 861968 | — |
| fn | — | 1611 | — | — | 1550 | — |

Replay-only val ranking: AUROC=0.8921172 AUPRC=0.07978222 (n=965524).

#### HxX

| Metric | Orig @0.5 | Replay@0.5 | Δ | Orig @val-thr | Replay @val-thr | Δ |
|--------|----------:|-----------:|--:|-------------:|----------------:|--:|
| auroc | 0.88879089 | 0.8154947 | -0.0732962 | 0.88879089 | 0.8154947 | -0.0732962 |
| auprc | 0.1011236 | 0.04804974 | -0.0530739 | 0.1011236 | 0.04804974 | -0.0530739 |
| f1 | 0 | 0 | +0 | 0.09937238 | 0.04588696 | -0.0534854 |
| precision | 0 | 0 | +0 | 0.31561462 | 0.23295455 | -0.0826601 |
| recall | 0 | 0 | +0 | 0.05896958 | 0.02545003 | -0.0335196 |
| threshold | 0.5 | 0.5 | +0 | 0.07 | 0.04 | -0.03 |
| positive_prediction_rate | — | 0 | — | — | 0.00020373 | — |
| tp | — | 0 | — | — | 41 | — |
| fp | — | 0 | — | — | 135 | — |
| tn | — | 862289 | — | — | 862154 | — |
| fn | — | 1611 | — | — | 1570 | — |

Replay-only val ranking: AUROC=0.91069504 AUPRC=0.1584297 (n=965524).

### B_gcpal — random-40 (diagnostic)

#### X

| Metric | Orig @0.5 | Replay@0.5 | Δ | Orig @val-thr | Replay @val-thr | Δ |
|--------|----------:|-----------:|--:|-------------:|----------------:|--:|
| auroc | 0.90920091 | 0.90920091 | +0 | 0.90920091 | 0.90920091 | +0 |
| auprc | 0.03499984 | 0.03499984 | +0 | 0.03499984 | 0.03499984 | +0 |
| f1 | 0 | 0 | +0 | 0.06773463 | 0.06773463 | +0 |
| precision | 0 | 0 | +0 | 0.03916239 | 0.03916239 | +0 |
| recall | 0 | 0 | +0 | 0.25048294 | 0.25048294 | +0 |
| threshold | 0.5 | 0.5 | +0 | 0.02 | 0.02 | +0 |
| positive_prediction_rate | — | 0 | — | — | 0.00651984 | — |
| tp | — | 0 | — | — | 778 | — |
| fp | — | 0 | — | — | 19088 | — |
| tn | — | 3043901 | — | — | 3024813 | — |
| fn | — | 3106 | — | — | 2328 | — |

Replay-only val ranking: AUROC=0.90928285 AUPRC=0.03604078 (n=507835).

#### H

| Metric | Orig @0.5 | Replay@0.5 | Δ | Orig @val-thr | Replay @val-thr | Δ |
|--------|----------:|-----------:|--:|-------------:|----------------:|--:|
| auroc | 0.90047219 | 0.90110006 | +0.000627861 | 0.90047219 | 0.90110006 | +0.000627861 |
| auprc | 0.16310425 | 0.12591451 | -0.0371897 | 0.16310425 | 0.12591451 | -0.0371897 |
| f1 | 0.01392846 | 0.00894283 | -0.00498563 | 0.26523778 | 0.20262954 | -0.0626082 |
| precision | 0.41509434 | 0.56 | +0.144906 | 0.43262928 | 0.5084088 | +0.0757795 |
| recall | 0.00708307 | 0.00450741 | -0.00257566 | 0.19124276 | 0.1265293 | -0.0647135 |
| threshold | 0.5 | 0.5 | +0 | 0.15 | 0.21 | +0.06 |
| positive_prediction_rate | — | 0.0000082 | — | — | 0.00025369 | — |
| tp | — | 14 | — | — | 393 | — |
| fp | — | 11 | — | — | 380 | — |
| tn | — | 3043890 | — | — | 3043521 | — |
| fn | — | 3092 | — | — | 2713 | — |

Replay-only val ranking: AUROC=0.89686364 AUPRC=0.13251861 (n=507835).

#### HxX

| Metric | Orig @0.5 | Replay@0.5 | Δ | Orig @val-thr | Replay @val-thr | Δ |
|--------|----------:|-----------:|--:|-------------:|----------------:|--:|
| auroc | 0.91249815 | 0.91318924 | +0.000691085 | 0.91249815 | 0.91318924 | +0.000691085 |
| auprc | 0.21373887 | 0.15726714 | -0.0564717 | 0.21373887 | 0.15726714 | -0.0564717 |
| f1 | 0.05666769 | 0.00955718 | -0.0471105 | 0.33905985 | 0.2785641 | -0.0604957 |
| precision | 0.65248227 | 0.45454545 | -0.197937 | 0.40985851 | 0.38383267 | -0.0260258 |
| recall | 0.02962009 | 0.00482936 | -0.0247907 | 0.28911784 | 0.21860914 | -0.0705087 |
| threshold | 0.5 | 0.5 | +0 | 0.08 | 0.18 | +0.1 |
| positive_prediction_rate | — | 0.00001083 | — | — | 0.00058057 | — |
| tp | — | 15 | — | — | 679 | — |
| fp | — | 18 | — | — | 1090 | — |
| tn | — | 3043883 | — | — | 3042811 | — |
| fn | — | 3091 | — | — | 2427 | — |

Replay-only val ranking: AUROC=0.9118053 AUPRC=0.16124354 (n=507835).

## 4. Root-cause classification

**Exact resume claim: false.**

| Code | Label | Role here |
|------|-------|-----------|
| **C** | Embedding extraction changed | **Primary** — explains H/HxX gaps with X fixed |
| **A** | Training replay not exact | Secondary — full-precision loss drift within loose tol |
| **B** | Training nondeterministic despite matched seeds | Secondary — residual float/CUDA drift |
| **D** | Downstream eval changed / nondeterministic | **Ruled out as primary** (X core metrics identical) |
| **E** | Unresolved | Missing hashes; no original weight snapshot to compare to `epoch_05.pt` |

Evidence summaries are in the JSON under `root_cause_classification.evidence`.

## 5. Correct learning-curve interpretation (replay suite)

Distinguish three statements:

1. **Validation-selected checkpoint** (selection rule): temporal **HxX validation AUPRC** → **epoch 5 for both A and B**.
2. **Fixed epoch-20 comparison** (not used for selection): under the *replay* eval protocol, **B temporal test H and HxX AUPRC improve from replay ep5 → ep20**:
   - H test AUPRC: **{B['temporal_test_H_auprc_ep5']:.6f} → {B['temporal_test_H_auprc_ep20']:.6f}**
   - HxX test AUPRC: **{B['temporal_test_HxX_auprc_ep5']:.6f} → {B['temporal_test_HxX_auprc_ep20']:.6f}**
3. **Test metrics must not drive selection.** Val HxX AUPRC still prefers ep5 for B (0.158 → 0.103 by ep20).

### Correction

Do **not** use the unqualified claim “longer training does not help.”  
More precise:

- Longer training **does not improve the declared validation selection metric** (both arms select ep5).
- For **B**, longer training **can improve temporal test H / HxX AUPRC** at fixed epoch 20 vs replay epoch 5; that is a separate, non-selection comparison and may be unstable relative to val.

A still degrades badly on temporal H/HxX with longer training under the replay protocol.

## Bottom line

The original 5ep scout metrics and the replay `epoch_05` evaluations are **not interchangeable**. Treat original 5ep notes as a distinct extraction protocol; treat the 20ep learning curve as an internally consistent **legacy-chunked** checkpoint series. Prefer **val-selected ep5** for protocol-faithful claims within that series; report ep20 test gains for B only as a fixed-horizon diagnostic. **Do not** promote either artifact family to thesis tables until canonical full-split frozen extraction is run from the saved checkpoints.
