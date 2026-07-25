# Positive-complete txn-node resume to 20ep (`B_gcpal`)

**Not an exact GCPAL reproduction.**

- Resume method: `deterministic_replay_epochs_1_to_5`
- seed=2 cap=2048 steps/epoch=1587
- train_seconds=12666.8 opt_steps=31740 anchor_exposures=3927951
- checkpoints: `checkpoints/gcpal_txn_node_poscomplete_B_gcpal_20ep_seed2`
- val-selected checkpoint: epoch **5** (temporal HxX val AUPRC=0.158430)

## Preflight (from 5ep JSON)

```json
{
  "mode": "B_gcpal",
  "frac_anchors_knn_ge1": 1.0,
  "frac_anchors_knn_all_available": 1.0,
  "frac_anchors_struct_ge1": 0.47580645161290325,
  "mean_knn_pos_growth": 15.0,
  "mean_structural_pos_growth": 0.47580645161290325,
  "mean_identity_pos_growth": 1.0,
  "loss_mask_mean_knn_pos": 15.0,
  "loss_mask_mean_structural_pos": 0.47580644488334656,
  "loss_mask_mean_total_pos": 16.475805282592773,
  "rejected_not_allowed": 0.0,
  "realized_n_anchors": 124,
  "n_nodes": 2042
}
```

## Loss curve

- ep 1: loss=3.217575
- ep 2: loss=3.157150
- ep 3: loss=3.152609
- ep 4: loss=3.149379
- ep 5: loss=3.145820
- ep 6: loss=3.145102
- ep 7: loss=3.144730
- ep 8: loss=3.143209
- ep 9: loss=3.142802
- ep 10: loss=3.142896
- ep 11: loss=3.142207
- ep 12: loss=3.142357
- ep 13: loss=3.143831
- ep 14: loss=3.141499
- ep 15: loss=3.141702
- ep 16: loss=3.141717
- ep 17: loss=3.141847
- ep 18: loss=3.141097
- ep 19: loss=3.142078
- ep 20: loss=3.141507

## Learning curve evaluations

### Epoch 5 — temporal primary

- **X** val AUPRC=0.0243041570727699 | @0.5 auprc=0.0140 f1=0.0000 ppr=0.0 tp/fp/tn/fn=0/0/862289/1611
- **X** @val-thr=0.01: auprc=0.0140 f1=0.0070 p=0.0571 r=0.0037 ppr=0.00012154184512096307 tp/fp/tn/fn=6/99/862190/1605
- **H** val AUPRC=0.07978221880935554 | @0.5 auprc=0.0177 f1=0.0000 ppr=0.0 tp/fp/tn/fn=0/0/862289/1611
- **H** @val-thr=0.02: auprc=0.0177 f1=0.0612 p=0.1597 r=0.0379 ppr=0.00044218080796388473 tp/fp/tn/fn=61/321/861968/1550
- **HxX** val AUPRC=0.15842969877557828 | @0.5 auprc=0.0480 f1=0.0000 ppr=0.0 tp/fp/tn/fn=0/0/862289/1611
- **HxX** @val-thr=0.04: auprc=0.0480 f1=0.0459 p=0.2330 r=0.0255 ppr=0.0002037272832503762 tp/fp/tn/fn=41/135/862154/1570

### Epoch 5 — random-40 diagnostic

- **X** val AUPRC=0.03604077922637433 | @0.5 auprc=0.0350 f1=0.0000 ppr=0.0 tp/fp/tn/fn=0/0/3043901/3106
- **X** @val-thr=0.02: auprc=0.0350 f1=0.0677 p=0.0392 r=0.2505 ppr=0.006519840617366484 tp/fp/tn/fn=778/19088/3024813/2328
- **H** val AUPRC=0.13251860739275387 | @0.5 auprc=0.1259 f1=0.0089 ppr=8.204772749127258e-06 tp/fp/tn/fn=14/11/3043890/3092
- **H** @val-thr=0.21000000000000002: auprc=0.1259 f1=0.2026 p=0.5084 r=0.1265 ppr=0.0002536915734030148 tp/fp/tn/fn=393/380/3043521/2713
- **HxX** val AUPRC=0.16124353629706056 | @0.5 auprc=0.1573 f1=0.0096 ppr=1.083030002884798e-05 tp/fp/tn/fn=15/18/3043883/3091
- **HxX** @val-thr=0.18000000000000002: auprc=0.1573 f1=0.2786 p=0.3838 r=0.2186 ppr=0.0005805697197282448 tp/fp/tn/fn=679/1090/3042811/2427

### Epoch 10 — temporal primary

- **X** val AUPRC=0.0243041570727699 | @0.5 auprc=0.0140 f1=0.0000 ppr=0.0 tp/fp/tn/fn=0/0/862289/1611
- **X** @val-thr=0.01: auprc=0.0140 f1=0.0070 p=0.0571 r=0.0037 ppr=0.00012154184512096307 tp/fp/tn/fn=6/99/862190/1605
- **H** val AUPRC=0.05644448285728984 | @0.5 auprc=0.0254 f1=0.0000 ppr=4.630165528417641e-06 tp/fp/tn/fn=0/4/862285/1611
- **H** @val-thr=0.02: auprc=0.0254 f1=0.0911 p=0.1092 r=0.0782 ppr=0.0013358027549484894 tp/fp/tn/fn=126/1028/861261/1485
- **HxX** val AUPRC=0.11808533118212225 | @0.5 auprc=0.0388 f1=0.0000 ppr=0.0 tp/fp/tn/fn=0/0/862289/1611
- **HxX** @val-thr=0.05: auprc=0.0388 f1=0.0503 p=0.2110 r=0.0286 ppr=0.0002523440212987614 tp/fp/tn/fn=46/172/862117/1565

### Epoch 10 — random-40 diagnostic

- **X** val AUPRC=0.03604077922637433 | @0.5 auprc=0.0350 f1=0.0000 ppr=0.0 tp/fp/tn/fn=0/0/3043901/3106
- **X** @val-thr=0.02: auprc=0.0350 f1=0.0677 p=0.0392 r=0.2505 ppr=0.006519840617366484 tp/fp/tn/fn=778/19088/3024813/2328
- **H** val AUPRC=0.12157867055290839 | @0.5 auprc=0.1215 f1=0.0228 ppr=1.8050500048079968e-05 tp/fp/tn/fn=36/19/3043882/3070
- **H** @val-thr=0.15000000000000002: auprc=0.1215 f1=0.2083 p=0.1976 r=0.2202 ppr=0.0011361969302991427 tp/fp/tn/fn=684/2778/3041123/2422
- **HxX** val AUPRC=0.1292994624710438 | @0.5 auprc=0.1385 f1=0.0883 ppr=7.351476383218023e-05 tp/fp/tn/fn=147/77/3043824/2959
- **HxX** @val-thr=0.2: auprc=0.1385 f1=0.2045 p=0.2035 r=0.2054 ppr=0.0010288785027405582 tp/fp/tn/fn=638/2497/3041404/2468

### Epoch 15 — temporal primary

- **X** val AUPRC=0.0243041570727699 | @0.5 auprc=0.0140 f1=0.0000 ppr=0.0 tp/fp/tn/fn=0/0/862289/1611
- **X** @val-thr=0.01: auprc=0.0140 f1=0.0070 p=0.0571 r=0.0037 ppr=0.00012154184512096307 tp/fp/tn/fn=6/99/862190/1605
- **H** val AUPRC=0.06169221720165434 | @0.5 auprc=0.0240 f1=0.0000 ppr=0.0 tp/fp/tn/fn=0/0/862289/1611
- **H** @val-thr=0.02: auprc=0.0240 f1=0.0794 p=0.0769 r=0.0819 ppr=0.001986341011691168 tp/fp/tn/fn=132/1584/860705/1479
- **HxX** val AUPRC=0.11445241766064951 | @0.5 auprc=0.0302 f1=0.0000 ppr=0.0 tp/fp/tn/fn=0/0/862289/1611
- **HxX** @val-thr=0.03: auprc=0.0302 f1=0.0376 p=0.1709 r=0.0211 ppr=0.00023035073503877764 tp/fp/tn/fn=34/165/862124/1577

### Epoch 15 — random-40 diagnostic

- **X** val AUPRC=0.03604077922637433 | @0.5 auprc=0.0350 f1=0.0000 ppr=0.0 tp/fp/tn/fn=0/0/3043901/3106
- **X** @val-thr=0.02: auprc=0.0350 f1=0.0677 p=0.0392 r=0.2505 ppr=0.006519840617366484 tp/fp/tn/fn=778/19088/3024813/2328
- **H** val AUPRC=0.10033970445834806 | @0.5 auprc=0.1066 f1=0.0566 ppr=4.791587285490319e-05 tp/fp/tn/fn=92/54/3043847/3014
- **H** @val-thr=0.06999999999999999: auprc=0.1066 f1=0.1862 p=0.1593 r=0.2241 ppr=0.0014338660856374797 tp/fp/tn/fn=696/3673/3040228/2410
- **HxX** val AUPRC=0.1402897183054813 | @0.5 auprc=0.1505 f1=0.0776 ppr=6.268446380333226e-05 tp/fp/tn/fn=128/63/3043838/2978
- **HxX** @val-thr=0.23: auprc=0.1505 f1=0.2289 p=0.3313 r=0.1748 ppr=0.0005379049014327831 tp/fp/tn/fn=543/1096/3042805/2563

### Epoch 20 — temporal primary

- **X** val AUPRC=0.0243041570727699 | @0.5 auprc=0.0140 f1=0.0000 ppr=0.0 tp/fp/tn/fn=0/0/862289/1611
- **X** @val-thr=0.01: auprc=0.0140 f1=0.0070 p=0.0571 r=0.0037 ppr=0.00012154184512096307 tp/fp/tn/fn=6/99/862190/1605
- **H** val AUPRC=0.06266670159413268 | @0.5 auprc=0.0347 f1=0.0000 ppr=0.0 tp/fp/tn/fn=0/0/862289/1611
- **H** @val-thr=0.060000000000000005: auprc=0.0347 f1=0.0392 p=0.0792 r=0.0261 ppr=0.0006134969325153375 tp/fp/tn/fn=42/488/861801/1569
- **HxX** val AUPRC=0.10306781936377693 | @0.5 auprc=0.0716 f1=0.0000 ppr=0.0 tp/fp/tn/fn=0/0/862289/1611
- **HxX** @val-thr=0.09999999999999999: auprc=0.0716 f1=0.0755 p=0.1572 r=0.0497 ppr=0.0005891885634911448 tp/fp/tn/fn=80/429/861860/1531

### Epoch 20 — random-40 diagnostic

- **X** val AUPRC=0.03604077922637433 | @0.5 auprc=0.0350 f1=0.0000 ppr=0.0 tp/fp/tn/fn=0/0/3043901/3106
- **X** @val-thr=0.02: auprc=0.0350 f1=0.0677 p=0.0392 r=0.2505 ppr=0.006519840617366484 tp/fp/tn/fn=778/19088/3024813/2328
- **H** val AUPRC=0.09496646373790212 | @0.5 auprc=0.1069 f1=0.0473 ppr=3.5116427366264665e-05 tp/fp/tn/fn=76/31/3043870/3030
- **H** @val-thr=0.17: auprc=0.1069 f1=0.1922 p=0.2365 r=0.1619 ppr=0.0006980620654957471 tp/fp/tn/fn=503/1624/3042277/2603
- **HxX** val AUPRC=0.1232853624689261 | @0.5 auprc=0.1239 f1=0.0923 ppr=7.548390929197077e-05 tp/fp/tn/fn=154/76/3043825/2952
- **HxX** @val-thr=0.16: auprc=0.1239 f1=0.1974 p=0.1998 r=0.1951 ppr=0.000995403029924119 tp/fp/tn/fn=606/2427/3041474/2500

