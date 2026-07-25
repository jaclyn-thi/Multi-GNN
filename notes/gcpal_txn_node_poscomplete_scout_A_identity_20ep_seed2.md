# Positive-complete txn-node resume to 20ep (`A_identity`)

**Not an exact GCPAL reproduction.**

- Resume method: `deterministic_replay_epochs_1_to_5`
- seed=2 cap=2048 steps/epoch=1587
- train_seconds=12679.2 opt_steps=31740 anchor_exposures=3927951
- checkpoints: `checkpoints/gcpal_txn_node_poscomplete_A_identity_20ep_seed2`
- val-selected checkpoint: epoch **5** (temporal HxX val AUPRC=0.088223)

## Preflight (from 5ep JSON)

```json
{
  "mode": "A_identity",
  "frac_anchors_knn_ge1": 1.0,
  "frac_anchors_knn_all_available": 1.0,
  "frac_anchors_struct_ge1": 0.47580645161290325,
  "mean_knn_pos_growth": 15.0,
  "mean_structural_pos_growth": 0.47580645161290325,
  "mean_identity_pos_growth": 1.0,
  "loss_mask_mean_knn_pos": 0.0,
  "loss_mask_mean_structural_pos": 0.0,
  "loss_mask_mean_total_pos": 1.0,
  "rejected_not_allowed": 0.0,
  "realized_n_anchors": 124,
  "n_nodes": 2042
}
```

## Loss curve

- ep 1: loss=6.157541
- ep 2: loss=6.094812
- ep 3: loss=6.089979
- ep 4: loss=6.083572
- ep 5: loss=6.078867
- ep 6: loss=6.077874
- ep 7: loss=6.074258
- ep 8: loss=6.069942
- ep 9: loss=6.069454
- ep 10: loss=6.071806
- ep 11: loss=6.068192
- ep 12: loss=6.067725
- ep 13: loss=6.069951
- ep 14: loss=6.066825
- ep 15: loss=6.064919
- ep 16: loss=6.063637
- ep 17: loss=6.064124
- ep 18: loss=6.064409
- ep 19: loss=6.065646
- ep 20: loss=6.063548

## Learning curve evaluations

### Epoch 5 — temporal primary

- **X** val AUPRC=0.0243041570727699 | @0.5 auprc=0.0140 f1=0.0000 ppr=0.0 tp/fp/tn/fn=0/0/862289/1611
- **X** @val-thr=0.01: auprc=0.0140 f1=0.0070 p=0.0571 r=0.0037 ppr=0.00012154184512096307 tp/fp/tn/fn=6/99/862190/1605
- **H** val AUPRC=0.014336226551215248 | @0.5 auprc=0.0020 f1=0.0000 ppr=0.0 tp/fp/tn/fn=0/0/862289/1611
- **H** @val-thr=0.05: auprc=0.0020 f1=0.0063 p=0.0052 r=0.0081 ppr=0.002920476907049427 tp/fp/tn/fn=13/2510/859779/1598
- **HxX** val AUPRC=0.0882234782149462 | @0.5 auprc=0.0125 f1=0.0000 ppr=0.0 tp/fp/tn/fn=0/0/862289/1611
- **HxX** @val-thr=0.09: auprc=0.0125 f1=0.0085 p=0.1750 r=0.0043 ppr=4.630165528417641e-05 tp/fp/tn/fn=7/33/862256/1604

### Epoch 5 — random-40 diagnostic

- **X** val AUPRC=0.03604077922637433 | @0.5 auprc=0.0350 f1=0.0000 ppr=0.0 tp/fp/tn/fn=0/0/3043901/3106
- **X** @val-thr=0.02: auprc=0.0350 f1=0.0677 p=0.0392 r=0.2505 ppr=0.006519840617366484 tp/fp/tn/fn=778/19088/3024813/2328
- **H** val AUPRC=0.0467995126209875 | @0.5 auprc=0.0429 f1=0.0006 ppr=1.3127636398603613e-06 tp/fp/tn/fn=1/3/3043898/3105
- **H** @val-thr=0.03: auprc=0.0429 f1=0.1109 p=0.2738 r=0.0695 ppr=0.00025894262796245625 tp/fp/tn/fn=216/573/3043328/2890
- **HxX** val AUPRC=0.1731410936771138 | @0.5 auprc=0.1778 f1=0.0089 ppr=1.3784018218533794e-05 tp/fp/tn/fn=14/28/3043873/3092
- **HxX** @val-thr=0.08: auprc=0.1778 f1=0.2901 p=0.3382 r=0.2540 ppr=0.0007656693929485557 tp/fp/tn/fn=789/1544/3042357/2317

### Epoch 10 — temporal primary

- **X** val AUPRC=0.0243041570727699 | @0.5 auprc=0.0140 f1=0.0000 ppr=0.0 tp/fp/tn/fn=0/0/862289/1611
- **X** @val-thr=0.01: auprc=0.0140 f1=0.0070 p=0.0571 r=0.0037 ppr=0.00012154184512096307 tp/fp/tn/fn=6/99/862190/1605
- **H** val AUPRC=0.008120651821936439 | @0.5 auprc=0.0017 f1=0.0000 ppr=0.0 tp/fp/tn/fn=0/0/862289/1611
- **H** @val-thr=0.06999999999999999: auprc=0.0017 f1=0.0000 p=0.0000 r=0.0000 ppr=0.0003495774973955319 tp/fp/tn/fn=0/302/861987/1611
- **HxX** val AUPRC=0.07193011739281069 | @0.5 auprc=0.0114 f1=0.0000 ppr=2.3150827642088205e-06 tp/fp/tn/fn=0/2/862287/1611
- **HxX** @val-thr=0.03: auprc=0.0114 f1=0.0354 p=0.1090 r=0.0211 ppr=0.000361152911216576 tp/fp/tn/fn=34/278/862011/1577

### Epoch 10 — random-40 diagnostic

- **X** val AUPRC=0.03604077922637433 | @0.5 auprc=0.0350 f1=0.0000 ppr=0.0 tp/fp/tn/fn=0/0/3043901/3106
- **X** @val-thr=0.02: auprc=0.0350 f1=0.0677 p=0.0392 r=0.2505 ppr=0.006519840617366484 tp/fp/tn/fn=778/19088/3024813/2328
- **H** val AUPRC=0.007830184989254505 | @0.5 auprc=0.0084 f1=0.0013 ppr=6.563818199301807e-07 tp/fp/tn/fn=2/0/3043901/3104
- **H** @val-thr=0.03: auprc=0.0084 f1=0.0258 p=0.0220 r=0.0312 ppr=0.001445352767486258 tp/fp/tn/fn=97/4307/3039594/3009
- **HxX** val AUPRC=0.17901252449856767 | @0.5 auprc=0.1885 f1=0.0314 ppr=2.6583463707172316e-05 tp/fp/tn/fn=50/31/3043870/3056
- **HxX** @val-thr=0.08: auprc=0.1885 f1=0.3009 p=0.4511 r=0.2257 ppr=0.0005100086740857504 tp/fp/tn/fn=701/853/3043048/2405

### Epoch 15 — temporal primary

- **X** val AUPRC=0.0243041570727699 | @0.5 auprc=0.0140 f1=0.0000 ppr=0.0 tp/fp/tn/fn=0/0/862289/1611
- **X** @val-thr=0.01: auprc=0.0140 f1=0.0070 p=0.0571 r=0.0037 ppr=0.00012154184512096307 tp/fp/tn/fn=6/99/862190/1605
- **H** val AUPRC=0.0021870779789774425 | @0.5 auprc=0.0012 f1=0.0000 ppr=0.0 tp/fp/tn/fn=0/0/862289/1611
- **H** @val-thr=0.08: auprc=0.0012 f1=0.0021 p=0.0060 r=0.0012 ppr=0.0003831461974765598 tp/fp/tn/fn=2/329/861960/1609
- **HxX** val AUPRC=0.02786733542391761 | @0.5 auprc=0.0063 f1=0.0000 ppr=0.0 tp/fp/tn/fn=0/0/862289/1611
- **HxX** @val-thr=0.03: auprc=0.0063 f1=0.0146 p=0.3636 r=0.0074 ppr=3.819886560944554e-05 tp/fp/tn/fn=12/21/862268/1599

### Epoch 15 — random-40 diagnostic

- **X** val AUPRC=0.03604077922637433 | @0.5 auprc=0.0350 f1=0.0000 ppr=0.0 tp/fp/tn/fn=0/0/3043901/3106
- **X** @val-thr=0.02: auprc=0.0350 f1=0.0677 p=0.0392 r=0.2505 ppr=0.006519840617366484 tp/fp/tn/fn=778/19088/3024813/2328
- **H** val AUPRC=0.02397433750406783 | @0.5 auprc=0.0114 f1=0.0013 ppr=9.84572729895271e-07 tp/fp/tn/fn=2/1/3043900/3104
- **H** @val-thr=0.03: auprc=0.0114 f1=0.0361 p=0.0498 r=0.0283 ppr=0.0005802415288182798 tp/fp/tn/fn=88/1680/3042221/3018
- **HxX** val AUPRC=0.10508515810807834 | @0.5 auprc=0.0915 f1=0.0026 ppr=2.2973363697556323e-06 tp/fp/tn/fn=4/3/3043898/3102
- **HxX** @val-thr=0.04: auprc=0.0915 f1=0.1685 p=0.3020 r=0.1169 ppr=0.0003944854737780386 tp/fp/tn/fn=363/839/3043062/2743

### Epoch 20 — temporal primary

- **X** val AUPRC=0.0243041570727699 | @0.5 auprc=0.0140 f1=0.0000 ppr=0.0 tp/fp/tn/fn=0/0/862289/1611
- **X** @val-thr=0.01: auprc=0.0140 f1=0.0070 p=0.0571 r=0.0037 ppr=0.00012154184512096307 tp/fp/tn/fn=6/99/862190/1605
- **H** val AUPRC=0.001512122560625324 | @0.5 auprc=0.0012 f1=0.0000 ppr=0.0 tp/fp/tn/fn=0/0/862289/1611
- **H** @val-thr=0.01: auprc=0.0012 f1=0.0000 p=0.0000 r=0.0000 ppr=0.0 tp/fp/tn/fn=0/0/862289/1611
- **HxX** val AUPRC=0.006326362822284401 | @0.5 auprc=0.0013 f1=0.0000 ppr=0.0 tp/fp/tn/fn=0/0/862289/1611
- **HxX** @val-thr=0.01: auprc=0.0013 f1=0.0000 p=0.0000 r=0.0000 ppr=0.0 tp/fp/tn/fn=0/0/862289/1611

### Epoch 20 — random-40 diagnostic

- **X** val AUPRC=0.03604077922637433 | @0.5 auprc=0.0350 f1=0.0000 ppr=0.0 tp/fp/tn/fn=0/0/3043901/3106
- **X** @val-thr=0.02: auprc=0.0350 f1=0.0677 p=0.0392 r=0.2505 ppr=0.006519840617366484 tp/fp/tn/fn=778/19088/3024813/2328
- **H** val AUPRC=0.0014885708842798527 | @0.5 auprc=0.0014 f1=0.0000 ppr=0.0 tp/fp/tn/fn=0/0/3043901/3106
- **H** @val-thr=0.01: auprc=0.0014 f1=0.0019 p=0.0010 r=0.0109 ppr=0.01066259447385582 tp/fp/tn/fn=34/32455/3011446/3072
- **HxX** val AUPRC=0.021908093158376257 | @0.5 auprc=0.0344 f1=0.0000 ppr=0.0 tp/fp/tn/fn=0/0/3043901/3106
- **HxX** @val-thr=0.02: auprc=0.0344 f1=0.0624 p=0.0395 r=0.1484 ppr=0.0038313006829324646 tp/fp/tn/fn=461/11213/3032688/2645

