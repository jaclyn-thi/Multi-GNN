# Formal aggregate: Multi-GIN+EU ports TDS-off 50ep (seeds 1–3)

Formal post-hoc evaluations of `checkpoint_best_val_f1.tar` with `model.eval()`, `paper_argmax` decision rule, `tds=False` / edge_dim=6, legacy head, ports/ego/reverse_mp/emlps. **Test metrics only** enter the aggregate; train-time metrics are diagnostic.

## Per-seed formal results (paper_argmax)

| Seed | Epoch | Test F1 | P | R | AUROC | AUPRC | Train cov | Val cov | Test cov |
|------|------:|--------:|--:|--:|------:|------:|----------:|--------:|---------:|
| 1 | 27 | 0.6633 | 0.6799 | 0.6474 | 0.9870 | 0.6747 | 0.999792 | 0.999953 | 0.999039 |
| 2 | 43 | 0.7176 | 0.8147 | 0.6412 | 0.9829 | 0.6742 | 0.999793 | 0.999940 | 0.999045 |
| 3 | 13 | 0.5984 | 0.5458 | 0.6623 | 0.9851 | 0.6399 | 0.999795 | 0.999944 | 0.999035 |

### Prediction counts (derived from n, prevalence, P/R)

| Seed | Split | n | n_pos | pred_pos | TP | FP | FN |
|------|-------|--:|------:|---------:|---:|---:|---:|
| 1 | train | 3248245 | 2530.0 | 1138.0 | 938.0 | 200.0 | 1592.0 |
| 1 | val | 965479 | 1036.0 | 639.0 | 511.0 | 128.0 | 525.0 |
| 1 | test | 863070 | 1611.0 | 1534.0 | 1043.0 | 491.0 | 568.0 |
| 2 | train | 3248249 | 2530.0 | 1092.0 | 952.0 | 140.0 | 1578.0 |
| 2 | val | 965466 | 1035.0 | 640.0 | 521.0 | 119.0 | 514.0 |
| 2 | test | 863075 | 1611.0 | 1268.0 | 1033.0 | 235.0 | 578.0 |
| 3 | train | 3248256 | 2530.0 | 1428.0 | 957.0 | 471.0 | 1573.0 |
| 3 | val | 965470 | 1036.0 | 833.0 | 510.0 | 323.0 | 526.0 |
| 3 | test | 863066 | 1611.0 | 1955.0 | 1067.0 | 888.0 | 544.0 |

### vs training-time metrics at best-val epoch (diagnostic)

| Seed | Train-time test F1 | Formal test F1 | Δ |
|------|-------------------:|---------------:|--:|
| 1 | 0.6510 | 0.6633 | +0.0123 |
| 2 | 0.6851 | 0.7176 | +0.0325 |
| 3 | 0.5906 | 0.5984 | +0.0078 |

## Formal test aggregate (paper_argmax F1 only)

- **mean ± sample SD:** 0.6598 ± 0.0597
- **median / range:** 0.6633 / [0.5984, 0.7176] (range=0.1192)
- **Paper Multi-GIN+EU:** 0.6479 ± 0.0122
- **Δ mean vs paper:** +0.0119
- **Mean reproduced?** yes (|Δ|≤paper σ → True)
- **Paper low variance reproduced?** no (sample σ=0.0597 vs paper σ=0.0122)

## vs old TDS-on formal (seed 1)

- Run: `small_hi_legacy_supervised_gin_emlps_tds_100ep_seed1` @ epoch 24
- Test paper_argmax F1 **0.5394** (P 0.4530, R 0.6667, AUROC 0.9837, AUPRC 0.6388)
- Formal TDS-off mean F1 − TDS-on: **+0.1203**

## Artifacts

- Aggregate JSON: `results/diagnostics/eval_small_hi_legacy_supervised_gin_emlps_ports_50ep_seeds1-3_formal_aggregate.json`
- Per-seed notes: `notes/eval_{run}.md`
- Train-time diagnostic aggregate (separate): `results/diagnostics/supervised_Small-HI_ports_50ep_seeds1-3_aggregate.json`
