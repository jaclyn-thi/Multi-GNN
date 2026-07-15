# Small-HI legacy supervised alert-budget (seed 1, formal 100ep)

**Checkpoint:** `/home/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN/saved-models/small_hi_legacy_supervised_gin_emlps_tds_100ep_seed1/checkpoint_best_val_f1.tar` @ epoch **24**

Metrics extracted from existing eval JSON — no model re-run.

**Source:** `results/diagnostics/eval_small_hi_legacy_supervised_gin_emlps_tds_100ep_seed1.json`

## Test split alert-budget

| K | Precision | Recall | Lift |
|---|-----------|--------|------|
| 100 | 0.99 | 0.061452513966480445 | 530.37 |
| 500 | 0.966 | 0.29981378026070765 | 517.51 |
| 1000 | 0.835 | 0.5183116076970825 | 447.33 |

## Caveats

- Uses **best-validation** checkpoint (epoch 24), not last epoch.
- Decision/ranking from **in-GNN supervised** model, not frozen SSL + linear probe.
- Comparable to SSL alert-budget only at ranking semantics level, not protocol level.