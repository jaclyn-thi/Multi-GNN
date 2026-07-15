# Small-LI legacy supervised alert-budget (seed 1, formal 100ep)

**Checkpoint:** `/home/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN/saved-models/small_li_legacy_supervised_gin_emlps_tds_100ep_seed1/checkpoint_best_val_f1.tar` @ epoch **35**

Metrics extracted from existing eval JSON — no model re-run.

**Source:** `results/diagnostics/eval_small_li_legacy_supervised_gin_emlps_tds_100ep_seed1.json`

## Test split alert-budget

| K | Precision | Recall | Lift |
|---|-----------|--------|------|
| 100 | 0.97 | 0.12094763092269327 | 1418.99 |
| 500 | 0.462 | 0.2880299251870324 | 675.85 |
| 1000 | 0.27 | 0.33665835411471323 | 394.98 |

## Caveats

- Uses **best-validation** checkpoint (epoch 35), not last epoch (collapse at ep 100).
- Decision/ranking from **in-GNN supervised** model, not frozen SSL + linear probe.
- Comparable to SSL alert-budget only at ranking semantics level, not protocol level.