# Positive-aggregation ablation (`C_logmeanexp`)

**B_gcpal only** · 5 epochs · seed 2 · expanding-window eval at ep5

- positive_aggregation: `logmeanexp_count_normalized`
- job: `18669617`
- config_hash: `dfdb4502b81593953c33abbc1646cd8a6cf29c93e459ca212ff2bc1b2a3f3933`
- mean |P|: 16.4950
- collapse_verdict: ok
- train_seconds: 3224.8

## Temporal HxX (ep5 expanding-window)

- val AUPRC: 0.08521364707431689
- test @0.5 AUPRC/AUROC/F1: 0.10364129473017063 / 0.801041064550565 / 0.0
- test @val-thr F1: 0.21217391304347827 (thr=0.09999999999999999)

Raw loss magnitudes are **not** comparable across aggregation modes.

