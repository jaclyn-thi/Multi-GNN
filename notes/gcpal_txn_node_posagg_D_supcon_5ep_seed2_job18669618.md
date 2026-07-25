# Positive-aggregation ablation (`D_supcon`)

**B_gcpal only** · 5 epochs · seed 2 · expanding-window eval at ep5

- positive_aggregation: `supcon_mean_logprob`
- job: `18669618`
- config_hash: `f8d50f51cc6d07a028d658b29df6e6664eb6f095501132089294e493eeac22ac`
- mean |P|: 16.4950
- collapse_verdict: ok
- train_seconds: 3092.1

## Temporal HxX (ep5 expanding-window)

- val AUPRC: 0.08567935744584639
- test @0.5 AUPRC/AUROC/F1: 0.08144180169943654 / 0.7972876145364538 / 0.0
- test @val-thr F1: 0.1758082092262986 (thr=0.09999999999999999)

Raw loss magnitudes are **not** comparable across aggregation modes.

