# Positive-aggregation ablation (B_gcpal)

Status: **complete** · A/B references read-only · C/D newly trained 5ep

Companion: [`results/diagnostics/gcpal_txn_node_posagg_ablation.json`](../results/diagnostics/gcpal_txn_node_posagg_ablation.json)

## Jobs

- C (`logmeanexp_count_normalized`): `18669617`
- D (`supcon_mean_logprob`): `18669618`

## Selection (validation HxX AUPRC only)

**Selected:** D neighbor / SupCon (`supcon_mean_logprob`) val AUPRC=0.085679

Test metrics for selected (after rule): AUPRC@0.5=0.081442, AUROC=0.797288, F1@val-thr=0.175808

## Comparison table (fixed epoch 5, expanding-window)

| Condition | Aggregation | mean\|P\| | val HxX AUPRC | test HxX AUPRC@0.5 | F1@val-thr | AUROC | collapse | Δval vs A | Δval vs B |
|-----------|-------------|----------:|--------------:|-------------------:|-----------:|------:|----------|----------:|----------:|
| A identity / sum | `sum_logsumexp` | 1.0000 | 0.017412 | 0.007159 | 0.031634 | 0.633526 | n/a (reference) | +0.000000 | -0.064630 |
| B neighbor / sum | `sum_logsumexp` | 16.4758 | 0.082042 | 0.136705 | 0.270476 | 0.799853 | n/a (reference) | +0.064630 | +0.000000 |
| C neighbor / logmeanexp | `logmeanexp_count_normalized` | 16.4950 | 0.085214 | 0.103641 | 0.212174 | 0.801041 | ok | +0.067802 | +0.003172 |
| D neighbor / SupCon | `supcon_mean_logprob` | 16.4950 | 0.085679 | 0.081442 | 0.175808 | 0.797288 | ok | +0.068267 | +0.003638 |

## Interpretation

All B variants beat A on val HxX AUPRC: positive-set result is robust to aggregation choice.

- Neighbor positives help after count normalization: **True**
- Multiseed replication justified: **True**

## Confirmation

- Default `sum_logsumexp` unchanged when `--positive_aggregation` omitted.
- Historical A/B checkpoints and scout artifacts not modified.
- Primary eval: `frozen_checkpoint_temporal_expanding_window_v1` only.
- Raw loss magnitudes not used as quality across modes.

