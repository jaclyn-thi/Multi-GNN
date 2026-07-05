
# Current-Protocol Recent Runs Summary

Updated after alert-budget resume job `17209094` completed.

## Executive Summary

The recent batch is complete and does not need reruns. The strongest thesis-safe result remains Small-HI: frozen GINe SSL embeddings, especially with raw transaction features, produce high AUPRC/F1 and very high alert-budget enrichment. Small-LI is substantially harder because positive prevalence is tiny, but augmented frozen embeddings still provide meaningful ranking enrichment. The safe framing is not that one recipe universally wins; it is that SSL embeddings become most useful when combined with simple transaction and morphology features, and alert-budget metrics are the most honest operational readout.

## Experiment Status

| Experiment | Newest job/log | Output status | Output path | Existing note | Needs scientific analysis? | Needs rerun? | Reason |
|---|---|---|---|---|---|---|---|
| Small-LI generic probe sweep | 17137814 | complete and analyzed | `results/diagnostics/probe_sweep_small_li_current_protocol.json` | `notes/probe_sweep_small_li_current_protocol.md` | No | No | 48/48 complete; superseded for weighting claims by explicit sweep. |
| Small-LI supervised GINe baseline | 17069764 | complete and analyzed | `results/diagnostics/supervised_small_li_gin_emlps_tds_seed1.json` | `notes/small_li_supervised_baseline_comparison.md` | No | No | Final checkpoint is weak operationally but useful as supervised baseline. |
| Small-LI FNF + emlps+tds scout | 17069933 | complete and analyzed | `results/diagnostics/probe_feature_ablation_small_li_fnf_current_protocol_seed1.json` | `notes/small_li_fnf_current_protocol_comparison.md` | No | No | FNF effect is mixed; current note includes alert-budget comparison. |
| Small-LI explicit positive-weight sweep | 17137922 | complete and analyzed | `results/diagnostics/probe_weight_sweep_small_li_current_protocol.json` | `notes/probe_weight_sweep_small_li_current_protocol.md` | No | No | 72/72 complete; current note interprets weighting vs ranking. |
| Small-HI explicit positive-weight sweep | 17137924 | complete and analyzed | `results/diagnostics/probe_weight_sweep_small_hi_key_runs.json` | `notes/probe_weight_sweep_small_hi_key_runs.md` | No | No | 60/60 complete; robust to probe-weight variation. |
| Current-protocol alert-budget metrics | 17209094 | complete and analyzed | `results/diagnostics/alert_budget_metrics_current_protocol.json` | `notes/alert_budget_metrics_current_protocol.md` | No | No | Resume completed Small-LI 7/7 and combined 17 rows. |

## Best Small-HI Results

- Probe weighting: best F1/AUPRC is `Small-HI GINe emlps+tds seed2 (40ep)` with `embedding+raw`, `pos_1`, C=1.0: F1 0.3677, AUPRC 0.3314, P@500 0.6980, lift@500 373.9.
- Alert-budget current protocol: `Small-HI GINe emlps+tds seed2 (40ep)` with `embedding+raw` gives F1 0.3464, AUPRC 0.2883, P@500 0.6380, lift@500 341.8.
- Small-HI claims are robust to reasonable probe class-weight changes.

## Best Small-LI Results

- Explicit weighting: best F1 is `embedding+raw+morph`, `pos_3`, C=0.1: F1 0.1130, AUPRC 0.0450.
- Best ranking/AUPRC is `embedding+raw+morph`, `pos_1`, C=1.0: AUPRC 0.0496, P@100 0.2900, lift@100 424.2.
- Alert-budget best practical GINe configuration is `embedding+raw+morph`, with P@100 0.2300, P@500 0.1160, and lift@500 169.7.

## Alert-Budget Takeaways

- Small-HI has strong absolute precision at practical budgets; Small-LI has lower absolute precision but very high enrichment above prevalence.
- Embedding-only is useful, but augmented feature stacks are more reliable for practical alerting.
- FNF on Small-LI is mixed: it helps some thresholded F1 comparisons, but plain GINe full-stack is stronger at fixed alert budgets.
- Alert-budget metrics are more thesis-safe than thresholded F1 because they reflect what an analyst would actually inspect.

## Thesis-Safe Claims

- Frozen GNN embeddings contain useful signal beyond raw/morphological engineered features, especially when stacked with those features.
- Small-HI current-protocol conclusions are robust across probe class-weight choices and alert-budget metrics.
- Small-LI is much harder in absolute precision, but augmented embeddings can still produce large lift above prevalence.
- Positive probe weighting is an operating-point tool; it can improve F1 but does not automatically improve ranking.

## Claims Needing Caveats

- Do not claim FNF universally improves transfer. Its Small-LI behavior is metric-dependent.
- Do not claim supervised Small-LI is categorically worse as a training paradigm from one final checkpoint; say this final checkpoint is not operationally competitive.
- Do not claim a universal best feature stack across datasets, seeds, and metrics. `embedding+raw` is strongest in some Small-HI rows, while `embedding+raw+morph` is strongest for several Small-LI rankings.

## Optional Follow-Ups

- If thesis space allows, rerun only selected Small-LI/FNF seeds to test stability.
- Consider reporting alert-budget tables alongside AUPRC/F1 for all final comparisons.
- Keep future large caches on Scratch and reusable checkpoints/data on Pool via the current symlinks.

## Artifacts Used

- `results/diagnostics/probe_sweep_small_li_current_protocol.json`
- `results/diagnostics/supervised_small_li_gin_emlps_tds_seed1.json`
- `results/diagnostics/probe_feature_ablation_small_li_fnf_current_protocol_seed1.json`
- `results/diagnostics/probe_weight_sweep_small_li_current_protocol.json`
- `results/diagnostics/probe_weight_sweep_small_hi_key_runs.json`
- `results/diagnostics/alert_budget_metrics_small_hi.json`
- `results/diagnostics/alert_budget_metrics_small_li.json`
- `results/diagnostics/alert_budget_metrics_current_protocol.json`
