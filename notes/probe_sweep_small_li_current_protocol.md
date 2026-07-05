
# Small-LI Current-Protocol Generic Probe Sweep

Updated from `results/diagnostics/probe_sweep_small_li_current_protocol.json`.

Status: complete, **48/48** cells. This is superseded for final probe-weight conclusions by the explicit positive-weight sweep, but it remains the baseline sweep over `model`, `none`, and `balanced` probe weights.

## Thesis-Relevant Takeaway

Frozen SSL embeddings are useful on Small-LI mainly when combined with transaction and morphology features. Embedding-only is better than pure engineered features on some ranking metrics, but the best results come from augmented stacks. The gap between AUPRC/lift and threshold-tuned F1 reinforces that Small-LI is a low-prevalence ranking problem.

| Criterion | Features | Weight | C | AUROC | AUPRC | F1 | F1@0.5 | P@100 | P@500 | P@1000 | lift@500 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Best F1 | `embedding+raw+morph` | none | 0.01 | 0.9028 | 0.0383 | 0.1052 | 0.0193 | 0.2200 | 0.1280 | 0.0930 | 187.2 |
| Best AUPRC | `embedding+raw+morph` | none | 1.0 | 0.9236 | 0.0496 | 0.0742 | 0.0925 | 0.2900 | 0.1540 | 0.1090 | 225.3 |

## Interpretation

- Best AUPRC and alert precision favor `embedding+raw+morph`, reaching AUPRC 0.0496, P@100 0.2900, and lift@100 424.2.
- Best F1 (0.1052) is threshold-sensitive, so it should not be the only headline.
- Use the explicit positive-weight sweep for current weighting claims; use this output as the baseline that showed augmented frozen embeddings matter.

Artifacts: `results/diagnostics/probe_sweep_small_li_current_protocol.json`.
