
# Small-LI Explicit Positive-Weight Probe Sweep

Updated from `results/diagnostics/probe_weight_sweep_small_li_current_protocol.json`.

Status: complete, **72/72** cells. This is the current source for Small-LI probe-weight conclusions.

## Thesis-Relevant Takeaway

Explicit probe weighting helps thresholded F1, but the best ranking metrics still come from the same augmented GINe feature stack. The best F1 is `embedding+raw+morph` with `pos_3`, C=0.1 (F1 0.1130), while the best AUPRC is `embedding+raw+morph` with `pos_1`, C=1.0 (AUPRC 0.0496). That split matters: stronger positive weighting can improve the operating threshold without necessarily improving the underlying ranking.

## Best Setting Per Feature Mode By F1

| Features | Run | Weight | C | AUROC | AUPRC | F1 | F1@0.5 | P@500 | lift@500 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `raw+morph` | Small-LI GINe emlps+tds seed1 (20ep) | pos_10 | 1.0 | 0.8580 | 0.0161 | 0.0572 | 0.0462 | 0.0620 | 90.7 |
| `embedding` | Small-LI GINe emlps+tds seed1 (20ep) | pos_1 | 0.1 | 0.8946 | 0.0271 | 0.0835 | 0.0000 | 0.1020 | 149.2 |
| `embedding+raw` | Small-LI GINe emlps+tds seed1 (20ep) | pos_3 | 0.1 | 0.9096 | 0.0386 | 0.0991 | 0.0817 | 0.1300 | 190.2 |
| `embedding+raw+morph` | Small-LI GINe emlps+tds seed1 (20ep) | pos_3 | 0.1 | 0.9214 | 0.0450 | 0.1130 | 0.1091 | 0.1440 | 210.7 |

## Interpretation

- The inherited `model` probe weighting was not catastrophic, but it was not the best thresholded operating point.
- Moderate explicit weights (`pos_1` to `pos_3`) are safer than heavier weights for the strongest augmented stack.
- The best F1 setting has lower AUPRC than the best ranking setting (0.0450 vs 0.0496), so this is partly threshold/operating-point tuning.
- Thesis-safe claim: Small-LI benefits from augmented frozen embeddings, and explicit positive weighting can improve F1, but ranking/alert metrics should remain primary.

Artifacts: `results/diagnostics/probe_weight_sweep_small_li_current_protocol.json`.
