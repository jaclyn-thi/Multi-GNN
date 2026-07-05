
# Small-HI Key-Run Explicit Positive-Weight Probe Sweep

Updated from `results/diagnostics/probe_weight_sweep_small_hi_key_runs.json`.

Status: complete, **60/60** cells. This is the current source for whether Small-HI conclusions depend on inherited probe weights.

## Thesis-Relevant Takeaway

Small-HI conclusions are robust to probe class weighting. The best row by both F1 and AUPRC is GINe emlps+tds seed2 with `embedding+raw`, `pos_1`, C=1.0: F1 0.3677, AUPRC 0.3314, P@500 0.6980, lift@500 373.9. This is not a fragile artifact of inherited class weights.

## Best Setting Per Feature Mode By F1

| Features | Run | Weight | C | AUROC | AUPRC | F1 | F1@0.5 | P@500 | lift@500 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `raw+morph` | Small-HI GINe emlps+tds seed1 (20ep) | pos_3 | 1.0 | 0.9050 | 0.0667 | 0.1375 | 0.1444 | 0.1880 | 100.7 |
| `embedding` | Small-HI GINe emlps+tds seed2 (40ep) | pos_1 | 1.0 | 0.9456 | 0.2717 | 0.3167 | 0.1896 | 0.6260 | 335.4 |
| `embedding+raw` | Small-HI GINe emlps+tds seed2 (40ep) | pos_1 | 1.0 | 0.9551 | 0.3314 | 0.3677 | 0.3627 | 0.6980 | 373.9 |
| `embedding+raw+morph` | Small-HI FNF + emlps+tds seed1 | pos_6.275 | 1.0 | 0.9586 | 0.2759 | 0.3477 | 0.3021 | 0.5720 | 306.4 |

## Interpretation

- `pos_1` is often strongest or near strongest. More aggressive weights can preserve reasonable F1 but usually reduce AUPRC/ranking.
- Embedding-only remains strong on Small-HI, but augmented stacks, especially `embedding+raw`, give the best overall F1/AUPRC for the strongest GINe seed.
- FNF is competitive in the full stack but not a clear universal winner.
- Thesis-safe claim: Small-HI has a robust high-signal SSL result, and the conclusion survives reasonable probe-weight changes.

Artifacts: `results/diagnostics/probe_weight_sweep_small_hi_key_runs.json`.
