# Target-supervised PaySim Multi-GIN+EU baseline using the paper-faithful architectural configuration.

> Twin: `results/diagnostics/paysim_supervised_multigin_eu.json`

## Protocol

- PaySim · GINe · supervised · legacy head · ports+emlps+reverse_mp+ego
- TDS / preserve / corrected reverse / TF-in: **off**
- edge_dim=6 · contract=`paysim_legacy_duplicate_v1` · legacy per-graph z-norm
- Selection: best validation minority F1 · decision: paper_argmax
- Test never used for selection (train-time test logs are diagnostic only)
- **Not** an exact published PaySim reproduction

## Aggregate (seeds 1–3)

- Test paper_argmax F1: 0.2020 ± 0.0070 (median 0.2040; per-seed [0.20404858299595138, 0.19430979978925184, 0.2077867999167187])
- Test AUROC: 0.9549 ± 0.0096 (median 0.9602; per-seed [0.9437911257211056, 0.960704676409646, 0.9601983571434103])
- Test AUPRC: 0.2553 ± 0.0267 (median 0.2531; per-seed [0.22975817249820313, 0.25311468951383675, 0.2830450212026745])
- Val paper_argmax F1: 0.1907 ± 0.0047 (median 0.1909; per-seed [0.19091751621872105, 0.18589743589743588, 0.19532044760935913])

## Comparisons (cautious; protocols differ)

- Supervised Multi-GIN **has PaySim fraud labels** — upper/reference ceiling, not a fair label-free transfer competitor.
- X-only val AUPRC@0.5: 0.004590890212575511
- Frozen P1 seed2 val AUPRC@0.5: 0.021656440218880312
- BN P2 seed2 val AUPRC@0.5: 0.02242386865062676
- Sequential SSL aggregate present: True

