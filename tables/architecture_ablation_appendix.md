# Appendix — Architecture ablation

| Encoder | Hidden dim | Pre dim | Post dim | Params | AUROC | AUPRC | F1 | Caveat |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gin | 66 | 198 | 128 | — | 0.944 | 0.213 | 0.259 | — |
| gat | — | — | 128 | — | 0.932 | 0.169 | 0.264 | — |
| pna | 20 | 60 | 128 | — | 0.946 | 0.112 | 0.208 | not capacity-matched to GIN |
| rgcn | — | — | 128 | — | 0.940 | 0.155 | 0.220 | — |
| pna (width-aligned) | 65 | 195 | 128 | — | 0.954 | 0.147 | 0.216 | GIN-matched LR/dropout; seed 1 scout |
| pna (width-aligned, best stack) | 65 | 195 | 128 | — | 0.982 | 0.407 | 0.410 | pre-3h+raw+temporal-flow; one seed; downstream-only diagnostic |

**Notes:**
- Comparable rows only: embedding-only, post-128, shared probe settings, Small-HI architecture sweep (results/diagnostics/architecture_sweep_shared_probe_weights.json).
- Default PNA (hidden 20, pre dim 60) was not capacity/hyperparameter matched to GIN (hidden 66, pre dim 198).
