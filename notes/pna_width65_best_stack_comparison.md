# PNA width65 best-stack comparison (diagnostic)

Downstream-only probe using validated temporal-flow stack `pre-3h + raw + temporal_flow_causal`. PNA was **not** retrained.

## Caveats

- Downstream-only probe; PNA SSL checkpoint was not retrained.
- PNA width65 is a one-seed scout (seed 1, 20ep); not a full architecture sweep.
- GIN comparison uses Small-HI 40ep seed2; epochs and seeds are not matched.
- Useful for architecture diagnostics, not a definitive architecture ranking.

## Compared rows

| label | AUROC | AUPRC | F1 | P@100 | R@100 | lift@100 | conv | n_iter | test_n |
|-------|------:|------:|---:|------:|------:|---------:|------|-------:|-------:|
| PNA width65 pre-3h | 0.9602 | 0.2303 | 0.2970 | 0.8200 | 0.0509 | 439.29 | converged | 1442 | 863049 |
| PNA width65 pre-3h + raw | 0.9689 | 0.2742 | 0.2792 | 0.8200 | 0.0509 | 439.29 | converged | 1357 | 863049 |
| PNA width65 pre-3h + raw + temporal_flow_causal | 0.9824 | 0.4065 | 0.4099 | 0.9300 | 0.0577 | 498.22 | converged | 1456 | 863049 |
| GIN Small-HI 40ep seed2 pre-3h + raw + temporal_flow_causal | 0.9789 | 0.5006 | 0.4649 | 0.9400 | 0.0583 | 503.58 | converged | 1164 | 863050 |
| PNA width65 post-128 + raw + temporal_flow_causal | 0.9817 | 0.3995 | 0.4225 | 0.8500 | 0.0528 | 455.36 | converged | 1843 | 863049 |

## Primary comparison (PNA − GIN, best stack)

- ΔAUPRC: **-0.0941**
- ΔF1: -0.0550
- ΔP@100: -0.0100
- ΔR@100: -0.0006
- Δlift@100: -5.36
