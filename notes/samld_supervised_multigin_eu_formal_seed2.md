# SAML-D supervised Multi-GIN+EU formal seed-2

> Twin: `results/diagnostics/samld_supervised_multigin_eu_formal_seed2.json`  
> Protocol: `samld_supervised_multigin_eu_v1` (Candidate A)  
> Train job: `19117881` · Eval job: `19204886`  
> Checkpoint: `/home/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN/saved-models/samld_supervised_multigin_eu_v1_formal_seed2/checkpoint_best_val_f1.tar` (best-val F1; selected epoch **22**)

## Protocol (locked)

- gin + legacy head + emlps + reverse_mp + ego + ports; TDS off; correct_reverse off; preserve off
- legacy per-graph z-norm; edge_dim=6; seed=2; 50 epochs
- Training used `--skip_test_eval` (test locked). This eval scores val+test **once** after selection.
- Gate: versioned **current get_data** seed counts/hashes + NeighborLoader coverage floors
  (min edge 0.85, min positive 0.9); scored ≠ integrity card.

## Primary (paper argmax)

| Split | AUROC | AUPRC | F1 | P | R | PPR |
|-------|------:|------:|---:|--:|--:|----:|
| Val | 1.0000 | 0.9954 | 0.9746 | 0.9841 | 0.9651 | 0.001106 |
| Test | 1.0000 | 0.9970 | 0.9789 | 0.9833 | 0.9745 | 0.001212 |

Confusion (argmax): val TP/FP/TN/FN = 1800/29/1651728/65;  
test TP/FP/TN/FN = 1946/33/1630672/51.

## Alert budgets (P@K)

Val P@100/500/1000: see JSON `splits.val.alert_budget`.  
Test P@100/500/1000: see JSON `splits.test.alert_budget`.

## Diagnostics (not selection)

- Max-validation-AUPRC epoch (diagnostic): **7**
- Val-tuned threshold F1 (NOT paper-compatible): thr=0.0549
- Fixed-0.5 equals paper argmax for two-class softmax

## Comparisons (validation)

| Reference | Val AUPRC / F1 |
|-----------|----------------|
| Prevalence | 0.001045 |
| X-only HGB | 0.7235 |
| Smoke ep1/ep2 AUPRC | 0.9840 / 0.9585 |
| Smoke selected F1 | 0.9044 |
| This formal (argmax) | 0.9954 / F1 0.9746 |

X-only gap is evidence the graph model outperforms the strongest audited feature-only control, not proof that the entire difference is exclusively message passing.

## Paper-comparability

Candidate-A / SAML-D locked protocol only. Do **not** claim IBM AMLWorld table parity beyond that definition.
