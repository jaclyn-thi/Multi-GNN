# PaySim native Multi-GIN formal seed-2 (`paysim_native_multigin_core_v1`)

> Twin: `results/diagnostics/paysim_native_multigin_core_v1_formal_seed2.json`  
> Train job: `19124783` · Eval job: `19124784`  
> Checkpoint: `/home/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN/saved-models/paysim_native_multigin_core_v1_formal_seed2/checkpoint_best_val_f1.tar` (best-val F1; selected epoch **42**)

## Protocol (locked to smoke)

- gin + legacy head + emlps + reverse_mp + ego + ports; TDS/preserve/correct_reverse off
- native 13-d edge contract; train-fit continuous z-norm; one-hots unchanged
- Adam + AML-derived class weights ~(1.0, 6.275); seed=2; 50 epochs
- Training used `--skip_test_eval`. This eval scores val+test **once** after selection.
- Scaler SHA256: `45ce032c08ae0f3ef73f11f3a778bbc351da7bd43b3316ab583c941d4bcbae27` (must match smoke `45ce032c08ae0f3ef73f11f3a778bbc351da7bd43b3316ab583c941d4bcbae27`)

## Deployment caveat

newbalanceOrig/newbalanceDest are post-transaction fields; may be unavailable pre-authorization.

## Primary (paper argmax / fixed-0.5)

| Split | AUROC | AUPRC | F1 | P | R | PPR |
|-------|------:|------:|---:|--:|--:|----:|
| Val | 0.9971 | 0.7845 | 0.7812 | 0.9538 | 0.6615 | 0.000424 |
| Test | 0.9953 | 0.8596 | 0.7872 | 0.9893 | 0.6536 | 0.002175 |

Confusion (argmax): val TP/FP/TN/FN = 516/25/1275470/264;  
test TP/FP/TN/FN = 2783/30/1289235/1475.

## Validation-selected threshold (not paper-compatible; reported only)

- thr=0.311901
- Test F1/P/R at val-tuned thr: 0.8173 / 0.9715 / 0.7053

## Alert budgets (P@K)

Test P@100=1.0, P@500=1.0, P@1000=1.0  
(full keys in JSON `splits.*.alert_budget`)

## Diagnostics (not selection)

- Max-validation-AUPRC epoch (diagnostic): **42**
- Smoke max val AUPRC: 0.6665 (job 19123387)

## Comparisons (protocol differences labeled)

| Reference | Metric | Value |
|-----------|--------|------:|
| Native HGB (`paysim_native_core_v1`, tabular) | test AUPRC / F1@0.5 | 0.6992 / 0.8134 |
| Balance-free Multi-GIN seed2 (legacy X, dim=6) | test AUPRC / F1 argmax | 0.2531 / 0.1943 |
| This formal native Multi-GIN | test AUPRC / F1 argmax | 0.8596 / 0.7872 |

Native HGB is tabular (no graph MP). Balance-free Multi-GIN uses paysim_legacy_duplicate_v1 edge_dim=6 without balances. This run uses paysim_native_multigin_core_v1 edge_dim=13 with train-fit continuous z-norm. Do not treat gaps as pure architecture effects.

## Selection integrity

- `test_used_for_selection`: false
- `test_evaluated_exactly_once`: true
