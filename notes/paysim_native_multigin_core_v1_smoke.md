# PaySim-native Multi-GIN smoke (`paysim_native_multigin_core_v1`)

- **Gate pass:** `True`
- **Job:** `19123387`
- **Run:** `paysim_native_multigin_core_v1_smoke_seed2`
- **Edge dim (ports on, TDS off):** `13`

## Scientific question

Does supervised Multi-GIN using PaySim native balance features materially improve over balance-free compatibility-contract Multi-GIN, and can it compete with native HGB val AUPRC 0.6616?

## Deployment caveat

newbalanceOrig/newbalanceDest make this a post-transaction supervised ceiling; may be unavailable pre-authorization.

## Protocol

- Candidate-A Multi-GIN (legacy head, emlps/reverse_mp/ego/ports on, tds off)
- Native feature contract (not AML transfer / not contrastive / not paper table claim)
- Adam + AML-derived class weights ~(1.0, 6.275); may need later weighting ablation
- 2 epochs, seed 2, `--skip_test_eval`, `--save_model`
- Locked steps: train 1–280 / val 281–354 / test 355–743 (test locked)

## Features (exact order)

1. `time`
2. `log1p_amount`
3. `type_PAYMENT`
4. `type_TRANSFER`
5. `type_CASH_OUT`
6. `type_DEBIT`
7. `type_CASH_IN`
8. `oldbalanceOrg`
9. `newbalanceOrig`
10. `oldbalanceDest`
11. `newbalanceDest`
12. `in_port`
13. `out_port`

## Epoch report

```json
[
  {
    "epoch": 1,
    "train_loss": 0.018499463016500302,
    "validation_auroc": 0.9717688688386362,
    "validation_auprc": 0.6126387420137948,
    "validation_minority_f1_argmax": 0.6006734006734006,
    "validation_precision_argmax": 0.6326241134751773,
    "validation_recall_argmax": 0.5717948717948718,
    "validation_positive_prediction_rate": 0.0005523883548699497,
    "validation_tp": 446.0,
    "validation_fp": 259.0,
    "validation_tn": 1275237.0,
    "validation_fn": 334.0,
    "validation_positive_coverage": 0.5717948717948718,
    "scores_finite": true
  },
  {
    "epoch": 2,
    "train_loss": 0.011050233593429274,
    "validation_auroc": 0.9816449668791978,
    "validation_auprc": 0.6664802592872943,
    "validation_minority_f1_argmax": 0.6753812636165577,
    "validation_precision_argmax": 0.7788944723618091,
    "validation_recall_argmax": 0.5961538461538461,
    "validation_positive_prediction_rate": 0.0004677678931013246,
    "validation_tp": 465.0,
    "validation_fp": 132.0,
    "validation_tn": 1275362.0,
    "validation_fn": 315.0,
    "validation_positive_coverage": 0.5961538461538461,
    "scores_finite": true
  }
]
```

- **Best val-F1 epoch:** `2` (F1=`0.6753812636165577`)
- **Max val-AUPRC epoch:** `2` (AUPRC=`0.6664802592872943`)

## Validation-only comparisons (different learners/protocols)

| Reference | Val AUPRC |
|-----------|----------:|
| Legacy compatibility X-only | ~0.0046 |
| Balance-free Multi-GIN Candidate A | ~0.168 |
| Native logistic | 0.5736 |
| Native MLP | 0.6476 |
| Native HGB | 0.6616 |
| **This smoke (max)** | **0.6664802592872943** |

- Pass threshold vs Multi-GIN: `0.21800000000000003` (~0.168 + 0.05)
- Material exceed Multi-GIN: `True`
- vs HGB: `{"native_hgb_val_auprc": 0.6616, "smoke_max_val_auprc": 0.6664802592872943, "delta": 0.004880259287294297, "approaches_or_exceeds": true, "exceeds": true, "note": "Learner/protocol differ (GNN vs HGB tabular); compare cautiously."}`

## Checks

- `no_historical_overwrite`: **PASS**
- `candidate_a_native_flags`: **PASS**
- `class_weights_aml_derived`: **PASS**
- `cohort_hashes`: **PASS**
- `checkpoints_reload`: **PASS**
- `finite_losses_and_scores`: **PASS**
- `validation_coverage_counts`: **PASS**
- `no_collapse`: **PASS**
- `material_vs_balance_free_multigin`: **PASS**
- `test_evaluated_false`: **PASS**

## Formal seed-2

- **Justified by this smoke:** `True`
- **Auto-submitted:** `false` (hard stop after smoke)

Twin JSON: `results/diagnostics/paysim_native_multigin_core_v1_smoke.json`

