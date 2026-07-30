# PaySim native tabular baseline (`paysim_native_tabular_baseline`)

> Supervised ceiling diagnostic under the locked temporal split. `table_eligible=false`. Not transfer evaluation.

- Formatted SHA256 verified: `03c2fa07b95d145e754b74a5e646c2d71cd4fed051210d6292a0bbab90112c93`
- Split: steps train 1–280 / val 281–354 / test 355–743 (hashes matched failure audit)
- Tree learner: HistGradientBoostingClassifier (xgboost/lightgbm not installed)
- Jobs: see `results/diagnostics/paysim_native_tabular_baseline/submission.json`

## Primary contract results (`paysim_native_core_v1`)

| Learner | val AUPRC | selected |
|---------|----------:|----------|
| logistic | 0.573588 | |
| mlp | 0.647617 | |
| hgb | 0.661602 | **primary** |

**Primary learner:** `hgb` (val AUPRC=0.661602)

### Locked test (after selection)

| rule | AUROC | AUPRC | F1 | P | R | PPR | AUPRC/π |
|------|------:|------:|---:|--:|--:|----:|--------:|
| 0.5 | 0.9485 | 0.6992 | 0.8134 | 0.9751 | 0.6977 | 0.002356 | 212.4 |
| val-tuned | 0.9485 | 0.6992 | 0.8143 | 0.9776 | 0.6977 | 0.002349 | 212.4 |

P@100/500/1000 @ scores: 1.000 / 1.000 / 1.000

## Gates

1. Native vs X-only val AUPRC improve ≥0.01? **True** (Δ=0.6570110947152569)
2. Exceeds Multi-GIN val AUPRC by ≥0.01? **True** (Δ=0.4937071806904121)
3. isFlaggedFraud Δ val AUPRC: **0.0**
4. GIN primarily feature-contract-limited? **True**
5. Native Multi-GIN justified? **True**

## Protocol caveats

- Temporal full-imbalance cohort (no resampling).
- Compatibility X-only and Multi-GIN use `paysim_legacy_duplicate_v1` (no balances).
- This diagnostic uses raw native balances; not an AMLWorld transfer setup.

## Artifacts

- `results/diagnostics/paysim_native_tabular_baseline.json`
- cells: `results/diagnostics/paysim_native_tabular_baseline/cells/`
