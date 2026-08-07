# Phase-4B frozen R198 four-cell validation eval

> Twin: `results/diagnostics/financial_multidataset_shared_core_phase4b_frozen_eval.json`
> Training integrity: `results/diagnostics/financial_multidataset_shared_core_phase4b_scout/training_integrity_summary.json`

**ok=True** — validation-only; no encoder retrain; no test.

## Classification: `POSITIVE_EXPANSION`

Phase-3 comparability: `COMPARABLE_REUSE_AUTHORIZED`

### Table A — 3-domain retention vs 2-domain mixed

| Target | 2-domain AUPRC | 3-domain AUPRC | Δ | retention | 2d F1@0.5 | 3d F1@0.5 | 2d BCE | 3d BCE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Small-HI | 0.3034 | 0.3232 | +0.0198 | 1.065 | 0.3763 | 0.3572 | 0.004903 | 0.004726 |
| SAML-D | 0.9307 | 0.9291 | -0.0016 | 0.998 | 0.8908 | 0.8875 | 0.000986 | 0.001386 |

### Table B — Small-LI mixed vs specialist

| Encoder | AUPRC | AUROC | F1@0.5 | F1@val-thr† | P | R | Final val BCE |
|---|---:|---:|---:|---:|---:|---:|---:|
| MIXED_3DOMAIN | 0.0761 | 0.9594 | 0.1144 | 0.1411 | 0.3103 | 0.0701 | 0.003779 |
| SMALL_LI_ONLY | 0.0238 | 0.7529 | 0.0432 | 0.0725 | 0.2857 | 0.0234 | 0.022276 |

† F1@val-thr is an optimistic validation-selected-threshold diagnostic.

### Table C — three-domain summary

| Target | 3-domain mixed AUPRC | specialist AUPRC | Δ | retention |
|---|---:|---:|---:|---:|
| Small-HI | 0.3232 | 0.3306 | -0.0074 | 0.978 |
| SAML-D | 0.9291 | 0.9121 | +0.0170 | 1.019 |
| Small-LI | 0.0761 | 0.0238 | +0.0523 | 3.199 |

## Caveats

- Training InfoNCE / final alpha are not representation-quality conclusions.
- Do not compare absolute AUPRC across Small-HI vs SAML-D vs Small-LI as equal difficulty.
- F1@val-thr is optimistic/in-sample, not a test estimate.
- SAML-D coverage uses floors due to extraction_loader_coverage_defect.
- Phase-3 MIXED_1TO1 metrics reused only after comparability card PASS.

Embeddings disk ≈ 16.85 GiB

Stop after this evaluation — no automatic next training phase.

## Addendum: LI specialist step-500 diagnostic

See [`results/diagnostics/financial_multidataset_shared_core_phase4b_frozen_eval/addendum_li_step500/README.md`](../results/diagnostics/financial_multidataset_shared_core_phase4b_frozen_eval/addendum_li_step500/README.md) and [`addendum_li_step500_summary.json`](../results/diagnostics/financial_multidataset_shared_core_phase4b_frozen_eval/addendum_li_step500_summary.json).
