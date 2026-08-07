# Phase-3 frozen R198 six-cell validation eval

> Twin: `results/diagnostics/smallhi_samld_mixed_ssl_phase3_frozen_eval.json`
> Integrity training: `results/diagnostics/smallhi_samld_mixed_ssl_phase3_scout/training_integrity_summary.json`

**ok=True** — validation-only; no encoder retrain; no test.

## Interpretation: `USEFUL_COMPROMISE`

### Small-HI validation table

| Encoder | Role | AUPRC | AUROC | F1@0.5 | F1@val-thr | P@0.5 | R@0.5 | Final val BCE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| SMALL_HI_ONLY | in_domain_specialist | 0.3306 | 0.9712 | 0.3782 | 0.4098 | 0.6038 | 0.2754 | 0.004888 |
| SAMLD_ONLY | frozen_cross_domain_transfer | 0.1352 | 0.9216 | 0.2414 | 0.2441 | 0.2959 | 0.2039 | 0.007594 |
| MIXED_1TO1 | mixed_multi_domain | 0.3034 | 0.9719 | 0.3763 | 0.3948 | 0.5397 | 0.2889 | 0.004903 |

### SAML-D validation table

| Encoder | Role | AUPRC | AUROC | F1@0.5 | F1@val-thr | P@0.5 | R@0.5 | Final val BCE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| SMALL_HI_ONLY | frozen_cross_domain_transfer | 0.8596 | 0.9891 | 0.8357 | 0.8372 | 0.9338 | 0.7562 | 0.002961 |
| SAMLD_ONLY | in_domain_specialist | 0.9121 | 0.9993 | 0.8733 | 0.8782 | 0.9671 | 0.7961 | 0.001742 |
| MIXED_1TO1 | mixed_multi_domain | 0.9307 | 0.9994 | 0.8908 | 0.9037 | 0.9928 | 0.8078 | 0.000986 |

## Deltas (AUPRC)

- Small-HI mixed−specialist: -0.0271 (retention 0.918)
- Small-HI mixed−transfer: +0.1683
- SAML-D mixed−specialist: +0.0186 (retention 1.020)
- SAML-D mixed−transfer: +0.0711

## Caveats

- Training InfoNCE is not a representation-quality conclusion.
- Do not compare absolute AUPRC across Small-HI vs SAML-D as equal difficulty.
- F1@val-thr is optimistic/in-sample, not a test estimate.
- SAML-D coverage uses floors due to extraction_loader_coverage_defect.

## Probe recipe

```json
{
  "learner": "PaperStyleMLP",
  "architecture": "Linear(d\u2192128)\u2192ReLU\u2192Dropout(0.1)\u2192Linear(128\u21921)",
  "input_dim": 198,
  "epochs": 20,
  "lr": 0.001,
  "batch_size": 8192,
  "seed": 2,
  "loss": "binary_cross_entropy_with_logits",
  "class_weights": null,
  "pos_weight": null,
  "reduction": "mean",
  "feature_scaler": "StandardScaler_fit_train_Z",
  "selection_within_probe": "best_val_auprc",
  "reported_bce": "final_probe_epoch_unweighted_mean_BCE"
}
```

Embeddings disk ≈ 24.60 GiB

