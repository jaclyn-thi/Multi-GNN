# Phase-4B specialist comparison addendum (LI step-500)

> Versioned addendum — does **not** overwrite historical Phase-4B frozen-eval JSON.  
> Artifacts:
> - [`specialist_comparison_with_li_step500.json`](../results/diagnostics/financial_multidataset_shared_core_phase4b_mixed_long_frozen_eval/specialist_comparison_with_li_step500.json)
> - [`specialist_comparison_with_li_step500.csv`](../results/diagnostics/financial_multidataset_shared_core_phase4b_mixed_long_frozen_eval/specialist_comparison_with_li_step500.csv)
> - [`specialist_comparison_three_domain_vs_long3000.csv`](../results/diagnostics/financial_multidataset_shared_core_phase4b_mixed_long_frozen_eval/specialist_comparison_three_domain_vs_long3000.csv)
>
> Sources (read-only reuse; no re-extract / no re-probe):
> - [`addendum_li_step500/summary.json`](../results/diagnostics/financial_multidataset_shared_core_phase4b_frozen_eval/addendum_li_step500/summary.json)
> - Phase-4B / LONG frozen-eval cell tables
> - Phase-3 specialist cells (HI / SAML)

**Protocol:** frozen R198 → PaperStyleMLP probe (20 ep, lr=1e-3, bs=8192, seed=2, StandardScaler on train Z, best-val-AUPRC selection). Feature contract `financial_multidataset_shared_core_v1`. No projection. No test access.

---

## Small-LI specialist ladder

| Encoder | LI updates | AUPRC | AUROC | F1@0.5 | P@0.5 | R@0.5 | F1@val-thr† | final BCE | train_n / val_n / val_pos | val EdgeID |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| SMALL_LI_ONLY@500 | 500 | 0.0588 | 0.9383 | 0.1215 | 0.1948 | 0.0883 | 0.1252 | 0.004552 | 4431773 / 1316415 / 770 | `fee6f1feb56c…` |
| SMALL_LI_ONLY@1000 | 1000 | 0.0238 | 0.7529 | 0.0432 | 0.2857 | 0.0234 | 0.0725 | 0.022276 | same | same |

Specialist **degraded** from step 500 → 1000 (AUPRC Δ = −0.0350).

## Small-LI mixed encoders (same matched cohort)

| Encoder | LI updates | AUPRC | AUROC | F1@0.5 | P@0.5 | R@0.5 | F1@val-thr† | final BCE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SHORT MIXED@1500 | 500 | 0.0761 | 0.9594 | 0.1144 | 0.3103 | 0.0701 | 0.1411 | 0.003779 |
| LONG MIXED@1500 | 500 | 0.0926 | 0.9634 | 0.0985 | 0.5060 | 0.0545 | 0.1596 | 0.003620 |
| LONG MIXED@3000 | 1000 | **0.1056** | 0.9613 | **0.1365** | 0.4919 | 0.0792 | 0.1798 | 0.003667 |

Coverage edge_min = 0.999738; positive_min = 1.0; EdgeID `fee6f1feb56c…` (shared with LI specialists).

† F1@val-thr is an optimistic validation-selected-threshold diagnostic.

---

## Two interpretations

### 1. Fixed-checkpoint specialist (predeclared final)

**SMALL_LI_ONLY@1000**

| Contrast | AUPRC Δ | AUPRC retention | F1@0.5 Δ |
|---|---:|---:|---:|
| LONG@3000 − LI@1000 | **+0.0818** | **4.44×** | +0.0932 |

LONG@3000 **exceeds** the fixed specialist.

### 2. Validation-selected specialist (mildly optimistic)

**SMALL_LI_ONLY@500** — best validation AUPRC among already-evaluated {500, 1000}.  
Not a test estimate; selection uses validation AUPRC only.

| Contrast | AUPRC Δ | AUPRC retention | F1@0.5 Δ |
|---|---:|---:|---:|
| LONG@3000 − LI@500 | **+0.0468** | **1.80×** | +0.0149 |

LONG@3000 **still exceeds** the strongest evaluated LI specialist.

Also: SHORT@1500 and LONG@1500 both beat LI@500 on AUPRC (+0.0173 and +0.0338).

---

## Three-domain specialist vs LONG MIXED@3000

| Target | Specialist | Spec AUPRC | LONG@3000 AUPRC | Δ | Retention | LONG wins? |
|---|---|---:|---:|---:|---:|---|
| Small-HI | SMALL_HI_ONLY (phase3) | 0.3306 | 0.3881 | +0.0575 | 1.174 | yes |
| SAML-D | SAMLD_ONLY (phase3) | 0.9121 | 0.9147 | +0.0026 | 1.003 | yes (marginal) |
| Small-LI | LI@1000 fixed | 0.0238 | 0.1056 | +0.0818 | 4.438 | yes |
| Small-LI | LI@500 val-selected† | 0.0588 | 0.1056 | +0.0468 | 1.796 | yes |

† Mildly optimistic validation selection among evaluated specialist checkpoints.

---

## Confirmation

- No re-extraction, no re-probe, no encoder retrain  
- No test access  
- Historical Phase-4B / LONG result JSON not overwritten  
