# PaySim native-feature transfer complementarity (seed 2) — smoke

> Twin: `results/diagnostics/paysim_native_feature_transfer_complementarity_seed2/smoke.json`
> Label: **frozen AMLWorld representation + target-native downstream features**
> Validation only. No encoder training. No test evaluation.

## Scientific question

Does a frozen AMLWorld-pretrained representation H add useful information beyond
PaySim-native downstream features X?

## Provenance

- Checkpoint SHA: `18e06f555aa4880dfc1e95caa3f54a207e5aa186d266887772640feb93a06ae6`
- H: P1 post-128 (`embeddings/final_corrected_no_preserve_multiseed/seed2_P1_strict_inductive_legacy`); matched random H reused
- X: `paysim_native_core_v1` (11 cols); continuous train-fit z-norm; one-hots unchanged
- Caveat: newbalanceOrig/newbalanceDest are post-transaction fields

## Primary gate (PaperStyleMLP, margin 0.003)

- Pass: **False**
- Δ(Hpre+X − X): `-0.2166913567202923`
- Δ(Hpre+X − Hrand+X): `0.033025112249169555`
- Δ(Hpre − Hrand) reported: `-0.011720539383139432`

## Val AUPRC by stack

| Stack | MLP | HGB |
|-------|----:|----:|
| `X_native` | 0.3159878516393002 | 0.6607035239932456 |
| `H_pretrained` | 0.0055193045944625894 | 0.007443667446298489 |
| `H_pretrained+X_native` | 0.09929649491900791 | 0.5849804722908452 |
| `H_random` | 0.01723984397760202 | 0.006911048783425 |
| `H_random+X_native` | 0.06627138266983836 | 0.6480635229094734 |

## End answers

1. Hpre+X beats native X? **False**
2. Hpre+X beats Hrand+X? **True**
3. Hpre beats Hrand? **False**
4. MLP/HGB consistent? **True**
5. Representation vs native-feature? **native_features_dominate_or_random_matches_pretrained**
6. Multiseed locked-test justified? **False**
7. Thesis-safe wording: On PaySim validation (seed 2), a frozen AMLWorld post-128 representation combined with PaySim-native tabular features did not meet the predeclared complementarity margins under PaperStyleMLP (Δ≥0.003 vs X and vs random H+X). This is not strict H-only zero-shot transfer. newbalance* fields make this a post-transaction monitoring setting. No PaySim test metrics; no encoder training.
8. No encoder train / no test? **True**

