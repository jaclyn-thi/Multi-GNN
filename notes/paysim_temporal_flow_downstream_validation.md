# PaySim temporal-flow downstream validation (Phase 1)

> Validation-only. Encoder frozen. No test metrics. Exploratory/post-hoc.
> Twin: `results/diagnostics/paysim_temporal_flow_downstream_validation.json`

## Status flags

- `encoder_training=false`
- `encoder_frozen=true`
- `validation_only=true`
- `test_evaluated=false`
- `exploratory_posthoc=true`
- `table_eligible=false` until confirmation

## Frozen representation

- Checkpoint: `gin_emlps_ports_tds_corrected_asym_proj_8192neg_queue0_40ep_seed2.tar`
- SHA256: `18e06f555aa4880dfc1e95caa3f54a207e5aa186d266887772640feb93a06ae6`
- Embeddings: `embeddings/final_corrected_no_preserve_multiseed/seed2_P1_strict_inductive_legacy`
- Contract: `paysim_legacy_duplicate_v1`
- BN: `frozen_aml_bn`
- Norm: `paysim_train_fit_edge_znorm`
- H: post-128

## TF cache

- Dir: `temporal_flow_cache/PaySim`
- Version: `temporal_flow_causal_paysim_v1`
- Features SHA256: `e9595a36a345f5c73088b1fa8d0e0db5cd2254226006b2cc6d7931fc573436d4`
- Tie policy: B simultaneous batch (strictly earlier timestamps)

## Validation AUPRC @ 0.5 (seed-2 logistic)

| Stack | Dim | Val AUPRC | Val AUROC | Val F1@0.5 |
|-------|-----|-----------|-----------|------------|
| X | 12 | 0.003339 | 0.872297 | 0.000000 |
| TF | 5 | 0.001455 | 0.591480 | 0.000000 |
| X+TF | 17 | 0.012764 | 0.906007 | 0.002503 |
| H | 128 | 0.022977 | 0.894399 | 0.057745 |
| H+X | 140 | 0.041084 | 0.877951 | 0.068419 |
| H+TF | 133 | 0.023375 | 0.896782 | 0.069514 |
| H+X+TF | 145 | 0.037926 | 0.876376 | 0.082856 |
| random_H+X+TF (control) | 145 | 0.070219 | — | — |

## Predeclared gate

- Margin: 0.003
- H+X+TF − H+X = -0.003159 (need ≥ 0.003): FAIL
- H+X+TF − X+TF = 0.025161 (need ≥ 0.003): PASS
- H+X+TF > H: PASS
- Gate overall: **FAIL**
- Central transfer criterion (H+X+TF > X+TF w/ margin): PASS

## Exact answers

1. Did TF improve H+X? **False** (Δ=-0.003159)
2. Did H improve X+TF? **True** (Δ=0.025161)
3. Winner (val AUPRC): **H+X**
4. Predeclared gate pass? **False**
5. Attribution: **transferred_H**
6. Multiseed/test confirmation justified? **False**
7. Follow-up jobs auto-submitted? **False**

