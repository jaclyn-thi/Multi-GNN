# GBT stdfloor full3000 frozen R198 validation eval

> Twin: `results/diagnostics/financial_multidataset_graph_barlow_twins_stdfloor_1e4_full3000_frozen_eval.json`
> Objective: `edge_aligned_graph_barlow_twins_r198_stdfloor_1e4` (gbt_std_floor=0.0001)

**ok=True** — validation-only; no encoder retrain; no test.

Baseline comparability: `COMPARABLE_REUSE_AUTHORIZED`

## Protocol locks

- contract: `financial_multidataset_shared_core_v1`
- probe: PaperStyleMLP 20ep lr=0.001 bs=8192 seed=2
- extract: full-subgraph R198 train/val; projection bypassed
- checkpoints: `results/checkpoints/financial_multidataset_graph_barlow_twins_stdfloor_1e4_full3000_seed2` steps 1500 + 3000 only

## Main table

| Arm | Step | Target | AUPRC | AUROC | F1@0.5 | P | R | F1@val-thr | Final val BCE |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| GBT_STDFLOOR_1500 | 1500 | Small-HI | 0.2040 | 0.9557 | 0.2480 | 0.4806 | 0.1671 | 0.2982 | 0.005707 |
| GBT_STDFLOOR_1500 | 1500 | SAML-D | 0.8718 | 0.9798 | 0.8562 | 0.9666 | 0.7685 | 0.8608 | 0.003765 |
| GBT_STDFLOOR_1500 | 1500 | Small-LI | 0.0444 | 0.9457 | 0.0698 | 0.2627 | 0.0403 | 0.1102 | 0.004448 |
| GBT_STDFLOOR_3000 | 3000 | Small-HI | 0.2071 | 0.9547 | 0.2353 | 0.4923 | 0.1546 | 0.2860 | 0.005704 |
| GBT_STDFLOOR_3000 | 3000 | SAML-D | 0.8150 | 0.9762 | 0.8228 | 0.9211 | 0.7435 | 0.8238 | 0.004601 |
| GBT_STDFLOOR_3000 | 3000 | Small-LI | 0.0491 | 0.9434 | 0.0963 | 0.2282 | 0.0610 | 0.1085 | 0.004357 |

## GBT@3000 vs GBT@1500

- **Small-HI**: ΔAUPRC=+0.0031 (0.2040 → 0.2071)
- **SAML-D**: ΔAUPRC=-0.0568 (0.8718 → 0.8150)
- **Small-LI**: ΔAUPRC=+0.0047 (0.0444 → 0.0491)

## Deltas vs reused baselines (AUPRC)

- **GBT_STDFLOOR_1500 / Small-HI vs INFONCE_ONLY@3000**: Δ=+0.1939 (retention 20.268)
- **GBT_STDFLOOR_1500 / Small-HI vs EXPERT_ONLY@3000**: Δ=-0.2135 (retention 0.489)
- **GBT_STDFLOOR_1500 / Small-HI vs PROJECTION_ON_ADAPTIVE@3000**: Δ=-0.1877 (retention 0.521)
- **GBT_STDFLOOR_1500 / Small-HI vs ADAPTIVE_LONG@3000**: Δ=-0.1841 (retention 0.526)
- **GBT_STDFLOOR_1500 / SAML-D vs INFONCE_ONLY@3000**: Δ=+0.2911 (retention 1.501)
- **GBT_STDFLOOR_1500 / SAML-D vs EXPERT_ONLY@3000**: Δ=-0.0783 (retention 0.918)
- **GBT_STDFLOOR_1500 / SAML-D vs PROJECTION_ON_ADAPTIVE@3000**: Δ=-0.0732 (retention 0.923)
- **GBT_STDFLOOR_1500 / SAML-D vs ADAPTIVE_LONG@3000**: Δ=-0.0429 (retention 0.953)
- **GBT_STDFLOOR_1500 / Small-LI vs INFONCE_ONLY@3000**: Δ=+0.0356 (retention 5.044)
- **GBT_STDFLOOR_1500 / Small-LI vs EXPERT_ONLY@3000**: Δ=-0.0674 (retention 0.397)
- **GBT_STDFLOOR_1500 / Small-LI vs PROJECTION_ON_ADAPTIVE@3000**: Δ=-0.0762 (retention 0.368)
- **GBT_STDFLOOR_1500 / Small-LI vs ADAPTIVE_LONG@3000**: Δ=-0.0612 (retention 0.420)
- **GBT_STDFLOOR_3000 / Small-HI vs INFONCE_ONLY@3000**: Δ=+0.1970 (retention 20.576)
- **GBT_STDFLOOR_3000 / Small-HI vs EXPERT_ONLY@3000**: Δ=-0.2104 (retention 0.496)
- **GBT_STDFLOOR_3000 / Small-HI vs PROJECTION_ON_ADAPTIVE@3000**: Δ=-0.1846 (retention 0.529)
- **GBT_STDFLOOR_3000 / Small-HI vs ADAPTIVE_LONG@3000**: Δ=-0.1810 (retention 0.534)
- **GBT_STDFLOOR_3000 / SAML-D vs INFONCE_ONLY@3000**: Δ=+0.2343 (retention 1.404)
- **GBT_STDFLOOR_3000 / SAML-D vs EXPERT_ONLY@3000**: Δ=-0.1351 (retention 0.858)
- **GBT_STDFLOOR_3000 / SAML-D vs PROJECTION_ON_ADAPTIVE@3000**: Δ=-0.1300 (retention 0.862)
- **GBT_STDFLOOR_3000 / SAML-D vs ADAPTIVE_LONG@3000**: Δ=-0.0996 (retention 0.891)
- **GBT_STDFLOOR_3000 / Small-LI vs INFONCE_ONLY@3000**: Δ=+0.0403 (retention 5.575)
- **GBT_STDFLOOR_3000 / Small-LI vs EXPERT_ONLY@3000**: Δ=-0.0627 (retention 0.439)
- **GBT_STDFLOOR_3000 / Small-LI vs PROJECTION_ON_ADAPTIVE@3000**: Δ=-0.0715 (retention 0.407)
- **GBT_STDFLOOR_3000 / Small-LI vs ADAPTIVE_LONG@3000**: Δ=-0.0566 (retention 0.464)

Confirmation: no test data loaded/scored; baselines not re-extracted; recovery / failed official full3000 checkpoints never evaluated.

