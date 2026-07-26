# PaySim transfer metric inventory (read-only)

Every result below is **AMLWorld → PaySim transfer** (or a PaySim-side control for that transfer protocol).
This inventory does **not** recompute predictions. AUROC/AUPRC are **threshold-independent**; when source JSON nests them under `threshold_0.5`, that is packaging only.

**Constraints honored:** no `sbatch`/`srun`, no GPU, no `.npz` loads, no probe retrain, no historical overwrite.

## Table 1 — Matched post-128 H + logistic (`class_weight=model`, C=1)

| Row | Encoder | Norm | BN | AUROC | AUPRC | F1@0.5 | F1@val-thr | P@100 | P@500 | P@1000 | Source cell |
|-----|---------|------|----|------:|------:|-------:|-----------:|------:|------:|-------:|-------------|
| `1_ports_only_sym_proj_pergraph` | `hi_contrastive_proj_sym_20ep_bestckpt` | per_graph_edge_znorm | frozen_aml_bn_assumed | 0.8643 | 0.1264 | 0.1262 | 0.0883 | 0.760 | 0.544 | 0.354 | `A_legacy_sym_logistic_model.json` |
| `2_tds_off_emlps_no_preserve_pergraph` | `gin_emlps_ports_tds_off_asym_proj_8192neg_queue0_40ep_seed2` | per_graph_edge_znorm | frozen_aml_bn_assumed | 0.9078 | 0.1239 | 0.1274 | 0.1078 | 0.750 | 0.582 | 0.390 | `L_tds_off_logistic_H.json` |
| `3_tds_off_emlps_preserve_pergraph` | `gin_emlps_ports_tds_off_preserve_seed_asym_proj_8192neg_queue0_40ep_seed2` | per_graph_edge_znorm | frozen_aml_bn_assumed | 0.8216 | 0.0728 | 0.0739 | 0.1370 | 0.600 | 0.380 | 0.256 | `L_tds_off_preserve_logistic_H.json` |
| `4a_corrected_no_preserve_pergraph` | `gin_emlps_ports_tds_corrected_asym_proj_8192neg_queue0_40ep_seed2` | per_graph_edge_znorm | frozen_aml_bn_assumed | 0.9201 | 0.1106 | 0.1266 | 0.2053 | 0.400 | 0.322 | 0.263 | `L_corrected_logistic_H.json` |
| `4b_corrected_no_preserve_trainfit` | `gin_emlps_ports_tds_corrected_asym_proj_8192neg_queue0_40ep_seed2` | paysim_train_fit_edge_znorm | none_frozen_eval | 0.8668 | 0.0262 | 0.0629 | 0.0673 | 0.000 | 0.008 | 0.022 | `A_corrected_trainfit_post128_logistic_cw_model.json` |
| `4c_corrected_no_preserve_trainfit_bnrecal` | `gin_emlps_ports_tds_corrected_asym_proj_8192neg_queue0_40ep_seed2` | paysim_train_fit_edge_znorm | target_train_only_running_stats | 0.8781 | 0.0480 | 0.0556 | 0.0873 | 0.050 | 0.034 | 0.040 | `A_corrected_trainfit_bnrecal_post128_logistic_cw_model.json` |
| `5a_dplus_preserve_pergraph` | `gin_emlps_ports_tds_corrected_preserve_seed_asym_proj_8192neg_queue0_40ep_seed2` | per_graph_edge_znorm | none_frozen_eval | 0.8331 | 0.1093 | 0.1424 | 0.1588 | 0.860 | 0.626 | 0.392 | `B_dplus_pergraph_post128_logistic_cw_model.json` |
| `5b_dplus_preserve_trainfit` | `gin_emlps_ports_tds_corrected_preserve_seed_asym_proj_8192neg_queue0_40ep_seed2` | paysim_train_fit_edge_znorm | frozen_aml_bn_from_dplus_final_extract | 0.7046 | 0.0261 | 0.0251 | 0.0192 | 0.000 | 0.110 | 0.176 | `C_dplus_seed2_post128_logistic_model.json` |
| `5c_dplus_preserve_pergraph_bnrecal` | `gin_emlps_ports_tds_corrected_preserve_seed_asym_proj_8192neg_queue0_40ep_seed2` | per_graph_edge_znorm | target_train_only_running_stats | 0.8906 | 0.1466 | 0.1480 | 0.1639 | 0.940 | 0.648 | 0.413 | `B_dplus_pergraph_bnrecal_post128_logistic_cw_model.json` |
| `6a_random_ports6_pergraph` | `random_init_gin_ports6` | per_graph_edge_znorm | n/a_random_init | 0.7298 | 0.0997 | 0.1350 | 0.1433 | 0.750 | 0.558 | 0.397 | `A_legacy_random_logistic_model.json` |
| `6b_random_edge8_pergraph` | `random_init_edge_dim8` | per_graph_edge_znorm | none_frozen_eval | 0.7347 | 0.0824 | 0.0662 | 0.1111 | 0.450 | 0.418 | 0.354 | `random_edge8_pergraph_post128_logistic_cw_model.json` |
| `6c_random_edge8_trainfit` | `random_init_edge_dim8` | paysim_train_fit_edge_znorm | none_frozen_eval | 0.5783 | 0.0261 | 0.0682 | 0.0481 | 0.020 | 0.062 | 0.114 | `random_edge8_trainfit_post128_logistic_cw_model.json` |
| `6d_random_dplus_arch_trainfit_from_final` | `random_init_dplus` | paysim_train_fit_edge_znorm | n/a_random_init | 0.5781 | 0.0261 | 0.0682 | 0.0481 | 0.020 | 0.062 | 0.115 | `C_dplus_random_post128_logistic_model.json` |

### Table 1 notes

- **AUROC/AUPRC** come from source key `threshold_0.5.auroc` / `.auprc` but are not threshold-dependent.
- **Val-thr** uses `validation_selected_threshold` chosen on validation (max-F1) then applied to test (`threshold_val_selected`).
- **ID cohorts differ** across groups — do not treat as paired:
  - `0d22bc07b85cee34…`: 1_ports_only_sym_proj_pergraph, 6a_random_ports6_pergraph
  - `592a102d5a062f6a…`: 2_tds_off_emlps_no_preserve_pergraph, 3_tds_off_emlps_preserve_pergraph
  - `95f1ab51aa08c43f…`: 4a_corrected_no_preserve_pergraph, 4b_corrected_no_preserve_trainfit, 5a_dplus_preserve_pergraph, 5b_dplus_preserve_trainfit, 6b_random_edge8_pergraph, 6c_random_edge8_trainfit, 6d_random_dplus_arch_trainfit_from_final
  - `5223fb2c8d8f6799…`: 4c_corrected_no_preserve_trainfit_bnrecal, 5c_dplus_preserve_pergraph_bnrecal

### Table 1 expanded fields (confusion @0.5, cohort)

| Row | n_test | n_pos | ID hash | thr_val | PPR@0.5 | TP/FP/TN/FN @0.5 | SHA256 |
|-----|-------:|------:|---------|--------:|--------:|-----------------|--------|
| `1_ports_only_sym_proj_pergraph` | 1293523 | 4258 | `0d22bc07b85c` | 0.7660 | 0.0006 | 315/421/1288844/3943 | `d578ab64edf2` |
| `2_tds_off_emlps_no_preserve_pergraph` | 1293522 | 4258 | `592a102d5a06` | 0.6099 | 0.0004 | 306/238/1289026/3952 | `null` |
| `3_tds_off_emlps_preserve_pergraph` | 1293522 | 4258 | `592a102d5a06` | 0.2070 | 0.0003 | 173/251/1289013/4085 | `null` |
| `4a_corrected_no_preserve_pergraph` | 1293522 | 4258 | `95f1ab51aa08` | 0.2046 | 0.0014 | 387/1467/1287797/3871 | `18e06f555aa4` |
| `4b_corrected_no_preserve_trainfit` | 1293522 | 4258 | `95f1ab51aa08` | 0.8749 | 0.0164 | 801/20409/1268855/3457 | `18e06f555aa4` |
| `4c_corrected_no_preserve_trainfit_bnrecal` | 1293522 | 4258 | `5223fb2c8d8f` | 0.8803 | 0.0640 | 2422/80401/1208863/1836 | `18e06f555aa4` |
| `5a_dplus_preserve_pergraph` | 1293522 | 4258 | `95f1ab51aa08` | 0.2435 | 0.0005 | 347/270/1288994/3911 | `a320920141f5` |
| `5b_dplus_preserve_trainfit` | 1293522 | 4258 | `95f1ab51aa08` | 0.6900 | 0.0004 | 60/465/1288799/4198 | `a320920141f5` |
| `5c_dplus_preserve_pergraph_bnrecal` | 1293522 | 4258 | `5223fb2c8d8f` | 0.2290 | 0.0005 | 363/285/1288979/3895 | `a320920141f5` |
| `6a_random_ports6_pergraph` | 1293523 | 4258 | `0d22bc07b85c` | 0.4533 | 0.0005 | 333/342/1288923/3925 | `null` |
| `6b_random_edge8_pergraph` | 1293522 | 4258 | `95f1ab51aa08` | 0.3380 | 0.0003 | 152/179/1289085/4106 | `null` |
| `6c_random_edge8_trainfit` | 1293522 | 4258 | `95f1ab51aa08` | 0.6909 | 0.0012 | 200/1409/1287855/4058 | `null` |
| `6d_random_dplus_arch_trainfit_from_final` | 1293522 | 4258 | `95f1ab51aa08` | 0.6884 | 0.0012 | 200/1406/1287858/4058 | `null` |

## Table 2 — Separate final D+ protocol (pre-3h H+X PaperStyleMLP)

**Not mixed with Table 1.** Primary stack `pre3h_HxX`; train-fit edge z-norm; frozen encoder (except FT secondary).

| Row | AUROC | AUPRC | F1@0.5 | F1@val-thr | P@100 | P@500 | P@1000 | Source |
|-----|------:|------:|-------:|-----------:|------:|------:|-------:|--------|
| seed1 pre3h_HxX | 0.7138 | 0.1393 | 0.0750 | 0.0986 | 1.000 | 0.760 | 0.497 | `role_seed1.json` |
| seed2 pre3h_HxX | 0.6698 | 0.0864 | 0.0205 | 0.0456 | 0.870 | 0.518 | 0.327 | `role_seed2.json` |
| seed3 pre3h_HxX | 0.5494 | 0.1090 | 0.1156 | 0.1224 | 1.000 | 0.716 | 0.459 | `role_seed3.json` |
| 3-seed mean±sd (AUROC/AUPRC/F1) | 0.6443±0.0851 | 0.1116±0.0265 | 0.0704±0.0478 | 0.0889±0.0393 | null | null | null | `paysim_dplus_transfer_final.json:primary_pre3h_HxX` |
| equal-weight ensemble | 0.7046 | 0.1325 | 0.0711 | 0.1112 | 1.000 | 0.744 | 0.474 | `ensemble_pre3h_HxX` |
| X-only control | 0.8517 | 0.0961 | 0.0000 | 0.0481 | null | null | null | `x_only` |
| random-encoder pre3h_HxX | 0.7537 | 0.1396 | 0.0654 | 0.1189 | 1.000 | 0.788 | 0.485 | `role_random_init.json` |
| FT seed2 (secondary) | 0.6004 | 0.0751 | 0.0223 | 0.0480 | 0.930 | 0.488 | 0.288 | `role_ft_seed2.json` |

## Best recorded results (inventory max; not a selection decision)

| Metric | Value | Protocol / row | Source |
|--------|------:|----------------|--------|
| Best AUROC | 0.9201 | `4a_corrected_no_preserve_pergraph` (per_graph_edge_znorm; frozen_aml_bn_assumed) | `L_corrected_logistic_H.json` |
| Best AUPRC | 0.1466 | `5c_dplus_preserve_pergraph_bnrecal` | `B_dplus_pergraph_bnrecal_post128_logistic_cw_model.json` |
| Best F1@0.5 | 0.1480 | `5c_dplus_preserve_pergraph_bnrecal` | `B_dplus_pergraph_bnrecal_post128_logistic_cw_model.json` |
| Best F1@val-thr | 0.2053 | `4a_corrected_no_preserve_pergraph` | `L_corrected_logistic_H.json` |
| Best P@100 | 1.000 | `table2_seed1_pre3h_HxX` | `role_seed1.json` |

Table-1-only bests (post-128 logistic): AUROC `4a_corrected_no_preserve_pergraph`=0.9201; AUPRC `5c_dplus_preserve_pergraph_bnrecal`=0.1466; F1@0.5 `5c_dplus_preserve_pergraph_bnrecal`=0.1480; F1@val `4a_corrected_no_preserve_pergraph`=0.2053; P@100 `5c_dplus_preserve_pergraph_bnrecal`=0.940.

## Missing metrics

- Total null fields across Table 1 required checklist: **6**
- By field: `{'checkpoint_sha256': 6}`

| Field | Rows affected | Recompute needed? |
|-------|---------------|-------------------|
| checkpoint_sha256 | 6 (2_tds_off_emlps_no_preserve_pergraph, 3_tds_off_emlps_preserve_pergraph, 6a_random_ports6_pergraph…) | only if hashing/recording absent SHA from checkpoint file is undesired here; else null |

## Cautious interpretation

- **Strict inductive:** `paysim_train_fit_edge_znorm` + frozen AML BN (e.g. rows `4b`, `5b`, `6c`, Table 2).
- **Target-train BN adaptation:** BN running stats updated on PaySim train only; no labels/gradients/learned-weight updates (`4c`, `5c`). Still uses target-graph unlabeled train traffic.
- **Transductive per-graph z-norm:** test-graph attributes enter test z-norm (`1`, `2`, `3`, `4a`, `5a`, `6a/b`). Higher scores are not inductive claims.
- Table 1 vs Table 2 differ in **representation** (post-128 H vs pre-3h H+X), **probe** (logistic vs MLP), and sometimes **ID hash** — not directly interchangeable.
- Encoders in Table 1/primary Table 2 rows are **frozen**; FT secondary is not. PaySim labels train the **probe** only (except FT). BN recal does **not** use labels.

## Fairest pretrained vs random evidence

Preferred strict-inductive matched pair: **`4b_corrected_no_preserve_trainfit` (AUROC 0.8668) vs `6c_random_edge8_trainfit` (0.5783)** — same edge_dim=8 contract, train-fit norm, frozen BN, post-128 logistic H-only.

Also strong (but transductive norm): `4a` 0.9201 vs `6b` 0.7347; legacy ports-only `1` 0.8643 vs `6a` 0.7298.

## Answers

1. **Best recorded PaySim AUROC:** 0.9201 under `4a_corrected_no_preserve_pergraph` (per_graph_edge_znorm, frozen_aml_bn_assumed, post-128 logistic). Source: `results/diagnostics/paysim_regression_audit/cells/L_corrected_logistic_H.json`.
2. **Best recorded AUPRC:** 0.1466 under `5c_dplus_preserve_pergraph_bnrecal`.
3. **Best F1@0.5:** 0.1480 under `5c_dplus_preserve_pergraph_bnrecal`.
4. **Best validation-threshold F1:** 0.2053 under `4a_corrected_no_preserve_pergraph`.
5. **Fairest pretrained>random evidence:** inductive matched pair `4b` vs `6c` (see above).
6. **Unavailable without reevaluation / re-recording:** primarily missing `checkpoint_sha256` on some random/lineage cells; optional explicit `test_n_positives` field on older cells (usually recoverable as tp+fn without recompute). No missing AUROC/AUPRC/F1/P@k for Table 1 cw=model cells inspected.
7. **Confirm:** no jobs submitted, no models loaded, no predictions recomputed, and no historical artifacts overwritten.

## Artifact

- JSON: `results/diagnostics/paysim_transfer_metric_inventory.json`
- MD: `notes/paysim_transfer_metric_inventory.md`

