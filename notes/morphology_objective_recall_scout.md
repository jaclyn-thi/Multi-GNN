# Morphology-objective recall scout

**Thesis role:** diagnostic_or_scout · **validation_status:** diagnostic_only · **table_eligible:** false · **table_group:** morphology_objective_recall_scout

Primary representation: **pre_embedding_3h**. Post-128 is diagnostic only.
SSL: morphology expert **regression** only (no M2 contrast, no TF soft positives, no tier2/betweenness, **no labels**).

## Verdict / recommendation

- Recommendation: **`scale_morph_only_skip_combo`**
- Recall improved at acceptable precision: `['degflow']`
- Precision-collapse variants (P@100 < 50% of baseline): `['clustering']`
- Best pre-3h embedding-only (A AUPRC): **degflow** (0.2828)
- Best pre-3h + raw (B AUPRC): **degflow** (0.3719)
- Best final D stack AUPRC: **degflow** (0.4740)

## Pre-3h primary metrics

| Variant | Arm | AUROC | AUPRC | F1 | P@100 | R@500 | R@1000 | R@P≥0.90 | R@P≥0.80 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | A_embedding | 0.9608 | 0.1888 | 0.2504 | 0.6600 | 0.1421 | 0.2061 | 0.0006 | 0.0006 |
| baseline | B_embedding_raw | 0.9617 | 0.2113 | 0.2872 | 0.6700 | 0.1558 | 0.2291 | 0.0006 | 0.0006 |
| baseline | D_embedding_raw_temporal_flow | 0.9786 | 0.4337 | 0.4482 | 0.9400 | 0.2495 | 0.3780 | 0.1291 | 0.2514 |
| degflow | A_embedding | 0.9331 | 0.2828 | 0.3175 | 0.8500 | 0.2030 | 0.2762 | 0.0168 | 0.1210 |
| degflow | B_embedding_raw | 0.9461 | 0.3719 | 0.0307 | 0.7400 | 0.2154 | 0.3575 | 0.0279 | 0.0354 |
| degflow | D_embedding_raw_temporal_flow | 0.9580 | 0.4740 | 0.0468 | 0.8200 | 0.2359 | 0.4202 | 0.0304 | 0.0571 |
| clustering | A_embedding | 0.9487 | 0.0861 | 0.1618 | 0.1400 | 0.0677 | 0.1210 | 0.0006 | 0.0006 |
| clustering | B_embedding_raw | 0.9605 | 0.1494 | 0.0825 | 0.3400 | 0.1024 | 0.1775 | 0.0031 | 0.0031 |
| clustering | D_embedding_raw_temporal_flow | 0.9759 | 0.2696 | 0.2015 | 0.6000 | 0.1359 | 0.2582 | 0.0006 | 0.0006 |
| degflow_tfreg | A_embedding | 0.9295 | 0.1275 | 0.2030 | 0.4600 | 0.1142 | 0.1719 | — | — |
| degflow_tfreg | B_embedding_raw | 0.9380 | 0.1819 | 0.1820 | 0.5900 | 0.1440 | 0.2154 | — | — |
| degflow_tfreg | D_embedding_raw_temporal_flow | 0.9621 | 0.3134 | 0.3878 | 0.7100 | 0.1912 | 0.3097 | — | — |

## Baseline re-probe

- Run: `hi_contrastive_gin_emlps_tds_proj_asym_8192neg_queue0_accum4_20ep`
- Pre-3h A AUPRC: 0.1888
- Pre-3h B AUPRC: 0.2113

## Notes

- ARM 1 (`degflow`): `--morph_expert --morph_targets local+global --morph_flow_balance --morph_target_groups degree_fan,flow_balance --morph_expert_weight 1.0`
- ARM 2 (`clustering`): `--morph_expert --morph_targets local+global --morph_local_subset clustering --morph_expert_weight 1.0` (matches best-known 11-dim clustering+proj style; excludes triangles)
- Optional ARM 3 (`degflow_tfreg`): InfoNCE + λ_morph=0.05 morph MSE + λ_tf=0.05 TF Huber regression
- Do not insert into main thesis tables yet.

