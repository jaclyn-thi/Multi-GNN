# Contrastive objective resource scout (pre-3h, seed2)

table_group=`contrastive_objective_resource_scout` · diagnostic_only · recommendation=`replicate_edge_drop_only`

| Variant | A AUPRC | B AUPRC | D AUPRC | A P@100 | A R@P≥0.80 | ΔA AUPRC | ΔB AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.2598 | 0.2725 | 0.5111 | 0.7900 | 0.0776 | — | — |
| large_bs | 0.1386 | 0.1116 | 0.3077 | 0.5100 | — | -0.1211 | -0.1610 |
| edge_drop | 0.2870 | 0.3251 | 0.4690 | 0.7900 | 0.0478 | 0.0273 | 0.0526 |
