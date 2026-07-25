# Temporal-flow aux scout — recall-oriented metrics

**Status:** CPU backfill jobs submitted (no GNN/SSL training, no GPU, no embedding regeneration).

| Job | ID | Scope |
|---|---|---|
| recall_backfill | 18179701 | tf_reg_w0.10 / tf_reg_w0.05 (post+pre) |
| recall_backfill | 18179702 | tf_bins5 / tf_bins10 (post+pre) |
| recall_backfill | 18179703 | baseline post-128 + validated TF HI/LI |
| recall_summary | 18179704 | write tables after backfill |

Original probe JSONs are **not** overwritten. Enriched outputs land in `results/diagnostics/enriched/`.

This page will be replaced when job 18179704 completes.
