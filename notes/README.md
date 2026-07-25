# Documentation index (`notes/`)

Navigation hub. Prefer purpose sections below over scanning the folder alphabetically.

**Status legend**

| Tag | Meaning |
|-----|---------|
| **canonical** | Authoritative reference; keep current. |
| **living** | Design/planning doc, updated as the stack evolves. |
| **current results** | Latest-protocol numbers + interpretation. |
| **run log** | Single-experiment / dated provenance; metrics frozen historically. |
| **diagnostic** | Engineering or protocol diagnostic; not main-table. |
| **preliminary** | Incomplete horizon / early scout. |
| **failed** | Job failed; no scientific conclusion. |
| **superseded** | Kept for history; do not use for current decisions. |

> **Comparability:** temporal vs random-40, and argmax vs fixed-0.5 vs val-tuned F1, must stay labeled. See [`evaluation_protocols.md`](evaluation_protocols.md).

> **Generated files:** `thesis_experiment_registry.md`, registry JSON/CSV, and `thesis_tables_preview.md` are produced by scripts — regenerate; do not hand-edit metrics.

---

## Getting started

| I want to… | File |
|------------|------|
| Onboard / run code | [`../README.md`](../README.md) |
| Prepare datasets | [`datasets.md`](datasets.md) |
| Look up a CLI flag | [`cli-reference.md`](cli-reference.md) |
| Cluster / Slurm paths | [`engaging-cluster-config.md`](engaging-cluster-config.md) |
| Documentation sync audit (2026-07-22) | [`documentation_audit_2026-07-22.md`](documentation_audit_2026-07-22.md) |

## Canonical protocols

| File | Purpose | Status |
|------|---------|--------|
| [`thesis_protocol_families.md`](thesis_protocol_families.md) | Edge-centric SSL A–D, batch E/F, supervised parity, txn-node GCPAL-inspired | canonical |
| [`evaluation_protocols.md`](evaluation_protocols.md) | Paper_argmax vs temporal frozen vs random-40 vs GCPAL labeling | canonical |
| [`contrastive-learning-plan.md`](contrastive-learning-plan.md) | GFM framing + design history | living |
| [`datasets.md`](datasets.md) | IBM / PaySim / SAML-D | canonical |
| [`cli-reference.md`](cli-reference.md) | Flags by owning command | canonical |
| [`morphology-reference.md`](morphology-reference.md) | Morphology SSL flags | canonical |
| [`knn-precompute-reference.md`](knn-precompute-reference.md) | Feature-KNN caches | canonical |

## Current thesis results

| File | Status |
|------|--------|
| [`current_protocol_recent_runs_summary.md`](current_protocol_recent_runs_summary.md) | current results (meta-index) |
| [`results.md`](results.md) | canonical (+ history) |
| [`strongest_runs_master_comparison.md`](strongest_runs_master_comparison.md) | current results |
| [`thesis_experiment_registry.md`](thesis_experiment_registry.md) | **generated** registry view |
| [`thesis_tables_preview.md`](thesis_tables_preview.md) | **generated** tables |

## Paper / reference reproductions

| File | Notes | Status |
|------|-------|--------|
| [`eval_small_hi_legacy_supervised_gin_emlps_ports_50ep_seeds1-3_formal_aggregate.md`](eval_small_hi_legacy_supervised_gin_emlps_ports_50ep_seeds1-3_formal_aggregate.md) | **Canonical** Multi-GIN+EU ports TDS-off aggregate | current results |
| [`supervised_Small-HI_small_hi_legacy_supervised_gin_emlps_ports_50ep_seed{1,2,3}_summary.md`](supervised_Small-HI_small_hi_legacy_supervised_gin_emlps_ports_50ep_seed1_summary.md) | Formal seeds | run log |
| [`multignn_supervised_parity_audit.md`](multignn_supervised_parity_audit.md) | Parity / reverse-feature audit | diagnostic |
| [`eval_small_hi_legacy_supervised_gin_emlps_tds_100ep_seed1.md`](eval_small_hi_legacy_supervised_gin_emlps_tds_100ep_seed1.md) | **Not paper-compatible** (TDS-on) | superseded for paper table |
| [`supervised_Small-LI_small_li_legacy_supervised_gin_emlps_tds_100ep_seed1_summary.md`](supervised_Small-LI_small_li_legacy_supervised_gin_emlps_tds_100ep_seed1_summary.md) | Small-LI legacy formal | current results |

## GCPAL diagnostics / reimplementation

| File | Status | Caveat |
|------|--------|--------|
| [`thesis_protocol_families.md`](thesis_protocol_families.md#transaction-node-gcpal-inspired-path) | canonical | **NOT exact reproduction** |
| [`gcpal_txn_node_extraction_scope_audit.md`](gcpal_txn_node_extraction_scope_audit.md) | canonical audit | Graph scope + extraction modes |
| [`gcpal_txn_node_canonical_reextraction.md`](gcpal_txn_node_canonical_reextraction.md) | candidate canonical re-extract | expanding-window + sensitivity + random-40; no GNN retrain |
| [`gcpal_txn_node_posagg_ablation.md`](gcpal_txn_node_posagg_ablation.md) | diagnostic | B/C/D positive aggregation; val selects D SupCon |
| [`gcpal_challenge_fullstack_eval.md`](gcpal_challenge_fullstack_eval.md) | diagnostic / challenge | No GNN train; temporal primary + reconstructed random 40/60; gate **PARTIAL**; job 18678029 |
| [`edge_dplus_neighbor_positive_preflight.md`](edge_dplus_neighbor_positive_preflight.md) | preflight | Neighbor-positive transfer into D+; matched identity required |
| [`edge_dplus_neighbor_positive_smoke.md`](edge_dplus_neighbor_positive_smoke.md) | smoke | Job 18719182 passed; D+-matched epoch budget |
| [`edge_dplus_neighbor_positive_10ep_seed2.md`](edge_dplus_neighbor_positive_10ep_seed2.md) | scout | Val gate job 18787415; neighbor does not beat identity; no 40ep |
| [`final_dplus_experiment_preflight.md`](final_dplus_experiment_preflight.md) | preflight | Locked D+ provenance; >40 UNRESOLVED; seed1/3 + FT templates |
| [`gcpal_txn_node_poscomplete_replay_audit.md`](gcpal_txn_node_poscomplete_replay_audit.md) | diagnostic | Original 5ep vs replay ep5 |
| [`gcpal_positive_set_audit.md`](gcpal_positive_set_audit.md) | diagnostic | coverage audit, not performance |
| [`gcpal_txn_node_smoke.md`](gcpal_txn_node_smoke.md) / poscomplete smoke | diagnostic | |
| [`gcpal_txn_node_scout_{control,gcpal}_5ep_seed2.md`](gcpal_txn_node_scout_control_5ep_seed2.md) | preliminary | ordinary batching |
| [`gcpal_txn_node_poscomplete_scout_A_identity_5ep_seed2.md`](gcpal_txn_node_poscomplete_scout_A_identity_5ep_seed2.md) | preliminary / **noncanonical extraction** | not table-eligible |
| [`gcpal_txn_node_poscomplete_scout_B_gcpal_5ep_seed2.md`](gcpal_txn_node_poscomplete_scout_B_gcpal_5ep_seed2.md) | preliminary / **noncanonical extraction** | not table-eligible |
| [`gcpal_txn_node_poscomplete_scout_{A,B}_*_20ep_seed2.md`](gcpal_txn_node_poscomplete_scout_B_gcpal_20ep_seed2.md) | replay-extension / legacy chunked encode | not table-eligible (B beats A under shared legacy-chunked frozen extract) |
| Forensic eval-protocol audit | **failed** | jobs 18558352 / 18566110 — no metric conclusion |

## Engineering audits (Jul 21–22 highlight)

| File | Status |
|------|--------|
| [`documentation_audit_2026-07-22.md`](documentation_audit_2026-07-22.md) | canonical audit |
| [`morning_jobs_2026-07-22_analysis.md`](morning_jobs_2026-07-22_analysis.md) | analysis |
| [`probe_feature_ablation_current_protocol_gin_40ep_seed2_tds_{off,off_preserve_seed,corrected,corrected_preserve_seed}.md`](probe_feature_ablation_current_protocol_gin_40ep_seed2_tds_corrected_preserve_seed.md) | A/B/C/D probes |
| [`probe_feature_ablation_gin_emlps_ports_tds_corrected_preserve_seed_asym_proj_allneg_bs*_10ep_seed2.md`](probe_feature_ablation_gin_emlps_ports_tds_corrected_preserve_seed_asym_proj_allneg_bs8192_accum4_10ep_seed2.md) | batch E/F diagnostic |
| Contrastive / reverse / seed-retention audits (`*audit*.md`, `seed_retention_*`) | diagnostic |

## Per-run provenance

Auto-generated probe / supervised notes under `notes/` and `notes/experiments/` stay as written. Index + registry supply status/caveats; **do not rewrite historical metrics** to match later conclusions.

Older ablation logs: [`experiments/ablation-runs/`](experiments/ablation-runs/), [`results-archive.md`](results-archive.md).

## Superseded / archive

| File | Status |
|------|--------|
| [`results-archive.md`](results-archive.md) | historical ablations |
| TDS-on Small-HI “paper-compatible” claims in older text | superseded — use ports TDS-off aggregate |
| Inherited TDS-on reverse semantics as “current best SSL” | superseded for semantic validity by corrected+preserve |

## Reference (stable runbooks)

| File | Purpose |
|------|---------|
| [`morphology-metrics-plan.md`](morphology-metrics-plan.md) | Morphology metric plan |
| [`downstream-eval-plan.md`](downstream-eval-plan.md) | PaySim / SAML-D / typology |
| [`lit-review-index.md`](lit-review-index.md) | Paper → implementation map |
| [`../morphology/IDS.md`](../morphology/IDS.md) | EdgeID join semantics |
