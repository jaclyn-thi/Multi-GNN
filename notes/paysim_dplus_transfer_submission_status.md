# PaySim D+ frozen transfer — submission status

**Date:** 2026-07-25  
**Status:** smoke PASSED; full A/B/C/D + aggregate submitted. Final metrics pending job completion.

Companion submission record: `results/diagnostics/paysim_dplus_transfer_final/submission.json`

## Corrected smoke (gate)

| Field | Value |
|-------|-------|
| Smoke job | **18854060** |
| Prior failed smoke | 18851404 (technical OK; bad projection only) |
| Gate | **PASSED** |
| Projected full-role hours | **0.74** (fits 6h) |
| Wall | 1245.5 s |

### Projection breakdown (seconds)

| Term | Value |
|------|------:|
| data_loading_csv | 42.8 |
| ports_construction | 843.3 |
| tds_construction | 344.7 |
| other_graph_prep | 2.8 |
| model_construct | 0.57 |
| checkpoint_load | 0.01 |
| paysim_x_build | 9.5 |
| **one_time_setup** | **1243.7** |
| dual extract (9 smoke batches) | 0.91 |
| extract_sec_per_batch | 0.101 |
| projected_pre3h_extract (1554 batches) | 157.4 |
| projected_post128_extract | 0.0 (same dual forward) |
| projected_downstream_eval | 49.3 |
| margin = max(1200, 0.25×core) | 1200.0 |
| **est_total** | **2650.3 s ≈ 0.74 h** |

Formula: `total = one_time_setup + projected_pre3h_extract + projected_post128_extract + projected_downstream_eval + margin`  
One-time setup is **never** multiplied by batch count.

## Submitted jobs (≤4 concurrent GPU)

| Role | Class | Job ID | Checkpoint |
|------|-------|-------:|------------|
| A seed1 | PRIMARY | **18855316** | `checkpoint_edge_dplus_corrected_preserve_40ep_seed1_final.tar` sha256 `7bc393f0…` |
| B seed2 (+X-only) | PRIMARY | **18855317** | `…8192neg_queue0_40ep_seed2.tar` sha256 `a3209201…` |
| C seed3 | PRIMARY | **18855318** | `checkpoint_edge_dplus_corrected_preserve_40ep_seed3_final.tar` sha256 `c8f95e98…` |
| D random + FT seed2 | CONTROL + SECONDARY | **18855319** | FT: `dplus_partial_finetune_hxxtf_seed2/checkpoint_best_val_auprc.tar` sha256 `5ed9503f…`; random = D+ arch seed 2 |
| Aggregate | post-hoc | **18855320** | afterok dependency on A–D |

Embeddings root: `embeddings/paysim_dplus_transfer_final/{dplus_seed1,dplus_seed2,dplus_seed3,random_init_dplus,ft_encoder_seed2}/`  
Legacy `embeddings/paysim/*` not written. No additional experiments beyond these four GPU roles.

## Claim language (locked; metrics TBD after aggregate)

PRIMARY encoders: AMLWorld self-supervised D+ weights learned **without AML labels**, frozen on PaySim. PaySim labels train **only** the downstream MLP. SECONDARY FT encoder had AML-supervised updates to its final encoder block before freeze; excluded from frozen three-seed mean.

Final thesis note/JSON (`notes/paysim_dplus_transfer_final.md`, `results/diagnostics/paysim_dplus_transfer_final.json`) will be written by job 18855320 after A–D complete.
