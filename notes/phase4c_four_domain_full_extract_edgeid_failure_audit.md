# Phase-4C full frozen extract EdgeID failure audit

> Read-only. Twin: `results/diagnostics/phase4c_four_domain_full_extract_edgeid_failure_audit.json`  
> Package: `results/diagnostics/phase4c_four_domain_full_extract_edgeid_failure_audit/`  
> No source edits, no Slurm submit, no re-extract, no temp deletion, exact EdgeID gate not weakened.

## Verdict

**Combination defect:** (1) preflight authorized full eval after bounded subset checks only; (2) production full extract uses NeighborLoader seed-survival incompleteness without AML/PaySim missing-seed reinjection; (3) Phase-4C exact-set join is stricter than the trusted Phase-4B floor policy on the **same** `extract_seed_embeddings_hetero` core.

All **10/10** extract jobs FAILED (~55 min); all **40** probes CANCELLED; finalizer **19746808** wrote `ok=false` (0/40).

Uniform error: `full extract EdgeID set != expected split`.

## 1. Counts from failed jobs

**Unavailable** for missing / extra / duplicate / intersection / coverage on every domain and split.

Failure JSON keys are only `arm, step, target, error, traceback, preserved_prior_domains, ok`.  
`verify_extracted_join` raises without logging set diffs. Staging NPZ write never runs.

### Historical same-loader proxy (not Phase-4C failed-job measurements)

| Domain | Split | source_n | extracted_n | missing≈ | coverage |
|---|---|---:|---:|---:|---:|
| Small-HI | train | 3,248,921 | 3,248,269 | 652 | 0.99980 |
| Small-HI | val | 965,524 | 965,466 | 58 | 0.99994 |
| SAML-D | train | 5,715,293 | 5,027,798 | 687,495 | 0.87971 |
| SAML-D | val | 1,900,105 | 1,653,322 | 246,783 | 0.87012 |
| Small-LI | train | 4,432,934 | 4,431,773 | 1,161 | 0.99974 |
| Small-LI | val | 1,316,442 | 1,316,415 | 27 | 0.99998 |
| PaySim | train | 3,792,821 | 3,792,809 | 12 | 0.999997 |
| PaySim | val | 1,276,276 | 1,276,275 | 1 | 0.999999 |

Sources: Phase-4B objective-ablation extract metas; PaySim from `expert_only_frozen_transfer_samld_paysim/probe_PaySim/coverage.json`.

Phase-4C preflight `requested` counts match these source_n values exactly → expected cohort construction looks correct.

## 2. Cross-checkpoint pattern

- Identical error on all 40 cells / 10 jobs.
- Within each domain, **train fails first** (code order); val not reached for that domain.
- Domains still attempted independently; all four fail per job.
- Recurrence of *specific* missing IDs across checkpoints: **unavailable** from failed artifacts.
- Historical EdgeID hashes are **identical across arms** for the same domain/split → missing set is loader/graph-deterministic, not checkpoint-weight-specific.

## 3. Root cause (source locations)

1. `train_util._extract_seed_embeddings_hetero_impl` — mask keeps seeds present in sampled subgraph; reinjection only for `Small_J`/`Small_Q`.
2. `phase4c_four_domain_frozen_eval/edgeid.verify_extracted_join` — full mode requires exact set equality.
3. `ops.extract_split_r198` — DAG calls with `max_batches=None` → exact gate.
4. Preflight used `max_batches=1` → bounded subset gate only, then `FULL_FROZEN_EVAL_AUTHORIZED`.

Extracted rows = **surviving sampled-subgraph seeds** (post-dedupe), not the full requested split.

## 4. vs trusted historical extractor

| Aspect | Phase-4B / GBT / ladder | Phase-4C DAG |
|---|---|---|
| Core extract | `extract_seed_embeddings_hetero` | same |
| `add_arange_ids` / reverse | same pattern | same |
| `preserve_seed_edges` | False | False |
| Join / coverage | floors + `log_seed_coverage` | **exact set equality** |
| SAML-D | documented defect, floor 0.85 | exact gate → hard fail |

Phase-4C copied the extract core but **did not** copy the trusted join policy.

## 5–6. Preflight false authorization / realized reductions

See `preflight_gap_analysis.md`. Bounded realized~8k is expected for one batch, not a production-defect warning. The unexercised gate is full exact-set equality.

## 7. Is exact full coverage achievable?

**Not** with the current AML/PaySim path (no reinjection).  
**Yes**, if missing-seed reinjection (already implemented for Small_J/Small_Q) is generalized, or an equivalent seed-complete extract path is used — then keep the exact gate.

Do **not** recommend floors as the preferred fix.

## 8. Ranked fixes

See `recommended_fix_and_v2_preflight.json`. Top: generalize missing-seed reinjection; then seed-independent extract; do not misuse `--preserve_seed_edges` as the loader fix.

## 9. v2 preflight (design only)

Representative: **PROJECTION SHORT @ 4000**. Full train/val × 4 domains, exact-set gate, write+reload reusable cells. Do not authorize other checkpoints from bounded batches.

## 10. Temporary artifact reuse

**No.** POOL emb root empty; no staging/NPZ/COMPLETE; failure JSON lack ID arrays.

## Confirmations

No source changed; no job submitted; no extract rerun; no test access; no historical artifact modified; temps not deleted; exact gate not weakened.
