# Documentation audit — 2026-07-22

Status: **canonical audit** · Documentation/provenance only · No training behavior changed.

This audit ranked findings by scientific risk before the synchronization pass on the same date.

## Critical

1. **Small-HI Multi-GIN+EU reproduction mislabeled.** Registry and several “paper-compatible” docs still centered the TDS-on 100ep legacy run (`tds=True`, edge_dim=8, F1≈0.54) as `thesis_primary` / table-eligible. Formal ports TDS-off 50ep seeds 1–3 (mean F1 **0.660 ± 0.060**) were absent from the registry.
2. **README legacy supervised command stale.** Claimed paper-compatible protocol but omitted `--emlps`, did not require TDS-off, and used 100 epochs instead of the 50ep reproduced protocol.
3. **`paper_comparable` training logger overstated.** `_is_configured_to_reproduce_egressy` only checked legacy+gin+not-testing — not TDS-off / emlps / ports. **Not changed in this pass** (would alter training log semantics); registry eligibility now encodes the correct rule explicitly.

## High

4. **Jul 21–22 contrastive A/B/C/D + corrected reverse + preserve-seed absent from index/registry.** Canonical “current protocol” still read as inherited `--emlps --tds` without reverse correction.
5. **CLI reference gaps:** `--preserve_seed_edges`, `--correct_reverse_edge_features`, `--supervised_head`, `--representation_source` (extraction), and all transaction-node script flags missing.
6. **Temporal vs random mixing risk** for GCPAL txn-node scouts (temporal primary vs random-40 diagnostic).
7. **Threshold protocol mixing** in synthesis docs (paper_argmax vs fixed-0.5 vs val-tuned F1).

## Medium

8. **Batch-size E/F** documented only as preflight; not classified as diagnostic-only in registry.
9. **GCPAL naming** (`B_gcpal`) is convenient but not an exact reproduction — disclaimers exist in run notes but not in README hierarchy.
10. **Failed forensic jobs** (18558352, 18566110) had no index entry; risk of treating them as results.
11. **Jul 21–22 notes** not listed in `notes/README.md`.

## Lower

12. Generated registry/tables vs hand notes: regenerate from scripts only.
13. Sampled relative links in core docs were intact at audit time.
14. Springer GCPAL URL typo risk in older citations (`924` vs `024`).

## Generated vs hand-edited

| Generated (do not hand-edit metrics) | Hand-edited canonical |
|--------------------------------------|------------------------|
| `notes/thesis_experiment_registry.md`, `results/diagnostics/thesis_experiment_registry.{json,csv}` | `README.md`, `notes/README.md`, `notes/cli-reference.md` |
| `notes/thesis_tables_preview.md`, `tables/*` | `notes/results.md`, `notes/datasets.md`, protocol docs |
| Per-run `supervised_*`, `probe_*`, `gcpal_txn_node_*` notes | Interpretation analyses (`morning_jobs_*`, parity audits) |

## Synchronization targets (this pass)

- Concise README + documentation map
- Canonical protocol families + evaluation protocol docs
- CLI reference + automated flag check
- Registry ingest for Jul 21–22 artifacts with correct eligibility
- Notes index refresh
- Failed forensic recorded as failed provenance only

## Synchronization completed (same date)

Implemented in-repo: audit note; README rewrite; `thesis_protocol_families.md`; `evaluation_protocols.md`; CLI sync + `scripts/check_documented_flags.py`; registry builder ingest + regenerate; notes index; tables regenerated. Training code / argparse defaults **unchanged**.
