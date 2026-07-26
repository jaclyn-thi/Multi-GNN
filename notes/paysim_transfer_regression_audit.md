# PaySim transfer regression audit

No encoder training or fine-tuning. Diagnostic extraction/evaluation only. Historical `embeddings/paysim/*` were not overwritten.

## Verdict (short)

1. **Old ~0.866 is reproducible** at thr0.5 (**0.8643** vs historical **0.8642**); matched random **0.7298** vs **0.7301**.
2. Cohort is the **same temporal PaySim protocol** with **off-by-one / feature-config ID-hash drift** (ports-only test n=1293523; D+/TDS n=1293522).
3. Old result is **preliminary but defensible** (PARTIAL/transductive edge z-norm), not an inductive D+ claim.
4. **Encoder/protocol is the major regression** under matched legacy logistic: **0.8643 → 0.7046** (Δ -0.1597).
5. Evaluator change is **secondary** for the old encoder (logistic → MLP H: 0.8643 → 0.8403).
6. Clearest matched-lineage reducer: **`preserve_seed_edges`** (tds_off 0.9078 → preserve 0.8216). Corrected TDS *without* preserve stays high (0.9201).
7. Old encoder does **not** beat matched random under current MLP H (0.8403 vs 0.9200).
8. D+ **does** beat matched random under legacy logistic post-128 (0.7046 vs 0.5781).
9. Feature-schema placeholders are **plausible contributors**, not a sole demonstrated cause (masking not run).
10. Prefer a **matched-protocol D+ re-extract** before any retrain; smallest training candidate is **D+ without preserve_seed** (one seed).

## Stage 0 — provenance

| Item | Old (Jun 2026) | Current D+ (Jul 2026) |
|------|----------------|------------------------|
| Jobs | pretrained 16036046, random 16043535 | seeds 18855316–18, random+FT 18855319 |
| Checkpoint | `checkpoint_hi_contrastive_proj_sym_20ep_bestckpt.tar` ep20, sha256 `d578ab64…` | `…tds_corrected_preserve_seed…seed2.tar` ep40 |
| Flags | reverse_mp+ego+ports; **no** tds/emlps/correct_reverse | ports+tds+emlps+correct_reverse+preserve_seed |
| edge_dim | 6 | 8 |
| Representation | post-128 | primary **pre-3h H+X** (also post-128 reported) |
| Edge z-norm | per-graph (test attrs in test z-norm) | **train_fit_edge_znorm** (inductive) |
| Probe | logistic H-only, `class_weight=model`, C=1, seed=1 | PaperStyleMLP, val-AUPRC epoch, val-F1 thr |

Full Stage-0 JSON: [`results/diagnostics/paysim_regression_audit/stage0_provenance.json`](../results/diagnostics/paysim_regression_audit/stage0_provenance.json)

## Stage 1 — reproduce old result

| Cell | Job | AUROC@0.5 | AUPRC | F1@0.5 | F1@val-thr |
|------|-----|----------:|------:|-------:|-----------:|
| A pretrained logistic | 18873221 | 0.8643 | 0.1264 | 0.1262 | 0.0883 |
| A random logistic | 18873388 | 0.7298 | 0.0997 | 0.1350 | 0.1433 |
| A pretrained logistic unweighted (diagnostic) | 18873221 | 0.9064 | 0.1483 | 0.0393 | 0.0850 |
| Historical thr0.5 JSON | 16036046 / 16043535 | 0.8642 / 0.7301 | — | — | — |

Artifacts:
- cells: [`A_legacy_sym_logistic_model.json`](../results/diagnostics/paysim_regression_audit/cells/A_legacy_sym_logistic_model.json), [`A_legacy_random_logistic_model.json`](../results/diagnostics/paysim_regression_audit/cells/A_legacy_random_logistic_model.json)
- embeds: [`embeddings/paysim_regression_audit/legacy_sym_post128/`](../embeddings/paysim_regression_audit/legacy_sym_post128/), [`…/legacy_random_ports6_post128/`](../embeddings/paysim_regression_audit/legacy_random_ports6_post128/)
- historical probes (`.npz` missing): [`embeddings/paysim/hi_contrastive_proj_sym_20ep_bestckpt/probe_results_cw_model_thr0.5.json`](../embeddings/paysim/hi_contrastive_proj_sym_20ep_bestckpt/probe_results_cw_model_thr0.5.json)

## Stage 2 — 2×2 crossover (AUROC primary)

| | Old H-only logistic | Current MLP |
|--|--------------------:|------------:|
| **Old sym+proj ports encoder** | A: **0.8643** | B H: 0.8403 · HxX: 0.8339 |
| **Old matched random** | A: 0.7298 | B H: 0.9200 |
| **D+ seed2** | C post128: **0.7046** · pre3h: 0.5181 | D post128 H: 0.7866 · pre3h H: 0.7558 · pre3h HxX: 0.6698 |
| **D+ matched random** | C post128: 0.5781 · pre3h: 0.6392 | (see [`role_random_init.json`](../results/diagnostics/paysim_dplus_transfer_final/role_random_init.json)) |

Cell JSONs: [`results/diagnostics/paysim_regression_audit/cells/`](../results/diagnostics/paysim_regression_audit/cells/)  
Current D+ evaluator stacks: [`role_seed2.json`](../results/diagnostics/paysim_dplus_transfer_final/role_seed2.json) · final writeup [`notes/paysim_dplus_transfer_final.md`](paysim_dplus_transfer_final.md)

## Stage 3 — lineage (legacy logistic H-only, post-128)

Fixed downstream protocol chosen before viewing test metrics: frozen encoder, H-only logistic, validation-selected threshold also reported.

| Checkpoint (seed2) | Job | Extract protocol | AUROC@0.5 | AUPRC |
|--------------------|-----|------------------|---------:|------:|
| 1. old sym ports-only | 18873221 | per-graph z-norm, edge_dim=6 | 0.8643 | 0.1264 |
| 2. TDS-off + emlps | 18873389 | per-graph, edge_dim=6 | 0.9078 | 0.1239 |
| 3. TDS-off + preserve | 18873391 | per-graph, edge_dim=6 | 0.8216 | 0.0728 |
| 4. corrected TDS, no preserve | 18873390 | per-graph, edge_dim=8, correct_reverse=True | 0.9201 | 0.1106 |
| 5. D+ corrected+preserve | 18869330 (CPU on existing embeds) | **train_fit** z-norm (prior final extract) | 0.7046 | 0.0261 |

**Interpretation:** Under matched lineage extracts, **preserve_seed** is the main drop; **corrected TDS alone is not**. Row 5 is confounded by extract normalization — do not treat 0.70 as a pure “preserve vs corrected” delta without a matched re-extract.

Embeds: [`embeddings/paysim_regression_audit/lineage_*/`](../embeddings/paysim_regression_audit/)

## Stage 4 — feature-contract audit

From `dataset_specs.PAYSIM_FEATURE_MAPPING`:

| AMLWorld channel | PaySim mapping | Alignment |
|------------------|----------------|-----------|
| Timestamp | step × 3600 | aligned (synthetic hours) |
| Amount Received | amount | aligned scalar |
| Received Currency | PaySim **type** integer code | **placeholder** |
| Payment Format | same type code | **placeholder** |
| ports / TDS | recomputed on PaySim graph | structural, not transferred AML stats |
| Transaction type (native PaySim) | folded into currency/payment slots | no dedicated AML channel |

Masking sensitivity diagnostics: **not run** in this pass (would be OOD zeroing; deferred to keep login/GPU load minimal).

## Decision answers

1. **Reproducible?** Yes — thr0.5 AUROC 0.8643 (hist 0.8642).
2. **Same cohort?** Same temporal protocol; off-by-one n and feature-config ID hashes differ.
3. **Validity?** Preliminary but defensible; PARTIAL/transductive z-norm; not D+.
4. **Encoder share?** Major (~−0.16 AUROC under matched legacy logistic).
5. **Evaluator share?** Secondary for old encoder AUROC; stack/learner change matters for the D+ primary claim.
6. **Worst D+ component?** `preserve_seed_edges` under matched lineage; full D+ row confounded by train_fit extract.
7. **Old > random under current eval?** **No** (MLP H).
8. **D+ > random under legacy eval?** **Yes** (post-128); **no** for pre-3h H.
9. **Schema mismatch cause?** Plausible contributor; not sole demonstrated cause.
10. **Smallest follow-up?** Matched-protocol D+ re-extract first; if training, one seed-2 **without preserve_seed**.

## Jobs and logs

| Role | JobID | Log |
|------|------:|-----|
| Stage2C logistic (CPU) | 18869330 | `logs/paysim_reg_audit_stage2c_18869330.out` |
| legacy_sym | 18873221 | `logs/paysim_reg_audit_legacy_sym_18873221.out` |
| legacy_random | 18873388 | `logs/paysim_reg_audit_legacy_rand_18873388.out` |
| lineage_tds_off | 18873389 | `logs/paysim_reg_audit_lin_toff_18873389.out` |
| lineage_corrected | 18873390 | `logs/paysim_reg_audit_lin_corr_18873390.out` |
| lineage_tds_off_preserve | 18873391 | `logs/paysim_reg_audit_lin_toffp_18873391.out` |

Submission record: [`results/diagnostics/paysim_regression_audit/submission.json`](../results/diagnostics/paysim_regression_audit/submission.json)

## Canonical outputs

- [`notes/paysim_transfer_regression_audit.md`](paysim_transfer_regression_audit.md) (this file)
- [`results/diagnostics/paysim_transfer_regression_audit.json`](../results/diagnostics/paysim_transfer_regression_audit.json)
