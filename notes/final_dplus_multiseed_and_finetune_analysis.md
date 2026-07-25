# Final D+ multiseed + fine-tune analysis

## Thesis-result hierarchy (locked)

1. **PRIMARY:** Self-supervised contrastive encoder evaluated using a supervised downstream classifier, with the encoder frozen (pre-3h H+X+TF MLP).
2. **SECONDARY:** SSL-pretrained D+ with supervised partial fine-tuning (seed 2).

## Provenance

### Training (encoder / FT) jobs

| Encoder | Train job | Checkpoint epoch | sha256 (full) |
|---------|-----------|------------------|---------------|
| seed 1 | 18801429 | 34 | `7bc393f02e552063524671837974991294423ab902786a898e3128489f68afb7` |
| seed 2 | 18514684 | 40 | `a320920141f585c5825cbd63ce760a845fb434a9b162d4c87270dc72b0442b87` |
| seed 3 | 18802579 | 29 | `c8f95e982e1e46f83cdcd0adc4533ddac6b996030669fee0b961def9a868e36b` |
| partial FT seed2 | 18801435 | 18 | `5ed9503f3c3def088ec3913e4e3021d35b03c69a83b17d3d66d1ee7347a15a0b` |

### Evaluation jobs (this locked stage; no GNN retrain)

| Role | Eval job | State | Exit | Start → End (Elapse) | Notes |
|------|----------|-------|------|----------------------|-------|
| Frozen seed1 extract+H+X+TF MLP | **18839150** | COMPLETED | 0:0 | 2026-07-25T07:12:19 → 07:27:28 (00:15:09) | wrote `frozen_dplus_hxxtf_mlp_seed1.json` |
| Frozen seed3 extract+H+X+TF MLP | **18839151** | COMPLETED | 0:0 | 2026-07-25T07:12:19 → 07:26:55 (00:14:36) | wrote `frozen_dplus_hxxtf_mlp_seed3.json` |
| Partial-FT locked test (no refit) | **18839152** | COMPLETED | 0:0 | 2026-07-25T07:12:19 → 07:24:49 (00:12:30) | wrote `dplus_partial_finetune_seed2_locked_test.json` |
| Aggregate analysis | **18839153** | COMPLETED | 0:0 | 2026-07-25T07:28:18 → 07:28:19 (00:00:01) | afterok on 50/51/52 |
| Frozen seed2 (reused) | **18678029** | historical | — | — | metrics reused; protocol REUSE |

Submit helper: `slurm/submit_final_dplus_eval_jobs.sh` → `logs/final_dplus_eval_submit_20260725_071210.txt`.

### Coverage denominators (test)

Do **not** read seed-2 `coverage: 1.0` in the reuse report as raw-split completeness.

| Denominator | Meaning | Seed1 | Seed2 (18678029) | Seed3 | FT online |
|-------------|---------|------:|-----------------:|------:|----------:|
| Raw temporal test split | calendar-day test IDs | 863900 | 863900 | 863900 | 863900 |
| Expected extractable seed-edge cohort | `expected_seed_edge_ids` / loader seed edges (= raw split here) | 863900 | 863900 | 863900 | 863900 |
| Actual evaluated rows | rows with H (or online FT scores) used for metrics | **863054** | **863054** | **863058** | **863055** |
| Raw-split / seed-edge coverage | actual / 863900 | 0.99902072 | **0.99902072** (same; reuse JSON wrongly labeled 1.0) | 0.99902535 | 0.99902188 |

Seed-edge extract logs (jobs 18839150/51): covered 863054 or 863058 / 863900 seed edges (99.90%); misses are loader-pass gaps under `--preserve_seed_edges` + neighbor sampling, not label filtering.

Frozen seeds differ by ≤157 test IDs (all **non-positive**). All **1611** positive test transactions are present in every frozen seed evaluation (`pos 1∩2∩3 = 1611`).

### Matched-cohort sensitivity (IDs only; no score arrays saved)

Prediction scores were **not** persisted by the eval scripts, so F1/AUPRC cannot be recomputed on intersections without refitting or re-inferring (forbidden here). Structural intersections from embedding `edge_id` arrays:

- Frozen 1∩2∩3 test IDs: **862849** (of ~863054–863058)
- Frozen seed2 ∩ FT seed2: **not computable** from saved FT artifacts (FT JSON stores only `edge_id_sum` / sha256 of IDs, not the ID list or scores)

**Common-cohort ΔF1 / ΔAUPRC vs primary:** **N/A — scores unavailable** (no evidence of >0.001 drift; also no evidence against).

## PRIMARY — frozen D+ multiseed (H+X+TF MLP)

| Seed | val AUPRC | val F1 | test AUROC | test AUPRC | F1@0.5 | F1@val-thr | P@100 | P@500 | P@1000 |
|-----:|----------:|-------:|-----------:|-----------:|-------:|-----------:|------:|------:|-------:|
| 1 | 0.5193 | 0.5519 | 0.9882 | 0.6567 | 0.6192 | 0.5940 | 0.960 | 0.920 | 0.837 |
| 2 | 0.5500 | 0.5751 | 0.9881 | 0.6742 | 0.6559 | 0.6118 | 0.990 | 0.934 | 0.851 |
| 3 | 0.5410 | 0.5730 | 0.9876 | 0.6599 | 0.6427 | 0.5824 | 0.950 | 0.952 | 0.861 |
| **mean±sd** | 0.5368 ± 0.0158 | 0.5667 ± 0.0129 | 0.9880 ± 0.0003 | 0.6636 ± 0.0093 | **0.6393 ± 0.0186** | 0.5961 ± 0.0148 | 0.9667 ± 0.0208 | 0.9353 ± 0.0160 | 0.8497 ± 0.0121 |

### Recommended primary claim

A self-supervised contrastive Multi-GIN encoder (D+: corrected reverse-edge semantics and preserve_seed_edges), evaluated with the encoder frozen and a supervised downstream MLP on pre-3h H+X+TF under a temporal split, achieves test F1@0.5 of 0.639 ± 0.019 over three encoder seeds.

## SECONDARY — partial fine-tune seed 2

- Stored val AUPRC @ ep 18: **0.5996**
- Test AUPRC: **0.7009**
- Test F1@0.5: **0.6971**
- Test F1@val-thr (0.46): **0.6934**
- vs frozen seed2 (val 0.550 / test AUPRC 0.674 / F1@0.5 0.656): ΔvalA=+0.0496, ΔtestA=+0.0269, ΔF1=+0.0411

As a secondary sensitivity analysis, SSL-pretrained D+ with supervised partial fine-tuning of the final GNN block reaches test F1@0.5 of 0.697 on seed 2; AML labels update both the classifier and the unfrozen encoder block.

## Cautious published comparisons (fixed-0.5 F1)

- Published Multi-GIN+EU: 0.6479 ± 0.0122
- Published Multi-PNA+EU: 0.6816 ± 0.0265
- Our supervised Multi-GIN+EU: 0.660 ± 0.060
- Frozen D+ mean: **0.6393** (does not exceed Multi-GIN+EU mean)
- Partial FT seed2: **0.6971** (numerically exceeds Multi-PNA+EU mean; encoder partially updated with AML labels)

## Final answers

- **1_frozen_multiseed_mean_pm_sd:** `{"test_f1_0.5": "0.6393 \u00b1 0.0186", "test_auprc": "0.6636 \u00b1 0.0093", "test_auroc": "0.9880 \u00b1 0.0003", "val_auprc": "0.5368 \u00b1 0.0158"}`
- **2_frozen_mean_numerically_exceeds_multigin_eu:** `false`
- **2_detail:** `{"frozen_mean_f1_0.5": 0.6392612830279665, "published_multigin_eu": 0.6479, "delta": -0.008638716972033555}`
- **3_locked_finetuned_seed2_test:** `{"test_auprc": 0.7009470591314606, "test_auroc": 0.9859688885641126, "test_f1_0.5": 0.6971279373368147, "test_f1_val_thr": 0.6933932007697243, "threshold": 0.46, "best_epoch": 18, "stored_val_auprc": 0.5995712482157797}`
- **4_finetuning_improves_over_frozen_seed2:** `{"val_auprc": true, "test_auprc": true, "test_f1_0.5": true, "deltas": {"delta_stored_val_auprc": 0.04957124821577963, "delta_test_auprc_vs_ref": 0.02694705913146056, "delta_test_f1_0.5_vs_ref": 0.041127937336814635}}`
- **5_finetuning_numerically_exceeds_multipna_eu:** `true`
- **5_detail:** `{"ft_f1_0.5": 0.6971279373368147, "published_multipna_eu": 0.6816, "delta": 0.015527937336814679}`
- **6_abstract_conclusion_result:** `frozen_dplus_multiseed_primary`
- **7_finetune_placement:** `secondary_sensitivity_or_appendix`
- **8_limitations:** `["Downstream MLP uses AML labels; pipeline is not wholly unsupervised.", "Feature stack / learner / downstream seed locked on seed-2 validation (18678029).", "Partial FT updates final encoder block with AML labels; not the primary claim.", "Paper baselines differ in protocol details; prefer 'numerically exceeds reported mean'.", "n=3 encoder seeds; sample SD is descriptive, not a formal superiority test."]`
- **9_test_metrics_did_not_influence_selection:** `true`
- **10_no_training_or_followup_jobs_in_this_eval:** `true`

## Confirmations

- Test metrics did **not** influence model/checkpoint/feature/learner/threshold selection.
- No GNN retraining and no automatic follow-up training jobs in this evaluation stage.
- Integrity check (2026-07-25): MD display metrics match JSON at reported precision; source eval JSONs equal final JSON at full float precision; all four checkpoint sha256s match training artifacts on disk; eval jobs 18839150–18839153 COMPLETED 0:0.
- Seed-2 reuse `coverage=1.0` clarified as tautological vs extracted rows; raw-split coverage is 863054/863900.
- Common-cohort metric sensitivity not run (scores not saved); primary metrics not replaced.

