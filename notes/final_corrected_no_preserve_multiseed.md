# Final corrected/no-preserve multiseed evaluation

> Final-results evaluation of frozen encoders (seeds 1–4). **Not** architecture search.
> Test metrics never selected protocol, checkpoint, learner, threshold, normalization, or feature contract.
> Aggregate completed **2026-07-27 01:37 EDT** (job `18952861`).

- Development seed: **2**
- Confirmation seeds: **1, 3, 4**
- Descriptive aggregate: seeds **1–4** (includes development seed; say so when citing)
- Seed 5: not included
- Encoder frozen everywhere; no GNN train/finetune
- Recipe: GIN + ports + TDS + emlps + ego + reverse_mp; **corrected** reverse; **preserve OFF**; asym proj 128; 8192 neg; queue 0; accum 4; T=0.5; 40 ep

---

## Verdict (thesis-facing)

1. **AMLWorld source retention is strong and stable.** Primary pre-3h H+X+TF PaperStyleMLP test AUPRC ≈ **0.67** (confirmation and descriptive agree within ~0.004).
2. **Post-128 H is only a diagnostic** (AUPRC ≈ 0.25) — do not replace the primary stack based on test.
3. **PaySim strict zero-shot (P1) is weak and high-variance** and does **not** beat the X-only control on mean AUPRC.
4. **Label-free BN adaptation (P2) is the stronger PaySim transfer protocol** in this pass (confirmation AUPRC ≈ 0.11), but it is **not** pure zero-shot — keep that wording hard.
5. **P3 type-only** is a sensitivity check; it does not improve on P1 and must not replace legacy-primary reporting based on test.
6. **n=4 does not support statistical superiority** vs supervised Multi-GIN+EU or preserve-seed D+.

**Recommendation:** Treat corrected/no-preserve as the **final primary frozen encoder family for AMLWorld**. For PaySim, report **P1 as the strict inductive baseline** and **P2 as label-free target adaptation**, without claiming P1 beats tabular X-only.

---

## 1. AMLWorld primary — pre-3h H+X+TF (PaperStyleMLP)

Test metrics; threshold 0.5 and validation-selected F1 threshold. Coverage ≈ 863k test edges, 1611 positives.

| Seed | Role | AUPRC | AUROC | F1@0.5 | F1@val-thr | P@100 |
|-----:|------|------:|------:|-------:|-----------:|------:|
| 1 | confirmation | 0.6555 | 0.9876 | 0.6006 | 0.6357 | 0.98 |
| 2 | development | 0.6870 | 0.9880 | 0.6928 | 0.6460 | 1.00 |
| 3 | confirmation | 0.6786 | 0.9879 | 0.6081 | 0.5900 | 0.98 |
| 4 | confirmation | 0.6750 | 0.9877 | 0.6435 | 0.6048 | 0.98 |

### Aggregates (test @0.5)

| Aggregate | AUPRC | AUROC | F1@0.5 | F1@val-thr |
|-----------|------:|------:|-------:|-----------:|
| Confirmation (1,3,4) | 0.6697 ± 0.0124 | 0.9877 ± 0.0002 | 0.6174 ± 0.0229 | 0.6102 ± 0.0233 |
| Descriptive (1–4) | 0.6740 ± 0.0134 | 0.9878 ± 0.0002 | 0.6363 ± 0.0421 | 0.6191 ± 0.0261 |

Seed-2 is the best single seed on AUPRC/F1@0.5; confirmation mean stays within ~0.004 of the descriptive mean → development-seed optimism is small on the primary stack.

### Ablation within AMLWorld (descriptive test AUPRC @0.5)

| Stack | Mean ± sd | Note |
|-------|----------:|------|
| pre3h H only | 0.5056 ± 0.0465 | encoder without X+TF |
| **pre3h H+X+TF (primary)** | **0.6740 ± 0.0134** | locked primary |
| post128 H only (diagnostic) | 0.2491 ± 0.0214 | much weaker |

---

## 2. AMLWorld post-128 H diagnostic

| Aggregate | Test AUPRC @0.5 |
|-----------|----------------:|
| Confirmation | 0.2477 ± 0.0260 |
| Descriptive | 0.2491 ± 0.0214 |

Per-seed AUPRC: s1=0.2774, s2=0.2535, s3=0.2294, s4=0.2362.

**Do not** promote post-128 over pre-3h H+X+TF from these test numbers.

---

## 3. PaySim P1 — strict inductive primary (`paysim_legacy_duplicate_v1`)

Legacy compatibility contract (type duplicated into currency + payment). Frozen AML BN. PaySim-train-fit edge z-norm. Logistic `cw=model`, C=1, downstream seed=1.

| Seed | AUPRC | AUROC | F1@0.5 | F1@val-thr |
|-----:|------:|------:|-------:|-----------:|
| 1 | 0.0840 | 0.9162 | 0.0847 | 0.1513 |
| 2 | 0.0201 | 0.8510 | 0.0518 | 0.0534 |
| 3 | 0.0185 | 0.7683 | 0.0350 | 0.0224 |
| 4 | 0.0575 | 0.7332 | 0.1255 | 0.1115 |

| Aggregate | AUPRC @0.5 |
|-----------|-----------:|
| Confirmation | 0.0533 ± 0.0329 |
| Descriptive | 0.0450 ± 0.0316 |

High seed variance; development seed **2 is among the worst** on P1 AUPRC (0.020), so descriptive mean is pulled down relative to confirmation.

vs controls (legacy): random AUPRC **0.0261**, X-only **0.0865**.  
→ Descriptive P1 mean (**0.045**) beats random but **loses to X-only**.

---

## 4. PaySim P2 — label-free target BN adaptation (legacy)

Same legacy contract + train-fit norm; BN running stats recalibrated on PaySim **train only** (no labels). Learned weights frozen.

| Seed | AUPRC | AUROC | F1@0.5 | F1@val-thr |
|-----:|------:|------:|-------:|-----------:|
| 1 | 0.1475 | 0.8629 | 0.1969 | 0.1876 |
| 2 | 0.0470 | 0.8784 | 0.0621 | 0.0816 |
| 3 | 0.1083 | 0.7597 | 0.1233 | 0.1569 |
| 4 | 0.0706 | 0.8050 | 0.0792 | 0.1488 |

| Aggregate | AUPRC @0.5 |
|-----------|-----------:|
| Confirmation | 0.1088 ± 0.0384 |
| Descriptive | 0.0934 ± 0.0440 |

P2 ≈ **2× P1** on mean AUPRC. Confirmation P2 (**0.109**) is above X-only (**0.087**); descriptive P2 (**0.093**) is roughly tied / slightly above X-only. Still **not** zero-shot.

---

## 5. PaySim P3 — type-only sensitivity (`paysim_type_only_v1`)

| Aggregate | AUPRC @0.5 |
|-----------|-----------:|
| Confirmation | 0.0485 ± 0.0123 |
| Descriptive | 0.0441 ± 0.0134 |

Per-seed AUPRC: s1=0.0623, s2=0.0309, s3=0.0444, s4=0.0386.

Similar to P1; **does not replace P1** based on test. Type-only random control AUPRC=0.0335; type-only X-only=0.0865 (same as legacy X-only within rounding).

---

## 6. Matched controls (computed once)

| Control | Contract | Test AUPRC | Test AUROC | F1@0.5 |
|---------|----------|-----------:|-----------:|-------:|
| Random encoder | legacy | 0.0261 | 0.5783 | 0.0681 |
| Random encoder | type_only | 0.0335 | 0.5699 | 0.0659 |
| X-only (edge_attr) | legacy | 0.0865 | 0.7428 | 0.1560 |
| X-only (edge_attr) | type_only | 0.0865 | 0.7429 | 0.1560 |

---

## 7. Equal-weight ensembles (secondary; not in robustness mean)

Val-selected threshold on ensemble val proba; applied once to test.

| Protocol | Test AUPRC @0.5 | Test F1@val-thr | n_test ∩ |
|----------|----------------:|----------------:|---------:|
| AML pre3h HxXTF | 0.7056 | 0.6400 | 862821 |
| AML pre3h H | 0.5930 | 0.5921 | 862821 |
| AML post128 H | 0.3377 | 0.3944 | 862821 |
| PaySim P1 | 0.1100 | 0.0264 | 1293521 |
| PaySim P2 | 0.1818 | 0.0212 | 1293521 |
| PaySim P3 | 0.0753 | 0.0215 | 1293521 |

Ensembles improve ranking AUPRC, especially on PaySim, but PaySim ensemble **F1@val-thr collapses** (very low positive prediction rate after val tuning) — report carefully; do not treat ensemble F1 as a robustness mean seed.

---

## 8. Cautious external comparators

| Reference | What it is | Caveat |
|-----------|------------|--------|
| Published Multi-GIN+EU fixed-decision F1 ≈ 64.8% | Supervised paper figure | Different objective / decision rule |
| Reproduced supervised Multi-GIN+EU | Thesis supervised parity notes | Not frozen SSL |
| Preserve-seed D+ frozen PaySim | [`paysim_dplus_transfer_final.json`](../results/diagnostics/paysim_dplus_transfer_final.json) primary AUPRC ≈ 0.11 ± 0.03 | **preserve ON**; different recipe |

No superiority claim from n=4.

---

## 9. Thesis-safe claim language

Frozen corrected/no-preserve Multi-GIN contrastive encoders (seeds 1–4) evaluated with locked AMLWorld PaperStyleMLP (pre-3h H+X+TF primary; post-128 H diagnostic) and locked PaySim logistic probes (P1 legacy frozen-BN primary; P2 label-free BN adaptation; P3 type-only sensitivity).

- n=4 does not support statistical superiority claims.
- Keep P1 (frozen BN / strict inductive) and P2 (label-free BN adaptation) clearly separated.
- Do not claim P1 beats X-only on mean AUPRC.
- Do not make an unqualified GCPAL comparison.

---

## 10. Job timeline (from Slurm logs)

| Job | ID | Wall |
|-----|---:|------|
| AML s1–s3 | 18952852–54 | ~21:26–22:01 EDT |
| AML s4 | 18952855 | ~22:04–22:35 |
| PaySim s1–s3 | 18952856–58 | ~22:04–23:47 |
| PaySim s4 | 18952859 | ~22:38–00:15 |
| Controls | 18952860 | ~23:42–01:30 |
| Aggregate | 18952861 | ~01:31–01:37 |

No traceback failures in these logs.

---

## Artifacts

| Role | Path |
|------|------|
| This note | [`notes/final_corrected_no_preserve_multiseed.md`](final_corrected_no_preserve_multiseed.md) |
| Canonical JSON | [`results/diagnostics/final_corrected_no_preserve_multiseed.json`](../results/diagnostics/final_corrected_no_preserve_multiseed.json) |
| Cells | [`results/diagnostics/final_corrected_no_preserve_multiseed/cells/`](../results/diagnostics/final_corrected_no_preserve_multiseed/cells/) |
| Probabilities | [`results/diagnostics/final_corrected_no_preserve_multiseed/probabilities/`](../results/diagnostics/final_corrected_no_preserve_multiseed/probabilities/) |
| Embeddings | [`embeddings/final_corrected_no_preserve_multiseed/`](../embeddings/final_corrected_no_preserve_multiseed/) |
| Preflight | [`notes/final_corrected_no_preserve_multiseed_preflight.md`](final_corrected_no_preserve_multiseed_preflight.md) |
| Submission | [`notes/final_corrected_no_preserve_multiseed_submission.md`](final_corrected_no_preserve_multiseed_submission.md) |
| Smoke | [`results/diagnostics/final_corrected_no_preserve_multiseed/smoke.json`](../results/diagnostics/final_corrected_no_preserve_multiseed/smoke.json) |
