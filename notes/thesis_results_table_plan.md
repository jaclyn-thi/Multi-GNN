# Thesis results table plan

Draft table structure organized by scientific question. Populated cells cite registry rows or named JSON sources; **—** means not available (do not infer).

**Registry:** `results/diagnostics/thesis_experiment_registry.csv`

---

## Table 1 — Dataset and task summary

**Placement:** Main

| dataset | train | val | test | pos (test) | prev (test) | split | source |
|---------|------:|----:|-----:|-----------:|------------:|-------|--------|
| Small-HI | 3,248,254 | 965,466 | 863,050 | 1,611 | 0.187% | calendar_day | `pre_embedding_3h_vs_post_embedding_small_hi.json` |
| Small-LI | 4,432,934 | 1,316,442 | 1,174,673 | 802 | 0.068% | calendar_day | `small_li_dataset_audit.json` |

---

## Table 2 — Main Small-HI comparison

**Placement:** Main | **Canonical pre/post source:** `pre3h_strong_run_comparison.json` (paired n≈862914 for 40ep s2)

| row | AUROC | AUPRC | F1 | P@100 | lift@100 | source |
|-----|------:|------:|---:|------:|---------:|--------|
| raw+morph baseline (no SSL) | 0.905 | 0.066 | 0.136 | — | — | feature ablation 20ep s1 |
| SSL post-128 embedding (20ep s1 baseline) | 0.944 | **0.213** | 0.259 | 0.85 | 455 | alert_budget / architecture |
| SSL post-128 embedding (40ep s2) | 0.949 | 0.245 | 0.304 | 0.80 | 429 | strong-run paired |
| SSL pre-3h embedding (40ep s2) | 0.958 | **0.295** | 0.340 | 0.83 | 445 | strong-run paired |
| SSL post-128 +raw (40ep s2) | 0.955 | 0.284 | 0.343 | 0.79 | 423 | strong-run paired |
| SSL pre-3h +raw (40ep s2) | **0.960** | **0.321** | 0.344 | 0.84 | 450 | strong-run paired |
| FNF full stack post-128 | 0.959 | 0.277 | 0.320 | **0.80** | **429** | strong-run paired |
| FNF full stack pre-3h | 0.968 | 0.291 | 0.314 | 0.73 | **391** | strong-run paired |

**Footnote (F1):** Under paired protocol, pre-3h +raw wins AUPRC (+0.037) and F1 (+0.001): 0.344 vs 0.343. Non-paired ablation post-128 F1=0.347 uses full test n≈863050 — cite separately, not in pre/post table.

**Thesis-critical missing:** Small-HI supervised baseline.

**Optional / not planned:** emb198 multiseed (see registry).

---

## Table 3 — Main Small-LI comparison

**Placement:** Main | Aggregates from multiseed JSON (n=3). **Mean ± sample standard deviation (ddof=1)** over three seeds.

| row | AUROC | AUPRC | F1 | P@100 | R@100 | lift@100 |
|-----|------:|------:|---:|------:|------:|---------:|
| legacy supervised (paper_argmax ep35) | 0.959 | **0.292** | **0.357** | 0.97 | 0.121 | 1419 |
| SSL post-128 +raw mean ± sample SD | 0.904 ± 0.014 | 0.032 ± 0.021 | 0.039 ± 0.028 | 0.227 ± 0.122 | 0.028 ± 0.015 | 331.583 ± 178.766 |
| SSL pre-3h +raw mean ± sample SD | 0.926 ± 0.013 | 0.061 ± 0.034 | 0.054 ± 0.007 | 0.343 ± 0.159 | 0.043 ± 0.020 | 502.251 ± 232.381 |
| SSL post-128 embedding mean ± sample SD | 0.888 ± 0.016 | 0.014 ± 0.010 | — | — | 0.015 ± 0.007 | — |
| SSL pre-3h embedding mean ± sample SD | 0.919 ± 0.009 | 0.039 ± 0.016 | — | — | 0.027 ± 0.010 | — |

Δpre-post (+raw): ΔAUPRC **0.0291±0.0262**, ΔR@100 **0.0145±0.0134**

Seed1 +raw pre-3h AUPRC: **0.0818** (multiseed); 0.0829 emb198 paired join only.

**Footnote:** Legacy F1 is paper_argmax; SSL F1 is val-tuned probe — do not mix without footnote.

---

## Table 4 — Representation-source ablation

**Placement:** Main summary + Appendix detail (unchanged structure; use strong-run HI + multiseed LI)

---

## Table 5 — Architecture ablation

**Placement:** Appendix (PNA not capacity-matched; fairness scout pending)

---

## Table 6 — Contrastive ablations

**Placement:** Appendix | degree_aware → negative_result

---

## Cross-table notes

| Topic | Status |
|-------|--------|
| FNF HI alert-budget | **Recovered** from `pre3h_strong_run_comparison.json` |
| emb198 multiseed | **Optional / not currently planned** |
| Legacy AUPRC | Use eval JSON **0.292** |
