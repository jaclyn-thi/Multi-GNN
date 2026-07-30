# PaySim supervised Multi-GIN+EU failure audit (read-only)

> Twin: `results/diagnostics/paysim_supervised_failure_audit.json`  
> Sources: `notes/paysim_supervised_multigin_eu.md`, aggregate + per-seed eval/train/epoch-history/logs.  
> **No jobs submitted. No retrain. No new test-based selection. Training code untouched.**

## Classification: **MULTIPLE_CONTRIBUTORS**

Primary: **VALID_BUT_FEATURE_INCOMPLETE** + **genuinely hard temporal evaluation**.  
Secondary: class-weight / checkpoint-selection limitations.  
**No graph/evaluation defect found.** Published high PaySim scores are **not protocol-comparable** here.

| Hypothesis | Support |
|------------|---------|
| A Hard temporal PaySim | **Major** — test π≈0.329% vs val 0.061% (~5.4×); late-step fraud densification |
| B AML schema information loss | **Major** — balances + `isFlaggedFraud` discarded; type duplicated |
| C Threshold / weights / selection | **Secondary** — val-tuned F1≈paper_argmax; AML w₁≪IF; seed1 picks epoch 3 |
| D Graph/eval bugs | **Not supported** |
| E Published-protocol mismatch | **Major for external comparisons** |

---

## 1. Dataset and cohort integrity

Formatted SHA256: `03c2fa07b95d145e754b74a5e646c2d71cd4fed051210d6292a0bbab90112c93`  
Raw: `aml-data/PaySim/PS_20174392719_1491204439457_log.csv` (columns include balances + `isFlaggedFraud`).

| Split | n | EdgeIDs unique | fraud | π | steps |
|-------|--:|---------------:|------:|--:|-------|
| Train | 3,792,821 | 3,792,821 | 3,175 | 0.083711% | 1–280 |
| Val | 1,276,276 | 1,276,276 | 780 | 0.061115% | 281–354 |
| Test | 1,293,523 | 1,293,523 | 4,258 | 0.329179% | 355–743 |
| All | 6,362,620 | all unique | 8,213 | 0.129082% | 1–743 |

| Check | Result |
|-------|--------|
| Duplicate EdgeIDs / rows | **0** / **0** |
| Missing / invalid accounts / self-loops | **0** / **0** / **0** |
| Accounts | all 9,073,900; val∩train **13.0%**; test∩train **11.6%** |
| Amount Sent = Received | all rows |
| Label in edge features / MP attrs | **No** |
| Eval coverage | ≈ **1.0** (≤11 dropped scored seeds on ~3.8M train) |

### Type × label (train)

| Type | n | fraud | π |
|------|--:|------:|--:|
| PAYMENT (0) | 1,277,764 | 0 | 0.0000% |
| TRANSFER (1) | 313,391 | 1,578 | 0.5035% |
| CASH_OUT (2) | 1,347,134 | 1,597 | 0.1185% |
| DEBIT (3) | 23,355 | 0 | 0.0000% |
| CASH_IN (4) | 831,177 | 0 | 0.0000% |

Fraud occurs **only** in TRANSFER + CASH_OUT (all splits). PAYMENT/DEBIT/CASH_IN are pure negatives.

### Amount (train)

| Class | mean | median | p99 |
|-------|-----:|-------:|----:|
| Neg | 154,984 | 74,800 | 1,259,093 |
| Pos | 1,352,101 | 407,834 | 10,000,000 |

Native **balance** distributions: **unavailable** in formatted CSV (discarded at format time).

Labels, EdgeIDs, and eval seed rows are aligned (coverage≈1; one txn → one EdgeID).

---

## 2. Native feature inventory → Candidate A contract

| Native column | Fate under Candidate A |
|---------------|------------------------|
| step | → Timestamp (=step×3600); split buckets |
| type | **Duplicated** into Sent/Received Currency **and** Payment Format |
| amount | → Amount Sent & Amount Received; encoder uses **Amount Received** |
| nameOrig / nameDest | Graph only (`from_id`/`to_id`) |
| oldbalanceOrg / newbalanceOrig | **Discarded** |
| oldbalanceDest / newbalanceDest | **Discarded** |
| isFraud | Label only (`Is Laundering`) |
| isFlaggedFraud | **Discarded** |

**Encoder edge_attr (dim=6):** `[Timestamp, Amount Received, Received Currency, Payment Format, in_port, out_port]`  
(`Sent Currency` / `Amount Sent` present in CSV but **not** in `DEFAULT_EDGE_FEATURE_COLS`.)

**Strong native balance features are missing.** AML compatibility contract is **not** suitable as a strong PaySim-supervised ceiling if the goal is to match tabular PaySim literature.

---

## 3. Graph-construction integrity

| Check | Status |
|-------|--------|
| One txn → one seed edge | **Yes** (n_txns = n_EdgeIDs = 6,362,620) |
| Directions nameOrig→nameDest | **Yes** |
| Parallel txns preserved | **Yes** (no coalescing; 0 duplicate rows) |
| Reverse edges | MP-only (`inherited_legacy`; correct_reverse **off**) |
| Supervised CE | Seed edges only (`input_id` mask) |
| Train MP sees val/test? | **No** |
| Ego / ports / emlps | On; TDS off — matches Candidate A |
| Account-ID collision | Factorize over all names; no negative/invalid IDs |

---

## 4. Training and selection (existing artifacts)

| | Seed1 | Seed2 | Seed3 |
|--|------:|------:|------:|
| Selected epoch (best val F1) | **3** | **34** | **36** |
| Val F1 @ select | 0.202 | 0.202 | 0.193 |
| Val AUPRC @ select | 0.158 | 0.165 | 0.156 |
| Max-val-AUPRC epoch | 7 (0.164) | 34 (0.165) | 45 (0.166) |
| Final-epoch val F1 | 0.179 | 0.169 | 0.149 |
| Eval test paper_argmax F1 | 0.204 | 0.194 | 0.208 |
| Eval test AUPRC | 0.230 | 0.253 | 0.283 |

**Shared:** Adam, lr≈0.00621 (constant), batch 8192, neighs 100/100, 50 epochs, no early stop.  
**Class weights:** fixed AML gin Bayesian `w=[1.000, 6.275]` from `model_settings.json` — **not** PaySim IF.

| Scheme | w₊ |
|--------|---:|
| None | 1 |
| Inverse-frequency (train) | ≈ **1195** |
| AML default used | **6.275** (~0.5% of IF) |

Seed1 selection at epoch **3** is a **noisy early maximum** (top-5 val F1 also includes epochs 7/40/20/50). Seeds 2–3 select mid/late epochs.

Positive exposure: ≈3175 positive seed edges per full train epoch.

---

## 5. Threshold and calibration

| Rule | Availability |
|------|----------------|
| paper_argmax (≡ P≥0.5) | In eval JSON — **primary** |
| Val-selected max-F1 threshold | In eval JSON (diagnostic) |
| Alert-budget P@k / R@k | In eval JSON |
| Saved score/prob arrays | **Absent** for supervised GIN |
| Full PR-curve / calibration plots | **Not saved** |

**Seed2 test example:** paper_argmax F1=0.194 (P≈0.95, R≈0.11); val-tuned F1≈0.205 — **not** a large lift.  
Alert budgets show high precision at low recall (P@100 etc.).

**Verdict:** F1≈0.20 is an **operating-point / recall** story under **good ranking** (test AUPRC≈0.255 ≈ **78×** prevalence 0.00329). Ranking is **not** poor relative to prevalence. Thresholding is **not** the major failure mode.

---

## 6. Existing non-GNN / probe controls (protocols labeled)

| Control | Features | Split | Norm | Headline | Match supervised GIN? |
|---------|----------|-------|------|----------|------------------------|
| X-only logistic | legacy_duplicate only | temporal | train-fit StandardScaler | test AUPRC **0.086**, F1@0.5 **0.156** | **No** (norm/learner) |
| Random H logistic | random encoder H | temporal | train-fit z-norm | test AUPRC **0.026** | **No** |
| Frozen capacity (seed2, val) | X / H / H+X logistic+MLP | temporal | P1 inductive | MLP X val AUPRC 0.066; log H+X 0.041 | **No** (val-only) |
| Supervised Multi-GIN+EU | legacy+ports GNN | temporal | legacy per-graph | test AUPRC **0.255**, F1 **0.202** | — |
| LightGBM/XGBoost | — | — | — | **Not present** | — |

Supervised GIN **beats** X-only ranking on the same compatibility features (0.255 vs 0.086 test AUPRC) despite protocol differences — graph model is learning something beyond raw X, but both are **balance-free**.

---

## 7. Published-protocol reconciliation

In-repo: **no** protocol-compatible published PaySim Multi-GIN numeric baseline (Egressy Multi-GNN is AMLWorld-directed; no PaySim numbers).

External high PaySim scores typically involve some mix of: random/IID splits, resampling, **full balance features**, accuracy on easier setups, tabular GBDT — **not** recoverable as comparable to this temporal, full-imbalance, balance-free, paper_argmax edge-GNN protocol.

---

## 8. Follow-ups (not submitted; ≤3)

1. **Full-native tabular control** under the **identical temporal split** (include balances) — clarifies B vs A.  
2. **Supervised Multi-GIN + versioned PaySim-native edge contract** (balances in; no type duplication) — tests B inside the GNN.  
3. **Random-split diagnostic only** for published comparability (not thesis primary) — clarifies E.

Optional later: probability dump of existing seed2 ckpt for offline calibration (does not fix feature incompleteness).

---

## Direct answers

1. **Is F1≈0.20 internally valid?** **Yes** — reproducible across seeds; coverage/labels/graph checks pass.  
2. **Is ranking poor vs prevalence?** **No** — test AUPRC≈0.255 ≫ π≈0.329%.  
3. **Which native features omitted?** `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`, `isFlaggedFraud`.  
4. **Is AML compatibility suitable for a strong supervised PaySim baseline?** **No** — as a literature-competitive supervised ceiling.  
5. **Is thresholding a major contributor?** **No**.  
6. **Is temporal shift a major contributor?** **Yes**.  
7. **Graph/eval bug evidence?** **No**.  
8. **Smallest clarifying experiment?** Native-feature **tabular** control on the **same temporal split**.  
9. **Files modified beyond the two audit artifacts?** **No** (intermediate cohort helper removed after merge).  
10. **Jobs submitted?** **No**.
