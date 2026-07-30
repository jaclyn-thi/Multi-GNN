# Multi-dataset next-steps preflight (read-only)

> Twin: `results/diagnostics/multidataset_next_steps_preflight.json`  
> Generated 2026-07-28. **No code changed. No jobs submitted.**  
> Does not train, extract, inspect new test predictions, or rewrite prior result artifacts.

---

## Track A — PaySim supervised Multi-GIN+EU audit

**Canonical:** [`results/diagnostics/paysim_supervised_multigin_eu.json`](../results/diagnostics/paysim_supervised_multigin_eu.json) · [`notes/paysim_supervised_multigin_eu.md`](paysim_supervised_multigin_eu.md) · per-seed [`results/diagnostics/paysim_supervised_multigin_eu/eval_seed{1,2,3}.json`](../results/diagnostics/paysim_supervised_multigin_eu/)

### 1. Split policy, sizes, positives

| Item | Value |
|------|--------|
| Policy | Temporal `hourly_step` (`dataset_splits.temporal_edge_split`); target 60/20/20; contiguous hour buckets |
| Label | `Is Laundering` ← PaySim `isFraud` at format time |
| Train | n=**3,792,821**, pos=**3,175**, π≈**0.0837%** |
| Val | n=**1,276,276**, pos=**780**, π≈**0.0611%** |
| Test | n=**1,293,523**, pos=**4,258**, π≈**0.329%** |
| Note | Strong **val→test prevalence shift** (test ~5× denser positives) |

### 2. Feature contract

- Contract: **`paysim_legacy_duplicate_v1`**
- PaySim **type duplicated** into AML slots `Received Currency` and `Payment Format` (compatibility mapping, **not** semantic AML equivalence)
- `edge_dim=6` = base4 + ports2; TDS off

### 3. Normalization fit scope

- Policy: **`legacy_per_graph_edge_znorm`** (`train_fit_edge_znorm=false`)
- Train / val / test graphs each z-norm **their own** edge attrs independently → **transductive** w.r.t. split-graph edge statistics
- Differs from frozen-transfer inductive `--train_fit_edge_znorm`

### 4. Graph / message-passing scope

| Flag | Value |
|------|--------|
| reverse_mp, ego, ports, emlps | **on** |
| tds, preserve, correct_reverse, TF-in | **off** |
| Train graph | train edges only |
| Val graph | train∪val edges; seeds = val |
| Test graph | **all** edges; seeds = test |
| Loss | CE on **seed edges only**; context edges MP-only |

### 5. Loss, weights, selection, stopping

- Loss: weighted `CrossEntropyLoss`
- Class weights: **AML gin Bayesian defaults** `w≈{0: 1.000, 1: 6.275}` — **not** PaySim inverse-frequency (~1/π≈1195)
- Stopping: fixed **50** epochs (no patience)
- Selection: **`best_validation_minority_class_f1_paper_argmax`** → `checkpoint_best_val_f1.tar`
- Selected epochs: seed1=**3**, seed2=**34**, seed3=**36**; test never used for selection

### 6. Meaning of paper_argmax F1

Minority-class F1 after **argmax over two-class logits** ≡ softmax P(y=1) ≥ 0.5. Primary paper-compatible Multi-GNN-style metric; **not** a published PaySim number.

### 7. Val-selected-threshold F1 without re-inference?

| Question | Answer |
|----------|--------|
| Saved probability arrays? | **No** under `results/diagnostics/paysim_supervised_multigin_eu/` |
| Val-tuned F1 already in JSON? | **Yes** — `per_seed_eval.*.splits.*.validation_tuned_threshold` |
| New thresholds / PR rebuild offline? | Requires **re-inference** (or a future prob dump) |

### 8. Metrics (aggregates; full per-seed in JSON twin)

| Split / rule | AUROC | AUPRC | F1 | Precision | Recall |
|--------------|------:|------:|---:|----------:|-------:|
| Test paper_argmax (mean±sample SD) | 0.955±0.010 | 0.255±0.027 | **0.202±0.007** | ~0.74–0.95 | ~0.11–0.12 |
| Val paper_argmax | — | — | 0.191±0.005 | — | — |
| Test val-tuned (per seed ~) | same ranking | same | ~0.19–0.23 | high | still low |

Also present: train metrics, confusion (tp/fp/tn/fn), PPR, alert-budget P@k / R@k. Seed2 test example: paper_argmax F1=0.194, AUPRC=0.253, AUROC=0.961; val-tuned thr≈0.406, F1≈0.205.

### 9. Is low F1 compatible with AUPRC + prevalence?

**Yes.** Test π≈0.33%; AUPRC≈0.255 ≫ π ⇒ strong ranking. Argmax operates at **high precision / low recall** → F1≈0.20. Val-tuned does not materially raise F1. Do **not** treat low F1 as proof the baseline is broken.

### 10. vs common published PaySim protocols

Temporal GNN edge task; excludes balance/`isFlaggedFraud`; type-duplication adapter; AML class weights; paper_argmax F1 vs accuracy; expanding/full-graph MP — **not** a Kaggle RF/XGB tabular reproduction. No in-repo published PaySim Multi-GIN numeric target.

### 11. Defects / unsupported assumptions (protocol caveats, not "broken")

1. Type duplication ≠ semantic AML features  
2. AML CE weights on PaySim  
3. Legacy per-graph z-norm (transductive attrs)  
4. Test MP on full timeline  
5. Uncorrected reverse features  
6. Val/test prevalence shift  
7. Seed1 best epoch=3 (volatile selection)  
8. No saved probs for offline re-thresholding  
9. Supervised has PaySim labels — ceiling only vs label-free transfer  

### 12. Smallest diagnostic if needed

**Optional, not required for core claim:** one-seed probability dump (eval-only, no retrain) for seed2 best-val ckpt → enables offline PR/threshold analysis.

---

## Track B — SAML-D integrity and readiness

### Binary verdict: **REPAIRABLE**

Data and load path work; **existing SAML-D supervised numbers/checkpoints are not thesis-citeable** until protocol repairs + val-only gates pass.

### 1–4. Data, unit, labels, splits

| Item | Finding |
|------|---------|
| Formatted | **Present:** `aml-data/SAML-D/formatted_transactions.csv` (~483 MB, ~9.50M edges) |
| Raw | **Present:** `raw-aml-data/SAML-D.csv` (also pool twin) |
| Prediction unit | **Edge / transaction** (`EdgeID`); accounts are nodes |
| Label | `Is Laundering`; overall π≈**0.104%** |
| Split | Temporal **`calendar_day`**, ~60/20/20 (days 0–191 / 192–255 / 256–320) |
| Approx sizes | train ~5.72M (5,764 pos); val ~1.90M (1,984); test ~1.89M (2,125) |

### 5–8. Integrity

| Check | Result |
|-------|--------|
| EdgeID cross-split | **0** overlap (clean) |
| Account overlap | High (~77–78% of val/test accounts also in train) — expected; **not** entity-inductive |
| Labels in X / edge attrs | **No** |
| Train MP sees val/test edges? | **No** (train graph = train edges only) |
| Default edge z-norm | Per-graph (transductive) unless `--train_fit_edge_znorm` |
| Test graph default | **All edges** (transductive MP) — declare before transfer claims |

### 9. Schema vs AMLWorld encoder

- Same ordered pipeline as AMLWorld under matching flags (`edge_dim=6` ports-only Multi-GIN+EU)
- Compromises: Amount Sent=Received; currency/payment are **SAML-local** ints (semantic mismatch for frozen transfer); no PaySim-style feature_contract; must match checkpoint ports/TDS/reverse recipe (`edge_dim` 6 vs 8)

### 10. Existing results

- Smoke F1 ~0.90 (1 ep) flagged **suspect** in `notes/datasets.md` / `notes/downstream-eval-plan.md`
- Checkpoints on pool (`checkpoint_multi-gin-eu-SAML-D-50epochs.tar` etc.) **without** formal diagnostic aggregates
- **Absent** from thesis inventory / registry as valid rows  
→ Treat as **obsolete / untraceable for claims**

### 11. Readiness

| Experiment | Ready? |
|------------|--------|
| X-only baseline | **No** — need build + val gates |
| Supervised Multi-GIN | Code yes / results **no** |
| Frozen AMLWorld transfer | **No** — recipe match + inductive protocol + gates |
| Matched-random | **No** |
| SAML-D-only SSL | **No** (path can run; no trusted runs) |

### 12. Predeclared validation gates (no test for selection)

1. Schema gate (columns, binary label, unique EdgeID, Amount Sent=Received documented)  
2. Split gate (calendar_day fracs; zero edge overlap; report account overlap)  
3. Leakage gate (label out of attrs/`x`; SSL `uses_labels=false`)  
4. Norm gate (`--train_fit_edge_znorm` for inductive/transfer claims)  
5. Graph-protocol gate (declare transductive full-graph vs inductive context)  
6. Dev selection on **val AUPRC** (and/or minority F1); lock test  
7. Smoke sanity vs Small-HI + prevalence baselines before trusting high F1  
8. Controls before claims: X-only + matched-random  
9. Supervised formal only after val-best + seeds ≥3  
10. Frozen transfer only after matching `edge_dim`/ports/TDS/BN protocol + val gates  

---

## Track C — Elliptic readiness

| Question | Answer |
|----------|--------|
| Data / loader | **Neither** present under `aml-data/` or formatters |
| Native task | **Node** classification (illicit transactions) |
| vs Multi-GIN | Different ontology: txn–txn flow / node labels vs account–account edges / edge labels |
| Adapter vs new rep | Needs **new representation** or GCPAL-like txn-node path — not a PaySim-style CSV adapter |
| Min work / runtime | Days–weeks eng + GPU; greenfield |
| Deadline credibility | **Low** (`notes/downstream-eval-plan.md` already deprioritizes) |
| **Verdict** | **DEFER** |

---

## Track D — Multi-domain SSL readiness (AML–PaySim → SAML-D)

**Completed scout:** [`joint_replay_scout/{shared_bn,domain_bn}.json`](../results/diagnostics/joint_replay_scout/) · notes `joint_replay_scout_*.md`  
- Shared weights; domain_bn wins on PaySim; shared_bn collapses PaySim  
- Labels excluded from SSL; val-only; matched streams  

**Smallest safe SAML-D extension (design only — do not implement here):**
- Extend **`scripts/joint_replay_scout.py`** (not `main.py`/`training.py`)
- Shared encoder+proj; **per-domain BN** buffers (aml / paysim / saml)
- Fixed documented ratio (e.g. 1:1:1); train-only norm per domain; no labels in SSL
- Separate val probes; no test; matched-random per domain
- Keep `exploratory_posthoc=true`, `table_eligible=false`

**3-domain support today?**  
- `joint_replay_scout.py`: **No** (hardcoded 2 domains) — needs script changes  
- `train_gnn` / main contrastive: **No** single-dataset loop  

---

## Final execution plan (strict order)

### P0 — before any GPU jobs (human / CPU-light)

1. Freeze SAML-D protocol card: calendar_day split hashes, `--train_fit_edge_znorm` for transfer/SSL claims, graph scope (inductive vs full-test), ports/TDS/reverse matching locked AML recipe.  
2. Write (or revive) SAML-D schema/split/leakage validate script outputs into `results/diagnostics/` — **CPU only**.  
3. Decide PaySim supervised story for thesis: keep as **ceiling with caveats** (Track A); optional later prob dump is **not** P0.  
4. Confirm Elliptic = **DEFER** with collaborator.  
5. Predeclare SAML-D val gates (§B.12) in writing before training.

**Stop for human review:** protocol card + gates signed off.

### First single smoke job (exactly one)

| Field | Spec |
|-------|------|
| Job | `SAML-D` supervised Multi-GIN+EU **smoke** (≤2 epochs, 1 GPU) with train-fit z-norm + documented flags |
| Pass | Finite loss/grads; seed-edge CE only; coverage OK; F1 not "magically" ≈0.9 without X-only/prevalence context; writes `results/diagnostics/saml_d_supervised_smoke.json` |
| Fail | Non-finite loss; label leakage; empty splits; unexplained F1≈0.9 vs X-only — **STOP**, no further GPU |

**Stop for human review** after smoke JSON.

### At most three independent GPU jobs after smoke (no DAG)

| # | Job | Est. resources | Artifacts |
|---|-----|----------------|-----------|
| 1 | SAML-D **X-only + matched-random** val-only controls (CPU or 1 GPU light) | ≤2 h, ≤128G CPU or 1 GPU | `results/diagnostics/saml_d_controls_seed2.json` |
| 2 | SAML-D **supervised Multi-GIN+EU** seed2 formal (50 ep) **or** freeze after smoke if collaborator prefers controls-first only | ≤6 h, 1 GPU, ~128G | ckpt under `saved-models/…`, `results/diagnostics/saml_d_supervised_seed2.json` |
| 3 | **Frozen AMLWorld→SAML-D P1** extract+logistic (seed2) **or** 3-domain domain-BN joint scout extension (choose one; not both in this tranche) | ≤6 h, 1 GPU | embeddings + `…/saml_d_frozen_p1_seed2.json` **or** `joint_replay_scout` 3-domain JSON |

**Hard stops:** after each job's JSON; no automatic follow-ups; no test unlock until val gates pass for that family.

**Do not schedule:** Elliptic; multi-seed SAML confirmation; joint 3-domain **and** frozen transfer in parallel before smoke+controls.

---

## One-page collaborator table

| Track | Verdict | Thesis role now | Next action | Risk if skipped |
|-------|---------|-----------------|-------------|-----------------|
| A PaySim supervised | **Intact ceiling** (low F1 ≠ broken) | Secondary / ceiling | Optional seed2 prob dump later | Over-claiming vs papers |
| B SAML-D | **REPAIRABLE** | Not ready to cite | P0 validate → 1 smoke → controls | Citing smoke F1≈0.90 |
| C Elliptic | **DEFER** | Out of scope | None | Deadline slip |
| D Joint→SAML SSL | Scout extendable; main trainer ≠ 3-domain | Exploratory only | After SAML smoke; domain-BN script change | Premature multi-domain claims |

**Recommended tranche:** P0 SAML integrity → **one** supervised smoke → human review → ≤3 jobs (controls, then either supervised seed2 **or** frozen P1 / joint-3domain — pick by thesis value).

---

## Confirmation

- **No training code modified**  
- **No Slurm jobs submitted**  
- **Only files written:** this note + twin JSON  
