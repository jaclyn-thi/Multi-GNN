# PaySim temporal-flow (TF) downstream audit

> Read-only audit. No training, feature construction, Slurm, or new test metrics.  
> Twin: `results/diagnostics/paysim_temporal_flow_downstream_audit.json`

## Executive answers (short)

| # | Question | Answer |
|---|----------|--------|
| 1 | Was PaySim TF ever evaluated downstream? | **No.** Explicitly deferred; no PaySim TF cache; no HxXTF PaySim result cells. |
| 2 | Is locked AML TF causal? | **Yes — past-only expanding-window causal** (val/test may use earlier-split history; never future or labels). |
| 3 | Can it map faithfully to PaySim? | **Yes for the five TF features** (needs `from_id`, `to_id`, `Timestamp`, amount). Schema placeholders affect **X**, not TF math. |
| 4 | Can final frozen H be reused without extraction? | **Post-128 yes** (join TF by `edge_id` once a PaySim cache exists). **Pre-3h no** for corrected/no-preserve PaySim (never extracted). |
| 5 | Is a new TF evaluation scientifically justified? | **Yes** — locked AML primary is H+X+TF; PaySim primary is H-only (or H+X without TF). Missing TF blocks like-for-like transfer claims. |
| 6 | Smallest next job? | **CPU: extend builder + build PaySim `temporal_flow_causal` cache + integrity/leakage checks.** Then seed-2 val logistic ablation. |
| 7 | What would positive results strengthen? | If **H+X+TF ≫ X+TF**: embedding transfer. If **X+TF ≥ H+X+TF**: engineered features, not transferred H. |
| 8 | Files modified / jobs submitted this turn? | **No** (only this audit MD/JSON written). |

---

## 1. Define AMLWorld TF exactly

### Shared feature set (`temporal_flow_causal_v1`)

**Code:** `morphology/temporal_flow_causal.py`  
**Builder:** `scripts/build_temporal_flow_causal_cache.py`  
**Cache:** `results/cache/temporal_flow_causal/{Small-HI,Small-LI}/`  
**Dim:** **5** (edge/transaction-level; CSV row order ≡ `edge_id`)  
**Labels:** never used (`uses_labels: false`)

| Name | Definition (as implemented) | Type |
|------|-----------------------------|------|
| `log1p_sender_interarrival` | `log1p(t − last_ts[sender])` if sender has prior activity; else **0** | time gap |
| `log1p_receiver_interarrival` | same for receiver | time gap |
| `log1p_sender_past_7d_count` | `log1p(count)` of **sender outgoing** timestamps in `(t−W, t)`, `W=604800` s; else **0** | count |
| `log1p_amount_vs_sender_past_mean` | `log1p(a / (mean_past_amount + ε))` if sender has prior amount history; else **0** | amount ratio |
| `pair_repeat_indicator` | **1** if prior ordered `(from_id,to_id)` at `t′<t`; else **0** | binary |

**Implementation notes (code evidence):**

- Global timestamp sort (`mergesort`); same-`Timestamp` edges are a **batch**: featurize all, then update account/pair state (no same-timestamp cross-influence).
- Account `last_ts`, `tx_count`, `amount_sum` update for **both** endpoints of each edge; `out_timestamps` (7d count) updates **sender only**.
- Required columns: `Timestamp`, `from_id`, `to_id`, amount (`Amount Received` via `resolve_amount_column`).
- Missing/no-history → **0.0** (not NaN). Cache stores **raw** (unnormalized) values.

### Mode A — TF as **encoder input**

| Item | Detail |
|------|--------|
| Flag | `--include_temporal_flow_edge_features` (default **false**) |
| Code | `morphology/temporal_flow_edge_features.py` |
| Behavior | Append train-z-scored TF (5 cols) to GNN `edge_attr` (both forward/reverse get **same** TF vector) |
| Dim effect | typically edge_dim 8 → **13** |
| Scaling | train-only mean/std from `split_train_edge_id` |
| Locked primary? | **No.** Locked extracts set `include_temporal_flow_edge_features: false`. |

### Mode B — TF in **downstream** stack H+X+TF

| Item | Detail |
|------|--------|
| Pattern | `concat([Z_H, X_raw[edge_id], tf_feat[edge_id]])` then `StandardScaler` fit on **train** |
| Scripts | `final_corrected_no_preserve_multiseed.py`, `gcpal_challenge_fullstack_eval.py`, `final_exploratory_ssl_scout.py`, `probe_temporal_flow_ablation.py`, … |
| Locked AML dims | pre-3h H **198** + X (~24 one-hot edge_native) + TF **5** → ~**227** |
| Locked primary? | **Yes.** `pre3h_HxXTF` PaperStyleMLP is the locked AMLWorld primary stack. |

**Locked AMLWorld primary = Mode B (downstream concat), not Mode A.**

---

## 2. Causality and leakage (locked AML TF)

Answers for a transaction at time **t**:

| Question | Answer |
|----------|--------|
| Can a train edge use later train edges? | **No** (past-only) |
| Can a train edge use validation/test edges? | **No** under temporal split (those have later timestamps); computation is global past-only |
| Can a validation edge use earlier train history? | **Yes** (expanding window; documented) |
| Can a validation edge use later validation edges? | **No** |
| Can a test edge use train/validation history? | **Yes** |
| Can a test edge use later test edges? | **No** |
| Are feature scalers fit on train only? | **Yes** at probe/MLP time; cache itself is unnormalized |
| Expanding-window / per-split / full-graph? | **Expanding-window over full CSV** in timestamp order; history does **not** reset at splits |
| Precomputed globally before splitting? | Features computed once on full CSV; split ID lists stored alongside |

**Classification: causal (streaming / expanding-window).** Not future-leaky. Not label-leaky.  
Not “train-isolated inductive” (val/test intentionally see earlier history).

Evidence: `morphology/temporal_flow_causal.py` docstring; `results/cache/temporal_flow_causal/Small-HI/meta.json`; `notes/temporal_flow_causal_leakage_audit.md` / `results/diagnostics/temporal_flow_causal_leakage_audit.json`.

**Separate caveat (not TF formula):** Multi-GNN test message-passing may use the full timeline graph — inherent GNN scope, orthogonal to TF feature causality.

---

## 3. PaySim support

| Check | Status |
|-------|--------|
| Required fields on formatted PaySim | Present: `from_id`, `to_id`, `Timestamp` (`step*3600`), `Amount Received` |
| Faithful TF math | **Yes** for all five features (amount/time/topology; label-free) |
| Schema placeholders | Affect currency/payment **X** slots (type codes), **not** TF definitions |
| Builder CLI | `--data` choices **`Small-HI`/`Small-LI` only** — PaySim blocked today |
| PaySim TF cache | **Does not exist** (`results/cache/temporal_flow_causal/` has only Small-HI, Small-LI) |
| Prior PaySim TF / HxXTF eval | **None found** in diagnostics/notes; explicitly deferred |

Evidence of deferral:

- `notes/paysim_frozen_transfer_preflight.md`: no PaySim TF cache; TF deferred  
- `notes/paysim_dplus_transfer_final.md`: “TF deferred”; “no TF”  
- `paysim_dplus_transfer_final.json` primary stack is **`pre3h_HxX`** (H+X, no TF)

7-day window on PaySim: with `Timestamp=step*3600`, `W=604800` ⇒ **168 steps** — still a valid past window, different calendar semantics than AMLWorld days (document as limitation).

---

## 4. Inventory of existing PaySim downstream stacks

### Final corrected / no-preserve (primary thesis encoder family)

Encoder: Multi-GIN, corrected reverse, **preserve OFF**, ports+tds+emlps; seeds 1–4.  
Root: `results/diagnostics/final_corrected_no_preserve_multiseed.json`  
PaySim embeds: `embeddings/final_corrected_no_preserve_multiseed/seed*_P*_*/` (**post-128 only**)

| Stack | H | X | TF | Morph | Learner | Norm | BN | Contract | Val/Test | Example metrics (seed2 val AUPRC @0.5) | Path | table_eligible |
|-------|---|---|----|-------|---------|------|-----|----------|----------|----------------------------------------|------|----------------|
| P1 primary | post128 | no | **no** | no | Logistic `cw=model`, C=1, seed=1 | train-fit z-norm | frozen AML BN | legacy_duplicate_v1 | both recorded | **0.0217** | `…/cells/seed2_P1_strict_inductive_legacy.json` | thesis_supporting (H-only transfer) |
| P2 | post128 | no | **no** | no | Logistic same | train-fit | PaySim-train BN recal | legacy | both | **0.0224** | `…/seed2_P2_…json` | adaptation (not pure ZS) |
| P3 | post128 | no | **no** | no | Logistic | train-fit | frozen AML BN | type_only_v1 | both | (see aggregate) | `…/seed*_P3_…` | sensitivity |
| X-only control | — | yes | **no** | no | Logistic | train-fit | n/a | legacy | both | val **0.0046** / test **0.0865** | `…/control_X_only_paysim_legacy_duplicate_v1.json` | control |

**PaySim H+X+TF under a valid causal protocol: never evaluated.**

### Preserve-ON D+ transfer final

Root: `results/diagnostics/paysim_dplus_transfer_final.json`  
Primary: **pre-3h H+X** PaperStyleMLP (**no TF**). Also H-only, post128 variants, X-only. TF deferred by design.

### AMLWorld (for contrast — not PaySim)

Locked primary: **pre-3h H+X+TF** PaperStyleMLP on Small-HI TF cache (Mode B). Encoder-input TF off.

---

## 5. Compatibility with final frozen embeddings

### Corrected/no-preserve seeds 1–4 (PaySim)

| Asset | Status |
|-------|--------|
| Train/val/test NPZ | Present for P1/P2/P3 (`Z`, `y`, `edge_id`) |
| post-128 H | **Yes** (128-d) |
| pre-3h H | **No** PaySim extracts |
| Encoder TF flag | `include_temporal_flow_edge_features: false` |
| Join TF by `edge_id` | **Yes**, once PaySim TF cache row/`edge_id` aligns with CSV indices |
| Re-extract required for post128 H+X+TF? | **No** (need X matrix + TF cache + join) |
| Re-extract for pre3h parity with AML primary? | **Yes** (new PaySim pre3h extract) |

Seed2 P1 shapes: train `(3792812, 128)`, val `(1276274, 128)`, test `(1293522, 128)`.

### `paysim_dplus_transfer_final`

Has PaySim **pre3h + post128** with IDs — good for H+X+TF join **after** PaySim TF cache, but **different encoder family** (preserve ON) than corrected/no-preserve.

---

## 6. Proposed locked downstream ablation (validation-first)

**Encoder:** final corrected/no-preserve frozen Multi-GIN (start with **seed 2**; expand 1–4 only after gate).  
**H:** **post-128** (matches locked final PaySim P1). Pre-3h = optional later sensitivity (requires new extract).  
**Cohort:** same PaySim train/val IDs as P1 embeds.  
**Learner:** locked logistic — `LogisticRegression`, `class_weight=model`, `C=1`, downstream seed **1**.  
**Norm:** train-fit edge z-norm for any H extract path already used; concat stacks use **train-only StandardScaler** on the stacked matrix.  
**Contract:** `paysim_legacy_duplicate_v1`.  
**BN:** frozen AML BN for H cells (P1 protocol).  
**TF:** new PaySim `temporal_flow_causal` cache (Mode B concat only; do **not** enable encoder-input TF for this ablation).

### Required stacks (same cohort)

1. X  
2. TF  
3. X + TF  
4. H (post128)  
5. H + X  
6. H + TF  
7. H + X + TF  

No MLP in the matched-logistic ablation. An MLP sensitivity (to mirror AML PaperStyleMLP) is **optional and separate**, only after logistic gate — AML H+X+TF used MLP, but the scientific gap vs PaySim P1 is “was TF ever included?”, answerable with logistic first.

**Do not** use test for stack/threshold/protocol selection.

---

## 7. Predeclared interpretation / validation gate

Before any new test look:

| Contrast | Question |
|----------|----------|
| H+X+TF vs H | Does TF+X help over embedding-only? |
| H+X+TF vs H+X | Does TF add over H+X? |
| H+X+TF vs X+TF | Does **H** add over engineered features? (**transfer claim key**) |
| H vs X / TF / X+TF | Does embedding beat features alone? |
| Consistency | Seed-2 gate first; then seeds 1–4 confirmation |

**Transferred-embedding claim:** requires **H+X+TF > X+TF** on validation AUPRC by a predeclared margin (suggest ≥ 0.003, or document alternative).  
If X+TF ≥ H+X+TF → supports engineered PaySim features, **not** embedding transfer value.

Also report AUROC / F1@0.5 / F1@val-thr as secondary; primary = **val AUPRC**.

---

## 8. Runtime and implementation plan

| Step | Resource | Notes |
|------|----------|-------|
| 1. Extend builder CLI for `PaySim` + build cache (~6.3M rows) | **CPU**, one job | One-time preprocessing; disk ~ features `(N,5) float32` + ids |
| 2. Leakage/integrity audit (reuse AML audit pattern) | CPU | Past-only, split history, train scaler |
| 3. Seed-2 validation logistic ablation (7 stacks) | CPU (or light GPU if joining H only) | Reuse existing post128 embeds |
| 4. If gate passes: seeds 1–4 confirmation | CPU | Still no test selection |
| 5. Optional: test report + optional pre3h extract + optional MLP | GPU for extract if needed | Only after val gate |

**Estimate:** TF build ≪ 1–3 h CPU; seed-2 ablation ≪ 1 h after cache; multiseed confirmation ≪ few hours total. Prefer **no GPU** until a pre3h extract is justified.

**Job count:** (1) TF construct+integrity → (2) seed-2 val ablation → (3 optional) seeds 1–4 / test report.

---

## 9. Thesis impact (if evaluated)

| Area | Impact |
|------|--------|
| Embedding-only transfer | Strengthened **only if** H+X+TF beats X+TF (and preferably H / H+X). Else weaken “H carries transferable signal beyond engineered flow.” |
| Full-system fraud detection | H+X+TF is the AML **system** stack; without PaySim TF, PaySim results are **not** stack-matched. Positive TF gains would make the PaySim full-system claim closer to AML methodology. |
| Methods | Must describe Mode B causal TF, expanding-window history, train-only scaling, and PaySim timestamp=`step*3600` semantics. |
| Experimental setup | Add PaySim TF cache construction + leakage audit; keep encoder-input TF (Mode A) clearly separate / off for locked extracts. |
| Results tables | New ablation table (7 stacks); do not silently replace H-only P1 primary until gate + policy decide. |
| Limitations | Schema placeholders still limit X semantics; 7d window≠calendar day on PaySim; MP graph scope caveat remains. |

**Do not rewrite thesis files from this audit alone.**

---

## Exact end-state checklist

1. **PaySim TF ever evaluated downstream?** No.  
2. **Existing AML TF causal?** Yes (past-only expanding-window).  
3. **Faithful PaySim mapping?** Yes for TF features; builder CLI/cache missing.  
4. **Reuse final frozen H without extraction?** Post-128 yes (after TF cache); pre-3h no for corrected/no-preserve.  
5. **New TF eval justified?** Yes.  
6. **Smallest next job?** PaySim TF cache construction + integrity (CPU).  
7. **Positive results strengthen?** Embedding transfer **iff** H+X+TF > X+TF; else engineered-feature performance.  
8. **Modified files / submitted jobs this turn?** **No** (audit artifacts only).
