# Final exploratory SSL scout — read-only feasibility preflight

> **Scope:** audit only. No code changes, no training scripts, no job submission,
> no full-dataset loads, no GPU work, no test-metric inspection.
> **Context:** locked primary = corrected/no-preserve multiseed (seeds 1–4). Any
> scout here is **exploratory / post-hoc** and cannot replace that primary without
> explicit caveats.
> **Timestamp (UTC):** 2026-07-27T12:00:00Z (approx; written in audit turn)

## Verdict: **GO** (first wave = C0 + M only)

| Arm | Rank |
|-----|------|
| **C0** (continue InfoNCE only) | **IMPLEMENT_NOW** |
| **M** (InfoNCE + low-weight morph aux) | **IMPLEMENT_NOW** |
| **J** (AML↔PaySim alternating continuation) | **OPTIONAL_LATER** (needs new trainer + BN policy lock) |
| **JM** (J + morph on both) | **OPTIONAL_LATER** |
| **JC** (J + CORAL) | **OPTIONAL_LATER** |
| Frozen structural-reliance diagnostic | **OPTIONAL_LATER** (new eval suite; not blocking C0/M) |

**STOP** conditions (do not proceed to implementation if these become goals of this scout):
- Replacing the locked multiseed primary from scout test scores.
- Re-running degflow at `morph_expert_weight=1.0` on this recipe (already failed multiseed on the older stack).
- Using validation/test graph structure or fraud labels in aux targets.

**REVISE before implementing J/JM/JC:** dual-loader step accounting, BN running-stat policy, and PaySim morph-cache provenance.

---

## Locked starting point

| Item | Value |
|------|-------|
| Checkpoint | `saved-models/checkpoint_gin_emlps_ports_tds_corrected_asym_proj_8192neg_queue0_40ep_seed2.tar` |
| SHA256 | `18e06f555aa4880dfc1e95caa3f54a207e5aa186d266887772640feb93a06ae6` |
| Recipe | GIN, ports+TDS+emlps+ego+reverse_mp, **corrected** reverse, **preserve OFF**, asym proj 128, 8192 neg, queue 0, accum 4, T=0.5, 40ep |
| Development seed | 2 |
| Continuation mechanism available today | `--finetune` weight load + **new** `unique_name` + fresh Adam (optimizer state reset); **no** contrastive `start_epoch` resume |

---

## Answers (1–18)

### 1. Exact historical morphology / expert-head experiments

Major families (all **uncorrected** unless noted; mostly 20ep):

| Family | Example unique / note | Objective |
|--------|----------------------|-----------|
| M1 local / M1b global | `hi_morphology_*_20ep` | morph expert MSE |
| +clustering / triangles / MAE / BC | `hi_morphology_global_{clustering,triangles,bc}_*` | morph expert |
| M2 morph contrast | `hi_morph_global_contrast_*`, `hi_morph_contrast_only_*` | soft-pos InfoNCE |
| Low-weight degree_fan | `morph_expert_emlps_tds_asym_proj_8192neg_queue0_20ep` | morph MSE **w=0.05** |
| Morph-obj recall scout | `hi_morph_obj_{degflow,clustering,degflow_tfreg}_*` | degflow **w=1.0**; clustering w=1.0; degflow_tfreg morph+TF **0.05** |
| Degflow multiseed | seeds 1–3 degflow w=1.0 | **stop** (seed2 collapse) |
| TF aux (not morph expert) | `hi_tf_aux_tf_reg_w0.05_*` | Huber/CE on TF feats @ post-128 |
| Corrected/no-preserve 40ep | `gin_emlps_ports_tds_corrected_asym_proj_8192neg_queue0_40ep_seed{1–4}` | **InfoNCE only** — no morph/TF aux |

Registry rows tagged `|morph|` on the corrected unique_name are **downstream probe feature stacks**, not SSL expert training.

### 2. Exact M objective on corrected/no-preserve?

**No.** Nothing trained morph aux (or TF aux) on the corrected/no-preserve 40ep recipe or on the seed-2 checkpoint above.

Closest: degflow@1.0 and degree_fan@0.05 on **uncorrected** emlps+tds 20ep stacks.

### 3. Structural targets — availability / cost / causal / nontriviality

| Target | AMLWorld | PaySim | Cost | Train-graph-only / causal | Nontrivial from post-128 H? |
|--------|:--------:|:------:|------|---------------------------|------------------------------|
| **degree_fan** (Tier0 global + Tier1 local degrees) | Yes | Yes (topology) | Cheap | Tier0: split-local train graph; Tier1: batch subgraph | Yes (prior morph SSL) |
| **flow_balance** (amount in/out lift) | Yes | Yes if Amount Received present | Cheap | Split-local | Yes (degflow scout interest) |
| local_density / clustering | Yes | Yes | Cheap (batch) | Batch-local | Mixed; clustering@1.0 scout weak |
| motif / triangles | Yes | Yes | Cheap | Batch-local | Historically hurt stacks |
| tier2 BC | Yes (cached Small-HI) | Unproven | **Expensive** | Split-local | Reject for scout |
| temporal_flow_causal bins/reg | Small-HI cached | Not established for SSL morph | Cache build | Past-only causal | Separate TF-aux line; not required for M |
| edge-native attrs | Schema-dependent | Contract-dependent | Free | Per-edge | Easy / leaky vs “structure” claim |

**On disk:** `morphology_cache/Small-HI/` exists. **No PaySim morphology_cache** found.

### 4. Fixed recommendation for M (no sweep)

**Target set (fixed):** `degree_fan` + `flow_balance` (i.e. degflow **groups**, not the failed w=1.0 protocol).

**CLI essence (fixed):**
```text
--morph_expert
--morph_targets local+global
--morph_flow_balance
--morph_target_groups degree_fan,flow_balance
--morph_expert_weight 0.05
--morph_expert_hidden 64
```

**Weight (fixed):** `λ_morph = 0.05` (matches the only prior low-weight morph precedent; explicitly avoids failed degflow@1.0).

**Attachment:** post-128 `z_seed` (post embedding head, pre-projection) — already how `MorphologyExpertHead` works. Discarded at extraction (head unused in `embedding_extraction.py`).

### 5. Rejected / leaking targets

| Reject | Reason |
|--------|--------|
| Fraud / Is Laundering / PaySim `isFraud` | Label leakage |
| Any target built from val/test edges or full-timeline future | Split leakage |
| Tier2 BC for this scout | Cost + prior negative stacking |
| Triangle-heavy / BC stacks | Historical hurt |
| Morph soft-pos (M2) / TF soft-pos | Prior negative / collapsed |
| degflow @ `weight=1.0` | Multiseed **stop** |
| Probe-time engineered morph as if it were SSL aux | Different experiment class |

### 6. Reuse vs new implementation (M)

**Reuse (sufficient for C0/M):**
- `morphology/expert.py`, `contrastive_train.py`, `tier0_*.py`, `target_registry.py`
- `training.py` morph loss wiring
- CLI in `util.py`
- Existing unit tests under `tests/test_morphology_*.py`

**New / light glue for C0/M:**
- Continuation Slurm + unique_name policy (`--finetune` from seed2 ckpt, abort-on-overwrite)
- Optional: precompute PaySim flow cache **only if** JM later; **not** required for AML-only M
- Val-only eval harness cloning final multiseed **validation** protocols (no test)

### 7. Shared encoder edge dimensionality (J)

**Yes.** Under ports+TDS, AMLWorld and PaySim contracts keep `edge_dim=8` (base-4 + ports + TDS). Legacy / type_only / structure_only preserve width; only slot policies differ.

Caveat: shared width ≠ shared semantics (AML currency/format vs PaySim type duplication / neutrals).

### 8. Locked PaySim contract + norm for J

Match final multiseed **P1** (strict inductive primary), not P3:

| Knob | Lock |
|------|------|
| Feature contract | `paysim_legacy_duplicate_v1` |
| Edge z-norm | `--train_fit_edge_znorm` (PaySim-train-fit) |
| Graph flags | reverse_mp, ego, ports, tds, emlps, correct_reverse; preserve OFF |
| Labels | unused in train and in any BN adaptation |

Do **not** silently use per-graph z-norm.

### 9. 1:1 alternating schedule = same optimizer-step budget as C0

From corrected train logs: **397** microbatches/epoch, **accum=4** → **100** optimizer steps/epoch.

Define continuation budget **N = 5** epochs of C0 (recommended scout length):

| Quantity | C0 / M | J |
|----------|--------|---|
| Optimizer steps | \(N \times 100 = 500\) | **500** (same) |
| Microbatches | \(N \times 397\) | \(500 \times 4 = 2000\) |
| Domain mix | AML only | **Alternate domains every optimizer step**: odd→AML, even→PaySim (or reverse, fixed in script) |
| Accumulators | 4 AML microbatches / step | 4 microbatches **from the active domain only** / step |
| Anchor exposure | All AML | ≈50% AML + 50% PaySim of the same total anchor count |

**Not** “AML epoch then PaySim epoch” unless step counts are forcibly equalized (PaySim loader length ≠ 397).

Wallclock: expect **≳ C0** (PaySim batches larger); still under 6h for N=5 if C0≈0.5h and J≲1.5–2h.

### 10. BN behavior under J + validation protocol

**Today’s contrastive loop leaves modules in train mode** → BN `running_*` update on every forward (including asymmetric no-grad view).

Under naive J, BN becomes a **mixed AML+PaySim** statistic — ambiguous for both domains.

**Predeclare (required before implementing J):**
1. **Training:** freeze BN affine+buffers during J continuation (`model.eval()` for BN modules only, or `track_running_stats` freeze) **or** update BN only on AML microbatches (document which).
2. **AML validation:** extract with **AML BN snapshot** (weights from continuation; BN = AML-only or frozen-at-init).
3. **PaySim validation:** two labeled protocols, val-only:
   - **J-strict:** frozen BN from (2) + legacy + train-fit (analogous to P1)
   - **J-adapt (optional diagnostic):** label-free PaySim-train BN recal (analogous to P2) — **not** claimed zero-shot

Do not use test for selection.

### 11. Is JM meaningful with the selected target set?

**Yes, conditionally.** `degree_fan` is topology-only (portable). `flow_balance` needs amount column — present on PaySim after formatting. JM is meaningful as “does morph aux still help when half the steps are target-domain?”

Blockers: no PaySim morph cache yet; must compute Tier0/flow from **PaySim train graph only**; do not share AML morph caches.

### 12. CORAL feasibility (JC)

**No CORAL/MMD code exists.** Feasible as a **small** add-on (not a large rewrite) **if and only if** J’s dual-batch loop exists.

**Exact loss (if later implemented — one weight only):**
Let \(H_A, H_P \in \mathbb{R}^{B\times 128}\) be L2-normalized post-128 seed embeddings from one AML and one PaySim microbatch (same B after truncate/pad).
\[
\mathcal{L}_{\mathrm{CORAL}} = \lVert C(H_A) - C(H_P)\rVert_F^2,\quad
C(H)=\tfrac{1}{B-1}(H-\bar H)^\top(H-\bar H)
\]
**Fixed weight:** \(\lambda_{\mathrm{CORAL}}=0.05\) (same order as morph low-weight; not swept).

Total: \(\mathcal{L}=\mathcal{L}_{\mathrm{InfoNCE}}+\lambda_{\mathrm{CORAL}}\mathcal{L}_{\mathrm{CORAL}}\) (plus morph if JM).

### 13. Runtime estimates (from existing logs)

| Proxy | Wallclock | Source |
|-------|----------:|--------|
| Integrity smoke (load+forward) | ~29 min | smoke `18951224` |
| AML extract+probe / seed | ~0.5 h | AML eval `18952852` |
| PaySim extract+P1/P2/P3 / seed | ~1.6–1.7 h | PS eval `18952856` |
| 40ep corrected train | ~2.7 h | train `18904237` |
| ≈5ep AML continuation (C0/M) | **~0.5 h** | ~12 min featurize + 5×~3.8 min/ep |
| J @ 5ep-equivalent steps | **~1–2 h** (est.) | PaySim microbatches slower; still ≪6h |
| Val-only AML+PaySim probe after train | ~0.5–1.0 h | half of full eval if test skipped |

### 14. Files that would need modification

**C0/M (minimal):**
- New: `slurm/run_final_exploratory_ssl_scout_{smoke,train,eval,aggregate}.sh`
- New: `scripts/final_exploratory_ssl_scout.py` (orchestration; calls `main.py` / extract / val probes)
- Possibly: thin notes/registry append only after aggregate

**J/JM/JC (additional):**
- `training.py` — dual loader / alternating step loop, BN freeze policy, optional CORAL
- `util.py` — CLI for dual data, coral weight, schedule
- `train_util.py` — checkpoint metadata for dual-domain continuation
- Possibly `data_loading.py` — only if dual-graph z-norm helpers needed
- New: `losses/coral.py` or `morphology/coral.py` (JC)

**Structural diagnostic (later):**
- New script under `scripts/` (graph mutations + extract + val probe); likely touch `embedding_extraction.py` only via CLI

**Must not edit after submit:** same freeze discipline as multiseed.

### 15. Tests

**Existing (reuse):** `tests/test_morphology_*.py`, TF aux tests, `tests/test_semantic_group_mask.py`, feature-contract tests.

**New required before train:**
- Continuation smoke: load seed2 → `--finetune` → 1–2 steps C0 and M; assert morph head present only for M; assert learned encoder grads flow; assert no label tensor in morph targets
- Target provenance: morph targets built only from train-split graph IDs
- Edge dim assert 8 under legacy PaySim (for J later)
- Alternating schedule unit test: 8 steps → 4 AML + 4 PaySim optimizer domains
- CORAL unit test: \(\mathcal{L}=0\) when covariances equal; shape checks
- Structural diagnostic (later): shuffle / drop-MP / neutralize change embeddings; random encoder baseline

### 16. Test data / labels required?

**No** for training (C0/M/J/JM/JC all label-free).  
**No** for gate selection (validation only).  
**Do not** inspect or select on test in this scout.

### 17. Ranking summary

| Arm | Rank | Rationale |
|-----|------|-----------|
| C0 | IMPLEMENT_NOW | Matched continuation control |
| M | IMPLEMENT_NOW | Code reuse; scientific gap on corrected recipe; fixed λ/targets |
| J | OPTIONAL_LATER | Scientifically motivated by P2 gains; needs new trainer + BN lock |
| JM | OPTIONAL_LATER | Meaningful after J; needs PaySim morph targets |
| JC | OPTIONAL_LATER | Small CORAL add-on after J; no existing code |
| Structural diagnostic | OPTIONAL_LATER | Useful frozen reliance suite; separate engineering |

### 18. Smallest scientifically controlled first wave

1. **C0** — 5ep continuation from seed2 ckpt, unchanged InfoNCE.  
2. **M** — same budget + fixed degflow groups @ \(\lambda=0.05\).  
3. Val-only AMLWorld primary stack (pre-3h H+X+TF PaperStyleMLP) + post-128 H diagnostic + PaySim P1-style logistic (legacy, train-fit, frozen BN) — **validation only**.  
4. Predeclared gate: M must beat C0 on **AML val** primary AUPRC by a small margin **or** improve PaySim **val** AUPRC without AML val regression beyond a fixed budget — exact thresholds written **before** any val scores are read.  
5. Explicit note: exploratory; cannot replace locked multiseed primary.

Defer J/JM/JC and structural diagnostics until C0/M resolve.

---

## Frozen structural-reliance diagnostic (audit only)

Desired cells (intact / shuffled endpoints / no MP edges / neutralized attrs / matched random / forward–reverse semantics / grad norms by component) are **mostly unimplemented**. Closest existing pieces: feature contracts (neutral slots), train-time edge/attr drop (not frozen diagnostic), global grad-norm smokes. Treat as a **new eval suite** (OPTIONAL_LATER), not part of wave-1 training.

---

## Caveats vs locked primary

- Scout uses **development seed 2** only unless a later confirmation wave is approved.  
- Continuation via `--finetune` resets Adam and epoch counter.  
- Results are **post-hoc** relative to [`notes/final_corrected_no_preserve_multiseed.md`](final_corrected_no_preserve_multiseed.md).

---

## GO / REVISE / STOP

**GO** — implement **C0 + M** only, with the fixed morph target set and \(\lambda=0.05\), val-only gates, and explicit exploratory labeling.

**REVISE** — before any J/JM/JC work: alternating step definition (above), BN freeze policy, PaySim morph cache plan.

**STOP** — if the goal shifts to replacing the locked multiseed primary, reintroducing degflow@1.0, or selecting on test.
