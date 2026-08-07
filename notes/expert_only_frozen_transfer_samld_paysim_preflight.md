# EXPERT_ONLY R198 frozen transfer preflight — SAML-D & PaySim (READ-ONLY)

**Date:** 2026-08-01  
**Scope:** Compatibility + protocol audit for frozen transfer of the AMLWorld Small-HI **EXPERT_ONLY** R198 encoder to **SAML-D** and **PaySim**.  
**Actions taken:** none beyond writing this note and its JSON twin.  
**Not done:** encoder training, embedding extraction, test-metric inspection, Slurm submission, training/eval code edits.

Twin: [`results/diagnostics/expert_only_frozen_transfer_samld_paysim_preflight.json`](../results/diagnostics/expert_only_frozen_transfer_samld_paysim_preflight.json)

---

## Executive verdicts

| Target | Verdict | One-line reason |
|--------|---------|-----------------|
| **PaySim** | **REPAIRABLE** | Geometry and P1 protocol are compatible; need a val-only R198 extract+logistic wiring path (do not silently use D+ PaperStyleMLP). |
| **SAML-D** | **REPAIRABLE** | Protocol B geometry matches edge_dim=8; no frozen-transfer runners exist; lock a **new/exploratory** val-only protocol on EXPERT_ONLY. |

Neither target is **BLOCKED** on data presence, checkpoint existence, or hard edge-dim incompatibility. Neither is **GO** without small implementation glue.

---

## 1. Source encoder (canonical EXPERT_ONLY)

Selected by completed TFMOE weighting ablation (`results/diagnostics/tfmoe_weight_ablation_lr2e-3/package/`): **EXPERT_ONLY**, peak LR **2e-3**, seed **2**, best SSL epoch by val AUPRC = **10** (AUPRC ≈ **0.4886**). Epoch **20** retained only as optional checkpoint sensitivity (AUPRC ≈ 0.4878).

| Role | Path (repo symlink → pool) | SHA256 | SSL epoch |
|------|----------------------------|--------|----------:|
| **Primary** | `saved-models/checkpoint_direct_r198_tfmoe_wtabl_expert_only_20ep_seed2_linear_lr2e-3_epoch10.tar` | `f0280e129c7bf0deb4c4a823fe24dd9e9b1c16ac2951aa87f0d81a55bc30c27c` | 10 |
| Optional sensitivity | `…_epoch20.tar` | `f31ab11495c9aad867fedea759f7a0958fcfe1364d23d7be6d394851f9d99d31` | 20 |

Real path prefix: `/orcd/pool/007/jthi/Multi-GNN/saved-models/` (via `saved-models` symlink).

### Architecture / training flags (from checkpoint payload + ablation notes)

| Flag / property | Value |
|-----------------|-------|
| Model | `gin` + `reverse_mp` + `ego` + `ports` + `emlps` + `tds` |
| `correct_reverse_edge_features` | **True** → reverse semantics `corrected` (swap `in_port`↔`out_port`, `in_td`↔`out_td`) |
| Edge dimension | **8** = `[Timestamp, Amount Received, Received Currency, Payment Format, in_port, out_port, in_td, out_td]` |
| Representation | `pre_embedding_3h` / R198 (`embedding_dim=198`, `n_hidden≈66`) |
| `include_temporal_flow_edge_features` | **False** — TF targets are **not** encoder inputs |
| `preserve_seed_edges` | **False** |
| Weight mode | `expert_only`: `w_contrast=0`, learned β among TF experts |
| InfoNCE encoder gradient | **None** — InfoNCE logged/excluded from `L_total` (see `notes/tfmoe_weight_ablation_lr2e-3.md`, `direct_r198/__init__.py`) |
| BN buffers | Present in `model_state_dict` (running_mean/var); extract must keep `model.eval()` / frozen AML BN |
| TF MoE heads | Stored separately as `direct_r198_tfmoe_state_dict`; meta `discard_at_extract=True` |
| Extract load path | `load_checkpoint_weights` loads **only** `model_state_dict` — MoE / αβ heads never enter the encoder graph |

**Confirmation:** temporal-flow expert heads are discarded at extraction; TF cache columns are not required as encoder inputs for this checkpoint.

---

## 2. Matched epoch-10 three-encoder comparison (separate track)

All three checkpoints exist on disk; **no retraining required**.

| Encoder | Run / checkpoint | SHA256 |
|---------|------------------|--------|
| EXPERT_ONLY | `…_tfmoe_wtabl_expert_only_20ep_…_epoch10.tar` | `f0280e129c7bf0…` |
| Adaptive TFMOE | `checkpoint_direct_r198_tfmoe_40ep_seed2_linear_lr2e-3_epoch10.tar` | `da73ddd5676e2c194a8f22632ac6f838f8ffdb638b5d5bd1f689adfd31d06b9c` |
| DIRECT_H InfoNCE | `checkpoint_direct_r198_infonce_40ep_seed2_linear_lr2e-3_epoch10.tar` | `c79e723d772e18748c0b675126cc0e5b7f2df01fde83d090243d70962521e06c` |

**Feasibility:** **YES** — report as a **matched epoch-10 R198** comparison, separate from older projected-H128 P1 transfer references.

Note: on AMLWorld, DIRECT_H’s *best* ablation epoch is SSL **3**, not 10. Matched-ep10 is still scientifically useful for equal-SSL-budget transfer comparison; do not conflate with “best-on-AML” selection.

---

## 3A. PaySim — graph & feature compatibility

| Item | Finding |
|------|---------|
| Loader | `--data PaySim` via `data_loading.get_data` / `PAYSIM_SPEC` — **available** |
| Formatted data | `aml-data/PaySim/formatted_transactions.csv` (~348 MiB) — **present** |
| Mapping | sender=`nameOrig`→`from_id`; receiver=`nameDest`→`to_id`; timestamp=`step*3600` (re-zeroed); amount=`amount`→`Amount Received` |
| Edge feature order / dim | Base4 + ports2 + tds2 → **edge_dim=8** under ports+tds — matches checkpoint |
| Ports / TDS | Constructed per split graph; required ON for strict load |
| Reverse-edge semantics | Must use `--correct_reverse_edge_features` (`corrected`); legacy inherited swap would be protocol-wrong |
| Categorical mapping | `paysim_legacy_duplicate_v1`: PaySim `type` duplicated into currency + payment slots — **schema-compatible, not semantic AML equivalence** (defensible as locked historical adapter; disclose) |
| Normalization | **`--train_fit_edge_znorm`** (fit mean/std on train edge_attr only) — required for strict inductive P1 |
| Label / future leakage | Labels not in X; balances / `isFlaggedFraud` excluded by leakage-safe contract. Within-split MP can see later edges in the same graph (repo standard); val graph = train∪val. **No test graph for this scout.** |
| Strict checkpoint load | Wrong edge_dim (6 vs 8) → shape failure on `edge_emb`; TF-in flag mismatch gated by `assert_checkpoint_tf_edge_features_flag`. No silent reorder if flags match schema. |

**Prohibited contract:** any post-transaction balance / native Multi-GIN balance feature stack (`paysim_native_*` with balances). Scout must stay on **`paysim_legacy_duplicate_v1`**.

Split sizes (canonical, from prior audits): train **3,792,821** (pos 3,175); val **1,276,276** (pos 780); test **1,293,523** — **test must not be loaded/extracted/evaluated**.

---

## 3B. PaySim — frozen-transfer protocol

| Policy | Locked choice |
|--------|----------------|
| Encoder weights | Frozen; no target-supervised encoder update |
| BN | **`frozen_aml_bn`** (P1 primary) — `model.eval()`, no target BN recal |
| Labels | Target labels train **downstream probe only** |
| Scout split | **Validation only**; `--skip_test_eval` + `extract_splits=train,val` |
| Feature contract | `paysim_legacy_duplicate_v1` |
| Norm | train-fit edge z-norm |
| Canonical protocol | **`P1_strict_inductive_legacy`** (learner=sklearn `LogisticRegression`) — prefer this; **do not** silently replace with PaperStyleMLP (D+ transfer used MLP + often H+X; different claim) |
| Representation | Frozen **R198** (`pre_embedding_3h`) only for primary |

Existing D+→PaySim R198 extracts (`embeddings/paysim_dplus_transfer_final/`) used preserve-ON D+ sources + MLP — **reference only**, not protocol-matched to EXPERT_ONLY (preserve off, expert_only objective).

---

## 4A. SAML-D — graph & feature compatibility

| Item | Finding |
|------|---------|
| Loader | `--data SAML-D` — **available** |
| Formatted / integrity | `aml-data/SAML-D/formatted_transactions.csv`; integrity **PASS** in `notes/samld_protocol_and_integrity.md` |
| Mapping | Standard formatted columns: sender/receiver accounts, Timestamp, Amount Sent/Received (equal by formatter), currency, payment, EdgeID, Is Laundering |
| Edge feature order / dim | Same AMLWorld 8-d schema under ports+tds+correct_reverse |
| Ports / TDS | Required ON for frozen AML SSL load |
| Reverse-edge | `corrected` required |
| Categorical mapping | SAML-local integer codes for currency/payment — **not** AMLWorld vocab-aligned; **disclose as schema transfer, not semantic currency transfer** |
| Normalization | Protocol B: **train-fit** |
| Label / future leakage | Label-in-X checks false in integrity audit; calendar_day split disjoint EdgeIDs; val sees train history. **No test for this scout.** |
| Strict load | Protocol A (`edge_dim=6`, TDS off) is **incompatible** and must be refused. Protocol B geometry matches. |

Split sizes: train **5,707,315** (pos 5,751); val **1,899,523** (pos 1,986); test **1,898,014** — **do not touch test**.

---

## 4B. SAML-D — frozen-transfer protocol

| Item | Finding |
|------|---------|
| Existing designed protocol | **`samld_frozen_aml_corrected_np_v1`** (ports+tds+correct_reverse+train-fit, edge_dim=8) — **design-only**; historically locked to older `checkpoint_gin_emlps_ports_tds_corrected_asym_proj_…_seed2.tar`, **not** EXPERT_ONLY |
| Runners | **None** found (no extract/probe Slurm for SAML-D frozen transfer) |
| Proposed scout protocol | **NEW / EXPLORATORY** — `samld_frozen_expert_only_r198_valonly_v1` (name locked here): same feature geometry as protocol B; source = EXPERT_ONLY ep10; frozen AML BN; LogisticRegression R198; train+val extract only; test forbidden |
| Labels | Probe-only |

---

## 5. Evaluation design (validation scout)

### Primary representation
Frozen **R198 only** (no H+X, no TF concat).

### Primary probe
- **PaySim:** sklearn **`LogisticRegression`** under P1-style settings (class_weight / C as in P1 cells; val-threshold selection on val only).
- **SAML-D:** same logistic learner for protocol parity (exploratory).

### Controls
1. **Target X-only** (raw edge features; no frozen encoder)  
2. **Matched random R198** (`--random_init` same gin+ports+tds+correct_reverse+train_fit architecture)  
3. **Existing frozen AML reference** where protocol-compatible: PaySim older **P1 H128** (`embeddings/final_corrected_no_preserve_multiseed/seed2_P1_strict_inductive_legacy`) as **projected-H128 reference only** — not matched to R198 three-encoder track. D+ PaySim R198+MLP is **not** a clean control for this scout.

### Extraction correctness (mandatory)
- Use **full-subgraph** path: `embedding_extraction.run_embedding_extraction` / generalized `extract_direct_r198_full_cell` pattern.  
- **Prohibit** invalid old seed-only EdgeID path (`extract_direct_r198_seed_only_cell` and any seed-only caches).  
- Gates: global EdgeID uniqueness; train∩val=0; expected n / positive counts; coverage vs loader seed sets; `z.shape[1]==198`; no `test.npz`.

### Proposed evaluation cells

**Primary scout (per dataset):**
1. Extract EXPERT_ONLY ep10 → train+val R198  
2. Extract random R198 → train+val  
3. Probe: R198 logistic + X-only + random (CPU)

**Optional matched three-encoder (per dataset, after primary):**
4. Extract DIRECT_H ep10  
5. Extract adaptive TFMOE ep10  
6. Probe logistic for each (CPU; can share X-only/random)

**Optional sensitivity:** EXPERT_ONLY ep20 extract+probe per dataset.

### Job count

| Bundle | GPU extract jobs | CPU probe jobs | Total jobs | Notes |
|--------|-----------------:|---------------:|-----------:|-------|
| Primary (both datasets) | 4 (2× expert + 2× random) | 2 | **6** | ≤2 concurrent GPU |
| + matched 3-encoder | +4 | +2 | **12** | Still ≤2 concurrent GPU |
| + ep20 sensitivity | +2 | +2 | **16** | Optional |

Cluster: **advanced account expired 2026-08-01**. Use standard only, e.g. `#SBATCH --partition=mit_preemptable` + `--account=mit_general` + `--qos=normal` (as in recent successful jobs), or bare `mit_normal_gpu` / `mit_normal` without advanced QoS. **Do not** request `mit_amf_advanced_*`. Max **≤2 concurrent GPU** jobs.

---

## 6. Resources & storage-conscious plan

| Resource | PaySim (per R198 ckpt, train+val) | SAML-D (per R198 ckpt, train+val) |
|----------|-----------------------------------|-----------------------------------|
| Embedding disk (float32 Z+y+eid, ~) | **~4.1–4.3 GiB** | **~6.1–6.5 GiB** |
| Peak host RAM (historical D+ extract) | **~41–42 GiB RSS** | **~50–65 GiB** (estimate; larger graph) |
| Peak GPU RAM (historical D+ extract) | **~0.22–0.23 GiB** | similar order if batch=8192 |
| Walltime / extract | ~0.5–2 h GPU after graph prep (PaySim ports+TDS setup ~15–20 min historically) | ~1–3 h GPU |
| Probe (logistic) | CPU minutes–~1 h | CPU ~1–2 h |

**Slurm mem recommendation:** PaySim `mem=64G` (floor 48G); SAML-D `mem=96G` (floor 64G); GPU `gres=gpu:1`; wall `04:00:00`–`06:00:00` on preemptable.

**Storage plan:**
1. Extract one dataset × one encoder at a time when possible; never more than two GPU jobs.  
2. Retain primary EXPERT_ONLY train+val for both datasets (~11 GiB) until metrics/logits/IDs/manifests/checkpoint provenance validated.  
3. Matched three-encoder peak if all retained: ~3×(~4.3+6.5) ≈ **~32 GiB** — prefer sequential extract → probe → delete embeddings **only after** validation of metrics, logits, EdgeIDs, manifests, and SHA256 provenance.  
4. **No automatic deletion** in this preflight.  
5. Do not duplicate AMLWorld source embeddings onto target dirs; do not keep seed-only paths.

---

## 7. Exact implementation changes required (next turn; not done here)

1. **Generalize full-subgraph extract** for `--data {PaySim,SAML-D}` (current `scripts/extract_direct_r198_full_cell.py` hardcodes `Small-HI`), plus `--feature_contract paysim_legacy_duplicate_v1` (PaySim), `--train_fit_edge_znorm`, `--skip_test_eval`, `extract_splits=train,val`, `representation_source=pre_embedding_3h`, correct unique_name + `_epoch10` suffix.  
2. **PaySim probe:** P1-style `LogisticRegression` on R198; refuse silent PaperStyleMLP swap; write val-only cell JSON.  
3. **SAML-D:** new exploratory protocol runner + ID/count gates from integrity tables; refuse protocol-A / edge_dim=6.  
4. **Strict load policy:** load `model_state_dict` only; assert edge_dim=8; assert `include_temporal_flow_edge_features=False`; BN eval.  
5. **Hard refuse** seed-only extract scripts and any path that builds/evaluates test.  
6. **Slurm wrappers** on `mit_preemptable`/`mit_general`/`normal` (or bare `mit_normal_gpu`), `MaxConcurrentGPU≤2`.  
7. Optional: matched ep10 DIRECT_H / adaptive extract cells reusing existing checkpoints.

---

## 8. Finish checklist answers

1. **GO / REPAIRABLE / BLOCKED:** PaySim = **REPAIRABLE**; SAML-D = **REPAIRABLE**.  
2. **Source checkpoint(s):** primary EXPERT_ONLY ep10 SHA `f0280e129c7bf0…`; optional ep20 `f31ab11495c9…`; matched DIRECT_H / adaptive ep10 as above.  
3. **Target feature contracts:** PaySim `paysim_legacy_duplicate_v1` edge_dim=8 ports+tds+correct_reverse+train_fit; SAML-D exploratory clone of `samld_frozen_aml_corrected_np_v1` geometry with EXPERT_ONLY source.  
4. **Primary probe / controls:** LogisticRegression R198; X-only; random R198; older P1 H128 as non-matched reference on PaySim only.  
5. **Cells / jobs:** primary **6** jobs (4 GPU + 2 CPU); +matched → **12**; +ep20 → **16**.  
6. **Resources:** see §6 (~4–6.5 GiB emb/ckpt; ~41–65 GiB host RAM; ~0.2 GiB GPU; hours-scale GPU).  
7. **Matched DIRECT_H/adaptive/EXPERT_ONLY:** **feasible without retraining** (separate track).  
8. **Implementation changes:** see §7.  
9. **Confirmation:** no test data accessed; no jobs submitted; no code modified outside this note and the JSON twin.
