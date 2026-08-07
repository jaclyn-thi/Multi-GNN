# PaySim four-domain readiness audit (read-only)

> Twin: [`results/diagnostics/paysim_four_domain_readiness_audit.json`](../results/diagnostics/paysim_four_domain_readiness_audit.json)  
> Package: [`results/diagnostics/paysim_four_domain_readiness_audit/`](../results/diagnostics/paysim_four_domain_readiness_audit/)  
> Reuses: [`notes/multidataset_ssl_compatibility_audit.md`](multidataset_ssl_compatibility_audit.md), [`notes/multidataset_expansion_compatibility_resource_audit.md`](multidataset_expansion_compatibility_resource_audit.md)  
> **No training, smoke, extraction, probe, Slurm, GPU, code edits, cache rebuilds, or test scoring.**  
> Checkpoint-ladder DAG was not read, modified, or depended upon.

---

## Verdict

**`READY_SMALL_REGISTRY_CHANGE`**

Leakage-safe PaySim can join the current N-domain shared-core stack with a **bounded** implementation delta (allowlist + domain registry + RR schedule/calib + checkpoint BN/LossNorm keys + cache path wiring). Raw/formatted data, loader, and MoE-compatible TF cache already exist and **do not need rebuilding**.

Not `READY_EXISTING_ASSETS` because `financial_multidataset_shared_core_v1` still lists PaySim under `excluded_from_this_phase` and Phase-4B `CANONICAL_DOMAINS` is the three-AML/SAML set only.

---

## 1. Existing assets (verified)

| Asset | Status |
|-------|--------|
| Raw CSV `aml-data/PaySim/PS_…_log.csv` | Present (~471 MiB); header includes balances + `isFraud`/`isFlaggedFraud` |
| Formatted `formatted_transactions.csv` | Present (~348 MiB); AMLWorld schema; SHA `03c2fa07…90112c93` |
| Formatter `format_paysim.py` | Maps `step→Timestamp=step*3600`, `amount`, type codes; **excludes balances + isFlaggedFraud** |
| Loader | `--data PaySim` via `dataset_specs.PAYSIM_SPEC` (`hourly_step`, 0.6/0.2/0.2) |
| TF cache `temporal_flow_cache/PaySim` | Present; `temporal_flow_causal_paysim_v1`; 5-col causal; EdgeID = row index |
| EdgeID | Contiguous `[0…N)`; join key for TF MoE |
| Splits | train/val/test EdgeID files; hourly buckets 0–279 / 280–353 / 354–742 |
| Prior frozen transfer | `embeddings/expert_only_frozen_transfer_samld_paysim` (POOL) + diagnostics (val-only) |
| Feature contracts | Shared-core dim-6 geometry defined; PaySim v1 legacy/type/structure contracts diagnostic/historical |
| Four-domain registry | **Not yet** — Phase-4A registry is HI+SAML+LI |

---

## 2. Leakage-safe pretraining contract

Proposed encoder inputs (exact order):

1. `Timestamp` ← `step * 3600` (synthetic seconds; **retain this documented PaySim unit**, then per-domain train-fit z-norm)  
2. `Amount Received` ← `amount`  
3. `in_port`  
4. `out_port`  
5. `in_td`  
6. `out_td`

**Exclude from encoder/pretraining inputs:** `isFraud` / `Is Laundering`, `isFlaggedFraud`, all balance fields, any fraud-derived features, and type-duplicated currency/payment categoricals (shared-core already drops them).

**Timestamp decision:** Do **not** leave raw `step` as the model channel; the formatter already converts to synthetic seconds so TDS/TF windows (`W=604800` s ⇒ 168 PaySim steps) stay consistent with AML/SAML second-based code. Document “PaySim hour-steps encoded as seconds,” then normalize train-only.

Ports/TDS: graph-structural on loaded endpoints/times (prior multi-dataset audits); TF cache is past-only with tie policy B.

---

## 3. Split and label policy

| Item | Policy |
|------|--------|
| Temporal boundary | `hourly_step` on `Timestamp//3600`; edge-fraction 60/20/20 → uneven step counts (train 280 steps, val 74, test 389) |
| Fraud label | `isFraud` → `Is Laundering` at format time; **fraud ≠ AML laundering** for claims |
| Test sealing | `skip_test_eval`; no test metrics; train-only SSL seeds |
| Existing splits vs pretrain | Compatible — same formatted CSV + TF split IDs used historically |
| TF targets/scalers | Cache stores raw causal values; `load_tf_moe_context` fits **train-only** mean/std on the three MoE columns |
| Balance cancellation | Balances absent from formatted CSV → cancellation leakage path closed for encoder |

---

## 4. TF-cache integrity

- Three MoE targets in order: sender timing → recent activity → amount deviation (`log1p_sender_interarrival`, `log1p_sender_past_7d_count`, `log1p_amount_vs_sender_past_mean`).  
- Full 5-col order **matches** `TEMPORAL_FLOW_CAUSAL_FEATURE_NAMES`; MoE indices `[0,2,3]`.  
- Version `temporal_flow_causal_paysim_v1`.  
- Train/val EdgeID SHAs recorded in package `split_cache_integrity.json`.  
- Test split **file present** (as on Small-HI); `load_tf_moe_context` does **not** load test IDs (features matrix is full-graph).  
- **Reuse without rebuild.**

---

## 5. Minimal trainer delta

See `minimal_code_delta.md`. Summary: extend allowlist + `DomainSpec` + `CANONICAL_DOMAINS` / RR + BN/LossNorm keys + PaySim TF path + no-test gates + α/β calib for 4×5 observations. No new objective, no cache rebuild, no native PaySim Multi-GIN default path.

---

## 6. Resource estimate

| Quantity | Estimate |
|----------|----------|
| Host RAM | Heuristic 4-domain ~160 GiB; Phase-4A **measured** 3-domain MaxRSS ~10 GiB — try **128 G** smoke first |
| GPU | ~13–20 GiB alloc expected on Phase-4 stack (Phase-4A ~13–15 GiB) |
| Graph build | ~1636 s (4/3 × 1227 s) |
| s/step | ~2.0–2.5 (Phase-4A mean 1.90; expansion 4-domain mean ~2.29) |
| 40-step smoke wall | ~29 min including build |
| 500 upd/domain (2000 steps) | ~1.7 h |
| 1000 upd/domain (4000 steps) | ~3.0 h |
| PaySim emb train+val R198 | ~3.7 GiB; ckpt bundle ~15 MB |

---

## 7. Proposed smoke (not submitted)

40 steps · 10/domain · RR · workers=0 · no test · unique `phase4c_paysim_four_domain_smoke` roots · advanced GPU flags.

**Objectives (both supported):**

| Option | Role |
|--------|------|
| **A. InfoNCE + temporal experts (adaptive)** | **Recommended** for first 4-domain **infrastructure** smoke — exercises full mixed path |
| **B. Temporal experts only** | Strongest fixed-3000 AML SSL in prior ablation/figures — preferred **scientific** follow-up arm |

---

## Blocking human decisions

1. Extend `financial_multidataset_shared_core_v1` to PaySim vs new contract ID (same geometry).  
2. Fraud-as-target framing (disclose ≠ laundering).  
3. 128 G simultaneous 4-domain vs higher mem / cyclic unload.  
4. Inherit Phase-3 init SHA vs fresh 4-domain init.  
5. Smoke objective A vs B (recommend A for stack check).

---

## Confirmations

No encoder training, smoke, extraction, probe, Slurm, GPU use, source edits, dataset/cache/checkpoint modification, test scoring, or unbounded CSV scans. Prior multi-dataset compatibility/resource audits reused. Checkpoint-ladder DAG untouched.
