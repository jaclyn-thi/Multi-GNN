# Small-HI ↔ SAML-D categorical mapping audit

> Twin: [`results/diagnostics/small_hi_samld_category_mapping_audit.json`](../results/diagnostics/small_hi_samld_category_mapping_audit.json)  
> **AUDIT ONLY** — exploratory. No data rewrite, adapters, jobs, training, or full CSV scans.  
> Git: `48bcb512415b45a2d1922fc2313bbc3cb065947e`

---

## Final verdict

**`DROP_FOR_FIRST_SHARED_CORE`**

**Recommended policy (one sentence):** For the first Small-HI + SAML-D mixed-SSL smoke, omit `Received Currency` and `Payment Format` and train a shared GIN on `[Timestamp, Amount Received, in_port, out_port, in_td, out_td]` (`edge_dim=6`), because current integer codes are independently first-seen-encoded per dataset and already collide semantically (e.g. code `0`), while complete integer↔label tables are not recoverable without a full ordered raw scan.

---

## 1. Formatting provenance

| | Small-HI (AMLWorld) | SAML-D |
|--|---------------------|--------|
| Raw | `raw-aml-data/HI-Small_Trans.csv` | `raw-aml-data/SAML-D.csv` |
| Formatted | `aml-data/Small-HI/formatted_transactions.csv` | `aml-data/SAML-D/formatted_transactions.csv` |
| Formatter | `format_kaggle_files.py` | `format_saml_d_files.py` |

**How codes are created (both formatters, identical pattern):**

```python
def get_dict_val(name, collection):
    if name in collection:
        return collection[name]
    val = len(collection)
    collection[name] = val
    return val
```

| Field | Small-HI raw column(s) | SAML-D raw column(s) | Written formatted column |
|-------|------------------------|----------------------|--------------------------|
| Currency (received) | `Receiving Currency` → `cur1` | `Received_currency` → `cur1` | `Received Currency` |
| Currency (sent) | `Payment Currency` → `cur2` | `Payment_currency` → `cur2` | `Sent Currency` |
| Payment | `Payment Format` | `Payment_type` | `Payment Format` |

- **Mechanism:** empty `dict()`, first-seen string → next integer (`0,1,2,…`). Not pandas `factorize`, not a fixed dictionary file, not source-native integers.
- **Shared currency dict** within a dataset: both sent and received currency strings populate the **same** `currency` dict (so Sent/Received share one codebook per dataset).
- **Determinism:** deterministic **iff** the same raw file is streamed in the same row order (`datatable.fread` then `range(nrows)`). Mapping is **not** saved to disk.
- **Fit scope:** encoding runs over the **entire raw CSV before temporal split**, then rows are sorted by Timestamp. Category IDs therefore depend on **global raw appearance order**, not train-only (uses strings from all time periods as they appear in the raw file).
- **No rewrite** performed in this audit.

Formatter SHA256:

- `format_kaggle_files.py` → `7f55b7a9e9421f294b6ab1684a5779537e8abac4ee352eb6ed6b5cf88fd8f415`
- `format_saml_d_files.py` → `c1f2d0e309485177e720cd2fe9444cdb06d334762b26c5ff85e82bfe42de6d6b`

---

## 2. Exact mappings — what is / is not recoverable

**Blocker for complete tables:** integer codes are defined by first-seen order over the full raw stream. Recovering a complete `{code → label}` table requires an ordered full pass of the raw CSV (or a persisted mapping file, which does not exist). Per hard limits, **that scan was not started**.

### Recoverable without full scan

| Item | Evidence |
|------|----------|
| Encoding algorithm | Formatter source (above) |
| Independent codebooks per dataset | Separate script runs / separate `currency` & `paymentFormat` dicts |
| Encoder does **not** use `Sent Currency` | `dataset_specs.DEFAULT_EDGE_FEATURE_COLS` |
| SAML-D integer support (counts only) | `samld_separability_audit.json`: Received Currency codes **0–12** (13); Payment Format **0–6** (7) |
| Small-HI currency cardinality (count only) | Prior reports: `"Received Currency": 15` in `temporal_flow_ablation_small_hi_*.json` |
| Partial first-seen labels (bounded sample) | Raw header + **5 rows** each (below) |

### Partial sample-derived first-seen assignments (incomplete; illustrative)

**Bounded read size:** 1 header + 5 data rows per raw file; 5 formatted rows each.

#### Small-HI (raw first-seen among sample only)

| Code | Field | Label (sample-inferred) | Evidence |
|------|-------|-------------------------|----------|
| 0 | Currency | `US Dollar` | First raw `Receiving Currency` / `Payment Currency` |
| 0 | Payment Format | `Reinvestment` | First raw `Payment Format` |
| 1 | Payment Format | `Cheque` | Second distinct payment in raw sample |

Formatted earliest-timestamp rows show currency/payment mostly `0` (consistent with early codes after sort, not a full map).

#### SAML-D (raw first-seen among sample only)

| Code | Field | Label (sample-inferred) | Evidence |
|------|-------|-------------------------|----------|
| 0 | Currency | `UK pounds` | First raw currency string |
| 1 | Currency | `Dirham` | Second raw `Received_currency` |
| 0 | Payment Format | `Cash Deposit` | First `Payment_type` |
| 1 | Payment Format | `Cross-border` | Second |
| 2 | Payment Format | `Cheque` | Third |
| 3 | Payment Format | `ACH` | Fourth |

Formatted sample rows show `Received Currency ∈ {0,1}` and `Payment Format ∈ {0,1,2,3}` matching this prefix.

### Not recoverable here

- Complete Small-HI / SAML-D `{code → original string}` tables  
- Full shared / dataset-only semantic category lists  
- Proof of every collision beyond the sample prefix  

**Evidence required for A (canonical remap):** authorized unique-label recovery (or re-format with a **fixed** name→id dictionary written to a mapping JSON).

---

## 3. Vocabulary comparison (from algorithm + sample + counts)

| Question | Finding |
|----------|---------|
| Identical semantic vocabularies? | **No** (different products/schemas: e.g. `US Dollar` vs `UK pounds` / `Dirham`; `Reinvestment` vs `Cash Deposit` / `Cross-border`) |
| Identical integer assignments? | **No** — independent first-seen dicts |
| Same integer, different meaning? | **Yes (sample-proven collisions):** currency `0` = US Dollar (HI) vs UK pounds (SAML); payment `0` = Reinvestment (HI) vs Cash Deposit (SAML) |
| Same meaning, different integers? | **Likely** (e.g. `Cheque` appears in both raw samples; assigned payment code **1** on HI sample prefix vs **2** on SAML) — full confirmation needs complete maps |
| UNKNOWN/OTHER reserved? | **No** in formatters |
| Unseen val category at format time? | Codes assigned on **full raw** before split; integrity reports for SAML show **0** unseen payment/currency codes val/test vs train (`samld_protocol_and_integrity.json`). Model-time “new code” only if raw formatting were redone on a subset |

**Do not conclude compatibility from category counts** (≈15 vs 13 currencies; SAML 7 payment types). Counts can match while meanings collide.

---

## 4. Current preprocessing treatment

| Aspect | Fact |
|--------|------|
| Slot indices in `edge_dim=8` | `2 = Received Currency`, `3 = Payment Format` (`feature_contracts.SLOT_*`; base then ports@4–5, tds@6–7) |
| Continuous vs categorical | Treated as **continuous floats** via `nn.Linear(edge_dim, n_hidden)` (`models.py` `edge_emb`) — **no** one-hot, **no** `nn.Embedding` in GINe path |
| Z-normalization | Standard SSL path applies **train-fit (or legacy) z-norm to the full `edge_attr`**, including these integer codes (`data_loading.py`) |
| Domain BN | Affinely rescales **activations**, not discrete code semantics; cannot make “0=USD” equal “0=UK pounds” |
| Shared encoder weights | **Yes** — same `edge_emb.weight[:, 2]` and `[:, 3]` multiply both domains’ currency/payment channels |

**Can per-dataset z-norm align categories semantically?** **No.** It only matches moments of the numeric code distribution. Evidence: codes are arbitrary ordinals; z-norm preserves relative ordering of integers within a domain, not cross-domain label identity.

**Can domain BN align them?** **No** for semantics — only feature-distribution / covariate shift at BN layers after a shared linear mix of misaligned codes.

**Would the shared encoder interpret the same slot with the same weights?** **Yes** — that is exactly the hazard under joint training.

---

## 5. Policy comparison (A / B / C) — not implemented

### A. Canonical semantic remapping
- Build fixed shared vocab from **string labels** (`US Dollar`, `Cheque`, …), reserve `UNKNOWN`/`OTHER`.
- **Possible in principle** from raw strings; **not** from current integers without recovery or re-format.
- Proposed direction (incomplete until unique pass): ISO-like currency names + payment-type names union; map OOV → UNKNOWN.
- Rank: highest semantic defensibility; medium effort after recovery; poor drop-in comparability with existing dim-8 R198 if vocab/geometry changes; low shortcut risk if UNKNOWN handled; **not** ready for first smoke without recovery job.

### B. Dataset-specific categorical adapters
- Separate emb tables (or one-hots) per domain → small shared latent (e.g. 4–8-D) concatenated with continuous channels; or replace slots 2–3 with adapter outputs.
- Minimal sketch: `Emb_d(|V_d|, k)` with `k∈{4,8}`; concat with z-scored `[time, amount, ports, tds]`.
- Rank: good defensibility; medium effort; breaks exact parity with legacy continuous-code R198; reduces cross-dataset code shortcuts; heavier than needed for first smoke.

### C. Shared-core ablation (recommended for first smoke)
- **Omit** currency/payment.
- Remaining contract: `[Timestamp, Amount Received, in_port, out_port, in_td, out_td]` → **`edge_dim=6`**.
- Tradeoff: loses categorical signal; **gains** a defensible shared geometry without silent collisions; not bit-comparable to historical `edge_dim=8` R198 checkpoints (acceptable for from-scratch mixed smoke).

| Criterion | A | B | C |
|-----------|---|---|---|
| Semantic defensibility | High (after recovery) | High | High (by omission) |
| Implementation effort | Med–high | Medium | **Small** |
| Comparability vs existing R198 | Low–med | Low | Low (dim 6) |
| Dataset-shortcut risk | Low | Low–med | **Lowest** |
| First mixed smoke | Not yet | Optional later | **Best** |

---

## 6. Required answers

| # | Answer |
|---|--------|
| 1. Identical semantic vocabularies? | **No** |
| 2. Identical integer assignments? | **No** |
| 3. Same-number codes semantically different? | **Yes** (sample-proven for code `0`; likely more) |
| 4. Deterministic shared semantic mapping constructible? | **Yes in principle from raw strings with a fixed dict**; **not** from current integers without full ordered recovery / re-format |
| 5. Adapter necessary for first smoke? | **No** if using **C** (drop categories). **Yes** if keeping categorical slots (then B, not raw shared codes) |
| 6. Smallest defensible adapter (if keeping cats)? | Per-dataset embedding (or one-hot) → shared latent **k≈4–8**, replacing slots 2–3; continuous time/amount/ports/tds unchanged |
| 7. If dropped, shared contract? | `[Timestamp, Amount Received, in_port, out_port, in_td, out_td]`, **edge_dim=6**, train-fit z-norm on these six, domain BN, labels still out of SSL |
| 8. Mapping uses val/test distribution? | **Yes at format time** — first-seen over **full raw** file order (not train-only). Temporal split happens after encoding |
| 9. Data/jobs/full scan? | **No** source/data modified; **no** jobs; **no** full dataset scan; only headers + 5-row samples + existing reports |

---

## 7. Provenance

- Git: `48bcb512415b45a2d1922fc2313bbc3cb065947e`
- Files read: `format_kaggle_files.py`, `format_saml_d_files.py`, `dataset_specs.py`, `feature_contracts.py`, `data_loading.py`, `models.py`, `notes/datasets.md`, `results/diagnostics/samld_separability_audit.json`, `results/diagnostics/samld_protocol_and_integrity.json`, `results/diagnostics/temporal_flow_ablation_small_hi_40ep_seed2.json` (cardinality only), `results/diagnostics/multidataset_ssl_compatibility_audit.md`
- Bounded samples: raw HI 5 rows; raw SAML 5 rows; formatted 5 rows each
- Uncertainty: complete codebooks and full intersection of category **names** unknown without authorized unique/recovery pass

---

## Stop

Human review gate. **Do not implement remapping, adapters, or dim-6 trainers until approved.**
