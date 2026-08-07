# AMLSim bounded loader-readiness audit

> Twin: [`results/diagnostics/amlsim_bounded_loader_readiness_audit.json`](../results/diagnostics/amlsim_bounded_loader_readiness_audit.json)  
> Package: [`results/diagnostics/amlsim_bounded_loader_readiness_audit/`](../results/diagnostics/amlsim_bounded_loader_readiness_audit/)  
> **Read-only.** No training, extraction, probe, GPU, Slurm, dataset/source edits, or full large-CSV scans.  
> Checkpoint-ladder DAG untouched.

---

## Verdict

**`IMPLEMENT_AFTER_PAYSIM`**

Archives are present and the schema is **inspectable and largely unambiguous** for an edge-level shared-core formatter. These packages are **example-scale** only (~17k and ~1M edges). PaySim is already formatted with TF cache and a small registry delta — do AMLSim **after** PaySim unless a human prioritizes AMLSim typology diversity over PaySim’s ready path. Not blocked by missing data or primary label ambiguity.

---

## Answers (1–10)

1. **Archives:** `aml-data/aml-sim/100vertices-10Kedges.7z`, `aml-data/aml-sim/10Kvertices-1Medges.7z` only.  
2. **Both examples** (IBM AMLSim demo sizes). No larger scenario dump found.  
3. **Files:** `transactions.csv`, `accounts.csv`, `alerts.csv`, `paramFiles/*`, `stat/*.png`.  
4. **Edges:** `transactions.csv`.  
5. **Fields:** `TIMESTAMP`, `TX_AMOUNT`, `SENDER_ACCOUNT_ID`, `RECEIVER_ACCOUNT_ID`, `TX_TYPE`; **no currency**; label `IS_FRAUD`; typology `alerts.ALERT_TYPE` via `TX_ID`/`ALERT_ID`.  
6. **Labels:** edge-level `IS_FRAUD` primary; alert rows for typology; account `IS_FRAUD` secondary (exclude from encoder). Tiny package: fraud ⇔ `ALERT_ID != -1`.  
7. **Shared core:** constructible causally once timestamp→seconds is locked.  
8. **Ports/TDS:** graph-generic for this directed account+time schema.  
9. **Split:** chronological on `TIMESTAMP`; 60/20/20 (or whole-step); train-only scalers; test sealed.  
10. **Human decisions:** timestamp unit; example-scale thesis value; never-use balances/account flags; typology eval scope; contract ID extension.

---

## Loader/formatter proposal

Design-only docs in package: `proposed_formatter_loader.md`, `proposed_split_protocol.md`, `proposed_tests.md`.  
Key: new `EdgeID` after sort; preserve `TX_ID` sidecar; shared-core dim-6; labels downstream-only; refuse ambiguous schemas; TF builder allowlist + separate cache version; tiny 100-vertex CPU smoke.

---

## Scientific comparison (vs PaySim)

| | AMLSim (here) | PaySim |
|--|---------------|--------|
| Generator diversity | High (IBM AMLSim) | High (mobile-money) |
| Task | AML alert patterns | Fraud (`isFraud`) |
| Scale on disk | Example (~1M max) | ~6.4M ready |
| Loader effort | Formatter + TF (days) | Small registry change |
| Thesis next step | After PaySim | Ready now |

---

## Byte budget (max touched)

Conservative upper bound **≈ 64.4 MiB** (both compressed archives + tiny full extract + large solid-block decompress upper bound ≤ listed 61.6 MiB uncompressed). **Under 1 GiB cap.**  
Large `transactions.csv` **not** written fully to disk; only 5-line stream. Commands logged in `bounded_samples_manifest.json`.

Temporary `7zz` used from `/tmp/jthi_amlsim_audit/` (not committed).

---

## Confirmations

No implementation, jobs, GPU, dataset/source modification, test evaluation, unbounded CSV scans, or ladder interaction.
