# Multi-dataset expansion compatibility and resource audit

> Twin: `results/diagnostics/multidataset_expansion_compatibility_resource_audit.json`
> Created: 2026-08-02T23:20:20.447207+00:00

**Read-only planning audit.** No encoder training, extraction, probes, test evaluation,
dataset rewrites, destructive ops, or GPU/Slurm jobs.

## Executive recommendations

1. **Third training domain:** Small-LI
2. **First independent domain after SAML-D:** PaySim (leakage-safe shared-core)
3. **If one Medium:** Medium-HI
4. **Include leakage-safe PaySim?** Yes — for generator diversity (shared-core drops type-dup categoricals).
5. **AMLSim attempt?** Yes, bounded (example archives); formatter/loader first — not a blocker by absence of code.
6. **Max realistic domains under current mem design:** 3 simultaneous; Medium needs special handling.
7. **N-domain trainer before next smoke?** **Yes.**
8. **Next implementation:** Extend SHARED_CORE_DATASETS + Phase-4 domain registry; wire existing Small-LI TF cache; no new TF build required for LI. For PaySim: register existing paysim TF cache under shared-core path.
9. **3-domain smoke:** `['Small-HI', 'SAML-D', 'Small-LI']`, 500/domain, 1500 mixed steps, 128G/1GPU, ~0.99 h mixed wall.
10. **Final collection:** ['Small-HI', 'SAML-D', 'Small-LI', 'PaySim']
11. **Human decisions:** Medium worth?, PaySim fraud≠AML claims, AMLSim vs PaySim priority, embedding storage, >128G for Medium.

## Family distinction (critical)

- **AMLWorld variants (same simulator):** Small-HI, Small-LI, Medium-HI, Medium-LI — **not** four independent sources.
- **Independent families:** SAML-D, PaySim, AMLSim (archives), plus on-disk AMLNet/BankSim (out of scope).
- **Label semantics differ:** AML laundering vs PaySim fraud vs AMLSim unresolved.

## Reference system

Contract `smallhi_samld_shared_core_v1` → `('Timestamp', 'Amount Received', 'in_port', 'out_port', 'in_td', 'out_td')`.
Shared GIN, R198 InfoNCE, projection off, adaptive TF-MoE (3 causal targets),
per-domain edge/TF scalers, LossNorm, BN; shared encoder/affine/experts; no test access.

## Shared-core compatibility summary

| Dataset | Class | Notes |
|---|---|---|
| Small-HI | DIRECT_COMPATIBLE | reference |
| SAML-D | DIRECT_COMPATIBLE | reference; EdgeID≠row index |
| Small-LI | DIRECT_COMPATIBLE | loader+TF exist; allowlist/registry only |
| Medium-HI/LI | DIRECT_COMPATIBLE | scale/TF-builder ops; not independent diversity |
| PaySim | DIRECT_COMPATIBLE | leakage-safe core from step/amount/ports/TDS |
| AMLSim | DATA_MISSING | extract+formatter+loader = implementation work |

Ports/TDS algorithms in `data_util.py` are graph-generic; reverse-edge swap via `correct_reverse_edge_features`.

## PaySim leakage-safe protocol

Exclude balances, `isFlaggedFraud`, and fraud labels from all encoder inputs.
Preserve source warning about cancelled-fraud balances.
Do not treat type-dup currency/payment slots as real AML fields; shared-core omits them.
All-native PaySim contracts remain diagnostic-only.

## Resource anchors (measured)

- Small-HI: 1.886 s/step (1000-step arm); TF ~183MB clean; emb train+val ~3.11 GiB
- SAML-D: 2.836 s/step; TF 262MB; emb train+val ~5.62 GiB
- Phase-3 HI+SAML simultaneous residency: **OK under 128G** (standard account).
- Phase-3 frozen-eval embeddings retained: **~25 GiB** for 3 encoders × 2 targets.
- Checkpoints: ~2.8–2.9 MB/file.
- Slurm: `mit_preemptable` / `mit_general` / `qos=normal`; do not assume expired advanced limits.

## Resident combinations (heuristic host GiB sum)

| Combo | Est RAM GiB | 128G OK? | OK w/ fewer workers? | Mixed steps | Est mixed wall h |
|---|---:|---|---|---:|---:|
| 1_HI_SAML_LI | 130 | False | True | 1500 | 0.99 |
| 2_HI_SAML_PaySim | 130 | False | True | 1500 | 0.94 |
| 3_HI_SAML_LI_PaySim | 160 | False | False | 2000 | 1.27 |
| 4_HI_SAML_MediumHI | 250 | False | False | 1500 | 2.3 |
| 5_HI_SAML_LI_MediumHI_PaySim | 310 | False | False | 2500 | 2.92 |
| 6_all_available_formatted | 460 | False | False | 3000 | 4.53 |

HI+SAML is measured OK at 128G. Three Small-scale domains are the practical max;
try 128G with reduced loader workers before requesting more memory.
Medium-inclusive combos need special residency. Alternatives by faithfulness:
fewer domains > lower workers > mmap > block cyclic > sharding.

## Rankings

1. Easiest next: **Small-LI**
2. Strongest diversity: **PaySim (available now); AMLSim after formatter**
3. Strongest scale: **Medium-HI**
4. Best overall next: **Small-LI**
5. Best realistic final set: **Small-HI + SAML-D + Small-LI + PaySim**

Per-axis scores (1–5) are in the twin JSON under `scores` (not collapsed to one number).

## N-domain trainer

Phase-3 hardcodes exactly two domains / 1:1 / three arms / fixed caches.
See `results/diagnostics/multidataset_expansion_compatibility_resource_audit/trainer_generalization_plan.md`.

## AMLSim

See `results/diagnostics/multidataset_expansion_compatibility_resource_audit/amlsim_loader_plan.md`.
Archives only; example-scale; formatter/loader are implementation work, not automatic blockers.

## Optional CPU metadata job

**Not submitted.** Existing metadata suffices for compatibility and ranking decisions.
A later streaming Medium count job is optional (see twin JSON `optional_cpu_metadata_job`).

## Confirmations

- no encoder training / embedding extraction / probes / test evaluation
- no destructive ops / no GPU jobs / no Slurm jobs
- no loader/formatter/model/training code modified
- no datasets modified
- no unbounded full CSV scans

