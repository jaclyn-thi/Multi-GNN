# EXPERT_ONLY frozen transfer — matched epoch-10 validation results

**Status:** probes complete (validation only).
**Smoke:** job 19475639 PASSED.
**Repair:** post-extract exact-count gate failure repaired via coverage-aware probe (embeddings reused; no re-extract).

## Predeclared answers

### PaySim

- Protocol: `P1_strict_inductive_legacy`
- Primary transfer signal (expert − random ≥ 0.003): **False** (ΔAUPRC=-0.021314)
- EXPERT_ONLY val AUPRC: 0.008777
- Random R198: 0.030091
- Adaptive TFMOE: 0.023702
- DIRECT_H: 0.027936
- X-only: 0.004590
- expert − adaptive: -0.014925
- expert − DIRECT_H: -0.019159
- adaptive − DIRECT_H: -0.004234
- test_evaluated: False
- Matched intersection n train/val: 3792809/1276275
- Edge coverage (min): 0.9999968361280429
- Positive coverage (min): 1.0

### SAML-D

- Protocol: `samld_frozen_expert_only_r198_valonly_v1`
- Primary transfer signal (expert − random ≥ 0.003): **True** (ΔAUPRC=0.032703)
- EXPERT_ONLY val AUPRC: 0.538498
- Random R198: 0.505795
- Adaptive TFMOE: 0.653018
- DIRECT_H: 0.600588
- X-only: 0.015381
- expert − adaptive: -0.114520
- expert − DIRECT_H: -0.062090
- adaptive − DIRECT_H: 0.052430
- test_evaluated: False
- Matched intersection n train/val: 5026799/1654261
- Edge coverage (min): 0.8708823215091368
- Positive coverage (min): 0.9365558912386707
- Cohort note: matched scored extraction cohort; incomplete relative to the locked integrity cohort because of the documented extraction_loader_coverage_defect.

## Integrity

- Full-subgraph R198; seed-only prohibited
- Locked integrity-card source counts preserved; coverage gated vs source (PaySim edge/pos ≥ 0.999; SAML-D edge ≥ 0.85, pos ≥ 0.9)
- global EdgeID unique + train∩val=0
- four-arm matched EdgeID intersection; train+val labels identical across arms
- no test.npz; BN policy frozen_aml_bn

## Artifacts

- `results/diagnostics/expert_only_frozen_transfer_samld_paysim/probe_PaySim/`
- `results/diagnostics/expert_only_frozen_transfer_samld_paysim/probe_SAML-D/`
- `results/diagnostics/expert_only_frozen_transfer_samld_paysim/gate_repair_report.md`
- `results/diagnostics/expert_only_frozen_transfer_samld_paysim/submission_probe_repair.json`

