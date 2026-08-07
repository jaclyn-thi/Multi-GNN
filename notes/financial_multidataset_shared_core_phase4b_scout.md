# Phase-4B financial multi-dataset shared-core scout

> Twin: [`results/diagnostics/financial_multidataset_shared_core_phase4b_scout.json`](../results/diagnostics/financial_multidataset_shared_core_phase4b_scout.json)  
> Integrity: [`training_integrity_summary.json`](../results/diagnostics/financial_multidataset_shared_core_phase4b_scout/training_integrity_summary.json)

**Training-integrity scout only.** No extraction, probes, test evaluation, or frozen-eval DAG submitted.

## Verdict

Both authorized arms **PASS** all training-integrity gates. Phase-3 shared init reused (`8821c986…`). First-500 Small-LI batches match across arms; first-500 Small-HI / SAML-D batches match Phase-3 `MIXED_1TO1`. Stop for human review before any frozen eval.

## Jobs

| Arm | Job | State | Wall | MaxRSS |
|---|---|---|---|---|
| MIXED_3DOMAIN | [`19532905_0`](../slurm/logs/phase4b_scout_19532905_0.out) (task `19532910`) | COMPLETED | 01:04:30 | 10.71 GiB |
| SMALL_LI_ONLY | [`19532951`](../slurm/logs/phase4b_scout_li_only_19532951.out) | COMPLETED | 00:33:23 | 7.25 GiB |
| SMALL_LI_ONLY (failed) | [`19532905_1`](../slurm/logs/phase4b_scout_19532905_1.err) | FAILED 15s | — | race on shared JSON write; fixed; **manual** resubmit only |

## Resolved recipe

- Contract: `financial_multidataset_shared_core_v1` · `edge_dim=6` · GIN · R198 InfoNCE · adaptive TF-MoE · `projection=false` · `AMP=false` · `workers=0`
- LR: Phase-3 **20%** warmup (not the provisional 10% guess) → MIXED 300/1200; LI-only 200/800
- α/β: MIXED freeze through 15 / update 16; LI-only freeze through 10 / update 11 (Phase-3 specialist convention)

## Stream matching

| Check | Result |
|---|---|
| MIXED vs LI-only first 500 Small-LI | **equal** |
| MIXED vs Phase-3 MIXED first 500 Small-HI | **equal** (`a8dc63a6…`) |
| MIXED vs Phase-3 MIXED first 500 SAML-D | **equal** (`d8f063f8…`) |

## Training losses (diagnostic only — not downstream quality)

| Arm | Final α / w_contrast | Final L_total |
|---|---|---|
| MIXED_3DOMAIN @1500 | 0.370 | 0.506 |
| SMALL_LI_ONLY @1000 | 0.446 | 0.542 |

Do **not** infer representation or AML quality from these curves.

## Proposed frozen eval (unsubmitted)

Four new cells only — see [`proposed_frozen_eval.json`](../results/diagnostics/financial_multidataset_shared_core_phase4b_scout/proposed_frozen_eval.json). Phase-3 validation reuse requires documented protocol/feature/scaler/probe comparability first.

## Confirmations

- no extraction / probe / test / PaySim / Medium / AMLSim / adapter / extra seed / DAG
