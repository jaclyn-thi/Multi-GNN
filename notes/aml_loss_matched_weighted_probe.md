# AML loss-matched weighted PaperStyleMLP probe sensitivity

## Status

Submitted (see `results/diagnostics/aml_loss_matched_weighted_probe/submission.json`).
Final tables/answers are written by the job into this note and:

- `results/diagnostics/aml_loss_matched_weighted_probe.json`
- `results/diagnostics/aml_loss_matched_weighted_probe/`

## Guardrails

- This does not retrain or finetune any encoder.
- This is a downstream-loss sensitivity, not a replacement for the established unweighted probe.
- Supervised Multi-GIN remains end-to-end trained; probes use frozen embeddings.
- A lower unweighted CE on an extremely imbalanced cohort does not by itself imply better minority detection.
- AUPRC remains the primary ranking metric.
- F1 at a validation-selected threshold is optimistic/diagnostic.
- No test claim may be made.
- Do not select which probe protocol to report based on test performance.

## Preflight (passed)

- Loss-equivalence tests passed (`tests/test_aml_loss_matched_weighted_ce.py`)
- Class weights loaded from supervised checkpoint config (not hardcoded):
  - w0 = 1.0000182882773443
  - w1 = 6.275014431494497
- All six embedding dirs present; no `test.npz`
- Supervised val predictions from job 19458946 present
- New namespace only; no overwrite of established probes/packages
- Slurm: `partition=mit_preemptable`, `account=mit_general`, `qos=normal`, mem=128G, gpu:1, time=06:00:00
