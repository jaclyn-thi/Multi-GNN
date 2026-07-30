# SAML-D supervised smoke v2 (protocol gate)

- **Gate pass:** `True`
- **Run name:** `samld_supervised_multigin_eu_v1_smoke_v2_seed2`
- **Job:** `19114673+gate19116341`
- **Protocol:** `samld_supervised_multigin_eu_v1` (Candidate A)
- **Prior deviant job:** 19109396 (embedding head / no save_model / test every epoch)

## Corrections vs 19109396

- `--supervised_head legacy`
- `--save_model`
- `--skip_test_eval` (no test graph materialization; no test metrics in history)
- Unique run name `*_smoke_v2_seed2`

## Checks

- `no_historical_overwrite`: **PASS**
- `supervised_head_legacy`: **PASS**
- `candidate_a_flags`: **PASS**
- `checkpoints_reload`: **PASS**
- `best_vs_last`: **PASS**
- `finite_losses`: **PASS**
- `validation_protocol_card`: **PASS**
- `test_evaluated_false`: **PASS**

## Checkpoints

- best sha256: `9c483b43c8dbf0c7a0c54ab111f92456a851eeee67079962d2a4d33474a2e1e7`
- last sha256: `30a6562cf80a15fece267628510dc5d7ac1f3a0b87082751e82a5c2ed47ee86b`
- hashes differ: `True`

## Epoch collapse diagnostic

```json
{
  "val_f1_by_epoch": [
    0.8801543284301906,
    0.9044324878298744
  ],
  "val_auprc_by_epoch": [
    0.9839860618615666,
    0.9585079160939624
  ],
  "epoch2_f1_near_zero": false,
  "epoch2_auprc_near_prevalence": false
}
```

## Formal 50-epoch

**Not authorized** by this gate. Do not auto-submit.

Twin JSON: `results/diagnostics/samld_supervised_smoke_v2.json`
