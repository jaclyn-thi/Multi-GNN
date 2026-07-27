# PaySim feature-contract validation gate (seed-2)

> **Scope:** seed-2 validation gate only. **Not** a final transfer result. PaySim test was never evaluated.

- Checkpoint: `checkpoint_gin_emlps_ports_tds_corrected_asym_proj_8192neg_queue0_40ep_seed2.tar` sha256 `18e06f555aa4…`
- Selected contract: **`paysim_legacy_duplicate_v1`**
- Numerical winner (val AUPRC): `paysim_legacy_duplicate_v1`

## Validation metrics (pretrained vs random)

| Contract | Pre AUROC | Pre AUPRC | Pre F1@0.5 | Pre F1@val-opt | Rand AUPRC | Δ AUPRC |
|----------|----------:|----------:|-----------:|---------------:|-----------:|--------:|
| `paysim_legacy_duplicate_v1` | 0.9045 | 0.0222 | 0.0517 | 0.0848 | 0.0115 | +0.0107 |
| `paysim_type_only_v1` | 0.8776 | 0.0145 | 0.0548 | 0.0641 | 0.0088 | +0.0057 |
| `paysim_structure_only_v1` | 0.6036 | 0.0085 | 0.0546 | 0.0575 | 0.0077 | +0.0009 |

## Selection notes

- paysim_legacy_duplicate_v1 beats matched random on val AUPRC (0.0222 > 0.0115)

## Protocol

- Frozen corrected/no-preserve seed-2 GIN; post-128 H; train-fit z-norm; frozen AML BN
- LogisticRegression `class_weight=model`, `C=1`; validation metrics only
- Matched random edge-dim-8 re-extracted under each contract

