# Final transfer capacity audit

> Exploratory / post-hoc. `table_eligible=false`. `exploratory_posthoc=true`.
> Twin JSON will be written by the aggregate job to `results/diagnostics/final_transfer_capacity_audit.json`.

## State snapshot (implementation)

- git HEAD: `61efb9004f4d75d22386babc200d0181497ae26b`
- Locked encoder ckpt: `saved-models/checkpoint_gin_emlps_ports_tds_corrected_asym_proj_8192neg_queue0_40ep_seed2.tar`
- SHA256: `18e06f555aa4880dfc1e95caa3f54a207e5aa186d266887772640feb93a06ae6`
- Feature contract (PaySim): `paysim_legacy_duplicate_v1`
- Primary embeddings: `embeddings/final_corrected_no_preserve_multiseed/seed{1–4}_P1_strict_inductive_legacy`
- Sensitivity: `…/seed{1–4}_P2_label_free_target_bn_legacy`
- Random control: `…/controls_random_paysim_legacy_duplicate_v1`
- Expected val ID hash: `a8de85f31dfe91bd767da6daedf9f2bab474d08c8412c796111e8767ebd0b1e3`

## Proposed / submitted DAG

| Job | Resources | Dependency |
|-----|-----------|------------|
| smoke | 16G / 2c / 30m | — |
| shift (Part A) | 128G / 16c / 6h | afterok:smoke |
| capacity_fit (Part B cells) | 256G / 16c / 10h | afterok:smoke (parallel with shift) |
| gate | 8G / 2c / 30m | afterok:capacity_fit (exit 0 on scientific fail) |
| confirm | 128G / 8c / 4h | afterok:gate (self-skips if gate fail) |
| aggregate | 8G / 2c / 30m | afterok:shift+confirm |

Partition/account: `mit_normal` + `mit_amf_advanced_cpu` (CPU MaxTime cap typically 12:00:00). No GPU jobs in this DAG.

Tree learner: `HistGradientBoostingClassifier` (predeclared; xgboost/lightgbm not installed).

## Results

Pending Slurm completion. See `notes/final_transfer_capacity_audit_submission.md` and `results/diagnostics/final_transfer_capacity_audit/submission.json` after successful `sbatch`.
