# Small-LI Current-Protocol Dataset Comparison

Small-LI scout for the current GINe emlps+tds SSL recipe. This is a dataset comparison, not a frozen benchmark.

Run: `slurm/scout_small_li_gin_emlps_tds_asym_proj_8192neg_queue0_20ep_seed1.sh` (job 17031798). Best checkpoint was **epoch 19**; last epoch (20) was extracted/probed separately and was worse on embedding-only (**0.029 F1 / 0.008 AUPRC**), so tables below report the best checkpoint.

## Dataset Audit

- Dataset key: `Small-LI`
- CSV: `aml-data/Small-LI/formatted_transactions.csv`
- Edges / nodes: 6,924,049 / 705,907
- Label: `Is Laundering`; overall positive rate 0.0515%
- Split mode: `calendar_day`; train/val/test positive rates 0.0450% / 0.0585% / 0.0683%
- Pattern metadata: not found
- Raw/morph generation: ok

## Small-LI Probes

Primary probe policy: `--class_weight model --model gin` (shared GIN weights). Metrics are test AUROC / AUPRC / F1 / F1@0.5.

| Features | Metrics | Threshold | Precision | Recall |
|----------|---------|----------:|----------:|-------:|
| `raw` | 0.724 / 0.009 / 0.008 / 0.000 | 0.019 | 0.004 | 0.111 |
| `morph` | 0.847 / 0.015 / 0.053 / 0.056 | 0.379 | 0.034 | 0.120 |
| `raw+morph` | 0.858 / 0.016 / 0.057 / 0.050 | 0.688 | 0.046 | 0.074 |
| `embedding` | 0.899 / 0.017 / 0.052 / 0.052 | 0.152 | 0.032 | 0.147 |
| `embedding+raw` | 0.909 / 0.027 / 0.076 / 0.081 | 0.473 | 0.054 | 0.127 |
| `embedding+raw+morph` | 0.925 / 0.039 / 0.056 / 0.073 | 0.327 | 0.031 | 0.305 |

Exploratory secondary probe (`cw=none`) is saved separately at `results/diagnostics/probe_feature_ablation_small_li_current_protocol_seed1_cw_none.json`.

## Small-HI References

| Reference | Metrics |
|-----------|---------|
| Small-HI GINe emlps+tds 20ep seed1 `embedding` | 0.944 / 0.213 / 0.259 / 0.257 |
| Small-HI GINe emlps+tds 20ep seed1 `embedding+raw+morph` | 0.945 / 0.276 / 0.298 / 0.327 |
| Small-HI FNF seed1 `embedding+raw+morph` | 0.959 / 0.276 / 0.319 / 0.303 |
| Small-HI `raw+morph` only | 0.905 / 0.066 / 0.136 / 0.132 |

## Interpretation

- Small-LI has lower test positive prevalence than the Small-HI reference (0.0683% vs 0.1867%), so AUPRC/F1 should be read with that class-balance shift in mind.
- Raw+morph-only matches or beats embedding-only on F1; inspect whether SSL transfer is weaker on Small-LI.
- `embedding+raw` changes F1 vs embedding-only by +0.023.
- Adding morphology on top of embedding+raw changes F1 by -0.020.

Artifacts:

- Audit: `results/diagnostics/small_li_dataset_audit.json`
- Embeddings: `embeddings/small_li_gin_emlps_tds_asym_proj_8192neg_queue0_20ep_seed1/`
- Feature ablation: `results/diagnostics/probe_feature_ablation_small_li_current_protocol_seed1.json`
- Last-checkpoint embedding probe: `embeddings/small_li_gin_emlps_tds_asym_proj_8192neg_queue0_20ep_seed1_last_ckpt/probe_results.json`
