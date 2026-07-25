# Probe feature ablation

- **data:** Small-HI
- **embedding_dir:** `embeddings/gin_emlps_ports_tds_corrected_preserve_seed_asym_proj_allneg_bs8192_accum4_10ep_seed2`
- **categorical_encoding:** ordinal

| features | dim | groups | AUROC | AUPRC | F1 | Prec | Recall | thr | val F1 | F1@0.5 |
|----------|-----|--------|-------|-------|-----|------|--------|-----|--------|--------|
| `raw` | 4 | edge_native | 0.860 | 0.009 | 0.009 | 0.004 | 0.943 | 0.014 | 0.006 | 0.000 |
| `morph` | 16 | degree_fan, flow_balance, temporal_behavior | 0.865 | 0.063 | 0.114 | 0.068 | 0.371 | 0.338 | 0.101 | 0.138 |
| `raw+morph` | 20 | edge_native, degree_fan, flow_balance, temporal_behavior | 0.905 | 0.065 | 0.135 | 0.086 | 0.315 | 0.537 | 0.099 | 0.132 |
| `embedding` | 128 | embedding | 0.936 | 0.162 | 0.244 | 0.207 | 0.298 | 0.406 | 0.281 | 0.258 |
| `embedding+raw` | 132 | embedding, edge_native | 0.950 | 0.255 | 0.264 | 0.178 | 0.508 | 0.781 | 0.305 | 0.133 |
| `embedding+raw+morph` | 148 | embedding, edge_native, degree_fan, flow_balance, temporal_behavior | 0.956 | 0.226 | 0.249 | 0.169 | 0.471 | 0.895 | 0.233 | 0.101 |

Scaling: non-embedding groups use `StandardScaler` fit on **train split rows only**; SSL embeddings are passed through unscaled.
