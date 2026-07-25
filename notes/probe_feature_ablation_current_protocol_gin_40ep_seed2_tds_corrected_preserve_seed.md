# Probe feature ablation

- **data:** Small-HI
- **embedding_dir:** `embeddings/gin_emlps_ports_tds_corrected_preserve_seed_asym_proj_8192neg_queue0_40ep_seed2`
- **categorical_encoding:** ordinal

| features | dim | groups | AUROC | AUPRC | F1 | Prec | Recall | thr | val F1 | F1@0.5 |
|----------|-----|--------|-------|-------|-----|------|--------|-----|--------|--------|
| `raw` | 4 | edge_native | 0.860 | 0.009 | 0.009 | 0.004 | 0.943 | 0.014 | 0.006 | 0.000 |
| `morph` | 16 | degree_fan, flow_balance, temporal_behavior | 0.865 | 0.063 | 0.114 | 0.068 | 0.371 | 0.338 | 0.101 | 0.138 |
| `raw+morph` | 20 | edge_native, degree_fan, flow_balance, temporal_behavior | 0.905 | 0.065 | 0.135 | 0.086 | 0.315 | 0.537 | 0.099 | 0.132 |
| `embedding` | 128 | embedding | 0.962 | 0.248 | 0.309 | 0.263 | 0.374 | 0.455 | 0.291 | 0.325 |
| `embedding+raw` | 132 | embedding, edge_native | 0.962 | 0.256 | 0.324 | 0.305 | 0.345 | 0.409 | 0.303 | 0.320 |
| `embedding+raw+morph` | 148 | embedding, edge_native, degree_fan, flow_balance, temporal_behavior | 0.960 | 0.261 | 0.314 | 0.302 | 0.327 | 0.293 | 0.288 | 0.286 |

Scaling: non-embedding groups use `StandardScaler` fit on **train split rows only**; SSL embeddings are passed through unscaled.
