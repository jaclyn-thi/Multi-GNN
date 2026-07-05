# Probe feature ablation

- **data:** Small-HI
- **embedding_dir:** `embeddings/gin_emlps_tds_asym_proj_8192neg_queue0_40ep_seed2`
- **categorical_encoding:** ordinal

| features | dim | groups | AUROC | AUPRC | F1 | Prec | Recall | thr | val F1 | F1@0.5 |
|----------|-----|--------|-------|-------|-----|------|--------|-----|--------|--------|
| `raw` | 4 | edge_native | 0.860 | 0.009 | 0.009 | 0.004 | 0.943 | 0.014 | 0.006 | 0.000 |
| `morph` | 16 | degree_fan, flow_balance, temporal_behavior | 0.865 | 0.063 | 0.114 | 0.068 | 0.371 | 0.338 | 0.101 | 0.138 |
| `raw+morph` | 20 | edge_native, degree_fan, flow_balance, temporal_behavior | 0.905 | 0.065 | 0.135 | 0.086 | 0.315 | 0.537 | 0.099 | 0.132 |
| `embedding` | 128 | embedding | 0.949 | 0.242 | 0.300 | 0.279 | 0.325 | 0.395 | 0.281 | 0.305 |
| `embedding+raw` | 132 | embedding, edge_native | 0.956 | 0.292 | 0.347 | 0.315 | 0.386 | 0.563 | 0.307 | 0.330 |
| `embedding+raw+morph` | 148 | embedding, edge_native, degree_fan, flow_balance, temporal_behavior | 0.958 | 0.219 | 0.275 | 0.211 | 0.397 | 0.408 | 0.252 | 0.283 |

Scaling: non-embedding groups use `StandardScaler` fit on **train split rows only**; SSL embeddings are passed through unscaled.
