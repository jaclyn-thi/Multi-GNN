# Probe feature ablation

- **data:** Small-HI
- **embedding_dir:** `embeddings/gin_emlps_tds_asym_proj_8192neg_queue0_40ep_seed1`
- **categorical_encoding:** ordinal

| features | dim | groups | AUROC | AUPRC | F1 | Prec | Recall | thr | val F1 | F1@0.5 |
|----------|-----|--------|-------|-------|-----|------|--------|-----|--------|--------|
| `raw` | 4 | edge_native | 0.860 | 0.009 | 0.009 | 0.004 | 0.943 | 0.014 | 0.006 | 0.000 |
| `morph` | 16 | degree_fan, flow_balance, temporal_behavior | 0.865 | 0.064 | 0.114 | 0.068 | 0.370 | 0.337 | 0.101 | 0.139 |
| `raw+morph` | 20 | edge_native, degree_fan, flow_balance, temporal_behavior | 0.905 | 0.066 | 0.136 | 0.087 | 0.318 | 0.537 | 0.099 | 0.132 |
| `embedding` | 128 | embedding | 0.948 | 0.199 | 0.292 | 0.310 | 0.276 | 0.502 | 0.300 | 0.292 |
| `embedding+raw` | 132 | embedding, edge_native | 0.961 | 0.316 | 0.269 | 0.177 | 0.565 | 0.786 | 0.315 | 0.134 |
| `embedding+raw+morph` | 148 | embedding, edge_native, degree_fan, flow_balance, temporal_behavior | 0.945 | 0.264 | 0.262 | 0.180 | 0.484 | 0.544 | 0.309 | 0.248 |

Scaling: non-embedding groups use `StandardScaler` fit on **train split rows only**; SSL embeddings are passed through unscaled.
