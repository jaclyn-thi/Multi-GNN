# Probe feature ablation

- **data:** Small-HI
- **embedding_dir:** `embeddings/gin_emlps_ports_tds_off_preserve_seed_asym_proj_8192neg_queue0_40ep_seed2`
- **categorical_encoding:** ordinal

| features | dim | groups | AUROC | AUPRC | F1 | Prec | Recall | thr | val F1 | F1@0.5 |
|----------|-----|--------|-------|-------|-----|------|--------|-----|--------|--------|
| `raw` | 4 | edge_native | 0.860 | 0.009 | 0.009 | 0.004 | 0.943 | 0.014 | 0.006 | 0.000 |
| `morph` | 16 | degree_fan, flow_balance, temporal_behavior | 0.865 | 0.064 | 0.115 | 0.068 | 0.371 | 0.340 | 0.101 | 0.138 |
| `raw+morph` | 20 | edge_native, degree_fan, flow_balance, temporal_behavior | 0.905 | 0.066 | 0.136 | 0.087 | 0.310 | 0.544 | 0.100 | 0.132 |
| `embedding` | 128 | embedding | 0.936 | 0.158 | 0.239 | 0.264 | 0.218 | 0.505 | 0.249 | 0.239 |
| `embedding+raw` | 132 | embedding, edge_native | 0.954 | 0.247 | 0.290 | 0.238 | 0.372 | 0.710 | 0.272 | 0.147 |
| `embedding+raw+morph` | 148 | embedding, edge_native, degree_fan, flow_balance, temporal_behavior | 0.954 | 0.226 | 0.275 | 0.221 | 0.363 | 0.498 | 0.254 | 0.275 |

Scaling: non-embedding groups use `StandardScaler` fit on **train split rows only**; SSL embeddings are passed through unscaled.
