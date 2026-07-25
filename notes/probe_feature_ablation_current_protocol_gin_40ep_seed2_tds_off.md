# Probe feature ablation

- **data:** Small-HI
- **embedding_dir:** `embeddings/gin_emlps_ports_tds_off_asym_proj_8192neg_queue0_40ep_seed2`
- **categorical_encoding:** ordinal

| features | dim | groups | AUROC | AUPRC | F1 | Prec | Recall | thr | val F1 | F1@0.5 |
|----------|-----|--------|-------|-------|-----|------|--------|-----|--------|--------|
| `raw` | 4 | edge_native | 0.860 | 0.009 | 0.009 | 0.004 | 0.943 | 0.014 | 0.006 | 0.000 |
| `morph` | 16 | degree_fan, flow_balance, temporal_behavior | 0.865 | 0.064 | 0.115 | 0.068 | 0.371 | 0.340 | 0.101 | 0.138 |
| `raw+morph` | 20 | edge_native, degree_fan, flow_balance, temporal_behavior | 0.905 | 0.066 | 0.136 | 0.087 | 0.310 | 0.544 | 0.100 | 0.132 |
| `embedding` | 128 | embedding | 0.929 | 0.116 | 0.193 | 0.173 | 0.217 | 0.460 | 0.225 | 0.194 |
| `embedding+raw` | 132 | embedding, edge_native | 0.948 | 0.174 | 0.203 | 0.137 | 0.396 | 0.637 | 0.265 | 0.159 |
| `embedding+raw+morph` | 148 | embedding, edge_native, degree_fan, flow_balance, temporal_behavior | 0.942 | 0.155 | 0.196 | 0.133 | 0.376 | 0.567 | 0.219 | 0.177 |

Scaling: non-embedding groups use `StandardScaler` fit on **train split rows only**; SSL embeddings are passed through unscaled.
