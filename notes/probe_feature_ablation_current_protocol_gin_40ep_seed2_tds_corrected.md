# Probe feature ablation

- **data:** Small-HI
- **embedding_dir:** `embeddings/gin_emlps_ports_tds_corrected_asym_proj_8192neg_queue0_40ep_seed2`
- **categorical_encoding:** ordinal

| features | dim | groups | AUROC | AUPRC | F1 | Prec | Recall | thr | val F1 | F1@0.5 |
|----------|-----|--------|-------|-------|-----|------|--------|-----|--------|--------|
| `raw` | 4 | edge_native | 0.860 | 0.009 | 0.009 | 0.004 | 0.943 | 0.014 | 0.006 | 0.000 |
| `morph` | 16 | degree_fan, flow_balance, temporal_behavior | 0.865 | 0.063 | 0.114 | 0.068 | 0.371 | 0.338 | 0.101 | 0.138 |
| `raw+morph` | 20 | edge_native, degree_fan, flow_balance, temporal_behavior | 0.905 | 0.065 | 0.135 | 0.086 | 0.315 | 0.537 | 0.099 | 0.132 |
| `embedding` | 128 | embedding | 0.950 | 0.134 | 0.195 | 0.138 | 0.328 | 0.511 | 0.259 | 0.190 |
| `embedding+raw` | 132 | embedding, edge_native | 0.959 | 0.213 | 0.228 | 0.155 | 0.431 | 0.692 | 0.292 | 0.138 |
| `embedding+raw+morph` | 148 | embedding, edge_native, degree_fan, flow_balance, temporal_behavior | 0.963 | 0.213 | 0.266 | 0.212 | 0.359 | 0.687 | 0.270 | 0.226 |

Scaling: non-embedding groups use `StandardScaler` fit on **train split rows only**; SSL embeddings are passed through unscaled.
