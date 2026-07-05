# Probe feature ablation

- **data:** Small-HI
- **embedding_dir:** `embeddings/degree_aware_edgedrop_emlps_tds_asym_proj_8192neg_queue0_20ep`
- **categorical_encoding:** ordinal

| features | dim | groups | AUROC | AUPRC | F1 | Prec | Recall | thr | val F1 | F1@0.5 |
|----------|-----|--------|-------|-------|-----|------|--------|-----|--------|--------|
| `raw` | 4 | edge_native | 0.860 | 0.009 | 0.009 | 0.004 | 0.943 | 0.014 | 0.006 | 0.000 |
| `morph` | 16 | degree_fan, flow_balance, temporal_behavior | 0.865 | 0.064 | 0.114 | 0.068 | 0.370 | 0.337 | 0.101 | 0.139 |
| `embedding` | 128 | embedding | 0.926 | 0.153 | 0.240 | 0.234 | 0.246 | 0.410 | 0.230 | 0.246 |
| `embedding+raw` | 132 | embedding, edge_native | 0.945 | 0.238 | 0.238 | 0.168 | 0.412 | 0.446 | 0.260 | 0.265 |
| `embedding+raw+morph` | 148 | embedding, edge_native, degree_fan, flow_balance, temporal_behavior | 0.957 | 0.253 | 0.291 | 0.218 | 0.435 | 0.704 | 0.255 | 0.207 |

Scaling: non-embedding groups use `StandardScaler` fit on **train split rows only**; SSL embeddings are passed through unscaled.
