# Probe feature ablation

- **data:** Small-HI
- **embedding_dir:** `embeddings/hi_contrastive_gin_emlps_tds_proj_asym_8192neg_queue0_accum4_20ep`
- **categorical_encoding:** ordinal

| features | dim | groups | AUROC | AUPRC | F1 | Prec | Recall | thr | val F1 | F1@0.5 |
|----------|-----|--------|-------|-------|-----|------|--------|-----|--------|--------|
| `raw` | 4 | edge_native | 0.860 | 0.009 | 0.009 | 0.004 | 0.943 | 0.014 | 0.006 | 0.000 |
| `morph` | 16 | degree_fan, flow_balance, temporal_behavior | 0.865 | 0.064 | 0.114 | 0.068 | 0.370 | 0.337 | 0.101 | 0.139 |
| `embedding` | 128 | embedding | 0.944 | 0.213 | 0.259 | 0.231 | 0.295 | 0.528 | 0.274 | 0.257 |
| `embedding+raw` | 132 | embedding, edge_native | 0.949 | 0.244 | 0.274 | 0.212 | 0.389 | 0.521 | 0.300 | 0.268 |
| `embedding+raw+morph` | 148 | embedding, edge_native, degree_fan, flow_balance, temporal_behavior | 0.945 | 0.276 | 0.298 | 0.231 | 0.420 | 0.366 | 0.325 | 0.327 |

Scaling: non-embedding groups use `StandardScaler` fit on **train split rows only**; SSL embeddings are passed through unscaled.
