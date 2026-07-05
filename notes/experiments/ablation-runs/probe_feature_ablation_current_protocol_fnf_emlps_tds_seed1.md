# Probe feature ablation

- **data:** Small-HI
- **embedding_dir:** `embeddings/same_pair_fnf_emlps_tds_asym_proj_8192neg_queue0_20ep`
- **categorical_encoding:** ordinal

| features | dim | groups | AUROC | AUPRC | F1 | Prec | Recall | thr | val F1 | F1@0.5 |
|----------|-----|--------|-------|-------|-----|------|--------|-----|--------|--------|
| `raw` | 4 | edge_native | 0.860 | 0.009 | 0.009 | 0.004 | 0.943 | 0.014 | 0.006 | 0.000 |
| `morph` | 16 | degree_fan, flow_balance, temporal_behavior | 0.865 | 0.064 | 0.114 | 0.068 | 0.370 | 0.337 | 0.101 | 0.139 |
| `raw+morph` | 20 | edge_native, degree_fan, flow_balance, temporal_behavior | 0.905 | 0.066 | 0.136 | 0.087 | 0.318 | 0.537 | 0.099 | 0.132 |
| `embedding` | 128 | embedding | 0.942 | 0.179 | 0.236 | 0.205 | 0.278 | 0.479 | 0.248 | 0.239 |
| `embedding+raw` | 132 | embedding, edge_native | 0.952 | 0.223 | 0.256 | 0.185 | 0.413 | 0.488 | 0.257 | 0.261 |
| `embedding+raw+morph` | 148 | embedding, edge_native, degree_fan, flow_balance, temporal_behavior | 0.959 | 0.276 | 0.319 | 0.260 | 0.413 | 0.566 | 0.265 | 0.303 |

Scaling: non-embedding groups use `StandardScaler` fit on **train split rows only**; SSL embeddings are passed through unscaled.
