# Probe feature ablation

- **data:** Small-HI
- **embedding_dir:** `embeddings/fnf_emlps_tds_asym_proj_8192neg_queue0_20ep_seed2`
- **categorical_encoding:** ordinal

| features | dim | groups | AUROC | AUPRC | F1 | Prec | Recall | thr | val F1 | F1@0.5 |
|----------|-----|--------|-------|-------|-----|------|--------|-----|--------|--------|
| `raw` | 4 | edge_native | 0.860 | 0.009 | 0.009 | 0.004 | 0.943 | 0.014 | 0.006 | 0.000 |
| `morph` | 16 | degree_fan, flow_balance, temporal_behavior | 0.865 | 0.063 | 0.114 | 0.068 | 0.371 | 0.338 | 0.101 | 0.138 |
| `raw+morph` | 20 | edge_native, degree_fan, flow_balance, temporal_behavior | 0.905 | 0.065 | 0.135 | 0.086 | 0.315 | 0.537 | 0.099 | 0.132 |
| `embedding` | 128 | embedding | 0.926 | 0.137 | 0.206 | 0.171 | 0.259 | 0.415 | 0.203 | 0.218 |
| `embedding+raw` | 132 | embedding, edge_native | 0.947 | 0.221 | 0.230 | 0.158 | 0.426 | 0.639 | 0.239 | 0.179 |
| `embedding+raw+morph` | 148 | embedding, edge_native, degree_fan, flow_balance, temporal_behavior | 0.955 | 0.243 | 0.262 | 0.189 | 0.429 | 0.743 | 0.212 | 0.170 |

Scaling: non-embedding groups use `StandardScaler` fit on **train split rows only**; SSL embeddings are passed through unscaled.
