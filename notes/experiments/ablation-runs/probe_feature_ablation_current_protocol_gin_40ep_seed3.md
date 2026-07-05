# Probe feature ablation

- **data:** Small-HI
- **embedding_dir:** `embeddings/gin_emlps_tds_asym_proj_8192neg_queue0_40ep_seed3`
- **categorical_encoding:** ordinal

| features | dim | groups | AUROC | AUPRC | F1 | Prec | Recall | thr | val F1 | F1@0.5 |
|----------|-----|--------|-------|-------|-----|------|--------|-----|--------|--------|
| `embedding` | 128 | embedding | 0.939 | 0.176 | 0.225 | 0.200 | 0.258 | 0.496 | 0.232 | 0.226 |
| `embedding+raw` | 132 | embedding, edge_native | 0.952 | 0.236 | 0.217 | 0.139 | 0.498 | 0.793 | 0.274 | 0.114 |
| `embedding+raw+morph` | 148 | embedding, edge_native, degree_fan, flow_balance, temporal_behavior | 0.953 | 0.196 | 0.187 | 0.116 | 0.493 | 0.693 | 0.239 | 0.136 |

Scaling: non-embedding groups use `StandardScaler` fit on **train split rows only**; SSL embeddings are passed through unscaled.
