# Probe feature ablation

- **data:** Small-HI
- **embedding_dir:** `embeddings/gin_emlps_tds_asym_proj_8192neg_queue0_40ep_seed4`
- **categorical_encoding:** ordinal

| features | dim | groups | AUROC | AUPRC | F1 | Prec | Recall | thr | val F1 | F1@0.5 |
|----------|-----|--------|-------|-------|-----|------|--------|-----|--------|--------|
| `embedding` | 128 | embedding | 0.931 | 0.148 | 0.235 | 0.188 | 0.313 | 0.498 | 0.280 | 0.236 |
| `embedding+raw` | 132 | embedding, edge_native | 0.947 | 0.227 | 0.254 | 0.179 | 0.435 | 0.626 | 0.319 | 0.165 |
| `embedding+raw+morph` | 148 | embedding, edge_native, degree_fan, flow_balance, temporal_behavior | 0.939 | 0.246 | 0.271 | 0.197 | 0.436 | 0.391 | 0.321 | 0.300 |

Scaling: non-embedding groups use `StandardScaler` fit on **train split rows only**; SSL embeddings are passed through unscaled.
