# Probe feature ablation

- **data:** Small-HI
- **embedding_dir:** `embeddings/hi_contrastive_proj_asym_8192neg_queue0_accum4_20ep_bestckpt`
- **categorical_encoding:** ordinal

| features | dim | groups | AUROC | F1 | Prec | Recall | thr | val F1 | F1@0.5 |
|----------|-----|--------|-------|-----|------|--------|-----|--------|--------|
| `raw` | 4 | edge_native | 0.860 | 0.009 | 0.004 | 0.943 | 0.014 | 0.006 | 0.000 |
| `morph` | 16 | degree_fan, flow_balance, temporal_behavior | 0.865 | 0.115 | 0.068 | 0.371 | 0.338 | 0.101 | 0.139 |
| `embedding` | 128 | embedding | 0.951 | 0.236 | 0.193 | 0.301 | 0.333 | 0.233 | 0.215 |
| `embedding+raw` | 132 | embedding, edge_native | 0.953 | 0.257 | 0.241 | 0.274 | 0.393 | 0.252 | 0.238 |
| `embedding+raw+morph` | 148 | embedding, edge_native, degree_fan, flow_balance, temporal_behavior | 0.952 | 0.233 | 0.204 | 0.273 | 0.507 | 0.169 | 0.231 |

Scaling: non-embedding groups use `StandardScaler` fit on **train split rows only**; SSL embeddings are passed through unscaled.
