# Probe feature ablation

- **data:** Small-HI
- **embedding_dir:** `embeddings/gin_emlps_ports_tds_corrected_preserve_seed_asym_proj_allneg_bs2048_accum16_10ep_seed2`
- **categorical_encoding:** ordinal

| features | dim | groups | AUROC | AUPRC | F1 | Prec | Recall | thr | val F1 | F1@0.5 |
|----------|-----|--------|-------|-------|-----|------|--------|-----|--------|--------|
| `raw` | 4 | edge_native | 0.860 | 0.009 | 0.009 | 0.004 | 0.943 | 0.014 | 0.006 | 0.000 |
| `morph` | 16 | degree_fan, flow_balance, temporal_behavior | 0.865 | 0.064 | 0.114 | 0.068 | 0.372 | 0.338 | 0.101 | 0.138 |
| `raw+morph` | 20 | edge_native, degree_fan, flow_balance, temporal_behavior | 0.905 | 0.065 | 0.135 | 0.086 | 0.310 | 0.543 | 0.100 | 0.132 |
| `embedding` | 128 | embedding | 0.946 | 0.150 | 0.255 | 0.230 | 0.287 | 0.536 | 0.263 | 0.250 |
| `embedding+raw` | 132 | embedding, edge_native | 0.954 | 0.217 | 0.280 | 0.211 | 0.417 | 0.714 | 0.300 | 0.188 |
| `embedding+raw+morph` | 148 | embedding, edge_native, degree_fan, flow_balance, temporal_behavior | 0.959 | 0.200 | 0.286 | 0.246 | 0.341 | 0.817 | 0.270 | 0.189 |

Scaling: non-embedding groups use `StandardScaler` fit on **train split rows only**; SSL embeddings are passed through unscaled.
