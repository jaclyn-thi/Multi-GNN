# Probe feature ablation

- **data:** Small-HI
- **embedding_dir:** `embeddings/hi_masked_edge_attr_gine_20ep_bestckpt`
- **categorical_encoding:** ordinal

| features | dim | groups | AUROC | F1 | Prec | Recall | thr | val F1 | F1@0.5 |
|----------|-----|--------|-------|-----|------|--------|-----|--------|--------|
| `embedding+raw` | 132 | embedding, edge_native | 0.943 | 0.114 | 0.062 | 0.664 | 0.917 | 0.225 | 0.032 |

Scaling: non-embedding groups use `StandardScaler` fit on **train split rows only**; SSL embeddings are passed through unscaled.
