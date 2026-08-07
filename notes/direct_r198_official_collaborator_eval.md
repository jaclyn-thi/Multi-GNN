# Official DIRECT_R198 collaborator evaluation

**Use this path for any collaborator-facing metrics.**  
Seed-only evaluation is diagnostic/provisional only and cannot merge into collaborator tables.

## One command (official)

```bash
# Single checkpoint cell (full-subgraph + ID checks + PaperStyleMLP)
python scripts/official_direct_r198_collaborator_eval.py \
  --run direct_r198_tfmoe_40ep_seed2_linear_lr2e-3 \
  --arm DIRECT_H_TFMOE \
  --peak_lr 0.002 \
  --epoch 10

# Matched SSL epochs 3,10,20,30,40
python scripts/official_direct_r198_collaborator_eval.py \
  --run direct_r198_tfmoe_40ep_seed2_linear_lr2e-3 \
  --arm DIRECT_H_TFMOE \
  --peak_lr 0.002 \
  --epochs 3,10,20,30,40

# Rebuild collaborator package (refuses protocol != full_subgraph)
python scripts/official_direct_r198_collaborator_eval.py --build-package
```

Defaults:

| Setting | Value |
|---------|-------|
| Extractor | `scripts/extract_direct_r198_full_cell.py` (full-subgraph) |
| Probe | PaperStyleMLP, 20 ep, lr=1e-3, bs=8192, seed=2, R198+X+TF, best-val-AUPRC |
| Embeddings | `embeddings/direct_r198_40ep_linear_lr_full_extract/` |
| Cells | `results/diagnostics/direct_r198_40ep_linear_lr_full_extract_reeval/cells/` |
| Protocol stamp | `protocol=full_subgraph`, `evaluation_tier=official_collaborator` |

Guards:

- refuses seed-only embedding path
- refuses overwriting existing official cell JSON unless `--allow_overwrite`
- does **not** train or modify checkpoints
- collaborator package builder raises if any cell is not `full_subgraph` / fails ID verify

## Diagnostic only (not official)

```bash
# Seed-only — stamped protocol=seed_only, collaborator_merge_allowed=false
# Refuses writing into the official full-subgraph out dir.
python scripts/eval_direct_r198_40ep_linear_arm.py \
  --run ... --arm DIRECT_R198_TFMOE --peak_lr ...
```

Outputs go under `results/diagnostics/direct_r198_tfmoe_40ep_linear_lr_sweep/` and are labeled diagnostic/provisional.

## Shared helpers

`scripts/direct_r198_eval_protocol.py` — protocol constants, merge gate, manifest helpers.
