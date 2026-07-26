# PaySim preserve vs normalization ablation

No encoder training. Primary diagnostic: **post-128 H-only logistic** (class_weight=model).

## 2×2 AUROC (cw=model @0.5)

| | per-graph z-norm | PaySim train-fit z-norm |
|--|-----------------:|------------------------:|
| **A corrected, no preserve** | 0.9201 (reused) | 0.8668 (new) |
| **B D+ + preserve** | 0.8331 (new) | 0.7046 (reused) |
| **random edge_dim=8** | 0.7347 | 0.5783 |

## BN recalibration (target-train-only running stats)

| Cell | AUROC |
|------|------:|
| A train-fit + BN recal | 0.8781 |
| B per-graph + BN recal | 0.8906 |

Deltas: A 0.0112, B 0.0575

## AMLWorld locked eval of A (secondary)

- pre-3h H: AUROC 0.9780 AUPRC 0.5761
- pre-3h H+X+TF: AUROC 0.9880 AUPRC 0.6871

## Deltas

- preserve | train-fit: -0.1622
- preserve | per-graph: -0.0870
- norm | A: -0.0532
- norm | B: -0.1284

## Artifacts

- `/orcd/home/002/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN/notes/paysim_preserve_normalization_ablation.md`
- `/orcd/home/002/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN/results/diagnostics/paysim_preserve_normalization_ablation.json`
- cells: `/orcd/home/002/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN/results/diagnostics/paysim_preserve_normalization_ablation/cells/`
- embeds: `/orcd/home/002/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN/embeddings/paysim_preserve_normalization_ablation/`
