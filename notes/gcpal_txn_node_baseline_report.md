# GCPAL-style transaction-node baseline — status report

**Not an exact GCPAL reproduction.** Standalone package: `gcpal_txn_node/`.

Spec / ambiguities: `notes/gcpal_txn_node_ambiguity_report.md`

## Delivered

| Stage | Artifact |
|-------|----------|
| 1 Spec | `notes/gcpal_txn_node_ambiguity_report.md`, `gcpal_txn_node/spec.py` |
| 2 Adjacency | `gcpal_txn_node/adjacency.py` (`immediate_next` default, `capped_next_k`) |
| 3 Features/views/KNN | `features.py`, `knn_adapter.py` |
| 4 Encoder/loss | `model.py`, `loss.py`, `sampling.py`, `train_step.py` |
| 5 Tests | `tests/test_gcpal_txn_node.py` (**11 passed**) |
| 6 Smoke | `scripts/gcpal_txn_node_smoke.py`, `slurm/gcpal_txn_node_smoke.sh` |
| 7 Scouts | `scripts/gcpal_txn_node_scout.py`, `slurm/gcpal_txn_node_scout.sh` (`MODE=control\|gcpal`) |

## Isolation

No `gcpal_txn_node` imports in `main.py`, `training.py`, `contrastive_loss.py`, or `embedding_extraction.py`.

## Documented deviations (selected)

- Adjacency = directed `immediate_next` flow (assumption)
- KNN cache = `edge_native+degree_fan` (degree features beyond raw columns)
- Identity included in positives by default (`I ∪ A ∪ A_knn`)
- Eval encode uses induced subgraph chunks (not a single full-graph forward)

## Pending runtime

Smoke + 5-epoch scouts require Advanced GPU Slurm. Submit:

```bash
sbatch slurm/gcpal_txn_node_smoke.sh
# after smoke JSON shows fits_advanced_gpu_6h_for_5ep=true:
sbatch --export=ALL,MODE=control --job-name=gcpal_txn_ctrl slurm/gcpal_txn_node_scout.sh
sbatch --export=ALL,MODE=gcpal --job-name=gcpal_txn_gcpal slurm/gcpal_txn_node_scout.sh
```

## Final questions (fill after scouts)

1. Feasibility of txn-node implementation — *(pending smoke)*
2. Effect of three-view training under temporal eval — *(pending)*
3. Effect of random vs temporal eval — *(pending)*
4. Remaining deviations — see ambiguity report
5. Is 40-epoch justified? — **default no** until 5-epoch scout shows material gain without collapse
