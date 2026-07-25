# Edge D+ neighbor-positive transfer — preflight

**NOT an exact GCPAL reproduction.** Distinct from `--enable_knn_soft_positives`.

## Locked D+ reference (job 18514684)

| Item | Value |
|------|-------|
| Slurm | `slurm/comparison_gin_emlps_ports_tds_corrected_preserve_seed_asym_proj_8192neg_queue0_40ep_seed2.sh` |
| Unique name | `gin_emlps_ports_tds_corrected_preserve_seed_asym_proj_8192neg_queue0_40ep_seed2` |
| Checkpoint | `saved-models/checkpoint_gin_emlps_ports_tds_corrected_preserve_seed_asym_proj_8192neg_queue0_40ep_seed2.tar` |
| Selected epoch | 40 (best == final) |
| sha256 | `a320920141f585c5825cbd63ce760a845fb434a9b162d4c87270dc72b0442b87` |
| Flags | gin + emlps + ports + tds + ego + reverse_mp + `--correct_reverse_edge_features` + `--preserve_seed_edges` + asym proj 128 + 8192 neg + queue 0 + accum 4 + seed 2 |
| Temp / lr / fanout | 0.5 / 0.006213… / `100 100` |

**Unchanged historical artifacts.** This experiment uses new `--unique_name` values only.

## Full-stack reference (job 18678029)

| Item | Value |
|------|-------|
| Winner | `edge_pre3h\|H+X+TF\|mlp\|none` |
| Temporal val AUPRC | 0.550 |
| Temporal test AUPRC | 0.674 |
| Temporal F1@0.5 | 0.656 |
| Pre-3h dir | `embeddings/...40ep_seed2/pre_embedding_3h` (extracted job 18558352; aligned by `edge_id`) |

These are **references**, not test-based selection targets for the scout.

## Matched-control requirement

Positive-complete retrieval **changes** batch membership, MP subgraph size, negative pool, optimizer-step count, and anchor exposure vs D+ identity InfoNCE on natural LinkNeighborLoader batches of 8192.

Therefore the existing D+ run **cannot** serve as the control.

**Required arms (≤2):**

1. `edge_dplus_identity_poscomplete_10ep_seed2` — same poscomplete batching, identity-only mask  
2. `edge_dplus_neighbor_supcon_poscomplete_10ep_seed2` — same batching, identity∪flow∪KNN + `supcon_mean_logprob`

## Implementation summary

| Piece | Location |
|-------|----------|
| Mask / expand / SupCon | `edge_neighbor_positives.py` |
| Training loop | `edge_neighbor_positives_train.py` |
| CLI | `--enable_edge_neighbor_positives` and `--edge_neighbor_*` in `util.py` |
| Dispatch | `training.py` → neighbor path when flag set |
| Tests | `tests/test_edge_neighbor_positives.py` |
| Smoke | `scripts/smoke_edge_dplus_neighbor_positives.py` |

KNN is **not** a third MP view; it only defines positives across the two edge-centric random views.

## Envelope / epoch budget

Copy Advanced-GPU D+ envelope: `mit_normal_gpu` / `mit_amf_advanced_gpu` / 64G / 16 CPU / 1 GPU / **6h**.

**Epoch streaming:** positive-complete batches fit ~120 anchors under `max_total=2048` (k=15 KNN). Full-stream coverage (~25k steps/epoch) exceeds 6h. Training therefore uses **D+-matched** `--edge_neighbor_max_batches_per_epoch = ceil(n_train/8192)` (~397 microbatches/epoch). Documented deviation from full-epoch coverage; both arms share the same budget.
