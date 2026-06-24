# Transaction feature-KNN precompute reference

Offline **sparse neighbor caches** for optional contrastive **negative exclusion**
(`--enable_knn_negative_filter`) or **soft positives** (`--enable_knn_soft_positives`).
Does **not** build a dense KNN graph view or change the GNN message-passing graph.

**Training integration:** [`cli-reference.md`](cli-reference.md) · design context:
[`contrastive-learning-plan.md`](contrastive-learning-plan.md). **Benchmark results:**
[`results.md` § Feature-KNN](results.md#feature-knn-small-hi).

**Outcome (Jun 2026):** precompute works. With random 8192 negs, **exclusion does not
help** and **soft positives hurt** (0.849 / 0.067 vs baseline 0.951 / 0.233) — neighbors
are too similar in feature space. See results doc before adopting either flag.

---

## What it does

1. Load **train-split** transactions only (temporal split; same as SSL).
2. Build **label-free** feature vectors per transaction.
3. Run **top-k similarity search** (cosine by default: standardize → L2 normalize → inner product).
4. Save sparse `(edge_id → neighbor_ids, neighbor_sims)` for contrastive training.

Neighbors are **excluded from negatives only** — never added as positives.

---

## Feature sets (`--feature_set`)

| Value | Contents | ~dims |
|-------|----------|-------|
| `edge_native` | Timestamp, log1p amount, currency, payment format | 4 |
| `degree_fan` | Sender/receiver in/out/total degrees on **train graph** (log1p) | 8 |
| `edge_native+degree_fan` | Concatenation of both | 12 |

Code: `transaction_knn/features.py` · registry groups in
[`morphology-reference.md`](morphology-reference.md).

---

## Cache format (`.npz`)

Written by `scripts/precompute_transaction_knn.py`; read by `knn_filter.py`.

| Array | Shape | Description |
|-------|-------|-------------|
| `edge_ids` | `(N,)` | Train **split-local** ids (0…N−1); matches contrastive `edge_id` after `add_arange_ids` |
| `csv_edge_ids` | `(N,)` | Original CSV `EdgeID` (audit only) |
| `neighbor_ids` | `(N, k)` | Neighbor split-local ids; `-1` if padding |
| `neighbor_sims` | `(N, k)` | Cosine similarity (or neg distance for euclidean) |
| `feature_names` | object array | Column names |
| `k` | scalar | Neighbors per row |
| `metadata_json` | string | JSON: data, feature_set, backend, shard info, etc. |

**ID space:** always train-split-local. Val/test transactions are not in the cache.

---

## Backends (`--backend`)

| Backend | Description |
|---------|-------------|
| `auto` | FAISS GPU if installed → else **PyTorch CUDA** → else CPU sklearn |
| `torch_gpu` | Batched `queries @ database_chunk.T` + merged `topk` on GPU (no full N×N matrix) |
| `faiss_gpu` / `faiss_cpu` | Exact inner-product (cosine) or L2 index |
| `faiss_ivf` | Approximate IVF; use `--approx_recall_subset N` to log recall@k vs exact |
| `cpu` | sklearn `NearestNeighbors` (fine for small N; ~70h est. on full Small-HI) |

Module: `transaction_knn/backends.py`.

---

## CLI

Subcommands: **`precompute`** (build cache) · **`merge`** (combine shards).

The `precompute` subcommand may be omitted for backward compatibility.

```bash
# Smoke (100k train rows)
python scripts/precompute_transaction_knn.py precompute \
  --data Small-HI --feature_set edge_native+degree_fan --k 15 \
  --backend auto --max_rows 100000 \
  --shard_dir morphology_cache/Small-HI/transaction_knn_edge_native_degree_fan_k15_smoke100k_shards \
  --output morphology_cache/Small-HI/transaction_knn_edge_native_degree_fan_k15_smoke100k.npz

# Full train (resumable shards)
python scripts/precompute_transaction_knn.py precompute \
  --data Small-HI --feature_set edge_native+degree_fan --k 15 \
  --backend auto --query_batch_size 8192 --shard_rows 250000 --resume \
  --shard_dir morphology_cache/Small-HI/transaction_knn_edge_native_degree_fan_k15_shards \
  --output morphology_cache/Small-HI/transaction_knn_edge_native_degree_fan_k15.npz

# Merge shards only (after TIME_LIMIT or manual stop)
python scripts/precompute_transaction_knn.py merge \
  --shard_dir morphology_cache/Small-HI/transaction_knn_edge_native_degree_fan_k15_shards \
  --output morphology_cache/Small-HI/transaction_knn_edge_native_degree_fan_k15.npz
```

### Useful flags

| Flag | Default | Purpose |
|------|---------|---------|
| `--k` | 50 | Neighbors per row (scout recipe: **15**) |
| `--query_batch_size` | 8192 | Query batch size (GPU memory) |
| `--shard_rows` | 250000 | Write shard every N rows (`0` = single file at end) |
| `--shard_dir` | `<output_stem>_shards` | Shard directory |
| `--resume` | off | Skip existing shards |
| `--max_rows` | 0 | Cap rows (smoke tests) |
| `--no_merge` | off | Shards only; no final `.npz` |

End of run prints sanity checks: feature names, row count, neighbor shape,
self-neighbor count (must be 0), sim min/mean/max.

---

## Slurm (Small-HI, k=15, edge_native+degree_fan)

| Script | GPU | Rows | Walltime | Output |
|--------|-----|------|----------|--------|
| `slurm/precompute_transaction_knn_small_hi_gpu_smoke100k.sh` | 1 | 100k | 1h | `..._smoke100k.npz` |
| `slurm/precompute_transaction_knn_small_hi_gpu_full_k15.sh` | 1 | full train | 6h | `..._k15.npz` |
| `slurm/precompute_transaction_knn_small_hi_edge_native_degree_fan_k15.sh` | none | full | 6h | CPU sklearn (legacy; slow) |

Shard + log paths mirror the `--output` / `--shard_dir` names above under
`morphology_cache/Small-HI/` and `logs/precompute_transaction_knn_*.log`.

If a full job hits `TIME_LIMIT`, resubmit the same script — `--resume` continues
from completed shards.

---

## Enable at training time (after cache exists)

```bash
--enable_knn_negative_filter \
--knn_cache_path morphology_cache/Small-HI/transaction_knn_edge_native_degree_fan_k15.npz \
--knn_filter_k 15   # use 5 for tighter exclusion from the same k=15 cache
```

`--knn_filter_k` truncates cached neighbors (cache may be built with `--k 15`).

**Slurm ablations (Small-HI, asym + projection backbone):**

| Script | `--knn_filter_k` |
|--------|------------------|
| `slurm/ablation_knn_filter_k5_asym_proj_8192neg_queue0_20ep.sh` | 5 |
| `slurm/ablation_knn_filter_k15_asym_proj_8192neg_queue0_20ep.sh` | 15 |

Each job: train → extract → linear probe. Artifacts under
`checkpoints/knn_filter_k*/`, `embeddings/knn_filter_k*/`, `logs/knn_filter_k*_*.log`.

**Training logs** (once per epoch):

```
hetero/train KNN negative filter: candidates=... removed=... (...),
  anchors_with_cache=..., anchors_with_knn_in_pool=..., fallback_rows=...
```

- `anchors_with_knn_in_pool` — fraction of anchors where a cached neighbor appeared in the sampled negative pool (typically **~1–3%** with 8192 random negs)
- `fallback_rows` — anchors that reverted to unfiltered negatives because too few candidates remained (0 in Jun 2026 runs)

Queue-based KNN filtering is **not** implemented.

### KNN soft positives (GCPAL-style)

Use cached feature neighbors as **low-weight positives** in the InfoNCE numerator (identity
positive stays weight `1.0`). Positives are materialized with an auxiliary
`LinkNeighborLoader` forward on explicit seed edge ids — not batch-overlap-only.

```bash
--enable_knn_soft_positives \
--knn_cache_path morphology_cache/Small-HI/transaction_knn_edge_native_degree_fan_k15.npz \
--knn_pos_source_k 15 --knn_pos_m 1 --knn_pos_weight 0.025 \
--knn_pos_weight_mode uniform --knn_pos_seed 1
```

Slurm: `slurm/ablation_knn_softpos_m1_w0025_asym_proj_8192neg_queue0_20ep.sh` (run;
**do not adopt** with current cache). `..._m3_...` not run.

Audit candidate feature sets before rebuilding caches:

```bash
python scripts/audit_transaction_knn_cache.py --data Small-HI --max_rows 50000
```

---

## Package layout

| Path | Role |
|------|------|
| `scripts/precompute_transaction_knn.py` | CLI entrypoint |
| `transaction_knn/features.py` | Feature matrices + train split load |
| `transaction_knn/backends.py` | CPU / FAISS / PyTorch GPU search |
| `transaction_knn/shards.py` | Shard write, manifest, merge, validate |
| `knn_filter.py` | Runtime negative mask during InfoNCE |

Tests: `tests/test_transaction_knn_precompute.py`.
