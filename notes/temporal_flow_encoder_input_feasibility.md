# Feasibility: temporal_flow_causal as GNN encoder edge inputs (SSL)

**Date:** 2026-07-20  
**Status:** Read-only audit. No jobs launched. No behavior changes.  
**Question:** Can causal temporal-flow features be appended to the edge feature tensor used by the GNN encoder during contrastive SSL pretraining?

**Verdict:** **Feasible, low–medium risk, label-free.** Best first step is a gated `--include_temporal_flow_edge_features` (default **false**) wired in `get_data`, then one Small-HI GIN seed2 scout. Do **not** combine with TF-reg aux / TF soft-positives on the first scout.

---

## 1. Where raw edge features are constructed (Small-HI / Small-LI)

| Step | Location | Behavior |
|------|----------|----------|
| Spec | `dataset_specs.py` `DEFAULT_EDGE_FEATURE_COLS` | `Timestamp`, `Amount Received`, `Received Currency`, `Payment Format` (4 cols) |
| Load | `data_loading.get_data` | CSV → `edge_attr` from those cols |
| Split graphs | same | train = `tr_inds`; val = `tr∪val`; test = all edges |
| Ports | `GraphData.add_ports` | +2 cols (in/out port) |
| TDs | `GraphData.add_time_deltas` | +2 cols (in/out Δt) |
| Norm | `z_norm` on each split’s `edge_attr` | **per-split** z-score (existing pattern) |
| Reverse MP | `create_hetero_obj` | copies attrs to `to` / `rev_to` |
| Synthetic id | `add_arange_ids` | prepends local arange; stripped before GNN (`edge_dim = cols − 1`) |

Typical contrastive recipe (`ports+tds+reverse_mp+ego`): **edge_dim = 8** (ego is a **node** channel only).

---

## 2. Can TF cache rows align by `edge_id`?

**Yes** for current AMLWorld Small-HI / Small-LI caches.

Cache: `results/cache/temporal_flow_causal/{Small-HI,Small-LI}/`

| Artifact | Role |
|----------|------|
| `features.npy` | `[N, 5]` float32, CSV row order |
| `edge_id.npy` | `arange(N)` |
| `split_{train,val,test}_edge_id.npy` | same indices as `temporal_edge_split` |
| `meta.json` | 5 feature names; `uses_labels: false`; `past_only: true` |

Verified: train / val / test IDs are **contiguous temporal prefixes** covering `[0..N)` on both HI and LI. Therefore:

- Train graph row `i` ↔ global `edge_id = i` ↔ `features[i]`
- Val graph (prefix `tr∪val`) local arange ↔ global id
- `add_arange_ids` local ids match global CSV `EdgeID` under this contiguous-prefix property

**Join recipe:** in `get_data`, after ports/tds (and preferably **after** base `z_norm`), append `features[e_tr]` / `features[e_val]` / `features[full]` with the same index arrays already used to slice `edge_attr`. Assert `meta.causal_history_policy.uses_labels is false` and `n_rows == len(df)`.

---

## 3. Flag surface

**Proposed:** `--include_temporal_flow_edge_features` (boolean, default **false**).

Optional reuse: `--aux_temporal_flow_cache` / a dedicated `--temporal_flow_edge_features_cache` pointing at `results/cache/temporal_flow_causal/{data}`.

**No such flag exists today.** Existing TF flags are orthogonal:

| Flag | Role today |
|------|------------|
| `--aux_temporal_flow` | Aux head on post-128 embeddings (targets from cache) |
| `--temporal_flow_soft_positives` | Soft positives in InfoNCE from TF bins |
| Probe `--temporal_flow_cache_dir` | Downstream arm C/D only |

**Wire point:** `data_loading.get_data` (not the probe scripts). `get_model` already infers `edge_dim` from the batch (`training.py` ~1494–1496), so the GNN `Linear(edge_dim, n_hidden)` grows automatically.

**Critical:** embedding **extract** must pass the **same** flag as training, or checkpoint `edge_emb` width will mismatch.

---

## 4. Pretraining only vs probe raw construction?

| Surface | Affected if flag on? |
|---------|----------------------|
| Contrastive SSL `edge_attr` → encoder | **Yes** (intended) |
| Embedding extract (same `get_data`) | **Yes** (must match train) |
| Probe **raw** (`edge_native` from CSV via `probe_feature_ablation`) | **No** — independent of GNN `edge_attr` |
| Probe **temporal_flow** arm (cache + train-only `StandardScaler`) | **No** construction change |

So this is an **encoder-input** change. Downstream A/B/D **feature builders** stay as-is; **interpretation** of the arms changes (see §5).

---

## 5. Avoiding double-counting in evaluation

When the encoder already saw TF:

| Arm | Role under TF-input encoder | Guidance |
|-----|-----------------------------|----------|
| **A** embedding-only | Primary: did TF-in-encoder improve the representation? | **Primary** |
| **B** embedding + raw | Secondary: raw is CSV `edge_native`, not GNN ports/tds/TF | **Primary** (matched baseline protocol) |
| **D** embedding + raw + TF | Explicit **TF double-count** (same 5 causal cols again in the linear probe) | **Diagnostic only** for redundancy; do not promote on D alone |

Do **not** change probe code for v1. Document in the scout note that D is expected to look more redundant than on the matched baseline. Skip post-128 unless free. Keep `table_eligible=false`.

---

## 6. Train-split safe and label-free?

| Check | Status |
|-------|--------|
| Features use labels? | **No** (`uses_labels: false`; compute uses ts / account ids / amount only) |
| Past-only / tie-safe? | **Yes** (`past_only: true`; tie batch uses history strictly before `t`) |
| Val/test history | Intentional: val sees train past; test sees train+val past (available-at-time, not future labels) |
| Leakage audit | `notes/temporal_flow_causal_leakage_audit.md` — recompute matches cache |

**Scaling caveat:** naive append **before** existing per-split `z_norm` would z-score TF on val/test graphs that already contain later edges (same pattern as ports/tds today). **Preferred:** append **after** `z_norm`, scale TF with a **train-only** scaler (match aux/probe). Refuse caches with `uses_labels: true`.

SSL pretraining must keep `--objective contrastive` with no label loss (unchanged).

---

## 7. Tests needed

1. **Default unchanged:** flag off → `edge_attr` shape/values identical to current recipe (bitwise or close under float).
2. **Flag on:** `edge_attr` width += 5; synthetic id still column 0; `get_model` builds with `e_dim_old + 5`.
3. **Alignment:** for HI/LI contiguous splits, `edge_attr[i, -5:]` equals train-scaled `features[global_id]` (or raw pre-scale if testing join only).
4. **Safety:** refuse `uses_labels: true` cache; missing cache → clear error.
5. **Hetero:** forward and reverse both carry TF; reverse port swap still only touches port cols.
6. **Extract parity:** loading a TF-input checkpoint without the flag fails loudly (dim mismatch), with the flag succeeds.

---

## Memory impact (expected)

| Item | Estimate |
|------|----------|
| Cache on disk (HI) | ~102 MB `features.npy` (already present) |
| Extra CPU `edge_attr` (HI train, +5×float32) | ~65 MB; full graph ~102 MB |
| Model | `edge_emb`: `8→13` inputs → tiny weight increase |
| Peak GPU | Dominated by neighbor sampling / activations; **modest** increase expected (not a new memory regime at bs=8192, fanout 100×100) |

No need for a special large-memory partition for the first scout.

---

## Implementation risk

| Risk | Level | Mitigation |
|------|-------|------------|
| Wrong join index | Medium | Index with same `e_tr` / `e_val` as `get_data`; assert contiguous or map by `edge_id.npy` |
| Per-split z_norm on TF | Low–med | Append after z_norm + train-only scaler |
| Checkpoint / extract dim mismatch | Medium | Same flag on train + extract; default false preserves old ckpts |
| Science overlap with TF-aux / probe D | Med (interpretation) | First scout: encoder TF **only**, no aux; treat D as redundancy diagnostic |
| GraphCL attr mask hits TF cols | Low | Accept for scout; optional `mask_cols` later |

**Overall implementation risk:** **low–medium**. Plumbing-localized; no architecture rewrite.

---

## Exact flags / scripts for a smallest seed2 scout (if implementing)

**Code (gated, default false):**

1. `util.py`: `--include_temporal_flow_edge_features` + optional cache path  
2. `data_loading.get_data`: load cache, assert label-free, append 5 cols (train-only scale preferred)  
3. Log new `edge_dim` and flag at train start  
4. Unit tests in §7  

**Do not** change probe builders for v1.

**Scout recipe (matched baseline):**

- Small-HI GIN seed2  
- `--include_temporal_flow_edge_features`  
- asym projection, 8192 neg, queue=0, temp=0.5, reverse_mp, ego, ports, emlps, tds  
- `batch_size=8192`, `contrastive_accum_steps=4`, 20 epochs, `checkpoint_policy best`  
- **No** `--aux_temporal_flow`, **no** soft-positives, **no** labels in SSL  
- Edge drop / attr-mask at **defaults** (0.10 / 0.10) unless a separate branch is intentionally crossed  

**Matched baseline:**  
`hi_contrastive_gin_emlps_tds_seed2_proj_asym_8192neg_queue0_accum4_20ep` / `morph_obj_baseline_pre3h_seed2.json`

**Eval:** pre-3h **A + B** primary; **D** optional diagnostic; skip post-128 unless free; `table_eligible=false`.

**Suggested outputs (when run later):**

- `results/diagnostics/tf_encoder_input_seed2_quickcheck.json`  
- `notes/tf_encoder_input_seed2_quickcheck.md`  

**Success heuristic (proposed):** A or B AUPRC improves vs matched seed2; P@100 does not collapse. If only B improves → keep diagnostic. If both A and B improve → optional one seed1 replication. If neither → stop TF-encoder-input branch.

---

## Final recommendation

| Item | Recommendation |
|------|----------------|
| Feasible now? | **Yes** |
| Implement risk | **Low–medium** (alignment + scaling + extract flag parity) |
| Memory | **Modest** (~+5 edge channels; ~65–100 MB CPU attrs; tiny `edge_emb`) |
| Flags | `--include_temporal_flow_edge_features` default **false**; reuse existing TF cache |
| Smallest scout | One Small-HI GIN **seed2** quickcheck as above — **after** the flag lands; **not launched in this audit** |
| Do not | Overwrite outputs; use labels; promote on D alone; mix with TF-aux on first scout; regenerate thesis tables |
