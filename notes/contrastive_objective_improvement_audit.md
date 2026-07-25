# Contrastive-objective improvement audit

**Date:** 2026-07-19  
**Scope:** Cheap, high-leverage changes enabled by larger GPU memory / advanced account.  
**Status:** Audit complete; seed2 resource scout launched and analyzed (2026-07-20). See § Outcomes below.

**Eval focus (current thesis policy):** primary = **pre_embedding_3h** arms **A** (embedding-only) and **B** (embedding+raw), with recall-oriented metrics (P@K / R@K / recall@P≥…). Post-128 diagnostic only. Do not rank by AUROC/F1 alone.

**Baseline recipe under audit:** Small-HI GIN, asym projection, 8192 negatives, queue=0, temp=0.5, reverse_mp+ego+ports+emlps+tds, `bs=8192`, `accum=4`, 20ep, `checkpoint_policy=best`.

---

## Executive verdict

The current objective is limited by a **memory-saving mismatch**: `--contrastive_accum_steps` enlarges the *optimizer* batch but **not** the InfoNCE negative pool. Negatives are computed only within each loader microbatch (+ optional queue). Historical “relax” runs already showed larger true batch helps F1, but the **current** strong recipe (`8192neg` + `queue=0` + emlps+tds) has **never** been run at `bs=16384`.

Highest-EV, lowest-risk next steps are flag-only:

1. True large InfoNCE batch (`bs=16384`, `accum=2`, keep 8192neg/queue0).
2. Milder edge drop (`--edge_drop_target_rate 0.05`).

Skip queue/neg sweeps, hard-negative mining (needs new code), degree-aware drop, masked-edge aux re-run, and broad multi-factor grids.

---

## 1. True batch size vs gradient accumulation

### Finding (critical)

InfoNCE is computed **per microbatch**. Accumulation only averages gradients.

Evidence:

- Loss call uses the current batch’s aligned seed embeddings only (`training.py` homo ~511–536; hetero ~same pattern before `loss_raw / accum_steps`).
- `contrastive_loss.edge_identity_infonce_loss` samples negatives from the **current aligned batch** (+ optional detached queue), not across accum windows (`contrastive_loss.py` ~1095–1136).
- Train loop scales then accumulates (`training.py` ~567–584 / ~1184–1201):

```text
loss = loss_raw / float(accum_steps)
loss.backward()
# optimizer.step() every accum_steps
```

| Quantity | Current recipe | What InfoNCE sees |
|----------|----------------|-------------------|
| `--batch_size` | 8192 seed edges | ~8192 requested; ~6600 shared after two-view edge-drop survival |
| `--contrastive_accum_steps` | 4 | **Does not enlarge negatives** |
| Effective optimizer batch | ≈ 8192×4 = 32768 | Gradients only |
| Queue | 0 | No cross-microbatch negatives |

First-batch hetero logs (current recipe): ~150k subgraph nodes, ~700k forward / ~880k reverse edges; `requested_seed_edges=8192`, `shared_seed_edges≈6570–6640`. Peak VRAM is dominated by subgraph GNN activations, not the InfoNCE matmul (`training.py` first-batch log).

### Advanced GPU headroom

Cluster `mit_normal_gpu` offers `gpu:l40s`, `gpu:h100`, `gpu:h200` (advanced account). Historical runs already fitted **`bs=16384 accum=2`** on older recipes (sym/asym confound). Current recipe at `bs=8192` is comfortable; **true** batch scale-up to 16384 is the plausible next step on H100/H200.

### Caveat on prior 16384 evidence

Prior `asym@16384` (`slurm/ablation_contrastive_proj_asym_16384_20ep.sh`) used **1024 negs + queue=32768** and **no emlps/tds** — not the current recipe. So “batch size helped” is suggestive, not a completed A/B against today’s baseline.

### Recommendation

**One large-actual-batch scout (highest EV):**

```text
--batch_size 16384 --contrastive_accum_steps 2
# keep: asym, 8192neg, queue=0, temp=0.5, reverse_mp ego ports emlps tds, 20ep, best ckpt
```

Optional OOM fallback only: `bs=12288 accum=2` or `bs=16384 accum=1` on H200. Do **not** “fix” by raising accum alone.

---

## 2. Neighbor / context sampling

### Current config

| Setting | Value | Source |
|---------|-------|--------|
| Loader | `LinkNeighborLoader` | `train_util.py` ~358–398 |
| Seed batch | `args.batch_size` (edges) | |
| Fanout | `--num_neighs` default `[100, 100]` | `util.py` ~66 |
| Hops | 2 | `len(num_neighs)` |
| Workers | recipes often 16 | |

Recipes consistently use `--num_neighs 100 100`. Increasing context is **flag-only**.

### Feasibility

Doubling fanout (~`200 200`) roughly grows the sampled subgraph and is the main VRAM risk. At current ~150k nodes / batch, a larger-context scout should use advanced GPU and prefer keeping `bs=8192` (or reduce batch if OOM). Three-hop (`100 100 50`) is a bigger untested jump — lower priority.

### Recommendation

**One larger-context scout (medium EV, medium risk):**

```text
--num_neighs 200 200
# else match current baseline (bs=8192 accum=4, 8192neg, queue=0, …)
```

Skip 3-hop and joint large-batch+large-fanout until single-factor scouts finish.

---

## 3. Augmentation strength

### Current parameters

| Transform | Value | CLI? |
|-----------|-------|------|
| Edge drop rate | **0.1** | `--edge_drop_target_rate` (`util.py` ~317–320) |
| Edge drop policy | `random` | `--edge_drop_policy` |
| Edge-attr mask rate | **0.1 hardcoded** | **No** (`training.py` `_contrastive_view_kwargs` ~226–236) |

Two independent views drop edges; seed edges can disappear from one view → shared seeds fall to ~80% of requested (~6600/8192). For AML flow motifs (fan-in/out, chains), aggressive random drop may destroy the very structure InfoNCE should preserve.

### Already-tested (do not repeat)

- `degree_aware` / `degree_flow_aware` edge drop: registered negative / no clear win vs random (`notes/results.md`).
- Do not invent new structure-preserving drop policies in this batch.

### Recommendation

**One lower-drop scout (high EV, very low risk):**

```text
--edge_drop_target_rate 0.05
# else match current baseline
```

Optional follow-up only if 0.05 helps: `0.0` (attr-mask still 0.1). Changing attr-mask needs a tiny code edit (hardcoded) — **defer** unless edge-drop scout is positive.

---

## 4. Negative sampling

### Current construction

- Primary: uniform in-batch negatives among aligned shared seeds (`--contrastive_num_neg_samples 8192`).
- Optional FIFO memory bank (`--contrastive_memory_bank_size`; recipe **0**).
- Asymmetric: view2 `no_grad`, z1→z2 only.
- Optional exclusion filters (FNF / KNN) and soft-positives exist; hard-negative *mining* does **not**.

### Skip (explicit)

| Idea | Why skip |
|------|----------|
| Larger neg count (10240/12288/…) | Already swept; no win; OOM at 12288@bs8192 |
| Queue re-enable / larger queue | Prefer queue=0 in current recipe |
| Hard-negative mix | **Not implemented**; needs new sampling logic — defer |
| KNN exclusion / soft-pos | Already negative |
| Blind FNF re-sweep | Out of scope for this tiny batch |

If hard-negatives are revisited later, prefer a minimal in-batch “top-k cosine hard mix” only after large-batch + low-drop results, not now.

---

## 5. Reconstruction-style auxiliary

### Already available (label-free)

| Aux | Flags | Status vs InfoNCE |
|-----|-------|-------------------|
| Temporal-flow regression/bins | `--aux_temporal_flow` | Separate multiseed confirmation in flight; **not** part of this batch |
| Morphology expert | `--morph_expert` | Degflow multiseed → stop; do not expand here |
| Masked edge attr recon | `--masked_edge_aux_weight` (hetero only) | Prior scout `w=0.1` ≈ contrastive baseline F1 (`notes/results.md` ~0.951 AUROC / 0.239 F1) — **no clear win** |
| Standalone masked_edge objective | `--objective masked_edge` | Replacement, not aux — out of scope |

### Recommendation

**Do not add a new recon aux in this scout batch.** Existing masked-edge aux is already tested and weak. Prefer finishing TF-reg multiseed separately. Any future recon aux should stay weight-small and InfoNCE-primary.

---

## Proposed tiny scout batch (≤4 runs)

All runs: Small-HI GIN 20ep, seed1, current graph stack, asym+proj+8192neg+queue0+temp0.5, `checkpoint_policy=best`, unique run names, no overwrite. Probe **pre-3h A/B** (+ D diagnostic if TF cache already available) with recall-oriented metrics. Matched baseline: existing `hi_contrastive_gin_emlps_tds_*` / morph_obj_baseline seed1 probes.

| Rank | Scout ID | Change (single factor) | Expected value | Impl. risk | Why |
|-----:|----------|------------------------|----------------|------------|-----|
| 1 | `large_bs_16384` | `bs=16384 accum=2` | **Highest** | Low (flags; historically fitted) | Fixes InfoNCE microbatch limit without changing objective |
| 2 | `edge_drop_0.05` | `--edge_drop_target_rate 0.05` | High | Very low | Motif-preserving; increases shared-seed count; CLI-ready |
| 3 | `fanout_200` | `--num_neighs 200 200` | Medium | Medium (VRAM) | More AML context; flag-only; use H100/H200 |
| 4 | *(optional)* `edge_drop_0.00` | `--edge_drop_target_rate 0.0` | Medium | Very low | Only if #2 helps; attr-mask still 0.1 |

**Run at most #1–#3 first** (or #1+#2 if budget is two). Do not combine factors in the first pass.

### Explicitly skip

- Raising `accum_steps` alone (does not help InfoNCE).
- Queue / larger-negative sweeps.
- Hard-negative mining (new code).
- Degree-aware / degree-flow-aware drop.
- Masked-edge aux re-run; morph/degflow/clustering; TF soft-positives; TF bins.
- Symmetric InfoNCE re-open; 3-hop fanout; attr-mask code change; broad grids.
- Joint `bs=16384` × `num_neighs 200 200` until singles land.

### Success read for these scouts

Promote interest only if **pre-3h A and/or B AUPRC** improve vs matched baseline on seed1 **and** P@100 / recall@P≥0.80–0.90 do not collapse. Do not count D-only or post-128-only gains as representation wins. If seed1 looks strong, replicate seeds 2–3 for the winner only.

---

## Outcomes — seed2 resource scout (2026-07-20)

Launched only ranks **#1 and #2** on **seed2** (matched baseline exists). Fanout / edge_drop_0.00 not run. Summary: `results/diagnostics/contrastive_objective_resource_scout.json`, `notes/contrastive_objective_resource_scout.md`.

| Scout | Pre-3h ΔA / ΔB AUPRC | A P@100 | Verdict |
|---|---|---|---|
| `large_bs_16384` (bs=16384 accum=2) | **−0.121 / −0.161** | 0.79→**0.51** | **stop** — true larger InfoNCE batch hurt under current recipe |
| `edge_drop_0.05` | **+0.027 / +0.053** | 0.79 (flat) | **replicate** seeds 1/3; D slightly down (−0.042 AUPRC) |

Sanity logs: large_bs shared seeds **13189**/16384 (~72 GiB peak, no OOM); edge_drop shared seeds **7420**/8192 (~40 GiB). Confirms InfoNCE microbatch grew with `bs`, and milder drop retained more shared seeds.

**Next:** replicate edge_drop_0.05 only. Do **not** pursue large_bs further. Defer fanout_200 and edge_drop_0.00.

### Closing update (2026-07-20)

Seed1 quickcheck completed and **closed**: edge_drop_0.05 improves **pre-3h+raw** on matched seeds 1–2; **embedding-only mixed**; **final D mixed**. Keep **diagnostic only** / not thesis-table eligible. **Do not** train seed3; **do not** run edge_drop_0.00 or fanout_200. See `notes/edge_drop_0.05_seed1_quickcheck.md`.

---

## Artifact paths

- This note: `notes/contrastive_objective_improvement_audit.md`
- Machine-readable twin: `results/diagnostics/contrastive_objective_improvement_audit.json`
- Scout results: `results/diagnostics/contrastive_objective_resource_scout.json`
- Seed1 quickcheck (closed): `results/diagnostics/edge_drop_0.05_seed1_quickcheck.json`
