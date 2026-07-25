# Contrastive augmentation feasibility audit (read-only)

**Date:** 2026-07-20  
**Status:** Read-only. No jobs launched. No code changes.  
**Context:** `edge_drop_0.05` closed as diagnostic/promising but **not promoted** (B improves seeds 1–2; A and D mixed). Do not continue rate sweeps (`0.00`, seed3, fanout). This note asks whether *small design changes* are feasible instead.

**Current default views** (`graph_augmentations.generate_views` via `training._contrastive_view_kwargs`):

| Knob | Current | CLI? |
|------|---------|------|
| Edge drop rate | 0.1 (scouted 0.05) | `--edge_drop_target_rate` |
| Edge drop policy | `random` | `--edge_drop_policy` |
| Edge-attr mask rate | **0.1 hardcoded** | **No** |
| View symmetry | Both views get the **same** drop/mask rates (independent draws) | N/A |
| Seed/anchor protection | **None** — seeds can be dropped | N/A |

Documented explicitly in `generate_views`:

> Seed/anchor edges are **not** protected: if a seed edge is dropped from one view, `select_shared_seed_edge_embeddings` excludes it from the contrastive loss.

Empirical cost: at `drop=0.1`, ~6600/8192 shared seeds; at `drop=0.05`, ~7400/8192 — milder drop already recovers seeds, but does not protect anchors by design.

---

## 1. Anchor-preserving edge drop

### What it would mean
When building each view, **force-keep** seed/anchor transaction `edge_id`s (the batch’s contrastive positives) while still dropping context/message-passing edges at the usual rate. Shared-seed count would approach `requested_seed_edges` (modulo other filters).

### Current code surface
- Drop helpers: `_random_edge_drop_view`, `_hetero_random_edge_drop_view`, policy variants (`graph_augmentations.py`).
- Call site already has `seed_edge_ids` before `generate_views` in both homo and hetero contrastive loops (`training.py`).
- Keep mask is Bernoulli over all forward edges; no `preserve_ids` argument today.

### Implementation size (estimate)
**Small–medium**, localized:

1. Add optional `preserve_edge_ids` (or `seed_edge_ids`) to `generate_views` / drop helpers.
2. After sampling `keep`, OR-in membership of preserved ids (hetero: preserve on forward; reverse sync already follows kept forward ids).
3. Thread `seed_edge_ids` from the train loop into `_contrastive_view_kwargs` / `generate_views`.
4. Unit test: with `drop_rate=1.0` and preserve set, all preserved ids remain in both views.

No new objective; still label-free. Does **not** require degree-aware caches.

### Risk / science notes
- Changes the InfoNCE denominator composition (more anchors survive) and the GNN context (seeds always present).
- Could undo some of the “harder view” pressure that GraphCL-style drop intends.
- Highest conceptual fit to the observed shared-seed leakage (~10–20% anchors lost per batch).

### Feasibility verdict
**Feasible with a small, well-scoped patch.** Best next *code* experiment if revisiting augmentation — better EV than another global drop-rate sweep.

---

## 2. Configurable edge-attribute mask rate

### What it would mean
Expose today’s hardcoded `edge_attr_mask_rate=0.1` as a CLI flag (e.g. `--edge_attr_mask_rate`), default **0.1** for backward compatibility. Optionally scout `0.0` / `0.05` without changing drop.

### Current code surface
- `mask_edge_attr(...)` already parameterized (`graph_augmentations.py` ~293–344).
- `generate_views(..., edge_attr_mask_rate=0.1, ...)` already accepts the rate.
- **Only** wiring gap: `training._contrastive_view_kwargs` hardcodes `"edge_attr_mask_rate": 0.1` (`training.py` ~226–236).
- **No** `--edge_attr_mask_rate` in `util.py` today.

### Implementation size (estimate)
**Trivial (flag-only):**

1. Add `parser.add_argument("--edge_attr_mask_rate", type=float, default=0.1)`.
2. Replace hardcoded `0.1` in `_contrastive_view_kwargs` with `getattr(args, "edge_attr_mask_rate", 0.1)`.
3. Smoke test / help string.

### Risk / science notes
- Orthogonal to edge topology drop; may matter for AML amount/currency/format features.
- Prior masked-edge *aux loss* was weak; this is GraphCL-style attr masking on views, not a reconstruction head — different lever.
- Lowest risk change; easy A/B vs baseline.

### Feasibility verdict
**Easiest win for enabling experiments.** Recommend implementing the flag before any new scout; a single seed2 probe at `mask=0.0` or `0.05` (holding drop=0.1 or 0.05) would be enough to test.

---

## 3. Asymmetric clean / noisy views

### What it would mean
Give the two views **different** augmentation strength, e.g.:

- View1 **clean**: `edge_drop_rate=0`, `edge_attr_mask_rate=0` (or light mask only)
- View2 **noisy**: current drop/mask (or stronger)

Often paired with existing `--contrastive_asymmetric` (view2 `no_grad`, z1→z2 only), but that flag today only changes **gradient flow**, not augmentation asymmetry.

### Current code surface
- `generate_views` always applies the **same** `edge_drop_rate` / `edge_attr_mask_rate` to both views (independent random draws).
- No `edge_drop_rate_v1/v2` or `view_noise_mode` parameters.
- Asymmetric InfoNCE path already exists and is the default recipe.

### Implementation size (estimate)
**Small**, slightly larger than attr-mask CLI:

1. Extend `generate_views` to accept per-view rates *or* a mode `symmetric | clean_noisy`.
2. Clean path: skip drop/mask for view1; apply rates only to view2 (or vice versa).
3. CLI: e.g. `--contrastive_view_mode {symmetric,clean_noisy}` + reuse existing rate flags for the noisy view.
4. Log realized drop/mask per view (stats hooks already partially exist for edge drop).

### Risk / science notes
- Interacts with anchor survival: clean view keeps all seeds; noisy view may still drop them unless combined with anchor-preserving.
- Changes positive difficulty and can bias the asymmetric loss toward the clean encoder path.
- Not previously scouted in this repo; one carefully named seed2 run would be enough — not a grid.

### Feasibility verdict
**Feasible with a small API extension.** Prefer after (or with) configurable attr-mask; consider combining with anchor-preserving on the noisy view only.

---

## Ranking (if any future work)

| Rank | Idea | Effort | Expected value | Depends on code? |
|-----:|------|--------|----------------|------------------|
| 1 | Configurable `--edge_attr_mask_rate` | Trivial | Enables cheap ablations | Flag wiring only |
| 2 | Anchor-preserving edge drop | Small–medium | Directly addresses seed leakage | Localized preserve mask |
| 3 | Clean/noisy asymmetric views | Small | New GraphCL-style lever | Per-view rates in `generate_views` |

### Explicitly skip (still)
- Further `edge_drop_target_rate` sweeps (0.00, etc.)
- Seed3 edge_drop without a new design
- `fanout_200`
- Degree-aware / degree-flow-aware drop (prior negative)
- Hard-neg mining, TF soft-pos, morph, reconstruction aux in this lane

### Jobs
**None launched** as part of this audit.
