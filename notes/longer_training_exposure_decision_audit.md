# Longer-training exposure & resource decision audit (read-only)

> Twin: [`results/diagnostics/longer_training_exposure_decision_audit.json`](../results/diagnostics/longer_training_exposure_decision_audit.json)  
> Package: [`results/diagnostics/longer_training_exposure_decision_audit/`](../results/diagnostics/longer_training_exposure_decision_audit/)  
> **Final decision: `PENDING_CHECKPOINT_LADDER`** — did not read in-progress ladder cell metrics.

No training, jobs, checkpoint edits, or resume logic.

---

## What “exposure” means here

Phase-4B mixed SSL does **not** run full-graph epochs. Each optimizer step pulls a `LinkNeighborLoader` seed batch (`batch_size=8192`, `num_neighbors=[100,100]`) from an **infinite** per-domain stream, then drops seeds that fail dual-view R198 alignment. Logged fields:

| Field | Meaning |
|--------|---------|
| `requested_seeds` | Seeds entering the step (almost always 8192; rare short wrap batches) |
| `realized_seeds` / `scored_seeds` | Seeds remaining after `align_seed_r198_pair` (~66xx HI/LI; ~58xx SAML-D) |

**Approximate passes** = (sum of seed counts) / `n_train_edges`. These are **not** epochs: neighborhood sampling, repetition, and realized≪requested make them non-equivalent to supervised epoch sweeps.

---

## Inventory (completed LONG@3000 logs)

Train-edge counts from scaler provenance (authoritative for training graphs):

| Dataset | Train edges | Config BS | Realized seeds (mean / 1000 upd) | Updates @1500 glob | Updates @3000 | Realized edge exposures @500 / @1000 upd | Approx passes @1000 (cfg 8192 / realized) |
|---------|-------------|-----------|----------------------------------|--------------------|---------------|------------------------------------------|-------------------------------------------|
| Small-HI | 3 248 921 | 8192 | 6629 | 500 | 1000 | 3.32M / 6.63M | 2.52 / **2.04** |
| SAML-D | 5 715 293 | 8192 | 5836 | 500 | 1000 | 2.92M / 5.84M | 1.43 / **1.02** |
| Small-LI | 4 432 934 | 8192 | 6627 | 500 | 1000 | 3.32M / 6.63M | 1.85 / **1.50** |

Probe-matched EdgeID counts differ slightly (esp. SAML-D 5.03M vs 5.72M) — use scaler counts for exposure; probe counts for frozen-eval cohorts.

Full table: `exposure_table.csv`. Longer projections: `projected_exposures_longer_schedules.csv`.

### Sampling / repetition caveats

- Infinite loader ⇒ seeds can repeat long before a full pass.
- Dual-view alignment drops ~19% (HI/LI) to ~29% (SAML-D) of requested seeds.
- Graph neighborhoods are subsampled (`100,100`), so each “exposure” is a local subgraph, not the full incident structure.
- Round-robin HI→SAML→LI: domain exposure is matched, but LR phase is shared globally.

---

## Historical wall times (not equal difficulty)

| Run | Wall | Notes |
|-----|------|--------|
| MIXED LONG 3000 steps | ~1.85 h (`elapsed_sec` 6651) + ~19 min graph build | `mean_sec_per_step` ≈ 1.82; submitted `04:00:00` on `mit_preemptable` |
| EXPERT_ONLY 3000 | ~1.24 h | Faster step (~1.10 s) |
| Supervised SAML-D 50 ep | ~8.82 h projected | Different objective, `batch_size=4096`, workers=16 — **duration context only** |
| Supervised Small-HI 50 ep | wall not in summary JSON | Same qualitative caveat |

Do not treat “SSL ≪ 50 supervised epochs” as under-training proof without the ladder’s step-wise frozen curves.

---

## Candidate longer schedules (design only)

Assume same MIXED step rate and ~1130 s three-domain graph build. Four-domain adds PaySim train edges 3 792 821 (TF cache) and an **unmeasured** ~400 s graph-build placeholder.

| Updates/domain | Domains | Global steps | Est. MIXED train wall | Suggest Slurm | `mit_normal_gpu` 6 h? | Continuation |
|----------------|---------|--------------|----------------------|---------------|------------------------|--------------|
| 2000 | 3 | 6000 | ~3.4 h | ~4.7 h | borderline / prefer 6 h | **Fresh schedule** |
| 2000 | 4 | 8000 | ~4.7 h | ~6.3 h | tight | Fresh |
| 3000 | 3 | 9000 | ~4.9 h | ~6.6 h | need ≥8 h or preemptable | Fresh |
| 3000 | 4 | 12000 | ~6.8 h | ~9 h | no (6 h) | Fresh |
| 5000 | 3 | 15000 | ~7.9 h | ~10.4 h | use preemptable/advanced longer | Fresh |
| 5000 | 4 | 20000 | ~11 h | ~14 h | preemptable | Fresh |

Checkpoint proposal: ~4–5 milestones spanning the horizon + `rolling_every=100` (~3 MiB each). Embedding extract for one final encoder × 3 targets ≈ **8.6 GiB** (historical). Host/GPU memory historically ~9.5 GiB RSS / ~13 GiB CUDA alloc under 128 G — 4-domain needs a residency smoke before trusting the same envelope.

Details: `schedule_options.csv`, `resource_estimates.csv`.

### Fresh vs resume

LONG@3000 was explicitly **fresh from shared init**, not a resume of SHORT@1500, with a **rescaled** LR (warmup 600 / decay 2400). Seed-batch hashes are unique per step; `NeighborLoader` stream position is not checkpointed for exact replay.

**Do not silently resume from step 3000.** Any longer run should be a new schedule (scaled 20% warmup) unless/until exact stream replay is implemented and verified.

### Confounds if a longer run is authorized

1. **Extra exposure** (more updates/domain)  
2. **Schedule horizon** (SHORT vs LONG at matched mid-checkpoint — already visible historically)  
3. **LR phase** at a given exposure (warmup/decay scale with total steps)  
4. **Fresh train vs resume**

Report all four; do not attribute gains to “more data” alone.

---

## Decision gate (await ladder finalize)

Ladder design (completed note only): MIXED milestones 750/1500/2250/3000; EXPERT 1500/3000; validation frozen R198. **Decision stays `PENDING_CHECKPOINT_LADDER`.**

| Ladder pattern | Action |
|----------------|--------|
| Peak ≤1500 | **No** longer arm |
| Peak @2250; 3000 flat/down | **No** longer arm (optional peak refinement only) |
| Improves through 3000 on ≥2/3 datasets | **One** arm = ladder winner; schedule **`U2000_D3`** only |
| Datasets disagree on best step | Longer only if HI+LI want more; never solely for SAML |
| Only one objective still rising | Longer **that** objective only |
| Strong late gains | Still start at U2000; escalate to U3000 only after U2000 frozen eval — **never jump to U5000** |

Hard caps: ≤1 longer arm, ≤1 schedule; 4-domain longer training out of scope until PaySim smoke + 3-domain decision.

Full JSON: `decision_rules.json`.

---

## Provisional (not authorized) pick

**If** finalize shows improvement through 3000 for one objective: that objective @ **2000 updates/domain / 6000 global**, fresh LR, milestones 1500/3000/4500/6000.

Until then: **`PENDING_CHECKPOINT_LADDER`**.

---

## Confirmations

- No encoder/expert training, no submits, no checkpoint modification, no resume code  
- Used completed LONG/EXPERT summaries + frozen-eval notes only  
- Did not consume partial checkpoint-ladder probe/extract metrics
