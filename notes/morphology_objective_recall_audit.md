# Morphology objective recall audit (pre-audit only)

**Scope:** read-only pre-audit for a future recall-oriented morphology/structural contrastive-objective batch.
**No jobs submitted, no training, no evaluation, no experiment JSONs modified, no model behavior changed.**

Companion machine-readable file: `results/diagnostics/morphology_objective_recall_audit.json`.

Primary representation for ranking (per current thesis policy): **pre-3h**; post-128 diagnostic only.
Recall-oriented metrics required for any probe summary (finalized in `ranking_metrics.py`):
AUROC, AUPRC, F1, P/R/Lift@{100,500,1000}, recall at precision ≥ {0.95, 0.90, 0.80, 0.70}.

---

## 1. Existing objective code and flags

| Objective | Loss family | Code | Enable flag | Status |
|---|---|---|---|---|
| Morphology **expert head** (Papagei-style) | **Regression** (MSE/MAE) | `morphology/expert.py` (`MorphologyExpertHead` :131, `MorphologyGroupedExpertHead` :151, `setup_morphology_expert` :651, `morphology_expert_step` :531) | `--morph_expert` | Mature; best morph SSL uses it |
| Morphology **contrast (M2)** | **Bin soft-positives** (contrastive) | `morphology/contrast.py` (`setup_morphology_contrast` :128, `estimate_morph_bin_edges` :250, `assign_morph_bin_ids` :219) | `--morph_contrast` | **Negative result** (see §6) |
| Temporal-flow **aux** | **Regression or bins** | `morphology/temporal_flow_aux.py` (`setup_temporal_flow_aux` :201, `temporal_flow_aux_loss` :292, `_fit_bins_train_only` :149) | `--aux_temporal_flow {none,regression,bins}` | Regression works (`tf_reg_w0.10`); bins fail |
| Temporal-flow **soft positives** | **Bin soft-positives** (contrastive) | `morphology/temporal_flow_soft_positives.py` | `--temporal_flow_soft_positives` | **Negative result** (this workstream) |

Wiring in `training.py`: `setup_morphology_expert` :1655, `setup_morphology_contrast` :1658, `setup_temporal_flow_aux` :1664, `setup_temporal_flow_soft_positives` :1674, tier0 :1710, tier0-flow :1713, tier2 :1717. Expert loss added to InfoNCE at homo :540 / hetero :1127; TF-aux loss at homo :558 / hetero :1145.

**Full morphology / aux CLI flags (defaults) — all in `util.py`:**

Expert head (:483–584): `--morph_expert` (False), `--morph_targets` (`local`; `local|local+global|local+tier2|local+global+tier2`), `--morph_tier0_cache` (None), `--morph_flow_balance` (False), `--morph_tier0_flow_cache` (None), `--morph_tier2_cache` (None), `--morph_tier2_lift` (`full`), `--morph_expert_loss` (`mse`; `mse|mae`), `--morph_expert_weight` (1.0), `--morph_expert_hidden` (64), `--morph_expert_layout` (`shared`; `shared|grouped`), `--morph_expert_group_weight_tier2` (1.0), `--morph_local_subset` (`all`; `all|degree|clustering|triangles`), `--morph_target_groups` (`all`), `--no_morph_edge_native` (False).

M2 contrast (:586–637): `--morph_contrast` (False), `--morph_contrast_features` (`local_ego,local_degree`), `--morph_contrast_scope` (`local`; `local|local+global`), `--morph_contrast_bins` (5), `--morph_contrast_calib_batches` (32), `--morph_contrast_max_soft_positives` (256), `--morph_val_every` (1), `--morph_val_max_batches` (0).

Loss-type selection: expert head is **always regression** (`--morph_expert_loss` → `mse→F.mse_loss`, `mae→F.l1_loss`; `expert.py` :261). M2 is **always binned contrastive**. TF-aux switches regression vs bins via `--aux_temporal_flow`.

---

## 2. Existing cached targets / features

| Cache / target | Producer | On disk now? | Notes |
|---|---|---|---|
| `results/cache/temporal_flow_causal/{Small-HI,Small-LI}/` (features.npy, edge_id.npy, split_*_edge_id.npy, meta.json) | `scripts/build_temporal_flow_causal_cache.py` | **Yes** | 5 causal features; `uses_labels:false`; used by TF-aux + soft positives |
| Tier-0 degrees `{split}_node_morphology.csv` (deg_in/out/total) | `scripts/precompute_morphology_tier0.py` | **No** (computed on-the-fly at startup) | O(nodes+edges), cheap |
| Tier-0 flow balance `{split}_node_flow_balance.csv` (amount_in/out) | `scripts/precompute_morphology_tier0_flow.py` | **No** | O(nodes+edges), cheap |
| Tier-2 betweenness `{split}_node_tier2.csv` (bc) | `scripts/precompute_morphology_tier2.py` | **No** | Brandes sampled; **~128G RAM**, can OOM |
| Tier-1 local (degree/clustering/triangle/ego) | `morphology/tier1_local.py` | N/A — **batch-local**, never persisted | Free during training |

Semantic target groups: `morphology/target_registry.py` (`MORPH_TARGET_GROUPS` :18). Structural-feature-group status:

| Group | Impl | Target/feature names | Cache |
|---|---|---|---|
| degree_fan | Yes (Tier0 global + Tier1 local) | `sender/receiver_deg_{in,out,total}`, `deg_sum_*` | tier0 CSV (on-the-fly) |
| flow_balance | Yes (Tier0 flow, 10-d) | `*_in/out_amount_log`, `*_flow_balance_ratio`, `*_abs_flow_imbalance_log`, edge-to-node ratios | tier0 flow CSV (on-the-fly) |
| local_motif (triangles) | Yes (batch-local); alias → `motif_participation,local_density,local_context_size` | `sender/receiver/mean_triangles_local` | none (batch-local) |
| triangle / clustering | Yes (batch-local) | `sender/receiver/mean_clustering_local` | none (batch-local) |
| centrality / betweenness | Yes (Tier2 BC); alias `centrality → global_role` | `sender/receiver_bc`, `bc_{sum,max}_global` | tier2 CSV (expensive) |

---

## 3. Which groups are regression-ready (expert head)

Ready **now** via `--morph_expert` (all consumed as regression targets on `z_seed`):

- **degree_fan** — `--morph_targets local` (Tier-1) or `local+global` (Tier-0 lift). Cheap, mature, best morph SSL uses it.
- **flow_balance** — `--morph_flow_balance` (10-d). Cheap on-the-fly cache.
- **local_density / clustering** — batch-local, included by default in `local` targets; `--morph_local_subset clustering`.
- **motif_participation / triangles** — batch-local; `--morph_local_subset triangles`.
- **global_role / centrality (BC)** — regression-ready via `--morph_targets ...+tier2`, but requires the expensive tier2 cache (see §5) and stacking hurt (see §6).

Loss weight/type: `--morph_expert_weight`, `--morph_expert_loss {mse,mae}` (MSE default; MAE not better on Small-HI).

## 4. Which groups are bin/contrastive-ready (M2 soft positives)

Ready via `--morph_contrast` + `--morph_contrast_features`: `local_ego`, `local_degree`, `local_clustering`, `local_triangles`, `global_degree` (needs `--morph_contrast_scope local+global`), `edge_native`. All feed bin-grouped InfoNCE soft positives (`contrastive_loss.edge_identity_infonce_loss`).

**Caveat:** every binned/soft-positive contrastive morphology objective tried so far is a **negative result** (M2 and TF soft positives). Bin-contrastive is *code-ready* but *evidence-negative* — not recommended for the recall batch.

## 5. Which groups are expensive or risky

- **centrality / betweenness (Tier-2 BC)** — **expensive**: Brandes sampled precompute needs ~128G RAM and OOMs on login nodes; and **risky**: stacking BC on degree lift consistently hurt the probe (see §6). Defer.
- **M2 morphology-bin contrast** — **risky**: negative result; contrast-only collapses (AUROC 0.680).
- **Temporal-flow soft positives** — **risky**: negative result; caps saturate.
- **Triangles stacked with clustering** — mildly risky: regressed vs clustering-only (−0.017 test AUROC), large val→test gap.
- Tier-1 local features are batch-size dependent (O(batch subgraph)); fine at scout scale but not a stable global signal.

## 6. Prior metrics: recall improved but precision hurt

From `notes/morphology-metrics-plan.md` (Small-HI, GIN, val-tuned F1 probe; P/R at the selected threshold):

| Config | Test AUROC | F1 | Precision | Recall | Recall↑ vs baseline? | Precision behavior |
|---|---:|---:|---:|---:|---|---|
| Contrastive baseline | 0.839 | 0.076 | 0.053 | 0.137 | — | — |
| **M1 (local expert)** | 0.910 | 0.079 | **0.046** | **0.290** | **Yes (0.137→0.290)** | **Collapsed (0.053→0.046)** |
| M1b (local+global) | 0.920 | 0.108 | 0.069 | 0.248 | Yes | Improved (no collapse) |
| M1b + triangles-only + proj | 0.910 | **0.067** | — | — | — | **F1 collapse is precision/threshold** (noted in plan) |
| M1b + sym + proj | 0.930 | **0.134** | — | — | over-flags | **−0.088 F1 vs sym-only (0.222)** — precision/threshold loss |
| M2 contrast-only | 0.680 | 0.012 | — | — | — | Degenerate |

Key precision-collapse cases: **M1 local-only expert** (recall nearly doubles, precision drops below baseline); **triangles-only + proj** and **M1b + sym + proj** over-flag (F1/precision collapse at the val-tuned threshold despite strong AUROC). Betweenness stacking (`M1b+BC`) *hurt AUROC* (0.896/0.861 vs 0.920) rather than trading recall for precision.

**Important gap:** no morphology run currently has precision-constrained recall metrics — recall@P≥0.90/0.80 and P/R@K were added to `ranking_metrics.py` *after* these runs and were only backfilled for TF-aux/baseline. So "recall improved but precision collapsed" above is inferred from threshold-selected P/R and F1, **not** from the alert-budget metrics we now require.

## 7. Recommended smallest recall-oriented batch (run AFTER current jobs finish)

Design goals: use the **regression-ready, cheap** groups only; **exclude** the negative-result contrastive-bin objectives (M2, soft positives) and the expensive/risky BC; evaluate on **pre-3h primary** with the finalized recall metrics.

Smallest sensible batch = **1 baseline re-probe (no training) + 2 training arms**, all GIN Small-HI 20ep, same strong recipe (asym proj, 8192 neg, queue0, temp0.5, reverse_mp+ego+ports+emlps+tds, bs=8192/accum=4, seed1), `--checkpoint_policy best`:

| # | Arm | Objective | Training? | Purpose |
|---|---|---|---|---|
| 0 | `hi_contrastive_gin_emlps_tds_proj_asym_8192neg_queue0_accum4_20ep` (existing ckpt) | plain contrastive+proj | **No — re-probe only** | Recall-metric baseline (currently missing recall@precision) |
| 1 | degree_fan + flow_balance expert + proj | `--morph_expert --morph_targets local+global --morph_flow_balance` | Yes | Cheapest regression morphology signal; degree+flow are the proven groups |
| 2 | clustering expert + proj (best known morph SSL) | `--morph_expert --morph_targets local+global` (default 14-local incl. clustering) | Yes | Reproduce best morph SSL (0.929 AUROC) under recall metrics |

Evaluation for all three (pre-3h primary; post-128 diagnostic), via a recall-metric-enabled probe (`scripts/probe_feature_ablation.py` or `probe_temporal_flow_ablation.py`), arms: pre-3h embedding-only, pre-3h+raw, pre-3h+raw+temporal-flow; report AUROC/AUPRC/F1, P/R/Lift@{100,500,1000}, recall@P≥{0.95,0.90,0.80,0.70}.

**Explicitly excluded from the smallest batch:**
- Tier-2 betweenness / centrality (expensive ~128G RAM; stacking hurt).
- M2 morphology-bin contrast and any bin/soft-positive contrastive morphology objective (negative results).
- Triangles stacked with clustering (regressed); grouped/per-metric expert layouts (M5) — no evidence they beat shared M1b.

**Caches to (optionally) precompute once before the batch** (cheap, avoids on-the-fly startup cost, no leakage — train-split only): `scripts/precompute_morphology_tier0.py` and `scripts/precompute_morphology_tier0_flow.py` → `morphology_cache/Small-HI`. Not required (training computes on the fly), and **not** to be run as part of this pre-audit.

**Stop rule alignment:** if arm 1/2 do not improve pre-3h A or B AUPRC over the (re-probed) baseline and do not improve recall at acceptable precision, morphology-objective expansion should stop before touching BC/M2.

---

## 8. Results — batch complete (seed 1) and stop-rule resolution

> Added after the planned batch ran. The §1–§7 pre-audit above was read-only and remains accurate as of submission; this section records outcomes only. All arms are **diagnostic/scout** (`table_eligible: false`, no labels in SSL, pre-3h primary, post-128 diagnostic). Machine-readable summary: `results/diagnostics/morphology_objective_recall_scout.json`; per-arm probes: `results/diagnostics/morph_obj_*_seed1.json`; rendered table: `notes/morphology_objective_recall_scout.md` / `tables/morphology_objective_recall_scout.md`.

**Pre-3h primary AUPRC (arm A embedding-only / B +raw / D +raw+TF-causal), best-val checkpoint epoch:**

| Variant | ckpt ep | A AUPRC | B AUPRC | D AUPRC | A P@100 | A R@P≥0.80 |
|---|---:|---:|---:|---:|---:|---:|
| baseline re-probe | 19 | 0.1888 | 0.2113 | 0.4337 | 0.66 | 0.0006 (1 alert) |
| **degflow** (deg_fan+flow_balance expert) | 13 | **0.2828** | **0.3719** | **0.4740** | **0.85** | **0.121** (243 alerts) |
| clustering (local+global expert) | 9 | 0.0861 | 0.1494 | 0.2696 | 0.14 | 0.0006 (1 alert) |
| degflow_tfreg (deg+flow + TF-reg λ=0.05) | 1 | 0.1275 | 0.1819 | 0.3134 | 0.46 | — (NaN) |

**Findings:**

1. **`degflow` is the clear winner and the only arm that clears the stop rule.** It improves pre-3h AUPRC on every stack over the re-probed baseline — embedding-only 0.1888→**0.2828** (+0.094), +raw 0.2113→**0.3719** (+0.161), +TF-causal 0.4337→**0.4740** (+0.040) — and lifts embedding-only precision@100 from 0.66→**0.85**. Critically for the recall objective, it opens a *usable precision-constrained operating point where the baseline had none*: baseline embedding-only R@P≥0.80 is degenerate (1 alert), while degflow reaches recall **0.121** at 80% precision (243 alerts) and **0.0168** at 90% precision.
2. **AUROC/AUPRC tradeoff (expected, benign).** degflow's AUROC is *lower* on every arm (A 0.9608→0.9331), i.e. it sacrifices some global ordering but concentrates ranking quality at the top of the list — the correct tradeoff for alert-budget deployment. Report AUPRC / P@K / recall@P, not AUROC, for this arm.
3. **`clustering` collapses — confirmed negative result.** Embedding-only AUPRC roughly halves (0.1888→0.0861) and P@100 falls 0.66→0.14. Consistent with the §6 prior that clustering-heavy expert targets over-flag / hurt precision. Keep excluded.
4. **`degflow_tfreg` (the combo) underperforms plain degflow and looks unstable.** Every stack is below degflow (A 0.1275, B 0.1819, D 0.3134), and its best-val checkpoint is **epoch 1** (vs degflow ep13, baseline ep19) — the TF-regression aux short-circuited SSL training. Skip the combo.
5. **Threshold caveat.** Several val-tuned F1 values are degenerate (degflow B/D F1 ≈ 0.03–0.05, clustering B ≈ 0.08) because the val-selected threshold lands near "flag-everything" (recall ≈ 0.9, precision ≈ 0.02). This is a threshold-transfer artifact, **not** a model-quality signal — the auto-generated scout table shows these raw F1s without the caveat, so rely on AUPRC / alert-budget / recall@P instead.
6. **post-128 is diagnostic only** and its recall@P metrics are mostly NaN (degenerate) — as expected under current policy.

**Stop-rule resolution (seed-1 only):** `scale_morph_only_skip_combo`.
- **Proceed to multiseed** with `degflow` (degree_fan + flow_balance expert regression): it passed both bars on seed 1 (AUPRC up on A *and* B; recall up at acceptable precision). Results stayed diagnostic until seeds 2–3 completed (see §9).
- **Stop / exclude** as planned: `clustering` (precision collapse), the `degflow_tfreg` combo (worse + unstable), and — unchanged — tier-2 betweenness, M2 bin-contrast, and TF soft positives. Do **not** expand to BC/M2.

---

## 9. Multiseed replication (seeds 2–3) — does **not** replicate; stop

> Focused degflow-only replication. No clustering / degflow_tfreg / M2 / soft positives / betweenness. Summary: `results/diagnostics/degflow_morphology_multiseed_scout.json`, `notes/degflow_morphology_multiseed_scout.md`, `tables/degflow_morphology_multiseed_scout.{md,tex}`. Registry rows marked `table_group=degflow_morphology_multiseed_scout`, `diagnostic_only`, `table_eligible=false`.

**Pre-3h degflow vs matched baseline (where available):**

| Seed | Variant | ckpt ep | A AUPRC | B AUPRC | D AUPRC | A P@100 | A R@P≥0.80 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | baseline | 19 | 0.1888 | 0.2113 | 0.4337 | 0.66 | ~0 (1 alert) |
| 1 | degflow | 13 | **0.2828** | **0.3719** | **0.4740** | **0.85** | **0.121** |
| 2 | baseline | 14 | 0.2598 | 0.2725 | **0.5111** | 0.79 | 0.078 |
| 2 | degflow | 13 | 0.0756 | 0.0362 | 0.1126 | 0.08 | — |
| 3 | baseline | — | unavailable (not retrained) | — | — | — | — |
| 3 | degflow | 20 | 0.0828 | 0.1185 | 0.2348 | 0.00 | — |

Degflow mean±SD (n=3): A AUPRC **0.147±0.118**, B **0.176±0.175**, D **0.274±0.184**. Seed-1 alone drives the means; seeds 2–3 sit near 0.08 AUPRC.

**Claim 1 — Representation improvement (A/B before TF features):** **Fails multiseed.**
- Seed 1: clear win (ΔA AUPRC +0.094, ΔB +0.161, A P@100 0.66→0.85, usable R@P≥0.80).
- Seed 2: **precision collapse** (A P@100 0.79→0.08; ΔA AUPRC **−0.184**, ΔB **−0.236**). Flagged by scout heuristic.
- Seed 3: no matched baseline, but absolute A/B AUPRC (~0.08 / 0.12) and P@100 (0.00 / 0.03) match the collapsed seed-2 pattern, not seed 1.

**Claim 2 — Final D-stack tradeoff:** **Not a win; often a loss.**
- Seed 1: degflow D raises AUPRC (+0.040) and R@1000 (+0.042) but **hurts** P@100 (−0.12) and R@P≥0.90 (−0.099) vs baseline D — the precision/recall tradeoff noted in the scout brief.
- Seed 2: degflow D is worse on essentially every metric (ΔAUPRC **−0.399**, ΔP@100 **−0.90**, ΔR@1000 **−0.296**). Best D stack overall is **baseline_seed2** (AUPRC 0.511, R@P≥0.90 0.136).
- Do **not** claim degflow is the new best final method.

**Training note (not a pipeline failure):** morph expert MSE trains normally on seeds 2–3 (`morph/expert_train` ~0.5 → ~0.01; ssl_labels_used=false; target_dim=25 degree_fan+flow_balance). Best ckpt ep13 (seed2) / ep20 (seed3). The expert fits; the AML probe ranking does not transfer stably across seeds.

**Final stop-rule resolution:** **`stop`**.
- Do **not** promote degflow from diagnostic to thesis result.
- Do **not** run a 40ep degflow scale-up.
- Do **not** expand to BC / M2 / soft positives on the basis of the seed-1 scout.
- Keep seed-1 degflow as a **single-seed diagnostic curiosity** (representation gain + D precision tradeoff), not a reproducible morphology-objective claim.
