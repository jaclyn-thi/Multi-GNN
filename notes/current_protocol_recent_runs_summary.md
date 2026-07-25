
# Current-Protocol Recent Runs Summary

Status: **current results** · Updated 2026-07-24 (GCPAL challenge full-stack eval indexed; no GNN training in that job).

**Canonical protocols:** [`thesis_protocol_families.md`](thesis_protocol_families.md) · **Evaluation rules:** [`evaluation_protocols.md`](evaluation_protocols.md) · **Paper Multi-GIN+EU:** ports TDS-off 50ep aggregate (mean F1 **0.660 ± 0.060**); TDS-on supervised is **not** paper-comparable · **Strongest valid edge-centric SSL tested:** corrected reverse TDS + `preserve_seed_edges` · **Txn-node GCPAL path:** preliminary diagnostics only (**not** exact reproduction) · **Challenge eval:** [`gcpal_challenge_fullstack_eval.md`](gcpal_challenge_fullstack_eval.md) (job 18678029) — temporal winner edge pre-3h `H+X+TF` MLP; random 40/60 below published F1 under reconstructed protocol (gate **PARTIAL**).

Historical body below retains Jul 19–20 overnight framing; Jul 21–22 A/B/C/D, batch E/F, poscomplete/posagg, and challenge eval are indexed in [`README.md`](README.md) and the experiment registry.


## Executive Summary

The recent batch is complete and does not need reruns. The strongest thesis-safe result remains Small-HI: frozen GINe SSL embeddings, especially with raw transaction features, produce high AUPRC/F1 and very high alert-budget enrichment. Small-LI is substantially harder because positive prevalence is tiny, but augmented frozen embeddings still provide meaningful ranking enrichment — and extracting `pre_embedding_3h` instead of the exported 128-d embedding now replicates across three plain-GINe seeds. A formal 100-epoch legacy supervised Small-LI run (Egressy-head, paper_argmax) reached test F1 **0.357** at best-val epoch 35, roughly doubling the 20-ep scout. The safe framing is not that one recipe universally wins; it is that SSL embeddings become most useful when combined with simple transaction features (and alert-budget metrics are the most honest operational readout for probes), while legacy supervised remains a separate, stronger end-to-end baseline on Small-LI when trained long enough.

## Experiment Status

| Experiment | Newest job/log | Output status | Output path | Existing note | Needs scientific analysis? | Needs rerun? | Reason |
|---|---|---|---|---|---|---|---|
| Small-LI generic probe sweep | 17137814 | complete and analyzed | `results/diagnostics/probe_sweep_small_li_current_protocol.json` | `notes/probe_sweep_small_li_current_protocol.md` | No | No | 48/48 complete; superseded for weighting claims by explicit sweep. |
| Small-LI supervised GINe baseline | 17069764 | complete and analyzed | `results/diagnostics/supervised_small_li_gin_emlps_tds_seed1.json` | `notes/small_li_supervised_baseline_comparison.md` | No | No | Final checkpoint is weak operationally but useful as supervised baseline. |
| Small-LI FNF + emlps+tds scout | 17069933 | complete and analyzed | `results/diagnostics/probe_feature_ablation_small_li_fnf_current_protocol_seed1.json` | `notes/small_li_fnf_current_protocol_comparison.md` | No | No | FNF effect is mixed; current note includes alert-budget comparison. |
| Small-LI explicit positive-weight sweep | 17137922 | complete and analyzed | `results/diagnostics/probe_weight_sweep_small_li_current_protocol.json` | `notes/probe_weight_sweep_small_li_current_protocol.md` | No | No | 72/72 complete; current note interprets weighting vs ranking. |
| Small-HI explicit positive-weight sweep | 17137924 | complete and analyzed | `results/diagnostics/probe_weight_sweep_small_hi_key_runs.json` | `notes/probe_weight_sweep_small_hi_key_runs.md` | No | No | 60/60 complete; robust to probe-weight variation. |
| Current-protocol alert-budget metrics | 17209094 | complete and analyzed | `results/diagnostics/alert_budget_metrics_current_protocol.json` | `notes/alert_budget_metrics_current_protocol.md` | No | No | Resume completed Small-LI 7/7 and combined 17 rows. |
| Representation location: pre-3h vs post-128 (HI + LI) | extract+probe (current protocol) | complete and analyzed | `results/diagnostics/pre_embedding_3h_vs_post_embedding_current_protocol.json` | `notes/pre_embedding_3h_vs_post_embedding_current_protocol.md` | No | No | Pre-`embedding_head` (`3×n_hidden`) ≥ exported 128-d; large Small-LI gain, same frozen checkpoint. |
| Small-LI exported-dim 128 vs 198 scout | train 17350448 → extract 17350449 → probe 17350450 | complete and analyzed | `results/diagnostics/small_li_embedding_dim_128_vs_198.json` | `notes/small_li_embedding_dim_128_vs_198.md` | No | No | Widening export is confounded with retraining; pre-3h of existing 128-d ckpt still best. |
| Pre-3h vs post-128 on strong checkpoints (HI 40ep s2, HI FNF s1, LI FNF s1) | extract 17409110/17411075/17409112 → probe 17409113/17411076/17409115 → summary 17411077 | complete and analyzed | `results/diagnostics/pre3h_strong_run_comparison.json` | `notes/pre3h_strong_run_comparison.md` | No | No | Pre-3h wins AUPRC/AUROC in all 8 run×stack cells; gain shrinks as raw/morph added; Small-LI FNF alert-budget ≈2×. One HI FNF extraction hit a GPU ECC fault (`node4104`) and was resubmitted with `--exclude`. |
| Small-LI plain GINe SSL seeds 2–3 + pre-3h multiseed | train 17495620/17495621 → extract 17495622/17495623 → probe 17495624/17495625 → summary 17495627 | complete and analyzed | `results/diagnostics/pre_embedding_3h_vs_post_embedding_small_li_multiseed.json` | `notes/pre_embedding_3h_vs_post_embedding_small_li_multiseed.md` | No | No | Pre-3h wins AUPRC 3/3 seeds (emb-only and +raw); mean ΔAUPRC +0.025 / +0.029; seed-1 advantage replicated. Pairing coverage 1.0000. |
| Small-LI legacy supervised 100 ep (formal) | train 17495885/17495886 (50+50 resume) | complete and analyzed | `results/diagnostics/supervised_Small-LI_small_li_legacy_supervised_gin_emlps_tds_100ep_seed1_summary.json` | `notes/supervised_Small-LI_small_li_legacy_supervised_gin_emlps_tds_100ep_seed1_summary.md` | No | No | paper_argmax test F1 **0.357** / AUPRC **0.292** @ best-val ep 35; final-epoch F1=0 — must use `checkpoint_best_val_f1.tar`. |
| Morphology-objective recall scout (Small-HI, seed 1) | train degflow/clustering/degflow_tfreg → extract → probe → summary | complete and analyzed | `results/diagnostics/morphology_objective_recall_scout.json` (+ `morph_obj_*_seed1.json`) | `notes/morphology_objective_recall_scout.md`; audit §8 `notes/morphology_objective_recall_audit.md` | No | No | Seed-1 only: `degflow` looked promising; `clustering`/`degflow_tfreg` negative. Superseded for claims by multiseed (§ below). |
| Degflow morphology multiseed (Small-HI, seeds 1–3) | train 18322903/18322906 → extract/probe → sum 18322909 (+ seed2 baseline extract/probe) | complete and analyzed | `results/diagnostics/degflow_morphology_multiseed_scout.json` | `notes/degflow_morphology_multiseed_scout.md`; audit §9 | No | **No** | **Does not replicate.** Seed1 win; seed2 precision-collapse; seed3 weak abs. Recommendation **`stop`**. No 40ep. Diagnostic only. |
| TF-reg aux multiseed (Small-HI, w0.05/w0.10 × seeds 1–3) | stage0 18358988; train/probe seed2–3 → sum **18359001** | complete and analyzed | `results/diagnostics/temporal_flow_regression_aux_multiseed.json` | `notes/temporal_flow_regression_aux_multiseed.md` | No | **No** (unless later promoting w0.05) | Claim 1 holds for **w0.05** on paired seeds 1–2; Claim 2 (D) fails; w0.10 **stop** (seed2 regresses). Overall **`keep_diagnostic`**. |
| Contrastive resource scout (seed2 large_bs + edge_drop_0.05) | train 18359809/18359805 → probe → sum **18359812** | complete and analyzed | `results/diagnostics/contrastive_objective_resource_scout.json` | `notes/contrastive_objective_resource_scout.md`; audit addendum | No | edge_drop: optional seeds 1/3 | **large_bs stop** (A/B collapse). **edge_drop_0.05** modest A/B gain → **`replicate_edge_drop_only`**. No fanout / 0.00 yet. |
| Edge-drop 0.05 seed1 quickcheck | train **18400345** → extract/probe → sum **18400348** | **closed** (diagnostic) | `results/diagnostics/edge_drop_0.05_seed1_quickcheck.json` | `notes/edge_drop_0.05_seed1_quickcheck.md` | No | **No** | **Closed / not promoted.** B improves seeds 1–2; A mixed; D mixed. No seed3 / 0.00 / fanout. |
| GCPAL txn-node posagg ablation (B/C/D) | train C/D **18669617/18669618** | complete and analyzed | `results/diagnostics/gcpal_txn_node_posagg_ablation.json` | `notes/gcpal_txn_node_posagg_ablation.md` | No | **No** | Val HxX AUPRC selects **D SupCon**; feeds challenge candidate. |
| GCPAL challenge full-stack (no GNN train) | eval **18678029** | complete and analyzed | `results/diagnostics/gcpal_challenge_fullstack_eval.json` | `notes/gcpal_challenge_fullstack_eval.md` | No | **No** (fine-tune optional, not submitted) | Temporal: `edge_pre3h\|H+X+TF\|mlp`. Random 40/60 below 0.581/0.658 under reconstruction. Gate **PARTIAL**. |

## Best Small-HI Results

- Probe weighting: best F1/AUPRC is `Small-HI GINe emlps+tds seed2 (40ep)` with `embedding+raw`, `pos_1`, C=1.0: F1 0.3677, AUPRC 0.3314, P@500 0.6980, lift@500 373.9.
- Alert-budget current protocol: `Small-HI GINe emlps+tds seed2 (40ep)` with `embedding+raw` gives F1 0.3464, AUPRC 0.2883, P@500 0.6380, lift@500 341.8.
- Small-HI claims are robust to reasonable probe class-weight changes.

## Best Small-LI Results

- **SSL probe (frozen, fair policy):** best ranking/AUPRC remains `embedding+raw+morph`, `pos_1`, C=1.0: AUPRC 0.0496, P@100 0.2900, lift@100 424.2.
- **SSL + pre-3h lever (3-seed mean, embedding+raw):** pre-3h AUPRC **0.061 ± 0.034**, P@100 **0.34 ± 0.16**, lift@100 **502 ± 232** vs post-128 AUPRC 0.032 ± 0.021 (see multiseed note).
- **Legacy supervised (in-GNN, paper_argmax):** `small_li_legacy_supervised_gin_emlps_tds_100ep_seed1` @ best-val ep 35: test F1 **0.357**, AUPRC **0.292**, AUROC **0.959** — much stronger than the embedding-head supervised control or the 20-ep legacy scout (~0.18 F1).
- Explicit weighting: best F1 is `embedding+raw+morph`, `pos_3`, C=0.1: F1 0.1130, AUPRC 0.0450.
- Alert-budget best practical **SSL probe** configuration is `embedding+raw+morph`, with P@100 0.2300, P@500 0.1160, and lift@500 169.7.

## Alert-Budget Takeaways

- Small-HI has strong absolute precision at practical budgets; Small-LI has lower absolute precision but very high enrichment above prevalence.
- Embedding-only is useful, but augmented feature stacks are more reliable for practical alerting.
- FNF on Small-LI is mixed: it helps some thresholded F1 comparisons, but plain GINe full-stack is stronger at fixed alert budgets.
- Alert-budget metrics are more thesis-safe than thresholded F1 because they reflect what an analyst would actually inspect.

## Representation-Location Findings (pre-embedding vs export width)

- **Pre-embedding beats the exported embedding**, especially for rare positives. Probing the `3×n_hidden` tensor fed into `embedding_head` (`pre_embedding_3h`) instead of the exported 128-d `z` improves Small-LI AUPRC from 0.014 → 0.045 (embedding-only) and 0.024 → 0.083 (+raw); Small-HI shows a smaller gain (embedding-only AUPRC ≈ 0.198 → 0.224). Both representations come from the **same frozen checkpoint** — no retraining, extraction-only.
- **Widening the exported dim to 198 (emb198 scout) is not a shortcut.** A seed-1 retrain with `embedding_dim=198` only modestly improved the export (AUPRC 0.014 → 0.022) and stayed well below the existing model's pre-embedding (0.045). At fixed dim (198), the emb198 encoder was actually *worse* (its own pre-embedding 0.028 < original 0.045), i.e. it landed at a worse SSL optimum — so the export gain is confounded with retraining, not attributable to width.
- **Practical guidance:** for downstream probes/deployment, extract `pre_embedding_3h` from existing checkpoints (`--representation_source pre_embedding_3h`) rather than retraining with a wider head.

### Strong-checkpoint replication (HI 40ep seed2, HI same-pair FNF seed1, LI FNF seed1)

- **Consistent ranking win, stack-dependent size.** Across all three strong checkpoints, `pre_embedding_3h` (198-d) beat the exported 128-d embedding on AUPRC **and** AUROC in every feature stack (8/8 run×stack cells). But the margin shrinks as engineered features are added — e.g. HI FNF ΔAUPRC +0.076 (embedding-only) → +0.052 (+raw) → +0.014 (+raw+morph). Pre-3h's extra signal partly overlaps with the raw/morph features.
- **Small-LI FNF is the operational standout.** Pre-3h roughly doubles top-budget precision: P@100 0.13→0.28 (embedding-only) and 0.18→0.34 (+raw); lift@100 190→410 and 263→497. The plain-Small-LI pre-3h advantage clearly extends to FNF.
- **Best-of-batch:** best AUPRC (0.321) and best F1 (0.344) both come from pre-3h + raw on the **ordinary** HI 40ep seed2 run; best alert-budget (lift@100 497) from pre-3h + raw on LI FNF seed1.
- **FNF still doesn't beat ordinary on Small-HI**, even with pre-3h (ordinary full-stack AUPRC 0.321 > FNF full-stack 0.291) — consistent with the earlier "FNF is mixed" conclusion.
- **Ranking vs threshold:** the wins are cleanest on AUPRC/AUROC/alert-budget; val-tuned F1 occasionally regresses at the full (+morph) stack because the tuned threshold transfers imperfectly to the wider 198-d probe.

### Plain Small-LI multiseed replication (seeds 1–3, Jul 8)

- **Seed-1 advantage replicated.** Pre-3h wins AUPRC in **3/3 seeds** for both embedding-only and embedding+raw (pairing coverage 1.0000). Mean ΔAUPRC: **+0.025 ± 0.009** (embedding-only), **+0.029 ± 0.026** (+raw). Mean Δlift@100 (+raw): **+171 ± 158**.
- **Magnitude is seed-dependent.** Seed 1 has the largest gains (+raw AUPRC 0.024→0.082); seed 2 is a weak encoder overall (post AUPRC 0.005) but pre-3h still wins; seed 3 is strong (post +raw AUPRC 0.056, pre 0.079).
- **Practical read:** extracting `pre_embedding_3h` is justified for plain Small-LI SSL checkpoints; do not overclaim uniform absolute lift when the underlying seed is weak.

### Legacy supervised 100-ep formal reproduction (Jul 9)

- **Formal run (no `--testing`, 100 ep, seed 1):** best-val epoch **35**; post-hoc `paper_argmax` test F1 **0.357**, AUPRC **0.292**, AUROC **0.959** (vs 20-ep scout ~0.18 F1).
- **Late collapse:** final-epoch test F1 = **0.000** (epochs 96–100 all zero). Reproduction eval **must** use `saved-models/.../checkpoint_best_val_f1.tar`, not last/flat checkpoint.
- **Separate paradigm from SSL probes:** legacy supervised trains end-to-end with the Egressy `3h→50→25→2` head; it is not comparable to frozen linear-probe numbers without careful framing.

### Morphology-objective recall scout (Small-HI, seed 1) — superseded for claims by multiseed

Diagnostic/scout batch (audit §8). Seed-1 only: `degflow` looked strong on A/B AUPRC and opened usable R@P≥0.80; `clustering` and `degflow_tfreg` were negative. **Do not cite seed-1 degflow as a thesis result** — see multiseed below.

### Degflow morphology multiseed (Small-HI, seeds 1–3) — stop

Focused replication of degree_fan+flow_balance expert only (jobs `18322901`–`18322909`). Pre-3h primary. Full writeup: `notes/degflow_morphology_multiseed_scout.md` + audit §9.

**Claim 1 (representation A/B before TF features):** does **not** replicate.
- Seed 1: AUPRC A 0.189→0.283, B 0.211→0.372; A P@100 0.66→0.85.
- Seed 2 (matched baseline): **precision collapse** — AUPRC A 0.260→**0.076**, B 0.273→**0.036**; A P@100 0.79→**0.08**.
- Seed 3 (no matched baseline; not retrained): absolute A/B AUPRC **0.083 / 0.119**, A P@100 **0.00** — same weak regime as seed 2.
- Degflow mean±SD A AUPRC: **0.147±0.118** (n=3); seed 1 alone carries the mean.

**Claim 2 (final D stack):** not a simple win; often a loss.
- Seed 1: D AUPRC up (+0.040) and R@1000 up, but P@100 and R@P≥0.90 **down** vs baseline D (precision/recall tradeoff).
- Seed 2: degflow D worse across the board (ΔAUPRC **−0.399**, ΔP@100 **−0.90**). Best D overall: **baseline_seed2** (AUPRC 0.511, R@P≥0.90 0.136).
- Do **not** claim degflow as the new best final method.

**Recommendation:** **`stop`**. Do not promote from diagnostic; do not run 40ep scale-up; do not expand to BC/M2/soft-positives on this evidence. Morph expert MSE trains fine on seeds 2–3 (`ssl_labels_used=false`); the AML probe ranking is seed-unstable.

### Temporal-flow regression aux multiseed (Small-HI) — keep diagnostic

InfoNCE + TF-causal **regression** (Huber) at λ∈{0.05, 0.10}; attach `post_embedding_head_pre_projection`; no labels. Seeds 2–3 trained; seed1 reused + recall backfill. Matched baselines seed1–2 only. Full writeup: `notes/temporal_flow_regression_aux_multiseed.md`.

**Claim 1 (pre-3h A/B before downstream TF features):**
| Weight | Paired seeds | ΔA AUPRC (s1 / s2) | ΔB AUPRC | Verdict |
|---|---|---|---|---|
| **w0.05** | 1–2 both up | +0.183 / +0.069 | +0.144 / +0.145 | Claim 1 **pass**; mean ΔA **+0.126** |
| w0.10 | seed1 up, seed2 **down** | +0.134 / **−0.070** | +0.129 / **−0.062** | Claim 1 **fail** (only-seed1 pattern) |

- w0.05 seed2 is strong on A/B (A 0.329, B **0.417**, A P@100 **0.91**) and keeps high-precision recall.
- Seed3 has **no matched baseline**; absolutes are mixed (w0.10 seed3 A **0.417** is the best single cell, but w0.05 seed3 A only 0.221).
- Mean±SD A AUPRC ~0.31±0.08–0.11 across weights — high variance; do not promote on means alone.

**Claim 2 (final D):** does **not** pass for either weight (paired D AUPRC not improved on most seeds; w0.10 D down on both paired seeds).

**Recommendation:** **`keep_diagnostic`** (via w0.05 Claim 1). Do **not** insert into main thesis tables. Do **not** promote w0.10. Optional later: more seeds only if a thesis chapter needs a soft TF-reg aux story — not required for core claims.

### Contrastive resource scout (seed2) — replicate edge_drop only

Tests whether memory-saving choices weaken InfoNCE. Matched baseline: `hi_contrastive_gin_emlps_tds_seed2_...`. Writeup: `notes/contrastive_objective_resource_scout.md`.

| Scout | Change | Pre-3h A / B AUPRC | A P@100 | Notes |
|---|---|---|---|---|
| baseline | bs=8192 accum=4 drop=0.1 | 0.260 / 0.273 | 0.79 | shared seeds ≈6600 typical |
| **large_bs** | bs=16384 accum=2 | **0.139 / 0.112** | **0.51** | **stop** — precision collapse; ckpt ep8; shared seeds 13189; peak ~72 GiB; no OOM fallback |
| **edge_drop_0.05** | drop=0.05 | **0.287 / 0.325** | 0.79 | A/B up (+0.027 / +0.053); D slightly down; shared seeds **7420**; peak ~40 GiB |

**Recommendation:** **`replicate_edge_drop_only`** on seeds 1/3. **Stop large_bs** (true larger InfoNCE batch did not help under the current recipe). Defer `fanout_200` and `edge_drop_0.00` until edge_drop multiseed lands.

## Thesis-Safe Claims

- Extracting the pre-`embedding_head` representation (`3×n_hidden`) from existing frozen checkpoints is at least as good as the exported 128-d embedding, and markedly better for Small-LI rare-positive ranking — with no retraining.
- The pre-`embedding_head` ranking advantage **replicates across three plain Small-LI SSL seeds** (pre-3h wins AUPRC 3/3 for embedding-only and +raw) and across three strong checkpoints (ordinary Small-HI, Small-HI FNF, Small-LI FNF).
- A formal **legacy supervised** Small-LI GINe run (100 ep, Egressy head, paper_argmax, best-val selection) achieves test F1 **0.357** and AUPRC **0.292** at epoch 35 — a usable supervised baseline distinct from the collapsed embedding-head control.
- Frozen GNN embeddings contain useful signal beyond raw/morphological engineered features, especially when stacked with those features.
- Small-HI current-protocol conclusions are robust across probe class-weight choices and alert-budget metrics.
- Small-LI is much harder in absolute precision, but augmented embeddings can still produce large lift above prevalence.
- Positive probe weighting is an operating-point tool; it can improve F1 but does not automatically improve ranking.

## Claims Needing Caveats

- Do not claim FNF universally improves transfer. Its Small-LI behavior is metric-dependent.
- Do not claim supervised Small-LI is categorically worse as a training paradigm. The **embedding-head** supervised control collapsed (F1≈0); the **legacy Egressy-head** 100-ep run reaches F1 **0.357** — but only at best-val epoch, not the final epoch.
- Do not claim a universal best feature stack across datasets, seeds, and metrics. `embedding+raw` is strongest in some Small-HI rows, while `embedding+raw+morph` is strongest for several Small-LI SSL probe rankings; legacy supervised is a separate stronger paradigm on Small-LI.
- Do not claim a wider exported embedding helps. The emb198 export-width scout is single-seed and confounded (the retrain hit a worse SSL optimum); the width question needs multi-seed emb198 replication before any claim.
- Do not overstate the pre-3h gain on full engineered stacks or weak SSL seeds. With `embedding+raw+morph` the AUPRC edge is small; seed 2 absolute gains are modest even though direction holds. Pre-3h (198-d) also has more dimensions than post (128-d), a partial confounder for a linear probe.
- Do not use the legacy supervised **last** checkpoint for any claim; final-epoch predictions collapse to all-negative argmax.
- Do **not** promote morphology-objective `degflow`: multiseed replication **failed** (seed-1 win; seed-2 precision collapse; seed-3 weak absolutes). Keep diagnostic only; no 40ep scale-up; do not expand BC/M2 from this line.
- Do **not** promote temporal-flow regression aux yet: w0.05 improves paired pre-3h A/B but Claim 2 (D) fails and seed3 absolutes are unstable; w0.10 fails Claim 1 on seed2. Keep `diagnostic_only` / `table_eligible=false`.
- Do **not** claim larger true InfoNCE batch helps the current recipe: seed2 `bs=16384` **hurt** A/B and collapsed P@100. Edge-drop 0.05 is **closed as diagnostic/promising but not promoted**: pre-3h+raw improves on seeds 1–2; embedding-only and final D are mixed; not table-eligible; no seed3 / 0.00 / fanout.

## Optional Follow-Ups

- Prefer new augmentation *design* audits (anchor-preserving drop, configurable attr-mask, clean/noisy views) over further edge_drop rate sweeps.
- If thesis space allows, rerun only selected Small-LI/FNF seeds to test stability.
- Consider reporting alert-budget tables alongside AUPRC/F1 for all final comparisons.
- Keep future large caches on Scratch and reusable checkpoints/data on Pool via the current symlinks.

## Artifacts Used

- `results/diagnostics/probe_sweep_small_li_current_protocol.json`
- `results/diagnostics/supervised_small_li_gin_emlps_tds_seed1.json`
- `results/diagnostics/probe_feature_ablation_small_li_fnf_current_protocol_seed1.json`
- `results/diagnostics/probe_weight_sweep_small_li_current_protocol.json`
- `results/diagnostics/probe_weight_sweep_small_hi_key_runs.json`
- `results/diagnostics/alert_budget_metrics_small_hi.json`
- `results/diagnostics/alert_budget_metrics_small_li.json`
- `results/diagnostics/alert_budget_metrics_current_protocol.json`
- `results/diagnostics/pre_embedding_3h_vs_post_embedding_current_protocol.json`
- `results/diagnostics/small_li_embedding_dim_128_vs_198.json`
- `results/diagnostics/pre3h_strong_run_comparison.json`
- `results/diagnostics/pre3h_vs_post128_small_hi_40ep_seed2.json`
- `results/diagnostics/pre3h_vs_post128_small_hi_fnf_seed1.json`
- `results/diagnostics/pre3h_vs_post128_small_li_fnf_seed1.json`
- `results/diagnostics/pre_embedding_3h_vs_post_embedding_small_li_multiseed.json`
- `results/diagnostics/pre_embedding_3h_vs_post_embedding_small_li_seed{2,3}.json`
- `results/diagnostics/supervised_Small-LI_small_li_legacy_supervised_gin_emlps_tds_100ep_seed1_summary.json`
- `results/diagnostics/supervised_Small-LI_small_li_legacy_supervised_gin_emlps_tds_100ep_seed1_epoch_history.json`
- `results/diagnostics/eval_small_li_legacy_supervised_gin_emlps_tds_100ep_seed1.json`
- `results/diagnostics/morphology_objective_recall_scout.json`
- `results/diagnostics/morph_obj_baseline_{pre3h,post128}_seed1.json`
- `results/diagnostics/morph_obj_degflow_{pre3h,post128}_seed1.json`
- `results/diagnostics/morph_obj_clustering_{pre3h,post128}_seed1.json`
- `results/diagnostics/morph_obj_degflow_tfreg_{pre3h,post128}_seed1.json`
- `results/diagnostics/degflow_morphology_multiseed_scout.json`
- `results/diagnostics/morph_obj_baseline_{pre3h,post128}_seed2.json`
- `results/diagnostics/temporal_flow_regression_aux_multiseed.json`
- `results/diagnostics/tf_aux_tf_reg_w0.{05,10}_{pre3h,post128}_seed{2,3}.json`
- `results/diagnostics/enriched/tf_aux_tf_reg_w0.{05,10}_{pre3h,post128}_recall_metrics.json`
- `results/diagnostics/contrastive_objective_resource_scout.json`
- `results/diagnostics/ctr_res_{large_bs,edge_drop}_{pre3h,post128}_seed2.json`
- `results/diagnostics/contrastive_resource_{large_bs,edge_drop}_resolved_run_seed2.json`
- `results/diagnostics/morph_obj_degflow_{pre3h,post128}_seed{2,3}.json`
