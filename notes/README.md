# Documentation index (`notes/`)

Navigation hub for all project docs. Start here to find the right file instead of scanning the folder.

**Status legend**

| Tag | Meaning |
|-----|---------|
| **canonical** | Authoritative reference; keep current. |
| **living** | Design/planning doc, updated as the stack evolves. |
| **current results** | Latest-protocol numbers + interpretation (Jun–Jul 2026). |
| **run log** | Single-experiment / dated record; provenance only. |
| **superseded** | Kept for history; do **not** use for current decisions. |

> **Results caveat:** all benchmark numbers in this folder are **development sanity checks**, not a frozen evaluation protocol. **Fair-comparison probe policy:** `--class_weight model --model gin`, C=1.0, val-tuned F1 threshold. Rows using other `cw`/C are exploratory and flagged as such.

> **Local-only tooling:** command examples across these docs that reference `sbatch slurm/*.sh` or `tests/` are maintainer convenience wrappers — `slurm/` and `tests/` are gitignored and **not in a public clone**. The canonical, runnable path is always the `python main.py …` / `python scripts/…` command each wrapper invokes.

---

## Start here

| I want to… | File | Status |
|------------|------|--------|
| See latest results & thesis-safe claims | [`current_protocol_recent_runs_summary.md`](current_protocol_recent_runs_summary.md) | current results |
| Compare our strongest runs (post-128 vs pre-3h, HI + LI) | [`strongest_runs_master_comparison.md`](strongest_runs_master_comparison.md) | current results |
| Read the full dev-results doc | [`results.md`](results.md) | canonical (+ history) |
| Get running / set up data | [`datasets.md`](datasets.md) | canonical |
| Look up a CLI flag | [`cli-reference.md`](cli-reference.md) | canonical |

## Reference (stable runbooks)

| File | Purpose | Status |
|------|---------|--------|
| [`datasets.md`](datasets.md) | IBM / PaySim / SAML-D setup, schema, splits | canonical |
| [`cli-reference.md`](cli-reference.md) | Full CLI flag tables (train / probe / label-efficiency) | canonical |
| [`morphology-reference.md`](morphology-reference.md) | M0–M5 flags, metric cheat sheet, tier columns | canonical |
| [`knn-precompute-reference.md`](knn-precompute-reference.md) | Feature-KNN cache: precompute, exclusion, soft positives | canonical |
| [`../morphology/IDS.md`](../morphology/IDS.md) | `EdgeID` / node-ID join semantics | canonical |
| [`engaging-cluster-config.md`](engaging-cluster-config.md) | Slurm accounts/QoS/partitions + verified walltime limits (Advanced account through 2026-08-01) | canonical |

## Design & research plans (living)

| File | Purpose | Status |
|------|---------|--------|
| [`contrastive-learning-plan.md`](contrastive-learning-plan.md) | GFM framing, contrastive design, phase history | living |
| [`morphology-metrics-plan.md`](morphology-metrics-plan.md) | Morphology metric selection + phased implementation | living |
| [`downstream-eval-plan.md`](downstream-eval-plan.md) | PaySim / SAML-D / typology eval strategy | living |
| [`lit-review-index.md`](lit-review-index.md) | Paper → implementation map (internal; not in README) | living |

> Plan docs carry design rationale and history. For **current** benchmark numbers, prefer the results docs below over any status tables inside the plans.

## Results — current protocol (Jun–Jul 2026)

| File | Scope | Status |
|------|-------|--------|
| [`current_protocol_recent_runs_summary.md`](current_protocol_recent_runs_summary.md) | Meta-index: experiment status + thesis-safe claims for the latest batch | current results |
| [`strongest_runs_master_comparison.md`](strongest_runs_master_comparison.md) | Master leaderboard of strongest configs; post-128 vs pre-3h side by side (HI + LI) | current results |
| [`results.md`](results.md) | Recommended configs + key-runs leaderboard (current protocol) | canonical |
| [`results-archive.md`](results-archive.md) | Historical ablations: queue/negatives, feature-KNN, edge-drop, PNA, masked-edge, morphology scouts, label-efficiency, PaySim | run log |
| [`probe_feature_ablation_current_protocol_comparison.md`](probe_feature_ablation_current_protocol_comparison.md) | Consolidated 6-mode feature ablation | current results |
| [`probe_feature_ablation_current_protocol_stack_comparison.md`](probe_feature_ablation_current_protocol_stack_comparison.md) | Focused stack comparison | current results |
| [`probe_sweep_40ep_current_protocol.md`](probe_sweep_40ep_current_protocol.md) | 40 ep probe hyperparameter sweep (seeds 1–4) | current results |
| [`probe_weight_sweep_small_hi_key_runs.md`](probe_weight_sweep_small_hi_key_runs.md) | Small-HI explicit positive-weight sweep | current results |
| [`probe_weight_sweep_small_li_current_protocol.md`](probe_weight_sweep_small_li_current_protocol.md) | Small-LI explicit positive-weight sweep | current results |
| [`probe_sweep_small_li_current_protocol.md`](probe_sweep_small_li_current_protocol.md) | Small-LI generic probe sweep (baseline; superseded for weighting by the explicit sweep) | current results |
| [`alert_budget_metrics_current_protocol.md`](alert_budget_metrics_current_protocol.md) | Alert-budget (P@k / lift@k) for HI + LI | current results |
| [`small_li_current_protocol_comparison.md`](small_li_current_protocol_comparison.md) | Small-LI dataset scout | current results |
| [`small_li_fnf_current_protocol_comparison.md`](small_li_fnf_current_protocol_comparison.md) | Small-LI false-negative-filter comparison | current results |
| [`small_li_supervised_baseline_comparison.md`](small_li_supervised_baseline_comparison.md) | Small-LI supervised baseline (embedding-head control) | current results |
| [`small_li_legacy_supervised_scout.md`](small_li_legacy_supervised_scout.md) | Small-LI legacy (Egressy-head) supervised scout | run log |
| [`pre_embedding_3h_vs_post_embedding_current_protocol.md`](pre_embedding_3h_vs_post_embedding_current_protocol.md) | Representation location: pre-`embedding_head` (`3×n_hidden`) vs exported 128-d (Small-HI + Small-LI) | current results |
| [`small_li_embedding_dim_128_vs_198.md`](small_li_embedding_dim_128_vs_198.md) | Small-LI exported-dim scout: 128 vs 198 (4-way paired probe) | run log |
| [`pre3h_strong_run_comparison.md`](pre3h_strong_run_comparison.md) | Pre-3h vs post-128 replicated across 3 strong checkpoints (HI 40ep s2, HI FNF s1, LI FNF s1) + interpreted conclusion | current results |
| [`pre_embedding_3h_vs_post_embedding_small_li_multiseed.md`](pre_embedding_3h_vs_post_embedding_small_li_multiseed.md) | Plain Small-LI pre-3h vs post-128 multiseed replication (seeds 1–3) | current results |
| [`supervised_Small-LI_small_li_legacy_supervised_gin_emlps_tds_100ep_seed1_summary.md`](supervised_Small-LI_small_li_legacy_supervised_gin_emlps_tds_100ep_seed1_summary.md) | Formal 100-ep legacy supervised Small-LI reproduction (best-val ep 35) | current results |
| [`eval_small_li_legacy_supervised_gin_emlps_tds_100ep_seed1.md`](eval_small_li_legacy_supervised_gin_emlps_tds_100ep_seed1.md) | Post-hoc eval of 100-ep legacy supervised checkpoint | run log |

## Per-run ablation logs (provenance)

In [`experiments/ablation-runs/`](experiments/ablation-runs/). Single-run tables auto-written by `scripts/probe_feature_ablation.py` (path passed per run). Rows are already folded into the consolidated tables above; use these only for provenance of a specific run.

- `probe_feature_ablation_current_protocol_gin_40ep_seed{1,2,3,4}.md`
- `probe_feature_ablation_current_protocol_fnf_emlps_tds_seed{1,2}.md`
- `probe_feature_ablation_current_protocol_gin_emlps_tds_seed1.md`
- `probe_feature_ablation_degree_aware_edgedrop_emlps_tds.md`
- `probe_feature_ablation_same_pair_fnf_emlps_tds.md`
- `probe_feature_ablation_hi_contrastive_gin_emlps_tds_embedding_raw.md`
- `probe_feature_ablation_hi_contrastive_proj_asym_8192neg_queue0_accum4_20ep_bestckpt.md`
- `probe_feature_ablation_masked_edge_embedding_raw.md`
- `probe_feature_ablation_small_hi.md`

Per-run pre-3h vs post-128 tables (folded into [`pre3h_strong_run_comparison.md`](pre3h_strong_run_comparison.md)) live directly in `notes/`:

- `pre3h_vs_post128_small_hi_40ep_seed2.md`, `pre3h_vs_post128_small_hi_fnf_seed1.md`, `pre3h_vs_post128_small_li_fnf_seed1.md`
- `pre_embedding_3h_vs_post_embedding_small_li_seed{2,3}.md` (folded into multiseed note above)

## KNN engineering audits

In [`experiments/knn-audits/`](experiments/knn-audits/). Outputs of `scripts/audit_transaction_knn_*.py` that informed the "feature-KNN does not help" conclusion (see [`results-archive.md` § Feature-KNN](results-archive.md#feature-knn-small-hi)).

- `knn_feature_audit_smoke.md`, `knn_feature_audit_50k.md`
- `knn_feature_audit_richer_v1_5k_smoke.md`, `knn_feature_audit_richer_v1_50k.md`
- `knn_metric_scaling_audit_50k.md`, `knn_mutual_hub_audit_50k.md`

## Superseded (history only)

In [`archive/`](archive/).

| File | Superseded by |
|------|---------------|
| [`archive/probe_feature_ablation_final_protocol_comparison.md`](archive/probe_feature_ablation_final_protocol_comparison.md) | [`probe_feature_ablation_current_protocol_comparison.md`](probe_feature_ablation_current_protocol_comparison.md) |
| [`archive/projection-head-ablation-jun2026.md`](archive/projection-head-ablation-jun2026.md) | [`results.md`](results.md) + [`morphology-metrics-plan.md`](morphology-metrics-plan.md) |

---

**Machine artifacts:** raw JSON and auto-generated tables live in [`../results/diagnostics/`](../results/diagnostics/). When a `notes/` file and a diagnostics file share a name, the `notes/` copy is the interpreted, canonical one.
