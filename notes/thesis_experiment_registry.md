# Thesis experiment registry

Traceable source of truth for thesis-relevant evaluated configurations.
Every metric is copied from a cited JSON file — **no inferred values**.

## Registry files

| File | Description |
|------|-------------|
| `results/diagnostics/thesis_experiment_registry.csv` | One row per evaluated config |
| `results/diagnostics/thesis_experiment_registry.json` | Rows + multiseed aggregates |

**Total rows:** 367 | **thesis_primary:** 27 | **thesis_supporting:** 112 | **diagnostic:** 119 | **negative_result:** 25 | **superseded:** 5

## Field conventions

- `representation_source`: `post_embedding_128` vs `pre_embedding_3h` (198-d)
- `threshold_rule`: `max_f1_on_val` (SSL probe) vs `paper_argmax` (legacy supervised)
- `paired_test_n`: test rows after inner-join when present in source JSON
- `thesis_role`: see classification rules below
- Legacy supervised **canonical AUPRC = 0.292** from `eval_..._100ep_seed1.json` (not summary JSON 0.260)
- Small-LI seed1 pre-3h +raw: **0.0818** in multiseed aggregate; **0.0829** only in emb198 paired join

### `thesis_role` classification rules (conservative)

| Value | Rule |
|-------|------|
| `thesis_primary` | Small-LI multiseed pre/post; **paper-faithful** Small-HI ports TDS-off supervised (paper_argmax); Small-HI strong-run paired pre/post for gin 40ep s2 and FNF HI |
| `thesis_supporting` | Architecture sweep; alert-budget; feature ablation; single-file HI pre/post; 20ep baseline; FNF/LI secondary |
| `diagnostic` | emb198; val-tuned supervised; batch-size E/F; GCPAL audits; A/B/C/D contrastive arms; txn-node scouts; random-40 |
| `negative_result` | degree-aware edge-drop; superseded non-legacy supervised |
| `historical` | Rows from superseded protocols not marked superseded=true |
| `superseded` | `superseded=true` (includes old Small-HI TDS-on supervised for paper table) |

### `paper_comparable` (Multi-GIN+EU)

`true` only for Small-HI legacy supervised with ports, **tds=False**, paper_argmax, formal seeds/aggregate.
Old TDS-on supervised → `false`. Contrastive / txn-node / random-40 → `false`.

---

## Current headline results

Frozen linear probe (`cw=model`, C=1.0) unless noted. Pre-vs-post tables use **paired** strong-run JSON (`pre3h_strong_run_comparison.json`).

### Small-HI embedding-only SSL

#### Best current (40ep seed2, strong-run paired)
- **run_id:** `gin_emlps_tds_asym_proj_8192neg_queue0_40ep_seed2|embedding|pre_embedding_3h`
- **seed:** 2 | **training_epochs:** 40 | **selected_epoch:** 36
- **rep:** pre_embedding_3h | **stack:** embedding
- **AUROC / AUPRC / F1:** 0.9581319256106977 / 0.2953354887979259 / 0.33975659229208927
- **P@100 / R@100 / lift@100:** 0.83 / 0.051520794537554315 / 444.5801489757914
- **paired_test_n:** 862914
- **threshold_rule:** max_f1_on_val | **thesis_role:** thesis_primary
- **source:** `results/diagnostics/pre3h_strong_run_comparison.json`
- **caveats:** strong-run paired pre/post batch; canonical for pre-vs-post tables; paired edge-ID inner-join per split

Post-128 same run: AUPRC **0.24488448136909816** F1 **0.30402722631877477** (paired n=862914)

#### Historical 20ep seed1 baseline (post-128)
- **run_id:** `hi_contrastive_gin_emlps_tds_proj_asym_8192neg_queue0_accum4_20ep|embedding|post_embedding_128`
- **seed:** 1 | **training_epochs:** 20 | **selected_epoch:** None
- **rep:** post_embedding_128 | **stack:** embedding
- **AUROC / AUPRC / F1:** 0.944004657677909 / 0.21330862216949034 / 0.2592794759825327
- **threshold_rule:** max_f1_on_val | **thesis_role:** thesis_supporting
- **source:** `results/diagnostics/probe_feature_ablation_current_protocol_comparison.json`
- **caveats:** 6-mode feature ablation; post_embedding_128; full test split (not paired pre/post intersection); historical_20ep_baseline; cross-source replication (sources: results/diagnostics/architecture_sweep_shared_probe_weights.json, results/diagnostics/probe_feature_ablation_current_protocol_comparison.json)

### Small-HI +raw F1: pre-3h vs post-128 (canonical paired comparison)

Both from `results/diagnostics/pre3h_strong_run_comparison.json`, gin 40ep seed2, val-tuned F1, paired n≈862914.

**Canonical paired (use in main pre/post table):**
- pre-3h +raw: AUPRC **0.3211782703234284**, F1 **0.34428571428571425**
- post-128 +raw: AUPRC **0.28375949571105374**, F1 **0.3429049344856426**

**Interpretation:** pre-3h has stronger **AUPRC** (+0.037) and slightly higher **tuned F1** (+0.001) under the *same paired rows*. The higher post-128 F1 **0.347** appears only in the non-paired feature-ablation file (full test n≈863050).

**Non-paired post-128 reference** (`probe_feature_ablation_current_protocol_comparison.json`, full test n≈863050): F1 **0.34719508791515485**, AUPRC **0.2921716129953516** — not comparable row-for-row to paired pre-3h.

### Small-HI FNF full-stack alert-budget (recovered from strong-run JSON)

- **post-128:** P@100=0.8, lift@100=428.6 | source `results/diagnostics/pre3h_strong_run_comparison.json`
- **pre-3h:** P@100=0.73, lift@100=391.1

### Small-LI multiseed aggregates (mean ± sample SD, n=3, ddof=1)

Convention: **mean ± sample standard deviation** over seeds 1–3.

#### `embedding`
- **post_embedding_128** AUROC 0.8883±0.0158 | AUPRC 0.0142±0.0096 | R@100 0.0150±0.0075
- **pre_embedding_3h** AUROC 0.9192±0.0087 | AUPRC 0.0387±0.0161 | R@100 0.0274±0.0102
- Δpre-post: delta_auprc_pre_minus_post mean=0.0245±0.0090, delta_recall_at_100_pre_minus_post mean=0.0125±0.0033, delta_auroc_pre_minus_post mean=0.0309±0.0078, delta_lift_at_100_pre_minus_post mean=146.2864±38.7055, delta_precision_at_100_pre_minus_post mean=0.1000±0.0265

#### `embedding+raw`
- **post_embedding_128** AUROC 0.9042±0.0137 | AUPRC 0.0321±0.0211 | R@100 0.0283±0.0152
- **pre_embedding_3h** AUROC 0.9260±0.0128 | AUPRC 0.0612±0.0336 | R@100 0.0428±0.0198
- Δpre-post: delta_auprc_pre_minus_post mean=0.0291±0.0262, delta_recall_at_100_pre_minus_post mean=0.0145±0.0134, delta_auroc_pre_minus_post mean=0.0218±0.0018, delta_lift_at_100_pre_minus_post mean=170.6684±157.7841, delta_precision_at_100_pre_minus_post mean=0.1167±0.1079

Seed1 +raw pre-3h AUPRC: multiseed **0.0818** | emb198 paired **0.08288958670149502**

### Legacy supervised Small-LI (formal)

- **run_id:** `small_li_legacy_supervised_gin_emlps_tds_100ep_seed1|supervised|paper_argmax`
- **seed:** 1 | **training_epochs:** 100 | **selected_epoch:** 35
- **rep:** logits_direct | **stack:** in_gnn_end_to_end
- **AUROC / AUPRC / F1:** 0.9587135629980161 / 0.2920792524625353 / 0.35701906412478335
- **P@100 / R@100 / lift@100:** 0.97 / 0.12094763092269327 / 1418.9878428927682
- **threshold_rule:** paper_argmax | **thesis_role:** thesis_primary
- **source:** `results/diagnostics/eval_small_li_legacy_supervised_gin_emlps_tds_100ep_seed1.json`
- **caveats:** paper_argmax in-GNN; NOT frozen probe; best-val checkpoint ep35 only

## Duplicate resolution

Rows sharing `run_id|rep|stack|threshold` across sources:

- **canonical_multiseed_source** (4 rows): Small-LI multiseed JSON is canonical; emb198 rows are **not** duplicates (different join).
- **canonical_prepost_source** (3 rows): Strong-run JSON is canonical for HI pre/post and FNF alert-budget; ablation JSON is cross-source replication.
- **pairing_intersection_diff** (4 rows): Same config, different paired row intersection (multiseed vs emb198).
- **cross_source_replication** (22 rows): Same metrics re-ingested from a second diagnostic file; keep both for provenance.

Distinct evaluations (same checkpoint, different stack/rep/threshold) are **not** duplicates.

## Optional / not currently planned

- **emb198 multiseed replication** — one-seed scout did not beat orig pre-3h; pre-3h replicates 3/3 on Small-LI without emb198.

## Remaining ambiguities

1. HI 40ep seed2: `training_epochs=40`, `selected_epoch=36` (best ckpt) — both fields preserved.
2. Feature ablation F1 on full test vs paired strong-run — use strong-run for pre/post tables only.
3. PNA capacity-matched comparison still pending (separate workstream).


## PaySim preserve/normalization ablation (diagnostic)

- See `/orcd/home/002/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN/notes/paysim_preserve_normalization_ablation.md` / `/orcd/home/002/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN/results/diagnostics/paysim_preserve_normalization_ablation.json`
- A train-fit AUROC=0.8668; B per-graph AUROC=0.8331
