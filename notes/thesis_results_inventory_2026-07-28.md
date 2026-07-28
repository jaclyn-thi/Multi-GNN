# Thesis results inventory (read-only reconstruction)

> Generated 2026-07-28. Source of truth = canonical diagnostic JSON + per-seed cells + checkpoint provenance.
> Existing registry / table preview / current-protocol summaries may be stale and were **not** modified.
> Twin: `results/diagnostics/thesis_results_inventory_2026-07-28.json`
>
> Scope counts (repo scan): **492** `results/diagnostics/**/*.json`, **251** `notes/**/*.md`, registry rows **371**.
> This inventory does **not** retrain, extract, evaluate, submit jobs, or rewrite prior artifacts.

---

## A. Canonical artifact map

Status legend: **primary** · **secondary** · **appendix** · **diagnostic/negative** · **superseded** · **missing**

| Role | Experiment family | Canonical JSON | Note | Status | Superseded / do-not-cite |
|------|-------------------|----------------|------|--------|---------------------------|
| A | AMLWorld supervised Multi-GIN+EU | `results/diagnostics/eval_small_hi_legacy_supervised_gin_emlps_ports_50ep_seeds1-3_formal_aggregate.json` | `notes/eval_small_hi_legacy_supervised_gin_emlps_ports_50ep_seeds1-3_formal_aggregate.md` | **primary** (supervised comparator) | TDS-on 100ep formal seed1; train-time-only aggregate |
| B | Published Multi-GNN comparator | Embedded in formal aggregate `paper_multigin_eu` + README / `notes/evaluation_protocols.md` | same | **primary** (external comparator, protocol caveat) | Upstream released-ckpt eval ≠ paper mean |
| C | Final frozen SSL AMLWorld | `results/diagnostics/final_corrected_no_preserve_multiseed.json` | `notes/final_corrected_no_preserve_multiseed.md` | **primary** (frozen SSL) | D+ preserve multiseed as *primary*; older LI/HI SSL “thesis_primary” registry rows |
| D | Representation ablation (pre-3h H / H+X+TF / post-128 H) | same JSON → `confirmation_aggregate.amlworld.*` | same note | **primary** (ablation columns) | Single-seed probe-feature ablations |
| E | Frozen equal-weight ensemble | Embedded `ensembles` in corrected JSON; also D+ `results/diagnostics/frozen_dplus_sprint_scores/equal_weight_ensemble.json` | corrected note; D+ sprint note | **secondary** | Must not replace seed-mean robustness |
| F | Supervised partial fine-tuning | `results/diagnostics/dplus_partial_finetune_seed2_locked_test.json` (+ D+ analysis) | `notes` twin if present | **appendix / secondary** | Not primary SSL claim |
| G | PaySim supervised Multi-GIN+EU | `results/diagnostics/paysim_supervised_multigin_eu.json` | `notes/paysim_supervised_multigin_eu.md` | **secondary** (target ceiling; `table_eligible=false` in embedded rows) | Not a published PaySim reproduction |
| H | PaySim strict frozen transfer (P1) | corrected JSON → `confirmation_aggregate.paysim.P1_strict_inductive_legacy` | corrected note | **primary** (PaySim transfer) | Capacity-check / TF / sequential (val-only scouts) |
| I | PaySim label-free BN (P2) | corrected JSON → `…P2_label_free_target_bn_legacy` | corrected note | **secondary** (adaptation, not zero-shot) | — |
| J | PaySim X-only / matched-random | `…/final_corrected_no_preserve_multiseed/cells/control_X_only_paysim_legacy_duplicate_v1.json`, `…/control_random_paysim_legacy_duplicate_v1.json` | corrected note / cells summary | **secondary** (controls) | type_only twins (sensitivity) |
| K | Sequential AML→PaySim SSL | `results/diagnostics/sequential_aml_to_paysim_ssl_scout.json` + `…/sequential_aml_to_paysim_ssl/eval_summary.json` | `notes/sequential_aml_to_paysim_ssl_scout.md` | **diagnostic/negative** (gate fail; exploratory) | Do not promote as final method |
| L | Joint replay / domain-BN | `results/diagnostics/joint_replay_scout/{shared_bn,domain_bn}.json` | `notes/joint_replay_scout_{shared,domain}_bn.md` | **diagnostic / exploratory** (1-seed, val-only) | — |
| M | GCPAL reconstruction/challenge | `results/diagnostics/gcpal_challenge_fullstack_eval.json` | `notes/gcpal_challenge_fullstack_eval.md` | **appendix / diagnostic** | Online txn-node scouts; legacy chunked extraction |
| N | Label-efficiency | `results/diagnostics/label_scarcity_temporal_flow_probe.json` | `notes/label_scarcity_temporal_flow_probe.md` | **appendix** (valid under caveats) | `embeddings/label_efficiency_summary.json` (historical) |
| O | AML typology/pattern | Per-run `results/diagnostics/hi_contrastive_*/pattern_typology_test.json` | CSVs in same dirs | **diagnostic**; **missing** on locked primary encoder | — |
| P | Architecture comparisons | `results/diagnostics/architecture_sweep_shared_probe_weights.json` | twin `.md` | **appendix** | Width-unmatched PNA default |
| Q | Negative ablations | See §Q below | various | **diagnostic/negative** | — |

### Q — major negative / closed scouts (canonical aggregates)

| Family | Canonical JSON | Outcome |
|--------|----------------|---------|
| Schema masking | `results/diagnostics/schema_mask_scout.json` | Gate fail; val-only |
| Morphology / degflow | `results/diagnostics/degflow_morphology_multiseed_scout.json` | Stop / not promoted |
| Degree-aware edge-drop | `results/diagnostics/probe_feature_ablation_degree_aware_edgedrop_emlps_tds.json` | Negative |
| Contrastive resource scout | `results/diagnostics/contrastive_objective_resource_scout.json` | large_bs negative; edge_drop not promoted |
| TF-on-PaySim | `results/diagnostics/paysim_temporal_flow_downstream_validation.json` | Gate fail (TF ≰ H+X) |
| Sequential SSL promote | sequential scout JSON | PaySim improves; AML retention fails |
| KNN filter / soft positives | **No diagnostic aggregate JSON** | Archive prose only (`notes/results-archive.md`) — **missing aggregate** |

### Older D+ preserve family (keep for history; not current primary)

| Artifact | Role under this inventory |
|----------|---------------------------|
| `results/diagnostics/final_dplus_multiseed_and_finetune_analysis.json` | **superseded as primary** by corrected/no-preserve (preserve ON vs OFF; seeds 1–3 vs confirmation 1/3/4) |
| Registry rows `primary_result` / `table_eligible=true` for D+ | **stale hierarchy** vs corrected note claim |

---

## B. Proposed Results chapter tables

### Table 1 — AMLWorld main comparison (main text)

**Title:** Supervised Multi-GIN+EU vs frozen self-supervised encoder (Small-HI)

| Row | Metric columns | Source JSON | Exact key path | Notes |
|-----|----------------|-------------|----------------|-------|
| Published Multi-GIN+EU | F1 mean±σ | formal aggregate | `paper_multigin_eu` → F1 **0.6479±0.0122** | External; protocol caveat |
| Supervised Multi-GIN+EU (reproduced) | paper_argmax F1, AUROC, AUPRC | `…/eval_small_hi_legacy_supervised_gin_emlps_ports_50ep_seeds1-3_formal_aggregate.json` | `formal_test_aggregate.paper_argmax_f1` / `.auroc` / `.auprc` | Seeds **1–3**; selection = best val minority F1; decision = paper_argmax; **test** |
| Frozen SSL (confirmation) | F1@0.5, AUROC, AUPRC | `…/final_corrected_no_preserve_multiseed.json` | `confirmation_aggregate.amlworld.pre3h_HxXTF.aggregate_test_threshold_0.5.{f1,auroc,auprc}` | Seeds **1,3,4**; encoder frozen; downstream uses labels; **test**; F1@**0.5** (not paper_argmax) |

**Values (traceable):**
- Supervised: F1 **0.6598±0.0597**, AUROC **0.9850±0.0021**, AUPRC **0.6629±0.0200**
- Frozen confirmation H+X+TF: F1 **0.6174±0.0229**, AUROC **0.9877±0.0002**, AUPRC **0.6697±0.0124**

**Do not** put seed-2 alone, descriptive n=4 mean, or ensemble in this table.

---

### Table 2 — AMLWorld representation stacks (main text or compact with Table 1)

**Title:** Frozen encoder stacks (confirmation seeds 1/3/4, test @ threshold 0.5)

| Row | AUPRC / AUROC / F1 keys under `confirmation_aggregate.amlworld.<stack>.aggregate_test_threshold_0.5` |
|-----|-----------------------------------------------------------------------------------------------------|
| pre-3h H+X+TF (primary) | AUPRC **0.6697±0.0124**, AUROC **0.9877±0.0002**, F1 **0.6174±0.0229** |
| pre-3h H-only | AUPRC **0.4826±0.0094**, AUROC **0.9731±0.0030**, F1 **0.4926±0.0441** |
| post-128 H-only (diagnostic) | AUPRC **0.2477±0.0260**, AUROC **0.9289±0.0069**, F1 **0.1504±0.0794** |

Source: `results/diagnostics/final_corrected_no_preserve_multiseed.json`.

---

### Table 3 — PaySim transfer (main text)

**Title:** PaySim under frozen AML encoder (legacy contract)

| Row | Role | Source | Key | Test AUPRC (confirmation 1/3/4) |
|-----|------|--------|-----|--------------------------------|
| P1 strict frozen H | primary transfer | corrected JSON | `confirmation_aggregate.paysim.P1_strict_inductive_legacy.aggregate_test_threshold_0.5.auprc` | **0.0533±0.0329** |
| P2 label-free BN H | secondary adaptation | same | `…P2_label_free_target_bn_legacy…auprc` | **0.1088±0.0384** |
| X-only control | control | `…/cells/control_X_only_paysim_legacy_duplicate_v1.json` | `test.threshold_0.5.auprc` | **0.0865** (single fit) |
| Matched-random H | control | `…/cells/control_random_paysim_legacy_duplicate_v1.json` | `test.threshold_0.5.auprc` | **0.0261** (single fit) |
| Supervised Multi-GIN+EU ceiling | secondary / separate panel | `paysim_supervised_multigin_eu.json` | `aggregate.test_paper_argmax_f1` / `test_auprc` | F1 **0.202±0.007**, AUPRC **0.255±0.027** (seeds 1–3; **paper_argmax**, not H-logistic) |

**Caveats:** P2 ≠ strict zero-shot. Supervised row is protocol-incomparable to P1/P2 logistic H. High seed variance on P1/P2.

---

### Table 4 — Appendix: ensembles & fine-tuning

| Row | Source | Key | Placement |
|-----|--------|-----|-----------|
| Corrected equal-weight ensembles | corrected JSON `ensembles.*` | `test.threshold_0.5` | appendix; secondary_only |
| D+ partial FT seed2 | `dplus_partial_finetune_seed2_locked_test.json` | `test_metrics_threshold_0.5` | appendix; not primary SSL |
| D+ preserve multiseed (historical) | `final_dplus_multiseed_and_finetune_analysis.json` | `aggregate.test_auprc` **0.6636±0.0093** | appendix history / recipe comparison only |

---

### Table 5 — Appendix: negatives & exploratory (one-line each)

Cite gate outcome only; no “method success” framing.

| Item | JSON | Status |
|------|------|--------|
| Sequential SSL | `sequential_aml_to_paysim_ssl_scout.json` | exploratory; promote **FAIL** (AML retention) |
| Joint domain-BN | `joint_replay_scout/domain_bn.json` | exploratory; val-only; 1-seed |
| Joint shared-BN | `joint_replay_scout/shared_bn.json` | exploratory; PaySim collapses |
| TF-on-PaySim | `paysim_temporal_flow_downstream_validation.json` | exploratory; gate **FAIL** |
| Schema mask / degflow / degree-drop / contrastive resource | listed in §Q | negative / closed |
| GCPAL challenge | `gcpal_challenge_fullstack_eval.json` | appendix diagnostic; not exact reproduction |
| Label scarcity | `label_scarcity_temporal_flow_probe.json` | appendix |
| Architecture sweep | `architecture_sweep_shared_probe_weights.json` | appendix |
| Frozen capacity check seed2 | `paysim_frozen_capacity_check_seed2.json` | exploratory probe capacity |

---

## C. Minimum recommended table set

**Main text (prefer 2–3 tables):**
1. **Table 1** — Supervised vs frozen SSL (AMLWorld), with published comparator.
2. **Table 2** — Stack ablation (can be a panel of Table 1).
3. **Table 3** — PaySim P1 (+ P2 as second column or footnote) + controls; supervised PaySim as separate ceiling callout, not same metric column as P1 H.

**Appendix only:** ensembles, fine-tuning, D+ preserve historical, GCPAL, label-scarcity, architecture sweep, all negative/exploratory scouts (sequential, joint, TF-PaySim, morphology, schema mask, capacity check).

**Combine:** Table 1+2; do **not** combine PaySim supervised paper_argmax F1 with P1 logistic AUPRC in one undifferentiated column.

---

## D. Missing or unresolved (no computation run)

1. **Typology/pattern results on the locked corrected/no-preserve encoder** — existing typology dirs are legacy `hi_contrastive_*` / morphology runs.
2. **KNN negative aggregate JSON** — archive narrative only; no canonical diagnostic aggregate.
3. **Registry alignment** — corrected/no-preserve not marked `table_eligible` / `primary_result` in registry (stale vs note).
4. **PaySim published Multi-GIN numeric target** — none in-repo.
5. **Whether thesis wants confirmation (n=3) or descriptive (n=4) as robustness mean** — inventory recommends **confirmation 1/3/4**; descriptive includes development seed 2.
6. **Val-selected-threshold F1** exists in corrected aggregates (`aggregate_test_threshold_val_selected`) but primary frozen reporting in the note uses **@0.5** — keep that distinction explicit.
7. **Joint / sequential / capacity / TF** have validation metrics only — no locked confirmation test for those scouts.

---

## E. Stale-document patch plan (do **not** apply in this turn)

| Document | Issue | Later action |
|----------|-------|--------------|
| `notes/thesis_experiment_registry.md` + `results/diagnostics/thesis_experiment_registry.{json,csv}` | D+ marked `primary_result`/`table_eligible`; corrected only `thesis_supporting`; old LI/HI still `thesis_primary`; row count MD≠JSON | Re-role: corrected = primary frozen; D+ = historical; demote obsolete SSL primaries |
| `notes/thesis_tables_preview.md` | Closer than plan but may still mix recipes | Align to Tables 1–3 above; confirmation seeds |
| `notes/thesis_results_table_plan.md` | Stale (missing supervised HI, wrong table layout) | Rewrite or mark superseded by this inventory |
| `notes/current_protocol_recent_runs_summary.md` | Likely incomplete vs 2026-07-27/28 artifacts | Refresh pointers only |
| `README.md` result pointers | May still emphasize older protocols | Point to corrected multiseed + formal supervised aggregate |
| `tables/*.md` / `tables/*.tex` | May cite D+ or descriptive means | Retarget keys to confirmation_aggregate / formal_test_aggregate |

---

## F. Thesis-safe result hierarchy (claim language)

- **Frozen AMLWorld primary:** A corrected/no-preserve Multi-GIN encoder pretrained without AML fraud labels, evaluated with a frozen encoder and a supervised downstream MLP on **pre-3h H+X+TF**, yields confirmation-seed (1/3/4) test AUPRC **0.670±0.012** and AUROC **0.988±0.0002** (`final_corrected_no_preserve_multiseed.json`). Downstream labels are used only for the probe, not for encoder updates.
- **Supervised AMLWorld comparator:** Paper-faithful Multi-GIN+EU (ports, TDS off, paper_argmax) reproduces test F1 **0.660±0.060** vs published **0.648±0.012** (`eval_small_hi_…formal_aggregate.json`). Not an SSL result.
- **PaySim supervised ceiling:** Target-supervised Multi-GIN+EU under `paysim_legacy_duplicate_v1` reaches test paper_argmax F1 **0.202±0.007** (`paysim_supervised_multigin_eu.json`) — label-rich ceiling, not transfer.
- **Strict transfer limitation:** Strict inductive P1 (frozen AML BN) confirmation test AUPRC **0.053±0.033** remains modest and high-variance; not competitive with the supervised ceiling.
- **Label-free BN adaptation:** P2 (target-train BN only) confirmation test AUPRC **0.109±0.038** improves over P1 but is **not** strict zero-shot.
- **Ensembles / fine-tuning:** Equal-weight ensembles and supervised partial fine-tuning are **secondary/appendix**; they do not define multiseed robustness.
- **Exploratory joint replay:** Domain-BN joint SSL (seed 2, val-only) approximately matches sequential PaySim H lift with better AML retention than sequential continuation, but is **one-seed, validation-only, not table-eligible**; shared-BN joint replay harms PaySim.

---

## Consistency flags (human review)

1. **Registry vs corrected note** on what is “primary” (largest process conflict).
2. **Metric type mismatch risk:** supervised **paper_argmax F1** vs SSL **F1@0.5** / AUPRC — never average.
3. **PaySim P1 seed variance** (confirmation AUPRC values ≈ 0.084 / 0.018 / 0.057) — report ±SD, avoid overclaiming.
4. **X-only test AUPRC (0.086) > P1 mean (0.053)** on confirmation — surprising; keep both with protocol notes (different learners/stacks; X-only is a single control fit).
5. **Capacity-check MLP** showed random H ≫ pretrained H — exploratory; do not override multiseed P1 test narrative without a locked re-eval.
6. **Formal supervised σ (0.060) ≫ paper σ (0.012)** — documented `low_variance_reproduced=false`.

---

## Return checklist

| Item | Result |
|------|--------|
| Artifacts inspected (scan) | 492 diagnostic JSON + 251 notes MD + registry (371 rows) + key canonical reports listed above |
| Minimum main-text tables | Table 1 (supervised vs frozen), Table 2 (stacks; optional merge), Table 3 (PaySim P1/P2/controls + supervised ceiling callout) |
| Exact canonical JSON paths | Listed in §A and §B |
| Conflicts | Registry primary≠corrected note; MD registry row count; D+ vs corrected recipes; metric-type confusion risk; X-only vs P1 ordering |
| Missing | Typology on primary encoder; KNN aggregate JSON; PaySim published target; registry eligibility flags for corrected |
| Mutations / jobs | **None** — only this note + twin JSON written |
