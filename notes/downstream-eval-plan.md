# Downstream Finance Task Considerations

This note summarizes current thinking around possible downstream tasks for evaluating our finance graph foundation model (GFM). The main goal is to understand whether pretraining on large synthetic transaction graphs produces reusable transaction/edge embeddings for finance tasks beyond the current AMLWorld binary edge-classification setup.

## Current model framing

Our current pipeline is primarily **edge/transaction-level**:

* nodes represent accounts/entities
* edges represent transactions
* transaction embeddings are constructed from:

  * source node embedding
  * destination node embedding
  * edge/transaction embedding
* downstream evaluation currently uses frozen embeddings with a linear/sklearn probe

This means the most natural downstream tasks are ones where the prediction unit is a **transaction edge**.

Because of this, we do not necessarily need to support node-level or subgraph-level benchmarks immediately. Those could be future extensions, but the core GFM claim can remain focused on reusable **transaction embeddings**.

## Most promising downstream directions

### 1. AMLWorld laundering-pattern tasks

AMLWorld includes metadata about laundering patterns such as cycles, fan-in, and fan-out. This could be a very useful downstream task because it stays within the current edge-level pipeline while testing something richer than binary laundering classification.

Possible tasks:

* binary classification for each pattern type

  * example: cycle vs non-cycle
* multi-class classification among laundering pattern types

  * example: cycle vs fan-in vs fan-out
* pattern-aware retrieval

  * example: do nearest-neighbor transaction embeddings retrieve transactions from the same laundering pattern?
* label-scarce pattern classification

This is appealing because it tests whether the model learns meaningful financial-crime structure and graph morphology, not just a binary laundering signal.

#### Laundering pattern metadata — status (Jun 2026)

Infrastructure for **auxiliary typology diagnostics** is in place on **Small-HI**. Pattern metadata is **not** used in training, inference, splits, or binary labels.

**Done**

* Parse AMLWorld `patterns.txt` → `aml-data/Small-HI/laundering_attempt_metadata.csv` (`scripts/parse_laundering_patterns.py`; needs raw Kaggle CSV + `formatted_transactions.csv` for `EdgeID` join).
* Load `EdgeID → {attempt_id, pattern_type, pattern_detail, …}` once in `get_data()` → `te_data.pattern_metadata_by_edge_id` / `te_data.csv_edge_ids` (`pattern_metadata.py`).
* Post-hoc test diagnostics on frozen-embedding linear-probe predictions (`pattern_diagnostics.py`, `scripts/evaluate_pattern_typology.py`; Slurm: `slurm/run_pattern_typology_diag.sh`, batch: `slurm/submit_pattern_typology_remaining.sh`, tier 2: `slurm/submit_pattern_typology_tier2.sh`, fixed @ 0.5: `slurm/submit_pattern_typology_fixed_thr0.5.sh`).

**Waiting on / not started**

* Actual **downstream typology probes** (multi-class pattern type, retrieval, label-scarce pattern classification) — diagnostics only for now.
* Pattern files for splits other than Small-HI.
* Policy for **~1,968 laundering edges** in the full dataset with no `patterns.txt` label (counted in audit; not assigned a typology).
* ~~Fixed @ 0.5 typology on SSL leaders (sym, 8192neg)~~ — done.
* ~~Fixed @ 0.5 typology on tier-2 encoders~~ — done (nine-way table below; **typology benchmark complete**).
* Richer attempt-level metrics (e.g. recall@k per attempt).

**Pattern typology results (Jun 2026)**

Test split: **1,611** laundering edges; **1,240** with known pattern metadata, **371** without. Protocol: refit linear probe (`--class_weight model`), typology on test predictions only. **Nineteen runs complete** (ten val-tuned + nine fixed @ 0.5).

#### Val-tuned threshold (max F1 on val) — overall test

| Run | Thr | AUROC | F1 | Prec | Recall | Known-meta recall* |
|-----|-----|-------|-----|------|--------|-------------------|
| `hi_contrastive_proj_sym_20ep_bestckpt` | 0.364 | 0.929 | **0.222** | **0.185** | 0.277 | 35% |
| `hi_contrastive_proj_asym_16384_20ep_bestckpt` | 0.329 | 0.920 | **0.206** | 0.161 | 0.286 | 36% |
| `hi_contrastive_proj_8192neg_20ep_bestckpt` | 0.393 | **0.930** | 0.191 | 0.149 | 0.267 | 34% |
| `hi_morphology_global_clustering_proj_20ep_bestckpt` | 0.273 | 0.929 | 0.156 | 0.117 | 0.235 | 29% |
| `hi_morphology_global_triangles_proj_20ep_bestckpt` | 0.204 | 0.912 | 0.145 | 0.095 | 0.305 | 37% |
| `hi_contrastive_proj_20ep_bestckpt` (asym baseline) | 0.331 | 0.927 | 0.144 | 0.098 | 0.272 | 34% |
| `hi_morphology_global_sym_proj_20ep_bestckpt` | 0.421 | 0.930 | 0.134 | 0.093 | 0.239 | 30% |
| `hi_morphology_global_20ep` (M1b) | 0.156 | 0.920 | 0.108 | 0.069 | 0.248 | 28% |
| `hi_morphology_global_proj_20ep_bestckpt` (M1b+proj) | 0.267 | 0.924 | 0.096 | 0.058 | 0.289 | 36% |
| `hi_morphology_global_triangles_only_proj_20ep_bestckpt` | 0.201 | 0.910 | 0.067 | 0.036 | **0.398** | **48%** |

\*Among 1,240 laundering edges with known pattern labels.

#### Tier 2 @ val-tuned — per-pattern recall (known metadata)

| Pattern | asym+proj | asym@16384 | sym+morph | M1b+proj | sym+proj (ref) |
|---------|-----------|------------|-----------|----------|----------------|
| FAN-OUT | **47%** | 44% | 39% | 32% | 49% |
| GATHER-SCATTER | 37% | **44%** | 34% | **44%** | 35% |
| FAN-IN | 40% | **41%** | 24% | 34% | 30% |
| STACK | 22% | **32%** | **18%** | 21% | 31% |

**Tier 2 takeaways:** **asym@16384** is the tier-2 F1 surprise (**0.206**, within 0.016 of sym) with **2× precision** of baseline asym (16.1% vs 9.8%) and half the flag volume (~2.9K vs ~4.5K). **sym+morph** confirms probe: AUROC ties sym but F1 **0.134** (−0.088). **M1b+proj** (no clustering) is recall-heavy like triangles-only (~8K flags, 5.8% precision). All projection runs flip FAN-OUT from M1b's worst type to best.

#### Per-pattern recall @ val-tuned — tier 1 + morph (known metadata; thresholds differ)

| Pattern | M1b | clust+proj | tri-only | tri+clust | sym+proj | 8192neg |
|---------|-----|------------|----------|-----------|----------|---------|
| FAN-OUT | **5%** | 19% | **68%** | 26% | 49% | **53%** |
| GATHER-SCATTER | 32% | 42% | **56%** | 39% | 35% | **44%** |
| FAN-IN | **40%** | 33% | 39% | 31% | 30% | 24% |
| SCATTER-GATHER | 30% | 23% | **46%** | 39% | 36% | 31% |
| STACK | 24% | **15%** | 33% | 33% | 31% | 21% |

#### Per-pattern one-vs-rest AUROC (full test split; fair across runs)

| Pattern | M1b | clust+proj | sym+proj | 8192neg | asym@16384 |
|---------|-----|------------|----------|---------|------------|
| GATHER-SCATTER | 0.941 | **0.971** | 0.950 | 0.934 | — |
| SCATTER-GATHER | 0.945 | 0.950 | **0.964** | 0.955 | — |
| FAN-IN | **0.959** | 0.949 | **0.968** | 0.916 | — |
| FAN-OUT | 0.884 | 0.962 | 0.922 | **0.970** | 0.959 |

#### Fixed threshold @ 0.5 — nine-way comparison (fair flagging; complete)

| Run | AUROC | F1 | Prec | Recall | Known-meta recall | FAN-OUT recall |
|-----|-------|-----|------|--------|-------------------|----------------|
| **asym@16384+proj** | 0.920 | **0.220** | **0.246** | 0.199 | 26% | 29% |
| sym+proj | 0.929 | 0.211 | 0.225 | 0.198 | 25% | 34% |
| 8192neg+proj | **0.930** | 0.196 | 0.181 | 0.213 | **27%** | **45%** |
| asym+proj (baseline) | 0.927 | 0.137 | 0.125 | 0.151 | 19% | 32% |
| sym+morph+proj | 0.930 | 0.133 | 0.104 | 0.185 | 23% | 35% |
| clust+proj | 0.929 | 0.132 | 0.182 | 0.103 | 13% | 8% |
| M1b+proj | 0.924 | 0.092 | 0.073 | 0.126 | 16% | 7% |
| M1b | 0.920 | 0.080 | 0.177 | 0.052 | 6% | 1% |
| tri-only+proj | 0.910 | 0.075 | 0.049 | 0.168 | 21% | 29% |

**@ fixed 0.5:** **asym@16384** best F1 (0.220) and precision (24.6%) — beats sym (0.211). **8192neg** best AUROC and FAN-OUT (45%). **Metric choice matters:** sym wins val-tuned F1 (0.222); asym@16384 wins shared-threshold F1.

#### Tier 2 @ fixed 0.5 — val-tuned vs shared threshold

| Run | F1 val | F1 @ 0.5 | Prec val | Prec @ 0.5 | Known-meta val | Known-meta @ 0.5 |
|-----|--------|----------|----------|------------|----------------|------------------|
| **asym@16384** | 0.206 | **0.220** | 0.161 | **0.246** | 36% | 26% |
| asym+proj | 0.144 | 0.137 | 0.098 | 0.125 | 34% | 19% |
| sym+morph | 0.134 | 0.133 | 0.093 | 0.104 | 30% | 23% |
| M1b+proj | 0.096 | 0.092 | 0.058 | 0.073 | 36% | 16% |

asym@16384 **improves** @ 0.5 (not a val-threshold artifact). Baseline asym and M1b+proj known-meta recall **halves** @ 0.5 — val-tuned thresholds inflated their typology story.

#### Main findings

1. **Projection unlocks FAN-OUT.** M1b without projection: FAN-OUT is the **worst** type (5% recall). SSL runs (sym, 8192neg): FAN-OUT becomes the **best** type (49–53% @ val-tuned; 34–45% @ 0.5).
2. **Val-tuned vs fixed @ 0.5 pick different SSL winners** — sym+proj best @ val-tuned F1 (**0.222**); **asym@16384 best @ fixed 0.5** (F1 **0.220**, prec **24.6%**). Use fixed @ 0.5 for fair cross-run flagging comparison.
3. **8192neg+proj** = best AUROC (0.930) and best FAN-OUT recall @ 0.5 (45%).
4. **asym@16384** = most balanced tier-2 run @ 0.5 (best gather/scatter recall 35%, not fan-out-dominated).
5. **triangles-only / M1b+proj over-flag** — high known-meta recall @ val-tuned, collapses @ 0.5 (M1b+proj: 36% → 16%); clustering expert needed for usable morphology SSL.
6. **sym+morph hurts flagging** — 0.133 @ 0.5 vs sym 0.211; typology confirms probe (−0.088 F1).

**Practical recipes (typology-informed):** best **F1 @ val-tuned** → sym+proj; best **F1 @ fixed 0.5** → asym@16384; best **AUROC** → 8192neg; best **FAN-OUT @ 0.5** → 8192neg; best **morph SSL** → clustering+proj; avoid triangles-only / M1b+proj for flagging.

Outputs: `results/diagnostics/<unique_name>/` (val-tuned) · `results/diagnostics/<unique_name>_thr0.5/` (fixed). Files: `pattern_typology_test.json`, `pattern_typology_by_type.csv`, `pattern_typology_by_detail.csv`, `pattern_typology_by_attempt.csv`.

### 2. PaySim fraud detection

PaySim is probably the easiest external downstream dataset to add first.

It naturally fits the current graph format:

* nodes: accounts
* edges: transactions
* edge label: `isFraud`
* timestamp: `step`

This would test transfer from AML-style synthetic transaction pretraining to mobile-money fraud detection. It is still synthetic and somewhat old, so it should probably be framed as a useful sanity check rather than the strongest evidence of real-world generalization.

Recommended protocol:

* temporal split by `step`
* frozen pretrained encoder + linear probe
* random-init frozen encoder control
* supervised-from-scratch baseline if feasible
* metrics: AUROC, AUPRC/AP, F1 with validation-tuned threshold, precision@k / recall@k

Important caveat: avoid using leaky PaySim fields such as balance fields or `isFlaggedFraud` in the main experiment.

#### PaySim — status (Jun 2026)

**Done**

* `format_paysim.py` — vectorized formatter; maps `isFraud` → `Is Laundering`, hourly timestamps, 4-d edge contract.
* `dataset_specs.py` — `PaySim` adapter (`hourly_step` splits).
* `dataset_splits.py` — shared temporal split logic (`calendar_day` / `hourly_step`).
* `data_loading.py` — `--data PaySim` via registry; writes `dataset_spec` to extraction `meta.json`.
* `embedding_extraction.py` — `--random_init` baseline (skip checkpoint load); `--embeddings_dir`.
* `scripts/validate_paysim_data.py` — format, stats, optional `get_data()` smoke test.
* `tests/test_format_paysim.py` — formatter unit tests.
* Slurm: `run_paysim_load_smoke.sh`, `run_paysim_extract_probe.sh`, `submit_paysim_transfer.sh`.

**Dev runs (not frozen benchmark protocol)**

* Load smoke test passed (~6.36M edges, 743 hourly buckets).
* First transfer: `hi_contrastive_proj_sym_20ep_bestckpt` → PaySim probe — test **AUROC 0.866**, F1 0.089 @ val-tuned / 0.127 @ 0.5.
* Random-init baseline: queued (`random_init_gin`, `--random_init`).

**Not started / deferred**

* AUPRC, precision@k in `linear_probe.py` (defer until official eval pass).
* PaySim label-efficiency probes.
* PaySim supervised-from-scratch baseline.
* Additional encoder comparisons via `submit_paysim_transfer.sh` tier `all`.

**Notes:** test fraud rate ~4× train (later PaySim steps). Prefer AUROC for cross-run comparison; val-tuned F1 can be pessimistic on test.

### 3. SAML-D / other AML-style transaction datasets

SAML-D seems like a good candidate if we can get it working. It is more directly aligned with transaction-monitoring/AML than PaySim and may already have a formatter in the repo.

If recoverable, it could be a strong second external dataset because it likely has:

* account-to-account transfer structure
* edge-level laundering labels
* similar graph ontology to our current pipeline

This would be a useful test of whether the GFM transfers beyond AMLWorld while staying in the same broad task family.

#### SAML-D — status (Jun 2026)

**Done**

* `format_saml_d_files.py` — maps SAML-D CSV to shared `formatted_transactions.csv` schema.
* `aml-data/SAML-D/formatted_transactions.csv` present (~9.5M edges).
* `--data SAML-D` uses AMLWorld default spec (`calendar_day`, `Is Laundering`) via `get_dataset_spec`.
* Slurm smoke: `run_saml_d_supervised_smoke.sh`, control `run_small_hi_supervised_smoke.sh`.

**Findings (1-epoch supervised smoke, dev only)**

| Dataset | Test F1 (1 ep) | Notes |
| ------- | ---------------- | ----- |
| Small-HI | ~0.00 | Expected — needs full training for ~0.49 F1 |
| SAML-D | ~0.90 | Anomalously high vs Small-HI under same flags |

Temporal splits look sane (~321 calendar days, ~60/20/20, similar pos rates per split). Likely drivers: transductive test graph (all edges visible at eval), long timeline (~320 days), high train/test account overlap — **not** obviously a random-split formatter bug.

**Not started / needed before trusting SAML-D numbers**

* Explicit `SAML-D` entry in `dataset_specs.py` (documentation only today).
* `scripts/validate_saml_d_data.py` smoke test.
* Strict temporal test graph / train-only normalization (pipeline-wide decision).
* Supervised baseline at full epochs vs Small-HI for fair comparison.
* External-AML transfer probes (frozen AML encoder → SAML-D), distinct from supervised-from-scratch.

### 4. Ranking and label-scarcity evaluations

These are high-value and relatively low-effort additions.

Instead of only reporting binary classification metrics, we should evaluate whether embeddings are useful for alert prioritization:

* AUPRC / Average Precision
* precision@k
* recall@k
* lift@k
* recall at fixed false-positive rate

Label-scarcity is also important because one of the main motivations for pretraining is that fraud/AML labels are sparse and expensive.

Useful label fractions:

* 1%
* 5%
* 10%
* 25%
* 100%

This can be run on AMLWorld, AMLWorld pattern tasks, PaySim, and SAML-D if available.

## Lower-priority / future datasets

### IEEE-CIS

IEEE-CIS is a real fraud dataset, but it is not naturally an account-to-account transaction graph. We could construct a customer-to-merchant or customer-to-product graph, but this requires more design choices and has more leakage risks.

This may be useful later as a real-world fraud supplement, but it is not the best first external dataset.

### Elliptic / Elliptic++

Elliptic is a canonical crypto illicit-transaction benchmark, but its native representation is different from ours:

* Elliptic nodes are transactions
* Elliptic edges are flows between transactions
* labels are node-level

This does not naturally fit our edge-centric pipeline.

Elliptic++ may be more useful later because it includes address/actor information and address-transaction views. However, using it for edge-level evaluation would likely require constructing new edge labels, such as “transaction touches an illicit actor,” which needs careful temporal censoring to avoid leakage.

Possible future use:

* constructed edge-risk task from Elliptic++ actor labels
* historical endpoint-risk labeling
* account/address-level classification if we later add node readout support

### Ethereum phishing / Ponzi datasets

Ethereum phishing and Ponzi datasets are interesting but often use address-level or subgraph-level labels. They may be useful later, but they require more task construction and leakage handling.

For the current edge-centric GFM, they are probably stretch goals rather than immediate next steps.

## Recommended next steps

### Immediate next experiment

Use AMLWorld laundering-pattern metadata to create pattern-aware **downstream probes** (see [laundering pattern metadata status](#laundering-pattern-metadata--status-jun-2026) above — parsing and post-hoc diagnostics are already implemented).

This is likely the best next step because it is:

* edge-level
* low-engineering
* directly connected to morphology
* more interesting than another binary AML result
* useful for showing that the embeddings encode laundering structure

### First external transfer experiment

~~Add PaySim as a non-AML edge-level fraud dataset.~~ **In progress (Jun 2026):** infrastructure + first transfer run complete (see [PaySim status](#paysim--status-jun-2026)). Random-init baseline queued.

This gives a clean first test of cross-domain transfer:

> AMLWorld-pretrained transaction GFM → PaySim mobile-money fraud detection

### Then, if feasible

Retry SAML-D — formatter and data exist; **supervised eval protocol needs review** before external-AML or in-domain claims (see [SAML-D status](#saml-d--status-jun-2026)).

## Tentative evaluation suite

A coherent thesis evaluation suite could be:

| Evaluation axis             | Dataset/task                              | What it demonstrates                          |
| --------------------------- | ----------------------------------------- | --------------------------------------------- |
| In-domain AML               | AMLWorld binary edge classification       | learns suspicious transaction representations |
| Typology awareness          | AMLWorld pattern classification/retrieval | captures laundering structure and morphology  |
| Label efficiency            | AMLWorld + pattern probes                 | useful when labels are scarce                 |
| Cross-domain fraud transfer | PaySim edge fraud                         | transfers to mobile-money fraud               |
| External AML transfer       | SAML-D if recoverable                     | transfers beyond AMLWorld                     |
| Alert prioritization        | ranking metrics on AMLWorld/PaySim/SAML-D | useful for investigation workflows            |

## Main takeaway

The project does not need to support every financial graph benchmark immediately. The cleanest story is:

> We pretrain an edge-centric finance GFM on synthetic transaction graphs and evaluate whether the resulting transaction embeddings transfer to multiple downstream edge-level finance tasks, including AML classification, laundering-pattern recognition, label-scarce fraud detection, and transaction-risk ranking.

Node-level and subgraph-level datasets like Elliptic, Elliptic++ actors, and Ethereum phishing/Ponzi can remain future extensions unless they become necessary for the thesis scope.
