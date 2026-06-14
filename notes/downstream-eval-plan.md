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
* Fixed @ 0.5 typology on SSL leaders (sym, 8192neg) and tier-2 encoders.
* Richer attempt-level metrics (e.g. recall@k per attempt).

**Pattern typology results (Jun 2026)**

Test split: **1,611** laundering edges; **1,240** with known pattern metadata, **371** without. Protocol: refit linear probe (`--class_weight model`), typology on test predictions only. **Nine runs complete** (six val-tuned + three fixed @ 0.5 on morph triangle ablation trio).

#### Val-tuned threshold (max F1 on val) — overall test

| Run | Thr | AUROC | F1 | Prec | Recall | Known-meta recall* |
|-----|-----|-------|-----|------|--------|-------------------|
| `hi_contrastive_proj_sym_20ep_bestckpt` | 0.364 | 0.929 | **0.222** | **0.185** | 0.277 | 35% |
| `hi_contrastive_proj_8192neg_20ep_bestckpt` | 0.393 | **0.930** | 0.191 | 0.149 | 0.267 | 34% |
| `hi_morphology_global_clustering_proj_20ep_bestckpt` | 0.273 | 0.929 | 0.156 | 0.117 | 0.235 | 29% |
| `hi_morphology_global_triangles_proj_20ep_bestckpt` | 0.204 | 0.912 | 0.145 | 0.095 | 0.305 | 37% |
| `hi_morphology_global_20ep` (M1b) | 0.156 | 0.920 | 0.108 | 0.069 | 0.248 | 28% |
| `hi_morphology_global_triangles_only_proj_20ep_bestckpt` | 0.201 | 0.910 | 0.067 | 0.036 | **0.398** | **48%** |

\*Among 1,240 laundering edges with known pattern labels.

#### Per-pattern recall @ val-tuned (known metadata; thresholds differ — see caveat)

| Pattern | M1b | clust+proj | tri-only | tri+clust | sym+proj | 8192neg |
|---------|-----|------------|----------|-----------|----------|---------|
| FAN-OUT | **5%** | 19% | **68%** | 26% | 49% | **53%** |
| GATHER-SCATTER | 32% | 42% | **56%** | 39% | 35% | **44%** |
| FAN-IN | **40%** | 33% | 39% | 31% | 30% | 24% |
| SCATTER-GATHER | 30% | 23% | **46%** | 39% | 36% | 31% |
| STACK | 24% | **15%** | 33% | 33% | 31% | 21% |

#### Per-pattern one-vs-rest AUROC (full test split; fair across runs)

| Pattern | M1b | clust+proj | tri-only | tri+clust | sym+proj | 8192neg |
|---------|-----|------------|----------|-----------|----------|---------|
| GATHER-SCATTER | 0.941 | **0.971** | 0.948 | 0.918 | 0.950 | 0.934 |
| SCATTER-GATHER | 0.945 | 0.950 | 0.919 | 0.948 | **0.964** | 0.955 |
| FAN-IN | **0.959** | 0.949 | 0.930 | 0.934 | **0.968** | 0.916 |
| FAN-OUT | 0.884 | 0.962 | 0.966 | 0.921 | 0.922 | **0.970** |
| CYCLE | 0.914 | 0.927 | 0.912 | 0.924 | 0.942 | 0.933 |

#### Fixed threshold @ 0.5 (morph triangle ablation trio)

| Run | F1 | Prec | Recall | Known-meta recall | FAN-OUT recall |
|-----|-----|------|--------|-------------------|----------------|
| clust+proj | **0.132** | **0.182** | 0.103 | 13% | 8% |
| M1b | 0.080 | 0.177 | 0.052 | 6% | 1% |
| tri-only+proj | 0.075 | 0.049 | 0.168 | 21% | **29%** |

At shared threshold, triangles-only's per-pattern recall advantage **shrinks** (known-meta 48% → 21%); clustering+proj **wins F1**. Triangles-only still leads FAN-OUT @ 0.5 (29% vs 8%).

#### Main findings

1. **Projection unlocks FAN-OUT.** M1b without projection: FAN-OUT is the **worst** type (5% recall). SSL runs (sym, 8192neg): FAN-OUT becomes the **best** type (49–53% recall). Morphology clustering+proj improves gather/scatter **ranking** (AUROC 0.971) but fan-out recall remains modest (19%).
2. **sym+proj = best operational flagger** — highest F1 (0.222) and precision (18.5%) with balanced typology. **8192neg+proj** = best AUROC (0.930) and best FAN-OUT / BIPARTITE ranking.
3. **triangles-only over-flags, not pattern-blind** — highest per-pattern recall at val-tuned thresholds on **every** type (~17.6K alerts, 3.6% precision) yet worst F1. Fixed @ 0.5 confirms threshold miscalibration.
4. **Stacking triangles + clustering (14 local) regresses** vs clustering-only on AUROC and gather/scatter ranking; sits between tri-only and clust on flagging.
5. **Cross-run recall at val-tuned thresholds is misleading** — use AUROC or fixed @ 0.5 for fair flagging comparison.

**Practical recipes (typology-informed):** best **F1** → sym+proj; best **AUROC** → 8192neg; best **morph SSL** → clustering+proj (gather/scatter); avoid triangles-only for flagging.

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

### 3. SAML-D / other AML-style transaction datasets

SAML-D seems like a good candidate if we can get it working. It is more directly aligned with transaction-monitoring/AML than PaySim and may already have a formatter in the repo.

If recoverable, it could be a strong second external dataset because it likely has:

* account-to-account transfer structure
* edge-level laundering labels
* similar graph ontology to our current pipeline

This would be a useful test of whether the GFM transfers beyond AMLWorld while staying in the same broad task family.

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

Add PaySim as a non-AML edge-level fraud dataset.

This gives a clean first test of cross-domain transfer:

> AMLWorld-pretrained transaction GFM → PaySim mobile-money fraud detection

### Then, if feasible

Retry SAML-D with help from Cursor/AI agent. If it works, it could become a strong thesis-aligned external AML transaction benchmark.

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
