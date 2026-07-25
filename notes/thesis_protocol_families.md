# Thesis protocol families

Status: **canonical** · Date: 2026-07-22 · Primary vs diagnostic marked per section.

This document is the single place for protocol definitions. Per-run notes remain provenance; do not treat them as the protocol source of truth.

Related: [`evaluation_protocols.md`](evaluation_protocols.md) · [`cli-reference.md`](cli-reference.md) · [`documentation_audit_2026-07-22.md`](documentation_audit_2026-07-22.md).

---

## A. Edge-centric thesis SSL (primary representation)

| Field | Value |
|-------|--------|
| Graph | Accounts = nodes; transactions = edges |
| Encoder | Hetero Multi-GIN (+ optional EU / ports / TDS / ego / reverse MP) |
| Edge embedding | Seed-edge readout → optional `embedding_head` (post-128) or raw pre-head concat (pre-3h) |
| Views | Two independently augmented graphs (edge drop + edge-attr mask) |
| Loss | Asymmetric InfoNCE with optional GraphCL projection head (extraction uses encoder `z`, not projection) |
| Reverse features | Inherited alias+trailing swap **or** `--correct_reverse_edge_features` (named directional swap) |
| Seed retention | Optional `--preserve_seed_edges` keeps target seed edge_ids through edge drop |
| Negatives | `--contrastive_num_neg_samples` (8192 typical; `0` = all in-batch negatives, chunked) |
| Accum | `--contrastive_accum_steps` (effective optimizer step = batch × accum) |
| Evaluation | Temporal frozen probe (primary); see evaluation protocols |

### Controlled A/B/C/D comparison (Small-HI, seed 2, 40ep scaffold)

| Arm | TDS | Reverse features | Preserve seeds | Role |
|-----|-----|------------------|----------------|------|
| A — inherited TDS-on | on | **malformed** inherited reverse under TDS | off | Historical / negative control |
| B — TDS-off | off | N/A (no TDS cols) | off | Ablation |
| C — TDS-off + preserve | off | N/A | on | Ablation |
| D — corrected TDS | on | `--correct_reverse_edge_features` | off | Semantic fix |
| D+ — corrected TDS + preserve | on | corrected | on | **Strongest semantically valid edge-centric condition tested so far** |

Do **not** claim any arm reproduces GCPAL. They are thesis contrastive ablations under the Multi-GNN edge-centric graph.

Probe artifacts (examples):

- `results/diagnostics/probe_feature_ablation_current_protocol_gin_40ep_seed2_tds_off.json`
- `…_tds_off_preserve_seed.json`
- `…_tds_corrected.json`
- `…_tds_corrected_preserve_seed.json`

Inherited TDS-on baseline remains the older current-protocol 40ep seed2 family (malformed reverse under TDS).

### Batch-size E/F diagnostic (not main-table)

Matched for optimizer steps and effective anchors per update:

| Arm | Batch | Accum | Negatives |
|-----|-------|-------|-----------|
| E | 8192 | 4 | all aligned in-batch (`num_neg=0`) |
| F | 2048 | 16 | all aligned in-batch |

Findings (10-epoch scout, seed 2, corrected+preserve scaffold): E beat F on emb / emb+raw AUPRC; F cut peak memory; **batch size alone is not the missing GCPAL ingredient**. Smaller cap remains useful for positive-complete **transaction-node** batching. Do not promote either scout to a main thesis result.

Artifacts: `probe_feature_ablation_gin_emlps_ports_tds_corrected_preserve_seed_asym_proj_allneg_bs{8192_accum4,2048_accum16}_10ep_seed2.*`

---

## B. Supervised Multi-GIN+EU paper parity (baseline)

Canonical Small-HI command (README): `--model gin --objective supervised --supervised_head legacy --emlps --reverse_mp --ego --ports`, **omit `--tds`**, 50 epochs, best-val minority F1 checkpoint, paper_argmax test.

| | Value |
|--|--------|
| Paper Multi-GIN+EU | 0.6479 ± 0.0122 minority F1 |
| Formal seeds 1–3 | 0.663, 0.718, 0.598 |
| Aggregate | 0.660 ± 0.060 |
| Claim | Mean reproduced; paper’s low variance **not** |

**Not paper-compatible:** older TDS-on 100ep Small-HI legacy run. Enabling non-paper TDS also activated incorrect reverse-edge feature semantics on the inherited path (see reverse-feature audits). Upstream reverse-MP reference: commit `252b025`; `fc751e8` alone is not the safest training reference.

---

## C. Transaction-node GCPAL-inspired path

### NOT AN EXACT GCPAL REPRODUCTION

Isolated package `gcpal_txn_node/` + `scripts/gcpal_txn_node_*.py`. Does **not** import Multi-GNN training defaults as GCPAL.

**Frozen extraction (2026-07-23):** `gcpal_txn_node/extraction.py` defines (1) per-split isolation v1 as **sensitivity**, (2) temporal expanding-window v1 as **candidate thesis-primary**, (3) joint full-graph random-40 v1 as **diagnostic-only**. Historical scout/resume encodes used legacy 4096-chunk induce — **not table-eligible**. Scope audit: [`gcpal_txn_node_extraction_scope_audit.md`](gcpal_txn_node_extraction_scope_audit.md). Re-extract suite: [`gcpal_txn_node_canonical_reextraction.md`](gcpal_txn_node_canonical_reextraction.md). Positive-aggregation ablation (B/C/D): [`gcpal_txn_node_posagg_ablation.md`](gcpal_txn_node_posagg_ablation.md) — val HxX AUPRC selects **D SupCon**. Original 5ep: noncanonical online-aug. Replay: legacy-chunked diagnostic (**B beats A under a shared frozen-checkpoint legacy-chunked extraction**).

**Challenge full-stack eval (2026-07-24, job 18678029, no GNN train):** [`gcpal_challenge_fullstack_eval.md`](gcpal_challenge_fullstack_eval.md). Candidates from val provenance only (edge D+ corrected+preserve post/pre-3h; txn D SupCon ep5 expanding-window; feature controls). Temporal primary winner: `edge_pre3h|H+X+TF|mlp|none`. Reconstructed random-40/60 **do not** exceed published F1 targets 0.581/0.658 under our protocol. Comparability gate **PARTIAL**. Next fine-tune recommendation recorded in that note (not auto-submitted).

| Choice | Implementation assumption |
|--------|---------------------------|
| Nodes | Transactions |
| Adjacency | Explicit **immediate-next financial-flow** policy |
| Views | Random graph views |
| KNN | Sparse **global** cached feature KNN (train-split cache) |
| Positives | Identity; structural (flow neighbors); KNN (k=15) |
| Loss mix | λ=0.3, τ=0.5 (GCPAL-style naming only) |
| Cap | `max_total_nodes=2048` unique nodes per step |
| Paper gaps | Missing official code → adjacency, positives, batching, and splits are reimplemented under documented assumptions |

### Ordinary vs positive-complete batching

| Mode | Behavior | KNN positive coverage |
|------|----------|----------------------|
| Ordinary minibatch | Loss/positives only among co-sampled anchors | ~1.3% anchors with any global KNN neighbor in-batch |
| Positive-complete | Greedy anchors + retrieve full global k-NN (+ structural) into batch; **anchor-only** loss rows; context as pos/neg/MP | Designed for ~100% KNN presence within cap |

### Preliminary five-epoch A/B (positive-complete, seed 2)

Matched ~7935 optimizer steps / ~981984 anchor exposures. Keep **temporal** and **random-40** in separate tables; random-40 is diagnostic-only.

| Mode | Positives |
|------|-----------|
| A_identity | Identity only |
| B_gcpal | Identity + structural + KNN (name is convenient — **not** exact reproduction) |

Artifacts: `notes/gcpal_txn_node_poscomplete_scout_{A_identity,B_gcpal}_5ep_seed2.md` + matching JSON under `results/diagnostics/`.

### Failed forensic (provenance only)

Jobs diagnosing F1 gap vs GCPAL Table 2 via D embeddings **failed** (API drift in extract helpers). Status: **failed / pending**; **no scientific conclusion**; do not ingest metric rows. See registry `status=failed` provenance entries after the 2026-07-22 sync.
