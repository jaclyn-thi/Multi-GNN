# Canonical transaction-node checkpoint re-extraction

Status: **complete** · Extraction + frozen MLP eval only · **No GNN retraining** · 2026-07-23

Companion: [`results/diagnostics/gcpal_txn_node_canonical_reextraction.json`](../results/diagnostics/gcpal_txn_node_canonical_reextraction.json)

## Job IDs

| Role | Job ID | Outcome | Wall |
|------|--------|---------|------|
| Smoke (B ep5) | **18662062** | pass | 182s |
| Full A_identity | **18662525** | COMPLETED 0:0 | 1480s |
| Full B_gcpal | **18662526** | COMPLETED 0:0 | 1584s |

## Feasibility / memory (smoke)

- Peak GPU allocated / reserved: **3503 / 5332 MiB** (L40S 46 GB)
- Full-forward max allocated (test scope): ~29.7 GiB during encode
- CPU RSS (smoke): **11650 MiB**
- Edge counts matched: train **1 614 187**, train∪val **2 086 777**, full **2 605 952**
- Coverage 1.0, finite embeddings, retained-edge fraction 1.0
- Suite estimate **2008s** ≪ 6h; `scalable_inference_required=false`
- Inference: one no-grad full-scope forward per graph scope (no graph-destructive chunking)

## Checkpoint selection (expanding-window temporal val HxX AUPRC)

Recomputed from scratch; **never** uses test; **does not** carry legacy ep5 selection forward (though ep5 wins again).

| Arm | Selected epoch | Val HxX AUPRC | Fixed ep20 val |
|-----|----------------|---------------|----------------|
| A_identity | **5** | 0.017412 | 0.008076 |
| B_gcpal | **5** | 0.082042 | 0.026858 |

Legacy-chunked selection (historical, unchanged): A ep5 (0.0882), B ep5 (0.1584).

## Canonical A/B learning curves — temporal expanding-window HxX

### Validation AUPRC (selection)

| Epoch | A | B |
|------:|--:|--:|
| 5 | 0.017412 | 0.082042 |
| 10 | 0.008088 | 0.057486 |
| 15 | 0.008579 | 0.053864 |
| 20 | 0.008076 | 0.026858 |

### Test @ threshold 0.5

| Epoch | A AUPRC | B AUPRC | A AUROC | B AUROC | A F1 | B F1 |
|------:|--------:|--------:|--------:|--------:|-----:|-----:|
| 5 | 0.007159 | 0.136705 | 0.633526 | 0.799853 | 0.000000 | 0.000000 |
| 10 | 0.002381 | 0.081615 | 0.394798 | 0.747538 | 0.000000 | 0.000000 |
| 15 | 0.002138 | 0.054733 | 0.321555 | 0.758045 | 0.000000 | 0.000000 |
| 20 | 0.002213 | 0.024919 | 0.333986 | 0.682690 | 0.000000 | 0.000000 |

### Val-selected threshold → test (selected epochs = 5)

- **A_identity** thr=0.02: test AUPRC=0.007159, AUROC=0.633526, F1=0.031634, P=0.023077, R=0.050279, PPR=0.004063, tp/fp/tn/fn=81/3429/858860/1530
- **B_gcpal** thr=0.10: test AUPRC=0.136705, AUROC=0.799853, F1=0.270476, P=0.276803, R=0.264432, PPR=0.001781, tp/fp/tn/fn=426/1113/861176/1185

## Controlled temporal comparisons (test HxX AUPRC @ 0.5)

| Epoch | legacy_chunked_4096 | per-split isolation v1 | expanding-window v1 | Δ restore cross-chunk (PS−leg) | Δ add history (EW−PS) |
|------:|--------------------:|-----------------------:|--------------------:|-------------------------------:|----------------------:|
| A@5 | 0.012492 | 0.007651 | 0.007159 | -0.004841 | -0.000492 |
| B@5 | 0.048050 | 0.181482 | 0.136705 | +0.133432 | -0.044777 |
| A@10 | 0.011352 | 0.001872 | 0.002381 | -0.009481 | +0.000510 |
| B@10 | 0.038758 | 0.138898 | 0.081615 | +0.100141 | -0.057283 |
| A@15 | 0.006346 | 0.001358 | 0.002138 | -0.004988 | +0.000780 |
| B@15 | 0.030178 | 0.116371 | 0.054733 | +0.086193 | -0.061638 |
| A@20 | 0.001256 | 0.001347 | 0.002213 | +0.000091 | +0.000865 |
| B@20 | 0.071570 | 0.069688 | 0.024919 | -0.001882 | -0.044768 |

Interpretation:

- **Restoring cross-chunk within-split edges** (legacy → per-split): large **gain for B** at ep5–15 (+0.09 to +0.13 AUPRC); A mostly flat/slightly down.
- **Adding historical train context** to val/test (per-split → expanding-window): **hurts B** on this metric (−0.04 to −0.06); A nearly unchanged.
- Absolute temporal performance remains weak for A; B is stronger but still far from supervised AML baselines.

## Random-40 joint full-graph (diagnostic only)

Label: **random-40, transductive, diagnostic-only, not thesis-primary.** Separate from temporal tables.

| Epoch | A val HxX AUPRC | B val | A test @0.5 | B test @0.5 |
|------:|----------------:|------:|------------:|------------:|
| 5 | 0.035893 | 0.060402 | 0.035347 | 0.059901 |
| 10 | 0.023501 | 0.048524 | 0.023851 | 0.051586 |
| 15 | 0.025828 | 0.036123 | 0.029060 | 0.044215 |
| 20 | 0.010702 | 0.015719 | 0.011174 | 0.017084 |

Joint full-graph encode uses one forward over all 5 078 345 nodes / 2 605 952 edges; probe split applied **after** extraction.

## Does B still beat A?

| Protocol | Criterion | B beats A? |
|----------|-----------|------------|
| Legacy-chunked (historical) | shared frozen encode | **Yes** (unchanged claim) |
| Expanding-window val-selected ep5 | val HxX AUPRC | **Yes** (0.082 > 0.017) |
| Expanding-window val-selected ep5 | test HxX AUPRC @0.5 | **Yes** (0.137 > 0.007) |
| Expanding-window fixed ep20 | test HxX AUPRC @0.5 | **Yes** (0.025 > 0.002) |
| Joint random-40 ep5 | test HxX AUPRC @0.5 | **Yes** (0.060 > 0.035) |

Corrected wording for legacy artifacts: **B beats A under a shared frozen-checkpoint legacy-chunked extraction; canonical graph-preserving re-extraction results are reported here.**

## Table eligibility

| Artifact | Eligible? |
|----------|-----------|
| Original online-aug 5ep | **No** |
| Replay legacy-chunked | **No** (diagnostic A/B only) |
| Per-split isolation v1 | **No** (sensitivity) |
| Expanding-window v1 | **Candidate** thesis-primary extraction; promote only after review |
| Joint random-40 v1 | **No** (never primary) |

## Artifact paths

- Smoke: `results/diagnostics/gcpal_txn_node_canonical_reextract_smoke_B_ep05_job18662062.json`
- A arm: `results/diagnostics/gcpal_txn_node_canonical_reextract_A_identity_seed2_job18662525.json`
- B arm: `results/diagnostics/gcpal_txn_node_canonical_reextract_B_gcpal_seed2_job18662526.json`
- Cached H: `embeddings/gcpal_txn_node_canonical/{A_identity,B_gcpal}/ep{05,10,15,20}/…`

## Confirmation

- **No GNN training** in smoke or arm jobs (`gnn_training_occurred: false`).
- Historical `*_5ep_*` / `*_20ep_*` scout JSON/MD **not rewritten**.
- Modes: expanding-window = thesis-primary candidate; per-split = sensitivity; joint random-40 = diagnostic.

