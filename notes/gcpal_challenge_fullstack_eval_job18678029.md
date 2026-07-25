# GCPAL challenge full-stack evaluation

**No GNN training.** NOT an exact GCPAL reproduction.

Companion: [`results/diagnostics/gcpal_challenge_fullstack_eval_job18678029.json`](../results/diagnostics/gcpal_challenge_fullstack_eval_job18678029.json)
Job: `18678029`
Indexed in: [`notes/README.md`](README.md) · [`thesis_protocol_families.md`](thesis_protocol_families.md) · experiment registry (`protocol_family=gcpal_challenge_fullstack`)

## Candidates (validation provenance)

- **Edge D+**: corrected reverse + `preserve_seed_edges`, post-128 & pre-3h (job 18514684; val-selected among A/B/C/D)
- **Txn D SupCon ep5**: job 18669618; val HxX AUPRC among B/C/D aggregations
- **Feature controls**: X / TF / morph

## Comparability gate

**PARTIAL** — GCPAL published raw/feature baselines are not recoverable from the paper text available in-repo (no code release). We report X-only logistic/MLP under our temporal protocol as the reconstruction gate, and compare method F1 to Table-2 targets 0.581 (40%) / 0.658 (60%) only under our reconstructed random-label protocol.

## Temporal primary (selected)

**Selected:** `edge_pre3h|H+X+TF|mlp|none`
- val AUPRC=0.550000  val F1@sel=0.575107
- test AUPRC@0.5=0.674189 AUROC=0.988146
- test F1@0.5=0.655917 F1@val-thr=0.611822
- P@100/500/1000=0.990/0.934/0.851

### Top-10 by val AUPRC

| tag | val AUPRC | test AUPRC@0.5 | F1@val-thr |
|-----|----------:|---------------:|-----------:|
| `edge_pre3h|H+X+TF|mlp|none` | 0.5500 | 0.6742 | 0.6118 |
| `edge_pre3h|H+X+TF|mlp_focal|none` | 0.5447 | 0.6797 | 0.6481 |
| `edge_pre3h|H+X|mlp_focal|none` | 0.5153 | 0.5748 | 0.5917 |
| `edge_pre3h|H+X|mlp|none` | 0.5075 | 0.5729 | 0.5847 |
| `edge_pre3h|H+TF|mlp_focal|none` | 0.4931 | 0.6419 | 0.6461 |
| `edge_pre3h|H+TF|mlp|none` | 0.4895 | 0.6360 | 0.6328 |
| `edge_post128|H+X+TF|mlp_focal|none` | 0.4671 | 0.6485 | 0.5934 |
| `edge_post128|H+X+TF|mlp|none` | 0.4645 | 0.6471 | 0.5711 |
| `edge_pre3h|H|mlp_focal|none` | 0.4619 | 0.5438 | 0.5376 |
| `edge_pre3h|H|mlp|none` | 0.4504 | 0.5289 | 0.5222 |

## Random-40 / Random-60 (diagnostic)

### random_40 (target F1=0.581)

StratifiedShuffleSplit over ALL transactions: 40% → train+val pool, remainder → test; inner 75/25 train/val on the pool (seed+1). Same construction as txn-node random-40 diagnostic.
- **Selected by val AUPRC:** `edge_pre3h|H+X+TF|mlp|none` mean val AUPRC=0.5145±0.0377
- Test F1@0.5 mean=0.4907±0.0144 (exceeds target: False)
- Test F1@val-thr mean=0.5468±0.0070 (exceeds target: False)

### random_60 (target F1=0.658)

StratifiedShuffleSplit over ALL transactions: 60% → train+val pool, remainder → test; inner 75/25 train/val on the pool (seed+1). Same construction as txn-node random-40 diagnostic.
- **Selected by val AUPRC:** `edge_pre3h|H+X+TF|mlp|none` mean val AUPRC=0.5464±0.0087
- Test F1@0.5 mean=0.5265±0.0130 (exceeds target: False)
- Test F1@val-thr mean=0.5628±0.0120 (exceeds target: False)

## Recommendation

Fine-tune the D+ edge-centric encoder (corrected reverse + preserve_seed) with a light supervised head on the winning feature stack `H+X+TF` under temporal val AUPRC early-stopping; do not change reverse/preserve semantics.

Rationale: Best temporal validation AUPRC came from edge-centric D+ frozen stack.

## Confirmation

- No GNN training in this job.
- No automatic fine-tune submissions.

