# SAML-D separability audit (Candidate A; train/val only)

> Twin: `results/diagnostics/samld_separability_audit.json`  
> CPU job **19114674** (`mit_normal`, COMPLETED, ~8m49s / 493.4s wall in-process)  
> Script: `scripts/samld_separability_audit.py`  
> **No new job submitted** — prior complete audit reused.

## Classification: **PIPELINE_HEALTHY_GRAPH_ADDS_VALUE**

No ID/label leakage in X; permutation control near prevalence; no exclusive categories; univariate features weak. Corrected legacy-head smoke is stable (val F1 0.9044, AUPRC 0.9585) and materially beats best X-only HGB AUPRC 0.7235 (delta=0.2350). Dataset has strong nonlinear tabular signal (HGB>>logistic) but graph still adds ranking value.

Formal 50-epoch supervised training is **justified** by this evidence; **not** auto-submitted.

---

## Cohorts (locked; test never loaded)

| Split | n | positives | π | index SHA256 |
|-------|--:|----------:|--:|--------------|
| Train | 5,707,315 | 5,751 | 0.001008 | `290713933cc655e9c70984bc3cb7f575ab26a03b8078a1337cda58892054935f` |
| Val | 1,899,523 | 1,986 | 0.001046 | `b08cdb815f82e6d37019e5e6ec9c5a6fd12c3f9d523f63b2768f6e4d0a99a38c` |

`matches_protocol=true` for both. `test_inspected=false`, `test_evaluated=false`.

## Candidate A features

- Timestamp, Amount Received, Received Currency, Payment Format, in_port, out_port  
- edge_dim=6; legacy per-graph z-norm (train / train∪val), matching smoke Candidate A

## X-only val AUPRC

| Control | Val AUPRC |
|---------|----------:|
| Prevalence baseline | 0.001046 |
| Best univariate (`in_port`) | 0.0279 |
| Logistic | 0.0128 |
| PaperStyle MLP (8 ep) | 0.4045 |
| HistGradientBoosting (LGBM/XGB absent) | **0.7235** |

## Permutation sanity

- Fixed first 200k train edges; labels permuted (seed 2); val labels preserved  
- Val AUPRC = **0.00600** vs prevalence **0.00105** → near prevalence (`falls_near_prevalence=true`)

## Smoke comparison (val only; no test)

| Metric | Value |
|--------|------:|
| Smoke ep1 val AUPRC | 0.983986 |
| Smoke ep2 val AUPRC | 0.958508 |
| Selected ep2 F1 | 0.904432 |
| Best X-only AUPRC | 0.7235 |
| GNN − X-only AUPRC | **0.2350** |

## Integrity (summary)

- Account / EdgeID / label in X: **false**  
- Conflicting-label feature hashes: **0**  
- Categories all-positive / high-pos-rate (≥50%): **none**  
- Amount ranges disjoint by label: **false** (pos max ≫ neg max, but not a near-identifier alone)  
- Val accounts also in train: **78.4%** (allowed overlap)  
- Ports: built per split graph; val seeds use train∪val adjacency (protocol A), not test

## Provenance

- Job: **19114674** · host node1601 · log `logs/samld_separability_19114674.out`  
- Script SHA256: `b9a0bb6bbb85b136a8bd2c353714748bc07f27344a37ca1d2419173817ecae56`  
- Optional cells dir: not written by this run (`results/diagnostics/samld_separability_audit/` absent)
