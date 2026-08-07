# Morphology downstream probe — isolated infrastructure

> Status twin: [`results/diagnostics/morphology_downstream_probe_infra_status.json`](../results/diagnostics/morphology_downstream_probe_infra_status.json)  
> Package: `morphology_downstream_probe/` (new only)

Implemented while checkpoint-ladder DAG may still be running. **No shared modules edited.**

## Semantics

- TRAIN edges only → simple **undirected** graph (direction ignored, parallels collapsed, self-loops dropped).
- Node \(T[v]\) = number of neighbor pairs of \(v\) that are themselves linked.
- Edge target \(y=\log(1+T[s]+T[r])\); validation joins the **train** table only.
- Unseen val endpoints: \(T=0\) + coverage log (`count_zero_with_coverage_log`).
- If train zero-target fraction \(>0.95\): gate fails for default log-regression; declare `any_triangle` binary + positive-only log regression (not executed here).

## Probe plan (not executed)

`EXPERT_ONLY@3000` and `MIXED_3DOMAIN_LONG@3000` Small-HI R198; PaperStyleMLP 198→128→1; 20 ep / 1e-3 / 8192 / seed 2; Spearman primary; baselines mean / raw-6 / degree-fan / R198; residualize triangles vs train-fit degree model for incremental claim.

## Tests

```bash
python -m pytest tests/test_morphology_downstream_probe.py -q
```

## NOT RUN (real)

```bash
# After ladder clears + authorized CSVs:
python scripts/precompute_smallhi_train_static_triangles.py --execute-full \
  --train-csv <TRAIN_ONLY.csv> --val-csv <VAL_ONLY.csv> \
  --out-dir results/diagnostics/morphology_downstream_probe_smallhi/triangle_cache/smallhi
python scripts/run_smallhi_morphology_probe.py --execute-probe --load-embeddings  # still refused until unlocked
```
