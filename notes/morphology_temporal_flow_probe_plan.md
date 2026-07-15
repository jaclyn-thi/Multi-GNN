# Temporal / flow morphology probe plan (downstream-only)

Design audit for a **small, leakage-safe** downstream feature group to complement `pre_embedding_3h + raw`. **No implementation** in this pass — definitions and ablation protocol only.

Primary baseline to beat: **`pre-3h + raw`** (paired probe), not embedding-only.

---

## Proposed feature group: `temporal_flow_causal` (5 features)

Selected from existing causal machinery in `transaction_knn/richer_features.py` — **not** the full 20+ KNN column set. Chosen for interpretability, past-only semantics, and low redundancy with train-static `degree_fan` / `flow_balance`.

| # | Feature | Definition | Side | Window | No-history default | Normalization | Complexity | Redundancy | Why complements pre-3h |
|---|---------|------------|------|--------|-------------------|---------------|------------|------------|-------------------------|
| 1 | `log1p_sender_interarrival` | log1p(t − t_prev) where t_prev is sender’s last tx timestamp **before** this edge | sender | all past | 0 | raw log1p | O(1) update | ≠ degree (timing not count) | SSL embedding may miss burst pacing |
| 2 | `log1p_receiver_interarrival` | same for receiver | receiver | all past | 0 | raw log1p | O(1) | ≠ degree | captures inbound activity rhythm |
| 3 | `log1p_sender_past_7d_count` | count of sender txs with t′ ∈ (t − W, t) | sender | W = 7 days (604800 s) | 0 | log1p(count) | O(k) per edge* | partial overlap w/ degree | **local** activity vs lifetime degree |
| 4 | `log1p_amount_vs_sender_past_mean` | log1p(a / (mean past sender amount + ε)) | sender | all past | 0 | log1p ratio | O(1) | extends raw amount | relative amount anomaly |
| 5 | `pair_repeat_indicator` | 1 if ∃ prior edge (same sender→receiver) with t′ < t; else 0 | pair | all past | 0 | binary | O(1) | ≠ triangles | repeat counterparty pattern |

\*Fixed-window counts require a deque per account or second pass — still O(n log n) sort + O(n) sweep with bucketed window index.

### Rejected candidates (this round)

| Candidate | Reason rejected |
|-----------|-----------------|
| Recent volume sums (7d amount) | High redundancy with `flow_balance` train-static totals |
| Counterparty diversity | Correlates with degree; expensive set maintenance |
| Burstiness (Fano factor) | Unstable on sparse histories; harder to normalize |
| Global `timestamp_norm` | Already in raw/temporal_behavior; not causal |
| Betweenness / triangles | Already inventoried; redundant or expensive |

### Leakage analysis

- All features computed in **global timestamp sort** with state updated **after** reading each edge (same pattern as `compute_causal_edge_stats`).
- **No** use of val/test labels.
- **No** future transactions in windows (strictly t′ < t for past; window (t−W, t) excludes t).
- Probe scaler fit on **train split rows only** (match `probe_feature_ablation.py` policy).
- Pair with `pre_embedding_3h` requires **same edge_id inner-join** as existing pre/post diagnostics.

### Prototype validation

Existing `compute_causal_edge_stats()` already implements features 1, 2, 4, and pair history — prototype = **column subset**, no new math required. Optional toy test: `tests/test_temporal_flow_causal_features.py` (future) on 5-node synthetic timeline.

---

## Task C — Downstream ablation plan

### Comparison arms (paired, same checkpoints)

| arm | feature stack |
|-----|----------------|
| A (baseline) | `pre_embedding_3h` |
| B | `pre_embedding_3h + raw` |
| C | `pre_embedding_3h + temporal_flow_causal` |
| D | `pre_embedding_3h + raw + temporal_flow_causal` |

Optional appendix: post-128 mirrors for HI only.

### Checkpoints / datasets (priority order)

| priority | checkpoint | dataset | seed | rationale |
|----------|------------|---------|------|-----------|
| 1 | `gin_emlps_tds_asym_proj_8192neg_queue0_40ep_seed2` | Small-HI | 2 | Strong-run AUPRC/F1 champion |
| 2 | `small_li_gin_emlps_tds_asym_proj_8192neg_queue0_20ep_seed{1,2,3}` | Small-LI | 1–3 | Multiseed pre-3h replication |
| 3 | `same_pair_fnf_emlps_tds_asym_proj_8192neg_queue0_20ep` | Small-HI | 1 | Best full-stack reference |

Extract dirs: existing `embeddings/{run}/pre_embedding_3h/` — **no SSL retraining**.

### Metrics

- **Primary:** ΔAUPRC and ΔF1 vs arm B on **test** (same paired rows).
- **Secondary:** AUROC, P@100, R@100, lift@100 (alert-budget).
- **LI reporting:** mean ± **sample SD** (ddof=1) across seeds where applicable.

### Protocol

- Probe: sklearn LogisticRegression lbfgs, `cw=model`, C=1.0, val max-F1 threshold.
- **Paired rows:** inner-join on `edge_id` across all arms (identical to `pre3h_strong_run_comparison.json`).
- Scaling: StandardScaler per non-embedding group, fit train only; embeddings unscaled.
- Cache metadata: extend `results/cache/probe_features_{sweep}/meta.json` with `feature_groups`, `causal_window_sec=604800`, `causal_impl=richer_features_v1_subset`.

### Compute (estimate)

| stage | CPU | GPU |
|-------|-----|-----|
| Causal feature compute (400k–3M train edges × 4 arms) | 1–4 h total | 0 |
| Logistic probe + metrics | <30 min | 0 |
| **No SSL training** | — | — |

### Stopping rule

Stop after **priority 1–2** if:

- Arm D does **not** beat arm B on AUPRC on **≥2/3 LI seeds** **and** HI seed2 paired test, **or**
- Arm C shows ≤ **+0.005 AUPRC** vs B everywhere (noise floor).

Proceed to FNF / post-128 only if D wins on AUPRC or alert-budget on HI **without** F1 regression > 0.01 vs B.

### Future scripts (do not submit)

```bash
# Feature compute + probe (to be implemented)
python scripts/probe_temporal_flow_ablation.py \
  --data Small-HI \
  --run-name gin_emlps_tds_asym_proj_8192neg_queue0_40ep_seed2 \
  --representation-source pre_embedding_3h \
  --feature-modes embedding+temporal_flow,embedding+raw+temporal_flow \
  --paired-edge-join \
  --out-json results/diagnostics/temporal_flow_ablation_small_hi_40ep_s2.json

python scripts/probe_temporal_flow_ablation.py \
  --data Small-LI \
  --run-name small_li_gin_emlps_tds_asym_proj_8192neg_queue0_20ep_seed1 \
  --seeds 1,2,3 \
  ...
```

Slurm template (prepare only): `slurm/probe_temporal_flow_ablation_small_hi_40ep_s2.sh` — **not created/submitted in this audit**.

---

## Recommended temporal/flow group (Task G preview)

**Ship `temporal_flow_causal` (5 features above)** as a downstream-only probe group — reuse `compute_causal_edge_stats` subset rather than new SSL expert work.
