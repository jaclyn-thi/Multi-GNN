# Development results — historical archive

> **Historical ablations (pre–Jul 2026 protocol).** These runs do **not** all use the current comparison protocol. For current recommendations and thesis-safe claims, see [`results.md`](results.md) and [`current_protocol_recent_runs_summary.md`](current_protocol_recent_runs_summary.md). Kept here for provenance and negative-result history.

Sections: [Queue and negative ablations](#queue-and-negative-ablations-small-hi) · [Label-efficiency](#label-efficiency-small-hi) · [PaySim transfer](#paysim-transfer-external-fraud).

---

## Queue and negative ablations (Small-HI)

Frozen linear probe after asym contrastive + projection, 8192 negatives, `bs=8192`, `accum=4`, 20 epochs.

| Queue size | Test AUROC | Test F1 | Precision | Recall |
|------------|------------|---------|-----------|--------|
| `0` | **0.951** | **0.233** | **0.209** | 0.263 |
| `8192` | 0.923 | 0.167 | 0.110 | **0.348** |
| `16384` | 0.930 | 0.165 | 0.117 | 0.282 |
| `32768` | 0.929 | 0.179 | 0.123 | 0.330 |
| `65536` | 0.922 | 0.143 | 0.102 | 0.238 |
| `131072` | 0.915 | 0.148 | 0.119 | 0.194 |

Interpretation: the queue increases contrastive difficulty but does not improve downstream probe quality in this setup. Prefer `--contrastive_memory_bank_size 0` for the current asym + projection recipe unless explicitly testing queue behavior.

### No-queue seed averages

| Mode | Mean AUROC | Mean F1 | Mean precision | Mean recall | Note |
|------|------------|---------|----------------|-------------|------|
| no filter | **0.946** | 0.216 | 0.168 | 0.319 | AUROC baseline |
| `same_pair` | 0.942 | **0.236** | **0.176** | **0.359** | leading filter candidate |
| `same_receiver` | 0.944 | 0.193 | 0.147 | 0.349 | unstable across seeds |
| `same_endpoint` | 0.938 | 0.240 | 0.203 | 0.297 | mixed replication; seed 1 was high |

`same_pair` is the cleanest false-negative filter on the **legacy** stack (no emlps/tds): improves mean F1/recall over no-filter while preserving most AUROC. On the **current protocol** (`--emlps --tds`), FNF does **not** beat plain emlps+tds on embedding-only — see [baseline comparisons](results.md#current-protocol-emlps--tds--interventions-jun-26). `same_receiver` produced one strong AUROC run but was unstable; `same_endpoint` improved precision/F1 on average but did not cleanly replicate the seed-1 jump.

### Larger negative / queue scouts with filtering

| Run | Test AUROC | Test F1 | Precision | Recall | Interpretation |
|-----|------------|---------|-----------|--------|----------------|
| `same_pair`, `queue=32768` | 0.935 | 0.185 | 0.142 | 0.266 | queue still hurts |
| `same_receiver`, `queue=32768` | 0.916 | 0.108 | 0.066 | 0.284 | queue hurts strongly |
| `same_pair`, `10240` negs, `queue=0` | 0.938 | 0.156 | 0.105 | 0.309 | more negatives hurt |
| `same_receiver`, `10240` negs, `queue=0` | 0.929 | 0.213 | 0.192 | 0.239 | no clear benefit |

The direct `12288` run at `bs=8192 accum=4` OOMed even with `queue=0`; reducing to `bs=4096 accum=8` made it fit but did not improve metrics. Current evidence favors `8192` negatives, `queue=0`, and optional `same_pair` filtering.

### Temperature sweep

All runs use asym contrastive + projection, 8192 negatives, `queue=0`,
`bs=8192`, `accum=4`, seed 1.

| Temperature | Test AUROC | Test F1 | Precision | Recall | Interpretation |
|-------------|------------|---------|-----------|--------|----------------|
| `0.05` | 0.898 | 0.103 | 0.070 | 0.189 | too sharp |
| `0.10` | 0.933 | 0.139 | 0.092 | 0.289 | below baseline |
| `0.20` | 0.942 | 0.198 | 0.149 | 0.299 | closest lower-temp run |
| `0.50` | **0.951** | **0.233** | **0.209** | 0.263 | current default/best |

Lower temperatures improved neither AUROC nor F1. Retain
`--contrastive_temperature 0.5` for the current best recipe.

### Symmetric memory rescue

Symmetric + projection with `8192` negatives, `queue=0`, `bs=4096`, and
`accum=8` fit without OOM, but underperformed the asym baseline
(0.931 AUROC / 0.161 F1). This is a viable fit recipe, not a current winner.

### Morphology target-group scouts (Small-HI)

Targeted morphology runs keep the current **asym contrastive + projection**
backbone (8192 negs, `queue=0`, `bs=8192 accum=4`, temp 0.5) and restrict the
shared expert head with `--morph_target_groups`. Default training uses
`--checkpoint_policy best` on morphology val loss (not AML probe). Artifacts:
`embeddings/{unique_name}/probe_results.json` (morph-val best checkpoint).

**No-morph references:** no filter **0.951 / 0.233** · `same_pair` filter
**~0.949 / 0.255** (best seed).

**Last-epoch reprobe:** CPU jobs copy `checkpoint_{run}_last.tar` and extract
under `{run}_lastckpt_probe` (`slurm/run_morph_lastckpt_extract_probe.sh`).
Summary: `python scripts/summarize_morph_ckpt_probe_comparison.py`.

#### Semantic groups @ w=0.05 (Jun 22, morph-val best checkpoint)

Slurm: `slurm/ablation_morph_degree_fan_only_asym_proj_20ep.sh`,
`slurm/ablation_morph_semantic_group_scout_10ep.sh`.

| Run | Groups | Ep | Weight | Test AUROC | Test F1 | Prec | Recall | Note |
|-----|--------|----|--------|------------|---------|------|--------|------|
| `motif_participation` only | 3 triangle targets | 9 | 0.05 | **0.937** | **0.190** | **0.173** | 0.211 | best AUROC among semantic scouts |
| `flow_balance` only | 10 amount targets | 9 | 0.05 | 0.930 | 0.157 | 0.109 | 0.279 | viable; recall-leaning |
| `degree_fan` only | degrees + global lift | 11 | 1.0 | 0.929 | 0.149 | 0.097 | 0.325 | 20 ep; F1 below 10 ep scout |
| `degree_fan` only | degrees + global lift | 1 | 0.05 | 0.902 | 0.090 | 0.055 | 0.238 | morph-val best ep 1; see last-epoch row |

#### w=1.0 scouts @ 10 ep (Jun 22–23)

| Run | Ckpt | Ep | Test AUROC | Test F1 | Note |
|-----|------|----|------------|---------|------|
| `motif_participation` | morph-val best | 3 | 0.894 | 0.044 | misleading; use last epoch |
| `motif_participation` | **last** | 10 | 0.925 | 0.128 | +0.084 F1 vs best; still below w=0.05 |
| `flow_balance` | best = last | 10 | 0.929 | 0.183 | same weights at final epoch |
| `flow_balance` | last reprobe | 10 | 0.929 | 0.183 | confirms best = last |

**Prefer w=0.05** for both groups. **Motif @ w=1.0:** morph-val best is unusable;
last epoch is recoverable but still trails **motif @ w=0.05 last** (0.937 / 0.198).
**Flow @ w=1.0:** viable when best equals final epoch; similar to w=0.05.

#### Morph-val best vs last epoch (Jun 22–23 reprobe)

`--checkpoint_policy best` minimizes morphology expert val MSE, not AML probe
quality. Low `morph_expert_weight` can lock to early epochs when morph val is
low but the encoder is under-trained.

| Run | Morph-val best (ep) | AUROC / F1 | Last epoch (ep) | AUROC / F1 | Δ F1 |
|-----|---------------------|------------|-----------------|------------|------|
| `degree_fan` 20 ep, w=0.05 | ep 1 | 0.902 / 0.090 | ep 20 | **0.943 / 0.199** | **+0.110** |
| `degree_fan` 20 ep, w=1.0 | ep 11 | **0.929** / 0.149 | ep 20 | 0.915 / 0.150 | +0.001 |
| `motif_participation` 10 ep, w=0.05 | ep 9 | 0.937 / 0.190 | ep 10 | 0.937 / **0.198** | +0.008 |
| `motif_participation` 10 ep, w=1.0 | ep 3 | 0.894 / 0.044 | ep 10 | **0.925 / 0.128** | **+0.084** |
| `flow_balance` 10 ep, w=0.05 | ep 9 | 0.930 / 0.157 | ep 10 | **0.932 / 0.174** | +0.017 |
| `flow_balance` 10 ep, w=1.0 | ep 10 | 0.929 / 0.183 | ep 10 | 0.929 / 0.183 | ~0 |

Last-epoch probes: `embeddings/{run}_lastckpt_probe/probe_results_lastckpt.json`.

**Takeaways:**

1. **Checkpoint policy matters** — for w=0.05 scouts, last epoch is as good or
   better on downstream F1; `degree_fan` 20 ep w=0.05 morph-val best (ep 1)
   is misleading (+0.11 F1 at last epoch).
2. **`motif_participation` @ w=0.05** — strongest semantic-group scout
   (0.937–0.938 AUROC, ~0.19–0.20 F1). **Motif @ w=1.0** morph-val best fails
   (ep 3); last epoch (0.925 / 0.128) is better but still below w=0.05.
3. **`flow_balance` @ w=0.05** — learnable; last epoch 0.932 / 0.174. **Flow @
   w=1.0** matches final epoch (0.929 / 0.183); similar to w=0.05.
4. **Best morphology F1** remains **`degree_fan` 10 ep, w=1.0** (0.208). Best
   morphology **AUROC** in targeted scouts is **`degree_fan` 20 ep w=0.05 last**
   (0.943) — still below no-morph baseline (0.951).
5. **All targeted morphology runs remain below** no-morph baseline and
   `same_pair` filtering on headline metrics.

#### Coarse grouping (Jun 2026, 10 ep, w=1.0)

Earlier scouts used legacy group names before the semantic registry split:

| Target groups | Test AUROC | Test F1 | Precision | Recall | Note |
|---------------|------------|---------|-----------|--------|------|
| `degree_fan` | 0.921 | **0.208** | 0.151 | 0.333 | **best morphology F1** |
| `local_motif` (legacy) | 0.909 | 0.105 | 0.067 | 0.243 | weak coarse bucket |
| `degree_fan,local_motif` | 0.922 | 0.049 | 0.027 | 0.253 | AUROC ok; F1 collapse |

#### Group loss diagnostics

`morph_group_diag_full_10ep` verified per-group MSE logging on the full M1b
expert (0.934 AUROC / 0.079 F1 — logging check only). Final group MSEs were
lowest for temporal/other and highest for the legacy `local_motif` bucket.

Group registry and flags: [`morphology-reference.md`](morphology-reference.md).

### Feature-KNN (Small-HI)

Offline sparse caches from `scripts/precompute_transaction_knn.py` support
**negative exclusion** (`--enable_knn_negative_filter`) or **soft positives**
(`--enable_knn_soft_positives`). How-to: [`knn-precompute-reference.md`](knn-precompute-reference.md).

#### Precompute

| Stage | Status | Note |
|-------|--------|------|
| CPU sklearn (full train) | too slow | ~70h estimate; 6h Slurm insufficient |
| GPU smoke 100k rows | **done** | `torch_gpu`, ~21s |
| GPU full train k=15 | **done** | ~8m; 3,248,921 rows; 0 self-neighbors; mean sim **0.991** |

Full cache: `morphology_cache/Small-HI/transaction_knn_edge_native_degree_fan_k15.npz`

#### Training ablation (Jun 23)

Same backbone as no-filter baseline: asym + projection, 8192 negs, `queue=0`,
`bs=8192 accum=4`, temp 0.5, seed 1, 20 ep. Slurm:
`slurm/ablation_knn_filter_k5_asym_proj_8192neg_queue0_20ep.sh`,
`slurm/ablation_knn_filter_k15_asym_proj_8192neg_queue0_20ep.sh`.

| Run | `--knn_filter_k` | Test AUROC | Test F1 | Prec | Recall |
|-----|------------------|------------|---------|------|--------|
| No filter (baseline) | — | **0.951** | **0.233** | 0.209 | 0.263 |
| KNN exclusion | 5 | 0.947 | 0.209 | 0.171 | 0.268 |
| KNN exclusion | 15 | 0.928 | 0.176 | 0.132 | 0.264 |

**Why it barely bites:** with 8192 negatives sampled from ~3.25M train edges, only
**~1%** (k=5) or **~3%** (k=15) of anchors had any cached neighbor in the sampled
pool per epoch; `fallback_rows=0`. Exclusion removes at most ~0.0005% of candidate
slots — too sparse to help, and may drop useful hard negatives when it fires.

**Exclusion takeaway:** do **not** adopt random-sampling + feature-KNN exclusion for the
current best recipe. Prefer `--false_neg_filter_mode same_pair` for F1/recall.
Revisit exclusion only with structured negative pools where overlap is guaranteed.

#### KNN soft positives (Jun 24)

GCPAL-style **low-weight KNN positives** in the InfoNCE numerator (identity stays
weight `1.0`). Positives are injected via an auxiliary seed forward pass — not
batch-overlap-only. Slurm: `slurm/ablation_knn_softpos_m1_w0025_asym_proj_8192neg_queue0_20ep.sh`
(`m=1`, `w=0.025`, job 16426890). `m=3` script exists but was **not** run.

| Run | Test AUROC | Test F1 | Prec | Recall | Note |
|-----|------------|---------|------|--------|------|
| No filter (baseline) | **0.951** | **0.233** | 0.209 | 0.263 | — |
| Soft-pos `m=1`, `w=0.025` | 0.849 | 0.067 | 0.041 | 0.182 | best ep 19 |

Training logs showed **~99% mean similarity** to injected neighbors (saturated
`edge_native+degree_fan` cache) and ~50% slower epochs vs no-KNN runs. Loss fell
to ~1.2 (vs ~7.0 on comparable runs) without better probe geometry.

**Soft-positive takeaway:** do **not** use feature-KNN soft positives with the
current cache. Neighbors are near-duplicates in raw feature space and pull
embeddings toward superficial transaction similarity. Pause this path unless
neighbor design changes (sparser/diverse positives, much lower weight, or
structure-based similarity — not raw feature KNN). Audit:
`python scripts/audit_transaction_knn_cache.py --data Small-HI --max_rows 50000`.

### Edge-drop augmentations (Small-HI)

Label-free, train-split-only edge-drop policies for contrastive views (default
`--edge_drop_policy random` unchanged). Precompute:
`scripts/precompute_edge_drop_scores.py` · audit:
`scripts/audit_edge_drop_scores.py` · flags: [`cli-reference.md`](cli-reference.md).

#### Training ablation (Jun 24)

Same backbone as no-filter baseline: asym + projection, 8192 negs, `queue=0`,
`bs=8192 accum=4`, temp 0.5, seed 1, 20 ep, `--edge_drop_target_rate 0.1`,
`--edge_drop_importance_alpha 2.0`. Slurm:
`slurm/ablation_degree_aware_edgedrop_asym_proj_8192neg_queue0_20ep.sh`,
`slurm/ablation_degree_flow_aware_edgedrop_asym_proj_8192neg_queue0_20ep.sh`
(jobs 16455755 / 16455756).

| Run | Best ep | Test AUROC | Test F1 | Prec | Recall | Note |
|-----|---------|------------|---------|------|--------|------|
| Random drop (baseline) | 19 | **0.951** | 0.233 | 0.209 | 0.263 | — |
| **`degree_aware`** | 18 | 0.949 | **0.237** | 0.198 | **0.297** | +F1/recall, −AUROC |
| **`degree_flow_aware`** | 20 | 0.947 | 0.234 | 0.207 | 0.269 | ≈ baseline |
| FNF `same_pair` seed 1 (ref) | — | 0.946 | 0.240 | 0.175 | 0.383 | stronger F1/recall tradeoff |

**Policy telemetry (epoch 1 buckets, stable across training):** global cache mean
drop prob **≈ 0.10** as intended. Per-batch realized drops were nonuniform:

- **`degree_aware`:** low-degree p0–20 **~1%** drop; high-degree p80–100 **~32%** drop.
- **`degree_flow_aware`:** high-amount p80–100 **~7%** drop (preserved); low-amount
  p0–20 **~23%** drop.

Training loss at best (**~7.06**) matched random-drop / KNN-filter runs. Wall clock
**~1.5 h** per job (incl. precompute + extract + probe).

**Takeaway:** do **not** replace the AUROC baseline — neither policy beats **0.951** on the **legacy** stack (`emlps=false`, `tds=false`).
**`degree_aware`** gave a small **F1 +0.004** and **recall +0.034** with **AUROC −0.002**. **`degree_flow_aware`**
did not improve on degree-only. Combining **`degree_aware`** with **`same_pair` FNF**
was tested and **failed** — see [below](#degree_aware-edge-drop--same_pair-fnf-small-hi).

**Current protocol (Jun 26):** with **`--emlps --tds`**, degree-aware edge drop **does not help**
(0.926 AUROC / 0.152 AUPRC / 0.240 F1 vs emlps+tds 0.944 / 0.213 / 0.259) — see
[current protocol](results.md#current-protocol-emlps--tds--interventions-jun-26). Legacy degree-aware numbers above
are still valid for the old graph stack only.
Per-view realized drop in epoch logs is **~0.16** on sampled batches (fixed Jun 24;
old headline `realized_v1≈0.58` was a logging bug). Bucket logs above are authoritative.

#### `degree_aware` edge drop + `same_pair` FNF (Small-HI)

Stack test: nonuniform **view corruption** (`degree_aware`) plus **negative-pool
hygiene** (`--false_neg_filter_mode same_pair`). Same backbone otherwise: asym +
projection, 8192 negs, `queue=0`, `bs=8192 accum=4`, temp 0.5, 20 ep,
`--edge_drop_target_rate 0.1`, `--edge_drop_importance_alpha 2.0`. Slurm:
`slurm/ablation_degree_aware_edgedrop_samepair_fnf_asym_proj_8192neg_queue0_20ep.sh`
(seed 1),
`slurm/ablation_degree_aware_edgedrop_samepair_fnf_seed2_asym_proj_8192neg_queue0_20ep.sh`
(seed 2).

| Run | Best ep | Test AUROC | Test F1 | Prec | Recall | Note |
|-----|---------|------------|---------|------|--------|------|
| Random drop (baseline, seed 1) | 19 | **0.951** | 0.233 | **0.209** | 0.263 | AUROC ref |
| **`degree_aware`** only (seed 1) | 18 | 0.949 | **0.237** | 0.198 | 0.297 | parent |
| **`same_pair` FNF** only (seed 1) | — | 0.946 | 0.240 | 0.175 | **0.383** | parent |
| **`same_pair` FNF** only (seed 2) | — | 0.949 | **0.255** | 0.189 | **0.394** | parent |
| **Combined** (seed 1) | 20 | 0.933 | 0.119 | 0.079 | 0.235 | large val→test F1 gap |
| **Combined** (seed 2) | 16 | 0.945 | 0.217 | 0.172 | 0.294 | below both parents |
| **Combined mean** (seeds 1–2) | — | 0.939 | 0.168 | 0.126 | 0.265 | — |

Training telemetry looked normal: contrastive loss **~7.056** at best (same as
degree-aware-only), FNF removal **~0.01%** with `fallback_rows=0`, degree buckets
unchanged (p0–20 **~1%** drop, p80–100 **~32%**). Problem is downstream embedding
geometry, not misconfigured flags.

**Stack takeaway:** interventions **do not combine**. Seed 1 collapsed (precision
**0.079**); seed 2 was more stable but still below both parents on every metric.
Use **`degree_aware`** or **`same_pair` FNF** alone — not both. High seed variance
on the combined recipe; no further combined seeds unless the design changes (e.g.
weaker drop policy or softer FNF).

### Multi-positive InfoNCE scouts

Multi-positive runs use weak endpoint/pair positives in the InfoNCE numerator;
identity positives remain weight `1.0`.

| Run | Test AUROC | Test F1 | Precision | Recall | Interpretation |
|-----|------------|---------|-----------|--------|----------------|
| `same_pair`, weight `0.1`, mean seeds 1–3 | 0.938 | 0.153 | 0.114 | 0.236 | weak |
| `same_pair`, weight `0.02`, seed 1 | 0.943 | 0.108 | 0.066 | 0.292 | too weak/noisy |
| `same_pair`, weight `0.05`, seed 1 | 0.944 | 0.224 | 0.183 | 0.287 | improved, still below filter |
| `same_receiver`, weight `0.1`, mean seeds 1–3 | 0.940 | 0.203 | 0.156 | 0.298 | below no-filter |

Endpoint/pair relationships appear more useful as **negative-exclusion rules**
than as weak positives in the numerator. If revisiting multi-positive InfoNCE,
start from a smaller and more targeted design rather than combining it with
queues or larger negative counts.

### Masked edge-attribute reconstruction (Small-HI)

GraphMAE-style label-free pretraining: mask transaction attributes on train seed
edges, reconstruct from GNN edge embeddings (`--objective masked_edge`). No
contrastive InfoNCE, no morphology, no KNN/FNF. Flags:
[`cli-reference.md`](cli-reference.md).

**Initial run:** `slurm/ablation_masked_edge_attr_gine_20ep.sh` (job 16481056) →
`hi_masked_edge_attr_gine_20ep_bestckpt`.

**Follow-ups (Jun 24):** seed-2 replicate
(`slurm/ablation_masked_edge_attr_gine_20ep_seed2.sh`, job 16490926),
40 ep seed-1 (`slurm/ablation_masked_edge_attr_gine_40ep_seed1.sh`, job 16490992),
CPU `embedding+raw` probe on bestckpt embeddings (job 16490482).

**Config (all GPU runs):** GINe hetero (`--reverse_mp --ego --ports`), `bs=8192`,
`num_neighs 100 100`, `--mask_edge_attr_rate 0.15`, fields
`amount,currency,payment_format`, zero mask tokens, decoder hidden 128,
`--checkpoint_policy best` (lowest train recon loss).

#### Primary + follow-up runs (embedding-only probe)

| Run | Best ep | Train recon loss | Test AUROC | Test F1 | Prec | Recall | F1@0.5 |
|-----|---------|------------------|------------|---------|------|--------|--------|
| **`hi_masked_edge_attr_gine_20ep_bestckpt` (seed 1)** | 20 | 0.032 | 0.932 | **0.241** | 0.180 | **0.363** | **0.257** |
| `hi_masked_edge_attr_gine_20ep_seed2` | 14 | 0.031 | 0.943 | 0.131 | 0.092 | 0.227 | 0.105 |
| `hi_masked_edge_attr_gine_40ep_seed1` | 29 | 0.029 | 0.925 | 0.166 | 0.114 | 0.303 | 0.159 |
| Contrastive + proj baseline | 19 | ~7.06 | **0.951** | 0.233 | **0.209** | 0.263 | 0.213 |
| Contrastive + masked-edge aux `w=0.1` | 20 | ~7.06 | 0.951 | 0.239 | 0.197 | 0.302 | 0.221 |

Seed-1 20 ep: recon loss 0.161 → 0.032; val AUROC **0.944**, val F1 **0.218**
(threshold 0.433). Seed-2: healthy training but val F1 **0.109**, threshold **0.863**
(poor val→test calibration). 40 ep: recon loss still improving at ep 29 but probe
metrics regress; ep 33 train-loss spike 0.047.

#### `embedding+raw` probe on masked-edge embeddings (CPU)

Same protocol as [probe feature ablation](#probe-feature-ablation-small-hi) on
`hi_masked_edge_attr_gine_20ep_bestckpt` (job 16490482). Detail:
[`probe_feature_ablation_masked_edge_embedding_raw.md`](experiments/ablation-runs/probe_feature_ablation_masked_edge_embedding_raw.md).

| Probe | Test AUROC | Test F1 | Prec | Recall | F1@0.5 |
|-------|------------|---------|------|--------|--------|
| Embedding only (seed-1 20 ep) | 0.932 | **0.241** | 0.180 | 0.363 | **0.257** |
| **`embedding+raw`** | 0.943 | **0.114** | 0.063 | 0.664 | 0.032 |

Contrastive reference on same probe script: `embedding+raw` **0.257 F1** (+2.1 pp
vs embedding-only). On masked-edge, raw attributes **hurt** — threshold jumps to
**0.917**; val F1 0.225 but test F1 collapses. Use **embedding-only** probes for
masked-edge runs.

**Takeaways:**

1. **Reference recipe:** `hi_masked_edge_attr_gine_20ep_bestckpt` (20 ep, seed 1)
   — first viable non-contrastive SSL objective; competitive F1 without InfoNCE.
2. **Trade-off vs contrastive (seed-1 20 ep):** **−1.9 pp AUROC** but **+0.8 pp F1**
   (val-tuned), **+4.4 pp F1@0.5**, **+10 pp recall**.
3. **Contrastive + aux @ 0.1 is a wash** — not worth the extra forward pass.
4. **Seed variance is high on F1** — seed 2 matches AUROC (~0.943) but F1 **0.131**
   (−11 pp vs seed 1); report ≥2 seeds if citing masked-edge F1.
5. **Do not extend to 40 ep** with recon-loss best checkpoint — lower recon loss
   (0.029) but worse probe (0.166 F1, −7.5 pp vs 20 ep seed 1).
6. **Do not use `embedding+raw`** downstream for masked-edge (unlike contrastive).
7. **Next sweeps (not longer training):** mask rate 0.30, `mean` tokens, include
   `timestamp`; optional last-epoch or probe-aligned checkpoint selection.

Artifacts:

- `embeddings/hi_masked_edge_attr_gine_20ep_bestckpt/probe_results.json`
- `embeddings/hi_masked_edge_attr_gine_20ep_seed2/probe_results.json`
- `embeddings/hi_masked_edge_attr_gine_40ep_seed1/probe_results.json`
- `embeddings/hi_contrastive_plus_masked_edge_asym_proj_8192neg_queue0_accum4_20ep_bestckpt/probe_results.json`
- `results/diagnostics/probe_feature_ablation_hi_masked_edge_attr_gine_20ep_bestckpt_embedding_raw.json`

### Probe feature ablation (Small-HI)

CPU downstream comparison on **frozen** contrastive embeddings
(`hi_contrastive_proj_asym_8192neg_queue0_accum4_20ep_bestckpt`): which feature
groups help a linear probe beyond SSL `z` alone. Script:
`scripts/probe_feature_ablation.py` · Slurm: `slurm/run_probe_feature_ablation_small_hi.sh`
(job 16479054). Detail: [`probe_feature_ablation_small_hi.md`](experiments/ablation-runs/probe_feature_ablation_small_hi.md).

Same protocol as `linear_probe.py` (val max-F1 threshold, class-weighted
logistic regression). Non-embedding groups use train-fit `StandardScaler`;
embeddings unscaled.

| Features | Dim | Test AUROC | AUPRC | Test F1 | Prec | Recall | F1@0.5 |
|----------|-----|------------|-------|---------|------|--------|--------|
| `raw` (edge_native) | 4 | 0.860 | 0.009 | 0.009 | 0.004 | 0.943 | 0.000 |
| `morph` (degree/flow/temporal) | 16 | 0.865 | 0.064 | 0.115 | 0.068 | 0.371 | 0.139 |
| **`embedding`** | 128 | **0.951** | 0.120 | 0.236 | 0.193 | 0.301 | 0.215 |
| **`embedding+raw`** | 132 | **0.953** | 0.137 | **0.257** | **0.241** | 0.274 | 0.238 |
| `embedding+raw+morph` | 148 | 0.952 | 0.153 | 0.233 | 0.204 | 0.273 | 0.231 |

**Takeaways:**

1. **SSL embeddings carry almost all signal** — raw/morph alone top out ~0.86 AUROC.
2. **Best probe in this study:** `embedding+raw` — **+2.1 pp F1** and +0.2 pp AUROC
   over embedding-only (0.257 vs 0.236).
3. **Adding morph on top of embedding+raw hurts** (F1 back to 0.233) — morphology
   features are largely redundant once the encoder has run.
4. **`embedding+raw` on masked-edge embeddings hurts** (0.114 F1 vs 0.241
   embedding-only) — see [masked edge section](#masked-edge-attribute-reconstruction-small-hi);
   contrastive +raw lift does **not** transfer.
5. **On emlps+tds embeddings, +raw and +morph both help** — see
   [Multi-GNN § emlps+tds probe ablation](#multi-gnn-graph-adaptations-small-hi)
   (0.298 F1 with `embedding+raw+morph` vs 0.257 on baseline contrastive +raw).

JSON (baseline contrastive): `results/diagnostics/probe_feature_ablation_hi_contrastive_proj_asym_8192neg_queue0_accum4_20ep_bestckpt.json`.
JSON (masked-edge +raw): `results/diagnostics/probe_feature_ablation_hi_masked_edge_attr_gine_20ep_bestckpt_embedding_raw.json`.
JSON (emlps+tds): `results/diagnostics/probe_feature_ablation_hi_contrastive_gin_emlps_tds_embedding_raw.json`.

### PNA encoder ablation (Small-HI)

Architecture swap on the best asymmetric contrastive + projection recipe: replace
**GINe** with **PNA** (`--model pna`), all else unchanged. Slurm:
`slurm/ablation_contrastive_pna_proj_asym_8192neg_queue0_accum4_20ep.sh` (job
16512658, `mit_preemptable`, 12h walltime).

**Config:** same as `hi_contrastive_proj_asym_8192neg_queue0_accum4_20ep_bestckpt`:
asym InfoNCE, projection head, 8192 negs, `queue=0`, `bs=8192 accum=4`, 20 ep,
seed 1, random edge-drop default, no morph/KNN/FNF/masked-edge aux.

**Hyperparameter caveat:** `extract_param()` loads **model-specific**
`model_settings.json` entries — this is **not** a capacity-matched architecture
swap:

| | GIN baseline | PNA run |
|--|--------------|---------|
| `n_hidden` | 66 | **20** |
| LR | 0.0062 | 0.00061 |
| `final_dropout` | 0.10 | 0.29 |
| Params | ~112k | ~71k |

#### Training

| | GIN | PNA |
|--|-----|-----|
| Wall time (20 ep) | ~1.5 h | **~2.7 h** (~7 min/ep) |
| InfoNCE loss (ep 1 → 20) | 7.31 → 7.06 | 7.27 → 7.07 |
| Best checkpoint | ep 19 | ep 20 |
| CUDA OOM at `bs=8192` | no | **no** |

#### Downstream probe (test)

| Metric | GIN baseline | PNA | Δ |
|--------|--------------|-----|---|
| **AUROC** | **0.951** | 0.942 | −0.9 pp |
| **F1** (val-tuned) | **0.233** | 0.186 | −4.7 pp |
| **F1@0.5** | **0.213** | 0.112 | −10.1 pp |
| Precision | 0.209 | 0.138 | −7.1 pp |
| Recall | 0.263 | **0.288** | +2.5 pp |
| Val F1 | **0.233** | 0.108 | −12.5 pp |
| Val AUROC | **0.959** | 0.942 | −1.7 pp |

PNA val→test calibration is weak (val F1 0.108, threshold 0.22 vs GIN 0.38 /
val F1 0.233). Does not beat masked-edge seed-1 20 ep on F1 (0.241).

**Takeaways:**

1. **PNA does not improve** the current best contrastive recipe on Small-HI.
2. **Keep GINe** as the default encoder for contrastive SSL in this stack.
3. **Confound:** PNA uses smaller `n_hidden` and different LR/dropout from
   supervised-tuned `model_settings.json` — a fairer test would match GIN
   hyperparams (`n_hidden=65`) or re-tune PNA for contrastive pretraining.
4. **Ops:** `bs=8192 accum=4` fits; PNA ~2× slower per epoch than GINe.

Artifact: `embeddings/hi_contrastive_pna_proj_asym_8192neg_queue0_accum4_20ep/probe_results.json`

### Multi-GNN graph adaptations (Small-HI)

Baseline contrastive runs use **`--reverse_mp --ego --ports`** but not
**`--emlps`** (edge-attribute MLP updates) or **`--tds`** (time-delta edge
features). Ablation on the best asym + projection + 8192 negs + `queue=0`
recipe (`bs=8192 accum=4`, 20 ep).

Slurm:
`slurm/ablation_contrastive_gin_emlps_proj_asym_8192neg_queue0_accum4_20ep.sh`,
`slurm/ablation_contrastive_gin_emlps_tds_proj_asym_8192neg_queue0_accum4_20ep.sh`,
`slurm/ablation_contrastive_gin_emlps_tds_seed2_proj_asym_8192neg_queue0_accum4_20ep.sh`,
`slurm/ablation_contrastive_gin_tds_proj_asym_8192neg_queue0_accum4_20ep.sh`.

Reference baseline: `hi_contrastive_proj_asym_8192neg_queue0_accum4_20ep_bestckpt`
(`emlps=false`, `tds=false`).

#### Downstream probe (test, val-tuned threshold)

| Run | Graph flags | Seed | Best ep | Thr | AUROC | AUPRC | F1 | Prec | Recall | F1@0.5 |
|-----|-------------|------|---------|-----|-------|-------|-----|------|--------|--------|
| **Baseline** | `ports` | 1 | 19 | 0.333 | **0.951** | 0.120 | 0.236 | **0.209** | 0.263 | 0.215 |
| **`--emlps` only** | `+emlps` | 1 | 20 | 0.357 | 0.915 | 0.093 | 0.186 | 0.189 | 0.182 | 0.170 |
| **`--tds` only** | `+tds` | 1 | 20 | 0.353 | 0.940 | 0.141 | 0.233 | 0.163 | **0.399** | 0.242 |
| **`--emlps --tds` (seed 1)** | `+emlps +tds` | 1 | 19 | 0.528 | 0.944 | **0.213** | **0.259** | 0.231 | 0.295 | **0.257** |
| **`--emlps --tds` (seed 2)** | `+emlps +tds` | 2 | 14 | 0.362 | 0.925 | 0.142 | 0.208 | 0.170 | 0.268 | 0.226 |
| **`--emlps --tds` (seed 3)** | `+emlps +tds` | 3 | 18 | 0.478 | 0.921 | 0.124 | 0.157 | 0.131 | 0.197 | 0.160 |
| **+ same_pair FNF** | final protocol | 1 | 19 | 0.507 | 0.942 | 0.178 | 0.241 | 0.220 | 0.266 | 0.239 |
| **+ degree_aware drop** | final protocol | 1 | 20 | 0.403 | 0.926 | 0.152 | 0.240 | 0.231 | 0.250 | 0.244 |
| **morph expert + emlps+tds (last ep)** | `degree_fan` w=0.05 | 1 | 20 | 0.541 | 0.947 | 0.184 | 0.288 | 0.282 | 0.294 | 0.283 |

Val AUROC: baseline **0.959** → emlps-only **0.960** (poor test transfer) →
tds-only **0.964** → emlps+tds seed-1 **0.963** → seed-2 **0.959**. All runs
fit at `bs=8192` without OOM.

#### Takeaways

1. **`--emlps --tds` together (seed 1)** give the best **embedding-only**
   contrastive F1 (**0.259**) and **AUPRC** (**0.213**) with modest AUROC cost
   (−0.7 pp vs baseline). Best **balanced** precision/recall among adaptation variants.
2. **`--emlps` alone hurts** — test 0.186 F1, 0.093 AUPRC, 0.915 AUROC.
3. **`--tds` alone does not explain the seed-1 lift** — test F1 matches baseline
   (0.233) with **recall-heavy** profile. Gain requires **interaction** with `--emlps`.
4. **Seed sensitivity:** seeds 1–3 mean **0.208 F1** / **0.160 AUPRC** (seed 3 worst: 0.157 / 0.124).
5. **Current-protocol embedding-only:** FNF and degree-aware do not beat emlps+tds s1 — but **FNF + full probe stack wins** (0.319 F1) — see [baseline comparisons](results.md#current-protocol-emlps--tds--interventions-jun-26).
6. **Morph expert during SSL:** morph-val best (ep 1) misleading; last epoch 0.288 F1 — below probe-time morph on frozen emlps+tds/FNF embeddings.
7. **Reference recipes:** AUROC-first → baseline (`ports` only). SSL embedding-only → **emlps+tds seed 1** (report seed variance). Best downstream → **FNF + emlps+tds + `embedding+raw+morph`**.

Artifacts:
`embeddings/hi_contrastive_gin_emlps_tds_proj_asym_8192neg_queue0_accum4_20ep/probe_results.json`,
`embeddings/hi_contrastive_gin_emlps_tds_seed2_proj_asym_8192neg_queue0_accum4_20ep/probe_results.json`,
`embeddings/emlps_tds_asym_proj_8192neg_queue0_20ep_seed3/probe_results.json`,
`embeddings/same_pair_fnf_emlps_tds_asym_proj_8192neg_queue0_20ep/probe_results.json`,
`embeddings/degree_aware_edgedrop_emlps_tds_asym_proj_8192neg_queue0_20ep/probe_results.json`,
`embeddings/morph_expert_emlps_tds_asym_proj_8192neg_queue0_20ep_lastckpt_probe/probe_results_lastckpt.json`.

#### emlps+tds probe feature ablation

CPU probe on **frozen** emlps+tds seed-1 embeddings (no retrain). Slurm:
`slurm/run_probe_feature_ablation_emlps_tds_embedding_raw.sh`,
`slurm/run_reprobe_probe_feature_ablation_auprc.sh` (AUPRC refresh, job 16615596). Detail:
[`probe_feature_ablation_hi_contrastive_gin_emlps_tds_embedding_raw.md`](experiments/ablation-runs/probe_feature_ablation_hi_contrastive_gin_emlps_tds_embedding_raw.md).

| Features | Dim | Test AUROC | AUPRC | Test F1 | Prec | Recall | F1@0.5 |
|----------|-----|------------|-------|---------|------|--------|--------|
| `embedding` | 128 | 0.944 | 0.213 | 0.259 | 0.231 | 0.295 | 0.257 |
| **`embedding+raw`** | 132 | **0.949** | 0.244 | **0.274** | 0.212 | 0.389 | 0.268 |
| **`embedding+raw+morph`** | 148 | 0.945 | **0.276** | **0.298** | 0.231 | **0.420** | **0.327** |

Compare baseline contrastive +raw (**0.257 F1**) and +raw+morph (**0.233 F1**,
morph hurts). On emlps+tds embeddings, **raw and morph both complement** SSL
`z`. **FNF + emlps+tds** full stack reaches **0.319 F1** — see
[current-protocol feature ablation](results.md#current-protocol-comparison-batch-jun-2829).

JSON (emlps+tds): `results/diagnostics/probe_feature_ablation_hi_contrastive_gin_emlps_tds_embedding_raw.json`.
JSON (FNF / degree-aware): `results/diagnostics/probe_feature_ablation_same_pair_fnf_emlps_tds.json`,
`results/diagnostics/probe_feature_ablation_degree_aware_edgedrop_emlps_tds.json`.
Comparison: `results/diagnostics/probe_feature_ablation_final_protocol_comparison.json`.

### SSL hyperparameter sweep (Small-HI)

Test whether the best contrastive recipe is limited by **supervised-tuned**
hyperparameters from `model_settings.json` (GIN: `lr≈0.0062`, `n_hidden≈66`,
`final_dropout≈0.10`, `dropout≈0.01`). Method fixed: asym InfoNCE + projection,
8192 negs, `queue=0`, `bs=8192 accum=4`, 20 ep, seed 1, random edge-drop default;
overrides via `--override_lr`, `--override_n_hidden`, `--override_final_dropout`
(no edits to `model_settings.json`).

Slurm:
`slurm/ablation_contrastive_gin_lr0p003_h66_proj_asym_8192neg_queue0_accum4_20ep.sh`,
`slurm/ablation_contrastive_gin_lr0p003_h128_proj_asym_8192neg_queue0_accum4_20ep.sh`,
`slurm/ablation_contrastive_gin_lrbase_h128_proj_asym_8192neg_queue0_accum4_20ep.sh`,
`slurm/ablation_contrastive_gin_lrbase_h66_finaldrop0_proj_asym_8192neg_queue0_accum4_20ep.sh`.

Sweep grid (LR × `n_hidden` + dropout ablation):

| Cell | LR | `n_hidden` | `final_dropout` | Status |
|------|-----|------------|-----------------|--------|
| baseline | 0.0062 | 66 | ~0.10 | reference |
| sweep | 0.003 | 66 | ~0.10 | **done** |
| sweep | 0.003 | 128 | ~0.10 | **done** |
| sweep | 0.0062 | 128 | ~0.10 | **done** |
| sweep | 0.0062 | 66 | **0.0** | **done** |
| sweep | 0.001 | 66 / 128 | ~0.10 | **cancelled** |

#### Downstream probe (test)

| Run | Best ep | Thr | AUROC | F1 | Prec | Recall | F1@0.5 |
|-----|---------|-----|-------|-----|------|--------|--------|
| **Baseline** (`lr≈0.0062`, h=66) | 19 | 0.378 | **0.951** | **0.233** | **0.209** | 0.263 | **0.213** |
| `lr=0.003`, h=66 | **20** | 0.360 | 0.932 | 0.212 | 0.163 | **0.303** | 0.204 |
| `lr=0.003`, h=128 | 19 | 0.420 | 0.940 | 0.205 | 0.158 | 0.289 | 0.204 |
| **baseline LR, h=128** | **20** | 0.404 | 0.939 | 0.153 | 0.117 | 0.223 | 0.149 |
| **baseline LR, final_dropout=0** | 18 | 0.388 | 0.945 | 0.213 | 0.165 | 0.299 | 0.215 |

Val AUROC: baseline **0.959** → finaldrop0 **0.957** → baseline LR h128 **0.958**
→ lr0.003 h66 **0.948** → lr0.003 h128 **0.952**. All h128 runs fit at
`bs=8192` without OOM.

#### Takeaways

1. **Inherited supervised hyperparams are well-matched** — baseline remains best
   on headline metrics (0.951 AUROC / 0.233 F1). No sweep cell beat it.
2. **Lower LR (0.003) hurts** ranking quality (AUROC −1.1–1.9 pp), not just
   calibration. h66 best ckpt at ep 20 suggests slower convergence but still
   below baseline at 20 ep.
3. **Larger capacity at baseline LR hurts badly** — h=128 @ baseline LR is the
   **worst F1** in the sweep (0.153 val-tuned; 0.149 @ 0.5). Val AUROC near
   baseline (0.958) but val→test F1 collapses (0.195 → 0.153) → poor
   generalization / overfitting, not threshold choice. Surprisingly, lr=0.003
   h128 (0.205 F1) generalizes better than baseline LR h128 — slower LR may act
   as implicit regularization for the wider model.
4. **`final_dropout=0` does not help** — small AUROC cost (−0.6 pp), val-tuned
   F1 −2.0 pp (0.213); F1@0.5 essentially tied (0.215 vs 0.213). Shifts toward
   higher recall / lower precision. **Keep `final_dropout≈0.10`.**
5. **Precision–recall shift** on non-baseline runs: higher recall, lower
   precision; at fixed 0.5 the gap vs baseline narrows on several cells,
   consistent with weaker separability.
6. **Decision:** **close the SSL LR/width/dropout sweep.** Keep GIN defaults
   from `model_settings.json`. For encoder gains, pivot to augmentation (e.g.
   `degree_aware`), objectives (masked-edge), or FNF — not capacity or dropout.

Artifacts:
`embeddings/hi_contrastive_gin_lr0p003_h66_proj_asym_8192neg_queue0_accum4_20ep/probe_results.json`,
`embeddings/hi_contrastive_gin_lr0p003_h128_proj_asym_8192neg_queue0_accum4_20ep/probe_results.json`,
`embeddings/hi_contrastive_gin_lrbase_h128_proj_asym_8192neg_queue0_accum4_20ep/probe_results.json`,
`embeddings/hi_contrastive_gin_lrbase_h66_finaldrop0_proj_asym_8192neg_queue0_accum4_20ep/probe_results.json`.

---


## Label-efficiency (Small-HI)

Stratified train-label fractions; threshold tuned on full val. Source: `embeddings/label_efficiency_summary.json`.

| Train labels | sym+proj | 8192neg+proj | clustering+proj | M1b+proj | Contrastive+proj |
|--------------|----------|--------------|-----------------|----------|------------------|
| 10% | **0.924** | 0.917 | 0.916 | 0.918 | 0.906 |
| 25% | **0.926** | 0.922 | **0.926** | 0.922 | 0.918 |
| 50% | 0.929 | **0.931** | 0.930 | 0.919 | 0.925 |
| 100% | 0.929 | **0.931** | 0.929 | 0.922 | 0.928 |

Full tables: [`morphology-metrics-plan.md` § Label-efficiency](morphology-metrics-plan.md).

---

## PaySim transfer (external fraud)

Frozen GIN extract + linear probe on ~6.36M edges. Canonical probe: `--class_weight model`.

| Encoder | Threshold | Test AUROC | Test F1 |
| ------- | --------- | ---------- | ------- |
| `hi_contrastive_proj_sym_20ep_bestckpt` | val-tuned | **0.866** | 0.089 |
| `hi_contrastive_proj_sym_20ep_bestckpt` | fixed 0.5 | **0.864** | **0.127** |
| `random_init_gin` | val-tuned / 0.5 | 0.730 | 0.135–0.143 |

In-domain Small-HI (same encoder): AUROC **0.929**, F1 **0.222** @ val-tuned.

Full tables and probe variants: [`downstream-eval-plan.md` § PaySim](downstream-eval-plan.md#paysim--status-jun-2026).
