# Development results (not frozen benchmarks)

Quick-run numbers for internal comparison while configs and code still change. **Not** a formal evaluation protocol — use fixed recipes for papers and PI updates once the stack stabilizes.

**Outputs:** `embeddings/{unique_name}/probe_results.json` · PaySim: `embeddings/paysim/{unique_name}/probe_results*.json`

---

## Recommended configs (Jun 2026)

**Full-label frozen probe (AUROC):** **8192neg + projection with queue disabled** is the current strongest asym AUROC recipe (**0.951 AUROC**, **0.233 F1**). Across seeds, `same_pair` false-negative filtering is the leading F1/recall variant (**0.236 mean F1**, **0.359 mean recall**) with a small AUROC tradeoff.

**Label-efficiency:** sym+proj best @ **10%** labels (0.924 AUROC); 8192neg+proj best @ **50–100%** (0.931). See [label-efficiency](#label-efficiency-small-hi) below.

**Morphology expert:** **M1b** @ 20 ep — best full-expert morph-only (0.920 AUROC). Targeted group scouts: best morphology **F1** still **`degree_fan` 10 ep, w=1.0** (0.208); best morphology **AUROC** is **`degree_fan` 20 ep, w=0.05, last epoch** (0.943) or **`motif_participation` 10 ep, w=0.05** (0.937). Use **`motif_participation` / `flow_balance` @ w=0.05**; avoid **motif @ w=1.0** with morph-val best (probe last epoch if needed). See [morphology target-group scouts](#morphology-target-group-scouts-small-hi). **Feature-KNN:** negative exclusion **did not help**; soft positives **hurt** (0.849 / 0.067) — see [feature-KNN](#feature-knn-small-hi). MAE expert loss did not beat MSE. Stacking BC on M1b hurts; M5a grouped heads did not fix interference (0.887 AUROC).

**Projection ablation:** [`projection-head-ablation-jun2026.md`](projection-head-ablation-jun2026.md). Earlier best contrastive+proj AUROC before the queue sweep: asym + 8192 negs (**0.930**). Earlier best F1: sym + 1024 @ `bs=16384` (**0.222**).

**Queue / negative ablations:** for asym + projection + 8192 negatives + `bs=8192 accum=4`, disabling the queue is best. Larger queues reduced AUROC/F1; increasing negatives beyond 8192 did not improve AUROC, even with false-negative filtering. See [queue and negative ablations](#queue-and-negative-ablations-small-hi) below.

**Temperature:** lower InfoNCE temperatures (`0.05`, `0.10`, `0.20`) underperformed the prior default `0.5`; keep `--contrastive_temperature 0.5` for the current recipe.

**Multi-positive InfoNCE:** endpoint weak-positive runs underperformed exclusion-only filtering. `same_pair` weak positives at weight `0.1` averaged **0.938 AUROC / 0.153 F1**; lowering the weight to `0.05` helped but still did not beat no-filter or `same_pair` exclusion.

---

## Small-HI SSL benchmark (linear probe, val-tuned F1, GIN hetero)

| Run | Config | Epochs | Test AUROC | Test F1 |
|-----|--------|--------|------------|---------|
| Contrastive + proj, **8192 negs, no queue** | asym; `queue=0`; `bs=8192 accum=4` | 20 → **ep 19** | **0.951** | **0.233** |
| Contrastive + proj, **KNN exclusion k=5** | asym; 8192 negs; `queue=0`; feature-KNN filter | 20 | 0.947 | 0.209 |
| Contrastive + proj, **KNN exclusion k=15** | asym; 8192 negs; `queue=0`; feature-KNN filter | 20 | 0.928 | 0.176 |
| Contrastive + proj, **KNN soft-pos m=1** | asym; 8192 negs; `queue=0`; `w=0.025` | 20 → **ep 19** | 0.849 | 0.067 |
| Contrastive + proj, **same-pair filter** | asym; 8192 negs; `queue=0`; mean over 3 seeds | 20 | 0.942 | **0.236** |
| Contrastive + proj, **same-endpoint filter** | asym; 8192 negs; `queue=0`; seed 1 only | 20 | 0.940 | 0.277 |
| Contrastive + proj, **8192 negs, queued** | asym; 8192 negs + queue; `bs=8192 accum=4` | 20 | **0.930** | 0.191 |
| **M1b + clustering + projection** | M1b + 11 local + `--contrast_projection_head` | 20 | **0.929** | 0.156 |
| M1b + sym + projection | M1b + 14 local + sym @ `bs=16384` | 20 | **0.930** | 0.134 |
| M1b + clustering + triangles + proj | 14 local (`--morph_local_subset all`) | 20 | 0.912 | 0.145 |
| Contrastive + proj, **symmetric** | sym InfoNCE + proj; `bs=16384 accum=2` | 20 | **0.929** | **0.222** |
| Contrastive + proj, asym @ 16384 | confound control; 1024 negs | 20 | 0.920 | 0.206 |
| Contrastive + projection | asym InfoNCE + proj; `bs=32768` | 20 | 0.927 | 0.144 |
| M1b + projection | M1b + projection head | 20 → **ep 15** | 0.924 | 0.096 |
| M1b | `--morph_expert`, `local+global` | 20 | 0.920 | 0.108 |
| M1b + MAE expert | `--morph_expert_loss mae` | 20 | 0.898 | 0.145 |
| M1b + clustering (MSE) | M1b, 11 local dims | 20 | 0.903 | 0.117 |
| M3 BC-only | `local+tier2` | 20 | 0.904 | 0.093 |
| M2 expert + contrast | + `--checkpoint_policy best` | 10 → **ep 9** | 0.906 | 0.058 |
| M2 expert + contrast | + `--checkpoint_policy best` | 20 | 0.891 | 0.107 |
| M1 | `--morph_expert`, `local` | 20 | 0.910 | 0.079 |
| M3 M1b + BC (4 lift cols) | `local+global+tier2`, best ckpt | 20 → **ep 14** | 0.896 | 0.033 |
| M3 M1b + bc_max | `lift max` | 20 | 0.889 | 0.086 |
| M5a grouped BC | `layout=grouped`, `w_tier2=1` | 20 | 0.887 | 0.028 |
| M2 expert + contrast | `morph_expert_weight=0.5` | 10 | 0.876 | 0.027 |
| M3 M1b + BC (last epoch) | same, no M4 | 20 | 0.861 | 0.029 |
| Contrastive baseline | identity InfoNCE | 20 | 0.839 | 0.076 |
| M2 expert + contrast | M1b + `--morph_contrast` | 20 (last) | 0.864 | 0.025 |
| M2 contrast only | `--morph_contrast` (no expert) | 10 | 0.680 | 0.012 |
| Supervised CE (in-GNN) | `--objective supervised` | — | ~0.972 | ~0.493 |

More detail: [`morphology-metrics-plan.md`](morphology-metrics-plan.md) · typology diagnostics: [`downstream-eval-plan.md`](downstream-eval-plan.md).

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

`same_pair` is the cleanest false-negative filter so far: it improves mean F1/recall over no-filter while preserving most AUROC. `same_receiver` produced one strong AUROC run but was unstable; `same_endpoint` improved precision/F1 on average but did not cleanly replicate the seed-1 jump.

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
