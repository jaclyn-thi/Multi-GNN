# Development results (not frozen benchmarks)

Quick-run numbers for internal comparison while configs and code still change. **Not** a formal evaluation protocol — numbers may change we refine the stack.

**New to the repo?** Start with [Recommended configs](#recommended-configs-jun-2026), the [key runs leaderboard](#key-runs-leaderboard-jun-26--jul-2), and the [Small-LI scout](#small-li-current-protocol-scout-jul-2) if you care about dataset transfer. Deeper history is in sections below (not all runs use the same protocol).

**Outputs:** `embeddings/{unique_name}/probe_results.json` · PaySim: `embeddings/paysim/{unique_name}/probe_results*.json` · AUPRC summary: `results/diagnostics/linear_probe_auprc_summary.json` · Feature ablation: `results/diagnostics/probe_feature_ablation_current_protocol_comparison.json` · Stack focus: [`probe_feature_ablation_current_protocol_stack_comparison.md`](probe_feature_ablation_current_protocol_stack_comparison.md) · **40 ep probe sweep:** [`probe_sweep_40ep_current_protocol.md`](probe_sweep_40ep_current_protocol.md) · JSON: `results/diagnostics/probe_sweep_40ep_current_protocol.json` · **Small-LI scout:** [`small_li_current_protocol_comparison.md`](small_li_current_protocol_comparison.md)

**Metrics (linear probe, test split):** **AUROC** — overall ranking. **AUPRC** (average precision) — better for rare positives (~0.19% test edges on Small-HI). **F1** — val-tuned threshold (primary operational metric in this doc). **F1@0.5** — fixed 0.5 threshold when reported. All from `linear_probe.py` unless noted (feature ablation uses same probe protocol).

---

## Recommended configs (Jun 2026)

**Current comparison-protocol backbone:** asym contrastive + projection, 8192 negs, `queue=0`, `bs=8192 accum=4`, 20 ep, seed 1, GINe, **`--reverse_mp --ego --ports --emlps --tds`**. No morphology expert, KNN, or multi-positive unless explicitly testing those. (Protocol may still change — treat numbers as dev comparisons, not frozen benchmarks.)

**Fair-comparison probe policy:** headline rows use **`--class_weight model --model gin`, C=1.0, val-tuned F1**. Rows at other `cw`/C (e.g. `cw=none`, `pos_3`) are exploratory operating-point tuning and can inflate F1 via extreme thresholds — check F1@0.5 before trusting them. Thesis-safe synthesis: [`current_protocol_recent_runs_summary.md`](current_protocol_recent_runs_summary.md).

| Goal | Recipe | Test metrics |
|------|--------|--------------|
| Best **AUROC** | Baseline (`ports` only, no emlps/tds) | **0.951** AUROC · 0.120 AUPRC · 0.236 F1 |
| Best **embedding-only AUPRC** | **emlps+tds** (seed 1, 20 ep) | 0.944 AUROC · **0.213 AUPRC** · 0.259 F1 |
| Best **embedding-only F1** | **emlps+tds** (seed 2, **40 ep**) | 0.949 AUROC · 0.245 AUPRC · **0.307 F1** |
| Best **`embedding+raw`** (scout) | **emlps+tds 40 ep seed2** @ `cw=model`, C=1.0 | 0.955 AUROC · 0.288 AUPRC · **0.346 F1** (F1@0.5 **0.339**) |
| Best **full stack (`embedding+raw+morph`)** | **FNF + emlps+tds seed1** | **0.959** AUROC · **0.276 AUPRC** · **0.319 F1** |
| Best **AUPRC (representation lever)** | **emlps+tds 40ep s2 + `raw`, `pre_embedding_3h`** | 0.960 AUROC · **0.321 AUPRC** · 0.344 F1 (extraction-only; see master note) |
| Best **Small-LI legacy supervised** (paper_argmax) | **legacy head, 100 ep, seed 1** @ best-val ep 35 | 0.959 AUROC · **0.292 AUPRC** · **0.357 F1** (in-GNN; not frozen probe) |

**Current protocol (Jun 26 – Jul 2):** **`same_pair` FNF** and **`degree_aware` edge drop** do **not** beat plain emlps+tds on **embedding-only** probes. **FNF seed1** still wins the standard full stack. **40 ep** improves mean embedding-only F1 (+0.5 pp vs 20 ep seed1) but **high seed variance** (seeds 3–4 weak). **`embedding+raw`** beats **`embedding+raw+morph`** on 3/4 GIN 40 ep seeds at shared probe settings (`cw=model`, C=1.0); mean F1 **0.273 ± 0.048** vs **0.248 ± 0.036** for full stack — but only seed2 reaches **0.35+ F1**. Details: [Jun 28–29 batch](#current-protocol-comparison-batch-jun-2829) · [40 ep probe sweep](#40-ep-targeted-probe-sweep-jul-2).

**Dataset transfer scout (Small-LI, Jul 2):** current GINe emlps+tds 20 ep seed1 is much weaker on Small-LI than Small-HI. Best shared-policy Small-LI F1 is **0.076** with `embedding+raw`; full stack gets better AUPRC (**0.039**) but lower F1 (**0.056**). Treat this as evidence that the Small-HI recipe does not transfer cleanly without dataset-specific tuning. Details: [Small-LI scout](#small-li-current-protocol-scout-jul-2).

**Do not use alone:** `--emlps` without `--tds` (0.915 / 0.093 AUPRC / 0.186 F1). **`--tds` alone** matches baseline F1 (0.233) with recall-heavy profile (0.395 recall), not the emlps+tds lift. **emlps+tds is seed-sensitive** — seeds 1–3 @ 20 ep embedding-only: mean **0.208 F1** / **0.160 AUPRC** (seed 1 best, seed 3 worst: 0.157 / 0.124).

**Label-efficiency:** sym+proj best @ **10%** labels (0.924 AUROC); 8192neg+proj best @ **50–100%** (0.931). See [label-efficiency](results-archive.md#label-efficiency-small-hi).

**Morphology:** two distinct paths — (1) **probe-time** morph on frozen emlps+tds/FNF embeddings (**helps** FNF full stack up to 0.319 F1; **can hurt** GIN 40 ep full stack vs embedding-only); (2) **SSL morphology expert** during contrastive (**unreliable** with morph-val best ckpt; use last epoch if reporting). Legacy scouts: [`morphology scouts`](results-archive.md#morphology-target-group-scouts-small-hi).

**Representation extraction lever (`pre_embedding_3h`):** every row in this doc probes the exported 128-d `post_embedding`. Probing the 198-d (`3×n_hidden`) tensor fed *into* `embedding_head` instead — same frozen checkpoint, `--representation_source pre_embedding_3h`, no retraining — is a **free ranking/alert-budget upgrade** for `embedding` and `embedding+raw` stacks. On **plain Small-LI SSL**, pre-3h wins AUPRC in **3/3 seeds** (mean ΔAUPRC +0.025 embedding-only, +0.029 +raw; see [`pre_embedding_3h_vs_post_embedding_small_li_multiseed.md`](pre_embedding_3h_vs_post_embedding_small_li_multiseed.md)). HI champion remains **40ep s2 + raw + pre-3h = 0.321 AUPRC**. Side-by-side: **[`strongest_runs_master_comparison.md`](strongest_runs_master_comparison.md)**.

**Legacy supervised Small-LI (Jul 9):** formal 100-ep reproduction with Egressy `3h→50→25→2` head reaches test **paper_argmax F1 0.357** / AUPRC **0.292** at best-val epoch 35 (~2× the 20-ep scout). Final epoch collapses (F1=0) — use `checkpoint_best_val_f1.tar` only. Details: [`supervised_Small-LI_small_li_legacy_supervised_gin_emlps_tds_100ep_seed1_summary.md`](supervised_Small-LI_small_li_legacy_supervised_gin_emlps_tds_100ep_seed1_summary.md).

**Closed / negative results:** feature-KNN; multi-positive InfoNCE; PNA/RGCN encoder swap (emlps+tds); symmetric contrastive scout (emlps+tds); SSL hyper sweep; `degree_aware` + FNF stack; degree-aware on emlps+tds (embedding-only); morph-expert SSL @ morph-val best. Details in linked sections.

**Pointers:** projection [`projection-head-ablation-jun2026.md`](archive/projection-head-ablation-jun2026.md) · queue/negs [archive](results-archive.md#queue-and-negative-ablations-small-hi) · masked-edge [archive](results-archive.md#masked-edge-attribute-reconstruction-small-hi)

---

## Key runs leaderboard (Jun 26 – Jul 2)

Frozen linear probe unless noted. Probes use `--class_weight model --model gin` unless stated. Feature-ablation rows: `scripts/probe_feature_ablation.py`.

| Run | AUROC | AUPRC | F1 | F1@0.5 | Note |
|-----|------:|------:|---:|-------:|------|
| Baseline (no emlps/tds) | **0.951** | 0.120 | 0.236 | 0.215 | AUROC reference |
| **emlps+tds** (seed 1, 20 ep) | 0.944 | **0.213** | 0.259 | 0.257 | best AUPRC @ 20 ep |
| **emlps+tds** (seed 1, **40 ep**) | 0.949 | 0.201 | 0.292 | 0.295 | embedding-only; best ckpt ep 40 |
| **emlps+tds** (seed 2, **40 ep**) | 0.949 | **0.245** | **0.307** | 0.312 | best embedding-only F1; best ckpt ep 36 |
| emlps+tds (seed 3, **40 ep**) | 0.938 | 0.176 | 0.228 | 0.226 | weak replicate |
| emlps+tds (seed 4, **40 ep**) | 0.932 | 0.148 | 0.239 | 0.236 | weak replicate |
| emlps+tds (seed 3, 20 ep) | 0.921 | 0.124 | 0.157 | 0.160 | worst 20 ep replicate |
| **FNF + emlps+tds + emb+raw+morph** (s1) | **0.959** | **0.276** | **0.319** | 0.303 | **best full stack** |
| emlps+tds + emb+raw+morph (s1, 20 ep) | 0.945 | 0.276 | 0.298 | 0.327 | strong without FNF |
| emlps+tds + emb+raw+morph (s1, **40 ep**) | 0.945 | 0.264 | 0.262 | 0.248 | 40 ep **hurts** full stack |
| **emlps+tds + emb+raw** (s2, **40 ep**) | 0.955 | 0.288 | **0.346** | 0.339 | sweep-verified @ `cw=model`, C=1.0 |
| FNF + emlps+tds (embedding-only s1) | 0.942 | 0.178 | 0.236 | 0.239 | wins only with full probe |
| FNF + emlps+tds (embedding-only **s2**) | 0.926 | 0.137 | 0.206 | 0.218 | seed replicate weak |
| FNF + emlps+tds + emb+raw+morph (**s2**) | 0.955 | 0.243 | 0.262 | 0.170 | high thr (0.74); poor F1@0.5 |
| sym scout (emlps+tds, 20 ep) | 0.936 | 0.176 | 0.230 | 0.225 | negative vs asym |
| degree-aware + emlps+tds | 0.926 | 0.152 | 0.240 | 0.244 | weak vs emlps+tds |
| morph expert + emlps+tds (**last ep**) | 0.947 | 0.184 | 0.288 | 0.283 | fair morph-SSL read |
| morph expert + emlps+tds (morph-val best) | 0.927 | 0.104 | 0.187 | 0.213 | **misleading** (ep 1) |
| Masked-edge GINE 20 ep | 0.932 | 0.232 | 0.247 | 0.272 | best non-contrastive AUPRC |
| **Legacy supervised LI** (100 ep, paper_argmax) | **0.959** | **0.292** | **0.357** | — | in-GNN; best-val ep 35; not frozen probe |
| Legacy supervised LI scout (20 ep) | 0.944 | 0.191 | ~0.18–0.20 | — | scout only; superseded by 100 ep formal |

Diagnostics: `results/diagnostics/linear_probe_auprc_summary.json` · Feature ablation (6 modes): [`probe_feature_ablation_current_protocol_comparison.md`](probe_feature_ablation_current_protocol_comparison.md) · Stack focus: [`probe_feature_ablation_current_protocol_stack_comparison.md`](probe_feature_ablation_current_protocol_stack_comparison.md) · **40 ep probe sweep:** [`probe_sweep_40ep_current_protocol.md`](probe_sweep_40ep_current_protocol.md) · **Small-LI pre-3h multiseed:** [`pre_embedding_3h_vs_post_embedding_small_li_multiseed.md`](pre_embedding_3h_vs_post_embedding_small_li_multiseed.md) · Architecture sweep: `results/diagnostics/architecture_sweep_shared_probe_weights.md`

### Representation-source variants (`pre_embedding_3h`, same checkpoints/policy)

Same frozen checkpoints and fair policy as above; only the extracted representation differs (198-d pre-`embedding_head` vs 128-d export). Full table + Small-LI: [`strongest_runs_master_comparison.md`](strongest_runs_master_comparison.md).

| Run (Small-HI, best stack) | AUPRC post→pre | F1 post→pre | lift@100 post→pre |
|-----|------:|------:|------:|
| **emlps+tds 40ep s2, `+raw`** | 0.288 → **0.321** | 0.346 → 0.344 | 423 → **450** |
| emlps+tds 40ep s2, `embedding` | 0.245 → **0.295** | 0.304 → **0.340** | 429 → 445 |
| FNF s1, `+raw+morph` | 0.277 → **0.291** | 0.320 → 0.314 | 429 → 391 |
| emlps+tds 40ep s1, `+raw` | **0.318** → 0.274 | 0.276 → 0.305 | 429 → 429 |

(pre-3h wins AUPRC in 3/4 HI cells above; the 40ep s1 +raw cell is the exception. pre-3h is 198-d vs 128-d — a dimensionality confounder; read alongside AUROC/alert-budget.)

**Small-LI plain GINe multiseed (seeds 1–3, paired pre vs post):**

| Seed | stack | AUPRC post→pre | lift@100 post→pre |
|-----:|-------|---------------:|------------------:|
| 1 | embedding_only | 0.013 → **0.046** | 176 → **351** |
| 1 | +raw | 0.024 → **0.082** | 293 → **644** |
| 2 | embedding_only | 0.005 → **0.020** | 88 → **190** |
| 2 | +raw | 0.016 → **0.022** | 176 → **234** |
| 3 | embedding_only | 0.024 → **0.050** | 263 → **424** |
| 3 | +raw | 0.056 → **0.079** | 527 → **629** |

Pre-3h wins AUPRC **3/3 seeds** in both stacks. Full tables: [`pre_embedding_3h_vs_post_embedding_small_li_multiseed.md`](pre_embedding_3h_vs_post_embedding_small_li_multiseed.md).

---

## Current-protocol comparison batch (Jun 28–29)

Follow-up on the [current-protocol backbone](#recommended-configs-jun-2026). Slurm: `slurm/comparison_*`, `slurm/scout_*`, `slurm/run_probe_feature_ablation_current_protocol_baselines.sh`, `slurm/comparison_gin_emlps_tds_asym_proj_8192neg_queue0_40ep_seed{2,3,4}.sh`, `slurm/run_probe_sweep_40ep_seeds_checkpointed.sh`.

### Embedding-only linear probe

| Run | `unique_name` | Ep | AUROC | AUPRC | F1 | Δ F1 vs 20ep s1 |
|-----|---------------|---:|------:|------:|---:|-----------------|
| emlps+tds s1 (ref) | `hi_contrastive_gin_emlps_tds_proj_asym_8192neg_queue0_accum4_20ep` | 19 | 0.944 | **0.213** | 0.259 | — |
| **emlps+tds 40 ep** | `gin_emlps_tds_asym_proj_8192neg_queue0_40ep_seed1` | 40 | 0.949 | 0.201 | **0.292** | **+3.3 pp** |
| FNF seed2 | `fnf_emlps_tds_asym_proj_8192neg_queue0_20ep_seed2` | 19 | 0.926 | 0.139 | 0.209 | −5.0 pp |
| **sym scout** | `gin_emlps_tds_sym_proj_8192neg_queue0_20ep_bs4096_accum8_seed1` | 20 | 0.936 | 0.176 | 0.230 | −2.9 pp |

**40 ep embedding-only takeaway (seed1):** train loss still improved through ep 40; best ckpt = last epoch (ep 40). **F1 +3.3 pp** (0.259 → 0.292) with **AUPRC −1.4 pp** (0.213 → 0.201) vs 20 ep seed1.

**40 ep seed2 replicate:** best ckpt **ep 36** (not ep 40); last-epoch embeddings much weaker (0.124 F1 / 0.075 AUPRC). Embedding-only probe: **0.307 F1 / 0.245 AUPRC** — stronger than seed1 40 ep on both metrics. Slurm: `slurm/comparison_gin_emlps_tds_asym_proj_8192neg_queue0_40ep_seed2.sh` (job 16822378).

**FNF seed2 takeaway:** does not replicate seed1 (embedding-only **0.206** F1; full stack **0.262** F1 with thr **0.74**). Do not treat FNF downstream win as seed-stable.

**Symmetric scout:** ran cleanly at `bs=4096 accum=8` with emlps+tds and 8192 negs (no OOM). Underperforms asymmetric GIN on F1/AUPRC — closed unless revisited with matched seed replicates.

### 40 ep full-stack & seed replicate (Jun 29)

CPU ablations on frozen 40 ep embeddings (`slurm/run_probe_feature_ablation_current_protocol_gin_40ep_seed1.sh`, job 16822377; seed2 ablation inline in GPU job 16822378). Focus table: [`probe_feature_ablation_current_protocol_stack_comparison.md`](probe_feature_ablation_current_protocol_stack_comparison.md).

#### Stack comparison (test F1 / AUPRC)

| Run | `raw+morph` | `embedding` | `embedding+raw+morph` |
|-----|------------:|------------:|----------------------:|
| GIN 20 ep seed1 | 0.136 / 0.066 | 0.259 / 0.213 | **0.298 / 0.276** |
| GIN 40 ep seed1 | 0.136 / 0.066 | **0.292 / 0.199** | 0.262 / 0.264 |
| GIN 40 ep seed2 | 0.135 / 0.065 | **0.300 / 0.242** | 0.275 / 0.219 |
| GIN 40 ep seed3 | — | 0.228 / 0.176 | 0.187 / 0.196 |
| GIN 40 ep seed4 | — | 0.239 / 0.148 | 0.271 / 0.246 |
| FNF 20 ep seed1 | 0.136 / 0.066 | 0.236 / 0.179 | **0.319 / 0.276** |
| FNF 20 ep seed2 | 0.135 / 0.065 | 0.206 / 0.137 | 0.262 / 0.243 |

(`raw+morph` is identical across runs — no SSL embedding.)

**Q1 — Does 40 ep improve the full downstream stack?** **No.** GIN 40 ep seed1 **`embedding+raw+morph` drops −3.6 pp F1** vs 20 ep (0.298 → 0.262) despite **+3.3 pp** on embedding-only. Morph/raw features do not compose cleanly with the longer-trained embedding (seed1 full stack becomes recall-heavy: 0.484 recall, thr 0.54).

**Q2 — Is the 40 ep embedding gain seed-stable?** **Mixed.** Seeds 1–2 beat 20 ep seed1 on embedding-only; **seeds 3–4 are weak** (0.228 / 0.239 F1). Four-seed GPU replicates (jobs 16882364–16882365) + [Jul 2 probe sweep](#40-ep-targeted-probe-sweep-jul-2) confirm high variance. **Checkpoint caveat:** best ckpt ≠ last epoch for seeds 2–3.

#### `embedding+raw` scout (40 ep, `cw=model`, C=1.0)

| Seed | F1 | AUPRC | vs `embedding+raw+morph` |
|------|---:|------:|--------------------------|
| 1 | 0.269 | **0.316** | raw +0.7 pp F1 |
| 2 | **0.346** | 0.288 | raw +7.0 pp F1 |
| 3 | 0.214 | 0.236 | raw +2.7 pp F1 |
| 4 | 0.263 | 0.228 | morph +0.3 pp F1 (tie) |
| **Mean ± std** | **0.273 ± 0.048** | **0.267 ± 0.037** | raw wins F1 on 3/4 seeds |

40 ep **`embedding+raw`** is the best **average** downstream stack for GIN, but **not a reliable default** — seed2 drives most gains. FNF seed1 full stack (**0.319 F1**) still leads for balanced end-to-end comparison.

JSON (single-setting ablations): `results/diagnostics/probe_feature_ablation_current_protocol_gin_40ep_seed{1,2,3,4}.json`

### 40 ep targeted probe sweep (Jul 2)

Checkpointed CPU sweep over GIN 40 ep **seeds 1–4** — tests whether strong **`embedding+raw`** results are robust to probe hyperparameters. **Frozen embeddings only** (no SSL retrain). Slurm: `slurm/run_probe_sweep_40ep_seeds_checkpointed.sh` (array job 16983960; ~4.2 h/task).

**Grid:** feature modes `embedding`, `embedding+raw`, `embedding+raw+morph` · class weights **`model`** (shared GIN `{0: ~1.0, 1: ~6.275}`) and **`none`** · C **`{0.1, 1.0, 10.0}`** · 78 cells total.

**Infrastructure:** incremental checkpoint JSON per seed (`probe_sweep_40ep_seed{N}_partial.json`); feature-matrix cache under `results/cache/probe_features_current_protocol/`. Script: `scripts/probe_sweep_40ep_current_protocol.py`.

#### Shared probe policy (`cw=model`, C=1.0) — mean ± std over 4 seeds

| Feature mode | F1 | AUPRC | F1@0.5 |
|--------------|---:|------:|-------:|
| `embedding` | 0.264 ± 0.032 | 0.192 ± 0.035 | 0.265 ± 0.034 |
| **`embedding+raw`** | **0.273 ± 0.048** | **0.267 ± 0.037** | 0.190 ± 0.089 |
| `embedding+raw+morph` | 0.248 ± 0.036 | 0.229 ± 0.026 | 0.239 ± 0.065 |

#### Key answers (Jul 2 sweep)

1. **`embedding+raw` vs full stack:** raw wins F1 on **3/4 seeds** at `cw=model`, C=1.0; wins AUPRC on **21/24** cw×C pairs.
2. **Seed2 `embedding+raw` robustness:** F1 stays **0.35–0.37** across all cw/C (not a threshold artifact; F1@0.5 ≈ 0.33–0.36 with `cw=model`).
3. **Cross-seed stability:** **No** — seeds 3–4 stay ~0.21–0.27 on `embedding+raw` regardless of probe settings.
4. **vs references @ `cw=model`, C=1.0:** below GIN 20 ep full stack (0.298) and FNF seed1 (0.319) on mean; seed2 scout peaks above both.

**Probe policy note:** `cw=none` can inflate val-tuned F1 via extreme thresholds (e.g. seed1 `embedding` hits 0.318 F1 but F1@0.5 **0.13**). Prefer **`cw=model`** for fair comparisons.

Consolidated: [`probe_sweep_40ep_current_protocol.md`](probe_sweep_40ep_current_protocol.md) · `results/diagnostics/probe_sweep_40ep_current_protocol.json`

### Small-LI current-protocol scout (Jul 2)

Dataset comparison using the same simple current-protocol SSL recipe as the Small-HI 20 ep reference: GINe, asym contrastive + projection, `--emlps --tds`, 8192 negs, `queue=0`, `bs=8192 accum=4`, 20 ep, seed1. Slurm: `slurm/scout_small_li_gin_emlps_tds_asym_proj_8192neg_queue0_20ep_seed1.sh` (job 17031798; completed in 3:56:52).

**Audit:** `Small-LI` is supported as an AMLWorld dataset key. `aml-data/Small-LI/formatted_transactions.csv` has the same label/edge schema as Small-HI. Splits are calendar-day temporal splits with lower test prevalence than Small-HI: **0.0683%** positives (Small-HI reference ≈ **0.1867%**). Pattern metadata was not present; raw/morph feature generation worked.

#### Small-LI probes (`cw=model --model gin`, best ckpt ep 19)

| Features | AUROC | AUPRC | F1 | F1@0.5 | Note |
|----------|------:|------:|---:|-------:|------|
| `raw+morph` | 0.858 | 0.016 | 0.057 | 0.050 | label-free baseline |
| `embedding` | 0.899 | 0.017 | 0.052 | 0.052 | SSL embedding alone weak |
| **`embedding+raw`** | 0.909 | 0.027 | **0.076** | **0.081** | best F1 |
| `embedding+raw+morph` | **0.925** | **0.039** | 0.056 | 0.073 | best ranking/AP; recall-heavy |

**Takeaway:** Small-LI is harder for this recipe. Compared with Small-HI GINe emlps+tds seed1 (embedding-only **0.259 F1 / 0.213 AUPRC**, full stack **0.298 / 0.276**), Small-LI drops sharply. `embedding+raw` is the best Small-LI F1 stack, while morphology on top improves AUROC/AUPRC but hurts val-tuned F1. The last epoch checkpoint (ep 20) is worse than best ckpt on embedding-only (**0.029 F1 / 0.008 AUPRC**), so keep reporting best ckpt for this scout.

Detailed note: [`small_li_current_protocol_comparison.md`](small_li_current_protocol_comparison.md) · Audit: `results/diagnostics/small_li_dataset_audit.json` · Feature ablation: `results/diagnostics/probe_feature_ablation_small_li_current_protocol_seed1.json`

### Probe feature ablation — label-free vs learned (CPU)

Six modes (`raw`, `morph`, `raw+morph`, `embedding`, `embedding+raw`, `embedding+raw+morph`); shared GIN class weights. JSON: `results/diagnostics/probe_feature_ablation_current_protocol_comparison.json`.

| Features | GIN 20ep s1 | GIN 40ep s1 | FNF s1 | FNF s2 |
|----------|------------:|------------:|-------:|-------:|
| raw only | 0.009 / 0.009 | (same) | (same) | (same) |
| morph only | 0.114 / 0.064 | (same) | (same) | (same) |
| **raw+morph** | 0.136 / 0.066 | (same) | (same) | (same) |
| embedding only | 0.259 / 0.213 | **0.292 / 0.199** | 0.236 / 0.179 | 0.206 / 0.137 |
| embedding+raw | 0.274 / 0.244 | 0.269 / **0.316** | 0.256 / 0.223 | 0.230 / 0.221 |
| **embedding+raw+morph** | 0.298 / 0.276 | 0.262 / 0.264 | **0.319 / 0.276** | 0.262 / 0.243 |

(F1 / AUPRC per cell; val-tuned threshold.)

**Takeaway:** engineered **raw+morph alone caps ~0.14 F1**; SSL embeddings are necessary for strong probes. **FNF seed1** remains best on the standard full stack. GIN **40 ep + `embedding+raw`** is a **seed-sensitive scout** (seed2 **0.346 F1** @ `cw=model`, C=1.0; see [Jul 2 sweep](#40-ep-targeted-probe-sweep-jul-2)).

### Architecture sweep (emlps+tds, shared probe weights)

GIN / GAT / PNA / RGCN @ 20 ep, seed 1; apples-to-apples reprobe with `--class_weight model --model gin`. **GIN best on AUPRC** (0.213); GAT closest on F1 (0.264); PNA/RGCN weaker on F1 despite competitive AUROC.

`results/diagnostics/architecture_sweep_shared_probe_weights.json` · Does not overwrite `embeddings/*/probe_results.json`.

---

## Current protocol: emlps + tds + interventions (Jun 26)

Fair comparison: same recipe as emlps+tds baseline, plus **one** intervention. GPU Slurm: `slurm/ablation_same_pair_fnf_emlps_tds_asym_proj_8192neg_queue0_20ep.sh`, `slurm/ablation_degree_aware_edgedrop_emlps_tds_asym_proj_8192neg_queue0_20ep.sh`.

### Embedding-only linear probe

| Run | `unique_name` | AUROC | AUPRC | F1 | Δ vs emlps+tds s1 |
|-----|---------------|------:|------:|---:|-------------------|
| emlps+tds s1 (reference) | `hi_contrastive_gin_emlps_tds_proj_asym_8192neg_queue0_accum4_20ep` | 0.944 | 0.213 | 0.259 | — |
| + `same_pair` FNF | `same_pair_fnf_emlps_tds_asym_proj_8192neg_queue0_20ep` | 0.942 | 0.178 | 0.241 | −0.002 / −0.035 / −0.019 |
| + `degree_aware` edge drop | `degree_aware_edgedrop_emlps_tds_asym_proj_8192neg_queue0_20ep` | 0.926 | 0.152 | 0.240 | −0.018 / −0.061 / −0.019 |

**Embedding-only takeaway:** plain emlps+tds seed 1 wins. Interventions that helped on the legacy stack do not add value here.

### Current-protocol probe feature ablation (Jun 27, refreshed Jun 29)

CPU ablation on frozen embeddings (`slurm/run_probe_feature_ablation_final_protocol_new_embeddings.sh`). Compares `embedding`, `embedding+raw`, `embedding+raw+morph`:

| Run | embedding | embedding+raw | embedding+raw+morph |
|-----|----------:|--------------:|--------------------:|
| emlps+tds baseline | 0.259 / 0.213 AUPRC | 0.274 / 0.244 | 0.298 / **0.276** |
| **FNF + emlps+tds** | 0.236 / 0.179 | 0.256 / 0.223 | **0.319 / 0.276** |
| degree-aware + emlps+tds | 0.240 / 0.153 | 0.238 / 0.238 | 0.291 / 0.253 |

(F1 / AUPRC per cell; val-tuned threshold.)

**Full-stack takeaway:** **FNF + emlps+tds + `embedding+raw+morph`** is the current best downstream recipe (**0.319 F1**, **0.959 AUROC**, AUPRC tied with emlps+tds full stack). Degree-aware full stack has weak calibration (thr **0.70**, F1@0.5 **0.207**).

JSON: `results/diagnostics/probe_feature_ablation_final_protocol_comparison.json` (legacy name) · Current 6-mode table: [`probe_feature_ablation_current_protocol_comparison.md`](probe_feature_ablation_current_protocol_comparison.md)

### emlps+tds seed replicates (Jun 27)

Plain emlps+tds baseline, embedding-only probe (`slurm/ablation_emlps_tds_asym_proj_8192neg_queue0_20ep_seed3.sh`):

| Seed | AUROC | AUPRC | F1 | F1@0.5 | Best ep |
|------|------:|------:|---:|-------:|--------|
| 1 | 0.944 | **0.213** | **0.259** | 0.257 | 19 |
| 2 | 0.925 | 0.142 | 0.208 | 0.226 | 14 |
| 3 | 0.921 | 0.124 | 0.157 | 0.160 | 18 |
| **mean ± range** | 0.930 | 0.160 | 0.208 | — | — |

Report seed 1 as best case; cite variance when generalizing emlps+tds embedding-only numbers.

### Morphology expert + emlps+tds (current protocol, Jun 27)

`degree_fan` only, `--morph_expert_weight 0.05`, emlps+tds graph stack (`slurm/ablation_morph_expert_emlps_tds_asym_proj_8192neg_queue0_20ep.sh`). Probes **morph-val best** and **last epoch**:

| Checkpoint | Ep | AUROC | AUPRC | F1 | F1@0.5 | Note |
|------------|---:|------:|------:|---:|-------:|------|
| morph-val best | **1** | 0.927 | 0.104 | 0.187 | 0.213 | **do not report** |
| **last epoch** | 20 | 0.947 | 0.184 | **0.288** | 0.283 | fair SSL morph read |

Last-epoch morph SSL (0.288 F1) **beats** emlps+tds seed-1 embedding-only on F1 but **loses** to probe-time morph on plain emlps+tds (0.298) or FNF (0.319). Morph-val `best` checkpoint policy remains misleading at w=0.05.

Artifacts: `embeddings/morph_expert_emlps_tds_asym_proj_8192neg_queue0_20ep/probe_results.json` (best) · `embeddings/morph_expert_emlps_tds_asym_proj_8192neg_queue0_20ep_lastckpt_probe/probe_results_lastckpt.json` (last)

---

## Small-HI SSL benchmark (linear probe, val-tuned F1, GIN hetero)

For **AUPRC** on headline runs, see [key runs leaderboard](#key-runs-leaderboard-jun-26--jul-2). Table below is AUROC / F1 unless noted.

| Run | Config | Epochs | Test AUROC | Test F1 |
|-----|--------|--------|------------|---------|
| Contrastive + proj, **8192 negs, no queue** | asym; `queue=0`; `bs=8192 accum=4`; **GINe**; `--reverse_mp --ego --ports` | 20 → **ep 19** | **0.951** | **0.233** |
| Contrastive + proj, **`--emlps --tds` (seed 1)** | full Multi-GNN stack; else same | 20 → **ep 19** | 0.944 | **0.259** |
| Contrastive + proj, **`--emlps --tds` (seed 3)** | final-protocol replicate | 20 → **ep 18** | 0.921 | 0.157 |
| Contrastive + proj, **`--emlps --tds` + same_pair FNF** | final protocol; FNF only | 20 → **ep 19** | 0.942 | 0.241 |
| Contrastive + proj, **`--emlps --tds` + degree_aware** | final protocol; edge drop only | 20 → **ep 20** | 0.926 | 0.240 |
| Contrastive + proj, **morph expert + emlps+tds** | `degree_fan` w=0.05; **last ep** probe | 20 → **ep 20** | 0.947 | **0.288** |
| Contrastive + proj, **`--emlps --tds` (seed 2)** | same; `--seed 2` | 20 → **ep 14** | 0.925 | 0.209 |
| Contrastive + proj, **`--tds` only** | `+tds`; no `--emlps` | 20 → **ep 20** | 0.940 | 0.233 |
| Contrastive + proj, **`--emlps` only** | `+emlps`; no `--tds` | 20 → **ep 20** | 0.915 | 0.188 |
| Contrastive + proj, **baseline LR, h=128** | `--override_n_hidden 128`; else baseline | 20 → **ep 20** | 0.939 | 0.153 |
| Contrastive + proj, **baseline LR, final_dropout=0** | `--override_final_dropout 0.0`; else baseline | 20 → **ep 18** | 0.945 | 0.213 |
| Contrastive + proj, **LR 0.003, h=66** | `--override_lr 0.003 --override_n_hidden 66`; else same | 20 → **ep 20** | 0.932 | 0.212 |
| Contrastive + proj, **LR 0.003, h=128** | `--override_lr 0.003 --override_n_hidden 128`; else same | 20 → **ep 19** | 0.940 | 0.205 |
| Contrastive + proj, **PNA encoder** | same recipe; `--model pna`; `mit_preemptable` | 20 → **ep 20** | 0.942 | 0.186 |
| **Masked edge reconstruction (seed 1)** | `--objective masked_edge`; GINe; mask rate 0.15; zero tokens | 20 → **ep 20** | 0.932 | **0.241** |
| Masked edge reconstruction (seed 2) | same; `--seed 2` | 20 → **ep 14** | 0.943 | 0.131 |
| Masked edge reconstruction (40 ep, seed 1) | same; 40 ep | 40 → **ep 29** | 0.925 | 0.166 |
| Contrastive + proj + **masked-edge aux** | asym; 8192 negs; `queue=0`; `masked_edge_aux_weight=0.1` | 20 → **ep 20** | 0.951 | 0.239 |
| Contrastive + proj, **`degree_aware` edge drop** | asym; 8192 negs; `queue=0`; `--edge_drop_policy degree_aware` | 20 → **ep 18** | 0.949 | **0.237** |
| Contrastive + proj, **`degree_aware` + same-pair FNF** | asym; 8192 negs; `queue=0`; mean over 2 seeds | 20 | 0.939 | 0.168 |
| Contrastive + proj, **`degree_flow_aware` edge drop** | asym; 8192 negs; `queue=0`; `--edge_drop_policy degree_flow_aware` | 20 → **ep 20** | 0.947 | 0.234 |
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

## Historical ablations (archived)

Older ablations that predate the current comparison protocol — queue/negative sweeps, feature-KNN, edge-drop, PNA/RGCN swaps, masked-edge reconstruction, morphology target-group scouts, label-efficiency, and PaySim transfer — now live in [`results-archive.md`](results-archive.md). Kept for provenance and negative-result history.
