# PNA SSL reproduction audit (read-only)

**Status:** audit only — no code changes, no jobs submitted.  
**Date:** 2026-07-09  
**Question:** Why did PNA not show the supervised Multi-PNA advantage (Egressy et al.) under our contrastive SSL protocol?

---

## 1. Executive summary

**No implementation bug is currently evident.** The PNA encoder core (`PNAConv` stack, aggregators, scalers, towers, edge-update MLPs, readout geometry) matches upstream IBM Multi-GNN at fork-point `fc751e8`. Existing contrastive PNA runs trained successfully, exported 128-d embeddings, and probed with the same downstream protocol as GIN after a shared-weight reprobe.

**The existing GIN vs PNA SSL evidence is not a fair architecture comparison.** PNA inherited **supervised Bayes-opt hyperparameters** from `model_settings.json` (`n_hidden=20`, LR `6.1e-4`, `final_dropout=0.29`) while GIN uses `n_hidden=66`, LR `6.2e-3`, `final_dropout=0.10`. That yields **~78k vs ~183k trainable encoder parameters** (~2.3× gap) and a **pre-embedding width of 60-d vs 198-d** before the shared 128-d `embedding_head`. The architecture sweep correctly reprobed with shared GIN class weights, but **did not retune or capacity-match the encoder**.

**Most plausible explanations (ranked):**

1. **Objective mismatch (supervised vs contrastive)** — PNA’s degree-aware multi-aggregation may help more when trained with CE on labels than with InfoNCE on augmented views.
2. **Inherited supervised hyperparameters unsuitable for SSL** — especially `n_hidden=20` and low LR; not re-tuned for contrastive pretraining.
3. **Severe capacity mismatch vs GIN** — largest confound in the only emlps+tds apples-to-oranges sweep.
4. **Representation geometry differs from GIN** — for PNA, `pre_embedding_3h` is **60-d** and `post_embedding` is **128-d** (expansion), the opposite of GIN (198→128 compression); the GIN pre-3h “free upgrade” story may not transfer.
5. **Weak ranking calibration** — PNA AUROC is competitive (~0.946) but AUPRC collapses (0.112 vs GIN 0.213); threshold transfer is poor (val-tuned thr 0.27 vs 0.53).
6. **Degree histogram from first training minibatch** — same as upstream; uncertain impact, not a regression.
7. **Unvalidated legacy/PNA parity** — only GINe has numerical fork-point validation; PNA wiring through `to_hetero` + SSL heads is plausible but unproven.

**Pre-3h extraction on existing PNA checkpoints is worthwhile and cheap** (checkpoint intact; post-128 `.npz` files were deleted but can be re-exported). Expect **`pre_embedding_3h` dim = 60**, not 198.

**Before claiming “PNA is worse for SSL,” run one capacity-matched PNA scout** (`n_hidden≈65`, GIN-matched LR/dropout, same emlps+tds contrastive recipe) plus optional upstream-weight parity test on the IBM supervised checkpoint.

---

## 2. Prior PNA experiment inventory

### 2.1 Contrastive runs (this project)

| Run | Objective | Dataset | Epochs | Seed | Hidden (`n_hidden`) | Params (encoder+heads) | Flags | Representation | Test metrics (fair probe) | Status |
|-----|-----------|---------|-------:|-----:|--------------------:|-------------------------:|-------|----------------|---------------------------|--------|
| `hi_contrastive_pna_proj_asym_8192neg_queue0_accum4_20ep` | Contrastive + proj | Small-HI | 20 | 1 | **20** (→60 pre) | **~71k** | `--reverse_mp --ego --ports`; **no** `--emlps --tds`; asym 8192 neg, queue 0, bs 8192 accum 4 | post_128 | AUROC 0.942, **AUPRC n/a in sweep**, F1 **0.186** (orig probe, PNA cw) | Complete. Ckpt ✓. **npz deleted**; probe JSON ✓ |
| `pna_emlps_tds_asym_proj_8192neg_queue0_20ep_seed1` | Contrastive + proj | Small-HI | 20 | 1 | **20** (→60 pre) | **~78k** | `--reverse_mp --ego --ports --emlps --tds`; same contrastive recipe | post_128 | AUROC **0.946**, AUPRC **0.112**, F1 **0.208** (shared GIN cw) | Complete. Ckpt ✓. **npz deleted**; probe JSON ✓ |

**Reference GIN row (same architecture sweep):** `hi_contrastive_gin_emlps_tds_proj_asym_8192neg_queue0_accum4_20ep` — `n_hidden=66`, **~183k** params, AUROC 0.944, AUPRC **0.213**, F1 0.259 (shared GIN cw).

### 2.2 Architecture sweep + reprobe

| Artifact | What it did |
|----------|-------------|
| `scripts/reprobe_architecture_sweep_shared_weights.py` | Re-ran linear probe on frozen embeddings with `--class_weight model --model gin` for all four encoders |
| `results/diagnostics/architecture_sweep_shared_probe_weights.json` | Canonical numbers |
| `results/diagnostics/architecture_sweep_shared_probe_weights.md` | Human table |

**Reprobe fixed probe settings only** (class weights, `--model gin` for weight lookup). Encoder training, checkpoint, and extraction were untouched.

### 2.3 Supervised PNA checkpoints (IBM / upstream-style)

| Checkpoint file | Objective | Dataset | Epoch (in ckpt) | Hidden | Head | Notes |
|-----------------|-----------|---------|----------------:|-------:|------|-------|
| `checkpoint_multi-pna-SmallHI-50epochs.tar` | Supervised CE | Small-HI | 38 | 20 | **legacy `mlp`** (no `embedding_head`) | ~58k params; **not evaluated** in project SSL pipeline |
| `checkpoint_multi-pna-SmallLI-50epochs.tar` | Supervised CE | Small-LI | ? | 20 | legacy `mlp` | Same family |
| `checkpoint_multi-pna-MediumHI-50epochs.tar` | Supervised CE | Medium-HI | ? | 20 | legacy `mlp` | Same family |
| (+ `multi-pna-eu-*`, SAML-D, etc.) | Supervised CE | various | ? | 20 | legacy `mlp` | Bundled upstream-style releases |

**No project notes, JSON diagnostics, or Slurm logs** were found for training or evaluating these supervised checkpoints under the current protocol. They are assets on disk only.

### 2.4 Not found

- PNA contrastive runs beyond the two above (no multiseed, no 40ep, no Small-LI SSL PNA)
- PNA feature ablations (`probe_feature_ablation_*` for PNA)
- PNA `pre_embedding_3h` extraction or `compare_representation_source.py` runs
- PNA hyperparameter scouts for SSL (`override_n_hidden` / LR sweeps for `--model pna`)
- Numerical parity tests for PNA vs upstream (GINe-only validation exists)

### 2.5 Where findings were recorded

| Topic | Primary note / artifact |
|-------|-------------------------|
| Early PNA encoder swap (no emlps/tds) | [`results-archive.md` § PNA encoder ablation](results-archive.md) |
| Architecture sweep summary | [`results.md` § Architecture sweep](results.md), [`architecture_sweep_shared_probe_weights.md`](../results/diagnostics/architecture_sweep_shared_probe_weights.md) |
| Closed negative result flag | [`results.md`](results.md) header (“PNA/RGCN encoder swap”) |
| Slurm recipes | `slurm/ablation_contrastive_pna_proj_asym_8192neg_queue0_accum4_20ep.sh`, `slurm/ablation_pna_emlps_tds_asym_proj_8192neg_queue0_20ep_seed1.sh` |
| Training logs | `slurm-logs/ablation_contrastive_pna_proj_asym_8192neg_queue0_accum4_20ep.sh_16512658.{out,err}`, `slurm-logs/pna_emlps_tds_16654962.{out,err}` |

---

## 3. Upstream vs current implementation

Comparison base: [IBM/Multi-GNN `models.py` @ main](https://github.com/IBM/Multi-GNN/blob/main/models.py) (matches fork `fc751e8` PNA body) vs `models.py` + `training.py` in this repo.

| Component | Upstream | Current | Classification |
|-----------|----------|---------|----------------|
| PNA aggregators | `mean, min, max, std` | same | Match — expected |
| PNA scalers | `identity, amplification, attenuation` | same | Match — expected |
| `n_hidden` rounding | `int((n_hidden // 5) * 5)` | same | Match — expected |
| Towers / pre / post layers | `towers=5, pre_layers=1, post_layers=1, divide_input=False` | same | Match — expected |
| Node / edge input MLPs | `Linear(num_features→h)`, `Linear(edge_dim→h)` | same | Match — expected |
| GNN residual | `(x + relu(BN(conv))) / 2` | same | Match — expected |
| Edge updates (emlps) | optional `Linear(3h→h)→ReLU→Linear(h→h)`, residual `/2` | same | Match — expected |
| Edge readout | `cat(relu(node_pair), edge_attr)` → `3*n_hidden` | same | Match — expected |
| Supervised head | `mlp: Linear(3h,50)→…→Linear(25,n_classes)` | `legacy` head restores same; default SSL uses `embedding_head` + `classifier` | **Expected project extension** (GFM/SSL) |
| `forward()` return | always `mlp(edge_rep)` logits path | `embedding_head(edge_rep)` → 128-d `z` (or raw `edge_rep` if `legacy`) | **Expected project extension** |
| Degree histogram | `degree(edge_index[1])` on homo; concat forward+reverse dst on hetero; `torch.bincount` | same in `get_model()` | Match upstream — **possible comparability issue** (see below) |
| Degree stored in checkpoint | No — recomputed at model init | No — recomputed at model init | Match upstream |
| `to_hetero` | `aggr='mean'` when `--reverse_mp` | same | Match — expected |
| Dropout on `PNAConv` | not passed (PyG default) | not passed | Match |
| `dropout` / `final_dropout` config fields | passed to model ctor; `final_dropout` used in head only | same | Match |
| Ports / ego / TDS | upstream supports via data pipeline | same flags; TDS increases `edge_dim` (logged as 8-d with emlps+tds vs 6-d without) | **Expected project extension** (TDS); wiring looks correct in logs |
| Contrastive projection head | N/A (upstream is supervised-only) | 128→128→128, not used at extraction | **Expected project extension** |
| GINe numerical validation | N/A | GINe `legacy` only | PNA explicitly **restored-but-unvalidated** (`models.py` comment, `util.py`) |
| RGCN `num_relations` | hardcoded `8` in upstream `get_model` bug | `2` if `reverse_mp` else `1` | Not PNA-specific; upstream typo |

### Degree histogram caveat

Both upstream and current code compute `deg` from the **first `LinkNeighborLoader` minibatch** (`sample_batch = next(iter(tr_loader))`), not the full training graph. This is inherited behavior, not a project regression. Impact is **uncertain** — if wrong, it would affect PNA and upstream supervised equally for a given loader config. **Requires a test** comparing deg from full graph vs minibatch vs val/test subgraph.

### Wiring sanity (from Slurm logs)

- Hetero graph with forward + reverse edges ✓
- `edge_emb` input 8 with `--tds`, 6 without ✓
- `embedding_head` maps `[*, 60] → [*, 128]` for `n_hidden=20` ✓
- Contrastive loss on forward-edge seeds only ✓
- InfoNCE loss decreases normally (7.27→7.07 over 20 ep) ✓

**No accidental `emlps=false` on the emlps+tds run** — `emlps` ModuleList present in summary (6,560 params).

---

## 4. Fairness of the existing GIN / PNA comparison

### 4.1 Side-by-side (emlps+tds architecture sweep)

| Knob | GIN (`hi_contrastive_gin_emlps_tds_…`) | PNA (`pna_emlps_tds_…`) |
|------|----------------------------------------|-------------------------|
| `n_hidden` | 66 | **20** |
| Pre-embedding dim (`3×h`) | **198** | **60** |
| Trainable params | **~183k** | **~78k** |
| LR | 0.00621 | **0.000612** (~10× lower) |
| Dropout | 0.0098 | 0.0834 |
| Final dropout | 0.105 | **0.288** |
| `n_gnn_layers` | 2 | 2 |
| Batch / accum | 8192 × 4 | 8192 × 4 |
| Projection head | 128→128→128 | same |
| Negatives / queue | 8192 / 0 | same |
| Checkpoint policy | best (train loss) | best (train loss) |
| Best epoch | 19 | 20 |
| Graph flags | `--reverse_mp --ego --ports --emlps --tds` | same |
| Probe (fair) | `cw=model, --model gin`, C=1.0, val-tuned F1 | same (reprobe) |

GAT/RGCN in the same sweep inherited GIN-like `n_hidden` (64 / 66) — **only PNA used the small supervised-tuned width**.

### 4.2 Answers

1. **Protocol-matched?** **Partially.** Same contrastive recipe, graph extensions, probe policy after reprobe. **Not** matched on encoder hyperparameters (by design of `extract_param` + `model_settings.json`).
2. **Capacity reasonably matched?** **No.** ~2.3× parameter gap; pre-embedding 3.3× narrower.
3. **PNA hyperparameters inherited from supervised tuning?** **Yes.** `model_settings.json` `"pna"` block is Bayes-opt for supervised CE (LR range 1e-4–1e-3, `n_hidden` 16–64, higher `final_dropout`).
4. **Did reprobes fix encoder training?** **No.** Only probe class weights / `--model gin` for weight lookup.
5. **Sufficient to say PNA is worse for SSL?** **No.** Evidence supports “**default supervised-tuned PNA underfits / mis-tunes for this SSL recipe relative to GIN**.” Not “PNA architecture cannot work for contrastive AML.”

### 4.3 Interpreting metrics

| Encoder | AUROC | AUPRC | F1 | Pattern |
|---------|------:|------:|---:|---------|
| GIN | 0.944 | **0.213** | 0.259 | Strong ranking + threshold |
| GAT | 0.932 | 0.169 | **0.264** | Best F1, weaker AUPRC |
| PNA | **0.946** | **0.112** | 0.208 | High separability, **poor precision-recall ranking** |
| RGCN | 0.940 | 0.155 | 0.220 | Middle |

PNA is **not** uniformly broken — it learns something (AUROC tops the table) but fails on AUPRC/F1 under a rare-positive probe. That pattern fits **capacity + calibration + objective** better than a wiring bug.

---

## 5. Existing-checkpoint pre-3h diagnostic (plan only)

### 5.1 Asset status

| Asset | `pna_emlps_tds_…_seed1` | `hi_contrastive_pna_…_20ep` |
|-------|-------------------------|-----------------------------|
| Checkpoint `.tar` | ✓ `saved-models/checkpoint_pna_emlps_tds_asym_proj_8192neg_queue0_20ep_seed1.tar` | ✓ `saved-models/checkpoint_hi_contrastive_pna_proj_asym_8192neg_queue0_accum4_20ep.tar` |
| `meta.json` | ✓ | ✓ |
| `probe_results.json` (post_128) | ✓ | ✓ |
| `train/val/test.npz` | **✗ deleted** (log shows they existed 2026-06-26) | **✗ deleted** |

Checkpoints are valid for re-extraction. Confirmed shapes from checkpoint state dict:

- `embedding_head.weight`: `(128, 60)` → **`n_hidden=20`, `pre_embedding_3h` dim = 60**
- `post_embedding` dim = **128**

### 5.2 Proposed workflow (do not submit until approved)

**Primary candidate:** `pna_emlps_tds_asym_proj_8192neg_queue0_20ep_seed1` (current-protocol match to GIN sweep).

```bash
# 1) Re-export post_128 (needed because npz were removed)
python embedding_extraction.py --data Small-HI --model pna \
  --unique_name pna_emlps_tds_asym_proj_8192neg_queue0_20ep_seed1 \
  --reverse_mp --ego --ports --emlps --tds --testing

# 2) Export pre_embedding_3h (writes embeddings/<run>/pre_embedding_3h/{train,val,test}.npz)
python embedding_extraction.py --data Small-HI --model pna \
  --unique_name pna_emlps_tds_asym_proj_8192neg_queue0_20ep_seed1 \
  --reverse_mp --ego --ports --emlps --tds \
  --representation_source pre_embedding_3h --testing

# 3) Paired compare (mirror GIN pre-3h scripts)
python scripts/compare_representation_source.py \
  --data Small-HI \
  --post_dir embeddings/pna_emlps_tds_asym_proj_8192neg_queue0_20ep_seed1 \
  --pre_dir embeddings/pna_emlps_tds_asym_proj_8192neg_queue0_20ep_seed1/pre_embedding_3h \
  --run_name pna_emlps_tds_20ep_seed1 \
  --class_weight model --model gin --seed 1 --with_raw \
  --output_json results/diagnostics/pre_embedding_3h_vs_post_embedding_pna_emlps_tds_seed1.json \
  --output_md notes/pre_embedding_3h_vs_post_embedding_pna_emlps_tds_seed1.md
```

**Secondary (optional):** repeat on `hi_contrastive_pna_proj_asym_8192neg_queue0_accum4_20ep` (no emlps/tds early ablation) for recipe alignment with [`results-archive.md`](results-archive.md).

**Expected outcome:** For GIN, pre-3h often wins on AUPRC because 198-d preserves ranking signal before the 128-d bottleneck. For PNA, the head **expands** 60→128, so pre-3h may **not** help — this is exactly the diagnostic value.

**Compute:** ~1× GPU extraction pass (~30–45 min) + CPU paired probe (~15 min). No retraining.

---

## 6. Parity and smoke-test plan (proposed, not implemented)

| Test | Purpose | Priority |
|------|---------|----------|
| **PNA forward parity vs upstream** | Load IBM `checkpoint_multi-pna-SmallHI-50epochs.tar` into homogeneous PNA with `supervised_head=legacy`, match upstream `mlp` keys, compare logits on fixed batch | **High** — only test that can confirm no silent regression |
| **Degree histogram audit** | `deg` from full train graph vs first loader batch vs extraction loader | **High** for PNA-specific correctness |
| **Aggregator/scaler config snapshot** | Assert `PNAConv` kwargs match upstream constants | Low (already code-identical) |
| **Flags → tensor shapes** | `--emlps --tds` gives `edge_emb` in_features=8; without TDS, 6 or 5 per recipe | Medium |
| **Legacy supervised logits** | `evaluate_supervised_gnn.py --model pna --supervised_head legacy` on IBM ckpt vs stored metrics if available | Medium |
| **post / pre embedding shapes** | Extraction smoke: `Z.shape[1] == 128` and `60` respectively | Medium (partially done via ckpt inspect) |
| **Checkpoint round-trip** | train 1 epoch `--testing`, save, load, max logit diff | Medium |
| **Train vs extraction model graph** | Same `unique_name`, compare `deg`, `n_hidden`, hetero metadata after load | Medium |

**Suggested location:** `tests/test_pna_upstream_parity.py` (new) — mirror any existing GINe fork-point test pattern if present in maintainer `tests/` (gitignored in public clone).

---

## 7. Likely explanations (ranked)

| Rank | Explanation | Evidence strength |
|------|-------------|-------------------|
| 1 | Supervised vs contrastive objective | Egressy advantage is on **supervised** Multi-PNA; we only tested SSL. No contradiction. |
| 2 | Inherited supervised hyperparams | Documented in archive; LR/h/dropout all mismatched; PNA Bayes opt prefers small `n_hidden`. |
| 3 | Capacity mismatch | 78k vs 183k params; PNA towers multiply width — small `n_hidden` especially hurts. |
| 4 | Post-128 head geometry | 60→128 is expansion for PNA vs compression for GIN; bottleneck story differs. |
| 5 | Weak AUPRC / calibration | High AUROC + low AUPRC + low val F1 (0.17) + low optimal threshold. |
| 6 | Degree histogram from minibatch | Shared with upstream; unvalidated. |
| 7 | Wiring bug (reverse_mp, ports, emlps, TDS) | Logs contradict; shapes consistent. **Low plausibility.** |
| 8 | Implementation regression | No parity test; core code matches upstream. **Unproven but unlikely** given training convergence. |

---

## 8. Is a bug currently evident?

**No.** Nothing in code review, checkpoint introspection, or training logs indicates miswired aggregators, wrong scalers, dropped emlps, or checkpoint/probe ID mismatch. The dominant issues are **experimental confounds** and **missing PNA-specific validation**, not an obvious coding error.

---

## 9. Is existing PNA pre-3h extraction worthwhile?

**Yes — as a cheap diagnostic, not as an expected fix.**

- Checkpoints exist and are compatible with `PreEmbeddingCapture` (`in_features=60`, `out_features=128`).
- Post-128 npz must be re-exported anyway (files removed from `embeddings/`).
- Result informs whether the GIN pre-3h lever generalizes or is GIN-specific / compression-specific.
- **Do not assume 198-d** — label artifacts `pre_embedding_3h_dim=60`.

---

## 10. Proposed minimal implementation / test work

### Phase A — Read-only / CPU-GPU light (no training)

1. Re-export post_128 + pre_3h for `pna_emlps_tds_…_seed1` (commands in §5.2).
2. Run `compare_representation_source.py` paired probe.
3. Add `tests/test_pna_upstream_parity.py` — forward pass vs IBM `checkpoint_multi-pna-SmallHI-50epochs.tar` on synthetic or cached batch.

### Phase B — Single capacity-matched scout (if Phase A still shows weak PNA)

One 20-epoch job:

```text
--model pna
--override_n_hidden 65    # PNA rounds 66→65; closest fair match to GIN h=66
--override_lr 0.006213266113989207
--override_final_dropout 0.10527690625126304
--reverse_mp --ego --ports --emlps --tds
(same contrastive flags as GIN emlps+tds seed1)
```

Compare param count after init (log `summary(model,…)`) to GIN ~183k before training.

### Phase C — Only if scout still loses

- Short SSL LR sweep for PNA at fixed `n_hidden=65`.
- Supervised legacy PNA eval on IBM ckpt via `evaluate_supervised_gnn.py` to re-anchor Egressy-class numbers in this codebase.

---

## 11. Files / scripts that would need changes

| Action | Files |
|--------|-------|
| Pre-3h extraction Slurm wrapper | **New** `slurm/extract_post_pre3h_pna_emlps_tds_20ep_seed1.sh` (mirror GIN scripts) |
| Paired probe | **New** `slurm/probe_pre3h_pna_emlps_tds_20ep_seed1.sh` or direct `compare_representation_source.py` invocation |
| Optional summarizer hook | Extend `scripts/summarize_pre3h_strong_runs.py` RUNS list — only if folding into master comparison |
| Parity test | **New** `tests/test_pna_upstream_parity.py` |
| Capacity scout | **New** `slurm/train_pna_capacity_matched_emlps_tds_20ep_seed1.sh` |
| Audit doc index | `notes/README.md` — add row for this file |
| **No changes required** to `models.py`, `embedding_extraction.py`, or `compare_representation_source.py` for basic pre-3h — infrastructure is model-agnostic |

---

## 12. Risks and estimated compute

| Step | Risk | GPU time | Notes |
|------|------|----------|-------|
| Re-export post_128 + pre_3h | `deg` recomputation drift; must pass identical flags as training | ~0.5–1 h | `mit_normal_gpu` |
| Paired probe | Low — standard CPU | ~15 min | |
| Upstream parity test | IBM ckpt may need legacy head + key remapping for `to_hetero` | CPU seconds | Dev-only |
| Capacity-matched 20ep scout | OOM if `n_hidden=65` PNA too wide at bs=8192 | ~3–4 h | PNA ~7 min/ep; have OOM fallback bs=4096 accum=8 |
| Full SSL PNA hyper sweep | Expensive, premature before single fair scout | days | Not recommended yet |

---

## 13. Thesis-safe wording (today)

- “Under the **current contrastive protocol** with **supervised-tuned PNA hyperparameters**, PNA underperforms GIN on AUPRC/F1 despite competitive AUROC.”
- “This does **not** refute Egressy et al.’s **supervised** Multi-PNA results; we have not run a fair capacity-matched SSL comparison or supervised PNA eval under the current data pipeline.”
- “A **capacity-matched PNA SSL scout** and **upstream parity test** are the minimal next steps before closing the architecture question.”
