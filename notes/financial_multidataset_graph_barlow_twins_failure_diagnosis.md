# Graph Barlow Twins failure diagnosis (job 19600042)

**Generated:** 2026-08-04T01:53:14.088586+00:00 (artifacts refreshed with narrative below)
**Logged steps:** 587 (RuntimeError on step 588)
**Primary hypothesis:** `mixed_std_sensitivity_amplified_as_LR_rises`

Near-zero batch std (often exactly 0 on ≥1 dim) amplifies GBT standardization gradients through `1/(std + 1e-15)`. View-repr grads intermittently hit ~1e13 from mid-warmup while losses stay moderate. Encoder grads usually remain O(10–10³) until a non-finite encoder grad at step 588 near peak LR (~1.96e-3).

## 1. Slurm / exit-code verdict

- `sacct`: unavailable on diagnosis host (`slurm001` unreachable)
- **Inferred Slurm state: `FAILED`** (not COMPLETED)
- Python: writes `failure.json`, then **re-raises** `RuntimeError`
- Wrapper `slurm/run_mixed_3domain_graph_barlow_twins_only_full.sh`: `set -euo pipefail`; final `echo end=…` never ran
- stderr: full traceback; stdout ends at `[ERROR] GBT full3000 failed`
- **Exit-code fix needed?: No** — exception propagates to nonzero shell exit; no catch-and-return-zero path

## 2. First gradient-spike thresholds

| Metric | >1e2 | >1e3 | >1e6 | >1e9 | >1e12 |
|--------|------|------|------|------|-------|
| max view-repr grad | step **247** Small-HI LR≈9.39e-4 (min_std=0, value≈2.46e13) | same | same | same | same |
| view2 | 247 Small-HI | same | same | same | same |
| view1 | 258 Small-LI | same | same | same | same |
| encoder | step **1** Small-HI LR=2e-4 (≈279) | step **297** Small-LI LR≈1.09e-3 (≈2656) | never | never | never |

- Spikes with max-view grad >1e12: **22 / 587** (3.7%)
- Domains on spikes: Small-HI 10, Small-LI 7, SAML-D 5 (not domain-exclusive)
- **0/22** spikes had `L_total > 20` → not loss-component explosion
- 18/22 spikes had LR ≥ 1e-3
- Every >1e12 spike had `min_std = 0`

## 3. Std-collapse vs LR-driven evidence

**For std-collapse / denominator instability**

- Overall `min_std` reaches 0; p01=0; median≈0.0037
- Median `min_std` on >1e12 spikes: **0** vs non-spikes ≈0.0039
- Spearman(log view-grad, log min-std) ≈ **−0.737** (descriptive)
- Pre-failure: step 586 Small-HI `view1_std_min≈2e-6`, `view1_grad≈1.15e4`, `min_diag_C≈−1.9e-4`; step 587 SAML-D **both** view `std_min=0`, `min_diag_C=0` (grads still finite ≈0.54); step **588 Small-LI** (inferred RR) → non-finite encoder grad

**For LR / update-size stress**

- View spikes cluster as LR rises through warmup toward 2e-3
- Spearman(log view-grad, LR) ≈ **0.623**; vs param-update ≈ **0.628**
- Failure at LR≈1.961e-3 (near scheduled peak)
- Param-update norms stay O(0.05–0.25) — not an explosion of Δθ by itself

**Against loss explosion / domain-only**

- Loss components decline overall; spike steps have moderate L≈3–6
- Spikes appear on all three domains

**Against encoder-grad explosion as the logged precursor**

- Encoder never logged >1e6 before NaN; last logged encoder grad (587) ≈13.7
- Intermittent **view** grads of ~1e13 are the clear numerical warning; final NaN is on encoder at 588 (batch not logged)

## 4. Window / domain summaries (compact)

| Window | n | LR range | min_std med | view_grad max | enc_grad max | >1e12 spikes | L_total mean |
|--------|---|----------|-------------|---------------|--------------|--------------|--------------|
| 1–30 | 30 | ~2e-4–2.9e-4 | healthy | modest | ~279 | 0 | high (~15–30) |
| 31–300 | 270 | →~1.1e-3 | declining mins | **~2.5e13** | ~2.7e3 | many start @247 | falling |
| 301–500 | 200 | →~1.7e-3 | mins hit 0 | ~2.5e13 | O(10²) | recurring | ~3–6 |
| 501–587 | 87 | 1.70e-3–1.96e-3 | med≈1e-3, min=0 | **~2.6e13** | ~137 | 6 | ~3.76 |
| last 20 | 20 | 1.90e-3–1.96e-3 | med≈1e-3, min=0 | ~1.2e4 | ~115 | **0** | ~3.60 |

501–587 by domain: all three show `min_std_min=0` and 2 spikes each; SAML-D keeps higher effective rank (~30) vs HI/LI (~5).

Median batch std often remains O(0.1–1) even when **minimum** dim std is 0 — collapse is sparse across dims, not global.

## 5. Checkpoint @500 integrity

- Path: `results/checkpoints/financial_multidataset_graph_barlow_twins_full3000_seed2/checkpoint_last.pt`
- SHA256: `b8e1b6eb0ca03fe6228d2db1dc7a21e61010028a12a7fd7350a971400081382f` (**unchanged**)
- Reloads; model/BN floating tensors finite; optimizer+scheduler present
- `global_step` / optimizer / scheduler completed = **500**
- Exposures: HI 167, SAML 167, LI 166 (matches RR at 500)
- Exact resume metadata present (RNG, loader generators, BN bundles)
- No encoder forward; checkpoint not modified

## 6. Ranked interventions

| Rank | ID | Intervention | Verdict |
|------|----|--------------|---------|
| 1 | **B** | Std floor / larger ε | Addresses `1/(std+ε)` amplification; `ε=1e-15` is unstable here |
| 2 | **A** | Lower peak LR | Faithful; cleanest if std were healthy — here secondary amplifier |
| 2 | **F** | B+A (≤2) | If B alone insufficient near peak LR |
| 3 | **E** | Longer warmup | Delays peak; does not fix denominator |
| 4 | **D** | Loss rescale | ≈ effective LR under Adam; less transparent than A |
| 5 | **C** | Grad clip alone | **Not recommended alone** — can hide std-denominator pathology |

## 7. Recommended recovery scout (NOT implemented / NOT submitted)

Isolated **step-500→700** resume from `checkpoint_last@500` with intervention **B** only: declared std floor (e.g. `max(std, 1e-4)` or `ε=1e-4`) keeping official λ=1/198 and GBT loss form. If view grads still explode near peak LR, a second scout adds **A** (lower peak LR) only.

## Artifacts

- Note: `notes/financial_multidataset_graph_barlow_twins_failure_diagnosis.md`
- JSON: `results/diagnostics/financial_multidataset_graph_barlow_twins_failure_diagnosis.json`
- CSV: `…/failure_diagnosis/per_step_compact.csv`, `spike_events_view_grad_gt_1e12.csv`
- Figures: `…/failure_diagnosis/figures/01_…` through `06_…`

## Confirmations

- no training resumed
- no hyperparameters changed / no job submitted
- no test-split access
- checkpoint_last.pt SHA unchanged
