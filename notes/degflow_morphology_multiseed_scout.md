# Degflow morphology multiseed scout

> **Hand analysis (Jul 19):** Recommendation **`stop`**. Seed-1 degflow gains do **not** replicate: seed 2 is a precision collapse vs its matched baseline; seed 3 is weak in absolute A/B/D metrics. Do not promote; do not 40ep scale-up. See audit §9 (`notes/morphology_objective_recall_audit.md`) and `notes/current_protocol_recent_runs_summary.md`. Auto-generated body below.

**Thesis role:** diagnostic_or_scout · **validation_status:** diagnostic_only · **table_eligible:** false · **table_group:** `degflow_morphology_multiseed_scout`

Primary representation: **pre_embedding_3h**. Post-128 diagnostic only.
SSL: InfoNCE + morphology expert **regression** only (degree_fan + flow_balance). **No labels.**

Exact flags: `--morph_expert --morph_targets local+global --morph_flow_balance --morph_target_groups degree_fan,flow_balance --morph_expert_weight 1.0 --morph_expert_hidden 64`

Excluded: clustering, degflow_tfreg, M2/bin contrast, TF soft positives, tier2/betweenness.

## Recommendation

- **`stop`**
- Promote as representation objective (even if baseline D wins strict precision): **False**
- Promote as overall best final method: **False**
- Scale to 40ep: **False**
- Precision-collapse seeds (A P@100 < 50% of matched baseline): `[2]`
- Paired seeds with matched baseline: **2**

## Claim 1 — Representation improvement

Evaluate whether degflow improves the learned pre-3h representation **before** downstream temporal-flow features are added.

- Best representation-only (A AUPRC): **degflow_seed1** (0.2828)
- Best +raw (B AUPRC): **degflow_seed1** (0.3719)
- Degflow A AUPRC mean±SD: 0.1471±0.1176 (n=3)
- Degflow B AUPRC mean±SD: 0.1755±0.1750 (n=3)
- Degflow A P@100 mean±SD: 0.3100±0.4694 (n=3)
- Degflow A R@P≥0.80 mean±SD: 0.1210 (n=1)
- Degflow A R@P≥0.90 mean±SD: 0.0168 (n=1)
- A AUPRC improved on all paired seeds: **False**
- B AUPRC improved on all paired seeds: **False**

## Claim 2 — Final-stack (D) tradeoff

Do **not** rank D by AUPRC alone. Seed-1 showed degflow D can raise AUPRC / R@1000 while lowering P@100 and high-precision recall vs baseline D.

- Best final D by AUPRC: **baseline_seed2** (0.5111)
- Best final D by strict high-precision (R@P≥0.90): **baseline_seed2** (0.1359)
- Best final D by R@P≥0.80: **baseline_seed2** (0.3352)
- Best final D by broader recall (R@1000): **baseline_seed2** (0.4401)
- Best final D by P@100: **baseline_seed1** (0.9400)
- Degflow D AUPRC mean±SD: 0.2738±0.1838 (n=3)
- Degflow D P@100 mean±SD: 0.3033±0.4477 (n=3)
- Degflow D R@1000 mean±SD: 0.2694±0.1396 (n=3)
- Degflow D R@P≥0.90 mean±SD: 0.0304 (n=1)

## Per-seed pre-3h metrics

| Seed | Variant | ckpt ep | Arm | AUROC | AUPRC | F1 | P@100 | R@500 | R@1000 | R@P≥0.90 | R@P≥0.80 |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | baseline | 19 | A_embedding | 0.9608 | 0.1888 | 0.2504 | 0.6600 | 0.1421 | 0.2061 | 0.0006 | 0.0006 |
| 1 | baseline | 19 | B_embedding_raw | 0.9617 | 0.2113 | 0.2872 | 0.6700 | 0.1558 | 0.2291 | 0.0006 | 0.0006 |
| 1 | baseline | 19 | D_embedding_raw_temporal_flow | 0.9786 | 0.4337 | 0.4482 | 0.9400 | 0.2495 | 0.3780 | 0.1291 | 0.2514 |
| 1 | degflow | 13 | A_embedding | 0.9331 | 0.2828 | 0.3175 | 0.8500 | 0.2030 | 0.2762 | 0.0168 | 0.1210 |
| 1 | degflow | 13 | B_embedding_raw | 0.9461 | 0.3719 | 0.0307 | 0.7400 | 0.2154 | 0.3575 | 0.0279 | 0.0354 |
| 1 | degflow | 13 | D_embedding_raw_temporal_flow | 0.9580 | 0.4740 | 0.0468 | 0.8200 | 0.2359 | 0.4202 | 0.0304 | 0.0571 |
| 2 | baseline | 14 | A_embedding | 0.9484 | 0.2598 | 0.2947 | 0.7900 | 0.1819 | 0.2694 | — | 0.0776 |
| 2 | baseline | 14 | B_embedding_raw | 0.9496 | 0.2725 | 0.3078 | 0.7900 | 0.1887 | 0.2800 | — | 0.0919 |
| 2 | baseline | 14 | D_embedding_raw_temporal_flow | 0.9734 | 0.5111 | 0.5012 | 0.9300 | 0.2626 | 0.4401 | 0.1359 | 0.3352 |
| 2 | degflow | 13 | A_embedding | 0.9378 | 0.0756 | 0.1803 | 0.0800 | 0.0484 | 0.1049 | — | — |
| 2 | degflow | 13 | B_embedding_raw | 0.9010 | 0.0362 | 0.0200 | 0.0500 | 0.0317 | 0.0670 | — | — |
| 2 | degflow | 13 | D_embedding_raw_temporal_flow | 0.9550 | 0.1126 | 0.0325 | 0.0300 | 0.0608 | 0.1446 | — | — |
| 3 | degflow | 20 | A_embedding | 0.9195 | 0.0828 | 0.1972 | 0.0000 | 0.0478 | 0.1148 | — | — |
| 3 | degflow | 20 | B_embedding_raw | 0.9287 | 0.1185 | 0.1797 | 0.0300 | 0.0720 | 0.1546 | — | — |
| 3 | degflow | 20 | D_embedding_raw_temporal_flow | 0.9513 | 0.2348 | 0.1552 | 0.0600 | 0.0999 | 0.2433 | — | — |

## Paired deltas (degflow − matched baseline)

| Seed | ΔA AUPRC | ΔA P@100 | ΔA R@P≥0.80 | ΔB AUPRC | ΔD AUPRC | ΔD P@100 | ΔD R@1000 | ΔD R@P≥0.90 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0941 | 0.1900 | 0.1204 | 0.1606 | 0.0403 | -0.1200 | 0.0422 | -0.0987 |
| 2 | -0.1842 | -0.7100 | — | -0.2364 | -0.3986 | -0.9000 | -0.2955 | — |
| 3 | — | — | — | — | — | — | — | — |

## Baseline availability

- Seed 1: `hi_contrastive_gin_emlps_tds_proj_asym_8192neg_queue0_accum4_20ep` (already probed).
- Seed 2: `hi_contrastive_gin_emlps_tds_seed2_proj_asym_8192neg_queue0_accum4_20ep` (extract+probe only; no retrain).
- Seed 3: no matched 20ep plain-contrastive checkpoint; **not retrained**.

## Notes

- Rank primarily by pre-3h A/B AUPRC and recall@P / P@K — not AUROC or F1 alone.
- Val-tuned F1 can be degenerate (flag-everything thresholds); treat with caution.
- Do not insert into main thesis tables yet.

