# Thesis tables preview

Auto-generated from `results/diagnostics/thesis_experiment_registry.json`.

- **Rows in registry:** 345
- **Include provisional:** False
- **Temporal-flow validation passed:** True

- Label-scarcity temporal-flow diagnostic available (`results/diagnostics/label_scarcity_temporal_flow_probe.json`; see appendix).
## PI-facing summary

### Short update

- Added a causal temporal-flow feature group: sender/receiver interarrival time, sender recent 7-day activity, amount relative to sender history, and repeat sender→receiver pair.
- Built a registry/table pipeline so results are generated from result JSONs instead of hand-copied.
- Added validated temporal-flow results, Small-HI supervised baseline, and width-aligned PNA diagnostics.
- Current best frozen SSL stack is pre-3h + raw + temporal-flow.
- Supervised GIN remains strongest, but frozen SSL + temporal-flow is competitive at top-alert precision.

### Compact AUPRC summary

| Method | Small-HI AUPRC | Small-LI AUPRC |
| --- | --- | --- |
| SSL post-128 | 0.245 | 0.014 ± 0.010 |
| SSL pre-3h | 0.295 | 0.039 ± 0.016 |
| SSL pre-3h + raw | 0.321 | 0.061 ± 0.034 |
| SSL pre-3h + raw + temporal-flow | 0.501 | 0.128 ± 0.027 |
| Legacy supervised GIN | 0.663 | 0.292 |

**Notes:**
- Small-LI SSL rows are mean ± sample SD (ddof=1) over seeds 1–3.
- Small-HI SSL rows use the validated strong-run / temporal-flow protocol.
- Supervised rows use end-to-end paper_argmax evaluation; SSL rows use frozen probe AUPRC.

### Compact temporal-flow ablation

| Dataset | Baseline stack | Baseline AUPRC | + temporal-flow AUPRC | Δ AUPRC |
| --- | --- | --- | --- | --- |
| Small-HI | pre-3h + raw | 0.320 | 0.501 | +0.180 |
| Small-LI | pre-3h + raw | 0.061 ± 0.033 | 0.128 ± 0.027 | +0.067 ± 0.010 |

**Notes:**
- Baseline = pre-3h + raw; +temporal-flow = pre-3h + raw + temporal_flow_causal.
- Uses validated max_iter=5000 temporal-flow results when available.
- Small-LI values are mean ± sample SD over seeds 1–3.

### Compact supervised versus frozen SSL

| Dataset | Method | AUPRC | F1 | P@100 | R@100 | Caveat |
| --- | --- | --- | --- | --- | --- | --- |
| Small-HI | SSL pre-3h + raw + temporal-flow | 0.501 | 0.465 | 0.940 | 0.058 | val-tuned F1; frozen linear probe |
| Small-HI | Legacy supervised GIN | 0.663 | 0.660 | — | — | paper_argmax F1 mean (seeds 1–3); supervised CE; not comparable to SSL val-tuned F1 |
| Small-LI | SSL pre-3h + raw + temporal-flow | 0.128 ± 0.027 | 0.092 ± 0.029 | 0.600 ± 0.056 | 0.075 ± 0.007 | val-tuned F1; frozen linear probe; mean ± sample SD (n=3) |
| Small-LI | Legacy supervised GIN | 0.292 | 0.357 | 0.970 | 0.121 | paper_argmax F1; supervised CE; not comparable to SSL val-tuned F1 |

**Notes:**
- SSL uses frozen linear probe with validation-tuned threshold.
- Supervised uses end-to-end supervised CE and paper_argmax F1.
- F1 values are not directly comparable without the protocol caveat.

### Compact PNA diagnostic

| Encoder / setup | Representation + features | Small-HI AUPRC | Takeaway |
| --- | --- | --- | --- |
| GIN main stack | pre-3h + raw + temporal-flow | 0.501 | main encoder/result |
| PNA width-aligned | post-128 embedding only | 0.147 | post-128 understates PNA |
| PNA width-aligned | pre-3h + raw + temporal-flow | 0.407 | competitive but below GIN; one-seed diagnostic |

**Notes:**
- PNA rows are one-seed width-aligned scouts; not a full architecture ranking.
- Best-stack PNA row is downstream-only (no PNA SSL retraining).

### Definitions

- **post-128:** exported 128-dimensional embedding after the learned embedding head.
- **pre-3h:** edge representation immediately before the embedding head, formed from source-node, destination-node, and edge representations. For GIN h=66, this is 3h=198 dimensions.
- **P@100:** precision among the top 100 scored test transactions.
- **R@100:** fraction of all positive test transactions recovered in the top 100 scored test transactions.
- **Lift@100:** P@100 divided by the test-set positive rate.
- **temporal-flow:** causal per-transaction history features using only prior transactions.

---

## Full detailed tables
## Table 1 — Dataset summary

| Dataset | Split | # Transactions | # Positives | Positive rate | Task |
| --- | --- | --- | --- | --- | --- |
| Small-HI | train | 3248254 | 2530 | 0.078% | edge-level AML detection |
| Small-HI | val | 965466 | 1035 | 0.107% | edge-level AML detection |
| Small-HI | test | 863050 | 1611 | 0.187% | edge-level AML detection |
| Small-LI | train | 4432934 | 1993 | 0.045% | edge-level AML detection |
| Small-LI | val | 1316442 | 770 | 0.058% | edge-level AML detection |
| Small-LI | test | 1174673 | 802 | 0.068% | edge-level AML detection |

**Notes:**
- Split counts from cited source JSON in registry dataset_metadata; node counts omitted when unavailable.

## Table 2 — Main Small-HI results

| Method | Representation | Features | AUROC | AUPRC | F1 | P@100 | R@100 | Lift@100 | Caveat |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Raw features only | — | raw | 0.860 | 0.009 | 0.009 | — | — | — | val-tuned F1; no SSL |
| Raw + morphology | — | raw+morph | 0.905 | 0.066 | 0.136 | 0.230 | 0.014 | 123 | val-tuned F1; no SSL |
| SSL post-128 | post-128 | embedding | 0.949 | 0.245 | 0.304 | 0.800 | 0.050 | 429 | val-tuned F1 |
| SSL pre-3h | pre-3h | embedding | 0.958 | 0.295 | 0.340 | 0.830 | 0.052 | 445 | val-tuned F1 |
| SSL post-128 + raw | post-128 | embedding+raw | 0.955 | 0.284 | 0.343 | 0.790 | 0.049 | 423 | val-tuned F1 |
| SSL pre-3h + raw | pre-3h | embedding+raw | 0.960 | 0.321 | 0.344 | 0.840 | 0.052 | 450 | val-tuned F1 |
| SSL pre-3h + raw + temporal-flow | pre-3h | embedding+raw+temporal_flow_causal | 0.979 | **0.501** | 0.465 | 0.940 | 0.058 | 504 | val-tuned F1; validated temporal-flow stack |
| Legacy supervised Multi-GIN+EU (ports TDS-off, 50ep seeds1–3 mean) | logits | in-GNN end-to-end | 0.985 | 0.663 | 0.660 | — | — | — | paper_argmax F1; mean seeds 1–3; not comparable to SSL val-tuned F1 |

**Notes:**
- Small-HI pre/post rows use paired strong-run protocol (results/diagnostics/pre3h_strong_run_comparison.json).
- F1 for SSL rows is validation-tuned; raw-feature rows use val-tuned probe.
- Legacy supervised row uses end-to-end labeled training and paper_argmax F1 (not val-tuned).
- Temporal-flow stack included only when validated or with --include_provisional.
- Contrastive-method variants such as FNF are reported in the appendix.
- P@100 = precision among the top 100 scored test transactions.
- R@100 = fraction of all positive test transactions recovered in the top 100 scored test transactions.
- Lift@100 = P@100 divided by the test-set positive rate for that dataset.

## Table 3 — Main Small-LI results

| Method | Representation | Features | AUROC | AUPRC | F1 | P@100 | R@100 | Lift@100 | Caveat |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SSL post-128 | post-128 | embedding | 0.888 ± 0.016 | 0.014 ± 0.010 | 0.046 ± 0.037 | 0.120 ± 0.060 | 0.015 ± 0.007 | 176 ± 88 | frozen probe; val-tuned F1; mean ± sample SD (n=3) |
| SSL pre-3h | pre-3h | embedding | 0.919 ± 0.009 | 0.039 ± 0.016 | 0.089 ± 0.026 | 0.220 ± 0.082 | 0.027 ± 0.010 | 322 ± 120 | frozen probe; val-tuned F1; mean ± sample SD (n=3) |
| SSL post-128 + raw | post-128 | embedding+raw | 0.904 ± 0.014 | 0.032 ± 0.021 | 0.039 ± 0.028 | 0.227 ± 0.122 | 0.028 ± 0.015 | 332 ± 179 | frozen probe; val-tuned F1; mean ± sample SD (n=3) |
| SSL pre-3h + raw | pre-3h | embedding+raw | 0.926 ± 0.013 | 0.061 ± 0.034 | 0.054 ± 0.007 | 0.343 ± 0.159 | 0.043 ± 0.020 | 502 ± 232 | frozen probe; val-tuned F1; mean ± sample SD (n=3) |
| SSL pre-3h + raw + temporal-flow | pre-3h | embedding+raw+temporal_flow_causal | 0.947 ± 0.006 | 0.128 ± 0.027 | 0.092 ± 0.029 | 0.600 ± 0.056 | 0.075 ± 0.007 | 878 ± 81 | val-tuned F1; mean ± sample SD (n=3) |
| Legacy supervised GIN (100ep seed1) | logits | in-GNN end-to-end | 0.959 | 0.292 | 0.357 | 0.970 | 0.121 | 1419 | paper_argmax F1 |

**Notes:**
- SSL multiseed rows: mean ± sample SD (ddof=1) over seeds 1–3; frozen linear probe with validation-tuned thresholds.
- SSL pre-3h + raw + temporal-flow uses validated temporal-flow multiseed aggregate (same as Table 5).
- Supervised row uses paper_argmax F1 from results/diagnostics/eval_small_li_legacy_supervised_gin_emlps_tds_100ep_seed1.json; not directly comparable to SSL F1 without footnote.
- P@100 = precision among the top 100 scored test transactions.
- R@100 = fraction of all positive test transactions recovered in the top 100 scored test transactions.
- Lift@100 = P@100 divided by the test-set positive rate for that dataset.

## Table 4 — Representation readout ablation

| Dataset / run | Feature stack | Post-128 AUPRC | Pre-3h AUPRC | Δ AUPRC | Post-128 F1 | Pre-3h F1 | Caveat |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Small-HI 40ep seed2 | embedding | 0.245 | 0.295 | 0.050 | 0.304 | 0.340 | paired strong-run |
| Small-HI 40ep seed2 | embedding+raw | 0.284 | 0.321 | 0.037 | 0.343 | 0.344 | paired strong-run |
| Small-LI multiseed | embedding | 0.014 ± 0.010 | 0.039 ± 0.016 | 0.025 ± 0.009 | 0.046 ± 0.037 | 0.089 ± 0.026 | paired; multiseed mean ± SD |
| Small-LI multiseed | embedding+raw | 0.032 ± 0.021 | 0.061 ± 0.034 | 0.029 ± 0.026 | 0.039 ± 0.028 | 0.054 ± 0.007 | paired; multiseed mean ± SD |

**Notes:**
- Δ AUPRC = pre-3h minus post-128 on paired rows.
- Small-LI Δ AUPRC uses mean ± sample SD over per-seed paired deltas when available.
- emb198 scout omitted from main table (diagnostic-only; see contrastive appendix if included).

## Table 5 — Temporal-flow ablation

| Dataset / run | Comparison | Feature stack | AUPRC | Δ AUPRC vs pre-3h + raw | F1 | P@100 | R@100 | Lift@100 | Validation status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Small-HI 40ep seed2 | pre-3h only | embedding | 0.295 | — | 0.336 | 0.830 | 0.052 | 445 | validated |
| Small-HI 40ep seed2 | pre-3h + raw | embedding+raw | 0.320 | — | 0.344 | 0.830 | 0.052 | 445 | validated |
| Small-HI 40ep seed2 | pre-3h + temporal-flow | embedding+temporal_flow_causal | 0.473 | — | 0.475 | 0.910 | 0.056 | 488 | validated |
| Small-HI 40ep seed2 | pre-3h + raw + temporal-flow | embedding+raw+temporal_flow_causal | 0.501 | +0.180 | 0.465 | 0.940 | 0.058 | 504 | validated |
| Small-LI multiseed | pre-3h + raw | embedding+raw | 0.061 ± 0.033 | — | 0.056 ± 0.006 | 0.337 ± 0.153 | 0.042 ± 0.019 | 492 ± 224 | validated |
| Small-LI multiseed | pre-3h + raw + temporal-flow | embedding+raw+temporal_flow_causal | 0.128 ± 0.027 | +0.067 ± 0.010 | 0.092 ± 0.029 | 0.600 ± 0.056 | 0.075 ± 0.007 | 878 ± 81 | validated |

**Notes:**
- Primary comparison: pre-3h + raw + temporal-flow versus pre-3h + raw.
- Provisional rows shown only with --include_provisional until validation summary passes.
- Validated max_iter=5000 JSONs preferred when validation summary passes.
- P@100 = precision among the top 100 scored test transactions.
- R@100 = fraction of all positive test transactions recovered in the top 100 scored test transactions.
- Lift@100 = P@100 divided by the test-set positive rate for that dataset.

## Table 6 — Supervised versus frozen SSL

| Dataset | Method | Training signal | Encoder updated with labels? | AUPRC | F1 | P@100 | R@100 | Lift@100 | Caveat |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Small-HI | SSL pre-3h + raw + temporal-flow | contrastive + frozen probe | no (frozen probe) | 0.501 | 0.465 | 0.940 | 0.058 | 504 | val-tuned F1 |
| Small-HI | Legacy supervised GIN | supervised CE (end-to-end) | yes | 0.663 | 0.660 | — | — | — | paper_argmax F1 |
| Small-LI | SSL pre-3h + raw (multiseed mean) | contrastive + frozen probe | no (frozen probe) | 0.061 ± 0.034 | 0.054 ± 0.007 | 0.343 ± 0.159 | 0.043 ± 0.020 | 502 ± 232 | val-tuned F1; frozen linear probe |
| Small-LI | SSL pre-3h + raw + temporal-flow (multiseed mean) | contrastive + frozen probe | no (frozen probe) | 0.128 ± 0.027 | 0.092 ± 0.029 | 0.600 ± 0.056 | 0.075 ± 0.007 | 878 ± 81 | val-tuned F1; mean ± sample SD (n=3) |
| Small-LI | Legacy supervised GIN | supervised CE (end-to-end) | yes | 0.292 | 0.357 | 0.970 | 0.121 | 1419 | paper_argmax F1 |

**Notes:**
- SSL rows use frozen linear probe with validation-tuned threshold.
- Supervised rows use end-to-end labeled training and paper_argmax F1.
- SSL and supervised F1 values are not directly comparable without the protocol caveat above.
- Small-LI SSL pre-3h + raw + temporal-flow uses validated temporal-flow multiseed aggregate (same as Table 5).
- Small-HI SSL temporal-flow row uses validated single-seed strong-run protocol when available.
- P@100 = precision among the top 100 scored test transactions.
- R@100 = fraction of all positive test transactions recovered in the top 100 scored test transactions.
- Lift@100 = P@100 divided by the test-set positive rate for that dataset.

## Appendix — Alert-budget performance

| Dataset | Method | P@100 | R@100 | Lift@100 | P@500 | R@500 | Lift@500 | P@1000 | R@1000 | Lift@1000 | Caveat |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Small-HI | Raw features only | — | — | — | — | — | — | — | — | — | val-tuned F1; no SSL |
| Small-HI | Raw + morphology | 0.230 | 0.014 | 123 | 0.188 | 0.058 | 101 | 0.153 | 0.095 | 82 | val-tuned F1; no SSL |
| Small-HI | SSL post-128 | 0.800 | 0.050 | 429 | 0.590 | 0.183 | 316 | 0.398 | 0.247 | 213 | val-tuned F1 |
| Small-HI | SSL pre-3h | 0.830 | 0.052 | 445 | 0.602 | 0.187 | 322 | 0.447 | 0.277 | 239 | val-tuned F1 |
| Small-HI | SSL post-128 + raw | 0.790 | 0.049 | 423 | 0.630 | 0.196 | 337 | 0.442 | 0.274 | 237 | val-tuned F1 |
| Small-HI | SSL pre-3h + raw | 0.840 | 0.052 | 450 | 0.640 | 0.199 | 343 | 0.490 | 0.304 | 262 | val-tuned F1 |
| Small-HI | SSL pre-3h + raw + temporal-flow | 0.940 | 0.058 | 504 | 0.842 | 0.261 | 451 | 0.682 | 0.423 | 365 | val-tuned F1; validated temporal-flow stack |
| Small-HI | Legacy supervised Multi-GIN+EU (ports TDS-off 50ep mean) | — | — | — | — | — | — | — | — | — | paper_argmax F1 |
| Small-LI | SSL post-128 | 0.120 ± 0.060 | 0.015 ± 0.007 | 176 ± 88 | 0.058 ± 0.034 | 0.036 ± 0.021 | 85 ± 50 | 0.046 ± 0.025 | 0.057 ± 0.031 | 67 ± 36 | frozen probe; mean ± sample SD (n=3) |
| Small-LI | SSL pre-3h | 0.220 ± 0.082 | 0.027 ± 0.010 | 322 ± 120 | 0.123 ± 0.059 | 0.077 ± 0.037 | 180 ± 87 | 0.087 ± 0.032 | 0.108 ± 0.040 | 127 ± 47 | frozen probe; mean ± sample SD (n=3) |
| Small-LI | SSL post-128 + raw | 0.227 ± 0.122 | 0.028 ± 0.015 | 332 ± 179 | 0.095 ± 0.059 | 0.059 ± 0.037 | 138 ± 86 | 0.070 ± 0.033 | 0.087 ± 0.041 | 102 ± 49 | frozen probe; mean ± sample SD (n=3) |
| Small-LI | SSL pre-3h + raw | 0.343 ± 0.159 | 0.043 ± 0.020 | 502 ± 232 | 0.163 ± 0.085 | 0.102 ± 0.053 | 239 ± 125 | 0.111 ± 0.048 | 0.139 ± 0.060 | 163 ± 70 | frozen probe; mean ± sample SD (n=3) |
| Small-LI | SSL pre-3h + raw + temporal-flow | 0.600 ± 0.056 | 0.075 ± 0.007 | 878 ± 81 | 0.271 ± 0.046 | 0.169 ± 0.029 | 396 ± 68 | 0.163 ± 0.023 | 0.203 ± 0.028 | 238 ± 33 | val-tuned F1; mean ± sample SD (n=3) |
| Small-LI | Legacy supervised GIN (100ep seed1) | 0.970 | 0.121 | 1419 | 0.462 | 0.288 | 676 | 0.270 | 0.337 | 395 | paper_argmax F1 |

**Notes:**
- Fixed top-K alert-budget metrics on the test split; threshold-tuned precision/recall are omitted.
- Small-LI SSL rows use mean ± sample SD (ddof=1) over seeds 1–3 where available.
- K=500 and K=1000 may be unavailable (—) for some multiseed aggregates when not present in registry summaries.
- P@100 = precision among the top 100 scored test transactions.
- R@100 = fraction of all positive test transactions recovered in the top 100 scored test transactions.
- Lift@100 = P@100 divided by the test-set positive rate for that dataset.

## Appendix — Architecture ablation

| Encoder | Hidden dim | Pre dim | Post dim | Params | AUROC | AUPRC | F1 | Caveat |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gin | 66 | 198 | 128 | — | 0.944 | 0.213 | 0.259 | — |
| gat | — | — | 128 | — | 0.932 | 0.169 | 0.264 | — |
| pna | 20 | 60 | 128 | — | 0.946 | 0.112 | 0.208 | not capacity-matched to GIN |
| rgcn | — | — | 128 | — | 0.940 | 0.155 | 0.220 | — |
| pna (width-aligned) | 65 | 195 | 128 | — | 0.954 | 0.147 | 0.216 | GIN-matched LR/dropout; seed 1 scout |
| pna (width-aligned, best stack) | 65 | 195 | 128 | — | 0.982 | 0.407 | 0.410 | pre-3h+raw+temporal-flow; one seed; downstream-only diagnostic |

**Notes:**
- Rows above the best-stack PNA diagnostic are embedding-only post-128 architecture comparisons under shared probe settings (results/diagnostics/architecture_sweep_shared_probe_weights.json).
- The width-aligned best-stack PNA row is a downstream-only diagnostic (pre-3h + raw + temporal-flow) and is not directly comparable to the embedding-only architecture sweep.
- Default PNA (hidden 20, pre dim 60) was not capacity/hyperparameter matched to GIN (hidden 66, pre dim 198).

## Appendix — Contrastive and diagnostic ablations

| Variant | Dataset | Representation | Feature stack | AUROC | AUPRC | F1 | Takeaway |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GIN baseline (20ep) | Small-HI | post-128 | embedding | 0.944 | 0.213 | 0.259 | embedding-only SSL baseline from architecture sweep |
| FNF full stack | Small-HI | post-128 | embedding+raw+morph | 0.959 | 0.277 | 0.320 | FNF contrastive variant; +raw+morph (not comparable to embedding-only baseline) |
| degree-aware edge-drop | Small-HI | post-128 | embedding | 0.926 | 0.153 | 0.240 | embedding-only negative result; no gain vs baseline |
| emb198 scout (Small-LI) | Small-LI | pre-3h emb198 scout | embedding+raw | 0.891 | 0.033 | 0.059 | one-seed diagnostic scout; not multiseed canonical |

**Notes:**
- Appendix rows are curated for interpretability; raw-only rows are not compared directly to embedding-only SSL baselines.
- Pending/manual review: queue-size contrastive variants; multi-positive contrastive variants; KNN positive variants; morphology auxiliary-loss variants.

## Appendix — Label-scarcity temporal-flow diagnostic

| Label fraction | Small-HI pre-3h+raw AUPRC | Small-HI pre-3h+raw+temporal AUPRC | Small-LI pre-3h+raw AUPRC | Small-LI pre-3h+raw+temporal AUPRC |
| --- | --- | --- | --- | --- |
| 1% | 0.058 | 0.139 | 0.004 ± 0.001 | 0.004 ± 0.001 |
| 5% | 0.143 | 0.300 | 0.036 ± 0.037 | 0.068 ± 0.053 |
| 10% | 0.306 | 0.430 | 0.023 ± 0.012 | 0.052 ± 0.018 |
| 25% | 0.298 | 0.475 | 0.047 ± 0.025 | 0.106 ± 0.028 |
| 50% | 0.306 | 0.496 | 0.061 ± 0.037 | 0.121 ± 0.040 |
| 100% | 0.320 | 0.501 | 0.065 ± 0.037 | 0.130 ± 0.029 |

**Notes:**
- Small-HI: temporal-flow improved AUPRC in 6/6 label fractions (mean ΔAUPRC=+0.152).
- Small-HI: temporal-flow gain appears larger at high label fractions.
- Small-LI: temporal-flow improved mean AUPRC in 6/6 label fractions (mean ΔAUPRC=+0.041).
- At 1% labels, performance is degraded vs 100% but not fully collapsed on the available datasets.
- Diagnostic appendix only; not inserted into main thesis tables.
- Train labels subsampled; validation/test unchanged. Frozen pre-3h embeddings.
- Small-LI values are mean ± sample SD over model seeds 1–3 when available.
