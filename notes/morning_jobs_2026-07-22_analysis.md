# Morning jobs analysis (2026-07-22)

All listed jobs finished successfully. Thesis contrastive C/D are **not** paper Multi-GIN+EU or GCPAL reproductions.

## 1. Formal supervised Multi-GIN+EU (ports, TDS-off, 50ep)

Jobs **18513757** (seed2), **18513758** (seed3) + prior seed1 eval.

| Seed | Epoch | Test paper_argmax F1 | P | R |
|------|------:|---------------------:|--:|--:|
| 1 | 27 | 0.663 | 0.680 | 0.647 |
| 2 | 43 | 0.718 | 0.815 | 0.641 |
| 3 | 13 | 0.598 | 0.546 | 0.662 |

**Formal aggregate:** 0.660 ± 0.060 (median 0.663). Paper 0.648 ± 0.012 → mean reproduced; low variance **not**.

- Aggregate: `notes/eval_small_hi_legacy_supervised_gin_emlps_ports_50ep_seeds1-3_formal_aggregate.md`
- JSON: `results/diagnostics/eval_small_hi_legacy_supervised_gin_emlps_ports_50ep_seeds1-3_formal_aggregate.json`
- Per-seed: `notes/eval_small_hi_legacy_supervised_gin_emlps_ports_50ep_seed{1,2,3}.md`

## 2. Contrastive corrected reverse-TDS (C / D) vs A / B / inherited

| Condition | Emb AUROC | Emb AUPRC | Emb F1 | emb+raw F1 | Shared seeds |
|-----------|----------:|----------:|-------:|-----------:|-------------:|
| Inherited TDS-on (malformed reverse) | 0.949 | 0.245 | 0.307 | 0.347 | — |
| A TDS-off | 0.929 | 0.117 | 0.193 | 0.203 | 6654/8192 |
| B TDS-off + preserve | 0.935 | 0.158 | 0.239 | 0.290 | 8190/8192 |
| **C corrected TDS** | 0.950 | 0.135 | 0.194 | 0.228 | 6681/8192 |
| **D corrected + preserve** | **0.963** | **0.242** | **0.320** | **0.324** | 8191/8192 |

**Takeaway:** Corrected TDS alone (C) does not beat inherited TDS-on. Corrected + seed preservation (D) matches/slightly exceeds inherited embedding F1 and is the strongest of the five on AUROC; inherited still leads emb+raw F1 (0.347 vs D 0.324).

Artifacts:
- C probe: `embeddings/gin_emlps_ports_tds_corrected_asym_proj_8192neg_queue0_40ep_seed2/probe_results.json`
- D probe: `embeddings/gin_emlps_ports_tds_corrected_preserve_seed_asym_proj_8192neg_queue0_40ep_seed2/probe_results.json`
- Ablations: `notes/probe_feature_ablation_current_protocol_gin_40ep_seed2_tds_corrected.md`, `..._preserve_seed.md`
- Logs: `slurm-logs/gin_40ep_s2_corrC_18514683.out`, `slurm-logs/gin_40ep_s2_corrD_18514684.out`
- Audit: `notes/correct_reverse_edge_features_audit.md`

## 3. GCPAL multi-batch KNN diagnostic

Job **18515104**. 64 batches, **83** minority anchors (target 100 not reached). Decision **D** (insufficient): best minority P@15 ≈ 0.003 despite ~4× lift over base rate; batch↔global cache overlap ≈ 0.0006; no multi-batch neighbor stability (seeds rarely recur).

- Note: `notes/gcpal_positive_set_multibatch_audit.md`
- JSON: `results/diagnostics/gcpal_positive_set_multibatch_audit.json`
