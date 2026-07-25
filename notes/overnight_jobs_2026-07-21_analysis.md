# Overnight jobs analysis (2026-07-21 → 2026-07-22)

Aggregate of finished overnight runs. Selection for supervised metrics is **best validation minority F1 (argmax)**; test metrics are reported at that epoch (not selected by test).

## 1. Supervised Multi-GIN+EU (ports, TDS-off, 50ep)

Paper target (Egressy et al. Multi-GIN+EU): **0.6479 ± 0.0122** minority F1.

| Seed | Job | Best val ep | Val F1 | Test F1 | P | R | AUROC | AUPRC | Final test F1 |
|------|-----|-------------|--------|---------|---|---|-------|-------|---------------|
| 1 | 18473402 | 27 | 0.603 | 0.651 | 0.650 | 0.652 | 0.986 | 0.665 | 0.547 |
| 2 | 18492402 | 43 | 0.610 | 0.685 | 0.739 | 0.639 | 0.982 | 0.656 | 0.454 |
| 3 | 18492403 | 13 | 0.543 | 0.591 | 0.531 | 0.665 | 0.985 | 0.629 | 0.000 |

**Train-time aggregate (n=3):** test F1 **0.642 ± 0.048** (Δ vs paper mean **−0.006**). Mean is near paper; sample SD is larger (paper ±0.012). Seed 3 collapses at epochs 49–50 (val/test F1 → 0); best-val selection still recovers a usable checkpoint at ep 13.

**Formal eval (seed 1 only, job 18491461):** test paper_argmax F1 **0.663**, P 0.680, R 0.647, AUROC 0.987, AUPRC 0.675.

Artifacts:
- Aggregate JSON: `results/diagnostics/supervised_Small-HI_ports_50ep_seeds1-3_aggregate.json`
- Per-seed notes: `notes/supervised_Small-HI_small_hi_legacy_supervised_gin_emlps_ports_50ep_seed{1,2,3}_summary.md`
- Formal eval: `notes/eval_small_hi_legacy_supervised_gin_emlps_ports_50ep_seed1.md`
- Slurm: `slurm-logs/hi_ports_50ep_s{1,2,3}_*.out`, `slurm-logs/hi_ports_eval_18491461.out`

## 2. GCPAL positive-set audit

Job **18490330**. Seed-only B≤2048, `tds=False`. Recommendation D: controlled positive-set ablation first; directed-chain most defensible.

- Note: `notes/gcpal_positive_set_audit.md`
- JSON: `results/diagnostics/gcpal_positive_set_audit.json`
- Log: `slurm-logs/gcpal_pos_audit_18490330.out`

## 3. Contrastive seed-retention ablation (Scout A vs B)

Same recipe otherwise: gin + ports/emlps, TDS-off, seed 2, bs 8192, 8192 negs, queue 0, T=0.5, projection, asymmetric, identity-only, 40ep. **Not** a GCPAL reproduction.

| | Scout A (identity) | Scout B (`--preserve_seed_edges`) |
|--|--------------------|-----------------------------------|
| Job | 18491335 | 18493007 |
| First-batch shared seeds | 8192 → **6654** (~81%) | 8192 → **8190** (~100%) |
| Wall (approx) | ~3.1 h | ~3.5 h |
| Embedding-only AUROC / AUPRC / F1 | 0.929 / 0.117 / 0.193 | 0.935 / 0.158 / 0.239 |
| embedding+raw F1 (ablation) | 0.203 | 0.290 |
| embedding+raw+morph F1 | 0.196 | 0.275 |

B improves AUPRC/F1 vs A but remains below the prior TDS-on seed2 40ep embedding-only reference (AUROC 0.949, AUPRC 0.245, F1 0.307).

Artifacts:
- Seed-retention note: `notes/seed_retention_view_construction_audit.md`
- Probes: `embeddings/gin_emlps_ports_tds_off_asym_proj_8192neg_queue0_40ep_seed2/probe_results.json`, `embeddings/gin_emlps_ports_tds_off_preserve_seed_asym_proj_8192neg_queue0_40ep_seed2/probe_results.json`
- Ablations: `notes/probe_feature_ablation_current_protocol_gin_40ep_seed2_tds_off.md`, `notes/probe_feature_ablation_current_protocol_gin_40ep_seed2_tds_off_preserve_seed.md`
- Logs: `slurm-logs/gin_40ep_s2_tdsoff_18491335.out`, `slurm-logs/gin_40ep_s2_pseed_18493007.out`
