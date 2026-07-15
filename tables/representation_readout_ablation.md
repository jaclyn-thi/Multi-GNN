# Table 4 — Representation readout ablation

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
