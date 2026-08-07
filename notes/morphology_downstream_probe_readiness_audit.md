# Morphology downstream probe readiness (read-only)

> Twin: [`results/diagnostics/morphology_downstream_probe_readiness_audit.json`](../results/diagnostics/morphology_downstream_probe_readiness_audit.json)  
> Package: [`results/diagnostics/morphology_downstream_probe_readiness_audit/`](../results/diagnostics/morphology_downstream_probe_readiness_audit/)  
> **No encoder/expert training, no probe fits, no jobs, no full embedding/dataset loads.**

---

## Final candidates (at most two)

### 1. Train-static triangle participation (primary)

Edge target on the **TRAIN graph only**:  
\(y = \log(1 + T[s] + T[r])\) where \(T[v]\) is the undirected triangle count of endpoint \(v\).

**Why independent enough:** Current multi-dataset R198 pretraining uses shared-core inputs (`Timestamp`, `Amount Received`, ports, TDS) and three causal **TF** experts only. Triangles are not encoder inputs, not TF targets, and not expert targets in Phase-4 runs. They are higher-order relative to ordinary 1–2 hop GIN aggregation, so predictability from frozen R198 is a meaningful structural-content test—not reconstructing a pretraining head.

**Cache:** No global triangle cache. Tier-1 triangles are batch-local only. One CPU precompute on Small-HI train edges is required (~MB node table).

**Frame:** log-regression (primary); optional train-quantile 4-way classification.

### 2. Train-static degree fan (control, not primary claim)

Use existing `degree_fan_features_train_static` / `morphology_cache/Small-HI/*_node_morphology.csv` with **train-only** degrees.

**Why second:** Unused as TF/expert in current R198 pretrain and fully cached, but message-passing **already** exposes degree-like signal—so beating baselines here is a sanity check. Prefer residualizing triangle performance on degree to show **incremental** higher-order content.

---

## Comparison snapshot

| Family | Verdict |
|--------|---------|
| Degree/fan | Control; cached; weaker independence |
| Triangle/clustering | **Best primary** (clustering ≈ substitute; SSL clustering objective previously collapsed—different question) |
| Flow balance | Cached; entangled with amount (encoder + TF amount expert) — not primary |
| Approx BC (k=256) | Cached on HI; **future work** (exact BC blocked; prior SSL harm) |

**Betweenness:** remain **future work**.

---

## Exclusions honored

- Three TF MoE targets → circular sanity only  
- `Amount Received` → encoder input  
- Cumulative val/test morph caches → do not use as-shipped for val targets without train-static rebuild  
- Exact betweenness → unsafe/expensive  

---

## Minimal evaluation (design only — not run)

- **Encoders:** `EXPERT_ONLY@3000` and `MIXED_LONG@3000` Small-HI frozen R198 (existing extracts).  
- **Data:** Small-HI matched train/val EdgeIDs; **no test**.  
- **Probe:** PaperStyleMLP 198→128→1; 20 ep; lr 1e-3; bs 8192; seed 2.  
- **Baselines:** mean · shared-core raw MLP · degree_fan MLP · R198.  
- **Success:** R198 Spearman on triangles > raw & mean; residual Spearman after degree > 0.05.  
- **Budget:** ≤8 cells; CPU; &lt;1 GiB new storage; one triangle precompute job.  

Details: `proposed_minimal_probe.json`.

---

## Confirmations

Nothing trained, probed, submitted, or loaded at full embedding/dataset scale. Metadata/headers/notes/diagnostics only.
