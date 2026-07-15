# Thesis figure plan

Recommended figures from **existing** diagnostics. No polished final figures generated in this pass unless noted.

---

## Figure 1 — SSL training and evaluation pipeline

| Field | Value |
|-------|-------|
| **Scientific question** | What is the end-to-end path from graph SSL pretraining to edge-level detection metrics? |
| **Source data** | Protocol docs: `notes/current_protocol_recent_runs_summary.md`, `notes/results.md`; code: `train.py`, `linear_probe.py`, extraction scripts |
| **Axes / layout** | Flow diagram: AML graph → GINe SSL (contrastive) → checkpoint → embedding export (post-128 / pre-3h) → feature stack → logistic probe → threshold tune on val → test metrics |
| **Main vs appendix** | **Main** (methods) |
| **Data exist?** | Yes (conceptual); no single JSON |
| **Plotting script needed?** | Yes — TikZ/diagram tool or simple matplotlib/Graphviz scaffold |

---

## Figure 2 — Pre-3h vs post-embedding readout

| Field | Value |
|-------|-------|
| **Scientific question** | Where is information lost when 3×hidden (198-d) tensors are compressed through `embedding_head` to 128-d? |
| **Source data** | `pre_embedding_3h_vs_post_embedding_small_hi.json`, `pre_embedding_3h_vs_post_embedding_small_li_multiseed.json` |
| **Axes / layout** | Block diagram: GINe layers → 3h tensor → [optional head] → 128-d export → probe; annotate dims and paired-probe policy |
| **Main vs appendix** | **Main** |
| **Data exist?** | Yes |
| **Plotting script needed?** | Yes (diagram); metrics table inset optional from registry |

---

## Figure 3 — Small-LI supervised validation/test F1 vs epoch (late collapse)

| Field | Value |
|-------|-------|
| **Scientific question** | Why must legacy supervised LI use best-val checkpoint rather than last epoch? |
| **Source data** | `supervised_Small-LI_small_li_legacy_supervised_gin_emlps_tds_100ep_seed1_epoch_history.json` |
| **Axes** | x: epoch (1–100); y: minority-class F1 (val + test); vertical line at epoch 35 |
| **Main vs appendix** | **Main** (Small-LI supervised subsection) |
| **Data exist?** | Yes — epoch series in JSON |
| **Plotting script needed?** | Yes — lightweight matplotlib (~30 lines); **trivial** |

---

## Figure 4 — Small-LI three-seed pre-3h vs post-128 AUPRC

| Field | Value |
|-------|-------|
| **Scientific question** | Does pre-3h AUPRC advantage replicate across seeds on Small-LI? |
| **Source data** | `pre_embedding_3h_vs_post_embedding_small_li_multiseed.json` (`per_seed` + `aggregate`) |
| **Axes** | Grouped bar: seeds 1–3 × {post-128, pre-3h}; stacks: embedding-only and +raw (two panels or hue) |
| **Main vs appendix** | **Main** |
| **Data exist?** | Yes |
| **Plotting script needed?** | Yes — straightforward bar chart from JSON |

---

## Figure 5 — Alert-budget comparison (top-K precision / lift)

| Field | Value |
|-------|-------|
| **Scientific question** | At fixed analyst budget (K=100/500/1000), how do SSL and legacy supervised rank alerts? |
| **Source data** | SSL: `alert_budget_metrics_current_protocol.json`, multiseed pre/post JSON; Supervised: `small_li_legacy_supervised_alert_budget_seed1.json` |
| **Axes** | Panels for Small-HI vs Small-LI; bars for P@K and/or lift@K; separate supervised vs SSL series |
| **Main vs appendix** | **Main** (operational metrics) |
| **Data exist?** | Partial — HI SSL complete; LI SSL multiseed has lift@100; legacy LI complete; FNF full-stack HI missing P@K |
| **Plotting script needed?** | Yes — after confirming comparable row set in registry |

---

## Figure 6 — Architecture comparison (encoder swap)

| Field | Value |
|-------|-------|
| **Scientific question** | Does encoder choice matter under shared SSL+probe protocol? |
| **Source data** | `architecture_sweep_shared_probe_weights.json` |
| **Axes** | Bar chart: AUROC / AUPRC / F1 for GIN, GAT, PNA, RGCN (embedding-only) |
| **Main vs appendix** | **Appendix** (footnote: PNA not matched); short call-out in main text |
| **Data exist?** | Yes |
| **Plotting script needed?** | Yes — trivial bar chart |

---

## Figure 7 — Feature-stack ablation (optional)

| Field | Value |
|-------|-------|
| **Scientific question** | Which probe features complement SSL embeddings on Small-HI? |
| **Source data** | `probe_feature_ablation_current_protocol_comparison.json` |
| **Axes** | Heatmap or grouped bars: 6 feature modes × {AUROC, AUPRC, F1} for key runs (20ep s1, 40ep s2, FNF) |
| **Main vs appendix** | **Appendix** |
| **Data exist?** | Yes |
| **Plotting script needed?** | Yes |

---

## Priority order for implementation

1. **Figure 3** — data ready, supports key supervised narrative  
2. **Figure 4** — multiseed SSL headline  
3. **Figure 5** — operational impact (partial data OK with explicit gaps)  
4. **Figures 1–2** — methods diagrams  
5. **Figures 6–7** — appendix support  

## Scripts to add (not created yet)

| Script | Output |
|--------|--------|
| `scripts/plot_small_li_supervised_epoch_curve.py` | PDF/PNG from epoch_history JSON |
| `scripts/plot_small_li_prepost_multiseed_auprc.py` | bar chart from multiseed JSON |
| `scripts/plot_alert_budget_comparison.py` | K=100/500/1000 panel figure from registry filter |

**Note:** Per task instructions, polished final figures were not rendered in this pass; all underlying numeric data are present in cited JSON files.
