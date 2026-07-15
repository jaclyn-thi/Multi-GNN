# Appendix — Contrastive and diagnostic ablations

| Variant | Dataset | Representation | Feature stack | AUROC | AUPRC | F1 | Takeaway |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GIN baseline (20ep) | Small-HI | post-128 | embedding | 0.944 | 0.213 | 0.259 | embedding-only SSL baseline from architecture sweep |
| FNF full stack | Small-HI | post-128 | embedding+raw+morph | 0.959 | 0.277 | 0.320 | FNF contrastive variant; +raw+morph (not comparable to embedding-only baseline) |
| degree-aware edge-drop | Small-HI | post-128 | embedding | 0.926 | 0.153 | 0.240 | embedding-only negative result; no gain vs baseline |
| emb198 scout (Small-LI) | Small-LI | pre-3h emb198 scout | embedding+raw | 0.891 | 0.033 | 0.059 | one-seed diagnostic scout; not multiseed canonical |

**Notes:**
- Appendix rows are curated for interpretability; raw-only rows are not compared directly to embedding-only SSL baselines.
- Pending/manual review: queue-size contrastive variants; multi-positive contrastive variants; KNN positive variants; morphology auxiliary-loss variants.
