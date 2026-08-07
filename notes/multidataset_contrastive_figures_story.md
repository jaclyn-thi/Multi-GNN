# Multi-dataset contrastive update — corrected story (v2)

## Executive summary

1. **Multi-domain round-robin training maintained useful in-domain representations** across Small-HI, SAML-D, and Small-LI relative to dataset-specific specialists (Figure 1). This does **not** directly test sequential catastrophic forgetting.
2. At the **fixed matched checkpoint** (1,000 updates/dataset), **temporal experts only** and **InfoNCE + temporal experts** dominate validation AUPRC; pure InfoNCE collapses on HI/LI.
3. **GBT only ≫ InfoNCE only**, but **learned GBT + experts** does not beat InfoNCE+experts or experts-only under matched exposure.
4. **Fixing GBT/expert mass at 50/50** (step 1,500 only) does not close that gap and must not be compared as matched to step 3,000.
5. Learned weights: **InfoNCE α falls (~0.209 at step 3,000)** while **GBT α rises (~0.875)** — optimizing normalized pretraining loss, not validation AUPRC.
6. In-domain HI/SAML-D/LI evaluation is **not transfer**.
7. Supervised Multi-GIN+EU is a **protocol-mismatched contextual ceiling**.

## Recommended figure order

Main 1 → 2 → 3 → 4; then supplemental S1 (F1), S5 (fixed-half), S3–S4 (losses), checkpoint table, S0/S2 as needed.

## One paragraph

We pretrained one GNN encoder on three AML graphs under several self-supervised objectives, froze it, and trained the same MLP probe on each dataset’s validation split. Expert-centric objectives yield the strongest shared encoders; Graph Barlow Twins helps as a pure contrastive substitute for InfoNCE but, jointly with temporal experts under the setups we tested, does not outperform InfoNCE-plus-experts or experts-only. Mixture weights track the pretraining loss, not downstream ranking, and supervised Multi-GIN comparisons remain informative only as a mismatched contextual benchmark.

## Claims that should not be made

- Calling in-domain HI/SAML-D/LI scores “transfer.”
- That InfoNCE+experts universally beats experts-only (false at fixed step-3,000 AUPRC on SAML-D and LI).
- That GBT+experts is competitive with InfoNCE+experts under matched exposure.
- That learned α/β maximize validation AUPRC.
- That supervised gaps are feature/protocol-matched.
- That best-checkpoint or F1@val-threshold results are test estimates.
- Averaging AUPRC across datasets as equal difficulty.

## Missing experimental cells (named)

- InfoNCE-only @ step 1,500 (multi-dataset)
- Fixed 50/50 GBT+experts @ step 3,000
- Matched two-domain encoder evaluated on Small-LI

Path-level frozen-eval cell misses in this regeneration: 0.

## Suggested supervised-gap language

> Relative to dataset-specific supervised Multi-GIN+EU reproductions (different feature schema and selection protocol; Small-LI supervised is seed 1 / TDS-on), the strongest fixed-checkpoint frozen multi-dataset SSL probes remain lower on Small-HI. Treat this as a contextual ceiling, not a controlled ablation.

See `claim_audit.csv` for **SUPPORTED** verdicts under fixed vs exploratory views.
