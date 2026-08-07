# Checkpoint-ladder frozen R198 evaluation (seed 2)

Validation-only. No encoder retrain. No test access. Historical cells reused by SHA.

- Extracted only: MIXED @750 and @2250 × 3 targets (6 cells).
- Reused: EXPERT @1500/@3000 and MIXED @1500/@3000 (12 cells).
- Expert-only milestones missing on disk: {'expert_only_step_750': 'not present under expert_only/', 'expert_only_step_2250': 'not present under expert_only/'}.

## Selection views
- Fixed primary: step 3000.
- Per-target validation-selected among available ladder checkpoints (exploratory; not a test estimate).
- Single common checkpoint: mean rank across datasets (exploratory; do not average raw AUPRC).

## Per-target validation-selected
### Temporal experts only
- Small-HI: step 3000 (AUPRC=0.4175, F1@0.5=0.4683626875407697, Δ vs 3000=+0.0000; n_cand=2)
- SAML-D: step 3000 (AUPRC=0.9501, F1@0.5=0.9073337123365549, Δ vs 3000=+0.0000; n_cand=2)
- Small-LI: step 3000 (AUPRC=0.1118, F1@0.5=0.08515815085158152, Δ vs 3000=+0.0000; n_cand=2)

### InfoNCE + temporal experts
- Small-HI: step 3000 (AUPRC=0.3881, F1@0.5=0.4309677419354839, Δ vs 3000=+0.0000; n_cand=4)
- SAML-D: step 1500 (AUPRC=0.9319, F1@0.5=0.8994285714285715, Δ vs 3000=+0.0172; n_cand=4)
- Small-LI: step 750 (AUPRC=0.1107, F1@0.5=0.1384790011350738, Δ vs 3000=+0.0051; n_cand=4)

## Single common checkpoint
- Temporal experts only: step 3000 (mean_rank=1.000)
- InfoNCE + temporal experts: step 3000 (mean_rank=2.333)

## Interpretation (Q1–Q10)

1. Peak before 3000? expert: Small-HI: no (→3000); SAML-D: no (→3000); Small-LI: no (→3000); mixed: Small-HI: no (→3000); SAML-D: yes (→1500); Small-LI: yes (→750)
2. AUPRC gain from val selection (experts-only): Small-HI=+0.0000, SAML-D=+0.0000, Small-LI=+0.0000
2. AUPRC gain from val selection (InfoNCE+experts): Small-HI=+0.0000, SAML-D=+0.0172, Small-LI=+0.0051
3. F1@0.5 also improves (experts-only)? Small-HI: no/flat (sel=0.4683626875407697, 3000=0.4683626875407697); SAML-D: no/flat (sel=0.9073337123365549, 3000=0.9073337123365549); Small-LI: no/flat (sel=0.08515815085158152, 3000=0.08515815085158152)
3. F1@0.5 also improves (InfoNCE+experts)? Small-HI: no/flat (sel=0.4309677419354839, 3000=0.4309677419354839); SAML-D: yes (sel=0.8994285714285715, 3000=0.872716894977169); Small-LI: yes (sel=0.1384790011350738, 3000=0.13646532438478748)
4. Same checkpoint all datasets? expert_only=True, infonce_tf_adaptive=False
5. Single common: experts=step 3000; mixed=step 3000
6. Is step 3000 still defensible as fixed? Yes as a predeclared fixed view; exploratory selection finds earlier peaks on some cells.
7. Longer training warranted by trajectory? Possible on: expert_only/Small-HI, expert_only/SAML-D, expert_only/Small-LI, infonce_tf_adaptive/Small-HI, infonce_tf_adaptive/Small-LI
8. If longer: expert_only/Small-HI, expert_only/SAML-D, expert_only/Small-LI, infonce_tf_adaptive/Small-HI, infonce_tf_adaptive/Small-LI
9. Val selection materially narrows supervised gap? Yes on: infonce_tf_adaptive/SAML-D, infonce_tf_adaptive/Small-LI
10. Do existing collaborator-figure conclusions change? Fixed-step-3000 figures remain valid for the predeclared view; exploratory selection may shift which SSL arm/step looks strongest on some datasets.

Package: `results/diagnostics/financial_multidataset_checkpoint_ladder_eval_seed2`
Embeddings physical: `/orcd/pool/007/jthi/Multi-GNN/embeddings_archive/financial_multidataset_checkpoint_ladder_eval_seed2`
Embeddings logical: `embeddings/financial_multidataset_checkpoint_ladder_eval_seed2`

Confirmation: no encoder training, no test access, no historical-artifact modification.

