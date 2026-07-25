# Partial fine-tune seed-2 — locked test evaluation (SECONDARY)

SSL-pretrained D+ with supervised partial fine-tuning.

- checkpoint: `/orcd/home/002/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN/saved-models/dplus_partial_finetune_hxxtf_seed2/checkpoint_best_val_auprc.tar`
- sha256: `5ed9503f3c3def088ec3913e4e3021d35b03c69a83b17d3d66d1ee7347a15a0b`
- best epoch: **18** (stored val AUPRC **0.599571**)
- stored val threshold: **0.46**
- online val AUPRC diagnostic: 0.598916 (neighbor-sampled; not for selection)
- test AUPRC: **0.700947**
- test AUROC: **0.985969**
- test F1@0.5: **0.697128**
- test F1@val-thr: **0.693393**
- P@100/500/1000: 0.990 / 0.976 / 0.893

Classifier was **not** refit. Test was **not** used for selection.

