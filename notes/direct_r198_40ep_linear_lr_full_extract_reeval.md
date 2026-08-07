# DIRECT_R198 40ep linear-LR sweep — corrected full-subgraph re-eval

> **Official eval command:** `python scripts/official_direct_r198_collaborator_eval.py`  
> (see `notes/direct_r198_official_collaborator_eval.md`). Collaborator tables refuse `protocol != full_subgraph`.

> **INVALID / DIAGNOSTIC (do not use for collaborator claims):** prior seed-only validation metrics under `results/diagnostics/direct_r198_tfmoe_40ep_linear_lr_sweep/` and `notes/direct_r198_tfmoe_40ep_linear_lr_sweep.md`. Those extracts treated loader `input_id` as a global edge index instead of `split_inds[input_id]`, so validation IDs landed in the train range (~100% train∩val overlap). Even ID-fixed seed-only remains provisional. Retraining was not redone; only extraction/probe.

## Protocol (matched to 10ep full-extract analysis)

- Extractor: `scripts/extract_direct_r198_full_cell.py` (full GraphModule R198)
- Embeddings root: `embeddings/direct_r198_40ep_linear_lr_full_extract/` (does **not** overwrite `embeddings/<run>_epochXX/` seed-only artifacts)
- Probe: PaperStyleMLP, 20 epochs, lr=1e-3, bs=8192, seed=2; features R198+X+TF; ranking metrics from best-val-AUPRC probe epoch; also report last probe-epoch train/val BCE
- ID gate before probe: train∩val=0, all val IDs above ref train max, no seed-only train-range signature; Jaccard≥0.999 and relative |n−n_ref|/n_ref≤1% vs prior full extract (exact set equality not required)
- Cells complete: **30/30**
- Matched SSL epoch grid: **3, 10, 20, 30, 40** (plots break on missing points)

## Primary results (R198+X+TF → PaperStyleMLP)

| Arm | Peak LR | SSL ep | Val AUPRC | F1@0.5 | F1@val-thr | Final train BCE | Final val BCE | Verify |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| DIRECT_H | 0.001 | 3 | 0.4782 | 0.5188 | 0.5201 | 0.0027 | 0.0041 | ok |
| DIRECT_H | 0.001 | 10 | 0.4795 | 0.5160 | 0.5179 | 0.0027 | 0.0039 | ok |
| DIRECT_H | 0.001 | 20 | 0.4815 | 0.5115 | 0.5148 | 0.0027 | 0.0043 | ok |
| DIRECT_H | 0.001 | 30 | 0.4491 | 0.4861 | 0.4968 | 0.0026 | 0.0042 | ok |
| DIRECT_H | 0.001 | 40 | 0.4592 | 0.4985 | 0.5053 | 0.0026 | 0.0042 | ok |
| DIRECT_H | 0.002 | 3 | 0.4583 | 0.4987 | 0.5047 | 0.0026 | 0.0041 | ok |
| DIRECT_H | 0.002 | 10 | 0.4776 | 0.5144 | 0.5161 | 0.0026 | 0.0040 | ok |
| DIRECT_H | 0.002 | 20 | 0.3825 | 0.4087 | 0.4291 | 0.0026 | 0.0049 | ok |
| DIRECT_H | 0.002 | 30 | 0.4369 | 0.4944 | 0.5011 | 0.0026 | 0.0040 | ok |
| DIRECT_H | 0.002 | 40 | 0.4480 | 0.4991 | 0.5051 | 0.0026 | 0.0042 | ok |
| DIRECT_H | 0.006213 | 3 | 0.4503 | 0.4884 | 0.4965 | 0.0027 | 0.0041 | ok |
| DIRECT_H | 0.006213 | 10 | 0.4933 | 0.5026 | 0.5277 | 0.0027 | 0.0037 | ok |
| DIRECT_H | 0.006213 | 20 | 0.4290 | 0.4586 | 0.4730 | 0.0027 | 0.0040 | ok |
| DIRECT_H | 0.006213 | 30 | 0.4404 | 0.4337 | 0.4858 | 0.0028 | 0.0040 | ok |
| DIRECT_H | 0.006213 | 40 | 0.4330 | 0.4587 | 0.4802 | 0.0028 | 0.0040 | ok |
| DIRECT_H_TFMOE | 0.001 | 3 | 0.5088 | 0.5195 | 0.5592 | 0.0027 | 0.0038 | ok |
| DIRECT_H_TFMOE | 0.001 | 10 | 0.5306 | 0.5232 | 0.5686 | 0.0027 | 0.0036 | ok |
| DIRECT_H_TFMOE | 0.001 | 20 | 0.4845 | 0.5134 | 0.5180 | 0.0026 | 0.0041 | ok |
| DIRECT_H_TFMOE | 0.001 | 30 | 0.4704 | 0.5059 | 0.5154 | 0.0025 | 0.0041 | ok |
| DIRECT_H_TFMOE | 0.001 | 40 | 0.4565 | 0.4770 | 0.4976 | 0.0026 | 0.0043 | ok |
| DIRECT_H_TFMOE | 0.002 | 3 | 0.5381 | 0.5576 | 0.5699 | 0.0026 | 0.0037 | ok |
| DIRECT_H_TFMOE | 0.002 | 10 | 0.5481 | 0.5360 | 0.5961 | 0.0025 | 0.0036 | ok |
| DIRECT_H_TFMOE | 0.002 | 20 | 0.5441 | 0.5630 | 0.5767 | 0.0024 | 0.0038 | ok |
| DIRECT_H_TFMOE | 0.002 | 30 | 0.5359 | 0.5515 | 0.5851 | 0.0024 | 0.0038 | ok |
| DIRECT_H_TFMOE | 0.002 | 40 | 0.5257 | 0.5135 | 0.5781 | 0.0024 | 0.0037 | ok |
| DIRECT_H_TFMOE | 0.006213 | 3 | 0.5373 | 0.5482 | 0.5701 | 0.0026 | 0.0036 | ok |
| DIRECT_H_TFMOE | 0.006213 | 10 | 0.5282 | 0.5108 | 0.5640 | 0.0024 | 0.0041 | ok |
| DIRECT_H_TFMOE | 0.006213 | 20 | 0.5478 | 0.5387 | 0.5786 | 0.0025 | 0.0035 | ok |
| DIRECT_H_TFMOE | 0.006213 | 30 | 0.4959 | 0.4473 | 0.5241 | 0.0024 | 0.0039 | ok |
| DIRECT_H_TFMOE | 0.006213 | 40 | 0.4976 | 0.4713 | 0.5287 | 0.0024 | 0.0040 | ok |

## Trajectory answers (corrected full-extract only)

- Grid complete: **True**
- Per-run peaks (by val AUPRC):
  - DIRECT_H lr=0.006213: SSL epoch **10** (AUPRC=0.4933, F1@0.5=0.5026, F1@val-thr=0.5277; 5/5 epochs observed)
  - DIRECT_H_TFMOE lr=0.006213: SSL epoch **20** (AUPRC=0.5478, F1@0.5=0.5387, F1@val-thr=0.5786; 5/5 epochs observed)
  - DIRECT_H lr=0.002: SSL epoch **10** (AUPRC=0.4776, F1@0.5=0.5144, F1@val-thr=0.5161; 5/5 epochs observed)
  - DIRECT_H_TFMOE lr=0.002: SSL epoch **10** (AUPRC=0.5481, F1@0.5=0.5360, F1@val-thr=0.5961; 5/5 epochs observed)
  - DIRECT_H lr=0.001: SSL epoch **20** (AUPRC=0.4815, F1@0.5=0.5115, F1@val-thr=0.5148; 5/5 epochs observed)
  - DIRECT_H_TFMOE lr=0.001: SSL epoch **10** (AUPRC=0.5306, F1@0.5=0.5232, F1@val-thr=0.5686; 5/5 epochs observed)
- TFMOE beats DIRECT_H on AUPRC at **14/15** matched (arm×LR×epoch) checkpoints with both present

## Learning-rate comparison (corrected only)

| Arm | SSL ep | AUPRC 6.21e-3 | AUPRC 2e-3 | AUPRC 1e-3 | Δ(1e-3−2e-3) | F1@0.5 2e-3 | F1@0.5 1e-3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| DIRECT_H | 3 | 0.4503 | 0.4583 | 0.4782 | 0.0200 | 0.4987 | 0.5188 |
| DIRECT_H | 10 | 0.4933 | 0.4776 | 0.4795 | 0.0019 | 0.5144 | 0.5160 |
| DIRECT_H | 20 | 0.4290 | 0.3825 | 0.4815 | 0.0990 | 0.4087 | 0.5115 |
| DIRECT_H | 30 | 0.4404 | 0.4369 | 0.4491 | 0.0122 | 0.4944 | 0.4861 |
| DIRECT_H | 40 | 0.4330 | 0.4480 | 0.4592 | 0.0113 | 0.4991 | 0.4985 |
| DIRECT_H_TFMOE | 3 | 0.5373 | 0.5381 | 0.5088 | -0.0293 | 0.5576 | 0.5195 |
| DIRECT_H_TFMOE | 10 | 0.5282 | 0.5481 | 0.5306 | -0.0175 | 0.5360 | 0.5232 |
| DIRECT_H_TFMOE | 20 | 0.5478 | 0.5441 | 0.4845 | -0.0596 | 0.5630 | 0.5134 |
| DIRECT_H_TFMOE | 30 | 0.4959 | 0.5359 | 0.4704 | -0.0655 | 0.5515 | 0.5059 |
| DIRECT_H_TFMOE | 40 | 0.4976 | 0.5257 | 0.4565 | -0.0691 | 0.5135 | 0.4770 |

## DIRECT_H vs DIRECT_H_TFMOE (corrected only)

| Peak LR | SSL ep | AUPRC H | AUPRC TFMOE | Δ (TF−H) | F1@0.5 H | F1@0.5 TF | Final val BCE H | Final val BCE TF |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.006213 | 3 | 0.4503 | 0.5373 | 0.0871 | 0.4884 | 0.5482 | 0.0041 | 0.0036 |
| 0.006213 | 10 | 0.4933 | 0.5282 | 0.0349 | 0.5026 | 0.5108 | 0.0037 | 0.0041 |
| 0.006213 | 20 | 0.4290 | 0.5478 | 0.1187 | 0.4586 | 0.5387 | 0.0040 | 0.0035 |
| 0.006213 | 30 | 0.4404 | 0.4959 | 0.0555 | 0.4337 | 0.4473 | 0.0040 | 0.0039 |
| 0.006213 | 40 | 0.4330 | 0.4976 | 0.0646 | 0.4587 | 0.4713 | 0.0040 | 0.0040 |
| 0.002 | 3 | 0.4583 | 0.5381 | 0.0798 | 0.4987 | 0.5576 | 0.0041 | 0.0037 |
| 0.002 | 10 | 0.4776 | 0.5481 | 0.0705 | 0.5144 | 0.5360 | 0.0040 | 0.0036 |
| 0.002 | 20 | 0.3825 | 0.5441 | 0.1615 | 0.4087 | 0.5630 | 0.0049 | 0.0038 |
| 0.002 | 30 | 0.4369 | 0.5359 | 0.0989 | 0.4944 | 0.5515 | 0.0040 | 0.0038 |
| 0.002 | 40 | 0.4480 | 0.5257 | 0.0777 | 0.4991 | 0.5135 | 0.0042 | 0.0037 |
| 0.001 | 3 | 0.4782 | 0.5088 | 0.0306 | 0.5188 | 0.5195 | 0.0041 | 0.0038 |
| 0.001 | 10 | 0.4795 | 0.5306 | 0.0511 | 0.5160 | 0.5232 | 0.0039 | 0.0036 |
| 0.001 | 20 | 0.4815 | 0.4845 | 0.0030 | 0.5115 | 0.5134 | 0.0043 | 0.0041 |
| 0.001 | 30 | 0.4491 | 0.4704 | 0.0212 | 0.4861 | 0.5059 | 0.0042 | 0.0041 |
| 0.001 | 40 | 0.4592 | 0.4565 | -0.0027 | 0.4985 | 0.4770 | 0.0042 | 0.0043 |

## Best corrected cell

- **DIRECT_H_TFMOE** peak_lr=0.002 SSL epoch 10: AUPRC=0.5481, F1@0.5=0.5360, F1@val-thr=0.5961, final val BCE=0.0036

## Artifacts

- **out_dir:** `results/diagnostics/direct_r198_40ep_linear_lr_full_extract_reeval`
- **embeddings:** `embeddings/direct_r198_40ep_linear_lr_full_extract/`
- **cells:** `results/diagnostics/direct_r198_40ep_linear_lr_full_extract_reeval/cells`
- **csv:** `results/diagnostics/direct_r198_40ep_linear_lr_full_extract_reeval/corrected_cells.csv`
- **aggregate_json:** `results/diagnostics/direct_r198_40ep_linear_lr_full_extract_reeval/aggregate.json`
- **lr_table:** `results/diagnostics/direct_r198_40ep_linear_lr_full_extract_reeval/lr_comparison_table.json`
- **h_vs_tf_table:** `results/diagnostics/direct_r198_40ep_linear_lr_full_extract_reeval/direct_h_vs_tfmoe_table.json`
- **trajectory_answers:** `results/diagnostics/direct_r198_40ep_linear_lr_full_extract_reeval/trajectory_answers.json`
- **fig_auprc:** `results/diagnostics/direct_r198_40ep_linear_lr_full_extract_reeval/figures/01_val_auprc_vs_epoch.png`
- **fig_f1_0.5:** `results/diagnostics/direct_r198_40ep_linear_lr_full_extract_reeval/figures/02_val_f1_at_0.5_vs_epoch.png`
- **fig_f1_val_thr:** `results/diagnostics/direct_r198_40ep_linear_lr_full_extract_reeval/figures/03_val_f1_at_val_thr_vs_epoch.png`
- **fig_bce:** `results/diagnostics/direct_r198_40ep_linear_lr_full_extract_reeval/figures/04_final_probe_val_bce_vs_epoch.png`
- **fig_combined:** `results/diagnostics/direct_r198_40ep_linear_lr_full_extract_reeval/figures/05_train_loss_lr_and_downstream_auprc.png`
- **note:** `notes/direct_r198_40ep_linear_lr_full_extract_reeval.md`
- **seed_only_invalid_marker:** `results/diagnostics/direct_r198_tfmoe_40ep_linear_lr_sweep/SEED_ONLY_VALIDATION_METRICS_INVALID.md`
- **seed_only_note_bannered:** `notes/direct_r198_tfmoe_40ep_linear_lr_sweep.md`
- **proposed_fix:** `scripts/extract_direct_r198_seed_only_cell.py (split_inds[input_id])`
- **proposed_test:** `tests/test_seed_only_val_edge_id_resolution.py`

## Proposed seed-only ID fix (not used for these numbers)

Smallest fix: in `scripts/extract_direct_r198_seed_only_cell.py` `extract_split_seed_only`, resolve `batch_edge_inds = split_inds[input_id]` then `edge_attr[batch_edge_inds, 0]` / `y[batch_edge_inds]` (same as full extract). Do **not** change training's `get_hetero_seed_edge_ids` without auditing train loaders (all-edge seeds). Regression: `tests/test_seed_only_val_edge_id_resolution.py`.

