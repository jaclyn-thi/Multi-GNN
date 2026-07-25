# Supervised run summary: small_hi_multignn_supervised_parity_smoke_tds_off_seed1

- **Supervised mode:** legacy supervised reproduction head (IBM Multi-GNN / Egressy et al.) scout/integration run (NOT the full reproduction)
- **Run kind:** scout/dev (--testing)  |  **Paper-comparable:** False
- **Model architecture:** gin (supervised_head=legacy)
- **Dataset:** Small-HI  |  **Seed:** 1
- **Graph flags:** emlps=True, reverse_mp=True, ports=True, tds=False, ego=True
- **Class weights (0,1):** (1.0000, 6.2750)
- **Optimizer / epochs:** adam / 1
- **Selection metric:** validation_minority_f1  |  **Decision rule:** argmax over two-class logits

## Best-validation selection

- **Best validation epoch:** 1
- **Validation minority F1 (argmax) at best:** 0.0000
- **Test minority F1 (argmax) at best epoch:** 0.0000  (primary reproduction metric)
- **Final-epoch test minority F1 (argmax):** 0.0000
- **Best vs final test F1 |delta|:** 0.0000  (differ substantially: False)

## Richer ranking metrics at best epoch (train-mode, diagnostic)

- validation AUROC: 0.9677  |  validation AUPRC: 0.0888
- test AUROC: 0.9728  |  test AUPRC: 0.1271

## Reproduction comparability

- **Configured to reproduce Egressy et al. setup:** False
- Legacy GINe head (numerically validated), but this is a scout/dev run (--testing) and/or not the full upstream setup: NOT paper-comparable. Run a non-testing, upstream epoch-count job before comparing to the Egressy et al. table.

## Artifacts

- Epoch history: `results/diagnostics/supervised_Small-HI_small_hi_multignn_supervised_parity_smoke_tds_off_seed1_epoch_history.json`
- Best-val checkpoint (use for reproduction eval): `/home/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN/saved-models/small_hi_multignn_supervised_parity_smoke_tds_off_seed1/checkpoint_best_val_f1.tar`
- Last checkpoint: `/home/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN/saved-models/small_hi_multignn_supervised_parity_smoke_tds_off_seed1/checkpoint_last.tar`
- Flat compatibility checkpoint (= last epoch): `/home/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN/saved-models/checkpoint_small_hi_multignn_supervised_parity_smoke_tds_off_seed1.tar`
