# Supervised run summary: small_li_legacy_supervised_gin_emlps_tds_seed1_scout

- **Supervised mode:** legacy supervised reproduction head (IBM Multi-GNN / Egressy et al.) scout/integration run (NOT the full reproduction)
- **Run kind:** scout/dev (--testing)  |  **Paper-comparable:** False
- **Model architecture:** gin (supervised_head=legacy)
- **Dataset:** Small-LI  |  **Seed:** 1
- **Graph flags:** emlps=True, reverse_mp=True, ports=True, tds=True, ego=True
- **Class weights (0,1):** (1.0000, 6.2750)
- **Optimizer / epochs:** adam / 20
- **Selection metric:** validation_minority_f1  |  **Decision rule:** argmax over two-class logits

## Best-validation selection

- **Best validation epoch:** 11
- **Validation minority F1 (argmax) at best:** 0.2424
- **Test minority F1 (argmax) at best epoch:** 0.1773  (primary reproduction metric)
- **Final-epoch test minority F1 (argmax):** 0.2293
- **Best vs final test F1 |delta|:** 0.0520  (differ substantially: True)

## Richer ranking metrics at best epoch (train-mode, diagnostic)

- validation AUROC: 0.9760  |  validation AUPRC: 0.1834
- test AUROC: 0.9429  |  test AUPRC: 0.1772

## Reproduction comparability

- **Configured to reproduce Egressy et al. setup:** False
- Legacy GINe head (numerically validated), but this is a scout/dev run (--testing) and/or not the full upstream setup: NOT paper-comparable. Run a non-testing, upstream epoch-count job before comparing to the Egressy et al. table.

## Artifacts

- Epoch history: `results/diagnostics/supervised_Small-LI_small_li_legacy_supervised_gin_emlps_tds_seed1_scout_epoch_history.json`
- Best-val checkpoint (use for reproduction eval): `/home/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN/saved-models/small_li_legacy_supervised_gin_emlps_tds_seed1_scout/checkpoint_best_val_f1.tar`
- Last checkpoint: `/home/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN/saved-models/small_li_legacy_supervised_gin_emlps_tds_seed1_scout/checkpoint_last.tar`
- Flat compatibility checkpoint (= last epoch): `/home/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN/saved-models/checkpoint_small_li_legacy_supervised_gin_emlps_tds_seed1_scout.tar`
