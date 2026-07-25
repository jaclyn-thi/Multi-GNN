# Supervised run summary: small_hi_legacy_supervised_gin_emlps_ports_50ep_seed3

- **Supervised mode:** legacy supervised reproduction head (IBM Multi-GNN / Egressy et al.)
- **Run kind:** standard  |  **Paper-comparable:** True
- **Model architecture:** gin (supervised_head=legacy)
- **Dataset:** Small-HI  |  **Seed:** 3
- **Graph flags:** emlps=True, reverse_mp=True, ports=True, tds=False, ego=True
- **Class weights (0,1):** (1.0000, 6.2750)
- **Optimizer / epochs:** adam / 50
- **Selection metric:** validation_minority_f1  |  **Decision rule:** argmax over two-class logits

## Best-validation selection

- **Best validation epoch:** 13
- **Validation minority F1 (argmax) at best:** 0.5434
- **Test minority F1 (argmax) at best epoch:** 0.5906  (primary reproduction metric)
- **Final-epoch test minority F1 (argmax):** 0.0000
- **Best vs final test F1 |delta|:** 0.5906  (differ substantially: True)

## Richer ranking metrics at best epoch (train-mode, diagnostic)

- validation AUROC: 0.9861  |  validation AUPRC: 0.5110
- test AUROC: 0.9852  |  test AUPRC: 0.6292

## Reproduction comparability

- **Configured to reproduce Egressy et al. setup:** True
- Legacy GINe head is a numerically validated restoration; 'configured to reproduce the corresponding Egressy et al. setup' still requires manual confirmation of data split, hyperparameters, class weights, optimizer, epoch count, selection rule, and decision rule.

## Artifacts

- Epoch history: `results/diagnostics/supervised_Small-HI_small_hi_legacy_supervised_gin_emlps_ports_50ep_seed3_epoch_history.json`
- Best-val checkpoint (use for reproduction eval): `/home/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN/saved-models/small_hi_legacy_supervised_gin_emlps_ports_50ep_seed3/checkpoint_best_val_f1.tar`
- Last checkpoint: `/home/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN/saved-models/small_hi_legacy_supervised_gin_emlps_ports_50ep_seed3/checkpoint_last.tar`
- Flat compatibility checkpoint (= last epoch): `/home/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN/saved-models/checkpoint_small_hi_legacy_supervised_gin_emlps_ports_50ep_seed3.tar`
