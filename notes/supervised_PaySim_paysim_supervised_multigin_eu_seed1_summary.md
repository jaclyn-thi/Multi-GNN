# Supervised run summary: paysim_supervised_multigin_eu_seed1

- **Supervised mode:** legacy supervised reproduction head (IBM Multi-GNN / Egressy et al.)
- **Run kind:** standard  |  **Paper-comparable:** True
- **Model architecture:** gin (supervised_head=legacy)
- **Dataset:** PaySim  |  **Seed:** 1
- **Graph flags:** emlps=True, reverse_mp=True, ports=True, tds=False, ego=True, correct_reverse_edge_features=False, preserve_seed_edges=False, reverse_edge_feature_semantics=inherited_legacy
- **Class weights (0,1):** (1.0000, 6.2750)
- **Optimizer / epochs:** adam / 50
- **Selection metric:** validation_minority_f1  |  **Decision rule:** argmax over two-class logits

## Best-validation selection

- **Best validation epoch:** 3
- **Validation minority F1 (argmax) at best:** 0.2018
- **Test minority F1 (argmax) at best epoch:** 0.1974  (primary reproduction metric)
- **Final-epoch test minority F1 (argmax):** 0.1982
- **Best vs final test F1 |delta|:** 0.0007  (differ substantially: False)

## Richer ranking metrics at best epoch (train-mode, diagnostic)

- validation AUROC: 0.9408  |  validation AUPRC: 0.1578
- test AUROC: 0.9233  |  test AUPRC: 0.2096

## Reproduction comparability

- **Configured to reproduce Egressy et al. setup:** True
- Legacy GINe head is a numerically validated restoration; 'configured to reproduce the corresponding Egressy et al. setup' still requires manual confirmation of data split, hyperparameters, class weights, optimizer, epoch count, selection rule, and decision rule.

## Artifacts

- Epoch history: `results/diagnostics/supervised_PaySim_paysim_supervised_multigin_eu_seed1_epoch_history.json`
- Best-val checkpoint (use for reproduction eval): `/home/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN/saved-models/paysim_supervised_multigin_eu_seed1/checkpoint_best_val_f1.tar`
- Last checkpoint: `/home/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN/saved-models/paysim_supervised_multigin_eu_seed1/checkpoint_last.tar`
- Flat compatibility checkpoint (= last epoch): `/home/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN/saved-models/checkpoint_paysim_supervised_multigin_eu_seed1.tar`
