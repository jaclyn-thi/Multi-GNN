# Supervised run summary: paysim_native_multigin_core_v1_formal_seed2

- **Supervised mode:** legacy supervised reproduction head (IBM Multi-GNN / Egressy et al.)
- **Run kind:** standard  |  **Paper-comparable:** True
- **Model architecture:** gin (supervised_head=legacy)
- **Dataset:** PaySim  |  **Seed:** 2
- **Graph flags:** emlps=True, reverse_mp=True, ports=True, tds=False, ego=True, correct_reverse_edge_features=False, preserve_seed_edges=False, train_fit_edge_znorm=True, skip_test_eval=True, save_model=True, reverse_edge_feature_semantics=inherited_legacy
- **Class weights (0,1):** (1.0000, 6.2750)
- **Optimizer / epochs:** adam / 50
- **Selection metric:** validation_minority_f1  |  **Decision rule:** argmax over two-class logits

## Best-validation selection

- **Best validation epoch:** 42
- **Validation minority F1 (argmax) at best:** 0.7779
- **Test minority F1 (argmax) at best epoch:** n/a  (primary reproduction metric)
- **Final-epoch test minority F1 (argmax):** n/a
- **Best vs final test F1 |delta|:** n/a  (differ substantially: False)

## Richer ranking metrics at best epoch (train-mode, diagnostic)

- validation AUROC: 0.9958  |  validation AUPRC: 0.7739
- test AUROC: n/a  |  test AUPRC: n/a

## Reproduction comparability

- **Configured to reproduce Egressy et al. setup:** True
- Legacy GINe head is a numerically validated restoration; 'configured to reproduce the corresponding Egressy et al. setup' still requires manual confirmation of data split, hyperparameters, class weights, optimizer, epoch count, selection rule, and decision rule.

## Artifacts

- Epoch history: `results/diagnostics/supervised_PaySim_paysim_native_multigin_core_v1_formal_seed2_epoch_history.json`
- Best-val checkpoint (use for reproduction eval): `/home/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN/saved-models/paysim_native_multigin_core_v1_formal_seed2/checkpoint_best_val_f1.tar`
- Last checkpoint: `/home/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN/saved-models/paysim_native_multigin_core_v1_formal_seed2/checkpoint_last.tar`
- Flat compatibility checkpoint (= last epoch): `/home/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN/saved-models/checkpoint_paysim_native_multigin_core_v1_formal_seed2.tar`
