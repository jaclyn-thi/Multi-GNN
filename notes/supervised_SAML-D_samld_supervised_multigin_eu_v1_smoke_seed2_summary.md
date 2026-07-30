# Supervised run summary: samld_supervised_multigin_eu_v1_smoke_seed2

- **Supervised mode:** current embedding-head supervised control (NOT the Egressy/Multi-GNN baseline)
- **Run kind:** standard  |  **Paper-comparable:** False
- **Model architecture:** gin (supervised_head=embedding)
- **Dataset:** SAML-D  |  **Seed:** 2
- **Graph flags:** emlps=True, reverse_mp=True, ports=True, tds=False, ego=True, correct_reverse_edge_features=False, preserve_seed_edges=False, reverse_edge_feature_semantics=inherited_legacy
- **Class weights (0,1):** (1.0000, 6.2750)
- **Optimizer / epochs:** adam / 2
- **Selection metric:** validation_minority_f1  |  **Decision rule:** argmax over two-class logits

## Best-validation selection

- **Best validation epoch:** 1
- **Validation minority F1 (argmax) at best:** 0.6167
- **Test minority F1 (argmax) at best epoch:** 0.5877  (primary reproduction metric)
- **Final-epoch test minority F1 (argmax):** 0.0000
- **Best vs final test F1 |delta|:** 0.5877  (differ substantially: True)

## Richer ranking metrics at best epoch (train-mode, diagnostic)

- validation AUROC: 0.9993  |  validation AUPRC: 0.8912
- test AUROC: 0.9995  |  test AUPRC: 0.8957

## Reproduction comparability

- **Configured to reproduce Egressy et al. setup:** False
- This is the current embedding-head supervised control; it is NOT the Egressy/Multi-GNN baseline.

## Artifacts

- Epoch history: `results/diagnostics/supervised_SAML-D_samld_supervised_multigin_eu_v1_smoke_seed2_epoch_history.json`
- Best-val checkpoint (use for reproduction eval): `/home/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN/saved-models/samld_supervised_multigin_eu_v1_smoke_seed2/checkpoint_best_val_f1.tar`
- Last checkpoint: `/home/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN/saved-models/samld_supervised_multigin_eu_v1_smoke_seed2/checkpoint_last.tar`
- Flat compatibility checkpoint (= last epoch): `/home/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN/saved-models/checkpoint_samld_supervised_multigin_eu_v1_smoke_seed2.tar`
