# Supervised evaluation: small_hi_legacy_supervised_gin_emlps_tds_100ep_seed1

- **Supervised mode:** legacy supervised reproduction (IBM Multi-GNN / Egressy et al. head) (`supervised_head=legacy`)
- **Model / data:** gin / Small-HI
- **Checkpoint:** `/home/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN/saved-models/small_hi_legacy_supervised_gin_emlps_tds_100ep_seed1/checkpoint_best_val_f1.tar` (source: best_val_f1)
- **Checkpoint epoch:** 24  |  **selected (best-val) epoch:** 24
- **CE class weights:** `{'0': 1.0000182882773443, '1': 6.275014431494497}`
- **Validation-tuned threshold (diagnostic only, NOT paper-compatible):** 0.5688579082489014

## paper_argmax (primary reproduction metric; decision rule = argmax over two-class logits)

| Split | AUROC | AUPRC | F1 | Precision | Recall | Pos Rate |
|-------|------:|------:|---:|----------:|-------:|---------:|
| train | 0.9836 | 0.4505 | 0.5140 | 0.7730 | 0.3850 | 0.000779 |
| val | 0.9864 | 0.5377 | 0.5602 | 0.6550 | 0.4894 | 0.001073 |
| test | 0.9837 | 0.6388 | 0.5394 | 0.4530 | 0.6667 | 0.001867 |

## validation_tuned_threshold (diagnostic only; NOT paper-compatible, do not compare to paper_argmax)

| Split | F1 | Precision | Recall | Threshold |
|-------|---:|----------:|-------:|----------:|
| train | 0.5045 | 0.8019 | 0.3680 | 0.5689 |
| val | 0.5699 | 0.7061 | 0.4778 | 0.5689 |
| test | 0.5723 | 0.5097 | 0.6524 | 0.5689 |

Best log epoch by argmax Validation F1: epoch 1 (val 0.0000, test 0.0000).
