# Supervised evaluation: small_hi_legacy_supervised_gin_emlps_ports_50ep_seed1

- **Supervised mode:** legacy supervised reproduction (IBM Multi-GNN / Egressy et al. head) (`supervised_head=legacy`)
- **Model / data:** gin / Small-HI
- **Checkpoint:** `/home/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN/saved-models/small_hi_legacy_supervised_gin_emlps_ports_50ep_seed1/checkpoint_best_val_f1.tar` (source: best_val_f1)
- **Checkpoint epoch:** 27  |  **selected (best-val) epoch:** 27
- **CE class weights:** `{'0': 1.0000182882773443, '1': 6.275014431494497}`
- **Validation-tuned threshold (diagnostic only, NOT paper-compatible):** 0.3500288128852844

## paper_argmax (primary reproduction metric; decision rule = argmax over two-class logits)

| Split | AUROC | AUPRC | F1 | Precision | Recall | Pos Rate |
|-------|------:|------:|---:|----------:|-------:|---------:|
| train | 0.9843 | 0.4641 | 0.5115 | 0.8243 | 0.3708 | 0.000779 |
| val | 0.9866 | 0.5665 | 0.6101 | 0.7997 | 0.4932 | 0.001073 |
| test | 0.9870 | 0.6747 | 0.6633 | 0.6799 | 0.6474 | 0.001867 |

## validation_tuned_threshold (diagnostic only; NOT paper-compatible, do not compare to paper_argmax)

| Split | F1 | Precision | Recall | Threshold |
|-------|---:|----------:|-------:|----------:|
| train | 0.5238 | 0.7674 | 0.3976 | 0.3500 |
| val | 0.6169 | 0.7456 | 0.5261 | 0.3500 |
| test | 0.6248 | 0.5808 | 0.6760 | 0.3500 |

Best log epoch by argmax Validation F1: epoch 27 (val 0.6027, test 0.6510).
