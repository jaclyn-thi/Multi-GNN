# Supervised evaluation: small_hi_legacy_supervised_gin_emlps_ports_50ep_seed2

- **Supervised mode:** legacy supervised reproduction (IBM Multi-GNN / Egressy et al. head) (`supervised_head=legacy`)
- **Model / data:** gin / Small-HI
- **Checkpoint:** `/home/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN/saved-models/small_hi_legacy_supervised_gin_emlps_ports_50ep_seed2/checkpoint_best_val_f1.tar` (source: best_val_f1)
- **Checkpoint epoch:** 43  |  **selected (best-val) epoch:** 43
- **CE class weights:** `{'0': 1.0000182882773443, '1': 6.275014431494497}`
- **Validation-tuned threshold (diagnostic only, NOT paper-compatible):** 0.39992237091064453

## paper_argmax (primary reproduction metric; decision rule = argmax over two-class logits)

| Split | AUROC | AUPRC | F1 | Precision | Recall | Pos Rate |
|-------|------:|------:|---:|----------:|-------:|---------:|
| train | 0.9848 | 0.4869 | 0.5257 | 0.8718 | 0.3763 | 0.000779 |
| val | 0.9876 | 0.5719 | 0.6221 | 0.8141 | 0.5034 | 0.001072 |
| test | 0.9829 | 0.6742 | 0.7176 | 0.8147 | 0.6412 | 0.001867 |

## validation_tuned_threshold (diagnostic only; NOT paper-compatible, do not compare to paper_argmax)

| Split | F1 | Precision | Recall | Threshold |
|-------|---:|----------:|-------:|----------:|
| train | 0.5453 | 0.8508 | 0.4012 | 0.3999 |
| val | 0.6271 | 0.7668 | 0.5304 | 0.3999 |
| test | 0.7036 | 0.7465 | 0.6654 | 0.3999 |

Best log epoch by argmax Validation F1: epoch 43 (val 0.6101, test 0.6851).
