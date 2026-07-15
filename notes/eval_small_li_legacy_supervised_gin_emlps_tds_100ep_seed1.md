# Supervised evaluation: small_li_legacy_supervised_gin_emlps_tds_100ep_seed1

- **Supervised mode:** legacy supervised reproduction (IBM Multi-GNN / Egressy et al. head) (`supervised_head=legacy`)
- **Model / data:** gin / Small-LI
- **Checkpoint:** `/home/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN/saved-models/small_li_legacy_supervised_gin_emlps_tds_100ep_seed1/checkpoint_best_val_f1.tar` (source: best_val_f1)
- **Checkpoint epoch:** 35  |  **selected (best-val) epoch:** 35
- **CE class weights:** `{'0': 1.0000182882773443, '1': 6.275014431494497}`
- **Validation-tuned threshold (diagnostic only, NOT paper-compatible):** 0.3260100483894348

## paper_argmax (primary reproduction metric; decision rule = argmax over two-class logits)

| Split | AUROC | AUPRC | F1 | Precision | Recall | Pos Rate |
|-------|------:|------:|---:|----------:|-------:|---------:|
| train | 0.9762 | 0.1905 | 0.2275 | 0.8245 | 0.1320 | 0.000450 |
| val | 0.9757 | 0.2154 | 0.2760 | 0.6000 | 0.1792 | 0.000585 |
| test | 0.9587 | 0.2921 | 0.3570 | 0.5852 | 0.2569 | 0.000684 |

## validation_tuned_threshold (diagnostic only; NOT paper-compatible, do not compare to paper_argmax)

| Split | F1 | Precision | Recall | Threshold |
|-------|---:|----------:|-------:|----------:|
| train | 0.2478 | 0.6904 | 0.1510 | 0.3260 |
| val | 0.2902 | 0.4953 | 0.2052 | 0.3260 |
| test | 0.3553 | 0.4455 | 0.2955 | 0.3260 |

Best log epoch by argmax Validation F1: epoch 1 (val 0.0000, test 0.0000).
