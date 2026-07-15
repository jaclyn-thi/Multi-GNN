# Supervised evaluation: small_li_legacy_supervised_gin_emlps_tds_seed1_scout

- **Supervised mode:** legacy supervised reproduction (IBM Multi-GNN / Egressy et al. head) (`supervised_head=legacy`)
- **Model / data:** gin / Small-LI
- **Checkpoint:** `/home/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN/saved-models/small_li_legacy_supervised_gin_emlps_tds_seed1_scout/checkpoint_best_val_f1.tar` (source: best_val_f1)
- **Checkpoint epoch:** 11  |  **selected (best-val) epoch:** 11
- **CE class weights:** `{'0': 1.0000182882773443, '1': 6.275014431494497}`
- **Validation-tuned threshold (diagnostic only, NOT paper-compatible):** 0.5689530372619629

## paper_argmax (primary reproduction metric; decision rule = argmax over two-class logits)

| Split | AUROC | AUPRC | F1 | Precision | Recall | Pos Rate |
|-------|------:|------:|---:|----------:|-------:|---------:|
| train | 0.9704 | 0.1371 | 0.1852 | 0.4382 | 0.1174 | 0.000450 |
| val | 0.9764 | 0.1805 | 0.2406 | 0.4107 | 0.1701 | 0.000585 |
| test | 0.9438 | 0.1906 | 0.2018 | 0.1816 | 0.2269 | 0.000684 |

## validation_tuned_threshold (diagnostic only; NOT paper-compatible, do not compare to paper_argmax)

| Split | F1 | Precision | Recall | Threshold |
|-------|---:|----------:|-------:|----------:|
| train | 0.1800 | 0.5307 | 0.1084 | 0.5690 |
| val | 0.2442 | 0.4704 | 0.1649 | 0.5690 |
| test | 0.2242 | 0.2306 | 0.2182 | 0.5690 |

Best log epoch by argmax Validation F1: epoch 11 (val 0.2424, test 0.1773).
