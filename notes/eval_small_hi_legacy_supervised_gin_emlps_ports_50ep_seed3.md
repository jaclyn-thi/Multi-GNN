# Supervised evaluation: small_hi_legacy_supervised_gin_emlps_ports_50ep_seed3

- **Supervised mode:** legacy supervised reproduction (IBM Multi-GNN / Egressy et al. head) (`supervised_head=legacy`)
- **Model / data:** gin / Small-HI
- **Checkpoint:** `/home/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN/saved-models/small_hi_legacy_supervised_gin_emlps_ports_50ep_seed3/checkpoint_best_val_f1.tar` (source: best_val_f1)
- **Checkpoint epoch:** 13  |  **selected (best-val) epoch:** 13
- **CE class weights:** `{'0': 1.0000182882773443, '1': 6.275014431494497}`
- **Validation-tuned threshold (diagnostic only, NOT paper-compatible):** 0.6663029789924622

## paper_argmax (primary reproduction metric; decision rule = argmax over two-class logits)

| Split | AUROC | AUPRC | F1 | Precision | Recall | Pos Rate |
|-------|------:|------:|---:|----------:|-------:|---------:|
| train | 0.9844 | 0.4188 | 0.4836 | 0.6702 | 0.3783 | 0.000779 |
| val | 0.9862 | 0.5116 | 0.5457 | 0.6122 | 0.4923 | 0.001073 |
| test | 0.9851 | 0.6399 | 0.5984 | 0.5458 | 0.6623 | 0.001867 |

## validation_tuned_threshold (diagnostic only; NOT paper-compatible, do not compare to paper_argmax)

| Split | F1 | Precision | Recall | Threshold |
|-------|---:|----------:|-------:|----------:|
| train | 0.4772 | 0.7523 | 0.3494 | 0.6663 |
| val | 0.5599 | 0.6943 | 0.4691 | 0.6663 |
| test | 0.6293 | 0.6248 | 0.6338 | 0.6663 |

Best log epoch by argmax Validation F1: epoch 13 (val 0.5434, test 0.5906).
