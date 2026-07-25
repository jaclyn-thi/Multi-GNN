# Secondary: SSL-pretrained D+ with supervised partial fine-tuning (seed 2)

| Metric | Frozen seed-2 ref | Partial FT best (ep 18) |
|--------|------------------:|------------------------:|
| val AUPRC | 0.550 | 0.5996 |
| test AUPRC | 0.674 | 0.7009 |
| test F1@0.5 | 0.656 | 0.6971 |

AML labels update the classifier and final encoder block (`convs.1`/`emlps.1`/`batch_norms.1`).

