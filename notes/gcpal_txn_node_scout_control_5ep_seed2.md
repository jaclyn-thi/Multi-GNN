# GCPAL-style txn-node scout (`control`)

**Not an exact GCPAL reproduction.**

- epochs=5 batch=2048 seed=2
- temporal MLP H||X @0.5: {"auroc": 0.4456621617162246, "auprc": 0.0037605767910725476, "f1": 0.0, "precision": 0.0, "recall": 0.0, "threshold": 0.5}
- random40 diagnostic: {"auroc": 0.9165770060078167, "auprc": 0.15371131468891153, "f1": 0.08338368580060422, "precision": 0.6764705882352942, "recall": 0.044430135222150675, "threshold": 0.5}
- train_seconds=3011.3
