# Positive-complete txn-node scout (`A_identity`)

**Not an exact GCPAL reproduction.**

- epochs=5 cap=2048 seed=2
- opt_steps=7935 anchor_exposures=981984
- train_seconds=3144.2

## Temporal (primary)

### temporal

- **X** @0.5: {"auroc": 0.874005942100123, "auprc": 0.013951571529468226, "f1": 0.0, "precision": 0.0, "recall": 0.0, "threshold": 0.5}
- **X** @val-thr: {"auroc": 0.874005942100123, "auprc": 0.013951571529468226, "f1": 0.006993006993006993, "precision": 0.05714285714285714, "recall": 0.0037243947858473, "threshold": 0.01}
- **H** @0.5: {"auroc": 0.6456133362343066, "auprc": 0.005060278512013024, "f1": 0.0, "precision": 0.0, "recall": 0.0, "threshold": 0.5}
- **H** @val-thr: {"auroc": 0.6456133362343066, "auprc": 0.005060278512013024, "f1": 0.0, "precision": 0.0, "recall": 0.0, "threshold": 0.060000000000000005}
- **HxX** @0.5: {"auroc": 0.8884183049783798, "auprc": 0.07930008315316212, "f1": 0.0, "precision": 0.0, "recall": 0.0, "threshold": 0.5}
- **HxX** @val-thr: {"auroc": 0.8884183049783798, "auprc": 0.07930008315316212, "f1": 0.03653506187389512, "precision": 0.36046511627906974, "recall": 0.019242706393544383, "threshold": 0.06999999999999999}

## Random-40 diagnostic

### random40

- **X** @0.5: {"auroc": 0.9092009107700558, "auprc": 0.034999842427103044, "f1": 0.0, "precision": 0.0, "recall": 0.0, "threshold": 0.5}
- **X** @val-thr: {"auroc": 0.9092009107700558, "auprc": 0.034999842427103044, "f1": 0.06773463346682917, "precision": 0.0391623879995973, "recall": 0.2504829362524147, "threshold": 0.02}
- **H** @0.5: {"auroc": 0.8506760067061087, "auprc": 0.06435385594719589, "f1": 0.0, "precision": 0.0, "recall": 0.0, "threshold": 0.5}
- **H** @val-thr: {"auroc": 0.8506760067061087, "auprc": 0.06435385594719589, "f1": 0.14617536179010093, "precision": 0.1174516318155169, "recall": 0.19349645846748229, "threshold": 0.02}
- **HxX** @0.5: {"auroc": 0.9090724121250946, "auprc": 0.14138593448761466, "f1": 0.048156049984760745, "precision": 0.4514285714285714, "recall": 0.025434642627173213, "threshold": 0.5}
- **HxX** @val-thr: {"auroc": 0.9090724121250946, "auprc": 0.14138593448761466, "f1": 0.2775899781268642, "precision": 0.3629745189807592, "recall": 0.2247263361236317, "threshold": 0.23}
