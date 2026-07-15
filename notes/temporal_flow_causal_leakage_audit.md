# temporal_flow_causal leakage audit

**Datasets:** Small-HI, Small-LI

## Small-HI

- Recompute matches cache: **True** (max |Δ|=0.00e+00)
- Edges in timestamp ties: 5078119

### Default-history fractions (true flags, not zero==no-NaN)

| split | sender no prior | receiver no prior | 7d count=0 | no amount hist | no pair prior | pair_repeat=1 |
|-------|----------------:|------------------:|-----------:|---------------:|--------------:|--------------:|
| train | 0.1338 | 0.1156 | 0.1535 | 0.1338 | 0.2933 | 0.7067 |
| val | 0.0009 | 0.0002 | 0.0059 | 0.0009 | 0.0394 | 0.9606 |
| test | 0.0010 | 0.0002 | 0.0660 | 0.0010 | 0.0432 | 0.9568 |

## Small-LI

- Recompute matches cache: **True** (max |Δ|=0.00e+00)
- Edges in timestamp ties: 6923996

### Default-history fractions (true flags, not zero==no-NaN)

| split | sender no prior | receiver no prior | 7d count=0 | no amount hist | no pair prior | pair_repeat=1 |
|-------|----------------:|------------------:|-----------:|---------------:|--------------:|--------------:|
| train | 0.1345 | 0.1162 | 0.1543 | 0.1345 | 0.2938 | 0.7062 |
| val | 0.0010 | 0.0003 | 0.0060 | 0.0010 | 0.0391 | 0.9609 |
| test | 0.0009 | 0.0002 | 0.0645 | 0.0009 | 0.0408 | 0.9592 |
