# Compact PNA diagnostic

| Encoder / setup | Representation + features | Small-HI AUPRC | Takeaway |
| --- | --- | --- | --- |
| GIN main stack | pre-3h + raw + temporal-flow | 0.501 | main encoder/result |
| PNA width-aligned | post-128 embedding only | 0.147 | post-128 understates PNA |
| PNA width-aligned | pre-3h + raw + temporal-flow | 0.407 | competitive but below GIN; one-seed diagnostic |

**Notes:**
- PNA rows are one-seed width-aligned scouts; not a full architecture ranking.
- Best-stack PNA row is downstream-only (no PNA SSL retraining).
