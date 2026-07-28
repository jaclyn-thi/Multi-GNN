# Sequential AMLWorld→PaySim SSL scout

> Exploratory / post-hoc. `table_eligible=false`. Validation only.
> `sequential_domain_adaptive_ssl=true`. Not joint multidomain pretraining.

- Promote AML→PaySim (`aml_init`): **FAIL**

## PaySim val AUPRC (post-128 H, logistic)

| Arm | Val AUPRC |
|-----|----------:|
| frozen_aml | 0.021013 |
| bn_only | 0.023255 |
| random_init_paysim_ssl | 0.019067 |
| aml_init_paysim_ssl | 0.043293 |
| x_only | 0.004591 |

## AMLWorld retention (pre-3h H+X+TF, restored original BN for continued)

| Encoder | Val AUPRC |
|---------|----------:|
| original AML | 0.533791 |
| aml_init + original BN | 0.389982 |

## Gate deltas

```json
{
  "passed": false,
  "checks": {
    "vs_frozen": true,
    "vs_bn_only": true,
    "vs_random": true,
    "vs_x_only": true,
    "aml_retention": false
  },
  "deltas": {
    "aml_init_minus_frozen": 0.022280902983994245,
    "aml_init_minus_bn_only": 0.02003819529530651,
    "aml_init_minus_random": 0.024226914430936098,
    "aml_init_minus_x_only": 0.03870252587708237,
    "aml_original_minus_continued": 0.14380969716556447
  },
  "thresholds": {
    "paysim_auprc_margin_abs": 0.003,
    "aml_regress_max": 0.02
  }
}
```

Artifacts: `/orcd/home/002/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN/results/diagnostics/sequential_aml_to_paysim_ssl_scout.json`, `/orcd/home/002/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN/results/diagnostics/sequential_aml_to_paysim_ssl/aggregate.json`, cells under `/orcd/home/002/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN/results/diagnostics/sequential_aml_to_paysim_ssl/cells`.
