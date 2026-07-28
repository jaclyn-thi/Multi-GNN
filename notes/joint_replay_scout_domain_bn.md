# Joint replay scout — `domain_bn`

> Exploratory / post-hoc. `table_eligible=false`. Validation only.
> Twin: `results/diagnostics/joint_replay_scout/domain_bn.json`

- Job ID: `19060693`
- Source ckpt SHA256: `18e06f555aa4880dfc1e95caa3f54a207e5aa186d266887772640feb93a06ae6`
- Steps: 500 (250 AML + 250 PaySim, 1:1)
- BN mode: **domain_bn**
- Wall sec: 6337.1; peak GPU GB: 33.23441553115845

## PaySim val (post-128 H logistic primary)

- H AUPRC@0.5: **0.040608**
- H+X AUPRC@0.5: **0.103377**
- Refs: frozen=0.0210, sequential_aml_init=0.0433, x_only=0.0046, random=0.0114

## AMLWorld val (pre-3h H+X+TF MLP)

- AUPRC@0.5: **0.524795**  F1@0.5: **0.535048**
- Ref original: 0.5338

- Seed-edge hash count logged: aml=32 paysim=32

