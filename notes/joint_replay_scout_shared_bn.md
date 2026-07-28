# Joint replay scout — `shared_bn`

> Exploratory / post-hoc. `table_eligible=false`. Validation only.
> Twin: `results/diagnostics/joint_replay_scout/shared_bn.json`

- Job ID: `19060692`
- Source ckpt SHA256: `18e06f555aa4880dfc1e95caa3f54a207e5aa186d266887772640feb93a06ae6`
- Steps: 500 (250 AML + 250 PaySim, 1:1)
- BN mode: **shared_bn**
- Wall sec: 5851.7; peak GPU GB: 33.23441553115845

## PaySim val (post-128 H logistic primary)

- H AUPRC@0.5: **0.008808**
- H+X AUPRC@0.5: **0.010606**
- Refs: frozen=0.0210, sequential_aml_init=0.0433, x_only=0.0046, random=0.0114

## AMLWorld val (pre-3h H+X+TF MLP)

- AUPRC@0.5: **0.535640**  F1@0.5: **0.563295**
- Ref original: 0.5338

- Seed-edge hash count logged: aml=32 paysim=32

