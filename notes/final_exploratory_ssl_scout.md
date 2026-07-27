# Final exploratory SSL scout (C0 + M)

> Exploratory / post-hoc. `table_eligible=false`. Validation only.

- M gate vs C0: **FAIL**
- PaySim val AUPRC Δ(M−C0)=-0.0024 (need ≥0.003 or F1 Δ≥0.01)
- PaySim val F1 Δ(M−C0)=+0.0008
- AML pre-3h H+X+TF val AUPRC regression (C0−M)=-0.0111 (max 0.02)
- Beats matched random: False

## C0 vs original uncontinued seed-2

- Original AML val AUPRC: 0.5333291631565786
- C0 AML val AUPRC: 0.5243
- Original PaySim val AUPRC: 0.021656440218880312
- C0 PaySim val AUPRC: 0.0130

Artifacts: `/orcd/home/002/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN/results/diagnostics/final_exploratory_ssl_scout.json`, `/orcd/home/002/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN/results/diagnostics/final_exploratory_ssl_scout/aggregate.json`, cells under `/orcd/home/002/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN/results/diagnostics/final_exploratory_ssl_scout/cells`.
