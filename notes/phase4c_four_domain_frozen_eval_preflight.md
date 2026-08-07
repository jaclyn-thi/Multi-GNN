# Phase-4C four-domain frozen-evaluation preflight

Authorized training manifest SHA: `7dcfaa52d38abd6e929633c028b2e2a21743385d26c6b2cdbd34e87b2f42d3aa`

This stage builds isolated frozen-evaluation infrastructure and runs one bounded
real-data Slurm preflight. It does **not** submit the full 40-cell extract/probe DAG.

## Protocol

- Contract: `financial_multidataset_shared_core_4domain_v1`
- Domains: Small-HI, SAML-D, Small-LI, PaySim
- Representation: direct frozen encoder R198 (`bypass_embedding_head=True`)
- Projection: reload-checked then bypassed (no projected H)
- Probe (later): PaperStyleMLP 198→128→1, 20 ep, lr 1e-3, bs 8192, seed 2
- PaySim label semantics: **fraud** (not laundering)
- Validation only; no test loading or scoring

## Artifacts

- Package: `phase4c_four_domain_frozen_eval/`
- CLI (preflight/inventory/plan): `scripts/run_phase4c_four_domain_frozen_eval_preflight.py`
- Legacy training-manifested extract/probe script (unchanged): `scripts/run_phase4c_four_domain_frozen_eval.py`
- Preflight Slurm: `slurm/run_phase4c_four_domain_frozen_eval_preflight.sh`
- Results: `results/diagnostics/phase4c_four_domain_frozen_eval_preflight_v1/`

## Commands

Login-safe:

```bash
python scripts/run_phase4c_four_domain_frozen_eval_preflight.py inventory
python scripts/run_phase4c_four_domain_frozen_eval_preflight.py plan
python -m pytest tests/test_phase4c_four_domain_frozen_eval.py -q
```

Real-data preflight (Slurm only):

```bash
sbatch --parsable slurm/run_phase4c_four_domain_frozen_eval_preflight.sh
```
