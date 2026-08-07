# EXPERT_ONLY R198 frozen transfer — implementation (primary scout)

**Date:** 2026-08-01  
**Preflight:** [`notes/expert_only_frozen_transfer_samld_paysim_preflight.md`](expert_only_frozen_transfer_samld_paysim_preflight.md)  
**This turn:** infrastructure + focused tests + **one** standard-account GPU smoke.  
**Not this turn:** six full extract/probe jobs; matched DIRECT_H/adaptive; epoch-20; test access.

---

## Source checkpoint (verified)

| Field | Value |
|-------|-------|
| Path | `saved-models/checkpoint_direct_r198_tfmoe_wtabl_expert_only_20ep_seed2_linear_lr2e-3_epoch10.tar` |
| SHA256 | `f0280e129c7bf0deb4c4a823fe24dd9e9b1c16ac2951aa87f0d81a55bc30c27c` |
| Representation | `pre_embedding_3h` / R198 |
| Load key | `model_state_dict` only (TF MoE discarded) |

---

## Code changes

| File | Role |
|------|------|
| `scripts/extract_direct_r198_full_cell.py` | Generalized `--data {Small-HI,PaySim,SAML-D}`; PaySim legacy contract + train-fit; SAML-D train-fit; hard train/val allowlist; refuse test + seed-only; SHA gate |
| `scripts/expert_only_frozen_transfer_scout.py` | Smoke / extract / probe orchestration; P1 logistic settings; SAML-D exploratory protocol id; integrity + interpretation gates |
| `embedding_extraction.py` | Freeze all params + BN eval after load; honor `extract_max_batches` for hetero smoke |
| `train_util.py` | Pass `max_batches` through `extract_seed_embeddings_hetero` |
| `tests/test_expert_only_frozen_transfer_scout.py` | Focused unit tests |
| `slurm/run_expert_only_frozen_transfer_smoke.sh` | Standard `mit_preemptable` / `mit_general` / `normal` smoke |

Small-HI default path preserved (no train-fit unless requested).

---

## Protocols

### PaySim — `P1_strict_inductive_legacy`
- Contract: `paysim_legacy_duplicate_v1` (balances / `isFlaggedFraud` prohibited)
- ports + TDS + corrected reverse; edge_dim=8; train-fit z-norm
- Frozen AML BN; LogisticRegression (`C=1`, `class_weight=model` gin CE weights, downstream seed=1)
- No PaperStyleMLP

### SAML-D — `samld_frozen_expert_only_r198_valonly_v1` (**NEW / EXPLORATORY**)
- Protocol-B geometry (ports+TDS+corrected reverse, edge_dim=8, train-fit)
- Same logistic learner settings for parity
- Refuse supervised protocol-A (edge_dim=6 / TDS off)

### Controls
- Matched random R198 (`--random_init`)
- Target X-only (8-d train-fit edge features)
- Predeclared transfer signal: expert AUPRC − random AUPRC ≥ **0.003** on validation

---

## Focused tests

```bash
python -m pytest -q tests/test_expert_only_frozen_transfer_scout.py
```

**Result (local, this turn):** **13 passed**.

Coverage: source SHA; seed-only refuse; test-split refuse; Small-HI default; PaySim contract/train-fit; balance-contract refuse; SAML-D edge_dim=8; edge_dim=6 refuse; geometry locks; logistic learner / no MLP; ID uniqueness/disjointness; BN snapshot helper.

---

## Smoke job

```bash
sbatch slurm/run_expert_only_frozen_transfer_smoke.sh
```

| Setting | Value |
|---------|-------|
| partition | `mit_preemptable` |
| account | `mit_general` |
| qos | `normal` |
| mem | 96G |
| gpu | 1 |
| time | 04:00:00 |
| batches | 2 train + 2 val per target × {expert_only, random_init} |
| outputs | `results/diagnostics/expert_only_frozen_transfer_samld_paysim/smoke.json` |

### Smoke gates
1. Source SHA matches lock  
2. Seed-only path refused  
3. Test splits refused  
4. edge_dim=6 / TDS-off refused  
5. PaySim + SAML-D each: expert_only and random_init load/extract  
6. edge_dim=8; te_inds empty; no test.npz  
7. z dim=198; finite; unique EdgeIDs; train∩val=0 (partial subset)  
8. BN buffers unchanged across extract; model.eval; requires_grad=False  
9. Load key = `model_state_dict`; TF MoE discarded  
10. LogisticRegression selected; PaperStyleMLP forbidden  
11. No full embeddings written  

---

## Proposed but **unsubmitted** six-job launch

Projected disk: PaySim ~4.3 GiB ×2 encoders + SAML-D ~6.5 GiB ×2 ≈ **~22 GiB** embeddings before probe artifacts.

```bash
# GPU extracts (≤2 concurrent)
sbatch --partition=mit_preemptable --account=mit_general --qos=normal \
  --gres=gpu:1 --mem=64G -c 8 -t 06:00:00 -J eo_ft_ps_exp \
  --wrap='module load miniforge; source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate multignn; cd /home/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN; export PYTHONPATH=$PWD; python scripts/expert_only_frozen_transfer_scout.py extract --data PaySim --encoder expert_only --embeddings_root embeddings/expert_only_frozen_transfer_samld_paysim'

sbatch --partition=mit_preemptable --account=mit_general --qos=normal \
  --gres=gpu:1 --mem=64G -c 8 -t 06:00:00 -J eo_ft_ps_rnd \
  --wrap='module load miniforge; source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate multignn; cd /home/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN; export PYTHONPATH=$PWD; python scripts/expert_only_frozen_transfer_scout.py extract --data PaySim --encoder random_init --embeddings_root embeddings/expert_only_frozen_transfer_samld_paysim'

sbatch --partition=mit_preemptable --account=mit_general --qos=normal \
  --gres=gpu:1 --mem=96G -c 8 -t 06:00:00 -J eo_ft_sd_exp \
  --wrap='module load miniforge; source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate multignn; cd /home/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN; export PYTHONPATH=$PWD; python scripts/expert_only_frozen_transfer_scout.py extract --data SAML-D --encoder expert_only --embeddings_root embeddings/expert_only_frozen_transfer_samld_paysim'

sbatch --partition=mit_preemptable --account=mit_general --qos=normal \
  --gres=gpu:1 --mem=96G -c 8 -t 06:00:00 -J eo_ft_sd_rnd \
  --wrap='module load miniforge; source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate multignn; cd /home/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN; export PYTHONPATH=$PWD; python scripts/expert_only_frozen_transfer_scout.py extract --data SAML-D --encoder random_init --embeddings_root embeddings/expert_only_frozen_transfer_samld_paysim'

# CPU probes (after corresponding extracts)
sbatch --partition=mit_preemptable --account=mit_general --qos=normal \
  --mem=64G -c 8 -t 06:00:00 -J eo_ft_ps_probe \
  --wrap='module load miniforge; source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate multignn; cd /home/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN; export PYTHONPATH=$PWD; python scripts/expert_only_frozen_transfer_scout.py probe --data PaySim --embeddings_root embeddings/expert_only_frozen_transfer_samld_paysim --out_dir results/diagnostics/expert_only_frozen_transfer_samld_paysim/probe_PaySim'

sbatch --partition=mit_preemptable --account=mit_general --qos=normal \
  --mem=96G -c 8 -t 06:00:00 -J eo_ft_sd_probe \
  --wrap='module load miniforge; source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate multignn; cd /home/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN; export PYTHONPATH=$PWD; python scripts/expert_only_frozen_transfer_scout.py probe --data SAML-D --embeddings_root embeddings/expert_only_frozen_transfer_samld_paysim --out_dir results/diagnostics/expert_only_frozen_transfer_samld_paysim/probe_SAML-D'
```

Printable via: `python scripts/expert_only_frozen_transfer_scout.py print_proposed_jobs`

---

## Confirmations

- No test graph/data accessed in unit tests or smoke design (`skip_test_eval`, train/val allowlist).  
- No full extract/probe jobs submitted this turn.  
- No matched DIRECT_H/adaptive / epoch-20 work.  
- Advanced account/QOS not used.
